import os
import io
import time
import queue
import math
import random
from collections import deque
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
from scipy.spatial.distance import cdist

# TTS offline
import pyttsx3

# WebRTC
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
import av

# JSON
import json
import unicodedata
import difflib

# ASR opcional
try:
    import vosk
    VOSK_AVAILABLE = True
except Exception:
    VOSK_AVAILABLE = False

# ----------------------------
# Configs
# ----------------------------
st.set_page_config(page_title="Fale Terena – Verificador (Robusto)", page_icon="🗣️", layout="centered")
DATA_DIR = Path("data")
LEXICON_CSV = DATA_DIR / "lexicon_terena.csv"
REF_AUDIO_DIR = DATA_DIR / "ref_audio"
DIC_JSON = DATA_DIR / "dicionario_terena.json"
VOSK_MODEL_DIR = DATA_DIR / "models" / "vosk-pt"
RTC_CONFIGURATION = RTCConfiguration(iceServers=[])

# ----------------------------
# Utils
# ----------------------------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def create_demo_lexicon():
    rows = [
        ("saud_01","Únati.","Tudo bem?"),
        ("saud_02","Ápeepo.","Vou bem."),
        ("perg_01","Na yéno?","Aonde vai?"),
        ("resp_01","Mbihópotine.","Estou voltando/indo embora."),
        ("nega_01","Ako yónongu.","Não vou a nenhuma parte."),
        ("desa_01","Ihárooti.","Até amanhã."),
        ("desa_02","Po’i káxe.","Até outro dia."),
        ("dir_01","Miranda-ke yónom.","Vou a Miranda."),
        ("obj_01","Áhara.","Enxada."),
        ("ani_01","Kachóro.","Cachorro."),
    ]
    return pd.DataFrame(rows, columns=["id","ter","pt"])

@st.cache_data(show_spinner=False)
def load_lexicon() -> pd.DataFrame:
    ensure_dirs()
    if not LEXICON_CSV.exists():
        df = create_demo_lexicon()
        df.to_csv(LEXICON_CSV, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(LEXICON_CSV)
    return df

def tts_reference(text: str, rate_delta: int = 0):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', max(100, rate + rate_delta))
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    try:
        engine.save_to_file(text, tmp_wav)
        engine.runAndWait()
        y, sr = sf.read(tmp_wav, dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        return y, sr
    finally:
        try: os.remove(tmp_wav)
        except Exception: pass

def load_native_reference(id_str: str):
    f = REF_AUDIO_DIR / f"{id_str}.wav"
    if f.exists():
        y, sr = sf.read(str(f), dtype="float32")
        if y.ndim > 1: y = np.mean(y, axis=1)
        return y, sr
    return None, None

def normalize_audio(y: np.ndarray) -> np.ndarray:
    if y is None or len(y) == 0: return y
    peak = np.max(np.abs(y))
    if peak < 1e-6: return y
    return y / peak

def trim(y: np.ndarray, sr: int, top_db: int = 35):
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y

def _safe_pitch(y: np.ndarray, sr: int):
    try:
        f0, vflag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0, nan=0.0)
        return f0.astype(np.float32)
    except Exception:
        return np.zeros(max(1, len(y)//512), dtype=np.float32)

def feature_stack(y: np.ndarray, sr: int):
    """
    Stack robusto: [MFCC(13) + Δ + ΔΔ, log-mel(40), pitch(1)] -> (T, D)
    """
    y = normalize_audio(y)
    y = trim(y, sr)
    if len(y) < sr * 0.2:
        return None
    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat_mfcc = np.vstack([mfcc, d1, d2])  # (39, T)
    # log-mel
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    logmel = librosa.power_to_db(mel + 1e-9)
    # pitch
    f0 = _safe_pitch(y, sr)
    # alinhar comprimentos por frame hop ~512
    T = min(feat_mfcc.shape[1], logmel.shape[1], len(f0))
    if T <= 5:
        return None
    X = np.vstack([feat_mfcc[:, :T], logmel[:, :T], f0[:T][None, :]])  # (39+40+1, T)
    return X.T.astype(np.float32)  # (T, D)

def dtw_distance(F_ref: np.ndarray, F_user: np.ndarray) -> float:
    if F_ref is None or F_user is None:
        return np.inf
    C = cdist(F_ref, F_user, metric="cosine")
    n, m = C.shape
    D = np.full((n+1, m+1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    return float(D[n, m] / (n + m))

def base_score(dist: float) -> int:
    if not np.isfinite(dist): return 0
    if dist <= 0.25: base = 95 + (0.25 - dist) * 20
    elif dist <= 0.45: base = 70 + (0.45 - dist) * 100
    elif dist <= 0.65: base = 50 + (0.65 - dist) * 100
    else: base = max(0, 50 - (dist - 0.65) * 60)
    return int(np.clip(base, 0, 100))

def softmax(xs, alpha=10.0):
    xs = np.array(xs, dtype=np.float32)
    z = np.exp(alpha * (xs - np.max(xs)))
    p = z / (np.sum(z) + 1e-9)
    return p

def contrastive_probability(dist_target: float, dist_negatives: list[float]) -> float:
    # Converte distâncias em similaridades s = -dist, aplica softmax
    sims = [-dist_target] + [-d for d in dist_negatives]
    p = softmax(sims, alpha=8.0)[0]  # prob do alvo
    return float(p)

def _strip_accents_lower(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s_norm = unicodedata.normalize("NFD", s)
    s_noacc = "".join(ch for ch in s_norm if unicodedata.category(ch) != "Mn")
    return s_noacc.casefold()

@st.cache_resource(show_spinner=False)
def load_dicionario_terena():
    if not DIC_JSON.exists():
        return [], {}, {}, {}
    with open(DIC_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx_ter, idx_ter_comp, idx_pt = {}, {}, {}
    for item in data:
        ter = _strip_accents_lower(item.get("terena", ""))
        terc = _strip_accents_lower(item.get("terena_completo", ""))
        pt  = _strip_accents_lower(item.get("portugues", ""))
        if ter:  idx_ter.setdefault(ter, []).append(item)
        if terc: idx_ter_comp.setdefault(terc, []).append(item)
        if pt:   idx_pt.setdefault(pt, []).append(item)
    return data, idx_ter, idx_ter_comp, idx_pt

def text_ref_from_json(terena_text: str, portugues_text: str):
    _, idx_ter, idx_ter_comp, idx_pt = load_dicionario_terena()
    keys = []
    if terena_text:
        k = _strip_accents_lower(terena_text)
        keys.append(("ter", k)); keys.append(("terc", k))
    if portugues_text:
        keys.append(("pt", _strip_accents_lower(portugues_text)))
    for kind, k in keys:
        if kind == "ter" and k in idx_ter:
            for item in idx_ter[k]:
                t = item.get("pronuncia") or item.get("terena_completo") or item.get("terena")
                if t and str(t).strip(): return str(t).strip()
        if kind == "terc" and k in idx_ter_comp:
            for item in idx_ter_comp[k]:
                t = item.get("pronuncia") or item.get("terena_completo") or item.get("terena")
                if t and str(t).strip(): return str(t).strip()
        if kind == "pt" and k in idx_pt:
            for item in idx_pt[k]:
                t = item.get("pronuncia") or item.get("terena_completo") or item.get("terena")
                if t and str(t).strip(): return str(t).strip()
    return None

@st.cache_resource(show_spinner=False)
def _load_vosk_model():
    if not VOSK_AVAILABLE or not VOSK_MODEL_DIR.exists():
        return None
    return vosk.Model(str(VOSK_MODEL_DIR))

def asr_text_pt(wav_bytes: bytes, sample_rate_hint: int = 16000) -> str | None:
    model = _load_vosk_model()
    if model is None:
        return None
    try:
        y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != sample_rate_hint:
            y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate_hint)
            sr = sample_rate_hint
        rec = vosk.KaldiRecognizer(model, sr); rec.SetWords(False)
        block = int(sr * 0.5); i = 0
        while i < len(y):
            chunk = (y[i:i+block] * 32767.0).astype(np.int16).tobytes()
            rec.AcceptWaveform(chunk); i += block
        final = rec.FinalResult()
        import json as _json
        txt = _json.loads(final).get("text", "").strip()
        return txt if txt else None
    except Exception:
        return None

def text_similarity(a: str, b: str) -> float:
    a = _strip_accents_lower(a); b = _strip_accents_lower(b)
    return difflib.SequenceMatcher(None, a, b).ratio()

# -------------- AudioProcessor --------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=48000 * 10)
        self.level_dbfs = -60.0
        self.speaking = False
        self.sample_rate = 48000

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        try:
            x = frame.to_ndarray().astype(np.float32) / 32768.0
            if x.ndim == 2: x = np.mean(x, axis=0)
            self.buffer.extend(x.tolist())
            rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
            db = 20.0 * math.log10(rms + 1e-12)
            self.level_dbfs = max(-60.0, min(0.0, db))
            self.speaking = rms > 0.01
        except Exception:
            pass
        return frame

    def reset_buffer(self):
        self.buffer.clear()

    def get_level_pct(self) -> int:
        return int(np.clip((self.level_dbfs + 60.0) / 60.0 * 100.0, 0, 100))

    def get_recent_waveform(self, max_points: int = 2000) -> np.ndarray:
        if not self.buffer: return np.array([])
        y = np.array(self.buffer, dtype=np.float32)
        if y.size > max_points:
            step = max(1, y.size // max_points)
            y = y[::step]
        return y

    def get_wav_bytes(self, sample_rate: int) -> bytes:
        if not self.buffer: return b""
        y = np.array(self.buffer, dtype=np.float32)
        buf = io.BytesIO(); sf.write(buf, y, sample_rate, format="WAV")
        return buf.getvalue()

# ----------------------------
# App
# ----------------------------
st.title("🗣️ Fale Terena – Verificador de Pronúncia (Robusto)")
st.caption("Agora com comparação **contrastiva** (alvo vs. negativas) e recursos anti-falso‑positivo.")

df = load_lexicon()

# estado
if "user_wav_bytes" not in st.session_state:
    st.session_state.user_wav_bytes = None

with st.sidebar:
    st.subheader("Opções")
    rate_delta = st.slider("Velocidade da referência (pyttsx3)", -60, 60, 0, 5)
    palavra_idx = st.number_input("Índice da palavra (0..n-1)", min_value=0, max_value=len(df)-1, value=0, step=1)
    st.markdown("---")
    st.markdown("**Referência:**")
    use_native = st.toggle("Usar .wav nativo se existir em data/ref_audio/<id>.wav", value=True)
    prefer_json = st.toggle("Preferir dicionário JSON (pronuncia → terena_completo → terena)", value=True)
    st.markdown("---")
    st.markdown("**Entrada:**")
    use_mic = st.toggle("Usar microfone via WebRTC", value=True)
    n_neg = st.slider("Negativas para contraste", 3, 10, 6, 1)
    st.caption("Mais negativas = avaliação mais seletiva (pode ficar mais lenta).")

row = df.iloc[palavra_idx]
target_id = str(row["id"])
target_ter = str(row["ter"])
target_pt = str(row["pt"])

st.markdown("### Palavra alvo")
st.write(f"**Terena:** {target_ter}")
st.caption(f"Português: {target_pt}")

# --- referência alvo ---
ref_y, ref_sr = (None, None)
if use_native:
    ref_y, ref_sr = load_native_reference(target_id)

reference_text = None
if prefer_json:
    reference_text = text_ref_from_json(terena_text=target_ter, portugues_text=target_pt)

if ref_y is None:
    chosen_text = reference_text or target_ter
    ref_y, ref_sr = tts_reference(chosen_text, rate_delta=rate_delta)
    st.caption("Referência: TTS offline (pyttsx3). Use fones/volume baixo.")

ref_wav = io.BytesIO(); sf.write(ref_wav, ref_y, ref_sr, format="WAV"); ref_wav.seek(0)
st.audio(ref_wav.read(), format="audio/wav")

st.info("Use **fones** ou baixe o volume do alto-falante para a referência não vazar no microfone.")

# --- captura ---
if use_mic:
    st.markdown("#### Gravar minha pronúncia")
    ctx = webrtc_streamer(
        key="pronuncia",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=4096,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
            "video": False,
        },
        audio_processor_factory=AudioProcessor,
        async_processing=False,
    )

    if ctx and ctx.state.playing and ctx.audio_processor:
        ap = ctx.audio_processor
        sr_stream = ap.sample_rate

        st.markdown("##### Monitor ao vivo")
        cvu, cplot = st.columns([1, 3])
        with cvu:
            vu_placeholder = st.empty()
            speak_indicator = st.empty()
            if st.button("Iniciar monitor (3 s)"):
                ap.reset_buffer(); start = time.time()
                while time.time() - start < 3 and ctx.state.playing:
                    vu_placeholder.progress(ap.get_level_pct(), text="Nível do microfone")
                    speak_indicator.write("Falando: " + ("🟢" if ap.speaking else "⚪"))
                    time.sleep(0.1)
        with cplot:
            wave_placeholder = st.empty()
            if st.button("Atualizar forma de onda"):
                y = ap.get_recent_waveform(1500)
                if y.size: wave_placeholder.line_chart(pd.DataFrame({"amp": y}))
                else: wave_placeholder.info("Sem dados ainda.")

        if st.button("Parar & Usar minha gravação", type="primary", key="btn_use_recording"):
            st.session_state.user_wav_bytes = ap.get_wav_bytes(sr_stream)
            if not st.session_state.user_wav_bytes:
                st.warning("Nada capturado.")
            else:   
                st.success("Gravação salva!"); st.audio(st.session_state.user_wav_bytes, format="audio/wav")
else:
    st.markdown("#### Upload de áudio (.wav)")
    f = st.file_uploader("Envie um .wav mono/16k~48kHz", type=["wav"], key="uploader_wav")
    if f is not None:
        st.session_state.user_wav_bytes = f.read()
        st.success("Arquivo carregado!"); st.audio(st.session_state.user_wav_bytes, format="audio/wav")

# Mostrar última
if st.session_state.user_wav_bytes:
    with st.expander("Minha última gravação"):
        st.audio(st.session_state.user_wav_bytes, format="audio/wav")

# --- avaliações helpers ---
def make_reference_for_text(txt: str, rate_delta: int = 0):
    y, sr = tts_reference(txt, rate_delta=rate_delta)
    return feature_stack(y, sr)

@st.cache_data(show_spinner=False)
def sample_negative_texts(df: pd.DataFrame, idx_exclude: int, k: int):
    # retorna lista de (id, ter, pt) diferentes do alvo
    ids = list(range(len(df)))
    ids.remove(idx_exclude)
    random.seed(42)  # determinístico por sessão
    picks = random.sample(ids, min(k, len(ids)))
    return [(str(df.iloc[i]["id"]), str(df.iloc[i]["ter"]), str(df.iloc[i]["pt"])) for i in picks]

# --- Avaliar ---
user_wav_bytes = st.session_state.get("user_wav_bytes", None)

if st.button("Avaliar pronúncia", use_container_width=True, key="btn_eval"):
    if ref_y is None or not ref_sr:
        st.error("Sem referência.")
        st.stop()
    if not user_wav_bytes:
        st.warning("Grave ou carregue um áudio primeiro.")
        st.stop()

    # features do alvo e do usuário
    F_ref = feature_stack(ref_y, ref_sr)
    y_u, sr_u = sf.read(io.BytesIO(user_wav_bytes), dtype="float32")
    if y_u.ndim > 1: y_u = np.mean(y_u, axis=1)
    F_user = feature_stack(y_u, sr_u)

    dist_t = dtw_distance(F_ref, F_user)
    n_frames_ref = 0 if F_ref is None else F_ref.shape[0]
    n_frames_user = 0 if F_user is None else F_user.shape[0]

    # NEGATIVAS (contrastivas)
    negatives = sample_negative_texts(df, palavra_idx, n_neg)
    dist_negs = []
    with st.spinner("Comparando a pronúncia..."):
        for _id, ter, pt in negatives:
            # usa JSON quando possível
            txtn = text_ref_from_json(ter, pt) or ter
            Yn, srn = load_native_reference(_id)
            if Yn is None:
                Yn, srn = tts_reference(txtn, rate_delta=rate_delta)
            Fn = feature_stack(Yn, srn)
            dn = dtw_distance(Fn, F_user)
            dist_negs.append(dn)

    # prob contrastiva
    p_target = contrastive_probability(dist_t, dist_negs)

    # ASR opcional
    asr_sim = None
    if VOSK_AVAILABLE and VOSK_MODEL_DIR.exists():
        txt = asr_text_pt(user_wav_bytes)
        if txt:
            ref_txt = reference_text or target_ter
            asr_sim = text_similarity(txt, ref_txt)
            st.caption(f"Reconhecido (ASR): “{txt}” · similaridade≈{asr_sim:.2f}")

    # score final
    s_base = base_score(dist_t)
    # mistura: 60% base + 40% prob contrastiva
    score = int(0.6 * s_base + 40.0 * p_target)
    # penalidade de duração
    if n_frames_ref > 0 and n_frames_user > 0:
        ratio = n_frames_user / n_frames_ref
        dev = abs(ratio - 1.0)
        if dev > 0.33:
            score -= min(20, int(60 * (dev - 0.33)))
    # teto por ASR (se disponível)
    if asr_sim is not None and asr_sim < 0.55:
        score = min(score, int(40 + 50 * asr_sim))

    score = int(np.clip(score, 0, 100))

    # feedback
    def feedback_from_score(s: int) -> str:
        if s >= 90: return "Excelente! Pronúncia muito próxima do alvo."
        if s >= 75: return "Muito bom! Pequenos ajustes de ritmo ou vogais."
        if s >= 60: return "Bom! Tente abrir/fechar mais as vogais e cuidar da nasalização."
        return "Vamos ajustar: foque nas vogais tônicas, nasalização e ritmo."

    colA, colB = st.columns(2)
    with colA:
        st.metric("Score de pronúncia", f"{score}/100")
        st.caption(f"DTW alvo: {dist_t:.3f} · P(alvo|usuário): {p_target:.2f}")
    with colB:
        st.success(feedback_from_score(score) if score>=60 else feedback_from_score(score))

    with st.expander("Negativas usadas e distâncias"):
        for (nid, ter, pt), dn in zip(negatives, dist_negs):
            st.write(f"- `{ter}` / `{pt}` → dist={dn:.3f}")

st.markdown("---")
if VOSK_AVAILABLE:
    if VOSK_MODEL_DIR.exists():
        st.caption("ASR offline habilitado (Vosk).")
    else:
        st.caption("Opcional: instale `vosk` e coloque um modelo PT-BR em `data/models/vosk-pt/`.")
else:
    st.caption("Opcional: `pip install vosk` para checagem textual offline.")

st.caption("Use gravações nativas em `data/ref_audio/<id>.wav` para melhor realismo e ajuste o campo 'pronuncia' no JSON.")

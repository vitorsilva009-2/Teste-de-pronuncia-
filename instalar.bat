@echo off
echo ========================================
echo  Nhe'e Pora - Vozes do Tupi-Guarani
echo  Script de Instalacao Automatica
echo ========================================
echo.

echo Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.8+ de https://python.org
    pause
    exit /b 1
)

echo Python encontrado!
echo.

echo Instalando dependencias...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERRO: Falha na instalacao das dependencias
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Instalacao concluida com sucesso!
echo ========================================
echo.
echo Para executar a aplicacao, digite:
echo   streamlit run app.py
echo.
echo Ou execute o arquivo: executar.bat
echo.
pause

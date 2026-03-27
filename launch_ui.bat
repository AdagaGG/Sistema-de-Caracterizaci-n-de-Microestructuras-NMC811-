@echo off
setlocal EnableExtensions
pushd "%~dp0"

set "ENV_NAME=nmc811-segmentation"
set "CONDA_BAT=%UserProfile%\miniconda3\condabin\conda.bat"

if exist "%CONDA_BAT%" (
    call "%CONDA_BAT%" activate "%ENV_NAME%"
) else (
    call conda activate "%ENV_NAME%"
)

if errorlevel 1 (
    echo [ERROR] Could not activate conda environment "%ENV_NAME%".
    echo [HINT] Open "Anaconda Prompt" and run: conda activate %ENV_NAME%
    popd
    exit /b 1
)

python -m pip install --disable-pip-version-check streamlit plotly
if errorlevel 1 (
    echo [ERROR] Failed installing streamlit/plotly.
    popd
    exit /b 1
)

python -m streamlit run src/ui/app.py --server.port 8501 --server.headless false
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%

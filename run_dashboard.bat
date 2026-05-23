@echo off
cd /d "%~dp0"
echo Starting Devin's Younger Brother dashboard...
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo If this failed, run: python -m pip install -r requirements.txt
    pause
)

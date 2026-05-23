# Devin's Younger Brother — Streamlit launcher (Windows)
# Uses `python -m streamlit` so PATH does not need streamlit.exe.

Set-Location $PSScriptRoot

Write-Host "Starting Devin's Younger Brother dashboard..." -ForegroundColor Cyan
python -m streamlit run app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "If Python is missing, install Python 3.9+ and run:" -ForegroundColor Yellow
    Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit $LASTEXITCODE
}

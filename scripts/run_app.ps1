$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
& "$projectRoot/.venv/Scripts/python.exe" -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

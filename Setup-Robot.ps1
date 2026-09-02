param(
    [switch]$ForceVoice
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
}

& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt
& $Python src\setup_assets.py
if ($ForceVoice) {
    & $Python src\setup_voice.py --force
} else {
    & $Python src\setup_voice.py
}

Write-Host "Setup complete. Run .\Start-Robot.ps1" -ForegroundColor Green

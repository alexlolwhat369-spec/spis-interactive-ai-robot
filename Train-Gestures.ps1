param(
    [int]$Camera = 0,
    [int]$Samples = 180,
    [switch]$Auto
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Labels = @("thumbs_up", "peace", "stop", "heart", "middle_finger", "ok", "mohan")

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment is missing. Run .\Setup-Robot.ps1 first."
}

foreach ($Label in $Labels) {
    Write-Host "Capturing $Label" -ForegroundColor Cyan
    $Arguments = @("src\collect_samples.py", "--label", $Label, "--samples", $Samples, "--camera", $Camera)
    if ($Auto) {
        $Arguments += "--auto"
    }
    & $Python @Arguments
}

& $Python src\train.py
Write-Host "Training complete. Run .\Start-Robot.ps1" -ForegroundColor Green

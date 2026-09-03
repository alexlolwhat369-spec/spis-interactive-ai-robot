param(
    [int]$Camera = 0,
    [int]$Port = 8000,
    [switch]$NoOllama,
    [string]$OllamaModel = "spis-robot"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"

$Required = @(
    $Python,
    (Join-Path $Root "models\hand_landmarker.task"),
    (Join-Path $Root "model\gesture_knn.npz")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file is missing: $Path. Run .\Setup-Robot.ps1 first."
    }
}

# The React frontend must be built before Flask can serve it.
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Npm) {
    throw "Node.js/npm is required to build the web interface. Install Node.js and reopen PowerShell."
}
Push-Location -LiteralPath (Join-Path $Root "web")
try {
    $InstalledLock = Get-Item "node_modules/.package-lock.json" -ErrorAction SilentlyContinue
    $SourceLock = Get-Item "package-lock.json"
    if (-not $InstalledLock -or $SourceLock.LastWriteTimeUtc -gt $InstalledLock.LastWriteTimeUtc) {
        & $Npm.Source ci
        if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    }
    & $Npm.Source run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
} finally {
    Pop-Location
}

$Url = "http://127.0.0.1:$Port"
$Arguments = @(
    "src\web_app.py",
    "--camera", $Camera,
    "--host", "127.0.0.1",
    "--port", $Port,
    "--ollama-model", $OllamaModel
)
if ($NoOllama) {
    $Arguments += "--no-ollama"
}

Write-Host "Starting the complete robot at $Url" -ForegroundColor Cyan
Start-Job -ScriptBlock {
    param($Address)
    Start-Sleep -Seconds 2
    Start-Process $Address
} -ArgumentList $Url | Out-Null

& $Python @Arguments

param(
    [string]$VenvPath = "zulong_env",
    [switch]$SkipInstall,
    [switch]$FullDoctor
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

Write-Host "ZULONG Windows setup"
Write-Host "Root: $Root"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found in PATH. Install Python 3.10-3.12 first."
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment: $VenvPath"
    python -m venv $VenvPath
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found: $Python"
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip setuptools wheel
    Write-Host "Installing Windows requirements..."
    & $Python -m pip install -r requirements-windows.txt
}

Write-Host "Running doctor..."
if ($FullDoctor) {
    & $Python scripts\doctor.py --full
} else {
    & $Python scripts\doctor.py
}

Write-Host "Windows setup complete."

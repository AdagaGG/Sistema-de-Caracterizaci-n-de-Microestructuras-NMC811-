param(
    [string]$CondaExe = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($CondaExe)) {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if ($cmd) {
        $CondaExe = $cmd.Source
    } elseif (Test-Path "$env:USERPROFILE\miniconda3\Scripts\conda.exe") {
        $CondaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
    } elseif (Test-Path "$env:USERPROFILE\miniconda3\condabin\conda.bat") {
        $CondaExe = "$env:USERPROFILE\miniconda3\condabin\conda.bat"
    } else {
        throw "Conda executable not found. Provide -CondaExe explicitly."
    }
}

Set-Location (Resolve-Path "$PSScriptRoot\..")
$EnvName = "nmc811-segmentation"

Write-Host "Using conda executable: $CondaExe"
Write-Host "Using conda environment: $EnvName"

# Verify required tooling exists in the target conda environment.
& $CondaExe run -n $EnvName python -m ruff --version
& $CondaExe run -n $EnvName python -m mypy --version
& $CondaExe run -n $EnvName pytest --version

# Deterministic quality gates (single command runner).
& $CondaExe run -n $EnvName python -m ruff check src tests
& $CondaExe run -n $EnvName python -m mypy src tests
& $CondaExe run -n $EnvName pytest -q tests/test_mobilesam_inference.py tests/test_error_codes.py

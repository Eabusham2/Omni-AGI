param(
  [ValidateSet("cpu", "cu124", "directml")]
  [string]$Backend = "cpu",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$VenvRoot = Join-Path $RuntimeRoot "python"

Write-Host "Preparing OmniCortex runtime at $VenvRoot"
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

if (-not (Test-Path (Join-Path $VenvRoot "Scripts\\python.exe"))) {
  & $Python -m venv $VenvRoot
}

$RuntimePython = Join-Path $VenvRoot "Scripts\\python.exe"
& $RuntimePython -m pip install --upgrade pip

if ($Backend -eq "cu124") {
  & $RuntimePython -m pip install torch --index-url https://download.pytorch.org/whl/cu124
}
elseif ($Backend -eq "directml") {
  & $RuntimePython -m pip install torch-directml
}
else {
  & $RuntimePython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}

& $RuntimePython -m pip install -r (Join-Path $RepoRoot "engine\\requirements.txt")
& $RuntimePython -m unittest discover -s (Join-Path $RepoRoot "engine\\tests") -p "test_*.py" -v

Write-Host "OmniCortex runtime is ready."
Write-Host "Set OMNI_PYTHON=$RuntimePython when launching a development build."

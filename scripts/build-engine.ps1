param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Worker = Join-Path $RepoRoot "engine\\worker.py"
$DistRoot = Join-Path $RepoRoot "engine-dist"
$WorkRoot = Join-Path $RepoRoot ".engine-build"

if (-not (Test-Path $Worker)) {
  throw "OmniCortex worker not found at $Worker"
}

& $Python -m pip install --disable-pip-version-check "pyinstaller>=6.10,<7"
if ($LASTEXITCODE -ne 0) {
  throw "Installing PyInstaller failed with exit code $LASTEXITCODE"
}

if (Test-Path $DistRoot) {
  Remove-Item -Recurse -Force $DistRoot
}
if (Test-Path $WorkRoot) {
  Remove-Item -Recurse -Force $WorkRoot
}

& $Python -m PyInstaller `
  --noconfirm `
  --clean `
  --onedir `
  --name omni-engine `
  --distpath $DistRoot `
  --workpath $WorkRoot `
  --specpath $WorkRoot `
  --paths (Join-Path $RepoRoot "engine") `
  --collect-all torch `
  --collect-all safetensors `
  --collect-all imageio_ffmpeg `
  --collect-all soundfile `
  $Worker
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$Executable = Join-Path $DistRoot "omni-engine\\omni-engine.exe"
if (-not (Test-Path $Executable)) {
  throw "Engine packaging completed without producing $Executable"
}

Write-Host "Packaged OmniCortex worker: $Executable"

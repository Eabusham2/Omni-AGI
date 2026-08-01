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

if ($env:OMNI_SKIP_BUILD_DEPENDENCY_INSTALL -eq "1") {
  & $Python -c "import PyInstaller; major=int(PyInstaller.__version__.split('.')[0]); assert major == 6"
  if ($LASTEXITCODE -ne 0) {
    throw (
      "Protected runtime evolution requires an existing PyInstaller 6.x; " +
      "automatic dependency installation is disabled."
    )
  }
}
else {
  & $Python -m pip install --disable-pip-version-check "pyinstaller>=6.10,<7"
  if ($LASTEXITCODE -ne 0) {
    throw "Installing PyInstaller failed with exit code $LASTEXITCODE"
  }
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

# Retain only the distributable. The PyInstaller analysis cache is
# reproducible and can otherwise exhaust the 14 GB native runner while
# electron-builder creates both NSIS and ZIP outputs.
if (Test-Path $WorkRoot) {
  Remove-Item -Recurse -Force $WorkRoot
}

Write-Host "Packaged OmniCortex worker: $Executable"

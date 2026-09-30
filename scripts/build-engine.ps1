param(
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Worker = Join-Path $RepoRoot "engine\\worker.py"
$DistRoot = Join-Path $RepoRoot "engine-dist"
$WorkRoot = Join-Path $RepoRoot ".engine-build"
$LockInstaller = Join-Path $RepoRoot "scripts\install-engine-lock.py"

if (-not (Test-Path $Worker)) {
  throw "OmniCortex worker not found at $Worker"
}

if (-not (Test-Path $LockInstaller)) {
  throw "Engine dependency lock installer not found at $LockInstaller"
}

if ($env:OMNI_SKIP_BUILD_DEPENDENCY_INSTALL -eq "1") {
  & $Python $LockInstaller --verify-only
  if ($LASTEXITCODE -ne 0) {
    throw "The existing Python environment does not match the reviewed engine release lock."
  }
}
else {
  & $Python $LockInstaller
  if ($LASTEXITCODE -ne 0) {
    throw "Installing the reviewed engine release lock failed with exit code $LASTEXITCODE"
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
  --collect-all jsonschema `
  --collect-all jsonschema_specifications `
  --collect-all referencing `
  --collect-all attrs `
  --collect-all rpds `
  $Worker
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller failed with exit code $LASTEXITCODE"
}

# Keep imageio-ffmpeg's BSD wrapper, but do not convey the separately licensed
# wheel-provided FFmpeg executable without complete corresponding source and
# build material. The verifier also rejects a binary moved elsewhere.
Get-ChildItem -LiteralPath (Join-Path $DistRoot "omni-engine") -Recurse -File |
  Where-Object {
    $_.FullName -match '[\\/]imageio_ffmpeg[\\/]binaries[\\/]' -and
    $_.Name -match '^ffmpeg(?:[-.]|$)' -and
    $_.Extension -notin @('.py', '.pyc', '.pyo', '.md', '.txt')
  } |
  ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

& node (Join-Path $RepoRoot "scripts\verify-packaged-compliance.mjs") `
  --repo-root $RepoRoot `
  --engine-dir (Join-Path $DistRoot "omni-engine")
if ($LASTEXITCODE -ne 0) {
  throw "Packaged worker compliance verification failed with exit code $LASTEXITCODE"
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

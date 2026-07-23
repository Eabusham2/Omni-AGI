param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("x64", "arm64")]
  [string]$Arch
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$HostArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
if ($HostArch -ne $Arch) {
  throw (
    "A native $Arch Electron package requires a $Arch Windows host. " +
    "The worker follows the architecture of the selected Python runtime. Current host: $HostArch."
  )
}

Push-Location $RepoRoot
try {
  & npm.cmd run build
  if ($LASTEXITCODE -ne 0) {
    throw "Desktop build failed with exit code $LASTEXITCODE."
  }
  & npm.cmd run build:engine:win
  if ($LASTEXITCODE -ne 0) {
    throw "Engine build failed with exit code $LASTEXITCODE."
  }
  & npx.cmd electron-builder --win nsis zip "--$Arch" --publish never
  if ($LASTEXITCODE -ne 0) {
    throw "Windows packaging failed with exit code $LASTEXITCODE."
  }
}
finally {
  Pop-Location
}

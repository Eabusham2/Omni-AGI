param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("x64", "arm64")]
  [string]$Arch,
  [switch]$SkipBuild
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
  if (-not $SkipBuild.IsPresent) {
    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
      throw "Desktop build failed with exit code $LASTEXITCODE."
    }
    & npm.cmd run build:engine:win
    if ($LASTEXITCODE -ne 0) {
      throw "Engine build failed with exit code $LASTEXITCODE."
    }
  } else {
    $EngineExecutable = Join-Path $RepoRoot "engine-dist\\omni-engine\\omni-engine.exe"
    if (-not (Test-Path -LiteralPath $EngineExecutable -PathType Leaf)) {
      throw "-SkipBuild requires an existing worker at $EngineExecutable."
    }
  }

  $SigningLink = if ($env:WIN_CSC_LINK) {
    $env:WIN_CSC_LINK
  } else {
    $env:CSC_LINK
  }
  $SigningPassword = if ($env:WIN_CSC_KEY_PASSWORD) {
    $env:WIN_CSC_KEY_PASSWORD
  } else {
    $env:CSC_KEY_PASSWORD
  }
  if (
    [string]::IsNullOrWhiteSpace($SigningLink) -and
    -not [string]::IsNullOrWhiteSpace($SigningPassword)
  ) {
    throw "A signing password was provided without a Windows signing certificate."
  }
  if ([string]::IsNullOrWhiteSpace($SigningLink)) {
    # Avoid silently signing with an unrelated certificate installed on the
    # runner. Unsigned output is explicitly recorded as signed-ready evidence.
    $env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
  }
  $ForceCodeSigning = if ([string]::IsNullOrWhiteSpace($SigningLink)) {
    "false"
  } else {
    "true"
  }
  $BuilderArguments = @(
    "electron-builder",
    "--win",
    "nsis",
    "zip",
    "--$Arch",
    "--publish",
    "never",
    "--config.forceCodeSigning=$ForceCodeSigning"
  )
  & npx.cmd @BuilderArguments
  if ($LASTEXITCODE -ne 0) {
    throw "Windows packaging failed with exit code $LASTEXITCODE."
  }
}
finally {
  Pop-Location
}

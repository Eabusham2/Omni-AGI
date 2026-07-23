param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("x64", "arm64")]
  [string]$Arch,
  [Parameter(Mandatory = $true)]
  [ValidateSet("x64", "arm64")]
  [string]$ExpectedWorkerArch,
  [string]$ReleaseRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ReleaseRoot) {
  $ReleaseRoot = Join-Path $RepoRoot "release"
}
$ReleaseRoot = (Resolve-Path $ReleaseRoot).Path
$Scratch = Join-Path ([System.IO.Path]::GetTempPath()) (
  "omni-package-smoke-$Arch-" + [System.Guid]::NewGuid().ToString("N")
)
$ZipRoot = Join-Path $Scratch "zip"
$InstallRoot = Join-Path $Scratch "installed"
[System.IO.Directory]::CreateDirectory($Scratch) | Out-Null

function Get-OneArtifact {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Extension
  )
  $Matches = @(
    Get-ChildItem -Path $ReleaseRoot -File |
      Where-Object {
        $_.Name -like "*-Windows-$Arch.$Extension"
      }
  )
  if ($Matches.Count -ne 1) {
    throw "Expected one Windows $Arch .$Extension artifact, found $($Matches.Count)."
  }
  return $Matches[0]
}

function Get-PeArchitecture {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path
  )
  $Stream = [System.IO.File]::OpenRead($Path)
  $Reader = [System.IO.BinaryReader]::new($Stream)
  try {
    if ($Reader.ReadUInt16() -ne 0x5A4D) {
      throw "$Path is not a PE executable."
    }
    $Stream.Seek(0x3C, [System.IO.SeekOrigin]::Begin) | Out-Null
    $PeOffset = $Reader.ReadInt32()
    $Stream.Seek($PeOffset, [System.IO.SeekOrigin]::Begin) | Out-Null
    if ($Reader.ReadUInt32() -ne 0x00004550) {
      throw "$Path has an invalid PE signature."
    }
    $Machine = $Reader.ReadUInt16()
    switch ($Machine) {
      0x8664 { return "x64" }
      0xAA64 { return "arm64" }
      default { throw ("Unsupported PE machine 0x{0:X4} in {1}." -f $Machine, $Path) }
    }
  }
  finally {
    $Reader.Dispose()
  }
}

try {
  $Zip = Get-OneArtifact -Extension "zip"
  $Installer = Get-OneArtifact -Extension "exe"
  Expand-Archive -LiteralPath $Zip.FullName -DestinationPath $ZipRoot
  $ZipWorkers = @(
    Get-ChildItem -Path $ZipRoot -Recurse -File -Filter "omni-engine.exe" |
      Where-Object { $_.FullName -like "*engine-runtime*" }
  )
  if ($ZipWorkers.Count -ne 1) {
    throw "ZIP package must contain exactly one resources/engine-runtime/omni-engine.exe."
  }
  $ZipWorkerArch = Get-PeArchitecture -Path $ZipWorkers[0].FullName
  if ($ZipWorkerArch -ne $ExpectedWorkerArch) {
    throw "ZIP worker architecture $ZipWorkerArch does not match expected $ExpectedWorkerArch."
  }
  $ZipSmoke = & (Join-Path $PSScriptRoot "smoke-engine.ps1") `
    -Executable $ZipWorkers[0].FullName `
    -BrainRoot (Join-Path $Scratch "zip-brain")

  $Install = Start-Process `
    -FilePath $Installer.FullName `
    -ArgumentList @("/S", "/D=$InstallRoot") `
    -Wait `
    -PassThru
  if ($Install.ExitCode -ne 0) {
    throw "NSIS silent installation failed with exit code $($Install.ExitCode)."
  }
  $InstalledWorkers = @(
    Get-ChildItem -Path $InstallRoot -Recurse -File -Filter "omni-engine.exe" |
      Where-Object { $_.FullName -like "*engine-runtime*" }
  )
  if ($InstalledWorkers.Count -ne 1) {
    throw "Installed package must contain exactly one resources/engine-runtime/omni-engine.exe."
  }
  $InstalledWorkerArch = Get-PeArchitecture -Path $InstalledWorkers[0].FullName
  if ($InstalledWorkerArch -ne $ExpectedWorkerArch) {
    throw "Installed worker architecture $InstalledWorkerArch does not match expected $ExpectedWorkerArch."
  }
  $InstalledSmoke = & (Join-Path $PSScriptRoot "smoke-engine.ps1") `
    -Executable $InstalledWorkers[0].FullName `
    -BrainRoot (Join-Path $Scratch "installed-brain") `
    -Comprehensive

  $AppExecutables = @(
    Get-ChildItem -Path $InstallRoot -File -Filter "Omni AGI Studio.exe"
  )
  if ($AppExecutables.Count -ne 1) {
    throw "Installed package is missing Omni AGI Studio.exe."
  }
  $AppArch = Get-PeArchitecture -Path $AppExecutables[0].FullName
  if ($AppArch -ne $Arch) {
    throw "Installed app architecture $AppArch does not match package architecture $Arch."
  }

  $AppLaunched = $false
  $DesktopE2E = $false
  $HostArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
  if ($Arch -eq $HostArch) {
    $PreviousExecutable = $env:OMNI_E2E_EXECUTABLE
    try {
      $env:OMNI_E2E_EXECUTABLE = $AppExecutables[0].FullName
      Push-Location $RepoRoot
      try {
        & npm.cmd run test:ui:built
        if ($LASTEXITCODE -ne 0) {
          throw "Installed $Arch desktop end-to-end test failed with exit code $LASTEXITCODE."
        }
      }
      finally {
        Pop-Location
      }
      $AppLaunched = $true
      $DesktopE2E = $true
    }
    finally {
      $env:OMNI_E2E_EXECUTABLE = $PreviousExecutable
    }
  }

  $EvidencePath = Join-Path $ReleaseRoot "windows-package-smoke-$Arch.json"
  $LaunchSkipReason = if ($AppLaunched) {
    ""
  } else {
    "Package architecture $Arch does not match runner architecture $HostArch."
  }
  @{
    architecture = $Arch
    zip = @{
      name = $Zip.Name
      sha256 = (Get-FileHash -Algorithm SHA256 $Zip.FullName).Hash.ToLowerInvariant()
      packagedWorker = $ZipWorkers[0].FullName.Substring($ZipRoot.Length)
      packagedWorkerArchitecture = $ZipWorkerArch
      rpcSmoke = $ZipSmoke | ConvertFrom-Json
    }
    nsis = @{
      name = $Installer.Name
      sha256 = (Get-FileHash -Algorithm SHA256 $Installer.FullName).Hash.ToLowerInvariant()
      silentInstall = $true
      packagedWorker = $InstalledWorkers[0].FullName.Substring($InstallRoot.Length)
      packagedWorkerArchitecture = $InstalledWorkerArch
      desktopArchitecture = $AppArch
      rpcSmoke = $InstalledSmoke | ConvertFrom-Json
      desktopLaunch = $AppLaunched
      desktopEndToEnd = $DesktopE2E
      desktopRestart = $DesktopE2E
      accessibilityNavigation = $DesktopE2E
      modalityGeneration = $DesktopE2E
      desktopLaunchSkippedReason = $LaunchSkipReason
    }
  } | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $EvidencePath
  Write-Host "Windows package smoke evidence: $EvidencePath"
}
finally {
  if (Test-Path $Scratch) {
    Remove-Item -Recurse -Force $Scratch
  }
}

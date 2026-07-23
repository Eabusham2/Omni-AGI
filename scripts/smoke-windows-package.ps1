param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("x64", "arm64")]
  [string]$Arch,
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
$ProfileRoot = Join-Path $Scratch "profile"
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
  $InstalledSmoke = & (Join-Path $PSScriptRoot "smoke-engine.ps1") `
    -Executable $InstalledWorkers[0].FullName `
    -BrainRoot (Join-Path $Scratch "installed-brain")

  $AppExecutables = @(
    Get-ChildItem -Path $InstallRoot -File -Filter "Omni AGI Studio.exe"
  )
  if ($AppExecutables.Count -ne 1) {
    throw "Installed package is missing Omni AGI Studio.exe."
  }

  $AppLaunched = $false
  if ($Arch -eq "x64") {
    [System.IO.Directory]::CreateDirectory($ProfileRoot) | Out-Null
    $App = Start-Process `
      -FilePath $AppExecutables[0].FullName `
      -ArgumentList @("--disable-gpu", "--user-data-dir=$ProfileRoot") `
      -PassThru
    Start-Sleep -Seconds 8
    if ($App.HasExited) {
      throw "Installed x64 desktop app exited during startup with code $($App.ExitCode)."
    }
    $AppLaunched = $true
    & taskkill.exe /PID $App.Id /T /F | Out-Null
  }

  $EvidencePath = Join-Path $ReleaseRoot "windows-package-smoke-$Arch.json"
  $LaunchSkipReason = if ($AppLaunched) {
    ""
  } else {
    "ARM64 Electron executable cannot run on the hosted x64 Windows runner."
  }
  @{
    architecture = $Arch
    zip = @{
      name = $Zip.Name
      sha256 = (Get-FileHash -Algorithm SHA256 $Zip.FullName).Hash.ToLowerInvariant()
      packagedWorker = $ZipWorkers[0].FullName.Substring($ZipRoot.Length)
      rpcSmoke = $ZipSmoke | ConvertFrom-Json
    }
    nsis = @{
      name = $Installer.Name
      sha256 = (Get-FileHash -Algorithm SHA256 $Installer.FullName).Hash.ToLowerInvariant()
      silentInstall = $true
      packagedWorker = $InstalledWorkers[0].FullName.Substring($InstallRoot.Length)
      rpcSmoke = $InstalledSmoke | ConvertFrom-Json
      desktopLaunch = $AppLaunched
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

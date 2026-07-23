param(
  [Parameter(Mandatory = $true)]
  [string]$Executable,
  [string]$BrainRoot = ""
)

$ErrorActionPreference = "Stop"
$Executable = (Resolve-Path $Executable).Path
if (-not $BrainRoot) {
  $BrainRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    "omni-engine-smoke-" + [System.Guid]::NewGuid().ToString("N")
  )
}
[System.IO.Directory]::CreateDirectory($BrainRoot) | Out-Null

$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Executable
$StartInfo.WorkingDirectory = Split-Path -Parent $Executable
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.RedirectStandardInput = $true
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$StartInfo.Environment["PYTHONUNBUFFERED"] = "1"
$StartInfo.Environment["OMNI_PROTOCOL_VERSION"] = "1"

$Worker = [System.Diagnostics.Process]::new()
$Worker.StartInfo = $StartInfo
if (-not $Worker.Start()) {
  throw "Failed to start packaged OmniCortex worker at $Executable"
}

function Invoke-WorkerRpc {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Id,
    [Parameter(Mandatory = $true)]
    [string]$Method,
    [Parameter(Mandatory = $true)]
    [hashtable]$Params
  )

  $Request = @{
    jsonrpc = "2.0"
    id = $Id
    method = $Method
    params = $Params
  } | ConvertTo-Json -Compress -Depth 30
  $Worker.StandardInput.WriteLine($Request)
  $Worker.StandardInput.Flush()

  while ($true) {
    $ReadTask = $Worker.StandardOutput.ReadLineAsync()
    if (-not $ReadTask.Wait([TimeSpan]::FromMinutes(3))) {
      throw "Timed out waiting for packaged worker RPC $Method"
    }
    $Line = $ReadTask.Result
    if ($null -eq $Line) {
      throw "Packaged worker closed stdout during $Method. $($Worker.StandardError.ReadToEnd())"
    }
    $Message = $Line | ConvertFrom-Json
    if ($Message.id -ne $Id) {
      continue
    }
    if ($null -ne $Message.error) {
      throw "Packaged worker RPC $Method failed: $($Message.error.message)"
    }
    return $Message.result
  }
}

try {
  $Health = Invoke-WorkerRpc -Id "health" -Method "health" -Params @{}
  if (-not $Health.ready -or $Health.worker -ne "python" -or $Health.protocolVersion -ne 1) {
    throw "Packaged worker returned an invalid health response."
  }

  $Storage = Join-Path $BrainRoot "packaged-smoke-brain"
  $Created = Invoke-WorkerRpc -Id "create" -Method "create" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
    hardwareTier = "micro"
    config = @{
      name = "Packaged worker smoke"
      hardwareTier = "micro"
      parallelThoughts = 1
      image_enabled = $false
      audio_enabled = $false
      video_enabled = $false
      vision_enabled = $false
    }
  }
  if ($Created.brainId -ne "packaged-smoke-brain") {
    throw "Packaged worker did not create the requested brain identity."
  }

  $Unloaded = Invoke-WorkerRpc -Id "unload" -Method "unload" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
  }
  if (-not $Unloaded.unloaded) {
    throw "Packaged worker did not unload its created brain."
  }
  $Loaded = Invoke-WorkerRpc -Id "load" -Method "load" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
  }
  if ($Loaded.brainId -ne "packaged-smoke-brain") {
    throw "Packaged worker could not reload its safe-tensor checkpoint."
  }
  $State = Invoke-WorkerRpc -Id "state" -Method "state" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
  }
  if ($State.brainId -ne "packaged-smoke-brain") {
    throw "Packaged worker could not inspect its reloaded brain state."
  }
  foreach ($File in @("brain.json", "core.safetensors", "plasticity.safetensors", "events.sqlite3")) {
    if (-not (Test-Path (Join-Path (Join-Path $Storage "engine") $File))) {
      throw "Packaged worker smoke is missing engine/$File"
    }
  }

  $Shutdown = Invoke-WorkerRpc -Id "shutdown" -Method "shutdown" -Params @{}
  if (-not $Shutdown.stopping) {
    throw "Packaged worker did not acknowledge shutdown."
  }
  if (-not $Worker.WaitForExit(30000)) {
    throw "Packaged worker did not exit after shutdown."
  }

  @{
    executable = $Executable
    engineVersion = $Health.engineVersion
    protocolVersion = $Health.protocolVersion
    pythonVersion = $Health.pythonVersion
    torchVersion = $Health.torchVersion
    platform = $Health.platform
    persistedBrain = $true
    safeTensorCheckpoint = $true
    sqliteEventLog = $true
  } | ConvertTo-Json -Depth 10
}
finally {
  if (-not $Worker.HasExited) {
    $Worker.Kill($true)
    $Worker.WaitForExit()
  }
  $Worker.Dispose()
}

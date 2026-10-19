param(
  [Parameter(Mandatory = $true)]
  [string]$Executable,
  [string]$BrainRoot = "",
  [switch]$HealthOnly,
  [switch]$Comprehensive,
  [switch]$StdioProbe
)

$ErrorActionPreference = "Stop"
if ($HealthOnly -and $Comprehensive) {
  throw "-HealthOnly and -Comprehensive cannot be used together."
}
if ($StdioProbe -and -not $HealthOnly) {
  throw "-StdioProbe requires -HealthOnly and never creates a brain."
}
$Executable = (Resolve-Path $Executable).Path
if (-not $HealthOnly) {
  if (-not $BrainRoot) {
    $BrainRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
      "omni-engine-smoke-" + [System.Guid]::NewGuid().ToString("N")
    )
  }
  [System.IO.Directory]::CreateDirectory($BrainRoot) | Out-Null
}

$StartInfo = [System.Diagnostics.ProcessStartInfo]::new()
$StartInfo.FileName = $Executable
$StartInfo.WorkingDirectory = Split-Path -Parent $Executable
$StartInfo.UseShellExecute = $false
$StartInfo.CreateNoWindow = $true
$StartInfo.RedirectStandardInput = $true
$StartInfo.RedirectStandardOutput = $true
$StartInfo.RedirectStandardError = $true
$WireEncoding = [System.Text.UTF8Encoding]::new($false)
$StartInfo.StandardOutputEncoding = $WireEncoding
$StartInfo.StandardErrorEncoding = $WireEncoding
$InputEncodingProperty = $StartInfo.GetType().GetProperty("StandardInputEncoding")
if ($null -ne $InputEncodingProperty) {
  $StartInfo.StandardInputEncoding = $WireEncoding
}
$StartInfo.Environment["PYTHONUNBUFFERED"] = "1"
$StartInfo.Environment["OMNI_PROTOCOL_VERSION"] = "1"
if ($HealthOnly) {
  $StartInfo.Environment["OMNI_WORKER_STARTUP_DIAGNOSTICS"] = "1"
}

$Worker = [System.Diagnostics.Process]::new()
$Worker.StartInfo = $StartInfo
$RestoreInputEncoding = $null
try {
  if ($null -eq $InputEncodingProperty) {
    # Framework's Process creates an AutoFlush StreamWriter using the console
    # encoding. Suppress its preamble before creation, then restore the host.
    $PreviousInputEncoding = [Console]::InputEncoding
    if ($PreviousInputEncoding.GetPreamble().Length -gt 0) {
      [Console]::InputEncoding = $WireEncoding
      $RestoreInputEncoding = $PreviousInputEncoding
    }
  }
  if (-not $Worker.Start()) {
    throw "Failed to start packaged OmniCortex worker at $Executable"
  }
}
finally {
  if ($null -ne $RestoreInputEncoding) { [Console]::InputEncoding = $RestoreInputEncoding }
}
$WorkerErrorReadTask = $Worker.StandardError.ReadToEndAsync()

function Get-WorkerExitDiagnostic {
  # Observe only the exact launched child. Never emit its environment, command
  # credentials or unrelated processes, and never turn a crash into success.
  $Exited = $Worker.HasExited
  if (-not $Exited) { $Exited = $Worker.WaitForExit(5000) }
  $Code = "unavailable"
  $Hex = "unavailable"
  if ($Exited) {
    $Code = [string]$Worker.ExitCode
    $Hex = "0x" + [Convert]::ToString([int]$Worker.ExitCode, 16).ToUpperInvariant().PadLeft(8, '0')
  }
  $StderrComplete = $WorkerErrorReadTask.IsCompleted
  if (-not $StderrComplete -and $Exited) { $StderrComplete = $WorkerErrorReadTask.Wait(1000) }
  $DiagnosticError = ""
  if ($StderrComplete) {
    $DiagnosticError = $WorkerErrorReadTask.GetAwaiter().GetResult()
    if ($DiagnosticError.Length -gt 12000) { $DiagnosticError = $DiagnosticError.Substring($DiagnosticError.Length - 12000) }
  }
  return "workerExited=$Exited; exitCode=$Code; exitHex=$Hex; stderrComplete=$StderrComplete`n$DiagnosticError"
}

function Stop-OwnedWorkerForFailure {
  if ($Worker.HasExited) { return }
  # Kill(bool) is absent on Windows PowerShell's .NET Framework. Never fall
  # back to a process-name/tree search or let cleanup replace the RPC failure.
  $KillTreeMethod = $Worker.GetType().GetMethod("Kill", [type[]]@([bool]))
  if ($null -ne $KillTreeMethod) { $Worker.Kill($true) }
  else { $Worker.Kill() }
  if (-not $Worker.WaitForExit(5000)) {
    throw "Exact packaged worker did not exit during bounded failure cleanup."
  }
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
  # Write canonical UTF-8 bytes independently of Framework console codepages.
  $RequestBytes = $WireEncoding.GetBytes($Request + "`n")
  $Worker.StandardInput.BaseStream.Write($RequestBytes, 0, $RequestBytes.Length)
  $Worker.StandardInput.BaseStream.Flush()

  while ($true) {
    $ReadTask = $Worker.StandardOutput.ReadLineAsync()
    if (-not $ReadTask.Wait([TimeSpan]::FromMinutes(3))) {
      throw "Timed out waiting for packaged worker RPC $Method"
    }
    $Line = $ReadTask.Result
    if ($null -eq $Line) {
      throw "Packaged worker closed stdout during $Method. $(Get-WorkerExitDiagnostic)"
    }
    $Message = $Line | ConvertFrom-Json
    if ($Message.id -ne $Id -and $null -ne $Message.error) {
      throw "Packaged worker returned an uncorrelated protocol error during $Method."
    }
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
  if ($StdioProbe) {
    if (-not $Health.ready -or $Health.worker -ne "windows-stdio-probe" -or
        $Health.protocolVersion -ne 1 -or $Health.loadedBrains -ne 0) {
      throw "Frozen Windows stdio probe did not return its distinct no-brain identity."
    }
  }
  elseif (
    -not $Health.ready -or
    $Health.worker -ne "python" -or
    $Health.protocolVersion -ne 1 -or
    $Health.operatingSystem -ne "win32" -or
    [string]::IsNullOrWhiteSpace([string]$Health.engineVersion) -or
    [string]::IsNullOrWhiteSpace([string]$Health.pythonVersion) -or
    [string]::IsNullOrWhiteSpace([string]$Health.torchVersion) -or
    [string]::IsNullOrWhiteSpace([string]$Health.platform)
  ) {
    throw "Packaged worker returned an invalid health response."
  }

  $TrainingLossDecreased = $false
  $PdfIngested = $false
  $ChatParameterMutation = $false
  $GeneratedModalities = @()
  if (-not $HealthOnly) {
  $Storage = Join-Path $BrainRoot "packaged-smoke-brain"
  $EnableModalities = $Comprehensive.IsPresent
  $Created = Invoke-WorkerRpc -Id "create" -Method "create" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
    hardwareTier = "micro"
    config = @{
      name = "Packaged worker smoke"
      hardwareTier = "micro"
      image_enabled = $EnableModalities
      audio_enabled = $EnableModalities
      video_enabled = $EnableModalities
      vision_enabled = $EnableModalities
    }
  }
  if ($Created.brainId -ne "packaged-smoke-brain") {
    throw "Packaged worker did not create the requested brain identity."
  }

  if ($Comprehensive) {
    $Training = Invoke-WorkerRpc -Id "train" -Method "train" -Params @{
      brainId = "packaged-smoke-brain"
      storagePath = $Storage
      epochs = 4
      texts = @(
        "Cobalt memory pathways connect a durable installed cortex."
        "Cobalt memory pathways grow through repeated local experience."
      )
    }
    $TrainingLossDecreased = (
      $Training.promoted -eq $true -and
      [double]$Training.finalLoss -le [double]$Training.baselineLoss -and
      $Training.parameterChecksumBefore -ne $Training.parameterChecksumAfter
    )
    if (-not $TrainingLossDecreased) {
      throw "Installed worker corpus training did not promote a decreasing-loss parameter update."
    }

    $PdfPath = Join-Path $BrainRoot "installed-knowledge.pdf"
    $PdfBase64 = (
      "JVBERi0xLjMKJeLjz9MKMSAwIG9iago8PAovUHJvZHVjZXIgKHB5cGRmKQo+PgplbmRvYmoK" +
      "MiAwIG9iago8PAovVHlwZSAvUGFnZXMKL0NvdW50IDEKL0tpZHMgWyA0IDAgUiBdCj4+CmVu" +
      "ZG9iagozIDAgb2JqCjw8Ci9UeXBlIC9DYXRhbG9nCi9QYWdlcyAyIDAgUgo+PgplbmRvYmoK" +
      "NCAwIG9iago8PAovVHlwZSAvUGFnZQovUmVzb3VyY2VzIDw8Ci9Gb250IDw8Ci9GMSA1IDAg" +
      "Ugo+Pgo+PgovTWVkaWFCb3ggWyAwLjAgMC4wIDYxMiA3OTIgXQovUGFyZW50IDIgMCBSCi9D" +
      "b250ZW50cyA2IDAgUgo+PgplbmRvYmoKNSAwIG9iago8PAovVHlwZSAvRm9udAovU3VidHlw" +
      "ZSAvVHlwZTEKL0Jhc2VGb250IC9IZWx2ZXRpY2EKPj4KZW5kb2JqCjYgMCBvYmoKPDwKL0xl" +
      "bmd0aCA4NAo+PgpzdHJlYW0KQlQgL0YxIDEyIFRmIDcyIDcyMCBUZCAoUGRmU2VudGluZWwg" +
      "aW5zdGFsbGVkIHBhY2thZ2UgbGVhcm5zIGNvYmFsdCBnZW9tZXRyeS4pIFRqIEVUCmVuZHN0" +
      "cmVhbQplbmRvYmoKeHJlZgowIDcKMDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAw" +
      "MDAwIG4gCjAwMDAwMDAwNTQgMDAwMDAgbiAKMDAwMDAwMDExMyAwMDAwMCBuIAowMDAwMDAw" +
      "MTYyIDAwMDAwIG4gCjAwMDAwMDAyOTQgMDAwMDAgbiAKMDAwMDAwMDM2NCAwMDAwMCBuIAp0" +
      "cmFpbGVyCjw8Ci9TaXplIDcKL1Jvb3QgMyAwIFIKL0luZm8gMSAwIFIKPj4Kc3RhcnR4cmVm" +
      "CjQ5OAolJUVPRgo="
    )
    [System.IO.File]::WriteAllBytes(
      $PdfPath,
      [System.Convert]::FromBase64String($PdfBase64)
    )
    $Ingested = Invoke-WorkerRpc -Id "ingest-pdf" -Method "ingest" -Params @{
      brainId = "packaged-smoke-brain"
      storagePath = $Storage
      path = $PdfPath
      kind = "pdf"
      policy = "encode"
    }
    $PdfIngested = (
      $Ingested.duplicate -eq $false -and
      [int]$Ingested.source.learned_ideas -gt 0 -and
      $Ingested.parameterChecksumBefore -ne $Ingested.parameterChecksumAfter
    )
    if (-not $PdfIngested) {
      throw "Installed worker did not extract the PDF into learned ideas and parameters."
    }

    $Chat = Invoke-WorkerRpc -Id "chat" -Method "chat" -Params @{
      brainId = "packaged-smoke-brain"
      storagePath = $Storage
      input = "Trace how the installed cortex adapts to this experience."
      maxNewTokens = 4
      seed = 23
    }
    $ChatParameterMutation = (
      $Chat.trace.parameter_checksum_before -ne
      $Chat.trace.parameter_checksum_after
    )
    if (-not $ChatParameterMutation) {
      throw "Installed worker chat did not record a persistent parameter mutation."
    }

    foreach ($Modality in @("image", "audio", "video")) {
      $Generated = Invoke-WorkerRpc `
        -Id "generate-$Modality" `
        -Method "generate_modality" `
        -Params @{
          brainId = "packaged-smoke-brain"
          storagePath = $Storage
          modality = $Modality
          prompt = "Cobalt memory pathway"
          seed = 31
        }
      if (
        $Generated.modality -ne $Modality -or
        -not $Generated.path -or
        -not (Test-Path $Generated.path)
      ) {
        throw "Installed worker did not generate a persisted $Modality artifact."
      }
      $GeneratedModalities += $Modality
    }

    $Traces = Invoke-WorkerRpc -Id "traces" -Method "trace" -Params @{
      brainId = "packaged-smoke-brain"
      storagePath = $Storage
      limit = 10
    }
    if (@($Traces.traces).Count -lt 1) {
      throw "Installed worker did not expose its operational trace."
    }
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
  }

  $Shutdown = Invoke-WorkerRpc -Id "shutdown" -Method "shutdown" -Params @{}
  if (-not $Shutdown.stopping) {
    throw "Packaged worker did not acknowledge shutdown."
  }
  if (-not $Worker.WaitForExit(30000)) {
    throw "Packaged worker did not exit after shutdown."
  }
  if ($Worker.ExitCode -ne 0) {
    throw "Packaged worker did not exit cleanly. $(Get-WorkerExitDiagnostic)"
  }

  @{
    executable = $Executable
    engineVersion = $Health.engineVersion
    protocolVersion = $Health.protocolVersion
    pythonVersion = $Health.pythonVersion
    torchVersion = $Health.torchVersion
    platform = $Health.platform
    operatingSystem = $Health.operatingSystem
    healthOnly = $HealthOnly.IsPresent
    transportProbeOnly = $StdioProbe.IsPresent
    persistedBrain = (-not $HealthOnly.IsPresent)
    safeTensorCheckpoint = (-not $HealthOnly.IsPresent)
    sqliteEventLog = (-not $HealthOnly.IsPresent)
    comprehensive = $Comprehensive.IsPresent
    trainingLossDecreased = $TrainingLossDecreased
    pdfIngested = $PdfIngested
    chatParameterMutation = $ChatParameterMutation
    generatedModalities = $GeneratedModalities
  } | ConvertTo-Json -Depth 10
}
catch {
  $PrimaryFailure = $_
  $FailureCleanupAttempted = -not $Worker.HasExited
  $CleanupFailureClass = "none"
  try { Stop-OwnedWorkerForFailure }
  catch { $CleanupFailureClass = $_.Exception.GetType().Name }
  try { $FailureDiagnostic = Get-WorkerExitDiagnostic }
  catch { $FailureDiagnostic = "diagnosticFailureClass=" + $_.Exception.GetType().Name }
  Write-Host "Packaged worker failure diagnostics: failureCleanupAttempted=$FailureCleanupAttempted; cleanupFailureClass=$CleanupFailureClass`n$FailureDiagnostic"
  throw $PrimaryFailure
}
finally {
  $Worker.Dispose()
}

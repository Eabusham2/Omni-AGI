param(
  [Parameter(Mandatory = $true)]
  [string]$Executable,
  [string]$BrainRoot = "",
  [switch]$Comprehensive
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
  $EnableModalities = $Comprehensive.IsPresent
  $Created = Invoke-WorkerRpc -Id "create" -Method "create" -Params @{
    brainId = "packaged-smoke-brain"
    storagePath = $Storage
    hardwareTier = "micro"
    config = @{
      name = "Packaged worker smoke"
      hardwareTier = "micro"
      parallelThoughts = 1
      image_enabled = $EnableModalities
      audio_enabled = $EnableModalities
      video_enabled = $EnableModalities
      vision_enabled = $EnableModalities
    }
  }
  if ($Created.brainId -ne "packaged-smoke-brain") {
    throw "Packaged worker did not create the requested brain identity."
  }

  $TrainingLossDecreased = $false
  $PdfIngested = $false
  $ChatParameterMutation = $false
  $GeneratedModalities = @()
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
    comprehensive = $Comprehensive.IsPresent
    trainingLossDecreased = $TrainingLossDecreased
    pdfIngested = $PdfIngested
    chatParameterMutation = $ChatParameterMutation
    generatedModalities = $GeneratedModalities
  } | ConvertTo-Json -Depth 10
}
finally {
  if (-not $Worker.HasExited) {
    $Worker.Kill($true)
    $Worker.WaitForExit()
  }
  $Worker.Dispose()
}

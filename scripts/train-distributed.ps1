param(
  [Parameter(Mandatory = $true)][string]$Dataset,
  [Parameter(Mandatory = $true)][string]$Output,
  [int]$Processes = 0,
  [string]$Python = "python",
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$TrainingArgs
)

$ErrorActionPreference = "Stop"
if ($Processes -le 0) {
  $gpuCount = & $Python -c "import torch; print(torch.cuda.device_count())"
  if ($LASTEXITCODE -ne 0) { throw "Python/PyTorch CUDA probe failed" }
  $Processes = [Math]::Max(1, [int]$gpuCount)
}

$env:PYTHONPATH = if ($env:PYTHONPATH) { "engine;$env:PYTHONPATH" } else { "engine" }
$arguments = @(
  "-m", "torch.distributed.run",
  "--standalone",
  "--nproc-per-node", "$Processes",
  "engine/distributed_train.py", "train",
  "--dataset", "$Dataset",
  "--output", "$Output"
) + $TrainingArgs

& $Python @arguments
exit $LASTEXITCODE

$ErrorActionPreference = "Stop"

$root = "E:\AI project"
$paths = @{
  RTXLLM_STORAGE_ROOT = $root
  RTXLLM_MODEL_DIR = Join-Path $root "models"
  RTXLLM_BENCHMARK_DIR = Join-Path $root "benchmarks"
  RTXLLM_TRACE_DIR = Join-Path $root "traces"
  HF_HOME = Join-Path $root "hf-cache"
  HF_HUB_CACHE = Join-Path $root "hf-cache\hub"
  TRANSFORMERS_CACHE = Join-Path $root "hf-cache\transformers"
  HF_XET_CACHE = Join-Path $root "hf-cache\xet"
  XDG_CACHE_HOME = Join-Path $root "tmp\xdg-cache"
}

foreach ($value in $paths.Values) {
  New-Item -ItemType Directory -Force -Path $value | Out-Null
}

foreach ($entry in $paths.GetEnumerator()) {
  Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
}

$env:TMP = Join-Path $root "tmp"
$env:TEMP = Join-Path $root "tmp"

$venvPython = Join-Path $root "venvs\rtxllm-py310\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
  $env:RTXLLM_PYTHON = $venvPython
  $venvScripts = Split-Path -Parent $venvPython
  if (($env:PATH -split ';') -notcontains $venvScripts) {
    $env:PATH = "$venvScripts;$env:PATH"
  }
}

Write-Host "RTXLLM storage root: $env:RTXLLM_STORAGE_ROOT"
Write-Host "HF cache: $env:HF_HOME"
Write-Host "Benchmarks: $env:RTXLLM_BENCHMARK_DIR"
if ($env:RTXLLM_PYTHON) {
  Write-Host "Python: $env:RTXLLM_PYTHON"
}

param(
  [string]$Python = "C:\Users\koufa\AppData\Local\Programs\Python\Python310\python.exe",
  [string]$StorageRoot = "E:\AI project"
)

$ErrorActionPreference = "Stop"

$venv = Join-Path $StorageRoot "venvs\rtxllm-py310"
$env:PIP_CACHE_DIR = Join-Path $StorageRoot "tmp\pip-cache"
New-Item -ItemType Directory -Force -Path $env:PIP_CACHE_DIR | Out-Null

if (-not (Test-Path -LiteralPath $venv)) {
  & $Python -m venv $venv
}

$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r "$PSScriptRoot\..\requirements-bench.txt"

Write-Host "Python environment ready: $venvPython"

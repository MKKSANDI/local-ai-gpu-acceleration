$ErrorActionPreference = "Continue"

Write-Host "== Storage =="
if (-not $env:RTXLLM_STORAGE_ROOT) {
  Write-Host "RTXLLM_STORAGE_ROOT is not set. Run: . .\scripts\env.ps1"
} else {
  Write-Host "RTXLLM_STORAGE_ROOT=$env:RTXLLM_STORAGE_ROOT"
}
Get-PSDrive -Name C,E | Select-Object Name,Free,Used,Root | Format-Table

Write-Host "`n== GPU =="
where.exe nvidia-smi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv,noheader

Write-Host "`n== CUDA =="
where.exe nvcc
nvcc --version
Get-ChildItem Env:CUDA_PATH* | Format-Table -AutoSize

Write-Host "`n== Native compiler =="
where.exe cl
if ($LASTEXITCODE -ne 0) {
  Write-Host "cl.exe is not visible in this shell."
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path -LiteralPath $vswhere) {
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($install) {
      $vcvars = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
      Write-Host "Visual C++ tools found at: $install"
      Write-Host "Use: .\scripts\build_native_vs.ps1"
      Write-Host "Developer env: $vcvars"
    } else {
      Write-Host "Visual Studio installation found, but VC++ x64 tools were not reported by vswhere."
    }
  } else {
    Write-Host "vswhere.exe not found. Install/enable Visual Studio C++ build tools for native C++/CUDA builds."
  }
}
where.exe cmake
cmake --version
where.exe ninja

Write-Host "`n== Python CUDA smoke =="
$python = if ($env:RTXLLM_PYTHON -and (Test-Path -LiteralPath $env:RTXLLM_PYTHON)) {
  $env:RTXLLM_PYTHON
} else {
  "python"
}
Write-Host "python=$python"
@'
import sys
from numba import cuda
print("python_version=", sys.version.split()[0])
print("numba_cuda_available=", cuda.is_available())
if cuda.is_available():
    cuda.detect()
'@ | & $python -

Write-Host "`n== Hugging Face =="
where.exe hf
if ($LASTEXITCODE -ne 0) {
  Write-Host "hf CLI not found; Python huggingface_hub will be used by project scripts."
} else {
  hf version
}

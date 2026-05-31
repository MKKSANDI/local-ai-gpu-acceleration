param(
  [string]$Preset = "windows-vs2022-cuda86-release",
  [string]$Config = "Release",
  [switch]$SkipConfigure
)

$ErrorActionPreference = "Stop"

function Find-VcVars64 {
  $candidates = @()
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (Test-Path -LiteralPath $vswhere) {
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($install) {
      $candidates += (Join-Path $install "VC\Auxiliary\Build\vcvars64.bat")
    }
  }

  $candidates += @(
    "D:\Visual build tools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
  )

  foreach ($candidate in $candidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
      return $candidate
    }
  }

  throw "Could not find vcvars64.bat. Install Visual Studio C++ build tools."
}

$vcvars = Find-VcVars64
Write-Host "Using MSVC environment: $vcvars"

$steps = @()
if (-not $SkipConfigure) {
  $steps += "cmake --preset $Preset"
}
$steps += "cmake --build --preset $Preset --config $Config"

$cmd = "call `"$vcvars`" && " + ($steps -join " && ")
cmd /d /s /c $cmd
if ($LASTEXITCODE -ne 0) {
  throw "Native build failed with exit code $LASTEXITCODE"
}

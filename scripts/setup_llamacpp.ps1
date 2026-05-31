param(
  [string]$StorageRoot = "E:\AI project",
  [string]$CudaVersion = "12.4",
  [string]$ReleaseTag = "latest"
)

$ErrorActionPreference = "Stop"

. "$PSScriptRoot\env.ps1"

$installRoot = Join-Path $StorageRoot "third_party\llama.cpp"
$archiveRoot = Join-Path $StorageRoot "third_party\archives"
New-Item -ItemType Directory -Force -Path $installRoot | Out-Null
New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

$headers = @{ "User-Agent" = "local-ai-gpu-runtime" }
if ($ReleaseTag -eq "latest") {
  $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -Headers $headers
} else {
  $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$ReleaseTag" -Headers $headers
}

$tag = $release.tag_name
$binName = "llama-$tag-bin-win-cuda-$CudaVersion-x64.zip"
$cudartName = "cudart-llama-bin-win-cuda-$CudaVersion-x64.zip"

$assets = @{}
foreach ($asset in $release.assets) {
  $assets[$asset.name] = $asset
}

foreach ($name in @($binName, $cudartName)) {
  if (-not $assets.ContainsKey($name)) {
    throw "Missing llama.cpp release asset: $name"
  }
  $asset = $assets[$name]
  $archive = Join-Path $archiveRoot $name
  if (-not (Test-Path -LiteralPath $archive) -or ((Get-Item -LiteralPath $archive).Length -ne $asset.size)) {
    Write-Host "Downloading $name to $archive"
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive -Headers $headers
  } else {
    Write-Host "Using existing archive $archive"
  }
  Write-Host "Extracting $name"
  Expand-Archive -LiteralPath $archive -DestinationPath $installRoot -Force
}

$manifest = [ordered]@{
  repo = "ggml-org/llama.cpp"
  tag = $tag
  published_at = $release.published_at
  html_url = $release.html_url
  cuda_version = $CudaVersion
  install_root = $installRoot
  assets = @($binName, $cudartName)
  generated_at = (Get-Date).ToUniversalTime().ToString("o")
}

$manifestPath = Join-Path $installRoot "rtxllm-llamacpp-install.json"
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding UTF8

Write-Host "llama.cpp installed at $installRoot"
Write-Host "manifest=$manifestPath"
Get-ChildItem -LiteralPath $installRoot -Recurse -Filter "llama*.exe" | Select-Object FullName

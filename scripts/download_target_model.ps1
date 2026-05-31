param(
  [string]$RepoId = "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
  [string]$Filename = "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
  [string]$StorageRoot = "E:\AI project"
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1"

$modelDir = Join-Path $StorageRoot "models"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

@"
from huggingface_hub import hf_hub_download
from pathlib import Path
repo_id = r"$RepoId"
filename = r"$Filename"
local_dir = Path(r"$modelDir")
path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    resume_download=True,
)
print(path)
"@ | python -

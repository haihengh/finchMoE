import os
import sys
import time
from huggingface_hub import snapshot_download

repo_id = "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit"
local_dir = "models/Qwen3.8-Flash-Next-125B.finch"

print(f"Downloading {repo_id} to {local_dir}...")
start_time = time.time()

try:
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        max_workers=16,
    )
    elapsed = time.time() - start_time
    print(f"\nDownload completed successfully in {elapsed:.1f}s ({elapsed/60:.2f} mins) to {path}")
except Exception as e:
    print(f"\nDownload failed with error: {e}", file=sys.stderr)
    sys.exit(1)

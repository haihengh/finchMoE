import os
import sys
import time
import json
import urllib.request
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ID = "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit"
LOCAL_DIR = "models/Qwen3.8-Flash-Next-125B-ple4bit.finch"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"

def get_repo_tree():
    url = f"https://huggingface.co/api/models/{REPO_ID}/tree/main?recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

class ProgressTracker:
    def __init__(self, total_bytes, total_files):
        self.total_bytes = total_bytes
        self.total_files = total_files
        self.downloaded_bytes = 0
        self.completed_files = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.last_print = 0

    def add_bytes(self, n):
        with self.lock:
            self.downloaded_bytes += n
            now = time.time()
            if now - self.last_print >= 1.5:
                self.print_progress()
                self.last_print = now

    def finish_file(self, path):
        with self.lock:
            self.completed_files += 1
            self.print_progress(force=True)

    def print_progress(self, force=False):
        elapsed = max(0.001, time.time() - self.start_time)
        speed_mb = (self.downloaded_bytes / (1024 * 1024)) / elapsed
        pct = (self.downloaded_bytes / max(1, self.total_bytes)) * 100
        cur_gb = self.downloaded_bytes / (1024**3)
        tot_gb = self.total_bytes / (1024**3)
        rem_sec = (self.total_bytes - self.downloaded_bytes) / (max(0.1, speed_mb) * 1024 * 1024)
        rem_min = rem_sec / 60
        sys.stdout.write(
            f"\r[{self.completed_files}/{self.total_files} files] "
            f"{cur_gb:.2f}/{tot_gb:.2f} GB ({pct:.1f}%) | "
            f"{speed_mb:.1f} MB/s | ETA: {rem_min:.1f} min   "
        )
        sys.stdout.flush()

def download_file(item, tracker):
    rel_path = item["path"]
    size = item.get("size", 0)
    dest_path = os.path.join(LOCAL_DIR, rel_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check if already complete
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == size and size > 0:
        tracker.add_bytes(size)
        tracker.finish_file(rel_path)
        return

    file_url = f"{BASE_URL}/{rel_path}"
    max_retries = 20

    for attempt in range(max_retries):
        cur_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if cur_size > size:
            try:
                os.remove(dest_path)
            except OSError:
                pass
            cur_size = 0

        # Account for bytes already on disk before starting download
        if attempt == 0 and cur_size > 0:
            tracker.add_bytes(cur_size)

        try:
            req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
            if cur_size > 0:
                req.add_header("Range", f"bytes={cur_size}-")

            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if cur_size > 0 and status == 200:
                    mode = "wb"
                else:
                    mode = "ab" if cur_size > 0 else "wb"

                with open(dest_path, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024) # 1MB chunk
                        if not chunk:
                            break
                        f.write(chunk)
                        tracker.add_bytes(len(chunk))

            # Verify size if known
            if size > 0 and os.path.getsize(dest_path) != size:
                time.sleep(1)
                continue

            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download {rel_path} after {max_retries} attempts: {e}")
            time.sleep(min(10, 1 + attempt))

    tracker.finish_file(rel_path)

def main():
    print(f"Fetching file list for {REPO_ID}...")
    tree = get_repo_tree()
    files = [item for item in tree if item.get("type") == "file"]
    total_bytes = sum(item.get("size", 0) for item in files)
    print(f"Found {len(files)} files, total size: {total_bytes / (1024**3):.2f} GB")

    tracker = ProgressTracker(total_bytes, len(files))

    # Download with 12 concurrent worker threads
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_file, f, tracker): f for f in files}
        for future in as_completed(futures):
            future.result()

    print("\n\nAll files downloaded successfully!")

if __name__ == "__main__":
    main()

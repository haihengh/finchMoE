import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ID = "haihengh/Qwen3.8-Flash-Next-125B-finch-4bit-ple4bit"
LOCAL_DIR = "models/Qwen3.8-Flash-Next-125B-ple4bit.finch"
BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main"
TREE_URL = f"https://huggingface.co/api/models/{REPO_ID}/tree/main?recursive=true"
WORKERS = 8
CHUNK_SIZE = 4 * 1024 * 1024
MAX_RETRIES = 50


def request(url, headers=None, timeout=120):
    merged = {"User-Agent": "finchmoe-resume-downloader/1.0"}
    if headers:
        merged.update(headers)
    return urllib.request.urlopen(urllib.request.Request(url, headers=merged), timeout=timeout)


def repo_files():
    with request(TREE_URL) as response:
        tree = json.loads(response.read().decode("utf-8"))
    return [item for item in tree if item.get("type") == "file"]


def local_size(path):
    try:
        return os.path.getsize(path)
    except FileNotFoundError:
        return 0


def audit(files):
    complete = []
    pending = []
    oversized = []
    total = 0
    present = 0
    for item in files:
        rel_path = item["path"]
        expected = item.get("size", 0)
        total += expected
        destination = os.path.join(LOCAL_DIR, rel_path)
        current = local_size(destination)
        if expected and current == expected:
            complete.append(item)
            present += expected
        else:
            if expected and current > expected:
                oversized.append((rel_path, current, expected))
            else:
                present += min(current, expected) if expected else current
            pending.append(item)
    return complete, pending, oversized, present, total


def progress_line(done_files, total_files, present, total, start_time):
    elapsed = max(time.time() - start_time, 0.001)
    rate = present / elapsed
    remain = max(total - present, 0)
    eta = remain / rate if rate > 1 else 0
    return (
        f"{done_files}/{total_files} files | "
        f"{present / 1024**3:.2f}/{total / 1024**3:.2f} GiB "
        f"({present / total * 100:.2f}%) | "
        f"{rate / 1024**2:.2f} MiB/s | ETA {eta / 60:.1f} min"
    )


def download_one(item, total_files, state):
    rel_path = item["path"]
    expected = item.get("size", 0)
    destination = os.path.join(LOCAL_DIR, rel_path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        current = local_size(destination)
        if expected and current == expected:
            return rel_path
        if expected and current > expected:
            os.remove(destination)
            current = 0

        headers = {}
        mode = "wb"
        if current > 0:
            headers["Range"] = f"bytes={current}-"
            mode = "ab"

        try:
            with request(f"{BASE_URL}/{rel_path}", headers=headers) as response:
                status = getattr(response, "status", 200)
                if current > 0 and status == 200:
                    mode = "wb"
                    current = 0
                with open(destination, mode) as output:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        state["present"] += len(chunk)
                        now = time.time()
                        if now - state["last_print"] > 5:
                            print("\r" + progress_line(state["done"], total_files, state["present"], state["total"], state["start"]), end="", flush=True)
                            state["last_print"] = now
            if not expected or local_size(destination) == expected:
                return rel_path
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"failed {rel_path} after {MAX_RETRIES} attempts: {error}") from error
            time.sleep(min(30, attempt))
    return rel_path


def main():
    print(f"Auditing {REPO_ID} into {LOCAL_DIR}")
    files = repo_files()
    complete, pending, oversized, present, total = audit(files)
    print(f"Expected: {len(files)} files, {total / 1024**3:.2f} GiB")
    print(f"Complete: {len(complete)} files")
    print(f"Pending: {len(pending)} files")
    print(f"Present: {present / 1024**3:.2f} GiB")
    if oversized:
        print("Oversized files will be redownloaded:")
        for rel_path, current, expected in oversized[:20]:
            print(f"  {rel_path}: {current} > {expected}")

    if not pending:
        print("Download already complete.")
        return

    state = {
        "present": present,
        "total": total,
        "done": len(complete),
        "start": time.time(),
        "last_print": 0.0,
    }
    print("Resuming pending files...")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(download_one, item, len(files), state) for item in pending]
        for future in as_completed(futures):
            future.result()
            state["done"] += 1
            print("\r" + progress_line(state["done"], len(files), state["present"], total, state["start"]), end="", flush=True)
    print()

    complete, pending, oversized, present, total = audit(files)
    print(f"Final complete: {len(complete)}/{len(files)} files")
    print(f"Final present: {present / 1024**3:.2f}/{total / 1024**3:.2f} GiB")
    if pending:
        print("Still pending:")
        for item in pending:
            rel_path = item["path"]
            expected = item.get("size", 0)
            current = local_size(os.path.join(LOCAL_DIR, rel_path))
            print(f"  {rel_path}: {current} / {expected}")
        sys.exit(1)
    print("Download complete.")


if __name__ == "__main__":
    main()

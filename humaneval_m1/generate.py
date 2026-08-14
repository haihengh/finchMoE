#!/usr/bin/env python3
"""generate.py — HumanEval generation against the local finchmoe-infer server.

Resumable: results append to he_results.jsonl; completed task_ids are skipped
on restart. Usage:
    python3 generate.py [--limit N] [--port 9000]
"""
import gzip, json, sys, time, urllib.request, argparse

BASE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
DATASET = f"{BASE}/HumanEval.jsonl.gz"
RESULTS = f"{BASE}/he_results.jsonl"

def load_dataset():
    rows = [json.loads(l) for l in gzip.open(DATASET, "rt")]
    return {r["task_id"]: r for r in rows}

def done_tasks():
    # Only skip tasks that actually have a completion — error records retry.
    try:
        return {json.loads(l)["task_id"] for l in open(RESULTS)
                if "completion" in json.loads(l)}
    except FileNotFoundError:
        return set()

def complete(port, prompt, max_tokens=512, timeout=900):
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    # Server always streams SSE: `data: {json}` lines, terminated by [DONE].
    parts = []
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        parts.append((chunk.get("choices") or [{}])[0].get("text") or "")
    return "".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all)")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args()

    tasks = load_dataset()
    done = done_tasks()
    todo = [t for t in tasks if t not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[generate] {len(todo)} problems to run ({len(done)} already done)", flush=True)
    out = open(RESULTS, "a")
    for i, task_id in enumerate(todo):
        t0 = time.time()
        rec = {"task_id": task_id}
        try:
            rec["completion"] = complete(args.port, tasks[task_id]["prompt"])
        except Exception as e:
            rec["error"] = str(e)
        rec["seconds"] = round(time.time() - t0, 1)
        out.write(json.dumps(rec) + "\n")
        out.flush()
        n_tok = len(rec.get("completion", "").split())
        print(f"[{i+1}/{len(todo)}] {task_id}  {rec['seconds']}s  ~{n_tok} tok"
              + (f"  ERROR: {rec['error']}" if "error" in rec else ""), flush=True)
    out.close()
    print("[generate] done", flush=True)

if __name__ == "__main__":
    main()

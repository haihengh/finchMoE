#!/usr/bin/env python3
"""generate.py — HumanEval generation against the local finchmoe-infer server.

Resumable: results append to the results file; completed task_ids are skipped
on restart.

Modes:
  default (raw):  POST /v1/completions with the bare prompt — the original
                  harness mode (raw code continuation, no template).
  --chat:         POST /v1/chat/completions with the prompt as a user message
                  (full chat template + system prompt). The completion is
                  post-processed: <think>...</think> blocks are stripped and
                  a markdown fenced code block is extracted when present.
                  This matches the standard instruct-model eval protocol.

Usage:
    python3 generate.py [--limit N] [--port 9000] [--chat] [--results FILE]
"""
import gzip, json, re, sys, time, urllib.request, argparse

BASE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
DATASET = f"{BASE}/HumanEval.jsonl.gz"
RESULTS = f"{BASE}/he_results.jsonl"


def load_dataset():
    rows = [json.loads(l) for l in gzip.open(DATASET, "rt")]
    return {r["task_id"]: r for r in rows}


def done_tasks(results_path):
    # Only skip tasks that actually have a completion — error records retry.
    try:
        return {json.loads(l)["task_id"] for l in open(results_path)
                if "completion" in json.loads(l)}
    except FileNotFoundError:
        return set()


def sse_collect(raw, field):
    """Parse the SSE stream; field = 'text' (completions) or 'content' (chat)."""
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
        if field == "content":
            delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
            parts.append(delta.get("content") or "")
        else:
            parts.append((chunk.get("choices") or [{}])[0].get("text") or "")
    return "".join(parts)


def strip_think(text):
    """Remove <think>...</think> blocks (unclosed: cut from <think>)."""
    out = []
    rest = text
    while True:
        i = rest.find("<think>")
        if i < 0:
            out.append(rest)
            break
        out.append(rest[:i])
        rest = rest[i + len("<think>"):]
        j = rest.find("</think>")
        if j < 0:
            rest = ""   # unclosed think — drop the remainder
            break
        rest = rest[j + len("</think>"):]
    return "".join(out)


def extract_code(text):
    """Extract the first markdown fenced code block when present."""
    fences = [m for m in re.finditer(r"```", text)]
    if len(fences) >= 2:
        body = text[fences[0].end():fences[1].start()]
        # drop a language tag on the opening fence line
        body = re.sub(r"^[a-zA-Z0-9_+-]*\s*\n?", "", body, count=1)
        return body
    return text


def complete(port, prompt, max_tokens=512, timeout=1200, chat=False):
    if chat:
        # Chat mode generates a think block before the code — the token
        # budget must cover both (512 left think ~350 + code ~160, cutting
        # completions mid-body).
        max_tokens = max(max_tokens, 1536)
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }).encode()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        field = "content"
    else:
        body = json.dumps({"prompt": prompt, "max_tokens": max_tokens,
                           "temperature": 0}).encode()
        url = f"http://127.0.0.1:{port}/v1/completions"
        field = "text"
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
    text = sse_collect(raw, field)
    if chat:
        text = strip_think(text)
        text = extract_code(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max problems (0 = all)")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--chat", action="store_true",
                    help="chat-templated mode (template + think, stripped)")
    ap.add_argument("--results", default=RESULTS, help="results file")
    args = ap.parse_args()
    results_path = args.results

    tasks = load_dataset()
    done = done_tasks(results_path)
    todo = [t for t in tasks if t not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[generate] {len(todo)} problems to run ({len(done)} already done) "
          f"mode={'chat' if args.chat else 'raw'}", flush=True)
    out = open(results_path, "a")
    for i, task_id in enumerate(todo):
        t0 = time.time()
        rec = {"task_id": task_id}
        try:
            rec["completion"] = complete(args.port, tasks[task_id]["prompt"],
                                         chat=args.chat)
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

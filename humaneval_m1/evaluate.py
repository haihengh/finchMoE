#!/usr/bin/env python3
"""evaluate.py — pass@1 evaluation of he_results.jsonl against the official
HumanEval tests. Standard truncation: cut the completion at the first
top-level def/class/if __name__/print after the prompt, then exec + check().
Usage:
    python3 evaluate.py [--verbose]
"""
import gzip, json, os, sys, argparse, traceback, re

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = f"{BASE}/HumanEval.jsonl.gz"
RESULTS = f"{BASE}/he_results.jsonl"

CUT_MARKERS = ["\nclass ", "\ndef ", "\nif __name__", "\nprint("]

def load_dataset():
    return {json.loads(l)["task_id"]: json.loads(l) for l in gzip.open(DATASET, "rt")}

def load_results(results_path):
    out = {}
    try:
        for l in open(results_path):
            r = json.loads(l)
            out[r["task_id"]] = r
    except FileNotFoundError:
        pass
    return out

def truncate_completion(completion):
    pos = len(completion)
    for m in CUT_MARKERS:
        i = completion.find(m)
        if i != -1:
            pos = min(pos, i)
    return completion[:pos].rstrip()

def build_code(task, completion):
    # Chat-templated models often restate the WHOLE function (imports +
    # signature + docstring). Concatenating that onto the prompt duplicates
    # the header and breaks the parse. If the completion is a complete
    # program whose first function is the entry point, use it alone — but
    # keep the prompt's imports (the model may omit them) and only apply the
    # cut markers AFTER the entry function's own header.
    entry = re.escape(task["entry_point"])
    if re.match(rf'(?s)\s*(?:(?:from\s+\S+\s+import\s+[^\n]+|import\s+[^\n]+)\n)*\s*def\s+{entry}\s*\(',
                completion):
        imports = "\n".join(l for l in task["prompt"].splitlines()
                             if l.startswith(("from ", "import ")))
        defpos = completion.find(f"def {task['entry_point']}(")
        head, tail = completion[:defpos], completion[defpos:]
        for m in ["\nclass ", "\nif __name__", "\nprint("]:
            i = tail.find(m)
            if i != -1:
                tail = tail[:i]
        return ((imports + "\n") if imports else "") + head + tail.rstrip()
    return task["prompt"] + truncate_completion(completion)


def run_check(task, completion, timeout_sec=10):
    code = build_code(task, completion)
    ns = {}
    try:
        exec(compile(code, f"<{task['task_id']}>", "exec"), ns)
        test = task["test"]
        exec(compile(test, f"<{task['task_id']}-test>", "exec"), ns)
        ns["check"](ns[task["entry_point"]])
        return True, "pass"
    except Exception:
        return False, traceback.format_exc(limit=1).strip().splitlines()[-1][:160]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--results", default=RESULTS, help="results file")
    args = ap.parse_args()
    results_path = args.results

    tasks = load_dataset()
    results = load_results(results_path)
    n_pass = n_run = 0
    failures = []
    for task_id, task in tasks.items():
        r = results.get(task_id)
        if r is None:
            print(f"  {task_id}: MISSING")
            continue
        if "error" in r:
            failures.append((task_id, f"gen-error: {r['error']}"))
            continue
        ok, msg = run_check(task, r["completion"])
        n_run += 1
        if ok:
            n_pass += 1
        else:
            failures.append((task_id, msg))
        if args.verbose or not ok:
            print(f"  {task_id}: {'PASS' if ok else 'FAIL'} — {msg if not ok else ''}")
    print(f"\npass@1: {n_pass}/{n_run} = {n_pass/n_run:.1%}" + (f"  (evaluated {n_run} of {len(tasks)})" if n_run < len(tasks) else ""))
    if failures and not args.verbose:
        print("failures:")
        for tid, msg in failures:
            print(f"  {tid}: {msg}")

if __name__ == "__main__":
    main()

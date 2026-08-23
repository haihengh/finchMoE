#!/usr/bin/env python3
"""smoke_check.py — protocol diagnostics on a (possibly partial) sample file.

evalplus.evaluate refuses to score a partial run ("Missing problems in
samples"), so a smoke run cannot produce pass@1. That is fine: the smoke run
exists to prove the PROTOCOL is right before committing hours to the sweep.

Checks, per the plan's go/no-go gate:
  1. protocol   — is the model returning fenced self-contained scripts, or
                  bare continuations? (the whole point of the EvalPlus harness)
  2. think leak — --no-think does not actually suppress reasoning in this
                  engine, and a think block eats the 768-token budget
  3. truncation — the engine always reports finish_reason "stop", so hitting
                  the token cap is the only truncation signal
  4. sanitizer  — did evalplus.sanitize extract a real function, or nothing?

Usage: smoke_check.py <model-id>
"""
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if not os.path.exists(path):
        sys.exit(f"ABORT: no such file: {path}")
    return [json.loads(l) for l in open(path)]


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "finchmoe-3bit"
    stem = os.path.join(BASE, "results", "humaneval",
                        f"{model_id}_openai_temp_0.0")
    raw = load(f"{stem}.raw.jsonl")
    san = {r["task_id"]: r["solution"] for r in load(f"{stem}.jsonl")}

    n = len(raw)
    fenced = think = empty_san = trivial_san = 0
    loops = []

    for r in raw:
        sol = r["solution"]
        tid = r["task_id"]
        if "```" in sol:
            fenced += 1
        if "<think>" in sol or "</think>" in sol:
            think += 1

        s = (san.get(tid) or "").strip()
        if not s:
            empty_san += 1
        elif not re.search(r"^\s*def\s+\w+", s, re.M):
            # only imports / no function body survived
            trivial_san += 1

        # crude repetition detector: same 40-char line repeated a lot
        lines = [l.strip() for l in sol.splitlines() if len(l.strip()) > 20]
        if lines:
            top = max(set(lines), key=lines.count)
            if lines.count(top) >= 5:
                loops.append((tid, lines.count(top), top[:60]))

    def pct(x):
        return f"{x}/{n} ({x / n:.0%})" if n else "0/0"

    print(f"\n=== smoke diagnostics: {model_id} ({n} samples) ===")
    print(f"  fenced code block   : {pct(fenced)}   <- protocol working if high")
    print(f"  <think> leaked      : {pct(think)}   <- should be 0")
    print(f"  sanitizer -> empty  : {pct(empty_san)}")
    print(f"  sanitizer -> no def : {pct(trivial_san)}   <- import-only, a real failure")
    print(f"  repetition loops    : {pct(len(loops))}")
    for tid, cnt, snippet in loops[:5]:
        print(f"      {tid}: line x{cnt}  {snippet!r}")

    print("\n  VERDICT:", end=" ")
    if n and fenced / n >= 0.8:
        print("protocol OK — model is answering EvalPlus-style.")
    else:
        print("PROTOCOL BROKEN — model is not returning fenced scripts. "
              "Do not start the sweep.")
    if think:
        print("  WARNING: think blocks leaked; they consume the token budget. "
              "Consider --think-budget on the engine.")


if __name__ == "__main__":
    main()

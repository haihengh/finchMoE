#!/usr/bin/env python3
"""e2_build_prompt.py — build the E2 differential prompt, exactly as the
HumanEval benchmark served it.

The benchmark's chat request contains a system message and a user message
(the EvalPlus instruction wrapper around the HumanEval task). Our engine
IGNORES the request's system role and reads ~/.flash-moe/system.md instead;
llama.cpp honors the request's system role. Both must therefore see the same
system prompt text. run_cell.sh wrote the 3090's system prompt to system.md
for the FinchMoE cells, and the EvalPlus SDK sent the same string to
llama.cpp — so "You are a helpful assistant good at coding." is the system
prompt on BOTH sides.

Outputs:
  /tmp/e2_prompt.txt    the user message (exactly what the SDK sent)
  /tmp/e2_system.txt    the system prompt (exactly what system.md held)
"""
import gzip
import json
import os

DATASET = "/Volumes/samsung 2t/code/finchMoE/humaneval_m1/HumanEval.jsonl.gz"
SYSTEM = "You are a helpful assistant good at coding."

# The 3090's evalplus driver: raw task prompt wrapped by EvalPlus's
# instruction prefix + a ```python fence (verified against the 3090's
# rendered template byte-for-byte on 2026-08-21).
INSTRUCTION = ("Please provide a self-contained Python script that solves the "
               "following problem in a markdown code block:\n```\n{prompt}\n```")


def main():
    rows = [json.loads(l) for l in gzip.open(DATASET, "rt")]
    task = next(r for r in rows if r["task_id"] == "HumanEval/0")
    user_msg = INSTRUCTION.format(prompt=task["prompt"].strip())

    with open("/tmp/e2_system.txt", "w") as f:
        f.write(SYSTEM)
    with open("/tmp/e2_prompt.txt", "w") as f:
        f.write(user_msg)
    print(f"[e2] user message: {len(user_msg)} chars")
    print(f"[e2] system prompt: {SYSTEM!r}")
    print("--- user message ---")
    print(user_msg)
    print("--- end ---")


if __name__ == "__main__":
    main()

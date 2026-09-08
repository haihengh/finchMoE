#!/usr/bin/env python3
"""Score evalplus solutions faithfully: sanitize(raw, entry_point), exec,
run fn against base_input/base_output. Mirrors evalplus untrusted_check.
Usage: score_slice.py <raw.jsonl> <start> <end>
"""
import json
import sys
import traceback

from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize


def run_one(code, entry, inputs, expected):
    ns = {}
    try:
        exec(compile(code, '<sol>', 'exec'), ns)
    except Exception as e:
        return f'EXEC_ERR({type(e).__name__})'
    if entry not in ns:
        return 'NO_ENTRY_POINT'
    fn = ns[entry]
    try:
        for inp, exp in zip(inputs, expected):
            out = fn(*inp)
            if out != exp:
                return f'WRONG({inp})'
        return 'PASS'
    except Exception as e:
        return f'RUN_ERR({type(e).__name__}: {e})'


def main():
    raw_path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    problems = get_human_eval_plus()
    raw = {}
    for line in open(raw_path):
        r = json.loads(line)
        raw[r['task_id']] = r.get('solution', '')

    results = {}
    for i in range(start, end):
        tid = f'HumanEval/{i}'
        if tid not in raw:
            results[tid] = 'NO_SAMPLE'
            continue
        p = problems[tid]
        code = sanitize(raw[tid], p['entry_point'])
        inputs = p.get('base_input') or p.get('test_inputs') or []
        expected = p.get('base_output') or p.get('test_outputs') or []
        results[tid] = run_one(code, p['entry_point'], inputs, expected)

    n = len(results)
    passed = sum(1 for v in results.values() if v == 'PASS')
    print(f'{raw_path}  tasks {start}-{end}: pass {passed}/{n} ({100.0*passed/n:.1f}%)')
    for tid, v in results.items():
        print(f'  {tid}: {v}')


if __name__ == '__main__':
    main()

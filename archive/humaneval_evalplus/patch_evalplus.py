#!/usr/bin/env python3
"""patch_evalplus.py — local patches to the installed evalplus.

Two patches, both idempotent, both keyed by a marker comment:

1. macOS compatibility (the 3090 run needed two Windows patches instead).

    evalplus/eval/utils.py :: reliability_guard()
    ValueError: current limit exceeds maximum limit

`resource.setrlimit(RLIMIT_AS, ...)` fails on macOS when the requested soft
limit exceeds the inherited hard limit. evalplus already special-cases Darwin
for RLIMIT_STACK but not for RLIMIT_AS / RLIMIT_DATA, so every test subprocess
dies before running a single test and the whole evaluation reports zero.

This wraps those setrlimit calls in try/except. The guard is explicitly NOT a
security sandbox (evalplus says so in its own docstring), so losing the memory
cap costs nothing for a benchmark we are running on our own generations.

2. Scoring with a known-excluded problem (evalplus/evaluate.py).

evalplus asserts that every problem in the dataset has a sample:

    assert len(completion_id) == len(problems), "Missing problems in samples"

That is the right default — it is what catches a sweep that silently lost
tasks. But HumanEval/38 wedges this engine deterministically (see
run_server_cell_resilient.sh), so the ple2bit cell will reach scoring with
exactly one known gap. The patch makes the assert env-gated: with
FINCHMOE_EXPECT_MISSING=38, that exact set may be missing and is recorded in
the results json as `finchmoe_excluded_problems`; any *other* gap still
raises. Excluded problems are absent from results["eval"], so pass@1 is
computed over the remaining 163 — which is why the exclusion has to be
visible in the artifact and in any published number.

Called by setup.sh. Safe to re-run.
"""
import os
import sys

MARKER = "# --- finchmoe macOS patch ---"
MISSING_MARKER = "# --- finchmoe missing-problem patch ---"

ORIGINAL = """        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )
"""

PATCHED = f"""        {MARKER}
        # macOS raises "current limit exceeds maximum limit" when the soft
        # limit is above the inherited hard limit. Best-effort: the guard is
        # not a security sandbox, so skipping the cap is acceptable here.
        for _lim in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_STACK"):
            if _lim == "RLIMIT_STACK" and platform.uname().system == "Darwin":
                continue
            try:
                resource.setrlimit(
                    getattr(resource, _lim),
                    (maximum_memory_bytes, maximum_memory_bytes),
                )
            except (ValueError, OSError):
                pass
"""

MISSING_ORIGINAL = """            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"
"""

# v1 of the patch, kept only so an install carrying it can be upgraded in place.
# It compared FINCHMOE_EXPECT_MISSING verbatim against evalplus's task ids, so
# the bare id the sweep uses ("38") never matched "HumanEval/38" and the run
# aborted instead of scoring. v2 normalises both spellings.
MISSING_PATCHED_V1 = f"""            assert n_samples == len(remainings), "Missing problems in unfinished"
            {MISSING_MARKER}
            # FINCHMOE_EXPECT_MISSING names task ids that may legitimately have
            # no sample (a deterministically wedging task the sweep stepped
            # over). Anything outside that set still fails the run: the whole
            # value of the assert is catching gaps nobody intended.
            _missing = sorted(set(problems) - set(completion_id))
            _expect = {{
                s.strip()
                for s in os.environ.get("FINCHMOE_EXPECT_MISSING", "").split(",")
                if s.strip()
            }}
            if _missing:
                if set(_missing) != _expect:
                    raise AssertionError(
                        f"Missing problems in samples: {{_missing}} "
                        f"(FINCHMOE_EXPECT_MISSING={{sorted(_expect)}})"
                    )
                warn(
                    f"finchmoe: {{len(_missing)}} problem(s) {{_missing}} have no "
                    "sample and are EXCLUDED from this score — pass@k is over the "
                    "remaining problems only; see finchmoe_excluded_problems"
                )
                results["finchmoe_excluded_problems"] = _missing
            else:
                assert len(completion_id) == len(problems), "Missing problems in samples"
"""

MISSING_PATCHED = f"""            assert n_samples == len(remainings), "Missing problems in unfinished"
            {MISSING_MARKER}
            # FINCHMOE_EXPECT_MISSING names task ids that may legitimately have
            # no sample (a deterministically wedging task the sweep stepped
            # over). A bare id ("38") is accepted alongside the full task id
            # ("HumanEval/38") because the sweep names tasks by bare id
            # everywhere else. Anything outside that set still fails the run:
            # the whole value of the assert is catching gaps nobody intended.
            _missing = sorted(set(problems) - set(completion_id))
            _expect = {{
                s if s.startswith("HumanEval/") else "HumanEval/" + s
                for s in (
                    x.strip()
                    for x in os.environ.get("FINCHMOE_EXPECT_MISSING", "").split(",")
                )
                if s
            }}
            if _missing:
                if set(_missing) != _expect:
                    raise AssertionError(
                        f"Missing problems in samples: {{_missing}} "
                        f"(FINCHMOE_EXPECT_MISSING={{sorted(_expect)}})"
                    )
                warn(
                    f"finchmoe: {{len(_missing)}} problem(s) {{_missing}} have no "
                    "sample and are EXCLUDED from this score — pass@k is over the "
                    "remaining problems only; see finchmoe_excluded_problems"
                )
                results["finchmoe_excluded_problems"] = _missing
            else:
                assert len(completion_id) == len(problems), "Missing problems in samples"
"""


def apply(target_path, marker, original, patched, upgrade_from=None):
    src = open(target_path).read()
    if marker in src:
        if upgrade_from is None or upgrade_from not in src:
            print(f"[patch] already applied: {target_path}")
            return
        open(target_path, "w").write(src.replace(upgrade_from, patched))
        print(f"[patch] upgraded to the current revision: {target_path}")
        return
    if original not in src:
        sys.exit(f"ABORT: source does not match the expected shape in\n"
                 f"       {target_path} — evalplus version drift. Patch by hand.")
    open(target_path, "w").write(src.replace(original, patched))
    print(f"[patch] applied: {target_path}")


def main():
    try:
        import evalplus.eval.utils as u
        import evalplus.evaluate as e
    except ImportError:
        sys.exit("ABORT: evalplus not importable — run setup.sh first")

    apply(u.__file__, MARKER, ORIGINAL, PATCHED)
    apply(e.__file__, MISSING_MARKER, MISSING_ORIGINAL, MISSING_PATCHED,
          upgrade_from=MISSING_PATCHED_V1)

    # prove the first one: the exact call that was failing
    import importlib
    importlib.reload(u)
    u.reliability_guard(maximum_memory_bytes=2 * 1024 * 1024 * 1024)
    print("[patch] verified: reliability_guard() no longer raises")
    # reliability_guard nulls builtins.exit and disables faulthandler, so leave
    # hard — but flush first, os._exit does not flush stdio.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

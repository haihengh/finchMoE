#!/usr/bin/env python3
"""patch_evalplus.py — macOS compatibility patch for the installed evalplus.

The 3090 run needed two Windows patches (missing SIGALRM, missing `resource`).
macOS has both of those, but it needs a different one:

    evalplus/eval/utils.py :: reliability_guard()
    ValueError: current limit exceeds maximum limit

`resource.setrlimit(RLIMIT_AS, ...)` fails on macOS when the requested soft
limit exceeds the inherited hard limit. evalplus already special-cases Darwin
for RLIMIT_STACK but not for RLIMIT_AS / RLIMIT_DATA, so every test subprocess
dies before running a single test and the whole evaluation reports zero.

This wraps those setrlimit calls in try/except. The guard is explicitly NOT a
security sandbox (evalplus says so in its own docstring), so losing the memory
cap costs nothing for a benchmark we are running on our own generations.

Idempotent — safe to re-run. Called by setup.sh.
"""
import os
import sys

MARKER = "# --- finchmoe macOS patch ---"

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


def main():
    try:
        import evalplus.eval.utils as u
    except ImportError:
        sys.exit("ABORT: evalplus not importable — run setup.sh first")

    path = u.__file__
    src = open(path).read()

    if MARKER in src:
        print(f"[patch] already applied: {path}")
        return

    if ORIGINAL not in src:
        sys.exit(f"ABORT: reliability_guard source does not match the expected\n"
                 f"       shape in {path} — evalplus version drift. Patch by hand.")

    open(path, "w").write(src.replace(ORIGINAL, PATCHED))
    print(f"[patch] applied macOS setrlimit patch: {path}")

    # prove it: the exact call that was failing
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

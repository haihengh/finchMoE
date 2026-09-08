# M4 Rebuild & Deploy Runbook

**When to use:** the binary deployed to the M1 (`finchmoe-m1/finchmoe-infer`) needs a
rebuild from current source. The M1's Command Line Tools are too old (clang 12 /
macOS 11.3 SDK) to compile the Metal 3.1 shader paths, so builds happen on the M4.

**Why this time (2026-08-14):** the deployed binary was built from an intermediate
source state with a broken `/v1/completions` SSE escape — generated newlines render
as the letter `n`. Chat-mode escaping works; only the completions endpoint is broken.
Current source is correct.

## On the M4

### 1. Get the repo to the current commit

```bash
cd ~/Desktop/code/finchMoE        # or wherever the repo lives
git pull                          # if it's a clone of github.com/haihengh/finchMoE
git log --oneline -1              # must show 3c781ca (or newer)
```

If `git pull` is not how the two machines sync, copy the repo (or at least
`finchmoe/infer.m`) from the M1 instead — the file must be newer than the 12:40
snapshot that produced the broken binary.

### 2. Sanity-check the fix is in the source (10 seconds)

```bash
grep -n "case '\\\\n'" finchmoe/infer.m
```

Both SSE writers must show `*w++ = '\\'; *w++ = 'n';` (backslash first). If either
shows only `*w++ = 'n';`, the source itself is stale — don't build.

**Also check the tokenizer byte table** (this bit us on 2026-08-14 — the rebuilt
binary encoded prompt newlines as the letter `n`):

```bash
git status --short finchmoe/          # list any uncommitted changes
git diff finchmoe/tokenizer.h         # MUST be empty
```

`build_byte_unicode_table` in `finchmoe/tokenizer.h` must use the standard GPT-2
mapping (`else { tok->byte_char[b] = 256 + n; n++; }`). A dirty variant maps
control bytes to C-escape letters (`\n` → `'n'`), which corrupts every prompt.
If the diff is non-empty, restore the committed version first:
`git checkout -- finchmoe/tokenizer.h` (then re-check `git status` for any other
uncommitted experiments before rebuilding).

### 3. Rebuild

```bash
cd finchmoe
make
ls -la finchmoe-infer            # fresh timestamp, should be ~270 KB
```

### 4. Copy to the M1

Overwrite the deployed binary (the M1 server can keep running; the file swap only
takes effect on next launch):

```bash
scp finchmoe-infer john@<m1-hostname>:/Users/john/Desktop/code/finchMoE/finchmoe-m1/finchmoe-infer
```

(or AirDrop / USB stick / any file transfer — destination is the same path).

## Back on the M1

Tell the Claude Code session on the M1 "copied". It will:

1. `chmod +x` the binary (file copies drop the +x bit)
2. Restart the server through the memory gate (`humaneval_m1/start_server.sh`)
3. Smoke-test `/v1/completions` newline rendering
4. Clear results and launch the full 164-problem HumanEval run

## Reference

- Escape bug history: `/v1/completions` was added in `df00099` with correct escaping;
  the broken binary predates that fix being built.
- M1 memory gate: refuses to boot below 3.0 GB available (free+inactive+purgeable+
  speculative); `sudo mdutil -a off` helps keep Spotlight from re-eating reclaimed RAM.

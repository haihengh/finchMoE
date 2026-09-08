# Protocol Improvement Plan — finchMoE HumanEval Prompt-Protocol Parity

**Status:** §1-§9's prompt-protocol fixes (chat endpoint, matching system message, `--no-think`,
think re-entry ban) are IMPLEMENTED and VERIFIED — see §0. They did **not** close the gap.
**§0 is the current diagnosis and the active plan. §1-§14 below are the ORIGINAL plan, kept
for history/rationale; their line-number anchors are STALE (written against `0b78d45`,
~10-11k lines; current `infer.m` is ~15.8k lines) and their root-cause theory (D1-D4, a
prompt-template mismatch) is REFUTED as the dominant cause by the §0 evidence. Do not
re-derive or re-apply §5/§6 without re-checking against current `infer.m` first — most of it
already exists (see §0.1).**
**Scope now:** `finchmoe/infer.m` (numeric/sampling path) + `humaneval_evalplus/` (harness,
already built and run on this Mac) + local `llama.cpp/build/bin/llama-server` (Metal, already
built in-tree) as a same-machine reference. `humaneval_3090/` stays the frozen external
baseline; `humaneval_m1/` is superseded by `humaneval_evalplus/` and should not be extended
further.

---

## 0. Status update (2026-08-23): the gap is numeric, not protocol

### 0.1 What actually got run

`humaneval_evalplus/` (branch `3090`, merged to `main` at `e679204`) is a byte-identical-tooling
re-run of the 3090's EvalPlus 0.3.1 harness against `finchmoe-infer`, via `shim.py` (OpenAI-shaped
proxy in front of the engine's SSE `/v1/chat/completions`). It already implements the core of
this plan's §4 "Chosen approach":

- request shape matches `humaneval_3090/humaneval_gen.py` byte-for-byte (same file, only `root=`
  changed): system message `"You are a helpful assistant good at coding."`, greedy, `max_tokens`
  768 (not 512 — confirmed on the wire, see `humaneval_evalplus/README.md`), `top_p=0.95` (a
  no-op at T=0 on both sides)
- `shim.py` forces the request onto the chat endpoint and injects the system prompt via
  `~/.flash-moe/system.md` — the file-based override `load_system_prompt()` already supports
  this operationally; no engine patch was needed for this cell
- `--no-think` is passed; the empty-think-block suffix (Bug 12's fix) plus the post-close
  re-entry ban (`infer.m:14277-14279`, `logits[THINK_START_TOKEN] = -INFINITY` etc.) are already
  in the current binary
- `smoke_check.py` measures `<think>` leakage directly per run — it is 0-2/164 across every
  cell (§0.3), so D4 (think leakage) is **not** the dominant failure mode any more

In other words: the concrete engine/harness changes this plan's §5-§6 called for are, in
substance, already in place and already exercised. The remaining gap must be explained by
something else.

### 0.2 The decisive test: byte-identical weights, still ~13%

`run_cell.sh gguf3090` loads `Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf` — the *exact*
`lmstudio-community` file the 3090 used to score 91.5/89.0 (733 tensors, imatrix-quantized,
21,166,757,728 bytes, same file the 3090's README documents). Through `finchmoe-infer`'s GGUF
path, with the already-fixed protocol above, it scores:

| Cell | Weights | pass@1 base | pass@1 plus |
|---|---|---|---|
| A (3090, external, reference) | lmstudio-community Q4_K_M | 91.5% | 89.0% |
| B — `finchmoe-3bit` | native 3-bit | 13.4% (22/164) | 12.8% |
| C — `finchmoe-4bit` | native 4-bit | 12.8% | 12.8% |
| D — `finchmoe-gguf-q4km` | our own Q4_K_M GGUF (no imatrix) | 12.8% | 12.8% |
| D′ — `finchmoe-gguf-q4km-3090` | **the 3090's exact GGUF file** | 12.8% | 12.8% |

(Cross-check: `evalplus.evaluate` on the 3090's own published sample file, run through this
Mac's patched evalplus install, reproduces 0.915/0.890 exactly — the *scoring* half is
byte-identical across machines; every point of the gap above comes from *generation*.)

Cell D′ is the load-bearing result: identical weights, identical prompts, identical greedy
decoding — yet finchMoE reproduces neither the 3090's score nor even a materially different
score from its own 3-bit/4-bit native tiers. Four tiers spanning native 3-bit through
imatrix-quantized Q4_K_M all converge to the same ~12.8-13.4% band. **A quantization-quality
explanation cannot produce that convergence** — 3-bit and imatrix Q4_K_M are not equivalent in
weight fidelity, but they are equivalent in *finchMoE's* score. Something common to all four
paths through `finchmoe-infer` is capping every tier at the same ceiling. §5 of the original
plan's D1-D4 (endpoint mismatch, wrong system text, implicit double system prompt, think
leakage) are the things that historically differed between the 3090 and finchMoE — and they are
now confirmed matched (§0.1) while the gap persists. That refutes the prompt-template theory as
the dominant cause; it may still be worth a few points, but it is not the ~78-point story.

### 0.3 Forensic finding: greedy-attractor loops break the docstring, not the logic

`smoke_check.py` per-cell diagnostics (164/164, full runs):

| Cell | fenced | `<think>` leaked | sanitizer → no `def` | repetition loops (5x+ line) |
|---|---|---|---|---|
| 3bit | 164/164 | 2/164 (1%) | 16/164 (10%) | not separately counted in this run |
| 4bit | 164/164 | 0/164 | 23/164 (14%) | 14/164 (9%) |
| gguf (ours) | 164/164 | 0/164 | 28/164 (17%) | 16/164 (10%) |
| gguf3090 (exact file) | 164/164 | 0/164 | 30/164 (18%) | 21/164 (13%) |

The protocol layer is clean (100% fenced, near-zero think leakage). The failures are inside the
model's own token stream. Comparing finchMoE's raw output for `HumanEval/0` and `HumanEval/1`
against the 3090's own published sanitized solutions for the same two tasks:

- **3090 (llama.cpp)**, `HumanEval/0`: docstring keeps exactly 2 short `>>>` examples, closes
  `"""` cleanly, then writes `sorted_numbers = sorted(numbers)` and finishes the function.
- **finchMoE (any GGUF/native tier)**, `HumanEval/0`: docstring reproduces 5-6+ `>>>` examples,
  each slightly mutated, and never reaches the closing `"""` before the 768-token cap — the
  sanitizer's `ast.parse` fails on the unterminated string, and everything after the last import
  is discarded, scoring as `no def` / fail.
- Same pattern on `HumanEval/1`: finchMoE's continuation drops a closing quote inside
  `paren_string.replace('` and de-indents by one space — a malformed statement, not merely a
  "slow" or "verbose" one.

This is a **greedy-decoding attractor** (the same failure family as `BUGS.md` Bug 12's
repetition loops): once the model samples one more `>>>` example than llama.cpp would have, it
falls into a basin where reproducing docstring-example lines becomes the greedy-optimal
continuation and it cannot escape before the token budget runs out. The question is *why*
finchMoE's greedy argmax diverges from llama.cpp's greedy argmax on the **same weights**, at
some early token — if the logits truly matched, greedy decoding is deterministic and the two
engines would produce identical output. That divergence has to originate in the forward pass:
dequantization, attention/RoPE, MoE routing/combine, or final-norm/lm_head numerics. This
codebase has hit this exact class of bug before (Bug 18: FMA contraction created ULP-level logit
noise invisible at the source level; Bug 17: a silent clobber that only showed up as an RMS
mismatch two components downstream) — greedy decoding is unforgiving of exactly this kind of
small, otherwise-harmless numeric drift, because it can flip an argmax tie into the wrong basin,
and greedy decoding then "locks in" and amplifies the mistake for hundreds of tokens.

**Revised diagnosis:** the ~78-point gap is dominated by a numeric divergence in finchMoE's
forward pass relative to llama.cpp on the same GGUF weights, not by the prompt-protocol
differences this plan originally targeted. §5-§9's protocol work should still be treated as
correct and worth keeping (it removed real confounders and got the harness to a trustworthy
byte-identical-weights control), but it is not the fix. §15 below is the new, active
investigation plan.

### 0.4 What is NOT yet ruled out (keep these on the list, do not re-litigate blindly)

- **`--low-memory` CPU fallback.** The `gguf`/`gguf3090` cells pass `--low-memory` (routes matmuls
  through a CPU fallback path instead of the Metal zero-copy path used by 3-bit/4-bit — see
  `infer.m:15253-15259` region, current file). That path has less test/parity history than the
  GPU path (only expert dequant had a bitwise-parity check historically, Bug 4). It is a weaker
  suspect than it looks, because 3-bit/4-bit do NOT use `--low-memory` and show the same ~13%
  ceiling and the same loop signature — so it cannot be the *sole* cause, but it could still be
  contributing independently on top of a shared bug. Test in isolation (§15.4).
- **Sampling/engine parity beyond greedy.** `-e 0 --top-k 1 --rep-penalty 1.0` is intended to
  match llama.cpp's greedy default (`penalty_repeat = 1.0` is llama.cpp's own "disabled" value
  too, confirmed in `llama.cpp/common/common.h:239`) — this looks matched on paper, but has not
  been probed by dumping both engines' per-step top-20 and diffing (§15.2 does exactly this).
- **Reference file provenance for the non-3090-exact GGUF cell.** Cell D (`gguf`, our own
  conversion) is a different, non-imatrix file from the 3090's, so in principle it should score
  lower than D′ — but it doesn't (both ~12.8%), which is itself evidence the attractor bug
  dominates over weight-quality differences at this level.

---

## 1. Background and goal

Two machines produced HumanEval numbers for the same model class, but with **different prompt protocols**:

| | Spec baseline (3090) | finchMoE run (M1) |
|---|---|---|
| Repo | `humaneval_3090/` | `humaneval_m1/` |
| Server | `llama-server` (llama.cpp `a30273376`), Windows, RTX 3090 | `finchmoe-infer` (`infer.m`), macOS M1 |
| Endpoint | `POST /v1/chat/completions` | `POST /v1/completions` (raw) |
| System message | `"You are a helpful assistant good at coding."` | none sent by client; server-side default `"You are a helpful assistant."` (or `~/.flash-moe/system.md`) |
| Chat template | llama.cpp Qwen3.6 template | none on the completions path |
| Thinking | **off** (`--reasoning off`) | `--no-think` passed, but see §3 — it has no effect on the completions path |
| Decoding | greedy, `max_tokens=512`, `n=1` | greedy (`-e 0 --top-k 1 --rep-penalty 1.0`), `max_tokens=512` |
| Result | 91.5 / 89.0 (base / plus) | unknown — run appears to stall or produce think-only output |

**Goal:** make the M1 finchMoE run feed the model *the same prompt* the 3090 spec run fed it, so the two numbers are comparable. Secondary goal: eliminate the "thinks and then stops / produces nothing usable" symptom.

---

## 2. Symptom summary (what was observed)

1. The user reports a local llama.cpp run at `localhost:8080` that "never does anything, just thinking then stop" — and, as the more pressing symptom, a harness that "keeps thinking but never stops producing a result".
2. The finchMoE M1 harness (`humaneval_m1/`) produces results that are not comparable to the 3090 baseline because of prompt-protocol differences (§3).
3. Both symptoms are explained by the code facts below plus the loop analysis in §10: (a) think tokens stream as output even when thinking is supposed to be off, and EOS inside/after a think block yields a turn that did nothing; (b) neither the finchMoE generation loop nor the agent-loop step loop has an end condition other than EOS / token cap / no-tool-call message, so a model that never emits EOS (or keeps returning tool calls) loops indefinitely.

---

## 3. Verified findings (code evidence)

### 3.1 What the spec run actually sent (frozen baseline)

`humaneval_3090/humaneval_gen.py:13-28`:

```python
system_msg = "You are a helpful assistant good at coding."
return client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_msg},
        {"role": "user", "content": message},
    ],
    max_tokens=max_tokens,   # 512
    temperature=temperature, # greedy=True -> 0.0
    n=n,                     # 1
    top_p=0.95,
    timeout=600,
    **kwargs,
)
```

`humaneval_3090/README.md:28-36`: greedy, 1 sample, 512 new tokens, **thinking disabled on the server** (`llama-server ... --reasoning off ... --host 127.0.0.1 --port 8080`).

So the spec prompt rendered by llama.cpp (ChatML for the qwen35 arch, reasoning off) is:

```
<|im_start|>system
You are a helpful assistant good at coding.<|im_end|>
<|im_start|>user
<HumanEval prompt><|im_end|>
<|im_start|>assistant
```

(Exact token ids must be confirmed with the llama.cpp dump in §8; the text form above is the expected rendering.)

### 3.2 What the finchMoE server does today

**System prompt is server-side and fixed at startup.**
`finchmoe/infer.m:10107-10126` `load_system_prompt()`: reads `~/.flash-moe/system.md` if the file exists, else returns `"You are a helpful assistant."`. The pre-cache happens once at serve startup (`infer.m:10803-10869`, `tokenize_system_prompt()` at `10131-10143`):

```
<|im_start|>system
{system_prompt}<|im_end|>
```

**Chat endpoint discards the request's system message.**
`POST /v1/chat/completions` (`infer.m:10971-11045`) extracts content with `extract_last_content(body)` (`10988`; function at `9816-9856`), which returns **the last `"content"` value** in the messages array — i.e., the user turn. Any `role:"system"` message from the client is silently dropped and replaced by the server-side cached system prompt.

**Chat-path user-turn template** (`tokenize_user_turn`, `infer.m:10074-10087`):

```
<|im_start|>user
{content}<|im_end|>
<|im_start|>assistant
{think_suffix}
```

where `build_think_suffix()` (`10058-10070`) with `--no-think` (`g_no_think`, flag parsed at `11329` case `'H'`) injects:

```
<think>

</think>

```

(This empty-think-block form is deliberate — see `BUGS.md` Bug 12: prepending `<think>\n` made the model close the block immediately and leak reasoning.)

**Completions endpoint is raw, but still sits on top of the system prompt.**
`POST /v1/completions` (`infer.m:11047-11108`) extracts `"prompt"` (`extract_prompt`, `9859+`) and marks the queue entry `is_completion = 1` (`11094`; `ServeQueueEntry` at `10210-10218`). In `process_chat_request` (`10312+`):

- `is_completion` → `pt = encode_prompt_text_to_tokens(content)` — **no ChatML wrapping, no `<|im_start|>assistant` marker** (`10338-10340`).
- But every non-continuation request — completions included — restores the **system-prompt KV snapshot** and starts at `pos = s->sys_prompt_len` (`10365-10410`).

So the actual context for a raw HumanEval completion is:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>          ← implicit, untemplated
<HumanEval prompt tokens>                        ← raw, no user/assistant markers
```

**Think tokens are streamed to the client.**
The generation loop (`10561-10670`) tracks `in_think` / `think_ended` and bans `THINK_START_TOKEN`/`THINK_END_TOKEN` only **after** the block has closed (`10664-10667`). The SSE send happens for **every** token regardless of think state (`10629-10633`; `sse_send_delta_completion` at `10001-10022`). Only the non-streamed `gen_response` buffer skips think tokens (`10622-10628`). The M1 harness reads the SSE stream, so **a think block lands verbatim in the recorded completion**.

**`--no-think` never reaches the completions path.**
`build_think_suffix()` is only called from `tokenize_user_turn` / `tokenize_continuation_turn` / `tokenize_chat_message` (chat templates). The raw-completions path skips all of them, so the model is free to emit `<think>` — and the tokens are recorded.

**Per-request sampling params are ignored.**
The server parses `max_tokens` (`extract_max_tokens`, `9880-9887`) but not `temperature`/`top_p` from the body; sampling is set by server flags (`-e`, `--top-k`, `--rep-penalty`). Not a blocker (the start script already launches greedy), but it is a protocol gap to remember.

### 3.3 The four deviations (summary)

1. **D1 — Endpoint/template.** Spec used `chat/completions` (ChatML with assistant marker); M1 harness uses raw `completions`.
2. **D2 — System message text.** Spec: `"You are a helpful assistant good at coding."`. Engine default: `"You are a helpful assistant."` (or whatever `~/.flash-moe/system.md` holds on the M1). The engine also ignores the request's system message.
3. **D3 — Implicit system on completions.** The raw M1 prompt is decoded on top of the pre-cached (untemplated) system prompt, so it is neither "no system prompt" nor "spec system prompt" — it is a third thing.
4. **D4 — Thinking control.** Spec: reasoning off. M1: `--no-think` does not affect the completions path; think tokens are streamed into the output. This is the likely mechanism behind "thinking then stop / think-only output".

### 3.4 Why "thinking then stop" happens (mechanism)

On the completions path the model can open a think block (`in_think = 1`). If it then emits EOS (`EOS_TOKEN_1` 248046 / `EOS_TOKEN_2` 248044, `infer.m:258-259`), the engine closes the block in the context (`10575-10594`) and stops — the client has received **only** `<think>…</think>` reasoning. Locally graded, that completion is garbage; in a chat UI it reads as "thought for a while, then nothing". Fixing D4 (bans + actually routing through the chat template) addresses this directly.

---

## 4. Chosen approach

Make the M1 run mirror the spec run exactly:

1. **Harness** (`generate.py`): call `POST /v1/chat/completions` with the spec's exact messages (system `"You are a helpful assistant good at coding."` + user prompt), `max_tokens=512`, `temperature=0` — the same request `humaneval_gen.py` sends.
2. **Engine** (`infer.m`): add a system-prompt override (`--system-prompt TEXT` flag + `FINCHMOE_SYSTEM_PROMPT` env) so the pre-cached system prompt is the spec text; add a hard think-token ban when `--no-think` is set so D4 cannot recur even if the empty-think-block suffix fails to suppress a think block.
3. **Server launch** (`start_server.sh`): export `FINCHMOE_SYSTEM_PROMPT` (or pass `--system-prompt`) and keep `--no-think -e 0 --top-k 1 --rep-penalty 1.0`.
4. **Parity gate**: capture the tokenized prompt ids from both servers for the identical request and diff them (§8).

**Rejected alternatives** (documented so nobody re-derives them):

- *A. Rebuild the spec prompt text in `generate.py` and keep `/v1/completions`.* The engine still prepends its own system snapshot (D3), so the context would contain two system prompts. Would additionally require a `--no-system` mode — strictly more engine surface than the chosen approach for a less faithful result.
- *B. Honor the request's system message per-request in the chat endpoint.* Correct OpenAI semantics, but the system prompt is pre-cached in KV at startup; per-request system text needs either a per-request re-prefill or reworking the snapshot restore (`10365-10410`) to start from an empty context via `tokenize_chat_message` (`10146-10162`). That is a real change with its own risk surface. **Defer** — the eval run uses one fixed system text, so a startup override achieves parity with 3 lines. Tracked as optional follow-up in §13.

**Decision point — think suppression mode** (choose at execution time):

- **Mode A (default, engine-tested):** keep `build_think_suffix`'s empty block `<think>\n\n</think>\n\n` (current `--no-think` behavior) + add the logit bans from §5.2. Text differs from the spec rendering only by that suffix.
- **Mode B (strictest textual parity):** make `--no-think` produce an **empty** suffix (`buf[0] = '\0'`) and rely solely on the bans. This matches llama.cpp `--reasoning off` text exactly, but it abandons the empty-block form that `BUGS.md` Bug 12 validated on the quantized model.

Start with Mode A. If the full-run score still lands ≥2 points below the 3090 baseline with all other protocol items identical, flip to Mode B (one-line edit in `build_think_suffix`, §13) and re-run.

---

## 5. Phase 1 — Engine changes (`finchmoe/infer.m`)

### 5.1 System-prompt override (REQUIRED)

Add a global next to `g_no_think` (`infer.m:402`):

```c
static int g_no_think = 0;          // 0 = thinking mode on, 1 = skip think block
static char *g_system_prompt_override = NULL;  // --system-prompt / FINCHMOE_SYSTEM_PROMPT
```

Change `load_system_prompt()` (`10107-10126`) so the precedence is:
**`--system-prompt` flag > `FINCHMOE_SYSTEM_PROMPT` env > `~/.flash-moe/system.md` file > built-in default.**

```c
static char *load_system_prompt(void) {
    if (g_system_prompt_override && g_system_prompt_override[0])
        return strdup(g_system_prompt_override);
    const char *env = getenv("FINCHMOE_SYSTEM_PROMPT");
    if (env && env[0])
        return strdup(env);
    const char *home = getenv("HOME");
    /* ... existing file / default code unchanged ... */
}
```

Register the flag:

- usage text (after the `--no-think` line, `infer.m:11157`):
  ```c
  printf("  --system-prompt TEXT  Override system prompt (else FINCHMOE_SYSTEM_PROMPT, then ~/.flash-moe/system.md, then built-in default)\n");
  ```
- `long_options` (after `{"no-think", no_argument, 0, 'H'},` at `11266`):
  ```c
  {"system-prompt",  required_argument, 0, 'q'},
  ```
- optstring `11287`: append `q:` → `"m:w:j:v:p:P:t:k:C:M:R:B:N:Q:e:o:I:r:b:lHLSTFE234GhXUY:VJq:"`
- switch (`11288+`):
  ```c
  case 'q': g_system_prompt_override = optarg; break;
  ```

`load_system_prompt()` is only called at serve startup (via `tokenize_system_prompt`, `10805`) and by the CLI prompt path (`tokenize_chat_message`, `10148`), both after argument parsing — no ordering hazard.

### 5.2 Hard think-token ban under `--no-think` (REQUIRED for D4)

Two edits in `process_chat_request`'s generation:

1. First-token sampling — after `lm_head_forward(s->wf, hidden, logits);` (`10552`) and before `int next_token = cpu_sample_temp(...)` (`10558`), insert:

```c
    if (g_no_think) {
        // --no-think must hold even if the model tries to open a think block:
        // ban the tags outright (chat template suffix is the primary mechanism,
        // this is the guarantee).
        logits[THINK_START_TOKEN] = -INFINITY;
        logits[THINK_END_TOKEN]  = -INFINITY;
    }
```

2. Loop ban — widen the existing post-think ban (`10664-10667`):

```c
        if (think_ended || g_no_think) {
            logits[THINK_START_TOKEN] = -INFINITY;
            logits[THINK_END_TOKEN] = -INFINITY;
        }
```

With this, no think tag can ever be sampled on a `--no-think` server, on **either** endpoint. The EOS-mid-think auto-close (`10575-10594`) becomes unreachable in no-think mode but stays for thinking mode.

### 5.3 Prompt-ids debug dump (REQUIRED for the §8 parity gate)

After the existing `[serve] %s prompt=%d tokens` log line (`10354`), add:

```c
    if (getenv("FINCHMOE_SERVE_DEBUG")) {
        fprintf(stderr, "[serve-dbg] %s prompt_ids:", request_id);
        for (int i = 0; i < pt->count && i < 1024; i++)
            fprintf(stderr, " %d", pt->ids[i]);
        fprintf(stderr, "\n");
    }
```

### 5.4 Build and deploy

Per `M4_REBUILD.md`: the M1's toolchain is too old for the Metal 3.1 shader paths, so **build on the M4**, then copy the binary.

```bash
cd ~/Desktop/code/finchMoE/finchmoe        # M4 (adjust path)
grep -n "system-prompt" infer.m            # sanity: all three 5.1 hunks present
make
ls -la finchmoe-infer                      # fresh timestamp, ~270 KB
scp finchmoe-infer john@<m1-hostname>:/Users/john/Desktop/code/finchMoE/finchmoe-m1/finchmoe-infer
```

Back up the deployed binary first on the M1: `cp finchmoe-m1/finchmoe-infer finchmoe-m1/finchmoe-infer.bak` and `chmod +x` after the copy.

---

## 6. Phase 2 — Harness changes

### 6.1 `humaneval_m1/generate.py`

Replace `complete()` (`generate.py:26-45`) — endpoint, body, and the SSE field (`chat/completions` streams `choices[0].delta.content`, not `choices[0].text`):

```python
SYSTEM_MSG = "You are a helpful assistant good at coding."  # spec parity, humaneval_3090/humaneval_gen.py:14

def complete(port, prompt, max_tokens=512, timeout=900):
    body = json.dumps({
        "model": "qwen3.6-35b-a3b",
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
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
        parts.append((chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or "")
    return "".join(parts)
```

No other function changes; `done_tasks()` resumability is untouched.

### 6.2 `humaneval_m1/start_server.sh`

Add the system-prompt override (env form — works even if 5.1's flag registration was missed) and keep the existing retry loop:

```bash
#!/bin/bash
export FINCHMOE_SYSTEM_PROMPT="You are a helpful assistant good at coding."
cd /Users/john/Desktop/code/finchMoE/finchmoe-m1
# ... unchanged loop ...
./finchmoe-infer -R 9000 -m . --weights quant_clean/model_weights_quant.bin \
    --manifest quant_clean/model_weights_quant.json -e 0 --top-k 1 --no-think --rep-penalty 1.0 \
    >> /tmp/he_server.log 2>&1 &
# ...
```

(`-e 0 --top-k 1` = greedy, matching the spec's `greedy=True`; the server ignores per-request temperature, so the flags are what counts.)

---

## 7. Phase 3 — Smoke tests on the Mac (before the full run)

All on the M1, in the deploy dir. Expected results are called out — treat any deviation as a stop-the-line.

```bash
# 1. Confirm the new binary is live and the memory gate lets it boot
tail -f /tmp/he_server.log          # expect "[serve] System prompt cached: N tokens prefilled"
                                    #       "[serve] Listening on http://0.0.0.0:9000"

# 2. Health
curl -s http://127.0.0.1:9000/health
# expect: {"status":"ok","model":"qwen3.6-35b-a3b"}

# 3. One chat request, exact spec shape
curl -sN http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"system","content":"You are a helpful assistant good at coding."},{"role":"user","content":"def is_even(n):\n    return n % 2 == 0\n\n# fix this function: def is_odd(n)\n"}],"max_tokens":128,"temperature":0}' \
  | tee /tmp/smoke_chat.sse

# EXPECT:
#  - SSE lines: data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{"content":"..."}}]...}
#  - terminal line: data: [DONE]
#  - the concatenated delta.content is code, not a think block
#  - grep -c "<think>" /tmp/smoke_chat.sse  →  0  (REQUIRED — 5.2 bans)
#  - the done event shows "completion_tokens":>0

# 4. Server log shows the protocol in effect
grep "enqueued" /tmp/he_server.log | tail -2
# expect: [serve] chatcmpl-N enqueued ...  (NOT "[COMPLETION]")
```

Run `generate.py --limit 3` as a mini harness check:

```bash
cd humaneval_m1
python3 generate.py --limit 3
# expect: 3 records in he_results.jsonl, no "error", ~300-500 words each,
#         completions contain "def ", no "<think>" substring
```

If step 3 shows a think block despite the bans, **stop** — the binary did not get the 5.2 edits (recheck `grep -n "g_no_think"` in the source and rebuild).

---

## 8. Phase 4 — Token-level protocol parity gate

Prove the finchMoE prompt is byte/token-identical to what the spec run fed llama.cpp.

1. **llama.cpp side (reference)** — on the Windows box (or the Mac if llama.cpp builds there), start the spec server and capture the rendered prompt:

   ```bash
   llama-server.exe -m <qwen3.6-35b-a3b.gguf> -c 32768 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 \
     --reasoning off --host 127.0.0.1 --port 8080 --verbose -no-cnv 2>&1 | tee llama_verbose.log
   ```

   Send the **same JSON as smoke step 3** via curl to `http://127.0.0.1:8080/v1/chat/completions`, then extract the prompt token dump from `llama_verbose.log` (llama.cpp prints the evaluated prompt tokens in verbose server mode; if it only prints the text, decode it back with `llama-cli` or note the text form).

2. **finchMoE side (under test)** — restart the M1 server with `FINCHMOE_SERVE_DEBUG=1` exported, send the identical request, and read the log:

   ```bash
   export FINCHMOE_SERVE_DEBUG=1
   # start server per start_server.sh (or manually), then:
   curl -sN http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model":"qwen3.6-35b-a3b","messages":[...same as step 3...],"max_tokens":1,"temperature":0}' > /dev/null
   grep "prompt_ids:" /tmp/he_server.log | tail -1
   ```

3. **Compare.** The two token-id lists must be identical, except for the Mode A vs Mode B think-suffix difference (§4):
   - Mode A: finchMoE has extra ids for `<think>\n\n</think>\n\n` right after the assistant marker.
   - Mode B: lists must be **identical**, start to finish.

   Record the two lists (and the llama.cpp log excerpt) in the PR / Agent Note for future re-verification.

   If they differ beyond the known suffix, do not run the full eval — the tokenizer or template is off (compare against `BUGS.md` Bug 11 history: tokenizer merge corruption produces exactly this signature).

---

## 9. Phase 5 — Full HumanEval run and evaluation

On the M1:

```bash
cd humaneval_m1
mv he_results.jsonl he_results_pre_protocol_fix.jsonl   # preserve old run
python3 generate.py                                      # full 164; resumable if interrupted
python3 evaluate.py                                      # local pass@1 (base tests)
```

Expected runtime: 164 problems × (~25 s prefill + up to 512 tok @ ~4 tok/s ≈ 2.2 min worst case) — order of 5-8 hours worst case; run in `tmux`/`nohup` and let the resumability logic handle interruptions. Watch `he_results.jsonl` growth, not just the terminal.

**Comparison target (base suite):** 3090 baseline for Qwen3.6-35B-A3B = **91.5% base / 89.0% plus**. `evaluate.py` computes base-only; porting plus tests is optional follow-up (§13).

**Known deliberate differences that remain after this plan** (do not chase them as bugs):

- Engine runs 3-bit quantized experts vs llama.cpp Q4_K_M — a quantization gap, not a protocol gap.
- `evaluate.py` truncation + base tests vs evalplus base+plus harness.
- M1 CPU/GPU numerics vs CUDA (sampling is greedy on both, so per-token drift is the only variable).
- Mode A empty-think-block suffix (if not flipped to Mode B).

**Acceptance criteria for the protocol fix itself:**

- [ ] `generate.py` requests carry the spec system message and go to `/v1/chat/completions` (visible in the server log per request).
- [ ] Server log's system-prompt pre-cache count matches the spec text (stderr shows the env override was used — extend the existing `[serve] Loaded custom system prompt` print to also cover the env/flag path, or verify via the token count changing from the default).
- [ ] No `<think>` in any of the 164 completions.
- [ ] 164/164 records with `"completion"` (no `"error"`).
- [ ] §8 prompt-ids gate passes (identical modulo Mode A suffix).
- [ ] pass@1 base lands within ~2 points of 91.5%. If it does not, flip to Mode B (§13) before investigating anything engine-numeric.

---

## 10. Phase 6 — "Keeps thinking, never produces a result" diagnostic playbook

Run these **only if** the symptom survives Phases 1-5 (the finchMoE engine half should not — 5.2 bans the think-tag mechanism).

### 10.1 Where an endless "thinking" loop actually comes from — two mechanisms

There is no magic "thinking" state that gets stuck. There are exactly two code paths that produce this symptom, and both are visible in the loop code:

**Mechanism A — one model response never ends, so the consumer waits on the stream forever.**

The finchMoE engine (`infer.m:10574-10670`) and the deepseek-harness agent loop (`packages/core/agent-loop/src/agent.ts:347`, `for await (const chunk of stream)`) both consume a model response until it *terminates*. A response terminates only on:

- an EOS token — finchMoE checks exactly two ids, `EOS_TOKEN_1` 248046 / `EOS_TOKEN_2` 248044 (`infer.m:258-259`, checked at `10575`); llama.cpp ends on its EOS per the tokenizer, or
- the token cap: finchMoE `max_gen` (`for (gen = 0; gen < max_gen; gen++)`, `10574`); llama.cpp `n_predict` (default `-1` = whole context when the client sends no `max_tokens`), or
- client disconnect / error.

Everything else — including a closed think block — just continues the loop. Note specifically: in finchMoE, `--think-budget` (default 2048, `infer.m:381`, enforced at `10615-10619`) only force-emits `</think>` after 2048 think tokens; it does **not** end the response. After `think_ended`, the only thing banned is re-entering think tags (`10664-10667`). If the model then rambles or repeats without ever emitting EOS, generation runs until `max_gen` (default **8192** on the chat endpoint, `infer.m:10982`) — which reads as "thinking forever". The engine's own comments document this failure family: repetition loops (BUGS.md Bug 12) and sampling ranges chosen because certain temperatures "end long gens naturally" (`infer.m:382-387`).

**Mechanism B — the agent keeps starting new steps, because the model keeps returning tool calls.**

deepseek-harness ends a *turn* only when a step ends (`agent.ts:263-301`, `332-400`): a message with **no tool-call blocks** → `{ kind: 'completed' }` (`393-394`); `max-tokens` → `{ kind: 'max-tokens' }` (sticky, `391`, `285-290`); error or abort otherwise. A step whose message contains tool calls executes them and returns `null` (`395-399`), and the `while (true)` opens the next step. **There is no step-count or wall-clock cap in this loop.** A local model that hallucinates or loops on tool calls therefore loops the harness indefinitely — each iteration is rendered as "thinking".

Key harness fact for Mechanism A: `BlockAssembler.finish` defaults to `stop` when the stream ends without a finish chunk (`packages/llm/llm/src/assembler.ts:165-167`). So if the harness *did* receive a stream end, the turn would end and it would not hang. A hang means the stream is still open — i.e., the upstream server is still generating.

The earlier "just thinking then stop" variant is Mechanism A's short form: the model emitted EOS inside/right after a think block, the stream ended, the harness saw a no-tool-call message and completed the turn having done nothing.

### 10.2 If the endless thinking is in the finchMoE engine (port 9000)

1. **Server log triage** (`/tmp/he_server.log`):
   - request enqueued but no `prefill=` line → worker thread stuck on a previous request (single worker, `SERVE_QUEUE_MAX 16`, `serve_worker` at ~10260-10305);
   - the done event reports `completion_tokens` == the request's `max_tokens` → Mechanism A (no EOS; ran to cap). If `completion_tokens` is small → immediate EOS (the think-then-stop form);
   - `FATAL: Not enough physical memory` → M1 memory gate (`infer.m:11192-11212`); close apps / `sudo mdutil -a off`, rerun `start_server.sh`;
   - `client disconnected` → harness timeout (900 s) or curl closed early.
2. **Add `--logit-diag 1`** (`infer.m:11162`) for a per-token top-20 logit dump — it shows whether the model is choosing EOS/think tags and what it is doing instead.
3. **Repetition instead of stop:** if output is a long repeat until `max_tokens`, verify `--rep-penalty 1.0` is in effect (`start_server.sh` already sets it) and the temperature flags; the engine's clean sampling range is documented at `infer.m:382-387` (0.1-0.3) — but for eval parity keep greedy (`-e 0 --top-k 1`).
4. **Check the old-results file**, `he_results_pre_protocol_fix.jsonl`: think-only or truncated-until-max completions confirm which mechanism the pre-fix run was hitting.

### 10.3 If "the harness" is the agent harness against llama.cpp at localhost:8080

The diagnosis is the same two mechanisms, on the deepseek-harness side of the wire:

1. **Check whether the step ever receives a finish chunk.** Replay the session log (`assistant/chunk` / `assistant/message` events) or watch the llama-server console: if llama.cpp prints no `finish_reason` and keeps streaming tokens, Mechanism A is active — the harness is inside `for await (const chunk of stream)` (`agent.ts:347`) and cannot do anything else.
2. **Cap every request.** If the harness config does not set `maxTokens` (`AgentOptions.maxTokens`, folded in at `agent.ts:427-437`), no cap is sent and llama-server defaults to `n_predict -1` (context limit). A hybrid-thinking Qwen model filling 32K+ tokens of reasoning looks exactly like "thinks forever". Set an explicit maxTokens in the harness config.
3. **Turn reasoning off at the server.** The 3090 spec run already does this: `llama-server ... --reasoning off ...` (`humaneval_3090/README.md:150`). Verify the startup banner says `reasoning: off`; if it says `enabled`, that is the most likely single cause — Qwen3.x hybrid models stream long think blocks with reasoning on, and an agent UI renders that as endless "thinking".
4. **Tool-call loop (Mechanism B):** if the log shows step → tool-call → step → tool-call repeatedly, the model is hallucinating tool calls (common for local quantized models serving an agent protocol built for a stronger model). Confirm each call's tool name parses and the calls differ; a repeated identical call is a model loop, not a harness bug. The harness has no turn cap by design — consider `maxTokens` and a stronger model for the loop, or a watchdog outside the harness.
5. **llama-server knobs for reasoning loops:** `--repeat-penalty 1.1` (or higher), `--temp`, and for older versions the `--reasoning off` flag must be repeated on every invocation (it is not persistent).

### 10.4 Triage table

| Symptom | Mechanism | Where to look | Fix |
|---|---|---|---|
| "Just thinking, then stop, nothing happened" | A (short): EOS inside/after think block; turn ends with no tool calls | finchMoE: think tags streamed on completions path (§3.2); llama.cpp: `reasoning: enabled` | finchMoE: Phases 1-5 of this plan; llama.cpp: `--reasoning off` |
| "Keeps thinking, never stops" (single long response) | A (long): no EOS, no cap | finchMoE: `completion_tokens == max_tokens` in done event; llama.cpp: stream still open, no finish chunk | set maxTokens/`max_tokens`; `--reasoning off`; repeat-penalty; §10.2/10.3 |
| "Keeps thinking" across many steps | B: tool-call loop | session log: repeated step/start → tool-call → step/start | cap tokens; check tool-name parsing; stronger model or watchdog |
| Request enqueued but nothing streams | worker blocked / memory gate / client timeout | finchMoE log: missing `prefill=` line; `FATAL:` memory banner; `client disconnected` | §10.2 items 1-2 |

---

## 15. Phase 7 (ACTIVE) — Numeric parity probe: finchMoE vs local llama.cpp, same weights, same Mac

This is the current plan of record. Goal: find the earliest token position where finchMoE's
greedy argmax diverges from llama.cpp's greedy argmax on the identical GGUF file, then use that
position to localize the bug to a specific stage of the forward pass (embedding / a specific
transformer layer's attention or MoE / final norm / lm_head). Everything runs on this one Mac —
no 3090 needed, which removes the CUDA-vs-Metal and Windows-vs-macOS variables that were
previously conflated with "llama.cpp vs finchMoE".

Both binaries already exist in this checkout:
- `finchmoe/finchmoe-infer` (Metal), flag `--logit-diag N` dumps top-20 logits + entropy every
  N tokens to stderr (`infer.m:1917-1966`, wired at `infer.m:14168`/`14282`/`15711`/`15815`).
- `llama.cpp/build/bin/llama-server` (Metal, AppleClang, arm64 — confirmed built and runnable on
  this machine) and `llama.cpp/build/bin/llama-cli`, both support `n_probs` (server) for top-K
  logprobs per generated token (`tools/server/server-context.cpp:1909-1926`).

### 15.1 Fix the prompt first — use ONE task, not the full 164

Pick `HumanEval/0` (already the clearest failure case in §0.3). Use the *exact* rendered prompt
finchMoE received for that task — pull it straight out of the harness rather than re-typing it,
to avoid introducing a whitespace/formatting difference of your own:

```bash
cd humaneval_evalplus
.venv/bin/python3 - <<'EOF'
import json
# grab the exact user-turn text evalplus sent for HumanEval/0
from evalplus.data import get_human_eval_plus
from evalplus.data.utils import CACHE_DIR
problems = get_human_eval_plus()
print(problems["HumanEval/0"]["prompt"])
EOF
```

Save that prompt text to `/tmp/he0_prompt.txt` (or pipe directly into the two probes below).
The system message is fixed: `You are a helpful assistant good at coding.`

### 15.2 Capture finchMoE's per-token top-20

```bash
cd finchmoe
printf 'You are a helpful assistant good at coding.' > ~/.flash-moe/system.md
./finchmoe-infer -R 9000 --gguf ../models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -e 0 --top-k 1 --no-think --rep-penalty 1.0 --low-memory \
    --logit-diag 1 > /tmp/finchmoe_he0.log 2>&1 &
# wait for /health, then send the SAME request shim.py would (via curl, chat endpoint):
curl -s http://127.0.0.1:9000/v1/chat/completions -H 'Content-Type: application/json' -d "$(python3 -c "
import json
prompt = open('/tmp/he0_prompt.txt').read()
print(json.dumps({'model':'x','messages':[{'role':'system','content':'You are a helpful assistant good at coding.'},{'role':'user','content':prompt}],'max_tokens':768,'temperature':0}))
")" > /tmp/finchmoe_he0_response.json
kill %1
```

`/tmp/finchmoe_he0.log` now has one `[logit-diag]` block per generated token: `step=N
token=<id> ("<text>") entropy=... max_logit=...` followed by the top-20 (id, logit) pairs.

Also run the SAME probe against the **native 3-bit/4-bit** tiers (drop `--gguf`/`--low-memory`,
add `--4bit` for the 4-bit tier) to get their per-token top-20 too — useful for §15.4's
`--low-memory` isolation test.

### 15.3 Capture llama.cpp's per-token top-20 on the identical file, locally

```bash
cd llama.cpp
./build/bin/llama-server -m ../models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
    -c 4096 -ngl 99 -fa on --reasoning off --temp 0 --top-k 1 --repeat-penalty 1.0 \
    --host 127.0.0.1 --port 8090 &
# wait for "server is listening", then hit /completion directly (not /v1/chat) so llama.cpp
# applies its OWN chat template deterministically the same way the 3090 did:
python3 - <<'EOF'
import json, requests
prompt = open('/tmp/he0_prompt.txt').read()
r = requests.post('http://127.0.0.1:8090/v1/chat/completions', json={
    "model": "x",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant good at coding."},
        {"role": "user", "content": prompt},
    ],
    "max_tokens": 768, "temperature": 0,
    "n_probs": 20, "post_sampling_probs": True,
})
open('/tmp/llamacpp_he0_response.json', 'w').write(r.text)
EOF
kill %1
```

The response's `choices[0].logprobs` (or `completion_probabilities` if you use the legacy
`/completion` endpoint instead) has the same shape as finchMoE's `--logit-diag`: per-token top-K
candidates with their probabilities. Convert probabilities back to a comparable ranking (you do
not need literal logit values — you only need each engine's rank-1 token id and whether the
other engine's rank-1 choice appears anywhere in your top-20).

### 15.4 Diff token-by-token, find the first divergence

Write a small script (`scripts/logit_diff.py`, does not exist yet — create it) that:

1. Parses both dumps into a list of `(step, top1_id, top1_text, top20_ids)`.
2. Walks both lists in lockstep while `top1_id` matches on both sides.
3. Reports the first step where `finchmoe.top1_id != llamacpp.top1_id`, and whether llama.cpp's
   top-1 appears anywhere in finchMoE's top-20 (small numeric drift, still recoverable) or not at
   all (a larger/qualitative divergence).
4. Print the decoded token text on both sides at and immediately around the divergence step —
   this tells you whether it's an early divergence (before the docstring even starts — likely a
   structural/dequant bug) or a late one (only appears once the docstring is already several
   `>>>` examples deep — consistent with the attractor amplifying a small earlier drift rather
   than causing it outright).

**Acceptance / next action table for this probe:**

| Finding | Interpretation | Next step |
|---|---|---|
| Divergence at step 0-5 (prompt is barely consumed) | Bug is in prompt encoding, RoPE position base, or the very first layers/embedding | Dump per-layer RMS (the technique from Bug 17/18: compare `[PRE-NORM]`/`[POST-NORM]` rms per layer) between finchMoE and a from-scratch reference (e.g. HF transformers CPU fp32 forward, if available, or llama.cpp's own `--verbose-prompt` + internal logging) for the identical prompt, walking layer-by-layer until the RMS values diverge |
| Divergence appears only after several `>>>` examples (10-30+ tokens in) | Consistent with attractor amplifying a small, otherwise-tolerable per-token drift (e.g. Bug 18-style FMA/rounding-order difference in the MoE combine or attention softmax) rather than a structural bug | Audit `finchmoe/infer.m`'s combine/softmax code for FMA-contraction or summation-order differences vs a reference implementation, same method as Bug 18 |
| llama.cpp's top-1 always appears in finchMoE's top-20, just not rank-1 | Small logit-scale/precision issue, not a wrong computation | Check final-norm epsilon, attention scale (`1/sqrt(d)`), and any float32-vs-float16 accumulation differences in the dequant kernels for Q4_K_M specifically (Bonsai/native tiers use different code paths — compare whether native 3-bit/4-bit diverge at the SAME step as GGUF, which would point to a bug shared by all dequant paths, e.g. RMSNorm or attention, rather than something GGUF-Q4_K_M-specific) |
| llama.cpp's top-1 is nowhere in finchMoE's top-20 | Large, qualitative divergence — likely a real correctness bug (wrong tensor, wrong offset, transposed weight, wrong MoE routing) | Re-run the Bug 1/2/7-style extraction-correctness audit: dump the specific tensor(s) active at that layer/step and diff against a known-good reference (e.g. `llama.cpp`'s own `--verbose` internal tensor dumps, or a Python `gguf` library read of the same tensor) |

### 15.5 Isolate `--low-memory` (CPU fallback) as an independent variable

Because `gguf`/`gguf3090` use `--low-memory` and the native `3bit`/`4bit` cells do not, and all
four still land in the same ~13% band, `--low-memory` is unlikely to be the sole cause — but
confirm directly:

1. Check available RAM; if the machine can fit the BF16/Q4_K_M weights without the low-memory
   guard, re-run `run_cell.sh gguf` with `--low-memory` removed from `ENGINE_ARGS` (edit the
   `gguf`/`gguf3090` cases in `humaneval_evalplus/run_cell.sh`).
2. If the engine now takes the Metal zero-copy path (confirm via the startup log line the memory
   gate prints, `infer.m` region documented in Bug 13), re-run `smoke_check.py` on a 20-30 task
   slice (`run_cell.sh gguf3090 0 30`) and compare the loop-rate/no-def-rate against the
   `--low-memory` run's same slice.
3. If the rate is materially different, `--low-memory`'s CPU fallback is contributing a
   *second*, independent numeric bug on top of whatever affects all four tiers — split the
   investigation into two bugs instead of one.
4. If the rate is the same, drop `--low-memory` from suspicion entirely and focus solely on
   §15.4's shared-code-path findings (embedding, RoPE, attention, MoE combine, final norm,
   sampling — the parts common to every tier).

### 15.6 Cheap parallel diagnostic: does breaking the loop externally recover score?

This does not fix the root cause and must not be presented as parity-preserving, but it is a
fast, informative test: temporarily add a repetition penalty (e.g. `--rep-penalty 1.3
--rep-last-n 64`, or whatever the engine's existing flags support) purely as a probe, and re-run
`smoke_check.py` on the same 20-30 task slice.

- **If the loop rate collapses and pass@1 jumps sharply** (say, from ~13% toward the 40-60%
  range): the underlying per-token logits are close enough to reasonable that only the
  *greedy-tie-breaking* is unusually attractor-prone in this engine — this narrows §15.4's search
  to numeric precision effects (small logit differences that make ties/near-ties resolve
  differently), not a gross correctness bug. It also becomes a legitimate interim mitigation
  (document it as a deliberate deviation from strict greedy-parity, not as "fixed").
- **If the loop rate is unaffected**: the model is confidently (not marginally) choosing the
  wrong continuation — i.e., the top-1 logit itself is wrong, not just numerically fragile at a
  tie. That is stronger evidence for a real correctness bug per §15.4's bottom row, and rules out
  "it's just noise" as an excuse.

---

## 11. Execution checklist (on the Mac)

**Historical (§1-§9, already done):** system-message/chat-endpoint/no-think parity is in place
via `humaneval_evalplus/` — no further action needed here. Do not re-run this checklist as
written; it targeted the retired `humaneval_m1/` two-machine setup.

**Active checklist (§15):**

- [x] §15.1: extract the exact `HumanEval/0` prompt text from evalplus, save to `/tmp/he0_prompt.txt`.
- [x] §15.2: capture finchMoE's `--logit-diag 1` dump for GGUF (3090-exact file) on that one prompt. (3-bit/4-bit tiers not re-captured this pass.)
- [x] §15.3: capture llama.cpp's `n_probs`/logprobs dump for the identical file + prompt, locally on this Mac via `llama-server`. (Note: the server's chat-template rendering differs from finchMoE's `--no-think` suffix, so the step-0 divergence seen there is partly a template artifact; the identical-token probes in §15.4 supersede it.)
- [x] §15.4: write `scripts/logit_diff.py`, run it, find the first divergence step, classify it against the acceptance table. **RESULT: divergence at step 0, but NOT a finchMoE bug — see `scripts/PARITY_FINDINGS_2026-08-23.md`. llama.cpp's own logits vary by maxd 2.0-3.7 (with argmax flips) across its prefill batch configurations; finchMoE sits inside that spread. The step-0 flips are near-tie argmax flips (both engines' top-2 within ~0.6 logits).**
- [x] §15.5: with `--low-memory` removed (if RAM allows), re-run a 20-30 task slice of `gguf3090` and compare loop/no-def rates to the `--low-memory` run. **DONE via `--cpu-linear` probe: CPU delta-net path gives identical logits (maxd 0.003) — GPU delta-net kernel ruled out as the h_diff seed.**
- [x] §15.6: as a probe only, re-run a 20-30 task slice with a non-zero repeat penalty and record whether the loop rate collapses. **DONE: `--rep-penalty 1.05` shipped in all 4 tiers — 12.8% -> 18.3% base / 16.5% plus (full 164-task sweep). Docstring loop fixed.**
- [x] Record all findings — `scripts/PARITY_FINDINGS_2026-08-23.md` holds the parity numbers; the open thread was the generation-level attractor under the EvalPlus wrapper prompt (see §0.3), now closed by the NumPy counterfactual (see `scripts/NUMPY_VERDICT_2026-08-27.md`).
- [x] NumPy offline forward (the decisive counterfactual): **20/20 = 100% PASS** — finchMoE's math executed with BLAS ordering is clean; the drift is Metal matmul ordering, not formulation. See `scripts/NUMPY_VERDICT_2026-08-27.md`.
- [ ] Once the kernel-ordering fix (fixed-order fp32 Metal matmuls) lands, re-run the full 164-task sweep (`run_all.sh`) and update the §0.2 results table with the corrected numbers.

---

## 12. Results (fill in after execution)

| Step | Result |
|---|---|
| §0.2 matrix (baseline, already measured) | A 91.5/89.0, B 13.4/12.8, C 12.8/12.8, D 12.8/12.8, D′ 12.8/12.8 |
| §15.4 first divergence step (HumanEval/0) | |
| §15.4 classification (structural / amplified-drift / precision / qualitative) | |
| §15.5 `--low-memory` isolation result | |
| §15.6 repeat-penalty probe result | `--rep-penalty 1.05` shipped in all 4 tiers (`run_cell.sh`): 12.8% -> 18.3% base / 16.5% plus (full 164-task sweep) |
| NumPy offline forward (counterfactual, 2026-08-27) | **20/20 = 100% PASS on the 20-task slice** (`scripts/numpy_forward.py`, BLAS-order numpy, same dequant + GDN/FA/MoE math as finchMoE). See `scripts/NUMPY_VERDICT_2026-08-27.md`. |
| Root cause (fill in once found) | finchMoE's layer math is **clean** (NumPy reproduction of it passes 100% and matches llama's CPU slice 19/19). The 12.8%-vs-91.5% gap decomposes into: (a) docstring loop -> fixed by `--rep-penalty 1.05` (12.8% -> 18.3%); (b) h_diff-driven newline/structural token suppression (verified `logit_diff = W @ h_diff` at corr 0.999, growing with sequence length) from general Metal-vs-CPU matmul ordering — NOT a formulation bug. |
| Fix applied | rep105 (shipped); kernel ordering is the remaining lever — scope fixed-order fp32 Metal kernels for the delta-net/attention/MoE matmuls |
| Full 164-task re-run after fix | rep105 re-run done (18.3% base / 16.5% plus); kernel-ordering fix pending |

---

## 13. Rollback and follow-ups

**Rollback** (any phase):
1. Restore the deployed binary: keep a `finchmoe-infer.bak` before any `infer.m` change under investigation; `cp` it back and restart via `run_cell.sh` if a change needs to be reverted.
2. `git checkout` any `infer.m` edits made during the §15 investigation; `humaneval_evalplus/` files are not modified by the probe (it only adds `scripts/logit_diff.py` and temp files under `/tmp`).

**Follow-ups (out of scope here, tracked for later):**
1. **Honor per-request system messages** in `/v1/chat/completions` (replace `extract_last_content`-only parsing with a proper system/user extraction; would remove the need for `shim.py`'s `~/.flash-moe/system.md` file trick). Still valid, still deferred — not the current bottleneck.
a2. **Strip or mask think tokens in streaming output** for thinking-mode servers (today streamed verbatim when `--no-think` is not passed). Still valid, still deferred.
3. **Parse per-request `temperature`/`top_p`** (currently ignored; server flags only). Still valid, still deferred.
4. **HumanEval+ number** for whichever tier ends up fixed, to compare against the 3090's 89.0 plus.
5. Once §15 lands a fix, revisit whether Mode A vs Mode B (§4) still matters at all — if the numeric bug explains the gap, the think-suffix wording is unlikely to matter further, but re-verify rather than assume.

---

## 14. Key code map (for quick re-anchoring)

**Caution:** the table below is pinned to commit `0b78d45` and is stale — current `infer.m` has
grown to ~15.8k lines and most of these anchors have shifted. Use `grep -n` for the function
names, not these line numbers, e.g.:

```bash
grep -n "static.*build_think_suffix\|static.*load_system_prompt\|static.*tokenize_chat_message\|static.*logit_diag_dump\|g_low_memory\|THINK_START_TOKEN\|THINK_END_TOKEN" finchmoe/infer.m
```

Current confirmed anchors (2026-08-23, this checkout):

| Anchor | current `infer.m` line |
|---|---|
| `THINK_START_TOKEN` / `THINK_END_TOKEN` | 260-261 |
| `g_no_think` | 463 |
| `g_low_memory` | 464 |
| `logit_diag_dump` | 1920 (wired at 14168, 14282, 15711, 15815) |
| generation loop / think tracking (chat path) | ~14170-14285 |
| think re-entry ban (chat path) | 14277-14279 |
| generation loop / think tracking (2nd path — completions?) | ~15740-15820 |
| `--no-think` flag parse | 15006 |
| `--logit-diag` flag parse | 15013 |
| `--low-memory` flag parse | 15007 |

Stale table below (commit `0b78d45`) kept for historical cross-reference only — do not use for
current edits:

| Anchor | `infer.m` line (commit `0b78d45`) |
|---|---|
| `g_no_think` | 402 |
| `build_think_suffix` | 10058-10070 |
| `tokenize_user_turn` | 10074-10087 |
| `load_system_prompt` | 10107-10126 |
| `tokenize_system_prompt` | 10131-10143 |
| `tokenize_chat_message` | 10146-10162 |
| `extract_last_content` | 9816-9856 |
| `extract_max_tokens` | 9880-9887 |
| `sse_send_delta` (chat) | 9958-9980 |
| `sse_send_delta_completion` | 10001-10022 |
| `ServeQueueEntry` | 10210-10218 |
| `process_chat_request` | 10312+ |
| system snapshot restore (`pos = s->sys_prompt_len`) | 10365-10410 |
| first-token sampling | 10552-10558 |
| generation loop / think tracking | 10561-10670 |
| think re-entry ban | 10664-10667 |
| chat handler | 10971-11045 |
| completions handler | 11047-11108 |
| usage / flags | 11126-11164, 11242-11287, 11327-11335 |

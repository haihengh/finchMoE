# Protocol Improvement Plan — finchMoE HumanEval Prompt-Protocol Parity

**Status:** PLAN ONLY — not yet executed. Execute on the Mac (M4 builds, M1 runs), per the checklists at the end.
**Pinned to:** finchMoE checkout `0b78d45` (`test(humaneval): add Ornith-1.5-35B-A3B results, ties Qwen3.6-35B-A3B`). All `infer.m` line numbers below refer to that commit; re-check anchors if the file has moved since.
**Scope:** `finchmoe/infer.m` (engine) + `humaneval_m1/generate.py` + `humaneval_m1/start_server.sh` (harness). No changes to `humaneval_3090/` (spec baseline, frozen).

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

## 11. Execution checklist (on the Mac)

### On the M4 (build machine)

- [ ] `cd ~/Desktop/code/finchMoE && git log --oneline -1` — source is at `0b78d45` or newer and contains all §5 hunks (`grep -n "system-prompt" finchmoe/infer.m`).
- [ ] Apply §5.1, §5.2, §5.3 edits to `finchmoe/infer.m`.
- [ ] `cd finchmoe && make` — clean build.
- [ ] Copy `finchmoe-infer` to the M1 deploy dir (back up old binary on the M1 first, `chmod +x` after copy).

### On the M1 (run machine)

- [ ] Update `humaneval_m1/generate.py` (§6.1) and `humaneval_m1/start_server.sh` (§6.2).
- [ ] Move old `he_results.jsonl` aside.
- [ ] Restart server via `start_server.sh` (memory-gate retry loop; tail `/tmp/he_server.log` until `Listening`).
- [ ] Run smoke tests (§7) — all EXPECT lines pass, in particular `grep -c "<think>"` = 0.
- [ ] Run the parity gate (§8) and record the two prompt-id lists.
- [ ] Launch full run (§9): `python3 generate.py`, then `python3 evaluate.py`.
- [ ] Record results in this file's §12, or in the repo's results table alongside the 3090 baseline.

---

## 12. Results (fill in after execution)

| Step | Result |
|---|---|
| Smoke §7.3 `<think>` count | |
| Prompt-ids parity (§8), Mode A/B | |
| Full-run records | /164 completed, errors |
| pass@1 base (m1) | |
| 3090 baseline | 91.5 base / 89.0 plus |

---

## 13. Rollback and follow-ups

**Rollback** (any phase):
1. Restore the deployed binary: `cp finchmoe-infer.bak finchmoe-infer` on the M1, restart server.
2. `git checkout` the §5 edits; `generate.py`/`start_server.sh` revert is a separate file-level revert (keep the old `he_results_pre_protocol_fix.jsonl` as the "before" record).

**Follow-ups (out of scope here, tracked for later):**
1. **Honor per-request system messages** in `/v1/chat/completions` (replaces `extract_last_content`-only parsing with a proper system/user extraction; requires reworking the startup system-prompt pre-cache or per-request full-message tokenization via `tokenize_chat_message`, `infer.m:10146-10162`).
2. **Strip or mask think tokens in streaming output** for thinking-mode servers (currently streamed verbatim, `10629-10633`) if a client needs reasoning-free responses without `--no-think`.
3. **Parse per-request `temperature`/`top_p`** (currently ignored; server flags only).
4. **HumanEval+ evaluation** on the M1 results (port `evalplus.evaluate` or extend `evaluate.py` with plus tests) to compare the 89.0 plus number.
5. **Mode B decision record**: if Mode B was used for the final run, log why in this file.

---

## 14. Key code map (for quick re-anchoring)

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

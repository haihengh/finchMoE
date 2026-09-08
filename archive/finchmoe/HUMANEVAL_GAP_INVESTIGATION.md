# HumanEval Gap Investigation — findings and plan

**Status 2026-08-21**: benchmark complete (5 cells), three root-cause hypotheses
eliminated with evidence, three remaining. This document is the input for planning
the next debugging session.

## 1. The measurement

EvalPlus 0.3.1, the exact harness and protocol that produced the 3090's published
numbers (`humaneval_3090/`), rebuilt on this Mac (`humaneval_evalplus/`). Scoring
parity is proven: our install reproduces **0.915 / 0.890** on the 3090's own sample
file, so any score difference comes from generation, not evaluation.

| Cell | Engine → weights | base / plus |
|---|---|---|
| A | llama.cpp (a30273376) → 3090's lmstudio Q4_K_M | **91.5% / 89.0%** |
| B | FinchMoE → 3-bit native | 13.4% / 12.8% |
| C | FinchMoE → 4-bit native | 12.8% / 12.8% |
| D | FinchMoE → our Q4_K_M GGUF (local quant, no imatrix) | 12.8% / 12.8% |
| D′ | FinchMoE → **the 3090's exact GGUF** | 11.6% / 11.6% |
| E1 | llama.cpp (this Mac, CPU) → **the 3090's exact GGUF** | **90.9% / 88.4%** |

**Every weight format through FinchMoE converges to ~13%.** The exact file that
scores 91.5% under llama.cpp scores 11.6% through FinchMoE — and, via E1, scores
90.9% through llama.cpp *on this same Mac* (CPU-only, same harness, same prompts).
The deficit is in the FinchMoE engine or its serving protocol — not the
quantization, not the harness, not the hardware, not the weights. Failure signature
is uniform across all four FinchMoE cells: docstring-reproduction loops under
greedy decoding (7–13% repetition loops, 10–18% sanitize to imports-only, 14–16%
exhaust the 768-token cap).

## 2. Eliminated hypotheses (with evidence)

### ✗ H1 — Chat template differs from llama.cpp's
**VERIFIED IDENTICAL, byte for byte.** Extracted `tokenizer.chat_template` from the
3090's GGUF (7,764-char Jinja) and rendered it for the exact EvalPlus conversation
(system + fenced user message) using llama.cpp's own template engine
(`test-chat-template`). Compared against the string our engine builds in
`tokenize_chat_message` (infer.m:13739):

```
<|im_start|>system\nYou are a helpful assistant good at coding.<|im_end|>\n
<|im_start|>user\nPlease provide a self-contained Python script ...<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The last line is the notable one: llama.cpp's template itself injects the same
empty think block when `enable_thinking` is false (the Jinja tail is literally
`{%- if enable_thinking is defined and enable_thinking is false %}...<think>\n\n</think>\n\n`),
which is what the 3090's `--reasoning off` selects. Our `--no-think` reproduces
this exactly. **The template hypothesis is dead.**

### ✗ H2 — Tokenizer maps tokens differently
**VERIFIED IDENTICAL.** Diffed `vocab.bin` against the 3090 GGUF's embedded
`tokenizer.ggml.tokens` (248,320 tokens):
- all **248,044** of our vocab entries match the gguf's strings at the same IDs — **0 mismatches**
- all 26 added tokens (248,044–248,069, including `<|im_start|>`, `<|im_end|>`,
  `<think>` at 248,068, `</think>` at 248,069) match — **0 mismatches**
- the gguf has 250 extra tail tokens (248,070–248,319) — multimodal/audio/TTS
  tokens irrelevant to text

Combined with the template match, **the prompt token sequence our engine feeds the
model in D′ is identical to what llama.cpp fed on the 3090.** Anything downstream
of the prompt is where the 80 points went.

### ✗ H3 — Quantization is the bottleneck
**DISPROVEN by D′.** The 3090's own GGUF through FinchMoE scores like every other
tier. (Still true: our local Q4_K_M has no imatrix and is objectively worse than
the lmstudio one — but that difference is invisible under FinchMoE.)

## 3. Remaining suspects

Ranked by prior probability given the evidence:

1. **Long-horizon decode accumulation.** All our bitwise/parity validations were
   short (1, 13, 90 tokens; raw completion). A small per-step error — attention
   accumulation, a KV-cache line read after N steps, a numeric drift in the
   recurrence — would be invisible at 90 tokens and catastrophic at 700+.
   **Predicts exactly what we see**: coherent starts, then entropy collapse into
   loops, uniform across weight formats, at every tier.

2. **Serve-mode state handling.** The chat serve path does things the raw path
   doesn't: system-prompt snapshot prefill at startup, KV restore per request,
   turn rollback, `--no-think` logits bans. A subtle state bug would degrade every
   request uniformly. (Note: the old harness at 22–32% also used serve mode, so
   this has never been isolated from the engine proper.)

3. **Sampling-path subtlety.** We run `-e 0 --top-k 1 --rep-penalty 1.0`.
   llama.cpp's greedy at T=0 differs in details worth auditing (rep-penalty
   application order, EOS min-keep, tie-breaking in argmax — the gate-score 0-ULP
   tie flip is a documented quirk of our path). Less likely to cause loops than
   (1), but cheap to check.

4. **Long-shot: first-token prefill corruption.** If the chat prefill (system
   snapshot + user turn) produced wrong logits even at step 1, all bets are off.
   A single chat-protocol logit dump eliminates 1-vs-4 decisively.

## 4. Experiment plan (cheapest first)

### E1 — DONE: llama.cpp reproduces on this Mac — 90.9% / 88.4%
Same GGUF, same harness, `llama-server --reasoning off -ngl 0` (CPU-only: Metal
OOMs on this 16 GB machine even with auto layer fit). Ran 2026-08-21→22
(~17 h at ~0.9 tok/s). Result **90.9% / 88.4%** — statistically identical to the
3090's 91.5% / 89.0%. The harness, file, and prompts are vindicated on this
hardware; the bug is definitively in FinchMoE. The 5-problem smoke was already
clean (5/5, zero loops, zero truncations — including HumanEval/0, which every
FinchMoE tier loops on). Continue to E2.

### E2 — RUN 2026-08-22: first-token differential — prefill is CLOSE but not bit-exact
Infrastructure: `e2_diff/` (e2_build_prompt.py, e2_run.sh) + `FINCHMOE_DUMP_PROMPT_TOKENS`
(infer.m:1539) + `llama.cpp/logit_dump.cpp` (batch cap raised 128→256 for the
180-token chat prompt).

Prompt: the HumanEval/0 chat conversation (system + EvalPlus user + assistant
turn with `--no-think`), tokenized by our engine's own tokenizer into 180 IDs —
the exact IDs both engines decode. First-token logits after prefill,
3090 GGUF on both sides:

| metric | value |
|---|---|
| cosine similarity | **0.9856** |
| max abs diff | 2.06 |
| argmax | **MATCH** (71093) |
| top-20 overlap | 19/20 |

**Interpretation.** The first sampled token is identical — so the prefill is
*mostly* right, and the 13%-vs-91% gap cannot be explained by a gross prefill
corruption. But cos 0.9856 is well below the 0.9982 the raw-completion
cross-validation achieved (90 tokens, OUR gguf, 2026-08-18). Two candidate
explanations: (a) the 3090 file's Q6_K output weight / different tensor layout
exercises a less-validated dequant path, or (b) the chat prompt's length/chunking
hits a prefill divergence the 90-token test never did. Either way there is a
REAL numeric difference to chase.

**Next (E2b):** extend `logit_dump` to generate N tokens and dump per-step
logits; do the same on our side with `--dump-logits` append mode (both already
write per-step). Diff the argmax sequences token by token and find the first
divergent token. That token's step pinpoints whether the divergence is born in
prefill (first step) or grows during decode (later steps).

### E3 — Sampling audit (hours)
Port llama.cpp's exact greedy pipeline semantics for comparison: rep-penalty
application order (llama.cpp applies it inside the sampler with 1-token history —
we apply ours before argmax with our own history window), EOS min-keep, argmax
tie-break. Verify our `-e 0 --top-k 1` is genuinely argmax on the *penalized*
logits with no hidden filter (the `--no-think` logits bans only touch think
tokens).

### E4 — Serve-state isolation (day)
Run the chat protocol against the engine's raw `/v1/completions` path with the
template hand-applied by the harness (no serve chat path, no snapshot restore,
no rollback). If quality jumps, the bug is in serve state handling (suspect 2);
if identical, it's the decode core (suspect 1). Cheap to run once E1 confirms
the harness is clean.

### E5 — Reproduce the loop locally (hours)
Pick one looping problem (HumanEval/0 loops on docstring examples in every
tier). Run it with `--logit-diag N` to watch entropy and top-token drift across
the loop. A loop with *rising* top-token probability and falling entropy is a
pure greedy attractor (model-level); a loop with oscillating/stable entropy is
accumulated numerical error. Distinguishes "our engine computes a wrong model"
from "the model genuinely loops under greedy."

## 5. Open questions to settle in planning

- Does the 3090's llama.cpp server strip or prefill `<think>` specially when
  `--reasoning off` (e.g. does it *remove* think tokens from output the way our
  shim does, or does it prevent them at sampling time with a grammar/ban)?
  Whatever it does, our D′ is equivalent at the prompt level but we have not
  verified output-side equivalence.
- What exactly does llama.cpp's T=0 greedy do that ours doesn't (E3)? The 3090
  numbers are greedy too — the same model does not loop there.
- Our raw-completion cross-validation (cos 0.998213, argmax match) was 90 tokens
  on our own GGUF — does that validation cover serve mode's system-prompt
  snapshot + restore path? (Read the 2026-08-18 cross-val notes: it covered the
  GGUF decode path, likely not the serve chat path.)

## 6. References

- `humaneval_evalplus/README.md` — full protocol, harness, and per-cell analysis
- `humaneval_3090/README.md` — the reference run
- `infer.m:13739` — `tokenize_chat_message` (verified identical to llama.cpp's render)
- `infer.m:1539` — `FINCHMOE_DUMP_PROMPT_TOKENS` cross-reference hook
- `llama.cpp/logit_dump.cpp` — reference logit dumper to extend
- Memory: `evalplus-3090-control` (result), `gguf-crossval-complete-2026-08-18`
  (90-token parity: cos 0.998213, argmax match, top10 10/10)

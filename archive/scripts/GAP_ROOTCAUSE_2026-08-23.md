# HumanEval Gap — Root-Cause Findings (2026-08-23)

## TL;DR

The 12.8%-vs-91.5% HumanEval gap is NOT a weights, protocol, or step-0
numerics problem. It is a **generation-trajectory problem**: finchMoE's greedy
decoding systematically produces structurally broken Python (missing newlines,
1-space indents) and falls into a docstring-completion loop that never closes
the ` ```python ` fence. evalplus then truncates those responses to
"from typing import List" and scores them as failures.

- llama.cpp on THIS Mac (same 3090-exact GGUF, CPU-only) scores **90.9%**.
- finchMoE baseline (rep-penalty 1.0): **12.8%**.
- finchMoE with `--rep-penalty 1.05`: docstring loops **completely gone**;
  20-task slice 40% -> **50%** (whitespace defect remains).

## Evidence chain

### 1. The failure mode (reproduced exactly)

Stored eval raw (2026-08-21) and a fresh harness run (2026-08-23) are
char-identical for HumanEval/0: 1428 chars, opens ` ```python `, writes the
def + a MUTATED docstring, then invents `>>> has_close_elements([1.2,2.3,...])`
examples forever and **never emits the closing fence** or the function body.
evalplus's `sanitize()` then returns "from typing import List"
(NO_ENTRY_POINT) -> fail.

The exact evalplus prompt matters: the user message is
`"Please provide a self-contained Python script ... markdown code block:\n```python\n{prompt}\n```"`
(fence is ` ```python `, NOT ` ``` `). The project's own probe scripts
(`e2_build_prompt.py`, `run_finchmoe_logit_probe.sh`) used the wrong fence
(` ``` `) — all E2/he0/sweep parity probes were on a slightly different
prompt than the real eval.

### 2. The loop is greedy + no repetition penalty

`--rep-penalty 1.0` disables the engine's penalty (its built-in default is
1.15). With the penalty re-enabled at 1.05, **0/20 slice tasks loop**; the
same inputs at 1.0 loop. 1.2 over-penalizes (15 fence fragments).

### 3. The remaining defect: whitespace/structural tokens

Even with the loop fixed, ~50% of outputs have broken whitespace:
- `result = []for i in range(...)` (missing newline - the model emits the BPE
  token `:result` / continues same line)
- ` sorted_numbers = sorted(numbers)` (1-space indent)
- `current depth -= 1` (space inside identifier)

These come from model token choices (verified at the token-stream level with
FINCHMOE_SERVE_DEBUG: the engine's decode/SSE assembly is correct; the model
emits newline-free tokens like id 91877 `:result`).

### 4. The logit-level mechanism

finchMoE's final-position logits have a **sequence-length-growing, consistent
suppression of newline/indent tokens** vs llama.cpp:

| length | newline-token mean diff (finch - llama) |
|---|---|
| 1 | +0.03 |
| 10 | +0.08 |
| 50 | -0.05 |
| 153 | -0.28 (individual tokens -0.4 to -0.95: `    \n` -0.95, `  \n` -0.50) |

The magnitude on those tokens is within llama.cpp's own batch-config spread
(b153 vs per-token varies whitespace tokens by up to 1.16), but llama.cpp's
variation is not consistently directional while finchMoE's is: newline
tokens down, space tokens slightly up. Greedy decoding converts this into
structurally broken code. The bias grows with sequence length -> it lives in
the stateful part of the forward pass (GDN recurrence / attention), not the
static layers.

## Parity investigation conclusion (supersedes earlier notes)

At the FINAL-LOGITS level finchMoE is within llama.cpp's own batch-config
spread (maxd 1.7-2.6 vs llama-vs-llama 2.0-3.6) — but that proximity is
misleading for generation quality: the differences concentrate on
structural tokens (newlines/indent) where argmax flips have outsized
consequences, and greedy decoding amplifies them. "Within the spread" at the
logit level does NOT mean "equivalent output quality". The earlier
PARITY_FINDINGS doc stands for the logit-level claim; this doc adds the
generation-level consequence.

## Reproducers

- Fast loop repro (curl): /tmp/repro_exact2 run - request /tmp/repro_exact.request.json
  (exact evalplus prompt + "You are a helpful assistant good at coding." via
  HOME=/tmp/.flash-moe/system.md; result: 1428-char unclosed-fence loop).
- Harness repro: /tmp/repro_harness.sh (shim.py path, char-identical output).
- Fix check: /tmp/fixtest.sh (rep105/110/120 vs baseline).
- Slice scorer: humaneval_evalplus/score_slice.py (faithful evalplus
  sanitize + base_input/base_output evaluation).

## Open threads / candidate fixes for the whitespace bias

1. Locate the source of the newline-token suppression in the GDN/attention
   accumulation (per-layer logit or hidden-state analysis vs llama.cpp's
   autoregressive reference).
2. Sampler-level compensation: bias newline/indent tokens by +0.3-0.5 (hack,
   testable in an afternoon; matches the observed -0.3..-0.95 deficit).
3. Full 164-task sweep with rep-penalty 1.05 to quantify the loop-fix-only
   improvement (running: finchmoe-rep105-full).

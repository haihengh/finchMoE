# §15.4 Parity Probe Results (2026-08-23)

Conclusion in one line: **finchMoE's logits sit INSIDE llama.cpp's own
configuration spread.** The per-step numerical "divergence" between finchMoE
and llama.cpp on this model is the same size as llama.cpp's divergence from
itself when you change its prefill batch size.

## Setup

- Model: `Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf` (lmstudio-community,
  the 3090-exact file; a ggml mix of 371 q4_K + 61 q6_K + 301 f32 tensors).
- Machine: Apple M4, 16 GB. All llama.cpp runs CPU-only (`-ngl 0`), zero-copy
  mmap — the GPU path OOMs on this box (20 GB of file-backed expert weights;
  reproduced with -c 8192 and with -c 4096 -b 256; `-cmoe` is worse, it
  allocates an 18.6 GB CPU_REPACK copy). See `run_llamacpp_logit_probe.sh`.
- Input: identical 153-token sequence (system+user chat template) fed raw to
  both engines (`logit_dump`/`llama-eval-callback` with `LLAMA_EVAL_TOKENS`,
  finchMoE `-p` + `--dump-logits`).

## The numbers (final-position logits, 248320-dim, L=153 tokens)

| comparison | cos | maxd | rmsd | argmax |
|---|---|---|---|---|
| finchMoE vs llama per-token (ub=1) | 0.9854 | 2.11 | 0.388 | 40 vs 71093 |
| finchMoE vs llama b153/u153 | 0.9914 | 1.74 | 0.294 | 40 vs 40 |
| finchMoE vs llama b256/u16 | 0.9898 | 2.08 | 0.322 | 40 vs 40 |
| finchMoE vs llama logit_dump (auto) | 0.9881 | 2.59 | 0.346 | 40 vs 71093 |
| **llama per-token vs llama b153/u153** | 0.9855 | **2.10** | 0.387 | 71093 vs 40 |
| **llama per-token vs llama b256/u16** | 0.9853 | **2.54** | 0.388 | 71093 vs 40 |
| **llama per-token vs llama logit_dump** | 0.9807 | **3.55** | 0.445 | 71093 vs 71093 |
| **llama b153/u153 vs llama b256/u16** | 0.9892 | **2.05** | 0.331 | 40 vs 40 |
| llama logit_dump vs llama b256/u16 | 0.9854 | **3.68** | 0.384 | 71093 vs 40 |

llama.cpp's own spread across batch configurations: maxd 2.05-3.68, with
argmax flipping between 40 ("I") and 71093 ("```"). finchMoE sits inside that
spread and agrees with the majority (server, b153, b256) on argmax 40.

## Length sweep (identical-token prefixes, logit_dump vs finchMoE)

| L | cos | maxd | rmsd |
|---|---|---|---|
| 1 | 0.99983 | 0.20 | 0.036 |
| 2 | 0.99942 | 0.46 | 0.088 |
| 5 | 0.99930 | 0.49 | 0.101 |
| 10 | 0.99887 | 1.04 | 0.179 |
| 50 | 0.99744 | 1.47 | 0.229 |
| 153 | 0.98806 | 2.59 | 0.346 |

Drift grows with sequence length — this is the GDN (delta-net) recurrent
state accumulating chunking/precision differences, in BOTH engines. It is not
specific to finchMoE: llama.cpp's own chunked-vs-autoregressive spread grows
the same way.

## What was ruled out

- **Embedding**: finchMoE's token embedding is bit-exact against a reference
  dequant of the GGUF file (corr 1.0, maxd 0.0 for token 248045). llama.cpp's
  eval-callback "model.input_embed" dump was garbage (ggml build_forward_select
  buffer artifact) — do not use the eval-callback dump for input_embed or for
  per-layer residual comparisons; the final-logit dumps from it are valid.
- **GGUF type mapping**: the file is q4_K/q6_K, which finchMoE's importer
  implements; no Q4_K_M-vs-Q4_K confusion (the "Q4_K_M" filename is the
  imatrix recipe, not a ggml type).
- **Prefill chunking inside finchMoE**: `--prefill-chunk 8/4/2/1` bit-identical,
  chunk 0 within maxd 0.003. Internal consistency is not the issue.
- **GDN math**: conv1d+SiLU, q/k RMS-vs-L2 norm (equivalent given the 1/sqrt(K)
  scale), softplus(alpha+dt)*ssm_a decay, sigmoid(beta), and the delta rule
  (decay-first, then delta from decayed state, rank-1 update, readout from the
  updated state) all structurally match llama.cpp's build_delta_net_* paths.

## Consequences

1. The earlier E2 "maxd=2.06" result and the §15.4 "first divergence at step
   0" are NOT evidence of a finchMoE-specific numerical bug. They are the
   GDN-chunking variance of the reference itself, which a plain
   `logit_dump` (auto ubatch 16) exhibits against its own sibling runs.
2. Step-0 argmax flips ("I" vs "```" vs "To") happen at near-ties: both
   engines' top-2 sit within ~0.6 logits. Greedy decoding amplifies such
   flips into different trajectories, but on the he0 probe both trajectories
   produced a correct solution.
3. The 12.8%-vs-91.5% HumanEval gap is therefore NOT explained by a ~2-logit
   per-step offset. Open threads, in order of likelihood:
   a. **Generation-level attractor**: §0.3 documents finchMoE writing 5-6
      mutated docstring examples and never closing the ``` fence within 768
      tokens under the EvalPlus wrapper prompt. That is a long-horizon greedy
      loop, not a step-0 numerics issue. Re-check with the wrapper prompt +
      "You are a helpful assistant good at coding." system (the he0 probe used
      the raw prompt + default system and did not show the loop).
   b. Repetition-penalty / logit-bias handling on the generation path.
   c. If a per-step parity target is still wanted, the fair reference is
      llama.cpp with ubatch=1 (autoregressive GDN) — or any fixed config, with
      the expectation that maxd ~2 is the floor for this model on this machine.

## How to reproduce

```bash
# finchMoE dump-logits (identical token input)
cd finchmoe
./finchmoe-infer --gguf Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -p /tmp/sweep_153.bin --dump-logits /tmp/fm.bin --no-think --low-memory \
  -e 0 --top-k 1 --rep-penalty 1.0 -t 1

# llama.cpp logit_dump (chunked-16 reference, the outlier)
./llama.cpp/build/bin/logit_dump <gguf> "$(cat /tmp/sweep_153.csv)" /tmp/lcpp.bin

# llama.cpp per-token autoregressive reference (eval-callback, local patch)
cd llama.cpp
LLAMA_EVAL_TOKENS=/tmp/sweep_153.csv LLAMA_TENSOR_DUMP=/tmp/pertok.bin \
  LLAMA_EVAL_FILTER=result_output LLAMA_EVAL_PER_TOKEN=1 \
  ./build/bin/llama-eval-callback -m <gguf> -p x -ngl 0 -b 1 -ub 1 -c 2048 -lv 0
# (last result_output record = final logits)

# compare with the he0 sweep harness: scripts/sweep.sh equivalent in /tmp
```

Local llama.cpp modifications (debug instrumentation only, not committed):
`common/debug.cpp` (LLAMA_TENSOR_DUMP full-tensor dump, ask returns the filter
match), `examples/eval-callback/eval-callback.cpp` (LLAMA_EVAL_TOKENS,
LLAMA_EVAL_FILTER, LLAMA_EVAL_PER_TOKEN).

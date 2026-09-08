# NumPy offline forward — verdict (2026-08-27)

## The decisive experiment

`scripts/numpy_forward.py` executes finchMoE's *exact* layer math — the same GGUF
dequant (Q4_K/Q6_K, verified bit-identical to the previous scalar reference), the same
GDN delta-net / full-attention / MoE formulation, the same norms, the same final
RMSNorm + lm_head — but with **numpy/BLAS-ordered matmuls** instead of finchMoE's
Metal kernels. It is the counterfactual that isolates arithmetic *ordering* from
arithmetic *formulation*.

## Result

- **Slice 0-20 (evalplus-exact prompts, greedy, `--rep-penalty 1.05`): 20/20 = 100% PASS.**
- Output file: `/tmp/numpy_slice.raw.jsonl` (each task flushed as completed; the run is
  resumable — completed `task_id`s are skipped on restart).
- HumanEval/0 in the full run reproduces the standalone smoke run character-for-character
  (675 chars, 1132s vs 1125s) — fully deterministic.

## Cross-engine comparison on the same 20 tasks

| Engine | Score |
|---|---|
| llama.cpp CPU (`-ngl 0`), this Mac, same 3090-exact GGUF | 19/19 = 100% (slice) |
| **NumPy forward (finchMoE's math, BLAS ordering)** | **20/20 = 100%** |
| finchMoE Metal (identical GGUF) | ~45% best slice / 18.3% official 164-task |

## Interpretation (per the user's decision rule)

>=~80% on the slice => **finchMoE's math is clean; the drift is purely kernel
ordering**. That is what we got: 100%.

- The layer-by-layer verification already showed finchMoE reproduces its own math
  (corr >= 0.999 per layer vs the NumPy forward, which is bit-faithful to finchmoe's
  formulation) and that the achievable corr vs llama is finchmoe-level (~0.985-0.999),
  with the L34 spike at positions 118-135 being inherent model sensitivity that
  finchMoE itself exhibits vs llama.
- The measured failure mechanism stands: `h_diff` (rms ~0.03 pre-norm at L38, 0.447
  post-norm) propagates to `logit_diff = output.weight @ (norm(h_fm) - norm(h_ll))`
  (corr 0.999, predicted == measured rms 0.3882), suppressing newline/indent logits by
  -0.4..-1.0 and flipping greedy whitespace decisions into broken Python.
- Since the *identical math* in BLAS order scores 100%, the h_diff seed is the Metal
  matmul ordering (general Metal-vs-CPU matmul reduction order), not a formulation,
  dequant, RoPE, GDN, or MoE bug. `--cpu-linear` already ruled out the GPU delta-net
  kernel as the seed; the NumPy result confirms it.

## Consequence / next lever

Scope **fixed-order fp32 Metal kernels** for the delta-net, attention, and MoE matmuls
(torch/llama.cpp-style reduction ordering) with the confidence that they can reach the
~90% band. rep105 stays as the shipped config until then: 18.3% base / 16.5% plus on
the full 164-task sweep.

## What was fixed/optimized in the tool along the way

- Tokenizer: classic GPT-2 byte-fallback table (printable ASCII -> literal chars, only
  control/extra bytes -> U+0100..) plus the qwen35 pre-tokenization regex; verified
  **0 mismatches** vs llama.cpp's own tokenization of the same 171-token prompt.
- `output_norm.weight` was accidentally skipped by the non-expert loader (name filter
  caught `output*`) — fixed.
- Expert LRU was thrashing (cache max 180 < 960 keys/token) — raised, then stored as
  fp16 to keep RSS in check on this 16 GB Mac.
- Dequant: zero-copy memmap views, uint8/uint16 intermediates, no concatenate copies —
  Q4_K/Q6_K ~2-3x faster, verified bit-identical (corr 1.0, maxdiff 0) against the
  previous reference.
- MoE expert dequant+matmul runs in a ThreadPoolExecutor (numpy releases the GIL on
  the large ops) — bit-identical output, generation ~1.6 s/token E2E (prefill ~3 s/tok,
  ~19 min/task, 5.3 h for the 20-task slice).

## Files

- `scripts/numpy_forward.py` — the offline forward (load, verify, slice modes).
- `humaneval_evalplus/score_slice.py` — faithful evalplus scorer (sanitize + exec +
  run against base_input/base_output).
- `/tmp/numpy_slice.raw.jsonl` — the 20-task raw output.

## Addendum (2026-08-28): truncated-weight counterfactual — 20/20

The earlier verdict showed the identical math in BLAS order scores 100%
where finchmoe scores ~50%. A follow-up counterfactual tested whether the
difference is the **weights** (finchmoe's GGUF importer BF16-truncates F32
small tensors: norms, conv1d, dt_bias, routers, alpha/beta) or the
**arithmetic**: the numpy forward was re-run with ALL those staged ops
truncated exactly as finchmoe stages them (`NUMPY_STAGED_TRUNC=1`, env-gated
in numpy_forward.py, `bits >> 16` truncation).

Result: **20/20 = 100%** — identical to the F32-weight run. The exact
weights that finchmoe's engine consumes give numpy a perfect slice score,
while finchmoe's own engine scores 50% (10/20) on the same 20 tasks and
GGUF. Combined with the 11-probe scalar-op/order sweep (DEEP_DIVE
2026-08-24 §"11-probe refutation"), the conclusion is:

- The h_diff / slice failure is NOT seeded by the BF16 weight staging.
- It is NOT any single always-CPU op or accumulation order (norm weights,
  output_norm, delta GPU-vs-cblas, GDN chain, RMS f32/f64, FA order, MoE
  routing, RoPE f32, conv1d order, expert cache, all-staged truncation).
- It is a distributed ~0.001-level-per-layer arithmetic difference in
  finchmoe's always-CPU implementation vs numpy/BLAS, amplified through the
  position-carrying GDN/residual state (L38 p0 0.003 → p135 0.18). No
  localized single-op fix exists in this (low-memory, mostly-CPU) eval
  config; the path to the llama band is matching the CPU arithmetic to
  numpy/BLAS across the board, not a weight-staging change.

Artifacts: /tmp/np_trunc_slice.raw.jsonl (20 tasks, 100%), scored with
humaneval_evalplus/score_slice.py.

## Addendum (2026-08-28, Step 1): expert-path bit-parity — EXONERATED

The last unverified execution path (routed experts: dequant + 3 GEMMs +
SwiGLU + weighted combine, the largest per-layer payload) was checked
bit-for-bit on the eval config:

- Engine GPU fused Q4_K expert kernel vs CPU gguf_cpu_matvec: **moe_out
  identical, maxd 1e-8** (rms 1e-9).
- numpy dequant_rows + BLAS reconstruction from finchmoe's OWN h_post and
  routing vs finchmoe's moe_out: **maxd ≤ 7e-7, rms ≤ 2e-7**; 0/2048
  elements exceed 1e-8 absolute. Top-8 routing agrees exactly.

Every path (GPU kernel, CPU fallback, numpy/BLAS) is machine-precision
identical given the same input. The expert path is NOT the seed. Combined
with the 11-probe sweep and the truncated-weight 20/20 slice, ALL single-op
and single-kernel hypotheses in this (low-memory, mostly-CPU) eval config
are closed. The verdict: the finchmoe-vs-numpy/llama divergence is a
distributed ~0.001-level-per-layer arithmetic difference in finchmoe's
always-CPU implementation, amplified through the position-carrying
GDN/residual state. No localized fix exists; the strategic options are
(A) ship as-is at ~50%, (B) route eval through the numpy forward at ~100%,
(C) wholesale CPU-arithmetic alignment to numpy/BLAS.

Probe artifacts: /tmp/famprobe_baseline.dump.bin, /tmp/famprobe_expcpu.dump.bin
(678,414,240 B each), /tmp/pt_experts.bin (routing + h_post, layers 0-1),
/tmp/expert_parity3.out. Engine source edit (env-gated h_post in the
FINCHMOE_GGUF_DBG routing dump, infer.m ~line 10104) is debug-only and
shipped-off.

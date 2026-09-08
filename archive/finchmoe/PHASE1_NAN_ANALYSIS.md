# Phase 1 Quantization — Resolution Analysis (2026-08-12)

**Status**: NaN RESOLVED. Root causes were data corruption, not GPU kernels.

---

## Root Causes Found (in order of discovery)

### Bug 1: 8-bit routing gate dispatched through 4-bit kernel

The routing gate (`mlp.gate.weight`, 8-bit packed, 4 values/u32) and
`shared_expert_gate.weight` (8-bit) were dispatched through the 4-bit
`dequant_matvec_4bit_v3` kernel in the fused CMD2 GPU path.
The CPU fallback path correctly used `bits=8`.

- Symptom: GPU routing CosSim=0.79 vs CPU (gate), -1.0 (shared gate).
- Fix: added `bits` field to `BatchMatvecSpec`; `gpu_encode_batch_matvec` /
  `gpu_batch_matvec` now select `matvec_8bit` (dequant_matvec_8bit) when bits==8.
- Verified: standalone CosSim=1.0 with real weight data.

### Bug 2: Expert packed files repacked from the WRONG model

`../models/Qwen3.6-35B-A3B-4bit/packed_experts/` was built by running
`repack_experts.py` against the stale cwd `expert_index.json` which pointed
at the **MTP model** (`Qwen3.6-35B-A3B-4bit-mtp`), whose routed experts are
**2-bit** ([256,512,128]) — while the repack used the 4-bit layout
(component size 524288 = 2× the source expert stride of 262144).
Result: overlapping reads of 2-bit data written into a 4-bit layout = garbage
experts, with huge/NaN values in early slots.

- Fix: `python3 generate_expert_index.py --model ../models/Qwen3.6-35B-A3B-4bit --output ../models/Qwen3.6-35B-A3B-4bit/expert_index.json`
  then `python3 repack_experts.py --index ../models/Qwen3.6-35B-A3B-4bit/expert_index.json`
- Verified byte-identical to source safetensors (0 mismatches).

### Bug 3: FP16→BF16 conversion corrupted genuine BF16 scales/biases

This model (`Qwen3.6-35B-A3B-4bit`) stores **genuine BF16** scale/bias data
with dtype='BF16' (the old "mlx-community stores FP16" rule does NOT apply
to it). The unconditional FP16→BF16 conversion in both `extract_weights.py`
and `repack_experts.py` misinterpreted BF16 values as FP16 and re-encoded:
e.g. BF16 -0.0041 (bits 0xBB87) → "FP16" -0.945 → BF16 -0.9375.

This corrupted ALL 4-bit and 8-bit scale/bias tensors in the extracted
weights AND the repacked experts. Evidence: `nibble*scale+bias` with the raw
BF16 bytes reproduces the BF16 reference model weights at CosSim=0.9978;
the converted values give CosSim=0.34.

- Fix: `--fp16-scales` flag (default OFF) added to both scripts.
- Verified: all tensor families (qkv, z, a, b, o_proj, shared expert, q/k/v,
  routing gate 8-bit) match the BF16 reference model at CosSim 0.995-1.000.

### Bug 4: NaN propagation through 0-weight experts

With garbage expert data (Bug 2), `moe_combine_residual` computed
`0.0 * NaN = NaN` for unselected experts, poisoning `buf_moe_hidden`.
This made the NaN visible only at the LAST prefill token (intermediate
tokens discard their deferred expert results). Fixed transitively by Bug 2.

---

## Current Status — PHASE 1 COMPLETE (2026-08-13)

- No NaN anywhere; engine **bit-exact** vs the numpy GDN reference
  (CosSim 1.000000 per stage); all kernels 1.0 vs CPU reference.
- **11.4-12.4 tok/s** decode (M4, K=8, greedy) with 1.95 GB weights
  (4-bit GDN tier + 3-bit experts, both default) and 2.25 GB peak GPU.
- Speed arc: 3.85 → 6.90 (self-quant) → 7.32 (CMD1+CMD2 fuse) →
  9.09 (3-bit experts) → 11.4 (4-bit GDN tier). The fused-GPU wait is
  weight memory bandwidth, not dispatch latency (proven by neutral
  micro-fusions).
- **Open quality issue (2026-08-13)**: typo'd/ambiguous prompts degrade
  into repetition loops. llama.cpp Q4_K_M on the SAME prompt reasons
  correctly → the base model is fine; the problem is our weights being
  quantized from the marginal `2bit-dense-v2` variant (double-quantized
  experts). Fix: re-quantize from pristine `Qwen3.6-35B-A3B-bf16`.
- Prefill = decode speed (~12 tok/s, 789 tokens = 66s) — batched GPU
  prefill is the agentic-workload priority.
- Server multi-turn continuation corrupts after turn 1 (stateless
  fallback active); MTP draft head math wrong (α=0%).
- **Output quality of the MLX-community 4bit model is the model's own
  property**, not an engine bug:
  - The GDN gated-norm sits at the eps-knee: 24/32 value-heads have
    delta_out rms < sqrt(eps)=1e-3, so tiny input differences are
    amplified ~2.5× per layer. Both the quant and BF16 setups share this
    structure; the MLX-4bit model's quantization noise lands worse.
  - llama.cpp Q4_K_M (same base model) generates fluently — the
    architecture tolerates 4-bit, but per-model quantization choices land
    differently in the sensitive regime.
  - MLX could not be run on this 16 GB machine (its own Metal OOM) to
    directly confirm — venv ready at /tmp/mlxvenv.
- **Recommendation**: self-quantize the known-good model (2bit-dense-v2's
  BF16 non-expert weights, fluent in our engine) with our own
  `quantize_non_experts.py` — controlled calibration + the now-verified
  engine, instead of the community MLX-4bit model.

## Reproduction

```bash
cd finchmoe
python3 extract_weights.py --model ../models/Qwen3.6-35B-A3B-4bit --output ./quant_test
ln -sf quant_test/model_weights.bin model_weights.bin
ln -sf quant_test/model_weights.json model_weights.json
./finchmoe-infer -t 40 -k 8   # NaN-free; quality = model property
```

## Debug Instrumentation Added (env-gated, inert by default)

- `FINCHMOE_DUMP_HPOST=1` — appends per-layer h_post to /tmp/hpost_dump.bin
- `FINCHMOE_DUMP_HIDDEN=1` — appends post-combine hidden to /tmp/hidden_dump.bin
- `FINCHMOE_DUMP_STAGES=1` — layer-0 token-0 stage dump to /tmp/stage_dump.bin
- `-X` adds ATTN-DBG/FAST-DBG/OPROJ-DBG/LC-DBG prints (layers 0-2)
- `-I` now dumps logits for EVERY decode step (append; step 0 truncates)
- `finchTool/tools/debug_gdn_reference.py` — numpy GDN reference for stage cross-validation

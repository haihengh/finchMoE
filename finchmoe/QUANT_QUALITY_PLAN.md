# Quantization Quality Plan — closing the HumanEval gap

**Status**: PLANNED 2026-08-19. Data-driven; nothing here is speculative.
Resume by starting Phase 0.

## Background: what the measurements say

Four-way HumanEval matrix (same M4, same harness, raw completions T=0 unless noted):

| engine + model | tier | decode (M4) | pass@1 |
|---|---|---|---|
| finchMoE + Qwen 3.6 35B | 4-bit native experts (`4bit-dense` pack) | ~5.5 tok/s | 53/164 = 32.3% |
| finchMoE + Qwen 3.6 35B | 3-bit native experts (default) | 10-16 tok/s | 49/164 = 29.9% |
| finchMoE + Qwen 3.6 35B | GGUF Q4_K_M | 3.4-5.5 tok/s | 36/164 = 22.0% |
| turbo-fieldfare + Gemma 4 26B-A4B | their 4-bit repack | ~3.5 (their mini) | 142/164 = 86.6% (chat protocol) |

Reference points that bound the goal:

- **FP16 reference for THIS model: 61.6%** (fraQtl's matched-FP16 table, N=164, ±3pp).
  A perfect quantization recovers to ~61.6%, NOT 86.6%. The rest of the gap to
  turbo-fieldfare is model choice — Gemma 4 26B is a stronger code model.
- **Community compressed build (fraQtl): 64.0%** — a proven recipe already
  exists for this exact model. Adopting/studying it is cheaper than
  re-deriving (Phase 3).
- Our custom pipeline beats llama.cpp Q4_K_M by +10/+13 points on the same
  model, so the pipeline is sound; the *quantization precision and scale
  calibration* are the ceiling.
- Chat-template protocols degrade our 3-bit tier badly (11.0% think,
  0/10 no-think) — instruct adherence is a quant-level casualty; watch it as
  a secondary signal in every phase.

## Constraints that shape the design

1. **Expert I/O budget**: decode is disk-bound. Expert bytes/token:
   3-bit 1.38 MB × 8 × 40 = ~440 MB; 4-bit 1.77 MB × 8 × 40 = ~566 MB.
   Raising expert precision costs decode speed directly (measured: 4-bit
   halves the tok/s). Non-expert tensors are streamed once per layer and
   stay page-cache-resident — their precision is nearly free.
2. **Engine format support (already present)**: the manifest
   (`model_weights.json`) carries per-tensor `bits`/`group_size`; non-expert
   kernels exist for BF16 (`bits=0`), 4-bit (`bits=4`), 8-bit (`bits=8`);
   expert packs are 1/2/3/4/8-bit slab formats (`packed_experts[_Nbit]/`).
   So mixed-bit allocation is a REQUANT-ONLY change — no engine work.
3. **Per-group scales are the kernel contract**: expert slabs store
   per-group scales/biases (see `repack_experts.py`); the dequant kernels
   read them. Activation-aware calibration must stay per-group to avoid
   kernel changes (per-channel would need format + kernel work — only
   justified if Phase 0 proves channels are the binding constraint).

## Phase 0 — Per-tensor quantization audit (1-2 days, analysis only)

Goal: a ranked map of where the quant loses fidelity, so Phase 1 spends
precision only where it pays.

Implementation:
1. Extend the clean-rebuild validation into a full audit script
   `finchmoe/finchTool/tools/quant_audit.py`:
   - Load the BF16 reference tensors (models/Qwen3.6-35B-A3B-bf16/
     model-*.safetensors) and the quantized pairs from the packed experts
     (`packed_experts_3bit`, `packed_experts_4bit` via the layer files) and
     `model_weights_quant.bin` (manifest-indexed).
   - For every tensor (1148): dequant to FP32 (reuse the dequant logic from
     `verify_clean_rebuild.py`), compute CosSim vs the BF16 reference, and
     report per-role aggregates:
     - `ffn_gate_exps/up_exps/down_exps` (experts, per layer)
     - `linear_attn.in_proj_qkv/z/b/a`, `gated_norm`, conv1d, A_log, dt_bias
     - full-attn `q/k/v/o`, router `ffn_gate` (the 256×2048 gate),
       shared expert `ffn_shared_gate/up/down`, embed + lm_head.
2. Outputs:
   - Worst-10 list per role and overall.
   - A "precision spend" table: for each role, expected bytes to raise it
     one notch (8-bit vs 4-bit vs 3-bit) and its share of per-token I/O.
3. Decisions this gates:
   - Which non-expert tensors go 8-bit/FP16 in Phase 1.
   - Whether expert gates need a 4-bit carve-out (e.g., gate+up 4-bit,
     down 3-bit) — note the GGUF lessons: gate/up type uniformity matters
     for the fused kernels.
4. Success gate: the audit explains ≥80% of the 32.3 → 61.6 gap by role
   before Phase 1 starts. If experts look uniform and healthy, the gap is
   in scale calibration (→ skip to Phase 2).

## Phase 1 — Mixed-bit allocation (1-2 days, requant-only)

Target the roles Phase 0 flags. Expected candidates (confirm with data):
- Router gate (`ffn_gate`, 256×2048) → 8-bit or FP16 (tiny; routing
  fidelity drives expert selection — highest leverage per byte).
- Attention q/k/v and GDN gated_norm + beta/alpha → 8-bit if degraded.
- Shared expert (already 4-bit?) → 8-bit if flagged.
- Experts stay 3-bit/4-bit per the I/O budget; consider gate+up 4-bit /
  down 3-bit asymmetric packs only if the audit shows gate/up degradation.

Implementation:
1. Re-run the requant for the flagged tensors at the chosen bits:
   `quantize_non_experts.py` (supports per-tensor bit selection — extend if
   needed), regenerate `model_weights_quant.bin` + manifest.
2. Expert-side changes (if any): `quantize_model.py` + `repack_experts.py`
   with the asymmetric plan; keep the slab layout constants
   (`GATE_W_OFF_*` etc. in `infer.m`) in sync — or reuse the existing
   `--4bit`/`--3bit` flag machinery per pack.
3. Verification gates (all must pass before default):
   - Per-tensor CosSim vs BF16 (target: ≥0.98 for flagged tensors).
   - Typo-prompt quality check (the clean-rebuild protocol).
   - 20-task HumanEval probe, raw protocol: expect a measurable gain over
     29.9% (3-bit) / 32.3% (4-bit).
   - Decode-speed check: non-expert changes must not move the 10-16 tok/s
     (3-bit tier).
4. Success gate: 3-bit ≥ 35%, 4-bit ≥ 40%.

## Phase 2 — Activation-aware per-group scales (AWQ-lite, 2-3 days, scripts only)

Goal: better scales without touching the kernel contract.

Implementation:
1. Build a calibration set: HumanEval prompts + the routing log
   (`FINCHMOE_ROUTING_LOG`-style captures we already have) + a code corpus
   (e.g., 200-500 Python files / the training split of HumanEval-style
   snippets). Record per-layer activation statistics (RMS per channel /
   per group) by running the engine with a `FINCHMOE_ACT_STATS` dump
   (new ~30-line env-gated probe in `infer.m`, same pattern as the S8
   probes — remember the lesson: probes must be verified on a REAL prompt).
2. Modify `quantize_model.py` / `quantize_non_experts.py`: per-group scale
   = f(group max |W|, group activation RMS) — the AWQ-lite formula, not
   plain max-abs. Keep the packed format unchanged (scales stay per-group).
3. Re-quantize experts at 3-bit and 4-bit with the new scales, run the same
   verification battery as Phase 1.
4. Success gate: 3-bit ≥ 40-45%, 4-bit ≥ 50-55% (approaching the FP16-adjacent
   ceiling). If Phase 2 falls short, the residual is 3-bit precision loss in
   experts themselves — then consider the fraQtl recipe (Phase 3) before any
   per-channel format work.

## Phase 3 — Study the fraQtl compressed recipe (parallel, 1 day)

The fraQtl compressed build scores 64.0% on this model. Actions:
- Fetch the model card + methodology (HF: fraQtl/Qwen3.6-35B-A3B-compressed;
  they disclose the matched-FP16 table and calibration corpus).
- Identify their quant scheme (formats per role, calibration set, any
  activation-aware step) and diff against ours.
- Adopt the portable parts into our requant pipeline (phases 1-2 may
  already cover them).
- Optionally: run their weights through our engine if the format maps
  (safetensors + our extractor) for a direct engine-level comparison.

## Explicitly out of scope

- **QAT**: requires training runs; no infra. Note that turbo-fieldfare's
  quality comes from Gemma 4 being a strong natively-released model — their
  runtime repack is still PTQ. QAT is a model-release decision, not an
  engine project.
- **Per-channel scale format changes**: only if Phases 0-2 show channels
  are the binding constraint.

## Regression battery (run at every phase gate)

```bash
cd finchmoe
# bitwise sanity: the 90-token soak + typo prompt on the native tier
./finchmoe-infer -m . --prompt "<typo prompt>" -t 40
# HumanEval probe (raw protocol)
# server: ./finchmoe-infer -R 9000 -m . -e 0 --top-k 1 --no-think --rep-penalty 1.0
# harness: cd ../humaneval_m1 && python3 generate.py --limit 20 --port 9000 \
#          --results he_results_probe.jsonl && python3 evaluate.py --results he_results_probe.jsonl
# decode speed check (3-bit tier must hold 10-16 tok/s)
```

## Resume notes (session state)

- 4-bit wiring already exists: `finchmoe/model_weights_4bitd.json`
  (model field → models/Qwen3.6-35B-A3B-4bit-dense) + `--4bit`.
- Harness modes: `--chat`, `--tf` (turbo-fieldfare, non-streaming JSON),
  `--results FILE`; scorer `evaluate.py` handles restated functions.
- Published results: `humaneval_m1/he_results*.jsonl` (gitignored artifacts)
  and the README matrix (commit e40fa43).
- All evals complete; machine idle. Servers stopped.

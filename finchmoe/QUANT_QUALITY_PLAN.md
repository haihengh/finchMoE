# Quantization Quality Plan — closing the HumanEval gap

**Status**: Phase 0 COMPLETE 2026-08-20. Phase 1 = E1 (GDN8) → E2 (near-lossless)
→ E3' (protocol probes). E3 (pristine 4-bit) dropped — see why below.
E1/E2 builds DONE 2026-08-20; the post-reboot verification session hit the
2026-08-20 kernel panic (two concurrent audit runs at 36 GB on the 16 GB
mini). Guardrails are now in the tools (job lock + memory guards + streamed
refs) — resume with the full sweep below.

## Phase 0 — COMPLETE 2026-08-20: per-tensor audit

Tool: `finchTool/tools/quant_audit.py` — dequantizes every tensor from the
LIVE pack (quant_clean_pi bin + packed_experts_3bit, plus the old 4-bit pack)
and computes CosSim vs the pristine BF16 reference. 259 non-expert tensors,
0 errors. Full per-tensor table: `/tmp/quant_audit_final.txt`.

| role | bit | mean cos | min cos |
|---|---|---|---|
| embed / lm_head | 8-bit | 0.99978 | — |
| linear_attn qkv / z / out (30 layers × 3) | 4-bit | 0.9949–0.9955 | 0.9949 |
| full_attn q/k/v/o (10 layers × 4) | 4-bit | 0.9920–0.9955 | 0.9920 |
| shared_expert gate/up/down (40 × 3) | 4-bit | 0.9924–0.9950 | 0.9924 |
| mtp head | 4-bit | 0.9947–0.9953 | 0.9947 |
| router gates, norms, GDN a/b/A_log/dt_bias | BF16 | 1.0000 | — |
| experts, 3-bit pack (40 × 256) | 3-bit | **0.9815** | **0.9536** |
| experts, 4-bit pack (40 × 256) | 4-bit | 0.9959 | 0.9891 |
| experts, 8-bit pack (E2, 40 × 256) | 8-bit | **0.99996** | **0.99988** |

### Verdict

1. **No role carries the gap at weight level.** The only sub-0.99 role is the
   3-bit experts (mean 0.9815). Every non-expert role ≥0.992.
2. **Quantization buys small, roughly linear deltas**: 3-bit → 4-bit experts
   (+0.014 cos) bought +2.4 HumanEval pts. Extrapolating, FP16-everything over
   the 4-bit tier buys only ~+4-6 pts → ~36-38%, not 61.6%.
3. **The 32.3 → 61.6 residual is a protocol cliff, not a weight cliff.** E2
   below pins our harness's FP16-adjacent ceiling; if it lands ~35%, the lever
   is prompt/sampling/think protocol (E3'), not quantization.
4. **Layout-contract finding (corrects 2026-08-18 note)**: the LIVE
   `model_weights.bin` (→ quant_clean_pi) carries the engine-contract head-pair
   interleave — v-heads de-interleaved evens-then-odds, audit unpermutes with
   PINV[j] = 16j mod 31. `quant_clean/` is the faithful HF layout. GGUF is a
   third (converter) order. All three are correct for their consumers; the
   earlier "BF16 dir is rotated" attribution was wrong — the rotation is in the
   LIVE pi file's v-blocks.
5. **The 32.3% (4-bit) run reused the LIVE non-expert bin + the OLD
   2bit-dense-v2-lineage 4-bit pack**, which measures cos 0.9959 (near-pristine)
   vs the BF16 reference. A fresh "pristine 4-bit" requant (old E3) has no
   expected gain → **E3 dropped**.
6. Phase 0's gate ("explains ≥80% of the gap by role") is NOT met — the plan's
   fallback triggers: *experts are uniform and healthy → the gap is in scale
   calibration and protocol*, tested empirically by E1/E2 and Phase 2.

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

## Phase 1 — Requant experiments (requant-only, no engine work)

Three experiments; each gated by a 20-task probe before any full 164 run:

- **E1 — GDN8 tier**: requant non-experts with `FINCHMOE_GDN8=1` (GDN
  qkv/z/out → 8-bit; embed/lm_head already 8-bit; everything else unchanged).
  Isolates whether the 4-bit GDN channel (cos ~0.995) costs anything through
  the 40-layer recurrence. Bin ~1.96 → ~3.2 GB (safe: <5 GB transient rule).
  Decode if adopted: ~11.4 → ~9.1 tok/s. ~30 min rebuild.
- **E2 — near-lossless baseline (the decisive experiment)**: E1's bin + 8-bit
  expert pack (`repack_experts.py --bits 8`, ~38 GB on the SATA drive, 1-2 h).
  Calibrates OUR harness's FP16-adjacent ceiling against fraQtl's 61.6%.
  Decode ~2.5-3 tok/s → full 164 run ~3.5 h (background, alone on the machine
  per the crash-history rule). Splits the hypothesis: "quantization is the
  ceiling" vs "protocol is the ceiling".
- **E3 — DROPPED**: the old 4-bit pack already measures cos 0.9959 vs
  pristine; a fresh pristine-4-bit rebuild has no expected gain.
- **E3' — protocol probes (no requant)**: on the best tier, chat template and
  think-allowed variants + T>0 sampling. The raw-protocol ceiling may sit well
  below 61.6% purely from prompt/sampling differences; Qwen 3.6 is a thinking
  model and `--no-think` is our raw protocol's ban — removing it is a free
  lever to test.

### Runbook (2026-08-20 state)

- E1 build: `FINCHMOE_GDN8=1 python3 quantize_non_experts.py --input
  ../models/Qwen3.6-35B-A3B-bf16 --output quant_clean_gdn8 --verify`
- E2 build: `python3 repack_experts.py --index
  ../models/Qwen3.6-35B-A3B-bf16/expert_index.json --bits 8` (writes
  `models/Qwen3.6-35B-A3B-bf16/packed_experts_8bit/`; symlinked from
  `finchmoe/packed_experts_8bit`). NOTE: 8-bit requires the dequant→requant
  path (bits in (3,8) routing + verify accepting BF16-byte sizes). DONE
  2026-08-20 (pre-reboot, 09:36); smoke verification (sample=64) reads
  cos ≥0.999; the FULL sweep was interrupted by the 2026-08-20 kernel
  panic and is pending.
- Full sweep: `python3 finchTool/tools/quant_audit.py --experts-only`
  — DONE 2026-08-20 (guardrails held: peak ~2 GB, exit 0). 30,720 tensors
  per pack: 3-bit mean 0.9815 (min down 0.9536); 4-bit 0.9959 (0.9891);
  **8-bit 0.99996 (min 0.99988)** — the E2 baseline is near-lossless.
  The worst-10 experts are the SAME across all three packs (L0e25, L36e2,
  L32e123, …) — per-expert difficulty is intrinsic to the weights, so any
  mixed-bit allocation should target exactly those experts.
- GOTCHA found 2026-08-20: the E1 build wrote the 2.4 GB bin but NOT the
  manifest (`model_weights_quant.json` missing — the 09:47 reboot cut the
  build between the bin and json writes). Restarted the runbook E1 command
  (deterministic — same bin, adds the json + verify).
- E1 probe server: `./finchmoe-infer -R 9000 -m . -w
  quant_clean_gdn8/model_weights_quant.bin -j
  quant_clean_gdn8/model_weights_quant.json -e 0 --top-k 1 --no-think
  --rep-penalty 1.0` (3-bit experts, GDN8 bin).
- E2 full run: same + `--int8-experts` (8-bit experts ≈ 3.2-4 tok/s; full 164
  ≈ 3-4 h, background, alone on the machine). Probe first: `--limit 20`.
- Harness: `cd ../humaneval_m1 && python3 generate.py --limit 20 --port 9000
  --results he_results_probe.jsonl && python3 evaluate.py --results
  he_results_probe.jsonl` — or `./probe_tier.sh <name> [extra args]` with
  `EXTRA_WEIGHTS` env for the bin/manifest pair.
- E3' variants: drop `--no-think` (think allowed), `--chat` on generate.py
  (chat template), `-e 0.3`/`0.7` (sampling).

Verification gates (every experiment, before defaulting anything):
- Per-tensor CosSim vs BF16 (target: ≥0.99 for any raised-bit tensor).
- Typo-prompt quality check (clean-rebuild protocol).
- 20-task HumanEval probe, raw protocol (regression battery below).
- Decode-speed check: 3-bit tier must stay 10-16 tok/s unless the tier is
  intentionally slower (E1/E2 by design).

Success gate (renegotiated after Phase 0): if E2 lands ≥50%, 8-bit experts
become the quality default (I/O cost accepted or mitigated by deferred
overlap); if E2 lands ~33-38%, quantization is near-maxed under the raw
protocol and the mission pivots to E3' (protocol) with Phase 2 as the quant
lever. 3-bit ≥35% / 4-bit ≥40% remain the gates for adopting E1's bin.

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

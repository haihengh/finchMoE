# Qwen 3.6 35B-A3B port

This repository is in the middle of a port. The upstream project is
[TurboFieldfare](https://github.com/drumih/turbo-fieldfare), a Swift + Metal
runtime for Gemma 4 26B-A4B. This fork's goal is to run
**Qwen 3.6 35B-A3B** (`model_type: qwen3_5_moe`) on the same engine. The
engine has since been rebranded to **FlashQwen** (upstream TurboFieldfare
base archived in `reference/`); the inherited docs in this directory describe
its runtime (`.fqturbo` layout, expert streaming, prefill/decode phases) and
remain the reference for the engine mechanics. This document covers the Qwen
3.6 model side and the port plan.

Directive (owner, 2026-08-30): drop the previous Qwen3-30B-A3B engine, run
Qwen 3.6 35B-A3B, use TurboFieldfare as the base, and feel free to rewrite
anything in this codebase.

Status: **port complete through end-to-end quality + benchmarks
(2026-09-05).** The engine is rebranded to **FlashQwen** — all
`TurboFieldfare`/`Fieldfare`/`.gturbo` identifiers, module/target names, and
the on-disk format are renamed (binary magic `FQTURBO`, extension `.fqturbo`);
the pristine upstream TurboFieldfare base is archived in `reference/`
(gitignored, like `models/`). The original `QwenFieldfare*` sources were
deleted; the bf16 Qwen 3.6 checkpoint is present at
`models/Qwen3.6-35B-A3B-bf16/` and the repacked int4 install at
`models/Qwen3.6-35B-A3B-4bit.fqturbo/` (19.5 GB, load-validated). The
Gated-DeltaNet unit is reference-checked against the `qwen3_5_moe`
transformers source, the full Qwen decode-layer path and the chunked Qwen
prefill path are wired into the forward pass (Gemma paths untouched), and
the bf16 → int4 quantizing repack is built and proven against the real
checkpoint. The interface is complete: the Qwen preset is auto-detected from
the installed manifest in app / CLI / server, the tokenizer family handles
Qwen special tokens + ChatML + the dual stop set (248046/248044), and the
softcap-0 sampling path is wired. End-to-end generation is coherent — greedy
decode ~10.5 tok/s, chunked prefill ~20 tok/s, ~1.1 GiB resident while the
19.5 GiB install streams out of core. The earlier degenerate-loop output was
the GDN readout-scale bug (missing `1/sqrt(head_dim)` after the l2norm),
resolved and verified 2026-09-05 — see item 1 under "Remaining work, in
order" for the evidence. The one open item is the ≈8 s fixed per-run prefill
cost (identical in debug/release, deterministic; untriaged — matters for
first-token latency). The [implementation plan](#implementation-plan)
records the verified state and the remaining work, in order.

## Target model

Local checkpoint: `models/Qwen3.6-35B-A3B-bf16/` (27 bf16 safetensors shards).
It is a vision-language model (`Qwen3_5MoeForConditionalGeneration` with a
`vision_config`); the port targets the `text_config` path only, consistent
with the engine being text-only.

| Property | Value |
| --- | --- |
| `model_type` | `qwen3_5_moe` |
| Layers | 40: `linear_attention` × 30 + `full_attention` × 10, pattern `[L,L,L,F]` × 10 |
| Hidden size | 2048 |
| Full attention | 16 Q heads, 2 KV heads, head_dim 256 |
| GDN heads | 16 key heads, 32 value heads, key/value head dim 128 |
| GDN conv | kernel dim 4 → 3-element causal state, `conv_dim = 8192` |
| GDN recurrent state | per value-head 128 × 128, fp32 (`mamba_ssm_dtype`) |
| MoE | 256 experts, top-8 per token, expert intermediate 512, shared expert 512 |
| Activation | silu (GeGLU experts) |
| Vocab | 248320, embeddings untied (`lm_head` is separate) |
| RoPE | partial rotary 0.25 of 256 dims, θ = 1e7, interleaved MRoPE, sections [11, 11, 10] |
| Attention output gate | yes (`attn_output_gate: true`) |
| Position limit | 262144 |
| BOS / EOS | 248044 |
| Extra | MTP head with 1 hidden layer (`mtp_num_hidden_layers: 1`) — not needed for greedy decode |
| Norm | RMSNorm, eps 1e-6; no attention bias |

The MoE shape differs from Gemma 4 in scale only, not in kind: more experts
(256 vs 128), same top-8, same shared-expert-plus-routed split, so the
expert-streaming runtime (16-slot LFU cache, `pread` on miss) transfers
directly. The real new compute is the GDN linear-attention layer, which
replaces the KV cache on 30 of the 40 layers with a fixed-size recurrent
state — smaller than the KV cache, but with math the engine has never
implemented.

## GDN decode math (locked)

Source of truth: `qwen3_5_moe` in Hugging Face Transformers —
`transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
(`Qwen3_5MoeGatedDeltaNet`, `torch_recurrent_gated_delta_rule`,
`causal_conv1d_update`, `l2norm`, `RMSNormGated`). The sibling `qwen3_next`
model family is **not** the reference even where its code looks similar; the
math below is the one pinned from `qwen3_5_moe`.

Per decode step, per value-head `hv` (its query/key head is `hv // 2` via
`repeat_interleave(2)`):

```text
# projections (per layer, all heads at once)
q, k, v = split(in_proj_qkv(x))     # q: 16×128, k: 16×128, v: 32×128
z      = in_proj_z(x)               # 32×128
b      = in_proj_b(x)               # 32
a      = in_proj_a(x)               # 32

# causal conv on the pre-norm q/k/v stream (state = last 3 inputs)
q, k, v = silu(conv_update(q, k, v))

# per value-head, K-major state S[128, 128] in fp32
qn = l2norm(q[hv//2]) / sqrt(128)   # l2norm = x * rsqrt(sum(x²) + 1e-6)
kn = l2norm(k[hv//2])
beta = sigmoid(b[hv])
g    = -exp(A_log[hv]) * softplus(a[hv] + dt_bias[hv])
decay = exp(g)

S    = S * decay
r    = S @ kn                          # residual read
delta = beta * (v[hv] - r)
S    = S + outer(kn, delta)            # rank-1 write
o[hv] = S @ qn                         # output read from the UPDATED state

# gated RMSNorm (mean-based) then output projection
out = out_proj( rmsnorm(x) * weight * silu(z) )
```

Details that matter for a Metal port:

- `RMSNormGated` in `qwen3_5_moe` has a **learned weight** `nn.Parameter(
  torch.ones(128))` and uses **mean** (not sum) for the variance:
  `y = x * rsqrt(mean(x²) + eps) * weight * silu(z)`.
- The recurrent state is fp32 and lives in a persistent device buffer per
  (layer, value-head): 32 heads × 128 × 128 × 4 B = 2 MiB per GDN layer,
  60 MiB for the 30 GDN layers. It is a cache, not scratch — allocate it
  like the KV cache, and keep it in device memory across decode steps
  (64 KB per head exceeds the default 32 KB threadgroup memory, so the
  kernel reads/writes it through a buffer, with the two matvecs done as
  two-stage block reductions in the style of `Metal/Primitives/rmsnorm.metal`).
- Conv state is per channel, 3 elements (kernel 4 − 1), on the 8192-dim
  qkv stream, updated each step.
- Prefill (chunked) runs the same recurrence sequentially over the chunk
  (wired 2026-09-04); there is no associative form for the gated delta rule,
  so the prefill cost is length × per-step cost. Bounded chunking keeps the
  activation scratch small as it does for the Gemma path.

## Implementation plan

Verified against the working tree on 2026-09-01 (commit `6b1e2ec`), updated
2026-09-03 after the decode-path wiring, 2026-09-04 after the Qwen chunked
prefill wiring, and 2026-09-04 again after the quantizing repack landed
(real 19.5 GB install built and load-validated), and 2026-09-05 after the
int8 linearAttention repack + the denormal-scale trap fix (20.0 GB install),
and 2026-09-05 again after the GDN readout-scale fix that resolved the
degenerate generation (item 1 below), and 2026-09-07 after the long-form
quality pass (item 3 below).
This section is the single
source of truth for what is done, what is wired in, and what remains, in the
order that unblocks an end-to-end Qwen 3.6 run.

### Done and verified

- **GDN decode unit (Phase 1, complete).** `Metal/LinearAttn/gdn.metal`
  (`gdn_conv_update`, `gdn_gate`, `gdn_recurrent`, `gdn_rmsnorm_gated`,
  `gdn_gate_gemv`), wrapper `Kernels/LinearAttn/GDN.swift`, fp32 CPU reference
  `FlashQwenValidation/Support/Reference/LinearAttn/GDNRef.swift`,
  registered in `MetalContext`, Swift-Testing tests in
  `Tests/FlashQwen/Core/Kernels/LinearAttn/GDNTests.swift` passing within
  `fp16ChainedReduction`. Covers the full decode step except the out_proj GEMV
  (shared with the existing int4 GEMV): causal conv → gate
  (`beta=sigmoid(b)`, `g=-exp(A_log)·softplus(a+dt_bias)`) → per-value-head
  recurrent recurrence (v-major `state[(hv*D+v)*D+k]`, fp32, two matvecs as
  two-stage block reductions) → gated RMSNorm (`x·rsqrt(mean(x²)+eps)·weight·silu(z)`,
  mean-based, shared bf16 weight). Reference pinned to `qwen3_5_moe`.
- **Arch preset + dispatch (Phase 6, core).** `ArchConfig.qwen3_6_35B_A3B`
  (`Infrastructure/ModelIO/ModelTypes.swift:139`): 40 layers, mask
  `fullAttentionLayerMask` set on the 10 `F` layers (`[L,L,L,F]` × 10), 256
  experts top-8, 16 Q / 2 KV, `fullHeadDim` 256, GDN 16 key / 32 value heads
  × 128, partial RoPE 0.25, `attnOutputGate`. The runtime already branches on
  it: `Model.swift` reads `fullAttentionLayerMask[layer]`;
  `RealForwardRunner.swift` consumes `fullHeadDim`, `partialRotaryFactor`, and
  `numFullKVHeads`.
- **Qwen decode-layer path wired into `produceToken` (2026-09-03).** The whole
  decode loop now branches on `cfg.modelFamily == "qwen3_6"`; the Gemma path is
  untouched (666-test suite green, byte-stable kernels). New pieces, each
  validated against an fp32 reference before wiring:
  - `Metal/Qwen/qwen.metal` (module `qwen`): `qwen_post_attn` (residual add +
    `post_attention_layernorm` → shared-expert/router input), `vec_add_fp16`
    (tail combine `hidden += h2`), `qwen_attn_output_gate`
    (`attn·sigmoid(gate)`), `qwen_shared_gate` (1-row int4 GEMV +
    sigmoid scalar scales h1), `qwen_full_attn_epilogue` (q|gate split from
    the doubled q_proj, q/k per-head norms, partial RoPE — rotates the first
    `rotary_dim = 0.25·head_dim` contiguous elements as adjacent pairs with
    the rotary-dim frequency denominator, matching
    `Qwen3_5MoeTextRotaryEmbedding`; the interleaved-MRoPE reordering is
    identity for text-only position ids). Wrapper
    `Kernels/Qwen/QwenDecodeFusions.swift`; refs in
    `FlashQwenValidation/Support/Reference/Qwen/QwenDecodeRef.swift`; tests in
    `Tests/FlashQwen/Core/Kernels/Qwen/`.
  - **GDN branch**: in_proj_qkv GEMV → silu causal conv (in-place state) →
    fused `gdn_gate_gemv` (a/b GEMVs + gate; the runner assembles the
    a|b-packed block once at init from the two resident entries) → recurrent
    step (scale 1/sqrt(128), l2norm q/k) → gated RMSNorm → out_proj →
    `qwen_post_attn`. q/k/v read from the conv output at offsets
    0 / keyDim·2 / keyDim·4 bytes — no split copies.
  - **Full-attention branch**: q_proj [8192,2048] → epilogue → attention
    (`encodeFull(scale: nil)` = rsqrt(256) — Qwen's scaling) → output gate →
    o_proj → `qwen_post_attn`.
  - **Qwen MoE tail** (shared with both layer types via the extracted
    `encodeRoutedTail`): silu routed experts (function-constant
    `FC_MOE_ACT_SILU` in `moe.metal` phase-1; gelu default preserved),
    silu shared expert (`silu_mul_fp16` in `utility.metal` +
    `SharedExpertActivation`), `qwen_shared_gate` scaling, phase-2
    `residual: h1Buf` so `h2 = shared + routed`, then `hidden += h2`. The
    router reuses `encodeRouterGemma4` with ones-filled `effectiveScale` and
    `perExpertScale` buffers — the kernel's top-8 softmax is mathematically
    identical to Qwen's full-softmax + top-8 renormalization. No
    layer_scalar/sandwich norms/router.scale on this path.
  - **State**: per-GDN-layer fp32 recurrent state (2 MiB) + fp16 conv state
    ([8192,3]), allocated at init, zeroed in `reset()` only (continuation
    keeps them, like the KV cache). `KVCacheManager` gained a `.linear`
    `LayerKind` that allocates no KV for Qwen GDN layers (~480 MB saved at
    32K context). Untied `lm_head.weight` is used for the head.
  - **Model**: family-branching accessors (Qwen `linear_attn.*`,
    `mlp.shared_expert.*`, `mlp.shared_expert_gate.weight`, `mlp.gate.weight`,
    untied `lm_head`) + `validateRuntimeSchema` per-family branches
    (`validateQwen36Layers` / `validateGemma4Layers`) + `requireRaw` for the
    fp32 `A_log`/`dt_bias` and fp16 `conv1d.weight` entries.
- **Qwen chunked prefill wired (2026-09-04).** `prefillChunked` no longer
  throws for `qwen3_6`; `executePrefillChunk` branches per layer on
  `cfg.modelFamily == "qwen3_6"` (the Gemma body is byte-for-byte untouched;
  the shared `encodeInt4Projection`/`copyPrefillKVToCache` helpers were
  promoted from nested funcs to private methods, behavior identical). New
  pieces, each validated against an fp32 reference before wiring:
  - `Metal/LinearAttn/gdn_prefill.metal` (module `gdn_prefill`):
    `prefill_gdn_conv_chunk` (batched kernel-4 conv over `[T][C]`; the last
    token's row commits the post-chunk conv state to a **separate** buffer —
    the wrapper blit-carries it into the persistent state — handling all T
    including 1 and 2), `prefill_gdn_recurrent_seq` (one 256-thread
    threadgroup per value head looping the chunk's T tokens through the exact
    decode recurrence; q/k/v read from the fused conv block at offsets
    0 / keyDim / 2·keyDim elements — no split copies), `prefill_gdn_gate`
    (batched `g`/`beta` from the fp16 a|b QMM output), and
    `prefill_gdn_rmsnorm_gated` (batched mean-based gated norm). Wrapper
    `Kernels/LinearAttn/GDNPrefill.swift`; reference `GDNPrefillRef.swift`;
    17 tests in `Tests/FlashQwen/Core/Kernels/LinearAttn/GDNPrefillTests.swift`,
    including a composed conv → gate → recurrent → norm chunk with the fp16
    boundary roundings the production path applies, and a two-chunk
    conv-state carry test. **The batched conv is not in-place-safe** (row
    t+3's taps read the raw x[t] while row t writes its output — no
    cross-threadgroup ordering), so the scratch keeps separate proj/conv-out
    buffers; the decode single-token conv stays in-place as before.
  - **GDN layer flow**: input norm → `in_proj_qkv`/`in_proj_z` (MPP/QMM) →
    fused `in_proj_a|b` `[2V,D]` QMM over the init-assembled decode gate
    block + batched gate → conv chunk → recurrent seq → gated rmsnorm →
    `out_proj` → per-token `qwen_post_attn` (decode kernel, per-token
    offsets).
  - **Full-attention layer flow**: input norm → doubled `q_proj` (the q
    scratch is sized T·2·qDim for Qwen) → per-token
    `qwen_full_attn_epilogue` (q|gate split, q/k norms, partial RoPE; q_out
    packed into the q scratch, gate into a dedicated buffer) → KV copy →
    `attention_prefill_causal_tiled` with Qwen's scale
    `1/sqrt(head_dim)` → per-token `qwen_attn_output_gate` → `o_proj` →
    per-token `qwen_post_attn`. No v_norm (Qwen has none); no ring (Qwen has
    no sliding window).
  - **MoE tail** (both layer types): router (ones-filled scales, as in
    decode) → silu shared expert (`PrefillSharedExpert.encodeBlock` gained an
    `activation:` pass-through) + per-token `qwen_shared_gate` → streamed
    routed tiles with a new silu phase-1 variant (`FC_PREFILL_MOE_ACT_SILU`,
    index 77, in `prefill.metal`; `encodeStreamedBatched` gained
    `activation:`) → reduce → `h2 += h1` → `hidden += h2` (per-token
    `vec_add_fp16`; the decode path's phase-2 residual restated for prefill).
    No layer_scalar, no sandwich norms.
  - **Scratch**: `PrefillChunkScratchLayout` gained family-aware Qwen buffers
    (qGate, qkv proj/conv-out, z, a|b, fp32 g|beta, recOut, conv newState)
    and a T·2·qDim q buffer for the doubled `q_proj`; Gemma layouts are
    unchanged.
- **Qwen quantizing repack (2026-09-04).** A from-scratch LOCAL bf16 → int4
  quantizing writer path — the Gemma remote byte-copy pipeline is untouched:
  - `Format/QwenLocalSnapshot.swift` — loads a local bf16 snapshot
    (index.json weight map + config.json via `ArchInfo` + all shard headers);
    no `quantization` slot required.
  - `Planning/QwenRepackPlanner.swift` — remaps checkpoint names to the
    engine namespace (`model.language_model.layers.{L}.*` →
    `language_model.model.layers.{L}.*`; `lm_head.weight` passes through),
    assigns per-tensor transforms (int4 affine / int8 router / `(1+w)` norm
    baking / bf16→fp16 conv1d / bf16→fp32 A_log+dt_bias), excludes the 352
    vision + MTP tensors, and plans the expert layers with the fused
    `gate_up_proj [E, 2F, D]` split into the gate/up role slices
    (per-expert source base offset f·d·2 for the up half).
  - `Writing/QwenQuantizedWriter.swift` — row-by-row transforms through the
    canonical `FQTurboQuantization` with scratch bounded by the widest row
    (2048 elements); `ResidentWriter.encodeIndex` was refactored to a shared
    record-based core (Gemma behavior identical; `PerExpertTensorSlice`
    gained a defaulted `sourceBaseOffset`).
  - `Workflow/LocalQwenRepacker.swift` — orchestrator: lock → plan → write
    resident + layer files → layout.json → tokenizer sidecars →
    manifest.json (qwen3_6 arch, quant slots 4/4/8/4/4, affine group 64) →
    verified-install receipt → atomic rename. CLI: `--input-snapshot <dir>`
    selects this path; the Gemma HF download stays the default.
  - **Validated**: 8 new tests — planner invariants (remap, dtypes, shapes,
    int8 router sizes, fp16/fp32 raw entries, page-aligned non-overlapping
    payloads, gate/up split) + full repack runs on a synthetic toy Qwen
    checkpoint + **the runtime gate**: the toy install passes
    `Model.load(expecting:)` with a matching toy preset (validateArch +
    validateRuntimeSchema) and every Qwen accessor resolves, plus byte-level
    checks that the packed payloads equal the canonical quantizer's output
    and that norm weights carry the baked `(1+w)`. Full suite: 692 tests
    green (only the pre-existing environment-driven AppModelTests flake
    fails).
  - **Real dry run DONE (2026-09-04):** `FlashQwenRepack --input-snapshot
    models/Qwen3.6-35B-A3B-bf16 --output models/Qwen3.6-35B-A3B-4bit.fqturbo`
    → **19.5 GB install** (`model_weights.bin`, 40 packed-expert layer files,
    layout.json, tokenizer sidecars, manifest, receipt). Three real-shape
    issues found and fixed by the run: the per-row scratch cap was 2048
    (widest real row is the 4096-column out_proj); per-row 1 KB pwrites were
    USB-bound (rows are now quantized in 64-row batches — three contiguous
    pwrites per batch — plus buffer-based quantizer overloads in
    `FQTurboQuantization`); and the 16 MB `layout.json` read caps (repack
    validator + engine `PackedExpertsLayoutReader`) were raised to 64 MB (the
    256-expert × 40-layer layout is 22.5 MB). The run must use the
    **release** binary — debug Swift numeric loops are ~50× slower.
    **Gate passed:** `QwenRealInstallLoadTests.realInstallLoadsWithQwenPreset`
    loads the real install with `.qwen3_6_35B_A3B` — validateArch +
    validateRuntimeSchema over all 613 entries + accessor shape checks green.
    Full suite: 694 tests green (only the pre-existing environment-driven
    AppModelTests flake fails).

### Wired in but not reachable for Qwen

Nothing. The Qwen preset is auto-detected from the installed manifest
(`ManifestReader.detectPreset` → `ArchConfig.preset(forModelFamily:)`), wired
into the app probe and the app/CLI/server model loads; the Qwen tokenizer
family, sampling softcap, and stop tokens are wired (see below).

### Done and verified (interface + end-to-end; committed bf1f5b0, 8e85605, f10b78b, 8004779)

- **Sampling / stop (Phase 4).** `final_logit_softcapping = 0` (Qwen 3.6
  config) is honored: `softcap_value` in `logit.metal` guards `softcap <= 0`
  (identity passthrough — `z/0 → ±inf` and `0·NaN` would poison the softmax),
  the Sampler's repetition-penalty path already branched on `logitSoftcap > 0`,
  and `RawCompletionScratch` now takes the softcap from
  `model.config.finalLogitSoftcap` (plumbed through CLI, server, and app).
  Validation: `zeroSoftcap_isPlainSoftmax` kernel-vs-reference test.
  **Stop set:** the checkpoint's `generation_config.json` stops on BOTH
  248046 (`<|im_end|>`) and 248044 (`<|endoftext|>`, the config's
  bos/pad/eos_token_id) — the Qwen tokenizer family carries both, plus
  `<|im_start|>` = 248045. Gemma special-token behavior is unchanged.
- **Qwen tokenizer family** (`Tokenizer.swift`): `GFTokenizerFamily` detected
  from the installed `config.json` (`model_type: qwen3_5_moe`); Qwen has no
  standalone BOS (`encode(addBOS: true)` is a no-op on that family — the
  ChatML template frames turns), pad/bos fall back to `<|endoftext|>`
  (248044), the Gemma tool/channel markers become inert -1 sentinels, and
  `applyChatTemplate` renders the standard
  `<|im_start|>role\n…<|im_end|>\n…<|im_start|>assistant\n` ChatML form.
  Five install-gated tests pin the family, stop set, special-token IDs, no-BOS
  encode, and template round-trip.
- **Preset selectable (Phase 6, finished).** `ManifestReader.detectPreset`
  peeks the installed manifest and `ArchConfig.preset(forModelFamily:)` picks
  `.qwen3_6_35B_A3B` for `qwen3_6`, Gemma otherwise — wired into
  `AppModelInstallationProbe`, the app `RealInferenceClient`, the CLI `Run`,
  and the server `ServerInference`. The app's KV memory estimator
  (`AppContextLengthOption`) is now family-aware: `slidingWindow == 0` means
  the non-full layers hold no KV (Qwen's cache is the 10 full-attention
  layers only, 2 KV heads × 256 — a Qwen-pinned test was added).
- **Decode full-attention KV offset fix.** The Qwen decode path passed the
  current token's k/v slot offset to `encodeFull`; the kernel walks positions
  `[0, seqLen)` from the buffer start, so any position > 0 read past the
  written rows. Now `kOffset: 0` / `vOffset: 0` (matching the Gemma path).
- **Real-install layer gate** (`RealForwardRunner.qwenLayerDebugHook` +
  `QwenLayer0DebugTests`): an internal, nil-cost-when-unset hook snapshots
  per-layer decode and prefill values; the test suite replays fp32 references
  against the real install and pins: per-kernel real-weight comparisons, all
  40 layers' decode mixers, the prefill per-row per-layer mixers, the
  prefill conv/recurrent states, and the full-vocab lm_head argmax. All match
  to fp16 noise.
- **Repack fidelity gate** (`repackWeightsMatchBf16Checkpoint`): install
  tensors dequantized and compared against the ORIGINAL bf16 checkpoint
  shards (parsed directly) — the kernel-vs-reference tests all read the same
  install bytes and cannot catch a repack mapping bug. Pinned: in_proj_qkv,
  conv1d (exact), A_log (exact), input norm / post-attn norm / q_norm /
  final norm with the `(1+w)` baking (the qwen3_5_moe RMSNorm computes
  `x·rsqrt(mean(x²)+eps)·(1+w)` — the baking is correct for ALL of them,
  including the final norm), lm_head rows, router (int8), embeddings for the
  actual prompt ids, and expert 0's gate/up/down slices of the fused
  `gate_up_proj`. All match to quantization noise.
- **Embedding scaling bug fixed (2026-09-04).** The Gemma path scales
  embeddings by `√hidden`; Qwen 3.5 does NOT (`embed_tokens` feeds the layers
  straight). The inherited scale left every residual embed-dominated by √D
  (45×) — each layer's contribution was attenuated by 1/√D and the model
  collapsed into fixed-point loops ("Of.\nOf.\n…"). Both embed sites
  (`prefillChunked`, `produceToken`) now pass
  `modelFamily == "qwen3_6" ? 1.0 : √D`; the debug-test references were
  updated to match.
- **Tokenizer ids independently verified.** A ground-truth BPE encoder in
  Python (vocab.json + merges + pretokenize regex + added tokens) reproduces
  the engine's token counts exactly (raw prompt 5 ids, ChatML prompt 21 ids),
  and the observed degenerate output decodes to a clean id cycle — the ids
  were right while the model's distribution was collapsed.
- **GDN linear-attention projections raised int4 → int8 (2026-09-05).** New
  required wire slot `linearAttention` (manifest between `attention` and
  `router`; fixture regenerated, hash pinned) and planner/writer/runner
  plumbing (`int8Affine` for all five `linear_attn.*` projections —
  in_proj_qkv / in_proj_z / in_proj_a|b / out_proj). Cut the recurrent-state
  quant-noise amplification: probe A/B on the real install shows the engine's
  per-layer isoMax vs the bf16 torch replay on linear layers dropping from
  mean 5.42 → 2.02 (worst L10 27.7 → 14.7), every linear layer improved
  (1.3–13.8×), full-attention layers unchanged at the ~0.03–0.4 noise floor.
- **Denormal-scale repack trap fixed (2026-09-05).** The real checkpoint
  carries subnormal residue rows (layer 0 `linear_attn.in_proj_qkv` row 143 =
  alternating ±2^-123 = BF16 0x0200/0x8200, an export-pipeline artifact). The
  canonical codecs divided via `× (1/scale)`; the FP32 reciprocal of a
  BF16-rounded subnormal scale overflowed to inf → `0 × inf` NaN → the
  `Int()` conversion trapped (deterministic SIGTRAP mid-repack at ~289 MiB,
  int8-only because `/15` keeps int4 scales in normal range). Both codecs now
  quantize with direct `(w − bias) / scale` division (`scale == 0 → q = 0`).
  Regression tests in `Tests/FlashQwenFormat/FQTurboQuantizationTests.swift`
  (3, passing). **Repack: exit 0, 20 014 114 816 bytes** (19.5 GiB install,
  `linearAttention` 8). Engine-load tests green on the fresh install
  (`QwenRealInstallLoadTests`, `QwenRepackEngineLoadTests`).
- **GDN readout-scale fix — degenerate generation RESOLVED (2026-09-05).**
  The GDN readout `q` was missing its `1/sqrt(head_dim)`: `qn = l2norm(q ·
  scale)` (in `GDNRef` and both Metal recurrent kernels) cancels the scale
  inside the norm, leaving the pre-gated-norm `o` √128 ≈ 11.31× too large;
  the per-head norm hid it on large-RMS heads but heads at the
  `mean(o²) ≲ eps/128` floor stayed scaled, and `out_proj` smeared the error
  over every residual. Fixed by applying the scale AFTER the l2norm
  (`qn[i] = qh[i] * qinv * scale`; ref: `l2Norm(q).map { $0 * scale }`) —
  the locked math above was already correct. Localized by the stage-fidelity
  drill (`tools/qwen-probe/stage_drill.py`, committed with this batch):
  conv/z/g/beta clean at quant noise, `o` a uniform 11.31× off. See item 1
  below for the full evidence and verification (CLI generation now
  coherent).

### Remaining work, in order

1. **End-to-end quality — RESOLVED (2026-09-05).** The degenerate output
   (" nothingParams the article as did尚liea") was **not** quantization
   noise: the GDN readout `q` was missing its `1/sqrt(head_dim)` scale.
   `GDNRef.recurrentStep` and both Metal kernels (`gdn_recurrent`,
   `prefill_gdn_recurrent_seq`) computed `qn = l2norm(q * scale)` — scale
   INSIDE the l2norm, where the constant factors out of the sum and cancels:
   `l2norm(c·q) = c·q / (c·‖q‖) = q/‖q‖`. The oracle applies it AFTER:
   `query = l2norm(query, eps=1e-6)` then `query = query * scale`
   (`torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`).
   The locked math in "GDN decode math" above was right; the code (and the
   `GDNRef` comment "applied before l2norm") drifted from it.
   - Consequence: the pre-gated-norm recurrent output `o` was √128 ≈ 11.31×
     torch's — elementwise-uniform (stage drill: head-RMS ratios 11.25–11.36
     on all 32 value heads at both L0 and L1). The per-head gated RMSNorm
     cancels a uniform per-head scale, so most heads looked fine — but heads
     with `mean(o²) ≲ eps/128 ≈ 1e-8` hit the `÷(mean + 1e-6)` floor
     asymmetrically (engine variance 128× torch's against the same eps), and
     those heads' normalized `h` rows stayed 11.31× too large (drill: h
     head-ratios bimodal ~1.0 vs ~11.3; 3 heads at L0, ~half at L1, whose
     activations are smaller). `out_proj` mixes every head into every `xa`
     element, so the error spread over the whole residual and compounded
     down the 30 linear layers — matching every earlier observation (linear
     layers diverge, full-attention clean, head/norm blameless).
   - Why the unit tests never caught it: GDN tests validate the Metal
     kernels against `GDNRef`/`GDNPrefillRef`, which share the wrong
     convention — self-consistent. The first oracle-crossed check was the
     stage-fidelity drill (`tools/qwen-probe/stage_drill.py`, committed with
     this batch): conv/z/g/beta matched the fp32 torch oracle at quant noise
     while `o` diverged by the uniform 11.31× — an exact localization to the
     recurrent readout.
   - Fix (engine-wide, 4 sites): `GDNRef.swift` →
     `qn = l2Norm(q).map { $0 * scale }`; `gdn.metal` + `gdn_prefill.metal`
     pass 1 accumulates the UNScaled q sum and writes
     `qn[i] = qh[i] * qinv * scale`; `GDNPrefillRef` inherits via
     `recurrentStep`. 28 GDN/GDNPrefill tests green.
   - Verification: drill rerun on a fresh engine dump — `o` rmsDiff now
     0.1–0.2% of ref RMS, `h` 0.1–0.7%, `xa` 0.5–1.8% (int8 out_proj
     floor), head ratios uniform ≈ 1.000 at both layers. CLI greedy is
     fixed: "The capital of France is" → " Paris, a city renowned for its
     iconic landmarks such as the Eiffel Tower, the Louvre Museum, and
     Notre-Dame Cathedral." (coherent through 150+ tokens, 10.6 tok/s).
     Note: raw-completion prompts without the chat template make this
     checkpoint emit its `<think>` preamble — model behavior, not engine
     error (bf16 reference does the same via the template).
   - Earlier probe evidence (engine-head reproduces engine logits,
     full-attn layers clean on any input, embedding int4 refuted as the
     driver, L1-vs-L0 t0 asymmetry) is all consistent with this one bug and
     stands closed out.
2. **Benchmarks — DONE (2026-09-05).** Measured on this 16 GB Mac mini with
   the release CLI (greedy, local Qwen install, warm page cache), recorded
   in the README "At a glance" table:
   - Decode **~10.5 tok/s**, flat over 200–300 tokens (debug ≈ release —
     decode is Metal-bound). Peak resident **~1.1 GiB** during decode (the
     ~20 GB install streams out of core). Wall for load + prefill + 300
     tokens ≈ 39 s.
   - Prefill **~20 tok/s** at 705 tokens, but with a **≈7.9 s fixed per-run
     cost inside the timed prefill** (5-token prompt: 7.9 s, 0.6 tok/s) —
     identical in debug and release, deterministic across runs. Worth a
     look if first-token latency matters (suspects: per-run expert-cache
     warm / per-layer out-of-core stream setup in the Qwen prefill path),
     not yet triaged.
   - CLI footer now prints `prefill=<s> (<tok/s>)` (was decode-only) so
     future runs report both.
3. **Long-form quality pass — DONE (2026-09-07).** Exercised the release CLI
   (local int4 install, raw + chat modes) to shake out residual quality or
   stopping issues before closing the port. Verdict: **no engine-side
   defects** — all runs coherent, no degenerate loops, no late-sequence
   drift (the degenerate symptom class) out to 1044 tokens; output style
   tracks checkpoint behavior. Runs (all greedy unless noted):
   - Raw, 300 tok: capital-cities continuation factually correct through
     Ottawa; byte-identical output on rerun (determinism).
   - Raw, 400 tok: computing-history prose clean end to end (drift check).
   - Raw prompts that trip the model's thinking mode spend the budget inside
     `<think>` (coherent chain-of-thought, no answer by 300 tok) — the
     known raw-completion behavior, not engine error.
   - Chat greedy: structured markdown/LaTeX explanations; two-turn
     follow-up keeps context (prefill 312 tok across both turns).
   - Chat greedy math: a train word problem solved end to end with
     verification — "meet at 11:48, 138 km from Station A" — and §2's
     head-start figure still consistent in §6 ~900 tokens later (good
     KV/recurrent-state continuity at 1000+ tokens). The model answered
     naturally (`stop=endOfTurn`) at 1044 tokens.
   - Chat sampled (default temp 0.2, seed): correct 5-7-5 haiku, natural
     stop at 29 tokens. Chinese raw prompt: coherent technical prose.
   - Footer stop reporting correct throughout (`stop=maxTokens|endOfTurn`);
     the ≈8 s fixed prefill cost reproduced in every run (7.8–8.6 s);
     decode 8.8–10.9 tok/s (longer contexts slower, mild thermal drift
     across back-to-back runs).
   - Practical notes: this checkpoint writes >1024-token worked solutions,
     so the CLI default `--max-new` 1024 truncates long reasoned answers
     (raise the budget); CLI usage line still claimed "Gemma 4 26B-A4B" —
     made model-neutral in this batch.

Known toolchain quirk (this machine, Xcode 26.6 / Swift 6.3.3): `swift test
-c release` discovers 0 tests under `-O` (the swift-testing section is linked
but not enumerated); run the suite with `-c debug` (or
`-c release -Xswiftc -Onone`). The heavy install-gated diagnostics are slow
in debug — ~5–15 min each.

### Machine: 2026-09-05 kernel panics and the 16 GB operating protocol

Two kernel panics this machine (16 GB Mac mini, Mac16,10), both identical
`watchdog timeout: no checkins from watchdogd in ~90 seconds` — a system-wide
hang from memory/IO thrash, not a process crash:

- 2026-09-04 21:46:18 (`panic-full-2026-09-04-214618`), mid-repack era.
- 2026-09-05 04:08:07 (`panic-full-2026-09-05-040807`) — followed by
  `apfsd` CPU-resource and `ResetCounter` diagnostics; GUI session restarted.

Unrelated to the panics: 8× `FlashQwenRepack` SIGTRAPs 2026-09-04 23:16–00:00
were the pre-fix denormal-scale traps (see above); the fix landed in 8004779
at 00:28:26, after the last trap.

Working-directory model is 19.5 GiB on a 16 GB machine (models live on the
same external volume, repack peak allocates bf16 source + output side by
side). Protocol that has kept this box alive since:

- `swift build` / `swift test` at `-j 3` max; incremental builds only.
- Gate heavy runs on `memory_pressure -Q` ≥ ~60% free, and never run the
  repack or an install-gated diagnostic while VS Code + Claude + simulators
  are also up (~3–4 GB of the 16 GB already gone).
- Watch RSS during install-gated tests (loads the real 19.5 GiB install as
  mmap/streamed layers + Metal buffers); abort if the test process RSS
  approaches ~13 GB or the box starts swapping (`sysctl vm.swapusage`).
- Watchdog panics leave no process-level crash report — the tell is a fresh
  `panic-full-*.panic` + `ResetCounter` pair in
  `/Library/Logs/DiagnosticReports/`.

## Repository state

- `Sources/FlashQwen*/`, `Tests/FlashQwen*/` — the working engine, rebranded
  from the TurboFieldfare base (upstream `drumih/turbo-fieldfare`). Modules,
  targets, product/binary names, and the on-disk format are all `FlashQwen`
  / `FQTurbo` / `.fqturbo` now.
- `reference/` — a pristine snapshot of the upstream TurboFieldfare source
  taken before the rebrand (gitignored, like `models/`), kept for reference.
- `Sources/QwenFieldfare*/` — the earlier Qwen3-30B-A3B engine, deleted.
- `models/Qwen3.6-35B-A3B-bf16/` — local bf16 checkpoint (27 shards), plus
  tokenizer files and `expert_index.json`.
- `docs/` — inherited TurboFieldfare documentation; `SYSTEM_DESIGN.md` and
  `IMPLEMENTATION_REFERENCES.md` describe the runtime and stay valid for the
  engine mechanics.

## References

- Authoritative GDN source: Hugging Face Transformers,
  `models/qwen3_5_moe/modeling_qwen3_5_moe.py` (local install under
  `transformers/models/qwen3_5_moe/`). Do **not** use `models/qwen3_next/`
  as the reference — it is a different model family with similar-looking
  code.
- [System design](SYSTEM_DESIGN.md) — `.fqturbo` layout, streaming,
  prefill/decode phases, Metal conventions the new kernels must follow.
- Kernel templates for the GDN unit:
  `Sources/FlashQwen/Metal/Primitives/rmsnorm.metal` (function-constant
  guards, two-stage block reduce), `Sources/FlashQwen/Kernels/Primitives/RMSNorm.swift`
  (PSO caching, dispatch), `Sources/FlashQwenValidation/Support/Reference/Primitives/RmsNorm.swift`
  (reference style), `Tests/FlashQwen/Core/Kernels/Primitives/RMSNormTests.swift`
  (test harness).

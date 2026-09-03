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

Status: **in progress.** The engine is rebranded to **FlashQwen** — all
`TurboFieldfare`/`Fieldfare`/`.gturbo` identifiers, module/target names, and
the on-disk format are renamed (binary magic `FQTURBO`, extension `.fqturbo`);
the pristine upstream TurboFieldfare base is archived in `reference/`
(gitignored, like `models/`). The original `QwenFieldfare*` sources were
deleted; the bf16 Qwen 3.6 checkpoint is present at
`models/Qwen3.6-35B-A3B-bf16/`. The Gated-DeltaNet decode unit (the part with
no Gemma analogue) is implemented and reference-checked against the
`qwen3_5_moe` transformers source (Phase 1, done), and the `qwen3_6_35B_A3B`
arch preset plus `fullAttentionLayerMask` dispatch are already in place. The
unit is **not yet wired into the forward pass** — that is the next step. The
[implementation plan](#implementation-plan) records the verified state and the
remaining work, in order.

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
- Prefill (chunked) must run the same recurrence sequentially over the
  chunk; there is no associative form for the gated delta rule, so the
  prefill cost is length × per-step cost. Bounded chunking keeps the
  activation scratch small as it does for the Gemma path.

## Implementation plan

Verified against the working tree on 2026-09-01 (commit `6b1e2ec`), updated
2026-09-03 after the decode-path wiring. This section is the single source of
truth for what is done, what is wired in, and what remains, in the order that
unblocks an end-to-end Qwen 3.6 run.

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

### Wired in but not reachable for Qwen

- **Prefill is Gemma-only.** `prefillChunked` throws
  `PrefillError.chunkedUnsupported` for `qwen3_6` — decode-only until the
  chunked prefill path (and the chunked GDN recurrence) is built.
- **Preset not selectable.** App, CLI, and server still select
  `ArchConfig.gemma4_26B_A4B` only
  (`AppContextLengthOption.swift:18`, `AppModelInstallationProbe.swift:22`).
- **Repack still pins Gemma, and can't ingest bf16 at all.**
  `Remote/SupportedModelSource.swift:5` points at
  `mlx-community/gemma-4-26b-a4b-it-4bit`. More importantly, the repack is a
  **byte-copy re-indexer**, not a quantizer: `Writing/WriterCore.swift:15`
  (`pwriteTensorRegion`) is a straight mmap→pwrite copy with no transform
  hook; `Format/IndexLoader.swift:53` hard-requires a `quantization` slot in
  the source `config.json` (the Qwen bf16 config has none); and
  `Planning/RepackPlanner.swift:346` rejects any routed expert that is not
  already `u32` with bf16 `.scales`/`.biases` companions, classifying routed
  experts only by Gemma's `.experts.switch_glu.` name (line 125). So the Qwen
  path is a **new quantizing code path**, not a config change. (The shared
  quantizer the new path needs is available — see step 2 — but the
  writer/planner themselves are still un-built.)

### Remaining work, in order

1. **Qwen prefill (chunked).** The decode path is wired; prefill must run the
   same GDN recurrence sequentially over each chunk (no associative form) and
   the full-attention chunk path for the `F` layers, then switch the
   `prefillChunked` family guard to the Qwen path. Until then end-to-end runs
   are blocked.
2. **Re-shape the `.fqturbo` manifest/layout to Qwen tensor names** and **write
   the bf16 → int4 group-64 `.fqturbo` repack** (Phases 3 + 5). Confirmed
   2026-09-02 this is a from-scratch quantizing writer, not a config tweak
   (see "Repack still pins Gemma"). **Prerequisite — DONE (2026-09-02):** the
   affine int4/int8 quantizer is now the **shared canonical implementation in
   `FlashQwenFormat`** (`FQTurboQuantization`), and the engine's `Quantization`
   enum is a thin forwarding facade over it. **Locked writer decisions from the
   2026-09-03 wiring:** the router is emitted as **int8 affine** (the existing
   router GEMV kernel reads one byte per weight and the manifest `router` slot
   requires weightBits 8 — so this also satisfies the "8-bit router" note);
   `Qwen3_5MoeRMSNorm` weights (`input_layernorm`, `post_attention_layernorm`,
   `q_norm`, `k_norm`, final norm) get the **`(1 + w)` form baked in at emit**
   so the runtime kernels stay weight-direct; `linear_attn.conv1d.weight` is
   emitted as **raw fp16** (bf16 → fp16 conversion at emit) and
   `A_log`/`dt_bias` as **raw fp32**; `in_proj_a`/`in_proj_b` stay separate
   entries (the runner fuses them at init). Source layout is **fused**
   (`mlp.experts.gate_up_proj [256,1024,2048]`), so the writer must split each
   expert's gate/up (`.chunk(2, -1)`: gate = first 512 of the feature dim, up =
   second) into the separate `gate`/`up` role blobs the decode kernel reads
   (ground truth: transformers `qwen3_5_moe` `modeling_*.py:729,751`). 256
   experts, top-8, shared expert 512, manifest with the `qwen3_6` architecture
   and SHA-256s.
3. **Sampling, stop on 248044** (Phase 4), and make the **Qwen preset
   selectable** in app / CLI / server (finish Phase 6). (Untied lm_head is
   already wired in the decode head.)
4. **End-to-end** (Phase 7): load → prefill → decode → sample; check a short
   reference generation for correctness; then measure and record tok/s so the
   "At a glance" table in the README gets real Qwen 3.6 numbers.

Each step gates the next: 1 unblocks real runs; 2 makes a Qwen install
loadable; 3 completes the interface; 4 proves it. Kernels stay validated
against the Swift fp32 reference with the `fp16ChainedReduction` tolerance
before they are wired into a layer.

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

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

Verified against the working tree on 2026-09-01 (commit `6b1e2ec`). This
section is the single source of truth for what is done, what is wired in, and
what remains, in the order that unblocks an end-to-end Qwen 3.6 run.

### Done and verified

- **GDN decode unit (Phase 1, complete).** `Metal/LinearAttn/gdn.metal`
  (`gdn_conv_update`, `gdn_gate`, `gdn_recurrent`, `gdn_rmsnorm_gated`),
  wrapper `Kernels/LinearAttn/GDN.swift`, fp32 CPU reference
  `FlashQwenValidation/Support/Reference/LinearAttn/GDNRef.swift`,
  registered in `MetalContext`, 9 Swift-Testing tests in
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
  it: `Model.swift:619` reads `fullAttentionLayerMask[layer]`;
  `RealForwardRunner.swift` consumes `fullHeadDim`, `partialRotaryFactor`, and
  `numFullKVHeads`.

### Wired in but not reachable for Qwen

- **GDN is not in the forward pass.** `RealForwardRunner.swift` / `Model.swift`
  have **no GDN branch** — the 30 `!isFull` layers currently run the standard
  (Gemma-style) attention path. The GDN unit is only reached from `MetalContext`
  registration and the tests. Until step 1 below is done, Qwen 3.6 output is
  wrong.
- **Preset not selectable.** App, CLI, and server still select
  `ArchConfig.gemma4_26B_A4B` only
  (`AppContextLengthOption.swift:18`, `AppModelInstallationProbe.swift:22`).
- **Manifest/layout is still Gemma-shaped.** `Model.swift:617-669` validates
  Gemma tensor names (`pre_feedforward_layernorm_1`, `router.scale`,
  `self_attn.q_norm`, `mlp.gate_proj`, …); the `.fqturbo` schema is not yet
  re-shaped to `qwen3_5_moe` tensor names.
- **Repack still pins Gemma.** `Remote/SupportedModelSource.swift:5` points at
  `mlx-community/gemma-4-26b-a4b-it-4bit`; there is no Qwen bf16 → `.fqturbo`
  writer yet.

### Remaining work, in order

1. **Wire GDN into the forward pass** (the load-bearing step). All four compute
   ops are now implemented and validated (see Done/verified); what remains is
   the wiring in `RealForwardRunner.produceToken`: route the 30 `!isFull`
   layers through `in_proj_qkv/z/b/a` GEMVs → `gdn_conv_update` → `gdn_gate` →
   `gdn_recurrent` → `gdn_rmsnorm_gated` → `out_proj`, replacing the
   standard-attention path (`gQKV`/`gQKVEpilogue`/`gAttention`) for those
   layers. Prerequisites: expose the GDN weight tensors (in_proj_qkv/z/b/a,
   `dt_bias`, `A_log`, norm weight, `out_proj`) — absent from the current
   Gemma-shaped layout (step 3 re-shapes them); and allocate the per-layer fp32
   recurrent-state buffer (32 value-heads × 128 × 128, ~2 MiB per GDN layer)
   like the KV cache, kept in device memory across decode steps.
2. **Full-attention path for the 10 `F` layers** (Phase 2). 16 Q / 2 KV,
   `head_dim` 256, partial RoPE 0.25 with interleaved MRoPE [11, 11, 10], and
   the `attn_output_gate`. Config plumbing is partly present (step above); the
   decode/prefill kernels still need it.
3. **Re-shape the `.fqturbo` manifest/layout to Qwen tensor names** and **write
   the bf16 → `.fqturbo` repack** (Phases 3 + 5): 256-expert blobs from the
   `models/Qwen3.6-35B-A3B-bf16/` shards (int4/int8 affine, group 64), 8-bit
   router, shared expert 512, manifest with the `qwen3_6` architecture and
   SHA-256s.
4. **Embedding + untied `lm_head` (vocab 248320), sampling, stop on 248044**
   (Phase 4), and make the **Qwen preset selectable** in app / CLI / server
   (finish Phase 6).
5. **End-to-end** (Phase 7): load → prefill → decode → sample; check a short
   reference generation for correctness; then measure and record tok/s so the
   "At a glance" table in the README gets real Qwen 3.6 numbers.

Each step gates the next: 1 makes the GDN layers produce the right hidden state;
2 completes the `F` layers; 3+4 make a real Qwen install loadable; 5 proves it.
Kernels stay validated against the Swift fp32 reference with the
`fp16ChainedReduction` tolerance before they are wired into a layer.

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

# Qwen 3.8 Flash-Next 125B port

This document covers the **Qwen 3.8 Flash-Next** model side of the FinchMoE
port (the engine mechanics live in the inherited docs — `.finch` layout,
expert streaming, prefill/decode phases). The engine runs **Qwen 3.6 35B-A3B**
and **Gemma 4 26B-A4B**; this port adds **Qwen 3.8 Flash-Next 125B**
(`model_type: qwen4_exp_text`, GGUF arch `qwen4exp`, manifest family
`qwen3_8`) to the same dual-family runtime. Both Qwen families share the
tokenizer and the Gated-DeltaNet (GDN) unit; 3.8 replaces every RMSNorm with
**hyper-connections**, adds **QSA** sparse-block attention on the full
attention layers, and adds a **PLE n-gram** hash-embedding on one layer.

Status: **M0 (data model) landed 2026-09-08 (commit `4de4774`); the engine
still refuses qwen3_8 installs at load** (`Model.validateRuntimeSchema`
throws "qwen3_8 installs need the Flash-Next engine (M2)"). The plan below
runs M1 (repack) → M2 (load/schema) → M3 (forward: hyper-connection → QSA →
PLE, decode then prefill) → M4 (real 352 GB repack + llama.cpp oracle
cross-check) under the machine protocol at the bottom. The user directive
(2026-09-09): full engine pass including repacking the local bf16 snapshot
into a `.finch` install and smoke-decode on this machine; correctness bar is
**numeric vs llama.cpp**; heavy runs happen here under the operating
protocol; the PLE table ships as **raw BF16 part files** (no PLE quant this
pass).

## Target model

Local checkpoint: `models/Qwen3.8-Flash-Next-bf16/` (352 GB, 131 bf16
safetensors shards; symlink → the flash-qwen checkout). It is a
vision-language checkpoint (`Qwen4ExpForConditionalGeneration` with
`model.visual.*`); the port targets the `text_config`/`language_model` path
only, like the 3.6 port. The MTP draft block (`mtp.*`, one decoder layer) and
the vision tower are **dropped** (llama.cpp's qwen4exp conversion also drops
both — `supports_mtp_export = False`, `no_mtp = True` in
`archive/llama.cpp/conversion/qwen4exp.py`).

Census (verified 2026-09-09 against shard headers, `model.language_model.*`
only): **48 layers**, `layer_types` pattern `linear_attention` × 3 +
`full_attention` × 1 repeated, full layers at 0-based **3, 7, …, 47** (12
layers, each with `self_attn.*` + `self_attn.indexer.*`); GDN on the other 36.
`ple_layer_ids: [2]` (1-based) → PLE on 0-based **layer 1 only** (a GDN
layer). `mtp.layers.0.*` also carries a full-attention set — it is root-level
and excluded.

| Property | Value |
| --- | --- |
| `model_type` | `qwen4_exp_text` |
| Layers | 48: 36 × GDN (`linear_attention`) + 12 × full attention at 3,7,…,47 |
| Hidden size | 2560 (LM), untied `lm_head` |
| Full attention | 24 Q heads (dim 256, **q\|gate interleaved**), 2 KV heads, head_dim 256, partial rotary 0.25 → **n_rot 64**, θ = 1e7, MRoPE sections [11, 11, 10] **in dim pairs** (= 32 pairs = the 64 rotary dims) |
| GDN | 16 key heads, 48 value heads, head dim 128, conv kernel 4, `conv_dim = 10240`, **sigmoid** output gate (3.6 uses silu) |
| Hyper-connection | 4 parallel streams × 2560 = 10240 plane, lowrank 320 |
| QSA indexer | 4 Q heads + 1 shared K head, dim 128; budget (top-k) 2048, compress ratio 4 |
| PLE | n-gram 3, heads per n-gram 8, conv kernel 4, row dim 160, 128 parts × 2,500,012 rows (320,001,536 rows total), layer 1 |
| MoE | 512 experts, top-10, expert/shared intermediate 640, shared gate sigmoid, `per_expert_scale` router |
| Vocab | 248320; embeddings untied; tokenizer shared with 3.6 |
| Norm | none in the residual path — hyper-connections replace every RMSNorm incl. the final (no `model.norm`) |

## Math authority

There is **no Hugging Face `transformers` module for `qwen4_exp`** on this
machine or upstream; llama.cpp is the only modeling source and is the
authority pinned here:

- `archive/llama.cpp/src/models/qwen4exp.cpp` — graph math (1213 lines).
- `archive/llama.cpp/conversion/qwen4exp.py` — HF→GGUF tensor transforms
  (which weights are zero-centred, split order, squeezes).
- `archive/llama.cpp/gguf-py/gguf/constants.py` — GGUF tensor/param names.
- Oracle at M4: the 79 GB quant GGUF `models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64`.

Converters: `qwen.py` (`_Qwen35MRopeMixin`, `_LinearAttentionVReorderBase`)
handle the shared Qwen-3.5-family machinery (mrope, GDN V-reorder) — the 3.6
port already validated those against llama; only the qwen4exp deltas are
locked below.

Line references below are to `qwen4exp.cpp` unless another file is named.

## Hyper-connection math (locked)

`build_hc_mix` (`:218-264`), `build_hc_combine` (`:266-286`), graph
(`:288-393`), tensor loads (`:108-209`). The residual is **4 parallel streams
of 2560** = one 10240 plane per token, `[n_embd, hc, T]` in ggml terms
(hc = 4 = `dsv4_hc_mult`; plane width `hc_dim` = 10240 = `hc_attn_norm.weight`
length). Per mixer:

```text
# per layer, before the token mixer (attn_hyper_connection) and before the
# MoE (mlp_hyper_connection); identical structure, separate weights
xn = rms_norm(x, eps) over each stream's own 2560 dim        # grouped RMS
xn = xn * w_norm[10240]        # hc_norm.weight — PLAIN scale, NOT zero-centred
                               # (conversion qwen4exp.py does not +1 it)
lo = w_down[320,10240] @ xn    # input_mix_weight_down
lo = silu(lo / 4)              # scale happens BEFORE silu
gate = sigmoid(w_up[10240,320] @ lo)        # input_mix_weight_up
gated = xn * gate
mixed = mean over the 4 streams of gated    # [2560, T] block input
inject = w_inject[4,10240] @ xn             # block_inject_weight — NO activation

# combine (per layer, after the token block and after the MoE):
#   block output b [2560,T] broadcast across streams, scaled per stream
res_c += b * 2*sigmoid(inject_c / 4)        # inject divided by hc before sigmoid
```

- Residual starts as **4 identical copies of the embedding** (`:324-326`); no
  embedding scale (`f_embd_norm` absent, same convention as 3.6).
- Root `hyper_connection_mixer` (`hc_head_norm/down/up`, `:380-393`) is the
  same mixer **without** inject (`nullptr` w_inject, `:382`): gate the normed
  10240 plane exactly as a per-layer mix does, then collapse to the **mean
  over the 4 streams → [2560]** — the collapsed vector *is* the mixer output
  (`result_norm` `:385`, comment `:380` "the final mixer is the output norm")
  and feeds `lm_head` directly. There is **no `model.norm`** anywhere in 3.8.
- The RMS eps is the model's `f_norm_rms_eps` (`:232`) — the same 1e-6 the
  GDN norms use (real `config.json` `text_config.rms_norm_eps` = 1e-6).
- Norm-bake caveat: llama's in-file comment at `:231` ("the converter folded
  each gamma to (1 + w)") is misleading for the HC norms — `qwen4exp.py`
  bakes `+1` **only** for the five zero-centred gammas
  (`ple.norm_key/query/conv`, `indexer.q_layernorm/k_layernorm`,
  `qwen4exp.py:134-136`); `hc_norm`/`hc_head_norm` are copied plain. The
  engine repack mirrors that exactly (raw BF16 hc_norm, `normOnePlusW` on
  the five), so GGUF-side llama and engine-side Metal multiply the same
  stored values.
- 2·sigmoid centres the scatter weights on 1: a zero injection is a plain
  residual add (`:274` comment).
- Weight shapes per mixer: `hc_norm` [10240], down [320,10240] row-major
  (out=320), up [10240,320], block_inject [4,10240] — HF stores
  `input_mix_weight_down/up` and `block_inject_weight`; the 4-block-inject
  row per stream: `res_c += b * 2*sigmoid(inject[c]/4)`.
- Per mixer the RMS reduction is per stream (`ggml_rms_norm` over ne0 2560,
  then the [10240] gamma scales stream c's plane at `[c·2560 : (c+1)·2560]`).

## QSA indexer math (locked)

`build_qsa_top_k` (`:477-616`), `build_attn_qsa` (`:620-698`),
`llama-memory-hybrid-idx.cpp` `set_input_qsa` (`:342-464`), tensor loads
(`:175-179`). Only full layers, only when the checkpoint carries indexer
tensors and `compress_ratios[layer] > 0`.

Key facts and engine consequences:

1. **Projections**: one HF tensor `index_qk_proj` [640, 2560]; the converter
   splits it **contiguously by row: q = rows [:512]** (4 heads × 128),
   **k = rows [512:]** (1 shared head × 128) (`qwen4exp.py:124-131`). Both
   derive from the **mixed** block input (same input as q/k/v/o), not from a
   per-stream plane.
2. **Zero-centred norms**: `indexer.q_layernorm/k_layernorm` (and
   `ple.norm_*`) are **Gemma zero-centred gammas**: the converter bakes
   `+1` (`qwen4exp.py:133-136`). Engine repack must bake `1+w` like the 3.6
   q/k_norm.
3. **Raw-key cache**: keys are stored raw (pooling precedes norm and RoPE,
   `:533-538` comment) in a separate MQA cache that tracks the attention
   cache cell for cell (assert `:308-311`). llama stores one per HC lane;
   since the keys derive from the shared mixed input the lanes are
   numerically identical copies — the engine keeps **one raw-key timeline of
   128 per token** (conceptual single-stream math; oracle confirms at M4).
4. **Blocks**: ratio r = 4. Block b of a stream covers cells/positions
   `[b·r, (b+1)·r)`. The pooled key of a block is the **mean over the valid
   member cells only when the block is complete**; an incomplete non-tail
   block's partial pool is garbage and is **never selected** (bias −INF,
   `:445`), so the engine may skip pooling it. Incomplete **tail** blocks
   (cells ≥ `(q+1)/r·r` of the query position q) are force-visible
   (`+1e9`, `:436-446`) — the engine masks them in directly.
5. **RoPE**: pooled keys are normed (`index_k_norm`, RMS over 128) and
   rotated with the block position `b·r` — the **first cell of the block**,
   not the last (`:374`, all four MRoPE sections carry it). Queries are
   normed (`index_q_norm`) and rotated per token. The rope call is the same
   `ggml_rope_multi(n_rot, sections, …)` used by full attention, with the same
   `n_rot` — the indexer does **not** scale the rotary width to its own head.
   `n_rot` is the context's `hparams.n_rot(il)` = the GGUF's
   `rope.dimension_count`, already carrying `partial_rotary_factor`
   (`llama-model.cpp:1335-1338`, `llama-hparams.cpp:85-91`), so for the 125B it
   is 0.25 × 256 = **64** and the 128-dim indexer head rotates its first half.
   The engine reuses its 3.6-validated MRoPE implementation on those heads.
6. **Scores**: `score[b, token] = relu-sum over the 4 q heads of
   q[head]·pooled[b]` — per-head relu **before** the sum (`:576-584`).
7. **Selection width**: `width = min(n_kv, indexer_top_k + r − 1)` =
   min(n_kv, 2051) (`:607`). Top-k runs on per-cell expanded scores; the
   causal mask is already added so future cells drop out naturally; blocks
   beyond the causal frontier of the query never survive, so the effective
   visible set is whole blocks from the tail back plus the tail itself. When
   `n_kv ≤ top_k + r − 1` (= 2050), width = n_kv → every cell is selected →
   the mask is the plain causal mask (**dense fast path**).
8. **Mask construction** (`build_attn_qsa:659-683`): start from the causal
   mask, fill −∞, set the top-k cell rows to 0 (`ggml_set_rows`), add the
   original causal mask back — visible cells = (selected ∩ causal).
9. **Attention** then runs over the visible cells with `kq_scale = 1/√256`
   and the standard V rotation/de-rotation the 3.6 dense path already does
   (`:628-637`, `:693-695`; rotation is self-inverse).

Block/cell layout internals llama keeps in the cache (per-stream cell
arrays, tail padding) reduce, for a single stream of real tokens, to: cell j
holds token j's raw key; block b pools tokens `[4b, 4b+4)`; `n_blocks =
ceil(n_kv/4)`; block positions are `4b`; the tail block (which holds the
last 1–3 tokens) is always visible.

## PLE n-gram math (locked)

`build_inp_ple` (`:1101-1120`), `llm_graph_input_ple::set_input`
(`:990-1051`), `build_ple` (`:1122-1213`), loads (`:127-141`, `:192-199`).
Host-side hash — ggml has no int64/xor (`:962-964` comment).

**Hash and head layout** (`:1036-1047`; heads 0–7 = bigram, 8–15 = trigram;
per-head constants from `ple_embedding.layer_multipliers`,
`ngram_heads_offsets`, `ngram_heads_vocab_sizes`, all int64 in the
checkpoint):

```text
ctx[0] = token t                       # the token's own EOS does not cut its context
for s in 1..2: ctx[s] = (EOS or missing) ? eos : token t-s   # sticky cut: the first
    # EOS (or the sequence start) going back freezes everything further back as EOS
for n in 2..3:                        # n = gram size
    mixed = (UInt64)ctx[0] * m[0]
    for j in 1..<n: mixed ^= (UInt64)ctx[j] * m[j]      # 64-bit wrap multiply
    base = (n-2) * 8
    for g in 0..<8:
        h = base + g
        row = mixed % vocab[h] + offset[h]              # each head has its OWN
                                                        # vocab size + offset
```

All arithmetic is **unsigned 64-bit wrap** (`m[0]` etc. are 45-bit
multipliers; the HF conversion keeps them exact and never floats them —
`qwen4exp.py:38-51`). `offset[h] + vocab[h] ≤ total rows`
(320,001,536). Row → part p = `row / 2_500_012`, offset in part =
`row % 2_500_012` (equal 128 parts; re-verify all 128 shard rows at M1
census).

**Gather**: per token, 16 rows in head order (row of head h at column
`t·16 + h`) → one 2560 = 16×160 vector per token (`:1114-1117`).

**Forward** (`build_ple:1122-1213`), layer 1 only, before that layer's
attn HC mixer (`:332-334`):

```text
key   = key_proj[10240,2560]   @ emb            # per token [10240]
value = value_proj[2560,2560]  @ emb            # [2560]
key    = grouped_rms(key,    norm_key)          # reduce over one 2560 stream,
query = grouped_rms(hidden, norm_query)         # scale by the [10240] gamma
s     = sum over 2560 of key*query per stream   # [4] per token
s    /= sqrt(2560)
gate = sigmoid(sign(s) * sqrt(clamp(|s|, 1e-6, inf)))      # per stream
v3   = value broadcast across the 4 streams
gated = v3 * gate                              # [10240] per token
normalized = grouped_rms(gated, norm_conv)
# dilated causal depthwise conv, kernel 4, dilation 3 = n-gram size:
#   out[c, t] = sum_k w[k, c] * x[c, t - (3-k)*3]          # taps t, t-3, t-6, t-9
conv_out = silu(conv(normalized))
hidden += gated + conv_out                      # BOTH terms are added
```

- Norm weights `norm_key/norm_query/norm_conv` are zero-centred (+1 baked,
  `qwen4exp.py:133-136`). `hc_norm` is not.
- `norm_query` normalizes the **HC plane itself** (the 4 streams of
  `hidden`, `:1143`).
- The conv weight is HF `[10240, 1, 4]` → squeezed to `[10240, 4]`
  (`qwen4exp.py:138-139`); ggml applies column k as one weight per channel
  with tap `(3−k)·3` (`:1183-1206`); the engine stores it squeezed like the
  GDN conv1d (fp16 raw) and applies the same orientation with dilation 3.
- Conv state: last `hist = (kernel−1)·dilation = 9` inputs per channel
  (`:1171`), causal, per sequence; a chunked prefill prepends the history so
  it matches a single-shot prefill (`:1168` comment).
- The MQA/EOS reset: hash ctx positions behind an EOS (or the sequence
  start) read as the PLE EOS token (checkpoint `eos_token_id`, `qwen4exp.py`
  takes `eos[-1]`); the current token's own EOS does not cut its context.
  Image batches (not used by the text engine) would stand in the image token
  id (`:996-1001`).

## 3.8 deltas vs the 3.6 GDN (locked)

GDN internals (projections, conv, l2-norm, beta/gamma gate, per-value-head
fp32 state, out projection) are **identical to 3.6** — see `QWEN36_PORT.md`
"GDN decode math". The one numerical difference (`build_norm_gated`,
`:411-421`): the output gate is **`sigmoid(z)`** where 3.6 uses `silu`. 3.8
runs no `input_layernorm`/`post_attention_layernorm` — HC mixers replace
them — and the gated-norm weight scales the whole 6144-wide value stream per
channel exactly as in 3.6 (`linear_attn.norm.weight`, plain RMS scale, no
+1).

The full-attention layer body is the 3.6 one (doubled-q `q_proj`, q|gate
interleaved per head, `q_norm`/`k_norm` zero-centred +1, sigmoid output
gate, per-head softmax 1/√256) with the QSA mask replacing the dense causal
mask on those layers, plus `out_proj`. MoE tail identical to 3.6 (router
per-expert scale, 512 experts top-10, shared expert with sigmoid gate).

## GGUF tensor-name map (M4 oracle harness)

From `constants.py` (per-layer names use `blk.{il}.`; qwen4exp uses no
`.weight` suffix on `mlp.experts.*`):

| GGUF | HF (`language_model.layers.{il}.`) |
| --- | --- |
| `blk.{il}.hc_attn_norm` / `_down` / `_up` / `_inject` | `attn_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}` |
| `blk.{il}.hc_ffn_norm` / `_down` / `_up` / `_inject` | `mlp_hyper_connection.{…}` (same four) |
| `output_hc_norm` / `_down` / `_up` | `hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` |
| `blk.{il}.indexer.q_proj` / `k_proj` | `self_attn.indexer.index_qk_proj` rows [:512] / [512:] |
| `blk.{il}.indexer.q_norm` / `k_norm` | `self_attn.indexer.q_layernorm` / `k_layernorm` (+1) |
| `blk.{il}.ple_key` / `ple_value` | `ple.key_proj` / `value_proj` |
| `blk.{il}.ple_norm_key` / `_query` / `_conv` | `ple.norm_key` / `norm_query` / `norm_conv` (+1) |
| `blk.{il}.ple_conv1d` | `ple.conv1d` (squeezed) |
| `per_layer_token_embd.weight` | 128 × `ple_embedding.ngram_embedding.shard_{0..127}.weight` concatenated in index order |
| (hyper params) | `hyper_connection.count` 4, `low_rank` 320; `attention.indexer.head_count` 4, `key_length` 128, `top_k` 2048; `attention.compress_ratios` [4 at full layers, else 0]; `ple.layers` [1], `ngram_size` 3, `heads_per_ngram` 8, `conv_kernel` 4, `layer_multipliers`/`head_offsets`/`head_vocab_sizes`, `eos_token_id`, `image_token_id`; `embedding_length_per_layer_input` 160 |

## Quant / packing plan (M1 target; mirrors the 3.6 policy per class)

Bits: embedding 4, attention 4, linear-attention 8, router 8, shared 4,
routed 4 (same widths as the 3.6 install). Quant classes:

| Tensor class | .finch treatment |
| --- | --- |
| `embed_tokens`, `lm_head`, `self_attn.{q,k,v,o}_proj`, HC `input_mix_weight_down/up`, `block_inject_weight`, root `hyper_connection_mixer.{down,up}` | int4 affine, row-major, same class as 3.6 attention projections |
| GDN `in_proj_qkv/z/b/a` | int8 affine (3.6 linear-attention policy) |
| `linear_attn.norm.weight` | raw BF16 (`requireBF16`) — 3.6 precedent |
| PLE `key_proj`/`value_proj` | int8 affine (3.6 linear-attention policy) |
| `ple.norm_key/query/conv`, `indexer.q/k_layernorm`, `attn.q_norm/k_norm` | **1+w baked** (zero-centred gammas; GGUF bakes +1) |
| HC `hc_norm.weight` (attn/ffn per layer + root) | raw BF16 — plain RMS scale, no bake |
| `conv1d` (GDN, PLE) | raw fp16, squeezed [C·4] (3.6 policy; PLE dilation applied in-kernel) |
| `A_log`, `dt_bias`, `router.per_expert_scale`, … | raw fp32 / int8 per 3.6 planner rules |
| `ple_embedding.ngram_embedding.shard_*.weight` | **raw BF16 part files** `ple_shards/shard_%03d.bin`, row-major straight copy, 128 × 2,500,012 × 160 ≈ 102.4 GB total (no quant this pass) |

Layout: the 512-expert/48-layer `layout.json` lands ≈ 45–55 MB — the reader
and repack-side validator caps must rise 64 → 128 MB in lockstep.

## Implementation plan

### Done and verified

M0 (commit `4de4774`, 2026-09-08): ArchConfig preset
`qwen3_8_flashNext_125B` (`ModelTypes.swift:290-333`) with the 14 new fields
zero-defaulted for the other presets; manifest wire keys optional +
omit-on-nil (format minor stays 0); `ManifestReader` arch enforcement;
repack-side `ArchInfo` family/field parse (1-based → 0-based
`pleLayerIndexes`; ngram geometry frozen 160 × 2,500,012 pending M1 census);
tokenizer maps `qwen4_exp` into the shared `.qwen3_6` family. Tests:
`Qwen38ArchModelTests` (incl. the bare-family-literal scan),
`Qwen38ManifestWireTests`, `ArchInfoTests` additions. The preset matches
`config.json` field-for-field (verified against the checkpoint's
`config.json` `text_config`).

### Remaining work, in order

0. **Machine facts (Stage-0 gates, all closed 2026-09-09).** `index_qk_proj`
   split = q rows [:512] then k [512:] and the +1 bakes verified from
   `qwen4exp.py`; PLE conv1d orientation + dilation 3 from `qwen4exp.cpp`;
   llama's per-block rope/bias fills from `llama-memory-hybrid-idx.cpp`
   (blk_pos = block's **first** cell position `b·r`, all four mrope
   sections; incomplete tail always visible); machine RAM 16 GB; CLI flags
   from `Sources/FinchMoERepack/Command/main.swift`: repack =
   `FinchMoERepack --input-snapshot <dir> --output <model.finch> --overwrite`
   (no family flag — M1 detects the family from the snapshot's `model_type`
   instead of growing the CLI surface).

1. **M1 repack support (toy-first; 3.6 output stays byte-identical).**
   Family parameter on `QwenRepackPlanner.plan(...)`; `residentName(for:)`
   family-aware (3.8: `model.language_model.` → bare `language_model.`, no
   inner `.model.`); exclusion keeps `model.language_model.*` + `lm_head`
   (drops `mtp.*` and `model.visual.*` — assert in a test); PLE part-file
   plan (128 raw-bf16 streaming copies, each part its own file
   `ple_shards/shard_%03d.bin` riding `manifest.files` — no new wire keys,
   no layout.json `pleShards` entry); transforms per the table above
   (`ple.`/`indexer.`/HC names
   bypass the generic int4 fallback); `qwenResidentOrdering()` without
   `model.language_model.norm.weight`; layout caps 64 → 128 MB in reader +
   validator + repack side, same commit; `LocalQwenRepacker` family
   parameter (default 3.6 → unchanged output), modelID
   `local/Qwen3.8-Flash-Next-125B`, PLE part-write phase + audit entries.
   Extend `SyntheticQwenSnapshot` with a scaled qwen3_8 tree (tiny PLE: 4
   parts × 32 rows). **Census re-verification items**: all 128 PLE shard
   shapes; whether layer tensors carry `layer_scalar`/`per_expert_scale` in
   3.8 (llama reads no layer scalar for this arch); exact names of the
   language_model root extra tensors the planner must skip or map.

2. **M2 engine load + runtime schema.** Generalize the per-family layer
   prefix through the hardcoded deep-prefix accessors in `Model.swift`
   (`qProj/kProj/vProj/oProj`, `qNorm/kNorm`, `inputNorm/postAttnNorm`,
   `finalNorm`, `layerPrefix`); `finalNorm` is nil for 3.8. New accessors:
   per-layer `attnHyperConnection`/`mlpHyperConnection` + root
   `hyperConnectionMixer`; `indexerQKProj`/`indexerQLayernorm`/
   `indexerKLayernorm` (full layers only); PLE head accessors + the
   128-part opener (names from the manifest's `ple_shards/shard_%03d.bin`
   entries, lazy open like `PreadExpertStreamer`). `validateQwen38Layers`
   (sibling of
   `validateQwen36Layers`) replaces the M2 throw: per-layer by kind (full:
   6 self_attn + 3 indexer, doubled q [12288,2560]; GDN: the 3.6-shaped set
   at qkvDim 10240), both HC 4-tuples per layer, layer-1-only PLE set + 128
   part entries, root mixer trio, no `model.norm`; readers reuse
   `requireBF16`/`requireRaw`/affine — no new reader classes.

3. **M3.1 hyper-connection (decode).** Kernels: grouped RMS over the 4
   stream planes with the 10240 gamma; down→silu(·/4)→up→sigmoid mix GEMMs;
   mean-input; 2·sigmoid(·/4) block-inject add; root-mixer variant.
   4-stream runner state + scratch; `encodeQwen38DecodeLayer` (full/GDN
   bodies cloned from `encodeQwenDecodeLayer` with HC replacing the norms
   and the GDN gate at sigmoid); family-aware head path through the root
   mixer; replace both layer-loop dispatch gates (`RealForwardRunner.swift`
   decode `:3012-3016`, prefill `:842-847`) using `isQwenHybrid`/
   `isQwen3_8` helpers only. fp32 `HyperConnectionRef` + real-shape tests
   before wiring; `qwenLayerDebugHook` "hc.pre/hc.post" snapshots.

4. **M3.2 QSA indexer (decode).** Built. Separate indexer raw-key timeline
   (128 per token) allocated like the KV cache on full layers only; kernels:
   index q/k projection, raw-key store, 4-token block mean pool, RMS +
   partial-rope at block position 4b, per-head relu-dot block scores, and a
   cell-granular top-k over the causal-masked candidate set (the cut is by
   *cell*, not by block — the `+r−1` slack only guarantees whole blocks fit);
   V rotation follows the 3.6 dense path.

   The selection reaches attention **by omission, not by an −INF mask**: a
   dedicated `attention_decode_cells_partial` walks the selected cell list,
   and an unselected cell never enters the running softmax maximum, so it
   carries no mass. This runtime therefore needs no `[n_kv]` mask buffer, and
   `encodeFull` — which never had a mask argument — is untouched; causality
   is implicit in its `[0, seq_len)` scan. While the selection width covers
   every causally-visible cell the ranking is skipped and the dense path
   answers, which for `n_kv ≤ 2050` is the same answer. `QSAIndexerRef` +
   edge-case tests (window boundary, budget crossing, incomplete tail).

5. **M3.3 PLE (decode).** Built. Host hash (UInt64 wrap, EOS window reset) →
   row addresses (`PLEHost`, CPU by necessity — ggml has no int64 xor to hash
   with, and the gather is 16 rows × 320 B = 5 KB per token, so nothing is
   cached and nothing is resident); ≤16-row gather through the part-file
   streamer; key/value projections (int8, riding the `linearAttention`
   manifest slot — `requireAffine(..., quant.linearAttention)` at load, so
   `int8GEMV` is provably the right kernel); grouped norms; sigmoid
   sign·√|s|/√2560 gate; 4-stream broadcast add; dilated causal depthwise conv
   (kernel 4, dilation 3, history 9, per-sequence state) + silu; both gated and
   conv terms add into the layer-1 HC plane **before its attn mixer** (llama's
   `t_layer_inp = res_hc` → PLE → `build_hc_mix` order; in the runner:
   `gPLE?(cb)` immediately ahead of `gAttnMix(cb)` in both layer bodies, so
   `hc.pre` holds the pre-PLE plane and `hc.mid` the post-PLE one).
   The window is an `ngramSize`-entry ring, not a token history: this engine
   never rewinds the KV cache (`ServerPromptCache.match` only hits where
   `kvPosition == kvBackedTokenIDs.count`), so a resume continues from the
   cursor with the last tokens still in the ring — O(1) memory is what the
   16 GB budget requires, and `PLEHostTests.windowStaysBounded` pins it.
   `PLERef` (locked to `qwen4exp.cpp`) + `PLEReferenceTests`; `PLEHostTests`
   cross-checks the ring against the reference's full-token-array window;
   `pleHashMetadataAndGatherMatchTheCheckpoint` pins the int64 constant
   round-trip and the row→part→byte gather against the repacked install;
   `pleTermsLandInTheLayerOnePlane` proves the table's contribution lands
   exactly there — two installs from one seed, one with zeroed parts (a
   provable no-op), bit-identical up to `hc.pre` of layer 1 and different from
   `hc.mid` on. Runner cost: ~490 KB of buffers at the real geometry, of which
   two 180 KiB conv-state rings kept separate rather than aliased.

6. **M3.4 prefill + M3.5 toy e2e.** Chunked counterparts (block-granular
   indexer over the chunk, conv-chunk separate buffers, scratch accounting,
   `prepareForContinuation`/reset clears PLE conv + indexer state);
   layer-debug-hook replay on layers 1 and a full layer; deterministic toy
   decode + prefill vs the fp32 replay within `fp16ChainedReduction`;
   toy CLI smoke.

7. **M4 real install + oracle.** Safe-run repack of
   `models/Qwen3.8-Flash-Next-bf16` → `models/Qwen3.8-Flash-Next-125B.finch`
   (~40 GB quantized core + 102.4 GB PLE + overhead ≈ 145 GB; disk free
   1.1 TiB; transient peak ≈ 290 GB during rename; 2–6 h streaming,
   release binary only). Real-shape gated tests (PLE part-boundary rows,
   real-vocab hash collisions, indexer crossing 2048). Oracle: build
   `archive/llama.cpp`, run the AD quant GGUF on a fixed prompt (mmap; on
   16 GB expect slow — it is a one-shot run and must be alone), dump
   logits; engine CLI same prompt + logits dump; bar: top-1 agreement
   ≥ 50/64 with top-5 overlap and rank-correlation sanity (engine int4
   group-64 vs IQ4XS → argmax agreement, not exact logits). Tokenizer
   `qwen3_8` case in the shared qwen family; `AppModelInstallationProbe`
   descriptor; full suite both families.

Deferred (documented here): PLE table quant; MTP; vision; indexer cache
compaction. `docs/QWEN36_PORT.md` remains the GDN/rope/mrope authority and
the 3.6 hardware findings carry over.

### Machine: 2026-09-05 kernel panics and the 16 GB operating protocol

Identical history and protocol as `QWEN36_PORT.md` (16 GB Mac mini,
Mac16,10; watchdog panics from memory/IO thrash — the tell is a fresh
`panic-full-*.panic` + `ResetCounter` pair). Additional 2026-09-09 facts:
`hw.memsize` 16 GB; disk free 1.1 TiB; the M4 oracle run on the 79 GB GGUF
must be the only heavy process (mmap pages the model; 16 GB RAM → slow but
bounded — keep `-c` small, no GPU offload); the repack reads 352 GB and
writes ~145 GB on the same external volume — staged checkpoints with fsync
per file class, kill-and-resume = rerun with stale-partial cleanup, never
two heavy jobs at once.

**2026-09-10:** a third panic of the same signature killed a `swift build`
mid-M3.2d (boot 07:20, stale `.build/.lock` left behind, Spotlight then
reindexed the external volume for ~40 min — load average 18). Three panics
from the same cause means the protocol needs to be a tool, not a habit:
`tools/memguard.sh` wraps any command, refuses to start below 60% free
(the documented gate), and during the run kills the whole process group if
free memory falls under 12%, the compressor passes 4 GB, or swap is in use
— the three states that precede the watchdog hang. Builds and test runs go
through it: `tools/memguard.sh -- swift build -j 3`.

**Indexer footprint (matters at the real geometry).** `QSAIndexerState`
allocates per full-attention layer: `rawKeys` = `maxContext · idxDim · 2`
bytes, `pooled` = `⌈maxContext/r⌉ · idxDim · 2`, plus ~8 KB of
`scores`/`cells`. At the engine's default `maxContext` 4096 that is ~16 MB
across the 12 full layers — nothing. It scales linearly with context
though, and at the model's full 262144 it is ~1.0 GB, i.e. it must be
charged against the KV budget before anyone sets a long context on this
box. The raw timeline is not windowed: every complete block is scored on
every step, so there is nothing to shrink it to.

## Repository state

- `Sources/FinchMoE*/` + `Tests/FinchMoE*/` — the working engine (M0 state
  described above). Format family strings single-sourced in
  `FinchFormatV1.swift`; preset/helpers in `ModelTypes.swift`; the load
  gate and accessors in `Model.swift`; family dispatch in
  `RealForwardRunner.swift`.
- `archive/llama.cpp/` — the math authority: `src/models/qwen4exp.cpp`,
  `conversion/qwen4exp.py`, `conversion/qwen.py`,
  `gguf-py/gguf/constants.py`, plus `src/llama-memory-hybrid-idx.cpp` for
  the QSA cell/block fills.
- `models/Qwen3.8-Flash-Next-bf16/` — 352 GB bf16 checkpoint (131 shards).
- `models/Qwen3.8-Flash-Next-AD-3.84bpw-IQ4_XS-M64/` — 79 GB oracle GGUF.
- `models/Qwen3.8-Flash-Next-125B.finch/` — M4 output (does not exist yet).

## References

- `QWEN36_PORT.md` — GDN/mrope/RMSNorm conventions, validated numerics, the
  machine protocol, and the plan template this document follows.
- `SYSTEM_DESIGN.md` — `.finch` layout, streaming, Metal conventions the new
  kernels must follow.
- llama.cpp qwen4exp sources listed under "Math authority".
- Kernel/reference/test templates for the new units: the GDN set named in
  `QWEN36_PORT.md` References, plus the existing full-attention encode path
  (`encodeQwenDecodeLayer`) and `qwenLayerDebugHook` replay harness that
  M3 extends.

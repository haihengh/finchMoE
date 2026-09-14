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

Status: **M0–M4 landed; the oracle cross-check ran and its evidence is
argmax-level, not cosine-level.** The real 174,403,168,940 B (162 GiB)
install at `models/Qwen3.8-Flash-Next-125B.finch` loads, prefill/decode are
coherent and inside the 16 GB budget. The oracle now runs reliably and both
prompts agree on tokenization (byte-exact), argmax, and generation
(`" Paris"`), with top-10/100 logit cosine 0.995/0.985 — but the whole-vocab
cosine is 0.88, under the plan's 0.95 bar, because the bar was taken from a
same-weights comparison and this one is cross-quantization. Full account and
what it does *not* establish under `### Remaining work` item 7 below. The qwen3_8
load-gate throw ("qwen3_8 installs need the Flash-Next engine (M2)") is
gone. **EvalPlus HumanEval on that install: base pass@1 0.945 (155/164),
HumanEval+ 0.921 (151/164)** — +3.7/+4.3 pt over the 3.6 install, with 3.8's
failure set a strict subset of 3.6's; the one apparent regression is a
768-token truncation artifact (item 8 below). The plan ran M1 (repack) → M2 (load/schema) → M3 (forward:
hyper-connection → QSA → PLE, decode then prefill) → M4 (real repack +
llama.cpp oracle cross-check) under the machine protocol at the bottom. The
user directive
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
`pleLayerIndexes`; ngram geometry 160 × 2,500,012 — **confirmed by the M4
census against the real tensor** `[2500012, 160]` in `model-00005`; the
config's `ngram_vocab_size_base: 20000000` is a nominal base, not a row
count);
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

6. **M3.4 prefill + M3.5 toy e2e. Done** (M3.4 `7973f83`; M3.5 with this
   commit). Chunked counterparts (block-granular indexer over the chunk,
   conv-chunk separate buffers, scratch accounting,
   `prepareForContinuation`/reset clears PLE conv + indexer state);
   layer-debug-hook replay on layers 1 and a full layer
   (`Qwen38DecodeWiringTests.prefillChunkMatchesDecodeSteps`); deterministic
   toy decode + prefill vs the fp32 replay
   (`Qwen38ToyReplayTests`, 50 and 45 stages compared, all inside
   `fp16ChainedReduction`); toy CLI smoke (`Qwen38ToyCLISmokeTests`).
   26 tests / 7 suites green, memory flat (72% free, 2.6 GB compressed peak).

   Three findings M3.5 produced, all now encoded in the tests:

   * **The CLI could not load any non-production geometry.** `Run.swift`
     resolved the expected arch from the *family's built-in preset*
     (`ManifestReader.detectPreset`), so an install declaring `qwen3_8` was
     required to be 2560-wide — a deliberate cross-check that is right for a
     real install and made the toy unloadable. `detectPreset` gained
     `allowManifestArch`, driven by `FINCHMOE_EXPECT_ARCH=1` and defaulting
     to off: the production path is byte-identical, and the override points
     the check at the arch the manifest declares about itself (so it still
     catches an internally inconsistent manifest, no longer a
     wrong-but-consistent one). `.fullSha256` content hashes are a separate
     check and still run.
   * **Layer 3's last-row prefill chain does not converge, and cannot.**
     The replay may only share a chunk's last row (`snapRow = t − 1`), never
     rows 8…10 of layer 3's KV timeline, and the QSA sparse selection turns
     that residue macroscopic — the row's *own* attention input is
     bit-identical between the two paths (`3|attnBlockIn` = 0.0) yet its
     output leaves at 0.07423675, so what the selection amplifies is the KV
     residue, not anything about the row's input. The control is the engine
     itself: its *own* decode and chunk paths differ at the same stage by
     0.09765625 — a *longer* distance than the replay's — seeded at layer 0
     where `MoeTailRef`'s chunked reduce takes fp16 `routePartials` while
     decode's fused `moe_phase2_down_reduce_k8` keeps fp32.
     `T38.prefillAmplified` pins the seven measured stages as a *subset*
     check — each ceiling is its measurement rounded up to two decimals, and
     a stage that joins the list fails — while the same stages hold at
     ≤ 3.4e-3 in decode, where every row is anchorable. Closing it is an
     engine change.
   * **That whole set was re-measured after the `hc_norm` fold was
     corrected (M4)**, because it had been measured in a wrong-gate regime.
     The hyper-connection mixer is the 3.8 residual backbone *and* its output
     norm; taking its gate raw left the residual plane sign-scrambled, so
     every ceiling above was a wrong-gate number. Under the old gate the same
     run measured 0.49819666 / 0.3041551 / 0.22048835 / 0.20131938 /
     0.1734365 / 0.062440872 for the layer-3 set and 0.50025904 for the
     control. The corrected gate is better conditioned as well as right: the
     layer-3 set shrank ~6-9×, `3|hc.mid` dropped out of it entirely
     (0.05141066 → 4.9e-3, so it is now held at `tolerance` — 0.010184287,
     tighter than the 0.06 ceiling it had), and only `2|recState` moved the
     other way (0.009862052 → 0.013246425).
   * **The toy's first greedy token is `<|im_end|>`**, so the CLI run ends
     `stop=eos new=1tok` with an empty stdout delta (a special token has no
     detokenized text). The smoke therefore asserts on the stderr footer
     (`prefill=13tok` — the sidecar tokenizer's 13 byte tokens, and proof the
     engine loaded and prefilled) rather than on streamed text, which is a
     property of the toy's seeded-noise argmax and not of the plumbing.

7. **M4 real install + oracle.** Safe-run repack of
   `models/Qwen3.8-Flash-Next-bf16` → `models/Qwen3.8-Flash-Next-125B.finch`
   (**174,403,168,940 B (162 GiB) measured**, not the ~145 GB estimated
   before the build: 3,877,265,688 B resident `model_weights.bin` + ~63 GB
   `packed_experts/` + 102.4 GB `ple_shards/`; disk free 980 GiB; the
   transient ≈ 290 GB peak applies only to the *overwrite* case, which this
   run was not; measured duration: see the M4 status entry below). Source
   snapshot is 360,000,192,888 B of safetensors (335.3 GiB) over 131 shards —
   that is what the repack reads, since it opens only the shards named in the
   index `weight_map`; the directory totals 377,446,183,588 B (351.5 GiB) but
   the 16 GiB difference is a HuggingFace `.cache/` the repack never touches.
   Real-shape
   gated tests (PLE part-boundary rows, real-vocab hash collisions, indexer
   crossing 2048). Oracle: build `archive/llama.cpp`, run the AD quant GGUF
   (**79 GB**, 28 shards) on a fixed prompt (mmap; on 16 GB expect slow — it
   is a one-shot run and must be alone), dump logits; engine CLI same prompt
   + logits dump. **The method is the 3.6-proven single-final-prefill-row one,
   but its bar does not transfer.** cos 0.998213 / argmax MATCH / top10 10/10
   (`archive/README.md:62`) was measured by running *one* weight file through
   both engines, so the only difference was bf16-vs-bf16 numerics. This
   comparison runs *two different quantizations* — engine int4 group-64
   repacked from the BF16 source vs IQ4_XS 3.84 bpw — through two engines, so
   a cosine drop is expected and "treat cos < 0.95 as failure" was the wrong
   threshold for it. The tokenizer side is already closed — M0 mapped
   `qwen4_exp` into the shared `.qwen3_6` family (see the M0 entry above), and
   3.8 shares the 3.6 tokenizer byte-for-byte, so no `qwen3_8` case is needed.

   **Oracle run, 2026-09-11 — engine consistent, plan's bar mis-derived.** Two
   prompts, each scored against `llama-debug --save-logits` (final prefill
   row, `examples/debug/debug.cpp:184-216`: one `llama_decode`, no sampling
   loop) and the engine's `FQ_DUMP_PREFILL_LOGITS`:

   | | 5-token | 15-token |
   |---|---|---|
   | tokenization | identical | identical |
   | argmax | `11751 ĠParis` MATCH | `11751 ĠParis` MATCH |
   | full-vocab cos | 0.889502 | 0.880557 |
   | top-10 cos | **0.99493** | **0.99565** |
   | top-100 cos | **0.98387** | **0.98468** |
   | top-1000 cos | 0.96568 | 0.96034 |
   | tail (10k-248k) cos | 0.89309 | 0.88018 |
   | top-10 overlap | 6/10 | 5/10 |
   | top-100 overlap | 55/100 | 64/100 |

   Both prompts generate `" Paris"`. Full-vocab cosine misses 0.95, and the
   *longer* prompt is slightly **worse** — so the "short prompt, noisy tail"
   reading is falsified as well. But the full-vocab number is measuring the
   wrong thing: **98.08% of the reference logit vector's squared magnitude
   lives in the bottom 247,000 of 248,320 dimensions**, so the whole-vector
   cosine is essentially just the tail band (0.8802 there vs 0.8806 overall),
   where both sides are near-uniform noise. The bands that decide behaviour
   agree at 0.96-0.996. This is the signature of quantization noise, whose
   absolute scale tracks logit magnitude: ~14 nats rms at the top leaves a
   ~1.4-nat perturbation, which preserves the 2.5-nat `ĠParis` margin but
   reshuffles ranks *within* the top ten (hence 5-6/10) and swamps the
   2.9-nat tail. A structural bug — wrong tensor, head, or index — would
   corrupt the top of the distribution too.

   Ruled out along the way: **the PLE is not an approximation on our side.**
   GGUF `per_layer_token_embd.weight` is `[160, 320001536]` at Q5_1 (6.0 bpw,
   24 B per 32 elements); ours is that same 320001536x160 table at **bf16**,
   sharded 128 x 2500012 rows. Dequantizing 15 sampled GGUF rows against ours
   gives corr **0.9993** (same weights, same rows — which also validates the
   Q5_1 reader and the row alignment) with **3.84% relative quantization
   error in the reference**. In this pathway llama.cpp is the lossy side. That
   is also where the install size goes: **95 of 162 GiB is PLE at bf16** —
   59% of the install to keep one tensor where the reference ships 6 bits.

   **What this does not establish.** There is no same-weights cross-engine
   comparison for 3.8 available, and there cannot be one today: the Swift
   engine has no GGUF reader (the 3.6 GGUF path was not carried over in the
   2.0 merge — `Sources/` matches `gguf` only in a comment). So the 3.6
   standard of evidence is unreachable without writing a 3.8 GGUF loader.
   Until then the honest reading is: tokenization byte-exact, argmax identical
   on both prompts, correct generation, top-of-distribution agreement at
   0.96-0.996 — consistent with a correct engine, short of proof.

   **Two operational traps, both cost a run.** `-nr` / `--no-repack` is
   load-bearing: `--repack` is on by default and makes *anonymous* tensor
   copies, which are compressible and feed the compressor (eager ~10 GB,
   killed at 5.0 GB with no output at all); with `-nr` the weights stay mmap'd
   file pages, which are clean and evictable, so RSS can read ~10.2 GB while
   the compressor peaks at only 2.2-2.3 GB. And `DYLD_*` is stripped at every
   SIP-protected exec boundary, so a `DYLD_LIBRARY_PATH` exported outside
   `tools/memguard.sh` never reaches `llama-debug` — the inner shell must set
   it and `exec` directly. A `pty.spawn` wrapper is also needed, because
   llama.cpp's INFO goes to stdout and is block-buffered when redirected, so a
   guard kill loses the whole log. See [[llamacpp-archive-tooling]].

   **Full suite green, both families (2026-09-11)** — 866 tests in 151 suites.
   Both real installs are covered (`QwenRealInstallLoadTests` for the 3.6,
   `Qwen38RealInstallLoadTests` for the 3.8), as are `Qwen38ToyReplayTests`,
   `Qwen38EngineLoadTests`, `Qwen38ToyCLISmokeTests` and
   `QwenLayer0DebugTests`. It does not run green as a single
   `swift test --no-parallel`, and the account below is the second attempt at
   explaining why — the first two explanations were both wrong.

   Three things had to be fixed before it would get there, all in
   `QwenLayer0DebugTests`:

   * **A `-1` index sentinel trapped the whole test process.** The sweep's
     `var idx = -1` only advances when `d > m`, so when *no* element differs —
     or every comparison is unordered because a plane carries a NaN — it stays
     at `-1` and the report's `a[idx]` raises `Index out of range`, which
     Swift's runtime turns into SIGTRAP: the suite dies with no `✘` line at
     all (this is why two earlier full-suite runs reported a bare
     `signal code 5`). Five sites subscripted arrays this way; they now route
     through `QwenLayer0DebugTests.worstDelta` / `describeDelta`, which report
     "no element differs (nan=…)" instead of trapping. The remaining `= -1`
     sites in that file are print-only or already guarded.
   * **`repackWeightsMatchBf16Checkpoint` widened 3.65 GB of bf16 into
     7.29 GB of fp32 to compare ~20 MB of rows.** `readBF16` reads a whole
     tensor, and three of its callers then used a handful of rows of it:
     `lm_head` and `embed_tokens` are `[248320, 2048]` — 1.017 GB each in
     bf16, 2.03 GB widened to `[Float]` — and the test looked at 8 rows and at
     5 rows respectively; `experts.gate_up_proj` (all 256 experts, 1.074 GB →
     2.15 GB) and `experts.down_proj` (0.537 GB → 1.07 GB) were read whole to
     compare **expert 0**, with both refs live in the same scope. A
     `readBF16Rows` sibling that bounds the widen to `rows * cols * 2` bytes
     — which matches each of those four tensors' `data_offsets` span exactly —
     leaves all 24 printed assertions byte-identical to the pre-fix run and
     takes the test from 13.46 s to 9.96 s. This is a real waste, but it was
     *not* the kill: measured alone the test peaks at 2.6-3.4 GB either way,
     because these are sub-5-second transients and the guard samples every 5 s.
   * **One process cannot hold all twelve heavy tests, and no ceiling fixes
     that.** With the trap gone the suite ran to completion for the first
     time, and four one-process runs bracketed the problem without explaining
     it. Run 3 at the default 4 GB died at 5.2 GB. Run 4 at
     `--max-compressed 6` was **green** — 866 tests in 151 suites, 200.1 s,
     low-water 47% free, swap flat at 342 MB. Run 5 at the *same* 6 GB died
     at 6.3 GB inside `multiStepStateContinuity`, right after
     `repackWeightsMatchBf16Checkpoint` passed at 14.18 s. Run 6 at 8 GB died
     at 8.9 GB inside `allLayerMixerSweep`. The peak tracks whatever ceiling
     is set and the outcome varies run to run — Swift Testing randomises test
     order — so the earlier claim here that "the ceiling that is actually
     needed is ~6 GB" is falsified. Measured one test per process under the
     **default** 4 GB ceiling, all twelve pass, 1.1-3.4 GB each:
     `multiStepStateContinuity` 1.8, `allLayerMixerSweep` 1.4,
     `layer0PostAttnMatchesFp32Reference` 1.4,
     `prefillSweepFindsFirstDivergence` 1.4,
     `prefillLayerTailsMatchReference` 1.3,
     `repackWeightsMatchBf16Checkpoint` 3.4, and the other five ≤ 1.5.
     It is cumulative, not a leak: the suite is a `struct` with no stored
     properties, `MetalContext` keeps only a per-instance pipeline cache, and
     the file has no static mutable state. Each test's working set is retained
     as compressed pages — free memory stays ~78%, so macOS never has a reason
     to drain them — and the next test allocates on top of the last.
   * **So the suite runs in two phases** (`tools/heavy-tests.sh`, both under
     the guard's defaults): everything except `QwenLayer0DebugTests` in one
     process (854 tests in 150 suites, 71.9 s, **1.5 GB** peak), then each of
     the twelve heavy tests in its own guarded process. That is the shape this
     16 GB box can finish. The guard's real protections were never approached
     in any run — worst **47% free** against a 12% floor, swap flat at
     342-883 MB against a 2048 MB limit — so the compressor ceiling was the
     only trigger that ever fired, and only for this one suite.

8. **EvalPlus HumanEval on the 3.8 install — DONE (2026-09-13).** The 3.6
   reference protocol (`QWEN36_PORT.md` §6) re-run against the 3.8 install
   through `archive/humaneval_evalplus/run_server_cell.sh` — a driver written
   for this run because the 3.6 cell was hand-driven and its command was never
   recorded, so neither that run nor its protocol could be re-checked
   afterwards. The protocol is deliberately *not* in the driver: it lives in
   `humaneval_gen.py` (system "You are a helpful assistant good at coding.",
   greedy T=0, 1 sample, top_p 0.95, 768-token cap via evalplus's
   `OpenAIChatDecoder`), and sharing that one file is what makes a 3.8 cell
   readable against a 3.6 one at all. EvalPlus 0.3.1, OpenAI backend on
   127.0.0.1:8080, no shim — `FinchMoEServer` is already OpenAI-compatible.
   - **Base pass@1 0.945 (155/164); HumanEval+ 0.921 (151/164)** — +3.7 pt and
     +4.3 pt over the 3.6 install's 0.909 (149/164) / 0.878 (144/164) on the
     identical rig.
   - Base fails (9): 32, 93, 116, 129, 130, 132, 145, 147, 163. 3.6 failed 15;
     the 7 it failed and 3.8 passes are 62, 95, 99, 113, 124, 134, 160. The
     only problem 3.8 fails that 3.6 passed is **116 — and that one is a
     truncation artifact, not a regression**: its solution ends mid-sentence
     on "Let me look at the", its `base_fail_tests` is `[]` (a crash
     signature, not a failed assertion — a truncated sample has nothing to
     fail on), and the server log puts it among the length-capped. **3.8's
     true failure set is a strict subset of 3.6's, so the larger model
     regresses nowhere.**
   - Reading the caps is the whole result. Of the 9 base fails, 5 are
     length-capped (32, 116, 129, 130, 147) and the other 4 (93, 132, 145,
     163) 3.6 failed too. Skip that check and 116 reads as the 125B
     regressing on a problem the 35B solved.
   - Generation: 164/164 in **23,771 s (~146 s/problem)**, against 3.6's
     6,786 s (41.4 s/problem) — 3.5× slower, which is the honest cost of the
     larger model at this decode path and worth stating plainly before anyone
     budgets another sweep. One continuous server instance, no restart: the
     164 per-request durations sum to 23,871 s over a 23,771 s wall-clock
     span, so there is no dead time to explain. 157 `finish=stop`, 7
     `length`-capped at 768 (3.6: 136 / 8); the capped set is 32, 64, 76,
     116, 129, 130, 147, from mapping the log's ordered `finish=` events onto
     task IDs (164 events, clean 1:1).
   - **What this is not.** There is no same-weights cross-engine comparison
     available for 3.8 and there cannot be one today — the Swift engine has no
     GGUF reader (see item 7) — so this is an engine-internal before/after,
     not the 3.6 standard of parity evidence. A larger model scoring higher
     through a correct engine and a larger model scoring higher through a
     subtly wrong one both look exactly like this.
   - Evidence: `quality/humaneval/finchmoe-qwen38_openai_temp_0.0.jsonl`,
     `.raw.jsonl`, `_eval_results.json`; scoring log and server log at
     `archive/humaneval_evalplus/results/finchmoe-qwen38_{eval.txt,server.log}`.
     The file is 164 lines with 164 unique task IDs — one clean sweep, no
     composed partial — because evalplus codegen **appends** rather than
     overwrites (the smoke slice's artifacts were moved to
     `results/humaneval/smoke_0_20/` first). The upside of that append is what
     made resume-by-range viable after a kill.
   - **Two harness findings, both cost a run.** (i) `tools/memguard.sh`'s
     default 4 GB compressor ceiling is *binding* for a 3.8 server run, not
     conservative: the server's steady-state working set is ~3.2 GB and this
     run started from a 0.8 GB baseline, so a pristine start lands at ~4.0 GB
     — exactly the ceiling — and the full sweep was killed at 4.1 GB a minute
     in. `FinchMoEServer` exposes no lever for it: `ServerArguments.swift` has
     no cache-slot flag and `--max-context` accepts only
     4096/8192/16384/32768/65536, so the 3.6-era `--max-context 2048` escape
     does not transfer — 4096 is the floor, not a tunable. This run therefore
     used a per-run `MEMGUARD_MAX_COMPRESSED_GB=6` env override with
     `tools/memguard.sh` itself unmodified, keeping the 12% free floor and the
     2 GB swap cap: the compressor holding more is a *symptom*, free% is the
     danger signal. (ii) **evalplus scoring, not the model, is the memory
     hog.** `evaluate.py` sets `n_workers = parallel or cpu_count() // 2` —
     five workers on this ten-core box, each compiling and executing test code
     — and run on top of the live server it reached **8.3 GB** and was killed
     at 161/164 on the progress bar. No data was lost (scoring re-executes
     stored solutions), and rescoring standalone with `--parallel 2`, no
     server, took **1:31 at a 2.0 GB peak**. Any future cell should score that
     way rather than inside the generation run.

Deferred (documented here): PLE table quant; MTP; vision; indexer cache
compaction. `docs/QWEN36_PORT.md` remains the GDN/rope/mrope authority and
the 3.6 hardware findings carry over.

### Machine: 2026-09-05 kernel panics and the 16 GB operating protocol

Identical history and protocol as `QWEN36_PORT.md` (16 GB Mac mini,
Mac16,10; watchdog panics from memory/IO thrash — the tell is a fresh
`panic-full-*.panic` + `ResetCounter` pair). Additional 2026-09-09 facts:
`hw.memsize` 16 GB; disk free 1.1 TiB; the M4 oracle run on the 79 GB GGUF
must be the only heavy process (mmap pages the model; 16 GB RAM → slow but
bounded — keep `-c` small, no GPU offload); the repack reads 360 GB
(335.3 GiB of shards) and writes 174 GB on the same external volume — staged
checkpoints with fsync
per file class, `--resume` (M4 Phase A) makes a kill cost only the in-flight
file, never two heavy jobs at once.

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
- `models/Qwen3.8-Flash-Next-125B.finch/` — the M4 output, now built and
  installable: 162 GiB (`manifest.json`, `model_weights.bin`,
  `packed_experts/`, `ple_shards/`, `tokenizer/`, `verified-install.json`).
  This is the install every measurement in item 7 and item 8 ran against.
  Untracked (weights), as are the two checkpoints above.

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

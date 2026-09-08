---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
language: [en]
tags: [qwen, moe, apple-silicon, metal, quantization, finchmoe]
library_name: finchmoe
---

# Qwen3.6-35B-A3B — FinchMoE 3-bit/4-bit quant (Apple Silicon)

Optimized weights for the [FinchMoE](https://github.com/haihengh/finchMoE) C/Metal
inference engine (SSD-streamed MoE, ~2 GB RAM, no llama.cpp needed).

**Why this quant**: 3-bit routed experts + 4-bit GDN/attention + 8-bit
embed/lm_head — 14.9 GB total (~4.8× smaller than the 71.9 GB BF16 source),
tuned for the engine's zero-copy expert streaming.

## Files (custom format — FinchMoE only)

| File | Size | Content |
|---|---|---|
| `model_weights_quant.bin` + `.json` | 1.95 GB | Non-expert weights: 4-bit affine (group 64) GDN/attention/shared experts, 8-bit embeddings + lm_head, BF16 norms/gates |
| `packed_experts_3bit/layer_00..39.bin` | 14 GB | Routed experts, 3-bit per layer (256 experts × 1.31 MB) |
| `vocab.bin`, `tokenizer.bin` | 8 MB each | Custom BPET tokenizer |
| `hot_sets.bin` | 10 KB | Optional prefill hot-set prefetch data |
| `shaders.metal` | 128 KB | Runtime-compiled Metal shaders |

The MTP files (`model_weights_mtp.bin`, `layer_40.bin`) are optional and not
included — the MTP head is not used.

## Usage

```bash
git clone https://github.com/haihengh/finchMoE
cd finchmoe && make          # requires macOS + Xcode CLT (Metal)

# download this repo's files next to the binary (or symlink them):
./finchmoe-infer -m . \
  --weights model_weights_quant.bin --manifest model_weights_quant.json \
  -P "Tell me a story about a lighthouse keeper." --tokens 200
```

## Quantization quality (measured, vs BF16)

| Component | Bits | CosSim vs BF16 |
|---|---|---|
| Routed experts (256×40) | 3-bit | 0.966-0.979 |
| Non-experts (GDN, attention, shared) | 4-bit | ≥ 0.995 |
| Embeddings + lm_head | 8-bit | near-lossless |

End-to-end quality passes the llama.cpp Q4_K_M baseline level on edge
prompts; no quantization-driven long-generation drift (fresh-prefill
differential cos 0.99942).

## Performance (M4, 16 GB, warm page cache)

- Decode: **16-22 tok/s** (K=8); cold-restart 10.3 tok/s (page-cache bound)
- Prefill (90 tokens): **1.9 s** chunked / 5.0-5.3 s per-token
- RAM: ~0.7 GB engine + ~3.7 GB expert page cache
- 8 GB machines (e.g. M1 mini): ~4.1 tok/s — SSD-I/O bound, see the engine
  README for the full analysis

## License

Weights derived from Qwen/Qwen3.6-35B-A3B (Apache-2.0).

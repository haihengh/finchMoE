# DeepSeek-V4-Flash Inference Engine — Design Document

## 1. Architecture Overview

DeepSeek-V4-Flash is a 155 GB MoE model with 43 layers, 256 experts per layer, and native FP4 quantization. It requires a new engine fundamentally different from FinchMoE's Qwen3.6-35B-A3B engine.

### 1.1 Model Specs

| Parameter | Value |
|-----------|-------|
| Hidden dim | 4096 |
| Layers | 43 + 1 MTP (DSPark) |
| Attention heads | 64 Q : 1 KV (extreme GQA) |
| Head dim | 512 |
| Q RoPE dim | 64 |
| Expert count | 256 routed, 1 shared |
| Active experts | 6 |
| Expert intermediate | 2048 |
| Expert format | MXFP4 (native, ue8m0 scales) |
| Vocab size | 129,280 |
| Context | 1M tokens |
| Attention window | Sliding window 128 |
| Routing | Hash-based (first 3 layers) + learned (remaining 40) |

### 1.2 Key Differences from Qwen3.6 Engine

| Component | Qwen3.6 | DeepSeek-V4 | Impact |
|-----------|---------|-------------|--------|
| Hidden dim | 2048 | **4096** | 2× larger matmuls |
| Attention | 32Q:2KV full window | **64Q:1KV sliding 128** | New Metal kernel needed |
| Q/O projection | Standard matmul | **LoRA (rank 1024, 8 groups)** | Two-stage matmul |
| KV cache | per full-attn layer | **Compressed KV (MLS)** | Multi-stage attention |
| Expert quant | 4-bit affine (q=bias+scale×nibble) | **MXFP4 (ue8m0 scale × e2m1 value)** | New dequant kernel |
| Expert layout | 1 file/layer (256 experts packed) | **Individual safetensors tensors** | Different I/O strategy |
| Routing | Softmax + Top-K | **Hash table (tid2eid) + sqrt(softplus)** | New routing kernel |
| FFN | gate + up + down | **w1 + w2 + w3 with SiLU clamp** | Different compute |
| Spec decode | MTP (1 layer) | **DSPark (block_size=5, Markov rank=256)** | Different architecture |
| RoPE | Standard | **YaRN (factor=16, compressed theta=160000)** | New RoPE kernel |

## 2. Component Design

### 2.1 Attention: Sliding Window MLA (Multi-Head Latent Attention)

DeepSeek uses a compressed KV cache with sliding window attention.

**Architecture**:
```
Input x [4096]
  → Q = wq_b [Q_DIM, 1024] @ wq_a [1024, 4096] @ x
     Q = [64 heads × 512 dim] but only first 64 dims per head get RoPE
  → K = wkv [512, 4096] @ x  
     K = [1 head × 512 dim], only first 64 dims get RoPE
  → V = wkv [512, 4096] @ x (same weight, different head)
  → Attention over sliding window (128 tokens)
  → O = wo_b [4096, 1024] @ wo_a [1024, O_DIM/8 groups] @ attn_out
```

**Borrowed from**: vllm `nvidia/model.py`, sglang `dsv4/attn.py`

**Implementation plan**:
- Phase 1: CPU-only with full attention (no sliding window, no KV compress)
- Phase 2: Metal sliding window kernel
- Phase 3: KV compression (MLS)

### 2.2 MoE: Hash Routing + MXFP4 Experts

**Hash Routing (first 3 layers)**:
```
Given token_id:
  expert_base = tid2eid[token_id * INDEX_TOPK]  // precomputed hash table
  for i in 1..ACTIVE_EXPERTS:
    eid = expert_base[i]
    score = sqrt(softplus(router_logits[token_id * 256 + eid]))
  weights = normalize(scores) * routed_scaling_factor
```

**Learned Routing (remaining 40 layers)**:
```
Standard softmax over 256 logits → top-6 → normalize weights
```

**Borrowed from**: sglang `hash_topk.cuh:16-68`, vllm `model.py:573-576`

**MXFP4 Expert Format** (per expert):
```
w1: [MOE_INTERMEDIATE*2, HIDDEN_DIM] FP4  →  SiLU gate+up fused
    scale: ue8m0, block_size=[128,128]
w2: [MOE_INTERMEDIATE, HIDDEN_DIM] FP4     →  down projection
    scale: ue8m0, block_size=[128,128]  
w3: [HIDDEN_DIM, MOE_INTERMEDIATE] FP4     →  output projection
    scale: ue8m0, block_size=[128,128]
```

**FP4 Dequant** (per block of 32 values, 17 bytes):
```
byte 0:        scale_ue8m0  →  scale_f32 = powf(2.0f, (int)scale_byte)
bytes 1-16:    32 nibbles (reordered: low nibbles [0:16], high nibbles [16:32])
value[i] = e2m1_table[nibble[i]] * scale_f32
```

**e2m1 value table**:
```
{0, 0.5, 1, 1.5}  // 2 exp bits, 1 mantissa bit
```

**Borrowed from**: llama.cpp `deepseek.py:713-735`, lmdeploy `weight_format.py:396-424`

**SwiGLU FFN**:
```
h = x
gate_up = w1 @ h                                    // [2*MOE_INTERMEDIATE] FP4 dequant
gate, up = split(gate_up)                            // [MOE_INTERMEDIATE] each
act = clamp(silu(gate) * up, -10.0, 10.0)           // SwiGLU with clamp
out = w2 @ act                                        // [HIDDEN_DIM] FP4 dequant
expert_out = w3 @ out                                 // [HIDDEN_DIM] FP4 dequant
```

**Borrowed from**: sglang `silu_and_mul_masked_post_quant.cuh`

### 2.3 DSPark Speculative Decoding

**Architecture**:
```
block_size: 5      // predict 5 tokens at once
markov_rank: 256   // Markov chain state size
noise_token: 128799 // special noise token ID
target_layers: 40, 41, 42  // last 3 of 43 layers for draft
```

**Implementation plan**:
- Phase 4: Port DSPark Markov chain logic
- Borrow from: llama.cpp `deepseek.py` (DSPark conversion), vllm `nvidia/dspark.py`

### 2.4 YaRN RoPE

**Parameters**:
```
rope_theta: 10000
compress_rope_theta: 160000
rope_scaling: {factor: 16, type: "yarn", beta_fast: 32, beta_slow: 1}
qk_rope_head_dim: 64  // only first 64 of 512 head dim get RoPE
```

**Implementation**:
- Standard RoPE with YaRN frequency scaling
- Only applied to first 64 dims of Q and K heads
- Borrow from: sglang `deepseek_v4_rope.py`

### 2.5 Weight Loading Strategy

Unlike Qwen3.6 where experts are packed in one file per layer, DeepSeek stores each expert as an individual safetensors tensor. Loading all 66,048 expert tensors at startup would use too much memory.

**Strategy**: Lazy loading via mmap + pread
```
Phase 1 (startup): Load all non-expert weights (1623 tensors) into memory
Phase 2 (per-token): Pread expert weights from safetensors shards as needed
Phase 3 (cache): Keep recently-used experts in an LRU cache
```

**Non-expert tensors to preload** (1623 total across 43 layers):
- embed.weight, head.weight, norm.weight
- Per layer: attn.* (LoRA + norms + wkv), ffn_norm, ffn.gate (routing)
- DSPark + MTP tensors

**Expert tensors** (66,048 across 43 layers × 256 experts × 6 tensors):
- Read on-demand from safetensors via pread
- Each expert: ~1.5 MB (FP4 packed)
- 6 active × 1.5 MB = 9 MB per token (acceptable)

## 3. Implementation Phases

### Phase 1: CPU Reference (1-2 days)
**Goal**: Single-token forward pass producing correct logits

- [ ] FP4 e2m1 dequant function (+ ue8m0 scale decode)
- [ ] Hash routing: tid2eid lookup + sqrt(softplus) scoring
- [ ] LoRA Q/O: two-stage matmul
- [ ] Sliding window attention (CPU, no compression)
- [ ] SwiGLU FFN: w1/w2/w3 with clamp
- [ ] YaRN RoPE
- [ ] Full 43-layer forward pass
- [ ] Token generation loop (greedy, no speculative)

**Validation**: Compare first-token logits against Python reference (HF/MLX)

### Phase 2: Expert I/O (1 day)
**Goal**: Efficient expert loading from safetensors

- [ ] Build safetensors shard index (which shard per expert per layer)
- [ ] LRU expert cache (Metal buffers if GPU, malloc if CPU)
- [ ] Parallel pread for 6 active experts
- [ ] Benchmark: expert I/O time vs compute time

### Phase 3: Metal GPU (2-3 days)
**Goal**: GPU-accelerated forward pass

- [ ] Metal FP4 dequant matvec kernel (MXFP4 format)
- [ ] Metal SwiGLU kernel (with clamp)
- [ ] Metal sliding window attention kernel
- [ ] Metal LoRA matmul dispatch (4 Metal dispatches per attention layer)
- [ ] Metal hash routing kernel
- [ ] GPU command buffer pipeline (CMD1/CMD2/CMD3 like Qwen3.6)

### Phase 4: DSPark Speculative Decoding (2 days)
**Goal**: Multi-token prediction for speed

- [ ] DSPark weight loading
- [ ] Markov chain token generation
- [ ] Batch verification with main model
- [ ] Acceptance heuristic
- [ ] Benchmark: tok/s with D=1,2,3,4,5

### Phase 5: Optimization (ongoing)
**Goal**: Production speed

- [ ] KV cache compression (MLS)
- [ ] Fused expert kernel (multiple experts in one dispatch)
- [ ] ICBs (Indirect Command Buffers)
- [ ] Async pipelining (overlap I/O with compute)
- [ ] FP16 KV cache
- [ ] Continuous batching

## 4. Source Attribution

| Algorithm | Primary Source | File |
|-----------|---------------|------|
| Hash routing | sglang | `csrc/deepseek_v4/hash_topk.cuh` |
| MXFP4 packing | llama.cpp | `conversion/deepseek.py:713-735` |
| MXFP4 format spec | lmdeploy | `turbomind/weight_format.py:396-424` |
| ue8m0 scale decode | vllm | `deepseek_v4/quant_config.py:311` |
| Model architecture | vllm | `deepseek_v4/nvidia/model.py` |
| SwiGLU clamp | sglang | `csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh` |
| DSPark spec | vllm | `deepseek_v4/nvidia/dspark.py` |
| LoRA attention | vllm | `deepseek_v4/attention.py` |
| YaRN RoPE | sglang | `deepseek_v4_rope.py` |

## 5. File Structure

```
finchmoe/
├── infer.m                    # Qwen3.6 engine (production)
├── infer_deepseek.m           # DeepSeek-V4 engine (new)
├── shaders.metal              # Qwen3.6 shaders
├── shaders_deepseek.metal     # DeepSeek-V4 shaders (new)
├── design.md                  # Qwen3.6 design doc
├── design_deepseek.md         # This file
└── finchTool/                 # Diagnostic suite (model-agnostic)
```

## 6. Build

```bash
# CPU-only reference
clang -O2 -Wall -fobjc-arc -framework Foundation \
      infer_deepseek.m -o finchmoe-deepseek

# Metal GPU
clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation \
      -framework Accelerate infer_deepseek.m -o finchmoe-deepseek

# With debug
clang -O0 -g -Wall -fobjc-arc -framework Metal -framework Foundation \
      infer_deepseek.m -o finchmoe-deepseek
```

# Bug Report: Qwen3.6-35B-A3B Inference Engine Produces Degenerate Output

## Summary

Custom C/Metal inference engine for Qwen3.6-35B-A3B (MoE, 4-bit quantized) produces **correct first token** then immediately degenerates into repeating token cycles during autoregressive generation. Both GPU and CPU execution paths fail identically. Both self-quantized and community 4-bit models fail.

## Environment

- **Hardware**: Apple M4 Mac mini 16GB
- **Model**: Qwen3.6-35B-A3B-Instruct (model_type: `qwen3_5_moe`)
- **Quantization**: MLX affine INT4 group-64, self-quantized from BF16 source
- **Engine**: C + Metal, ~7500 lines (`infer.m`), adapted from flash-moe

## Behavior

### Prompt: "The capital of France is" (temperature=0, greedy)

```
First token:  " the"     ✅ (correct continuation)
Second token: ","         ❌ (should be "Paris" or "capital")
Then enters 3-token cycle: [",", " ", "2"] repeating forever
```

### Prompt: "a" (temperature=0, greedy)

```
"ight a a ight a ight ta ight ta" — cycles between "ight", "a", "ta"
```

### Key observations:

1. **First token is always correct** — the prefill computation works.
2. **Degeneration starts at token 2** — the very first autoregressive step fails.
3. **Perfect cycles at temp=0** — the hidden state enters a deterministic limit cycle.
4. **Cycle length varies**: 3-token with K=4 experts, 4-token with K=0 (shared expert only).
5. **Both `--cpu-linear --cpu-experts` and default GPU path fail** — bug is in shared logic, not GPU-specific.
6. **Both self-quantized (`4bit-custom`) and community (`4bit`) models fail** — not a model corruption issue.

## Architecture (40 layers, 3:1 linear:full attention)

```
Layer forward (HuggingFace reference):
  residual = hidden
  hidden = input_layernorm(hidden)
  hidden = attention(hidden)          # GatedDeltaNet or full-attention (every 4th layer)
  hidden = residual + hidden
  residual = hidden
  hidden = post_attention_layernorm(hidden)
  hidden = MoE_block(hidden)          # router + shared_expert + routed experts
  hidden = residual + hidden

MoE_block:
  shared_out = shared_expert(hidden)  # dense SwiGLU FFN
  shared_out *= sigmoid(shared_expert_gate(hidden))  # scalar gate
  expert_out = weighted_sum(experts[topk(router(hidden))])
  return expert_out + shared_out
```

- **30× GatedDeltaNet layers**: conv1d(k=4) + delta-rule recurrence + gated RMSNorm
- **10× Full attention layers**: GQA (16Q:2KV), head_dim=256, RoPE, Q-output-gate
- **MoE**: 256 experts, top-8 (we use K=4), 1 shared expert
- **Dimensions**: hidden=2048, moe_inter=512, vocab=248320

## What Has Been Verified (Against HuggingFace Reference)

All of the following match the `Qwen3_5MoeForConditionalGeneration` reference implementation exactly:

### 1. RMSNorm (`cpu_rms_norm`, `infer.m:818`)
```c
// Qwen3_5RMSNorm: weight stored as (1 + original_param), engine applies:
out[i] = x[i] * inv_rms(x) * bf16_to_f32(weight[i]);
// Verified: BF16 weights ~0.1, quantized weights = +1.0 = ~1.1 ✅
```

### 2. 4-bit Dequant Matvec (`cpu_dequant_matvec`, `infer.m:780`)
```c
// MLX affine INT4: w_deq = w_q * scale + bias
// Group size 64, 8 int4 per uint32, LSB-first packing
for each row:
  for each group (in_dim/64 groups):
    scale = bf16_to_f32(scales[row][g])
    bias  = bf16_to_f32(biases[row][g])
    for each packed uint32:
      for n in 0..7:
        val = (packed >> (n*4)) & 0xF
        acc += ((float)val * scale + bias) * x[group*64 + p*8 + n]
// Verified: matches MLX packing convention ✅
```

### 3. Full Attention Output Gate (`infer.m:2365-2372, 2466-2472`)
```c
// Qwen3.6 full attention: Q projection outputs [num_heads * head_dim * 2]
// First half = queries, second half = sigmoid gate applied after attention
float *q   = q_proj_out + h * (2 * HEAD_DIM);       // first 256 values
float *gate = q_proj_out + h * (2 * HEAD_DIM) + HEAD_DIM; // next 256 values
// ... after scaled dot-product attention:
attn_out[i] *= 1.0f / (1.0f + expf(-q_gate[i]));   // sigmoid gate
// Verified: matches HuggingFace Qwen3_5MoeAttention.forward() ✅
```

### 4. Gated Delta Rule Recurrence (`infer.m:4868-4907`)
```c
// Matches torch_recurrent_gated_delta_rule exactly:
// Step 1: S *= g_decay          (cblas_sscal)
// Step 2: kv_mem = S @ k        (cblas_sgemv)
// Step 3: delta = (v - kv_mem) * beta
// Step 4: S += delta @ k^T      (cblas_sger, rank-1 update)
// Step 5: output = S @ q        (cblas_sgemv)
// Reference: transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:358-362
float A_val = expf(A_log[vh]);
float softplus_val = logf(1.0f + expf(alpha_out[vh] + dt_bias));
g_decay[vh] = expf(-A_val * softplus_val);  // exp(g) in reference
beta_gate[vh] = sigmoid(beta_out[vh]);
// Verified ✅
```

### 5. Conv1d (`cpu_conv1d_step`, `infer.m:1025`)
```c
// Depthwise conv1d, kernel=4, SiLU activation
// State: [kernel-1] * channels time-major, shifted before new input
for each channel c:
  acc = sum(state[k*channels + c] * weight[c*kernel + k] for k in 0..2)
      + new_input[c] * weight[c*kernel + 3]
  out[c] = silu(acc)
// Verified: matches torch_causal_conv1d_update ✅
```

### 6. RoPE (`apply_rotary_emb`, `infer.m:2187`)
```c
// Neox-style: pairs at (i, i + half_dim), half = rotary_dim/2 = 32
// For text-only, MRoPE interleaved [11,11,10] is equivalent to standard RoPE
// since T=H=W=pos for 1D position ids
freq = 1.0 / powf(ROPE_THETA, (2*i) / rotary_dim);  // theta=10M
angle = pos * freq;
q[i]        = q0 * cos(angle) - q1 * sin(angle);
q[i+half]   = q0 * sin(angle) + q1 * cos(angle);
// Verified ✅
```

### 7. All Dimensions Match Model Config
```
hidden=2048, num_heads=16, kv_heads=2, head_dim=256
num_experts=256, moe_inter=512, shared_inter=512
linear_num_v_heads=32, linear_num_k_heads=16
linear_key_dim=128, linear_value_dim=128
full_attention_interval=4, conv_kernel=4
vocab=248320, rope_theta=10M, partial_rotary=0.25
// All verified against config.json and HuggingFace model definition ✅
```

### 8. Quantization Pipeline
```python
# quantize_model.py:
# - RMSNorm weights: arr = arr + 1.0 before storing (Bug 1 fix)
# - Expert gate_up_proj fused → split into gate_proj + up_proj
# - 8-bit: mlp.gate.weight, mlp.shared_expert_gate.weight
# - 4-bit: everything else with in_dim % 64 == 0
# - 1D tensors (norms, A_log, dt_bias): stored as BF16
# extract_weights.py:
# - Correctly excludes switch_mlp (routed experts → packed_experts/)
# - Correctly skips vision_tower and MTP layers
# Verified ✅
```

## Where The Bug Likely Is

Since the prefill produces the correct first token but autoregressive generation fails immediately, the problem is in the **state transition** between prefill and generation, or in how the **first generated token** is processed.

### Pipeline Structure (`fused_layer_forward`, 3 Metal command buffers per layer):

```
CMD1: attention input projections (Q/K/V or QKV/Z/Beta/Alpha)
      [+ GPU delta-net if enabled]
CPU:  attention compute (RoPE + softmax/delta-net)
CMD2: o_proj + residual_add + rms_norm + routing gate + shared_expert gate/up
CPU:  softmax + top-K + pread experts
CMD3: expert forwards + shared SwiGLU + shared down
      [+ GPU-side combine (moe_combine_residual + rms_norm → buf_input)]
      [DEFERRED: async commit, no wait]
```

### Prefill vs Generation:

```c
// PREFILL (intermediate tokens):
for each prefill token (except last):
    for each layer: fused_layer_forward(...)
    discard_deferred_experts();  // ONLY waits for GPU, doesn't readback hidden
    pos++;

// PREFILL (last token):
for each layer: fused_layer_forward(...)
complete_deferred_experts();  // waits + reads back hidden state
final_norm(hidden); lm_head(hidden); sample → first_token ✅

// GENERATION (all subsequent tokens):
embed_lookup(first_token, hidden);
for each layer: fused_layer_forward(...)
complete_deferred_experts();  // reads back hidden
final_norm(hidden); lm_head(hidden); sample → second_token ❌
```

### Suspicious Area #1: Last Layer GPU/CPU Combine

For layers 0-38, `gpu_combine=true` — CMD3 includes GPU-side combine that writes to `buf_moe_hidden`. For layer 39 (last), `gpu_combine=false` — CPU combine path reads from `buf_multi_expert_out[k]` and `buf_shared_out`:

```c
// infer.m:5562-5567 — GPU combine disabled for last layer:
int gpu_combine = (g_metal->moe_combine_residual && ...
                   layer_idx < NUM_LAYERS - 1 &&  // ← FALSE for layer 39
                   layer_cache[layer_idx + 1].input_norm_w != NULL);

// infer.m:4014-4044 — CPU combine path (for last layer):
if (g_deferred.gpu_combined) {
    // GPU path: read from buf_moe_hidden
    memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents], HIDDEN_DIM * sizeof(float));
} else {
    // CPU path: read expert outputs, accumulate, combine
    float moe_out[HIDDEN_DIM];
    for (int k = 0; k < g_deferred.actual_K; k++) {
        float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
        cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], HIDDEN_DIM);
    }
    float shared_out[HIDDEN_DIM];
    memcpy(shared_out, [g_metal->buf_shared_out contents], HIDDEN_DIM * sizeof(float));
    shared_weight = sigmoid(g_deferred.shared_gate_score);
    for (int i = 0; i < HIDDEN_DIM; i++) shared_out[i] *= shared_weight;
    // Final: hidden = h_mid + moe_out + shared_out
    for (int i = 0; i < HIDDEN_DIM; i++)
        g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i] + shared_out[i];
}
```

**Question**: Is `buf_multi_expert_out[k]` correctly populated for the last layer? For layers 0-38, the GPU combine kernel reads these and writes to `buf_moe_hidden`. For the last layer, the CPU reads them directly. If the buffers are stale from a previous layer...

### Suspicious Area #2: Recurrent State Persistence

Between the last prefill token and the first generation token, the recurrent states must persist:
- `buf_delta_state[30]` (GPU) or `ssm_state` (CPU) — delta-net S matrices [32×128×128]
- `conv_state[30]` — conv1d history buffer [8192×3]
- `kv_caches[10]` — full attention KV cache

The prefill uses `discard_deferred_experts()` for intermediate tokens which only waits for GPU. The first generation token uses `complete_deferred_experts()` which reads back hidden. Both modify `g_deferred`.

**Question**: Could `discard_deferred_experts()` leave `g_deferred` in a state that corrupts the first generation step? It sets `g_deferred.active=0` but doesn't clear `g_deferred.hidden`, `g_deferred.h_mid`, or `g_deferred.expert_weights`.

### Suspicious Area #3: Hidden State After Final Norm

```c
// infer.m:7541-7542 — final norm MODIFIES hidden in place:
cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
// hidden is now the NORMED version
lm_head_forward(wf, hidden, logits);  // reads normed hidden
// ...
// Next iteration:
embed_lookup(wf, next_token, hidden);  // overwrites with new embedding ✅
```

But the second iteration's `fused_layer_forward` reads `hidden` which was just set to the embedding. This should be correct since `embed_lookup` overwrites it completely.

## How to Reproduce

```bash
cd finchmoe
make
./finchmoe-infer --prompt "The capital of France is" \
    --tokens 5 --temperature 0 --no-think
```

Expected: " Paris, the capital..."
Actual: " the,  2 ,"

## Debugging Output

With `--debug-layers`:
```
[DEBUG-L0] input: mean=-0.000199 rms=0.009896   ← embedding of first prompt token
[DEBUG-L0] h_mid: mean=0.002268 rms=0.036489    ← after attention + residual
[DEBUG-L0] h_post: mean=0.045347 rms=0.609793   ← after post-attention norm
...
[DEBUG-L0] input: mean=0.000149 rms=0.012598   ← second token (generated)
[DEBUG-L0] h_mid: mean=0.001988 rms=0.046010    ← different from prefill (expected)
```

Note: With GPU fused delta-net path (default), `qkv-proj`, `z-proj`, `beta-proj`, `alpha-proj` debug prints show zeros because GPU writes to Metal buffers, not the CPU scratch buffers being printed.

## Files

- `finchmoe/infer.m` — Main engine (~7500 lines)
- `finchmoe/shaders.metal` — Metal GPU kernels (~1300 lines)
- `finchmoe/quantize_model.py` — BF16→4bit quantization
- `finchmoe/extract_weights.py` — Weight extraction to model_weights.bin
- `finchmoe/repack_experts.py` — Expert packing to packed_experts/
- `finchmoe/Makefile` — Build system
- Reference: `/Library/.../transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`

## CRITICAL FINDING (2026-08-08): Both flash-moe AND FinchMoE Produce Garbage on M4

The **original flash-moe** code (which FinchMoE was adapted from) also produces garbage on this machine, even with its intended 397B model:

```
# flash-moe with its own 397B model:
$ ./infer --model ../../models/Qwen3.5-397B-A17B-4bit --prompt "The capital of France is" --tokens 5
[debug] hidden rms after final_norm=nan, logits rms=nan
Output: !!! ! ! !   (all token_id=0)

# FinchMoE with 35B model:
$ ./finchmoe-infer --prompt "The capital of France is" --tokens 5 --temperature 0
[debug] hidden rms after final_norm=1.8899, logits rms=2.1034  (no NaN)
Output: " the,  2 ,"  (wrong but finite)
```

**Key observations:**
- flash-moe (397B, 60 layers, hidden=4096): hidden state becomes **NaN** → logits NaN → stuck on token 0
- FinchMoE (35B, 40 layers, hidden=2048): finite but wrong → degenerate cycle
- **Metal GPU expert kernels verified bit-identical to CPU** via `--compare-experts` ✅
- **Metal shader source is identical** between flash-moe and FinchMoE ✅
- **CPU-only path also fails** (`--cpu-linear --cpu-experts`) ✅

**Implication**: The bug is NOT in FinchMoE's adaptations. It exists in the original flash-moe code and manifests differently at different model scales (NaN for 397B, finite-but-wrong for 35B). The issue may be specific to M4 hardware or macOS 26.

## Hardware/Software Environment

- **This machine**: M4 Mac mini 16GB, macOS 26.2 (Darwin 25.5.0), Metal compiler 21.x
- **Original flash-moe dev machine**: M3 Max 48GB, macOS 26.2 (Darwin 25.2.0)
- Both use JIT Metal shader compilation (`newLibraryWithSource`, MTLLanguageVersion3_1)

## Help Needed

1. Has anyone run flash-moe or similar Metal inference engines on **M4** hardware? Are there known M4-specific Metal issues?
2. Are there known issues with `newBufferWithBytesNoCopy` for large (5.5GB) Metal buffers on M4?
3. What could cause NaN propagation that gets worse with more layers (60 layers → NaN, 40 layers → finite but wrong)?
4. Any suggestions for adding NaN-detection instrumentation to isolate the first layer that produces NaN?

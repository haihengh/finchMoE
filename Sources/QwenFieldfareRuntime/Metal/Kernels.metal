#include <metal_stdlib>
using namespace metal;

// =============================================================================
// QwenFieldfare Metal kernels for Qwen3-30B-A3B (MLX affine 4-bit).
//
// Conventions:
//  - fp16 (half) for activations, norms, scales, biases, embeddings.
//  - int4 packed as uint32 (8 nibbles per u32), group_size = 64, affine
//    dequant: value = (nibble - 8) * scale + bias.
//  - Vectors are single-token activation vectors during decode.
// =============================================================================

// -----------------------------------------------------------------------------
// 1. gemv_int4_q64 — GEMV: affine 4-bit quantized matrix [rows, cols] × fp16
//    vector [cols] -> fp16 vector [rows]. One thread computes one output row.
// -----------------------------------------------------------------------------
kernel void gemv_int4_q64(
    device const uint*  weights [[buffer(0)]],   // [rows * (cols/8)] packed nibbles
    device const half*  scales  [[buffer(1)]],   // [rows * (cols/64)]
    device const half*  biases  [[buffer(2)]],   // [rows * (cols/64)]
    device const half*  x       [[buffer(3)]],   // [cols]
    device half*        out     [[buffer(4)]],   // [rows]
    constant uint&      rows    [[buffer(5)]],
    constant uint&      cols    [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;

    const uint groupsPerRow = cols / 64u;   // affine groups of 64
    const uint u32PerRow    = cols / 8u;     // 8 nibbles per u32
    const uint wBase = gid * u32PerRow;
    const uint sBase = gid * groupsPerRow;

    float acc = 0.0f;
    for (uint g = 0; g < groupsPerRow; ++g) {
        const float scale = (float)scales[sBase + g];
        const float bias  = (float)biases[sBase + g];
        const uint  colStart = g * 64u;
        // 64 values == 8 u32 words.
        for (uint j = 0; j < 8u; ++j) {
            uint packed = weights[wBase + g * 8u + j];
            for (uint n = 0; n < 8u; ++n) {
                int nib = (int)((packed >> (n * 4u)) & 0xFu);
                float w = ((float)(nib - 8)) * scale + bias;
                float xv = (float)x[colStart + j * 8u + n];
                acc += w * xv;
            }
        }
    }
    out[gid] = (half)acc;
}

// -----------------------------------------------------------------------------
// 2. rms_norm — RMSNorm over a single vector. Dispatch with ONE threadgroup.
// -----------------------------------------------------------------------------
kernel void rms_norm(
    device const half* x       [[buffer(0)]],
    device const half* weight  [[buffer(1)]],
    device half*       out     [[buffer(2)]],
    constant uint&     n       [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    threadgroup float partial[256];

    float local = 0.0f;
    for (uint i = tid; i < n; i += tcount) {
        float v = (float)x[i];
        local += v * v;
    }
    partial[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = partial[0] / (float)n;
    float inv  = rsqrt(mean + eps);

    for (uint i = tid; i < n; i += tcount) {
        out[i] = (half)(((float)x[i]) * inv * (float)weight[i]);
    }
}

// -----------------------------------------------------------------------------
// 3. rope_neox — NeoX RoPE applied in place to [numHeads, headDim] for a single
//    token at absolute `pos`. One thread per (head, pair).
// -----------------------------------------------------------------------------
kernel void rope_neox(
    device half*    x        [[buffer(0)]],  // [numHeads * headDim]
    constant uint&  numHeads [[buffer(1)]],
    constant uint&  headDim  [[buffer(2)]],
    constant float& theta    [[buffer(3)]],
    constant uint&  pos      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint half_ = headDim / 2u;
    const uint total = numHeads * half_;
    if (gid >= total) return;

    const uint head = gid / half_;
    const uint i    = gid % half_;   // pair index in [0, headDim/2)

    float freq  = 1.0f / pow(theta, (2.0f * (float)i) / (float)headDim);
    float angle = (float)pos * freq;
    float c = cos(angle);
    float s = sin(angle);

    const uint base = head * headDim;
    float x0 = (float)x[base + i];
    float x1 = (float)x[base + i + half_];
    x[base + i]          = (half)(x0 * c - x1 * s);
    x[base + i + half_]  = (half)(x1 * c + x0 * s);
}

// -----------------------------------------------------------------------------
// 4. gqa_attention_causal — full causal GQA attention for a single query token.
//    Q: [numQHeads, headDim]; K/V cache: [length, numKVHeads, headDim].
//    One thread per Q head, online (streaming) softmax. headDim <= 128.
// -----------------------------------------------------------------------------
kernel void gqa_attention_causal(
    device const half* q          [[buffer(0)]],  // [numQHeads * headDim]
    device const half* kcache     [[buffer(1)]],  // [length * numKVHeads * headDim]
    device const half* vcache     [[buffer(2)]],  // [length * numKVHeads * headDim]
    device half*       out        [[buffer(3)]],  // [numQHeads * headDim]
    constant uint&     numQHeads  [[buffer(4)]],
    constant uint&     numKVHeads [[buffer(5)]],
    constant uint&     headDim    [[buffer(6)]],
    constant uint&     length     [[buffer(7)]],  // valid tokens incl. current
    uint qh [[thread_position_in_grid]])
{
    if (qh >= numQHeads) return;

    const uint groupSize = numQHeads / numKVHeads;   // Q heads per KV head
    const uint kvh       = qh / groupSize;
    const float scale    = 1.0f / sqrt((float)headDim);
    const uint qBase     = qh * headDim;
    const uint kvStride  = numKVHeads * headDim;

    float m = -INFINITY;
    float l = 0.0f;
    float acc[128];
    for (uint d = 0; d < headDim; ++d) acc[d] = 0.0f;

    for (uint t = 0; t < length; ++t) {
        const uint kBase = t * kvStride + kvh * headDim;
        float score = 0.0f;
        for (uint d = 0; d < headDim; ++d) {
            score += (float)q[qBase + d] * (float)kcache[kBase + d];
        }
        score *= scale;

        float mNew = max(m, score);
        float corr = exp(m - mNew);
        float p    = exp(score - mNew);
        l = l * corr + p;
        for (uint d = 0; d < headDim; ++d) {
            acc[d] = acc[d] * corr + p * (float)vcache[kBase + d];
        }
        m = mNew;
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (uint d = 0; d < headDim; ++d) {
        out[qBase + d] = (half)(acc[d] * inv);
    }
}

// -----------------------------------------------------------------------------
// 5. silu_mul — SwiGLU elementwise: out = silu(gate) * up.
// -----------------------------------------------------------------------------
kernel void silu_mul(
    device const half* gate [[buffer(0)]],
    device const half* up   [[buffer(1)]],
    device half*       out  [[buffer(2)]],
    constant uint&     n    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float g = (float)gate[gid];
    float s = g / (1.0f + exp(-g));   // silu(g) = g * sigmoid(g)
    out[gid] = (half)(s * (float)up[gid]);
}

// -----------------------------------------------------------------------------
// 6. moe_combine — accumulate weighted expert output: out += weight * expert.
// -----------------------------------------------------------------------------
kernel void moe_combine(
    device half*       out    [[buffer(0)]],
    device const half* expert [[buffer(1)]],
    constant float&    weight [[buffer(2)]],
    constant uint&     n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[gid] = (half)((float)out[gid] + weight * (float)expert[gid]);
}

// -----------------------------------------------------------------------------
// 7. sample_argmax — greedy: index of max logit. Dispatch ONE threadgroup.
// -----------------------------------------------------------------------------
kernel void sample_argmax(
    device const half* logits [[buffer(0)]],
    device uint*       result [[buffer(1)]],
    constant uint&     n      [[buffer(2)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    threadgroup float vals[256];
    threadgroup uint  idxs[256];

    float best = -INFINITY;
    uint  bi = 0;
    for (uint i = tid; i < n; i += tcount) {
        float v = (float)logits[i];
        if (v > best) { best = v; bi = i; }
    }
    vals[tid] = best;
    idxs[tid] = bi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            if (vals[tid + s] > vals[tid]) {
                vals[tid] = vals[tid + s];
                idxs[tid] = idxs[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[0] = idxs[0];
}

// -----------------------------------------------------------------------------
// 8. sample_top_p — temperature + nucleus (top-p) sampling. Dispatch ONE
//    threadgroup. Parallel max/sum reductions, then thread 0 does a binary
//    search for the nucleus probability threshold and samples proportionally.
// -----------------------------------------------------------------------------
kernel void sample_top_p(
    device const half* logits        [[buffer(0)]],
    device uint*       result        [[buffer(1)]],
    constant uint&     n             [[buffer(2)]],
    constant float&    temperature   [[buffer(3)]],
    constant float&    topP          [[buffer(4)]],
    constant float&    randomUniform [[buffer(5)]],
    uint tid    [[thread_position_in_threadgroup]],
    uint tcount [[threads_per_threadgroup]])
{
    threadgroup float red[256];
    threadgroup float gmaxTG;
    threadgroup float sumExpTG;

    const float invT = (temperature > 0.0f) ? (1.0f / temperature) : 1.0f;

    // --- parallel max of scaled logits ---
    float localMax = -INFINITY;
    for (uint i = tid; i < n; i += tcount) {
        localMax = max(localMax, (float)logits[i] * invT);
    }
    red[tid] = localMax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) gmaxTG = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gmax = gmaxTG;

    // --- parallel sum of exp ---
    float localSum = 0.0f;
    for (uint i = tid; i < n; i += tcount) {
        localSum += exp(((float)logits[i] * invT) - gmax);
    }
    red[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tcount / 2u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) sumExpTG = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumExp = sumExpTG;

    if (tid != 0) return;

    // Max probability corresponds to gmax (exp(0)/sumExp).
    float pmax = 1.0f / sumExp;

    // Binary search for the largest threshold τ with mass(>=τ) >= topP.
    float lo = 0.0f;
    float hi = pmax;
    for (uint iter = 0; iter < 40u; ++iter) {
        float mid = 0.5f * (lo + hi);
        float mass = 0.0f;
        for (uint i = 0; i < n; ++i) {
            float p = exp(((float)logits[i] * invT) - gmax) / sumExp;
            if (p >= mid) mass += p;
        }
        if (mass >= topP) lo = mid; else hi = mid;
    }
    float tau = lo;

    // Total nucleus mass at threshold τ.
    float total = 0.0f;
    for (uint i = 0; i < n; ++i) {
        float p = exp(((float)logits[i] * invT) - gmax) / sumExp;
        if (p >= tau) total += p;
    }
    if (total <= 0.0f) {
        // Fallback to argmax.
        float best = -INFINITY; uint bi = 0;
        for (uint i = 0; i < n; ++i) {
            float v = (float)logits[i];
            if (v > best) { best = v; bi = i; }
        }
        result[0] = bi;
        return;
    }

    // Sample proportionally within the nucleus.
    float r = randomUniform * total;
    float cum = 0.0f;
    uint chosen = 0;
    for (uint i = 0; i < n; ++i) {
        float p = exp(((float)logits[i] * invT) - gmax) / sumExp;
        if (p >= tau) {
            cum += p;
            chosen = i;
            if (cum >= r) break;
        }
    }
    result[0] = chosen;
}

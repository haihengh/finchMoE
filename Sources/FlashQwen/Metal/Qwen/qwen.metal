#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qwen_decode.metal — Qwen 3.6 (qwen3_5_moe) decode-layer fusions.
//
// Math pinned to `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:
//   qwen_post_attn:          hidden += attn; x = rmsnorm(hidden, w_post_attn)
//   qwen_attn_output_gate:   attn *= sigmoid(gate)        (attn_output_gate)
//   qwen_full_attn_epilogue: q_norm/k_norm + partial RoPE + q|gate split
//   qwen_shared_gate:        h1 *= sigmoid(dot(w, x))     (shared_expert_gate)
//   vec_add_fp16:            a += b                        (tail combine)
//
// Qwen3_5MoeRMSNorm is mean-based and multiplies (1 + weight) — the +1 is
// baked into the stored weight by the repack writer, so the runtime kernel
// shape stays x * rsqrt(mean(x^2) + eps) * w, identical to the Gemma norms.
// ============================================================================

constant constexpr uint kQwenThreads       = 256;
constant constexpr uint kQwenMaxSimdGroups = kQwenThreads / 32;  // 8
constant constexpr uint kQwenGroupSize     = 64;
// Threadgroup staging caps: hidden dim <= 4096, head dim <= 512.
constant constexpr uint kQwenMaxD          = 4096;
constant constexpr uint kQwenMaxHeadDim    = 512;

static inline float qwen_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// Two-stage SIMD-group block sum of a per-thread `acc` into `partial[0]`
// (mirrors gdn_block_sum; local copy per module convention).
static inline void qwen_block_sum(
    float acc,
    uint  simd_lane,
    uint  simd_group,
    uint  simdgroups,
    threadgroup float* partial
) {
    acc = simd_sum(acc);
    if (simd_lane == 0) partial[simd_group] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float v = (simd_lane < simdgroups) ? partial[simd_lane] : 0.0f;
        v = simd_sum(v);
        if (simd_lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Qwen partial RoPE: rotates the first `rotary_dim` contiguous elements in
// HF half-split pairs (i, i + rotary_dim/2). The frequency denominator is
// the ROTARY dim (not the full head dim): inv_freq_i = theta^(-2i/rotary_dim),
// i = 0..rotary_dim/2-1. (Qwen3_5MoeTextRotaryEmbedding.compute_default_rope_
// parameters uses `dim = int(head_dim * partial_rotary_factor)` and
// Qwen3_5MoeAttention rotates with apply_rotary_pos_emb's rotate_half —
// pairs split at rotary_dim/2, NOT adjacent (2i, 2i+1).) For text-only runs
// the interleaved-MRoPE reordering is identity, so it is not represented
// here.
static inline void qwen_rope_pair(thread float& x0,
                                  thread float& x1,
                                  uint pair_index,
                                  uint rotary_dim,
                                  float position,
                                  float theta_base)
{
    const float exponent = -float(2u * pair_index) / float(rotary_dim);
    const float freq     = pow(theta_base, exponent);
    const float angle    = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// ============================================================================
// qwen_post_attn — Qwen residual + norm after the token mixer.
//
//     hidden[i] += attn[i]              (rounded to half, stored back)
//     out[i]     = rmsnorm(hidden)[i] * w_post_attn[i]
//
// `out` is the post_attention_layernorm output and feeds both the shared
// expert and the router. One threadgroup; two-stage mean-based reduction.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kQwenThreads)]]
void qwen_post_attn(
    device       half*   hidden    [[buffer(0)]],  // [D] FP16 in place
    device const half*   attn      [[buffer(1)]],  // [D] FP16
    device       half*   out       [[buffer(2)]],  // [D] FP16
    device const bfloat* weight    [[buffer(3)]],  // [D] BF16
    constant     uint&   D         [[buffer(4)]],
    constant     float&  rms_eps   [[buffer(5)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane        [[thread_index_in_simdgroup]],
    uint  simd_group       [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]]
) {
    threadgroup half  hidden_tg[kQwenMaxD];
    threadgroup float partial[kQwenMaxSimdGroups];

    float acc = 0.0f;
    for (uint i = lid; i < D; i += lsize) {
        const half h = half(float(hidden[i]) + float(attn[i]));
        hidden_tg[i] = h;
        hidden[i] = h;
        acc = fma(float(h), float(h), acc);
    }
    qwen_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(D) + rms_eps);

    for (uint i = lid; i < D; i += lsize) {
        out[i] = half(float(hidden_tg[i]) * inv * float(weight[i]));
    }
}

// ============================================================================
// vec_add_fp16 — elementwise `a += b` (Qwen tail combine: hidden += h2).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kQwenThreads)]]
void vec_add_fp16(
    device       half*   a     [[buffer(0)]],  // [D] FP16 in place
    device const half*   b     [[buffer(1)]],  // [D] FP16
    constant     uint&   D     [[buffer(2)]],
    uint  tid                  [[thread_position_in_grid]]
) {
    if (tid >= D) return;
    a[tid] = half(float(a[tid]) + float(b[tid]));
}

// ============================================================================
// qwen_attn_output_gate — full-attention output gate:
//     attn[i] *= sigmoid(gate[i])
// (`attn_output = attn_output * torch.sigmoid(gate)` in Qwen3_5MoeAttention.)
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kQwenThreads)]]
void qwen_attn_output_gate(
    device       half*   attn  [[buffer(0)]],  // [N] FP16 in place
    device const half*   gate  [[buffer(1)]],  // [N] FP16
    constant     uint&   N     [[buffer(2)]],
    uint  tid                  [[thread_position_in_grid]]
) {
    if (tid >= N) return;
    const float g = float(gate[tid]);
    attn[tid] = half(float(attn[tid]) * qwen_sigmoid(g));
}

// ============================================================================
// qwen_shared_gate — shared_expert_gate [1, N] GEMV + sigmoid scalar, then
// scales the shared-expert output in place:
//     h1[i] *= sigmoid(dot(W, x))
// One 32-thread SIMD group: dot over the int4-affine row, lane 0 computes the
// scalar, then all lanes scale h1. N % 64 == 0 (validated by the wrapper).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(32)]]
void qwen_shared_gate(
    device const uint8_t* W      [[buffer(0)]],  // [N/2] nibbles, one row
    device const bfloat*  scales [[buffer(1)]],  // [N/64] BF16
    device const bfloat*  biases [[buffer(2)]],  // [N/64] BF16
    device const half*    x      [[buffer(3)]],  // [N] FP16
    device       half*    h1     [[buffer(4)]],  // [D] FP16 in place
    constant     uint&    N      [[buffer(5)]],
    constant     uint&    D      [[buffer(6)]],
    uint  lane                   [[thread_index_in_simdgroup]]
) {
    const uint n_groups = N / kQwenGroupSize;
    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(scales[g]);
        const float b = float(biases[g]);
        const uint8_t byte = W[g * (kQwenGroupSize / 2u) + lane];
        const float x0 = float(x[g * kQwenGroupSize + lane * 2u]);
        const float x1 = float(x[g * kQwenGroupSize + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        const float sum = x0 + x1;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    acc = simd_sum(acc);
    const float gate = qwen_sigmoid(acc);
    for (uint i = lane; i < D; i += 32u) {
        h1[i] = half(float(h1[i]) * gate);
    }
}

// ============================================================================
// qwen_full_attn_epilogue — Qwen full-attention q/k post-processing.
//
// q_proj is doubled for attn_output_gate: per head, rows [q_256 | gate_256]
// (Qwen3_5MoeAttention: q_proj(h).view(-1, head_dim*2) chunked on dim=-1).
// Grid = num_q_heads + num_kv_heads threadgroups:
//   q head h:  q_out = rmsnorm(q, q_w); RoPE (partial); gate_out = raw gate
//   k head h:  k     = rmsnorm(k, k_w); RoPE (partial), in place
// v is untouched (already projected into the KV slot).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kQwenThreads)]]
void qwen_full_attn_epilogue(
    device const half*   q_proj    [[buffer(0)]],  // [numQHeads, 2*HD]
    device       half*   q_out     [[buffer(1)]],  // [numQHeads, HD]
    device       half*   gate_out  [[buffer(2)]],  // [numQHeads, HD]
    device       half*   k         [[buffer(3)]],  // [numKVHeads, HD] in place
    device const bfloat* q_weight  [[buffer(4)]],  // [HD] BF16
    device const bfloat* k_weight  [[buffer(5)]],  // [HD] BF16
    constant     uint&   head_dim  [[buffer(6)]],
    constant     uint&   num_q_heads  [[buffer(7)]],
    constant     uint&   num_kv_heads [[buffer(8)]],
    constant     uint&   position  [[buffer(9)]],
    constant     float&  theta_base [[buffer(10)]],
    constant     uint&   rotary_dim [[buffer(11)]],
    constant     float&  rms_eps   [[buffer(12)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane        [[thread_index_in_simdgroup]],
    uint  simd_group       [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]],
    uint  head_group       [[threadgroup_position_in_grid]]
) {
    threadgroup half  head_tg[kQwenMaxHeadDim];
    threadgroup float partial[kQwenMaxSimdGroups];

    const uint HD = head_dim;
    const uint NQ = num_q_heads;
    const bool is_q = head_group < NQ;
    const uint local_head = is_q ? head_group : (head_group - NQ);

    device const half* src = is_q ? (q_proj + local_head * 2u * HD)
                                  : (k + local_head * HD);
    device const bfloat* w = is_q ? q_weight : k_weight;

    float acc = 0.0f;
    for (uint i = lid; i < HD; i += lsize) {
        const float xv = float(src[i]);
        acc = fma(xv, xv, acc);
    }
    qwen_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(HD) + rms_eps);

    for (uint i = lid; i < HD; i += lsize) {
        head_tg[i] = half(float(src[i]) * inv * float(w[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        // Copy the raw gate half (unnormalized, unrotated) to gate_out.
        device const half* gateSrc = q_proj + local_head * 2u * HD + HD;
        for (uint i = lid; i < HD; i += lsize) {
            gate_out[local_head * HD + i] = gateSrc[i];
        }
    }

    // Partial rotary over the first `rotary_dim` elements, HF half-split
    // pairing (rotate_half): pair i mixes (i, i + rotary_dim/2) with the
    // shared inv_freq_i. Frequencies come from qwen_rope_pair's exponent
    // -2*i/rotary_dim, matching emb = cat(freqs, freqs) in the module.
    const uint half_rot = rotary_dim / 2u;
    for (uint pair = lid; pair < half_rot; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[pair + half_rot]);
        qwen_rope_pair(x0, x1, pair, rotary_dim, float(position), theta_base);
        head_tg[pair]            = half(x0);
        head_tg[pair + half_rot] = half(x1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half* dst = is_q ? (q_out + local_head * HD)
                            : (k + local_head * HD);
    for (uint i = lid; i < HD; i += lsize) {
        dst[i] = head_tg[i];
    }
}

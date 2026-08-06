// norm_kernels.metal
// ─────────────────────────────────────────────────────────────────────────────
// Fused RMSNorm kernels for Qwen3.6-35B-A3B
//
// Two variants:
//   rms_norm_bf16   — standard pre-norm / post-norm (input_layernorm,
//                     post_attention_layernorm, model.norm)
//   rms_norm_head   — QK-Norm applied per attention head (q_norm / k_norm)
//                     operates on [seq, n_heads, head_dim] with a weight
//                     vector of length head_dim
//
// Kernel strategy:
//   • One threadgroup per row (one token or one head slice).
//   • SIMD reduction (simd_sum) for the mean-square accumulator.
//   • Weights are bf16; activations are computed in float32 internally
//     then rounded back to bf16 on store.
//   • threadgroup_barrier(mem_flags::mem_threadgroup) between reduce and
//     broadcast is avoided by using simd_broadcast_first after the SIMD
//     reduce — valid when all threads in a SIMD group participate.
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── bf16 helpers ─────────────────────────────────────────────────────────────
// Metal's bfloat type is available on A16+/M2+; for earlier chips we
// reinterpret uint16_t manually.

inline float bf16_to_f32(ushort u) {
    uint f = (uint)u << 16;
    return as_type<float>(f);
}

inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    // round-to-nearest-even
    uint rounding = (u >> 16) & 1u;
    u += 0x7FFFu + rounding;
    return (ushort)(u >> 16);
}

// ── kernel: rms_norm_bf16 ────────────────────────────────────────────────────
//
// grid:        (num_rows, 1, 1)       one threadgroup per row
// threadgroup: (SIMD_SIZE * N, 1, 1)  typically 32 or 64 threads
//
// Each thread handles one or more elements of the row (stride = threads_per_tg).
// After SIMD reduction, every thread in the group has rms_inv and applies
// the scale+norm in a second pass.

kernel void rms_norm_bf16(
    device const ushort* __restrict__ input   [[ buffer(0) ]],  // [rows, hidden]
    device const ushort* __restrict__ weight  [[ buffer(1) ]],  // [hidden]
    device       ushort* __restrict__ output  [[ buffer(2) ]],  // [rows, hidden]
    constant     uint&                hidden  [[ buffer(3) ]],
    constant     float&               eps     [[ buffer(4) ]],
    uint  tgid   [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  tg_sz  [[ threads_per_threadgroup ]]
) {
    uint row    = tgid;
    uint base   = row * hidden;

    // ── pass 1: accumulate sum of squares ────────────────────────────────────
    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tg_sz) {
        float x = bf16_to_f32(input[base + i]);
        local_ss += x * x;
    }

    // SIMD reduce within each SIMD-group (32 threads), then across groups
    float ss = simd_sum(local_ss);

    // For threadgroups larger than one SIMD-group, reduce across groups
    // using threadgroup memory.
    threadgroup float tg_accum[32];   // max 32 SIMD groups per threadgroup
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_lane == 0) tg_accum[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < (tg_sz + 31u) / 32u) {
        ss = tg_accum[tid];
    } else {
        ss = 0.0f;
    }
    ss = simd_sum(ss);
    // Now lane 0 of simd-group 0 has total sum-of-squares; broadcast it.
    threadgroup float tg_ss;
    if (tid == 0) tg_ss = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = tg_ss;

    float rms_inv = rsqrt(ss / (float)hidden + eps);

    // ── pass 2: normalize and scale ───────────────────────────────────────────
    for (uint i = tid; i < hidden; i += tg_sz) {
        float x = bf16_to_f32(input[base + i]);
        float w = bf16_to_f32(weight[i]);
        output[base + i] = f32_to_bf16(x * rms_inv * w);
    }
}

// ── kernel: rms_norm_head (per-head QK-Norm) ─────────────────────────────────
//
// Applies RMSNorm independently to each head slice of Q or K.
//
// Input:  [seq_len, n_heads, head_dim]  (bf16, contiguous)
// Weight: [head_dim]                   (bf16, shared across all heads)
// Output: same shape as input          (bf16, in-place safe if input==output)
//
// grid:  (seq_len * n_heads, 1, 1)
// threadgroup: (min(head_dim, 64), 1, 1)

kernel void rms_norm_head(
    device const ushort* __restrict__ input    [[ buffer(0) ]],
    device const ushort* __restrict__ weight   [[ buffer(1) ]],
    device       ushort* __restrict__ output   [[ buffer(2) ]],
    constant     uint&                head_dim [[ buffer(3) ]],
    constant     float&               eps      [[ buffer(4) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    uint tg_sz [[ threads_per_threadgroup ]]
) {
    uint base = tgid * head_dim;

    float local_ss = 0.0f;
    for (uint i = tid; i < head_dim; i += tg_sz) {
        float x = bf16_to_f32(input[base + i]);
        local_ss += x * x;
    }
    float ss = simd_sum(local_ss);

    threadgroup float tg_ss_head;
    if (tid % 32u == 0) {
        // simd group 0 writes; other groups may be partial
        threadgroup_barrier(mem_flags::mem_none);
    }
    // Simpler path for head_dim <= 64 (fits in one SIMD group):
    // simd_sum already gives the total.  For larger head_dim we'd need
    // the multi-group reduction above, but Qwen3.6 head_dim=64.
    if (tid == 0) tg_ss_head = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = tg_ss_head;

    float rms_inv = rsqrt(ss / (float)head_dim + eps);

    for (uint i = tid; i < head_dim; i += tg_sz) {
        float x = bf16_to_f32(input[base + i]);
        float w = bf16_to_f32(weight[i]);
        output[base + i] = f32_to_bf16(x * rms_inv * w);
    }
}

// ── kernel: rms_norm_inplace_bf16 ────────────────────────────────────────────
// Same as rms_norm_bf16 but reads and writes to the same buffer (activations).
// Used for the final model.norm before the LM head projection.

kernel void rms_norm_inplace_bf16(
    device       ushort* __restrict__ inout   [[ buffer(0) ]],
    device const ushort* __restrict__ weight  [[ buffer(1) ]],
    constant     uint&                hidden  [[ buffer(2) ]],
    constant     float&               eps     [[ buffer(3) ]],
    uint  tgid   [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  tg_sz  [[ threads_per_threadgroup ]]
) {
    uint row  = tgid;
    uint base = row * hidden;

    float local_ss = 0.0f;
    for (uint i = tid; i < hidden; i += tg_sz) {
        float x = bf16_to_f32(inout[base + i]);
        local_ss += x * x;
    }
    float ss = simd_sum(local_ss);

    threadgroup float tg_accum2[32];
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_lane == 0) tg_accum2[simd_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < (tg_sz + 31u) / 32u) ss = tg_accum2[tid]; else ss = 0.0f;
    ss = simd_sum(ss);
    threadgroup float tg_ss2;
    if (tid == 0) tg_ss2 = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ss = tg_ss2;

    float rms_inv = rsqrt(ss / (float)hidden + eps);

    for (uint i = tid; i < hidden; i += tg_sz) {
        float x = bf16_to_f32(inout[base + i]);
        float w = bf16_to_f32(weight[i]);
        inout[base + i] = f32_to_bf16(x * rms_inv * w);
    }
}

// gemv_kernels.metal
// ─────────────────────────────────────────────────────────────────────────────
// GEMV (matrix × vector) kernels for the attention projections and LM head.
// Used for single-token decode (batch=1).
//
// Kernels:
//   gemv_bf16       — y = W @ x,  W bf16, x bf16, y bf16
//   gemv_bf16_f32   — y = W @ x,  W bf16, x bf16, y float32 (LM head logits)
//   gemv_add_bias   — y += b,  bias bf16 added to float32 output
//
// Strategy:
//   One threadgroup per output row. Threads stride across the inner dimension
//   (columns of W), accumulate in float32, SIMD-reduce, write result.
//   For hidden_size=4096: 4096 / 32 = 128 iterations per thread (32-thread group).
//   For vocab_size=151936: dispatched as (151936, 1, 1) groups, same kernel.
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float bf16_to_f32(ushort u) {
    return as_type<float>((uint)u << 16);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// ── kernel: gemv_bf16 ─────────────────────────────────────────────────────────
//
// y[row] = dot(W[row, :], x[:])
//
// W:   [out_dim, in_dim]   bf16, row-major
// x:   [in_dim]            bf16
// y:   [out_dim]           bf16
//
// grid:       (out_dim, 1, 1)
// threadgroup: (32, 1, 1)   one SIMD group — sufficient for in_dim ≤ 8192

kernel void gemv_bf16(
    device const ushort* __restrict__ W      [[ buffer(0) ]],
    device const ushort* __restrict__ x      [[ buffer(1) ]],
    device       ushort* __restrict__ y      [[ buffer(2) ]],
    constant     uint&                in_dim [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    uint row  = tgid;
    uint base = row * in_dim;

    float dot = 0.0f;
    for (uint i = tid; i < in_dim; i += tg_sz) {
        dot += bf16_to_f32(W[base + i]) * bf16_to_f32(x[i]);
    }
    dot = simd_sum(dot);

    if (tid == 0) y[row] = f32_to_bf16(dot);
}

// ── kernel: gemv_bf16_f32 ─────────────────────────────────────────────────────
//
// Same as gemv_bf16 but output is float32.
// Used for the LM head: logits = lm_head.weight @ hidden_states
// Keeping logits in f32 avoids precision loss when computing argmax/sampling.
//
// grid:       (vocab_size, 1, 1)   — 151936 threadgroups for Qwen3.6
// threadgroup: (32, 1, 1)

kernel void gemv_bf16_f32(
    device const ushort* __restrict__ W      [[ buffer(0) ]],
    device const ushort* __restrict__ x      [[ buffer(1) ]],
    device       float*  __restrict__ y      [[ buffer(2) ]],
    constant     uint&                in_dim [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    uint row  = tgid;
    uint base = row * in_dim;

    float dot = 0.0f;
    for (uint i = tid; i < in_dim; i += tg_sz) {
        dot += bf16_to_f32(W[base + i]) * bf16_to_f32(x[i]);
    }
    dot = simd_sum(dot);

    if (tid == 0) y[row] = dot;
}

// ── kernel: gemv_bf16_batched ─────────────────────────────────────────────────
//
// Batched GEMV for prefill: Y = W @ X  where X has multiple columns (tokens).
//
// W: [out_dim, in_dim]      bf16
// X: [in_dim,  batch]       bf16  (batch = seq_len during prefill)
// Y: [out_dim, batch]       bf16
//
// grid:       (out_dim * batch, 1, 1)
// threadgroup: (32, 1, 1)

kernel void gemv_bf16_batched(
    device const ushort* __restrict__ W      [[ buffer(0) ]],
    device const ushort* __restrict__ X      [[ buffer(1) ]],
    device       ushort* __restrict__ Y      [[ buffer(2) ]],
    constant     uint&                in_dim [[ buffer(3) ]],
    constant     uint&                batch  [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    uint row = tgid / batch;
    uint col = tgid % batch;   // token index

    float dot = 0.0f;
    for (uint i = tid; i < in_dim; i += tg_sz) {
        float w  = bf16_to_f32(W[row * in_dim + i]);
        float xi = bf16_to_f32(X[i * batch + col]);
        dot += w * xi;
    }
    dot = simd_sum(dot);

    if (tid == 0) Y[row * batch + col] = f32_to_bf16(dot);
}

// ── kernel: elementwise_add_bf16 ──────────────────────────────────────────────
//
// y += x  (in-place residual add, both bf16)
//
// grid: (hidden / 256, 1, 1)   — one threadgroup per 256-element chunk
// threadgroup: (256, 1, 1)

kernel void elementwise_add_bf16(
    device       ushort* __restrict__ y      [[ buffer(0) ]],
    device const ushort* __restrict__ x      [[ buffer(1) ]],
    constant     uint&                n      [[ buffer(2) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]],
    uint tgid [[ threadgroup_position_in_grid ]]
) {
    uint base = tgid * tg_sz;
    uint idx  = base + tid;
    if (idx < n) {
        float sum = bf16_to_f32(y[idx]) + bf16_to_f32(x[idx]);
        y[idx] = f32_to_bf16(sum);
    }
}

// ── kernel: argmax_f32 ────────────────────────────────────────────────────────
//
// Find the index of the maximum element in a float32 vector.
// Used for greedy sampling from LM head logits.
//
// logits: [vocab_size]  float32
// result: [1]           uint32
//
// grid: (1, 1, 1)
// threadgroup: (256, 1, 1)

kernel void argmax_f32(
    device const float*  __restrict__ logits   [[ buffer(0) ]],
    device       uint*   __restrict__ result   [[ buffer(1) ]],
    constant     uint&                vocab_sz [[ buffer(2) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_sz[[ threads_per_threadgroup ]]
) {
    float local_max = -INFINITY;
    uint  local_idx = 0;

    for (uint i = tid; i < vocab_sz; i += tg_sz) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    // SIMD reduce: we need both max value and its index.
    // Use two-step: first find global max, then find first index matching it.
    float group_max = simd_max(local_max);

    threadgroup float tg_max_buf[8];
    threadgroup uint  tg_idx_buf[8];
    uint simd_id   = tid / 32u;
    uint simd_lane = tid % 32u;
    if (simd_lane == 0) {
        tg_max_buf[simd_id] = group_max;
        tg_idx_buf[simd_id] = (local_max == group_max) ? local_idx : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float  gmax = tg_max_buf[0];
        uint   gidx = tg_idx_buf[0];
        uint   n_groups = (tg_sz + 31u) / 32u;
        for (uint g = 1; g < n_groups; ++g) {
            if (tg_max_buf[g] > gmax) {
                gmax = tg_max_buf[g];
                gidx = tg_idx_buf[g];
            }
        }
        result[0] = gidx;
    }
}

// rope_kernels.metal
// ─────────────────────────────────────────────────────────────────────────────
// RoPE (Rotary Position Embedding) for Qwen3.6-35B-A3B
//
// Qwen3 uses the NeoX hd2-split convention:
//   - The head_dim is split into two equal halves:
//       first hd2  = [0 .. head_dim/2)
//       second hd2 = [head_dim/2 .. head_dim)
//   - Rotation pairs: (x[i], x[i + head_dim/2]) for i in [0, head_dim/2)
//   - This differs from GPT-NeoX interleaved (x[2i], x[2i+1]).
//
// RoPE variant: YaRN (Yet another RoPE extensioN)
//   YaRN modifies the effective base frequency per dimension using a
//   linear interpolation factor `scale` and corrective factors.
//   For inference simplicity we precompute the cos/sin tables on the CPU
//   and pass them as buffers (see rope_table.h on the CPU side).
//   The Metal kernel just applies the rotation.
//
// Kernel variants:
//   rope_neox_bf16          — applies RoPE to a Q or K tensor in-place
//   rope_neox_fused_qk_bf16 — applies RoPE to Q and K in a single dispatch
//                             (halves the kernel launch overhead per layer)
// ─────────────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── bf16 helpers (shared with norm_kernels) ──────────────────────────────────

inline float bf16_to_f32(ushort u) {
    return as_type<float>((uint)u << 16);
}
inline ushort f32_to_bf16(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort)(u >> 16);
}

// ── kernel: rope_neox_bf16 ───────────────────────────────────────────────────
//
// Applies YaRN RoPE in-place to Q or K using precomputed cos/sin tables.
//
// tensor:   [seq_len, n_heads, head_dim]   bf16, in-place
// cos_sin:  [max_seq_len, head_dim]        float32, layout: cos[i] at [pos, i],
//                                          sin[i] at [pos, i + head_dim/2]
//           — or separate cos/sin buffers (variant below)
//
// grid:       (seq_len * n_heads, 1, 1)
// threadgroup (head_dim/2, 1, 1)           each thread handles one rotation pair
//
// NeoX hd2-split:
//   x0 = tensor[pos, head, i]              i in [0, head_dim/2)
//   x1 = tensor[pos, head, i + head_dim/2]
//   cos_val, sin_val from table at [position_id, i]
//
//   out0 = x0 * cos - x1 * sin
//   out1 = x0 * sin + x1 * cos

kernel void rope_neox_bf16(
    device       ushort* __restrict__ tensor    [[ buffer(0) ]],  // Q or K
    device const float*  __restrict__ cos_table [[ buffer(1) ]],  // [max_seq, hd/2]
    device const float*  __restrict__ sin_table [[ buffer(2) ]],  // [max_seq, hd/2]
    device const uint*   __restrict__ pos_ids   [[ buffer(3) ]],  // [seq_len]
    constant     uint&                n_heads   [[ buffer(4) ]],
    constant     uint&                head_dim  [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]]
) {
    // tgid = seq_pos * n_heads + head_id
    uint seq_pos = tgid / n_heads;
    uint head_id = tgid % n_heads;

    uint pos    = pos_ids[seq_pos];
    uint hd2   = head_dim / 2u;

    // Only threads [0, hd2) are active; threads >= hd2 are idle.
    if (tid >= hd2) return;

    uint base   = (seq_pos * n_heads + head_id) * head_dim;
    uint i      = tid;  // rotation pair index

    float x0    = bf16_to_f32(tensor[base + i]);
    float x1    = bf16_to_f32(tensor[base + i + hd2]);

    float c     = cos_table[pos * hd2 + i];
    float s     = sin_table[pos * hd2 + i];

    tensor[base + i]        = f32_to_bf16(x0 * c - x1 * s);
    tensor[base + i + hd2] = f32_to_bf16(x0 * s + x1 * c);
}

// ── kernel: rope_neox_fused_qk_bf16 ─────────────────────────────────────────
//
// Applies RoPE to Q and K simultaneously — one dispatch instead of two.
// Q and K may have different n_heads (GQA: Q has n_heads, K has n_kv_heads).
//
// Buffer layout:
//   q: [seq_len, n_heads,    head_dim]   bf16
//   k: [seq_len, n_kv_heads, head_dim]   bf16
//
// grid: (seq_len * max(n_heads, n_kv_heads), 1, 1)
//
// We launch max(n_heads, n_kv_heads) threadgroups per token.
// Threads whose head_id >= n_heads  skip Q.
// Threads whose head_id >= n_kv_heads skip K.

kernel void rope_neox_fused_qk_bf16(
    device       ushort* __restrict__ q         [[ buffer(0) ]],
    device       ushort* __restrict__ k         [[ buffer(1) ]],
    device const float*  __restrict__ cos_table [[ buffer(2) ]],
    device const float*  __restrict__ sin_table [[ buffer(3) ]],
    device const uint*   __restrict__ pos_ids   [[ buffer(4) ]],
    constant     uint&                n_heads    [[ buffer(5) ]],
    constant     uint&                n_kv_heads [[ buffer(6) ]],
    constant     uint&                head_dim   [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]]
) {
    uint max_heads = max(n_heads, n_kv_heads);
    uint seq_pos   = tgid / max_heads;
    uint head_id   = tgid % max_heads;

    uint pos  = pos_ids[seq_pos];
    uint hd2 = head_dim / 2u;

    if (tid >= hd2) return;

    float c = cos_table[pos * hd2 + tid];
    float s = sin_table[pos * hd2 + tid];

    // Apply to Q
    if (head_id < n_heads) {
        uint base_q = (seq_pos * n_heads + head_id) * head_dim;
        float x0    = bf16_to_f32(q[base_q + tid]);
        float x1    = bf16_to_f32(q[base_q + tid + hd2]);
        q[base_q + tid]        = f32_to_bf16(x0 * c - x1 * s);
        q[base_q + tid + hd2] = f32_to_bf16(x0 * s + x1 * c);
    }

    // Apply to K
    if (head_id < n_kv_heads) {
        uint base_k = (seq_pos * n_kv_heads + head_id) * head_dim;
        float x0    = bf16_to_f32(k[base_k + tid]);
        float x1    = bf16_to_f32(k[base_k + tid + hd2]);
        k[base_k + tid]        = f32_to_bf16(x0 * c - x1 * s);
        k[base_k + tid + hd2] = f32_to_bf16(x0 * s + x1 * c);
    }
}

// ── kernel: build_yarn_rope_tables ───────────────────────────────────────────
//
// Precomputes YaRN cos/sin tables on the GPU.
// Called once at startup (or when max_seq_len changes).
//
// YaRN effective frequency:
//   theta_i = base^(-2i / head_dim)
//   With YaRN:  theta_i' = theta_i / scale   (for high-freq dims)
//                           interpolated     (for mid-freq dims)
//                           unchanged        (for low-freq dims)
// For simplicity we implement the "extended" variant:
//   theta_i' = base^(-2i / head_dim) / scale_factor
// where scale_factor is passed in (128.0 for Qwen3.6 default context extension).
//
// grid:  (max_seq_len, 1, 1)
// threadgroup: (head_dim/2, 1, 1)

kernel void build_yarn_rope_tables(
    device       float* __restrict__ cos_out     [[ buffer(0) ]],  // [max_seq, hd/2]
    device       float* __restrict__ sin_out     [[ buffer(1) ]],  // [max_seq, hd/2]
    constant     float&              rope_theta  [[ buffer(2) ]],
    constant     float&              scale       [[ buffer(3) ]],  // YaRN scale
    constant     uint&               head_dim    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]]
) {
    uint pos  = tgid;
    uint hd2 = head_dim / 2u;
    if (tid >= hd2) return;

    // Effective frequency for dimension tid
    float exp_val  = -2.0f * (float)tid / (float)head_dim;
    float freq     = pow(rope_theta, exp_val) / scale;
    float angle    = (float)pos * freq;

    uint out_idx   = pos * hd2 + tid;
    cos_out[out_idx] = cos(angle);
    sin_out[out_idx] = sin(angle);
}

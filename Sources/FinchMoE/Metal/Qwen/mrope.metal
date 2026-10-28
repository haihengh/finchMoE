#include <metal_stdlib>
using namespace metal;

// Interleaved mRoPE for the Qwen multimodality path.
//
// This module is compiled lazily, only when a multimodal prefill asks for the
// kernel, so a text-only application never pays to build it. That is why the
// constants and the block-sum helper are local copies instead of the ones in
// qwen.metal (the two sources are not concatenated).
//
// Layout: the H section owns pair indices 1 (mod 3) below 3*sectionH, the W
// section owns 2 (mod 3) below 3*sectionW, and T keeps every other pair. For
// the shipped [11, 11, 10] sections and 32 rotary pairs that is the reference
// interleaved [T,H,W,T,H,W,...] order. Text-only runs never reach this kernel.

constant constexpr uint kMRoPEThreads = 256;
constant constexpr uint kMRoPEMaxSimdGroups = kMRoPEThreads / 32;
constant constexpr uint kMRoPEMaxHeadDim = 512;
constant constexpr uint kMRoPEIdxThreads = 128;
constant constexpr uint kMRoPEIdxMaxDim = 512;

static inline void mrope_block_sum(float acc,
                                   uint simd_lane,
                                   uint simd_group,
                                   uint simdgroups,
                                   threadgroup float* partial) {
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

static inline void qwen_rope_pair_mrope(thread float& x0,
                                        thread float& x1,
                                        uint pair_index,
                                        uint rotary_dim,
                                        float position_t,
                                        float position_h,
                                        float position_w,
                                        uint section_h,
                                        uint section_w,
                                        float theta_base)
{
    uint axis = 0;
    if ((pair_index % 3u) == 1u && pair_index < section_h * 3u) {
        axis = 1;
    } else if ((pair_index % 3u) == 2u && pair_index < section_w * 3u) {
        axis = 2;
    }
    const float position = axis == 0 ? position_t
                         : (axis == 1 ? position_h : position_w);
    const float exponent = -float(2u * pair_index) / float(rotary_dim);
    const float freq = pow(theta_base, exponent);
    const float angle = position * freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const float r0 = x0 * c - x1 * s;
    const float r1 = x0 * s + x1 * c;
    x0 = r0;
    x1 = r1;
}

// Clone of qwen_full_attn_epilogue with the mRoPE position triple. The
// text-only kernel is untouched, so a nil multimodal input still runs the
// original code and its original arguments.
[[kernel, max_total_threads_per_threadgroup(kMRoPEThreads)]]
void qwen_full_attn_epilogue_mrope(
    device const half*   q_proj    [[buffer(0)]],
    device       half*   q_out     [[buffer(1)]],
    device       half*   gate_out  [[buffer(2)]],
    device       half*   k         [[buffer(3)]],
    device const bfloat* q_weight  [[buffer(4)]],
    device const bfloat* k_weight  [[buffer(5)]],
    constant     uint&   head_dim  [[buffer(6)]],
    constant     uint&   num_q_heads  [[buffer(7)]],
    constant     uint&   num_kv_heads [[buffer(8)]],
    constant     uint&   position_t [[buffer(9)]],
    constant     float&  theta_base [[buffer(10)]],
    constant     uint&   rotary_dim [[buffer(11)]],
    constant     float&  rms_eps   [[buffer(12)]],
    constant     uint&   position_h [[buffer(13)]],
    constant     uint&   position_w [[buffer(14)]],
    constant     uint&   section_h [[buffer(15)]],
    constant     uint&   section_w [[buffer(16)]],
    uint  lid              [[thread_position_in_threadgroup]],
    uint  lsize            [[threads_per_threadgroup]],
    uint  simd_lane        [[thread_index_in_simdgroup]],
    uint  simd_group       [[simdgroup_index_in_threadgroup]],
    uint  simdgroups       [[simdgroups_per_threadgroup]],
    uint  head_group       [[threadgroup_position_in_grid]]
) {
    threadgroup half  head_tg[kMRoPEMaxHeadDim];
    threadgroup float partial[kMRoPEMaxSimdGroups];

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
    mrope_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(HD) + rms_eps);

    for (uint i = lid; i < HD; i += lsize) {
        head_tg[i] = half(float(src[i]) * inv * float(w[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (is_q) {
        device const half* gateSrc = q_proj + local_head * 2u * HD + HD;
        for (uint i = lid; i < HD; i += lsize) {
            gate_out[local_head * HD + i] = gateSrc[i];
        }
    }

    const uint half_rot = rotary_dim / 2u;
    for (uint pair = lid; pair < half_rot; pair += lsize) {
        float x0 = float(head_tg[pair]);
        float x1 = float(head_tg[pair + half_rot]);
        qwen_rope_pair_mrope(x0, x1, pair, rotary_dim,
                             float(position_t), float(position_h),
                             float(position_w), section_h, section_w,
                             theta_base);
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

// ============================================================================
// QSA indexer mRoPE variants. The text-only kernels in qsa_indexer.metal use
// one scalar position for both the raw-key store and the rotation; with
// multimodality those are different: keys stay on the KV timeline, while the
// rotation uses the token's (t,h,w) triple. The pooled block key is rotated
// at its first cell's triple, read from the prefill's position timeline.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kMRoPEIdxThreads)]]
void idx_qk_post_mrope(
    device const half*   qk        [[buffer(0)]],
    device const bfloat* q_gamma   [[buffer(1)]],
    device       half*   q_out     [[buffer(2)]],
    device       half*   k_raw     [[buffer(3)]],
    constant     uint&   store_pos [[buffer(4)]],
    constant     uint&   n_heads   [[buffer(5)]],
    constant     uint&   idx_dim   [[buffer(6)]],
    constant     uint&   n_rot     [[buffer(7)]],
    constant     float&  theta     [[buffer(8)]],
    constant     float&  eps       [[buffer(9)]],
    constant     uint&   pos_t     [[buffer(10)]],
    constant     uint&   pos_h     [[buffer(11)]],
    constant     uint&   pos_w     [[buffer(12)]],
    constant     uint&   section_h [[buffer(13)]],
    constant     uint&   section_w [[buffer(14)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_group [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    threadgroup half  staged[kMRoPEIdxMaxDim];
    threadgroup float partial[kMRoPEMaxSimdGroups];

    if (tg == n_heads) {
        device const half* src = qk + n_heads * idx_dim;
        device       half* dst = k_raw + (size_t)store_pos * idx_dim;
        for (uint i = lid; i < idx_dim; i += lsize) {
            dst[i] = src[i];
        }
        return;
    }

    device const half* src = qk + tg * idx_dim;
    float acc = 0.0f;
    for (uint i = lid; i < idx_dim; i += lsize) {
        const float v = float(src[i]);
        staged[i] = half(v);
        acc = fma(v, v, acc);
    }
    mrope_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(idx_dim) + eps);

    for (uint i = lid; i < idx_dim; i += lsize) {
        staged[i] = half(float(staged[i]) * inv * float(q_gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_rot = n_rot / 2u;
    device half* out = q_out + tg * idx_dim;
    if (lid < half_rot) {
        float x0 = float(staged[lid]);
        float x1 = float(staged[lid + half_rot]);
        qwen_rope_pair_mrope(x0, x1, lid, n_rot,
                             float(pos_t), float(pos_h), float(pos_w),
                             section_h, section_w, theta);
        out[lid]            = half(x0);
        out[lid + half_rot] = half(x1);
    } else if (lid >= n_rot && lid < idx_dim) {
        out[lid] = staged[lid];
    }
}

[[kernel, max_total_threads_per_threadgroup(kMRoPEIdxThreads)]]
void idx_block_pool_norm_rope_mrope(
    device const half*   k_raw    [[buffer(0)]],
    device const bfloat* k_gamma  [[buffer(1)]],
    device       half*   pooled   [[buffer(2)]],
    constant     uint&   first    [[buffer(3)]],
    constant     uint&   r        [[buffer(4)]],
    constant     uint&   idx_dim  [[buffer(5)]],
    constant     uint&   n_rot    [[buffer(6)]],
    constant     float&  theta    [[buffer(7)]],
    constant     float&  eps      [[buffer(8)]],
    /// Three UInt32 per prompt cell: (t, h, w), index 0 == position 0.
    device const uint*   positions [[buffer(9)]],
    constant     uint&   prompt_length [[buffer(10)]],
    /// RoPE position for decode cells past the prompt is cell + this.
    constant     int&    decode_offset [[buffer(11)]],
    constant     uint&   section_h [[buffer(12)]],
    constant     uint&   section_w [[buffer(13)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_group [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    threadgroup half  staged[kMRoPEIdxMaxDim];
    threadgroup float partial[kMRoPEMaxSimdGroups];

    const uint b = first + tg;
    const uint cell = b * r;
    uint pos_t, pos_h, pos_w;
    if (cell < prompt_length) {
        pos_t = positions[(size_t)cell * 3u];
        pos_h = positions[(size_t)cell * 3u + 1u];
        pos_w = positions[(size_t)cell * 3u + 2u];
    } else {
        const int shifted = int(cell) + decode_offset;
        pos_t = uint(max(shifted, 0));
        pos_h = pos_t;
        pos_w = pos_t;
    }

    device const half* base = k_raw + (size_t)b * r * idx_dim;
    float acc = 0.0f;
    for (uint i = lid; i < idx_dim; i += lsize) {
        float sum = 0.0f;
        for (uint j = 0; j < r; ++j) {
            sum += float(base[(size_t)j * idx_dim + i]);
        }
        staged[i] = half(sum * (1.0f / float(r)));
        const float v = float(staged[i]);
        acc = fma(v, v, acc);
    }
    mrope_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
    const float inv = rsqrt(partial[0] / float(idx_dim) + eps);

    for (uint i = lid; i < idx_dim; i += lsize) {
        staged[i] = half(float(staged[i]) * inv * float(k_gamma[i]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_rot = n_rot / 2u;
    device half* out = pooled + (size_t)b * idx_dim;
    if (lid < half_rot) {
        float x0 = float(staged[lid]);
        float x1 = float(staged[lid + half_rot]);
        qwen_rope_pair_mrope(x0, x1, lid, n_rot,
                             float(pos_t), float(pos_h), float(pos_w),
                             section_h, section_w, theta);
        out[lid]            = half(x0);
        out[lid + half_rot] = half(x1);
    } else if (lid >= n_rot && lid < idx_dim) {
        out[lid] = staged[lid];
    }
}

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// attention — split-KV tiled softmax attention for single-token decode.
//
// Decode path only: M_q = 1 (one query token), arbitrary seq_len history.
// The MPP prefill path handles M_q > 1 separately.
//
// Layout (caller-side contract):
//   Q   : [num_q_heads,  head_dim]                      FP16, contiguous.
//   K   : [seq_len, num_kv_heads, head_dim]             FP16, contiguous.
//   V   : [seq_len, num_kv_heads, head_dim]             FP16, same shape as K.
//         Full attention reuses the raw K projection for V, but its separate
//         normalization and RoPE paths make these buffers distinct here.
//   out : [num_q_heads,  head_dim]                      FP16.
//
// GQA: q_head -> kv_head = q_head / (num_q_heads / num_kv_heads).
//      Multiple Q heads share one KV head; the dispatch indexes Q heads.
//
// Online softmax recurrence (FP32 accumulators) — Milakov & Gimelshein 2018,
// also FlashAttention:
//   m_new   = max(m, s)
//   alpha   = exp(m - m_new)                 // rescale factor for past state
//   d       = d * alpha + exp(s - m_new)
//   o[i]    = o[i] * alpha + exp(s - m_new) * V[p, i]
//   m       = m_new
// Final normalization: out[i] = o[i] / d.
//
// ============================================================================

constant constexpr uint kAttnThreads      = 256;
// kAttnMaxSimdGroups must cover kAttnThreads / 32 = 8.
constant constexpr uint kAttnMaxSimdGroups = 8;
constant constexpr uint kAttnMaxQPerKV     = 2;
constant constexpr uint kAttnMaxFullQPerKV = 8;
constant constexpr uint kAttnFullQPerThreadgroup = 2;
// Largest head_dim we run with (full-attention layers). SWA uses 256 — the
// kernel still allocates the 512-slot scratch but only touches the live half.
constant constexpr uint kAttnMaxHeadDim   = 512;
constant uint FC_ATTN_HEAD_DIM [[function_constant(60)]];
constant uint FC_ATTN_NUM_Q_HEADS [[function_constant(61)]];
constant uint FC_ATTN_NUM_KV_HEADS [[function_constant(62)]];
constant bool FC_ATTN_USE_FC [[function_constant(63)]];
constant float FC_ATTN_SCALE [[function_constant(64)]];
constant uint FC_ATTN_NUM_CHUNKS [[function_constant(65)]];
constant uint FC_ATTN_RING_CAP [[function_constant(69)]];
// int8 K/V timelines (values + one fp16 scale per 64-element block). Set only
// for full-attention layers on a KVCacheManager running in `.int8` storage;
// the fp16 specialization never reads buffers 14-17.
constant bool FC_ATTN_INT8_KV [[function_constant(74)]];

static inline uint attn_fc_head_dim(constant uint& head_dim) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_HEAD_DIM))
        ? FC_ATTN_HEAD_DIM
        : head_dim;
}
static inline uint attn_fc_num_q_heads(constant uint& num_q_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_Q_HEADS))
        ? FC_ATTN_NUM_Q_HEADS
        : num_q_heads;
}

static inline uint attn_fc_num_kv_heads(constant uint& num_kv_heads) {
    return (is_function_constant_defined(FC_ATTN_USE_FC) &&
            FC_ATTN_USE_FC &&
            is_function_constant_defined(FC_ATTN_NUM_KV_HEADS))
        ? FC_ATTN_NUM_KV_HEADS
        : num_kv_heads;
}

static inline float attn_fc_scale(float scale) {
    return is_function_constant_defined(FC_ATTN_SCALE) ? FC_ATTN_SCALE : scale;
}

static inline uint attn_fc_num_chunks(constant uint& num_chunks) {
    return is_function_constant_defined(FC_ATTN_NUM_CHUNKS) ? FC_ATTN_NUM_CHUNKS : num_chunks;
}

static inline uint attn_ring_slot(uint p) {
    return (is_function_constant_defined(FC_ATTN_RING_CAP) &&
            FC_ATTN_RING_CAP != 0u)
        ? (p % FC_ATTN_RING_CAP)
        : p;
}

static inline float attn_softmax_exp(float x) {
    return fast::exp(x);
}

static inline bool attn_int8_kv(void) {
    return is_function_constant_defined(FC_ATTN_INT8_KV) && FC_ATTN_INT8_KV;
}

// One int8 element scaled by its 64-element block scale. `i` indexes within a
// single KV head's row, and a head is a whole number of blocks (256/64 = 4),
// so the block offset within the token row is head * blocksPerHead + i/64.
static inline float attn_kv_element(bool int8_kv,
                                    device const half* row16,
                                    device const char* row8,
                                    device const half* scale_row,
                                    uint i) {
    return int8_kv ? (float(row8[i]) * float(scale_row[i >> 6]))
                   : float(row16[i]);
}

// Block reduce: per-SIMD-group simd_sum, write partial to scratch, lane 0 of
// SIMD-group 0 finishes the merge with a second simd_sum and broadcasts.
// `scratch` must hold at least `simdgroups` floats; `bcast` is one float used
// to publish the final reduced value to all threads.
inline float block_reduce_sum(float v,
                              uint simd_lane_id,
                              uint simd_group_id,
                              uint simdgroups,
                              threadgroup float* scratch,
                              threadgroup float* bcast) {
    float s = simd_sum(v);
    if (simd_lane_id == 0) { scratch[simd_group_id] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        float t = (simd_lane_id < simdgroups) ? scratch[simd_lane_id] : 0.0f;
        t = simd_sum(t);
        if (simd_lane_id == 0) { *bcast = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *bcast;
}


// ============================================================================
// Split-KV (Flash-Decoding) decode attention — the default path.
//

// Pass 1 (attention_decode_partial): grid = num_q_heads * num_chunks. Each TG
//   runs the same online-softmax recurrence over its chunk [p_start, p_end) and
//   writes the UN-normalized partial state (m_chunk, d_chunk, o_chunk[head_dim])
//   to scratch — no division yet.
// Pass 2 (attention_decode_combine): grid = num_q_heads. Each TG merges its
//   head's num_chunks partials with the standard online-softmax rescale
//   (m_glob = max_c m_c; D = Σ d_c·e^{m_c−m_glob}; O = Σ o_c·e^{m_c−m_glob}) and
//   writes out[i] = O[i] / D in FP16.
//
// At num_chunks == 1 the chunk spans the whole [kv_start, seq_len) range and
// the partial is the exact single-pass accumulation; the combine's only chunk
// has m_glob == m_chunk so e^0 == 1 and out == o/d — byte-identical to the
// single-pass kernels above. num_chunks > 1 changes the FP rounding of the
// partial sums only (same position summation order), not the algorithm.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    device const char*  K8            [[buffer(14)]],  // int8 K timeline
    device const half*  KScale        [[buffer(15)]],  // fp16 block scales
    device const char*  V8            [[buffer(16)]],  // int8 V timeline
    device const half*  VScale        [[buffer(17)]],  // fp16 block scales
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const bool int8KV = attn_int8_kv();
    const uint blocksPerHead = HD >> 6;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // p_start can land past the end when num_chunks > range length (the tail
    // chunks are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;
        device const char* K_row8 = K8 + (phys_p * NKV + kv_head) * HD;
        device const char* V_row8 = V8 + (phys_p * NKV + kv_head) * HD;
        device const half* K_scale = KScale + phys_p * (NKV * blocksPerHead)
                                   + kv_head * blocksPerHead;
        device const half* V_scale = VScale + phys_p * (NKV * blocksPerHead)
                                   + kv_head * blocksPerHead;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i],
                          attn_kv_element(int8KV, K_row, K_row8, K_scale, i),
                          partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha
                + p_exp * attn_kv_element(int8KV, V_row, V_row8, V_scale, i);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

// ============================================================================
// attention_decode_cells_partial — the same online-softmax partial as
// attention_decode_partial, but the positions it walks come from an explicit
// ascending cell list instead of the contiguous range [kv_start, seq_len).
//
// This is the body of QSA (query-selection attention): the indexer picks the
// cells that may carry softmax mass and everything else is treated as -inf,
// which is exactly what "not in the list" means here. A cell never enters the
// running max, so it contributes no mass and no output — the mask is applied
// by omission rather than by a large negative bias.
//
// Chunking is over the LIST, not the timeline: chunk c owns list entries
// [c * chunk_len, min((c+1) * chunk_len, n_cells)). That keeps the split-KV
// shape identical to the dense kernel, so pass 2 is the very same
// attention_decode_combine — it merges num_chunks partials per head and has no
// idea what a chunk's entries mean.
//
// Passing the cell list 0…n-1 with n_cells == seq_len reproduces
// attention_decode_partial exactly, instruction for instruction: the same
// chunk geometry yields the same j ranges, and cells[j] == j makes every
// K/V row hit the same address as the dense p. That equivalence is what the
// tests lean on.
//
// No ring mapping here. `attn_ring_slot` exists for the SWA FP16 ring, and the
// indexer only runs on full-attention layers, which are unringed — applying it
// would silently wrap a cell list into a smaller buffer.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_cells_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    device const uint*  cells         [[buffer(6)]],   // [n_cells], ascending
    constant     uint&  head_dim      [[buffer(7)]],
    constant     uint&  num_q_heads   [[buffer(8)]],
    constant     uint&  num_kv_heads  [[buffer(9)]],
    constant     uint&  n_cells       [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    device const char*  K8            [[buffer(14)]],  // int8 K timeline
    device const half*  KScale        [[buffer(15)]],  // fp16 block scales
    device const char*  V8            [[buffer(16)]],  // int8 V timeline
    device const half*  VScale        [[buffer(17)]],  // fp16 block scales
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxSimdGroups];
    threadgroup float bcast;
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const bool int8KV = attn_int8_kv();
    const uint blocksPerHead = HD >> 6;

    const uint q_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint j_start = chunk * chunk_len;
    uint j_end = j_start + chunk_len;
    if (j_end > n_cells) { j_end = n_cells; }

    const uint kv_head = q_head / (NQ / NKV);

    device const half* Q_row = Q + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        q_smem[i] = float(Q_row[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr uint kPerThread = (kAttnMaxHeadDim + kAttnThreads - 1) / kAttnThreads;
    float o_local[kPerThread];
    for (uint k = 0; k < kPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    // j_start can land past the end when num_chunks > n_cells (the tail chunks
    // are empty); the loop simply does not execute and the partial is
    // (-inf, 0, 0), which the combine weights to zero via e^{-inf}.
    for (uint j = j_start; j < j_end; ++j) {
        const uint p = cells[j];
        device const half* K_row = K + (p * NKV + kv_head) * HD;
        device const half* V_row = V + (p * NKV + kv_head) * HD;
        device const char* K_row8 = K8 + (p * NKV + kv_head) * HD;
        device const char* V_row8 = V8 + (p * NKV + kv_head) * HD;
        device const half* K_scale = KScale + p * (NKV * blocksPerHead)
                                   + kv_head * blocksPerHead;
        device const half* V_scale = VScale + p * (NKV * blocksPerHead)
                                   + kv_head * blocksPerHead;

        float partial = 0.0f;
        for (uint i = lid; i < HD; i += lsize) {
            partial = fma(q_smem[i],
                          attn_kv_element(int8KV, K_row, K_row8, K_scale, i),
                          partial);
        }
        float s = block_reduce_sum(partial,
                                   simd_lane_id, simd_group_id, simdgroups,
                                   reduce_scratch, &bcast);
        s *= attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s     - m_new);
        d_run = d_run * alpha + p_exp;

        uint slot = 0;
        for (uint i = lid; i < HD; i += lsize) {
            o_local[slot] = o_local[slot] * alpha
                + p_exp * attn_kv_element(int8KV, V_row, V_row8, V_scale, i);
            slot += 1;
        }
        m_run = m_new;
    }

    const uint base = uint(q_head) * NC + chunk;
    if (lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = lid; i < HD; i += lsize) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_gqa_swa_partial(
    device const half*  Q             [[buffer(0)]],
    device const half*  K             [[buffer(1)]],
    device const half*  V             [[buffer(2)]],
    device       float* m_out         [[buffer(3)]],   // [num_q_heads * num_chunks]
    device       float* d_out         [[buffer(4)]],   // [num_q_heads * num_chunks]
    device       float* o_out         [[buffer(5)]],   // [num_q_heads * num_chunks * head_dim]
    constant     uint&  head_dim      [[buffer(6)]],
    constant     uint&  num_q_heads   [[buffer(7)]],
    constant     uint&  num_kv_heads  [[buffer(8)]],
    constant     uint&  seq_len       [[buffer(9)]],
    constant     uint&  kv_start      [[buffer(10)]],
    constant     uint&  chunk_len     [[buffer(11)]],
    constant     uint&  num_chunks    [[buffer(12)]],
    constant     float& scale         [[buffer(13)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]],
    uint simd_lane_id    [[thread_index_in_simdgroup]],
    uint simd_group_id   [[simdgroup_index_in_threadgroup]],
    uint simdgroups      [[simdgroups_per_threadgroup]]
) {
    threadgroup float q_smem[kAttnMaxQPerKV * kAttnMaxHeadDim];
    threadgroup float reduce_scratch[kAttnMaxQPerKV * kAttnMaxSimdGroups];
    threadgroup float bcast[kAttnMaxQPerKV];
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NQ = attn_fc_num_q_heads(num_q_heads);
    const uint NKV = attn_fc_num_kv_heads(num_kv_heads);
    const uint NC = attn_fc_num_chunks(num_chunks);

    const uint q_per_kv = NQ / NKV;
    if (q_per_kv > kAttnMaxQPerKV) { return; }

    const uint kv_head = tg_id / NC;
    const uint chunk  = tg_id % NC;
    const uint p_start = kv_start + chunk * chunk_len;
    uint p_end = p_start + chunk_len;
    if (p_end > seq_len) { p_end = seq_len; }

    const uint q_base = kv_head * q_per_kv;
    for (uint qg = 0; qg < q_per_kv; ++qg) {
        device const half* Q_row = Q + (q_base + qg) * HD;
        threadgroup float* Q_s = q_smem + qg * kAttnMaxHeadDim;
        for (uint i = lid; i < HD; i += lsize) {
            Q_s[i] = float(Q_row[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint groups_per_q = max(1u, simdgroups / q_per_kv);
    const uint active_q = min(q_per_kv - 1u, simd_group_id / groups_per_q);
    const uint local_group = simd_group_id - active_q * groups_per_q;
    const uint threads_per_q = groups_per_q * 32u;
    const uint local_lid = local_group * 32u + simd_lane_id;

    constexpr uint kGQAPerThread =
        (kAttnMaxHeadDim + (kAttnThreads / kAttnMaxQPerKV) - 1) /
        (kAttnThreads / kAttnMaxQPerKV);
    float o_local[kGQAPerThread];
    for (uint k = 0; k < kGQAPerThread; ++k) { o_local[k] = 0.0f; }

    float m_run = -INFINITY;
    float d_run = 0.0f;

    for (uint p = p_start; p < p_end; ++p) {
        const uint phys_p = attn_ring_slot(p);
        device const half* K_row = K + (phys_p * NKV + kv_head) * HD;
        device const half* V_row = V + (phys_p * NKV + kv_head) * HD;

        float partial = 0.0f;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            const float k_val = float(K_row[i]);
            partial = fma(q_smem[active_q * kAttnMaxHeadDim + i], k_val, partial);
        }

        float s = simd_sum(partial);
        if (simd_lane_id == 0) {
            reduce_scratch[active_q * kAttnMaxSimdGroups + local_group] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_group == 0) {
            float t = (simd_lane_id < groups_per_q)
                ? reduce_scratch[active_q * kAttnMaxSimdGroups + simd_lane_id]
                : 0.0f;
            t = simd_sum(t);
            if (simd_lane_id == 0) { bcast[active_q] = t; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        s = bcast[active_q] * attn_fc_scale(scale);

        const float m_new = max(m_run, s);
        const float alpha = attn_softmax_exp(m_run - m_new);
        const float p_exp = attn_softmax_exp(s - m_new);
        d_run = d_run * alpha + p_exp;
        for (uint slot = 0; slot < kGQAPerThread; ++slot) { o_local[slot] *= alpha; }
        m_run = m_new;

        uint slot = 0;
        for (uint i = local_lid; i < HD; i += threads_per_q) {
            o_local[slot] += p_exp * float(V_row[i]);
            slot += 1;
        }
    }

    const uint q_head = q_base + active_q;
    const uint base = uint(q_head) * NC + chunk;
    if (local_lid == 0) { m_out[base] = m_run; d_out[base] = d_run; }
    device float* o_row = o_out + base * HD;
    uint slot = 0;
    for (uint i = local_lid; i < HD; i += threads_per_q) {
        o_row[i] = o_local[slot];
        slot += 1;
    }
}

[[kernel, max_total_threads_per_threadgroup(kAttnThreads)]]
void attention_decode_combine(
    device const float* m_in         [[buffer(0)]],    // [num_q_heads * num_chunks]
    device const float* d_in         [[buffer(1)]],
    device const float* o_in         [[buffer(2)]],    // [num_q_heads * num_chunks * head_dim]
    device       half*  out          [[buffer(3)]],    // [num_q_heads * head_dim]
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  num_chunks   [[buffer(5)]],
    uint tg_id           [[threadgroup_position_in_grid]],
    uint lid             [[thread_position_in_threadgroup]],
    uint lsize           [[threads_per_threadgroup]]
) {
    const uint HD = attn_fc_head_dim(head_dim);
    const uint NC = attn_fc_num_chunks(num_chunks);
    const uint q_head = tg_id;
    device const float* m_row  = m_in + uint(q_head) * NC;
    device const float* d_row  = d_in + uint(q_head) * NC;
    device const float* o_base = o_in + uint(q_head) * NC * HD;

    // num_chunks is small (<= a few dozen); each thread recomputes the global
    // max and denominator rather than pay a threadgroup reduction + barriers.
    float m_glob = -INFINITY;
    for (uint c = 0; c < NC; ++c) { m_glob = max(m_glob, m_row[c]); }
    float D = 0.0f;
    for (uint c = 0; c < NC; ++c) { D += d_row[c] * attn_softmax_exp(m_row[c] - m_glob); }
    const float inv_d = (D > 0.0f) ? (1.0f / D) : 0.0f;

    device half* out_row = out + uint(q_head) * HD;
    for (uint i = lid; i < HD; i += lsize) {
        float acc = 0.0f;
        for (uint c = 0; c < NC; ++c) {
            acc += o_base[c * HD + i] * attn_softmax_exp(m_row[c] - m_glob);
        }
        out_row[i] = half(acc * inv_d);
    }
}

// ============================================================================
// kv_quantize_int8_rows — the int8 KV storage writer.
//
// One threadgroup per row (a token's K or V row for one layer), eight
// SIMD-groups per threadgroup, each SIMD-group owning one 64-element block of a
// 512-element row. Symmetric int8: scale = absmax / 127, values rounded and
// clamped to [-127, 127]; every reader multiplies by the same fp16 scale.
//
// Symmetric rather than affine because a KV row is zero-centred (RMS-normed
// keys, raw values), so an affine zero point would spend a second fp16 per
// block on a range the data does not use.
// ============================================================================

constant constexpr uint kKVQuantBlockElements = 64;
constant constexpr uint kKVQuantThreads       = 256;

[[kernel, max_total_threads_per_threadgroup(kKVQuantThreads)]]
kernel void kv_quantize_int8_rows(
    device const half*  src         [[buffer(0)]],
    device       char*  dst         [[buffer(1)]],
    device       half*  scales      [[buffer(2)]],
    constant     uint&  rowDim      [[buffer(3)]],
    constant     uint&  blockElems  [[buffer(4)]],
    uint tg          [[threadgroup_position_in_grid]],
    uint sg          [[simdgroup_index_in_threadgroup]],
    uint lane        [[thread_index_in_simdgroup]],
    uint simdgroups  [[simdgroups_per_threadgroup]]
) {
    // One SIMD-group per block; the threadgroup walks however many blocks the
    // row has (8 for Qwen's 512-element full KV row, 16 for Gemma's 1024).
    const uint blocks = rowDim / blockElems;
    for (uint b = sg; b < blocks; b += simdgroups) {
        const uint base = tg * rowDim + b * blockElems;
        float mx = 0.0f;
        for (uint i = lane; i < blockElems; i += 32) {
            mx = max(mx, fabs(float(src[base + i])));
        }
        mx = simd_max(mx);

        const float scale = mx > 0.0f ? (mx / 127.0f) : 0.0f;
        const float inv   = scale > 0.0f ? (1.0f / scale) : 0.0f;
        for (uint i = lane; i < blockElems; i += 32) {
            const int q = int(round(float(src[base + i]) * inv));
            dst[base + i] = char(clamp(q, -127, 127));
        }
        if (lane == 0) {
            scales[tg * blocks + b] = half(scale);
        }
    }
}

// ============================================================================
// kv_simulate_int4_roundtrip_rows — a precision probe, not a storage format.
//
// Rewrites fp16 K/V rows in place as dequantize(quantize(x)) at `bits` bits
// with a symmetric scale per `groupElems`-element group. The result stays fp16
// in the fp16 cache, so every attention kernel, every layout and every byte
// count is exactly what ships today: the only variable is how much precision
// the K/V values carry. That is what makes it a quality probe for a 4-bit KV
// format that does not exist yet — if the answer is that 4-bit K/V breaks the
// model, the packing, the unpack kernels and the metadata layout were never
// worth writing.
//
// `FQ_KV_INT4_SIM=<group>` selects group 32/64/128; unset means off.
// ============================================================================

[[kernel, max_total_threads_per_threadgroup(256)]]
kernel void kv_simulate_int4_roundtrip_rows(
    device half*    rows        [[buffer(0)]],   // [rows][rowDim], modified
    constant uint&  rowDim      [[buffer(1)]],
    constant uint&  groupElems  [[buffer(2)]],
    constant uint&  bits        [[buffer(3)]],
    uint tg         [[threadgroup_position_in_grid]],
    uint sg         [[simdgroup_index_in_threadgroup]],
    uint lane       [[thread_index_in_simdgroup]],
    uint simdgroups [[simdgroups_per_threadgroup]]
) {
    const uint groups = rowDim / groupElems;
    const float levels = float((1u << (bits - 1u)) - 1u);   // 7 at 4 bits
    for (uint g = sg; g < groups; g += simdgroups) {
        const uint base = tg * rowDim + g * groupElems;
        float mx = 0.0f;
        for (uint i = lane; i < groupElems; i += 32) {
            mx = max(mx, fabs(float(rows[base + i])));
        }
        mx = simd_max(mx);
        const float scale = mx > 0.0f ? (mx / levels) : 0.0f;
        const float inv   = scale > 0.0f ? (1.0f / scale) : 0.0f;
        // Read-modify-write stays inside one lane's own indices, so the in-place
        // update needs no barrier behind the absmax reduction.
        for (uint i = lane; i < groupElems; i += 32) {
            const float x = float(rows[base + i]);
            const float q = clamp(round(x * inv), -levels, levels);
            rows[base + i] = half(q * scale);
        }
    }
}

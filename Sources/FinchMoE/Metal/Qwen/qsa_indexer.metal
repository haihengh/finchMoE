#include <metal_stdlib>
using namespace metal;

// ============================================================================
// qsa_indexer.metal — Qwen 3.8 Flash-Next QSA sparse-block indexer (decode).
//
// Math pinned to `archive/llama.cpp/src/models/qwen4exp.cpp`
// `build_qsa_top_k` (:477-616) and `llama-memory-hybrid-idx.cpp`
// `set_input_qsa` (:342-464). Every kernel here is cross-checked against the
// fp32 `QSAIndexerRef` by `QSAIndexerTests`.
//
//   idx_qk_post   one GEMV emits the whole indexer projection: rows [0:nH)
//                 are the query heads, rows [nH:nH+1] the single shared key
//                 head. Query heads: rmsnorm(index_q_norm) then partial rope
//                 at the token position (:588-593). The key head is stored
//                 RAW — pooled keys carry the norm and the rotation, and
//                 pooling precedes both, so the cache must hold untouched
//                 projections (:533-538).
//
//   idx_block_pool_norm_rope
//                 pooled[b] = mean over the r raw keys of block b, then
//                 rmsnorm(index_k_norm), then partial rope at the block's
//                 FIRST cell b*r (:374 — all four MRoPE sections carry that
//                 one position, so a text run rotates at b*r) (:543-560).
//
//   idx_block_scores
//                 score[b] = Σ_h relu(q[h] · pooled[b]) — the relu sits on
//                 each head dot BEFORE the sum, as in the DeepSeek lightning
//                 indexer (:576-585) — then + blk_bias.
//
// The block bias is folded into the score because llama folds it there: "one
// value per block, so it is cheaper to bias here than after the cells are
// expanded" (:594-596). Per cell the bias is
//
//   tail_start = (pos + 1) / r * r                       (:436, int division)
//   bias[b]    = b*r >= tail_start ? +1e9  :   force-visible incomplete tail
//                filled[b] < r     ? -INF :   unusable partial pool (:421-428)
//                                       0
//
// `tail_start` names the first cell past the query's own block, so the +1e9
// arm is exactly the block that cannot be pooled yet. It is finite so it can
// never meet a -INF mask and produce a NaN (:444). In a decode step the
// timeline is contiguous from cell 0 and one token is appended, so at most
// one block is incomplete — the query's own, and only when the query is not
// its last cell — which makes that block force-visible and its pool
// unreachable: the kernel returns +1e9 without reading it. Every other block
// is complete, so the -INF arm is llama's cache-hole case (`:397-399`) and
// stays unreachable for a timeline the engine holds.
//
// Causality is NOT here. llama puts it on the expanded per-cell scores
// (`:597-600`), and the plan's engine equivalent is the per-cell visibility
// mask the selection step builds; the bias kernel is deliberately blind to it.
//
// Activation storage is FP16 (the decode convention) with FP32 kernel
// internals. The pooling mean and the RMS sums accumulate in FP32; the
// normalised vector is staged as FP16 before the rope, the way
// qwen_full_attn_epilogue stages a head — llama has the same round trip
// through its f16 pooled/normed tensors, so the two pipelines round at the
// same points. Only the vector exchange the rope needs is staged; no other
// intermediate leaves FP32.
// ============================================================================

constant constexpr uint kIdxDim          = 128;  // indexer head width (indexer_head_dim)
constant constexpr uint kIdxThreads      = 128;  // one thread per indexer dim
constant constexpr uint kIdxMaxDim       = 512;  // threadgroup staging cap
constant constexpr uint kIdxScoreThreads = 256;  // 8 SIMD groups = up to 8 query heads
constant constexpr uint kIdxMaxHeads     = 8;    // threadgroup-memory cap, one slot per head

// The post/pool kernels are dispatched with kIdxThreads and address the head
// dim one thread per element, so they require idx_dim <= kIdxThreads: a
// threadgroup wider than the head would either leave dims unwritten or (worse)
// write past the head's slice. `QSAIndexer` carries the same bound as a
// precondition, and the real indexer head is exactly kIdxDim.

// ============================================================================
// idx_qk_post — indexer query/key post-processing.
//
// Grid = nHeads + 1 threadgroups of kIdxThreads:
//   tg < nHeads: q_out[tg] = rope(rmsnorm(qk[tg], q_gamma), pos)
//   tg == nHeads: k_raw[pos] = qk[nHeads]        (verbatim — the raw timeline)
//
// The RMS reduction is two-stage SIMD-group (qwen_block_sum); the rope needs
// the normalised vector, so it lands in threadgroup memory the way
// qwen_full_attn_epilogue stages a head.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kIdxThreads)]]
void idx_qk_post(
    device const half*   qk        [[buffer(0)]],  // [(nHeads+kvHeads)*idxDim] FP16 GEMV out
    device const bfloat* q_gamma   [[buffer(1)]],  // [idxDim] BF16, 1+w already baked
    device       half*   q_out     [[buffer(2)]],  // [nHeads*idxDim] FP16 norm+roped
    device       half*   k_raw     [[buffer(3)]],  // [cap*idxDim] FP16 raw-key timeline
    constant     uint&   pos       [[buffer(4)]],  // query position
    constant     uint&   n_heads   [[buffer(5)]],
    constant     uint&   idx_dim   [[buffer(6)]],
    constant     uint&   n_rot     [[buffer(7)]],  // rotated leading dims (32)
    constant     float&  theta     [[buffer(8)]],
    constant     float&  eps       [[buffer(9)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_group [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    threadgroup half  staged[kIdxMaxDim];
    threadgroup float partial[kQwenMaxSimdGroups];

    if (tg == n_heads) {
        // Shared key head: the indexer cache holds keys raw, because pooling
        // precedes both the norm and the rotation (:533-538).
        device const half* src = qk + n_heads * idx_dim;
        device       half* dst = k_raw + (size_t)pos * idx_dim;
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
    qwen_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
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
        qwen_rope_pair(x0, x1, lid, n_rot, float(pos), theta);
        out[lid]            = half(x0);
        out[lid + half_rot] = half(x1);
    } else if (lid >= n_rot && lid < idx_dim) {
        // dims from n_rot on are carried through unrotated (partial rope);
        // [half_rot, n_rot) is written by those dims' pair partners. The
        // `lid < idx_dim` bound matters when the threadgroup is wider than
        // the head (a narrow test geometry): threads past idx_dim must not
        // write, or they would spill into the next head's slice.
        out[lid] = staged[lid];
    }
}

// ============================================================================
// idx_block_pool_norm_rope — finalise `count` whole blocks starting at
// `first`: mean over the r raw keys (fp32 accumulator), RMS with the k gamma,
// partial rope at the block's first cell b*r. One threadgroup per block.
//
// Only COMPLETE blocks are finalised — the mean divides by r, and a block's
// pooled key is written once and never recomputed, which is what lets the
// decode step skip llama's per-step re-pooling of the whole timeline. An
// incomplete block is exactly the block the bias force-visibles, so its pool
// is never scored (and by extension never needed).
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kIdxThreads)]]
void idx_block_pool_norm_rope(
    device const half*   k_raw    [[buffer(0)]],  // [n_kv*idxDim] FP16 raw keys
    device const bfloat* k_gamma  [[buffer(1)]],  // [idxDim] BF16, 1+w already baked
    device       half*   pooled   [[buffer(2)]],  // [n_blocks*idxDim] FP16 norm+roped
    constant     uint&   first    [[buffer(3)]],  // first block this dispatch finalises
    constant     uint&   r        [[buffer(4)]],
    constant     uint&   idx_dim  [[buffer(5)]],
    constant     uint&   n_rot    [[buffer(6)]],
    constant     float&  theta    [[buffer(7)]],
    constant     float&  eps      [[buffer(8)]],
    uint  tg   [[threadgroup_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  lsize [[threads_per_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_group [[simdgroup_index_in_threadgroup]],
    uint  simdgroups [[simdgroups_per_threadgroup]]
) {
    threadgroup half  staged[kIdxMaxDim];
    threadgroup float partial[kQwenMaxSimdGroups];

    const uint b = first + tg;
    device const half* base = k_raw + (size_t)b * r * idx_dim;

    float acc = 0.0f;
    for (uint i = lid; i < idx_dim; i += lsize) {
        float sum = 0.0f;
        for (uint j = 0; j < r; ++j) {
            sum += float(base[(size_t)j * idx_dim + i]);
        }
        // llama scales the f16 pooled tensor, so the mean is rounded to the
        // cache dtype before the norm reduces over it; reduce on that value.
        staged[i] = half(sum * (1.0f / float(r)));
        const float v = float(staged[i]);
        acc = fma(v, v, acc);
    }
    qwen_block_sum(acc, simd_lane, simd_group, simdgroups, partial);
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
        qwen_rope_pair(x0, x1, lid, n_rot, float(b * r), theta);
        out[lid]            = half(x0);
        out[lid + half_rot] = half(x1);
    } else if (lid >= n_rot && lid < idx_dim) {
        out[lid] = staged[lid];
    }
}

// ============================================================================
// idx_block_scores — score[b] = Σ_h relu(q[h] · pooled[b]) + blk_bias.
// One threadgroup per block, one SIMD group per query head: each group
// reduces its own head's dot over the head dim, so the per-head relu lands
// before the sum with no threadgroup-wide reduction at all.
// Grid = n_blocks threadgroups of 32*n_heads threads.
// ============================================================================
[[kernel, max_total_threads_per_threadgroup(kIdxScoreThreads)]]
void idx_block_scores(
    device const half*  q        [[buffer(0)]],  // [n_heads*idxDim] FP16 norm+roped
    device const half*  pooled   [[buffer(1)]],  // [n_blocks*idxDim] FP16 norm+roped
    device       float* scores   [[buffer(2)]],  // [n_blocks] FP32 out, biased
    constant     uint&  n_heads  [[buffer(3)]],
    constant     uint&  idx_dim  [[buffer(4)]],
    constant     uint&  r        [[buffer(5)]],
    constant     uint&  n_kv     [[buffer(6)]],
    constant     uint&  pos      [[buffer(7)]],
    uint  b    [[threadgroup_position_in_grid]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]
) {
    threadgroup float head_score[kIdxMaxHeads];

    // The tail block cannot be pooled yet, so the bias forces it visible and
    // its score is never read for a decision: llama stores 1e9 + a finite
    // pool score, and no complete block reaches 1e9, so the tail outranks
    // every scoreable block either way. Returning here also means the
    // kernel never touches a pooled slot that was never written.
    const uint tail_start = ((pos + 1u) / r) * r;
    if (b * r >= tail_start) {
        if (sg == 0 && lane == 0) scores[b] = 1e9f;
        return;
    }

    device const half* kb = pooled + (size_t)b * idx_dim;
    if (sg < n_heads) {
        device const half* qh = q + sg * idx_dim;
        float acc = 0.0f;
        for (uint i = lane; i < idx_dim; i += 32u) {
            acc = fma(float(qh[i]), float(kb[i]), acc);
        }
        acc = simd_sum(acc);
        if (lane == 0) head_score[sg] = max(acc, 0.0f);   // relu, per head
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0 && lane == 0) {
        float s = 0.0f;
        for (uint h = 0; h < n_heads; ++h) {
            s += head_score[h];
        }
        // filled[b] < r is llama's unusable-partial-pool case (:421-428),
        // unreachable while the timeline is contiguous from cell 0.
        const uint filled = n_kv > b * r ? min(r, n_kv - b * r) : 0u;
        scores[b] = filled < r ? -INFINITY : s;
    }
}

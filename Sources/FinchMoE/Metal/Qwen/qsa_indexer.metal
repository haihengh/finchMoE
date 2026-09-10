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

// ============================================================================
// idx_select_cells — the sparse-path selection, from the biased block scores
// to the cell indices the attention body may attend to.
//
// llama runs `ggml_top_k(expanded, width)` over per-cell scores and turns the
// result into a mask (`build_attn_qsa:659-683`: fill -INF, `ggml_set_rows`
// zeros into the selected rows, add the causal mask back). Read through the
// mask, that is just
//
//     visible(c)  ⟺  c ≤ pos  ∧  c ∈ top_k
//
// and because a block's cells all carry that block's score, the ranking is
// block-granular and the answer is a compact run per kept block. So the
// engine emits the selected cells as an ascending index list rather than a
// dense per-cell mask: the attention then gathers over `width` cells instead
// of walking n_kv, which is the entire point of the indexer, and the list is
// what `QSAIndexerRef.topKCells` returns.
//
// Shape of the answer. Write nvis = pos + 1 (cells up to and including the
// query), nc = nvis / r complete blocks, t = nvis % r the tail block's cells,
// target = min(n_kv, budget + r - 1) cells. The tail block scores +1e9, so it
// ranks first and is kept whole — it is the query's own block and an
// attention that cannot see its own token is not attention (when the budget
// cannot even hold the tail, target < t, the tail's first `target` cells are
// all that is kept and nothing is ranked). That leaves
//
//     K = target - min(t, target)   cells to take from the complete blocks
//     q = K / r                     complete blocks kept whole
//     m = K % r                     cells of the boundary block, the (q+1)-th
//
// so the selection is one rank: the (q+1)-th largest complete-block score is
// the boundary block, which contributes its lowest m cell indices — and only
// when m > 0, since m == 0 means the whole blocks land exactly on K and there
// is no partial block at all. Ties rank to the lower block index, and within a
// block to the lower cell index, which is what the reference's stable sort
// produces.
//
// Note the query's own cell is not forced visible in the m == 0 case: when
// pos ends a complete block there is no tail, that block carries the ordinary
// zero bias, and it competes with the rest. llama's mask has no diagonal term
// (`build_attn_qsa` combines `ggml_set_rows`-unmasked top_k with the plain
// causal mask and nothing else), so this is the model's own behaviour and not
// an oversight here.
//
// Finding that rank exactly is a 4-bit-digit MSB radix select over the block
// keys, one threadgroup scanning the score array eight times. A threshold
// computed any other way — a histogram, a sampled pivot, an approximated
// quantile — cannot distinguish the boundary block from the ones tied with
// it, and the difference is not cosmetic: it is a whole block of attended
// cells. The pass count is the reason this is a single threadgroup and a
// single dispatch: a device-wide histogram would need a dispatch per digit,
// and twelve QSA layers per token make dispatch count the budget that matters.
//
// The dense fast path (nvis ≤ target, i.e. context ≤ budget + r - 1 = 2050
// for the real geometry) emits the causal prefix and never reaches the radix
// select — at that size QSA is plain causal attention and the budget selects
// every visible cell (`qwen4exp.cpp:606-607`).
// ============================================================================

constant constexpr uint kIdxSelThreads = 1024;

// Monotone float → uint32 key: unsigned comparison of the key reproduces
// float ordering, so the radix select can treat the keys as integers. The
// flip is needed because the bias puts -INFINITY (llama's unusable-partial-
// pool arm) in the same array as the finite scores, and -inf's raw bit
// pattern is larger than any positive float's.
static inline uint idx_sel_key(float v) {
    const uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

[[kernel, max_total_threads_per_threadgroup(kIdxSelThreads)]]
void idx_select_cells(
    device const float* scores [[buffer(0)]],  // [n_blocks] biased, from idx_block_scores
    device       uint*  cells  [[buffer(1)]],  // [width] selected cells, ascending
    device       uint*  count  [[buffer(2)]],  // [1] how many were written
    constant     uint&  pos    [[buffer(3)]],
    constant     uint&  n_kv   [[buffer(4)]],
    constant     uint&  r      [[buffer(5)]],
    constant     uint&  budget [[buffer(6)]],
    uint tid        [[thread_position_in_threadgroup]],
    uint lsize      [[threads_per_threadgroup]],
    uint simd_lane  [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint tg_simd[32 * 16];              // per-SIMD-group digit histograms
    threadgroup uint tg_hist[16];                   // threadgroup digit histogram
    threadgroup uint tg_comm[2];                    // chosen digit, count above it
    threadgroup uint scan_a[kIdxSelThreads + 1];    // exclusive scan of strictly-greater counts
    threadgroup uint scan_e[kIdxSelThreads + 1];    // exclusive scan of equal-to-boundary counts

    const uint nvis   = min(pos + 1u, n_kv);
    const uint target = min(n_kv, budget + r - 1u);

    if (nvis <= target) {
        // Dense fast path: every causal cell is selected, so QSA degenerates
        // to plain causal attention and there is no ranking to do.
        for (uint i = tid; i < nvis; i += lsize) cells[i] = i;
        if (tid == 0) count[0] = nvis;
        return;
    }

    const uint nc = nvis / r;        // complete causal blocks
    const uint t  = nvis - nc * r;   // the tail block's causal cells (0 = last block is complete)
    const uint tailKeep = min(t, target);
    const uint K = target - tailKeep;
    const uint q = K / r;
    const uint m = K - q * r;

    // ---- rank select: the (q+1)-th largest complete-block key -------------
    // Guarded so a degenerate geometry (K == 0, or q >= nc) never reads a
    // rank that does not exist; the emit below then keeps whatever fits.
    uint tau = 0;
    const bool needRank = (K > 0u) && (q < nc);
    if (needRank) {
        uint hi = 0;          // decided high bits, right-aligned
        uint k  = q + 1u;     // remaining 1-based rank, counting down
        // The histogram fold is per SIMD group, so the cross-group sum spans
        // exactly the groups this dispatch has rather than a literal 32: a
        // pipeline narrower than kIdxSelThreads would otherwise read
        // uninitialised bins and take them for counts.
        const uint ngroups = lsize / 32u;
        for (uint pass = 0; pass < 8u; ++pass) {
            const uint shift = 28u - 4u * pass;
            uint c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0, c6 = 0, c7 = 0;
            uint c8 = 0, c9 = 0, ca = 0, cb = 0, cc = 0, cd = 0, ce = 0, cf = 0;
            for (uint b = tid; b < nc; b += lsize) {
                const uint key = idx_sel_key(scores[b]);
                // Only keys still matching the decided prefix can hold the
                // rank, so each pass narrows the field it scans.
                if ((ulong)key >> (shift + 4u) != (ulong)hi) continue;
                switch ((key >> shift) & 0xFu) {
                    case  0: c0++; break;  case  1: c1++; break;
                    case  2: c2++; break;  case  3: c3++; break;
                    case  4: c4++; break;  case  5: c5++; break;
                    case  6: c6++; break;  case  7: c7++; break;
                    case  8: c8++; break;  case  9: c9++; break;
                    case 10: ca++; break;  case 11: cb++; break;
                    case 12: cc++; break;  case 13: cd++; break;
                    case 14: ce++; break;  default: cf++; break;
                }
            }
            // Fold the 16 private counters down to one histogram: SIMD-group
            // sums first, then bin l summed over the groups by lane l of the
            // first group. No atomics, so the result does not depend on the
            // scheduler.
            //
            // Both barriers below sit at the same level for every thread. The
            // tempting nesting — fold, barrier, digit choice, all inside
            // `if (simd_group == 0)` — is NOT equivalent: that barrier is
            // reached by 32 of 1024 threads, which Metal leaves undefined, and
            // here it let thread 0 read `tg_hist` before lanes 0-15 wrote it.
            // The bins were then one pass stale, the digit choice followed a
            // stale count, and the select returned the 6th largest key for a
            // rank of 3 — deterministic, data-dependent and silent, and wrong
            // by whole blocks of attended cells.
            uint v[16] = { c0, c1, c2, c3, c4, c5, c6, c7,
                           c8, c9, ca, cb, cc, cd, ce, cf };
            for (uint d = 0; d < 16u; ++d) v[d] = simd_sum(v[d]);
            if (simd_lane == 0) {
                for (uint d = 0; d < 16u; ++d) tg_simd[simd_group * 16u + d] = v[d];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0 && simd_lane < 16u) {
                uint s = 0;
                for (uint g = 0; g < ngroups; ++g) s += tg_simd[g * 16u + simd_lane];
                tg_hist[simd_lane] = s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                // Digits count from the largest down: the rank lands in the
                // first digit whose running total reaches it.
                uint cum = 0;
                uint chosen = 0;
                for (int d = 15; d >= 0; --d) {
                    if (cum + tg_hist[d] >= k) { chosen = (uint)d; break; }
                    cum += tg_hist[d];
                }
                tg_comm[0] = chosen;
                tg_comm[1] = cum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k -= tg_comm[1];
            hi = (hi << 4) | tg_comm[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        tau = hi;
    }

    // ---- how many complete blocks strictly outrank the boundary -----------
    uint a = 0;
    if (needRank) {
        for (uint b = tid; b < nc; b += lsize) {
            if (idx_sel_key(scores[b]) > tau) a++;
        }
        a = simd_sum(a);
        if (simd_lane == 0) tg_simd[simd_group] = a;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint s = 0;
            for (uint g = 0; g < lsize / 32u; ++g) s += tg_simd[g];
            tg_comm[0] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a = tg_comm[0];
    }
    // Blocks tied with the boundary are taken in ascending block order until
    // the budget is full — the same tie rule the reference's stable sort has.
    // Strictly-greater blocks are all kept whole: a <= q by construction of
    // the rank, so they account for a*r <= q*r = K - m cells and never
    // overrun the budget.
    const uint need = (needRank && q > a) ? q - a : 0u;
    // Cells left once those whole blocks are placed. R = (q - a)*r + m, so the
    // tied blocks contribute `need` whole ones and then `m` cells of the one
    // after them — the boundary block, the only block the ranking cuts in
    // half. When the budget divides by r (m == 0) there is no boundary block
    // and the whole blocks land exactly on K.
    const uint boundaryCells = m;

    // ---- emit, ascending by cell index ------------------------------------
    // Threads own contiguous block ranges so the emitted runs stay ordered;
    // each thread needs to know how many cells the kept blocks above its range
    // contribute, which is the exclusive scan of the two counts it tallied —
    // plus the boundary's `m` when that block lies above the range, since it
    // is the one kept block that is not worth r cells.
    const uint chunk = (nc + lsize - 1u) / lsize;
    const uint lo = min(tid * chunk, nc);
    const uint hi_b = min(lo + chunk, nc);
    uint myA = 0, myE = 0;
    for (uint b = lo; b < hi_b; ++b) {
        const uint key = idx_sel_key(scores[b]);
        if (key > tau) myA++;
        else if (key == tau) myE++;
    }
    scan_a[tid] = myA;
    scan_e[tid] = myE;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = 1; off < lsize; off <<= 1u) {
        const uint va = (tid >= off) ? scan_a[tid - off] : 0u;
        const uint ve = (tid >= off) ? scan_e[tid - off] : 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scan_a[tid] += va;
        scan_e[tid] += ve;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint baseA = scan_a[tid] - myA;
    const uint baseE = scan_e[tid] - myE;

    // Cells the kept blocks above this range contribute: r for each kept-whole
    // block, m for the boundary. The boundary is the tied block with exactly
    // `need` tied blocks before it, so it lies above the range exactly when
    // more than `need` tied blocks do. Every kept block contributes a nonzero
    // run, so the run starts of the whole grid are a partition of [0, K).
    const uint wholeAbove = baseA + min(baseE, need);
    const uint emitted0 = min(r * wholeAbove + (baseE > need ? boundaryCells : 0u), K);
    uint emitted = emitted0;
    uint j = baseE;                       // boundary-tied blocks seen so far
    for (uint b = lo; b < hi_b; ++b) {
        const uint key = idx_sel_key(scores[b]);
        const bool tied = (key == tau);
        // The tied blocks up to `need` fill the budget whole; the one after
        // them is the boundary and contributes its lowest `boundaryCells` cell
        // indices, which is the per-cell cut the reference's ordering makes.
        const bool isBoundary = tied && (j == need) && boundaryCells > 0u;
        const bool keep = (key > tau) || (tied && j < need) || isBoundary;
        if (tied) j++;
        if (!keep) continue;
        // The run is the block's own contribution. Clamping to K instead would
        // give the boundary r cells rather than m and shift every later run,
        // and since the runs are written by different threads concurrently
        // that overlap is a race, not a harmless overwrite.
        const uint stop = emitted + (isBoundary ? boundaryCells : r);
        for (uint u = 0; emitted < stop; ++u) cells[emitted++] = b * r + u;
        if (emitted >= K) break;
    }
    // The tail block is kept whole and sits past every complete block, so its
    // cells close the list.
    for (uint u = tid; u < tailKeep; u += lsize) cells[K + u] = nc * r + u;
    if (tid == 0) count[0] = K + tailKeep;
}

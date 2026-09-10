import Foundation

/// FP32 reference for the Qwen 3.8 Flash-Next QSA indexer (the sparse-
/// attention selector on full-attention layers), grounded in
/// `archive/llama.cpp/src/models/qwen4exp.cpp` `build_qsa_top_k` (:477-616).
///
/// Real 3.8 geometry (`config.json` text_config):
///   idx_dim (indexer_head_dim)  = 128
///   n_idx_h (indexer_n_heads)   = 4     // query heads
///   kv heads                    = 1     // one key head; raw keys contiguous
///   r (compress_ratio)          = 4     // whole blocks of 4 tokens
///   budget (indexer_top_k)      = 2048  // whole blocks
///   n_rot                       = idx_dim * 0.25 = 32  // 16 rope pairs
///   theta                       = 1e7 (rope_parameters.rope_theta)
///   rms eps                     = 1e-6
///
/// The indexer scores whole blocks of `r` tokens with a single mean-pooled
/// key per block, and a query of `n_idx_h` heads. Each *cell* (token) carries
/// its block's **biased** score (`blockBias`: the block holding the query is
/// forced visible, the complete ones score on their pool alone), and
/// `ggml_top_k(expanded, min(n_kv, budget + r - 1))` keeps the top cells —
/// `budget` (`indexer_top_k`, real 2048) is in **tokens**, the `+ r - 1`
/// rounding to a whole block so a block is never split (qwen4exp.cpp:606-609).
/// The selected set is returned sorted by cell position.
///
/// Dense fast path: when the number of causally-visible cells is ≤ `budget +
/// r - 1` (always, for context ≤ ~budget), every visible cell is chosen and
/// QSA degenerates to plain causal self-attention. `topKCells` returns the
/// visible cells in position order in that regime; the scoring path is only
/// exercised when more cells are visible than the budget can select.
///
/// Every step is a deliberate scalar loop — a different op-tree from the
/// Metal kernels it validates (no SIMD group reductions, no fused dot),
/// which is what makes the kernel-vs-reference comparison meaningful.
public enum QSAIndexerRef {
    public static let rmsEps: Float = 1e-6

    // Real 3.8 constants (defaults for the toy tests; the runner passes the
    // same values from config so the reference and kernels share one source).
    public static let idxDim    = 128
    public static let nIdxHeads = 4
    public static let compressRatio = 4
    public static let budget    = 2048
    public static let nRot      = 32
    public static let ropeTheta: Float = 1.0e7

    // MARK: - block mean-pool

    /// Mean-pool one block of up to `r` raw keys (cell-major, `idxDim` each).
    /// `cells` lists the block's member token indices, in order; the last
    /// block may be incomplete (fewer than `r` members) — the mean is over
    /// the members actually present. Returns one `idxDim` vector.
    public static func meanPool(
        _ cells: [Int], raw: [Float], idxDim: Int
    ) -> [Float] {
        precondition(!cells.isEmpty)
        var out = [Float](repeating: 0, count: idxDim)
        for c in cells {
            let base = c * idxDim
            for i in 0..<idxDim { out[i] += raw[base + i] }
        }
        let inv = 1.0 / Float(cells.count)
        for i in 0..<idxDim { out[i] *= inv }
        return out
    }

    /// Mean-pool every block of the `[n_kv]` raw keys. Block `b` covers cells
    /// `[b*r, min((b+1)*r, n_kv))`. Returns `[n_blocks * idxDim]`, block-major.
    public static func meanPoolAll(
        _ raw: [Float], n_kv: Int, r: Int, idxDim: Int
    ) -> [Float] {
        precondition(raw.count == n_kv * idxDim)
        let n_blocks = (n_kv + r - 1) / r
        var out = [Float](repeating: 0, count: n_blocks * idxDim)
        for b in 0..<n_blocks {
            let lo = b * r
            let hi = min(lo + r, n_kv)
            let pooled = meanPool(Array(lo..<hi), raw: raw, idxDim: idxDim)
            for i in 0..<idxDim { out[b * idxDim + i] = pooled[i] }
        }
        return out
    }

    // MARK: - RMSNorm

    /// Plain RMSNorm over one `dim` vector, scaled by `gamma` (a plain
    /// multiplier — the indexer norms are never 1+w-baked; `qwen4exp.py`
    /// bakes `+1` only for the five zero-centred PLE/indexer-norm gammas, and
    /// those arrive here already baked in the repack).
    public static func rms(
        _ x: [Float], gamma: [Float], eps: Float = rmsEps
    ) -> [Float] {
        precondition(x.count == gamma.count)
        let n = x.count
        var sumSq: Float = 0
        for i in 0..<n { let v = x[i]; sumSq += v * v }
        let inv = 1.0 / (sumSq / Float(n) + eps).squareRoot()
        var y = [Float](repeating: 0, count: n)
        for i in 0..<n { y[i] = x[i] * inv * gamma[i] }
        return y
    }

    // MARK: - partial RoPE

    /// Partial-RoPE over the first `nRot` dims, HF **rotate_half** (half-split)
    /// convention: pair `i` mixes `(i, i + nRot/2)` with the shared frequency
    /// `theta^(-2i/nRot)` (denominator = rotary dim, NOT the full head dim),
    /// leaving dims ≥ `nRot` unchanged. This is the 3.6/3.8 text convention
    /// (mrope degenerates to this for a 1-D text position) and must match the
    /// cross-validated 3.6 `qwen_rope_pair` kernel exactly. `pos` is the
    /// token (or block) position.
    public static func rope(
        _ x: [Float], pos: Int, nRot: Int = Self.nRot,
        theta: Float = Self.ropeTheta
    ) -> [Float] {
        var y = x
        let half = nRot / 2
        let logTheta = logf(theta) / Float(nRot)
        for i in 0..<half {
            let a = x[i]
            let b = x[i + half]
            let freq = expf(-Float(2 * i) * logTheta)   // theta^(-2i/nRot)
            let ang = Float(pos) * freq
            let c = cosf(ang), s = sinf(ang)
            y[i]       = a * c - b * s
            y[i + half] = a * s + b * c
        }
        return y
    }

    /// Apply `rms` + `rope` to `perVector` vectors of length `dim`, each with
    /// its own position (block-major for the pooled keys, head-major for the
    /// query heads). Returns the same layout.
    public static func normRope(
        _ vectors: [Float], perVector: Int, gamma: [Float], pos: [Int],
        dim: Int, eps: Float = rmsEps
    ) -> [Float] {
        precondition(vectors.count == perVector * dim)
        precondition(gamma.count == dim)
        precondition(pos.count == perVector)
        var out = [Float](repeating: 0, count: perVector * dim)
        for v in 0..<perVector {
            let base = v * dim
            let one = (0..<dim).map { vectors[base + $0] }
            let normed = rms(one, gamma: gamma, eps: eps)
            let rotated = rope(normed, pos: pos[v])
            for i in 0..<dim { out[base + i] = rotated[i] }
        }
        return out
    }

    // MARK: - block scores

    /// Score one block against the query. `q` is `[nIdxHeads * idxDim]`
    /// (already norm+roped); `kBlock` is the `[idxDim]` norm+roped block key.
    /// `score = Σ_heads relu(dot(qHead, kBlock))` (qwen4exp.cpp:576-585) —
    /// relu is applied per head *before* the sum, not to the sum.
    public static func blockScore(
        q: [Float], kBlock: [Float], idxDim: Int, nIdxHeads: Int
    ) -> Float {
        precondition(q.count == nIdxHeads * idxDim)
        precondition(kBlock.count == idxDim)
        var score: Float = 0
        for h in 0..<nIdxHeads {
            let qb = h * idxDim
            var dot: Float = 0
            for i in 0..<idxDim { dot += q[qb + i] * kBlock[i] }
            if dot > 0 { score += dot }
        }
        return score
    }

    /// Score every block at the given position (no causal mask here — causality
    /// is applied per cell in `topKCells`, matching llama's per-cell KQ mask).
    /// Returns `[n_blocks]`.
    public static func blockScores(
        _ q: [Float], _ kPooledNormRope: [Float],
        n_kv: Int, r: Int, idxDim: Int, nIdxHeads: Int
    ) -> [Float] {
        let n_blocks = (n_kv + r - 1) / r
        var out = [Float](repeating: 0, count: n_blocks)
        for b in 0..<n_blocks {
            let kb = Array(kPooledNormRope[(b * idxDim)..<((b + 1) * idxDim)])
            out[b] = blockScore(q: q, kBlock: kb, idxDim: idxDim,
                                nIdxHeads: nIdxHeads)
        }
        return out
    }

    // MARK: - per-block visibility bias

    /// llama's per-block bias — the half of the QSA input that carries the
    /// whole-block rules (`llama-memory-hybrid-idx.cpp:438-449`, the `blk_bias`
    /// branch B chooses whenever a causal KQ mask is available,
    /// `qwen4exp.cpp:504-506`). Per-cell causality is *not* here; the mask
    /// carries it (`:439-440` "the caller adds the attention mask, which drops
    /// empty, foreign and future cells"):
    ///
    ///   tail_start = (pos + 1) / r * r              // :436, integer division
    ///   bias[b]    = b*r >= tail_start ? +1e9       // :445 force-visible
    ///              : filled[b] < r     ? -INFINITY  // unusable partial pool
    ///              : 0
    ///
    /// `tail_start` is the first cell past the query's own block, so it names
    /// exactly the blocks that cannot be pooled completely yet: the block
    /// holding `pos` (incomplete, and always visible — an attention that cannot
    /// see its own token is not attention) and anything behind it. +1e9 is
    /// finite on purpose, so a tail cell the causal mask also rejects sums to
    /// -INFINITY rather than to a NaN (`:444`).
    ///
    /// `filled[b]` counts the cells of block `b` present in the key timeline.
    /// The engine's timeline is contiguous from cell 0, so the only incomplete
    /// block is the last one, and `b*r >= tail_start` already covers it when
    /// the query sits in it — the -INFINITY arm is llama's *cache-hole* case
    /// (`:397-399`, `:421-428`: a partial block before the tail whose pool
    /// would be a mean over a hole). It is kept here for completeness and is
    /// unreachable for a timeline the engine holds.
    public static func blockBias(n_kv: Int, r: Int, pos: Int) -> [Float] {
        let n_blocks = (n_kv + r - 1) / r
        let tailStart = ((pos + 1) / r) * r
        var bias = [Float](repeating: 0, count: n_blocks)
        for b in 0..<n_blocks {
            if b * r >= tailStart {
                bias[b] = 1e9
            } else {
                let filled = min(r, max(0, n_kv - b * r))
                bias[b] = filled < r ? -.infinity : 0
            }
        }
        return bias
    }

    // MARK: - top-k cell selection

    /// Select the cells the dense attention body may attend to, exactly as
    /// llama `build_qsa_top_k` ranks them (`ggml_top_k(expanded, width)`):
    ///
    ///   width = min(n_kv, budget + r - 1)          // qwen4exp.cpp:607
    ///
    /// Each *cell* (token) carries its block's score (`expanded` = block score
    /// gathered per cell via `cell_blk`) **plus its block's bias**
    /// (`blockBias`), causality is a per-cell test (`cell ≤ pos`, llama's KQ
    /// mask), and the top-`width` cells are kept (ties → lower index,
    /// deterministic). `budget` (`indexer_top_k`, real 2048) is in **tokens**,
    /// and the `+ r - 1` rounds to a whole block so a block is never split
    /// (qwen4exp.cpp:606 "whole blocks plus the tail").
    ///
    /// The bias is what keeps the tail block in: without it a query whose own
    /// block scores badly would drop the token it is attending *from*, which is
    /// exactly the failure the reference must make impossible to reproduce.
    ///
    /// The returned set is already `selected ∩ causal` — the cells
    /// `build_attn_qsa:659-683` ends up making visible, since a selected cell
    /// the causal mask rejects is dropped there and cannot be returned here.
    ///
    /// Dense fast path: if the number of causally-visible cells ≤ width, every
    /// visible cell is kept and QSA == plain causal attention. This is the
    /// `context ≤ budget + r - 1` regime, where no scoring is needed.
    public static func topKCells(
        _ blockScores: [Float], pos: Int, n_kv: Int, r: Int, budget: Int
    ) -> [Int] {
        let n_blocks = (n_kv + r - 1) / r
        precondition(blockScores.count == n_blocks)
        let bias = blockBias(n_kv: n_kv, r: r, pos: pos)
        let width = min(n_kv, budget + r - 1)
        let visible = Array(0..<n_kv).filter { $0 <= pos }   // causal, ascending
        if visible.count <= width {
            return visible   // dense fast path: everything visible is kept
        }
        // Sparse path: rank visible cells by their block's biased score desc
        // (tie → lower cell index), keep the top `width`.
        var ranked = visible
        ranked.sort { a, b in
            let sa = blockScores[a / r] + bias[a / r]
            let sb = blockScores[b / r] + bias[b / r]
            if sa != sb { return sa > sb }
            return a < b
        }
        return Array(ranked.prefix(width)).sorted()
    }

    /// One full decode-step indexer pass at position `pos`, given the
    /// (already norm+roped) query and pooled keys. Returns the selected cell
    /// indices — the set the dense attention body may attend to.
    public static func select(
        q: [Float], kPooledNormRope: [Float],
        pos: Int, n_kv: Int,
        r: Int = compressRatio, idxDim: Int = Self.idxDim,
        nIdxHeads: Int = Self.nIdxHeads, budget: Int = Self.budget
    ) -> [Int] {
        let scores = blockScores(q, kPooledNormRope,
                                 n_kv: n_kv, r: r, idxDim: idxDim,
                                 nIdxHeads: nIdxHeads)
        return topKCells(scores, pos: pos, n_kv: n_kv, r: r, budget: budget)
    }
}

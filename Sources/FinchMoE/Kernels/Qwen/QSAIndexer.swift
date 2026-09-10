import Foundation
import Metal

/// Swift wrapper for the Qwen 3.8 Flash-Next QSA indexer kernels in
/// `Metal/Qwen/qsa_indexer.metal`:
///
///   idx_qk_post               q = rope(rmsnorm(idx_q, gamma), pos)
///                             k_raw[pos] = idx_k        (verbatim)
///   idx_block_pool_norm_rope  pooled[b] = rope(rmsnorm(mean of block b), b·r)
///   idx_block_scores          score[b] = Σ_h relu(q_h · pooled[b]) + blk_bias
///
/// The projections themselves are the engine's existing `dequant_int4_gemv`
/// path (one [640, 2560] row block per full layer), so they are not part of
/// this wrapper: it starts from the GEMV output `[(nHeads + kvHeads)·idxDim]`.
///
/// Storage follows the decode convention — FP16 activations, FP32 kernel
/// internals. `scores` is FP32 because the selection compares it against the
/// bias, and llama keeps its expanded scores in f32 (qwen4exp.cpp:597-600).
/// The indexer norm gammas are raw BF16 payloads carrying the repack's +1
/// bake, so they multiply as stored (no `1 + w` fold here).
///
/// Geometry (real Qwen 3.8 Flash-Next): idxDim 128, nHeads 4, kvHeads 1,
/// r 4, budget 2048, nRot 128·0.25 = 32, theta 1e7, eps 1e-6. The runner
/// passes every one of them from `ArchConfig`; the defaults below only make
/// the real geometry's values visible at the call site.
final class QSAIndexer {

    /// Partial-rope width (`partialRotaryFactor` × `indexerHeadDim`).
    static let defaultNRot = 32
    static let defaultTheta: Float = 1.0e7
    static let defaultEps: Float = 1e-6

    /// Threadgroup-memory cap on the score kernel's per-head slots; the
    /// indexer's head count is 4 and the kernel is dispatched one SIMD group
    /// per head.
    static let maxHeads = 8

    /// One thread per indexer dim in the post and pool kernels, so the head
    /// width may not exceed the threadgroup height (`kIdxThreads`). The real
    /// indexer head is 128.
    static let maxDim = 128

    private let psoQKPost: MTLComputePipelineState
    private let psoPool: MTLComputePipelineState
    private let psoScores: MTLComputePipelineState
    private let psoSelect: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoQKPost  = try context.pipeline("idx_qk_post")
        self.psoPool    = try context.pipeline("idx_block_pool_norm_rope")
        self.psoScores  = try context.pipeline("idx_block_scores")
        self.psoSelect  = try context.pipeline("idx_select_cells")
    }

    private static func width(_ pso: MTLComputePipelineState) -> Int {
        min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
    }

    /// Indexer q/k post-processing for one token.
    ///
    /// `qk` is the int4 GEMV output of `index_qk_proj` — rows `[0, nHeads)`
    /// are the query heads, rows `[nHeads, nHeads + kvHeads)` the shared key
    /// head(s). Query heads are RMS-normed with `qGamma` and partially
    /// rotated at `pos` into `qOut`; the key head is copied verbatim into the
    /// raw timeline at cell `pos`, because pooling precedes both the norm and
    /// the rotation (qwen4exp.cpp:533-538).
    func encodeQKPost(
        commandBuffer: MTLCommandBuffer,
        qk: MTLBuffer, qkOffset: Int = 0,
        qGamma: MTLBuffer, qGammaOffset: Int = 0,
        qOut: MTLBuffer, qOutOffset: Int = 0,
        kRaw: MTLBuffer, kRawOffset: Int = 0,
        pos: UInt32,
        nHeads: UInt32,
        idxDim: UInt32,
        nRot: UInt32 = UInt32(defaultNRot),
        theta: Float = defaultTheta,
        eps: Float = defaultEps
    ) {
        precondition(idxDim <= UInt32(Self.maxDim),
                     "indexer head dim \(idxDim) exceeds the kernels' one-thread-per-dim width (\(Self.maxDim))")
        precondition(nRot <= idxDim, "rotary width \(nRot) exceeds the head dim \(idxDim)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoQKPost)
        enc.setBuffer(qk,     offset: qkOffset,     index: 0)
        enc.setBuffer(qGamma, offset: qGammaOffset, index: 1)
        enc.setBuffer(qOut,   offset: qOutOffset,   index: 2)
        enc.setBuffer(kRaw,   offset: kRawOffset,   index: 3)
        var posVar = pos
        var nHeadsVar = nHeads
        var idxDimVar = idxDim
        var nRotVar = nRot
        var thetaVar = theta
        var epsVar = eps
        enc.setBytes(&posVar,    length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nHeadsVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&idxDimVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&nRotVar,   length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&thetaVar,  length: MemoryLayout<Float>.size,  index: 8)
        enc.setBytes(&epsVar,    length: MemoryLayout<Float>.size,  index: 9)
        // One threadgroup per query head, plus one for the shared key head.
        enc.dispatchThreadgroups(
            MTLSize(width: Int(nHeads) + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.width(psoQKPost), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Finalise `blockCount` whole blocks starting at `firstBlock`: pool the
    /// `r` raw keys, RMS with `kGamma`, partial rope at the block's first
    /// cell `b·r`. One threadgroup per block; nothing is carried between
    /// calls, so a block's pooled key is computed once, when it fills.
    ///
    /// Blocks are passed here only when complete — the mean divides by `r`.
    /// An incomplete block is exactly the one the score kernel force-visibles
    /// (its bias is +1e9), so its pooled key is never read.
    func encodeBlockPoolNormRope(
        commandBuffer: MTLCommandBuffer,
        kRaw: MTLBuffer, kRawOffset: Int = 0,
        kGamma: MTLBuffer, kGammaOffset: Int = 0,
        pooled: MTLBuffer, pooledOffset: Int = 0,
        firstBlock: UInt32,
        blockCount: UInt32,
        r: UInt32,
        idxDim: UInt32,
        nRot: UInt32 = UInt32(defaultNRot),
        theta: Float = defaultTheta,
        eps: Float = defaultEps
    ) {
        precondition(blockCount > 0, "pool dispatch needs at least one block")
        precondition(idxDim <= UInt32(Self.maxDim),
                     "indexer head dim \(idxDim) exceeds the kernels' one-thread-per-dim width (\(Self.maxDim))")
        precondition(nRot <= idxDim, "rotary width \(nRot) exceeds the head dim \(idxDim)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPool)
        enc.setBuffer(kRaw,   offset: kRawOffset,   index: 0)
        enc.setBuffer(kGamma, offset: kGammaOffset, index: 1)
        enc.setBuffer(pooled, offset: pooledOffset, index: 2)
        var firstVar = firstBlock
        var rVar = r
        var idxDimVar = idxDim
        var nRotVar = nRot
        var thetaVar = theta
        var epsVar = eps
        enc.setBytes(&firstVar,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rVar,      length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&idxDimVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nRotVar,   length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&thetaVar,  length: MemoryLayout<Float>.size,  index: 7)
        enc.setBytes(&epsVar,    length: MemoryLayout<Float>.size,  index: 8)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(blockCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.width(psoPool), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Score every block of the timeline against the query and fold in the
    /// per-block visibility bias:
    ///
    ///     score[b] = Σ_h relu(q[h] · pooled[b]) + bias(b, pos, n_kv, r)
    ///
    /// The block holding `pos` is force-visible while it is incomplete
    /// (`+1e9`) and is written without reading `pooled`; see
    /// `QSAIndexerRef.blockBias` for the arms and `qwen4exp.cpp:594-600` for
    /// why the bias lands on the block score. Per-cell causality is the
    /// caller's mask, not this kernel's.
    ///
    /// Grid = `n_blocks` threadgroups of `32 · nHeads` threads.
    func encodeBlockScores(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, qOffset: Int = 0,
        pooled: MTLBuffer, pooledOffset: Int = 0,
        scores: MTLBuffer, scoresOffset: Int = 0,
        nHeads: UInt32,
        idxDim: UInt32,
        r: UInt32,
        nKv: UInt32,
        pos: UInt32
    ) {
        precondition(nHeads > 0 && nHeads <= UInt32(Self.maxHeads),
                     "indexer heads \(nHeads) exceed the score kernel's \(Self.maxHeads)")
        precondition(idxDim % 32 == 0,
                     "indexer head dim must be a multiple of 32 for the per-head SIMD dot")
        precondition(r > 0 && nKv > 0, "indexer scoring needs r > 0 and n_kv > 0")
        let nBlocks = (Int(nKv) + Int(r) - 1) / Int(r)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoScores)
        enc.setBuffer(q,      offset: qOffset,      index: 0)
        enc.setBuffer(pooled, offset: pooledOffset, index: 1)
        enc.setBuffer(scores, offset: scoresOffset, index: 2)
        var nHeadsVar = nHeads
        var idxDimVar = idxDim
        var rVar = r
        var nKvVar = nKv
        var posVar = pos
        enc.setBytes(&nHeadsVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&idxDimVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&rVar,      length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nKvVar,    length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&posVar,    length: MemoryLayout<UInt32>.size, index: 7)
        enc.dispatchThreadgroups(
            MTLSize(width: nBlocks, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32 * Int(nHeads), height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Turn the biased block scores into the cell indices the attention body
    /// may attend to — the engine's `ggml_top_k` (`qwen4exp.cpp:606`) and the
    /// selection half of `build_attn_qsa`'s mask (`:659-683`).
    ///
    /// One threadgroup, one dispatch: a 4-bit-digit radix select finds the
    /// boundary block's score exactly, then the kept blocks' cells are written
    /// ascending. `cells` must hold `min(nKv, budget + r - 1)` entries, which
    /// is what `selectCapacity` returns; `count` receives how many were
    /// written. In the dense regime (`nKv <= budget + r - 1`) every causal
    /// cell is selected, so the list is just `0..<nKv` and nothing is ranked.
    ///
    /// The list is the answer because it doubles as the mask: selecting a cell
    /// is exactly zeroing its row in llama's `-INF`-filled mask, so an
    /// attention that walks this list sees `selected ∩ causal` and no more.
    func encodeSelectCells(
        commandBuffer: MTLCommandBuffer,
        scores: MTLBuffer, scoresOffset: Int = 0,
        cells: MTLBuffer, cellsOffset: Int = 0,
        count: MTLBuffer, countOffset: Int = 0,
        pos: UInt32,
        nKv: UInt32,
        r: UInt32,
        budget: UInt32
    ) {
        precondition(r > 0, "the block ratio must be positive")
        precondition(nKv > 0, "selection needs a non-empty timeline")
        precondition(pos < nKv, "the query sits past the end of the timeline")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSelect)
        enc.setBuffer(scores, offset: scoresOffset, index: 0)
        enc.setBuffer(cells,  offset: cellsOffset,  index: 1)
        enc.setBuffer(count,  offset: countOffset,  index: 2)
        var posVar = pos
        var nKvVar = nKv
        var rVar = r
        var budgetVar = budget
        enc.setBytes(&posVar,    length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&nKvVar,    length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&rVar,      length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&budgetVar, length: MemoryLayout<UInt32>.size, index: 6)
        // Deliberately one threadgroup: the radix select re-scans the score
        // array across its passes, so spreading it over a grid would need a
        // dispatch per digit, and twelve QSA layers per token make dispatch
        // count the budget that matters.
        enc.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(Self.selectThreads, psoSelect.maxTotalThreadsPerThreadgroup),
                height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Threads the select kernel is dispatched with (`kIdxSelThreads`).
    static let selectThreads = 1024

    /// Entries `cells` must hold: llama's `width = min(n_kv, top_k + r - 1)`
    /// (`qwen4exp.cpp:607`), the whole-block rounding of the token budget.
    static func selectCapacity(nKv: Int, r: Int, budget: Int) -> Int {
        min(nKv, budget + r - 1)
    }
}

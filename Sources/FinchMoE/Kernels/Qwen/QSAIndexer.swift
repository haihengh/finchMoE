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
/// r 4, budget 2048, nRot 64 (the model's 256·0.25 rope width, inherited),
/// theta 1e7, eps 1e-6. The runner
/// passes every one of them from `ArchConfig`; the defaults below only make
/// the real geometry's values visible at the call site.
final class QSAIndexer {

    /// Rotated leading dims of the indexer head — the *model's* rope width
    /// (`partialRotaryFactor` × `fullHeadDim` = 64), not the indexer's own
    /// head scaled. `build_qsa_top_k` passes the indexer's rope call the same
    /// `n_rot` as the layer's full attention (qwen4exp.cpp:563/572 vs
    /// :749/755), so the 128-dim indexer head rotates its first half.
    static let defaultNRot = 64
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
    /// mRoPE variants are compiled on the first multimodal turn, so a
    /// text-only 3.8 run never builds the lazy shader module.
    private let context: MetalContext
    private var mropeLibrary: MTLLibrary?
    private var psoQKPostMRoPE: MTLComputePipelineState?
    private var psoPoolMRoPE: MTLComputePipelineState?

    init(context: MetalContext) throws {
        self.context = context
        self.psoQKPost  = try context.pipeline("idx_qk_post")
        self.psoPool    = try context.pipeline("idx_block_pool_norm_rope")
        self.psoScores  = try context.pipeline("idx_block_scores")
        self.psoSelect  = try context.pipeline("idx_select_cells")
    }

    private func mropePipeline(named name: String,
                               cache: inout MTLComputePipelineState?) throws
        -> MTLComputePipelineState {
        if let cached = cache { return cached }
        let library: MTLLibrary
        if let existing = mropeLibrary {
            library = existing
        } else {
            library = try MetalContext.moduleLibrary(device: context.device,
                                                     module: "mrope")
            mropeLibrary = library
        }
        guard let function = library.makeFunction(name: name) else {
            throw MetalError.missingFunction(name)
        }
        let pipeline = try context.device.makeComputePipelineState(function: function)
        cache = pipeline
        return pipeline
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
    ///
    /// `kRaw` is the layer's raw timeline *from its start*, and `pos` is the
    /// token's absolute position — the kernel addresses the store as
    /// `k_raw + pos · idxDim`, so a caller that also binds `kRaw` at an offset
    /// would shift the store twice. This encoder used to take a `kRawOffset`,
    /// and the chunked prefill passed `pos · idxDim` into it: every prefill
    /// key landed at `2 · pos`, which mis-pools every block the prefill
    /// completes *and* writes past the end of the timeline once `pos` passes
    /// half the context length. The parameter is gone rather than defaulted,
    /// because a prefill chunk has no reason to want a base: the pool kernel
    /// indexes cells absolutely, and the decode path reuses the same slots.
    func encodeQKPost(
        commandBuffer: MTLCommandBuffer,
        qk: MTLBuffer, qkOffset: Int = 0,
        qGamma: MTLBuffer, qGammaOffset: Int = 0,
        qOut: MTLBuffer, qOutOffset: Int = 0,
        kRaw: MTLBuffer,
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
        // The store is the one write in this file that can leave its buffer: a
        // caller handing over a timeline shorter than `pos + 1` rows (a
        // chunk-local slice, say) would have the kernel write past its end,
        // where the driver cannot order that write against anything. Say the
        // bound here rather than leaving it to the kernel's arithmetic.
        precondition(kRaw.length >= Int(pos + 1) * Int(idxDim) * MemoryLayout<Float16>.stride,
                     "raw timeline is \(kRaw.length) bytes, too short for position \(pos) "
                     + "at idxDim \(idxDim)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoQKPost)
        enc.setBuffer(qk,     offset: qkOffset,     index: 0)
        enc.setBuffer(qGamma, offset: qGammaOffset, index: 1)
        enc.setBuffer(qOut,   offset: qOutOffset,   index: 2)
        enc.setBuffer(kRaw,   offset: 0,            index: 3)
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

    /// Multimodal variant of `encodeQKPost`: the raw key is stored at the KV
    /// position, while the query heads rotate at the token's mRoPE triple.
    func encodeQKPostMRoPE(
        commandBuffer: MTLCommandBuffer,
        qk: MTLBuffer, qkOffset: Int = 0,
        qGamma: MTLBuffer, qGammaOffset: Int = 0,
        qOut: MTLBuffer, qOutOffset: Int = 0,
        kRaw: MTLBuffer,
        storePos: UInt32,
        positionT: UInt32,
        positionH: UInt32,
        positionW: UInt32,
        sections: [Int],
        nHeads: UInt32,
        idxDim: UInt32,
        nRot: UInt32 = UInt32(defaultNRot),
        theta: Float = defaultTheta,
        eps: Float = defaultEps
    ) throws {
        precondition(idxDim <= UInt32(Self.maxDim),
                     "indexer head dim \(idxDim) exceeds the kernels' width")
        precondition(nRot <= idxDim, "rotary width \(nRot) exceeds the head dim \(idxDim)")
        precondition(sections.count >= 2, "mRoPE needs H and W sections")
        precondition(kRaw.length >= Int(storePos + 1) * Int(idxDim)
                        * MemoryLayout<Float16>.stride,
                     "raw timeline is too short for position \(storePos)")
        let pipeline = try mropePipeline(named: "idx_qk_post_mrope",
                                         cache: &psoQKPostMRoPE)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qk,     offset: qkOffset,     index: 0)
        enc.setBuffer(qGamma, offset: qGammaOffset, index: 1)
        enc.setBuffer(qOut,   offset: qOutOffset,   index: 2)
        enc.setBuffer(kRaw,   offset: 0,            index: 3)
        var storeVar = storePos
        var nHeadsVar = nHeads
        var idxDimVar = idxDim
        var nRotVar = nRot
        var thetaVar = theta
        var epsVar = eps
        var posTVar = positionT
        var posHVar = positionH
        var posWVar = positionW
        var sectionH = UInt32(sections[1])
        var sectionW = UInt32(sections[2])
        enc.setBytes(&storeVar, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&nHeadsVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&idxDimVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&nRotVar, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&thetaVar, length: MemoryLayout<Float>.size, index: 8)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 9)
        enc.setBytes(&posTVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&posHVar, length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&posWVar, length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&sectionH, length: MemoryLayout<UInt32>.size, index: 13)
        enc.setBytes(&sectionW, length: MemoryLayout<UInt32>.size, index: 14)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(nHeads) + 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.width(pipeline),
                                           height: 1, depth: 1))
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
    ///
    /// `kRaw` is the whole timeline, as in `encodeQKPost`: the kernel addresses
    /// block `b` as `k_raw + b·r·idxDim`, so a binding offset shifts every
    /// block by one. Neither encoder takes one now — the two cannot disagree
    /// about where the timeline starts.
    func encodeBlockPoolNormRope(
        commandBuffer: MTLCommandBuffer,
        kRaw: MTLBuffer,
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
        enc.setBuffer(kRaw,   offset: 0,            index: 0)
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

    /// Multimodal variant of `encodeBlockPoolNormRope`: each pooled block
    /// rotates at its first cell's mRoPE triple. `positions` packs three
    /// UInt32 per prompt cell (t, h, w), index 0 == position 0; cells past
    /// `promptLength` (decode) use `cell + decodeOffset` on all axes.
    func encodeBlockPoolNormRopeMRoPE(
        commandBuffer: MTLCommandBuffer,
        kRaw: MTLBuffer,
        kGamma: MTLBuffer, kGammaOffset: Int = 0,
        pooled: MTLBuffer, pooledOffset: Int = 0,
        positions: MTLBuffer,
        promptLength: Int,
        decodeOffset: Int,
        sections: [Int],
        firstBlock: UInt32,
        blockCount: UInt32,
        r: UInt32,
        idxDim: UInt32,
        nRot: UInt32 = UInt32(defaultNRot),
        theta: Float = defaultTheta,
        eps: Float = defaultEps
    ) throws {
        precondition(blockCount > 0, "pool dispatch needs at least one block")
        precondition(idxDim <= UInt32(Self.maxDim),
                     "indexer head dim \(idxDim) exceeds the kernels' width")
        precondition(nRot <= idxDim, "rotary width \(nRot) exceeds the head dim \(idxDim)")
        precondition(sections.count >= 2, "mRoPE needs H and W sections")
        precondition(promptLength >= 0)
        precondition(positions.length >= promptLength * 3 * MemoryLayout<UInt32>.stride,
                     "position timeline is too short for \(promptLength) cells")
        let pipeline = try mropePipeline(named: "idx_block_pool_norm_rope_mrope",
                                         cache: &psoPoolMRoPE)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(kRaw,   offset: 0,            index: 0)
        enc.setBuffer(kGamma, offset: kGammaOffset, index: 1)
        enc.setBuffer(pooled, offset: pooledOffset, index: 2)
        enc.setBuffer(positions, offset: 0,         index: 9)
        var firstVar = firstBlock
        var rVar = r
        var idxDimVar = idxDim
        var nRotVar = nRot
        var thetaVar = theta
        var epsVar = eps
        var promptVar = UInt32(promptLength)
        var offsetVar = Int32(clamping: decodeOffset)
        var sectionH = UInt32(sections[1])
        var sectionW = UInt32(sections[2])
        enc.setBytes(&firstVar,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rVar,      length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&idxDimVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&nRotVar,   length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&thetaVar,  length: MemoryLayout<Float>.size,  index: 7)
        enc.setBytes(&epsVar,    length: MemoryLayout<Float>.size,  index: 8)
        enc.setBytes(&promptVar, length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&offsetVar, length: MemoryLayout<Int32>.size,  index: 11)
        enc.setBytes(&sectionH,  length: MemoryLayout<UInt32>.size, index: 12)
        enc.setBytes(&sectionW,  length: MemoryLayout<UInt32>.size, index: 13)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(blockCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.width(pipeline),
                                           height: 1, depth: 1))
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

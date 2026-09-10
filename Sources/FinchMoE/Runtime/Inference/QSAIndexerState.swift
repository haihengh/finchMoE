import Foundation
import Metal

/// Per-full-layer storage for the Qwen 3.8 Flash-Next QSA indexer's decode
/// path, plus the small per-layer scratch the four kernels need.
///
/// The indexer is a second, much smaller attention: four query heads and one
/// shared key head of width 128, whose job is not to produce output but to
/// decide *which cells of the real KV cache the layer may read*. It keeps its
/// own key timeline (`rawKeys`), pooled into blocks of `r` (`pooled`), scored
/// against the query (`scores`), and reduced to an explicit cell list
/// (`cells`) that `Attention.encodeFullCells` consumes as its mask.
///
/// The two timelines have different lifetimes, which is why they are separate
/// buffers rather than one:
///
///   - `rawKeys[p]` is written once, verbatim, when cell `p` is appended, and
///     never rewritten. It is the state that must survive across steps.
///   - `pooled[b]` is written once, when block `b`'s last cell lands. Also
///     persistent, and derived from `rawKeys` — but only the *pooled* form is
///     ever scored, so the raw timeline is written for exactly one reason.
///
/// `scores` and `cells` are transient: rewritten from scratch every step by
/// `idx_block_scores` and `idx_select_cells`. `qkProj`/`qIdx` are the
/// indexer's projection output and post-processed query, likewise per step.
/// All four are per-layer rather than shared because a layer's indexer chain
/// and its attention run inside one command buffer, while the previous
/// layer's may still be in flight.
///
/// Sizing: `rawKeys` and `pooled` are bounded by `maxContext` — every complete
/// block is scored on every step (the selection then discards most of them),
/// so there is no window to shrink them to. `cells` is bounded by the
/// selection width, `min(n_kv, budget + r − 1)`, which for the real geometry
/// is 2051 and does not depend on `n_kv` once past it.
final class QSAIndexerState {

    /// One full-attention layer's indexer buffers and scratch.
    struct Layer {
        /// Raw indexer keys, `[maxContext, idxDim]` FP16 — cell `p` holds the
        /// key head at position `p`, unnormed and unrotated.
        let rawKeys: MTLBuffer
        /// Pooled block keys, `[maxBlocks, idxDim]` FP16 — the mean of a
        /// block's raw keys, then normed and rotated at the block's first cell.
        let pooled: MTLBuffer
        /// Biased block scores, `[maxBlocks]` FP32, rewritten each step.
        let scores: MTLBuffer
        /// Selected ascending cell indices, `[capacity]` UInt32.
        let cells: MTLBuffer
        /// `[1]` UInt32 — how many entries of `cells` are live.
        let cellCount: MTLBuffer
        /// `index_qk_proj` GEMV output, `[(nHeads + kvHeads) · idxDim]` FP16.
        let qkProj: MTLBuffer
        /// Post-processed indexer queries, `[nHeads · idxDim]` FP16.
        let qIdx: MTLBuffer
        /// Blocks of this layer pooled so far — bookkeeping, not state the
        /// kernels read. A block is pooled on the step that fills it, so in a
        /// straight decode this counts up by one every `r` tokens; a sequence
        /// that rewinds and re-pools a block just counts it again, which is
        /// why nothing asserts monotonicity here.
        var pooledBlocks: Int
    }

    /// One entry per full-attention layer, in layer order.
    private(set) var layers: [Layer]
    /// Model layer → index into `layers`, or -1 for layers with no indexer.
    let layerIndexByLayer: [Int]

    let numQHeads: Int
    let numKVHeads: Int
    let idxDim: Int
    let r: Int
    let budget: Int
    let nRot: Int
    let theta: Float
    let eps: Float
    /// `min(maxContext, budget + r − 1)` — the allocation size of `cells`,
    /// and the count the dense path writes.
    let capacity: Int
    /// `ceil(maxContext / r)` — the allocation size of `pooled`/`scores`.
    let maxBlocks: Int

    /// Build the indexer storage for a Qwen 3.8 install, or return nil when
    /// the family carries no indexer (every other family, and the guard the
    /// runner uses to keep the dense attention path).
    init?(device: MTLDevice, config: ArchConfig, maxContext: Int) throws {
        guard config.isQwen3_8, config.indexerHeadDim > 0,
              config.indexerNumHeads > 0, config.indexerCompressRatio > 0 else {
            return nil
        }
        let r = config.indexerCompressRatio
        let idxDim = config.indexerHeadDim
        let nHeads = config.indexerNumHeads
        let nKV = config.indexerKVHeads
        // A full layer with no indexer would silently fall back to dense
        // attention, so the two must agree about which layers are full.
        let fullLayers = (0..<config.numLayers).filter { config.fullAttentionLayerMask[$0] != 0 }
        guard !fullLayers.isEmpty else { return nil }

        let maxBlocks = (maxContext + r - 1) / r
        let capacity = min(maxContext, config.indexerBudget + r - 1)
        precondition(capacity > 0, "QSA indexer needs a positive selection width")
        precondition(idxDim <= QSAIndexer.maxDim,
                     "indexer head dim \(idxDim) exceeds the kernels' per-dim thread width")
        precondition(nHeads <= QSAIndexer.maxHeads,
                     "indexer heads \(nHeads) exceed the score kernel's cap")
        precondition(idxDim % 32 == 0,
                     "indexer head dim must be a multiple of 32 for the per-head SIMD dot")

        func half(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * MemoryLayout<Float16>.size,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        func word(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * MemoryLayout<UInt32>.size,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }
        func single(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * MemoryLayout<Float>.size,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            b.label = label
            return b
        }

        var built: [Layer] = []
        built.reserveCapacity(fullLayers.count)
        for L in fullLayers {
            built.append(Layer(
                rawKeys:   try half(maxContext * idxDim, "qsa.\(L).rawKeys"),
                pooled:    try half(maxBlocks * idxDim, "qsa.\(L).pooled"),
                scores:    try single(maxBlocks, "qsa.\(L).scores"),
                cells:     try word(capacity, "qsa.\(L).cells"),
                cellCount: try word(1, "qsa.\(L).cellCount"),
                qkProj:    try half((nHeads + nKV) * idxDim, "qsa.\(L).qkProj"),
                qIdx:      try half(nHeads * idxDim, "qsa.\(L).qIdx"),
                pooledBlocks: 0))
        }

        // The select kernel is the only writer of `cellCount`, and it does not
        // run while the context still sits inside the selection width. Zeroing
        // here makes that state say what it means: 0 is "no ranking happened",
        // never a count left over from an earlier sequence.
        for l in built {
            memset(l.cellCount.contents(), 0, l.cellCount.length)
        }

        var indexByLayer = [Int](repeating: -1, count: config.numLayers)
        for (i, L) in fullLayers.enumerated() { indexByLayer[L] = i }

        self.layers = built
        self.layerIndexByLayer = indexByLayer
        self.numQHeads = nHeads
        self.numKVHeads = nKV
        self.idxDim = idxDim
        self.r = r
        self.budget = config.indexerBudget
        // The indexer inherits the *model's* rotary width — it does not derive
        // one from its own head. `build_qsa_top_k` passes the same bare `n_rot`
        // to both of its rope calls as the layer's full attention does
        // (qwen4exp.cpp:563/572 vs :749/755), and `n_rot` is the context's
        // `hparams.n_rot(il)` = the GGUF's `rope.dimension_count`, which the
        // converter has already scaled by `partial_rotary_factor`
        // (llama-model.cpp:1335-1338, llama-hparams.cpp:85-91). For the real
        // 125B that is 0.25 × the 256-dim full head = 64, so the 128-dim
        // indexer head rotates its first half. Deriving from `idxDim` instead
        // would give 32 and quietly rotate half as much.
        self.nRot = Int(Double(config.fullHeadDim) * config.partialRotaryFactor)
        self.theta = Float(config.fullRopeTheta)
        self.eps = 1e-6
        self.capacity = capacity
        self.maxBlocks = maxBlocks
    }

    func index(ofLayer L: Int) -> Int? {
        guard L >= 0, L < layerIndexByLayer.count else { return nil }
        let i = layerIndexByLayer[L]
        return i >= 0 ? i : nil
    }

    /// Record that `delta` more blocks of layer `index` have been pooled.
    /// Pooling is idempotent — a block's pooled key is a function of its raw
    /// keys and its first cell — so this is bookkeeping the caller keeps on
    /// the way past, not state any kernel reads.
    func advancePooledBlocks(_ index: Int, by delta: Int = 1) {
        layers[index].pooledBlocks += delta
    }

    /// Drop the timelines — the next prefill re-seeds them.
    ///
    /// Only the *pooled* bookkeeping and the raw timeline need clearing: the
    /// caches are positional and every position is rewritten before it is
    /// read, so zeroing the payloads buys nothing, but `pooledBlocks` must go
    /// back to 0 or the next run would skip pooling the blocks it re-fills.
    func reset() {
        for i in layers.indices {
            layers[i].pooledBlocks = 0
            memset(layers[i].rawKeys.contents(), 0, layers[i].rawKeys.length)
            memset(layers[i].cellCount.contents(), 0, layers[i].cellCount.length)
        }
    }
}

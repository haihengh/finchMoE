import Foundation
import Metal

/// GPU-side per-row fingerprints of a prefill tensor, one 64-bit FNV-1a hash
/// per row, written into a side buffer the host reads once after the run.
///
/// This exists to reach below the chunk boundary. `prefillChunked` encodes a
/// chunk's rows into a single command buffer and returns with it in flight, so
/// a host read of an intermediate row would need a commit and a wait per chunk
/// — and the correlate under investigation is how long the prefill takes, so an
/// instrument that changes the timing changes its own subject. Hashing on the
/// GPU costs one small dispatch per (layer, stage) instead.
///
/// The algorithm and constants match the QSA selection fingerprint in
/// `FQ_QSA_DUMP` deliberately, so values from the two instruments are
/// comparable. See KV-15 in
/// `docs/experiments/summaries/05-attention-and-kv-cache.md`.
final class PrefillRowHash {
    /// The layer stages fingerprinted, in encoding order. These are the points
    /// `RealForwardRunner` already snapshots one row of for the toy replay
    /// tests, so a diff reads in the vocabulary those tests use.
    ///
    /// `in` at layer L agreeing while `in` at layer L+1 differs localizes the
    /// divergence to layer L; `attn` and `post` then split that layer into its
    /// attention block and its routed-expert tail.
    /// Stages 0-2 are the residual plane at three points of every layer; 3-6 are
    /// the attention block's own sub-stages, on full-attention layers only. They
    /// share one dimension so one dump and one diff cover both — a difference in
    /// `qkv` with none in `idxcells` puts the divergence in the indexer, and one
    /// in `in` at layer L puts it inside layer L-1.
    static let stageNames = ["in", "attn", "post", "qkv", "idxcells", "core", "oproj",
                             "krot", "vrot"]
    static var stageCount: Int { stageNames.count }

    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("row_hash_fnv1a_64")
    }

    /// Hash `rowCount` rows of `src`, each `rowBytes` long with
    /// `rowStrideBytes` between row starts, into `dst`, beginning at row
    /// `dstRowBase` of that stage's region.
    ///
    /// `dstRowBase` is the row's *position in the sequence*, not its index in
    /// the chunk: every chunk reuses the same plane rows, so a per-chunk index
    /// would let each chunk overwrite the previous one's fingerprints.
    ///
    /// `dstOffsetBytes` must be 256-byte aligned; the stage regions a caller
    /// writes into are sized to keep it that way (see `RowHashLayout`).
    func encode(commandBuffer: MTLCommandBuffer,
                src: MTLBuffer,
                srcOffsetBytes: Int = 0,
                dst: MTLBuffer,
                dstOffsetBytes: Int,
                dstRowBase: Int,
                rowCount: UInt32,
                rowStrideBytes: UInt32,
                rowBytes: UInt32) {
        guard rowCount > 0 else { return }
        precondition(rowStrideBytes >= rowBytes, "row stride is smaller than the row")
        precondition(dstOffsetBytes % 256 == 0,
                     "row-hash regions must stay 256-byte aligned")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(src, offset: srcOffsetBytes, index: 0)
        enc.setBuffer(dst, offset: dstOffsetBytes, index: 1)
        var bytes = rowBytes
        var stride = rowStrideBytes
        var rows = rowCount
        var rowBase = UInt32(dstRowBase)
        enc.setBytes(&bytes, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&stride, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&rows, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&rowBase, length: MemoryLayout<UInt32>.size, index: 5)
        let threads = min(pso.maxTotalThreadsPerThreadgroup, 256)
        let groups = (Int(rowCount) + threads - 1) / threads
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads,
                                                                height: 1, depth: 1))
        enc.endEncoding()
    }
}

/// Where a stage's hashes live in the side buffer.
///
/// Laid out `[layer][stage][row]` with `row` the position in the sequence, so a
/// dump walks the layers in order and a diff reports the first layer and the
/// first token that moved. A stride of `maxRows` per (layer, stage) keeps every
/// region 256-byte aligned as long as `maxRows` is a multiple of 32 — every
/// legal chunk size is a multiple of 32, and the row dimension is the context
/// length.
struct RowHashLayout {
    let maxRows: Int
    let layerCount: Int

    var stageBytes: Int { maxRows * MemoryLayout<UInt64>.stride }
    var layerBytes: Int { stageBytes * PrefillRowHash.stageCount }
    var totalBytes: Int { layerBytes * layerCount }

    func offsetBytes(layer: Int, stage: Int) -> Int {
        layer * layerBytes + stage * stageBytes
    }

    func elementIndex(layer: Int, stage: Int, row: Int) -> Int {
        (offsetBytes(layer: layer, stage: stage) + row * MemoryLayout<UInt64>.stride)
            / MemoryLayout<UInt64>.stride
    }
}

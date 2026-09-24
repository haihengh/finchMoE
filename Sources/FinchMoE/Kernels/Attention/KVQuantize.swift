import Foundation
import Metal

/// Encodes the fp16 → int8 KV row quantizer (`kv_quantize_int8_rows`).
///
/// The int8 KV store keeps one fp16 scale per 64-element block, so a row of
/// `numFullKVHeads * fullHeadDim` elements becomes `rowDim` int8 bytes plus
/// `rowDim / 64` fp16 scales. The attention kernels multiply each element by
/// its block scale on read; this is the only writer.
final class KVQuantize {
    /// Elements per scale block. A row may hold any number of them (8 for
    /// Qwen's 512-element full KV row, 16 for Gemma's 1024-element one).
    static let blockElements = 64
    static let threadsPerGroup = 256

    private let context: MetalContext
    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.context = context
        self.pso = try context.pipeline("kv_quantize_int8_rows")
    }

    /// Quantizes `rows` contiguous fp16 rows into int8 + block scales.
    ///
    /// - Parameters:
    ///   - rowDim: elements per row; must be a multiple of `blockElements`.
    ///   - destination/scales: shared row-major buffers; `rows * rowDim` bytes
    ///     and `rows * (rowDim / blockElements) * 2` bytes are written.
    func encode(commandBuffer: MTLCommandBuffer,
                source: MTLBuffer, sourceOffset: Int,
                rows: Int,
                destination: MTLBuffer, destinationOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                rowDim: Int) {
        precondition(rows > 0, "rows must be positive")
        precondition(rowDim > 0 && rowDim % Self.blockElements == 0,
                     "int8 KV quantizer expects a multiple of \(Self.blockElements) elements per row, got \(rowDim)")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(source, offset: sourceOffset, index: 0)
        enc.setBuffer(destination, offset: destinationOffset, index: 1)
        enc.setBuffer(scales, offset: scalesOffset, index: 2)
        var rd = UInt32(rowDim)
        var be = UInt32(Self.blockElements)
        enc.setBytes(&rd, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&be, length: MemoryLayout<UInt32>.size, index: 4)
        enc.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadsPerGroup,
                                           height: 1, depth: 1))
        enc.endEncoding()
    }
}

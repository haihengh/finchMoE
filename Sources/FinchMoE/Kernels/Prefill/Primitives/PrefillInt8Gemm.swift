import Foundation
import Metal

/// Batched int8 affine projection for the prefill path.
///
/// Why it exists: the linear-attention (GDN) projections are int8 on every
/// shipped Qwen 3.8 install, and the prefill fed them to `dequant_int8_gemv_simd`
/// once per token — one dispatch per token, each walking the whole weight matrix
/// again. The GDN split measured that at 11.9 s of a 29.5 s prefill across the
/// four projection stages, with both projection stages landing on ~150 GFLOP/s
/// and only 88 MB/s of effective weight traffic, which is the signature of a
/// kernel that gets no reuse across the batch.
///
/// This dispatches `prefill_dequant_int8_gemm_f16_block` instead: a threadgroup
/// tile of 64 output rows by 32 tokens, the weight tile dequantized once into
/// threadgroup memory per K-step and reused across the token dimension, with W,
/// X and Y all coalesced.
///
/// It covers the sites whose output row stride equals the row count (qkv, z and
/// the output projection — the three that carry the time). The small a/b
/// projections keep the repeated-GEMV path: their two vectors interleave in one
/// buffer with a doubled stride, which this kernel's store does not express, and
/// they are 2 x 32 rows of work.
final class PrefillInt8Gemm {
    static let rowsPerTile = 64
    static let tokensPerTile = 32
    /// The kernel indexes its threadgroup as `tid.y * 16 + tid.x`, so the
    /// dispatch below must match this shape exactly.
    static let threadgroupWidth = 16
    static let threadgroupHeight = 8

    private let pso: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.pso = try context.pipeline("prefill_dequant_int8_gemm_f16_block")
    }

    /// K must be a whole number of int8 groups; M and T are free, because the
    /// tile edges are masked in-kernel.
    static func supports(columns k: Int) -> Bool {
        k > 0 && k % 64 == 0
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int,
                scales: MTLBuffer, scalesOffset: Int,
                biases: MTLBuffer, biasesOffset: Int,
                x: MTLBuffer, xOffset: Int,
                y: MTLBuffer, yOffset: Int,
                tokens: Int, rows: Int, columns: Int) {
        guard tokens > 0, rows > 0, Self.supports(columns: columns) else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(y, offset: yOffset, index: 4)
        var t = UInt32(tokens)
        var m = UInt32(rows)
        var k = UInt32(columns)
        enc.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 7)
        let wide = (rows + Self.rowsPerTile - 1) / Self.rowsPerTile
        let tall = (tokens + Self.tokensPerTile - 1) / Self.tokensPerTile
        enc.dispatchThreadgroups(
            MTLSize(width: wide, height: tall, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth,
                                           height: Self.threadgroupHeight,
                                           depth: 1))
        enc.endEncoding()
    }
}

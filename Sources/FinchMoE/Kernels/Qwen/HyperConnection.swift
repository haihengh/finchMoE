import Foundation
import Metal

/// Swift wrapper for the Qwen 3.8 Flash-Next hyper-connection kernels in
/// `Metal/Qwen/hyper_connection.metal`:
///
///   hc_plane_init     plane = hc copies of the [D] hidden stream
///   hc_grouped_rms    per-stream RMS over the hc planes, BF16 gamma
///   hc_silu_scale     lo = silu(down·xn · 1/hc)        (÷hc before silu)
///   hc_gate_mul       gated = xn · sigmoid(up·lo)
///   hc_stream_mean    blockInput = mean over streams of gated
///   hc_combine        plane += blockOut · 2·sigmoid(inject · 1/hc)
///
/// Activation buffers are FP16 (decode convention); the grouped-RMS gamma is
/// raw BF16 (HC norms are never 1+w-baked). The two GEMVs feeding the silu/
/// sigmoid activation kernels (down [lowrank, hc·D] int4, up [hc·D, lowrank]
/// int4, block_inject rows [hc, hc·D] int4) are the engine's existing
/// `dequant_int4_gemv` path, so they are not part of this wrapper.
final class HyperConnection {

    private let psoPlaneInit: MTLComputePipelineState
    private let psoGroupedRMS: MTLComputePipelineState
    private let psoSiluScale: MTLComputePipelineState
    private let psoGateMul: MTLComputePipelineState
    private let psoStreamMean: MTLComputePipelineState
    private let psoCombine: MTLComputePipelineState
    // Chunked (prefill) forms — see the `_seq` note in `hyper_connection.metal`.
    private let psoSeqPlaneInit: MTLComputePipelineState
    private let psoSeqGroupedRMS: MTLComputePipelineState
    private let psoSeqStreamMean: MTLComputePipelineState
    private let psoSeqCombine: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoPlaneInit  = try context.pipeline("hc_plane_init")
        self.psoGroupedRMS = try context.pipeline("hc_grouped_rms")
        self.psoSiluScale  = try context.pipeline("hc_silu_scale")
        self.psoGateMul    = try context.pipeline("hc_gate_mul")
        self.psoStreamMean = try context.pipeline("hc_stream_mean")
        self.psoCombine    = try context.pipeline("hc_combine")
        self.psoSeqPlaneInit  = try context.pipeline("hc_seq_plane_init")
        self.psoSeqGroupedRMS = try context.pipeline("hc_seq_grouped_rms")
        self.psoSeqStreamMean = try context.pipeline("hc_seq_stream_mean")
        self.psoSeqCombine    = try context.pipeline("hc_seq_combine")
    }

    private static func width(_ pso: MTLComputePipelineState) -> Int {
        min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
    }

    /// `plane[c*D + i] = hidden[i]` for every stream c — the decode-step
    /// residual starts as hc identical copies of the [D] hidden state.
    func encodePlaneInit(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        plane: MTLBuffer, planeOffset: Int = 0,
        d: UInt32,
        hc: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPlaneInit)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(plane,  offset: planeOffset,  index: 1)
        var dVar = d
        var hcVar = hc
        enc.setBytes(&dVar,  length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 3)
        let w = Self.width(psoPlaneInit)
        enc.dispatchThreads(MTLSize(width: Int(d), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-stream grouped RMSNorm over the [hc*D] plane:
    /// `out[c*D+i] = x[c*D+i] * rsqrt(mean_sq(stream c) + eps) * gamma[c*D+i]`.
    /// One threadgroup per stream. Gamma is raw BF16 (plain scale, no 1+w
    /// fold). `x` may alias `out` — each stream's scale pass re-reads the
    /// element it writes after the reduction barrier.
    func encodeGroupedRMS(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        gamma: MTLBuffer, gammaOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        eps: Float
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGroupedRMS)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(gamma,  offset: gammaOffset,  index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        var dVar = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 4)
        // One 256-thread threadgroup per stream (threadgroup index = stream).
        enc.dispatchThreadgroups(MTLSize(width: Int(hc), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `lo[i] = silu(z[i] * invHc)` — the ÷hc sits before the silu.
    /// `z` is the down-projection GEMV output (raw dot), `lo` feeds the up
    /// GEMV. Pass `invHc = 1.0 / Float(streamCount)`.
    func encodeSiluScale(
        commandBuffer: MTLCommandBuffer,
        z: MTLBuffer, zOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        n: UInt32,
        invHc: Float
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSiluScale)
        enc.setBuffer(z,   offset: zOffset,   index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var nVar = n
        var invHcVar = invHc
        enc.setBytes(&nVar,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&invHcVar, length: MemoryLayout<Float>.size,  index: 3)
        let w = Self.width(psoSiluScale)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `gated[i] = xn[i] * sigmoid(z[i])` — the up-GEMV raw gate dot fused
    /// with the read gate and the xn product. `gated` then collapses to the
    /// block input via `encodeStreamMean`.
    func encodeGateMul(
        commandBuffer: MTLCommandBuffer,
        xn: MTLBuffer, xnOffset: Int = 0,
        z: MTLBuffer, zOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        n: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGateMul)
        enc.setBuffer(xn,  offset: xnOffset,  index: 0)
        enc.setBuffer(z,   offset: zOffset,   index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nVar = n
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 3)
        let w = Self.width(psoGateMul)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `out[i] = (1/hc) · Σ_c gated[c*D + i]` — the [hc*D] gated plane
    /// collapses to the [D] block input (and, at the root mixer, to the
    /// model output). Pass `invHc = 1.0 / Float(streamCount)`.
    func encodeStreamMean(
        commandBuffer: MTLCommandBuffer,
        gated: MTLBuffer, gatedOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        invHc: Float
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoStreamMean)
        enc.setBuffer(gated, offset: gatedOffset, index: 0)
        enc.setBuffer(out,   offset: outOffset,   index: 1)
        var dVar = d
        var hcVar = hc
        var invHcVar = invHc
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hcVar,    length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&invHcVar, length: MemoryLayout<Float>.size,  index: 4)
        let w = Self.width(psoStreamMean)
        enc.dispatchThreads(MTLSize(width: Int(d), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `plane[c*D + i] += blockOut[i] * 2·sigmoid(inject[c] * invHc)`, in
    /// place. `blockOut` is the one-stream-wide block output; `inject` is
    /// the [hc] per-stream scatter weight vector. One threadgroup per stream.
    func encodeCombine(
        commandBuffer: MTLCommandBuffer,
        plane: MTLBuffer, planeOffset: Int = 0,
        blockOut: MTLBuffer, blockOutOffset: Int = 0,
        inject: MTLBuffer, injectOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        invHc: Float
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoCombine)
        enc.setBuffer(plane,    offset: planeOffset,    index: 0)
        enc.setBuffer(blockOut, offset: blockOutOffset, index: 1)
        enc.setBuffer(inject,   offset: injectOffset,   index: 2)
        var dVar = d
        var invHcVar = invHc
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&invHcVar, length: MemoryLayout<Float>.size,  index: 4)
        enc.dispatchThreadgroups(MTLSize(width: Int(hc), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}

// MARK: - Chunked (prefill) forms

extension HyperConnection {

    private static let hcThreads = 256

    /// `plane[t][c*D + i] = hidden[t][i]` — the chunked plane seed: `t`
    /// identical stream copies of each of the `t` hidden rows. The chunked
    /// prefill twin of `encodePlaneInit`.
    func encodeSeqPlaneInit(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        plane: MTLBuffer, planeOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        tokens: UInt32
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqPlaneInit)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(plane,  offset: planeOffset,  index: 1)
        var dVar = d
        var hcVar = hc
        enc.setBytes(&dVar,  length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 3)
        let n = Int(tokens) * Int(d) * Int(hc)
        let w = Self.width(psoSeqPlaneInit)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-stream grouped RMSNorm over a chunk of planes. `x` and `out` are
    /// `[tokens][hc*D]` with the whole-plane gamma shared across tokens; token
    /// `t`'s rows are read and written at `t · hc · D`. `x` may alias `out`.
    func encodeSeqGroupedRMS(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer, xOffset: Int = 0,
        gamma: MTLBuffer, gammaOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        tokens: UInt32,
        eps: Float
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqGroupedRMS)
        enc.setBuffer(x,     offset: xOffset,     index: 0)
        enc.setBuffer(gamma, offset: gammaOffset, index: 1)
        enc.setBuffer(out,   offset: outOffset,   index: 2)
        var dVar = d
        var hcVar = hc
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar,  length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 5)
        // One threadgroup per (stream, token), flat: threadgroup `t·hc + c`.
        enc.dispatchThreadgroups(
            MTLSize(width: Int(tokens) * Int(hc), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.hcThreads, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `out[t][i] = (1/hc) · Σ_c gated[t][c*D + i]` — the chunked stream mean:
    /// each token's `[hc*D]` gated plane collapses to its own `[D]` block
    /// input. Pass `invHc = 1.0 / Float(streamCount)`.
    func encodeSeqStreamMean(
        commandBuffer: MTLCommandBuffer,
        gated: MTLBuffer, gatedOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        tokens: UInt32,
        invHc: Float
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqStreamMean)
        enc.setBuffer(gated, offset: gatedOffset, index: 0)
        enc.setBuffer(out,   offset: outOffset,   index: 1)
        var dVar = d
        var hcVar = hc
        var invHcVar = invHc
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&hcVar,    length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&invHcVar, length: MemoryLayout<Float>.size,  index: 4)
        let n = Int(tokens) * Int(d)
        let w = Self.width(psoSeqStreamMean)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `plane[t][c*D + i] += blockOut[t][i] · 2·sigmoid(inject[t][c] · invHc)`,
    /// in place. `blockOut` is `[tokens][D]`, `inject` is `[tokens][hc]` — the
    /// per-token scatter weights.
    func encodeSeqCombine(
        commandBuffer: MTLCommandBuffer,
        plane: MTLBuffer, planeOffset: Int = 0,
        blockOut: MTLBuffer, blockOutOffset: Int = 0,
        inject: MTLBuffer, injectOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        tokens: UInt32,
        invHc: Float
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqCombine)
        enc.setBuffer(plane,    offset: planeOffset,    index: 0)
        enc.setBuffer(blockOut, offset: blockOutOffset, index: 1)
        enc.setBuffer(inject,   offset: injectOffset,   index: 2)
        var dVar = d
        var hcVar = hc
        var invHcVar = invHc
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar,    length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&invHcVar, length: MemoryLayout<Float>.size,  index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(tokens) * Int(hc), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.hcThreads, height: 1, depth: 1))
        enc.endEncoding()
    }
}

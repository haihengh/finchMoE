import Foundation
import Metal

/// Swift wrapper for the Qwen 3.8 Flash-Next PLE n-gram head in
/// `Metal/Qwen/ple.metal`:
///
///   ple_gate          gate[c] = sigmoid(sgn(s)·√max(|s|,1e-6)),  s = (1/√D)·⟨k_c, q_c⟩
///   ple_gated_value   gated[c*D+d] = value[d] · gate[c]
///   ple_conv_update   dilated causal depthwise conv + silu, and the history roll
///   ple_plane_add     plane += gated + conv_out
///
/// The two projections and the three grouped norms around these kernels are
/// **not** here: the projections are the engine's int8 GEMV path, and each norm
/// is `hc_grouped_rms` (one stream's RMS under a whole-plane BF16 gamma — the
/// same operator, on the same layout). This wrapper is only the four steps that
/// have no existing equivalent.
///
/// The n-gram hash and the ≤16-row table gather are host-side by design, not by
/// omission: the table is 102.4 GB across the part files, so the rows are
/// chosen and pread'd on the CPU and this file starts from the gathered
/// `[nHeads · 160]` vector.
///
/// Buffer types matter at the seams: activations are FP16, but the gate is
/// FP32 — it is a multiplier derived from a dot product, and its `hc` values
/// cost nothing to keep exact.
final class PLE {

    private let psoGate: MTLComputePipelineState
    private let psoGatedValue: MTLComputePipelineState
    private let psoConvUpdate: MTLComputePipelineState
    private let psoPlaneAdd: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoGate       = try context.pipeline("ple_gate")
        self.psoGatedValue = try context.pipeline("ple_gated_value")
        self.psoConvUpdate = try context.pipeline("ple_conv_update")
        self.psoPlaneAdd   = try context.pipeline("ple_plane_add")
    }

    private static func width(_ pso: MTLComputePipelineState) -> Int {
        min(Int(pso.maxTotalThreadsPerThreadgroup), 256)
    }

    /// One threadgroup per stream; each reduces its own `d`-wide dot.
    /// `key` and `query` are both the *normed* `[hc*D]` planes. Pass
    /// `invSqrtD = 1.0 / Float(d).squareRoot()`.
    func encodeGate(
        commandBuffer: MTLCommandBuffer,
        key: MTLBuffer, keyOffset: Int = 0,
        query: MTLBuffer, queryOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        d: UInt32,
        invSqrtD: Float,
        hc: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGate)
        enc.setBuffer(key,   offset: keyOffset,   index: 0)
        enc.setBuffer(query, offset: queryOffset, index: 1)
        enc.setBuffer(gate,  offset: gateOffset,  index: 2)
        var dVar = d
        var invSqrtDVar = invSqrtD
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&invSqrtDVar, length: MemoryLayout<Float>.size, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: Int(hc), height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `gated[c*D + d] = value[d] * gate[c]` — the `[D]` value vector fans out
    /// across the `hc` streams, each scaled by its own gate.
    func encodeGatedValue(
        commandBuffer: MTLCommandBuffer,
        value: MTLBuffer, valueOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        gated: MTLBuffer, gatedOffset: Int = 0,
        d: UInt32,
        hc: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGatedValue)
        enc.setBuffer(value, offset: valueOffset, index: 0)
        enc.setBuffer(gate,  offset: gateOffset,  index: 1)
        enc.setBuffer(gated, offset: gatedOffset, index: 2)
        var dVar = d
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 3)
        let n = Int(d) * Int(hc)
        let w = Self.width(psoGatedValue)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Dilated causal depthwise conv (`kernel` taps, `dilation` apart, so the
    /// receptive field is `(kernel-1)·dilation` back) plus silu, **and** the
    /// history roll into `newState`.
    ///
    /// `state` is `[(kernel-1)·dilation, C]` row-major, oldest row first — the
    /// *normed gated value* of the preceding positions. At a sequence start
    /// pass the same buffer for `state` and `newState` zero-filled, or two
    /// distinct zeroed buffers; `newState` may alias `state` only if every
    /// thread reaches the shift (it does when `C` is a multiple of the
    /// threadgroup width, which the real `hc·D` always is).
    func encodeConvUpdate(
        commandBuffer: MTLCommandBuffer,
        weight: MTLBuffer, weightOffset: Int = 0,
        state: MTLBuffer, stateOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        newState: MTLBuffer, newStateOffset: Int = 0,
        c: UInt32,
        kernel: UInt32,
        dilation: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoConvUpdate)
        enc.setBuffer(weight,   offset: weightOffset,   index: 0)
        enc.setBuffer(state,    offset: stateOffset,    index: 1)
        enc.setBuffer(x,        offset: xOffset,        index: 2)
        enc.setBuffer(out,      offset: outOffset,      index: 3)
        enc.setBuffer(newState, offset: newStateOffset, index: 4)
        var cVar = c
        var kVar = kernel
        var dilVar = dilation
        enc.setBytes(&cVar,   length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&kVar,   length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&dilVar, length: MemoryLayout<UInt32>.size, index: 7)
        let w = Self.width(psoConvUpdate)
        let groups = (Int(c) + w - 1) / w
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `plane[i] += gated[i] + conv[i]`, in place — both PLE terms land in the
    /// layer's HC plane before its attention mixer.
    func encodePlaneAdd(
        commandBuffer: MTLCommandBuffer,
        plane: MTLBuffer, planeOffset: Int = 0,
        gated: MTLBuffer, gatedOffset: Int = 0,
        conv: MTLBuffer, convOffset: Int = 0,
        n: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPlaneAdd)
        enc.setBuffer(plane, offset: planeOffset, index: 0)
        enc.setBuffer(gated, offset: gatedOffset, index: 1)
        enc.setBuffer(conv,  offset: convOffset,  index: 2)
        var nVar = n
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 3)
        let w = Self.width(psoPlaneAdd)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }
}

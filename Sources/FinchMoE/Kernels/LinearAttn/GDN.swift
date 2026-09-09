import Foundation
import Metal

/// Swift wrapper for the GDN (gated-delta-net) decode unit: the causal-conv1d
/// update and the per-value-head recurrent step. Math pinned to `qwen3_5_moe`.
///
/// Buffer conventions (fp16 activations, fp32 state, per-head fp32 scalars):
///   q, k: `[numKeyHeads * headDim]`, v, out: `[numValueHeads * headDim]`
///   g, beta: `[numValueHeads]`
///   state: `[numValueHeads * headDim * headDim]` fp32, v-major
///   conv:  w `[C,4]`, state `[C,3]`, x/out `[C]`, newState `[C,3]`
final class GDN {

    /// The gated output norm's z-activation. Qwen 3.5/3.6 use silu; Qwen 3.8
    /// Flash-Next uses sigmoid (`qwen4exp build_norm_gated`) — the sole GDN
    /// numerical delta between the families.
    enum RMSNormGateActivation {
        case silu
        case sigmoid
    }

    /// Function-constant index selecting the sigmoid gate (see gdn.metal).
    /// 66 is free in the merged library — 60 is FC_ATTN_HEAD_DIM.
    private static let sigmoidGateConstantIndex = 66

    private let psoConv: MTLComputePipelineState
    private let psoRecurrent: MTLComputePipelineState
    private let psoGate: MTLComputePipelineState
    private let psoNormGated: MTLComputePipelineState
    private let psoNormGatedSigmoid: MTLComputePipelineState
    private let psoGateGEMV: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoConv      = try context.pipeline("gdn_conv_update")
        self.psoRecurrent = try context.pipeline("gdn_recurrent")
        self.psoGate      = try context.pipeline("gdn_gate")
        self.psoNormGated = try context.pipeline("gdn_rmsnorm_gated")
        self.psoNormGatedSigmoid = try context.pipeline(
            "gdn_rmsnorm_gated",
            constants: [MetalFunctionConstant(
                index: Self.sigmoidGateConstantIndex, value: .bool(true))])
        self.psoGateGEMV  = try context.pipeline("gdn_gate_gemv")
    }

    /// Fused in_proj_a/in_proj_b int4-affine GEMVs + gate formula.
    /// `weights` is [2V, N/2] nibbles (a-rows first, then b-rows), `scales`/
    /// `biases` [2V, N/64] BF16, `x` [N] fp16, `A_log`/`dt_bias` [V] fp32,
    /// `g`/`beta` [V] fp32 outputs. One 256-thread threadgroup.
    func encodeGateGEMV(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        A_log: MTLBuffer, A_logOffset: Int = 0,
        dt_bias: MTLBuffer, dt_biasOffset: Int = 0,
        g: MTLBuffer, gOffset: Int = 0,
        beta: MTLBuffer, betaOffset: Int = 0,
        numValueHeads: Int,
        n: UInt32
    ) {
        precondition(numValueHeads <= 32, "gdn_gate_gemv caps V at 32")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGateGEMV)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales,  offset: scalesOffset,  index: 1)
        enc.setBuffer(biases,  offset: biasesOffset,  index: 2)
        enc.setBuffer(x,       offset: xOffset,       index: 3)
        enc.setBuffer(A_log,   offset: A_logOffset,   index: 4)
        enc.setBuffer(dt_bias, offset: dt_biasOffset, index: 5)
        enc.setBuffer(g,       offset: gOffset,       index: 6)
        enc.setBuffer(beta,    offset: betaOffset,    index: 7)
        var vVar = UInt32(numValueHeads)
        var nVar = n
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 9)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Causal conv1d decode update over `channels` channels (kernel 4).
    func encodeCausalConvUpdate(
        commandBuffer: MTLCommandBuffer,
        w: MTLBuffer,      wOffset: Int = 0,
        state: MTLBuffer,  stateOffset: Int = 0,
        x: MTLBuffer,      xOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        newState: MTLBuffer, newStateOffset: Int = 0,
        channels: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoConv)
        enc.setBuffer(w,        offset: wOffset,      index: 0)
        enc.setBuffer(state,    offset: stateOffset,  index: 1)
        enc.setBuffer(x,        offset: xOffset,      index: 2)
        enc.setBuffer(out,      offset: outOffset,    index: 3)
        enc.setBuffer(newState, offset: newStateOffset, index: 4)
        var cVar = UInt32(channels)
        enc.setBytes(&cVar, length: MemoryLayout<UInt32>.size, index: 5)

        let width = Int(psoConv.maxTotalThreadsPerThreadgroup)
        let groups = (channels + width - 1) / width
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Recurrent gated-delta-rule decode step for all `numValueHeads`.
    /// `state` is mutated in place. `q`/`k` are key-head indexed (head `hv/2`).
    func encodeRecurrent(
        commandBuffer: MTLCommandBuffer,
        state: MTLBuffer,  stateOffset: Int = 0,
        q: MTLBuffer,      qOffset: Int = 0,
        k: MTLBuffer,      kOffset: Int = 0,
        v: MTLBuffer,      vOffset: Int = 0,
        g: MTLBuffer,      gOffset: Int = 0,
        beta: MTLBuffer,   betaOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        numValueHeads: Int,
        headDim: UInt32,
        scale: Float,
        l2eps: Float = 1e-6
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoRecurrent)
        enc.setBuffer(state, offset: stateOffset, index: 0)
        enc.setBuffer(q,     offset: qOffset,     index: 1)
        enc.setBuffer(k,     offset: kOffset,     index: 2)
        enc.setBuffer(v,     offset: vOffset,     index: 3)
        enc.setBuffer(g,     offset: gOffset,     index: 4)
        enc.setBuffer(beta,  offset: betaOffset,  index: 5)
        enc.setBuffer(out,   offset: outOffset,   index: 6)
        var dVar = headDim
        var scaleVar = scale
        var l2epsVar = l2eps
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size,  index: 8)
        enc.setBytes(&l2epsVar, length: MemoryLayout<Float>.size,  index: 9)

        let width = min(Int(psoRecurrent.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: numValueHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Per-value-head gate: `beta = sigmoid(b)`,
    /// `g = -exp(A_log) * softplus(a + dt_bias)`. `a/b/A_log/dt_bias` are
    /// `[numValueHeads]` fp32; `g`/`beta` are `[numValueHeads]` fp32 outputs.
    func encodeGate(
        commandBuffer: MTLCommandBuffer,
        a: MTLBuffer,      aOffset: Int = 0,
        b: MTLBuffer,      bOffset: Int = 0,
        A_log: MTLBuffer,  A_logOffset: Int = 0,
        dt_bias: MTLBuffer, dt_biasOffset: Int = 0,
        g: MTLBuffer,      gOffset: Int = 0,
        beta: MTLBuffer,   betaOffset: Int = 0,
        numValueHeads: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGate)
        enc.setBuffer(a,       offset: aOffset,       index: 0)
        enc.setBuffer(b,       offset: bOffset,       index: 1)
        enc.setBuffer(A_log,   offset: A_logOffset,   index: 2)
        enc.setBuffer(dt_bias, offset: dt_biasOffset, index: 3)
        enc.setBuffer(g,       offset: gOffset,       index: 4)
        enc.setBuffer(beta,    offset: betaOffset,    index: 5)
        var vVar = UInt32(numValueHeads)
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 6)

        let width = Int(psoGate.maxTotalThreadsPerThreadgroup)
        let groups = (numValueHeads + width - 1) / width
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Gated RMSNorm over each value head's vector:
    /// `y = x * rsqrt(mean(x^2) + eps) * weight * act(z)` with act = silu
    /// (`.silu`, 3.5/3.6) or sigmoid (`.sigmoid`, Qwen 3.8). `x`/`z` are
    /// `[numValueHeads * headDim]` fp16; `weight` is `[headDim]` bf16 shared
    /// across heads; `out` is `[numValueHeads * headDim]` fp16.
    func encodeRMSNormGated(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer,      xOffset: Int = 0,
        z: MTLBuffer,      zOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        numValueHeads: Int,
        headDim: UInt32,
        eps: Float = 1e-6,
        activation: RMSNormGateActivation = .silu
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(
            activation == .sigmoid ? psoNormGatedSigmoid : psoNormGated)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(z,      offset: zOffset,      index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var dVar = headDim
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 5)

        let width = min(Int(psoNormGated.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: numValueHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }
}

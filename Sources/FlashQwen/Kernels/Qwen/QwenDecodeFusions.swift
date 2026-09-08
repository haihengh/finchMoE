import Foundation
import Metal

/// Swift wrapper for the Qwen 3.6 (qwen3_5_moe) decode-layer fusions in
/// `Metal/Qwen/qwen_decode.metal`:
///
///   qwen_post_attn          hidden += attn; out = rmsnorm(hidden, w)
///   vec_add_fp16            a += b (tail combine: hidden += h2)
///   qwen_attn_output_gate   attn *= sigmoid(gate)   (full-attention output gate)
///   qwen_shared_gate        h1 *= sigmoid(dot(W, x)) (shared_expert_gate [1, N])
///   qwen_full_attn_epilogue q_norm/k_norm + partial RoPE + q|gate split
///
/// Norm weights are expected with the Qwen `(1 + w)` form already baked in by
/// the repack writer, so the kernels apply the weight directly.
final class QwenDecodeFusions {

    private let psoPostAttn: MTLComputePipelineState
    private let psoVecAdd: MTLComputePipelineState
    private let psoOutputGate: MTLComputePipelineState
    private let psoSharedGate: MTLComputePipelineState
    private let psoFullAttnEpilogue: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoPostAttn         = try context.pipeline("qwen_post_attn")
        self.psoVecAdd           = try context.pipeline("vec_add_fp16")
        self.psoOutputGate       = try context.pipeline("qwen_attn_output_gate")
        self.psoSharedGate       = try context.pipeline("qwen_shared_gate")
        self.psoFullAttnEpilogue = try context.pipeline("qwen_full_attn_epilogue")
    }

    /// `hidden[i] += attn[i]` (in place), then `out = rmsnorm(hidden) * weight`.
    /// `out` is the post_attention_layernorm output; it feeds both the shared
    /// expert and the router.
    func encodePostAttn(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer, hiddenOffset: Int = 0,
        attn: MTLBuffer, attnOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        d: UInt32,
        eps: Float = 1e-6
    ) {
        precondition(d <= 4096, "qwen_post_attn caps D at 4096")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoPostAttn)
        enc.setBuffer(hidden, offset: hiddenOffset, index: 0)
        enc.setBuffer(attn,   offset: attnOffset,   index: 1)
        enc.setBuffer(out,    offset: outOffset,    index: 2)
        enc.setBuffer(weight, offset: weightOffset, index: 3)
        var dVar = d
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 5)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Elementwise `a += b` in place.
    func encodeVecAdd(
        commandBuffer: MTLCommandBuffer,
        a: MTLBuffer, aOffset: Int = 0,
        b: MTLBuffer, bOffset: Int = 0,
        d: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoVecAdd)
        enc.setBuffer(a, offset: aOffset, index: 0)
        enc.setBuffer(b, offset: bOffset, index: 1)
        var dVar = d
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 2)
        let width = min(Int(psoVecAdd.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: Int(d), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Full-attention output gate: `attn[i] *= sigmoid(gate[i])` in place.
    func encodeAttnOutputGate(
        commandBuffer: MTLCommandBuffer,
        attn: MTLBuffer, attnOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        n: UInt32
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoOutputGate)
        enc.setBuffer(attn, offset: attnOffset, index: 0)
        enc.setBuffer(gate, offset: gateOffset, index: 1)
        var nVar = n
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 2)
        let width = min(Int(psoOutputGate.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: Int(n), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// shared_expert_gate: one-row int4-affine GEMV over `x`, sigmoid, then
    /// `h1[i] *= gate` in place. `weights` is [N/2] nibbles, `scales`/`biases`
    /// [N/64] BF16; `N % 64 == 0` required.
    func encodeSharedGate(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer, weightsOffset: Int = 0,
        scales: MTLBuffer, scalesOffset: Int = 0,
        biases: MTLBuffer, biasesOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        h1: MTLBuffer, h1Offset: Int = 0,
        n: UInt32,
        d: UInt32
    ) {
        precondition(n.isMultiple(of: UInt32(Quantization.groupSize)))
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSharedGate)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales,  offset: scalesOffset,  index: 1)
        enc.setBuffer(biases,  offset: biasesOffset,  index: 2)
        enc.setBuffer(x,       offset: xOffset,       index: 3)
        enc.setBuffer(h1,      offset: h1Offset,      index: 4)
        var nVar = n
        var dVar = d
        enc.setBytes(&nVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&dVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Qwen full-attention q/k epilogue. `qProj` is [numQHeads, 2*headDim]
    /// (q|gate pairs per head); `qOut`/`gateOut` are [numQHeads, headDim];
    /// `k` is [numKVHeads, headDim] normalized + partially rotated in place.
    /// RoPE rotates the first `rotaryDim` contiguous elements of each head.
    func encodeFullAttnEpilogue(
        commandBuffer: MTLCommandBuffer,
        qProj: MTLBuffer, qProjOffset: Int = 0,
        qOut: MTLBuffer, qOutOffset: Int = 0,
        gateOut: MTLBuffer, gateOutOffset: Int = 0,
        k: MTLBuffer, kOffset: Int = 0,
        qWeight: MTLBuffer, qWeightOffset: Int = 0,
        kWeight: MTLBuffer, kWeightOffset: Int = 0,
        headDim: UInt32,
        numQHeads: UInt32,
        numKVHeads: UInt32,
        position: UInt32,
        theta: Float,
        rotaryDim: UInt32,
        eps: Float = 1e-6
    ) {
        precondition(headDim <= 512, "qwen_full_attn_epilogue caps head_dim at 512")
        precondition(rotaryDim <= headDim && rotaryDim.isMultiple(of: 2),
                     "rotary_dim must be an even divisor bound of head_dim")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoFullAttnEpilogue)
        enc.setBuffer(qProj,   offset: qProjOffset,   index: 0)
        enc.setBuffer(qOut,    offset: qOutOffset,    index: 1)
        enc.setBuffer(gateOut, offset: gateOutOffset, index: 2)
        enc.setBuffer(k,       offset: kOffset,       index: 3)
        enc.setBuffer(qWeight, offset: qWeightOffset, index: 4)
        enc.setBuffer(kWeight, offset: kWeightOffset, index: 5)
        var hd = headDim
        var nq = numQHeads
        var nkv = numKVHeads
        var pos = position
        var th = theta
        var rd = rotaryDim
        var epsVar = eps
        enc.setBytes(&hd,   length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&nq,   length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&nkv,  length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&pos,  length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&th,   length: MemoryLayout<Float>.size,  index: 10)
        enc.setBytes(&rd,   length: MemoryLayout<UInt32>.size, index: 11)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size, index: 12)
        let groups = Int(numQHeads + numKVHeads)
        enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}

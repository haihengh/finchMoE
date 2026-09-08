import Foundation
import Metal

/// Swift wrapper for the GDN (gated-delta-net) prefill unit: the batched
/// causal-conv1d chunk, the per-value-head sequential recurrence over the
/// chunk's T tokens, the batched gate, and the batched gated RMSNorm. Math
/// pinned to `qwen3_5_moe`; each kernel bit-mirrors its decode counterpart in
/// `gdn.metal` / `GDN.swift` (see `docs/QWEN36_PORT.md`).
///
/// Buffer conventions (fp16 activations, fp32 state):
///   conv in/out: `[T][C]` fp16, the fused [q|k|v] block per token
///               (q at 0, k at keyDim, v at 2*keyDim elements)
///   conv w `[C,4]` fp16, conv state `[C,3]` fp16
///   z: `[T][valueDim]` fp16 (= [T][V][D]); out: `[T][valueDim]` fp16
///   ab: `[T][2V]` fp16 (a rows then b rows — the fused in_proj_a|b QMM out)
///   g, beta: `[T][V]` fp32
///   state: `[V][D][D]` fp32, v-major
final class GDNPrefill {

    private let psoConvChunk: MTLComputePipelineState
    private let psoRecurrentSeq: MTLComputePipelineState
    private let psoGate: MTLComputePipelineState
    private let psoNormGated: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoConvChunk     = try context.pipeline("prefill_gdn_conv_chunk")
        self.psoRecurrentSeq  = try context.pipeline("prefill_gdn_recurrent_seq")
        self.psoGate          = try context.pipeline("prefill_gdn_gate")
        self.psoNormGated     = try context.pipeline("prefill_gdn_rmsnorm_gated")
    }

    /// Batched causal conv1d over the chunk (kernel 4, silu). `x`/`out` are
    /// `[T][C]` fp16 and MUST be distinct buffers: taps at row t+3 read the
    /// raw input x[t] while row t writes out[t], so aliasing races across
    /// threads. `newState` is a separate `[C,3]` buffer (the kernel's state
    /// readers and state writers live in different threads of the same grid,
    /// so the post-chunk state must not alias the pre-chunk `state`); after
    /// encoding, it is blit-copied into the persistent `state`.
    func encodeConvChunk(
        commandBuffer: MTLCommandBuffer,
        w: MTLBuffer,      wOffset: Int = 0,
        state: MTLBuffer,  stateOffset: Int = 0,
        x: MTLBuffer,      xOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        newState: MTLBuffer, newStateOffset: Int = 0,
        channels: Int,
        tokens: Int
    ) {
        precondition(tokens > 0, "conv chunk requires at least one token")
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoConvChunk)
        enc.setBuffer(w,        offset: wOffset,       index: 0)
        enc.setBuffer(state,    offset: stateOffset,   index: 1)
        enc.setBuffer(x,        offset: xOffset,       index: 2)
        enc.setBuffer(out,      offset: outOffset,     index: 3)
        enc.setBuffer(newState, offset: newStateOffset, index: 4)
        var cVar = UInt32(channels)
        var tVar = UInt32(tokens)
        enc.setBytes(&cVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(
            MTLSize(width: channels, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        enc.endEncoding()

        // The post-chunk state lands in `newState`; carry it into the
        // persistent buffer. Same command buffer: ordered after the kernel.
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(from: newState, sourceOffset: newStateOffset,
                  to: state, destinationOffset: stateOffset,
                  size: channels * 3 * MemoryLayout<Float16>.stride)
        blit.endEncoding()
    }

    /// Sequential gated-delta-rule recurrence over the chunk's T tokens, one
    /// threadgroup per value head. `conv` is the fused [T][C] conv output
    /// (q at 0, k at `kOffset`, v at `vOffset` elements); `g`/`beta` are
    /// [T][V] fp32; `out` is [T][V][D] fp16; `state` is mutated in place.
    func encodeRecurrentSeq(
        commandBuffer: MTLCommandBuffer,
        state: MTLBuffer,  stateOffset: Int = 0,
        conv: MTLBuffer,   convOffset: Int = 0,
        g: MTLBuffer,      gOffset: Int = 0,
        beta: MTLBuffer,   betaOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        headDim: UInt32,
        channels: UInt32,          // conv row stride (elements)
        kOffset: UInt32,           // k block offset (elements)
        vOffset: UInt32,           // v block offset (elements)
        numValueHeads: Int,
        tokens: Int,
        scale: Float,
        l2eps: Float = 1e-6
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoRecurrentSeq)
        enc.setBuffer(state, offset: stateOffset, index: 0)
        enc.setBuffer(conv,  offset: convOffset,  index: 1)
        enc.setBuffer(g,     offset: gOffset,     index: 2)
        enc.setBuffer(beta,  offset: betaOffset,  index: 3)
        enc.setBuffer(out,   offset: outOffset,   index: 4)
        var dVar = headDim
        var cVar = channels
        var kOffVar = kOffset
        var vOffVar = vOffset
        var vVar = UInt32(numValueHeads)
        var tVar = UInt32(tokens)
        var scaleVar = scale
        var l2epsVar = l2eps
        enc.setBytes(&dVar,     length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&cVar,     length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&kOffVar,  length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&vOffVar,  length: MemoryLayout<UInt32>.size, index: 8)
        enc.setBytes(&vVar,     length: MemoryLayout<UInt32>.size, index: 9)
        enc.setBytes(&tVar,     length: MemoryLayout<UInt32>.size, index: 10)
        enc.setBytes(&scaleVar, length: MemoryLayout<Float>.size,  index: 11)
        enc.setBytes(&l2epsVar, length: MemoryLayout<Float>.size,  index: 12)

        let width = min(Int(psoRecurrentSeq.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: numValueHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Batched per-(token, value-head) gate: `beta = sigmoid(b)`,
    /// `g = -exp(A_log) * softplus(a + dt_bias)` from the fp16 QMM output
    /// `ab` ([T][2V], a rows then b rows). `A_log`/`dt_bias` are [V] fp32;
    /// `g`/`beta` are [T][V] fp32 outputs.
    func encodeGateBatch(
        commandBuffer: MTLCommandBuffer,
        ab: MTLBuffer,      abOffset: Int = 0,
        A_log: MTLBuffer,   A_logOffset: Int = 0,
        dt_bias: MTLBuffer, dt_biasOffset: Int = 0,
        g: MTLBuffer,       gOffset: Int = 0,
        beta: MTLBuffer,    betaOffset: Int = 0,
        numValueHeads: Int,
        tokens: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoGate)
        enc.setBuffer(ab,      offset: abOffset,      index: 0)
        enc.setBuffer(A_log,   offset: A_logOffset,   index: 1)
        enc.setBuffer(dt_bias, offset: dt_biasOffset, index: 2)
        enc.setBuffer(g,       offset: gOffset,       index: 3)
        enc.setBuffer(beta,    offset: betaOffset,    index: 4)
        var vVar = UInt32(numValueHeads)
        var tVar = UInt32(tokens)
        enc.setBytes(&vVar, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 6)
        enc.dispatchThreads(
            MTLSize(width: numValueHeads, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
        enc.endEncoding()
    }

    /// Batched gated RMSNorm over each (token, value head) vector:
    /// `y = x * rsqrt(mean(x^2) + eps) * weight * silu(z)`. `x`/`z`/`out` are
    /// [T][V][D] fp16 (out may alias x); `weight` is [D] bf16 shared across
    /// heads.
    func encodeRMSNormGatedBatch(
        commandBuffer: MTLCommandBuffer,
        x: MTLBuffer,      xOffset: Int = 0,
        z: MTLBuffer,      zOffset: Int = 0,
        weight: MTLBuffer, weightOffset: Int = 0,
        out: MTLBuffer,    outOffset: Int = 0,
        headDim: UInt32,
        numValueHeads: Int,
        tokens: Int,
        eps: Float = 1e-6
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoNormGated)
        enc.setBuffer(x,      offset: xOffset,      index: 0)
        enc.setBuffer(z,      offset: zOffset,      index: 1)
        enc.setBuffer(weight, offset: weightOffset, index: 2)
        enc.setBuffer(out,    offset: outOffset,    index: 3)
        var dVar = headDim
        var vVar = UInt32(numValueHeads)
        var tVar = UInt32(tokens)
        var epsVar = eps
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&vVar,   length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&tVar,   length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&epsVar, length: MemoryLayout<Float>.size,  index: 7)

        let width = min(Int(psoNormGated.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreadgroups(MTLSize(width: tokens * numValueHeads, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }
}

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

    private let psoConv: MTLComputePipelineState
    private let psoRecurrent: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoConv      = try context.pipeline("gdn_conv_update")
        self.psoRecurrent = try context.pipeline("gdn_recurrent")
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
}

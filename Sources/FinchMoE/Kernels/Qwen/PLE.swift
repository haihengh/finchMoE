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
    // Chunked (prefill) forms — see the note in `ple.metal`.
    private let psoSeqGate: MTLComputePipelineState
    private let psoSeqGatedValue: MTLComputePipelineState
    private let psoSeqConv: MTLComputePipelineState
    private let psoSeqConvRoll: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.psoGate       = try context.pipeline("ple_gate")
        self.psoGatedValue = try context.pipeline("ple_gated_value")
        self.psoConvUpdate = try context.pipeline("ple_conv_update")
        self.psoPlaneAdd   = try context.pipeline("ple_plane_add")
        self.psoSeqGate       = try context.pipeline("ple_seq_gate")
        self.psoSeqGatedValue = try context.pipeline("ple_seq_gated_value")
        self.psoSeqConv       = try context.pipeline("ple_seq_conv")
        self.psoSeqConvRoll   = try context.pipeline("ple_seq_conv_roll")
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
    /// history roll — on return `state` holds the rolled history and is ready
    /// for the next position.
    ///
    /// `state` is `[(kernel-1)·dilation, C]` row-major, oldest row first — the
    /// *normed gated value* of the preceding positions. At a sequence start
    /// pass a zero-filled buffer.
    ///
    /// The kernel writes the roll to `newState` rather than in place because
    /// its state readers and state writers live in different threads of the
    /// same grid: the shift is only alias-safe when every thread reaches it,
    /// which the divisible real geometry satisfies but the kernel does not
    /// promise. The write-back into `state` is therefore a blit encoded here,
    /// after the kernel — the same contract, and the same reason, as
    /// `GDNPrefill.encodeConvChunk`. Passing one buffer for both is allowed
    /// and needs no blit, but then the shift *is* in place and the divisibility
    /// caveat applies to the caller.
    ///
    /// Without this write-back the history never advances: every position
    /// would read the same stale rows, which is only the right answer while
    /// the history is still all zeros (positions `0 ..< (kernel-1)·dilation`).
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

        // The roll lands in `newState`; carry it into `state` so the next
        // position reads it. Same command buffer, so this is ordered after the
        // kernel. Aliased buffers need nothing — the roll was in place.
        if state !== newState {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            let bpr = Int(c) * MemoryLayout<Float16>.stride
            let rows = Int(kernel - 1) * Int(dilation)
            blit.copy(from: newState, sourceOffset: newStateOffset,
                      to: state, destinationOffset: stateOffset,
                      size: rows * bpr)
            blit.endEncoding()
        }
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

// MARK: - Chunked (prefill) forms

extension PLE {

    /// `ple_gate` over a chunk: `key`/`query` are `[tokens][hc*D]` normed
    /// planes and `gate` is `[tokens][hc]` FP32 — each token's streams gate
    /// against that token's own normed key/query pair. One threadgroup per
    /// (stream, token), flat: threadgroup `t·hc + c`.
    func encodeSeqGate(
        commandBuffer: MTLCommandBuffer,
        key: MTLBuffer, keyOffset: Int = 0,
        query: MTLBuffer, queryOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        d: UInt32,
        invSqrtD: Float,
        hc: UInt32,
        tokens: UInt32
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqGate)
        enc.setBuffer(key,   offset: keyOffset,   index: 0)
        enc.setBuffer(query, offset: queryOffset, index: 1)
        enc.setBuffer(gate,  offset: gateOffset,  index: 2)
        var dVar = d
        var hcVar = hc
        var invVar = invSqrtD
        enc.setBytes(&dVar,   length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar,  length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&invVar, length: MemoryLayout<Float>.size,  index: 5)
        enc.dispatchThreadgroups(
            MTLSize(width: Int(tokens) * Int(hc), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `ple_gated_value` over a chunk: `value` is `[tokens][D]`, `gate` is
    /// `[tokens][hc]`, `gated` is `[tokens][hc*D]`. Each token broadcasts its
    /// own value vector under its own per-stream gates.
    func encodeSeqGatedValue(
        commandBuffer: MTLCommandBuffer,
        value: MTLBuffer, valueOffset: Int = 0,
        gate: MTLBuffer, gateOffset: Int = 0,
        gated: MTLBuffer, gatedOffset: Int = 0,
        d: UInt32,
        hc: UInt32,
        tokens: UInt32
    ) {
        guard tokens > 0 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(psoSeqGatedValue)
        enc.setBuffer(value, offset: valueOffset, index: 0)
        enc.setBuffer(gate,  offset: gateOffset,  index: 1)
        enc.setBuffer(gated, offset: gatedOffset, index: 2)
        var dVar = d
        var hcVar = hc
        enc.setBytes(&dVar,  length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&hcVar, length: MemoryLayout<UInt32>.size, index: 4)
        let n = Int(tokens) * Int(hc) * Int(d)
        let w = Self.width(psoSeqGatedValue)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// The dilated causal conv over a whole chunk, plus the history roll.
    ///
    /// `state` is `[(kernel-1)·dilation, C]` fp16, oldest row first, holding
    /// the *normed gated value* of the positions preceding the chunk — the
    /// same buffer the decode path carries. On return it holds the chunk's own
    /// tail, ready for the next chunk. `x` and `out` are `[tokens][C]` and
    /// must be distinct (`ple_seq_conv` writes row `t` while later rows read
    /// it as a tap); `newState` must be distinct from `state`.
    ///
    /// The kernel can read a tap out of either the history or the chunk
    /// itself, so a chunk shorter than the receptive field is exact — every
    /// tap it cannot see inside the chunk comes from `state`.
    func encodeSeqConv(
        commandBuffer: MTLCommandBuffer,
        weight: MTLBuffer, weightOffset: Int = 0,
        state: MTLBuffer, stateOffset: Int = 0,
        x: MTLBuffer, xOffset: Int = 0,
        out: MTLBuffer, outOffset: Int = 0,
        newState: MTLBuffer, newStateOffset: Int = 0,
        c: UInt32,
        kernel: UInt32,
        dilation: UInt32,
        tokens: UInt32
    ) {
        guard tokens > 0 else { return }
        let hist = (Int(kernel) - 1) * Int(dilation)
        guard hist > 0 else { return }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(psoSeqConv)
            enc.setBuffer(weight, offset: weightOffset, index: 0)
            enc.setBuffer(state,  offset: stateOffset,  index: 1)
            enc.setBuffer(x,      offset: xOffset,      index: 2)
            enc.setBuffer(out,    offset: outOffset,    index: 3)
            var cVar = c
            var kVar = kernel
            var dilVar = dilation
            enc.setBytes(&cVar,   length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&kVar,   length: MemoryLayout<UInt32>.size, index: 5)
            enc.setBytes(&dilVar, length: MemoryLayout<UInt32>.size, index: 6)
            // Flat grid: `token·groups + channelGroup`, recovered in-kernel
            // from the threadgroup size.
            let w = Self.width(psoSeqConv)
            let groups = (Int(c) + w - 1) / w
            enc.dispatchThreadgroups(
                MTLSize(width: groups * Int(tokens), height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let enc = commandBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(psoSeqConvRoll)
            enc.setBuffer(state,    offset: stateOffset,    index: 0)
            enc.setBuffer(x,        offset: xOffset,        index: 1)
            enc.setBuffer(newState, offset: newStateOffset, index: 2)
            var cVar = c
            var tVar = tokens
            var hVar = UInt32(hist)
            enc.setBytes(&cVar, length: MemoryLayout<UInt32>.size, index: 3)
            enc.setBytes(&tVar, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&hVar, length: MemoryLayout<UInt32>.size, index: 5)
            let w = Self.width(psoSeqConvRoll)
            let n = hist * Int(c)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        // Carry the rolled history into the persistent buffer, after every
        // read of it in this chunk. Aliased buffers need nothing.
        if state !== newState {
            guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
            blit.copy(from: newState, sourceOffset: newStateOffset,
                      to: state, destinationOffset: stateOffset,
                      size: hist * Int(c) * MemoryLayout<Float16>.stride)
            blit.endEncoding()
        }
    }
}

import Foundation
import Metal

/// Compiles and caches `MTLComputePipelineState` objects for each kernel, and
/// provides typed dispatch helpers used by the forward pass.
public final class KernelPipelines {

    public enum Kernel: String, CaseIterable {
        case gemvInt4Q64        = "gemv_int4_q64"
        case rmsNorm            = "rms_norm"
        case ropeNeox           = "rope_neox"
        case gqaAttentionCausal = "gqa_attention_causal"
        case siluMul            = "silu_mul"
        case moeCombine         = "moe_combine"
        case sampleArgmax       = "sample_argmax"
        case sampleTopP         = "sample_top_p"
    }

    public let context: MetalContext
    private var cache: [String: MTLComputePipelineState] = [:]
    private let lock = NSLock()

    public init(context: MetalContext) throws {
        self.context = context
        // Eagerly build all pipelines to surface any compile errors up front.
        for k in Kernel.allCases {
            _ = try pipeline(k)
        }
    }

    public func pipeline(_ kernel: Kernel) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let pso = cache[kernel.rawValue] { return pso }
        guard let fn = context.library.makeFunction(name: kernel.rawValue) else {
            throw MetalContext.MetalError.libraryCompileFailed("missing function \(kernel.rawValue)")
        }
        let pso = try context.device.makeComputePipelineState(function: fn)
        cache[kernel.rawValue] = pso
        return pso
    }

    // MARK: - Dispatch helpers

    /// GEMV: int4 [rows, cols] × x[cols] -> out[rows].
    public func encodeGemvInt4(_ enc: MTLComputeCommandEncoder,
                               weights: MTLBuffer, weightsOffset: Int,
                               scales: MTLBuffer, scalesOffset: Int,
                               biases: MTLBuffer, biasesOffset: Int,
                               x: MTLBuffer, xOffset: Int,
                               out: MTLBuffer, outOffset: Int,
                               rows: Int, cols: Int) throws {
        let pso = try pipeline(.gemvInt4Q64)
        enc.setComputePipelineState(pso)
        enc.setBuffer(weights, offset: weightsOffset, index: 0)
        enc.setBuffer(scales, offset: scalesOffset, index: 1)
        enc.setBuffer(biases, offset: biasesOffset, index: 2)
        enc.setBuffer(x, offset: xOffset, index: 3)
        enc.setBuffer(out, offset: outOffset, index: 4)
        var r = UInt32(rows), c = UInt32(cols)
        enc.setBytes(&r, length: 4, index: 5)
        enc.setBytes(&c, length: 4, index: 6)
        dispatch1D(enc, pso: pso, count: rows)
    }

    /// RMSNorm over a vector of length n. Uses one threadgroup.
    public func encodeRMSNorm(_ enc: MTLComputeCommandEncoder,
                              x: MTLBuffer, xOffset: Int,
                              weight: MTLBuffer, weightOffset: Int,
                              out: MTLBuffer, outOffset: Int,
                              n: Int, eps: Float) throws {
        let pso = try pipeline(.rmsNorm)
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n), e = eps
        enc.setBytes(&nn, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        let threads = min(256, pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// NeoX RoPE in-place on [numHeads, headDim] for token at `pos`.
    public func encodeRoPE(_ enc: MTLComputeCommandEncoder,
                           x: MTLBuffer, xOffset: Int,
                           numHeads: Int, headDim: Int, theta: Float, pos: Int) throws {
        let pso = try pipeline(.ropeNeox)
        enc.setComputePipelineState(pso)
        enc.setBuffer(x, offset: xOffset, index: 0)
        var h = UInt32(numHeads), d = UInt32(headDim), t = theta, p = UInt32(pos)
        enc.setBytes(&h, length: 4, index: 1)
        enc.setBytes(&d, length: 4, index: 2)
        enc.setBytes(&t, length: 4, index: 3)
        enc.setBytes(&p, length: 4, index: 4)
        dispatch1D(enc, pso: pso, count: numHeads * (headDim / 2))
    }

    /// Causal GQA attention for a single query token.
    public func encodeAttention(_ enc: MTLComputeCommandEncoder,
                                q: MTLBuffer, qOffset: Int,
                                kcache: MTLBuffer, vcache: MTLBuffer,
                                out: MTLBuffer, outOffset: Int,
                                numQHeads: Int, numKVHeads: Int,
                                headDim: Int, length: Int) throws {
        let pso = try pipeline(.gqaAttentionCausal)
        enc.setComputePipelineState(pso)
        enc.setBuffer(q, offset: qOffset, index: 0)
        enc.setBuffer(kcache, offset: 0, index: 1)
        enc.setBuffer(vcache, offset: 0, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var qh = UInt32(numQHeads), kvh = UInt32(numKVHeads), d = UInt32(headDim), len = UInt32(length)
        enc.setBytes(&qh, length: 4, index: 4)
        enc.setBytes(&kvh, length: 4, index: 5)
        enc.setBytes(&d, length: 4, index: 6)
        enc.setBytes(&len, length: 4, index: 7)
        dispatch1D(enc, pso: pso, count: numQHeads)
    }

    /// SwiGLU: out = silu(gate) * up.
    public func encodeSiluMul(_ enc: MTLComputeCommandEncoder,
                              gate: MTLBuffer, gateOffset: Int,
                              up: MTLBuffer, upOffset: Int,
                              out: MTLBuffer, outOffset: Int,
                              n: Int) throws {
        let pso = try pipeline(.siluMul)
        enc.setComputePipelineState(pso)
        enc.setBuffer(gate, offset: gateOffset, index: 0)
        enc.setBuffer(up, offset: upOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var nn = UInt32(n)
        enc.setBytes(&nn, length: 4, index: 3)
        dispatch1D(enc, pso: pso, count: n)
    }

    /// out += weight * expert.
    public func encodeMoeCombine(_ enc: MTLComputeCommandEncoder,
                                 out: MTLBuffer, outOffset: Int,
                                 expert: MTLBuffer, expertOffset: Int,
                                 weight: Float, n: Int) throws {
        let pso = try pipeline(.moeCombine)
        enc.setComputePipelineState(pso)
        enc.setBuffer(out, offset: outOffset, index: 0)
        enc.setBuffer(expert, offset: expertOffset, index: 1)
        var w = weight, nn = UInt32(n)
        enc.setBytes(&w, length: 4, index: 2)
        enc.setBytes(&nn, length: 4, index: 3)
        dispatch1D(enc, pso: pso, count: n)
    }

    /// Greedy argmax over logits. Uses one threadgroup.
    public func encodeArgmax(_ enc: MTLComputeCommandEncoder,
                             logits: MTLBuffer, result: MTLBuffer, n: Int) throws {
        let pso = try pipeline(.sampleArgmax)
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(result, offset: 0, index: 1)
        var nn = UInt32(n)
        enc.setBytes(&nn, length: 4, index: 2)
        let threads = min(256, pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    /// Temperature + top-p nucleus sampling. Uses one threadgroup.
    public func encodeTopP(_ enc: MTLComputeCommandEncoder,
                           logits: MTLBuffer, result: MTLBuffer, n: Int,
                           temperature: Float, topP: Float, randomUniform: Float) throws {
        let pso = try pipeline(.sampleTopP)
        enc.setComputePipelineState(pso)
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(result, offset: 0, index: 1)
        var nn = UInt32(n), t = temperature, p = topP, r = randomUniform
        enc.setBytes(&nn, length: 4, index: 2)
        enc.setBytes(&t, length: 4, index: 3)
        enc.setBytes(&p, length: 4, index: 4)
        enc.setBytes(&r, length: 4, index: 5)
        let threads = min(256, pso.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    }

    // MARK: - Private

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState, count: Int) {
        let w = min(pso.maxTotalThreadsPerThreadgroup, 256)
        let tpg = MTLSize(width: w, height: 1, depth: 1)
        let groups = MTLSize(width: (count + w - 1) / w, height: 1, depth: 1)
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)
    }
}

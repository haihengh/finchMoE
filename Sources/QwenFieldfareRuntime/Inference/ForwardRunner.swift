import Foundation
import Metal
import QwenFieldfareFormat

/// Executes the Qwen3-30B-A3B forward pass for a single token using the Metal
/// kernels. Attention/FFN projections and expert matmuls run as int4 GEMV on
/// the GPU; the router and (optional) fp16 lm_head fall back to CPU math.
///
/// Prefill and decode both process one token per `step`, which keeps the code
/// path uniform and correct for arbitrary context lengths.
public final class ForwardRunner {

    public let model: Model
    private let cfg: QTurboModelConfig
    private let pipe: KernelPipelines
    private let device: MTLDevice
    private let queue: MTLCommandQueue

    // Scratch GPU buffers (allocated once, reused every step).
    private let bufHidden: MTLBuffer      // [hidden]
    private let bufNormed: MTLBuffer      // [hidden]
    private let bufQ: MTLBuffer           // [qDim]
    private let bufK: MTLBuffer           // [kvDim]
    private let bufV: MTLBuffer           // [kvDim]
    private let bufAttn: MTLBuffer        // [qDim]
    private let bufAttnProj: MTLBuffer    // [hidden]
    private let bufFfnNormed: MTLBuffer   // [hidden]
    private let bufSharedGate: MTLBuffer  // [sharedInter]
    private let bufSharedUp: MTLBuffer    // [sharedInter]
    private let bufSharedAct: MTLBuffer   // [sharedInter]
    private let bufSharedOut: MTLBuffer   // [hidden]
    private let bufExpertGate: MTLBuffer  // [moe]
    private let bufExpertUp: MTLBuffer    // [moe]
    private let bufExpertAct: MTLBuffer   // [moe]
    private let bufExpertOut: MTLBuffer   // [hidden]
    private let bufRoutedOut: MTLBuffer   // [hidden]
    private let bufExpertBlob: MTLBuffer  // [expert blob size]
    private let bufLogits: MTLBuffer      // [vocab]

    // Cached resident regions.
    private let embedEntry: QTurboTensorEntry
    private let lmHeadName = "lm_head.weight"
    private let finalNormName = "model.norm.weight"

    public init(model: Model) throws {
        self.model = model
        self.cfg = model.config
        self.pipe = model.pipelines
        self.device = model.metal.device
        self.queue = model.metal.commandQueue

        func mk(_ count: Int, _ label: String) throws -> MTLBuffer {
            guard let b = model.metal.makeBuffer(length: count * MemoryLayout<UInt16>.size, label: label) else {
                throw Model.ModelError.bufferCreationFailed
            }
            return b
        }

        let hidden = cfg.hiddenSize
        bufHidden     = try mk(hidden, "hidden")
        bufNormed     = try mk(hidden, "normed")
        bufQ          = try mk(cfg.qDim, "q")
        bufK          = try mk(cfg.kvDim, "k")
        bufV          = try mk(cfg.kvDim, "v")
        bufAttn       = try mk(cfg.qDim, "attn")
        bufAttnProj   = try mk(hidden, "attnProj")
        bufFfnNormed  = try mk(hidden, "ffnNormed")
        bufSharedGate = try mk(cfg.sharedExpertIntermediateSize, "sharedGate")
        bufSharedUp   = try mk(cfg.sharedExpertIntermediateSize, "sharedUp")
        bufSharedAct  = try mk(cfg.sharedExpertIntermediateSize, "sharedAct")
        bufSharedOut  = try mk(hidden, "sharedOut")
        bufExpertGate = try mk(cfg.moeIntermediateSize, "expertGate")
        bufExpertUp   = try mk(cfg.moeIntermediateSize, "expertUp")
        bufExpertAct  = try mk(cfg.moeIntermediateSize, "expertAct")
        bufExpertOut  = try mk(hidden, "expertOut")
        bufRoutedOut  = try mk(hidden, "routedOut")
        bufLogits     = try mk(cfg.vocabSize, "logits")

        guard let blob = model.metal.makeBuffer(length: model.manifest.expertLayout.blobSize,
                                                label: "expertBlob") else {
            throw Model.ModelError.bufferCreationFailed
        }
        bufExpertBlob = blob

        guard let embed = model.entry("model.embed_tokens.weight") else {
            throw Model.ModelError.tensorMissing("model.embed_tokens.weight")
        }
        self.embedEntry = embed
    }

    // MARK: - Public API

    /// Runs a full forward step for `token` at absolute `position`, updating the
    /// KV cache and leaving fp16 logits in the returned buffer.
    @discardableResult
    public func step(token: Int, position: Int) throws -> MTLBuffer {
        loadEmbedding(token: token, into: bufHidden)
        for layer in 0..<cfg.numHiddenLayers {
            try runLayer(layer, position: position)
        }
        try finalizeLogits()
        model.kvCache.advance(to: position)
        return bufLogits
    }

    // MARK: - Embedding

    private func loadEmbedding(token: Int, into out: MTLBuffer) {
        let hidden = cfg.hiddenSize
        let dst = out.contents()
        if embedEntry.dtype == .fp16 {
            let rowBytes = hidden * 2
            let src = model.residentPointer(offset: embedEntry.offset + token * rowBytes)
            memcpy(dst, src, rowBytes)
        } else {
            // Quantized embedding row → dequantize into fp16.
            dequantEmbeddingRow(token: token, into: dst)
        }
    }

    /// Dequantizes a single int4 embedding row into fp16 (used when the source
    /// model quantizes `embed_tokens`).
    private func dequantEmbeddingRow(token: Int, into dst: UnsafeMutableRawPointer) {
        let hidden = cfg.hiddenSize
        let group = cfg.quantGroupSize
        let groups = hidden / group
        let u32PerRow = hidden / 8
        guard let wEntry = model.entry("model.embed_tokens.weight"),
              let sEntry = model.entry("model.embed_tokens.scales"),
              let bEntry = model.entry("model.embed_tokens.biases") else {
            return
        }
        let w = model.residentPointer(offset: wEntry.offset + token * u32PerRow * 4)
            .bindMemory(to: UInt32.self, capacity: u32PerRow)
        let s = model.residentPointer(offset: sEntry.offset + token * groups * 2)
            .bindMemory(to: UInt16.self, capacity: groups)
        let b = model.residentPointer(offset: bEntry.offset + token * groups * 2)
            .bindMemory(to: UInt16.self, capacity: groups)
        let outHalf = dst.bindMemory(to: UInt16.self, capacity: hidden)
        for gI in 0..<groups {
            let scale = Sampler.halfToFloat(s[gI])
            let bias = Sampler.halfToFloat(b[gI])
            for j in 0..<8 {
                let packed = w[gI * 8 + j]
                for n in 0..<8 {
                    let nib = Int((packed >> (UInt32(n) * 4)) & 0xF)
                    let val = Float(nib - 8) * scale + bias
                    outHalf[gI * group + j * 8 + n] = floatToHalf(val)
                }
            }
        }
    }

    // MARK: - Layer

    private func runLayer(_ layer: Int, position: Int) throws {
        let p = model.layerPrefix(layer)
        let hidden = cfg.hiddenSize

        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw Model.ModelError.bufferCreationFailed
        }

        // (a) RMSNorm(hidden) -> normed
        try rms(enc, x: bufHidden, weightName: "\(p).input_layernorm.weight", out: bufNormed)

        // (b) Q/K/V projections (int4 GEMV).
        try gemvQuant(enc, weightBase: "\(p).self_attn.q_proj", x: bufNormed, out: bufQ,
                      rows: cfg.qDim, cols: hidden)
        try gemvQuant(enc, weightBase: "\(p).self_attn.k_proj", x: bufNormed, out: bufK,
                      rows: cfg.kvDim, cols: hidden)
        try gemvQuant(enc, weightBase: "\(p).self_attn.v_proj", x: bufNormed, out: bufV,
                      rows: cfg.kvDim, cols: hidden)

        // (c) RoPE on Q and K (NeoX, theta from config).
        try pipe.encodeRoPE(enc, x: bufQ, xOffset: 0,
                            numHeads: cfg.numAttentionHeads, headDim: cfg.headDim,
                            theta: cfg.ropeTheta, pos: position)
        try pipe.encodeRoPE(enc, x: bufK, xOffset: 0,
                            numHeads: cfg.numKeyValueHeads, headDim: cfg.headDim,
                            theta: cfg.ropeTheta, pos: position)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // (d) Append K,V to the KV cache at the current position.
        try model.kvCache.append(layer: layer, position: position,
                                 keys: bufK.contents(), values: bufV.contents())

        // (e) GQA causal attention over [0, position].
        let (kBuf, vBuf, length) = model.kvCache.slice(layer: layer, upTo: position + 1)
        guard let cmd2 = queue.makeCommandBuffer(), let enc2 = cmd2.makeComputeCommandEncoder() else {
            throw Model.ModelError.bufferCreationFailed
        }
        try pipe.encodeAttention(enc2, q: bufQ, qOffset: 0, kcache: kBuf, vcache: vBuf,
                                 out: bufAttn, outOffset: 0,
                                 numQHeads: cfg.numAttentionHeads, numKVHeads: cfg.numKeyValueHeads,
                                 headDim: cfg.headDim, length: length)

        // (f) o_proj(attn) -> attnProj
        try gemvQuant(enc2, weightBase: "\(p).self_attn.o_proj", x: bufAttn, out: bufAttnProj,
                      rows: hidden, cols: cfg.qDim)

        // (g) hidden = hidden + attnProj
        try pipe.encodeMoeCombine(enc2, out: bufHidden, outOffset: 0,
                                  expert: bufAttnProj, expertOffset: 0, weight: 1.0, n: hidden)

        // (h) RMSNorm(hidden) -> ffnNormed
        try rms(enc2, x: bufHidden, weightName: "\(p).post_attention_layernorm.weight", out: bufFfnNormed)

        // (i) shared_expert(ffnNormed) -> sharedOut
        try runSharedExpert(enc2, prefix: p)

        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        // (j) Router: top-8 experts + normalized weights (CPU).
        let routing = computeRouting(prefix: p)

        // Prefetch this layer's selected experts (and warm cache).
        model.streamer.prefetch(layer: layer, experts: routing.map { $0.expert })

        // (k) Zero routedOut, then accumulate top-8 expert outputs.
        zero(bufRoutedOut, count: hidden)
        for sel in routing {
            try runRoutedExpert(layer: layer, expert: sel.expert, weight: sel.weight)
        }

        // (l) hidden = hidden + sharedOut + routedOut
        guard let cmd3 = queue.makeCommandBuffer(), let enc3 = cmd3.makeComputeCommandEncoder() else {
            throw Model.ModelError.bufferCreationFailed
        }
        try pipe.encodeMoeCombine(enc3, out: bufHidden, outOffset: 0,
                                  expert: bufSharedOut, expertOffset: 0, weight: 1.0, n: hidden)
        try pipe.encodeMoeCombine(enc3, out: bufHidden, outOffset: 0,
                                  expert: bufRoutedOut, expertOffset: 0, weight: 1.0, n: hidden)
        enc3.endEncoding()
        cmd3.commit()
        cmd3.waitUntilCompleted()
    }

    // MARK: - Shared expert

    private func runSharedExpert(_ enc: MTLComputeCommandEncoder, prefix p: String) throws {
        let hidden = cfg.hiddenSize
        let inter = cfg.sharedExpertIntermediateSize
        try gemvQuant(enc, weightBase: "\(p).mlp.shared_expert.gate_proj", x: bufFfnNormed,
                      out: bufSharedGate, rows: inter, cols: hidden)
        try gemvQuant(enc, weightBase: "\(p).mlp.shared_expert.up_proj", x: bufFfnNormed,
                      out: bufSharedUp, rows: inter, cols: hidden)
        try pipe.encodeSiluMul(enc, gate: bufSharedGate, gateOffset: 0,
                               up: bufSharedUp, upOffset: 0, out: bufSharedAct, outOffset: 0, n: inter)
        try gemvQuant(enc, weightBase: "\(p).mlp.shared_expert.down_proj", x: bufSharedAct,
                      out: bufSharedOut, rows: hidden, cols: inter)
    }

    // MARK: - Routed expert

    private func runRoutedExpert(layer: Int, expert: Int, weight: Float) throws {
        let hidden = cfg.hiddenSize
        let moe = cfg.moeIntermediateSize
        let layout = model.manifest.expertLayout

        // Stream the expert blob and upload it to the GPU buffer.
        let blob = try model.streamer.load(layer: layer, expert: expert)
        memcpy(bufExpertBlob.contents(), blob.baseAddress!, min(blob.count, layout.blobSize))

        func sub(_ kind: String) -> QTurboExpertSubTensor { layout.subTensor(kind)! }

        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw Model.ModelError.bufferCreationFailed
        }

        // gate_proj: [moe, hidden]
        let gw = sub("gate_proj.weight"), gs = sub("gate_proj.scales"), gb = sub("gate_proj.biases")
        try pipe.encodeGemvInt4(enc,
            weights: bufExpertBlob, weightsOffset: gw.offset,
            scales: bufExpertBlob, scalesOffset: gs.offset,
            biases: bufExpertBlob, biasesOffset: gb.offset,
            x: bufFfnNormed, xOffset: 0, out: bufExpertGate, outOffset: 0,
            rows: moe, cols: hidden)

        // up_proj: [moe, hidden]
        let uw = sub("up_proj.weight"), us = sub("up_proj.scales"), ub = sub("up_proj.biases")
        try pipe.encodeGemvInt4(enc,
            weights: bufExpertBlob, weightsOffset: uw.offset,
            scales: bufExpertBlob, scalesOffset: us.offset,
            biases: bufExpertBlob, biasesOffset: ub.offset,
            x: bufFfnNormed, xOffset: 0, out: bufExpertUp, outOffset: 0,
            rows: moe, cols: hidden)

        // silu(gate) * up
        try pipe.encodeSiluMul(enc, gate: bufExpertGate, gateOffset: 0,
                               up: bufExpertUp, upOffset: 0, out: bufExpertAct, outOffset: 0, n: moe)

        // down_proj: [hidden, moe]
        let dw = sub("down_proj.weight"), ds = sub("down_proj.scales"), db = sub("down_proj.biases")
        try pipe.encodeGemvInt4(enc,
            weights: bufExpertBlob, weightsOffset: dw.offset,
            scales: bufExpertBlob, scalesOffset: ds.offset,
            biases: bufExpertBlob, biasesOffset: db.offset,
            x: bufExpertAct, xOffset: 0, out: bufExpertOut, outOffset: 0,
            rows: hidden, cols: moe)

        // routedOut += weight * expertOut
        try pipe.encodeMoeCombine(enc, out: bufRoutedOut, outOffset: 0,
                                  expert: bufExpertOut, expertOffset: 0, weight: weight, n: hidden)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
    }

    // MARK: - Router (CPU)

    private struct ExpertSelection { let expert: Int; let weight: Float }

    private func computeRouting(prefix p: String) -> [ExpertSelection] {
        let hidden = cfg.hiddenSize
        let numExperts = cfg.numExperts
        let topK = cfg.numExpertsPerTok

        guard let gate = model.entry("\(p).mlp.gate.weight") else { return [] }
        // gate.weight is fp16 [numExperts, hidden].
        let gPtr = model.residentPointer(offset: gate.offset).bindMemory(to: UInt16.self, capacity: numExperts * hidden)
        let xPtr = bufFfnNormed.contents().bindMemory(to: UInt16.self, capacity: hidden)

        var logits = [Float](repeating: 0, count: numExperts)
        for e in 0..<numExperts {
            var acc: Float = 0
            let base = e * hidden
            for i in 0..<hidden {
                acc += Sampler.halfToFloat(gPtr[base + i]) * Sampler.halfToFloat(xPtr[i])
            }
            logits[e] = acc
        }

        // Softmax over all experts.
        let maxL = logits.max() ?? 0
        var exps = logits.map { expf($0 - maxL) }
        let sum = exps.reduce(0, +)
        if sum > 0 { for i in 0..<numExperts { exps[i] /= sum } }

        // Top-k selection.
        let ordered = (0..<numExperts).sorted { exps[$0] > exps[$1] }
        let chosen = Array(ordered.prefix(topK))

        var weights = chosen.map { exps[$0] }
        if cfg.normTopkProb {
            let wsum = weights.reduce(0, +)
            if wsum > 0 { weights = weights.map { $0 / wsum } }
        }
        return zip(chosen, weights).map { ExpertSelection(expert: $0.0, weight: $0.1) }
    }

    // MARK: - Final logits

    private func finalizeLogits() throws {
        let hidden = cfg.hiddenSize
        // Final RMSNorm into normed.
        guard let cmd = queue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            throw Model.ModelError.bufferCreationFailed
        }
        try rms(enc, x: bufHidden, weightName: finalNormName, out: bufNormed)
        enc.endEncoding(); cmd.commit(); cmd.waitUntilCompleted()

        // lm_head: quantized int4 GEMV if scales present, else fp16 CPU matmul.
        if model.entry("\(lmHeadName)") != nil, model.entry("lm_head.scales") != nil {
            guard let cmd2 = queue.makeCommandBuffer(), let enc2 = cmd2.makeComputeCommandEncoder() else {
                throw Model.ModelError.bufferCreationFailed
            }
            try gemvQuant(enc2, weightBase: "lm_head", x: bufNormed, out: bufLogits,
                          rows: cfg.vocabSize, cols: hidden)
            enc2.endEncoding(); cmd2.commit(); cmd2.waitUntilCompleted()
        } else {
            lmHeadFP16CPU()
        }
    }

    /// fp16 lm_head matmul on the CPU (used when lm_head is not quantized).
    private func lmHeadFP16CPU() {
        let hidden = cfg.hiddenSize
        let vocab = cfg.vocabSize
        guard let lm = model.entry(lmHeadName) else { return }
        let wPtr = model.residentPointer(offset: lm.offset).bindMemory(to: UInt16.self, capacity: vocab * hidden)
        let xPtr = bufNormed.contents().bindMemory(to: UInt16.self, capacity: hidden)
        let outPtr = bufLogits.contents().bindMemory(to: UInt16.self, capacity: vocab)

        // Cache x as float.
        var xf = [Float](repeating: 0, count: hidden)
        for i in 0..<hidden { xf[i] = Sampler.halfToFloat(xPtr[i]) }

        DispatchQueue.concurrentPerform(iterations: vocab) { v in
            var acc: Float = 0
            let base = v * hidden
            for i in 0..<hidden { acc += Sampler.halfToFloat(wPtr[base + i]) * xf[i] }
            outPtr[v] = self.floatToHalf(acc)
        }
    }

    // MARK: - Helpers

    private func rms(_ enc: MTLComputeCommandEncoder, x: MTLBuffer, weightName: String, out: MTLBuffer) throws {
        let (wOff, _) = try model.region(weightName)
        try pipe.encodeRMSNorm(enc, x: x, xOffset: 0,
                               weight: model.residentBuffer, weightOffset: wOff,
                               out: out, outOffset: 0, n: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    /// Encodes an int4 GEMV for a resident quantized weight identified by
    /// `weightBase` (expects `.weight`, `.scales`, `.biases` companions).
    private func gemvQuant(_ enc: MTLComputeCommandEncoder, weightBase: String,
                           x: MTLBuffer, out: MTLBuffer, rows: Int, cols: Int) throws {
        let (wOff, _) = try model.region("\(weightBase).weight")
        let (sOff, _) = try model.region("\(weightBase).scales")
        let (bOff, _) = try model.region("\(weightBase).biases")
        try pipe.encodeGemvInt4(enc,
            weights: model.residentBuffer, weightsOffset: wOff,
            scales: model.residentBuffer, scalesOffset: sOff,
            biases: model.residentBuffer, biasesOffset: bOff,
            x: x, xOffset: 0, out: out, outOffset: 0,
            rows: rows, cols: cols)
    }

    private func zero(_ buf: MTLBuffer, count: Int) {
        memset(buf.contents(), 0, count * MemoryLayout<UInt16>.size)
    }

    @inline(__always)
    func floatToHalf(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        var exp = Int32((bits >> 23) & 0xFF) - 127 + 15
        let mant = bits & 0x7FFFFF
        if exp <= 0 {
            if exp < -10 { return sign } // too small → zero
            let m = (mant | 0x800000) >> UInt32(14 - exp)
            return sign | UInt16(m)
        } else if exp >= 0x1F {
            return sign | 0x7C00 // Inf
        }
        return sign | UInt16(exp << 10) | UInt16(mant >> 13)
    }
}

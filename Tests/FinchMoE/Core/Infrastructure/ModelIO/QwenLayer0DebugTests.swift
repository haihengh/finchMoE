import Testing
import Foundation
import Metal
import FinchMoEFormat
import FinchMoEValidationSupport
@testable import FinchMoE

/// Diagnostic gate on the REAL Qwen install: runs one decode step through
/// layer 0 (GDN) with the production kernels and compares the
/// post_attention_layernorm output against an fp32 reference of the same
/// chain built from the install's own (dequantized) weights. The layer input
/// is captured from the engine itself ("preLayer" hook), so the comparison
/// isolates exactly the layer-0 mixer: input norm → qkv/z/a/b projections →
/// causal conv → gate → recurrence → gated RMSNorm → out_proj → post-attn.
/// Skipped when the install is not present.
@Suite struct QwenLayer0DebugTests {

    private static let installPath =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-4bit.finch"

    private static var installExists: Bool {
        FileManager.default.fileExists(atPath: installPath + "/manifest.json")
    }

    private static let bf16Dir =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-bf16"

    private static var bf16DirExists: Bool {
        FileManager.default.fileExists(atPath: bf16Dir + "/config.json")
    }

    // MARK: - Dequant helpers (TensorView → fp32 rows)

    private static func int4Rows(_ view: TensorView, rows: Int, cols: Int) -> [[Float]] {
        let base = view.buffer.contents()
        let wBytes = base.advanced(by: Int(view.offset))
        let sWords = base.advanced(by: Int(view.scaleOffset))
            .assumingMemoryBound(to: UInt16.self)
        let bWords = base.advanced(by: Int(view.biasOffset))
            .assumingMemoryBound(to: UInt16.self)
        let groups = cols / 64
        var out = [[Float]](repeating: [], count: rows)
        for r in 0..<rows {
            var row = [Float](repeating: 0, count: cols)
            for g in 0..<groups {
                let scale = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                let byteBase = r * (cols / 2) + g * 32
                for k in 0..<64 {
                    let byte = wBytes.load(fromByteOffset: byteBase + k / 2,
                                           as: UInt8.self)
                    let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                    row[g * 64 + k] = Float(nibble) * scale + bias
                }
            }
            out[r] = row
        }
        return out
    }

    private static func int8Rows(_ view: TensorView, rows: Int, cols: Int) -> [[Float]] {
        let base = view.buffer.contents()
        let wBytes = base.advanced(by: Int(view.offset))
        let sWords = base.advanced(by: Int(view.scaleOffset))
            .assumingMemoryBound(to: UInt16.self)
        let bWords = base.advanced(by: Int(view.biasOffset))
            .assumingMemoryBound(to: UInt16.self)
        let groups = cols / 64
        var out = [[Float]](repeating: [], count: rows)
        for r in 0..<rows {
            var row = [Float](repeating: 0, count: cols)
            for g in 0..<groups {
                let scale = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                for k in 0..<64 {
                    let q = wBytes.load(fromByteOffset: r * cols + g * 64 + k,
                                        as: UInt8.self)
                    row[g * 64 + k] = Float(q) * scale + bias
                }
            }
            out[r] = row
        }
        return out
    }

    private static func bf16Values(_ view: TensorView, count: Int) -> [Float] {
        let words = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: UInt16.self)
        return (0..<count).map { FinchQuantization.bf16ToFloat(words[$0]) }
    }

    private static func fp16Values(_ view: TensorView, count: Int) -> [Float] {
        let halves = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(halves[$0]) }
    }

    private static func fp32Values(_ view: TensorView, count: Int) -> [Float] {
        let words = view.buffer.contents().advanced(by: Int(view.offset))
            .assumingMemoryBound(to: Float.self)
        return (0..<count).map { words[$0] }
    }

    private static func gemv(_ w: [[Float]], _ x: [Float]) -> [Float] {
        w.map { row in
            var acc: Float = 0
            for i in 0..<x.count { acc += row[i] * x[i] }
            return acc
        }
    }

    private static func rms(_ x: [Float], weight: [Float]) -> [Float] {
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1.0 / (ss / Float(x.count) + 1e-6).squareRoot()
        return zip(x, weight).map { $0 * inv * $1 }
    }

    private static func f16(_ x: [Float]) -> [Float] { x.map { Float(Float16($0)) } }

    /// FP16 → FP32 (the bitPattern<<16 trick only works for BF16).
    private static func toF32(_ h: Float16) -> Float {
        let bits = UInt32(h.bitPattern)
        let sign = (bits >> 15) & 1
        let exp = Int((bits >> 10) & 0x1F)
        let mant = bits & 0x3FF
        let m: Float
        let e: Float
        if exp == 0 {
            m = Float(mant) / 1024.0
            e = 1.0 / 16384.0   // 2^-14 · 2^-10
        } else {
            m = 1.0 + Float(mant) / 1024.0
            e = Float(1 << exp) / 32768.0   // 2^(exp-15), no negative shift
        }
        return (sign == 1 ? -1 : 1) * m * e
    }

    // MARK: - Test

    @Test(.enabled(if: installExists))
    func kernelsMatchReferenceOnRealWeights() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)

        let D = 2048
        // Fixed pseudo-random input (deterministic, fp16).
        var rng: UInt64 = 0xABC123
        var xF32 = [Float](repeating: 0, count: D)
        for i in 0..<D {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            xF32[i] = Float(rng >> 40) / Float(UInt64(1) << 24) - 0.5
        }
        let x16 = xF32.map { Float16($0) }
        let xRef = x16.map { Float($0) }

        // -- input norm: kernel vs reference.
        let inNormView = try model.inputNorm(layer: 0)
        let wF32 = Self.bf16Values(inNormView, count: D)
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = ctx.device.makeBuffer(
                  bytes: x16, length: D * 2, options: .storageModeShared),
              let oBuf = ctx.device.makeBuffer(
                  length: D * 2, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBF16W(commandBuffer: cb, x: xBuf,
                           weight: inNormView.buffer,
                           weightOffset: Int(inNormView.offset),
                           out: oBuf, d: UInt32(D), eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()
        let got = (0..<D).map { oBuf.contents().advanced(by: $0 * 2)
            .load(as: Float16.self) }
        let refNorm = Self.f16(Self.rms(xRef, weight: wF32))
        var maxAbs: Float = 0
        for i in 0..<D {
            maxAbs = max(maxAbs, abs(Float(got[i]) - refNorm[i]))
        }
        print("NORM: maxAbs=\(maxAbs) got[1875]=\(got[1875]) ref[1875]=\(refNorm[1875])")

        // -- int4 GEMV (in_proj_a rows): kernel vs reference.
        let aView = try model.gdnInProjA(layer: 0)
        let gemv = try DequantInt4GEMV(context: ctx)
        let rows = 32
        guard let yBuf = ctx.device.makeBuffer(
                  length: rows * 2, options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb2 = ctx.queue.makeCommandBuffer()!
        gemv.encode(commandBuffer: cb2,
                    weights: aView.buffer, weightsOffset: Int(aView.offset),
                    scales: aView.buffer, scalesOffset: Int(aView.scaleOffset),
                    biases: aView.buffer, biasesOffset: Int(aView.biasOffset),
                    x: xBuf, y: yBuf, m: UInt32(rows), n: UInt32(D))
        cb2.commit(); cb2.waitUntilCompleted()
        let gotRows = (0..<rows).map { yBuf.contents().advanced(by: $0 * 2)
            .load(as: Float16.self) }
        let refRows = Self.gemv(Self.int4Rows(aView, rows: rows, cols: D), xRef)
        var maxRow: Float = 0
        for i in 0..<rows {
            maxRow = max(maxRow, abs(Float(gotRows[i]) - refRows[i]))
        }
        print("GEMV(a): maxAbs=\(maxRow) got[0]=\(gotRows[0]) ref[0]=\(refRows[0])")
        #expect(maxAbs < 0.02, "norm kernel diverges: \(maxAbs)")
        #expect(maxRow < 0.02, "gemv kernel diverges: \(maxRow)")
    }

    @Test(.enabled(if: installExists))
    func layer0PostAttnMatchesFp32Reference() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)

        // Capture the layer-0 snapshots (decode + prefill).
        var phases: [String: [Float16]] = [:]
        runner.qwenLayerDebugHook = { layer, phase, values in
            phases["\(phase).\(layer)"] = values
        }

        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The", addBOS: false)
        let nextIds = tokenizer.encode(" capital", addBOS: false)

        // Reference: embedding row for the first prompt token (dequant × √D).
        let embedView = model.embedding
        let tokenID = Int(promptIds[0])
        let embedRow = Self.int4Rows(embedView, rows: tokenID + 1,
                                     cols: model.config.hiddenSize)[tokenID]
        let embedRef = Self.f16(embedRow)   // Qwen does not scale embeddings

        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        // Prefill one token first, so the prefill path is exercised and the
        // prefill hook snapshots (prefillHidden0/prefillDense0) are captured.
        try await runner.prefillChunked(
            tokens: promptIds[0..<1], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })
        try await runner.produce(token: nextIds[0], position: 1, into: logits)

        let preLayer = try #require(phases["preLayer.0"]).map { Self.toF32($0) }
        // The embed for the DECODE token (the runner re-embeds each token).
        let nextEmbedRow = Self.int4Rows(embedView, rows: Int(nextIds[0]) + 1,
                                         cols: model.config.hiddenSize)[Int(nextIds[0])]
        let nextEmbed = Self.f16(nextEmbedRow)
        print("decode hidden[0..8]=\(preLayer[0..<8])")
        print("nextEmbed[0..8]=\(nextEmbed[0..<8])")
        var preSS: Float = 0
        for v in preLayer { preSS += v * v }
        print("decode hidden ss=\(preSS)")
        var nonzero = 0
        for (i, v) in preLayer.enumerated() where v != 0 { nonzero += 1 }
        print("decode hidden nonzero count: \(nonzero) of \(preLayer.count)")
        print("decode hidden[0..<64]=\(preLayer[0..<64])")
        print("nextEmbed[0..<64]=\(nextEmbed[0..<64])")
        do {
            // Direct probe: engine embed kernel vs reference dequant.
            let embedKernel = try EmbedLookupInt4(context: ctx)
            guard let eBuf = ctx.device.makeBuffer(
                      length: model.config.hiddenSize * 2,
                      options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            let cb = ctx.queue.makeCommandBuffer()!
            embedKernel.encode(commandBuffer: cb,
                               table: embedView.buffer,
                               tableOffset: Int(embedView.offset),
                               scales: embedView.buffer,
                               scalesOffset: Int(embedView.scaleOffset),
                               biases: embedView.buffer,
                               biasesOffset: Int(embedView.biasOffset),
                               out: eBuf,
                               tokenId: UInt32(bitPattern: nextIds[0]),
                               d: UInt32(model.config.hiddenSize),
                               outScale: 1.0)
            cb.commit(); await cb.completed()
            let kernelOut = (0..<model.config.hiddenSize).map { Self.toF32(
                eBuf.contents().advanced(by: $0 * 2).load(as: Float16.self)) }
            print("embedKernel[0..8]=\(kernelOut[0..<8])")
            var m: Float = 0
            for i in 0..<model.config.hiddenSize {
                m = max(m, abs(kernelOut[i] - nextEmbed[i]))
            }
            print("embedKernel vs ref: maxAbs=\(m)")
        }
        let postAttn = try #require(phases["postAttn.0"]).map { Self.toF32($0) }

        // fp32 reference of the layer-0 GDN mixer.
        let D = 2048
        let V = 32, HD = 128
        let keyDim = 16 * HD   // 2048
        let valueDim = V * HD   // 4096
        let qkvDim = 2 * keyDim + valueDim

        let inNormW = Self.bf16Values(try model.inputNorm(layer: 0), count: D)
        let x = Self.f16(Self.rms(preLayer, weight: inNormW))   // input norm (fp16 out)

        let qkvW = Self.int4Rows(try model.gdnInProjQKV(layer: 0), rows: qkvDim, cols: D)
        let zW = Self.int4Rows(try model.gdnInProjZ(layer: 0), rows: valueDim, cols: D)
        let aW = Self.int4Rows(try model.gdnInProjA(layer: 0), rows: V, cols: D)
        let bW = Self.int4Rows(try model.gdnInProjB(layer: 0), rows: V, cols: D)
        let outW = Self.int4Rows(try model.gdnOutProj(layer: 0), rows: D, cols: valueDim)

        let convW = Self.fp16Values(try model.gdnConv1D(layer: 0), count: qkvDim * 4)
        let aLog = Self.fp32Values(try model.gdnALog(layer: 0), count: V)
        let dt = Self.fp32Values(try model.gdnDtBias(layer: 0), count: V)
        let normW = Self.bf16Values(try model.gdnNormWeight(layer: 0), count: HD)
        let postW = Self.bf16Values(try model.postAttnNorm(layer: 0), count: D)

        let qkv = Self.f16(Self.gemv(qkvW, x))
        let z = Self.f16(Self.gemv(zW, x))
        let a = Self.gemv(aW, x)
        let b = Self.gemv(bW, x)

        // Causal conv over the qkv stream (kernel 4), seeded from the
        // PREFILL's conv state (zero when no prefill ran).
        let pfConv = (phases["pfConvState.0"].map { $0.map { Self.toF32($0) } })
            ?? [Float](repeating: 0, count: qkvDim * 3)
        var conv = [Float](repeating: 0, count: qkvDim)
        for c in 0..<qkvDim {
            let acc = convW[c * 4 + 0] * pfConv[c * 3 + 0]
                + convW[c * 4 + 1] * pfConv[c * 3 + 1]
                + convW[c * 4 + 2] * pfConv[c * 3 + 2]
                + convW[c * 4 + 3] * qkv[c]
            conv[c] = acc / (1.0 + expf(-acc))
        }
        conv = Self.f16(conv)

        // Gate + recurrence per value head, seeded from the PREFILL's
        // recurrent state (zero when no prefill ran).
        let scale = 1.0 / Float(HD).squareRoot()
        let pfS = (phases["pfState.0"].map { $0.map { Self.toF32($0) } })
            ?? [Float](repeating: 0, count: V * HD * HD)
        var rec = [Float](repeating: 0, count: valueDim)
        for hv in 0..<V {
            let kh = hv / 2
            let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
            let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
            let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
            var qss: Float = 0, kss: Float = 0
            for i in 0..<HD {
                qss += (qHead[i] * scale) * (qHead[i] * scale)
                kss += kHead[i] * kHead[i]
            }
            let qinv = 1.0 / (qss + 1e-6).squareRoot()
            let kinv = 1.0 / (kss + 1e-6).squareRoot()
            let qn = qHead.map { $0 * scale * qinv }
            let kn = kHead.map { $0 * kinv }
            let g = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
            let beta = 1.0 / (1.0 + expf(-b[hv]))
            let decay = expf(g)
            let sBase = hv * HD * HD
            for i in 0..<(HD * HD) { pfS[sBase + i]  }  // no-op to avoid unused
            var knqn: Float = 0
            for i in 0..<HD { knqn += kn[i] * qn[i] }
            for vIdx in 0..<HD {
                let row = vIdx * HD
                var base: Float = 0
                for kk in 0..<HD {
                    let sv = pfS[sBase + row + kk] * decay
                    base += sv * qn[kk]
                }
                var r: Float = 0
                for kk in 0..<HD {
                    r += pfS[sBase + row + kk] * decay * kn[kk]
                }
                let delta = beta * (vHead[vIdx] - r)
                rec[hv * HD + vIdx] = base + delta * knqn
            }
        }
        rec = Self.f16(rec)

        // Gated RMSNorm (mean-based).
        var gated = [Float](repeating: 0, count: valueDim)
        for hv in 0..<V {
            var ss: Float = 0
            for i in 0..<HD { ss += rec[hv * HD + i] * rec[hv * HD + i] }
            let inv = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
            for i in 0..<HD {
                let zVal = z[hv * HD + i]
                let silu = zVal / (1.0 + expf(-zVal))
                gated[hv * HD + i] = rec[hv * HD + i] * inv * normW[i] * silu
            }
        }
        gated = Self.f16(gated)

        let attn = Self.f16(Self.gemv(outW, gated))
        let hidden1 = Self.f16(zip(preLayer, attn).map { $0 + $1 })
        let refDense = Self.f16(Self.rms(hidden1, weight: postW))

        func compareStage(_ name: String, _ engineValues: [Float16],
                          _ ref: [Float], count: Int) {
            var maxAbs: Float = 0
            var worst = -1
            for i in 0..<count {
                let e = Self.toF32(engineValues[i])
                let diff = abs(e - ref[i])
                if diff > maxAbs { maxAbs = diff; worst = i }
            }
            print("\(name): maxAbsDiff=\(maxAbs) at \(worst): "
                + "engine=\(engineValues[worst]) ref=\(ref[worst])")
        }

        let normedEngine = try #require(phases["normed.0"])
        compareStage("normed", normedEngine, x, count: D)
        compareStage("qkvConv", try #require(phases["qkvConv.0"]), conv, count: qkvDim)
        do {
            let recEngine = try #require(phases["recurrentOut.0"]).map { Self.toF32($0) }
            for hv in 0..<V {
                var hMax: Float = 0
                for i in 0..<HD {
                    hMax = max(hMax, abs(recEngine[hv * HD + i] - rec[hv * HD + i]))
                }
                if hMax > 0.01 {
                    print("REC hv=\(hv) maxAbs=\(hMax) engine[0..3]=\(recEngine[(hv*HD)..<(hv*HD+3)]) ref[0..3]=\(rec[(hv*HD)..<(hv*HD+3)])")
                }
            }
        }
        compareStage("recurrentOut", try #require(phases["recurrentOut.0"]), rec, count: valueDim)
        do {
            // Per-head gate comparison: g | beta (fp32 → fp16 in the hook).
            let gBetaEngine = try #require(phases["gFloat.0"]).map { Self.toF32($0) }
            var gRef = [Float](repeating: 0, count: V)
            var bRef = [Float](repeating: 0, count: V)
            for hv in 0..<V {
                gRef[hv] = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
                bRef[hv] = 1.0 / (1.0 + expf(-b[hv]))
            }
            for hv in 0..<V {
                let ge = gBetaEngine[hv]
                let be = gBetaEngine[V + hv]
                if abs(ge - gRef[hv]) > 0.01 || abs(be - bRef[hv]) > 0.01 {
                    print("GATE hv=\(hv): g engine=\(ge) ref=\(gRef[hv]) | beta engine=\(be) ref=\(bRef[hv])")
                }
            }
            print("g[0..4] engine=\(gBetaEngine[0..<4]) ref=\(gRef[0..<4])")
            print("beta[0..4] engine=\(gBetaEngine[V..<(V+4)]) ref=\(bRef[0..<4])")
        }
        do {
            // Decisive recurrent probe: kernel on the snapshot conv/g/beta
            // with a zero state buffer, vs the reference recurrence.
            let convEngine = try #require(phases["qkvConv.0"]).map { Self.toF32($0) }
            let gBetaEngine = try #require(phases["gFloat.0"]).map { Self.toF32($0) }
            let conv16 = convEngine.map { Float16($0) }
            let g16 = (0..<V).map { Float(gBetaEngine[$0]) }
            let b16 = (0..<V).map { Float(gBetaEngine[V + $0]) }
            guard let convBuf = ctx.device.makeBuffer(
                      bytes: conv16, length: qkvDim * 2, options: .storageModeShared),
                  let gBuf = ctx.device.makeBuffer(
                      bytes: g16, length: V * 4, options: .storageModeShared),
                  let bBuf = ctx.device.makeBuffer(
                      bytes: b16, length: V * 4, options: .storageModeShared),
                  let stBuf = ctx.device.makeBuffer(
                      length: V * HD * HD * 4, options: .storageModeShared),
                  let outBuf = ctx.device.makeBuffer(
                      length: valueDim * 2, options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            memset(stBuf.contents(), 0, stBuf.length)
            let gdn = try GDN(context: ctx)
            let cb = ctx.queue.makeCommandBuffer()!
            gdn.encodeRecurrent(commandBuffer: cb, state: stBuf,
                                q: convBuf,
                                k: convBuf, kOffset: keyDim * 2,
                                v: convBuf, vOffset: keyDim * 4,
                                g: gBuf, beta: bBuf, out: outBuf,
                                numValueHeads: V, headDim: UInt32(HD),
                                scale: scale, l2eps: 1e-6)
            cb.commit(); await cb.completed()
            let kOut = (0..<valueDim).map { Self.toF32(outBuf.contents()
                .advanced(by: $0 * 2).load(as: Float16.self)) }
            var kMax: Float = 0
            for i in 0..<valueDim {
                kMax = max(kMax, abs(kOut[i] - rec[i]))
            }
            print("recurrentKernel vs myRef: maxAbs=\(kMax)")
            do {
                let st = try #require(phases["recState.0"]).map { Self.toF32($0) }
                var stMax: Float = 0
                for v in st { stMax = max(stMax, abs(v)) }
                var stNonzero = 0
                for v in st where v != 0 { stNonzero += 1 }
                print("runner recState: maxAbs=\(stMax) nonzero=\(stNonzero) of \(st.count)")
            }
            print("recurrentKernel[0..3]=\(kOut[0..<3]) myRef[0..3]=\(rec[0..<3])")
        }
        compareStage("oOut", try #require(phases["oOut.0"]), attn, count: D)
        compareStage("postAttn", try #require(phases["postAttn.0"]), refDense, count: D)

        // ---- MoE tail reference: router + shared expert + routed experts.
        let dense = refDense   // the post-attn output (fp16) feeds the MoE

        // Router: int8 affine dequant → softmax → top-8 renormalized.
        let routerView = try model.router(layer: 0)
        let routerW = Self.int8Rows(routerView, rows: model.config.numExperts,
                                    cols: D)
        let routerLogits = Self.gemv(routerW, dense)
        var m = routerLogits.max() ?? 0
        var sum: Float = 0
        for i in 0..<routerLogits.count { sum += expf(routerLogits[i] - m) }
        let inv = 1.0 / sum
        let probs = routerLogits.map { expf($0 - m) * inv }
        let order = probs.indices.sorted { probs[$0] > probs[$1] }
        let topK = min(model.config.topKExperts, order.count)
        let topIDs = Array(order[0..<topK])
        let topW = topIDs.map { probs[$0] }
        var wSum: Float = 0
        for w in topW { wSum += w }
        let renorm = topW.map { $0 / wSum }
        print("ROUTER top8 ids=\(topIDs) weights=\(renorm)")

        // Shared expert: int4 FFN with silu + sigmoid gate.
        let sGateW = Self.int4Rows(try model.sharedExpertGate(layer: 0),
                                   rows: model.config.intermediateSize, cols: D)
        let sUpW = Self.int4Rows(try model.sharedExpertUp(layer: 0),
                                 rows: model.config.intermediateSize, cols: D)
        let sDownW = Self.int4Rows(try model.sharedExpertDown(layer: 0),
                                   rows: D, cols: model.config.intermediateSize)
        let gateVec = Self.f16(Self.gemv(sGateW, dense))
        let upVec = Self.f16(Self.gemv(sUpW, dense))
        var act = [Float](repeating: 0, count: model.config.intermediateSize)
        for i in 0..<act.count {
            let silu = gateVec[i] / (1.0 + expf(-gateVec[i]))
            act[i] = Float(Float16(silu * upVec[i]))
        }
        var h1 = Self.f16(Self.gemv(sDownW, act))
        let sharedGateW = Self.int4Rows(try model.sharedExpertGateProj(layer: 0),
                                        rows: 1, cols: D)[0]
        var gateDot: Float = 0
        for i in 0..<D { gateDot += sharedGateW[i] * dense[i] }
        let gateScale = 1.0 / (1.0 + expf(-gateDot))
        h1 = h1.map { Float(Float16($0 * gateScale)) }

        // Routed experts: read the packed blobs from the layer file.
        let layout = model.packedExpertsLayout
        let layer0Layout = layout.layers[0]
        let layerFile = URL(fileURLWithPath: Self.installPath)
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(layer0Layout.file)
        let fileData = try Data(contentsOf: layerFile)
        var h2 = [Float](repeating: 0, count: D)
        for (slot, expertID) in topIDs.enumerated() {
            let exp = layout.expert(layer: 0, expert: expertID)
            func rows(_ role: String, _ r: Int, _ c: Int) -> [[Float]] {
                let w = exp.subTensors[role]!
                let sT = exp.subTensors[role + "_scales"]!
                let bT = exp.subTensors[role + "_biases"]!
                let wBytes = [UInt8](fileData.subdata(
                    in: Data.Index(w.offset)..<Data.Index(w.offset + w.size)))
                let sWords = fileData.subdata(
                    in: Data.Index(sT.offset)..<Data.Index(sT.offset + sT.size))
                let bWords = fileData.subdata(
                    in: Data.Index(bT.offset)..<Data.Index(bT.offset + bT.size))
                let sVals = sWords.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt16.self))
                }
                let bVals = bWords.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt16.self))
                }
                var out = [[Float]](repeating: [], count: r)
                let groups = c / 64
                for row in 0..<r {
                    var vals = [Float](repeating: 0, count: c)
                    for g in 0..<groups {
                        let scale = FinchQuantization.bf16ToFloat(sVals[row * groups + g])
                        let bias = FinchQuantization.bf16ToFloat(bVals[row * groups + g])
                        let byteBase = row * (c / 2) + g * 32
                        for k in 0..<64 {
                            let byte = wBytes[byteBase + k / 2]
                            let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                            vals[g * 64 + k] = Float(nibble) * scale + bias
                        }
                    }
                    out[row] = vals
                }
                return out
            }
            let gateRows = rows("gate", model.config.moeIntermediateSize, D)
            let upRows = rows("up", model.config.moeIntermediateSize, D)
            let downRows = rows("down", D, model.config.moeIntermediateSize)
            let eGate = Self.f16(Self.gemv(gateRows, dense))
            let eUp = Self.f16(Self.gemv(upRows, dense))
            var eAct = [Float](repeating: 0, count: model.config.moeIntermediateSize)
            for i in 0..<eAct.count {
                let silu = eGate[i] / (1.0 + expf(-eGate[i]))
                eAct[i] = Float(Float16(silu * eUp[i]))
            }
            let eOut = Self.f16(Self.gemv(downRows, eAct))
            for i in 0..<D { h2[i] += renorm[slot] * eOut[i] }
        }
        for i in 0..<D { h2[i] = Float(Float16(h2[i] + h1[i])) }
        let refHidden = zip(hidden1, h2).map { Float(Float16($0 + $1)) }

        compareStage("postLayer", try #require(phases["postLayer.0"]), refHidden, count: D)

        // ---- Prefill layer-0 mixer: embed → mixer → denseX row 0.
        do {
            let pfDense = try #require(phases["prefillDense0.0"])
                .map { Self.toF32($0) }
            // Reuse the layer-0 reference chain with the embed as input.
            let xP = Self.f16(Self.rms(embedRef, weight: inNormW))
            let qkvP = Self.f16(Self.gemv(qkvW, xP))
            let zP = Self.f16(Self.gemv(zW, xP))
            let aP = Self.gemv(aW, xP)
            let bP = Self.gemv(bW, xP)
            var convP = [Float](repeating: 0, count: qkvDim)
            for c in 0..<qkvDim {
                let acc = convW[c * 4 + 3] * qkvP[c]
                convP[c] = acc / (1.0 + expf(-acc))
            }
            convP = Self.f16(convP)
            var recP = [Float](repeating: 0, count: valueDim)
            for hv in 0..<V {
                let kh = hv / 2
                let qHead = Array(convP[(kh * HD)..<(kh * HD + HD)])
                let kHead = Array(convP[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                let vHead = Array(convP[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                var qss: Float = 0, kss: Float = 0
                for i in 0..<HD {
                    qss += (qHead[i] * scale) * (qHead[i] * scale)
                    kss += kHead[i] * kHead[i]
                }
                let qinv = 1.0 / (qss + 1e-6).squareRoot()
                let kinv = 1.0 / (kss + 1e-6).squareRoot()
                let qn = qHead.map { $0 * scale * qinv }
                let kn = kHead.map { $0 * kinv }
                let g = -expf(aLog[hv]) * GDNRef.softplus(aP[hv] + dt[hv])
                let beta = 1.0 / (1.0 + expf(-bP[hv]))
                var knqn: Float = 0
                for i in 0..<HD { knqn += kn[i] * qn[i] }
                for vIdx in 0..<HD {
                    recP[hv * HD + vIdx] = beta * vHead[vIdx] * knqn
                }
            }
            recP = Self.f16(recP)
            var gatedP = [Float](repeating: 0, count: valueDim)
            for hv in 0..<V {
                var ss: Float = 0
                for i in 0..<HD { ss += recP[hv * HD + i] * recP[hv * HD + i] }
                let inv = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
                for i in 0..<HD {
                    let zVal = zP[hv * HD + i]
                    let silu = zVal / (1.0 + expf(-zVal))
                    gatedP[hv * HD + i] = recP[hv * HD + i] * inv * normW[i] * silu
                }
            }
            gatedP = Self.f16(gatedP)
            let attnP = Self.f16(Self.gemv(outW, gatedP))
            let hidden1P = Self.f16(zip(embedRef, attnP).map { $0 + $1 })
            let refDenseP = Self.f16(Self.rms(hidden1P, weight: postW))
            compareStage("prefillDense0", try #require(phases["prefillDense0.0"]),
                         refDenseP, count: D)
        }

        // ---- Layer 3: full-attention mixer reference (single token, pos 0).
        let L3 = 3
        let dense3 = try #require(phases["postAttn.3"]).map { Self.toF32($0) }
        // Layer 3's input = layer 2's FINAL hidden (the preLayer snapshot
        // races the deferred previous-layer tail; postLayer is clean).
        let hidden3 = try #require(phases["postLayer.2"]).map { Self.toF32($0) }
        let HD3 = model.config.fullHeadDim          // 256
        let numQ = model.config.numHeads            // 16
        let numKV = model.config.numFullKVHeads     // 2
        let qDim = numQ * HD3                       // 4096
        let kvDim = numKV * HD3                     // 512

        let inW3 = Self.bf16Values(try model.inputNorm(layer: L3), count: D)
        let x3 = Self.f16(Self.rms(hidden3, weight: inW3))
        let qVec = Self.f16(Self.gemv(
            Self.int4Rows(try model.qProj(layer: L3), rows: 2 * qDim, cols: D), x3))
        let kVec = Self.f16(Self.gemv(
            Self.int4Rows(try model.kProj(layer: L3), rows: kvDim, cols: D), x3))
        let vVec = Self.f16(Self.gemv(
            Self.int4Rows(try model.vProj(layer: L3), rows: kvDim, cols: D), x3))
        let oP = Self.int4Rows(try model.oProj(layer: L3), rows: D, cols: qDim)
        let qNW = Self.bf16Values(try model.qNorm(layer: L3), count: HD3)
        let kNW = Self.bf16Values(try model.kNorm(layer: L3), count: HD3)
        let postW3 = Self.bf16Values(try model.postAttnNorm(layer: L3), count: D)

        // Per-head norms; position 0 → RoPE identity; single KV token →
        // attention output = v. Then the output gate + o_proj.
        var attn3 = [Float](repeating: 0, count: qDim)
        for h in 0..<numQ {
            let kvh = h / (numQ / numKV)
            let qSlice = Array(qVec[(h * 2 * HD3)..<(h * 2 * HD3 + HD3)])
            let gateSlice = Array(qVec[(h * 2 * HD3 + HD3)..<(h * 2 * HD3 + 2 * HD3)])
            let qn = Self.rms(qSlice, weight: qNW)
            for i in 0..<HD3 {
                let g = gateSlice[i]
                let v = vVec[kvh * HD3 + i]
                let sgm = 1.0 / (1.0 + expf(-g))
                attn3[h * HD3 + i] = Float(Float16(v * sgm))
            }
        }
        let o3 = Self.f16(Self.gemv(oP, attn3))
        let hidden13 = Self.f16(zip(hidden3, o3).map { $0 + $1 })
        let refDense3 = Self.f16(Self.rms(hidden13, weight: postW3))
        compareStage("postAttnL3", try #require(phases["postAttn.3"]), refDense3, count: D)
        compareStage("normedL3", try #require(phases["normed.3"]), x3, count: D)
        do {
            // Compare the engine's raw q_proj output: qOut = normalized q,
            // gate = raw. The gate reveals whether the GEMV output matched.
            let gF = try #require(phases["gateF.3"]).map { Self.toF32($0) }
            var gateRef = [Float](repeating: 0, count: qDim)
            for h in 0..<numQ {
                for i in 0..<HD3 {
                    gateRef[h * HD3 + i] = Float(Float16(qVec[h * 2 * HD3 + HD3 + i]))
                }
            }
            var m: Float = 0
            var w = -1
            for i in 0..<qDim {
                let d = abs(gF[i] - gateRef[i])
                if d > m { m = d; w = i }
            }
            print("L3 gateRaw maxAbs=\(m) at \(w): engine=\(gF[w]) ref=\(gateRef[w])")
        }
        do {
            let qF = try #require(phases["qOutF.3"]).map { Self.toF32($0) }
            let gF = try #require(phases["gateF.3"]).map { Self.toF32($0) }
            let kF = try #require(phases["kF.3"]).map { Self.toF32($0) }
            var qnRef = [Float](repeating: 0, count: qDim)
            var gateRef = [Float](repeating: 0, count: qDim)
            var knRef = [Float](repeating: 0, count: kvDim)
            for h in 0..<numQ {
                let qSlice = Array(qVec[(h * 2 * HD3)..<(h * 2 * HD3 + HD3)])
                let qn = Self.rms(qSlice, weight: qNW)
                for i in 0..<HD3 {
                    qnRef[h * HD3 + i] = Float(Float16(qn[i]))
                    gateRef[h * HD3 + i] = Float(Float16(qVec[h * 2 * HD3 + HD3 + i]))
                }
            }
            for h in 0..<numKV {
                let kSlice = Array(kVec[(h * HD3)..<(h * HD3 + HD3)])
                let kn = Self.rms(kSlice, weight: kNW)
                for i in 0..<HD3 {
                    knRef[h * HD3 + i] = Float(Float16(kn[i]))
                }
            }
            func maxDiff(_ a: [Float], _ b: [Float]) -> (Float, Int) {
                var m: Float = 0
                var w = -1
                for i in 0..<min(a.count, b.count) {
                    let d = abs(a[i] - b[i])
                    if d > m { m = d; w = i }
                }
                return (m, w)
            }
            let (qm, qw) = maxDiff(qF, qnRef)
            let (gm, gw) = maxDiff(gF, gateRef)
            let (km, kw) = maxDiff(kF, knRef)
            print("L3 qOut maxAbs=\(qm) at \(qw): engine=\(qF[qw]) ref=\(qnRef[qw])")
            print("L3 gate maxAbs=\(gm) at \(gw): engine=\(gF[gw]) ref=\(gateRef[gw])")
            print("L3 k maxAbs=\(km) at \(kw): engine=\(kF[kw]) ref=\(knRef[kw])")
        }
    }
}

extension QwenLayer0DebugTests {

    /// Sweeps every layer's mixer (GDN + full-attention) for the first
    /// decode token after a 1-token prefill, comparing each layer's
    /// post_attention_layernorm output against a state-aware fp32 reference.
    /// Prints the first diverging layer.
    @Test(.enabled(if: installExists))
    func allLayerMixerSweep() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        var phases: [String: [Float16]] = [:]
        runner.qwenLayerDebugHook = { layer, phase, values in
            phases["\(phase).\(layer)"] = values
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        // The CLI's exact prompt: 5-token prefill, then one decode step.
        let promptIds = tokenizer.encode("The capital of France is", addBOS: false)
        let nextIds = tokenizer.encode(" Paris", addBOS: false)
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<promptIds.count], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })
        try await runner.produce(token: nextIds[0], position: promptIds.count,
                                 into: logits)

        let D = model.config.hiddenSize
        let embedView = model.embedding
        let decodeEmbedRow = Self.int4Rows(embedView, rows: Int(nextIds[0]) + 1,
                                           cols: D)[Int(nextIds[0])]
        let decodeEmbed = Self.f16(decodeEmbedRow)

        // Position of the decode token (RoPE matters for full layers).
        let position = promptIds.count
        let V = model.config.linearNumValueHeads
        let HD = model.config.linearValueHeadDim
        let keyDim = model.config.linearNumKeyHeads * HD
        let valueDim = V * HD
        let qkvDim = 2 * keyDim + valueDim
        let rotaryDim = Int(Double(model.config.fullHeadDim)
            * model.config.partialRotaryFactor)
        let theta = Float(model.config.fullRopeTheta)

        func ropePairs(_ x: inout [Float], headBase: Int, rotaryDim: Int) {
            for pair in 0..<(rotaryDim / 2) {
                let exponent = -Float(2 * pair) / Float(rotaryDim)
                let freq = powf(theta, exponent)
                let angle = Float(position) * freq
                let c = cosf(angle), sn = sinf(angle)
                let i0 = headBase + 2 * pair
                let i1 = headBase + 2 * pair + 1
                let x0 = x[i0], x1 = x[i1]
                x[i0] = x0 * c - x1 * sn
                x[i1] = x0 * sn + x1 * c
            }
        }

        func mixerReference(layer: Int, input: [Float]) -> [Float] {
            var x = Self.f16(Self.rms(input,
                weight: Self.bf16Values(try! model.inputNorm(layer: layer),
                                        count: D)))
            if model.config.fullAttentionLayerMask[layer] == 1 {
                let HDf = model.config.fullHeadDim
                let numQ = model.config.numHeads
                let numKV = model.config.numFullKVHeads
                let qDim = numQ * HDf
                let kvDim = numKV * HDf
                let oP = Self.int4Rows(try! model.oProj(layer: layer), rows: D, cols: qDim)
                let postW = Self.bf16Values(try! model.postAttnNorm(layer: layer),
                                            count: D)
                // Real attention over all T prefill rows + the decode row,
                // using the engine's own rotated q/k and raw v (no RoPE in
                // the reference).
                let qF = (phases["qOutF.\(layer)"] ?? []).map { Self.toF32($0) }
                let kF = (phases["kF.\(layer)"] ?? []).map { Self.toF32($0) }
                let vF = (phases["vF.\(layer)"] ?? []).map { Self.toF32($0) }
                var pfK: [[Float]] = []
                var pfV: [[Float]] = []
                for p in 0..<promptIds.count {
                    pfK.append((phases["pfK.\(p).\(layer)"] ?? []).map { Self.toF32($0) })
                    pfV.append((phases["pfV.\(p).\(layer)"] ?? []).map { Self.toF32($0) })
                }
                let gF = (phases["gateF.\(layer)"] ?? []).map { Self.toF32($0) }
                let scaleAttn = 1.0 / Float(HDf).squareRoot()
                var attn = [Float](repeating: 0, count: qDim)
                for h in 0..<numQ {
                    let kvh = h / (numQ / numKV)
                    for i in 0..<HDf {
                        // scores over the T+1 keys
                        var scores = [Float](repeating: 0, count: promptIds.count + 1)
                        var m: Float = -.infinity
                        for p in 0..<promptIds.count {
                            var s: Float = 0
                            for d in 0..<HDf {
                                s += qF[h * HDf + d] * pfK[p][kvh * HDf + d]
                            }
                            s *= scaleAttn
                            scores[p] = s
                            m = max(m, s)
                        }
                        var s: Float = 0
                        for d in 0..<HDf {
                            s += qF[h * HDf + d] * kF[kvh * HDf + d]
                        }
                        s *= scaleAttn
                        scores[promptIds.count] = s
                        m = max(m, s)
                        var sum: Float = 0
                        for p in 0...promptIds.count {
                            scores[p] = expf(scores[p] - m)
                            sum += scores[p]
                        }
                        var o: Float = 0
                        for p in 0..<promptIds.count {
                            o += (scores[p] / sum) * pfV[p][kvh * HDf + i]
                        }
                        o += (scores[promptIds.count] / sum) * vF[kvh * HDf + i]
                        let gate = gF[h * HDf + i]
                        let sgm = 1.0 / (1.0 + expf(-gate))
                        attn[h * HDf + i] = Float(Float16(o * sgm))
                    }
                }
                if layer == 3 {
                    // Direct attention kernel probe on the snapshot inputs.
                    let attnK = try! Attention(context: ctx)
                    let q16 = qF.map { Float16($0) }
                    let k16 = pfK[0] + kF
                    let v16 = pfV[0] + vF
                    guard let qB = ctx.device.makeBuffer(bytes: q16, length: qDim * 2, options: .storageModeShared),
                          let kB = ctx.device.makeBuffer(bytes: k16.map { Float16($0) }, length: kvDim * 2 * 2, options: .storageModeShared),
                          let vB = ctx.device.makeBuffer(bytes: v16.map { Float16($0) }, length: kvDim * 2 * 2, options: .storageModeShared),
                          let oB = ctx.device.makeBuffer(length: qDim * 2, options: .storageModeShared) else {
                        Issue.record("alloc failed"); return [Float](repeating: 0, count: D)
                    }
                    let cb = ctx.queue.makeCommandBuffer()!
                    attnK.encodeFull(commandBuffer: cb, q: qB,
                                     k: kB, v: vB, out: oB,
                                     headDim: UInt32(HDf),
                                     numQHeads: UInt32(numQ),
                                     numKVHeads: UInt32(numKV),
                                     seqLen: 2, scale: nil)
                    cb.commit(); cb.waitUntilCompleted()
                    let kOut = (0..<qDim).map { Self.toF32(oB.contents().advanced(by: $0 * 2).load(as: Float16.self)) }
                    var am: Float = 0, aw = -1
                    for i in 0..<qDim {
                        let d = abs(kOut[i] - attn[i])
                        if d > am { am = d; aw = i }
                    }
                    print("L3 attnKernel vs ref: maxAbs=\(am) at \(aw): kernel=\(kOut[aw]) ref=\(attn[aw])")
                    let engAttn = (phases["recurrentOut.3"] ?? []).map { Self.toF32($0) }
                    var em: Float = 0, ew = -1
                    for i in 0..<qDim {
                        let d = abs(engAttn[i] - kOut[i])
                        if d > em { em = d; ew = i }
                    }
                    print("L3 attnKernel vs engine: maxAbs=\(em) at \(ew): kernel=\(kOut[ew]) engine=\(engAttn[ew])")
                }
                let o = Self.f16(Self.gemv(oP, attn))
                let h1 = Self.f16(zip(input, o).map { $0 + $1 })
                return Self.f16(Self.rms(h1, weight: postW))
            } else {
                let qkvW = Self.int4Rows(try! model.gdnInProjQKV(layer: layer),
                                         rows: qkvDim, cols: D)
                let zW = Self.int4Rows(try! model.gdnInProjZ(layer: layer),
                                       rows: valueDim, cols: D)
                let aW = Self.int4Rows(try! model.gdnInProjA(layer: layer),
                                       rows: V, cols: D)
                let bW = Self.int4Rows(try! model.gdnInProjB(layer: layer),
                                       rows: V, cols: D)
                let outW = Self.int4Rows(try! model.gdnOutProj(layer: layer),
                                         rows: D, cols: valueDim)
                let convW = Self.fp16Values(try! model.gdnConv1D(layer: layer),
                                            count: qkvDim * 4)
                let aLog = Self.fp32Values(try! model.gdnALog(layer: layer), count: V)
                let dt = Self.fp32Values(try! model.gdnDtBias(layer: layer), count: V)
                let normW = Self.bf16Values(try! model.gdnNormWeight(layer: layer),
                                            count: HD)
                let postW = Self.bf16Values(try! model.postAttnNorm(layer: layer),
                                            count: D)
                let qkv = Self.f16(Self.gemv(qkvW, x))
                let z = Self.f16(Self.gemv(zW, x))
                let a = Self.gemv(aW, x)
                let b = Self.gemv(bW, x)
                let pfConv = phases["pfConvState.\(layer)"]?.map { Self.toF32($0) }
                    ?? [Float](repeating: 0, count: qkvDim * 3)
                var conv = [Float](repeating: 0, count: qkvDim)
                for c in 0..<qkvDim {
                    let acc = convW[c * 4 + 0] * pfConv[c * 3 + 0]
                        + convW[c * 4 + 1] * pfConv[c * 3 + 1]
                        + convW[c * 4 + 2] * pfConv[c * 3 + 2]
                        + convW[c * 4 + 3] * qkv[c]
                    conv[c] = acc / (1.0 + expf(-acc))
                }
                conv = Self.f16(conv)
                let pfS = phases["pfState.\(layer)"]?.map { Self.toF32($0) }
                    ?? [Float](repeating: 0, count: V * HD * HD)
                let scale = 1.0 / Float(HD).squareRoot()
                var rec = [Float](repeating: 0, count: valueDim)
                for hv in 0..<V {
                    let kh = hv / 2
                    let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
                    let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                    let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                    var qss: Float = 0, kss: Float = 0
                    for i in 0..<HD {
                        qss += (qHead[i] * scale) * (qHead[i] * scale)
                        kss += kHead[i] * kHead[i]
                    }
                    let qinv = 1.0 / (qss + 1e-6).squareRoot()
                    let kinv = 1.0 / (kss + 1e-6).squareRoot()
                    let qn = qHead.map { $0 * scale * qinv }
                    let kn = kHead.map { $0 * kinv }
                    let g = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
                    let beta = 1.0 / (1.0 + expf(-b[hv]))
                    let decay = expf(g)
                    let sBase = hv * HD * HD
                    var knqn: Float = 0
                    for i in 0..<HD { knqn += kn[i] * qn[i] }
                    for vIdx in 0..<HD {
                        let row = vIdx * HD
                        var base: Float = 0
                        var r: Float = 0
                        for kk in 0..<HD {
                            let sv = pfS[sBase + row + kk] * decay
                            base += sv * qn[kk]
                            r += sv * kn[kk]
                        }
                        let delta = beta * (vHead[vIdx] - r)
                        rec[hv * HD + vIdx] = base + delta * knqn
                    }
                }
                rec = Self.f16(rec)
                var gated = [Float](repeating: 0, count: valueDim)
                for hv in 0..<V {
                    var ss: Float = 0
                    for i in 0..<HD { ss += rec[hv * HD + i] * rec[hv * HD + i] }
                    let inv = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
                    for i in 0..<HD {
                        let zVal = z[hv * HD + i]
                        let silu = zVal / (1.0 + expf(-zVal))
                        gated[hv * HD + i] = rec[hv * HD + i] * inv * normW[i] * silu
                    }
                }
                gated = Self.f16(gated)
                let attn = Self.f16(Self.gemv(outW, gated))
                let h1 = Self.f16(zip(input, attn).map { $0 + $1 })
                return Self.f16(Self.rms(h1, weight: postW))
            }
        }

        var input = decodeEmbed
        for layer in 0..<model.config.numLayers {
            let ref = mixerReference(layer: layer, input: input)
            let engineDense = try #require(phases["postAttn.\(layer)"])
                .map { Self.toF32($0) }
            var m: Float = 0
            var w = -1
            for i in 0..<D {
                let d = abs(engineDense[i] - ref[i])
                if d > m { m = d; w = i }
            }
            print("L\(layer) mixer: maxAbs=\(m) at \(w): engine=\(engineDense[w]) ref=\(ref[w])")
            if layer == 1 {
                // Sub-bisect layer 1: norm → qkv proj → conv.
                let inW1 = Self.bf16Values(try! model.inputNorm(layer: 1), count: D)
                let x1 = Self.f16(Self.rms(input, weight: inW1))
                let normed1 = try #require(phases["normed.1"]).map { Self.toF32($0) }
                var nm: Float = 0, nw = -1
                for i in 0..<D {
                    let d = abs(normed1[i] - x1[i])
                    if d > nm { nm = d; nw = i }
                }
                print("L1 normed: maxAbs=\(nm) at \(nw): engine=\(normed1[nw]) ref=\(x1[nw])")
                let qkv1 = try #require(phases["qkvConv.1"]).map { Self.toF32($0) }
                let qkvW1 = Self.int4Rows(try! model.gdnInProjQKV(layer: 1),
                                          rows: qkvDim, cols: D)
                let qkvRef1 = Self.f16(Self.gemv(qkvW1, x1))
                let pfConv1 = phases["pfConvState.1"]?.map { Self.toF32($0) }
                    ?? [Float](repeating: 0, count: qkvDim * 3)
                let convW1 = Self.fp16Values(try! model.gdnConv1D(layer: 1),
                                             count: qkvDim * 4)
                var convRef1 = [Float](repeating: 0, count: qkvDim)
                for c in 0..<qkvDim {
                    let acc = convW1[c * 4 + 0] * pfConv1[c * 3 + 0]
                        + convW1[c * 4 + 1] * pfConv1[c * 3 + 1]
                        + convW1[c * 4 + 2] * pfConv1[c * 3 + 2]
                        + convW1[c * 4 + 3] * qkvRef1[c]
                    convRef1[c] = acc / (1.0 + expf(-acc))
                }
                convRef1 = Self.f16(convRef1)
                var cm: Float = 0, cw = -1
                for i in 0..<qkvDim {
                    let d = abs(qkv1[i] - convRef1[i])
                    if d > cm { cm = d; cw = i }
                }
                print("L1 qkvConv: maxAbs=\(cm) at \(cw): engine=\(qkv1[cw]) ref=\(convRef1[cw])")
                let pfS1 = phases["pfState.1"]?.map { Self.toF32($0) }
                    ?? [Float](repeating: 0, count: V * HD * HD)
                var sMax: Float = 0
                for v in pfS1 { sMax = max(sMax, abs(v)) }
                print("L1 pfState maxAbs=\(sMax) count=\(pfS1.count)")
            }
            input = (try #require(phases["postLayer.\(layer)"])).map { Self.toF32($0) }
        }

        // ---- Second decode step: layer-0 mixer with the step-1 states.
        do {
            // The runner's step-2 layer-0 input = step-1's postLayer.0 — but
            // the hook phases were overwritten by step 2's snapshots. Capture
            // the step-2 snapshots separately by re-running? Instead, verify
            // the state transition rule: the step-1 post-decode state was
            // captured by recState.0 (step-1, post-update) — compare the
            // step-2 preLayer against a reference built from step-1's
            // postLayer.0 and step-1's states.
            let s2Input = try #require(phases["postLayer.39"]).map { Self.toF32($0) }
            let s2Pre = try #require(phases["preLayer.0"]).map { Self.toF32($0) }
            var m: Float = 0
            for i in 0..<D {
                m = max(m, abs(s2Pre[i] - s2Input[i]))
            }
            print("step2 input vs step1 postLayer.39: maxAbs=\(m)")
            print("step2 input[0..4]=\(s2Pre[0..<4])")
            print("step1 postLayer39[0..4]=\(s2Input[0..<4])")
        }

        // ---- PREFILL STATE VALIDATION: layer 0's conv + recurrent state,
        // computed from scratch over the 5 prefill tokens (the layer-0
        // inputs are the embeds; no MoE feedback into its states).
        do {
            let V0 = model.config.linearNumValueHeads
            let HD0 = model.config.linearValueHeadDim
            let keyDim0 = model.config.linearNumKeyHeads * HD0
            let valueDim0 = V0 * HD0
            let qkvDim0 = 2 * keyDim0 + valueDim0
            let inNormW0 = Self.bf16Values(try model.inputNorm(layer: 0), count: D)
            let qkvW0 = Self.int4Rows(try model.gdnInProjQKV(layer: 0),
                                      rows: qkvDim0, cols: D)
            let aW0 = Self.int4Rows(try model.gdnInProjA(layer: 0), rows: V0, cols: D)
            let bW0 = Self.int4Rows(try model.gdnInProjB(layer: 0), rows: V0, cols: D)
            let convW0 = Self.fp16Values(try model.gdnConv1D(layer: 0),
                                         count: qkvDim0 * 4)
            let aLog0 = Self.fp32Values(try model.gdnALog(layer: 0), count: V0)
            let dt0 = Self.fp32Values(try model.gdnDtBias(layer: 0), count: V0)
            let scale0 = 1.0 / Float(HD0).squareRoot()

            var convState = [Float](repeating: 0, count: qkvDim0 * 3)
            var S = [Float](repeating: 0, count: V0 * HD0 * HD0)
            for t in 0..<promptIds.count {
                let tokenID = Int(promptIds[t])
                let embRow = Self.int4Rows(embedView, rows: tokenID + 1, cols: D)[tokenID]
                let emb = Self.f16(embRow)
                let x = Self.f16(Self.rms(emb, weight: inNormW0))
                let qkv = Self.f16(Self.gemv(qkvW0, x))
                let a = Self.gemv(aW0, x)
                let b = Self.gemv(bW0, x)
                // conv: taps from convState, carry raw qkv
                var conv = [Float](repeating: 0, count: qkvDim0)
                for c in 0..<qkvDim0 {
                    let acc = convW0[c * 4 + 0] * convState[c * 3 + 0]
                        + convW0[c * 4 + 1] * convState[c * 3 + 1]
                        + convW0[c * 4 + 2] * convState[c * 3 + 2]
                        + convW0[c * 4 + 3] * qkv[c]
                    conv[c] = acc / (1.0 + expf(-acc))
                }
                conv = Self.f16(conv)
                for c in 0..<qkvDim0 {
                    convState[c * 3 + 0] = convState[c * 3 + 1]
                    convState[c * 3 + 1] = convState[c * 3 + 2]
                    convState[c * 3 + 2] = qkv[c]
                }
                // recurrent step
                for hv in 0..<V0 {
                    let kh = hv / 2
                    let qHead = Array(conv[(kh * HD0)..<(kh * HD0 + HD0)])
                    let kHead = Array(conv[(keyDim0 + kh * HD0)..<(keyDim0 + kh * HD0 + HD0)])
                    let vHead = Array(conv[(2 * keyDim0 + hv * HD0)..<(2 * keyDim0 + hv * HD0 + HD0)])
                    var qss: Float = 0, kss: Float = 0
                    for i in 0..<HD0 {
                        qss += (qHead[i] * scale0) * (qHead[i] * scale0)
                        kss += kHead[i] * kHead[i]
                    }
                    let qinv = 1.0 / (qss + 1e-6).squareRoot()
                    let kinv = 1.0 / (kss + 1e-6).squareRoot()
                    let qn = qHead.map { $0 * scale0 * qinv }
                    let kn = kHead.map { $0 * kinv }
                    let g = -expf(aLog0[hv]) * GDNRef.softplus(a[hv] + dt0[hv])
                    let beta = 1.0 / (1.0 + expf(-b[hv]))
                    let decay = expf(g)
                    let sBase = hv * HD0 * HD0
                    for i in 0..<(HD0 * HD0) { S[sBase + i] *= decay }
                    var r = [Float](repeating: 0, count: HD0)
                    for vIdx in 0..<HD0 {
                        var acc: Float = 0
                        let row = vIdx * HD0
                        for kk in 0..<HD0 { acc += S[sBase + row + kk] * kn[kk] }
                        r[vIdx] = acc
                    }
                    for vIdx in 0..<HD0 {
                        let delta = beta * (vHead[vIdx] - r[vIdx])
                        let row = vIdx * HD0
                        for kk in 0..<HD0 { S[sBase + row + kk] += kn[kk] * delta }
                    }
                }
            }
            // Compare against the engine's prefill states for layer 0.
            let pfS = (phases["pfState.0"] ?? []).map { Self.toF32($0) }
            var sMax: Float = 0, sW = -1
            for i in 0..<S.count {
                let d = abs(pfS[i] - S[i])
                if d > sMax { sMax = d; sW = i }
            }
            print("pfState.0 vs ref: maxAbs=\(sMax) at \(sW)")
            let pfC = (phases["pfConvState.0"] ?? []).map { Self.toF32($0) }
            var cMax: Float = 0, cW = -1
            for i in 0..<convState.count {
                let d = abs(pfC[i] - convState[i])
                if d > cMax { cMax = d; cW = i }
            }
            print("pfConvState.0 vs ref: maxAbs=\(cMax) at \(cW)")
        }

        // ---- LM head reference: argmax over all lm_head rows.
        do {
            let hidden = try #require(phases["postLayer.39"]).map { Self.toF32($0) }
            let normW = Self.bf16Values(model.finalNorm, count: D)
            let normed = Self.f16(Self.rms(hidden, weight: normW))
            let lmView = model.lmHead
            let base = lmView.buffer.contents()
            let wBytes = base.advanced(by: Int(lmView.offset))
            let sWords = base.advanced(by: Int(lmView.scaleOffset))
                .assumingMemoryBound(to: UInt16.self)
            let bWords = base.advanced(by: Int(lmView.biasOffset))
                .assumingMemoryBound(to: UInt16.self)
            let vocab = model.config.vocabSize
            let groups = D / 64
            var best: Float = -.infinity
            var bestRow = -1
            for r in 0..<vocab {
                var acc: Float = 0
                for g in 0..<groups {
                    let scale = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                    let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                    let byteBase = r * (D / 2) + g * 32
                    for k in 0..<64 {
                        let byte = wBytes.load(fromByteOffset: byteBase + k / 2,
                                               as: UInt8.self)
                        let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                        let w = Float(nibble) * scale + bias
                        acc += w * normed[g * 64 + k]
                    }
                }
                if acc > best {
                    best = acc
                    bestRow = r
                }
            }
            print("HEAD argmax ref: row=\(bestRow) score=\(best)")
            print("HEAD argmax engine: \(runner.lastGreedyToken)")
        }
    }
}

extension QwenLayer0DebugTests {

    /// Hunts the FIRST diverging prefill layer: replays the 5-token prefill
    /// layer-by-layer in fp32, seeding each layer's reference from the
    /// ENGINE's own per-row input snapshots (so a layer is judged in
    /// isolation given its real input), and compares the engine's denseX row
    /// (post_attention_layernorm, pre-MoE) against the reference. GDN layers
    /// replay their conv/recurrent state from zero per layer; full-attention
    /// layers replay the doubled q-proj + epilogue + causal softmax. Prints
    /// per-layer/token maxAbs and the first exceeding 0.05.
    @Test(.enabled(if: installExists))
    func prefillSweepFindsFirstDivergence() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        var phases: [String: [Float16]] = [:]
        runner.qwenLayerDebugHook = { layer, phase, values in
            phases["\(phase).\(layer)"] = values
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is",
                                         addBOS: false)
        let T = promptIds.count
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<T], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })

        // Layer L's INPUT is layer L-1's post-MoE hidden (the prefill hook
        // snapshots `prefillHidden.<t>.<L>` AFTER layer L's own MoE tail, so
        // that row is L's OUTPUT). Layer 0's input is the embedding row.
        let embedView = model.embedding
        let embeds = (0..<T).map { t -> [Float] in
            let row = Self.int4Rows(embedView, rows: Int(promptIds[t]) + 1,
                                    cols: model.config.hiddenSize)[Int(promptIds[t])]
            return Self.f16(row)   // Qwen does not scale embeddings
        }
        func layerInput(_ L: Int, _ t: Int) -> [Float] {
            L == 0 ? embeds[t]
                : (phases["prefillHidden.\(t).\(L - 1)"] ?? []).map { Self.toF32($0) }
        }

        let D = model.config.hiddenSize
        let V = model.config.linearNumValueHeads
        let HD = model.config.linearValueHeadDim
        let keyDim = model.config.linearNumKeyHeads * HD
        let valueDim = V * HD
        let qkvDim = 2 * keyDim + valueDim
        let scale = 1.0 / Float(HD).squareRoot()
        let HDf = model.config.fullHeadDim
        let numQ = model.config.numHeads
        let numKV = model.config.numFullKVHeads
        let rotaryDim = Int(Double(HDf) * model.config.partialRotaryFactor)
        let theta = Float(model.config.fullRopeTheta)

        var firstDivergence: String? = nil

        for L in 0..<model.config.numLayers {
            let isFull = model.config.fullAttentionLayerMask[L] == 1
            let inNormW = Self.bf16Values(try model.inputNorm(layer: L),
                                          count: D)
            let postW = Self.bf16Values(try model.postAttnNorm(layer: L),
                                        count: D)
            // Cached per-layer weights.
            let qkvW: [[Float]]? = isFull ? nil
                : Self.int4Rows(try model.gdnInProjQKV(layer: L),
                                rows: qkvDim, cols: D)
            let zW: [[Float]]? = isFull ? nil
                : Self.int4Rows(try model.gdnInProjZ(layer: L),
                                rows: valueDim, cols: D)
            let aW: [[Float]]? = isFull ? nil
                : Self.int4Rows(try model.gdnInProjA(layer: L), rows: V, cols: D)
            let bW: [[Float]]? = isFull ? nil
                : Self.int4Rows(try model.gdnInProjB(layer: L), rows: V, cols: D)
            let outW: [[Float]]? = isFull ? nil
                : Self.int4Rows(try model.gdnOutProj(layer: L), rows: D,
                                cols: valueDim)
            let convW = isFull ? [] : Self.fp16Values(try model.gdnConv1D(layer: L),
                                                     count: qkvDim * 4)
            let aLog = isFull ? [] : Self.fp32Values(try model.gdnALog(layer: L),
                                                     count: V)
            let dt = isFull ? [] : Self.fp32Values(try model.gdnDtBias(layer: L),
                                                   count: V)
            let normW = isFull ? [] : Self.bf16Values(
                try model.gdnNormWeight(layer: L), count: HD)
            let qProjW: [[Float]]? = !isFull ? nil
                : Self.int4Rows(try model.qProj(layer: L),
                                rows: 2 * numQ * HDf, cols: D)
            let kProjW: [[Float]]? = !isFull ? nil
                : Self.int4Rows(try model.kProj(layer: L),
                                rows: numKV * HDf, cols: D)
            let vProjW: [[Float]]? = !isFull ? nil
                : Self.int4Rows(try model.vProj(layer: L),
                                rows: numKV * HDf, cols: D)
            let oProjW: [[Float]]? = !isFull ? nil
                : Self.int4Rows(try model.oProj(layer: L), rows: D,
                                cols: numQ * HDf)
            let qNW = !isFull ? [] : Self.bf16Values(try model.qNorm(layer: L),
                                                     count: HDf)
            let kNW = !isFull ? [] : Self.bf16Values(try model.kNorm(layer: L),
                                                     count: HDf)

            // Sequential GDN states, replayed per layer from zero.
            var convState = [Float](repeating: 0, count: qkvDim * 3)
            var recState = [Float](repeating: 0, count: V * HD * HD)

            for t in 0..<T {
                let input = layerInput(L, t)
                let engineDense = (phases["prefillDense.\(t).\(L)"] ?? [])
                    .map { Self.toF32($0) }
                var denseRef: [Float]
                if isFull {
                    let x = Self.f16(Self.rms(input, weight: inNormW))
                    let qVec = Self.f16(Self.gemv(qProjW!, x))
                    let kVec = Self.f16(Self.gemv(kProjW!, x))
                    let vVec = Self.f16(Self.gemv(vProjW!, x))
                    // Epilogue: per-head norms + partial RoPE (q and k).
                    var qn = [Float](repeating: 0, count: numQ * HDf)
                    var gate = [Float](repeating: 0, count: numQ * HDf)
                    var kn = [Float](repeating: 0, count: numKV * HDf)
                    for h in 0..<numQ {
                        let qSlice = Array(qVec[(h * 2 * HDf)..<(h * 2 * HDf + HDf)])
                        let gSlice = Array(qVec[(h * 2 * HDf + HDf)..<(h * 2 * HDf + 2 * HDf)])
                        let qnH = Self.rms(qSlice, weight: qNW)
                        for i in 0..<HDf {
                            qn[h * HDf + i] = qnH[i]
                            gate[h * HDf + i] = gSlice[i]
                        }
                        Self.ropePairs(&qn, headBase: h * HDf,
                                       rotaryDim: rotaryDim, position: t,
                                       theta: theta)
                    }
                    for h in 0..<numKV {
                        let kSlice = Array(kVec[(h * HDf)..<(h * HDf + HDf)])
                        let knH = Self.rms(kSlice, weight: kNW)
                        for i in 0..<HDf { kn[h * HDf + i] = knH[i] }
                        Self.ropePairs(&kn, headBase: h * HDf,
                                       rotaryDim: rotaryDim, position: t,
                                       theta: theta)
                    }
                    // Reference KV rows for causal attention: recompute the
                    // epilogue for every PREVIOUS token at ITS position.
                    var kRows = [[Float]]()
                    var vRows = [[Float]]()
                    for p in 0...t {
                        let xP = Self.f16(Self.rms(layerInput(L, p),
                                                   weight: inNormW))
                        let kP = Self.f16(Self.gemv(kProjW!, xP))
                        let vP = Self.f16(Self.gemv(vProjW!, xP))
                        var knP = [Float](repeating: 0, count: numKV * HDf)
                        for h in 0..<numKV {
                            let kSlice = Array(kP[(h * HDf)..<(h * HDf + HDf)])
                            let knH = Self.rms(kSlice, weight: kNW)
                            for i in 0..<HDf { knP[h * HDf + i] = knH[i] }
                            Self.ropePairs(&knP, headBase: h * HDf,
                                           rotaryDim: rotaryDim,
                                           position: p, theta: theta)
                        }
                        kRows.append(knP)
                        vRows.append(vP)
                    }
                    // Causal softmax attention (scale = 1/sqrt(head_dim)).
                    let attnScale = 1.0 / Float(HDf).squareRoot()
                    var attn = [Float](repeating: 0, count: numQ * HDf)
                    for h in 0..<numQ {
                        let kvh = h / (numQ / numKV)
                        for i in 0..<HDf {
                            var scores = [Float](repeating: 0, count: t + 1)
                            var m: Float = -.infinity
                            for p in 0...t {
                                var s: Float = 0
                                for d in 0..<HDf {
                                    s += qn[h * HDf + d] * kRows[p][kvh * HDf + d]
                                }
                                s *= attnScale
                                scores[p] = s
                                m = max(m, s)
                            }
                            var sum: Float = 0
                            for p in 0...t {
                                scores[p] = expf(scores[p] - m)
                                sum += scores[p]
                            }
                            var o: Float = 0
                            for p in 0...t {
                                o += (scores[p] / sum) * vRows[p][kvh * HDf + i]
                            }
                            let sgm = 1.0 / (1.0 + expf(-gate[h * HDf + i]))
                            attn[h * HDf + i] = Float(Float16(o * sgm))
                        }
                    }
                    let o = Self.f16(Self.gemv(oProjW!, attn))
                    let hidden1 = Self.f16(zip(input, o).map { $0 + $1 })
                    denseRef = Self.f16(Self.rms(hidden1, weight: postW))
                } else {
                    let x = Self.f16(Self.rms(input, weight: inNormW))
                    let qkv = Self.f16(Self.gemv(qkvW!, x))
                    let z = Self.f16(Self.gemv(zW!, x))
                    // The engine's fused a|b QMM writes fp16; the gate kernel
                    // reads the rounded values.
                    let a = Self.f16(Self.gemv(aW!, x))
                    let b = Self.f16(Self.gemv(bW!, x))
                    var conv = [Float](repeating: 0, count: qkvDim)
                    for c in 0..<qkvDim {
                        let acc = convW[c * 4 + 0] * convState[c * 3 + 0]
                            + convW[c * 4 + 1] * convState[c * 3 + 1]
                            + convW[c * 4 + 2] * convState[c * 3 + 2]
                            + convW[c * 4 + 3] * qkv[c]
                        conv[c] = acc / (1.0 + expf(-acc))
                    }
                    conv = Self.f16(conv)
                    for c in 0..<qkvDim {
                        convState[c * 3 + 0] = convState[c * 3 + 1]
                        convState[c * 3 + 1] = convState[c * 3 + 2]
                        convState[c * 3 + 2] = qkv[c]
                    }
                    var rec = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        let kh = hv / 2
                        let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
                        let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                        let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                        var qss: Float = 0, kss: Float = 0
                        for i in 0..<HD {
                            qss += (qHead[i] * scale) * (qHead[i] * scale)
                            kss += kHead[i] * kHead[i]
                        }
                        let qinv = 1.0 / (qss + 1e-6).squareRoot()
                        let kinv = 1.0 / (kss + 1e-6).squareRoot()
                        let qn = qHead.map { $0 * scale * qinv }
                        let kn = kHead.map { $0 * kinv }
                        let g = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
                        let beta = 1.0 / (1.0 + expf(-b[hv]))
                        let decay = expf(g)
                        let sBase = hv * HD * HD
                        for i in 0..<(HD * HD) { recState[sBase + i] *= decay }
                        var r = [Float](repeating: 0, count: HD)
                        for vIdx in 0..<HD {
                            var acc: Float = 0
                            let row = vIdx * HD
                            for kk in 0..<HD { acc += recState[sBase + row + kk] * kn[kk] }
                            r[vIdx] = acc
                        }
                        var knqn: Float = 0
                        for i in 0..<HD { knqn += kn[i] * qn[i] }
                        for vIdx in 0..<HD {
                            let delta = beta * (vHead[vIdx] - r[vIdx])
                            let row = vIdx * HD
                            for kk in 0..<HD {
                                recState[sBase + row + kk] += kn[kk] * delta
                            }
                        }
                        // o[hv] = updated_state @ qn.
                        for vIdx in 0..<HD {
                            var oVal: Float = 0
                            let row = vIdx * HD
                            for kk in 0..<HD {
                                oVal += recState[sBase + row + kk] * qn[kk]
                            }
                            rec[hv * HD + vIdx] = oVal
                        }
                    }
                    rec = Self.f16(rec)
                    var gated = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        var ss: Float = 0
                        for i in 0..<HD { ss += rec[hv * HD + i] * rec[hv * HD + i] }
                        let inv = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
                        for i in 0..<HD {
                            let zVal = z[hv * HD + i]
                            let silu = zVal / (1.0 + expf(-zVal))
                            gated[hv * HD + i] = rec[hv * HD + i] * inv * normW[i] * silu
                        }
                    }
                    gated = Self.f16(gated)
                    let attn = Self.f16(Self.gemv(outW!, gated))
                    let hidden1 = Self.f16(zip(input, attn).map { $0 + $1 })
                    denseRef = Self.f16(Self.rms(hidden1, weight: postW))
                }
                var m: Float = 0
                var w = -1
                for i in 0..<D {
                    let d = abs(engineDense[i] - denseRef[i])
                    if d > m { m = d; w = i }
                }
                print("pf L\(L).t\(t) dense: maxAbs=\(m) at \(w)")
                if m > 0.05 && firstDivergence == nil {
                    firstDivergence = "L\(L).t\(t) maxAbs=\(m) at \(w)"
                }
            }
            if firstDivergence != nil { break }
        }
        print("FIRST DIVERGENCE: \(firstDivergence ?? "none — all mixers match")")
    }

    /// Per-layer engine-vs-reference probe of the WHOLE MoE layer (mixer +
    /// router + shared expert + top-8 routed experts) for the LAST prompt row
    /// (row 4 of the 5-token prompt — the row the CLI sampled token 1 from).
    /// Unlike prefillMoETailHunt this replays the GDN conv/recurrent states
    /// FROM ZERO through rows 0..4 (the end-of-chunk pfState/pfConvState
    /// snapshots include row 4's own write, which polluted the old hunt's
    /// row-4 read), and the tail is fed the REPLAYED dense row (the engine's
    /// own denseX is still printed for attribution). Per-layer isolation: the
    /// mixer input rows are the engine's own snapshots at that layer.
    @Test(.enabled(if: installExists))
    func prefillLayerTailsMatchReference() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        var phases: [String: [Float16]] = [:]
        runner.qwenLayerDebugHook = { layer, phase, values in
            phases["\(phase).\(layer)"] = values
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is",
                                         addBOS: false)
        let T = promptIds.count
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<T], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })
        func say(_ s: String) { print(s); fflush(stdout) }

        let cfg = model.config
        let D = cfg.hiddenSize
        let I = cfg.intermediateSize
        let moeI = cfg.moeIntermediateSize
        let V = cfg.linearNumValueHeads
        let HD = cfg.linearValueHeadDim
        let keyDim = cfg.linearNumKeyHeads * HD
        let valueDim = V * HD
        let qkvDim = 2 * keyDim + valueDim
        let scale = 1.0 / Float(cfg.linearKeyHeadDim).squareRoot()
        let HDf = cfg.fullHeadDim
        let numQ = cfg.numHeads
        let numKV = cfg.numFullKVHeads
        let numExperts = cfg.numExperts
        let topK = cfg.topKExperts

        let embedView = model.embedding
        let embeds = (0..<T).map { t -> [Float] in
            Self.f16(Self.int4Rows(embedView, rows: Int(promptIds[t]) + 1,
                                   cols: D)[Int(promptIds[t])])
        }
        func layerInput(_ L: Int, _ t: Int) -> [Float] {
            L == 0 ? embeds[t]
                : (phases["prefillHidden.\(t).\(L - 1)"] ?? []).map { Self.toF32($0) }
        }
        // FQ_CHAINED=1: instead of per-layer isolation (each layer re-seeded
        // from the engine's own rows), chain the replay's own fp16-rounded
        // output rows into the next layer — cumulative-drift measurement.
        let chained = ProcessInfo.processInfo.environment["FQ_CHAINED"] != nil
        var chain = chained ? embeds : []
        if chained { say("CHAINED MODE: cumulative row-4 replay vs engine") }
        var chainOut = [[Float]](repeating: [], count: T)

        let layout = model.packedExpertsLayout
        var firstDivergence: String? = nil

        for L in 0..<cfg.numLayers {
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            say("probe L\(L) start")
            let inNormW = Self.bf16Values(try model.inputNorm(layer: L), count: D)
            let postW = Self.bf16Values(try model.postAttnNorm(layer: L), count: D)
            // Weights (mirror the sweep's per-layer cache).
            var qkvW: [[Float]]? = nil; var zW: [[Float]]? = nil
            var aW: [[Float]]? = nil; var bW: [[Float]]? = nil
            var outW: [[Float]]? = nil
            var convW: [Float] = []; var aLog: [Float] = []
            var dt: [Float] = []; var normW: [Float] = []
            var qProjW: [[Float]]? = nil; var kProjW: [[Float]]? = nil
            var vProjW: [[Float]]? = nil; var oProjW: [[Float]]? = nil
            var qNW: [Float] = []; var kNW: [Float] = []
            if isFull {
                qProjW = Self.int4Rows(try model.qProj(layer: L),
                                       rows: 2 * numQ * HDf, cols: D)
                kProjW = Self.int4Rows(try model.kProj(layer: L),
                                       rows: numKV * HDf, cols: D)
                vProjW = Self.int4Rows(try model.vProj(layer: L),
                                       rows: numKV * HDf, cols: D)
                oProjW = Self.int4Rows(try model.oProj(layer: L),
                                       rows: D, cols: numQ * HDf)
                qNW = Self.bf16Values(try model.qNorm(layer: L), count: HDf)
                kNW = Self.bf16Values(try model.kNorm(layer: L), count: HDf)
            } else {
                qkvW = Self.int4Rows(try model.gdnInProjQKV(layer: L),
                                     rows: qkvDim, cols: D)
                zW = Self.int4Rows(try model.gdnInProjZ(layer: L),
                                   rows: valueDim, cols: D)
                aW = Self.int4Rows(try model.gdnInProjA(layer: L),
                                   rows: V, cols: D)
                bW = Self.int4Rows(try model.gdnInProjB(layer: L),
                                   rows: V, cols: D)
                outW = Self.int4Rows(try model.gdnOutProj(layer: L),
                                     rows: D, cols: valueDim)
                convW = Self.fp16Values(try model.gdnConv1D(layer: L),
                                        count: qkvDim * 4)
                aLog = Self.fp32Values(try model.gdnALog(layer: L), count: V)
                dt = Self.fp32Values(try model.gdnDtBias(layer: L), count: V)
                normW = Self.bf16Values(try model.gdnNormWeight(layer: L),
                                        count: HD)
            }
            // Router + shared expert weights (any layer).
            let routerW = Self.int8Rows(try model.router(layer: L),
                                        rows: numExperts, cols: D)
            let sGateW = Self.int4Rows(try model.sharedExpertGate(layer: L),
                                       rows: I, cols: D)
            let sUpW = Self.int4Rows(try model.sharedExpertUp(layer: L),
                                     rows: I, cols: D)
            let sDownW = Self.int4Rows(try model.sharedExpertDown(layer: L),
                                       rows: D, cols: I)
            let sharedGateW = Self.int4Rows(try model.sharedExpertGateProj(layer: L),
                                            rows: 1, cols: D)[0]
            // Routed expert bytes for this layer (one file handle, reused).
            let l0 = layout.layers[L]
            let layerURL = URL(fileURLWithPath: Self.installPath)
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(l0.file)
            let fh = try FileHandle(forReadingFrom: layerURL)
            defer { try? fh.close() }
            /// Dequant rows of one role of one expert (`rows` x `cols`).
            /// Layout offsets are blob-relative; the blob starts at
            /// `exp.offset` inside the layer file, so reads seek there.
            func expertRows(_ exp: ExpertEntry, _ role: String,
                            _ r: Int, _ c: Int) -> [[Float]] {
                let blob = Int(exp.offset)
                let w = exp.subTensors[role]!
                let sT = exp.subTensors[role + "_scales"]!
                let bT = exp.subTensors[role + "_biases"]!
                let wBytes = Self.readRange(fh, blob + Int(w.offset), Int(w.size))
                let sVals = Self.readRange(fh, blob + Int(sT.offset), Int(sT.size))
                    .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                let bVals = Self.readRange(fh, blob + Int(bT.offset), Int(bT.size))
                    .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                let groups = c / 64
                var out = [[Float]](repeating: [], count: r)
                for row in 0..<r {
                    var vals = [Float](repeating: 0, count: c)
                    for g in 0..<groups {
                        let scl = FinchQuantization.bf16ToFloat(sVals[row * groups + g])
                        let bias = FinchQuantization.bf16ToFloat(bVals[row * groups + g])
                        let byteBase = row * (c / 2) + g * 32
                        for k in 0..<64 {
                            let byte = wBytes[byteBase + k / 2]
                            let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                            vals[g * 64 + k] = Float(nibble) * scl + bias
                        }
                    }
                    out[row] = vals
                }
                return out
            }

            // Sequential states replayed from zero through rows 0..<T.
            var convState = [Float](repeating: 0, count: qkvDim * 3)
            var recState = [Float](repeating: 0, count: V * HD * HD)
            var worstDense: Float = 0
            var worstHidden: Float = 0
            var hiddenRefRow: [Float] = []

            for t in 0..<T {
                let input = chained ? chain[t] : layerInput(L, t)
                let engineDense = (phases["prefillDense.\(t).\(L)"] ?? [])
                    .map { Self.toF32($0) }
                var denseRef: [Float]
                var hidden1: [Float]
                if isFull {
                    let x = Self.f16(Self.rms(input, weight: inNormW))
                    let qVec = Self.f16(Self.gemv(qProjW!, x))
                    var qn = [Float](repeating: 0, count: numQ * HDf)
                    var gate = [Float](repeating: 0, count: numQ * HDf)
                    for h in 0..<numQ {
                        let qSlice = Array(qVec[(h * 2 * HDf)..<(h * 2 * HDf + HDf)])
                        let gSlice = Array(qVec[(h * 2 * HDf + HDf)..<(h * 2 * HDf + 2 * HDf)])
                        let qnH = Self.rms(qSlice, weight: qNW)
                        for i in 0..<HDf {
                            qn[h * HDf + i] = qnH[i]
                            gate[h * HDf + i] = gSlice[i]
                        }
                        Self.ropePairs(&qn, headBase: h * HDf,
                                       rotaryDim: Int(Double(HDf) * cfg.partialRotaryFactor),
                                       position: t, theta: Float(cfg.fullRopeTheta))
                    }
                    // Engine KV rows (already normalized + rotated at each p).
                    var kRows = [[Float]]()
                    var vRows = [[Float]]()
                    for p in 0...t {
                        kRows.append((phases["pfK.\(p).\(L)"] ?? []).map { Self.toF32($0) })
                        vRows.append((phases["pfV.\(p).\(L)"] ?? []).map { Self.toF32($0) })
                    }
                    let attnScale = 1.0 / Float(HDf).squareRoot()
                    var attnOut = [Float](repeating: 0, count: numQ * HDf)
                    for h in 0..<numQ {
                        let kvh = h / (numQ / numKV)
                        for i in 0..<HDf {
                            var scores = [Float](repeating: 0, count: t + 1)
                            var mx: Float = -.infinity
                            for p in 0...t {
                                var s: Float = 0
                                for d in 0..<HDf {
                                    s += qn[h * HDf + d] * kRows[p][kvh * HDf + d]
                                }
                                s *= attnScale
                                scores[p] = s
                                mx = max(mx, s)
                            }
                            var sm: Float = 0
                            for p in 0...t {
                                scores[p] = expf(scores[p] - mx)
                                sm += scores[p]
                            }
                            var o: Float = 0
                            for p in 0...t {
                                o += (scores[p] / sm) * vRows[p][kvh * HDf + i]
                            }
                            let sgm = 1.0 / (1.0 + expf(-gate[h * HDf + i]))
                            attnOut[h * HDf + i] = Float(Float16(o * sgm))
                        }
                    }
                    let o = Self.f16(Self.gemv(oProjW!, attnOut))
                    hidden1 = Self.f16(zip(input, o).map { $0 + $1 })
                    denseRef = Self.f16(Self.rms(hidden1, weight: postW))
                } else {
                    let x = Self.f16(Self.rms(input, weight: inNormW))
                    let qkv = Self.f16(Self.gemv(qkvW!, x))
                    let z = Self.f16(Self.gemv(zW!, x))
                    let a = Self.f16(Self.gemv(aW!, x))
                    let b = Self.f16(Self.gemv(bW!, x))
                    var conv = [Float](repeating: 0, count: qkvDim)
                    for c in 0..<qkvDim {
                        let acc = convW[c * 4 + 0] * convState[c * 3 + 0]
                            + convW[c * 4 + 1] * convState[c * 3 + 1]
                            + convW[c * 4 + 2] * convState[c * 3 + 2]
                            + convW[c * 4 + 3] * qkv[c]
                        conv[c] = acc / (1.0 + expf(-acc))
                    }
                    conv = Self.f16(conv)
                    for c in 0..<qkvDim {
                        convState[c * 3 + 0] = convState[c * 3 + 1]
                        convState[c * 3 + 1] = convState[c * 3 + 2]
                        convState[c * 3 + 2] = qkv[c]
                    }
                    var rec = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        let kh = hv / 2
                        let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
                        let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                        let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                        var qss: Float = 0, kss: Float = 0
                        for i in 0..<HD {
                            qss += (qHead[i] * scale) * (qHead[i] * scale)
                            kss += kHead[i] * kHead[i]
                        }
                        let qinv = 1.0 / (qss + 1e-6).squareRoot()
                        let kinv = 1.0 / (kss + 1e-6).squareRoot()
                        let qn = qHead.map { $0 * scale * qinv }
                        let kn = kHead.map { $0 * kinv }
                        let g = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
                        let beta = 1.0 / (1.0 + expf(-b[hv]))
                        let decay = expf(g)
                        let sBase = hv * HD * HD
                        for i in 0..<(HD * HD) { recState[sBase + i] *= decay }
                        var r = [Float](repeating: 0, count: HD)
                        for vIdx in 0..<HD {
                            var acc: Float = 0
                            let row = vIdx * HD
                            for kk in 0..<HD { acc += recState[sBase + row + kk] * kn[kk] }
                            r[vIdx] = acc
                        }
                        var knqn: Float = 0
                        for i in 0..<HD { knqn += kn[i] * qn[i] }
                        for vIdx in 0..<HD {
                            let delta = beta * (vHead[vIdx] - r[vIdx])
                            let row = vIdx * HD
                            for kk in 0..<HD {
                                recState[sBase + row + kk] += kn[kk] * delta
                            }
                        }
                        for vIdx in 0..<HD {
                            var oVal: Float = 0
                            let row = vIdx * HD
                            for kk in 0..<HD {
                                oVal += recState[sBase + row + kk] * qn[kk]
                            }
                            rec[hv * HD + vIdx] = oVal
                        }
                    }
                    rec = Self.f16(rec)
                    var gated = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        var ss: Float = 0
                        for i in 0..<HD { ss += rec[hv * HD + i] * rec[hv * HD + i] }
                        let inv = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
                        for i in 0..<HD {
                            let zVal = z[hv * HD + i]
                            let silu = zVal / (1.0 + expf(-zVal))
                            gated[hv * HD + i] = rec[hv * HD + i] * inv * normW[i] * silu
                        }
                    }
                    gated = Self.f16(gated)
                    let attn = Self.f16(Self.gemv(outW!, gated))
                    hidden1 = Self.f16(zip(input, attn).map { $0 + $1 })
                    denseRef = Self.f16(Self.rms(hidden1, weight: postW))
                }
                var dMax: Float = 0
                for i in 0..<D { dMax = max(dMax, abs(engineDense[i] - denseRef[i])) }
                worstDense = max(worstDense, dMax)

                if chained || t == T - 1 {
                    // fp32 MoE tail on the REPLAYED dense row: router →
                    // top-8 renorm, shared expert + sigmoid gate, routed
                    // experts (packed file), all per the hunt's replay.
                    // (Every row in chained mode — the tail feeds the chain.)
                    let routerLogits = Self.gemv(routerW, denseRef)
                    var mxR = routerLogits.max() ?? 0
                    var sumR: Float = 0
                    for i in 0..<numExperts { sumR += expf(routerLogits[i] - mxR) }
                    let inv = 1.0 / sumR
                    let probs = routerLogits.map { expf($0 - mxR) * inv }
                    let order = probs.indices.sorted { probs[$0] > probs[$1] }
                    let topIDs = Array(order[0..<topK])
                    let topW = topIDs.map { probs[$0] }
                    var wSum: Float = 0
                    for wv in topW { wSum += wv }
                    let renorm = topW.map { $0 / wSum }

                    let gateVec = Self.f16(Self.gemv(sGateW, denseRef))
                    let upVec = Self.f16(Self.gemv(sUpW, denseRef))
                    var act = [Float](repeating: 0, count: I)
                    for i in 0..<I {
                        let silu = gateVec[i] / (1.0 + expf(-gateVec[i]))
                        act[i] = Float(Float16(silu * upVec[i]))
                    }
                    var h1 = Self.f16(Self.gemv(sDownW, act))
                    var gateDot: Float = 0
                    for i in 0..<D { gateDot += sharedGateW[i] * denseRef[i] }
                    let gateScale = 1.0 / (1.0 + expf(-gateDot))
                    h1 = h1.map { Float(Float16($0 * gateScale)) }

                    var h2 = [Float](repeating: 0, count: D)
                    for (slot, expertID) in topIDs.enumerated() {
                        let exp = layout.expert(layer: L, expert: expertID)
                        let gateRows = expertRows(exp, "gate", moeI, D)
                        let upRows = expertRows(exp, "up", moeI, D)
                        let downRows = expertRows(exp, "down", D, moeI)
                        let eGate = Self.f16(Self.gemv(gateRows, denseRef))
                        let eUp = Self.f16(Self.gemv(upRows, denseRef))
                        var eAct = [Float](repeating: 0, count: moeI)
                        for i in 0..<moeI {
                            let silu = eGate[i] / (1.0 + expf(-eGate[i]))
                            eAct[i] = Float(Float16(silu * eUp[i]))
                        }
                        let eOut = Self.f16(Self.gemv(downRows, eAct))
                        for i in 0..<D { h2[i] += renorm[slot] * eOut[i] }
                    }
                    for i in 0..<D { h2[i] = Float(Float16(h2[i] + h1[i])) }
                    hiddenRefRow = zip(hidden1, h2).map { Float(Float16($0 + $1)) }
                    if chained { chainOut[t] = hiddenRefRow }

                    if t == T - 1 {
                        let engineHidden = (phases["prefillHidden.\(t).\(L)"] ?? [])
                            .map { Self.toF32($0) }
                        var hMax: Float = 0
                        for i in 0..<D {
                            hMax = max(hMax, abs(engineHidden[i] - hiddenRefRow[i]))
                        }
                        worstHidden = hMax
                    }
                }
                // FQ_HEAD=1: at the last layer, replay the final head (final
                // rmsnorm + full-vocab int4 lm_head GEMV) on the replay's
                // row-4 hidden and compare against the engine's logits buffer.
                if chained && L == cfg.numLayers - 1 && t == T - 1
                    && ProcessInfo.processInfo.environment["FQ_HEAD"] != nil {
                    let fNormW = Self.bf16Values(model.finalNorm, count: D)
                    let xh = Self.f16(Self.rms(hiddenRefRow, weight: fNormW))
                    let lm = model.lmHead
                    let base = lm.buffer.contents()
                    let wBytes = base.advanced(by: Int(lm.offset))
                    let sWords = base.advanced(by: Int(lm.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    let bWords = base.advanced(by: Int(lm.biasOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    let vocab = cfg.vocabSize
                    let groups = D / 64
                    let lg = logits.contents().bindMemory(to: Float16.self,
                                                          capacity: vocab)
                    var idealTop: [(Int, Float)] = []
                    var maxDiff: Float = 0
                    var diffAt = -1
                    for r in 0..<vocab {
                        var acc: Float = 0
                        for g in 0..<groups {
                            let scl = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                            let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                            let byteBase = r * (D / 2) + g * 32
                            for k in 0..<64 {
                                let byte = wBytes.load(fromByteOffset: byteBase + k / 2,
                                                       as: UInt8.self)
                                let nibble = (k & 1) == 0 ? Int(byte & 0x0F)
                                    : Int(byte >> 4)
                                acc += (Float(nibble) * scl + bias) * xh[g * 64 + k]
                            }
                        }
                        let d = abs(acc - Float(lg[r]))
                        if d > maxDiff { maxDiff = d; diffAt = r }
                        if idealTop.count < 6
                            || acc > idealTop.last!.1 {
                            idealTop.append((r, acc))
                            idealTop.sort { $0.1 > $1.1 }
                            if idealTop.count > 6 { idealTop.removeLast() }
                        }
                    }
                    var engTop: [(Int, Float)] = []
                    for r in 0..<vocab {
                        let v = Float(lg[r])
                        if engTop.count < 6 || v > engTop.last!.1 {
                            engTop.append((r, v))
                            engTop.sort { $0.1 > $1.1 }
                            if engTop.count > 6 { engTop.removeLast() }
                        }
                    }
                    let piece = { (i: Int) -> String in
                        (try? tokenizer.decode([Int32(i)], skipSpecialTokens: false))
                            ?? "?" }
                    say("HEAD: ideal row4 top="
                        + idealTop.map { "\($0.0)[\(piece($0.0))]:\(String(format: "%.2f", $0.1))" }
                            .joined(separator: " ")
                        + " engine top="
                        + engTop.map { "\($0.0)[\(piece($0.0))]:\(String(format: "%.2f", $0.1))" }
                            .joined(separator: " ")
                        + " maxAbs=\(maxDiff) at id \(diffAt)")
                }
            }
            if chained { chain = chainOut }
            say("probe L\(L): dense worst=\(worstDense) hidden(row\(T - 1))="
                + "\(worstHidden) route=[\(phases["pfRouteIDs.\(L)"]?.prefix(8).map { Int($0) } ?? [])]")
            if !chained && worstHidden > 0.25 {
                firstDivergence = "L\(L) hidden=\(worstHidden) dense=\(worstDense)"
                break
            }
            if !chained && firstDivergence == nil && worstDense > 0.1 {
                firstDivergence = "L\(L) dense=\(worstDense) hidden=\(worstHidden)"
                break
            }
        }
        let verdict = firstDivergence
            ?? "none — every layer's mixer+MoE tail matches"
            + (chained ? " (cumulative row-4 drift by layer above)" : "")
        say("TAIL VERDICT: \(verdict)")
    }

    /// Applies Qwen's partial RoPE (HF half-split pairs over the first
    /// `rotaryDim` elements of one head: pair i mixes (i, i + rotaryDim/2)
    /// with inv_freq_i = theta^(-2i/rotaryDim)) at `position`, in place.
    private static func ropePairs(_ x: inout [Float], headBase: Int,
                                  rotaryDim: Int, position: Int,
                                  theta: Float) {
        let half = rotaryDim / 2
        for pair in 0..<half {
            let exponent = -Float(2 * pair) / Float(rotaryDim)
            let freq = powf(theta, exponent)
            let angle = Float(position) * freq
            let c = cosf(angle), sn = sinf(angle)
            let i0 = headBase + pair
            let i1 = headBase + half + pair
            let x0 = x[i0], x1 = x[i1]
            x[i0] = x0 * c - x1 * sn
            x[i1] = x0 * sn + x1 * c
        }
    }
}

extension QwenLayer0DebugTests {

    /// Repack-fidelity gate: dequantizes install tensors and compares them
    /// against the ORIGINAL bf16 checkpoint shards (parsed directly — the
    /// kernel-vs-reference tests all read the same install bytes, so they
    /// cannot catch a repack mapping bug). Covers the name remap, the affine
    /// quantizer, the (1+w) norm baking, and one expert's gate/up split.
    @Test(.enabled(if: installExists))
    func repackWeightsMatchBf16Checkpoint() throws {
        let ckpt = "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-bf16"
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: MetalContext().device,
            expecting: .qwen3_6_35B_A3B)

        func shard(_ name: String) throws -> Data {
            try Data(contentsOf: URL(fileURLWithPath: ckpt + "/" + name),
                     options: .mappedIfSafe)
        }
        /// Reads one tensor from one shard as [Float] (bf16→fp32 bit trick).
        func readBF16(_ shardName: String, _ tensorName: String) throws -> [Float] {
            let data = try shard(shardName)
            let n = Int(data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
            let header = try JSONSerialization.jsonObject(
                with: data[8..<8 + n]) as! [String: Any]
            let info = try #require(header[tensorName] as? [String: Any])
            let offs = try #require(info["data_offsets"] as? [Int])
            let raw = data[(8 + n + offs[0])..<(8 + n + offs[1])]
            return raw.withUnsafeBytes { ptr in
                let words = ptr.bindMemory(to: UInt16.self)
                return (0..<(raw.count / 2)).map {
                    FinchQuantization.bf16ToFloat(words[$0])
                }
            }
        }
        func compare(_ name: String, _ got: [Float], _ ref: [Float]) {
            var maxAbs: Float = 0
            var worst = -1
            var meanAbs: Float = 0
            for i in 0..<min(got.count, ref.count) {
                let d = abs(got[i] - ref[i])
                meanAbs += d / Float(min(got.count, ref.count))
                if d > maxAbs { maxAbs = d; worst = i }
            }
            print("\(name): maxAbs=\(maxAbs) at \(worst): got=\(worst >= 0 ? got[worst] : 0) ref=\(worst >= 0 ? ref[worst] : 0) meanAbs=\(meanAbs)")
        }

        // 1. in_proj_qkv [8192, 2048] — int4 affine round-trip.
        do {
            let ref = try readBF16("model-00001-of-00026.safetensors",
                                   "model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
            let view = try model.gdnInProjQKV(layer: 0)
            let got = Self.int4Rows(view, rows: 8192, cols: 2048).flatMap { $0 }
            compare("in_proj_qkv[0]", got, ref)
        }
        // 2. conv1d [8192, 4] — bf16→fp16 copy.
        do {
            let ref = try readBF16("model-00002-of-00026.safetensors",
                                   "model.language_model.layers.0.linear_attn.conv1d.weight")
            let view = try model.gdnConv1D(layer: 0)
            let got = Self.fp16Values(view, count: ref.count)
            compare("conv1d[0]", got, ref)
        }
        // 3. A_log [32] — bf16→fp32 copy.
        do {
            let ref = try readBF16("model-00002-of-00026.safetensors",
                                   "model.language_model.layers.0.linear_attn.A_log")
            let view = try model.gdnALog(layer: 0)
            let got = Self.fp32Values(view, count: ref.count)
            compare("A_log[0]", got, ref)
        }
        // 4. input_layernorm [2048] — baked (1+w).
        do {
            let ref = try readBF16("model-00002-of-00026.safetensors",
                                   "model.language_model.layers.0.input_layernorm.weight")
            let view = try model.inputNorm(layer: 0)
            let got = Self.bf16Values(view, count: ref.count)
            compare("inputNorm[0]", got, ref.map { 1.0 + $0 })
        }
        // 4b. Final norm (model.language_model.norm) — the lm_head input
        // norm; must be RAW (a plain RMSNorm weight, not baked).
        do {
            let ref = try readBF16("model-00026-of-00026.safetensors",
                                   "model.language_model.norm.weight")
            let view = model.finalNorm
            let got = Self.bf16Values(view, count: ref.count)
            compare("finalNorm (vs raw)", got, ref)
            compare("finalNorm (vs 1+w)", got, ref.map { 1.0 + $0 })
        }
        // 4c. post_attention_layernorm of a full layer — raw vs baked.
        do {
            let ref = try readBF16("model-00003-of-00026.safetensors",
                                   "model.language_model.layers.3.post_attention_layernorm.weight")
            let view = try model.postAttnNorm(layer: 3)
            let got = Self.bf16Values(view, count: ref.count)
            compare("postAttnNorm[3] (vs raw)", got, ref)
            compare("postAttnNorm[3] (vs 1+w)", got, ref.map { 1.0 + $0 })
        }
        // 4d. q_norm of a full layer.
        do {
            let ref = try readBF16("model-00003-of-00026.safetensors",
                                   "model.language_model.layers.3.self_attn.q_norm.weight")
            let view = try model.qNorm(layer: 3)
            let got = Self.bf16Values(view, count: ref.count)
            compare("qNorm[3] (vs raw)", got, ref)
            compare("qNorm[3] (vs 1+w)", got, ref.map { 1.0 + $0 })
        }
        // 5. lm_head rows 0..7 [248320, 2048].
        do {
            let ref = try readBF16("model-00026-of-00026.safetensors", "lm_head.weight")
            let view = model.lmHead
            let got = Self.int4Rows(view, rows: 8, cols: 2048).flatMap { $0 }
            compare("lm_head[0..8]", got, Array(ref[0..<(8 * 2048)]))
        }
        // 5b. Embedding rows for the actual prompt ids [760, 6511, 314,
        // 9338, 369] ("The capital of France is") — the model's input
        // vectors, against the checkpoint's embed_tokens.
        do {
            let ref = try readBF16("model-00001-of-00026.safetensors",
                                   "model.language_model.embed_tokens.weight")
            let view = model.embedding
            for tid in [760, 6511, 314, 9338, 369] {
                let got = Self.int4Rows(view, rows: tid + 1, cols: 2048)[tid]
                compare("embed[\(tid)]",
                        got, Array(ref[(tid * 2048)..<((tid + 1) * 2048)]))
            }
        }
        // 6. Router (mlp.gate) [256, 2048] — int8 affine.
        do {
            let ref = try readBF16("model-00002-of-00026.safetensors",
                                   "model.language_model.layers.0.mlp.gate.weight")
            let view = try model.router(layer: 0)
            let got = Self.int8Rows(view, rows: 256, cols: 2048).flatMap { $0 }
            compare("router[0]", got, ref)
        }
        // 6b. Shared expert (affects EVERY token) + its gate — layer 0.
        do {
            let gate = try readBF16("model-00002-of-00026.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight")
            let up = try readBF16("model-00002-of-00026.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight")
            let down = try readBF16("model-00002-of-00026.safetensors",
                "model.language_model.layers.0.mlp.shared_expert.down_proj.weight")
            let gateW = try readBF16("model-00002-of-00026.safetensors",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight")
            compare("sharedExpert[0].gate",
                    Self.int4Rows(try model.sharedExpertGate(layer: 0),
                                  rows: model.config.intermediateSize,
                                  cols: model.config.hiddenSize).flatMap { $0 },
                    gate)
            compare("sharedExpert[0].up",
                    Self.int4Rows(try model.sharedExpertUp(layer: 0),
                                  rows: model.config.intermediateSize,
                                  cols: model.config.hiddenSize).flatMap { $0 },
                    up)
            compare("sharedExpert[0].down",
                    Self.int4Rows(try model.sharedExpertDown(layer: 0),
                                  rows: model.config.hiddenSize,
                                  cols: model.config.intermediateSize).flatMap { $0 },
                    down)
            compare("sharedExpertGate[0]",
                    Self.int4Rows(try model.sharedExpertGateProj(layer: 0),
                                  rows: 1, cols: model.config.hiddenSize).flatMap { $0 },
                    gateW)
        }
        // 7. Expert 0 gate/up/down vs the fused checkpoint tensors.
        do {
            let gateRef = try readBF16("model-00001-of-00026.safetensors",
                "model.language_model.layers.0.mlp.experts.gate_up_proj")
            let downRef = try readBF16("model-00002-of-00026.safetensors",
                "model.language_model.layers.0.mlp.experts.down_proj")
            let layout = model.packedExpertsLayout
            let l0 = layout.layers[0]
            let fileData = try Data(contentsOf: URL(fileURLWithPath: Self.installPath)
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(l0.file))
            let exp = layout.expert(layer: 0, expert: 0)
            func dequant(_ role: String, _ r: Int, _ c: Int) -> [Float] {
                let w = exp.subTensors[role]!
                let sT = exp.subTensors[role + "_scales"]!
                let bT = exp.subTensors[role + "_biases"]!
                let wBytes = [UInt8](fileData.subdata(
                    in: Data.Index(w.offset)..<Data.Index(w.offset + w.size)))
                let sWords = fileData.subdata(
                    in: Data.Index(sT.offset)..<Data.Index(sT.offset + sT.size))
                let bWords = fileData.subdata(
                    in: Data.Index(bT.offset)..<Data.Index(bT.offset + bT.size))
                let sVals = sWords.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt16.self))
                }
                let bVals = bWords.withUnsafeBytes { raw in
                    Array(raw.bindMemory(to: UInt16.self))
                }
                var out = [Float](repeating: 0, count: r * c)
                let groups = c / 64
                for row in 0..<r {
                    for g in 0..<groups {
                        let scale = FinchQuantization.bf16ToFloat(sVals[row * groups + g])
                        let bias = FinchQuantization.bf16ToFloat(bVals[row * groups + g])
                        let byteBase = row * (c / 2) + g * 32
                        for k in 0..<64 {
                            let byte = wBytes[byteBase + k / 2]
                            let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                            out[row * c + g * 64 + k] = Float(nibble) * scale + bias
                        }
                    }
                }
                return out
            }
            // Expert 0 gate rows [0,512) and up rows [512,1024) of the fused
            // gate_up_proj [256, 1024, 2048] — flattened rows 0..511 = gate,
            // 512..1023 = up.
            compare("expert0.gate", dequant("gate", 512, 2048),
                    Array(gateRef[0..<(512 * 2048)]))
            compare("expert0.up", dequant("up", 512, 2048),
                    Array(gateRef[(512 * 2048)..<(1024 * 2048)]))
            compare("expert0.down", dequant("down", 2048, 512),
                    Array(downRef[0..<(2048 * 512)]))
        }
    }
}

extension QwenLayer0DebugTests {

    /// Multi-step decode continuity: after the validated prefill, drives 6
    /// decode steps and replays layer 0 (GDN) and layer 3 (full attention)
    /// with references that maintain THEIR OWN conv/recurrent/KV states from
    /// the validated prefill state — so a state-update bug in the engine
    /// (which per-step isolated checks cannot see) shows up as growing
    /// divergence. Prints per-step maxAbs for the recurrent output, the
    /// recurrent state, and (layer 3) the attention output.
    @Test(.enabled(if: installExists))
    func multiStepStateContinuity() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is",
                                         addBOS: false)
        let T = promptIds.count
        // Decode tokens: " Paris" (whatever it encodes to) + more text.
        let genText = " Paris is the capital of France. "
        let genIds = tokenizer.encode(genText, addBOS: false)
        let steps = min(6, genIds.count)

        // The hook cannot know which step it is in — a manual counter.
        let counter = StepCounter()
        runner.qwenLayerDebugHook = { layer, phase, values in
            counter.capture(step: counter.currentStep,
                            layer: layer, phase: phase, values: values)
        }
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<T], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })
        for k in 0..<steps {
            try await runner.produce(token: genIds[k], position: T + k,
                                     into: logits)
            counter.currentStep += 1
        }

        let D = model.config.hiddenSize
        let V = model.config.linearNumValueHeads
        let HD = model.config.linearValueHeadDim
        let keyDim = model.config.linearNumKeyHeads * HD
        let valueDim = V * HD
        let qkvDim = 2 * keyDim + valueDim
        let scale = 1.0 / Float(HD).squareRoot()
        let HDf = model.config.fullHeadDim
        let numQ = model.config.numHeads
        let numKV = model.config.numFullKVHeads
        let rotaryDim = Int(Double(HDf) * model.config.partialRotaryFactor)
        let theta = Float(model.config.fullRopeTheta)
        let embedView = model.embedding
        func embed(_ id: Int32) -> [Float] {
            let row = Self.int4Rows(embedView, rows: Int(id) + 1, cols: D)[Int(id)]
            return Self.f16(row)
        }

        // Layer 0 reference state, seeded from the engine's prefill
        // snapshots (captured under step 0 by the counter hook).
        let pfConv = (counter.value(step: 0, layer: 0, phase: "pfConvState") ?? [])
            .map { Self.toF32($0) }
        let pfState = (counter.value(step: 0, layer: 0, phase: "pfState") ?? [])
            .map { Self.toF32($0) }
        var refConvState = pfConv.count == qkvDim * 3
            ? pfConv : [Float](repeating: 0, count: qkvDim * 3)
        var refRecState = pfState.count == V * HD * HD
            ? pfState : [Float](repeating: 0, count: V * HD * HD)

        let inW0 = Self.bf16Values(try model.inputNorm(layer: 0), count: D)
        let qkvW0 = Self.int4Rows(try model.gdnInProjQKV(layer: 0),
                                  rows: qkvDim, cols: D)
        let zW0 = Self.int4Rows(try model.gdnInProjZ(layer: 0),
                                rows: valueDim, cols: D)
        let aW0 = Self.int4Rows(try model.gdnInProjA(layer: 0), rows: V, cols: D)
        let bW0 = Self.int4Rows(try model.gdnInProjB(layer: 0), rows: V, cols: D)
        let outW0 = Self.int4Rows(try model.gdnOutProj(layer: 0),
                                  rows: D, cols: valueDim)
        let convW0 = Self.fp16Values(try model.gdnConv1D(layer: 0),
                                     count: qkvDim * 4)
        let aLog0 = Self.fp32Values(try model.gdnALog(layer: 0), count: V)
        let dt0 = Self.fp32Values(try model.gdnDtBias(layer: 0), count: V)
        let normW0 = Self.bf16Values(try model.gdnNormWeight(layer: 0),
                                     count: HD)
        let postW0 = Self.bf16Values(try model.postAttnNorm(layer: 0),
                                     count: D)

        // Layer 3 reference state: KV rows for prefill tokens.
        let inW3 = Self.bf16Values(try model.inputNorm(layer: 3), count: D)
        let qW3 = Self.int4Rows(try model.qProj(layer: 3),
                                rows: 2 * numQ * HDf, cols: D)
        let kW3 = Self.int4Rows(try model.kProj(layer: 3),
                                rows: numKV * HDf, cols: D)
        let vW3 = Self.int4Rows(try model.vProj(layer: 3),
                                rows: numKV * HDf, cols: D)
        let oW3 = Self.int4Rows(try model.oProj(layer: 3), rows: D,
                                cols: numQ * HDf)
        let qN3 = Self.bf16Values(try model.qNorm(layer: 3), count: HDf)
        let kN3 = Self.bf16Values(try model.kNorm(layer: 3), count: HDf)
        let postW3 = Self.bf16Values(try model.postAttnNorm(layer: 3),
                                     count: D)
        // Prefill KV rows for layer 3 from the reference's own chain.
        var refK: [[Float]] = []
        var refV: [[Float]] = []
        for p in 0..<T {
            let l3in = (counter.value(step: 0, layer: 2,
                                      phase: "prefillHidden.\(p)") ?? [])
                .map { Self.toF32($0) }
            let x = Self.f16(Self.rms(l3in, weight: inW3))
            let kVec = Self.f16(Self.gemv(kW3, x))
            let vVec = Self.f16(Self.gemv(vW3, x))
            var kn = [Float](repeating: 0, count: numKV * HDf)
            for h in 0..<numKV {
                let kSlice = Array(kVec[(h * HDf)..<(h * HDf + HDf)])
                let knH = Self.rms(kSlice, weight: kN3)
                for i in 0..<HDf { kn[h * HDf + i] = knH[i] }
                Self.ropePairs(&kn, headBase: h * HDf, rotaryDim: rotaryDim,
                               position: p, theta: theta)
            }
            refK.append(kn)
            refV.append(vVec)
        }

        var worstL0: Float = 0
        var worstL3: Float = 0
        for k in 0..<steps {
            // ---- Layer 0 (GDN): full mixer with the reference's own state.
            let input0 = (counter.value(step: k, layer: 0, phase: "preLayer")
                ?? []).map { Self.toF32($0) }
            if k == 0 {
                var m: Float = 0
                for i in 0..<D { m = max(m, abs(input0[i] - embed(genIds[0])[i])) }
                print("step\(k) L0 input vs embed: maxAbs=\(m)")
            }
            let x0 = Self.f16(Self.rms(input0, weight: inW0))
            let qkv = Self.f16(Self.gemv(qkvW0, x0))
            let z = Self.f16(Self.gemv(zW0, x0))
            let a = Self.f16(Self.gemv(aW0, x0))
            let b = Self.f16(Self.gemv(bW0, x0))
            var conv = [Float](repeating: 0, count: qkvDim)
            for c in 0..<qkvDim {
                let acc = convW0[c * 4 + 0] * refConvState[c * 3 + 0]
                    + convW0[c * 4 + 1] * refConvState[c * 3 + 1]
                    + convW0[c * 4 + 2] * refConvState[c * 3 + 2]
                    + convW0[c * 4 + 3] * qkv[c]
                conv[c] = acc / (1.0 + expf(-acc))
            }
            conv = Self.f16(conv)
            for c in 0..<qkvDim {
                refConvState[c * 3 + 0] = refConvState[c * 3 + 1]
                refConvState[c * 3 + 1] = refConvState[c * 3 + 2]
                refConvState[c * 3 + 2] = qkv[c]
            }
            var rec = [Float](repeating: 0, count: valueDim)
            for hv in 0..<V {
                let kh = hv / 2
                let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
                let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                var qss: Float = 0, kss: Float = 0
                for i in 0..<HD {
                    qss += (qHead[i] * scale) * (qHead[i] * scale)
                    kss += kHead[i] * kHead[i]
                }
                let qinv = 1.0 / (qss + 1e-6).squareRoot()
                let kinv = 1.0 / (kss + 1e-6).squareRoot()
                let qn = qHead.map { $0 * scale * qinv }
                let kn = kHead.map { $0 * kinv }
                let g = -expf(aLog0[hv]) * GDNRef.softplus(a[hv] + dt0[hv])
                let beta = 1.0 / (1.0 + expf(-b[hv]))
                let decay = expf(g)
                let sBase = hv * HD * HD
                for i in 0..<(HD * HD) { refRecState[sBase + i] *= decay }
                var r = [Float](repeating: 0, count: HD)
                for vIdx in 0..<HD {
                    var acc: Float = 0
                    let row = vIdx * HD
                    for kk in 0..<HD { acc += refRecState[sBase + row + kk] * kn[kk] }
                    r[vIdx] = acc
                }
                var knqn: Float = 0
                for i in 0..<HD { knqn += kn[i] * qn[i] }
                for vIdx in 0..<HD {
                    let delta = beta * (vHead[vIdx] - r[vIdx])
                    let row = vIdx * HD
                    for kk in 0..<HD {
                        refRecState[sBase + row + kk] += kn[kk] * delta
                    }
                    var oVal: Float = 0
                    for kk in 0..<HD {
                        oVal += refRecState[sBase + row + kk] * qn[kk]
                    }
                    rec[hv * HD + vIdx] = oVal
                }
            }
            rec = Self.f16(rec)
            let recEngine = (counter.value(step: k, layer: 0,
                                           phase: "recurrentOut") ?? [])
                .map { Self.toF32($0) }
            var mRec: Float = 0
            for i in 0..<valueDim { mRec = max(mRec, abs(recEngine[i] - rec[i])) }
            let stEngine = (counter.value(step: k, layer: 0,
                                           phase: "recState") ?? [])
                .map { Self.toF32($0) }
            var mSt: Float = 0
            for i in 0..<refRecState.count {
                mSt = max(mSt, abs(stEngine[i] - refRecState[i]))
            }
            print("step\(k) L0 recurrentOut: maxAbs=\(mRec) recState: maxAbs=\(mSt)")
            worstL0 = max(worstL0, mRec)

            // ---- Layer 3 (full attention) with the reference's own KV.
            let input3 = (counter.value(step: k, layer: 3, phase: "preLayer")
                ?? []).map { Self.toF32($0) }
            let x3 = Self.f16(Self.rms(input3, weight: inW3))
            let qVec = Self.f16(Self.gemv(qW3, x3))
            var qn = [Float](repeating: 0, count: numQ * HDf)
            var gate = [Float](repeating: 0, count: numQ * HDf)
            for h in 0..<numQ {
                let qSlice = Array(qVec[(h * 2 * HDf)..<(h * 2 * HDf + HDf)])
                let gSlice = Array(qVec[(h * 2 * HDf + HDf)..<(h * 2 * HDf + 2 * HDf)])
                let qnH = Self.rms(qSlice, weight: qN3)
                for i in 0..<HDf {
                    qn[h * HDf + i] = qnH[i]
                    gate[h * HDf + i] = gSlice[i]
                }
                Self.ropePairs(&qn, headBase: h * HDf, rotaryDim: rotaryDim,
                               position: T + k, theta: theta)
            }
            let kVec = Self.f16(Self.gemv(kW3, x3))
            let vVec = Self.f16(Self.gemv(vW3, x3))
            var kn = [Float](repeating: 0, count: numKV * HDf)
            for h in 0..<numKV {
                let kSlice = Array(kVec[(h * HDf)..<(h * HDf + HDf)])
                let knH = Self.rms(kSlice, weight: kN3)
                for i in 0..<HDf { kn[h * HDf + i] = knH[i] }
                Self.ropePairs(&kn, headBase: h * HDf, rotaryDim: rotaryDim,
                               position: T + k, theta: theta)
            }
            refK.append(kn)
            refV.append(vVec)
            let attnScale = 1.0 / Float(HDf).squareRoot()
            var attn = [Float](repeating: 0, count: numQ * HDf)
            for h in 0..<numQ {
                let kvh = h / (numQ / numKV)
                for i in 0..<HDf {
                    var scores = [Float](repeating: 0, count: refK.count)
                    var m: Float = -.infinity
                    for p in 0..<refK.count {
                        var s: Float = 0
                        for d in 0..<HDf {
                            s += qn[h * HDf + d] * refK[p][kvh * HDf + d]
                        }
                        s *= attnScale
                        scores[p] = s
                        m = max(m, s)
                    }
                    var sum: Float = 0
                    for p in 0..<refK.count {
                        scores[p] = expf(scores[p] - m)
                        sum += scores[p]
                    }
                    var o: Float = 0
                    for p in 0..<refK.count {
                        o += (scores[p] / sum) * refV[p][kvh * HDf + i]
                    }
                    let sgm = 1.0 / (1.0 + expf(-gate[h * HDf + i]))
                    attn[h * HDf + i] = Float(Float16(o * sgm))
                }
            }
            let attnEngine = (counter.value(step: k, layer: 3,
                                            phase: "recurrentOut") ?? [])
                .map { Self.toF32($0) }
            var mA: Float = 0
            for i in 0..<attn.count {
                mA = max(mA, abs(attnEngine[i] - attn[i]))
            }
            print("step\(k) L3 attnOut: maxAbs=\(mA)")
            worstL3 = max(worstL3, mA)
        }
        print("MULTI-STEP worst L0=\(worstL0) L3=\(worstL3)")
    }

    /// Thread-unsafe per-step capture for the multi-step test.
    private final class StepCounter: @unchecked Sendable {
        var currentStep = 0
        private var store: [String: [Float16]] = [:]
        func capture(step: Int, layer: Int, phase: String, values: [Float16]) {
            store["\(step).\(phase).\(layer)"] = values
        }
        func value(step: Int, layer: Int, phase: String) -> [Float16]? {
            store["\(step).\(phase).\(layer)"]
        }
    }
}

extension QwenLayer0DebugTests {

    /// The MoE-tail hunt: every earlier check validated each layer's MIXER in
    /// isolation given the engine's own post-MoE input rows — a broken MoE
    /// tail (router readback, shared expert, streamed routed tiles, h2/hidden
    /// combine) is invisible to those checks. This replays, per layer, the
    /// last prefill token's (t=4) full tail in fp32 from the engine's mixer
    /// output and compares the engine's post-MoE hidden row. Early-exits at
    /// the first layer exceeding 0.05.
    @Test(.enabled(if: installExists))
    func prefillMoETailHunt() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        let counter = StepCounter()
        runner.qwenLayerDebugHook = { layer, phase, values in
            counter.capture(step: counter.currentStep,
                            layer: layer, phase: phase, values: values)
        }
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is",
                                         addBOS: false)
        let T = promptIds.count
        let t = T - 1   // the last token — the one the first new token samples
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<T], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })

        let D = model.config.hiddenSize
        let I = model.config.intermediateSize
        let layout = model.packedExpertsLayout

        // Staged probe: [0] first (cheapest GDN layer) to bisect the tail
        // math; widen to [0, 3, 7] then the full 0..<numLayers once clean.
        let probeLayers = [0]
        for L in probeLayers {
            let isFull = model.config.fullAttentionLayerMask[L] == 1
            // Engine rows: layer input (post-MoE of L-1), mixer out, MoE out.
            // Layer 0's input is the embedding row for the token.
            let input: [Float]
            if L == 0 {
                let row = Self.int4Rows(model.embedding, rows: Int(promptIds[t]) + 1,
                                        cols: D)[Int(promptIds[t])]
                input = Self.f16(row)
            } else {
                input = (counter.value(step: 0, layer: L - 1,
                                       phase: "prefillHidden.\(t)") ?? [])
                    .map { Self.toF32($0) }
            }
            let dense = (counter.value(step: 0, layer: L,
                                       phase: "prefillDense.\(t)") ?? [])
                .map { Self.toF32($0) }
            let hiddenEngine = (counter.value(step: 0, layer: L,
                                              phase: "prefillHidden.\(t)") ?? [])
                .map { Self.toF32($0) }
            if dense.count < D || hiddenEngine.count < D {
                print("L\(L): missing snapshots — layer \(L) is full? \(isFull); skip")
                break
            }

            // ---- Reference MoE tail (dense → hidden).
            let routerView = try model.router(layer: L)
            let routerW = Self.int8Rows(routerView, rows: model.config.numExperts,
                                        cols: D)
            let routerLogits = Self.gemv(routerW, dense)
            var m = routerLogits.max() ?? 0
            var sum: Float = 0
            for i in 0..<routerLogits.count { sum += expf(routerLogits[i] - m) }
            let inv = 1.0 / sum
            let probs = routerLogits.map { expf($0 - m) * inv }
            let order = probs.indices.sorted { probs[$0] > probs[$1] }
            let topK = min(model.config.topKExperts, order.count)
            let topIDs = Array(order[0..<topK])
            let topW = topIDs.map { probs[$0] }
            var wSum: Float = 0
            for w in topW { wSum += w }
            let renorm = topW.map { $0 / wSum }

            // Shared expert (silu) + sigmoid gate.
            let sGateW = Self.int4Rows(try model.sharedExpertGate(layer: L),
                                       rows: I, cols: D)
            let sUpW = Self.int4Rows(try model.sharedExpertUp(layer: L),
                                     rows: I, cols: D)
            let sDownW = Self.int4Rows(try model.sharedExpertDown(layer: L),
                                       rows: D, cols: I)
            let gateVec = Self.f16(Self.gemv(sGateW, dense))
            let upVec = Self.f16(Self.gemv(sUpW, dense))
            var act = [Float](repeating: 0, count: I)
            for i in 0..<I {
                let silu = gateVec[i] / (1.0 + expf(-gateVec[i]))
                act[i] = Float(Float16(silu * upVec[i]))
            }
            var h1 = Self.f16(Self.gemv(sDownW, act))
            let sharedGateW = Self.int4Rows(try model.sharedExpertGateProj(layer: L),
                                            rows: 1, cols: D)[0]
            var gateDot: Float = 0
            for i in 0..<D { gateDot += sharedGateW[i] * dense[i] }
            let gateScale = 1.0 / (1.0 + expf(-gateDot))
            h1 = h1.map { Float(Float16($0 * gateScale)) }

            // Shared-expert act-formula variants (same rows): swap = silu(up)*gate,
            // prod = silu(gate*up).
            var h1Swap: [Float]
            var h1Prod: [Float]
            do {
                var s1Swap = [Float](repeating: 0, count: I)
                var s1Prod = [Float](repeating: 0, count: I)
                for i in 0..<I {
                    let sU = upVec[i] / (1.0 + expf(-upVec[i]))
                    s1Swap[i] = Float(Float16(sU * gateVec[i]))
                    let p = gateVec[i] * upVec[i]
                    s1Prod[i] = Float(Float16(p / (1.0 + expf(-p))))
                }
                h1Swap = Self.f16(Self.gemv(sDownW, s1Swap))
                    .map { Float(Float16($0 * gateScale)) }
                h1Prod = Self.f16(Self.gemv(sDownW, s1Prod))
                    .map { Float(Float16($0 * gateScale)) }
            }

            // Routed experts: range-read the packed tensors for the top-8.
            let l0 = layout.layers[L]
            let layerURL = URL(fileURLWithPath: Self.installPath)
                .appendingPathComponent("packed_experts")
                .appendingPathComponent(l0.file)
            let fh = try FileHandle(forReadingFrom: layerURL)
            defer { try? fh.close() }
            var h2 = [Float](repeating: 0, count: D)
            var h2Swap = [Float](repeating: 0, count: D)
            var h2Prod = [Float](repeating: 0, count: D)
            for (slot, expertID) in topIDs.enumerated() {
                let exp = layout.expert(layer: L, expert: expertID)
                func rows(_ role: String, _ r: Int, _ c: Int) -> [[Float]] {
                    let w = exp.subTensors[role]!
                    let sT = exp.subTensors[role + "_scales"]!
                    let bT = exp.subTensors[role + "_biases"]!
                    let wBytes = Self.readRange(fh, Int(w.offset), Int(w.size))
                    let sVals = Self.readRange(fh, Int(sT.offset), Int(sT.size))
                        .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    let bVals = Self.readRange(fh, Int(bT.offset), Int(bT.size))
                        .withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
                    var out = [[Float]](repeating: [], count: r)
                    let groups = c / 64
                    for row in 0..<r {
                        var vals = [Float](repeating: 0, count: c)
                        for g in 0..<groups {
                            let scale = FinchQuantization.bf16ToFloat(sVals[row * groups + g])
                            let bias = FinchQuantization.bf16ToFloat(bVals[row * groups + g])
                            let byteBase = row * (c / 2) + g * 32
                            for k in 0..<64 {
                                let byte = wBytes[byteBase + k / 2]
                                let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                                vals[g * 64 + k] = Float(nibble) * scale + bias
                            }
                        }
                        out[row] = vals
                    }
                    return out
                }
                let gateRows = rows("gate", model.config.moeIntermediateSize, D)
                let upRows = rows("up", model.config.moeIntermediateSize, D)
                let downRows = rows("down", D, model.config.moeIntermediateSize)
                let eGate = Self.f16(Self.gemv(gateRows, dense))
                let eUp = Self.f16(Self.gemv(upRows, dense))
                var eAct = [Float](repeating: 0, count: model.config.moeIntermediateSize)
                for i in 0..<eAct.count {
                    let silu = eGate[i] / (1.0 + expf(-eGate[i]))
                    eAct[i] = Float(Float16(silu * eUp[i]))
                }
                let eOut = Self.f16(Self.gemv(downRows, eAct))
                for i in 0..<D { h2[i] += renorm[slot] * eOut[i] }
                // Routed act-formula variants: swap = silu(up)*gate, prod = silu(gate*up).
                var swAct = [Float](repeating: 0,
                                    count: model.config.moeIntermediateSize)
                var prAct = [Float](repeating: 0,
                                    count: model.config.moeIntermediateSize)
                for i in 0..<eAct.count {
                    let sU = eUp[i] / (1.0 + expf(-eUp[i]))
                    swAct[i] = Float(Float16(sU * eGate[i]))
                    let p = eGate[i] * eUp[i]
                    prAct[i] = Float(Float16(p / (1.0 + expf(-p))))
                }
                let eOutSwap = Self.f16(Self.gemv(downRows, swAct))
                let eOutProd = Self.f16(Self.gemv(downRows, prAct))
                for i in 0..<D {
                    h2Swap[i] += renorm[slot] * eOutSwap[i]
                    h2Prod[i] += renorm[slot] * eOutProd[i]
                }
            }
            for i in 0..<D { h2[i] = Float(Float16(h2[i] + h1[i])) }
            // hidden = input + attn + h2. input + attn is the pre-norm hidden;
            // the engine's dense = rms(input + attn) — recover the residual
            // from the mixer: attn = mixer output. Recompute from the
            // reference mixer? The mixer was validated — use the engine's
            // own dense and norm weight to recover input+attn? Simpler: the
            // engine hidden BEFORE the tail = prefillHidden? No — that is
            // AFTER the tail. Recompute attn via the validated mixer for t.
            // To keep this test self-contained, rebuild the hidden1 from the
            // mixer replay (same code as the mixer sweep).
            var hiddenRef: [Float]
            do {
                let inNormW = Self.bf16Values(try model.inputNorm(layer: L),
                                              count: D)
                let postW = Self.bf16Values(try model.postAttnNorm(layer: L),
                                            count: D)
                let x = Self.f16(Self.rms(input, weight: inNormW))
                var attn: [Float]
                if isFull {
                    let HDf = model.config.fullHeadDim
                    let numQ = model.config.numHeads
                    let numKV = model.config.numFullKVHeads
                    let qDim = numQ * HDf
                    let rotaryDim = Int(Double(HDf) * model.config.partialRotaryFactor)
                    let theta = Float(model.config.fullRopeTheta)
                    let qVec = Self.f16(Self.gemv(
                        Self.int4Rows(try model.qProj(layer: L),
                                      rows: 2 * qDim, cols: D), x))
                    let qNW = Self.bf16Values(try model.qNorm(layer: L),
                                              count: HDf)
                    let kNW = Self.bf16Values(try model.kNorm(layer: L),
                                              count: HDf)
                    var qn = [Float](repeating: 0, count: qDim)
                    var gate = [Float](repeating: 0, count: qDim)
                    for h in 0..<numQ {
                        let qSlice = Array(qVec[(h * 2 * HDf)..<(h * 2 * HDf + HDf)])
                        let gSlice = Array(qVec[(h * 2 * HDf + HDf)..<(h * 2 * HDf + 2 * HDf)])
                        let qnH = Self.rms(qSlice, weight: qNW)
                        for i in 0..<HDf {
                            qn[h * HDf + i] = qnH[i]
                            gate[h * HDf + i] = gSlice[i]
                        }
                        Self.ropePairs(&qn, headBase: h * HDf,
                                       rotaryDim: rotaryDim, position: t,
                                       theta: theta)
                    }
                    // KV rows for this layer: engine snapshots (prefill).
                    var kRows = [[Float]]()
                    var vRows = [[Float]]()
                    for p in 0...t {
                        kRows.append((counter.value(step: 0, layer: L,
                                                    phase: "pfK.\(p)") ?? [])
                            .map { Self.toF32($0) })
                        vRows.append((counter.value(step: 0, layer: L,
                                                    phase: "pfV.\(p)") ?? [])
                            .map { Self.toF32($0) })
                    }
                    let attnScale = 1.0 / Float(HDf).squareRoot()
                    var attnOut = [Float](repeating: 0, count: qDim)
                    for h in 0..<numQ {
                        let kvh = h / (numQ / numKV)
                        for i in 0..<HDf {
                            var scores = [Float](repeating: 0, count: t + 1)
                            var mx: Float = -.infinity
                            for p in 0...t {
                                var s: Float = 0
                                for d in 0..<HDf {
                                    s += qn[h * HDf + d] * kRows[p][kvh * HDf + d]
                                }
                                s *= attnScale
                                scores[p] = s
                                mx = max(mx, s)
                            }
                            var sm: Float = 0
                            for p in 0...t {
                                scores[p] = expf(scores[p] - mx)
                                sm += scores[p]
                            }
                            var o: Float = 0
                            for p in 0...t {
                                o += (scores[p] / sm) * vRows[p][kvh * HDf + i]
                            }
                            let sgm = 1.0 / (1.0 + expf(-gate[h * HDf + i]))
                            attnOut[h * HDf + i] = Float(Float16(o * sgm))
                        }
                    }
                    attn = Self.f16(Self.gemv(
                        Self.int4Rows(try model.oProj(layer: L),
                                      rows: D, cols: qDim), attnOut))
                } else {
                    let V = model.config.linearNumValueHeads
                    let HD = model.config.linearValueHeadDim
                    let keyDim = model.config.linearNumKeyHeads * HD
                    let valueDim = V * HD
                    let qkvDim = 2 * keyDim + valueDim
                    let qkvW = Self.int4Rows(try model.gdnInProjQKV(layer: L),
                                             rows: qkvDim, cols: D)
                    let zW = Self.int4Rows(try model.gdnInProjZ(layer: L),
                                           rows: valueDim, cols: D)
                    let aW = Self.int4Rows(try model.gdnInProjA(layer: L),
                                           rows: V, cols: D)
                    let bW = Self.int4Rows(try model.gdnInProjB(layer: L),
                                           rows: V, cols: D)
                    let outW = Self.int4Rows(try model.gdnOutProj(layer: L),
                                             rows: D, cols: valueDim)
                    let convW = Self.fp16Values(try model.gdnConv1D(layer: L),
                                                count: qkvDim * 4)
                    let aLog = Self.fp32Values(try model.gdnALog(layer: L),
                                               count: V)
                    let dt = Self.fp32Values(try model.gdnDtBias(layer: L),
                                             count: V)
                    let normW = Self.bf16Values(try model.gdnNormWeight(layer: L),
                                                count: HD)
                    let qkv = Self.f16(Self.gemv(qkvW, x))
                    let z = Self.f16(Self.gemv(zW, x))
                    let a = Self.f16(Self.gemv(aW, x))
                    let b = Self.f16(Self.gemv(bW, x))
                    let pfConv = (counter.value(step: 0, layer: L,
                                                phase: "pfConvState") ?? [])
                        .map { Self.toF32($0) }
                    var conv = [Float](repeating: 0, count: qkvDim)
                    for c in 0..<qkvDim {
                        let acc = convW[c * 4 + 0] * pfConv[c * 3 + 0]
                            + convW[c * 4 + 1] * pfConv[c * 3 + 1]
                            + convW[c * 4 + 2] * pfConv[c * 3 + 2]
                            + convW[c * 4 + 3] * qkv[c]
                        conv[c] = acc / (1.0 + expf(-acc))
                    }
                    conv = Self.f16(conv)
                    let pfS = (counter.value(step: 0, layer: L,
                                             phase: "pfState") ?? [])
                        .map { Self.toF32($0) }
                    let scale = 1.0 / Float(HD).squareRoot()
                    var rec = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        let kh = hv / 2
                        let qHead = Array(conv[(kh * HD)..<(kh * HD + HD)])
                        let kHead = Array(conv[(keyDim + kh * HD)..<(keyDim + kh * HD + HD)])
                        let vHead = Array(conv[(2 * keyDim + hv * HD)..<(2 * keyDim + hv * HD + HD)])
                        var qss: Float = 0, kss: Float = 0
                        for i in 0..<HD {
                            qss += (qHead[i] * scale) * (qHead[i] * scale)
                            kss += kHead[i] * kHead[i]
                        }
                        let qinv = 1.0 / (qss + 1e-6).squareRoot()
                        let kinv = 1.0 / (kss + 1e-6).squareRoot()
                        let qn = qHead.map { $0 * scale * qinv }
                        let kn = kHead.map { $0 * kinv }
                        let g = -expf(aLog[hv]) * GDNRef.softplus(a[hv] + dt[hv])
                        let beta = 1.0 / (1.0 + expf(-b[hv]))
                        let decay = expf(g)
                        let sBase = hv * HD * HD
                        var knqn: Float = 0
                        for i in 0..<HD { knqn += kn[i] * qn[i] }
                        for vIdx in 0..<HD {
                            let row = vIdx * HD
                            var base: Float = 0
                            var r: Float = 0
                            for kk in 0..<HD {
                                let sv = pfS[sBase + row + kk] * decay
                                base += sv * qn[kk]
                                r += sv * kn[kk]
                            }
                            let delta = beta * (vHead[vIdx] - r)
                            rec[hv * HD + vIdx] = base + delta * knqn
                        }
                    }
                    rec = Self.f16(rec)
                    var gated = [Float](repeating: 0, count: valueDim)
                    for hv in 0..<V {
                        var ss: Float = 0
                        for i in 0..<HD { ss += rec[hv * HD + i] * rec[hv * HD + i] }
                        let invN = 1.0 / (ss / Float(HD) + 1e-6).squareRoot()
                        for i in 0..<HD {
                            let zVal = z[hv * HD + i]
                            let silu = zVal / (1.0 + expf(-zVal))
                            gated[hv * HD + i] = rec[hv * HD + i] * invN * normW[i] * silu
                        }
                    }
                    gated = Self.f16(gated)
                    attn = Self.f16(Self.gemv(outW, gated))
                }
                let hidden1 = Self.f16(zip(input, attn).map { $0 + $1 })
                hiddenRef = zip(hidden1, h2).map { Float(Float16($0 + $1)) }
                _ = postW
            }
            var maxAbs: Float = 0
            var worst = -1
            for i in 0..<D {
                let d = abs(hiddenEngine[i] - hiddenRef[i])
                if d > maxAbs { maxAbs = d; worst = i }
            }
            print("pfMoE L\(L): hidden maxAbs=\(maxAbs) at \(worst)")
            // Formula-variant bisect: which act formula makes engine == ref?
            // hidden1 = hiddenRef - full normal tail; each variant re-adds its
            // own routed sum + shared vector.
            var hidden1 = [Float](repeating: 0, count: D)
            var routedN = [Float](repeating: 0, count: D)
            var routedS = [Float](repeating: 0, count: D)
            var routedP = [Float](repeating: 0, count: D)
            for i in 0..<D {
                hidden1[i] = hiddenRef[i] - h2[i]
                routedN[i] = h2[i] - h1[i]
                routedS[i] = h2Swap[i] - h1Swap[i]
                routedP[i] = h2Prod[i] - h1Prod[i]
            }
            func variantScore(_ routed: [Float], _ shared: [Float]) -> Float {
                var mx: Float = 0
                for i in 0..<D {
                    let v = Float(Float16(hidden1[i]
                        + Float(Float16(routed[i] + shared[i]))))
                    mx = max(mx, abs(hiddenEngine[i] - v))
                }
                return mx
            }
            print("pfMoE L\(L): variants "
                + "normalR+normalS=\(variantScore(routedN, h1)) "
                + "swapR+swapS=\(variantScore(routedS, h1Swap)) "
                + "prodR+prodS=\(variantScore(routedP, h1Prod)) "
                + "swapR+normalS=\(variantScore(routedS, h1)) "
                + "normalR+swapS=\(variantScore(routedN, h1Swap)) "
                + "prodR+normalS=\(variantScore(routedP, h1)) "
                + "normalR+prodS=\(variantScore(routedN, h1Prod))")
            // Bisect: shared-only and routed-only mismatches.
            do {
                var sharedOnly: Float = 0
                var routedOnly: Float = 0
                for i in 0..<D {
                    let residual = hiddenEngine[i] - hiddenRef[i]
                    // h2 = h1(shared) + routed; shared-only mismatch uses
                    // h1 alone.
                    sharedOnly = max(sharedOnly, abs(residual + h2[i] - h1[i]))
                    routedOnly = max(routedOnly, abs(residual + h1[i]))
                }
                print("pfMoE L\(L): sharedOnly=\(sharedOnly) routedOnly=\(routedOnly)")
                let engIDs = (counter.value(step: 0, layer: L,
                                            phase: "pfRouteIDs") ?? [])
                    .map { UInt32(Self.toF32($0)) }
                let engW = (counter.value(step: 0, layer: L,
                                          phase: "pfRouteW") ?? [])
                    .map { Self.toF32($0) }
                print("pfMoE L\(L): engine route ids=\(engIDs) w=\(engW)")
                print("pfMoE L\(L): ref    route ids=\(topIDs) w=\(renorm)")
            }
            if maxAbs > 0.05 {
                print("FIRST MoE DIVERGENCE: L\(L) maxAbs=\(maxAbs) at \(worst)")
                break
            }
        }
    }

    private static func readRange(_ fh: FileHandle, _ offset: Int,
                                  _ size: Int) -> [UInt8] {
        try? fh.seek(toOffset: UInt64(offset))
        guard let data = try? fh.read(upToCount: size), data.count == size else {
            return []
        }
        return [UInt8](data)
    }
}

extension QwenLayer0DebugTests {

    /// Differential across the prefill/decode seam (FinchMoE "fresh-prefill"
    /// pattern), extended to TWO consecutive decode steps:
    /// - Run A7 chunked-prefills [prompt + " Paris" + " is"] (7 tokens);
    /// - Run B2 prefills [prompt] then `produce()`s " Paris" at position 5 and
    ///   " is" at position 6 (forceLogitsHead so logits are written).
    /// Every layer's output row at position 5 and 6 must agree between the
    /// two runs to fp16 rounding. Step 2 is the check no other instrument
    /// makes: it exercises the decode conv/rec state carry, the KV append,
    /// and the DECODE MoE tail a SECOND time, from state written by a decode
    /// step rather than by prefill.
    @Test(.enabled(if: installExists))
    func decodeEqualsPrefillAtPosition5() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is", addBOS: false)
        let nextIds = tokenizer.encode(" Paris", addBOS: false)
        let step2Ids = tokenizer.encode(" is", addBOS: false)
        let vocab = model.config.vocabSize
        let logitsBytes = vocab * MemoryLayout<Float16>.size

        // One scenario = a runner + hook + logits buffers.
        struct Run {
            var phases: [String: [Float16]] = [:]
            var logits: MTLBuffer? = nil
            var logitsPerStep: [MTLBuffer] = []
        }

        /// Chunked prefill of `tokens` with a hook; last-row logits.
        func prefillRun(_ tokens: [Int32]) async throws -> Run {
            var run = Run()
            run.logits = ctx.device.makeBuffer(length: logitsBytes,
                                               options: .storageModeShared)
            let runner = try RealForwardRunner(model: model, context: ctx,
                                               maxContext: 256,
                                               runtimeConfiguration: .production)
            runner.qwenLayerDebugHook = { layer, phase, values in
                run.phases["\(phase).\(layer)"] = values
            }
            try await runner.prefillChunked(
                tokens: tokens[0..<tokens.count], startPosition: 0,
                outputMode: .logits, config: .defaultChunked,
                into: run.logits!, onProgress: { _ in })
            run.logitsPerStep = [run.logits!]
            return run
        }

        /// Prefill `tokens`, then `produce()` each of `produced` at successive
        /// positions. Decode-step hook phases are tagged `step<N>.` so
        /// consecutive steps' per-layer rows are separable; each step's logits
        /// buffer is kept in `logitsPerStep`.
        func prefillPlusDecodes(_ tokens: [Int32],
                                _ produced: [Int32]) async throws -> Run {
            var run = Run()
            let runner = try RealForwardRunner(
                model: model, context: ctx, maxContext: 256,
                runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
            var step = 0
            runner.qwenLayerDebugHook = { layer, phase, values in
                let key = step == 0
                    ? "\(phase).\(layer)"
                    : "step\(step).\(phase).\(layer)"
                run.phases[key] = values
            }
            try await runner.prefillChunked(
                tokens: tokens[0..<tokens.count], startPosition: 0,
                outputMode: .logits, config: .defaultChunked,
                into: try #require(ctx.device.makeBuffer(
                    length: logitsBytes, options: .storageModeShared)),
                onProgress: { _ in })
            for (i, tok) in produced.enumerated() {
                step = i + 1
                let buf = try #require(ctx.device.makeBuffer(
                    length: logitsBytes, options: .storageModeShared))
                try await runner.produce(token: tok,
                                         position: tokens.count + i, into: buf)
                run.logitsPerStep.append(buf)
            }
            run.logits = run.logitsPerStep.last
            return run
        }

        func maxAbs(_ a: [Float16], _ b: [Float16]) -> Float {
            guard a.count == b.count, !a.isEmpty else { return -1 }
            var m: Float = 0
            for i in 0..<a.count { m = max(m, abs(Float(a[i]) - Float(b[i]))) }
            return m
        }
        func argmax(_ buf: MTLBuffer?) -> Int {
            guard let buf else { return -1 }
            let p = buf.contents().bindMemory(to: Float16.self, capacity: vocab)
            var best = -Float.greatestFiniteMagnitude
            var at = -1
            for i in 0..<vocab {
                let v = Float(p[i])
                if v > best { best = v; at = i }
            }
            return at
        }
        func logitsStats(_ buf: MTLBuffer?) -> (max: Float, mean: Float, std: Float) {
            guard let buf else { return (0, 0, 0) }
            let p = buf.contents().bindMemory(to: Float16.self, capacity: vocab)
            var mx = -Float.greatestFiniteMagnitude
            var sum: Float = 0, sumsq: Float = 0
            for i in 0..<vocab {
                let v = Float(p[i])
                mx = max(mx, v)
                sum += v
                sumsq += v * v
            }
            let mean = sum / Float(vocab)
            let var_ = max(0, sumsq / Float(vocab) - mean * mean)
            return (mx, mean, var_.squareRoot())
        }
        func maxAbsBuffers(_ a: MTLBuffer?, _ b: MTLBuffer?) -> Float {
            guard let a, let b else { return -1 }
            let ap = a.contents().bindMemory(to: Float16.self, capacity: vocab)
            let bp = b.contents().bindMemory(to: Float16.self, capacity: vocab)
            var m: Float = 0
            for i in 0..<vocab { m = max(m, abs(Float(ap[i]) - Float(bp[i]))) }
            return m
        }

        // Runs:
        //  prefill5 = CLI prompt alone (5 tokens); its logits = row 4, the
        //    source the CLI actually sampled its first token from.
        //  prefill6 / prefill7 = chunked references for decode steps 1 and 2
        //    (positions 5 and 6).
        //  dec2 = 5-token prefill + produce() at 5, then produce() at 6 —
        //    step 2 runs on conv/rec state written by step 1, not by prefill.
        let prefill5 = try await prefillRun(promptIds)
        let prefill6 = try await prefillRun(promptIds + nextIds.prefix(1))
        let prefill7 = try await prefillRun(promptIds + nextIds.prefix(1)
                                            + step2Ids.prefix(1))
        let dec2 = try await prefillPlusDecodes(promptIds,
                                                [nextIds[0], step2Ids[0]])

        /// Compare the chunked prefill's `row` against decode `step`'s
        /// postLayer row at every layer.
        func compareRow(_ label: String, prefill: [String: [Float16]],
                        decode: [String: [Float16]], row: Int, step: Int) {
            var worst: Float = 0
            var worstAt = "none"
            var diverged = -1
            for L in 0..<model.config.numLayers {
                guard let aRow = prefill["prefillHidden.\(row).\(L)"],
                      let bRow = decode["step\(step).postLayer.\(L)"] else {
                    print("\(label): L\(L) MISSING "
                        + "a=\(prefill["prefillHidden.\(row).\(L)"] != nil) "
                        + "b=\(decode["step\(step).postLayer.\(L)"] != nil)")
                    continue
                }
                let d = maxAbs(aRow, bRow)
                print("\(label): L\(L) maxAbs=\(d)")
                if d > worst { worst = d; worstAt = "L\(L)" }
                if d > 0.05 && diverged < 0 { diverged = L }
            }
            print("\(label): VERDICT worst=\(worst) at \(worstAt) "
                + "firstDiverged(>0.05)=\(diverged)")
        }
        // Step 1 (position 5) vs a 6-token prefill.
        compareRow("seam1 pos5", prefill: prefill6.phases,
                   decode: dec2.phases, row: 5, step: 1)
        // Step 2 (position 6) vs a 7-token prefill — the second decode step.
        compareRow("seam2 pos6", prefill: prefill7.phases,
                   decode: dec2.phases, row: 6, step: 2)

        func report(_ label: String, _ buf: MTLBuffer?) {
            let t = argmax(buf)
            let s = logitsStats(buf)
            print("\(label): argmax=\(t) "
                + "'\(tokenizer.decode([Int32(t)], skipSpecialTokens: false))'"
                + " max=\(s.max) mean=\(s.mean) std=\(s.std)")
        }
        report("row4 logits (CLI token-1 source)", prefill5.logits)
        report("row5 logits prefill", prefill6.logits)
        report("row5 logits decode step1", dec2.logitsPerStep.first)
        report("row6 logits prefill", prefill7.logits)
        report("row6 logits decode step2", dec2.logitsPerStep.last)
        print("row5 logits decode-vs-prefill maxAbs="
            + "\(maxAbsBuffers(prefill6.logits, dec2.logitsPerStep.first))")
        print("row6 logits decode-vs-prefill maxAbs="
            + "\(maxAbsBuffers(prefill7.logits, dec2.logitsPerStep.last))")
    }

    // MARK: - Repack-integrity gate: install LAYER tensors vs original bf16

    /// The embed + lm_head tables were aligned in
    /// `installRowsAlignWithOriginalVocab`, but no probe has ever compared a
    /// LAYER tensor of the int4 install against its original bf16 row. Every
    /// engine-vs-replay probe replays the SAME install bytes through
    /// layout.json, so a systematic repack corruption in layer tensors would
    /// look perfectly "consistent" — and the degenerate space-loop output
    /// (engine == grounded replay at every layer, top-1 = id 220 at every
    /// position) is exactly the signature of consistently-wrong layer weights.
    /// This gate dequantizes install rows (int4/int8 affine, fp16/fp32 raw
    /// copies, (1+w)-baked bf16 norms) and compares against the original bf16.
    @Test(.enabled(if: installExists && bf16DirExists))
    func installLayerTensorsAlignWithOriginalBf16() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let bf16Dir = "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-bf16"

        struct Located {
            let path: String
            let absOffset: UInt64
            let rows: Int
            let cols: Int
        }
        /// Scan the source shards for the tensor whose name ENDS with
        /// `suffix`; 1-D and 2-D tensors both accepted (1-D → rows=1).
        func locate(_ suffix: String) throws -> Located? {
            let files = try FileManager.default.contentsOfDirectory(atPath: bf16Dir)
                .filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }
                .sorted()
            for file in files {
                let path = bf16Dir + "/" + file
                let fh = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
                defer { try? fh.close() }
                let lenBytes = try fh.read(upToCount: 8) ?? Data()
                guard lenBytes.count == 8 else { continue }
                let headerLen = lenBytes.withUnsafeBytes { $0.load(as: UInt64.self) }
                guard headerLen < 1 << 24 else { continue }
                let header = try fh.read(upToCount: Int(headerLen)) ?? Data()
                let obj = try JSONSerialization.jsonObject(with: header)
                    as? [String: Any]
                for (name, value) in obj ?? [:] {
                    guard name.hasSuffix(suffix),
                          let entry = value as? [String: Any],
                          let shape = entry["shape"] as? [Any],
                          (1...2).contains(shape.count),
                          let offs = entry["data_offsets"] as? [Any],
                          offs.count == 2 else { continue }
                    func num(_ v: Any) -> Int? {
                        (v as? NSNumber)?.intValue ?? (v as? Int)
                    }
                    guard let begin = num(offs[0]) else { continue }
                    let rows = shape.count == 2 ? (num(shape[0]) ?? 0) : 1
                    let cols = shape.count == 2
                        ? (num(shape[1]) ?? 0) : (num(shape[0]) ?? 0)
                    return Located(path: path,
                                   absOffset: 8 + headerLen + UInt64(begin),
                                   rows: rows, cols: cols)
                }
            }
            return nil
        }

        /// Original bf16 flat row range → fp32 (whole tensor in row order).
        func origFlat(_ loc: Located) -> [Float] {
            let n = loc.rows * loc.cols
            let fh = try! FileHandle(forReadingFrom: URL(fileURLWithPath: loc.path))
            defer { try? fh.close() }
            try! fh.seek(toOffset: loc.absOffset)
            let data = try! fh.read(upToCount: n * 2) ?? Data()
            return data.withUnsafeBytes { raw in
                let words = raw.bindMemory(to: UInt16.self)
                return (0..<n).map { FinchQuantization.bf16ToFloat(words[$0]) }
            }
        }

        /// One tensor pair + verdict. `inst` rows come from a caller-supplied
        /// builder so each transform reads its own layout; `orig` is
        /// optionally transformed (e.g. +1 for the (1+w) norms).
        func compare(_ label: String,
                     _ inst: () -> [Float],
                     _ orig: [Float],
                     _ transform: String) {
            let a = inst()
            guard a.count == orig.count, !orig.isEmpty else {
                print("\(label): size mismatch \(a.count) vs \(orig.count)")
                return
            }
            var sumAbs: Float = 0
            var sumSq: Float = 0
            var mx: Float = 0
            var worstI = -1
            for i in 0..<a.count {
                let d = abs(a[i] - orig[i])
                sumAbs += d
                sumSq += orig[i] * orig[i]
                if d > mx { mx = d; worstI = i }
            }
            let avgAbs = sumAbs / Float(a.count)
            let rms = (sumSq / Float(a.count)).squareRoot()
            let verdict = avgAbs * 8 < rms ? "SAME" : "DIFFERENT"
            print("\(label) [\(transform)]: avgAbs=\(avgAbs) "
                + "maxAbs=\(mx)@\(worstI) origRMS=\(rms) → \(verdict)")
        }

        // Rows to dequant for the big int4/int8 matrices (whole-tensor
        // compare for anything small). Stride samples catch offset bugs that
        // shift everything from some row onward.
        func sampleRows(_ n: Int, fullIfBelow: Int = 800) -> [Int] {
            if n <= fullIfBelow { return Array(0..<n) }
            var rows = [0, 1, n / 4, n / 2, 3 * n / 4, n - 1]
            var r = 0
            while r < n { rows.append(r); r += max(1, n / 97) }
            return Array(Set(rows)).sorted()
        }
        /// Install int4/int8 rows for the sampled row set.
        func instQuant(_ view: TensorView, rows: Int, cols: Int,
                       int8: Bool) -> [Float] {
            let base = view.buffer.contents()
            let wBytes = base.advanced(by: Int(view.offset))
            let sWords = base.advanced(by: Int(view.scaleOffset))
                .assumingMemoryBound(to: UInt16.self)
            let bWords = base.advanced(by: Int(view.biasOffset))
                .assumingMemoryBound(to: UInt16.self)
            let groups = cols / 64
            var out: [Float] = []
            out.reserveCapacity(rows * cols)
            for r in sampleRows(rows) {
                for g in 0..<groups {
                    let scale = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                    let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                    let byteBase = int8 ? r * cols + g * 64 : r * (cols / 2) + g * 32
                    for k in 0..<64 {
                        if int8 {
                            let q = wBytes.load(fromByteOffset: byteBase + k,
                                                as: UInt8.self)
                            out.append(Float(q) * scale + bias)
                        } else {
                            let byte = wBytes.load(fromByteOffset: byteBase + k / 2,
                                                   as: UInt8.self)
                            let nibble = (k & 1) == 0 ? Int(byte & 0x0F)
                                : Int(byte >> 4)
                            out.append(Float(nibble) * scale + bias)
                        }
                    }
                }
            }
            return out
        }
        /// Sampled rows of the original (must match instQuant's sampling).
        func origSampled(_ loc: Located) -> [Float] {
            var out: [Float] = []
            out.reserveCapacity(loc.rows * loc.cols)
            let fh = try! FileHandle(forReadingFrom: URL(fileURLWithPath: loc.path))
            defer { try? fh.close() }
            let rowBytes = loc.cols * 2
            for r in sampleRows(loc.rows) {
                try! fh.seek(toOffset: loc.absOffset + UInt64(r * rowBytes))
                let data = try! fh.read(upToCount: rowBytes) ?? Data()
                out += data.withUnsafeBytes { raw in
                    let words = raw.bindMemory(to: UInt16.self)
                    return (0..<loc.cols).map {
                        FinchQuantization.bf16ToFloat(words[$0])
                    }
                }
            }
            return out
        }

        let cfg = model.config
        func check(_ label: String, _ suffix: String, _ view: TensorView,
                   kind: String) throws {
            guard let loc = try locate(suffix) else {
                print("\(label): bf16 '\(suffix)' NOT FOUND"); return
            }
            let n = loc.rows * loc.cols
            switch kind {
            case "i4", "i8":
                let orig = origSampled(loc)
                compare(label, {
                    instQuant(view, rows: loc.rows, cols: loc.cols,
                              int8: kind == "i8")
                }, orig, kind)
            case "bf16Raw", "onePlusW":
                let orig = origFlat(loc)
                let transformed = kind == "onePlusW" ? orig.map { 1 + $0 } : orig
                compare(label, {
                    Self.bf16Values(view, count: n)
                }, transformed, kind)
            case "fp16Raw":
                compare(label, { Self.fp16Values(view, count: n) },
                        origFlat(loc), kind)
            case "fp32Raw":
                compare(label, { Self.fp32Values(view, count: n) },
                        origFlat(loc), kind)
            default: break
            }
        }

        // Layer 0 (GDN) — norms, gates, conv, projections.
        try check("L0 input_layernorm", "layers.0.input_layernorm.weight",
                  try model.inputNorm(layer: 0), kind: "onePlusW")
        try check("L0 post_attention_layernorm",
                  "layers.0.post_attention_layernorm.weight",
                  try model.postAttnNorm(layer: 0), kind: "onePlusW")
        try check("L0 linear_attn.norm", "layers.0.linear_attn.norm.weight",
                  try model.gdnNormWeight(layer: 0), kind: "onePlusW")
        try check("L0 conv1d", "layers.0.linear_attn.conv1d.weight",
                  try model.gdnConv1D(layer: 0), kind: "fp16Raw")
        try check("L0 A_log", "layers.0.linear_attn.A_log",
                  try model.gdnALog(layer: 0), kind: "fp32Raw")
        try check("L0 dt_bias", "layers.0.linear_attn.dt_bias",
                  try model.gdnDtBias(layer: 0), kind: "fp32Raw")
        try check("L0 in_proj_qkv", "layers.0.linear_attn.in_proj_qkv.weight",
                  try model.gdnInProjQKV(layer: 0), kind: "i4")
        try check("L0 in_proj_z", "layers.0.linear_attn.in_proj_z.weight",
                  try model.gdnInProjZ(layer: 0), kind: "i4")
        try check("L0 in_proj_a", "layers.0.linear_attn.in_proj_a.weight",
                  try model.gdnInProjA(layer: 0), kind: "i4")
        try check("L0 in_proj_b", "layers.0.linear_attn.in_proj_b.weight",
                  try model.gdnInProjB(layer: 0), kind: "i4")
        try check("L0 out_proj", "layers.0.linear_attn.out_proj.weight",
                  try model.gdnOutProj(layer: 0), kind: "i4")

        // Layer 21 (GDN + MoE): projections, router (int8), shared expert.
        let L21 = 21
        try check("L21 in_proj_qkv", "layers.21.linear_attn.in_proj_qkv.weight",
                  try model.gdnInProjQKV(layer: L21), kind: "i4")
        try check("L21 out_proj", "layers.21.linear_attn.out_proj.weight",
                  try model.gdnOutProj(layer: L21), kind: "i4")
        try check("L21 A_log", "layers.21.linear_attn.A_log",
                  try model.gdnALog(layer: L21), kind: "fp32Raw")
        try check("L21 conv1d", "layers.21.linear_attn.conv1d.weight",
                  try model.gdnConv1D(layer: L21), kind: "fp16Raw")
        try check("L21 router", "layers.21.mlp.gate.weight",
                  try model.router(layer: L21), kind: "i8")
        try check("L21 shared.gate", "layers.21.mlp.shared_expert.gate_proj.weight",
                  try model.sharedExpertGate(layer: L21), kind: "i4")
        try check("L21 shared.up", "layers.21.mlp.shared_expert.up_proj.weight",
                  try model.sharedExpertUp(layer: L21), kind: "i4")
        try check("L21 shared.down", "layers.21.mlp.shared_expert.down_proj.weight",
                  try model.sharedExpertDown(layer: L21), kind: "i4")
        try check("L21 shared_expert_gate",
                  "layers.21.mlp.shared_expert_gate.weight",
                  try model.sharedExpertGateProj(layer: L21), kind: "i4")

        // Layer 39 (full attention): projections + q/k norms.
        let L39 = 39
        for suffix in ["layers.39.self_attn.q_proj.weight",
                       "layers.39.self_attn.k_proj.weight",
                       "layers.39.self_attn.v_proj.weight",
                       "layers.39.self_attn.o_proj.weight"] {
            let view = try suffix.hasSuffix("q_proj.weight") ? model.qProj(layer: L39)
                : suffix.hasSuffix("k_proj.weight") ? model.kProj(layer: L39)
                : suffix.hasSuffix("v_proj.weight") ? model.vProj(layer: L39)
                : model.oProj(layer: L39)
            try check("L39 " + suffix, suffix, view, kind: "i4")
        }
        try check("L39 q_norm", "layers.39.self_attn.q_norm.weight",
                  try model.qNorm(layer: L39), kind: "onePlusW")
        try check("L39 k_norm", "layers.39.self_attn.k_norm.weight",
                  try model.kNorm(layer: L39), kind: "onePlusW")
        try check("L39 input_layernorm", "layers.39.input_layernorm.weight",
                  try model.inputNorm(layer: L39), kind: "onePlusW")

        // Final norm + embed (embed re-checked here at sampled rows only).
        try check("final norm", "model.language_model.norm.weight",
                  model.finalNorm, kind: "onePlusW")
        try check("embed", "model.language_model.embed_tokens.weight",
                  model.embedding, kind: "i4")
        print("layer-integrity sweep rows: \(cfg.hiddenSize) hidden, "
            + "\(cfg.numLayers) layers — see per-tensor verdicts above")
    }

    // MARK: - Vocab-order check: install rows vs original bf16 shards

    /// Hypothesis check: is the CLI garbage caused by the install's
    /// embedding / lm_head tables being permuted or shifted relative to the
    /// tokenizer's id order? (All engine-internal and fp32-replay checks read
    /// the same repacked files, so a consistent reorder would stay invisible
    /// while the output TEXT becomes nonsense.) Compares dequantized install
    /// rows against the original bf16 safetensor rows for the same ids, and
    /// also scores the ±1 shifts that an inserted/removed pad row would cause.
    @Test(.enabled(if: installExists && bf16DirExists))
    func installRowsAlignWithOriginalVocab() async throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let D = model.config.hiddenSize
        let bf16Dir = "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-bf16"

        struct Located {
            let path: String
            let absOffset: UInt64
            let rows: Int
            let cols: Int
        }
        /// Scan the source shards for the tensor whose name ends with
        /// `substring`; return its absolute file coordinates.
        func locate(_ substring: String) throws -> Located? {
            let files = try FileManager.default.contentsOfDirectory(atPath: bf16Dir)
                .filter { $0.hasPrefix("model-") && $0.hasSuffix(".safetensors") }
                .sorted()
            for file in files {
                let path = bf16Dir + "/" + file
                let fh = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
                defer { try? fh.close() }
                let lenBytes = try fh.read(upToCount: 8) ?? Data()
                guard lenBytes.count == 8 else { continue }
                let headerLen = lenBytes.withUnsafeBytes { $0.load(as: UInt64.self) }
                guard headerLen < 1 << 24 else { continue }
                let header = try fh.read(upToCount: Int(headerLen)) ?? Data()
                let obj = try JSONSerialization.jsonObject(with: header)
                    as? [String: Any]
                for (name, value) in obj ?? [:] {
                    guard name.hasSuffix(substring),
                          let entry = value as? [String: Any],
                          let shape = entry["shape"] as? [Any],
                          shape.count == 2,
                          let offs = entry["data_offsets"] as? [Any],
                          offs.count == 2 else { continue }
                    func num(_ v: Any) -> Int? {
                        (v as? NSNumber)?.intValue ?? (v as? Int)
                    }
                    guard let rows = num(shape[0]), let cols = num(shape[1]),
                          let begin = num(offs[0]) else { continue }
                    return Located(path: path,
                                   absOffset: 8 + headerLen + UInt64(begin),
                                   rows: rows, cols: cols)
                }
            }
            return nil
        }

        /// Original bf16 row `r` → fp32.
        func origRow(_ loc: Located, _ r: Int) -> [Float] {
            let rowBytes = loc.cols * 2
            let fh = try! FileHandle(forReadingFrom: URL(fileURLWithPath: loc.path))
            defer { try? fh.close() }
            try! fh.seek(toOffset: loc.absOffset + UInt64(r * rowBytes))
            let data = try! fh.read(upToCount: rowBytes) ?? Data()
            return data.withUnsafeBytes { raw in
                let words = raw.bindMemory(to: UInt16.self)
                return (0..<loc.cols).map {
                    FinchQuantization.bf16ToFloat(words[$0])
                }
            }
        }

        /// Install-side dequantized row `r` of a resident int4 view (same
        /// layout math as `int4Rows`, single row so large ids are cheap).
        func instRow(_ view: TensorView, _ r: Int) -> [Float] {
            let base = view.buffer.contents()
            let wBytes = base.advanced(by: Int(view.offset))
            let sWords = base.advanced(by: Int(view.scaleOffset))
                .assumingMemoryBound(to: UInt16.self)
            let bWords = base.advanced(by: Int(view.biasOffset))
                .assumingMemoryBound(to: UInt16.self)
            let groups = D / 64
            var row = [Float](repeating: 0, count: D)
            for g in 0..<groups {
                let scale = FinchQuantization.bf16ToFloat(sWords[r * groups + g])
                let bias = FinchQuantization.bf16ToFloat(bWords[r * groups + g])
                let byteBase = r * (D / 2) + g * 32
                for k in 0..<64 {
                    let byte = wBytes.load(fromByteOffset: byteBase + k / 2,
                                           as: UInt8.self)
                    let nibble = (k & 1) == 0 ? Int(byte & 0x0F) : Int(byte >> 4)
                    row[g * 64 + k] = Float(nibble) * scale + bias
                }
            }
            return row
        }

        /// L1-style separation: average |Δ| vs the row's own RMS. Same-token
        /// int4 noise is ~0.005 avg|Δ| at embed scale; a wrong token (or a
        /// shift) is several times the row RMS.
        func report(_ label: String, _ inst: [Float], _ orig: [Float]) {
            guard inst.count == orig.count else {
                print("\(label): size mismatch \(inst.count) vs \(orig.count)")
                return
            }
            var sumAbs: Float = 0
            var sumSq: Float = 0
            for i in 0..<inst.count {
                sumAbs += abs(inst[i] - orig[i])
                sumSq += orig[i] * orig[i]
            }
            let avgAbs = sumAbs / Float(inst.count)
            let rms = (sumSq / Float(orig.count)).squareRoot()
            let verdict = avgAbs * 8 < rms ? "SAME" : "DIFFERENT"
            print("\(label): avgAbs=\(avgAbs) origRMS=\(rms) → \(verdict)")
        }

        guard let embed = try locate("embed_tokens.weight"),
              let head = try locate("lm_head.weight"),
              embed.cols == D, head.cols == D else {
            print("embed/lm_head not located")
            return
        }
        let maxID = min(embed.rows, head.rows, model.config.vocabSize) - 1
        print("original embed rows=\(embed.rows) lm_head rows=\(head.rows) "
            + "maxID=\(maxID)")

        // Dense identity sweep of the first 256 ids (both tables) — catches a
        // uniform shift instantly.
        for (name, loc, view) in [
            ("embed", embed, model.embedding),
            ("lmHead", head, model.lmHead),
        ] {
            var worst: Float = 0
            var worstID = -1
            var best: Float = .greatestFiniteMagnitude
            var bestID = -1
            let n = min(256, maxID + 1)
            for id in 0..<n {
                let inst = instRow(view, id)
                let ref = origRow(loc, id)
                var mx: Float = 0
                for i in 0..<D { mx = max(mx, abs(inst[i] - ref[i])) }
                if mx > worst { worst = mx; worstID = id }
                if mx < best { best = mx; bestID = id }
            }
            print("\(name) rows 0..<\(n): maxAbs worst=\(worst) at \(worstID) "
                + "best=\(best) at \(bestID)")
        }

        // Sampled ids across the vocab, with ±1 shift probes.
        let probes = [0, 32, 220, 315, 5265, 95744, 248044,
                      model.config.vocabSize - 1]
        for id in probes where id <= maxID {
            for (name, loc, view) in [
                ("embed", embed, model.embedding),
                ("lmHead", head, model.lmHead),
            ] {
                let inst = instRow(view, id)
                let atID = origRow(loc, id)
                report("\(name)[\(id)]@id", inst, atID)
                if id > 0 {
                    report("\(name)[\(id)]@id-1", inst, origRow(loc, id - 1))
                }
                if id < maxID {
                    report("\(name)[\(id)]@id+1", inst, origRow(loc, id + 1))
                }
            }
        }
    }

    // MARK: - Torch probe row dump

    /// Writes the engine's full prefill chain to a binary file for the
    /// per-layer torch ground-truth probe (`torch_layer_probe.py`): embed rows,
    /// every layer's post-MoE hidden row AND post-attention-norm dense row (all
    /// positions, fp16-rounded to f32), plus per-layer router top-8 ids/weights
    /// for the last token.
    /// File layout (little-endian): Int32[4] header (magic 0x00A10003, T, D,
    /// numLayers), Int32[T] token ids, f32[T·D] embed, f32[numLayers·T·D]
    /// hidden rows, f32[numLayers·T·D] dense rows, Int32 count + Int32[8·L]
    /// route ids, f32[8·L] route weights, then a trailing GDN-branch block:
    /// Int32[4] (magic 0x00A20003, numLayers, T, D) + f32[numLayers·T·D]
    /// pre-residual linear-attn output rows `xa` (zeros on full-attention
    /// layers), then a per-stage block: Int32[7] (magic 0x00A20005,
    /// linearCount, T, D, qkvDim, valueDim, numValueHeads) followed per
    /// linear layer in ascending order by Int32[1] layer + f32[T·D] normed +
    /// f32[T·qkvDim] post-conv qkv + f32[T·valueDim] o (pre-gated-norm
    /// recurrent output, captured before the in-place norm) + f32[T·valueDim]
    /// h (post-gated-norm, pre-out_proj) + f32[T·valueDim] z + f32[T·numV] g
    /// + f32[T·numV] beta. Gated on FQ_TORCH_DUMP=1 (no-op otherwise so the
    /// suite stays green in plain runs); path override via FQ_ROW_PATH.
    @Test(.enabled(if: installExists))
    func dumpRowsForTorchProbe() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["FQ_TORCH_DUMP"] == "1" else { return }
        func say(_ s: String) { print(s); fflush(stdout) }
        let outPath = env["FQ_ROW_PATH"] ?? "/tmp/fq_rows.bin"

        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        let D = model.config.hiddenSize
        let numLayers = model.config.numLayers
        var hidden: [String: [Float16]] = [:]   // key "\(L).\(t)"
        var dense: [String: [Float16]] = [:]    // post-attn-norm rows
        var xa: [String: [Float16]] = [:]       // GDN pre-residual branch rows
        var stages: [String: [Float16]] = [:]   // key "\(L)|\(stage)"
        var routeIDs: [Int: [Float16]] = [:]    // key "\(L)", last token only
        var routeW: [Int: [Float16]] = [:]
        runner.qwenLayerDebugHook = { layer, phase, values in
            if phase.hasPrefix("pfGdn.") {
                stages["\(layer)|\(phase.dropFirst("pfGdn.".count))"] = values
            } else if phase.hasPrefix("prefillHidden.") {
                if let t = Int(phase.dropFirst("prefillHidden.".count)) {
                    hidden["\(layer).\(t)"] = values
                }
            } else if phase.hasPrefix("prefillDense.") {
                if let t = Int(phase.dropFirst("prefillDense.".count)) {
                    dense["\(layer).\(t)"] = values
                }
            } else if phase.hasPrefix("prefillXA.") {
                if let t = Int(phase.dropFirst("prefillXA.".count)) {
                    xa["\(layer).\(t)"] = values
                }
            } else if phase == "pfRouteIDs" {
                routeIDs[layer] = values
            } else if phase == "pfRouteW" {
                routeW[layer] = values
            }
        }

        let tokenizer = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let promptIds = tokenizer.encode("The capital of France is",
                                         addBOS: false)
        let T = promptIds.count
        say("dumpRowsForTorchProbe: T=\(T) ids=\(promptIds)")
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        try await runner.prefillChunked(
            tokens: promptIds[0..<T], startPosition: 0,
            outputMode: .logits, config: .defaultChunked,
            into: logits, onProgress: { _ in })

        // Engine greedy first token from the prefill logits (row T-1).
        let lp = logits.contents().assumingMemoryBound(to: Float16.self)
        var top8: [(Float, Int)] = []
        for v in 0..<model.config.vocabSize {
            let f = Self.toF32(lp[v])
            if top8.count < 8 { top8.append((f, v)); top8.sort { $0.0 > $1.0 } }
            else if f > top8[7].0 { top8[7] = (f, v); top8.sort { $0.0 > $1.0 } }
        }
        say("engine row-\(T - 1) top8: " + top8.map { "\($0.1)(\($0.0))" }
            .joined(separator: " "))

        // Embed rows: the layer-0 input (dequant int4, fp16-rounded).
        let embedView = model.embedding
        var embed: [[Float]] = []
        embed.reserveCapacity(T)
        for t in 0..<T {
            embed.append(Self.f16(Self.int4Rows(
                embedView, rows: Int(promptIds[t]) + 1,
                cols: D)[Int(promptIds[t])]))
        }

        var data = Data()
        func put(_ xs: [Float]) {
            xs.withUnsafeBytes { data.append(contentsOf: $0) }
        }
        var header: [Int32] = [0x00A10003, Int32(T), Int32(D),
                               Int32(numLayers)]
        header.withUnsafeBytes { data.append(contentsOf: $0) }
        var ids = promptIds.map { Int32($0) }
        ids.withUnsafeBytes { data.append(contentsOf: $0) }
        for row in embed { put(row) }
        var missing = 0
        var missingDense = 0
        for L in 0..<numLayers {
            for t in 0..<T {
                guard let v = hidden["\(L).\(t)"] else { missing += 1; continue }
                put(v.map { Float($0) })
            }
        }
        for L in 0..<numLayers {
            for t in 0..<T {
                guard let v = dense["\(L).\(t)"] else { missingDense += 1; continue }
                put(v.map { Float($0) })
            }
        }
        // Per-layer router top-8 ids (float16-packaged) + weights, last token.
        var routeIds = [Int32]()
        var wts = [Float]()
        for L in 0..<numLayers {
            let rid = (routeIDs[L] ?? []).map { Int32(Self.toF32($0)) }
            if rid.count == 8 {
                routeIds.append(contentsOf: rid)
            } else {
                routeIds.append(contentsOf: [Int32](repeating: -1, count: 8))
            }
            let rw = (routeW[L] ?? []).map { Float($0) }
            if rw.count == 8 { wts.append(contentsOf: rw) }
            else { wts.append(contentsOf: [Float](repeating: 0, count: 8)) }
        }
        var ridHeader: [Int32] = [Int32(routeIds.count)]
        ridHeader.withUnsafeBytes { data.append(contentsOf: $0) }
        routeIds.withUnsafeBytes { data.append(contentsOf: $0) }
        put(wts)
        // GDN pre-residual branch rows (xa), zeros where a layer did not emit
        // (full-attention layers / missed snapshots).
        var xaBlock: [Int32] = [0x00A20003, Int32(numLayers), Int32(T),
                                Int32(D)]
        xaBlock.withUnsafeBytes { data.append(contentsOf: $0) }
        var missingXA = 0
        let zeros = [Float](repeating: 0, count: D)
        for L in 0..<numLayers {
            for t in 0..<T {
                if let v = xa["\(L).\(t)"] {
                    put(v.map { Float($0) })
                } else {
                    missingXA += 1
                    put(zeros)
                }
            }
        }
        // GDN branch per-stage snapshots (whole chunk per stage), one per
        // linear layer, in ascending layer order: [L] + f32[T·D] normed,
        // f32[T·qkvDim] post-conv qkv, f32[T·valueDim] o (pre-gated-norm),
        // f32[T·valueDim] h (post-gated-norm, pre-out_proj), f32[T·valueDim]
        // z, f32[T·numV] g, f32[T·numV] beta. Header: magic 0x00A20005,
        // linearCount, T, D, qkvDim, valueDim, numValueHeads.
        let mask = model.config.fullAttentionLayerMask
        let linearLayers = (0..<numLayers).filter { mask[$0] == 0 }
        let keyDim = model.config.linearNumKeyHeads
            * model.config.linearKeyHeadDim
        let valueDim = model.config.linearNumValueHeads
            * model.config.linearValueHeadDim
        let numV = model.config.linearNumValueHeads
        let qkvDim = 2 * keyDim + valueDim
        var stHeader: [Int32] = [0x00A20005, Int32(linearLayers.count),
                                 Int32(T), Int32(D), Int32(qkvDim),
                                 Int32(valueDim), Int32(numV)]
        stHeader.withUnsafeBytes { data.append(contentsOf: $0) }
        var missingStage = 0
        func putStage(_ key: String, _ count: Int) {
            if let v = stages[key] {
                put(v.map { Float($0) })
            } else {
                missingStage += 1
                put([Float](repeating: 0, count: count))
            }
        }
        for L in linearLayers {
            var lt: [Int32] = [Int32(L)]
            lt.withUnsafeBytes { data.append(contentsOf: $0) }
            putStage("\(L)|normed", T * D)
            putStage("\(L)|conv", T * qkvDim)
            putStage("\(L)|o", T * valueDim)
            putStage("\(L)|h", T * valueDim)
            putStage("\(L)|z", T * valueDim)
            putStage("\(L)|g", T * numV)
            putStage("\(L)|beta", T * numV)
        }
        try data.write(to: URL(fileURLWithPath: outPath))
        say("wrote \(outPath) bytes=\(data.count) missingRows=\(missing) "
            + "missingDense=\(missingDense) missingXA=\(missingXA) "
            + "missingStage=\(missingStage) linearLayers=\(linearLayers.count) "
            + "routeLayers=\(routeIDs.count)")
    }
}

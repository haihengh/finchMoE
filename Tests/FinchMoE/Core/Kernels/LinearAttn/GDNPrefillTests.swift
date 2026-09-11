import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares the Metal `gdn_prefill` kernels against the fp32 `GDNPrefillRef`
/// reference — the same sequential GDN recurrence as the decode unit, batched
/// over the chunk's T tokens.
///
/// Validated operations, all grounded in `qwen3_5_moe`:
///   * `prefill_gdn_conv_chunk`  — batched causal conv1d (kernel 4), including
///     the post-chunk state commit for every T (1, 2, and larger).
///   * `prefill_gdn_recurrent_seq` — per-value-head gated-delta-rule recurrence
///     unrolled over the chunk (reads q/k/v from the fused conv block).
///   * `prefill_gdn_gate`        — batched `g`/`beta` from the fp16 a|b QMM out.
///   * `prefill_gdn_rmsnorm_gated` — batched gated RMSNorm (mean-based).
///
/// The recurrent state is fp32 and the kernel does two-stage block reductions;
/// the reference is a naive per-element fp32 loop. Different op-trees, so a
/// matching result within `fp16ChainedReduction` is a real check.
@Suite struct GDNPrefillTests {

    private static let l2Eps: Float = 1e-6

    // MARK: - Batched causal conv1d chunk

    private static func runConvChunk(channels C: Int, tokens T: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-prefill-conv-C\(C)-T\(T)")
        let wF32   = (0..<(C*4)).map { _ in rng.uniform(-1.0, 1.0) }
        let sF32   = (0..<(C*3)).map { _ in rng.uniform(-1.0, 1.0) }
        let xF32   = (0..<(T*C)).map { _ in rng.uniform(-1.0, 1.0) }

        // The kernel reads fp16, so the reference uses the fp16-rounded inputs.
        let w16 = wF32.map { Float16($0) }; let wRef = w16.map { Float($0) }
        let s16 = sF32.map { Float16($0) }; let sRef = s16.map { Float($0) }
        let x16 = xF32.map { Float16($0) }; let xRef = x16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let wBuf = Fp16Buffer.make(ctx.device, halves: w16),
              let sBuf = Fp16Buffer.make(ctx.device, halves: s16),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let oBuf = Fp16Buffer.make(ctx.device, count: T*C),
              let nsBuf = Fp16Buffer.make(ctx.device, count: C*3) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeConvChunk(
            commandBuffer: cb, w: wBuf, state: sBuf, x: xBuf,
            out: oBuf, newState: nsBuf, channels: C, tokens: T)
        cb.commit(); cb.waitUntilCompleted()

        let (refOut, refNS) = GDNPrefillRef.convChunk(
            w: wRef, state: sRef, x: xRef, channels: C, tokens: T)

        let outActual = Fp16Buffer.read(oBuf, count: T*C)
        let outRel = RelError.compute(actual: outActual, reference: refOut)
        #expect(outRel < Tolerance.fp16ChainedReduction,
                "conv C=\(C) T=\(T): out relErr=\(outRel) maxAbs=\(RelError.maxAbsDiff(outActual, refOut))")

        let nsActual = Fp16Buffer.read(nsBuf, count: C*3)
        let nsRel = RelError.compute(actual: nsActual, reference: refNS)
        #expect(nsRel < Tolerance.fp16ChainedReduction,
                "conv C=\(C) T=\(T): newState relErr=\(nsRel) maxAbs=\(RelError.maxAbsDiff(nsActual, refNS))")
    }

    @Test func prefill_conv_t1()    throws { try Self.runConvChunk(channels: 128,  tokens: 1,   seed: 0x1101) }
    @Test func prefill_conv_t2()    throws { try Self.runConvChunk(channels: 128,  tokens: 2,   seed: 0x1102) }
    @Test func prefill_conv_t3()    throws { try Self.runConvChunk(channels: 128,  tokens: 3,   seed: 0x1103) }
    @Test func prefill_conv_full()  throws { try Self.runConvChunk(channels: 8192, tokens: 1,   seed: 0x1104) }
    @Test func prefill_conv_fullT() throws { try Self.runConvChunk(channels: 8192, tokens: 128, seed: 0x1105) }

    /// Two chunks through the wrapper on one command buffer: the second chunk's
    /// conv must see the first chunk's committed state (the wrapper blit-copies
    /// `newState` into the persistent buffer after each kernel).
    @Test func prefill_conv_stateCarriesAcrossChunks() throws {
        var rng = SeedTree(0x1106).key("gdn-prefill-conv-carry")
        let C = 256, T1 = 5, T2 = 3
        let wF32 = (0..<(C*4)).map { _ in rng.uniform(-1.0, 1.0) }
        let sF32 = (0..<(C*3)).map { _ in rng.uniform(-1.0, 1.0) }
        let x1F32 = (0..<(T1*C)).map { _ in rng.uniform(-1.0, 1.0) }
        let x2F32 = (0..<(T2*C)).map { _ in rng.uniform(-1.0, 1.0) }

        let w16 = wF32.map { Float16($0) }; let wRef = w16.map { Float($0) }
        let s16 = sF32.map { Float16($0) }; let sRef = s16.map { Float($0) }
        let x1 = x1F32.map { Float16($0) }; let x1Ref = x1.map { Float($0) }
        let x2 = x2F32.map { Float16($0) }; let x2Ref = x2.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let wBuf = Fp16Buffer.make(ctx.device, halves: w16),
              let sBuf = Fp16Buffer.make(ctx.device, halves: s16),
              let x1Buf = Fp16Buffer.make(ctx.device, halves: x1),
              let x2Buf = Fp16Buffer.make(ctx.device, halves: x2),
              let o1Buf = Fp16Buffer.make(ctx.device, count: T1*C),
              let o2Buf = Fp16Buffer.make(ctx.device, count: T2*C),
              let nsBuf = Fp16Buffer.make(ctx.device, count: C*3) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeConvChunk(commandBuffer: cb, w: wBuf, state: sBuf,
                               x: x1Buf, out: o1Buf, newState: nsBuf,
                               channels: C, tokens: T1)
        kernel.encodeConvChunk(commandBuffer: cb, w: wBuf, state: sBuf,
                               x: x2Buf, out: o2Buf, newState: nsBuf,
                               channels: C, tokens: T2)
        cb.commit(); cb.waitUntilCompleted()

        let (refOut1, refNS1) = GDNPrefillRef.convChunk(
            w: wRef, state: sRef, x: x1Ref, channels: C, tokens: T1)
        let (refOut2, refNS2) = GDNPrefillRef.convChunk(
            w: wRef, state: refNS1, x: x2Ref, channels: C, tokens: T2)

        let out1Actual = Fp16Buffer.read(o1Buf, count: T1*C)
        let out2Actual = Fp16Buffer.read(o2Buf, count: T2*C)
        #expect(RelError.compute(actual: out1Actual, reference: refOut1) < Tolerance.fp16ChainedReduction,
                "carry chunk1 out relErr=\(RelError.compute(actual: out1Actual, reference: refOut1))")
        #expect(RelError.compute(actual: out2Actual, reference: refOut2) < Tolerance.fp16ChainedReduction,
                "carry chunk2 out relErr=\(RelError.compute(actual: out2Actual, reference: refOut2))")
        #expect(RelError.compute(actual: Fp16Buffer.read(sBuf, count: C*3), reference: refNS2) < Tolerance.fp16ChainedReduction,
                "carry final state relErr=\(RelError.compute(actual: Fp16Buffer.read(sBuf, count: C*3), reference: refNS2))")
    }

    // MARK: - Sequential gated-delta-rule recurrence over the chunk
    //
    // Real Qwen3.6 GDN dims: 16 key heads, 32 value heads, head dim 128. The
    // kernel reads q/k/v from the fused [T][C] conv block (k at keyDim, v at
    // 2*keyDim elements) — the production "no split copies" layout.

    private static func runRecurrentSeq(
        numKeyHeads K: Int, numValueHeads V: Int, headDim D: Int, tokens T: Int, seed: UInt64
    ) throws {
        var rng = SeedTree(seed).key("gdn-prefill-rec-V\(V)-D\(D)-T\(T)")
        let scale = 1.0 / sqrtf(Float(D))
        let keyDim = K * D
        let valueDim = V * D
        let C = 2 * keyDim + valueDim
        let kOff = keyDim
        let vOff = 2 * keyDim

        var convF32 = [Float](repeating: 0, count: T * C)
        for t in 0..<T {
            for i in 0..<(2*keyDim + valueDim) {
                convF32[t * C + i] = rng.uniform(-1.0, 1.0)
            }
        }
        let g    = (0..<(T*V)).map { _ in rng.uniform(-3.0, 0.0) }   // decay in [0.05, 1]
        let beta = (0..<(T*V)).map { _ in rng.uniform(0.0, 1.0) }
        var state = (0..<(V*D*D)).map { _ in rng.uniform(-0.5, 0.5) }

        // Kernel reads the conv block as fp16 → reference uses the rounded values.
        let conv16 = convF32.map { Float16($0) }; let convRef = conv16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let convBuf = Fp16Buffer.make(ctx.device, halves: conv16),
              let oBuf = Fp16Buffer.make(ctx.device, count: T*valueDim),
              let gBuf = makeFp32Buffer(ctx.device, g),
              let bBuf = makeFp32Buffer(ctx.device, beta),
              let sBuf = makeFp32Buffer(ctx.device, state) else {
            Issue.record("alloc failed"); return
        }

        var refState = state
        let refOut = GDNPrefillRef.recurrentChunk(
            state: &refState, conv: convRef, g: g, beta: beta,
            channels: C, kOffset: kOff, vOffset: vOff,
            numValueHeads: V, numKeyHeads: K, headDim: D, tokens: T, scale: scale)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRecurrentSeq(
            commandBuffer: cb, state: sBuf, conv: convBuf,
            g: gBuf, beta: bBuf, out: oBuf,
            headDim: UInt32(D), channels: UInt32(C),
            kOffset: UInt32(kOff), vOffset: UInt32(vOff),
            numValueHeads: V, numKeyHeads: K, tokens: T, scale: scale, l2eps: l2Eps)
        cb.commit(); cb.waitUntilCompleted()

        let outActual = Fp16Buffer.read(oBuf, count: T*valueDim)
        let outRel = RelError.compute(actual: outActual, reference: refOut)
        #expect(outRel < Tolerance.fp16ChainedReduction,
                "recurrent V=\(V) D=\(D) T=\(T): out relErr=\(outRel) maxAbs=\(RelError.maxAbsDiff(outActual, refOut))")

        let stateActual = readFp32(sBuf, count: V*D*D)
        let stateRel = RelError.compute(actual: stateActual, reference: refState)
        #expect(stateRel < Tolerance.fp16ChainedReduction,
                "recurrent V=\(V) D=\(D) T=\(T): state relErr=\(stateRel) maxAbs=\(RelError.maxAbsDiff(stateActual, refState))")
    }

    @Test func prefill_recurrent_fullT1() throws {
        try Self.runRecurrentSeq(numKeyHeads: 16, numValueHeads: 32, headDim: 128, tokens: 1, seed: 0x1201)
    }
    @Test func prefill_recurrent_fullT2() throws {
        try Self.runRecurrentSeq(numKeyHeads: 16, numValueHeads: 32, headDim: 128, tokens: 2, seed: 0x1202)
    }
    @Test func prefill_recurrent_fullT7() throws {
        try Self.runRecurrentSeq(numKeyHeads: 16, numValueHeads: 32, headDim: 128, tokens: 7, seed: 0x1203)
    }
    @Test func prefill_recurrent_smallT3() throws {
        try Self.runRecurrentSeq(numKeyHeads: 4, numValueHeads: 8, headDim: 32, tokens: 3, seed: 0x1204)
    }
    @Test func prefill_recurrent_gqa1() throws {
        // K == V: key-head index equals value-head index (no repeat).
        try Self.runRecurrentSeq(numKeyHeads: 8, numValueHeads: 8, headDim: 64, tokens: 5, seed: 0x1205)
    }
    @Test func prefill_recurrent_gqa3() throws {
        // The real Qwen 3.8 ratio: 48 value heads over 16 key heads.
        try Self.runRecurrentSeq(numKeyHeads: 16, numValueHeads: 48, headDim: 128, tokens: 4, seed: 0x1206)
        try Self.runRecurrentSeq(numKeyHeads: 4, numValueHeads: 12, headDim: 32, tokens: 3, seed: 0x1207)
    }

    // MARK: - Batched gate (fp16 a|b QMM output → g/beta)

    private static func runGateBatch(numValueHeads V: Int, tokens T: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-prefill-gate-V\(V)-T\(T)")
        let abF32  = (0..<(T*2*V)).map { _ in rng.uniform(-3.0, 3.0) }
        let A_log  = (0..<V).map { _ in rng.uniform(-2.0, 2.0) }
        let dt_bias = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }

        // Kernel reads the QMM output as fp16 → reference uses rounded values.
        let ab16 = abF32.map { Float16($0) }; let abRef = ab16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let abBuf = Fp16Buffer.make(ctx.device, halves: ab16),
              let AlogBuf = makeFp32Buffer(ctx.device, A_log),
              let dtBuf   = makeFp32Buffer(ctx.device, dt_bias),
              let gBuf    = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: T*V)),
              let betaBuf = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: T*V)) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGateBatch(
            commandBuffer: cb, ab: abBuf, A_log: AlogBuf, dt_bias: dtBuf,
            g: gBuf, beta: betaBuf, numValueHeads: V, tokens: T)
        cb.commit(); cb.waitUntilCompleted()

        let ref = GDNPrefillRef.gateBatch(
            ab: abRef, A_log: A_log, dt_bias: dt_bias, numValueHeads: V, tokens: T)
        let gActual    = readFp32(gBuf, count: T*V)
        let betaActual = readFp32(betaBuf, count: T*V)
        #expect(RelError.compute(actual: gActual, reference: ref.g) < Tolerance.fp16ChainedReduction,
                "gate V=\(V) T=\(T): g relErr=\(RelError.compute(actual: gActual, reference: ref.g)) maxAbs=\(RelError.maxAbsDiff(gActual, ref.g))")
        #expect(RelError.compute(actual: betaActual, reference: ref.beta) < Tolerance.fp16ChainedReduction,
                "gate V=\(V) T=\(T): beta relErr=\(RelError.compute(actual: betaActual, reference: ref.beta)) maxAbs=\(RelError.maxAbsDiff(betaActual, ref.beta))")
    }

    @Test func prefill_gate_full()  throws { try Self.runGateBatch(numValueHeads: 32, tokens: 128, seed: 0x1301) }
    @Test func prefill_gate_small() throws { try Self.runGateBatch(numValueHeads: 8,  tokens: 3,   seed: 0x1302) }

    // MARK: - Batched gated RMSNorm

    private static func runRMSNormGatedBatch(numValueHeads V: Int, headDim D: Int, tokens T: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-prefill-norm-V\(V)-D\(D)-T\(T)")
        let xF32 = (0..<(T*V*D)).map { _ in rng.uniform(-2.0, 2.0) }
        let zF32 = (0..<(T*V*D)).map { _ in rng.uniform(-2.0, 2.0) }
        let wF32 = (0..<D).map   { _ in rng.uniform(0.5, 1.5) }

        // Kernel reads fp16 x/z and bf16 weight → reference uses those rounded inputs.
        let x16 = xF32.map { Float16($0) }; let xRef = x16.map { Float($0) }
        let z16 = zF32.map { Float16($0) }; let zRef = z16.map { Float($0) }
        let wBits = wF32.map { Quantization.bf16Bits($0) }
        let wRef = wBits.map { Float(Quantization.bf16ToFloat($0)) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let xBuf  = Fp16Buffer.make(ctx.device, halves: x16),
              let zBuf  = Fp16Buffer.make(ctx.device, halves: z16),
              let wBuf  = makeBF16Buffer(ctx.device, wF32),
              let oBuf  = Fp16Buffer.make(ctx.device, count: T*V*D) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRMSNormGatedBatch(
            commandBuffer: cb, x: xBuf, z: zBuf, weight: wBuf, out: oBuf,
            headDim: UInt32(D), numValueHeads: V, tokens: T)
        cb.commit(); cb.waitUntilCompleted()

        let ref = GDNPrefillRef.rmsNormGatedBatch(
            x: xRef, z: zRef, weight: wRef, numValueHeads: V, headDim: D, tokens: T)

        let outActual = Fp16Buffer.read(oBuf, count: T*V*D)
        let rel = RelError.compute(actual: outActual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "norm-gated V=\(V) D=\(D) T=\(T): relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(outActual, ref))")
    }

    @Test func prefill_rmsnorm_gated_full() throws {
        try Self.runRMSNormGatedBatch(numValueHeads: 32, headDim: 128, tokens: 5, seed: 0x1401)
    }
    @Test func prefill_rmsnorm_gated_small() throws {
        try Self.runRMSNormGatedBatch(numValueHeads: 8, headDim: 32, tokens: 3, seed: 0x1402)
    }

    // MARK: - Composed GDN chunk (conv → gate → recurrent → gated norm)
    //
    // One GDN layer chunk at full shape, with the fp16 rounding at each stage
    // boundary the production path applies (conv out is fp16, recurrent out is
    // fp16 and feeds the norm in place).

    private static func runComposedChunk(tokens T: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-prefill-composed-T\(T)")
        let K = 16, V = 32, D = 128
        let scale = 1.0 / sqrtf(Float(D))
        let keyDim = K * D
        let valueDim = V * D
        let C = 2 * keyDim + valueDim
        let kOff = keyDim
        let vOff = 2 * keyDim

        let wF32   = (0..<(C*4)).map { _ in rng.uniform(-0.5, 0.5) }
        let sF32   = (0..<(C*3)).map { _ in rng.uniform(-0.5, 0.5) }
        let xF32   = (0..<(T*C)).map { _ in rng.uniform(-1.0, 1.0) }
        let zF32   = (0..<(T*valueDim)).map { _ in rng.uniform(-2.0, 2.0) }
        let nwF32  = (0..<D).map { _ in rng.uniform(0.5, 1.5) }
        let abF32  = (0..<(T*2*V)).map { _ in rng.uniform(-3.0, 3.0) }
        let A_log  = (0..<V).map { _ in rng.uniform(-2.0, 2.0) }
        let dt_bias = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }
        var stateF32 = (0..<(V*D*D)).map { _ in rng.uniform(-0.5, 0.5) }

        // fp16/bf16 rounding of every kernel input.
        let w16 = wF32.map { Float16($0) }; let wRef = w16.map { Float($0) }
        let s16 = sF32.map { Float16($0) }; let sRef = s16.map { Float($0) }
        let x16 = xF32.map { Float16($0) }; let xRef = x16.map { Float($0) }
        let z16 = zF32.map { Float16($0) }; let zRef = z16.map { Float($0) }
        let ab16 = abF32.map { Float16($0) }; let abRef = ab16.map { Float($0) }
        let nwBits = nwF32.map { Quantization.bf16Bits($0) }
        let nwRef = nwBits.map { Float(Quantization.bf16ToFloat($0)) }

        let ctx = try MetalContext()
        let kernel = try GDNPrefill(context: ctx)

        guard let wBuf = Fp16Buffer.make(ctx.device, halves: w16),
              let sBuf = Fp16Buffer.make(ctx.device, halves: s16),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let convOutBuf = Fp16Buffer.make(ctx.device, count: T*C),
              let zBuf = Fp16Buffer.make(ctx.device, halves: z16),
              let abBuf = Fp16Buffer.make(ctx.device, halves: ab16),
              let nwBuf = makeBF16Buffer(ctx.device, nwF32),
              let AlogBuf = makeFp32Buffer(ctx.device, A_log),
              let dtBuf   = makeFp32Buffer(ctx.device, dt_bias),
              let gBuf    = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: T*V)),
              let betaBuf = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: T*V)),
              let nsBuf = Fp16Buffer.make(ctx.device, count: C*3),
              let recOutBuf = Fp16Buffer.make(ctx.device, count: T*valueDim),
              let stBuf = makeFp32Buffer(ctx.device, stateF32) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        // conv: separate in/out (the batched form races if aliased), state
        // committed to sBuf via blit
        kernel.encodeConvChunk(commandBuffer: cb, w: wBuf, state: sBuf,
                               x: xBuf, out: convOutBuf, newState: nsBuf,
                               channels: C, tokens: T)
        kernel.encodeGateBatch(commandBuffer: cb, ab: abBuf, A_log: AlogBuf,
                               dt_bias: dtBuf, g: gBuf, beta: betaBuf,
                               numValueHeads: V, tokens: T)
        kernel.encodeRecurrentSeq(commandBuffer: cb, state: stBuf, conv: convOutBuf,
                                  g: gBuf, beta: betaBuf, out: recOutBuf,
                                  headDim: UInt32(D), channels: UInt32(C),
                                  kOffset: UInt32(kOff), vOffset: UInt32(vOff),
                                  numValueHeads: V, numKeyHeads: K,
                                  tokens: T, scale: scale, l2eps: l2Eps)
        // gated norm in place on recOutBuf (out == x)
        kernel.encodeRMSNormGatedBatch(commandBuffer: cb, x: recOutBuf, z: zBuf,
                                       weight: nwBuf, out: recOutBuf,
                                       headDim: UInt32(D), numValueHeads: V, tokens: T)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: same stages with the fp16 boundary roundings.
        let (convOutRef, _) = GDNPrefillRef.convChunk(
            w: wRef, state: sRef, x: xRef, channels: C, tokens: T)
        let convOut16 = convOutRef.map { Float(Float16($0)) }
        let (gRef, betaRef) = GDNPrefillRef.gateBatch(
            ab: abRef, A_log: A_log, dt_bias: dt_bias, numValueHeads: V, tokens: T)
        var refState = stateF32
        let recOutRef = GDNPrefillRef.recurrentChunk(
            state: &refState, conv: convOut16, g: gRef, beta: betaRef,
            channels: C, kOffset: kOff, vOffset: vOff,
            numValueHeads: V, numKeyHeads: K, headDim: D, tokens: T, scale: scale)
        let recOut16 = recOutRef.map { Float(Float16($0)) }
        let normRef = GDNPrefillRef.rmsNormGatedBatch(
            x: recOut16, z: zRef, weight: nwRef, numValueHeads: V, headDim: D, tokens: T)

        let normActual = Fp16Buffer.read(recOutBuf, count: T*valueDim)
        let rel = RelError.compute(actual: normActual, reference: normRef)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "composed T=\(T): norm relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(normActual, normRef))")

        let stateActual = readFp32(stBuf, count: V*D*D)
        let stateRel = RelError.compute(actual: stateActual, reference: refState)
        #expect(stateRel < Tolerance.fp16ChainedReduction,
                "composed T=\(T): state relErr=\(stateRel) maxAbs=\(RelError.maxAbsDiff(stateActual, refState))")
    }

    @Test func prefill_composedChunk_full() throws {
        try Self.runComposedChunk(tokens: 4, seed: 0x1501)
    }
    @Test func prefill_composedChunk_t1() throws {
        try Self.runComposedChunk(tokens: 1, seed: 0x1502)
    }

    // MARK: - Buffer helpers

    private static func makeFp32Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        let buf = device.makeBuffer(length: values.count * MemoryLayout<Float>.size,
                                    options: .storageModeShared)
        guard let buf else { return nil }
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: values.count)
        for (i, v) in values.enumerated() { ptr[i] = v }
        return buf
    }

    private static func readFp32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { ptr[$0] }
    }

    private static func makeBF16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { Quantization.bf16Bits($0) }
        let buf = device.makeBuffer(length: bits.count * MemoryLayout<UInt16>.size,
                                    options: .storageModeShared)
        guard let buf else { return nil }
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for (i, v) in bits.enumerated() { ptr[i] = v }
        return buf
    }
}

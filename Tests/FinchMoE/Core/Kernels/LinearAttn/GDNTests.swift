import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares the Metal `gdn` kernels against the fp32 `GDNRef` reference.
///
/// Two operations are validated, both grounded in `qwen3_5_moe`:
///   * `gdn_conv_update`  — causal conv1d decode update (kernel 4).
///   * `gdn_recurrent`    — per-value-head gated-delta-rule recurrence
///     (decay → residual read → rank-1 write → updated-state readout).
///
/// The recurrent state is fp32 and the kernel does a two-stage block reduce
/// for the two l2norm sums; the reference is a naive per-element fp32 loop.
/// Different op-trees, so a matching result within `fp16ChainedReduction` is
/// a real check on the recurrence order and the v-major state layout.
@Suite struct GDNTests {

    // MARK: - Causal conv1d update
    private static let l2Eps: Float = 1e-6

    private static func runConv(channels: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-conv-c\(channels)")
        let C = channels
        let wF32  = (0..<(C*4)).map { _ in rng.uniform(-1.0, 1.0) }
        let sF32  = (0..<(C*3)).map { _ in rng.uniform(-1.0, 1.0) }
        let xF32  = (0..<C).map   { _ in rng.uniform(-1.0, 1.0) }

        // The kernel reads fp16, so the reference uses the fp16-rounded inputs.
        let w16 = wF32.map { Float16($0) }; let wRef = w16.map { Float($0) }
        let s16 = sF32.map { Float16($0) }; let sRef = s16.map { Float($0) }
        let x16 = xF32.map { Float16($0) }; let xRef = x16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDN(context: ctx)

        guard let wBuf = Fp16Buffer.make(ctx.device, halves: w16),
              let sBuf = Fp16Buffer.make(ctx.device, halves: s16),
              let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let oBuf = Fp16Buffer.make(ctx.device, count: C),
              let nsBuf = Fp16Buffer.make(ctx.device, count: C*3) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCausalConvUpdate(
            commandBuffer: cb, w: wBuf, state: sBuf, x: xBuf,
            out: oBuf, newState: nsBuf, channels: C)
        cb.commit(); cb.waitUntilCompleted()

        let (refOut, refNS) = GDNRef.causalConvUpdate(w: wRef, state: sRef, x: xRef)

        let outActual = Fp16Buffer.read(oBuf, count: C)
        let outRel = RelError.compute(actual: outActual, reference: refOut)
        #expect(outRel < Tolerance.fp16ChainedReduction,
                "conv C=\(C): out relErr=\(outRel) maxAbs=\(RelError.maxAbsDiff(outActual, refOut))")

        let nsActual = Fp16Buffer.read(nsBuf, count: C*3)
        let nsRel = RelError.compute(actual: nsActual, reference: refNS)
        #expect(nsRel < Tolerance.fp16ChainedReduction,
                "conv C=\(C): newState relErr=\(nsRel) maxAbs=\(RelError.maxAbsDiff(nsActual, refNS))")
    }

    @Test func gdn_conv_update_small() throws { try Self.runConv(channels: 128, seed: 0x101) }
    @Test func gdn_conv_update_8192() throws { try Self.runConv(channels: 8192, seed: 0x102) }

    // MARK: - Recurrent gated-delta-rule decode step
    //
    // Real Qwen3.6 GDN dims: 16 key heads, 32 value heads, head dim 128.
    // Validates the readout vector AND the mutated fp32 state (decay + rank-1
    // write), which is the part most likely to be subtly wrong.

    private static func runRecurrent(
        numKeyHeads K: Int, numValueHeads V: Int, headDim D: Int, seed: UInt64
    ) throws {
        var rng = SeedTree(seed).key("gdn-rec-V\(V)-D\(D)")
        let scale = 1.0 / sqrtf(Float(D))

        let qF32 = (0..<(K*D)).map { _ in rng.uniform(-1.0, 1.0) }
        let kF32 = (0..<(K*D)).map { _ in rng.uniform(-1.0, 1.0) }
        let vF32 = (0..<(V*D)).map { _ in rng.uniform(-1.0, 1.0) }
        let g    = (0..<V).map   { _ in rng.uniform(-3.0, 0.0) }   // decay in [0.05, 1]
        let beta = (0..<V).map   { _ in rng.uniform(0.0, 1.0) }
        var state = (0..<(V*D*D)).map { _ in rng.uniform(-0.5, 0.5) }

        // Kernel reads q/k/v as fp16 → reference uses the rounded values.
        let q16 = qF32.map { Float16($0) }; let qRef = q16.map { Float($0) }
        let k16 = kF32.map { Float16($0) }; let kRef = k16.map { Float($0) }
        let v16 = vF32.map { Float16($0) }; let vRef = v16.map { Float($0) }

        let ctx = try MetalContext()
        let kernel = try GDN(context: ctx)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k16),
              let vBuf = Fp16Buffer.make(ctx.device, halves: v16),
              let oBuf = Fp16Buffer.make(ctx.device, count: V*D),
              let gBuf = makeFp32Buffer(ctx.device, g),
              let bBuf = makeFp32Buffer(ctx.device, beta),
              let sBuf = makeFp32Buffer(ctx.device, state) else {
            Issue.record("alloc failed"); return
        }

        // Reference: per value-head, mutate the fp32 state copy, collect out.
        // The pairing is the checkpoint's grouped (repeat_interleave) order,
        // so the divisor is `V / K` — not a literal (Qwen 3.6 is 32/16, but
        // 3.8 is 48/16, and a hardcoded 2 reads the wrong q/k there).
        let vPerK = V / K
        var refState = state
        var refOut = [Float](repeating: 0, count: V*D)
        for hv in 0..<V {
            let kh = hv / vPerK
            let sBase = hv * D * D
            var slice = Array(refState[sBase..<(sBase + D*D)])
            let o = GDNRef.recurrentStep(
                state: &slice, q: Array(qRef[(kh*D)..<(kh*D + D)]), k: Array(kRef[(kh*D)..<(kh*D + D)]),
                v: Array(vRef[(hv*D)..<(hv*D + D)]), g: g[hv], beta: beta[hv], scale: scale)
            for (i, val) in o.enumerated() { refOut[hv*D + i] = val }
            refState.replaceSubrange(sBase..<(sBase + D*D), with: slice)
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRecurrent(
            commandBuffer: cb, state: sBuf, q: qBuf, k: kBuf, v: vBuf,
            g: gBuf, beta: bBuf, out: oBuf,
            numValueHeads: V, numKeyHeads: K,
            headDim: UInt32(D), scale: scale, l2eps: l2Eps)
        cb.commit(); cb.waitUntilCompleted()

        let outActual = Fp16Buffer.read(oBuf, count: V*D)
        let outRel = RelError.compute(actual: outActual, reference: refOut)
        #expect(outRel < Tolerance.fp16ChainedReduction,
                "recurrent V=\(V) D=\(D): out relErr=\(outRel) maxAbs=\(RelError.maxAbsDiff(outActual, refOut))")

        let stateActual = readFp32(sBuf, count: V*D*D)
        let stateRel = RelError.compute(actual: stateActual, reference: refState)
        #expect(stateRel < Tolerance.fp16ChainedReduction,
                "recurrent V=\(V) D=\(D): state relErr=\(stateRel) maxAbs=\(RelError.maxAbsDiff(stateActual, refState))")
    }

    @Test func gdn_recurrent_fullShape() throws {
        try Self.runRecurrent(numKeyHeads: 16, numValueHeads: 32, headDim: 128, seed: 0x201)
    }
    @Test func gdn_recurrent_smallShape() throws {
        try Self.runRecurrent(numKeyHeads: 4, numValueHeads: 8, headDim: 32, seed: 0x202)
    }
    @Test func gdn_recurrent_gqa1() throws {
        // K == V: key-head index equals value-head index (no repeat).
        try Self.runRecurrent(numKeyHeads: 8, numValueHeads: 8, headDim: 64, seed: 0x203)
    }
    @Test func gdn_recurrent_gqa3() throws {
        // The real Qwen 3.8 ratio: 48 value heads over 16 key heads. This is
        // the case the hardcoded `hv / 2` got wrong for every layer of the
        // real model while every ratio-2 toy passed.
        try Self.runRecurrent(numKeyHeads: 16, numValueHeads: 48, headDim: 128, seed: 0x204)
        try Self.runRecurrent(numKeyHeads: 4, numValueHeads: 12, headDim: 32, seed: 0x205)
    }

    // MARK: - Per-value-head gate
    //
    //   beta = sigmoid(b)
    //   g    = -exp(A_log) * softplus(a + dt_bias)
    //
    // a/b/A_log/dt_bias are small per-head fp32 vectors; the kernel computes in
    // fp32, so the reference uses the same fp32 inputs directly.

    private static func runGate(numValueHeads V: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-gate-V\(V)")
        let a      = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }
        let b      = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }
        let A_log  = (0..<V).map { _ in rng.uniform(-2.0, 2.0) }
        let dt_bias = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }

        let ctx = try MetalContext()
        let kernel = try GDN(context: ctx)

        guard let aBuf    = makeFp32Buffer(ctx.device, a),
              let bBuf    = makeFp32Buffer(ctx.device, b),
              let AlogBuf = makeFp32Buffer(ctx.device, A_log),
              let dtBuf   = makeFp32Buffer(ctx.device, dt_bias),
              let gBuf    = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: V)),
              let betaBuf = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: V)) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGate(
            commandBuffer: cb,
            a: aBuf, b: bBuf, A_log: AlogBuf, dt_bias: dtBuf,
            g: gBuf, beta: betaBuf, numValueHeads: V)
        cb.commit(); cb.waitUntilCompleted()

        let ref = GDNRef.gate(a: a, b: b, A_log: A_log, dt_bias: dt_bias)
        let gActual    = readFp32(gBuf, count: V)
        let betaActual = readFp32(betaBuf, count: V)
        #expect(RelError.compute(actual: gActual, reference: ref.g) < Tolerance.fp16ChainedReduction,
                "gate V=\(V): g relErr=\(RelError.compute(actual: gActual, reference: ref.g)) maxAbs=\(RelError.maxAbsDiff(gActual, ref.g))")
        #expect(RelError.compute(actual: betaActual, reference: ref.beta) < Tolerance.fp16ChainedReduction,
                "gate V=\(V): beta relErr=\(RelError.compute(actual: betaActual, reference: ref.beta)) maxAbs=\(RelError.maxAbsDiff(betaActual, ref.beta))")
    }

    @Test func gdn_gate_32heads() throws { try Self.runGate(numValueHeads: 32, seed: 0x301) }
    @Test func gdn_gate_8heads()  throws { try Self.runGate(numValueHeads: 8,  seed: 0x302) }

    // MARK: - Fused in_proj_a/in_proj_b GEMV + gate (`gdn_gate_gemv`)
    //
    // The production path runs the two [V, N] int4-affine GEMVs and the gate
    // formula in one 256-thread threadgroup (8 SIMD groups, 8 sequential
    // row passes into threadgroup fp32 staging). The reference GEMVs go
    // through `DequantInt4GemvRef` (bulk-dequant + vDSP_dotpr) and
    // `GDNRef.gate` — different op-trees on both stages.

    private static func runGateGEMV(numValueHeads V: Int, n: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("gdn-gate-gemv-V\(V)-n\(n)")
        let aW = (0..<V).map { _ in (0..<n).map { _ in rng.uniform(-0.2, 0.2) } }
        let bW = (0..<V).map { _ in (0..<n).map { _ in rng.uniform(-0.2, 0.2) } }
        let xF32 = (0..<n).map { _ in rng.uniform(-0.4, 0.4) }
        let A_log  = (0..<V).map { _ in rng.uniform(-2.0, 2.0) }
        let dt_bias = (0..<V).map { _ in rng.uniform(-3.0, 3.0) }

        let x16 = xF32.map { Float16($0) }
        let xRef = x16.map { Float($0) }
        let aRows = aW.map(Quantization.quantizeInt4Affine)
        let bRows = bW.map(Quantization.quantizeInt4Affine)
        let allRows = aRows + bRows
        let packed = allRows.flatMap(\.packed)
        let scales = allRows.flatMap(\.scales)
        let biases = allRows.flatMap(\.biases)

        let ctx = try MetalContext()
        let kernel = try GDN(context: ctx)

        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let wBuf = ctx.device.makeBuffer(bytes: packed,
                                               length: packed.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: scales,
                                               length: scales.count * 2,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: biases,
                                               length: biases.count * 2,
                                               options: .storageModeShared),
              let AlogBuf = makeFp32Buffer(ctx.device, A_log),
              let dtBuf   = makeFp32Buffer(ctx.device, dt_bias),
              let gBuf    = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: V)),
              let betaBuf = makeFp32Buffer(ctx.device, [Float](repeating: 0, count: V)) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGateGEMV(
            commandBuffer: cb,
            weights: wBuf, scales: sBuf, biases: bBuf, x: xBuf,
            A_log: AlogBuf, dt_bias: dtBuf,
            g: gBuf, beta: betaBuf,
            numValueHeads: V, n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let ref = GDNRef.gateGEMV(aRows: aRows, bRows: bRows,
                                  x: xRef, A_log: A_log, dt_bias: dt_bias)
        let gActual    = readFp32(gBuf, count: V)
        let betaActual = readFp32(betaBuf, count: V)
        let gRel    = RelError.compute(actual: gActual, reference: ref.g)
        let betaRel = RelError.compute(actual: betaActual, reference: ref.beta)
        #expect(gRel < Tolerance.fp16ChainedReduction,
                "gate-gemv V=\(V) n=\(n): g relErr=\(gRel) maxAbs=\(RelError.maxAbsDiff(gActual, ref.g))")
        #expect(betaRel < Tolerance.fp16ChainedReduction,
                "gate-gemv V=\(V) n=\(n): beta relErr=\(betaRel) maxAbs=\(RelError.maxAbsDiff(betaActual, ref.beta))")
    }

    @Test func gdn_gate_gemv_fullShape() throws { try Self.runGateGEMV(numValueHeads: 32, n: 2048, seed: 0x303) }
    @Test func gdn_gate_gemv_smallShape() throws { try Self.runGateGEMV(numValueHeads: 8, n: 256, seed: 0x304) }

    // MARK: - Gated RMSNorm (per value head)
    //
    //   y[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i] * act(z[i])
    //
    // act = silu on Qwen 3.5/3.6, sigmoid on Qwen 3.8 Flash-Next (`qwen4exp`
    // `build_norm_gated` — the sigmoid is a function-constant PSO variant of
    // the same kernel). `weight` is a single bf16 vector of length headDim
    // shared across the V heads (Qwen3_5MoeRMSNormGated(head_v_dim)). The
    // reference is a naive per-head fp32 loop — a different op-tree than the
    // kernel's block reduce.

    private static func runRMSNormGated(numValueHeads V: Int, headDim D: Int,
                                        seed: UInt64, sigmoidGate: Bool = false) throws {
        var rng = SeedTree(seed).key("gdn-norm-V\(V)-D\(D)")
        let xF32 = (0..<(V*D)).map { _ in rng.uniform(-2.0, 2.0) }
        let zF32 = (0..<(V*D)).map { _ in rng.uniform(-2.0, 2.0) }
        let wF32 = (0..<D).map   { _ in rng.uniform(0.5, 1.5) }

        // Kernel reads fp16 x/z and bf16 weight, so the reference uses those
        // exact rounded inputs.
        let x16 = xF32.map { Float16($0) }; let xRef = x16.map { Float($0) }
        let z16 = zF32.map { Float16($0) }; let zRef = z16.map { Float($0) }
        let wBits = wF32.map { Quantization.bf16Bits($0) }
        let wRef = wBits.map { Float(Quantization.bf16ToFloat($0)) }

        let ctx = try MetalContext()
        let kernel = try GDN(context: ctx)

        guard let xBuf  = Fp16Buffer.make(ctx.device, halves: x16),
              let zBuf  = Fp16Buffer.make(ctx.device, halves: z16),
              let wBuf  = makeBF16Buffer(ctx.device, wF32),
              let oBuf  = Fp16Buffer.make(ctx.device, count: V*D) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeRMSNormGated(
            commandBuffer: cb,
            x: xBuf, z: zBuf, weight: wBuf, out: oBuf,
            numValueHeads: V, headDim: UInt32(D),
            activation: sigmoidGate ? .sigmoid : .silu)
        cb.commit(); cb.waitUntilCompleted()

        var ref = [Float](repeating: 0, count: V*D)
        for h in 0..<V {
            let xs = Array(xRef[(h*D)..<(h*D+D)])
            let zs = Array(zRef[(h*D)..<(h*D+D)])
            if sigmoidGate {
                let n = xs.count
                var ss: Float = 0
                for v in xs { ss += v * v }
                let inv = 1.0 / (ss / Float(n) + 1e-6).squareRoot()
                let y = (0..<n).map { i in
                    xs[i] * inv * wRef[i] * (1.0 / (1.0 + exp(-zs[i])))
                }
                ref.replaceSubrange((h*D)..<(h*D+D), with: y)
            } else {
                let y = GDNRef.rmsNormGated(xs, weight: wRef, z: zs)
                ref.replaceSubrange((h*D)..<(h*D+D), with: y)
            }
        }

        let outActual = Fp16Buffer.read(oBuf, count: V*D)
        let rel = RelError.compute(actual: outActual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "norm-gated V=\(V) D=\(D): relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(outActual, ref))")
    }

    @Test func gdn_rmsnorm_gated_fullShape() throws {
        try Self.runRMSNormGated(numValueHeads: 32, headDim: 128, seed: 0x401)
    }
    @Test func gdn_rmsnorm_gated_smallShape() throws {
        try Self.runRMSNormGated(numValueHeads: 8, headDim: 32, seed: 0x402)
    }
    @Test func gdn_rmsnorm_gated_sigmoid_fullShape() throws {
        try Self.runRMSNormGated(numValueHeads: 32, headDim: 128, seed: 0x411,
                                 sigmoidGate: true)
    }
    @Test func gdn_rmsnorm_gated_sigmoid_smallShape() throws {
        try Self.runRMSNormGated(numValueHeads: 8, headDim: 32, seed: 0x412,
                                 sigmoidGate: true)
    }

    // MARK: - fp32 buffer helpers (state, g, beta are fp32)
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

    // bf16 weight buffer (shared per-head norm weight is a bf16 vector).
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

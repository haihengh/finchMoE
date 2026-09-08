import Testing
import Foundation
import Metal
@testable import FlashQwen
import FlashQwenValidationSupport

/// Compares the Qwen 3.6 decode-layer fusions (`Metal/Qwen/qwen_decode.metal`)
/// against the fp32 `QwenDecodeRef`. Inputs are fp16/bf16-rounded and the
/// rounded values feed the reference, matching the discipline of the other
/// kernel tests. The reference op-trees differ deliberately (Accelerate norm,
/// bulk vForce trig, bulk-dequant GEMV vs the kernel's block reductions and
/// scalar loops).
@Suite struct QwenDecodeFusionsTests {

    private static let eps: Float = 1e-6

    // MARK: - qwen_post_attn (residual add + post_attention_layernorm)

    private static func runPostAttn(d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("qwen-post-attn-d\(d)")
        let hF32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let aF32 = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let wF32 = (0..<d).map { _ in rng.uniform(0.5, 1.5) }
        let h16 = hF32.map { Float16($0) }; let hRef = h16.map { Float($0) }
        let a16 = aF32.map { Float16($0) }; let aRef = a16.map { Float($0) }
        let wBits = wF32.map { Quantization.bf16Bits($0) }
        let wRef = wBits.map { Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try QwenDecodeFusions(context: ctx)
        guard let hBuf = Fp16Buffer.make(ctx.device, halves: h16),
              let aBuf = Fp16Buffer.make(ctx.device, halves: a16),
              let oBuf = Fp16Buffer.make(ctx.device, count: d),
              let wBuf = ctx.device.makeBuffer(length: d * 2,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: d)
        for i in 0..<d { wPtr[i] = wBits[i] }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodePostAttn(commandBuffer: cb, hidden: hBuf, attn: aBuf,
                              out: oBuf, weight: wBuf, d: UInt32(d), eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDecodeRef.postAttn(hidden: hRef, attn: aRef, weight: wRef)
        let hActual = Fp16Buffer.read(hBuf, count: d)
        let oActual = Fp16Buffer.read(oBuf, count: d)
        let hRel = RelError.compute(actual: hActual, reference: ref.hidden)
        let oRel = RelError.compute(actual: oActual, reference: ref.out)
        #expect(hRel < Tolerance.fp16Reduction,
                "post-attn D=\(d): hidden relErr=\(hRel) maxAbs=\(RelError.maxAbsDiff(hActual, ref.hidden))")
        #expect(oRel < Tolerance.fp16Reduction,
                "post-attn D=\(d): out relErr=\(oRel) maxAbs=\(RelError.maxAbsDiff(oActual, ref.out))")
    }

    @Test func qwenPostAttn_d2048() throws { try Self.runPostAttn(d: 2048, seed: 0x711) }
    @Test func qwenPostAttn_d512()  throws { try Self.runPostAttn(d: 512, seed: 0x712) }

    // MARK: - vec_add_fp16

    @Test func vecAdd_d2048() throws {
        var rng = SeedTree(0x721).key("qwen-vec-add")
        let d = 2048
        let a16 = (0..<d).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
        let b16 = (0..<d).map { _ in Float16(rng.uniform(-2.0, 2.0)) }

        let ctx = try MetalContext()
        let kernel = try QwenDecodeFusions(context: ctx)
        guard let aBuf = Fp16Buffer.make(ctx.device, halves: a16),
              let bBuf = Fp16Buffer.make(ctx.device, halves: b16) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeVecAdd(commandBuffer: cb, a: aBuf, b: bBuf, d: UInt32(d))
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDecodeRef.vecAdd(a16.map { Float($0) }, b16.map { Float($0) })
        let actual = Fp16Buffer.read(aBuf, count: d)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16Reduction,
                "vec-add: relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    // MARK: - qwen_attn_output_gate

    @Test func attnOutputGate_n4096() throws {
        var rng = SeedTree(0x731).key("qwen-attn-gate")
        let n = 4096
        let a16 = (0..<n).map { _ in Float16(rng.uniform(-3.0, 3.0)) }
        let g16 = (0..<n).map { _ in Float16(rng.uniform(-6.0, 6.0)) }

        let ctx = try MetalContext()
        let kernel = try QwenDecodeFusions(context: ctx)
        guard let aBuf = Fp16Buffer.make(ctx.device, halves: a16),
              let gBuf = Fp16Buffer.make(ctx.device, halves: g16) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeAttnOutputGate(commandBuffer: cb, attn: aBuf, gate: gBuf,
                                    n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDecodeRef.attnOutputGate(attn: a16.map { Float($0) },
                                               gate: g16.map { Float($0) })
        let actual = Fp16Buffer.read(aBuf, count: n)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16Reduction,
                "attn-gate: relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    // MARK: - qwen_shared_gate (shared_expert_gate GEMV + sigmoid scale)

    @Test func sharedGate_n2048() throws {
        var rng = SeedTree(0x741).key("qwen-shared-gate")
        let n = 2048
        let wRow = (0..<n).map { _ in rng.uniform(-0.2, 0.2) }
        let xF32 = (0..<n).map { _ in rng.uniform(-0.4, 0.4) }
        let hF32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let x16 = xF32.map { Float16($0) }
        let xRef = x16.map { Float($0) }
        let h16 = hF32.map { Float16($0) }
        let hRef = h16.map { Float($0) }
        let packed = Quantization.quantizeInt4Affine(wRow)

        let ctx = try MetalContext()
        let kernel = try QwenDecodeFusions(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: x16),
              let hBuf = Fp16Buffer.make(ctx.device, halves: h16),
              let wBuf = ctx.device.makeBuffer(bytes: packed.packed,
                                               length: packed.packed.count,
                                               options: .storageModeShared),
              let sBuf = ctx.device.makeBuffer(bytes: packed.scales,
                                               length: packed.scales.count * 2,
                                               options: .storageModeShared),
              let bBuf = ctx.device.makeBuffer(bytes: packed.biases,
                                               length: packed.biases.count * 2,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSharedGate(commandBuffer: cb,
                                weights: wBuf, scales: sBuf, biases: bBuf,
                                x: xBuf, h1: hBuf,
                                n: UInt32(n), d: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDecodeRef.sharedGate(weightRow: packed, x: xRef, h1: hRef)
        let actual = Fp16Buffer.read(hBuf, count: n)
        let rel = RelError.compute(actual: actual, reference: ref)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "shared-gate: relErr=\(rel) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    // MARK: - qwen_full_attn_epilogue

    private static func runFullAttnEpilogue(
        numQHeads: Int, numKVHeads: Int, headDim: Int, rotaryDim: Int, seed: UInt64
    ) throws {
        var rng = SeedTree(seed).key("qwen-full-attn-q\(numQHeads)-kv\(numKVHeads)-hd\(headDim)")
        let qLen = numQHeads * 2 * headDim
        let kLen = numKVHeads * headDim
        let qF32 = (0..<qLen).map { _ in rng.uniform(-2.0, 2.0) }
        let kF32 = (0..<kLen).map { _ in rng.uniform(-2.0, 2.0) }
        let qwF32 = (0..<headDim).map { _ in rng.uniform(0.5, 1.5) }
        let kwF32 = (0..<headDim).map { _ in rng.uniform(0.5, 1.5) }
        let position = 7
        let theta: Float = 10_000_000.0

        let q16 = qF32.map { Float16($0) }; let qRef = q16.map { Float($0) }
        let k16 = kF32.map { Float16($0) }; let kRef = k16.map { Float($0) }
        let qwBits = qwF32.map { Quantization.bf16Bits($0) }
        let qwRef = qwBits.map { Quantization.bf16ToFloat($0) }
        let kwBits = kwF32.map { Quantization.bf16Bits($0) }
        let kwRef = kwBits.map { Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try QwenDecodeFusions(context: ctx)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q16),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k16),
              let qoBuf = Fp16Buffer.make(ctx.device, count: numQHeads * headDim),
              let goBuf = Fp16Buffer.make(ctx.device, count: numQHeads * headDim),
              let qwBuf = ctx.device.makeBuffer(length: headDim * 2,
                                                options: .storageModeShared),
              let kwBuf = ctx.device.makeBuffer(length: headDim * 2,
                                                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let qwPtr = qwBuf.contents().bindMemory(to: UInt16.self, capacity: headDim)
        for i in 0..<headDim { qwPtr[i] = qwBits[i] }
        let kwPtr = kwBuf.contents().bindMemory(to: UInt16.self, capacity: headDim)
        for i in 0..<headDim { kwPtr[i] = kwBits[i] }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeFullAttnEpilogue(
            commandBuffer: cb,
            qProj: qBuf, qOut: qoBuf, gateOut: goBuf, k: kBuf,
            qWeight: qwBuf, kWeight: kwBuf,
            headDim: UInt32(headDim),
            numQHeads: UInt32(numQHeads), numKVHeads: UInt32(numKVHeads),
            position: UInt32(position), theta: theta,
            rotaryDim: UInt32(rotaryDim), eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = QwenDecodeRef.fullAttnEpilogue(
            qProj: qRef, kIn: kRef, qWeight: qwRef, kWeight: kwRef,
            headDim: headDim, numQHeads: numQHeads, numKVHeads: numKVHeads,
            rotaryDim: rotaryDim, position: position, theta: theta)
        let qActual  = Fp16Buffer.read(qoBuf, count: numQHeads * headDim)
        let gActual  = Fp16Buffer.read(goBuf, count: numQHeads * headDim)
        let kActual  = Fp16Buffer.read(kBuf, count: kLen)
        #expect(RelError.compute(actual: qActual, reference: ref.qOut) < Tolerance.fp16ChainedReduction,
                "full-attn epilogue q: relErr=\(RelError.compute(actual: qActual, reference: ref.qOut))")
        #expect(RelError.compute(actual: gActual, reference: ref.gateOut) < Tolerance.fp16Reduction,
                "full-attn epilogue gate copy: relErr=\(RelError.compute(actual: gActual, reference: ref.gateOut))")
        #expect(RelError.compute(actual: kActual, reference: ref.kOut) < Tolerance.fp16ChainedReduction,
                "full-attn epilogue k: relErr=\(RelError.compute(actual: kActual, reference: ref.kOut))")
    }

    /// Production shape: 16 q heads, 2 kv heads, head dim 256, rotary 64
    /// (partial_rotary_factor 0.25), theta 1e7.
    @Test func fullAttnEpilogue_productionShape() throws {
        try Self.runFullAttnEpilogue(numQHeads: 16, numKVHeads: 2,
                                     headDim: 256, rotaryDim: 64, seed: 0x751)
    }
    /// Small shape with full rotary (rotaryDim == headDim).
    @Test func fullAttnEpilogue_fullRotary() throws {
        try Self.runFullAttnEpilogue(numQHeads: 4, numKVHeads: 2,
                                     headDim: 64, rotaryDim: 64, seed: 0x752)
    }
}

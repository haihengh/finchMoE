import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares the Qwen 3.8 hyper-connection kernels (`Metal/Qwen/
/// hyper_connection.metal`) against the fp32 `HyperConnectionRef` and inline
/// scalar transcriptions. Inputs are fp16-rounded (activations) and the
/// grouped-RMS gamma is BF16-rounded, and the rounded values feed the
/// reference — the discipline of the other kernel suites. Where a reference
/// op exists (`groupedRMS`, `combine`) it has a deliberately different op
/// tree (scalar loops vs SIMD block reductions).
///
/// Geometries mirror the reference suite: tiny (hand-checkable), toy
/// (hc 4 × 256, lowrank 64) and real (hc 4 × 2560 = 10240 plane, lowrank
/// 320) — the real widths exercise dispatch beyond one 256-thread group.
@Suite struct HyperConnectionTests {

    private static let eps: Float = 1e-6

    private static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    private static func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }

    private static func makeBF16Buffer(_ device: MTLDevice,
                                       _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { Quantization.bf16Bits($0) }
        guard let buf = device.makeBuffer(length: bits.count * 2,
                                          options: .storageModeShared) else { return nil }
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for i in 0..<bits.count { p[i] = bits[i] }
        return buf
    }

    /// Round fp16 activations, bf16 the gamma; returns fp32 inputs for the
    /// reference plus the prepared buffers (nil on allocation failure).
    private static func seedVectors(
        _ rng: inout SplitMix64, hc: Int, d: Int, device: MTLDevice
    ) -> (x: [Float], gammaRef: [Float], xBuf: MTLBuffer?, gammaBuf: MTLBuffer?,
          outBuf: MTLBuffer?) {
        let n = hc * d
        let xH = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let gH = (0..<n).map { _ in Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(0.5, 1.5))) }
        let xRef = xH.map { Float($0) }
        return (xRef, gH,
                Fp16Buffer.make(device, halves: xH),
                Self.makeBF16Buffer(device, gH),
                Fp16Buffer.make(device, count: n))
    }

    // MARK: - hc_plane_init

    @Test("plane init replicates the hidden stream into every stream")
    func planeInit_replicates() throws {
        for (hc, d, seed) in [(4, 2560, UInt64(0x501)), (3, 333, UInt64(0x502))] {
            var rng = SeedTree(seed).key("hc-plane-init")
            let hidden = (0..<d).map { _ in Float16(rng.uniform(-2.0, 2.0)) }
            let ctx = try MetalContext()
            let kernel = try HyperConnection(context: ctx)
            guard let hBuf = Fp16Buffer.make(ctx.device, halves: hidden),
                  let pBuf = Fp16Buffer.make(ctx.device, count: hc * d) else {
                Issue.record("alloc failed"); continue
            }
            let cb = ctx.queue.makeCommandBuffer()!
            kernel.encodePlaneInit(commandBuffer: cb, hidden: hBuf, plane: pBuf,
                                   d: UInt32(d), hc: UInt32(hc))
            cb.commit(); cb.waitUntilCompleted()

            var expected = [Float](repeating: 0, count: hc * d)
            for c in 0..<hc {
                for i in 0..<d { expected[c * d + i] = Float(hidden[i]) }
            }
            let actual = Fp16Buffer.read(pBuf, count: hc * d)
            let relErr = RelError.compute(actual: actual, reference: expected)
            #expect(relErr < Tolerance.identity,
                    "hc=\(hc) d=\(d): pure copy must be exact, relErr=\(relErr)")
        }
    }

    // MARK: - hc_grouped_rms

    @Test("grouped RMS matches HyperConnectionRef", arguments: [
        (2, 4, UInt64(0x511)),       // tiny
        (4, 256, UInt64(0x512)),     // toy
        (4, 2560, UInt64(0x513)),    // real stream width
    ])
    func groupedRms_matchesRef(hc: Int, d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("hc-grouped-rms")
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        let v = Self.seedVectors(&rng, hc: hc, d: d, device: ctx.device)
        guard let xBuf = v.xBuf, let gBuf = v.gammaBuf, let oBuf = v.outBuf else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGroupedRMS(commandBuffer: cb, x: xBuf, gamma: gBuf,
                                out: oBuf, d: UInt32(d), hc: UInt32(hc),
                                eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = HyperConnectionRef.groupedRMS(x: v.x, gamma: v.gammaRef,
                                                streamCount: hc, eps: Self.eps)
        let actual = Fp16Buffer.read(oBuf, count: hc * d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "hc=\(hc) d=\(d): relErr=\(relErr) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    @Test("grouped RMS normalises each stream independently (no whole-plane leak)")
    func groupedRms_perStreamIsolation() throws {
        // Stream 0 = [1, 1] and stream 1 = [100, 100] with unit gamma: a
        // whole-plane reduction would scale both to ~1/√5000; per-stream
        // reduction leaves both streams at ~1.
        let hc = 2, d = 2
        let xH: [Float16] = [1, 1, 100, 100].map { Float16($0) }
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xH),
              let outBuf = Fp16Buffer.make(ctx.device, count: hc * d),
              let gBuf = Self.makeBF16Buffer(ctx.device, [Float](repeating: 1, count: 4)) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGroupedRMS(commandBuffer: cb, x: xBuf, gamma: gBuf,
                                out: outBuf, d: UInt32(d), hc: UInt32(hc),
                                eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(outBuf, count: 4)
        let inv0 = 1.0 / (1.0 + Self.eps).squareRoot()  // ~1 - 5e-7
        #expect(abs(actual[0] - Float(inv0)) < 1e-4, "stream 0 first element: \(actual[0])")
        #expect(abs(actual[1] - Float(inv0)) < 1e-4, "stream 0 second element: \(actual[1])")
        #expect(abs(actual[2] - 1.0) < 1e-4, "stream [100,100] must self-normalise: \(actual[2])")
        #expect(abs(actual[3] - 1.0) < 1e-4, "stream [100,100] must self-normalise: \(actual[3])")
    }

    @Test("grouped RMS is in-place safe (x aliases out)")
    func groupedRms_inPlaceAlias() throws {
        let hc = 2, d = 256, seed = UInt64(0x514)
        let ctx = try MetalContext()
        var rng = SeedTree(seed).key("hc-grouped-rms-inplace")
        let v = Self.seedVectors(&rng, hc: hc, d: d, device: ctx.device)
        let kernel = try HyperConnection(context: ctx)
        guard let xBuf = v.xBuf, let gBuf = v.gammaBuf else {
            Issue.record("alloc failed"); return
        }

        // Same MTLBuffer for x and out (same offset): the reduction completes
        // before the scale pass writes, and each element is read and written
        // by the same thread.
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGroupedRMS(commandBuffer: cb, x: xBuf, gamma: gBuf,
                                out: xBuf, d: UInt32(d), hc: UInt32(hc),
                                eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = HyperConnectionRef.groupedRMS(x: v.x, gamma: v.gammaRef,
                                                streamCount: hc, eps: Self.eps)
        let actual = Fp16Buffer.read(xBuf, count: hc * d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction, "in-place relErr=\(relErr)")
    }

    // MARK: - hc_silu_scale

    @Test("silu scale matches silu(z · 1/hc)", arguments: [
        (1, UInt64(0x521)),      // tiny lowrank
        (64, UInt64(0x522)),     // toy lowrank
        (320, UInt64(0x523)),    // real lowrank
    ])
    func siluScale_matchesNaive(n: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("hc-silu-scale")
        let hc = 4
        let zH = (0..<n).map { _ in Float16(rng.uniform(-8.0, 8.0)) }
        let zRef = zH.map { Float($0) }
        let invHc: Float = 1.0 / Float(hc)
        let expected = zRef.map { Self.silu($0 * invHc) }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let zBuf = Fp16Buffer.make(ctx.device, halves: zH),
              let oBuf = Fp16Buffer.make(ctx.device, count: n) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSiluScale(commandBuffer: cb, z: zBuf, out: oBuf,
                               n: UInt32(n), invHc: invHc)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(oBuf, count: n)
        let relErr = RelError.compute(actual: actual, reference: expected)
        #expect(relErr < Tolerance.fp16Reduction, "n=\(n): relErr=\(relErr)")
    }

    @Test("silu(0) is exactly 0 (÷hc before silu is a no-op at z = 0)")
    func siluScale_zero() throws {
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let zBuf = Fp16Buffer.make(ctx.device, halves: [Float16](repeating: 0, count: 8)),
              let oBuf = Fp16Buffer.make(ctx.device, count: 8) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSiluScale(commandBuffer: cb, z: zBuf, out: oBuf,
                               n: 8, invHc: 0.25)
        cb.commit(); cb.waitUntilCompleted()
        let actual = Fp16Buffer.read(oBuf, count: 8)
        for v in actual {
            #expect(v == 0, "silu(0) = 0, got \(v)")
        }
    }

    // MARK: - hc_gate_mul

    @Test("gate mul matches xn · sigmoid(z)", arguments: [
        (8, UInt64(0x531)),          // tiny plane
        (1024, UInt64(0x532)),       // toy plane (4 × 256)
        (10240, UInt64(0x533)),      // real plane (4 × 2560)
    ])
    func gateMul_matchesNaive(n: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("hc-gate-mul")
        let xnH = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let zH = (0..<n).map { _ in Float16(rng.uniform(-6.0, 6.0)) }
        let xnRef = xnH.map { Float($0) }
        let zRef = zH.map { Float($0) }
        let expected = zip(xnRef, zRef).map { $0 * Self.sigmoid($1) }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xnH),
              let zBuf = Fp16Buffer.make(ctx.device, halves: zH),
              let oBuf = Fp16Buffer.make(ctx.device, count: n) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGateMul(commandBuffer: cb, xn: xBuf, z: zBuf, out: oBuf,
                             n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(oBuf, count: n)
        let relErr = RelError.compute(actual: actual, reference: expected)
        #expect(relErr < Tolerance.fp16Reduction, "n=\(n): relErr=\(relErr)")
    }

    // MARK: - hc_stream_mean

    @Test("stream mean matches the per-element stream average", arguments: [
        (2, 4, UInt64(0x541)),      // tiny
        (4, 256, UInt64(0x542)),    // toy
        (4, 2560, UInt64(0x543)),   // real
    ])
    func streamMean_matchesNaive(hc: Int, d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("hc-stream-mean")
        let gH = (0..<(hc * d)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let gRef = gH.map { Float($0) }
        let invHc: Float = 1.0 / Float(hc)
        var expected = [Float](repeating: 0, count: d)
        for i in 0..<d {
            var acc: Float = 0
            for c in 0..<hc { acc += gRef[c * d + i] }
            expected[i] = acc * invHc
        }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let gBuf = Fp16Buffer.make(ctx.device, halves: gH),
              let oBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeStreamMean(commandBuffer: cb, gated: gBuf, out: oBuf,
                                d: UInt32(d), hc: UInt32(hc), invHc: invHc)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(oBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: expected)
        #expect(relErr < Tolerance.fp16Reduction,
                "hc=\(hc) d=\(d): relErr=\(relErr) maxAbs=\(RelError.maxAbsDiff(actual, expected))")
    }

    @Test("stream mean hand case: [1,2,3,4] over 2 streams → [2,3]")
    func streamMean_handCase() throws {
        let hc = 2, d = 2
        let gH: [Float16] = [1, 2, 3, 4].map { Float16($0) }
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let gBuf = Fp16Buffer.make(ctx.device, halves: gH),
              let oBuf = Fp16Buffer.make(ctx.device, count: d) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeStreamMean(commandBuffer: cb, gated: gBuf, out: oBuf,
                                d: UInt32(d), hc: UInt32(hc), invHc: 0.5)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(oBuf, count: d)
        #expect(abs(actual[0] - 2) < 1e-6, "mean of stream column 0 = (1+3)/2, got \(actual[0])")
        #expect(abs(actual[1] - 3) < 1e-6, "mean of stream column 1 = (2+4)/2, got \(actual[1])")
    }

    // MARK: - hc_combine

    @Test("combine matches HyperConnectionRef scatter-add", arguments: [
        (2, 4, UInt64(0x551)),      // tiny
        (4, 256, UInt64(0x552)),    // toy
        (4, 2560, UInt64(0x553)),   // real
    ])
    func combine_matchesRef(hc: Int, d: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("hc-combine")
        let n = hc * d
        let pH = (0..<n).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let bH = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let iH = (0..<hc).map { _ in Float16(rng.uniform(-4.0, 4.0)) }
        let pRef = pH.map { Float($0) }
        let bRef = bH.map { Float($0) }
        let iRef = iH.map { Float($0) }
        let invHc: Float = 1.0 / Float(hc)

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let pBuf = Fp16Buffer.make(ctx.device, halves: pH),
              let bBuf = Fp16Buffer.make(ctx.device, halves: bH),
              let iBuf = Fp16Buffer.make(ctx.device, halves: iH) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCombine(commandBuffer: cb, plane: pBuf, blockOut: bBuf,
                             inject: iBuf, d: UInt32(d), hc: UInt32(hc),
                             invHc: invHc)
        cb.commit(); cb.waitUntilCompleted()

        let ref = HyperConnectionRef.combine(plane: pRef, blockOut: bRef,
                                             inject: iRef, streamCount: hc)
        let actual = Fp16Buffer.read(pBuf, count: n)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "hc=\(hc) d=\(d): relErr=\(relErr) maxAbs=\(RelError.maxAbsDiff(actual, ref))")
    }

    @Test("zero inject is a plain residual add through the kernel")
    func combine_zeroInject() throws {
        let hc = 4, d = 256, seed = UInt64(0x554)
        var rng = SeedTree(seed).key("hc-combine-zero")
        let pH = (0..<(hc * d)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let bH = (0..<d).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        let zero = [Float16](repeating: 0, count: hc)

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let pBuf = Fp16Buffer.make(ctx.device, halves: pH),
              let bBuf = Fp16Buffer.make(ctx.device, halves: bH),
              let iBuf = Fp16Buffer.make(ctx.device, halves: zero) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCombine(commandBuffer: cb, plane: pBuf, blockOut: bBuf,
                             inject: iBuf, d: UInt32(d), hc: UInt32(hc),
                             invHc: 0.25)
        cb.commit(); cb.waitUntilCompleted()

        // 2·sigmoid(0) = 1 exactly in fp32: the kernel adds block to each
        // stream with a unit weight and one fp16 store, so the result must
        // equal the single-rounding fp32 sum.
        var expected = [Float](repeating: 0, count: hc * d)
        let pRef = pH.map { Float($0) }
        let bRef = bH.map { Float($0) }
        for c in 0..<hc {
            for i in 0..<d {
                expected[c * d + i] = Float(Float16(pRef[c * d + i] + bRef[i]))
            }
        }
        let actual = Fp16Buffer.read(pBuf, count: hc * d)
        let relErr = RelError.compute(actual: actual, reference: expected)
        #expect(relErr < Tolerance.identity, "zero inject relErr=\(relErr)")
    }
}

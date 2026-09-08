import Testing
import Foundation
import Metal
@testable import FlashQwen
import FlashQwenValidationSupport

/// Compares the Metal `silu_mul_fp16` kernel against the fp32
/// `SiluMulRef`. Inputs are generated in fp32, rounded to FP16, and the
/// rounded values feed the reference — the same discipline as the other
/// primitive tests. The kernel is dispatched raw (by PSO name) here; the
/// shared-expert runtime path through it is covered in
/// `SharedExpertInt4Tests`.
@Suite struct SiluMulTests {

    private static func runAndCompare(count: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("silu-mul-\(count)")
        let gateFp16 = (0..<count).map { _ in Float16(rng.uniform(-4.0, 4.0)) }
        let upFp16   = (0..<count).map { _ in Float16(rng.uniform(-4.0, 4.0)) }

        let ctx = try MetalContext()
        let pso = try ctx.pipeline("silu_mul_fp16")
        guard let gateBuf = Fp16Buffer.make(ctx.device, halves: gateFp16),
              let upBuf   = Fp16Buffer.make(ctx.device, halves: upFp16),
              let outBuf  = Fp16Buffer.make(ctx.device, count: count) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        guard let enc = cb.makeComputeCommandEncoder() else {
            Issue.record("no encoder"); return
        }
        enc.setComputePipelineState(pso)
        enc.setBuffer(gateBuf, offset: 0, index: 0)
        enc.setBuffer(upBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        var c = UInt32(count)
        enc.setBytes(&c, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(pso.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()

        let ref = SiluMulRef.apply(gate: gateFp16.map { Float($0) },
                                   up: upFp16.map { Float($0) })
        let actual = Fp16Buffer.read(outBuf, count: count)
        let relErr = RelError.compute(actual: actual, reference: ref)
        let maxAbs = RelError.maxAbsDiff(actual, ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "silu_mul count=\(count): relErr=\(relErr) maxAbsDiff=\(maxAbs)")
    }

    @Test func siluMul_count512() throws {
        try Self.runAndCompare(count: 512, seed: 0x51)
    }
    @Test func siluMul_count2112() throws {
        try Self.runAndCompare(count: 2112, seed: 0x52)
    }
    /// Non-multiple-of-256 count exercises the tail guard (`tid >= count`).
    @Test func siluMul_countOddTail() throws {
        try Self.runAndCompare(count: 1000, seed: 0x53)
    }
}

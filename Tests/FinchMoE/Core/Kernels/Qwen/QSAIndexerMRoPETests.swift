import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// mRoPE variants of the QSA indexer. The text path must be unchanged when the
/// three axes agree (identity to the scalar kernels), and image tokens must
/// rotate each pair on the axis the interleaved section selects.
@Suite struct QSAIndexerMRoPETests {

    private static let eps: Float = 1e-6
    private static let theta: Float = 1.0e7
    private static let sections = [11, 11, 10]

    private static func bf16(_ rng: inout SplitMix64, _ n: Int,
                             _ lo: Float, _ hi: Float) -> [Float] {
        (0..<n).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(lo, hi)))
        }
    }

    private static func bf16Buffer(_ device: MTLDevice,
                                   _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { Quantization.bf16Bits($0) }
        guard let buf = device.makeBuffer(length: bits.count * 2,
                                          options: .storageModeShared) else { return nil }
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for i in 0..<bits.count { p[i] = bits[i] }
        return buf
    }

    private static func fp16(_ rng: inout SplitMix64, _ n: Int,
                             _ lo: Float, _ hi: Float) -> [Float16] {
        (0..<n).map { _ in Float16(rng.uniform(lo, hi)) }
    }

    /// HF half-split rotation with the interleaved T/H/W axis per pair.
    private static func rotate(_ vector: [Float],
                               t: Int, h: Int, w: Int,
                               nRot: Int,
                               sections: [Int],
                               theta: Float) -> [Float] {
        var out = vector
        let half = nRot / 2
        let logTheta = logf(theta) / Float(nRot)
        for pair in 0..<half {
            let axis: Int
            if pair % 3 == 1 && pair < sections[1] * 3 {
                axis = 1
            } else if pair % 3 == 2 && pair < sections[2] * 3 {
                axis = 2
            } else {
                axis = 0
            }
            let position = [t, h, w][axis]
            let freq = expf(-Float(2 * pair) * logTheta)
            let angle = Float(position) * freq
            let c = cosf(angle)
            let s = sinf(angle)
            let a = vector[pair]
            let b = vector[pair + half]
            out[pair] = a * c - b * s
            out[pair + half] = a * s + b * c
        }
        return out
    }

    private static func normedRope(_ vector: [Float],
                                   gamma: [Float],
                                   t: Int, h: Int, w: Int,
                                   nRot: Int) -> [Float] {
        let normed = QSAIndexerRef.rms(vector, gamma: gamma, eps: eps)
        return rotate(normed, t: t, h: h, w: w,
                      nRot: nRot, sections: sections, theta: theta)
    }

    // MARK: - QK post

    @Test func qkPostTextTripleIsIdentityToTheScalarKernel() throws {
        let nHeads = 4, idxDim = 128, nRot = 64, pos = 37
        var rng = SeedTree(0xA01).key("qk-post-text")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qk = Self.fp16(&rng, (nHeads + 1) * idxDim, -1, 1)
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qk),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let scalarQ = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let mropeQ = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let scalarK = Fp16Buffer.make(ctx.device, count: (pos + 1) * idxDim),
              let mropeK = Fp16Buffer.make(ctx.device, count: (pos + 1) * idxDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeQKPost(commandBuffer: cb, qk: qkBuf, qGamma: gBuf,
                            qOut: scalarQ, kRaw: scalarK,
                            pos: UInt32(pos), nHeads: UInt32(nHeads),
                            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                            theta: Self.theta, eps: Self.eps)
        try kernel.encodeQKPostMRoPE(
            commandBuffer: cb, qk: qkBuf, qGamma: gBuf, qOut: mropeQ,
            kRaw: mropeK, storePos: UInt32(pos),
            positionT: UInt32(pos), positionH: UInt32(pos),
            positionW: UInt32(pos), sections: Self.sections,
            nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
            nRot: UInt32(nRot), theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let scalar = Fp16Buffer.read(scalarQ, count: nHeads * idxDim)
        let mrope = Fp16Buffer.read(mropeQ, count: nHeads * idxDim)
        #expect(scalar == mrope)
        #expect(Fp16Buffer.read(scalarK, count: (pos + 1) * idxDim)
                == Fp16Buffer.read(mropeK, count: (pos + 1) * idxDim))
    }

    @Test func qkPostRotatesAnImageTriplePerAxisAndStoresAtTheKVPosition() throws {
        let nHeads = 4, idxDim = 128, nRot = 64
        let storePos = 37
        let t = 5, h = 7, w = 9
        var rng = SeedTree(0xA02).key("qk-post-image")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qk = Self.fp16(&rng, (nHeads + 1) * idxDim, -1, 1)
        let qkRef = qk.map { Float($0) }
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qk),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let qOut = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let kRaw = Fp16Buffer.make(ctx.device, count: (storePos + 1) * idxDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeQKPostMRoPE(
            commandBuffer: cb, qk: qkBuf, qGamma: gBuf, qOut: qOut,
            kRaw: kRaw, storePos: UInt32(storePos),
            positionT: UInt32(t), positionH: UInt32(h), positionW: UInt32(w),
            sections: Self.sections, nHeads: UInt32(nHeads),
            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        var reference: [Float] = []
        for head in 0..<nHeads {
            let row = Array(qkRef[(head * idxDim)..<((head + 1) * idxDim)])
            reference += Self.normedRope(row, gamma: gamma, t: t, h: h, w: w,
                                         nRot: nRot)
        }
        let actual = Fp16Buffer.read(qOut, count: nHeads * idxDim)
        #expect(RelError.compute(actual: actual, reference: reference)
                < Tolerance.fp16Reduction)

        // The key head is raw and lands at the KV store position, never at the
        // rope position.
        let keys = Fp16Buffer.read(kRaw, count: (storePos + 1) * idxDim)
        let keyRow = Array(qkRef[(nHeads * idxDim)..<((nHeads + 1) * idxDim)])
        let stored = Array(keys[(storePos * idxDim)..<((storePos + 1) * idxDim)])
        #expect(stored == keyRow.map { Float(Float16($0)) })
    }

    // MARK: - Block pool

    private static func poolReference(rawKeys: [Float16],
                                      gamma: [Float],
                                      block: Int,
                                      r: Int,
                                      idxDim: Int,
                                      t: Int, h: Int, w: Int,
                                      nRot: Int) -> [Float] {
        var mean = [Float](repeating: 0, count: idxDim)
        for d in 0..<idxDim {
            var sum: Float = 0
            for j in 0..<r {
                sum += Float(rawKeys[(block * r + j) * idxDim + d])
            }
            // The kernel rounds the mean to the cache dtype before the norm.
            mean[d] = Float(Float16(sum * (1.0 / Float(r))))
        }
        return normedRope(mean, gamma: gamma, t: t, h: h, w: w, nRot: nRot)
    }

    @Test func poolTextTimelineIsIdentityToTheScalarKernel() throws {
        let idxDim = 128, nRot = 64, r = 4, blocks = 4
        var rng = SeedTree(0xA03).key("pool-text")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let raw = Self.fp16(&rng, blocks * r * idxDim, -1, 1)
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        guard let rawBuf = Fp16Buffer.make(ctx.device, halves: raw),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let scalarPool = Fp16Buffer.make(ctx.device, count: blocks * idxDim),
              let mropePool = Fp16Buffer.make(ctx.device, count: blocks * idxDim) else {
            Issue.record("alloc failed"); return
        }
        var packed: [UInt32] = []
        for cell in 0..<(blocks * r) {
            packed += [UInt32(cell), UInt32(cell), UInt32(cell)]
        }
        let positionBuffer = packed.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                  options: .storageModeShared)
        }
        let positions = try #require(positionBuffer)

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockPoolNormRope(
            commandBuffer: cb, kRaw: rawBuf, kGamma: gBuf, pooled: scalarPool,
            firstBlock: 0, blockCount: UInt32(blocks), r: UInt32(r),
            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
            theta: Self.theta, eps: Self.eps)
        try kernel.encodeBlockPoolNormRopeMRoPE(
            commandBuffer: cb, kRaw: rawBuf, kGamma: gBuf, pooled: mropePool,
            positions: positions, promptLength: blocks * r, decodeOffset: 0,
            sections: Self.sections, firstBlock: 0, blockCount: UInt32(blocks),
            r: UInt32(r), idxDim: UInt32(idxDim), nRot: UInt32(nRot),
            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        #expect(Fp16Buffer.read(scalarPool, count: blocks * idxDim)
                == Fp16Buffer.read(mropePool, count: blocks * idxDim))
    }

    @Test func poolRotatesAtTheImageTripleOfTheBlockStart() throws {
        let idxDim = 128, nRot = 64, r = 4, blocks = 4
        let block = 2
        var rng = SeedTree(0xA04).key("pool-image")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let raw = Self.fp16(&rng, blocks * r * idxDim, -1, 1)
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        guard let rawBuf = Fp16Buffer.make(ctx.device, halves: raw),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let pooled = Fp16Buffer.make(ctx.device, count: blocks * idxDim) else {
            Issue.record("alloc failed"); return
        }
        // Every cell gets a distinct triple; block 2 starts at cell 8.
        var packed: [UInt32] = []
        for cell in 0..<(blocks * r) {
            packed += [UInt32(cell), UInt32(cell + 100), UInt32(cell + 200)]
        }
        let positionBuffer = packed.withUnsafeBytes {
            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count,
                                  options: .storageModeShared)
        }
        let positions = try #require(positionBuffer)

        let cb = ctx.queue.makeCommandBuffer()!
        try kernel.encodeBlockPoolNormRopeMRoPE(
            commandBuffer: cb, kRaw: rawBuf, kGamma: gBuf, pooled: pooled,
            positions: positions, promptLength: blocks * r, decodeOffset: 0,
            sections: Self.sections, firstBlock: UInt32(block), blockCount: 1,
            r: UInt32(r), idxDim: UInt32(idxDim), nRot: UInt32(nRot),
            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let reference = Self.poolReference(
            rawKeys: raw, gamma: gamma, block: block, r: r, idxDim: idxDim,
            t: block * r, h: block * r + 100, w: block * r + 200, nRot: nRot)
        let actual = Fp16Buffer.read(
            pooled, count: blocks * idxDim)[(block * idxDim)..<((block + 1) * idxDim)]
        #expect(RelError.compute(actual: Array(actual), reference: reference)
                < Tolerance.fp16Reduction)
    }
}

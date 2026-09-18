import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares the Qwen 3.8 QSA indexer kernels (`Metal/Qwen/qsa_indexer.metal`)
/// against the fp32 `QSAIndexerRef` committed with M3.2a. Inputs are
/// fp16-rounded (activations, raw keys, queries, pooled keys) and the indexer
/// norm gammas are BF16-rounded, and the rounded values feed the reference —
/// the discipline of the other kernel suites.
///
/// Geometries: the real indexer shape (idxDim 128, 4 query heads, r 4,
/// nRot 64 — the model's 256·0.25 rope width, not a quarter of the indexer's
/// own 128-dim head) plus a tiny one where a hand check is possible. Cases cover the
/// dense tail (the query ends a complete block, no force-visible block), the
/// sparse tail (the query's own block is incomplete → bias +1e9), the -INF
/// partial-pool arm llama reserves for a cache hole, and the two ordering
/// traps a plausible reimplementation gets wrong: per-head relu, and rope at
/// the block's FIRST cell rather than its last.
@Suite struct QSAIndexerTests {

    private static let eps: Float = 1e-6
    private static let theta: Float = 1.0e7

    private static func bf16(_ rng: inout SplitMix64, _ n: Int,
                             _ lo: Float, _ hi: Float) -> [Float] {
        (0..<n).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(lo, hi)))
        }
    }

    private static func bf16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
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

    /// The score buffer is FP32 (llama keeps the expanded scores in f32).
    private static func readF32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { p[$0] }
    }

    private static func f32Buffer(_ device: MTLDevice, count: Int) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)
    }

    // MARK: - idx_qk_post (query norm + rope, raw key store)

    @Test("indexer q head: RMS + partial rope matches the reference at `pos`",
          arguments: [
            (4, 128, 64, 37, UInt64(0x901)),   // real geometry, mid-timeline
            (4, 128, 64, 0, UInt64(0x902)),    // position 0 (rope ≈ identity)
            (2, 32, 16, 5, UInt64(0x903)),     // tiny, partial rotary
            (4, 128, 64, 4097, UInt64(0x904)), // past the dense window
            (4, 128, 128, 9, UInt64(0x905)),   // fully rotated: no carry-through
          ])
    func qPost_matchesRef(nHeads: Int, idxDim: Int, nRot: Int,
                          pos: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("idx-q-post")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qkH = Self.fp16(&rng, (nHeads + 1) * idxDim, -1.0, 1.0)
        let qkRef = qkH.map { Float($0) }
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qkH),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let qOutBuf = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let kBuf = Fp16Buffer.make(ctx.device, count: (pos + 1) * idxDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeQKPost(commandBuffer: cb, qk: qkBuf, qGamma: gBuf,
                            qOut: qOutBuf, kRaw: kBuf,
                            pos: UInt32(pos), nHeads: UInt32(nHeads),
                            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        // Every query head is normed with the same [idxDim] gamma and rotated
        // at the token position. The key head (row nHeads) is not part of this
        // path — it goes to the timeline raw.
        let qRows = Array(qkRef[0..<(nHeads * idxDim)])
        let ref = QSAIndexerRef.normRope(
            qRows, perVector: nHeads, gamma: gamma,
            pos: [Int](repeating: pos, count: nHeads), dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)

        let actual = Fp16Buffer.read(qOutBuf, count: nHeads * idxDim)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "nHeads=\(nHeads) idxDim=\(idxDim) pos=\(pos): relErr=\(relErr)")
    }

    @Test("indexer key head reaches the timeline raw — no norm, no rotation")
    func qPost_keyHeadIsRaw() throws {
        let nHeads = 4, idxDim = 128, nRot = 64, pos = 11
        var rng = SeedTree(0x911).key("idx-k-raw")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qkH = Self.fp16(&rng, (nHeads + 1) * idxDim, -1.0, 1.0)
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qkH),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let qOutBuf = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let kBuf = Fp16Buffer.make(ctx.device, count: (pos + 1) * idxDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeQKPost(commandBuffer: cb, qk: qkBuf, qGamma: gBuf,
                            qOut: qOutBuf, kRaw: kBuf,
                            pos: UInt32(pos), nHeads: UInt32(nHeads),
                            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        // The pooled keys carry the norm and the rotation, and pooling
        // precedes both, so the cached key must be the verbatim projection
        // (qwen4exp.cpp:533-538). A normalised or rotated store would show up
        // here as a mismatch — and would silently corrupt every later block.
        let stored = Fp16Buffer.read(kBuf, count: (pos + 1) * idxDim)
        let expected = qkH[(nHeads * idxDim)...].map { Float($0) }
        for i in 0..<idxDim {
            #expect(stored[pos * idxDim + i] == expected[i],
                    "raw key [\(i)] = \(stored[pos * idxDim + i]), expected \(expected[i])")
        }
        // Cells the step did not touch stay zero — the store is a slot write,
        // not a shift.
        for i in 0..<(pos * idxDim) {
            #expect(stored[i] == 0, "cell \(i / idxDim) was written: \(stored[i])")
        }
    }

    /// The chunk form of the same store: a prefill chunk hands the kernel a
    /// chunk-local projection buffer (`qkOffset` by row) and a timeline the
    /// chunk only partly fills (`kRaw` is the layer's persistent buffer, and
    /// `pos` is the token's *absolute* position). The store must still land at
    /// slot `pos` — the pool kernel indexes cells absolutely, and a decode
    /// step later reuses the same slots — so a chunk's posts and a chunk's
    /// pools have to agree with the positions they name, not with the chunk's
    /// own origin.
    @Test("chunk-form posts store each key at its absolute position, and the chunk's pools read them back")
    func chunkFormPostsUseAbsolutePositions() throws {
        let nHeads = 4, idxDim = 128, nRot = 64, r = 4
        let base = 8          // the chunk starts inside block 2, so pos != row
        let T = 8             // two complete blocks (2 and 3)
        var rng = SeedTree(0x9A1).key("idx-chunk-post")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        // One row per token, chunk-local — the shape `qwen38IdxQKProj` has.
        let projDim = (nHeads + 1) * idxDim
        let qk = Self.fp16(&rng, T * projDim, -1.0, 1.0)
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        let nCells = base + T

        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qk),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let qOutBuf = Fp16Buffer.make(ctx.device, count: T * nHeads * idxDim),
              let kBuf = Fp16Buffer.make(ctx.device, count: nCells * idxDim),
              let pBuf = Fp16Buffer.make(ctx.device, count: (nCells / r) * idxDim) else {
            Issue.record("alloc failed"); return
        }

        let cb = ctx.queue.makeCommandBuffer()!
        for row in 0..<T {
            let pos = base + row
            kernel.encodeQKPost(commandBuffer: cb,
                                qk: qkBuf, qkOffset: row * projDim * MemoryLayout<Float16>.stride,
                                qGamma: gBuf,
                                qOut: qOutBuf,
                                qOutOffset: row * nHeads * idxDim * MemoryLayout<Float16>.stride,
                                kRaw: kBuf,
                                pos: UInt32(pos), nHeads: UInt32(nHeads),
                                idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                                theta: Self.theta, eps: Self.eps)
        }
        kernel.encodeBlockPoolNormRope(commandBuffer: cb, kRaw: kBuf, kGamma: gBuf,
                                       pooled: pBuf, firstBlock: UInt32(base / r),
                                       blockCount: UInt32(T / r), r: UInt32(r),
                                       idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                                       theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        // The timeline the reference pools from: each row's key head verbatim
        // at its own absolute slot, and nothing anywhere else.
        var rawRef = [Float](repeating: 0, count: nCells * idxDim)
        for row in 0..<T {
            let pos = base + row
            for i in 0..<idxDim {
                rawRef[pos * idxDim + i] = Float(qk[row * projDim + nHeads * idxDim + i])
            }
        }
        let stored = Fp16Buffer.read(kBuf, count: nCells * idxDim)
        let expected = rawRef.map { Float(Float16($0)) }
        for i in 0..<(nCells * idxDim) where stored[i] != expected[i] {
            Issue.record("timeline slot \(i / idxDim) [\(i % idxDim)] = \(stored[i]) but the chunk's post for that position stored nothing there, so the posts are not using absolute positions")
            break
        }

        // …and the chunk's own pools must be the mean of those cells.
        var pooledRef: [Float] = []
        for b in (base / r)..<(base / r + T / r) {
            pooledRef += QSAIndexerRef.meanPool(Array((b * r)..<((b + 1) * r)),
                                                raw: rawRef, idxDim: idxDim)
        }
        let ref = QSAIndexerRef.normRope(
            pooledRef, perVector: T / r, gamma: gamma,
            pos: ((base / r)..<(base / r + T / r)).map { $0 * r },
            dim: idxDim, eps: Self.eps, nRot: nRot, theta: Self.theta)
        let pooled = Fp16Buffer.read(pBuf, count: (nCells / r) * idxDim)
        let window = Array(pooled[((base / r) * idxDim)..<(((base / r) + T / r) * idxDim)])
        let relErr = RelError.compute(actual: window, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction, "chunk-form pool relErr=\(relErr)")
    }

    @Test("indexer query rope tracks the token position, not a fixed one")
    func qPost_ropeTracksPosition() throws {
        let nHeads = 4, idxDim = 128, nRot = 64, pos = 13
        var rng = SeedTree(0x921).key("idx-q-rope-pos")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qkH = Self.fp16(&rng, (nHeads + 1) * idxDim, -1.0, 1.0)
        let qkRef = qkH.map { Float($0) }
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        guard let qkBuf = Fp16Buffer.make(ctx.device, halves: qkH),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let qOutBuf = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let kBuf = Fp16Buffer.make(ctx.device, count: (pos + 1) * idxDim) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeQKPost(commandBuffer: cb, qk: qkBuf, qGamma: gBuf,
                            qOut: qOutBuf, kRaw: kBuf,
                            pos: UInt32(pos), nHeads: UInt32(nHeads),
                            idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                            theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(qOutBuf, count: nHeads * idxDim)
        let qRows = Array(qkRef[0..<(nHeads * idxDim)])
        let atPos = QSAIndexerRef.normRope(
            qRows, perVector: nHeads, gamma: gamma,
            pos: [Int](repeating: pos, count: nHeads), dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)
        let atOther = QSAIndexerRef.normRope(
            qRows, perVector: nHeads, gamma: gamma,
            pos: [Int](repeating: pos + 1, count: nHeads), dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)

        let right = RelError.compute(actual: actual, reference: atPos)
        let wrong = RelError.compute(actual: actual, reference: atOther)
        #expect(right < Tolerance.fp16Reduction, "relErr at pos=\(pos): \(right)")
        #expect(wrong > 1e-2,
                "trap: the query must rotate at its own position (relErr=\(wrong))")
    }

    // MARK: - idx_block_pool_norm_rope

    @Test("block pooling: mean, RMS and rope match the reference",
          arguments: [
            (4, 128, 64, 6, 0, UInt64(0x931)),   // real geometry, whole timeline
            (4, 128, 64, 3, 2, UInt64(0x932)),   // finalise a suffix from block 2
            (2, 32, 16, 2, 0, UInt64(0x933)),    // tiny
          ])
    func blockPool_matchesRef(r: Int, idxDim: Int, nRot: Int,
                              blockCount: Int, first: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("idx-block-pool")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let nCells = (first + blockCount) * r
        let rawH = Self.fp16(&rng, nCells * idxDim, -1.0, 1.0)
        let rawRef = rawH.map { Float($0) }
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        // Sentinel payload, so a write outside [first, first + blockCount)
        // shows up instead of passing silently.
        let sentinel: Float16 = 12345
        var pooledH = [Float16](repeating: sentinel, count: nCells * idxDim)
        for i in 0..<(blockCount * idxDim) {
            pooledH[first * idxDim + i] = Float16(rng.uniform(-1.0, 1.0))
        }
        let before = pooledH.map { Float($0) }

        guard let rawBuf = Fp16Buffer.make(ctx.device, halves: rawH),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let pBuf = Fp16Buffer.make(ctx.device, halves: pooledH) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockPoolNormRope(commandBuffer: cb, kRaw: rawBuf, kGamma: gBuf,
                                       pooled: pBuf, firstBlock: UInt32(first),
                                       blockCount: UInt32(blockCount), r: UInt32(r),
                                       idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                                       theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        // Reference: pool each finalised block over its own members (mean over
        // exactly r), norm with the k gamma, rotate at the block's first cell.
        var pooledRef: [Float] = []
        for b in first..<(first + blockCount) {
            pooledRef += QSAIndexerRef.meanPool(Array((b * r)..<((b + 1) * r)),
                                                raw: rawRef, idxDim: idxDim)
        }
        let ref = QSAIndexerRef.normRope(
            pooledRef, perVector: blockCount, gamma: gamma,
            pos: (first..<(first + blockCount)).map { $0 * r }, dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)

        let actual = Fp16Buffer.read(pBuf, count: nCells * idxDim)
        let window = Array(actual[(first * idxDim)..<((first + blockCount) * idxDim)])
        let relErr = RelError.compute(actual: window, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "r=\(r) idxDim=\(idxDim) first=\(first): relErr=\(relErr)")

        for i in 0..<(first * idxDim) {
            #expect(actual[i] == before[i],
                    "slot \(i) outside the finalised range was rewritten")
        }
    }

    @Test("the pooled key rotates at the block's FIRST cell, not its last")
    func blockPool_ropeAtFirstCell() throws {
        let r = 4, idxDim = 128, nRot = 64, nBlocks = 3
        var rng = SeedTree(0x941).key("idx-block-rope-pos")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let rawH = Self.fp16(&rng, nBlocks * r * idxDim, -1.0, 1.0)
        let rawRef = rawH.map { Float($0) }
        let gamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        guard let rawBuf = Fp16Buffer.make(ctx.device, halves: rawH),
              let gBuf = Self.bf16Buffer(ctx.device, gamma),
              let pBuf = Fp16Buffer.make(ctx.device, count: nBlocks * idxDim) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockPoolNormRope(commandBuffer: cb, kRaw: rawBuf, kGamma: gBuf,
                                       pooled: pBuf, firstBlock: 0,
                                       blockCount: UInt32(nBlocks), r: UInt32(r),
                                       idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                                       theta: Self.theta, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let pooledRef = QSAIndexerRef.meanPoolAll(rawRef, n_kv: nBlocks * r,
                                                  r: r, idxDim: idxDim)
        let atFirst = QSAIndexerRef.normRope(
            pooledRef, perVector: nBlocks, gamma: gamma,
            pos: (0..<nBlocks).map { $0 * r }, dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)
        let atLast = QSAIndexerRef.normRope(
            pooledRef, perVector: nBlocks, gamma: gamma,
            pos: (0..<nBlocks).map { $0 * r + r - 1 }, dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)

        let actual = Fp16Buffer.read(pBuf, count: nBlocks * idxDim)
        let right = RelError.compute(actual: actual, reference: atFirst)
        let wrong = RelError.compute(actual: actual, reference: atLast)
        #expect(right < Tolerance.fp16Reduction, "relErr at b·r: \(right)")
        #expect(wrong > 1e-2,
                "trap: all four MRoPE sections carry b·r, so blocks rotate at their first cell (relErr=\(wrong))")
    }

    // MARK: - idx_block_scores

    @Test("block scores: per-head relu-sum matches QSAIndexerRef.blockScores",
          arguments: [
            (4, 128, 4, 9, UInt64(0xA01)),   // real geometry
            (2, 32, 2, 3, UInt64(0xA02)),    // tiny
          ])
    func blockScores_matchesRef(nHeads: Int, idxDim: Int, r: Int,
                                nBlocks: Int, seed: UInt64) throws {
        var rng = SeedTree(seed).key("idx-block-scores")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qH = Self.fp16(&rng, nHeads * idxDim, -1.0, 1.0)
        let poolH = Self.fp16(&rng, nBlocks * idxDim, -1.0, 1.0)
        let qRef = qH.map { Float($0) }
        let poolRef = poolH.map { Float($0) }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qH),
              let pBuf = Fp16Buffer.make(ctx.device, halves: poolH),
              let sBuf = Self.f32Buffer(ctx.device, count: nBlocks) else {
            Issue.record("alloc failed"); return
        }
        // A whole timeline whose last cell is the query: every block is
        // complete, so every bias is 0 and the scores are the raw relu-sums.
        let nKv = nBlocks * r
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockScores(commandBuffer: cb, q: qBuf, pooled: pBuf, scores: sBuf,
                                 nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
                                 r: UInt32(r), nKv: UInt32(nKv), pos: UInt32(nKv - 1))
        cb.commit(); cb.waitUntilCompleted()

        let ref = QSAIndexerRef.blockScores(qRef, poolRef, n_kv: nKv, r: r,
                                            idxDim: idxDim, nIdxHeads: nHeads)
        let actual = Self.readF32(sBuf, count: nBlocks)
        // A bias leak would be unmistakable here: +1e9 or -INF dwarfs every
        // relu-sum, so the relative error check covers the bias arm too.
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.identity,
                "nHeads=\(nHeads) idxDim=\(idxDim): relErr=\(relErr) (a biased block would blow this up)")
    }

    @Test("score rectifies each head dot before summing (relu-of-each, not relu-of-sum)")
    func blockScores_reluIsPerHead() throws {
        // Two heads whose dots cancel: q = [+1…, −1…], pooled = [+1…]. The
        // doc formula gives relu(32) + relu(−32) = 32; rectifying the sum
        // would give relu(0) = 0, and so would dropping the relu entirely.
        let nHeads = 2, idxDim = 32, r = 4, nBlocks = 1
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        var qH = [Float16](repeating: 1, count: nHeads * idxDim)
        for i in idxDim..<(2 * idxDim) { qH[i] = -1 }
        let poolH = [Float16](repeating: 1, count: nBlocks * idxDim)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qH),
              let pBuf = Fp16Buffer.make(ctx.device, halves: poolH),
              let sBuf = Self.f32Buffer(ctx.device, count: nBlocks) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockScores(commandBuffer: cb, q: qBuf, pooled: pBuf, scores: sBuf,
                                 nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
                                 r: UInt32(r), nKv: UInt32(r * nBlocks),
                                 pos: UInt32(r * nBlocks - 1))
        cb.commit(); cb.waitUntilCompleted()

        let score = Self.readF32(sBuf, count: nBlocks)[0]
        #expect(abs(score - Float(idxDim)) < 1e-3,
                "per-head relu gives relu(32) + relu(−32) = 32, got \(score)")
        #expect(abs(score - 0) > 1e-2,
                "trap: rectifying the summed dot gives 0, got \(score)")

        let ref = QSAIndexerRef.blockScores(
            qH.map { Float($0) }, poolH.map { Float($0) },
            n_kv: r * nBlocks, r: r, idxDim: idxDim, nIdxHeads: nHeads)
        #expect(abs(score - ref[0]) < 1e-3, "kernel \(score) vs reference \(ref[0])")
    }

    @Test("the query's own incomplete block is force-visible and its pool never read")
    func blockScores_tailIsForcedVisible() throws {
        let nHeads = 4, idxDim = 128, r = 4, nKv = 42, pos = 41
        let nBlocks = (nKv + r - 1) / r          // 11: block 10 holds cells 40..41
        var rng = SeedTree(0xA11).key("idx-tail-forced")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qH = Self.fp16(&rng, nHeads * idxDim, -1.0, 1.0)
        let poolH = Self.fp16(&rng, nBlocks * idxDim, -1.0, 1.0)

        // Poison the tail block's pooled slot: a kernel that scored it would
        // return NaN, and a forced 1e9 cannot be reached from NaN.
        var poisoned = poolH
        for i in 0..<idxDim { poisoned[10 * idxDim + i] = Float16.nan }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qH),
              let pBuf = Fp16Buffer.make(ctx.device, halves: poisoned),
              let sBuf = Self.f32Buffer(ctx.device, count: nBlocks) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockScores(commandBuffer: cb, q: qBuf, pooled: pBuf, scores: sBuf,
                                 nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
                                 r: UInt32(r), nKv: UInt32(nKv), pos: UInt32(pos))
        cb.commit(); cb.waitUntilCompleted()

        let actual = Self.readF32(sBuf, count: nBlocks)
        #expect(actual[10] == 1e9,
                "the incomplete tail block must be forced visible, got \(actual[10])")

        // Blocks 0..9 are complete: unbiased relu-sums.
        let ref = QSAIndexerRef.blockScores(qH.map { Float($0) }, poolH.map { Float($0) },
                                            n_kv: 10 * r, r: r, idxDim: idxDim,
                                            nIdxHeads: nHeads)
        let relErr = RelError.compute(actual: Array(actual[0..<10]), reference: ref)
        #expect(relErr < Tolerance.identity, "complete-block relErr=\(relErr)")

        // llama stores 1e9 + the tail's own (finite, non-negative) pool score,
        // where the kernel stores 1e9 flat. The selection only ever compares
        // this value against the complete blocks, which cannot reach 1e9 —
        // so the two rank identically and the engine's shortcut is safe.
        let bias = QSAIndexerRef.blockBias(n_kv: nKv, r: r, pos: pos)
        #expect(bias[10] == 1e9, "reference bias disagrees: \(bias[10])")
        #expect(bias[0..<10].allSatisfy { $0 == 0 },
                "complete blocks must carry no bias: \(bias[0..<10])")
        #expect(ref.allSatisfy { $0 < 1e9 },
                "no complete block may reach the forced value")
    }

    @Test("a partial pool before the query is -INF (llama's cache-hole arm)")
    func blockScores_partialPoolIsNegInf() throws {
        // Decode cannot reach this — the timeline is contiguous, so the only
        // incomplete block is the query's own — but the arm is part of the
        // bias contract (llama-memory-hybrid-idx.cpp:421-428) and a chunked
        // path will reach it once prefill lands (M3.4).
        let nHeads = 4, idxDim = 128, r = 4, nKv = 10, pos = 20
        let nBlocks = (nKv + r - 1) / r          // 3: block 2 holds only cells 8..9
        var rng = SeedTree(0xA21).key("idx-partial-pool")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let qH = Self.fp16(&rng, nHeads * idxDim, -1.0, 1.0)
        let poolH = Self.fp16(&rng, nBlocks * idxDim, -1.0, 1.0)

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qH),
              let pBuf = Fp16Buffer.make(ctx.device, halves: poolH),
              let sBuf = Self.f32Buffer(ctx.device, count: nBlocks) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockScores(commandBuffer: cb, q: qBuf, pooled: pBuf, scores: sBuf,
                                 nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
                                 r: UInt32(r), nKv: UInt32(nKv), pos: UInt32(pos))
        cb.commit(); cb.waitUntilCompleted()

        let actual = Self.readF32(sBuf, count: nBlocks)
        let bias = QSAIndexerRef.blockBias(n_kv: nKv, r: r, pos: pos)
        #expect(bias[2] == -.infinity, "reference bias disagrees: \(bias[2])")
        #expect(actual[2] == -.infinity,
                "a partial block before the query is unusable, got \(actual[2])")

        let ref = QSAIndexerRef.blockScores(qH.map { Float($0) }, poolH.map { Float($0) },
                                            n_kv: nKv, r: r, idxDim: idxDim,
                                            nIdxHeads: nHeads)
        let relErr = RelError.compute(actual: Array(actual[0..<2]),
                                      reference: Array(ref[0..<2]))
        #expect(relErr < Tolerance.identity, "complete-block relErr=\(relErr)")
    }

    // MARK: - the whole chain, driven the way a decode step will drive it

    @Test("decode chain: q/k post → block pool → scores matches the reference pipeline",
          arguments: [
            (40, UInt64(0xB01)),   // query ends a complete block: no forced block
            (42, UInt64(0xB02)),   // query inside an incomplete tail block
            (43, UInt64(0xB03)),   // tail block one cell in
          ])
    func decodeChain_matchesRef(nKv: Int, seed: UInt64) throws {
        let nHeads = 4, idxDim = 128, r = 4, nRot = 64
        let nComplete = nKv / r
        let nBlocks = (nKv + r - 1) / r
        let pos = nKv - 1
        var rng = SeedTree(seed).key("idx-decode-chain")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        // One GEMV row per token, as the layer would emit them step by step.
        var qkBufs: [MTLBuffer] = []
        var qkRows: [[Float]] = []
        for _ in 0..<nKv {
            let row = Self.fp16(&rng, (nHeads + 1) * idxDim, -1.0, 1.0)
            guard let b = Fp16Buffer.make(ctx.device, halves: row) else {
                Issue.record("alloc failed"); return
            }
            qkBufs.append(b)
            qkRows.append(row.map { Float($0) })
        }
        let qGamma = Self.bf16(&rng, idxDim, 0.5, 1.5)
        let kGamma = Self.bf16(&rng, idxDim, 0.5, 1.5)

        guard let qGBuf = Self.bf16Buffer(ctx.device, qGamma),
              let kGBuf = Self.bf16Buffer(ctx.device, kGamma),
              let qOutBuf = Fp16Buffer.make(ctx.device, count: nHeads * idxDim),
              let kRawBuf = Fp16Buffer.make(ctx.device, count: nKv * idxDim),
              let pooledBuf = Fp16Buffer.make(ctx.device, count: nBlocks * idxDim),
              let scoresBuf = Self.f32Buffer(ctx.device, count: nBlocks) else {
            Issue.record("alloc failed"); return
        }

        // One decode step per token: post-process q/k, then finalise the block
        // that just filled (decode appends one token, so at most one does).
        // The score step reads the query the last post left in qOut, which is
        // exactly the one at `pos`.
        let cb = ctx.queue.makeCommandBuffer()!
        for t in 0..<nKv {
            kernel.encodeQKPost(commandBuffer: cb, qk: qkBufs[t], qGamma: qGBuf,
                                qOut: qOutBuf, kRaw: kRawBuf,
                                pos: UInt32(t), nHeads: UInt32(nHeads),
                                idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                                theta: Self.theta, eps: Self.eps)
            if (t + 1) % r == 0 {
                kernel.encodeBlockPoolNormRope(
                    commandBuffer: cb, kRaw: kRawBuf, kGamma: kGBuf, pooled: pooledBuf,
                    firstBlock: UInt32(t / r), blockCount: 1, r: UInt32(r),
                    idxDim: UInt32(idxDim), nRot: UInt32(nRot),
                    theta: Self.theta, eps: Self.eps)
            }
        }
        kernel.encodeBlockScores(commandBuffer: cb, q: qOutBuf, pooled: pooledBuf,
                                 scores: scoresBuf, nHeads: UInt32(nHeads),
                                 idxDim: UInt32(idxDim), r: UInt32(r),
                                 nKv: UInt32(nKv), pos: UInt32(pos))
        cb.commit(); cb.waitUntilCompleted()

        // Reference: the same timeline, built by QSAIndexerRef's own steps.
        var rawRef = [Float](repeating: 0, count: nKv * idxDim)
        for t in 0..<nKv {
            let kRow = Array(qkRows[t][(nHeads * idxDim)...])
            for i in 0..<idxDim { rawRef[t * idxDim + i] = kRow[i] }
        }
        let qRef = QSAIndexerRef.normRope(
            Array(qkRows[pos][0..<(nHeads * idxDim)]), perVector: nHeads, gamma: qGamma,
            pos: [Int](repeating: pos, count: nHeads), dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)
        let pooledRef = QSAIndexerRef.normRope(
            QSAIndexerRef.meanPoolAll(Array(rawRef[0..<(nComplete * r * idxDim)]),
                                      n_kv: nComplete * r, r: r, idxDim: idxDim),
            perVector: nComplete, gamma: kGamma,
            pos: (0..<nComplete).map { $0 * r }, dim: idxDim, eps: Self.eps,
            nRot: nRot, theta: Self.theta)
        let refScores = QSAIndexerRef.blockScores(
            qRef, pooledRef, n_kv: nComplete * r, r: r, idxDim: idxDim, nIdxHeads: nHeads)
        let bias = QSAIndexerRef.blockBias(n_kv: nKv, r: r, pos: pos)

        let actual = Self.readF32(scoresBuf, count: nBlocks)
        let produced = Array(actual[0..<nComplete])
        // Complete blocks carry bias 0, so this compares the full chain:
        // raw key → pool → norm+rope → per-head relu dot.
        let relErr = RelError.compute(actual: produced, reference: refScores)
        #expect(relErr < Tolerance.fp16ChainedReduction,
                "nKv=\(nKv): pooled+score relErr=\(relErr)")

        // Everything past the last complete block is the query's own tail
        // (forced) — or nothing at all when the query ends a complete block.
        let tail = Array(actual[nComplete..<nBlocks])
        if nKv > nComplete * r {
            #expect(bias[nComplete] == 1e9,
                    "nKv=\(nKv): the tail block must be the forced one, bias=\(bias[nComplete])")
            #expect(tail == [1e9],
                    "nKv=\(nKv): expected exactly one forced block, got \(tail)")
        } else {
            #expect(tail.isEmpty,
                    "nKv=\(nKv): a complete timeline leaves nothing past the query")
            #expect(bias.allSatisfy { $0 == 0 },
                    "nKv=\(nKv): no block may be forced, bias=\(bias)")
        }
    }
}

import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares `idx_select_cells` against `QSAIndexerRef.topKCells` — llama's
/// `ggml_top_k(expanded, min(n_kv, top_k + r - 1))` and the selection half of
/// `build_attn_qsa`'s mask.
///
/// The kernel gets the *biased* block scores (what `idx_block_scores` emits)
/// and returns the selected cell indices ascending. The reference takes
/// unbiased scores and applies the same bias internally, so a case feeds it
/// the unbiased array and the kernel the biased one — the two must then agree
/// cell for cell, exactly, because the whole selection is integers.
@Suite struct QSAIndexerSelectTests {

    private static func readU32(_ buf: MTLBuffer, count: Int) -> [UInt32] {
        let p = buf.contents().bindMemory(to: UInt32.self, capacity: count)
        return (0..<count).map { p[$0] }
    }

    private static func f32Buffer(_ device: MTLDevice, values: [Float]) -> MTLBuffer? {
        guard let buf = device.makeBuffer(length: values.count * MemoryLayout<Float>.size,
                                          options: .storageModeShared) else { return nil }
        let p = buf.contents().bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
        return buf
    }

    private static func u32Buffer(_ device: MTLDevice, count: Int) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<UInt32>.size, options: .storageModeShared)
    }

    /// Run the kernel over `unbiased` and return the emitted cell list.
    private static func select(
        _ ctx: MetalContext, _ kernel: QSAIndexer,
        unbiased: [Float], pos: Int, nKv: Int, r: Int, budget: Int
    ) throws -> (cells: [UInt32], count: Int) {
        let nBlocks = (nKv + r - 1) / r
        precondition(unbiased.count == nBlocks)
        let bias = QSAIndexerRef.blockBias(n_kv: nKv, r: r, pos: pos)
        let biased = (0..<nBlocks).map { unbiased[$0] + bias[$0] }
        let capacity = QSAIndexer.selectCapacity(nKv: nKv, r: r, budget: budget)

        guard let sBuf = f32Buffer(ctx.device, values: biased),
              let cBuf = u32Buffer(ctx.device, count: capacity),
              let nBuf = u32Buffer(ctx.device, count: 1) else {
            Issue.record("alloc failed")
            return ([], 0)
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeSelectCells(commandBuffer: cb, scores: sBuf, cells: cBuf, count: nBuf,
                                 pos: UInt32(pos), nKv: UInt32(nKv),
                                 r: UInt32(r), budget: UInt32(budget))
        cb.commit(); cb.waitUntilCompleted()

        let n = Int(readU32(nBuf, count: 1)[0])
        return (Array(readU32(cBuf, count: capacity).prefix(n)), n)
    }

    // MARK: - dense fast path

    // Every case here has nvis <= width = min(n_kv, budget + r - 1), which is
    // the only thing that puts the kernel on the dense path — budget alone
    // says nothing. (40, 4, 8, 39) is *not* dense: width is 11, not 40.
    @Test("n_kv within the budget selects every causal cell, in order",
          arguments: [
            (11, 4, 8, 10),    // exactly at the width: budget + r - 1 == nvis
            (40, 4, 37, 39),   // width == n_kv, the widest the budget reaches
            (128, 4, 125, 127),
            (4, 4, 8, 3),      // one whole block
          ])
    func select_denseFastPath(nKv: Int, r: Int, budget: Int, pos: Int) throws {
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let nBlocks = (nKv + r - 1) / r
        var rng = SeedTree(UInt64(nKv &* 31 &+ budget)).key("idx-select-dense")
        let unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(0, 100)) }

        let (cells, count) = try Self.select(ctx, kernel, unbiased: unbiased,
                                             pos: pos, nKv: nKv, r: r, budget: budget)
        #expect(count == pos + 1, "dense path emits the causal prefix, got \(count)")
        #expect(cells == Array(0..<UInt32(pos + 1)).map { UInt32($0) },
                "dense path must emit 0…pos in order")

        // The reference agrees, and never ranks anything in this regime.
        let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)
        #expect(cells.map { Int($0) } == ref, "reference disagrees: \(ref)")
    }

    // MARK: - sparse path

    @Test("sparse selection matches QSAIndexerRef.topKCells exactly",
          arguments: [
            // (nKv, r, budget, pos) — pos = nKv - 1 is the decode shape.
            (12, 4, 8, 11),     // one cell past dense: the narrowest sparse case
            (40, 4, 8, 39),     // 4 blocks over budget, last block aligned (no tail)
            (43, 4, 8, 42),     // tail block 3 cells in
            (41, 4, 8, 40),     // tail block 1 cell in
            (100, 4, 16, 99),   // budget a whole number of blocks
            (100, 4, 15, 99),   // budget not a multiple of r
            (37, 5, 9, 36),     // r that does not divide the budget
            (300, 4, 8, 299),   // many blocks, small budget
          ])
    func select_matchesRef(nKv: Int, r: Int, budget: Int, pos: Int) throws {
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let nBlocks = (nKv + r - 1) / r
        var rng = SeedTree(UInt64(nKv &* 1_000 &+ budget &* 10 &+ r)).key("idx-select-sparse")
        let unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(0, 100)) }

        let (cells, count) = try Self.select(ctx, kernel, unbiased: unbiased,
                                             pos: pos, nKv: nKv, r: r, budget: budget)
        let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)

        #expect(cells.map { Int($0) } == ref,
                "nKv=\(nKv) r=\(r) budget=\(budget): kernel \(cells) vs reference \(ref)")

        // The sparse path fills the budget exactly: width cells.
        let width = min(nKv, budget + r - 1)
        #expect(count == width, "expected \(width) cells, got \(count)")
        #expect(cells == cells.sorted(), "the list must be ascending by construction")
        #expect(Set(cells).count == cells.count, "no cell may be emitted twice")
    }

    /// The same comparison at the real budget. Every case above uses a budget
    /// of 8–16 cells, and the budget is what sizes the emitted-cell buffer
    /// (`QSAIndexer.selectCapacity`) and what the kernel's radix select is
    /// bounded by — the two places this family has already been bitten by a
    /// ceiling chosen for a smaller model (`Attention.maxQHeads`,
    /// `kPrefillRouterMaxExperts`). Budgets that fit in a toy block count can
    /// pass while 2048 cells overrun a buffer or truncate a selection.
    ///
    /// `r = 4` is the model's `indexerCompressRatio`, so the crossing sits
    /// where the engine actually crosses it: at `n_kv` 2052, the first length
    /// whose visible cells exceed `budget + r − 1` and leave the dense path.
    @Test("the real 2048-cell budget selects the same cells as the reference",
          arguments: [
            // (nKv, r, budget, pos) — pos = nKv - 1 is the decode shape.
            (2_048, 4, 2_048, 2_047),  // exactly the budget: still dense
            (2_051, 4, 2_048, 2_050),  // exactly the dense width
            (2_052, 4, 2_048, 2_051),  // one cell past it: the first sparse case
            (4_096, 4, 2_048, 4_095),  // blocks divide evenly, no tail
            (4_097, 4, 2_048, 4_096),  // tail block one cell in
            (8_192, 4, 2_048, 8_191),  // well past the budget
          ])
    func select_realBudgetMatchesRef(nKv: Int, r: Int, budget: Int, pos: Int) throws {
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let nBlocks = (nKv + r - 1) / r
        var rng = SeedTree(UInt64(nKv &* 1_000 &+ budget &* 10 &+ r))
            .key("idx-select-realbudget")
        let unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(0, 100)) }

        // The buffer the caller has to size. It must hold the widest selection
        // the kernel can emit, which is `width` — not the budget: the boundary
        // block contributes up to `r − 1` cells on top.
        let capacity = QSAIndexer.selectCapacity(nKv: nKv, r: r, budget: budget)
        let width = min(nKv, budget + r - 1)
        #expect(capacity >= width,
                "capacity \(capacity) cannot hold a \(width)-cell selection")

        let (cells, count) = try Self.select(ctx, kernel, unbiased: unbiased,
                                             pos: pos, nKv: nKv, r: r, budget: budget)
        let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)

        #expect(cells.map { Int($0) } == ref,
                "nKv=\(nKv) budget=\(budget): kernel and reference disagree")
        #expect(count == width, "expected \(width) cells, got \(count)")
        #expect(count <= capacity, "the kernel wrote past the buffer it was given")
        #expect(cells == cells.sorted(), "the list must be ascending by construction")
        #expect(Set(cells).count == cells.count, "no cell may be emitted twice")
        // Causality is a per-cell test, so nothing beyond the query may be
        // selected — whether the query's *own* cell survives is not asserted:
        // `blockBias` forces the block containing `pos` in only when `pos` does
        // not end a block, which several of these cases do.
        #expect(cells.allSatisfy { $0 <= UInt32(pos) })
    }

    @Test("the boundary block is cut mid-block, keeping its lowest cell indices")
    func select_cutsTheBoundaryBlock() throws {
        // r = 4, budget = 8 → width = 11: two whole blocks plus 3 cells. The
        // query ends a block (t = 0), so nothing is force-visible and the whole
        // selection is decided by the ranking.
        let r = 4, budget = 8, nKv = 40, pos = 39
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        let nBlocks = nKv / r                      // 10, all complete
        var rng = SeedTree(0xC01).key("idx-select-boundary")
        let unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(0, 100)) }

        let (cells, _) = try Self.select(ctx, kernel, unbiased: unbiased,
                                         pos: pos, nKv: nKv, r: r, budget: budget)

        // The expectation is derived here, from the scores alone. Writing
        // `0..<(q*r)` for the whole blocks instead would quietly assume the
        // top-ranked blocks are the low-indexed ones, and pass only when the
        // seed happens to make that true.
        let width = min(nKv, budget + r - 1)       // 11
        let nvis = pos + 1
        let t = nvis % r
        #expect(t == 0, "this case is built around the query ending a block")
        let K = width - min(t, width)
        let q = K / r                              // whole blocks kept
        let m = K - q * r                          // cells of the boundary block
        #expect(q == 2 && m == 3)

        let ranked = (0..<nBlocks).sorted { unbiased[$0] > unbiased[$1] }
        var expected = (0..<q).flatMap { k in (0..<r).map { ranked[k] * r + $0 } }
        expected += (0..<m).map { ranked[q] * r + $0 }
        #expect(cells.map { Int($0) } == expected.sorted(),
                "expected the top \(q) blocks whole plus \(m) low cells of the boundary (blocks \(ranked.prefix(q + 1))), got \(cells)")
        #expect(cells.count == q * r + m, "width is whole blocks plus the partial one")

        // The boundary's last cell is the one the cut drops — the proof that
        // this is a cell-level cut and not a whole-block one.
        #expect(!cells.contains(UInt32(ranked[q] * r + m)),
                "the boundary block must lose its highest cell index")
    }

    @Test("equal block scores rank to the lowest block index")
    func select_tiesRankToLowerBlockIndex() throws {
        let r = 4, budget = 8, nKv = 40, pos = 39
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        // Every complete block scores the same, so the ranking is decided
        // entirely by the tie rule. A radix select finds the tied score
        // correctly but says nothing about *which* tied blocks to keep —
        // getting this wrong would still produce 11 cells, just of the wrong
        // blocks, with no other test able to see it.
        let nBlocks = nKv / r
        let unbiased = [Float](repeating: 7.5, count: nBlocks)

        let (cells, _) = try Self.select(ctx, kernel, unbiased: unbiased,
                                         pos: pos, nKv: nKv, r: r, budget: budget)
        // width = 11 = blocks 0 and 1 whole, plus cells 0..2 of block 2.
        #expect(cells.map { Int($0) } == Array(0..<11),
                "ties must keep the lowest-indexed blocks, got \(cells)")

        let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)
        #expect(cells.map { Int($0) } == ref, "reference disagrees: \(ref)")
    }

    @Test("the query's own tail block is kept whatever the rest score")
    func select_tailBlockSurvives() throws {
        let r = 4, budget = 8
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        // nKv = 43 → nvis 43, ten complete blocks, a 3-cell tail. The complete
        // blocks score in the thousands, far above anything a relu-sum
        // reaches — and still lose to the tail's +1e9, which is the point:
        // without the bias the query could drop the block it sits in.
        for pos in [42, 41, 40, 39] {
            let nKv = pos + 1
            let nBlocks = (nKv + r - 1) / r
            var rng = SeedTree(UInt64(0xD00 + pos)).key("idx-select-tail")
            var unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(1000, 2000)) }
            unbiased[nBlocks - 1] = 0            // the tail block itself scores nothing

            let (cells, _) = try Self.select(ctx, kernel, unbiased: unbiased,
                                             pos: pos, nKv: nKv, r: r, budget: budget)

            let nc = (pos + 1) / r
            let t = (pos + 1) - nc * r
            if t > 0 {
                let tail = (0..<t).map { UInt32(nc * r + $0) }
                #expect(tail.allSatisfy { cells.contains($0) },
                        "pos=\(pos): the tail block \(nc) must be kept, got \(cells)")
            } else {
                // The query ended a complete block, so there is no tail at all
                // and the budget goes entirely to ranked blocks.
                #expect(cells.count == budget + r - 1, "pos=\(pos): width cells")
            }

            let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)
            #expect(cells.map { Int($0) } == ref, "pos=\(pos): reference disagrees")
        }
    }

    @Test("selection is stable as the timeline grows by one cell at a time")
    func select_isStableAcrossGrowth() throws {
        // A decode appends one cell per step, so the selected set moves by at
        // most a few entries between steps. This walks across a block boundary
        // and across the point where the tail block fills and a new one opens.
        let r = 4, budget = 8
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        for nKv in 30...48 {
            let pos = nKv - 1
            let nBlocks = (nKv + r - 1) / r
            var rng = SeedTree(UInt64(0xE00 + nKv)).key("idx-select-growth")
            let unbiased = (0..<nBlocks).map { _ in Float(rng.uniform(0, 100)) }

            let (cells, count) = try Self.select(ctx, kernel, unbiased: unbiased,
                                                 pos: pos, nKv: nKv, r: r, budget: budget)
            let ref = QSAIndexerRef.topKCells(unbiased, pos: pos, n_kv: nKv, r: r, budget: budget)
            #expect(cells.map { Int($0) } == ref, "nKv=\(nKv): kernel vs reference")
            #expect(count == min(nKv, budget + r - 1), "nKv=\(nKv): width")
            #expect(cells.allSatisfy { Int($0) <= pos }, "nKv=\(nKv): a future cell leaked in")
        }
    }

    // MARK: - the whole chain, as a decode step will drive it

    @Test("block scores → selection matches the reference pipeline end to end",
          arguments: [
            (40, 8, UInt64(0xF01)),    // boundary at width, tail aligned
            (43, 8, UInt64(0xF02)),    // 3-cell tail
            (41, 8, UInt64(0xF03)),    // 1-cell tail
            (200, 12, UInt64(0xF04)),  // deeper timeline, larger budget
          ])
    func select_chainFromBlockScores(nKv: Int, budget: Int, seed: UInt64) throws {
        let nHeads = 4, idxDim = 128, r = 4
        let pos = nKv - 1
        let nBlocks = (nKv + r - 1) / r
        let nComplete = nKv / r
        var rng = SeedTree(seed).key("idx-select-chain")
        let ctx = try MetalContext()
        let kernel = try QSAIndexer(context: ctx)

        // Pooled keys for the complete blocks; the tail block's slot is left
        // unwritten-and-poisoned, since the score kernel must not read it.
        let qH = (0..<(nHeads * idxDim)).map { _ in Float16(rng.uniform(-1.0, 1.0)) }
        var poolH = [Float16](repeating: .nan, count: nBlocks * idxDim)
        for i in 0..<(nComplete * idxDim) {
            poolH[i] = Float16(rng.uniform(-1.0, 1.0))
        }

        let capacity = QSAIndexer.selectCapacity(nKv: nKv, r: r, budget: budget)
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qH),
              let pBuf = Fp16Buffer.make(ctx.device, halves: poolH),
              let sBuf = Self.f32Buffer(ctx.device,
                                        values: [Float](repeating: 0, count: nBlocks)),
              let cBuf = Self.u32Buffer(ctx.device, count: capacity),
              let nBuf = Self.u32Buffer(ctx.device, count: 1) else {
            Issue.record("alloc failed"); return
        }

        // One command buffer, the way the layer encodes it: score, then select.
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeBlockScores(commandBuffer: cb, q: qBuf, pooled: pBuf, scores: sBuf,
                                 nHeads: UInt32(nHeads), idxDim: UInt32(idxDim),
                                 r: UInt32(r), nKv: UInt32(nKv), pos: UInt32(pos))
        kernel.encodeSelectCells(commandBuffer: cb, scores: sBuf, cells: cBuf, count: nBuf,
                                 pos: UInt32(pos), nKv: UInt32(nKv),
                                 r: UInt32(r), budget: UInt32(budget))
        cb.commit(); cb.waitUntilCompleted()

        let n = Int(Self.readU32(nBuf, count: 1)[0])
        let cells = Array(Self.readU32(cBuf, count: capacity).prefix(n))

        // Reference over the same inputs: score the complete blocks, then the
        // full top-k over cells with the bias and the causal test. The forced
        // blocks carry no score of their own — the kernel writes a flat 1e9
        // without reading the unusable pool, so their unbiased score is 0 and
        // the bias alone makes both sides 1e9 exactly. Handing the reference
        // the NaN the poisoned slot would have produced is not equivalent: a
        // NaN sorts unpredictably and the comparison would be meaningless.
        let qRef = qH.map { Float($0) }
        let poolRef = poolH.map { Float($0) }
        let refScores = QSAIndexerRef.blockScores(qRef, poolRef, n_kv: nComplete * r,
                                                  r: r, idxDim: idxDim, nIdxHeads: nHeads)
        var refBlockScores = [Float](repeating: 0, count: nBlocks)
        for b in 0..<nComplete { refBlockScores[b] = refScores[b] }
        let ref = QSAIndexerRef.topKCells(refBlockScores, pos: pos, n_kv: nKv,
                                          r: r, budget: budget)

        let width = min(nKv, budget + r - 1)
        #expect(cells.map { Int($0) } == ref,
                "nKv=\(nKv): kernel \(cells) vs reference \(ref)")
        #expect(n == width, "nKv=\(nKv): expected \(width) cells, got \(n)")
        #expect(cells.allSatisfy { $0 < UInt32(nKv) }, "nKv=\(nKv): past the timeline")
        #expect(cells.allSatisfy { Int($0) <= pos }, "nKv=\(nKv): a future cell leaked in")

        // With a tail the query's own block is force-visible, so the cell it
        // attends from must be in the list. When pos ends a complete block
        // (t == 0) there is no tail, that block carries the ordinary zero bias
        // and competes on its score like every other — llama's mask has no
        // diagonal term — so nothing is asserted about it there.
        if (pos + 1) % r > 0 {
            #expect(cells.contains(UInt32(pos)),
                    "nKv=\(nKv): the query's own cell is missing from the forced tail")
        }
    }
}

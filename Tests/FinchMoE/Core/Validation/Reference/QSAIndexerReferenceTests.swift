import Testing
import Foundation
import FinchMoEValidationSupport

/// Cross-validates `QSAIndexerRef` (the Qwen 3.8 QSA indexer selection,
/// locked to `archive/llama.cpp/src/models/qwen4exp.cpp` `build_qsa_top_k`
/// :477-616) against a second formulation written from scratch. The point is
/// to prove the *reference* right, so the Metal kernels (M3.2b) have a
/// trustworthy comparator.
///
/// The independent formulations share no code path with the reference and
/// differ in summation order / activation identity, so agreement to
/// `Tolerance.identity` is a real check on the math, not on shared code.
/// Geometries mirror the reference suite: tiny (hand-checkable), and the
/// real 3.8 widths (idxDim 128, nIdxHeads 4, r 4).
@Suite struct QSAIndexerReferenceTests {

    // MARK: - mean-pool

    @Test("mean pool divides by the members actually present (tail block)")
    func meanPool_tailBlock() {
        let idxDim = 4
        // 5 cells: block 0 = [0,1,2,3] full, block 1 = [4] tail.
        var raw = [Float](repeating: 0, count: 5 * idxDim)
        for c in 0..<5 {
            for i in 0..<idxDim { raw[c * idxDim + i] = Float(c + 1) }
        }
        // Block 0's members hold 1,2,3,4, so their mean is 2.5 — dividing by
        // the full block size instead would give 2.0.
        let b0 = QSAIndexerRef.meanPool([0, 1, 2, 3], raw: raw, idxDim: idxDim)
        for v in b0 { #expect(abs(v - 2.5) < 1e-6, "block 0: \(v)") }
        // Block 1 (tail, single member [5,5,5,5]): mean → [5,5,5,5].
        let b1 = QSAIndexerRef.meanPool([4], raw: raw, idxDim: idxDim)
        for v in b1 { #expect(abs(v - 5) < 1e-6, "block 1: \(v)") }
    }

    @Test("mean pool of a full block equals the naive sum/r")
    func meanPool_fullBlock_matchesNaive() {
        let idxDim = 8, r = 4
        var raw = [Float](repeating: 0, count: r * idxDim)
        for c in 0..<r { for i in 0..<idxDim { raw[c * idxDim + i] = Float(c * 3 + i) } }

        let ref = QSAIndexerRef.meanPool(Array(0..<r), raw: raw, idxDim: idxDim)
        // Naive: sum members in reverse, divide last.
        var naive = [Float](repeating: 0, count: idxDim)
        for c in (0..<r).reversed() {
            for i in 0..<idxDim { naive[i] += raw[c * idxDim + i] }
        }
        for i in 0..<idxDim { naive[i] /= Float(r) }
        let rel = RelError.compute(actual: ref, reference: naive)
        #expect(rel < Tolerance.identity, "meanPool rel=\(rel)")
    }

    // MARK: - RMSNorm

    @Test("rms normalises each vector to unit RMS then scales by gamma")
    func rms_unitRMSAndGamma() {
        let dim = 16
        let x = (0..<dim).map { Float($0) }          // 0..15
        let gamma = (0..<dim).map { Float($0 % 3) }  // 0,1,2,0,1,2,...
        let y = QSAIndexerRef.rms(x, gamma: gamma)
        // RMS of x = sqrt(mean(x^2)) = sqrt((0^2+...+15^2)/16) = sqrt(120).
        let rmsX = x.reduce(0) { $0 + $1 * $1 } / Float(dim)
        let inv = 1.0 / rmsX.squareRoot()
        for i in 0..<dim {
            let expected = x[i] * inv * gamma[i]
            #expect(abs(y[i] - expected) < 1e-5, "i=\(i): \(y[i]) vs \(expected)")
        }
    }

    // MARK: - RoPE

    @Test("rope rotates the first nRot dims half-split; the tail is untouched")
    func rope_partialAndHalfSplit() {
        let idxDim = 32, nRot = 8, half = 4
        var x = [Float](repeating: 0, count: idxDim)
        // Put a known value in one first-half and one second-half slot.
        x[1] = 1.0
        x[1 + half] = 2.0   // = x[5]
        let pos = 3
        let y = QSAIndexerRef.rope(x, pos: pos, nRot: nRot)

        // freq for pair i=1: theta^(-2*1/nRot) = 1e7^(-2/8) = 1e7^(-0.25)
        let freq = powf(1e7, -0.25)
        let ang = Float(pos) * freq
        let c = cosf(ang), s = sinf(ang)
        #expect(abs(y[1] - (1.0 * c - 2.0 * s)) < 1e-4, "y[1]=\(y[1])")
        #expect(abs(y[1 + half] - (1.0 * s + 2.0 * c)) < 1e-4, "y[5]=\(y[5])")
        // Dims beyond nRot are untouched.
        for i in nRot..<idxDim { #expect(y[i] == 0, "tail dim \(i) must be 0") }
        // pos 0 is an identity rotation.
        let id = QSAIndexerRef.rope(x, pos: 0, nRot: nRot)
        for i in 0..<nRot { #expect(abs(id[i] - x[i]) < 1e-6, "id[\(i)]=\(id[i])") }
    }

    @Test("rope matches a half-split rotate_half transcription (real geometry)")
    func rope_matchesRotateHalf() {
        let idxDim = 128, nRot = 32, half = 16
        var x = [Float](repeating: 0, count: idxDim)
        for i in 0..<nRot { x[i] = Float(i) / 16 }
        let pos = 10

        let ref = QSAIndexerRef.rope(x, pos: pos, nRot: nRot)
        // Independent: rotate_half then x·cos + rot(x)·sin, freq denom = nRot.
        var rot = [Float](repeating: 0, count: idxDim)
        for i in 0..<half {
            rot[i] = -x[i + half]
            rot[i + half] = x[i]
        }
        var naive = [Float](repeating: 0, count: idxDim)
        for i in 0..<half {
            let freq = powf(1e7, -Float(2 * i) / Float(nRot))
            let ang = Float(pos) * freq
            let c = cosf(ang), s = sinf(ang)
            naive[i] = x[i] * c + rot[i] * s
            naive[i + half] = x[i + half] * c + rot[i + half] * s
        }
        for i in nRot..<idxDim { naive[i] = x[i] }
        let rel = RelError.compute(actual: ref, reference: naive)
        #expect(rel < Tolerance.identity, "rope rel=\(rel)")
    }

    // MARK: - block score

    @Test("block score rectifies each head before summing (relu-of-each, not relu-of-sum)")
    func blockScore_reluPerHead() {
        let idxDim = 8, nH = 3
        // Head 0 dot = +5, head 1 dot = -5, head 2 dot = +2.
        // Per-head relu then sum = 5 + 0 + 2 = 7. relu-of-sum would be 2.
        var q = [Float](repeating: 0, count: nH * idxDim)
        var k = [Float](repeating: 0, count: idxDim)
        q[0] = 1; q[idxDim] = -1; q[2 * idxDim + 2] = 1   // q heads tap k[0,k[0],k[2]]
        k[0] = 5; k[1] = -5; k[2] = 2                     // k (shared) → dots 5,-5,2
        let score = QSAIndexerRef.blockScore(q: q, kBlock: k,
                                             idxDim: idxDim, nIdxHeads: nH)
        #expect(abs(score - 7) < 1e-5, "score=\(score), expected 7 (not 2)")
    }

    @Test("block scores carry no mask: every block is scored, visible or not")
    func blockScores_unmasked() {
        let idxDim = 8, r = 2, n_kv = 6   // 3 blocks: [0,1],[2,3],[4,5]
        let nH = 2
        var q = [Float](repeating: 0, count: nH * idxDim)
        q[0] = 1                          // every block scores ~ k·1
        var k = [Float](repeating: 0, count: 3 * idxDim)
        for b in 0..<3 { k[b * idxDim + 0] = Float(b + 1) }

        // Causality is the mask's job — llama adds it to the *expanded* cell
        // scores (qwen4exp.cpp:597-600), not to the block scores — so a block
        // past the query is still scored here rather than masked away.
        let scores = QSAIndexerRef.blockScores(q, k, n_kv: n_kv, r: r,
                                              idxDim: idxDim, nIdxHeads: nH)
        for b in 0..<3 { #expect(scores[b].isFinite, "block \(b) scored") }
        #expect(scores[2] > scores[1] && scores[1] > scores[0],
                "scores increase with b")
    }

    // MARK: - block bias

    @Test("bias leaves complete blocks at 0 and forces the query's own block visible")
    func blockBias_tailVersusComplete() {
        let r = 4
        // Query at the end of a complete block: n_kv 12, pos 11.
        // tail_start = (12/4)*4 = 12 → every block is complete and behind it.
        #expect(QSAIndexerRef.blockBias(n_kv: 12, r: r, pos: 11) == [0, 0, 0])

        // Short last block: n_kv 11, pos 10. tail_start = (11/4)*4 = 8, so
        // block 2 (cells 8..10) is the tail and the two complete ones stay 0.
        let tail = QSAIndexerRef.blockBias(n_kv: 11, r: r, pos: 10)
        #expect(tail == [0, 0, 1e9], "got \(tail)")
    }

    @Test("a mid-block query biases every block at or past it, and the mask trims")
    func blockBias_midBlockQuery() {
        // A cell window reaching past the query (a prefill chunk): pos 2 with
        // 6 cells. tail_start = (3/4)*4 = 0, so no block sits behind the query
        // and every block takes +1e9. The bias may overshoot here because the
        // causal mask, not the bias, is what bounds visibility.
        let bias = QSAIndexerRef.blockBias(n_kv: 6, r: 4, pos: 2)
        #expect(bias == [1e9, 1e9], "got \(bias)")

        let cells = QSAIndexerRef.topKCells([1, 5], pos: 2, n_kv: 6, r: 4,
                                            budget: 2048)
        #expect(cells == [0, 1, 2], "mask trims to the causal prefix: \(cells)")
    }

    // MARK: - top-k selection

    @Test("dense fast path: n_kv ≤ budget selects every visible cell")
    func topK_denseFastPath() {
        let r = 4, budget = 2048
        let n_kv = 1000   // 250 blocks, 1000 cells — well under the budget
        let scores = [Float](repeating: 0.5, count: n_kv / r)
        let cells = QSAIndexerRef.topKCells(scores, pos: n_kv - 1,
                                            n_kv: n_kv, r: r, budget: budget)
        #expect(cells == Array(0..<n_kv), "dense path keeps all \(cells.count) cells")
    }

    @Test("sparse path keeps whole blocks and cuts the last one mid-block")
    func topK_sparseSelectsTopBlocks() {
        // budget is in cells (qwen4exp.cpp:607), so width = 8 + 4 - 1 = 11.
        let r = 4, budget = 8
        let n_kv = 20                // 5 complete blocks, 0..4
        // Scores: block 2 highest, then 4, then 0, then 3, then 1.
        let scores: [Float] = [3, 0, 5, 1, 4]
        let pos = n_kv - 1           // n_kv = pos + 1: no future cells
        let cells = QSAIndexerRef.topKCells(scores, pos: pos, n_kv: n_kv,
                                            r: r, budget: budget)
        // 20 visible cells > width 11, and no block is the tail at pos 19
        // (tail_start = 20), so the ranking is by score alone: blocks 2 (5)
        // and 4 (4) fill 8 cells, and block 0 (3) supplies the last 3 — a tie
        // inside block 0, broken toward its lower cells.
        let expected = ([0, 1, 2] + Array(8..<12) + Array(16..<20)).sorted()
        #expect(cells == expected, "got \(cells)")
    }

    @Test("sparse path respects causality: future blocks can never be selected")
    func topK_sparseCausal() {
        let r = 4, budget = 4        // width = 7
        let n_kv = 20
        let scores: [Float] = [3, 0, 5, 1, 4]   // blocks 3 and 4 outscore 3
        let pos = 8                  // visible cells 0..8; tail_start = 8
        let cells = QSAIndexerRef.topKCells(scores, pos: pos, n_kv: n_kv,
                                            r: r, budget: budget)
        // Cell 8 is block 2's only causally-valid member and block 2 is the
        // tail (b*r = 8 >= tail_start = 8), so it is forced in whatever it
        // scores. The other 6 cells come from block 0 (score 3) first, then
        // block 1 (score 0, tie → lower cells). Blocks 3 and 4 — the two
        // highest-scoring blocks — lie wholly in the future.
        let expected = ([0, 1, 2, 3] + [4, 5] + [8]).sorted()
        #expect(cells == expected, "got \(cells)")
    }

    @Test("the query's own block is kept even when it scores worst")
    func topK_incompleteTailIsForcedVisible() {
        let r = 4, budget = 4        // width = 7
        let n_kv = 11                // blocks 0,1 complete; block 2 = cells 8..10
        let scores: [Float] = [5, 4, 0]   // the tail block scores worst
        let pos = n_kv - 1
        let cells = QSAIndexerRef.topKCells(scores, pos: pos, n_kv: n_kv,
                                            r: r, budget: budget)
        // tail_start = (11/4)*4 = 8, so block 2 is the tail and its 3 cells
        // rank ahead of both complete blocks. Without the bias this returns
        // block 1's cells instead and loses the token the query sits on.
        let expected = ([0, 1, 2, 3] + [8, 9, 10]).sorted()
        #expect(cells == expected, "got \(cells)")
        #expect(!cells.contains(7), "block 1 must not displace the tail")
    }

    @Test("ties break deterministically to the lower cell index")
    func topK_tieBreakLowerIndex() {
        let r = 2, budget = 2        // width = 2 + 2 - 1 = 3
        let n_kv = 8                 // 4 complete blocks
        let scores: [Float] = [1, 1, 1, 1]   // all tied
        let pos = n_kv - 1
        let cells = QSAIndexerRef.topKCells(scores, pos: pos, n_kv: n_kv,
                                            r: r, budget: budget)
        #expect(cells == [0, 1, 2], "3 lowest cells, got \(cells)")
    }

    @Test("the tail block contributes only its present members")
    func topK_tailBlockPartial() {
        // width = 8 + 4 - 1 = 11 >= n_kv, so this is the dense fast path and
        // the short tail block simply keeps its 2 members.
        let r = 4, budget = 8
        let n_kv = 10   // blocks 0..2; block 2 = cells [8,9] (tail, 2 members)
        let scores: [Float] = [1, 2, 3]
        let pos = n_kv - 1
        let cells = QSAIndexerRef.topKCells(scores, pos: pos, n_kv: n_kv,
                                            r: r, budget: budget)
        #expect(cells == Array(0..<10), "tail block keeps only [8,9]; got \(cells)")
    }

    @Test("the dense fast path ends exactly at budget + r - 1 cells")
    func topK_budgetCrossing() {
        let r = 4, budget = 2048     // width = min(n_kv, 2051)

        // At the limit every cell is visible and every one is kept.
        let atLimit = QSAIndexerRef.topKCells(
            [Float](repeating: 1, count: 513), pos: 2050, n_kv: 2051,
            r: r, budget: budget)
        #expect(atLimit == Array(0..<2051), "dense at the limit: \(atLimit.count)")

        // One cell past it the cut lands inside the *complete* blocks: 2050 of
        // their cells (block 0's first, by score) plus the forced tail cell.
        var scores = [Float](repeating: 1, count: 514)   // n_blocks = 514
        scores[0] = 100
        let past = QSAIndexerRef.topKCells(scores, pos: 2052, n_kv: 2053,
                                           r: r, budget: budget)
        // tail_start = (2053/4)*4 = 2052, so block 513 (cell 2052 alone) is the
        // tail: forced ahead of everything despite scoring 1. Width caps the
        // rest at 2050 cells, taken lowest-index-first. Cells 2050 and 2051 are
        // the ones that fall off.
        #expect(past == Array(0...2049) + [2052], "got \(past.count) cells")
    }

    // MARK: - end-to-end select (real geometry)

    @Test("select: a hand-built query/key pair picks the intended block")
    func select_picksStrongestBlock() {
        let idxDim = 16, nH = 2, r = 2, budget = 2
        let n_kv = 6   // 3 blocks
        // One key head, so a block's pooled key is just its mean key. Build
        // keys so block 1 is the strongest match for the query.
        var kPooled = [Float](repeating: 0, count: 3 * idxDim)
        for i in 0..<idxDim { kPooled[i] = 0.2 }              // block 0 weak
        for i in 0..<idxDim { kPooled[idxDim + i] = 1.0 }     // block 1 strong
        for i in 0..<idxDim { kPooled[2 * idxDim + i] = 0.3 } // block 2 weak

        // Query: both heads align with block 1's direction.
        var q = [Float](repeating: 0, count: nH * idxDim)
        for i in 0..<idxDim { q[i] = 1.0; q[idxDim + i] = 1.0 }

        let pos = n_kv - 1
        let cells = QSAIndexerRef.select(q: q, kPooledNormRope: kPooled,
                                         pos: pos, n_kv: n_kv, r: r,
                                         idxDim: idxDim, nIdxHeads: nH,
                                         budget: budget)
        // Both query heads align with block 1, so the scores run block 1 ~ 32,
        // block 2 ~ 9.6, block 0 ~ 6.4. budget 2 is in cells, so width = 2 + 2
        // - 1 = 3: block 1 takes both its cells and block 2 supplies the third
        // (tie inside it → its lower cell).
        let expected = [2, 3, 4]   // block 1 whole, block 2's first cell
        #expect(cells == expected, "expected blocks 1&2, got \(cells)")
    }
}

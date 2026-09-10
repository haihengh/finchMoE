import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// `Attention.encodeFullCells` — full attention restricted to an explicit
/// ascending cell list, which is how the QSA indexer's selection reaches the
/// attention body.
///
/// The reference is `AttentionRef.apply` over a *gathered* K/V: the selected
/// rows compacted into a dense `[nCells, numKVHeads, headDim]` array. That is
/// the right comparison rather than a mask full of `-inf`, because the kernel
/// omits unselected cells from the softmax maximum entirely — and `-inf` also
/// yields zero mass, so the two agree exactly where it matters, while the
/// gathered form additionally pins the summation order to something a CPU
/// reference can reproduce.
@Suite struct AttentionCellsTests {

    private static let headDim = 128
    private static let numQHeads = 16
    private static let numKVHeads = 2

    /// Gather the `cells` rows of a `[seqLen, numKVHeads, headDim]` buffer.
    private static func gather(_ src: [Float16], cells: [UInt32],
                               numKVHeads: Int, headDim: Int) -> [Float16] {
        var out = [Float16]()
        out.reserveCapacity(cells.count * numKVHeads * headDim)
        for c in cells {
            let base = Int(c) * numKVHeads * headDim
            out.append(contentsOf: src[base..<(base + numKVHeads * headDim)])
        }
        return out
    }

    /// Relative L2 difference, matching the convention in `AttentionTests`.
    private static func relError(_ got: [Float], _ want: [Float]) -> Float {
        var num: Float = 0, den: Float = 0
        for i in 0..<want.count {
            num += (got[i] - want[i]) * (got[i] - want[i])
            den += want[i] * want[i]
        }
        return den > 0 ? (num / den).squareRoot() : num.squareRoot()
    }

    private static func runCells(_ ctx: MetalContext, _ kernel: Attention,
                                 q: [Float16], k: [Float16], v: [Float16],
                                 cells: [UInt32], out: MTLBuffer,
                                 headDim: Int, numQHeads: Int, numKVHeads: Int) throws {
        guard let qBuf = Fp16Buffer.make(ctx.device, halves: q),
              let kBuf = Fp16Buffer.make(ctx.device, halves: k),
              let vBuf = Fp16Buffer.make(ctx.device, halves: v),
              let cBuf = ctx.device.makeBuffer(
                bytes: cells, length: cells.count * MemoryLayout<UInt32>.size,
                options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeFullCells(commandBuffer: cb,
                               q: qBuf, k: kBuf, v: vBuf,
                               cells: cBuf, out: out,
                               headDim: UInt32(headDim),
                               numQHeads: UInt32(numQHeads),
                               numKVHeads: UInt32(numKVHeads),
                               nCells: UInt32(cells.count))
        cb.commit(); cb.waitUntilCompleted()
    }

    // MARK: - the equivalence that anchors the kernel

    @Test("a cell list covering the whole timeline reproduces dense full attention exactly")
    func cells_fullTimelineIsBitwiseDense() throws {
        // The two kernels share a body and derive their chunking from the same
        // `chunkCount`, so with cells = 0…n-1 the partials are computed by the
        // same instructions on the same addresses. Anything short of bit
        // equality here means the sparse path diverged somewhere structural —
        // an off-by-one in the chunk split, a different threadgroup width —
        // which a tolerance-based test would absorb silently.
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)
        let headDim = Self.headDim, numQ = Self.numQHeads, numKV = Self.numKVHeads
        let seqLen = 100

        var rng = SeedTree(0xB01).key("attn-cells-full")
        let qF = (0..<(numQ * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        guard let qBuf = Fp16Buffer.make(ctx.device, halves: qF),
              let kBuf = Fp16Buffer.make(ctx.device, halves: kF),
              let vBuf = Fp16Buffer.make(ctx.device, halves: vF),
              let dense = Fp16Buffer.make(ctx.device, count: numQ * headDim),
              let sparse = Fp16Buffer.make(ctx.device, count: numQ * headDim) else {
            Issue.record("alloc failed"); return
        }
        let cells = Array(0..<UInt32(seqLen))

        // Separate command buffers: the two encoders share the split-KV
        // scratch, and this test is about the kernels, not about the
        // scratch's hazard tracking.
        let cb1 = ctx.queue.makeCommandBuffer()!
        attention.encodeFull(commandBuffer: cb1, q: qBuf, k: kBuf, v: vBuf, out: dense,
                             headDim: UInt32(headDim), numQHeads: UInt32(numQ),
                             numKVHeads: UInt32(numKV), seqLen: UInt32(seqLen))
        cb1.commit(); cb1.waitUntilCompleted()

        try Self.runCells(ctx, attention, q: qF, k: kF, v: vF, cells: cells, out: sparse,
                          headDim: headDim, numQHeads: numQ, numKVHeads: numKV)

        let a = Fp16Buffer.read(dense, count: numQ * headDim)
        let b = Fp16Buffer.read(sparse, count: numQ * headDim)
        #expect(a == b, "sparse over the full timeline must equal dense, bit for bit")
    }

    // MARK: - sparse selection vs the gathered reference

    @Test("sparse attention matches the reference over the selected cells",
          arguments: [
            // (seqLen, selected description) — built inside the test.
            (40, 0),
            (100, 1),
            (200, 2),
            (37, 3),
          ])
    func cells_matchesReference(seqLen: Int, variant: Int) throws {
        let headDim = Self.headDim, numQ = Self.numQHeads, numKV = Self.numKVHeads
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)

        var rng = SeedTree(UInt64(0xB10 + variant)).key("attn-cells-ref")
        let qF = (0..<(numQ * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        let cells: [UInt32]
        switch variant {
        case 0:
            cells = [0, 3, 7, 11, 12, 30, 39]                 // scattered, sparse
        case 1:
            cells = Array(stride(from: 0, to: seqLen, by: 3)).map { UInt32($0) }
        case 2:
            // Deep enough to need more than one pass-1 chunk (chunkCount
            // caps at 16, so 200 cells split 13 per chunk).
            cells = Array(0..<UInt32(seqLen)).filter { $0 % 5 != 4 }
        default:
            cells = [UInt32(seqLen - 1)]                      // only the last cell
        }

        guard let out = Fp16Buffer.make(ctx.device, count: numQ * headDim) else {
            Issue.record("alloc failed"); return
        }
        try Self.runCells(ctx, attention, q: qF, k: kF, v: vF, cells: cells, out: out,
                          headDim: headDim, numQHeads: numQ, numKVHeads: numKV)

        let gk = Self.gather(kF, cells: cells, numKVHeads: numKV, headDim: headDim)
        let gv = Self.gather(vF, cells: cells, numKVHeads: numKV, headDim: headDim)
        let want = AttentionRef.apply(q: qF.map { Float($0) },
                                      k: gk.map { Float($0) },
                                      v: gv.map { Float($0) },
                                      headDim: headDim, numQHeads: numQ,
                                      numKVHeads: numKV, seqLen: cells.count)

        let got = Fp16Buffer.read(out, count: numQ * headDim)
        let rel = Self.relError(got, want)
        #expect(rel < Tolerance.fp16ChainedReduction,
                "seqLen=\(seqLen) variant=\(variant) nCells=\(cells.count): rel \(rel)")
    }

    // MARK: - the mask is omission, not a bias

    @Test("an unselected cell contributes no mass however it scores")
    func cells_excludedCellIsIgnored() throws {
        // A large-negative bias implementation and an omission implementation
        // agree everywhere except here: if the kernel "masked" by adding a big
        // number to the score, a key engineered to score very high would still
        // reach the softmax through the running maximum and shift every
        // weight. Omission cannot see it at all.
        let headDim = Self.headDim, numQ = Self.numQHeads, numKV = Self.numKVHeads
        let seqLen = 20
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)

        var rng = SeedTree(0xB20).key("attn-cells-excluded")
        let qF = (0..<(numQ * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        var kF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        // Cell 5 is the trap: aligned with q so every head scores it hugely.
        let trap = 5
        for kvh in 0..<numKV {
            for i in 0..<headDim {
                kF[(trap * numKV + kvh) * headDim + i] = Float16(30.0)
            }
        }
        let cells = Array(0..<UInt32(seqLen)).filter { $0 != UInt32(trap) }

        guard let out = Fp16Buffer.make(ctx.device, count: numQ * headDim) else {
            Issue.record("alloc failed"); return
        }
        try Self.runCells(ctx, attention, q: qF, k: kF, v: vF, cells: cells, out: out,
                          headDim: headDim, numQHeads: numQ, numKVHeads: numKV)

        // Reference over the list WITHOUT the trap row.
        let gk = Self.gather(kF, cells: cells, numKVHeads: numKV, headDim: headDim)
        let gv = Self.gather(vF, cells: cells, numKVHeads: numKV, headDim: headDim)
        let want = AttentionRef.apply(q: qF.map { Float($0) },
                                      k: gk.map { Float($0) },
                                      v: gv.map { Float($0) },
                                      headDim: headDim, numQHeads: numQ,
                                      numKVHeads: numKV, seqLen: cells.count)
        let got = Fp16Buffer.read(out, count: numQ * headDim)
        let rel = Self.relError(got, want)
        #expect(rel < Tolerance.fp16ChainedReduction, "the trap cell leaked in: rel \(rel)")

        // And the trap is genuinely tempting: attending to it would give the
        // same q a completely different answer, so the check above has teeth.
        let withTrap = Array(0..<UInt32(seqLen))
        let gkT = Self.gather(kF, cells: withTrap, numKVHeads: numKV, headDim: headDim)
        let gvT = Self.gather(vF, cells: withTrap, numKVHeads: numKV, headDim: headDim)
        let wantTrap = AttentionRef.apply(q: qF.map { Float($0) },
                                          k: gkT.map { Float($0) },
                                          v: gvT.map { Float($0) },
                                          headDim: headDim, numQHeads: numQ,
                                          numKVHeads: numKV, seqLen: seqLen)
        let gap = Self.relError(got, wantTrap)
        #expect(gap > 0.1, "including the trap should change the answer a lot, got rel \(gap)")
    }

    @Test("a single selected cell returns that cell's value")
    func cells_singleCellIsItsValue() throws {
        // Softmax over one position is exactly 1, so the output is V[p] with no
        // interpolation — the sharpest possible check that the cell list
        // indexes K and V at the same row and that no other cell sneaks in.
        let headDim = Self.headDim, numQ = Self.numQHeads, numKV = Self.numKVHeads
        let seqLen = 12
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)

        var rng = SeedTree(0xB30).key("attn-cells-single")
        let qF = (0..<(numQ * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let kF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }
        let vF = (0..<(seqLen * numKV * headDim)).map { _ in Float16(rng.uniform(-0.5, 0.5)) }

        for cell in [0, 5, seqLen - 1] {
            let cells = [UInt32(cell)]
            guard let out = Fp16Buffer.make(ctx.device, count: numQ * headDim) else {
                Issue.record("alloc failed"); return
            }
            try Self.runCells(ctx, attention, q: qF, k: kF, v: vF, cells: cells, out: out,
                              headDim: headDim, numQHeads: numQ, numKVHeads: numKV)
            // readHalf, not read: this comparison is equality on the stored
            // FP16, and the point is that the value comes back untouched.
            let got = Fp16Buffer.readHalf(out, count: numQ * headDim)
            for qh in 0..<numQ {
                // Every Q head reads its own KV head's row; with one cell the
                // softmax weight is 1, so this must come back untouched — the
                // output is FP16 stored from an FP32 accumulator, and 1.0 × v
                // is exact, which is why this can be an equality.
                let kvHead = qh / (numQ / numKV)
                let base = (cell * numKV + kvHead) * headDim
                let want = Array(vF[base..<(base + headDim)])
                let row = Array(got[qh * headDim..<(qh * headDim + headDim)])
                #expect(row == want, "cell=\(cell) qh=\(qh) is not a clean copy of V")
            }
        }
    }
}

import Testing
import Foundation
import FinchMoEValidationSupport

/// Cross-validates `PLERef` (the Qwen 3.8 PLE n-gram head, locked to
/// `archive/llama.cpp/src/models/qwen4exp.cpp` `llm_graph_input_ple::set_input`
/// :990-1047 and `build_ple` :1122-1213) against second formulations written
/// from scratch. The point is to prove the *reference* right, so the M3.3b
/// kernels and the M3.3c wiring have a trustworthy comparator.
///
/// The host hash is **exact integer work**: a wrong row is a different row,
/// not a slightly-off vector, so those cases carry hand-computed expectations
/// rather than tolerances, and the wrap cases are chosen so that a
/// wider-than-64-bit implementation would give a *different* answer.
@Suite struct PLEReferenceTests {

    // MARK: - Context window

    @Test("the window walks back over the real predecessors")
    func contextWindow_walksBack() {
        let tokens: [Int32] = [10, 11, 12, 13, 14]
        let ctx = PLERef.contextWindow(tokens: tokens, position: 4,
                                       ngramSize: 3, eos: 99)
        #expect(ctx == [14, 13, 12])
    }

    @Test("a token's own EOS does not cut its own context")
    func contextWindow_ownEOSDoesNotCut() {
        // The generation's final EOS still gets a full real n-gram window —
        // only tokens *behind* an EOS are frozen, never the one in front.
        let tokens: [Int32] = [10, 11, 12, 99]
        let ctx = PLERef.contextWindow(tokens: tokens, position: 3,
                                       ngramSize: 3, eos: 99)
        #expect(ctx == [99, 12, 11])
    }

    @Test("an EOS in the window freezes everything behind it (sticky cut)")
    func contextWindow_stickyCut() {
        let tokens: [Int32] = [10, 99, 12, 13]
        let ctx = PLERef.contextWindow(tokens: tokens, position: 3,
                                       ngramSize: 3, eos: 99)
        // [13, 12, 99] would be the non-sticky reading — it would reach back
        // *past* the EOS to token 10. The cut makes the whole tail EOS.
        #expect(ctx == [13, 12, 99])

        let deeper = PLERef.contextWindow(tokens: [7, 8, 99, 12, 13], position: 4,
                                          ngramSize: 4, eos: 99)
        #expect(deeper == [13, 12, 99, 99], "cut keeps applying past its position")
    }

    @Test("the sequence start reads as EOS, position 0 entirely so")
    func contextWindow_sequenceStart() {
        let tokens: [Int32] = [10, 11, 12]
        #expect(PLERef.contextWindow(tokens: tokens, position: 0,
                                     ngramSize: 3, eos: 99) == [10, 99, 99])
        #expect(PLERef.contextWindow(tokens: tokens, position: 1,
                                     ngramSize: 3, eos: 99) == [11, 10, 99])
        #expect(PLERef.contextWindow(tokens: tokens, position: 2,
                                     ngramSize: 3, eos: 99) == [12, 11, 10])
    }

    // MARK: - Hash

    @Test("head order and rows are hand-checkable")
    func rowIndices_handChecked() {
        // Two heads per gram: heads 0,1 are the bigram, heads 2,3 the trigram.
        // bigram  mixed = 2·3 ^ 4·5         = 6 ^ 20            = 18
        // trigram mixed = 18 ^ 6·7          = 18 ^ 42           = 56
        // rows: 18 % 10 + 0, 18 % 20 + 100, 56 % 30 + 200, 56 % 40 + 300
        let rows = PLERef.rowIndices(
            context: [2, 4, 6],
            multipliers: [3, 5, 7],
            vocabSizes: [10, 20, 30, 40],
            offsets: [0, 100, 200, 300],
            headsPerNGram: 2)
        #expect(rows == [8, 118, 226, 316], "got \(rows)")
    }

    @Test("the multiply wraps at 64 bits, and a wider type would differ")
    func rowIndices_unsignedWrap() {
        // 2 · (2^64 − 1) wraps to 2^64 − 2. Modulo 3: the wrapped value is
        // ≡ 2 (2^64 ≡ 1 mod 3), while the 65-bit product 2^65 − 2 ≡ 0 — so
        // this case *fails* if the wrap is not unsigned 64-bit.
        let rows = PLERef.rowIndices(
            context: [2, 0],
            multipliers: [UInt64.max, 1],
            vocabSizes: [3, 3],
            offsets: [0, 0],
            headsPerNGram: 2)
        #expect(rows == [2, 2], "got \(rows)")
    }

    @Test("a multiplier above 2^63 keeps its bit pattern")
    func rowIndices_highBitMultiplier() {
        // (1 << 63) as a signed int64 is negative; reading it back as a
        // magnitude would give a different row. ctx 1 × 2^63 = 2^63; the
        // low bit is 0, so modulo 2 is 0 — but a sign-extended reading
        // would produce (−2^63) % 2 == 0 too, so pin the value explicitly.
        let m0 = UInt64(1) << 63
        let rows = PLERef.rowIndices(
            context: [3, 0],
            multipliers: [m0, 0],
            vocabSizes: [1024, 1024],
            offsets: [5, 5],
            headsPerNGram: 2)
        // 3 · 2^63 wraps to 2^63 + 2^64 ≡ 2^63; 2^63 % 1024 == 0.
        #expect(rows == [5, 5], "got \(rows)")
    }

    @Test("hash matches an independent wrap-explicit transcription")
    func rowIndices_matchesNaive() {
        var rng = SeedTree(0xE01).key("ple-hash")
        for trial in 0..<24 {
            let nGram = 2 + trial % 2
            let perGram = [1, 2, 4][trial % 3]
            let nHeads = (nGram - 1) * perGram
            let ctx = (0..<nGram).map { _ in Int32(rng.next() % 90_000) }
            // Multipliers exercise the full 64-bit range, high bit included.
            let mult = (0..<nGram).map { _ in rng.next() }
            let vocab = (0..<nHeads).map { _ in 7 + rng.next() % 4_000_000 }
            let offsets = (0..<nHeads).map { _ in rng.next() % 100_000 }

            let got = PLERef.rowIndices(context: ctx, multipliers: mult,
                                        vocabSizes: vocab, offsets: offsets,
                                        headsPerNGram: perGram)
            let want = Self.naiveRowIndices(context: ctx, multipliers: mult,
                                            vocabSizes: vocab, offsets: offsets,
                                            headsPerNGram: perGram)
            #expect(got == want, "trial \(trial): \(got) vs \(want)")
        }
    }

    /// Independent transcription: grams iterated largest-first, each mixed
    /// product folded with explicit overflow-reporting calls instead of `&*`,
    /// and the head map built by walking heads in reverse.
    private static func naiveRowIndices(
        context: [Int32], multipliers: [UInt64], vocabSizes: [UInt64],
        offsets: [UInt64], headsPerNGram: Int
    ) -> [Int] {
        let nGram = context.count
        let nHeads = (nGram - 1) * headsPerNGram
        var rows = [Int](repeating: 0, count: nHeads)
        for n in stride(from: nGram, through: 2, by: -1) {
            var mixed: UInt64 = wrapMul(UInt64(context[0]), multipliers[0])
            for j in 1..<n {
                let prod = wrapMul(UInt64(context[j]), multipliers[j])
                mixed = mixed ^ prod
            }
            for g in stride(from: headsPerNGram - 1, through: 0, by: -1) {
                let h = (n - 2) * headsPerNGram + g
                let (sum, _) = (mixed % vocabSizes[h]).addingReportingOverflow(offsets[h])
                rows[h] = Int(Int32(truncatingIfNeeded: sum))
            }
        }
        return rows
    }

    /// The low word of the full 128-bit product — the wrap made explicit.
    private static func wrapMul(_ a: UInt64, _ b: UInt64) -> UInt64 {
        a.multipliedFullWidth(by: b).low
    }

    // MARK: - Forward

    @Test("conv taps are the dilated ones: history row r feeds tap (3 − r/3)")
    func forward_dilatedTaps() {
        // The dilation is the n-gram size, so with kernel 4 only history rows
        // 0, 3 and 6 are inside the receptive field (t−9, t−6, t−3); rows
        // 1,2,4,5,7,8 feed no tap at all. A dilation of 1 would light up
        // every row — that is what this case is built to catch.
        let streamCount = 2, nEmbd = 4, hcDim = streamCount * nEmbd
        let kernel = 4, dilation = 3
        let history = (kernel - 1) * dilation            // 9
        let embDim = 3

        var rng = SeedTree(0xE11).key("ple-taps")
        let emb = (0..<embDim).map { _ in rng.uniform(-1, 1) }
        let plane = (0..<hcDim).map { _ in rng.uniform(-1, 1) }
        let keyProj = (0..<hcDim * embDim).map { _ in rng.uniform(-1, 1) }
        let valueProj = (0..<nEmbd * embDim).map { _ in rng.uniform(-1, 1) }
        let normKey = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let normQuery = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let normConv = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let convWeight = (0..<hcDim * kernel).map { _ in rng.uniform(-1, 1) }

        func run(historyRows: [Float]) -> [Float] {
            PLERef.forward(embedding: emb, plane: plane,
                           keyProj: keyProj, valueProj: valueProj,
                           normKey: normKey, normQuery: normQuery,
                           normConv: normConv, convWeight: convWeight,
                           convHistory: historyRows,
                           streamCount: streamCount,
                           convKernel: kernel, dilation: dilation).plane
        }

        let zeroHistory = [Float](repeating: 0, count: history * hcDim)
        let baseline = run(historyRows: zeroHistory)

        for row in 0..<history {
            // A single nonzero history row, at tap distance (history − row).
            var h = zeroHistory
            let back = history - row                     // 9 − row
            for c in 0..<hcDim { h[row * hcDim + c] = Float(c + 1) }
            let out = run(historyRows: h)
            let moved = zip(out, baseline).contains { abs($0 - $1) > 1e-6 }
            let inField = back % dilation == 0 && back < kernel * dilation
            #expect(moved == inField,
                    "history row \(row) (t−\(back)) moved=\(moved), expected \(inField)")
        }
    }

    @Test("conv history rolls one row per token and matches a direct conv")
    func forward_streamingStateMatchesDirect() {
        // Feeding tokens one at a time must equal computing each token's conv
        // straight from the normalized values that preceded it: the state is
        // exactly the receptive field, so nothing older can matter.
        let streamCount = 2, nEmbd = 4, hcDim = streamCount * nEmbd
        let kernel = 4, dilation = 3
        let embDim = 3
        let tokens = 12

        var rng = SeedTree(0xE21).key("ple-stream")
        let keyProj = (0..<hcDim * embDim).map { _ in rng.uniform(-1, 1) }
        let valueProj = (0..<nEmbd * embDim).map { _ in rng.uniform(-1, 1) }
        let normKey = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let normQuery = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let normConv = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let convWeight = (0..<hcDim * kernel).map { _ in rng.uniform(-1, 1) }

        let embs = (0..<tokens).map { _ in (0..<embDim).map { _ in rng.uniform(-1, 1) } }
        let planes = (0..<tokens).map { _ in (0..<hcDim).map { _ in rng.uniform(-1, 1) } }
        let zeroConv = [Float](repeating: 0, count: hcDim * kernel)
        let histRows = (kernel - 1) * dilation

        /// One streaming pass; returns each token's plane and the `normalized`
        /// row of that token, which is what a zero conv weight leaves at the
        /// tail of the state.
        func run(convWeight w: [Float]) -> (planes: [[Float]], normalized: [[Float]]) {
            var state = [Float]()
            var outs: [[Float]] = []
            var norms: [[Float]] = []
            for t in 0..<tokens {
                let out = PLERef.forward(embedding: embs[t], plane: planes[t],
                                         keyProj: keyProj, valueProj: valueProj,
                                         normKey: normKey, normQuery: normQuery,
                                         normConv: normConv, convWeight: w,
                                         convHistory: state,
                                         streamCount: streamCount,
                                         convKernel: kernel, dilation: dilation)
                state = out.history
                outs.append(out.plane)
                norms.append(Array(state[((histRows - 1) * hcDim)...]))
            }
            return (outs, norms)
        }

        let withConv = run(convWeight: convWeight)
        let noConv = run(convWeight: zeroConv)
        // The two runs must agree on `normalized` — the conv weights scale the
        // taps, not the input they read.
        for t in 0..<tokens {
            #expect(RelError.compute(actual: withConv.normalized[t],
                                     reference: noConv.normalized[t]) < Tolerance.identity,
                    "normalized diverged at t=\(t) between the two runs")
        }

        // Direct: tap k of token t reads t − (kernel−1−k)·dilation, and reads
        // zero before the sequence start.
        for t in 0..<tokens {
            for c in 0..<hcDim {
                var acc: Float = 0
                for k in 0..<kernel {
                    let src = t - (kernel - 1 - k) * dilation
                    let tap = src >= 0 ? withConv.normalized[src][c] : 0
                    acc += convWeight[c * kernel + k] * tap
                }
                let want = noConv.planes[t][c] + silu(acc)
                #expect(abs(withConv.planes[t][c] - want) < 1e-5,
                        "t=\(t) c=\(c): \(withConv.planes[t][c]) vs \(want)")
            }
        }
    }

    @Test("a zero key/query dot gives gate exactly 0.5")
    func forward_zeroDotGate() {
        let streamCount = 2, nEmbd = 4, hcDim = streamCount * nEmbd
        let embDim = 3, kernel = 4, dilation = 3
        var rng = SeedTree(0xE31).key("ple-gate")
        // key_proj = 0 ⇒ key = 0 ⇒ the grouped norm of 0 is 0 (the RMS
        // divisor is eps-floored, not zero) ⇒ every per-stream dot is 0 ⇒
        // sgn(0)·√1e-6 = 0 ⇒ gate = sigmoid(0) = 0.5.
        let keyProj = [Float](repeating: 0, count: hcDim * embDim)
        let valueProj = (0..<nEmbd * embDim).map { _ in rng.uniform(-1, 1) }
        let emb = (0..<embDim).map { _ in rng.uniform(-1, 1) }
        let plane = (0..<hcDim).map { _ in rng.uniform(-1, 1) }
        let ones = [Float](repeating: 1, count: hcDim)

        let out = PLERef.forward(embedding: emb, plane: plane,
                                 keyProj: keyProj, valueProj: valueProj,
                                 normKey: ones, normQuery: ones, normConv: ones,
                                 convWeight: [Float](repeating: 0, count: hcDim * kernel),
                                 convHistory: [],
                                 streamCount: streamCount,
                                 convKernel: kernel, dilation: dilation)
        // value[:, d]·0.5 per stream, plus the plane (conv weight is 0).
        var value = [Float](repeating: 0, count: nEmbd)
        for d in 0..<nEmbd {
            var acc: Float = 0
            for i in 0..<embDim { acc += valueProj[d * embDim + i] * emb[i] }
            value[d] = acc
        }
        for c in 0..<streamCount {
            for d in 0..<nEmbd {
                let want = plane[c * nEmbd + d] + value[d] * 0.5
                #expect(abs(out.plane[c * nEmbd + d] - want) < 1e-6,
                        "stream \(c) dim \(d)")
            }
        }
    }
}

private func silu(_ x: Float) -> Float {
    x / (1.0 + exp(-x))
}

import Testing
import Foundation
@testable import FinchMoE
import FinchMoEValidationSupport

/// Cross-validates `PLEHost` — the host-side PLE n-gram routing — against
/// `PLERef`, which was locked to `qwen4exp.cpp` in M3.3a.
///
/// The two disagree about *how the predecessors are stored*, which is the
/// point: the reference walks a full token array, the host walks a two-entry
/// positional ring sized to the window. They have to agree on every window,
/// including at the sequence start where the ring is short and the array is
/// not. The hash itself is exact integer work — a wrong index reads a
/// different row — so those cases compare exactly, with no tolerance.
@Suite struct PLEHostTests {

    // MARK: - Fixtures

    private static let eos: Int32 = 99

    /// A toy 3.8 config: 3-gram, 2 heads per gram (4 heads), 4-wide rows,
    /// 2 parts of 8 rows = a 16-row table. Small enough that a test can write
    /// the whole table by hand and know which row each index names.
    private static func toyConfig(ngramSize: Int = 3,
                                  headsPerNgram: Int = 2,
                                  rowDim: Int = 4,
                                  partCount: Int = 2,
                                  partRows: Int = 8,
                                  eosTokenId: Int = Int(eos)) -> ArchConfig {
        let b = ArchConfig.qwen3_8_flashNext_125B
        return ArchConfig(
            hiddenSize: b.hiddenSize,
            intermediateSize: b.intermediateSize,
            moeIntermediateSize: b.moeIntermediateSize,
            numHeads: b.numHeads,
            numKVHeads: b.numKVHeads,
            numFullKVHeads: b.numFullKVHeads,
            headDim: b.headDim,
            fullHeadDim: b.fullHeadDim,
            vocabSize: b.vocabSize,
            slidingWindow: b.slidingWindow,
            finalLogitSoftcap: b.finalLogitSoftcap,
            ropeTheta: b.ropeTheta,
            fullRopeTheta: b.fullRopeTheta,
            partialRotaryFactor: b.partialRotaryFactor,
            numLayers: b.numLayers,
            numExperts: b.numExperts,
            topKExperts: b.topKExperts,
            tieWordEmbeddings: b.tieWordEmbeddings,
            attentionKEqV: b.attentionKEqV,
            fullAttentionLayerMask: b.fullAttentionLayerMask,
            hiddenActivation: b.hiddenActivation,
            modelFamily: b.modelFamily,
            attnOutputGate: b.attnOutputGate,
            linearNumKeyHeads: b.linearNumKeyHeads,
            linearNumValueHeads: b.linearNumValueHeads,
            linearKeyHeadDim: b.linearKeyHeadDim,
            linearValueHeadDim: b.linearValueHeadDim,
            linearConvKernelDim: b.linearConvKernelDim,
            hyperConnectionCount: b.hyperConnectionCount,
            hyperConnectionLowrank: b.hyperConnectionLowrank,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            ngramRowDim: rowDim,
            ngramPartCount: partCount,
            ngramPartRows: partRows,
            pleLayerIndexes: [1],
            pleConvKernelSize: 4,
            pleEosTokenId: eosTokenId)
    }

    /// Small vocabularies so the modulo is exercised but a hand check stays
    /// possible; offsets place heads 2 and 3 in the second part file, which is
    /// what the gather test needs to cross a part boundary.
    private static func toyConstants(heads: Int = 4)
        -> (multipliers: [UInt64], offsets: [UInt64], vocabs: [UInt64]) {
        (multipliers: [3, 5, 7],
         offsets: [0, 0, 8, 8],
         vocabs: Array(repeating: UInt64(8), count: heads))
    }

    private static func makeHost(config: ArchConfig? = nil) -> PLEHost {
        let c = toyConstants()
        guard let host = PLEHost(config: config ?? toyConfig(),
                                 multipliers: c.multipliers,
                                 headOffsets: c.offsets,
                                 headVocabSizes: c.vocabs) else {
            preconditionFailure("toy config must build a PLE host")
        }
        return host
    }

    // MARK: - Window and hash vs the reference

    @Test("the ring reproduces the reference window over a whole sequence")
    func windowMatchesReference() {
        // Tokens chosen to hit all four cases: a clean window, a token whose
        // own value is EOS (which must NOT cut itself), an EOS that the next
        // two tokens have to see as sticky, and deep history behind it.
        let tokens: [Int32] = [10, 11, 12, Self.eos, 13, 14, 15, 16]
        let host = Self.makeHost()
        for p in 0..<tokens.count {
            host.record(position: p, token: tokens[p])
            let got = host.contextWindow(atPosition: p)
            let want = PLERef.contextWindow(tokens: tokens, position: p,
                                            ngramSize: 3, eos: Self.eos)
            #expect(got == want, "position \(p): \(got) vs \(want)")
        }
    }

    @Test("row indices match the reference exactly")
    func rowIndicesMatchReference() {
        let tokens: [Int32] = [7, Self.eos, 3, 21, 4, 5, 6, 7, 8, 9]
        let c = Self.toyConstants()
        let host = Self.makeHost()
        for p in 0..<tokens.count {
            host.record(position: p, token: tokens[p])
            let ctx = PLERef.contextWindow(tokens: tokens, position: p,
                                           ngramSize: 3, eos: Self.eos)
            let want = PLERef.rowIndices(context: ctx,
                                         multipliers: c.multipliers,
                                         vocabSizes: c.vocabs,
                                         offsets: c.offsets,
                                         headsPerNGram: 2)
            #expect(host.rowIndices(atPosition: p) == want,
                    "position \(p) (ctx \(ctx)): \(host.rowIndices(atPosition: p)) vs \(want)")
        }
    }

    @Test("the ring stays bounded however long the sequence runs")
    func windowStaysBounded() {
        // The 16 GB budget rides on this: the PLE host may not accumulate a
        // token history. 10k tokens in, the ring is still `ngramSize` deep.
        let host = Self.makeHost()
        for p in 0..<10_000 { host.record(position: p, token: Int32(p % 997)) }
        #expect(host.windowDepth == 3)
        #expect(host.contextWindow(atPosition: 9_999)
                == [Int32(9_999 % 997), Int32(9_998 % 997), Int32(9_997 % 997)])
    }

    @Test("a sequence start reads as EOS without any predecessor recorded")
    func sequenceStart() {
        let host = Self.makeHost()
        host.record(position: 0, token: 42)
        #expect(host.contextWindow(atPosition: 0) == [42, Self.eos, Self.eos])
        host.record(position: 1, token: 43)
        #expect(host.contextWindow(atPosition: 1) == [43, 42, Self.eos])
    }

    @Test("reset drops the ring with the KV cache")
    func resetClears() {
        let host = Self.makeHost()
        for p in 0..<4 { host.record(position: p, token: Int32(10 + p)) }
        host.reset()
        host.record(position: 0, token: 5)
        #expect(host.contextWindow(atPosition: 0) == [5, Self.eos, Self.eos],
                "the first token of a new sequence must not see the old one")
    }

    // MARK: - Row placement

    @Test("rows split into parts by divmod, and the offset region is honoured")
    func locationSplitsByPart() {
        let host = Self.makeHost()          // 2 parts × 8 rows
        #expect(host.geometry.totalRows == 16)
        for (row, want) in [(0, (0, 0)), (7, (0, 7)), (8, (1, 0)), (15, (1, 7))] {
            let got = host.location(ofRow: row)
            #expect(got == want, "row \(row): \(got) vs \(want)")
        }
    }

    // MARK: - Gather

    /// Writes a table whose every row is `[row, row+1, row+2, row+3]` as BF16,
    /// so a gathered vector says exactly which rows it read and in what order.
    private static func writeToyTable(rowDim: Int, partCount: Int, partRows: Int,
                                      in directory: URL) throws {
        for part in 0..<partCount {
            var bytes = Data()
            for r in 0..<partRows {
                let row = part * partRows + r
                for d in 0..<rowDim {
                    let bits = Quantization.bf16Bits(Float(row + d))
                    bytes.append(UInt8(bits & 0xFF))
                    bytes.append(UInt8(bits >> 8))
                }
            }
            let name = String(format: "shard_%03d.bin", part)
            try bytes.write(to: directory.appendingPathComponent(name))
        }
    }

    @Test("the gather reads the right row from the right part, in head order")
    func gatherReadsHeadMajor() throws {
        let rowDim = 4, partCount = 2, partRows = 8
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("plehost-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeToyTable(rowDim: rowDim, partCount: partCount,
                               partRows: partRows, in: dir)

        let config = Self.toyConfig(rowDim: rowDim, partCount: partCount, partRows: partRows)
        let host = Self.makeHost(config: config)
        host.record(position: 0, token: 3)
        let rows = host.rowIndices(atPosition: 0)

        let gathered = try host.gather(atPosition: 0) { part in
            let fd = Darwin.open(dir.appendingPathComponent(
                String(format: "shard_%03d.bin", part)).path, O_RDONLY)
            guard fd >= 0 else { throw CocoaError(.fileNoSuchFile) }
            defer { close(fd) }
            return try PLEPartStreamer(partIndex: part, rows: partRows,
                                       columns: rowDim, fileDescriptor: fd)
        }

        #expect(gathered.count == rows.count * rowDim)
        for (h, row) in rows.enumerated() {
            for d in 0..<rowDim {
                let want = Float16(Float(row + d))
                let got = gathered[h * rowDim + d]
                #expect(got == want,
                        "head \(h) (row \(row), part \(row / partRows)) dim \(d): \(got) vs \(want)")
            }
        }
    }
}

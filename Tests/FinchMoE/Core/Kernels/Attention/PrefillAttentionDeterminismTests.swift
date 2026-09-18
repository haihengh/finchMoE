import Foundation
import Metal
import Testing

@testable import FinchMoE

/// Does the dense prefill attention path repeat bit-exactly?
///
/// The row-fingerprint map localized the engine's long-prompt nondeterminism to
/// **one row of one layer's attention output**, with the layer's queries, keys,
/// values and QSA selection all bit-identical at that layer. If that reading
/// holds, the kernel is nondeterministic on fixed inputs, and the way to find it
/// is to run the same dispatch many times and compare — a single occurrence per
/// ~350 s of prefill is far too rare to catch by reading or by re-running
/// prefills.
///
/// The shape is the shipping one for the 125B's full layers (head_dim 256, 24
/// query heads, 2 KV heads) and the lengths straddle the boundaries the map
/// pointed at: 1024, 2048, and the rows either side.
///
/// **These tests are a fuzz, not a proof.** A pass bounds the rate at the number
/// of dispatches run; it does not clear the kernel. `FQ_ATTN_FUZZ_ROUNDS` raises
/// the count for a longer hunt.
@Suite struct PrefillAttentionDeterminismTests {
    static let headDim = 256
    static let qHeads = 24
    static let kvHeads = 2
    static var rounds: Int {
        ProcessInfo.processInfo.environment["FQ_ATTN_FUZZ_ROUNDS"]
            .flatMap(Int.init) ?? 200
    }
    /// Straddling both boundaries the maps landed on — 1024 (below the QSA
    /// selection width, so the dense path) and 2051 (above it, so the cells
    /// path) — plus neighbours as controls.
    static let lengths = [1023, 1024, 1025, 2049, 2050, 2051, 2052, 2064, 2065]

    static func fill(_ buf: MTLBuffer, count: Int, seed: UInt64) {
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
        var s = seed
        for i in 0..<count {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            // Small values: the softmax is what is under test, not overflow.
            let v = Float(Int((s >> 40) & 0xFF) - 128) / 512.0
            ptr[i] = Float16(v)
        }
    }

    /// The cells path: `encodeFullCells` over an ascending list of every
    /// position, which the encoder documents as reproducing `encodeFull`
    /// instruction for instruction. Same question, the other path — and the
    /// second map landed above the selection width, which is the cells path.
    @Test func cellsAttentionRepeatsBitExactly() throws {
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)
        let hd = Self.headDim, nq = Self.qHeads, nkv = Self.kvHeads
        let qDim = nq * hd
        let maxLen = (Self.lengths.max() ?? 2065) + 1

        let q = try #require(ctx.device.makeBuffer(length: qDim * 2,
                                                   options: .storageModeShared))
        let k = try #require(ctx.device.makeBuffer(length: maxLen * nkv * hd * 2,
                                                   options: .storageModeShared))
        let v = try #require(ctx.device.makeBuffer(length: maxLen * nkv * hd * 2,
                                                   options: .storageModeShared))
        let out = try #require(ctx.device.makeBuffer(length: qDim * 2,
                                                    options: .storageModeShared))
        let cells = try #require(ctx.device.makeBuffer(length: maxLen * 4,
                                                       options: .storageModeShared))
        Self.fill(q, count: qDim, seed: 11)
        Self.fill(k, count: maxLen * nkv * hd, seed: 22)
        Self.fill(v, count: maxLen * nkv * hd, seed: 33)
        let cellPtr = cells.contents().bindMemory(to: UInt32.self, capacity: maxLen)
        for i in 0..<maxLen { cellPtr[i] = UInt32(i) }

        func run(seqLen: Int) throws -> [UInt8] {
            memset(out.contents(), 0, qDim * 2)
            guard let cb = ctx.queue.makeCommandBuffer() else {
                Issue.record("no command buffer"); return []
            }
            attention.encodeFullCells(commandBuffer: cb,
                                      q: q, qOffset: 0, k: k, kOffset: 0,
                                      v: v, vOffset: 0, cells: cells, cellsOffset: 0,
                                      out: out, outOffset: 0,
                                      headDim: UInt32(hd), numQHeads: UInt32(nq),
                                      numKVHeads: UInt32(nkv), nCells: UInt32(seqLen))
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { Issue.record("kernel error: \(e)"); return [] }
            let p = out.contents().bindMemory(to: UInt8.self, capacity: qDim * 2)
            return Array(UnsafeBufferPointer(start: p, count: qDim * 2))
        }

        var mismatches: [String] = []
        for len in Self.lengths {
            let reference = try run(seqLen: len)
            for round in 1...Self.rounds {
                let got = try run(seqLen: len)
                if got != reference, mismatches.count < 8 {
                    let differing = zip(got, reference).filter { $0 != $1 }.count
                    mismatches.append("nCells \(len) round \(round): \(differing) bytes differ")
                }
            }
        }
        #expect(mismatches.isEmpty,
                "\(Self.rounds) rounds x \(Self.lengths.count) lengths: \(mismatches.joined(separator: "; "))")
    }

    @Test func denseAttentionRepeatsBitExactly() throws {
        let ctx = try MetalContext()
        let attention = try Attention(context: ctx)
        let hd = Self.headDim, nq = Self.qHeads, nkv = Self.kvHeads
        let qDim = nq * hd
        let maxLen = (Self.lengths.max() ?? 2049) + 1
        let kvCount = maxLen * nkv * hd

        let q = try #require(ctx.device.makeBuffer(length: qDim * 2,
                                                   options: .storageModeShared))
        let k = try #require(ctx.device.makeBuffer(length: kvCount * 2,
                                                   options: .storageModeShared))
        let v = try #require(ctx.device.makeBuffer(length: kvCount * 2,
                                                   options: .storageModeShared))
        let out = try #require(ctx.device.makeBuffer(length: qDim * 2,
                                                    options: .storageModeShared))
        Self.fill(q, count: qDim, seed: 11)
        Self.fill(k, count: kvCount, seed: 22)
        Self.fill(v, count: kvCount, seed: 33)

        func run(seqLen: Int) throws -> [UInt8] {
            memset(out.contents(), 0, qDim * 2)
            guard let cb = ctx.queue.makeCommandBuffer() else {
                Issue.record("no command buffer"); return []
            }
            attention.encodeFull(commandBuffer: cb,
                                 q: q, qOffset: 0, k: k, kOffset: 0,
                                 v: v, vOffset: 0, out: out, outOffset: 0,
                                 headDim: UInt32(hd), numQHeads: UInt32(nq),
                                 numKVHeads: UInt32(nkv), seqLen: UInt32(seqLen))
            cb.commit()
            cb.waitUntilCompleted()
            if let e = cb.error { Issue.record("kernel error: \(e)"); return [] }
            let p = out.contents().bindMemory(to: UInt8.self, capacity: qDim * 2)
            return Array(UnsafeBufferPointer(start: p, count: qDim * 2))
        }

        var mismatches: [String] = []
        for len in Self.lengths {
            let reference = try run(seqLen: len)
            for round in 1...Self.rounds {
                let got = try run(seqLen: len)
                if got != reference {
                    let differing = zip(got, reference).filter { $0 != $1 }.count
                    if mismatches.count < 8 {
                        mismatches.append("seqLen \(len) round \(round): \(differing) bytes differ")
                    } else { break }
                }
            }
        }
        #expect(mismatches.isEmpty,
                "\(Self.rounds) rounds x \(Self.lengths.count) lengths: \(mismatches.joined(separator: "; "))")
    }
}

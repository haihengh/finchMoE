import Foundation
import Metal
import Testing

@testable import FinchMoE

/// The row-fingerprint instrument is a diagnostic, so its failure mode is
/// silent: a hash that is wrong, or that quietly ignores part of a row, still
/// produces a file of plausible numbers and a diff that reports the wrong
/// answer. These tests pin it to a host implementation of the same algorithm.
@Suite struct PrefillRowHashTests {
    /// FNV-1a, 64-bit — the algorithm the kernel implements, written out here
    /// independently rather than shared with the engine under test.
    static func hostFNV1a(_ bytes: [UInt8]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for b in bytes { h = (h ^ UInt64(b)) &* 0x0000_0100_0000_01b3 }
        return h
    }

    /// Runs the kernel over `rows` (each `strideBytes` long, `rowBytes` of them
    /// meaningful) and returns the hashes written at `rowBase`, reading
    /// `rowCount` of them.
    static func runKernel(rows: [[UInt8]], rowBytes: Int, strideBytes: Int,
                          rowBase: Int, rowCount: Int) throws -> [UInt64] {
        let ctx = try MetalContext()
        let hash = try PrefillRowHash(context: ctx)
        let layout = RowHashLayout(maxRows: 64, layerCount: 1)
        guard let src = ctx.device.makeBuffer(
                length: max(rows.count * strideBytes, 8), options: .storageModeShared),
              let dst = ctx.device.makeBuffer(
                length: layout.totalBytes, options: .storageModeShared) else {
            Issue.record("buffer allocation failed")
            return []
        }
        // Lay the rows out at the requested stride, padding each with a marker
        // byte so a kernel that hashed the whole stride would be caught.
        let srcPtr = src.contents().bindMemory(to: UInt8.self,
                                               capacity: rows.count * strideBytes)
        for (i, row) in rows.enumerated() {
            for j in 0..<strideBytes { srcPtr[i * strideBytes + j] = 0xAB }
            for j in 0..<min(row.count, rowBytes) {
                srcPtr[i * strideBytes + j] = row[j]
            }
        }
        memset(dst.contents(), 0, layout.totalBytes)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("no command buffer"); return []
        }
        hash.encode(commandBuffer: cb, src: src, dst: dst,
                    dstOffsetBytes: layout.offsetBytes(layer: 0, stage: 0),
                    dstRowBase: rowBase,
                    rowCount: UInt32(rows.count),
                    rowStrideBytes: UInt32(strideBytes),
                    rowBytes: UInt32(rowBytes))
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { Issue.record("kernel error: \(error)"); return [] }

        let dstPtr = dst.contents().bindMemory(to: UInt64.self,
                                               capacity: layout.totalBytes / 8)
        let base = layout.elementIndex(layer: 0, stage: 0, row: rowBase)
        return (0..<rowCount).map { dstPtr[base + $0] }
    }

    @Test func hashesMatchAHandComputedFNV1a() throws {
        let rows: [[UInt8]] = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [255, 254, 253, 252, 251, 250, 249, 248],
        ]
        let got = try Self.runKernel(rows: rows, rowBytes: 8, strideBytes: 8,
                                     rowBase: 0, rowCount: 3)
        let want = rows.map { Self.hostFNV1a($0) }
        #expect(got == want, "kernel \(got) vs host \(want)")
    }

    /// The row is `rowBytes` long inside a wider stride. The padding is written
    /// with a sentinel, so a kernel that hashed the full stride — which would
    /// make every hash depend on whatever the plane happens to hold beyond the
    /// tensor, and turn the instrument into a second source of noise — reports a
    /// different value here.
    @Test func paddingBeyondTheRowIsNotHashed() throws {
        let rows: [[UInt8]] = [[9, 8, 7, 6], [1, 1, 1, 1]]
        let got = try Self.runKernel(rows: rows, rowBytes: 4, strideBytes: 32,
                                     rowBase: 0, rowCount: 2)
        let want = rows.map { Self.hostFNV1a($0) }
        #expect(got == want, "kernel \(got) vs host \(want)")
    }

    /// `dstRowBase` is what keeps a multi-chunk prefill from erasing itself:
    /// each chunk writes its rows at their position, not at the start of the
    /// stage region.
    @Test func rowsLandAtTheirSequencePosition() throws {
        let rows: [[UInt8]] = [[42, 42, 42, 42]]
        let got = try Self.runKernel(rows: rows, rowBytes: 4, strideBytes: 4,
                                     rowBase: 1000, rowCount: 1)
        #expect(got == [Self.hostFNV1a(rows[0])])
    }

    /// Identical rows hash identically and differing rows do not — the property
    /// the whole diff rests on. A single flipped bit anywhere in the row must
    /// change the hash.
    @Test func oneFlippedBitChangesTheHash() throws {
        var a = [UInt8](repeating: 0x5A, count: 64)
        var b = a
        b[37] ^= 0x01
        let got = try Self.runKernel(rows: [a, b, a], rowBytes: 64, strideBytes: 64,
                                     rowBase: 0, rowCount: 3)
        #expect(got[0] == got[2], "identical rows must hash identically")
        #expect(got[0] != got[1], "one bit is enough to change the hash")
        a[63] ^= 0x80
        let got2 = try Self.runKernel(rows: [a], rowBytes: 64, strideBytes: 64,
                                      rowBase: 0, rowCount: 1)
        #expect(got2[0] != got[0], "the last byte must be covered")
    }

    /// The kernel guards `gid >= rowCount`, so a dispatch rounded up to whole
    /// threadgroups must leave the region past the rows untouched.
    @Test func rowsPastTheCountAreLeftAlone() throws {
        let rows = [[UInt8]](repeating: [1, 2, 3, 4], count: 3)
        let got = try Self.runKernel(rows: rows, rowBytes: 4, strideBytes: 4,
                                     rowBase: 0, rowCount: 3)
        #expect(got.count == 3)
        let sentinel = Self.hostFNV1a([0, 0, 0, 0])
        #expect(!got.contains(sentinel), "a row past rowCount was written")
    }
}

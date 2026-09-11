import Testing
import Foundation
import Darwin
import Metal
@testable import FinchMoE

/// Unit tests for the synchronous `pread` backend: round-trip correctness,
/// exact file-byte reads, the short-read failure path,
/// and round-robin slot reuse. No real Gemma weights — a synthetic layer file
/// of tagged expert blobs.
@Suite struct PreadExpertStreamerTests {

    static let pageSize = Int(getpagesize())

    /// One expert blob = 2 pages, every byte tagged with `tagByte(expert)` so
    /// a readback uniquely identifies which expert landed in a slot.
    static let expertStride = 2 * pageSize
    static let numExperts = 4
    /// A non-zero stream offset exercises `streamOffset` in the file-offset math.
    static let headerPages = 1
    static var streamOffset: UInt64 { UInt64(headerPages * pageSize) }
    static var streamSize: UInt64 { UInt64(numExperts * expertStride) }

    static func tagByte(_ expert: Int) -> UInt8 { UInt8(0xA0 + expert) }

    /// Write a synthetic layer file: `headerPages` of zeros, then `numExperts`
    /// blobs each filled with their tag byte. Returns the file URL.
    static func writeSyntheticLayer() throws -> URL {
        let total = Int(streamOffset) + Int(streamSize)
        var bytes = [UInt8](repeating: 0, count: total)
        for e in 0..<numExperts {
            let start = Int(streamOffset) + e * expertStride
            for i in 0..<expertStride { bytes[start + i] = tagByte(e) }
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pread-streamer-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return url
    }

    /// The byte a file at absolute offset `i` carries in the offset-tagged
    /// fixture. Any position-dependence will do; this one is cheap and spreads
    /// the values over the full byte range.
    static func patternByte(_ i: Int) -> UInt8 {
        UInt8(truncatingIfNeeded: i &* 31 &+ 7)
    }

    /// Write a layer where every byte depends on its absolute file offset.
    ///
    /// The tag fixture above cannot see a tiling bug in a chunked read: every
    /// byte of an expert is the same byte, so a skipped or repeated chunk still
    /// reads as valid data. This one makes a gap read as a *wrong* byte, which
    /// is the only way an assertion can catch the chunk arithmetic being off by
    /// one.
    static func writeOffsetTaggedLayer() throws -> URL {
        let total = Int(streamOffset) + Int(streamSize)
        var bytes = [UInt8](repeating: 0, count: total)
        for i in 0..<total { bytes[i] = patternByte(i) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pread-streamer-pattern-\(UUID().uuidString).bin")
        try Data(bytes).write(to: url)
        return url
    }

    static func makeLayout(path: String) -> StreamLayout {

        StreamLayout(path: path,
                     streamOffset: streamOffset,
                     streamSize: streamSize,
                     expertsPerLayer: numExperts,
                     expertStride: UInt64(expertStride))
    }

    static func bytes(of buffer: MTLBuffer, offset: UInt64, count: Int) -> [UInt8] {
        let base = buffer.contents().advanced(by: Int(offset))
        return [UInt8](UnsafeRawBufferPointer(start: base, count: count))
    }

}

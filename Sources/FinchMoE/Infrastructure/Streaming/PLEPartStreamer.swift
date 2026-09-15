import Foundation
import Darwin

/// One PLE n-gram part file (`ple_shards/shard_%03d.bin`): row-major
/// `[rows, columns]`, opened lazily by `Model.openPLEPart(_:)` and shared for
/// the model lifetime — the per-part analog of a routed-expert layer streamer,
/// minus the GPU slot cache (the M3.3 gather decides row caching once the host
/// hash layout is in; this type only guarantees verified, bounds-checked row
/// access).
///
/// A part is either **raw BF16** (`Layout.rawBF16`, the legacy 162 GiB install
/// and every test fixture) or the **int4 affine** layout the PLE-quant writer
/// produces (`Layout.quantized(groupSize:)`), which is
/// `[packed nibbles][scale BF16 × nGroups][bias BF16 × nGroups]` at a fixed
/// stride. The caller passes the layout from the manifest's `pleNgram` slot, so
/// the streamer never guesses the on-disk stride.
///
/// Mirrors `PreadExpertStreamer`'s fd discipline: the caller's file
/// descriptor is duplicated with `F_DUPFD_CLOEXEC` at init, so the opener
/// may `close` its own copy immediately; this instance owns the dup and
/// closes it on deinit. `pread` is thread-safe on the shared fd, so
/// concurrent row reads from different callers are fine.
public final class PLEPartStreamer: @unchecked Sendable {
    /// How the rows are stored on disk. `rawBF16` rows are `columns` BF16
    /// (stride `columns × 2`); `quantized` rows are the int4 affine layout
    /// whose stride depends on the group size, so the group size is carried
    /// here rather than derived from `columns`.
    public enum Layout: Sendable, Equatable {
        case rawBF16
        case quantized(groupSize: Int)
    }

    public let partIndex: Int
    /// Row count of this part file (`ngramPartRows`; 2,500,012 on the real
    /// snapshot, small on the synthetic toys).
    public let rows: Int
    /// Columns per row (`ngramRowDim`, 160 on this snapshot).
    public let columns: Int
    public let layout: Layout
    /// Bytes per row, fixed for the file's lifetime: `columns × 2` for raw BF16,
    /// `columns/2 + 2 · nGroups · 2` for the quantized layout.
    public var byteStride: Int {
        switch layout {
        case .rawBF16:
            return columns * MemoryLayout<UInt16>.size
        case .quantized(let groupSize):
            let nGroups = columns / groupSize
            return (columns >> 1) + nGroups * MemoryLayout<UInt16>.size * 2
        }
    }
    public let sizeBytes: UInt64

    private let fileDescriptor: Int32

    public init(partIndex: Int,
                rows: Int,
                columns: Int,
                layout: Layout = .rawBF16,
                fileDescriptor: Int32) throws {
        precondition(rows > 0 && columns > 0, "PLE part geometry must be positive")
        let openedFD = fcntl(fileDescriptor, F_DUPFD_CLOEXEC, 0)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(
                path: "ple_shards/shard_\(String(format: "%03d", partIndex)).bin",
                errno: errno)
        }
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: "ple_shards/shard_\(String(format: "%03d", partIndex)).bin",
                errno: errno == 0 ? EINVAL : errno)
        }
        // Resolved before `self` is written: `byteStride` is a computed property
        // that reads `layout` and `columns`, so it cannot be called from inside
        // `init` before those stored properties are initialized.
        let byteStride: Int
        switch layout {
        case .rawBF16:
            byteStride = columns * MemoryLayout<UInt16>.size
        case .quantized(let groupSize):
            let nGroups = columns / groupSize
            byteStride = (columns >> 1) + nGroups * MemoryLayout<UInt16>.size * 2
        }
        let expected = UInt64(rows) * UInt64(byteStride)
        guard UInt64(fileStats.st_size) == expected else {
            throw StreamerError.sizeMismatch(expected: expected,
                                             actual: UInt64(fileStats.st_size))
        }

        self.partIndex = partIndex
        self.rows = rows
        self.columns = columns
        self.layout = layout
        self.sizeBytes = expected
        self.fileDescriptor = openedFD
        closeFDOnFailure = false
    }

    deinit {
        close(fileDescriptor)
    }

    /// Row bytes for `rows[lower, upper)`, bounds-checked. A straight
    /// byte-range `pread` keyed off `byteStride`, identical in shape whether the
    /// rows are raw BF16 or the int4 affine layout — decoding is the reader's
    /// job, not this streamer's.
    public func readRows(_ range: Range<Int>) throws -> Data {
        guard range.lowerBound >= 0,
              range.upperBound <= rows,
              range.lowerBound <= range.upperBound else {
            throw StreamerError.invalidRowRange(
                part: partIndex,
                lower: range.lowerBound,
                upper: range.upperBound,
                rows: rows)
        }
        guard !range.isEmpty else { return Data() }
        let length = (range.upperBound - range.lowerBound) * byteStride
        var data = Data(count: length)
        var failureErrno: Int32 = 0
        let read = data.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) -> Int in
            var base = buffer.baseAddress!
            var remaining = length
            var offset = off_t(range.lowerBound) * off_t(byteStride)
            while remaining > 0 {
                let n = pread(fileDescriptor, base, remaining, offset)
                if n < 0 {
                    if errno == EINTR { continue }
                    failureErrno = errno
                    return -1
                }
                if n == 0 { break }
                base = base.advanced(by: n)
                remaining -= n
                offset += off_t(n)
            }
            return length - remaining
        }
        guard read == length else {
            throw StreamerError.preadFailed(errno: failureErrno == 0 ? EIO : failureErrno)
        }
        return data
    }
}

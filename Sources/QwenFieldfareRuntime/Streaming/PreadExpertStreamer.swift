import Foundation
import QwenFieldfareFormat

/// Streams routed expert blobs from per-layer packed files on SSD using
/// thread-safe `pread`. Maintains a bounded LFU cache of decoded blobs and
/// issues SSD prefetch hints (`F_RDADVISE` / `posix_fadvise` analogue) so the
/// next layer's experts are warm in the OS page cache while the GPU computes
/// the current layer.
///
/// Blobs are returned as raw buffer pointers into cache-owned memory that stays
/// valid until evicted. Callers must copy or use the pointer before the slot is
/// reused; in practice the runtime consumes the blob immediately (uploads to a
/// Metal buffer) before requesting further experts than the cache can hold.
public final class PreadExpertStreamer: @unchecked Sendable {

    public enum StreamError: Error, CustomStringConvertible {
        case cannotOpen(Int, String)
        case readFailed(layer: Int, expert: Int, errno: Int32)
        case shortRead(layer: Int, expert: Int, want: Int, got: Int)

        public var description: String {
            switch self {
            case .cannotOpen(let l, let p): return "Streamer: cannot open layer \(l) file \(p)"
            case .readFailed(let l, let e, let en): return "Streamer: pread failed layer \(l) expert \(e) errno \(en)"
            case .shortRead(let l, let e, let w, let g): return "Streamer: short read layer \(l) expert \(e) want \(w) got \(g)"
            }
        }
    }

    /// A cached expert blob (owns its backing memory).
    private final class CacheEntry {
        let key: UInt64          // (layer << 32) | expert
        let pointer: UnsafeMutableRawPointer
        let length: Int
        var frequency: Int       // LFU counter
        init(key: UInt64, pointer: UnsafeMutableRawPointer, length: Int) {
            self.key = key
            self.pointer = pointer
            self.length = length
            self.frequency = 1
        }
        deinit { pointer.deallocate() }
    }

    public let modelDir: URL
    public let layout: QTurboExpertLayout
    public let numLayers: Int
    public let capacity: Int

    private var fds: [Int32]                       // per-layer file descriptors
    private var cache: [UInt64: CacheEntry] = [:]
    private let lock = NSLock()
    private let prefetchQueue = DispatchQueue(label: "qwen.expert.prefetch", attributes: .concurrent)

    public init(modelDir: URL,
                layout: QTurboExpertLayout,
                numLayers: Int,
                capacity: Int = 16) throws {
        self.modelDir = modelDir
        self.layout = layout
        self.numLayers = numLayers
        self.capacity = max(1, capacity)
        self.fds = [Int32](repeating: -1, count: numLayers)

        let packedDir = modelDir.appendingPathComponent(QTurboFormatV1.packedExpertsDir)
        for l in 0..<numLayers {
            let path = packedDir.appendingPathComponent(QTurboFormatV1.packedExpertFilename(layer: l)).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw StreamError.cannotOpen(l, path) }
            #if os(macOS)
            // Disable additional read-ahead; we manage prefetch explicitly.
            fcntl(fd, F_NOCACHE, 0)
            #endif
            fds[l] = fd
        }
    }

    deinit {
        for fd in fds where fd >= 0 { close(fd) }
    }

    @inline(__always)
    private func key(_ layer: Int, _ expert: Int) -> UInt64 {
        (UInt64(layer) << 32) | UInt64(UInt32(expert))
    }

    @inline(__always)
    private func fileOffset(expert: Int) -> Int {
        expert * layout.alignedStride
    }

    // MARK: - Loading

    /// Returns the blob for (layer, expert), loading from disk on a cache miss.
    /// The returned pointer references cache-owned memory valid until eviction.
    public func load(layer: Int, expert: Int) throws -> UnsafeRawBufferPointer {
        let k = key(layer, expert)

        lock.lock()
        if let entry = cache[k] {
            entry.frequency &+= 1
            let buf = UnsafeRawBufferPointer(start: entry.pointer, count: entry.length)
            lock.unlock()
            return buf
        }
        lock.unlock()

        // Miss: read outside the lock (pread is thread-safe).
        let (ptr, len) = try readBlob(layer: layer, expert: expert)

        lock.lock()
        // Another thread may have loaded it concurrently.
        if let existing = cache[k] {
            ptr.deallocate()
            existing.frequency &+= 1
            let buf = UnsafeRawBufferPointer(start: existing.pointer, count: existing.length)
            lock.unlock()
            return buf
        }
        evictIfNeededLocked(inserting: 1)
        let entry = CacheEntry(key: k, pointer: ptr, length: len)
        cache[k] = entry
        let buf = UnsafeRawBufferPointer(start: entry.pointer, count: entry.length)
        lock.unlock()
        return buf
    }

    /// Reads a single blob from disk via pread into freshly allocated memory.
    private func readBlob(layer: Int, expert: Int) throws
        -> (UnsafeMutableRawPointer, Int) {
        let fd = fds[layer]
        let len = layout.blobSize
        let off = fileOffset(expert: expert)
        let ptr = UnsafeMutableRawPointer.allocate(byteCount: len, alignment: QTurboFormatV1.pageSize)

        var total = 0
        while total < len {
            let n = pread(fd, ptr.advanced(by: total), len - total, off_t(off + total))
            if n < 0 {
                let en = errno
                if en == EINTR { continue }
                ptr.deallocate()
                throw StreamError.readFailed(layer: layer, expert: expert, errno: en)
            }
            if n == 0 { break }
            total += n
        }
        guard total == len else {
            ptr.deallocate()
            throw StreamError.shortRead(layer: layer, expert: expert, want: len, got: total)
        }
        return (ptr, len)
    }

    // MARK: - Prefetch

    /// Issues SSD read-ahead hints for the given experts of a layer, and warms
    /// the cache in the background so subsequent `load` calls hit memory.
    public func prefetch(layer: Int, experts: [Int]) {
        guard layer >= 0 && layer < numLayers else { return }
        let fd = fds[layer]

        for e in experts {
            let off = fileOffset(expert: e)
            let len = layout.blobSize
            issueReadAdvice(fd: fd, offset: off, length: len)
        }

        // Warm the LFU cache asynchronously.
        prefetchQueue.async { [weak self] in
            guard let self else { return }
            for e in experts {
                let k = self.key(layer, e)
                self.lock.lock()
                let present = self.cache[k] != nil
                self.lock.unlock()
                if present { continue }
                if let (ptr, len) = try? self.readBlob(layer: layer, expert: e) {
                    self.lock.lock()
                    if self.cache[k] == nil {
                        self.evictIfNeededLocked(inserting: 1)
                        self.cache[k] = CacheEntry(key: k, pointer: ptr, length: len)
                    } else {
                        ptr.deallocate()
                    }
                    self.lock.unlock()
                }
            }
        }
    }

    /// Issues an OS-level read-ahead hint for a byte range of an fd.
    private func issueReadAdvice(fd: Int32, offset: Int, length: Int) {
        #if os(macOS)
        var ra = radvisory(ra_offset: off_t(offset), ra_count: Int32(min(length, Int(Int32.max))))
        _ = fcntl(fd, F_RDADVISE, &ra)
        #else
        _ = posix_fadvise(fd, off_t(offset), off_t(length), POSIX_FADV_WILLNEED)
        #endif
    }

    // MARK: - Eviction (LFU)

    /// Evicts least-frequently-used entries until there is room for `inserting`
    /// new entries. Must be called with the lock held.
    private func evictIfNeededLocked(inserting: Int) {
        while cache.count + inserting > capacity {
            // Find the least-frequently-used entry.
            var victimKey: UInt64?
            var minFreq = Int.max
            for (k, e) in cache where e.frequency < minFreq {
                minFreq = e.frequency
                victimKey = k
            }
            guard let vk = victimKey else { break }
            cache.removeValue(forKey: vk) // deinit frees memory
        }
    }

    /// Current number of cached blobs.
    public var cachedCount: Int {
        lock.lock(); defer { lock.unlock() }
        return cache.count
    }

    /// Drops all cached blobs (frees memory).
    public func clearCache() {
        lock.lock(); defer { lock.unlock() }
        cache.removeAll()
    }
}

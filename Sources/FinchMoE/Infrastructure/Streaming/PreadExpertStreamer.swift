import Darwin
import Foundation
import Metal

public struct ExpertIOAdviceResult: Sendable, Equatable {
    public let requested: Int
    public let failed: Int
    public let calls: Int
    public let bytes: UInt64
    public let skipped: Int
    public let maxCallNanos: UInt64

    public init(requested: Int,
                failed: Int,
                calls: Int? = nil,
                bytes: UInt64 = 0,
                skipped: Int = 0,
                maxCallNanos: UInt64 = 0) {
        self.requested = requested
        self.failed = failed
        self.calls = calls ?? requested
        self.bytes = bytes
        self.skipped = skipped
        self.maxCallNanos = maxCallNanos
    }

    public static func skipped(requested: Int, bytes: UInt64 = 0) -> ExpertIOAdviceResult {
        ExpertIOAdviceResult(requested: requested,
                             failed: 0,
                             calls: 0,
                             bytes: bytes,
                             skipped: requested)
    }

}

public struct ExpertCachePlan: Sendable, Equatable {
    public let experts: [Int]
    public let assignedSlots: [Int]
    public let misses: [Int]
    public let hits: Int

    public init(experts: [Int], assignedSlots: [Int], misses: [Int], hits: Int) {
        self.experts = experts
        self.assignedSlots = assignedSlots
        self.misses = misses
        self.hits = hits
    }
}

public enum ExpertCachePolicy: String, Sendable {
    case lru
    case lfu
}

/// `pread`-based routed-expert streamer with a fixed per-layer slot cache.
public final class PreadExpertStreamer: @unchecked Sendable {
    public static let scratchAlignment = 2 * 1024 * 1024
    public static var cachePolicyDefault: ExpertCachePolicy { .lfu }

    /// Words per miss in the read-split scratch: 16 `UInt64`s = 128 bytes, one
    /// Apple Silicon cache line. Each miss gets its own line so that the threads
    /// of a fan-out storing their entry and exit stamps are not also contending
    /// for the line those stamps live on.
    private static let markStride = 16

    /// The staging buffer's per-thread slot, owned by `pthread_key_t` so that a
    /// pool thread which exits frees its buffer instead of leaking it. The
    /// destructor is the only owner and the only free.
    ///
    /// One key for the whole process, so the buffer is shared by every streamer
    /// that runs on a thread rather than held per instance. That is safe here
    /// for a reason worth writing down, because nothing enforces it: there is
    /// one construction site (`Model.swift`, one streamer per layer) and every
    /// layer draws the same `packedExpertsLayout.expertStride`, so all streamers
    /// in a process agree on the staging length. A future second construction
    /// site with a *larger* stride traps on the precondition in `readFull`
    /// rather than overrunning; one with a smaller stride reuses an oversized
    /// buffer, which is harmless because only `count` bytes are ever written to
    /// it or copied out. What would not be safe is a smaller-then-larger pair
    /// with the precondition removed.
    private static let stageKey: pthread_key_t? = {
        var key = pthread_key_t()
        return pthread_key_create(&key) { raw in free(raw) } == 0 ? key : nil
    }()

    public let layout: StreamLayout
    public let slotCount: Int
    public let cachePolicy: ExpertCachePolicy

    private let fd: Int32
    private let slotPointers: [UnsafeMutableRawPointer]
    private let slotBuffers: [MTLBuffer]

    /// `FINCHMOE_IO_READ_SPLIT` -- how many `pread`s one expert read is issued
    /// as, sequentially within the calling thread. Default 1, which is the
    /// shape every number before this knob was measured in.
    ///
    /// It exists to test exactly one thing: whether this drive serves a smaller
    /// request faster *per byte*. Qwen 3.8 reads 2.64 MiB experts at 3.05 GB/s
    /// where Qwen 3.6 reads 1.69 MiB experts at 6.12 GB/s, same engine, same
    /// drive, and every other explanation has been measured away -- the reads
    /// are 99.7% of the window at 100% of the fan-out width asked for, and
    /// bypassing the buffer cache leaves the gap at 1.95x against a cached
    /// 2.01x. Splitting the read is the only way to change request size while
    /// holding the install, the offsets and the bytes fixed.
    ///
    /// It is not a throughput knob to reach for on faith. It converts one
    /// `pread` into K and trades per-stream queue depth for request count, so
    /// it can as easily lose -- see `FINCHMOE_IO_NOCACHE` above for what this
    /// file does with a knob that measures well offline and badly here.
    private let readSplit: Int

    /// `FINCHMOE_IO_READ_WAVE` -- how many misses are left outstanding at once.
    /// Zero or less means the whole batch, which is the single unbroken
    /// `concurrentPerform` every number before this knob was measured through.
    ///
    /// It exists to test the one structural difference left between the two
    /// installs' read rates, and it is a *depth* knob where `readSplit` is a
    /// *shape* knob: this one changes nothing about which bytes are asked for,
    /// only how many requests are in the drive at a time.
    ///
    /// Qwen 3.8 reads at 3.07 GB/s with 5.59 reads outstanding, 14.76 MiB in
    /// flight; Qwen 3.6 reads at 5.95 GB/s with 2.95 outstanding, 4.99 MiB.
    /// The deeper one is the slower one, which is backwards for a queue, and
    /// that is the whole of the prior: two installs that differ in stride,
    /// expert count, layer count and cache size, compared on one observation
    /// each.
    ///
    /// The offline replay is *against* this, and the record should say so. It
    /// varied the pool width over 3 to 10 on 3.8's own captured offsets and
    /// found nothing -- 147.3 against 149.6 ms/step, a 1.6% spread
    /// (`docs/OPTIMIZATION_PLAN.md`, the replay table). Pool width was called
    /// flat there, and "the queue-depth knee" is its phrase. So this knob is
    /// testing a hypothesis the best available control already discounts, and
    /// it is worth running only because a clean negative closes the question on
    /// the engine the way it is closed offline.
    ///
    /// What the replay does not explain, and what is left when depth goes: the
    /// same captured offsets at the same per-layer depth, destination pages
    /// shaped like the engine's, run at 149.6 ms/step offline against the
    /// engine's own 225.4 (`io` window). Same drive, same offsets, same depth,
    /// 1.5x apart. That gap is not a queueing effect under any reading of the
    /// depth curve, and it is the number to chase if this comes back flat.
    ///
    /// This is not the `IO-04`/`IO-05` rejection. Those were a dedicated
    /// executor and a custom worker pool -- new threading machinery on a path
    /// whose dispatch was later measured at 0.29% of its own window. The
    /// dispatch is unchanged here; only the number of iterations handed to it
    /// at a time is.
    ///
    /// The cost is real and belongs in the reading: `concurrentPerform` returns
    /// only when every iteration has, so each wave is a barrier. A flat result
    /// at a low width is therefore ambiguous -- it can mean the drive does not
    /// care, or that it liked the width and paid it back at the barriers. The
    /// `io_conc` counter and the summed thread time are what tell those apart.
    private let readWave: Int

    /// `FINCHMOE_IO_STAGE` -- read into a plain staging buffer and copy into the
    /// slot, instead of reading straight into the slot.
    ///
    /// One variable changes and it is the one every previous experiment held
    /// fixed: *where the bytes land*. The staging buffer is allocated with the
    /// same `posix_memalign(Self.scratchAlignment, ...)` call and the same
    /// padded length as a slot, so alignment and page size match the slot
    /// exactly. The only difference left is that the slot is handed to
    /// `device.makeBuffer(bytesNoCopy:options:.storageModeShared)` and the
    /// staging buffer is not.
    ///
    /// **Measured, and the answer is no.** `docs/OPTIMIZATION_PLAN.md` named
    /// this as the next discriminator: the engine reads at 3.60 GB/s where the
    /// offline replay reads the same offsets at 5.42, and the last difference
    /// left standing between them was that "the engine's slot pages are
    /// GPU-shared `MTLBuffer`s under a live Metal heap, which changes the vm
    /// object's reclamation behaviour" (`:193`, the 5.42 at `:177`). Staging
    /// holds the drive, the offsets, the depth, the slot and the alignment
    /// fixed and moves only the destination. Qwen 3.8 reads at 3.28 GB/s with
    /// it off and 3.28 with it on, two rounds each agreeing to 0.3%; the window
    /// grows 2.5% and the added memcpy is 2.4% of it. The slot mapping is not
    /// the cost.
    ///
    /// The control is what makes that a finding rather than an artifact: Qwen
    /// 3.6 reads at 6.24/6.55 GB/s off against 6.90/6.87 on, so the staging
    /// path is not itself expensive and its null result on 3.8 belongs to the
    /// read, not to the knob. The copy separates by install too -- 0.043
    /// ms/MiB on 3.8 against 0.050 on 3.6, within 15%, while the *read* differs
    /// 3.3x (1.68 ms/MiB against 0.51). Same memory system, same bytes, same
    /// pages: symmetric on the copy, asymmetric on the read. Whatever the 2.2x
    /// between the installs is, it is upstream of the destination.
    ///
    /// One earlier reason to suspect the destination did not survive
    /// re-reading. The slot sweep reported Qwen 3.6 losing a quarter to a third
    /// of its per-byte rate when its pool doubled, but its *absolute* read
    /// thread time is flat across that change (146.5/138.6 ms/step at 16 slots
    /// against 136.9/144.1 at 32) and the window moves only 3-10%; the rate
    /// falls because 32 slots cache-hit more and so read 30% fewer bytes in
    /// about the same window. Nothing per byte got worse. What does rise is
    /// per-read latency, 38%, across 30% fewer reads -- and that is a statement
    /// about the drive's response to fewer, sparser requests, not about where
    /// the bytes land, which staging confirms by decoupling the two: 3.6 at 32
    /// slots reads at 4.16/4.02 GB/s off against 4.25/4.04 on, same byte count
    /// either way. So the corrected reading is not "a bigger pool reads slower"
    /// but "a bigger pool issues fewer reads, and each costs more"; the per-byte
    /// version is how this becomes the reason to try staging a second time.
    ///
    /// It costs a memcpy of every expert read, so it is not a knob to leave on.
    private let stageReads: Bool

    /// The padded byte length of one slot, which is also the staging buffer's
    /// length. Kept so the two allocations can be made identically.
    private let stageAllocationSize: Int

    /// The staging experiment's two timers, summed across the batch's threads
    /// under a lock. This is not the per-call lock the comment above rejects:
    /// it is taken once per *read*, and the read it brackets is three orders of
    /// magnitude longer than the acquisition is.
    private let stageLock = NSLock()
    private var batchPreadNanos: UInt64 = 0
    private var batchCopyNanos: UInt64 = 0

    private var nextSlot = 0
    private let cursorLock = NSLock()

    private var slotExpert: [Int]
    private var slotLastUse: [Int]
    private var expertUseCount: [Int]
    private var useClock = 0
    private let cacheLock = NSLock()

    public convenience init(layout: StreamLayout,
                            device: MTLDevice,
                            slotCount: Int,
                            cachePolicy: ExpertCachePolicy = .lfu) throws {
        try self.init(layout: layout,
                      device: device,
                      slotCount: slotCount,
                      cachePolicy: cachePolicy,
                      fileDescriptor: nil)
    }

    package init(layout: StreamLayout,
                 device: MTLDevice,
                 slotCount: Int,
                 cachePolicy: ExpertCachePolicy = .lfu,
                 fileDescriptor: Int32?,
                 readSplit: Int? = nil,
                 readWave: Int? = nil,
                 stageReads: Bool? = nil) throws {
        precondition(slotCount > 0, "slotCount must be positive")
        self.layout = layout
        self.slotCount = slotCount
        self.cachePolicy = cachePolicy
        let pageSize = Int(getpagesize())

        let openedFD = fileDescriptor.map { fcntl($0, F_DUPFD_CLOEXEC, 0) }
            ?? open(layout.path, O_RDONLY | O_NONBLOCK | O_CLOEXEC)
        guard openedFD >= 0 else {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        // `F_NOCACHE` -- measured on the engine, and it loses. Replaying this
        // streamer's exact pread sequence offline says bypassing the buffer
        // cache is worth 42-61 ms/step -- a fifth to a quarter of the engine's
        // own read window -- on identical offsets at identical depth,
        // interleaved and run in both orders. On the engine it costs instead:
        // 225.4 -> 261.0 ms/step of `io`, -12% tok/s, reproduced in both orders
        // with identical token IDs.
        //
        // The offline harness is what is wrong, and the trace says how. 41.7%
        // of this streamer's reads are repeats of a (layer, expert) pair
        // already read during the run, but the *median* reuse distance is 3.97
        // GiB and not one repeat lands within 512 MiB -- so the harness's ten
        // recycling destination pages were holding reuse the engine's cache
        // cannot, and bypassing threw away a hit rate the engine actually has.
        // Worth re-testing only if the reuse distance ever shortens.
        //
        // Kept, off by default, as the falsifier for that claim rather than as
        // a knob anyone should turn on.
        if ProcessInfo.processInfo.environment["FINCHMOE_IO_NOCACHE"] == "1",
           fcntl(openedFD, F_NOCACHE, 1) != 0 {
            throw StreamerError.openFailed(path: layout.path, errno: errno)
        }
        self.fd = openedFD
        var closeFDOnFailure = true
        defer { if closeFDOnFailure { close(openedFD) } }

        var fileStats = stat()
        guard fstat(openedFD, &fileStats) == 0,
              (fileStats.st_mode & S_IFMT) == S_IFREG,
              fileStats.st_size >= 0 else {
            throw StreamerError.openFailed(
                path: layout.path, errno: errno == 0 ? EINVAL : errno)
        }
        let (required, requiredOverflow) = layout.streamOffset
            .addingReportingOverflow(layout.streamSize)
        guard !requiredOverflow, UInt64(fileStats.st_size) >= required else {
            throw StreamerError.sizeMismatch(
                expected: requiredOverflow ? UInt64.max : required,
                actual: UInt64(fileStats.st_size))
        }
        guard layout.expertStride > 0,
              layout.expertStride <= UInt64(Int.max - (pageSize - 1)) else {
            throw StreamerError.invalidIOSplitConfiguration(
                "expertStride \(layout.expertStride) is not addressable")
        }

        let allocationSize = ((Int(layout.expertStride) + pageSize - 1) / pageSize) * pageSize
        var pointers: [UnsafeMutableRawPointer] = []
        var buffers: [MTLBuffer] = []
        pointers.reserveCapacity(slotCount)
        buffers.reserveCapacity(slotCount)

        func unwind() {
            for index in buffers.count..<pointers.count {
                free(pointers[index])
            }
        }

        for _ in 0..<slotCount {
            var raw: UnsafeMutableRawPointer?
            let result = posix_memalign(&raw, Self.scratchAlignment, allocationSize)
            guard result == 0, let pointer = raw else {
                unwind()
                throw StreamerError.allocFailed(errno: result)
            }
            pointers.append(pointer)
            nonisolated(unsafe) let capturedPointer = pointer
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: allocationSize,
                options: .storageModeShared,
                deallocator: { _, _ in free(capturedPointer) })
            else {
                unwind()
                throw StreamerError.bufferWrapFailed
            }
            buffers.append(buffer)
        }

        // Read once per layer open rather than per read: this sits on the hot
        // path, and `getenv` there would be measured by the numbers it exists
        // to produce. The parameter wins over the environment so tests can set
        // it without mutating process state.
        self.readSplit = max(
            1,
            readSplit
                ?? ProcessInfo.processInfo.environment["FINCHMOE_IO_READ_SPLIT"]
                    .flatMap(Int.init)
                ?? 1)
        // Not clamped to a positive minimum the way `readSplit` is: zero is a
        // meaningful setting here, and it is the default. Any width at or above
        // the miss count lands on the same single batch.
        self.readWave = max(
            0,
            readWave
                ?? ProcessInfo.processInfo.environment["FINCHMOE_IO_READ_WAVE"]
                    .flatMap(Int.init)
                ?? 0)
        // `FINCHMOE_IO_STAGE=0` is a meaningful setting, so this reads presence
        // plus value rather than presence alone.
        // Parenthesised because `??` binds tighter than `!=` in Swift: without
        // the outer pair this parses as `(stageReads ?? Int) != 0`.
        self.stageReads = stageReads
            ?? ((ProcessInfo.processInfo.environment["FINCHMOE_IO_STAGE"]
                .flatMap(Int.init) ?? 0) != 0)
        self.stageAllocationSize = allocationSize
        self.slotPointers = pointers
        self.slotBuffers = buffers
        self.slotExpert = [Int](repeating: -1, count: slotCount)
        self.slotLastUse = [Int](repeating: 0, count: slotCount)
        self.expertUseCount = [Int](repeating: 0, count: max(1, layout.expertsPerLayer))
        closeFDOnFailure = false
    }

    deinit {
        close(fd)
    }

    public func loadExpert(layer: Int, expert: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        cursorLock.lock()
        let slot = nextSlot
        nextSlot = (nextSlot + 1) % slotCount
        cursorLock.unlock()
        return try loadExpert(layer: layer, expert: expert, slot: slot)
    }

    public func loadExpert(layer: Int, expert: Int, slot: Int) throws
        -> (buffer: MTLBuffer, offset: UInt64, size: UInt64) {
        guard slot >= 0 && slot < slotCount else {
            throw StreamerError.slotOutOfRange(slot)
        }
        let regionOffset = layout.expertOffset(layer: layer, expert: expert)
        guard regionOffset + layout.expertStride <= layout.streamSize else {
            throw StreamerError.offsetOutOfRange(regionOffset)
        }
        try readFull(
            into: slotPointers[slot],
            fileOffset: layout.streamOffset + regionOffset,
            count: Int(layout.expertStride))
        return (slotBuffers[slot], 0, layout.expertStride)
    }

    public func loadExpertsCached(experts: [Int]) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        try executeExpertCachePlan(planExpertsCached(experts: experts))
    }

    public func planExpertsCached(experts: [Int],
                                  avoidingSlots: Set<Int> = []) -> ExpertCachePlan {
        guard let plan = makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots) else {
            preconditionFailure("expert cache cannot place requested misses")
        }
        return plan
    }

    public func planExpertsCachedIfPossible(experts: [Int],
                                            avoidingSlots: Set<Int> = []) -> ExpertCachePlan? {
        makeExpertCachePlan(experts: experts, avoidingSlots: avoidingSlots)
    }

    private func makeExpertCachePlan(experts: [Int],
                                     avoidingSlots rawAvoidingSlots: Set<Int>) -> ExpertCachePlan? {
        precondition(experts.count <= slotCount,
                     "expert cache needs at least \(experts.count) slots")
        let avoidingSlots = Set(rawAvoidingSlots.filter { $0 >= 0 && $0 < slotCount })

        cacheLock.lock()
        defer { cacheLock.unlock() }

        let clock = useClock + 1
        var assignedSlots = [Int](repeating: -1, count: experts.count)
        var reserved = [Bool](repeating: false, count: slotCount)

        for index in experts.indices {
            for slot in 0..<slotCount
                where !reserved[slot] && slotExpert[slot] == experts[index] {
                assignedSlots[index] = slot
                reserved[slot] = true
                break
            }
        }
        for slot in avoidingSlots where !reserved[slot] {
            reserved[slot] = true
        }

        let misses = experts.indices.filter { assignedSlots[$0] == -1 }
        let evictable = (0..<slotCount)
            .filter { !reserved[$0] }
            .sorted { shouldEvictSlot($0, before: $1) }
        guard misses.count <= evictable.count else { return nil }

        useClock = clock
        for expert in experts where expert >= 0 && expert < expertUseCount.count {
            expertUseCount[expert] &+= 1
        }
        for slot in assignedSlots where slot >= 0 {
            slotLastUse[slot] = clock
        }
        for (offset, index) in misses.enumerated() {
            let slot = evictable[offset]
            assignedSlots[index] = slot
            reserved[slot] = true
            slotExpert[slot] = -1
            slotLastUse[slot] = clock
        }

        return ExpertCachePlan(
            experts: experts,
            assignedSlots: assignedSlots,
            misses: misses,
            hits: experts.count - misses.count)
    }

    /// The last `executeExpertCachePlan` call's split, in nanoseconds: the
    /// `concurrentPerform` fan-out plus its preads, then the cache bookkeeping
    /// and the view construction that follow. Read immediately by the caller
    /// that owns the running totals (`ModelExpertIO`), on the same thread that
    /// made the call and before any other call can start — which is why these
    /// are plain properties and not a lock-guarded accumulator. A per-call
    /// lock here would be measured by the very numbers it produces.
    public private(set) var lastReadNanos: UInt64 = 0
    public private(set) var lastTailNanos: UInt64 = 0

    /// The read's own split, in nanoseconds. `lastReadNanos` above is the whole
    /// `concurrentPerform`; these are its parts, and they tile it exactly:
    ///
    /// - `fanout` -- batch start to the first iteration's entry: what the price
    ///   of the nested dispatch actually is, measured rather than assumed. It is
    ///   the window in which the batch has been handed to the global queue and
    ///   no iteration has begun yet.
    /// - `span` -- first entry to last exit: the window in which reads were
    ///   actually in flight.
    /// - `drain` -- last exit to the return, i.e. the straggler.
    /// - `threadNanos` -- the sum of each iteration's own pread time across all
    ///   threads, so `threadNanos / span` is the parallelism the fan-out
    ///   actually achieved. Near 1.0 the reads ran one at a time however many
    ///   threads were asked for; near the miss count they ran fully wide. That
    ///   ratio is the one thing a wall clock around the batch cannot show.
    ///
    ///   It reads slightly differently once `readWave` caps the width: the span
    ///   then also contains the barriers between waves, in which nothing is in
    ///   flight, so the ratio comes out *below* the width rather than at it.
    ///   `io_conc` is the average over the whole window, not the peak.
    public private(set) var lastReadFanoutNanos: UInt64 = 0
    public private(set) var lastReadSpanNanos: UInt64 = 0
    public private(set) var lastReadDrainNanos: UInt64 = 0
    public private(set) var lastReadThreadNanos: UInt64 = 0

    /// The read split by *operation* rather than by thread, and only
    /// meaningful as a pair. With `FINCHMOE_IO_STAGE` off the whole of
    /// `lastReadThreadNanos` is charged to `pread` and `copy` is zero, which is
    /// the honest reading -- there was no copy. With it on they are the two
    /// spans actually taken. They tile the read portion of the iteration and
    /// fall a few nanoseconds short of `lastReadThreadNanos`, which also
    /// contains the slot bounds check and the offset arithmetic around them.
    public private(set) var lastReadPreadNanos: UInt64 = 0
    public private(set) var lastReadCopyNanos: UInt64 = 0

    /// Per-read latency as a log2 histogram, because every other read number
    /// here is a sum, and a sum cannot say what shape its mean has.
    ///
    /// `io_thread_wall / misses` is 4.65 ms on 3.8 against an offline replay's
    /// 2.62 ms on the identical offsets at the same depth, and no aggregate
    /// distinguishes "every read is slower" from "the same reads plus a tail".
    /// Bucket `b` counts the reads whose iteration took `2^b ..< 2^(b+1)`
    /// nanoseconds, so the two shapes separate on sight: a uniform shift moves
    /// the median, a tail leaves the median where it was and stretches the p99.
    ///
    /// Filled from the same walk over `marks` that computes
    /// `lastReadThreadNanos` -- on the calling thread, after the fan-out has
    /// returned -- so the read path pays nothing for it. An iteration's span is
    /// the slot bounds check, the offset arithmetic, the pread and, when
    /// staging is on, the copy: exactly the quantity `io_thread_wall` sums, so
    /// the histogram and the mean describe the same reads.
    public private(set) var lastReadLatencyHistogram =
        [UInt64](repeating: 0, count: PreadExpertStreamer.latencyBucketCount)
    /// `2^0 .. 2^33` ns, i.e. up to 8.6 s. A read slower than that is a hang
    /// rather than a read, and it clamps into the top bucket instead of being
    /// dropped -- a dropped read would silently pull every percentile down.
    public static let latencyBucketCount = 34

    /// Reads counted in `lastReadLatencyHistogram`, so a percentile can be
    /// quoted against its own denominator rather than against `misses`.
    public private(set) var lastReadLatencyCount: UInt64 = 0

    public func executeExpertCachePlan(_ plan: ExpertCachePlan) throws
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.experts.count <= slotCount,
                     "expert cache plan exceeds slot count")
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")

        let errorLock = NSLock()
        nonisolated(unsafe) var firstError: Error?
        // Two timestamps per miss, each miss on its own cache line: the whole
        // point is that several threads store these at once, and six stores
        // into one line is a contention the reads would then be measured
        // through. Raw memory rather than an array because the indices are
        // distinct by construction. Deliberately not zeroed: the iteration that
        // owns a miss stores its entry before entering and its exit after, so
        // every slot is written before anything reads it.
        // `nonisolated(unsafe)` for the same reason `firstError` carries it: the
        // closure is `@Sendable`, and what makes the sharing safe is not the type
        // but the disjointness of the indices, which the compiler cannot see.
        nonisolated(unsafe) let marks = UnsafeMutablePointer<UInt64>
            .allocate(capacity: max(plan.misses.count, 1) * Self.markStride)
        defer { marks.deallocate() }

        let tRead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // One wave, unless a width is set. `concurrentPerform` returns only
        // once every iteration has returned, so the loop below is a barrier
        // between waves: at most `waveWidth` reads are ever outstanding, and
        // the price is the serialisation point. With no width the loop runs
        // exactly once and this is the unbroken fan-out as it was.
        //
        // The marks stay indexed by the *global* miss offset, not by the offset
        // within the wave, so the tiling below is unchanged by the split -- a
        // wave boundary falls inside `span` and nowhere else.
        stageLock.lock()
        batchPreadNanos = 0
        batchCopyNanos = 0
        stageLock.unlock()
        let waveWidth = readWave > 0 ? min(readWave, plan.misses.count) : plan.misses.count
        var waveStart = 0
        while waveStart < plan.misses.count {
            let waveCount = min(waveWidth, plan.misses.count - waveStart)
            let waveBase = waveStart
            DispatchQueue.concurrentPerform(iterations: waveCount) { offset in
                let missOffset = waveBase + offset
                let index = plan.misses[missOffset]
                let base = missOffset * Self.markStride
                marks[base] = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                do {
                    _ = try self.loadExpert(
                        layer: 0,
                        expert: plan.experts[index],
                        slot: plan.assignedSlots[index])
                } catch {
                    errorLock.lock()
                    if firstError == nil { firstError = error }
                    errorLock.unlock()
                }
                // Stored on the throwing path too: an iteration that failed still
                // consumed wall time inside the batch, and leaving its mark at zero
                // would make the slowest iteration look instantaneous.
                marks[base + 1] = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            }
            waveStart += waveCount
        }
        let tReadEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        lastReadNanos = tReadEnd &- tRead

        // The split. Telescoping exactly: fanout + span + drain is
        // (firstEnter - tRead) + (lastExit - firstEnter) + (tReadEnd - lastExit)
        // = tReadEnd - tRead = lastReadNanos, by construction and not by
        // assertion. An empty plan never enters the closure at all, so it has no
        // first entry to measure and its whole window is drain.
        for bucket in 0..<Self.latencyBucketCount { lastReadLatencyHistogram[bucket] = 0 }
        lastReadLatencyCount = 0
        if plan.misses.count == 0 {
            lastReadFanoutNanos = 0
            lastReadSpanNanos = 0
            lastReadDrainNanos = lastReadNanos
            lastReadThreadNanos = 0
        } else {
            var firstEnter = UInt64.max
            var lastExit: UInt64 = 0
            var threadNanos: UInt64 = 0
            for missOffset in 0..<plan.misses.count {
                let enter = marks[missOffset * Self.markStride]
                let exit = marks[missOffset * Self.markStride + 1]
                firstEnter = min(firstEnter, enter)
                lastExit = max(lastExit, exit)
                let nanos = exit &- enter
                threadNanos &+= nanos
                // `floor(log2(nanos))`, zero-safe: `leadingZeroBitCount` of 0 is
                // 64, which would index below the array. The clamp is the hang
                // case the bucket count documents.
                let bucket = nanos == 0
                    ? 0
                    : min(63 - nanos.leadingZeroBitCount, Self.latencyBucketCount - 1)
                lastReadLatencyHistogram[bucket] &+= 1
                lastReadLatencyCount &+= 1
            }
            // `CLOCK_UPTIME_RAW` is monotonic, so these three differences cannot
            // go negative: `tRead` precedes every entry, and every exit follows
            // its own entry, which is what bounds `firstEnter` from below. Taken
            // as plain differences rather than clamped ones so that the tiling
            // stays exact -- a clamp would silently hand the reader three parts
            // that do not add up to the whole.
            lastReadFanoutNanos = firstEnter &- tRead
            lastReadSpanNanos = lastExit &- firstEnter
            lastReadDrainNanos = tReadEnd &- lastExit
            lastReadThreadNanos = threadNanos
        }
        // Published after the thread-time split, not before, because the
        // no-staging branch is defined in terms of it.
        if stageReads {
            stageLock.lock()
            lastReadPreadNanos = batchPreadNanos
            lastReadCopyNanos = batchCopyNanos
            stageLock.unlock()
        } else {
            lastReadPreadNanos = lastReadThreadNanos
            lastReadCopyNanos = 0
        }
        if let firstError { throw firstError }

        let tTail = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        cacheLock.lock()
        for index in plan.misses {
            slotExpert[plan.assignedSlots[index]] = plan.experts[index]
        }
        cacheLock.unlock()

        let views = expertCachePlanBuffers(plan)
        lastTailNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tTail
        return views
    }

    public func expertCachePlanBuffers(_ plan: ExpertCachePlan)
        -> [(buffer: MTLBuffer, offset: UInt64, size: UInt64)] {
        precondition(plan.assignedSlots.count == plan.experts.count,
                     "expert cache plan slot count mismatch")
        return plan.assignedSlots.map { slot in
            (slotBuffers[slot], UInt64(0), layout.expertStride)
        }
    }

    public func adviseExpertCachePlanMisses(_ plan: ExpertCachePlan) -> ExpertIOAdviceResult {
        let experts = plan.misses.map { plan.experts[$0] }
        return adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExperts(experts: [Int]) -> ExpertIOAdviceResult {
        adviseRanges(expertAdviceRanges(experts: experts), requested: experts.count)
    }

    public func adviseExpertMisses(experts: [Int]) -> ExpertIOAdviceResult {
        cacheLock.lock()
        let misses = experts.filter { !slotExpert.contains($0) }
        cacheLock.unlock()
        return adviseRanges(expertAdviceRanges(experts: misses), requested: misses.count)
    }

    static func coalescedAdjacentAdviceRanges(_ ranges: [(offset: UInt64, count: UInt64)])
        -> [(offset: UInt64, count: UInt64)] {
        let sorted = ranges.filter { $0.count > 0 }.sorted {
            $0.offset == $1.offset ? $0.count < $1.count : $0.offset < $1.offset
        }
        var result: [(offset: UInt64, count: UInt64)] = []
        for range in sorted {
            guard var last = result.popLast() else {
                result.append(range)
                continue
            }
            let lastEnd = last.offset &+ last.count
            let rangeEnd = range.offset &+ range.count
            if range.offset <= lastEnd {
                last.count = max(lastEnd, rangeEnd) - last.offset
                result.append(last)
            } else {
                result.append(last)
                result.append(range)
            }
        }
        return result
    }

    private func shouldEvictSlot(_ lhs: Int, before rhs: Int) -> Bool {
        if cachePolicy == .lru {
            return slotLastUse[lhs] < slotLastUse[rhs]
        }
        let lhsExpert = slotExpert[lhs]
        let rhsExpert = slotExpert[rhs]
        if lhsExpert < 0 || rhsExpert < 0 {
            return lhsExpert < rhsExpert
        }
        let lhsCount = lhsExpert < expertUseCount.count ? expertUseCount[lhsExpert] : 0
        let rhsCount = rhsExpert < expertUseCount.count ? expertUseCount[rhsExpert] : 0
        if lhsCount != rhsCount { return lhsCount < rhsCount }
        return slotLastUse[lhs] < slotLastUse[rhs]
    }

    private func expertAdviceRanges(experts: [Int]) -> [(offset: UInt64, count: UInt64)] {
        experts.compactMap { expert in
            let regionOffset = layout.expertOffset(layer: 0, expert: expert)
            guard regionOffset + layout.expertStride <= layout.streamSize else { return nil }
            return (layout.streamOffset + regionOffset, layout.expertStride)
        }
    }

    private func adviseRanges(_ ranges: [(offset: UInt64, count: UInt64)],
                              requested: Int) -> ExpertIOAdviceResult {
        let coalesced = Self.coalescedAdjacentAdviceRanges(ranges)
        var failed = 0
        var bytes: UInt64 = 0
        var maxCallNanos: UInt64 = 0
        for range in coalesced {
            let result = RDAdvice.call(fd: fd, offset: range.offset, byteCount: range.count)
            if !result.succeeded { failed += 1 }
            bytes &+= result.requestedBytes
            maxCallNanos = max(maxCallNanos, result.elapsedNanos)
        }
        return ExpertIOAdviceResult(
            requested: requested,
            failed: failed,
            calls: coalesced.count,
            bytes: bytes,
            maxCallNanos: maxCallNanos)
    }

    /// The thread's staging buffer, allocated on first use at the same
    /// alignment and length as a slot and registered with the key before it is
    /// returned, so exactly one owner can ever free it.
    private func stagingBuffer() -> UnsafeMutableRawPointer? {
        guard let key = Self.stageKey else { return nil }
        if let existing = pthread_getspecific(key) { return existing }
        var raw: UnsafeMutableRawPointer?
        guard posix_memalign(&raw, Self.scratchAlignment, stageAllocationSize) == 0,
              let buffer = raw else { return nil }
        pthread_setspecific(key, buffer)
        return buffer
    }

    /// Read the expert into `destination`, optionally by way of the thread's
    /// plain staging buffer.
    ///
    /// Timed as two spans rather than one so the experiment can say where the
    /// time went: `pread` is the drive and the staging write, `copy` is the
    /// write into the slot's mapping. With staging off there is no second span
    /// and the whole of it is charged to `pread`, which keeps the pair tiling
    /// the read in both modes rather than only in the one under test.
    private func readFull(into destination: UnsafeMutableRawPointer,
                          fileOffset: UInt64,
                          count: Int) throws {
        guard count > 0 else { return }
        guard stageReads, let staging = stagingBuffer() else {
            try readChunks(into: destination, fileOffset: fileOffset, count: count)
            return
        }
        // The staging buffer is sized once from `expertStride` at init, while
        // `count` is chosen per call. Today they agree by construction; the
        // precondition is here so that a second caller with a larger read
        // fails loudly instead of overrunning a thread-local allocation.
        precondition(
            count <= stageAllocationSize,
            "staged read of \(count) bytes into a \(stageAllocationSize)-byte staging buffer")
        let tPreadStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try readChunks(into: staging, fileOffset: fileOffset, count: count)
        let tPreadEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        memcpy(destination, staging, count)
        let tCopyEnd = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        stageLock.lock()
        batchPreadNanos &+= tPreadEnd &- tPreadStart
        batchCopyNanos &+= tCopyEnd &- tPreadEnd
        stageLock.unlock()
    }

    private func readChunks(into destination: UnsafeMutableRawPointer,
                            fileOffset: UInt64,
                            count: Int) throws {
        guard count > 0 else { return }
        // Split into `readSplit` sequential chunks. With the default of 1 this
        // is exactly the single loop it has always been; the loop is kept
        // per-chunk rather than around the whole read so that a chunk which
        // comes back short is resumed at its own offset instead of restarting
        // the read -- the existing short-read semantics, applied per piece.
        let chunks = min(readSplit, count)
        let base = count / chunks
        let remainder = count % chunks
        var chunkStart = 0
        for chunk in 0..<chunks {
            // The remainder is spread over the first few chunks so the pieces
            // tile `count` exactly whatever the split divides into.
            let chunkCount = base + (chunk < remainder ? 1 : 0)
            var filled = 0
            while filled < chunkCount {
                let readCount = pread(
                    fd,
                    destination.advanced(by: chunkStart + filled),
                    chunkCount - filled,
                    off_t(fileOffset) + off_t(chunkStart + filled))
                if readCount < 0 {
                    throw StreamerError.preadFailed(errno: errno)
                }
                if readCount == 0 {
                    throw StreamerError.sizeMismatch(
                        expected: UInt64(count), actual: UInt64(chunkStart + filled))
                }
                filled += readCount
            }
            chunkStart += chunkCount
        }
    }
}

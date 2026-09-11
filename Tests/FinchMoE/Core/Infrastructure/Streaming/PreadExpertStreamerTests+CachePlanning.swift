import Darwin
import Foundation
import Metal
import Testing

@testable import FinchMoE

extension PreadExpertStreamerTests {
  @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
    for (index, result) in results.enumerated() {
      let expert = [3, 1, 2][index]
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  /// The read split has to tile the window it claims to split, on the real
  /// code path and not just in the formatter.
  ///
  /// `fanout + span + drain == lastReadNanos` is the property the whole
  /// measurement rests on -- if it does not hold here, the engine's `io` split
  /// is four numbers that look reasonable and cannot be added up, which is worse
  /// than one number that is merely coarse.
  @Test func readSplitTilesTheBatchWindow() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [3, 1, 2])

    let read = streamer.lastReadNanos
    let tiled = streamer.lastReadFanoutNanos
      &+ streamer.lastReadSpanNanos &+ streamer.lastReadDrainNanos
    #expect(read > 0, "a batch of real preads must take measurable time")
    #expect(tiled == read, "fanout + span + drain must tile the read window")

    // Every miss ran inside the span, so summed thread time is positive. It is
    // deliberately not asserted to be at least the span: with three reads this
    // short, skew between the pool threads can make the sum come out under the
    // span, and a bound that holds only on the real workload is not a bound.
    #expect(streamer.lastReadThreadNanos > 0)
  }

  /// Waving the batch must move the same bytes to the same places as one
  /// unbroken fan-out.
  ///
  /// The wave arithmetic adds a second index -- `waveBase + offset` -- on top of
  /// the miss index, and the remainder path runs whenever the width does not
  /// divide the miss count. Both are the kind of off-by-one a uniformly tagged
  /// fixture cannot see, so the check is against the offset-tagged layer: every
  /// expert carries a distinct byte, and a miss sent out in the wrong wave reads
  /// back as the wrong expert instead of passing.
  ///
  /// Width 1 is also the only width with an assertion available about the read
  /// split itself. One read at a time makes the per-miss intervals disjoint, so
  /// their sum cannot exceed the span that contains them -- a consequence of the
  /// width rather than of the machine's timing, which is why it is assertable
  /// here and not for a wide batch.
  @Test func wavedReadsDeliverTheSameBytesAndTileTheirWindow() throws {
    let url = try Self.writeOffsetTaggedLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let requested = [3, 1, 2]

    for width in [1, 2, 3] {
      // A fresh streamer per width, so every batch is all misses and no wave
      // boundary is shifted by a cache that has already seen an expert.
      let streamer = try PreadExpertStreamer(
        layout: Self.makeLayout(path: url.path), device: device, slotCount: 4,
        fileDescriptor: nil, readWave: width)

      let results = try streamer.loadExpertsCached(experts: requested)
      #expect(results.count == requested.count)
      for (index, result) in results.enumerated() {
        let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
        let base = Int(Self.streamOffset) + requested[index] * Self.expertStride
        let mismatch = (0..<Self.expertStride).first {
          got[$0] != Self.patternByte(base + $0)
        }
        if let j = mismatch {
          Issue.record(
            "width \(width): slot \(index) byte \(j) is \(got[j]), expected expert \(requested[index])")
        }
      }

      let read = streamer.lastReadNanos
      let tiled = streamer.lastReadFanoutNanos
        &+ streamer.lastReadSpanNanos &+ streamer.lastReadDrainNanos
      #expect(tiled == read, "width \(width): fanout + span + drain must tile the read")
      if width == 1 {
        #expect(
          streamer.lastReadThreadNanos <= streamer.lastReadSpanNanos,
          "one read outstanding at a time cannot sum to more time than the span holds")
      }
    }
  }

  /// Staging must be byte-for-byte invisible.
  ///
  /// The staged path reads into a thread-local scratch page and `memcpy`s it
  /// into the slot, so it is a second route for the bytes and a second place
  /// for an offset or length error to hide. Comparing against the unstaged
  /// streamer's own output is the only check that covers both -- asserting the
  /// staged buffer alone would pass just as well if the unstaged path were the
  /// one that had broken.
  ///
  /// The accounting has to hold in both modes, and that is the other half of
  /// the test: with staging off the whole read is charged to `pread` and the
  /// copy is zero, so `pread + copy` equals `io_thread_wall` whichever mode ran
  /// and a reader never sees part of a read charged to nothing. `threadNanos`
  /// is the outer envelope here: it also contains the per-iteration bounds
  /// check and the offset arithmetic, so the pair sums to *at most* the thread
  /// time rather than to exactly it.
  @Test func stagedReadsAreByteIdenticalAndKeepTheAccountingWhole() throws {
    let url = try Self.writeOffsetTaggedLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let requested = [3, 1, 2]

    // A fresh streamer per mode, so every batch is all misses in both.
    func readAll(staging: Bool) throws -> ([[UInt8]], UInt64, UInt64, UInt64) {
      let streamer = try PreadExpertStreamer(
        layout: Self.makeLayout(path: url.path), device: device, slotCount: 4,
        fileDescriptor: nil, stageReads: staging)
      let results = try streamer.loadExpertsCached(experts: requested)
      let bytes = results.map { Self.bytes(of: $0.buffer, offset: 0, count: Self.expertStride) }
      return (
        bytes, streamer.lastReadPreadNanos, streamer.lastReadCopyNanos,
        streamer.lastReadThreadNanos
      )
    }

    let (plain, plainPread, plainCopy, plainThread) = try readAll(staging: false)
    let (staged, stagedPread, stagedCopy, stagedThread) = try readAll(staging: true)

    for (index, expert) in requested.enumerated() {
      let base = Int(Self.streamOffset) + expert * Self.expertStride
      let expected = (0..<Self.expertStride).map { Self.patternByte(base + $0) }
      #expect(plain[index] == expected, "unstaged slot \(index) is not the tagged expert")
      #expect(staged[index] == expected, "staged slot \(index) is not the tagged expert")
      let mismatch = (0..<Self.expertStride).first { staged[index][$0] != plain[index][$0] }
      if let j = mismatch {
        Issue.record(
          "slot \(index) byte \(j): staged \(staged[index][j]) against unstaged \(plain[index][j])")
      }
    }

    // Staging off: nothing was copied, and nothing was dropped either.
    #expect(plainCopy == 0, "with staging off there is no copy to charge")
    #expect(plainPread == plainThread, "with staging off the whole read is pread")
    #expect(plainPread + plainCopy <= plainThread)

    // Staging on: the copy is real (the knob engaged rather than falling back)
    // and the pair still fits inside the thread envelope it was taken from.
    #expect(stagedCopy > 0, "staging on must charge a copy, or it did not stage")
    #expect(stagedPread + stagedCopy <= stagedThread)
  }

  /// An all-hits plan has no misses to fan out, so there is no first entry to
  /// measure and the whole window is drain. It must still tile: a zero-miss plan
  /// that reported a nonzero fanout would be inventing dispatch cost that was
  /// never paid.
  @Test func anAllHitsPlanPutsTheWholeWindowInDrain() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    // The first call is all misses; the second repeats the same experts, so
    // with four slots and two experts the plan is entirely hits.
    _ = try streamer.loadExpertsCached(experts: [2, 0])
    _ = try streamer.loadExpertsCached(experts: [2, 0])

    let read = streamer.lastReadNanos
    #expect(streamer.lastReadFanoutNanos == 0)
    #expect(streamer.lastReadSpanNanos == 0)
    #expect(streamer.lastReadDrainNanos == read)
    #expect(streamer.lastReadThreadNanos == 0)
  }

  @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 2, 3]

    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.requested == experts.count)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == experts.count)
    #endif

    let results = try streamer.loadExpertsCached(experts: experts)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func adviseExpertMissesSkipsResidentSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

    #expect(advice.requested == 2)
    #expect(advice.calls == 1)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == 1)
    #endif
  }

  @Test func plannedCacheLoadExecutesSameMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)

    #expect(plan.hits == 1)
    #expect(plan.misses.map { experts[$0] } == [1, 2])

    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)
    let reserved = streamer.expertCachePlanBuffers(plan)

    let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
    #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

    let executed = try streamer.executeExpertCachePlan(plan)
    for i in 0..<experts.count {
      #expect(reserved[i].buffer === executed[i].buffer)
      let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
    }
  }

  @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCached(
      experts: [0, 2],
      avoidingSlots: [0, 1])

    #expect(plan.assignedSlots == [0, 2])
    #expect(plan.hits == 1)
    #expect(plan.misses == [1])

    let executed = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in plan.experts.enumerated() {
      let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }

    let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
    #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
  }

  /// The invariant the `io` counters rest on.
  ///
  /// `RunnerCounterValues.expertHits/expertMisses` are read off one
  /// `planRoutedExperts` call per layer, and the byte figure is
  /// `misses x expertStride` — which is only the *whole* routed-expert traffic
  /// if every expert the layer routes to is either a hit or a miss. A plan that
  /// dropped an expert (or listed one twice) would make the readout look
  /// plausible while under- or over-counting the disk it is meant to explain.
  ///
  /// Every plan the streamer can return has to satisfy this, not just the
  /// no-cache-only plan, so both the cache-warm and the cold path are checked
  /// here — the cold one also pins that `misses` is not silently the empty set
  /// when nothing is resident.
  @Test func everyPlannedExpertIsEitherAHitOrAMiss() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    // Cold: nothing resident, so every expert is a miss and the miss list is
    // the whole request.
    let cold = streamer.planExpertsCached(experts: [0, 1, 2, 3])
    #expect(cold.hits == 0)
    #expect(cold.misses.count == cold.experts.count)
    #expect(cold.hits + cold.misses.count == cold.experts.count)

    // Warm: a mixed plan, where the split is the thing under test.
    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let mixed = streamer.planExpertsCached(experts: [0, 2])
    #expect(mixed.hits == 1)
    #expect(mixed.misses == [1])
    #expect(mixed.hits + mixed.misses.count == mixed.experts.count)

    // And the assignment the byte figure assumes: one slot per requested
    // expert, so `assignedSlots` and `experts` stay in step.
    #expect(mixed.assignedSlots.count == mixed.experts.count)
  }

  @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCachedIfPossible(
      experts: [0, 2, 3, 4],
      avoidingSlots: [0, 1])

    #expect(plan == nil)
  }

}

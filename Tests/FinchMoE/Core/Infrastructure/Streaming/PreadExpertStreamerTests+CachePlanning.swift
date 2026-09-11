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

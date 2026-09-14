import Testing
@testable import FinchMoECLICore
@testable import FinchMoE

/// The formatter is a plain function over a POD, so none of this needs a
/// Metal device or an install — which is the reason `RunnerCounterValues` is
/// mirrored into the CLI target rather than reached through the runner.
@Suite struct RunnerCountersTests {
    /// The last field carries the line's closing bracket, so trim it here
    /// rather than making every caller remember to.
    private static func field(_ key: String, in line: String) -> String? {
        for token in line.split(separator: " ") {
            let trimmed = token.hasSuffix("]") ? token.dropLast() : token[...]
            let parts = trimmed.split(separator: "=", maxSplits: 1)
            if parts.count == 2, parts[0] == Substring(key) { return String(parts[1]) }
        }
        return nil
    }

    /// A snapshot that tiles exactly, as a real run's must.
    private static func tiled(forwards: UInt64,
                              cb1: UInt64,
                              misses: UInt64 = 0) -> RunnerCounterValues {
        // other carries whatever is left after the named buckets, so the sum is
        // `cb1` by construction — the same way the cursor produces it.
        let attention: UInt64 = 8_000
        let gdnProj: UInt64 = 3_000
        let gdnConvGate: UInt64 = 1_000
        let gdnRecurrent: UInt64 = 5_000
        let router: UInt64 = 2_000
        let other = cb1 - (attention + gdnProj + gdnConvGate + gdnRecurrent + router)
        return RunnerCounterValues(forwards: forwards,
                                   cb1Nanos: cb1,
                                   cb1OtherNanos: other,
                                   cb1AttentionNanos: attention,
                                   cb1GdnProjNanos: gdnProj,
                                   cb1GdnConvGateNanos: gdnConvGate,
                                   cb1GdnRecurrentNanos: gdnRecurrent,
                                   cb1RouterNanos: router,
                                   cb1WaitNanos: 40_000,
                                   expertMisses: misses)
    }

    @Test func timingsDivideByTheForwardCount() {
        let line = RunnerCounters.line(Self.tiled(forwards: 8, cb1: 80_000_000),
                                       expertStride: nil)
        // 80 ms over 8 steps is 10 ms per step, not 80.
        #expect(Self.field("cb1_cpu_ms/step", in: line) == "10.00")
        #expect(Self.field("forwards", in: line) == "8")
        #expect(Self.field("identity", in: line) == "exact")
    }

    /// `forwards == 0` is the *normal* case for a short greedy run, not an edge
    /// case: the toy smoke test stops on its first token, and a prefill seed
    /// can produce a token with no forward at all. It must not divide by zero.
    ///
    /// `forwards == 1` is checked beside it because it is the other way the
    /// division is easy to get wrong — the per-step value and the total are the
    /// same number there, so a formatter that forgot to divide would pass.
    @Test func zeroAndOneForwardDoNotDivideByZero() {
        let zero = RunnerCounters.line(Self.tiled(forwards: 0, cb1: 80_000_000),
                                       expertStride: nil)
        #expect(Self.field("forwards", in: zero) == "0")
        #expect(Self.field("cb1_cpu_ms/step", in: zero) == "80.00")
        #expect(Self.field("cbs/step", in: zero) == "0.0")

        let one = RunnerCounters.line(Self.tiled(forwards: 1, cb1: 80_000_000),
                                      expertStride: nil)
        #expect(Self.field("cb1_cpu_ms/step", in: one) == "80.00")
    }

    @Test func missBytesUseTheSameUnitAsTheAdviceCounter() {
        let stride: UInt64 = 2_768_896
        let line = RunnerCounters.line(
            Self.tiled(forwards: 2, cb1: 80_000_000, misses: 40),
            expertStride: stride)
        // 40 misses x 2,768,896 B = 110,755,840 B = 105.625 MiB, / 2 steps.
        #expect(Self.field("expert_stride", in: line) == "2768896")
        #expect(Self.field("io_mb/step", in: line) == "52.8")
        #expect(Self.field("misses", in: line) == "40")
    }

    /// Without a stride the byte figure is absent rather than zero — a model
    /// whose stride was never consulted has no bytes to report, and printing
    /// `io_mb/step=0.0` would read as a real measurement of no I/O.
    @Test func byteFigureIsOmittedWithoutAStride() {
        let line = RunnerCounters.line(Self.tiled(forwards: 2, cb1: 80_000_000),
                                       expertStride: nil)
        #expect(Self.field("io_mb/step", in: line) == nil)
        #expect(Self.field("expert_stride", in: line) == nil)
    }

    /// A mis-tiled snapshot has to stay visible. Clamping it would turn the one
    /// check that can catch a double-counted span into a field that always
    /// agrees.
    @Test func aSumAboveCb1IsReportedNotClamped() {
        var values = Self.tiled(forwards: 4, cb1: 40_000)
        values.cb1RouterNanos &+= 5_000        // now tiles to 45,000
        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("identity", in: line) == "OFF(5000ns)")

        var short = Self.tiled(forwards: 4, cb1: 40_000)
        short.cb1OtherNanos -= 900             // a gap the cursor cannot leave
        let gapLine = RunnerCounters.line(short, expertStride: nil)
        #expect(Self.field("identity", in: gapLine) == "OFF(-900ns)")
    }

    /// Gemma is not instrumented, so a Gemma run reports zero buckets against a
    /// real `cb1`. "The sum matches" is vacuous there and must not print as
    /// `exact`, which would claim the split was verified.
    @Test func anUninstrumentedFamilyDoesNotClaimAnExactSplit() {
        let line = RunnerCounters.line(
            RunnerCounterValues(forwards: 3, cb1Nanos: 90_000_000),
            expertStride: nil)
        #expect(Self.field("identity", in: line) == "none")
        #expect(Self.field("cb1_cpu_ms/step", in: line) == "30.00")
    }

    @Test func deltaIsPerCounterAndSaturatesAtZero() {
        let base = Self.tiled(forwards: 2, cb1: 80_000, misses: 10)
        var now = Self.tiled(forwards: 5, cb1: 200_000, misses: 46)
        now.cb1RouterNanos += 7_000

        let d = now.delta(from: base)
        #expect(d.forwards == 3)
        #expect(d.cb1Nanos == 120_000)
        #expect(d.expertMisses == 36)
        #expect(d.cb1RouterNanos == 7_000)

        // A counter that goes backwards is a snapshot-ordering bug, not a
        // negative duration: saturating keeps it from printing as ~1.8e19.
        #expect(base.delta(from: now).cb1Nanos == 0)
    }

    /// The GPU split reports both layer kinds, because they are alternatives: a
    /// layer runs the attention stack or the GDN stack, never both. On Qwen 3.8
    /// that is 12 full-attention layers against 36 GDN ones, so the pair is read
    /// per layer — dividing each by its own layer count — and a single summed
    /// `gpu_cb1` cannot say which stack owns the time.
    @Test func gpuSplitReportsBothLayerKinds() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.gpuCb1Nanos = 72_000_000
        values.gpuCb1FullAttnNanos = 12_000_000
        values.gpuCb1GdnNanos = 60_000_000
        let line = RunnerCounters.line(values, expertStride: nil)

        #expect(Self.field("gpu_cb1_wall_ms/step", in: line) == "36.00")
        #expect(Self.field("gpu_cb1_fullattn_wall_ms/step", in: line) == "6.00")
        #expect(Self.field("gpu_cb1_gdn_wall_ms/step", in: line) == "30.00")

        // Both buckets come from one `recordGpuTime` sample and a branch on the
        // layer kind, so they tile the total by construction — asserting the sum
        // is checking the wiring, not re-deriving a number the formatter made.
        #expect(values.gpuCb1FullAttnNanos + values.gpuCb1GdnNanos
                    == values.gpuCb1Nanos)

        // Per-counter like every other field, so a snapshot pair cannot drop
        // the new buckets silently.
        var later = values
        later.gpuCb1GdnNanos = 90_000_000
        #expect(later.delta(from: values).gpuCb1GdnNanos == 30_000_000)
        #expect(later.delta(from: values).gpuCb1FullAttnNanos == 0)
    }

    /// The `io` window's three parts leave a remainder, and that remainder is
    /// where a per-layer fixed cost would hide — the continuation hops, the
    /// `streamersQueue.sync`, the `ensureLayerOpened` check, all of which run
    /// once per layer per token. It is derived by subtraction, which is the
    /// arithmetic that wraps to ~1.8e19 in `UInt64`, so saturation is the
    /// property under test rather than an implementation detail.
    @Test func ioPartsTileTheWindowAndTheRemainderSaturates() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioNanos = 20_000_000
        values.ioDispatchNanos = 200_000
        values.ioReadNanos = 19_600_000
        values.ioTailNanos = 40_000
        // 20,000,000 - 19,840,000 = 160,000 ns of handoff, over 2 steps.
        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_wall_ms/step", in: line) == "10.00")
        #expect(Self.field("io_read_wall_ms/step", in: line) == "9.80")
        #expect(Self.field("io_handoff_wall_ms/step", in: line) == "0.08")

        // Parts that exceed the window cannot be subtracted. This is the shape a
        // snapshot takes when a fetch completes between the two reads, and a
        // wrapped `UInt64` would print as ~1.8e19 ms — reading as a units bug
        // rather than as the ordering bug it is.
        var incoherent = values
        incoherent.ioReadNanos = 30_000_000
        let wrapped = RunnerCounters.line(incoherent, expertStride: nil)
        #expect(Self.field("io_handoff_wall_ms/step", in: wrapped) == "0.00")
        // The parts themselves stay honest: clamping them to `io` would hide the
        // discrepancy rather than expose it.
        #expect(Self.field("io_read_wall_ms/step", in: wrapped) == "15.00")
    }

    /// `plan` is CPU work *between* the router readback and the fetch, so it
    /// happens outside the window and must not be subtracted from it. Tiling
    /// `io` with it would drive the remainder to zero on every real run — and
    /// the remainder is exactly the per-layer fixed cost this split exists to
    /// measure.
    @Test func planTimeIsOutsideTheWindowAndNeverSubtracted() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioNanos = 20_000_000
        values.ioDispatchNanos = 200_000
        values.ioReadNanos = 19_600_000
        values.ioTailNanos = 40_000
        values.ioPlanNanos = 5_000_000

        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_plan_cpu_ms/step", in: line) == "2.50")
        #expect(Self.field("io_handoff_wall_ms/step", in: line) == "0.08")
    }

    /// The read split tiles `io_read_wall` exactly, and `io_conc` is the summed
    /// thread time over the span rather than a fourth part of the window.
    ///
    /// The distinction is the whole point of the field: `io_thread_wall` is
    /// normally *larger* than the span it sits inside, so a reader who added it
    /// to the other three would get a total that doubles the read. The test
    /// pins both the tiling and the fact that the thread figure is outside it.
    @Test func readSplitTilesTheReadWindowAndConcurrencyIsARatio() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioReadNanos = 12_000_000
        values.ioFanoutNanos = 1_000_000
        values.ioSpanNanos = 9_000_000
        values.ioDrainNanos = 2_000_000
        // Six iterations' worth of thread time inside a 9 ms span: 2.33 deep.
        values.ioThreadNanos = 21_000_000

        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_read_identity", in: line) == "exact")
        #expect(Self.field("io_conc", in: line) == "2.33")
        #expect(Self.field("io_span_wall_ms/step", in: line) == "4.50")
        #expect(Self.field("io_thread_wall_ms/step", in: line) == "10.50")
        // Thread time exceeds the read window it was spent inside, which is the
        // property that makes it not a part of that window.
        #expect(values.ioThreadNanos > values.ioReadNanos)
    }

    /// A snapshot taken across a fetch in flight shows up as a broken tiling
    /// rather than as a plausible wrong number, and the parts stay honest --
    /// clamping them would hide the ordering bug instead of exposing it.
    @Test func aMisTiledReadSplitIsReportedNotClamped() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioReadNanos = 10_000_000
        values.ioFanoutNanos = 4_000_000
        values.ioSpanNanos = 9_000_000
        values.ioDrainNanos = 2_000_000

        let line = RunnerCounters.line(values, expertStride: nil)
        // 15,000,000 - 10,000,000 = 5,000,000 ns over the window.
        #expect(Self.field("io_read_identity", in: line) == "OFF(5000000ns)")
        #expect(Self.field("io_read_wall_ms/step", in: line) == "5.00")
    }

    /// The pread/copy pair is a split of `io_thread_wall`, so the two must not
    /// be added to it and must not be read as a timeline of their own.
    ///
    /// With staging off the streamer charges the whole read to `pread` and
    /// reports no copy, which is what makes the pair sum to the thread time
    /// rather than to something smaller; a run that reported both halves of an
    /// unstaged read would be inventing a copy that never happened.
    @Test func thePreadCopySplitChargesAnUnstagedReadEntirelyToPread() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioReadNanos = 12_000_000
        values.ioSpanNanos = 9_000_000
        values.ioThreadNanos = 21_000_000
        values.ioPreadNanos = 21_000_000
        values.ioCopyNanos = 0

        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_pread_wall_ms/step", in: line) == "10.50")
        #expect(Self.field("io_copy_wall_ms/step", in: line) == "0.00")
        // Both halves divide by `forwards` like every other timing field, and
        // together they tile the thread time they were split out of.
        #expect(values.ioPreadNanos + values.ioCopyNanos == values.ioThreadNanos)
    }

    /// Staged, the two spans are disjoint sub-intervals of the iteration, so
    /// they fall short of the thread time by the loop's own bounds check and
    /// offset arithmetic. The gap is expected, not an error -- which is the
    /// opposite of the `io_read_identity` case, where any gap is a bug.
    @Test func aStagedReadLeavesTheLoopOverheadUnattributed() {
        var values = Self.tiled(forwards: 2, cb1: 80_000)
        values.ioReadNanos = 12_000_000
        values.ioFanoutNanos = 1_000_000
        values.ioSpanNanos = 9_000_000
        values.ioDrainNanos = 2_000_000
        values.ioThreadNanos = 21_000_000
        values.ioPreadNanos = 20_000_000
        values.ioCopyNanos = 500_000

        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_pread_wall_ms/step", in: line) == "10.00")
        #expect(Self.field("io_copy_wall_ms/step", in: line) == "0.25")
        #expect(values.ioPreadNanos + values.ioCopyNanos < values.ioThreadNanos)
        // The read tiling is unaffected by the operational split: staging
        // changes where the bytes land, not the shape of the window.
        #expect(Self.field("io_read_identity", in: line) == "exact")
    }

    /// An uninstrumented run reports zeros, and zero tiles zero -- so the
    /// identity must not claim `exact` for a family that measured nothing. This
    /// mirrors `identity=none` on the `cb1` side, and matters for the same
    /// reason: Gemma is uninstrumented on both.
    @Test func anUnmeasuredReadSplitDoesNotClaimAnExactTiling() {
        let values = Self.tiled(forwards: 2, cb1: 80_000)
        let line = RunnerCounters.line(values, expertStride: nil)
        #expect(Self.field("io_read_identity", in: line) == "none")
        // No span to divide by, so the ratio is absent rather than infinite.
        #expect(Self.field("io_conc", in: line) == "n/a")
    }
}

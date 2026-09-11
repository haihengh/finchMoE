import Foundation
import FinchMoE

/// A whole-number snapshot of `RealForwardRunner`'s decode counters.
///
/// Deliberately a plain POD with a memberwise init: the app target's
/// `RunnerCounterSnapshot` (`FinchMoEApp/Core/Inference/RealInferenceClient.swift`)
/// is private to that module, so the half that is worth sharing is mirrored
/// here rather than imported. Keeping it free of `RealForwardRunner` also makes
/// the formatter testable without a Metal device, which is the whole reason
/// `RunnerCounters.line` takes this type and not the runner.
public struct RunnerCounterValues: Equatable, Sendable {
    public var forwards: UInt64
    public var cb1Nanos: UInt64
    public var cb1OtherNanos: UInt64
    public var cb1AttentionNanos: UInt64
    public var cb1GdnProjNanos: UInt64
    public var cb1GdnConvGateNanos: UInt64
    public var cb1GdnRecurrentNanos: UInt64
    public var cb1RouterNanos: UInt64
    public var cb1WaitNanos: UInt64
    public var cb2Nanos: UInt64
    public var ioNanos: UInt64
    public var ioPlanNanos: UInt64
    public var ioDispatchNanos: UInt64
    public var ioReadNanos: UInt64
    public var ioTailNanos: UInt64
    public var ioFanoutNanos: UInt64
    public var ioSpanNanos: UInt64
    public var ioDrainNanos: UInt64
    public var ioThreadNanos: UInt64
    public var headNanos: UInt64
    public var expertHits: UInt64
    public var expertMisses: UInt64
    public var pleGatherNanos: UInt64
    public var plePartOpens: UInt64
    public var indexerRankedLayers: UInt64
    public var commandBuffers: UInt64
    public var gpuCb1Nanos: UInt64
    public var gpuCb1FullAttnNanos: UInt64
    public var gpuCb1GdnNanos: UInt64
    public var gpuRoutedNanos: UInt64
    public var gpuSamples: UInt64

    public init(forwards: UInt64 = 0,
                cb1Nanos: UInt64 = 0,
                cb1OtherNanos: UInt64 = 0,
                cb1AttentionNanos: UInt64 = 0,
                cb1GdnProjNanos: UInt64 = 0,
                cb1GdnConvGateNanos: UInt64 = 0,
                cb1GdnRecurrentNanos: UInt64 = 0,
                cb1RouterNanos: UInt64 = 0,
                cb1WaitNanos: UInt64 = 0,
                cb2Nanos: UInt64 = 0,
                ioNanos: UInt64 = 0,
                ioPlanNanos: UInt64 = 0,
                ioDispatchNanos: UInt64 = 0,
                ioReadNanos: UInt64 = 0,
                ioTailNanos: UInt64 = 0,
                ioFanoutNanos: UInt64 = 0,
                ioSpanNanos: UInt64 = 0,
                ioDrainNanos: UInt64 = 0,
                ioThreadNanos: UInt64 = 0,
                headNanos: UInt64 = 0,
                expertHits: UInt64 = 0,
                expertMisses: UInt64 = 0,
                pleGatherNanos: UInt64 = 0,
                plePartOpens: UInt64 = 0,
                indexerRankedLayers: UInt64 = 0,
                commandBuffers: UInt64 = 0,
                gpuCb1Nanos: UInt64 = 0,
                gpuCb1FullAttnNanos: UInt64 = 0,
                gpuCb1GdnNanos: UInt64 = 0,
                gpuRoutedNanos: UInt64 = 0,
                gpuSamples: UInt64 = 0) {
        self.forwards = forwards
        self.cb1Nanos = cb1Nanos
        self.cb1OtherNanos = cb1OtherNanos
        self.cb1AttentionNanos = cb1AttentionNanos
        self.cb1GdnProjNanos = cb1GdnProjNanos
        self.cb1GdnConvGateNanos = cb1GdnConvGateNanos
        self.cb1GdnRecurrentNanos = cb1GdnRecurrentNanos
        self.cb1RouterNanos = cb1RouterNanos
        self.cb1WaitNanos = cb1WaitNanos
        self.cb2Nanos = cb2Nanos
        self.ioNanos = ioNanos
        self.ioPlanNanos = ioPlanNanos
        self.ioDispatchNanos = ioDispatchNanos
        self.ioReadNanos = ioReadNanos
        self.ioTailNanos = ioTailNanos
        self.ioFanoutNanos = ioFanoutNanos
        self.ioSpanNanos = ioSpanNanos
        self.ioDrainNanos = ioDrainNanos
        self.ioThreadNanos = ioThreadNanos
        self.headNanos = headNanos
        self.expertHits = expertHits
        self.expertMisses = expertMisses
        self.pleGatherNanos = pleGatherNanos
        self.plePartOpens = plePartOpens
        self.indexerRankedLayers = indexerRankedLayers
        self.commandBuffers = commandBuffers
        self.gpuCb1Nanos = gpuCb1Nanos
        self.gpuCb1FullAttnNanos = gpuCb1FullAttnNanos
        self.gpuCb1GdnNanos = gpuCb1GdnNanos
        self.gpuRoutedNanos = gpuRoutedNanos
        self.gpuSamples = gpuSamples
    }

    public static let zero = RunnerCounterValues()

    /// Component-wise difference, saturating at zero.
    ///
    /// Saturating rather than `&-`: a counter that went backwards means the
    /// snapshot is not a prefix of the later one, and a wrapped `UInt64` would
    /// print as ~1.8e19 — a number that reads like a units bug rather than like
    /// the mistake it is. Zero is at least visibly wrong for a timing field.
    public func delta(from base: RunnerCounterValues) -> RunnerCounterValues {
        func d(_ now: UInt64, _ was: UInt64) -> UInt64 { now > was ? now - was : 0 }
        return RunnerCounterValues(
            forwards: d(forwards, base.forwards),
            cb1Nanos: d(cb1Nanos, base.cb1Nanos),
            cb1OtherNanos: d(cb1OtherNanos, base.cb1OtherNanos),
            cb1AttentionNanos: d(cb1AttentionNanos, base.cb1AttentionNanos),
            cb1GdnProjNanos: d(cb1GdnProjNanos, base.cb1GdnProjNanos),
            cb1GdnConvGateNanos: d(cb1GdnConvGateNanos, base.cb1GdnConvGateNanos),
            cb1GdnRecurrentNanos: d(cb1GdnRecurrentNanos, base.cb1GdnRecurrentNanos),
            cb1RouterNanos: d(cb1RouterNanos, base.cb1RouterNanos),
            cb1WaitNanos: d(cb1WaitNanos, base.cb1WaitNanos),
            cb2Nanos: d(cb2Nanos, base.cb2Nanos),
            ioNanos: d(ioNanos, base.ioNanos),
            ioPlanNanos: d(ioPlanNanos, base.ioPlanNanos),
            ioDispatchNanos: d(ioDispatchNanos, base.ioDispatchNanos),
            ioReadNanos: d(ioReadNanos, base.ioReadNanos),
            ioTailNanos: d(ioTailNanos, base.ioTailNanos),
            ioFanoutNanos: d(ioFanoutNanos, base.ioFanoutNanos),
            ioSpanNanos: d(ioSpanNanos, base.ioSpanNanos),
            ioDrainNanos: d(ioDrainNanos, base.ioDrainNanos),
            ioThreadNanos: d(ioThreadNanos, base.ioThreadNanos),
            headNanos: d(headNanos, base.headNanos),
            expertHits: d(expertHits, base.expertHits),
            expertMisses: d(expertMisses, base.expertMisses),
            pleGatherNanos: d(pleGatherNanos, base.pleGatherNanos),
            plePartOpens: d(plePartOpens, base.plePartOpens),
            indexerRankedLayers: d(indexerRankedLayers, base.indexerRankedLayers),
            commandBuffers: d(commandBuffers, base.commandBuffers),
            gpuCb1Nanos: d(gpuCb1Nanos, base.gpuCb1Nanos),
            gpuCb1FullAttnNanos: d(gpuCb1FullAttnNanos, base.gpuCb1FullAttnNanos),
            gpuCb1GdnNanos: d(gpuCb1GdnNanos, base.gpuCb1GdnNanos),
            gpuRoutedNanos: d(gpuRoutedNanos, base.gpuRoutedNanos),
            gpuSamples: d(gpuSamples, base.gpuSamples))
    }
}

extension RunnerCounterValues {
    /// The only Metal-touching part of this file: everything else works on the
    /// POD above, so the formatter can be tested without a device.
    public init(_ runner: RealForwardRunner) {
        self.init(forwards: runner.totalForwards,
                  cb1Nanos: runner.totalCb1Nanos,
                  cb1OtherNanos: runner.totalCb1OtherNanos,
                  cb1AttentionNanos: runner.totalCb1AttentionNanos,
                  cb1GdnProjNanos: runner.totalCb1GdnProjNanos,
                  cb1GdnConvGateNanos: runner.totalCb1GdnConvGateNanos,
                  cb1GdnRecurrentNanos: runner.totalCb1GdnRecurrentNanos,
                  cb1RouterNanos: runner.totalCb1RouterNanos,
                  cb1WaitNanos: runner.totalCb1WaitNanos,
                  cb2Nanos: runner.totalCb2Nanos,
                  ioNanos: runner.totalIoNanos,
                  ioPlanNanos: runner.totalIoPlanNanos,
                  ioDispatchNanos: runner.totalIoDispatchNanos,
                  ioReadNanos: runner.totalIoReadNanos,
                  ioTailNanos: runner.totalIoTailNanos,
                  ioFanoutNanos: runner.totalIoFanoutNanos,
                  ioSpanNanos: runner.totalIoSpanNanos,
                  ioDrainNanos: runner.totalIoDrainNanos,
                  ioThreadNanos: runner.totalIoThreadNanos,
                  // Summed, as the app does: which of the two carries the head
                  // depends on whether the run took the fused-greedy path, and
                  // a reader wants the cost, not the path.
                  headNanos: runner.totalHeadNanos &+ runner.totalHeadFusedNanos,
                  expertHits: runner.totalExpertHits,
                  expertMisses: runner.totalExpertMisses,
                  pleGatherNanos: runner.totalPleGatherNanos,
                  plePartOpens: runner.totalPlePartOpens,
                  indexerRankedLayers: runner.totalIndexerRankedLayers,
                  commandBuffers: runner.totalDecodeCommandBuffers,
                  gpuCb1Nanos: runner.totalGpuCb1Nanos,
                  gpuCb1FullAttnNanos: runner.totalGpuCb1FullAttnNanos,
                  gpuCb1GdnNanos: runner.totalGpuCb1GdnNanos,
                  gpuRoutedNanos: runner.totalGpuRoutedNanos,
                  gpuSamples: runner.totalGpuSamples)
    }
}

/// Renders `RunnerCounterValues` as the one-line `--counters` readout.
public enum RunnerCounters {
    /// Nanoseconds per millisecond, as a `Double` divisor.
    private static let nanosPerMilli = 1_000_000.0

    /// What is left of the `io` window once its three measured parts are
    /// subtracted. Saturating for the same reason `delta` is: a negative here
    /// means the parts were snapshotted across a fetch that completed between
    /// the two reads, not that the window is shorter than its own contents, and
    /// a wrapped `UInt64` would print as ~1.8e19.
    private static func ioHandoff(_ values: RunnerCounterValues) -> UInt64 {
        let accounted = values.ioDispatchNanos &+ values.ioReadNanos
            &+ values.ioTailNanos
        return values.ioNanos > accounted ? values.ioNanos - accounted : 0
    }

    /// `io_thread / io_span`: how many reads were in flight at once, on average.
    ///
    /// A ratio and not a duration, which is why it carries no `_wall` suffix —
    /// it is not another part of the window. It is also the one quantity in the
    /// read split that no wall clock can show on its own: a batch that ran
    /// six-up and a batch that ran one at a time for six times as long have the
    /// same span, and only the summed thread time tells them apart.
    private static func ioConcurrency(_ values: RunnerCounterValues) -> String {
        guard values.ioSpanNanos > 0 else { return "n/a" }
        return String(format: "%.2f",
                      Double(values.ioThreadNanos) / Double(values.ioSpanNanos))
    }

    /// Whether `io_fanout + io_span + io_drain` is exactly `io_read`.
    ///
    /// It always should be: the streamer computes the three from one timestamp
    /// pair per miss, so they telescope into the batch window by construction.
    /// That is exactly why anything other than `exact` is worth printing — it
    /// means the snapshot was taken across a fetch in flight, and the four
    /// numbers beside it do not all belong to the same call.
    private static func ioReadIdentity(_ values: RunnerCounterValues) -> String {
        let tiled = values.ioFanoutNanos &+ values.ioSpanNanos &+ values.ioDrainNanos
        if tiled == values.ioReadNanos {
            return tiled == 0 ? "none" : "exact"
        }
        return "OFF(\(Int64(bitPattern: tiled) - Int64(bitPattern: values.ioReadNanos))ns)"
    }

    /// Fields that are per-step are suffixed `/step`; counts are not. Every
    /// duration key carries its clock kind, because these numbers are not one
    /// timescale:
    ///
    /// - `_cpu` — an encode-and-commit clock. It measures *encoding the
    ///   dispatch*, not running the kernel, and it excludes the pipeline wait
    ///   that `cb1` itself excludes. Printing these beside a wall clock without
    ///   the suffix is exactly the confusion `docs/SYSTEM_DESIGN.md` warns
    ///   about.
    /// - `_wall` — a wall clock. `io` is awaited read time and `head` wraps a
    ///   synchronous submit-and-wait, so both include the time they waited.
    ///
    /// `scope` says what the numbers cover. `decode` means the caller took a
    /// snapshot at the prefill/decode boundary and passed the difference, which
    /// is the only way the counts come out decode-only: an `.off` prefill runs
    /// the decode path once per prompt token.
    ///
    /// `identity` is the reconciliation check described in
    /// `docs/SYSTEM_DESIGN.md`. It compares `cb1` against the sum of its
    /// sub-buckets, which the cursor tiling makes exact — so anything other
    /// than `exact` means a span is double-counted and the split must not be
    /// quoted.
    /// `slots` is printed because it is the swept variable in the item 2.1
    /// study: a line without it is only identifiable by which file captured it.
    public static func line(_ values: RunnerCounterValues,
                            expertStride: UInt64?,
                            slots: Int? = nil,
                            scope: String = "decode") -> String {
        let steps = max(values.forwards, 1)
        let n = Double(steps)

        func msPerStep(_ nanos: UInt64) -> String {
            String(format: "%.2f", Double(nanos) / nanosPerMilli / n)
        }

        // The sub-buckets must tile `cb1` exactly: the cursor skips the wait
        // rather than attributing it, and `cb1` is computed from the same
        // clock reading the final lapse leaves behind.
        //
        // The discriminator is `tiled == 0`, not `cb1 == 0`. Only the Qwen
        // bodies are instrumented, so a Gemma run reports zero for every
        // bucket against a real `cb1` — and there "the sum matches" would be
        // vacuous. `other` is accumulated unconditionally in the instrumented
        // bodies precisely so that a zero sum can only mean "not instrumented",
        // never "every span happened to be empty".
        let tiled = values.cb1OtherNanos &+ values.cb1AttentionNanos
            &+ values.cb1GdnProjNanos &+ values.cb1GdnConvGateNanos
            &+ values.cb1GdnRecurrentNanos &+ values.cb1RouterNanos
        let identity: String
        if tiled == 0 {
            identity = "none"
        } else if tiled == values.cb1Nanos {
            identity = "exact"
        } else {
            identity = "OFF(\(Int64(bitPattern: tiled) - Int64(bitPattern: values.cb1Nanos))ns)"
        }

        var fields: [String] = [
            "scope=\(scope)",
            "forwards=\(values.forwards)",
        ]
        if let slots { fields.append("slots=\(slots)") }
        fields.append(contentsOf: [
            "cb1_cpu_ms/step=\(msPerStep(values.cb1Nanos))",
            "other_cpu_ms/step=\(msPerStep(values.cb1OtherNanos))",
            "attention_cpu_ms/step=\(msPerStep(values.cb1AttentionNanos))",
            "gdn_proj_cpu_ms/step=\(msPerStep(values.cb1GdnProjNanos))",
            "gdn_conv_gate_cpu_ms/step=\(msPerStep(values.cb1GdnConvGateNanos))",
            "gdn_recurrent_cpu_ms/step=\(msPerStep(values.cb1GdnRecurrentNanos))",
            "router_cpu_ms/step=\(msPerStep(values.cb1RouterNanos))",
            "wait_cpu_ms/step=\(msPerStep(values.cb1WaitNanos))",
            "identity=\(identity)",
            "cb2_cpu_ms/step=\(msPerStep(values.cb2Nanos))",
            "io_wall_ms/step=\(msPerStep(values.ioNanos))",
            // The `io` window split, and the one place the suffix rule earns
            // its keep: `plan` is CPU work that happens *outside* the window
            // (between the router readback and the fetch) and is printed beside
            // it rather than inside it. The three `_wall` terms below do not
            // tile `io` — `handoff` is the remainder, and it is the
            // continuation hops, the `streamersQueue.sync` and the
            // `ensureLayerOpened` check. That remainder is per-layer fixed cost
            // on a path that runs once per layer per token, which is the
            // quantity this split exists to expose.
            "io_plan_cpu_ms/step=\(msPerStep(values.ioPlanNanos))",
            "io_dispatch_wall_ms/step=\(msPerStep(values.ioDispatchNanos))",
            "io_read_wall_ms/step=\(msPerStep(values.ioReadNanos))",
            // The read window split. `fanout + span + drain` is `io_read_wall`
            // exactly, and `read_identity` is the check that says so; `io_conc`
            // is the achieved parallelism, summed thread time over the span.
            // The four together replace a single number that could not
            // distinguish a slow drive from a fan-out that never widened.
            "io_fanout_wall_ms/step=\(msPerStep(values.ioFanoutNanos))",
            "io_span_wall_ms/step=\(msPerStep(values.ioSpanNanos))",
            "io_drain_wall_ms/step=\(msPerStep(values.ioDrainNanos))",
            "io_thread_wall_ms/step=\(msPerStep(values.ioThreadNanos))",
            "io_conc=\(ioConcurrency(values))",
            "io_read_identity=\(ioReadIdentity(values))",
            "io_tail_wall_ms/step=\(msPerStep(values.ioTailNanos))",
            "io_handoff_wall_ms/step=\(msPerStep(ioHandoff(values)))",
            "head_wall_ms/step=\(msPerStep(values.headNanos))",
            "ple_wall_ms/step=\(msPerStep(values.pleGatherNanos))",
            "hits=\(values.expertHits)",
            "misses=\(values.expertMisses)",
            "ple_opens=\(values.plePartOpens)",
            "indexer_ranked=\(values.indexerRankedLayers)",
            "cbs=\(values.commandBuffers)",
            "cbs/step=\(String(format: "%.1f", Double(values.commandBuffers) / n))",
            "gpu_cb1_wall_ms/step=\(msPerStep(values.gpuCb1Nanos))",
            // The two stacks are alternatives, so these are compared per layer,
            // not per step: divide by the full-attention and GDN layer counts.
            // They sum to `gpu_cb1` on a Qwen install (and are both zero on
            // Gemma, which is not instrumented — the same signature its CPU
            // buckets show).
            "gpu_cb1_fullattn_wall_ms/step=\(msPerStep(values.gpuCb1FullAttnNanos))",
            "gpu_cb1_gdn_wall_ms/step=\(msPerStep(values.gpuCb1GdnNanos))",
            "gpu_routed_wall_ms/step=\(msPerStep(values.gpuRoutedNanos))",
            "gpu_samples=\(values.gpuSamples)",
        ])

        if let stride = expertStride {
            // Exact, not an estimate: every miss reads exactly one expert
            // stride (`PreadExpertStreamer.readFull(count:)`). Multiplied here,
            // at print time, rather than per layer — the only per-layer byte
            // accessor takes a lock and opens the layer, which would add the
            // very per-layer overhead item 2.2 exists to remove.
            let bytes = Double(values.expertMisses &* stride)
            fields.append("io_mb/step=\(String(format: "%.1f", bytes / 1_048_576.0 / n))")
            fields.append("expert_stride=\(stride)")
        }

        return "[counters " + fields.joined(separator: " ") + "]"
    }
}

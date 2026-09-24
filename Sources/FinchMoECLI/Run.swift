import Foundation
import Metal
import FinchMoE

private struct MessageJSON: Decodable {
    let role: String
    let content: String
}

/// Guards the one `--expert-cache-slots` value that is on the allowed list and
/// still fatal.
///
/// `RuntimeConfiguration.allowedExpertCacheSlots` is a fixed list, so `Args`
/// can reject everything off it — but it cannot reject a count that is merely
/// *too small for this model*: it parses before anything is loaded, and the
/// bound is the model's top-k. The failure it misses is not a thrown error:
/// `makeExpertCachePlan` answers a layer that routes to more experts than there
/// are slots with `preconditionFailure` (`PreadExpertStreamer.swift:219-220`),
/// which traps the process.
///
/// On the Qwen 3.8 install (`topKExperts: 10`) this makes 8 unusable and leaves
/// {16, 24, 32}; on Qwen 3.6 (top-k 8) all four are legal. Split out as a pure
/// function because the alternative is a test that has to trap to prove itself.
public enum ExpertCacheSlotCheck {
    public static func error(slots: Int, topKExperts: Int) -> String? {
        guard slots < topKExperts else { return nil }
        return "expert cache slots \(slots) is below this model's top-k "
            + "\(topKExperts); a single layer would route to more experts than "
            + "the cache can hold"
    }
}

public struct RunResult: Equatable, Sendable {
    public let exitCode: Int32
    public init(exitCode: Int32) { self.exitCode = exitCode }
}

/// `environment` is a parameter rather than a direct `ProcessInfo` read so a
/// test can drive the `FINCHMOE_EXPECT_ARCH` override without mutating global
/// process state — Swift Testing runs cases in parallel, so `setenv` in one
/// case would race every other case that reads the environment.
public func run(args: Args,
                stdout: FileHandle = .standardOutput,
                stderr: FileHandle = .standardError,
                environment: [String: String] = ProcessInfo.processInfo.environment) async -> RunResult {
    do {
        let modelURL = URL(fileURLWithPath: args.model)
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let promptIds: [Int32]
        if let rawPrompt = args.prompt {
            promptIds = tokenizer.encode(rawPrompt, addBOS: true)
        } else if let messagesFile = args.messagesFile {
            let data = try Data(contentsOf: URL(fileURLWithPath: messagesFile),
                                options: [.mappedIfSafe])
            let rows = try JSONDecoder().decode([MessageJSON].self, from: data)
            let messages = try rows.map { row -> GFTokenizer.Message in
                guard let role = GFTokenizer.Role(rawValue: row.role) else {
                    throw GFTokenizerError.invalidChatTemplate("unsupported role \(row.role)")
                }
                return GFTokenizer.Message(role: role, content: row.content)
            }
            let rendered = try tokenizer.applyChatTemplate(messages)
            promptIds = tokenizer.encode(rendered, addBOS: false)
        } else {
            return errored(stderr, "one of --prompt or --messages-file is required", 2)
        }
        guard !promptIds.isEmpty else { return errored(stderr, "empty prompt", 2) }
        guard promptIds.count < args.maxContext else {
            return errored(
                stderr,
                "context overflow: prompt \(promptIds.count) reaches maxContext \(args.maxContext)",
                2)
        }
        let effectiveMaxNew = min(args.maxNew, args.maxContext - promptIds.count)
        let config = GenerationConfig(
            maxNewTokens: effectiveMaxNew,
            temperature: args.temperature,
            topK: args.topK,
            topP: args.topP,
            repetitionPenalty: args.repetitionPenalty,
            seed: args.seed,
            stopStrings: args.stops,
            extraStopTokens: [])
        // A dump has to force the logits head. Under `--temperature 0` the
        // fused head never materializes logits, so the file would hold
        // whatever the buffer last did. `Sampler` keys greedy on temperature
        // alone, so forcing the head still yields the same argmax token — it
        // is just reached through the logits path instead of the fused one.
        let prefillLogitsDumpPath = environment["FQ_DUMP_PREFILL_LOGITS"]
        let runtime = RuntimeConfiguration(
            expertCacheSlots: args.expertCacheSlots ?? RuntimeConfiguration.production.expertCacheSlots,
            prefillChunkTokens: args.prefillChunkTokens,
            forceLogitsHead: !config.isPureGreedy || prefillLogitsDumpPath != nil,
            prefillTileDepth: args.prefillTileDepth,
            prefillTileExperts: args.prefillTileExperts,
            kvStorageMode: args.kvInt8 ? .int8 : .fp16)

        guard MTLCreateSystemDefaultDevice() != nil else {
            return errored(stderr, "no Metal device", 1)
        }
        let context = try MetalContext()
        let model = try Model.load(
            directoryURL: modelURL,
            device: context.device,
            expecting: try ManifestReader.detectPreset(
                directoryURL: modelURL,
                allowManifestArch: environment["FINCHMOE_EXPECT_ARCH"] == "1"),
            streamingMode: .pread(slotCount: runtime.expertCacheSlots),
            expertCachePolicy: runtime.modelExpertCachePolicy,
            integrityPolicy: args.verify)
        // A receipt that is present but unusable is the one case worth
        // interrupting for: the caller may have been relying on it. An absent
        // one is silent. Goes to the injected `stderr` so tests capture it, and
        // is not gated on `--quiet`, which documents itself as suppressing the
        // timing footer only.
        if let warning = model.integrityOutcome.warningMessage {
            stderr.write(Data("warning: \(warning)\n".utf8))
        }
        if let message = ExpertCacheSlotCheck.error(slots: runtime.expertCacheSlots,
                                                    topKExperts: model.config.topKExperts) {
            return errored(stderr, message, 2)
        }
        // The Phase 5.1 isolation knob changes what the model reads without
        // changing what is on disk, so a run has to say it was on: a dump whose
        // log does not name the mode cannot be compared against another.
        if let sim = environment["FQ_PLE_QUANT_SIM"], !sim.isEmpty {
            stderr.write(Data(("ple_quant_sim group=\(sim): PLE rows are quantized "
                + "and decoded in memory after the read; the installed table is "
                + "unchanged\n").utf8))
        }
        // Same reason: a run with the sparse-block selector disabled computes
        // different logits from one with it, so the log has to say which it is.
        if environment["FQ_QSA_OFF"] == "1" {
            stderr.write(Data(("qsa_off: built without the QSA indexer; attention "
                + "takes the dense path for every layer\n").utf8))
        }
        let runner = try RealForwardRunner(
            model: model,
            context: context,
            maxContext: args.maxContext,
            runtimeConfiguration: runtime)
        let scratch = try RawCompletionScratch(context: context,
                                               vocab: model.config.vocabSize,
                                               logitSoftcap: Float(model.config.finalLogitSoftcap))
        // Taken at the prefill/decode boundary, the same point the app uses.
        // Without it the counters would not be decode-only: the `.off` prefill
        // path runs the decode forward once per prompt token, so on that path
        // (and only that one) the totals would carry the prompt's work too.
        var countersAtDecodeStart: RunnerCounterValues?
        let stats = try await runRawCompletion(
            producer: runner,
            tokenizer: tokenizer,
            promptIds: promptIds,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: runtime.prefillConfig,
            prefillLogitsDumpPath: prefillLogitsDumpPath) { progress in
                switch progress {
                case .prefill(let done, let total):
                    if done == total { countersAtDecodeStart = RunnerCounterValues(runner) }
                case .token(_, let id, let delta):
                    if ProcessInfo.processInfo.environment["FQ_TOKEN_IDS"] != nil {
                        let piece = tokenizer.decode([id], skipSpecialTokens: false)
                        stdout.write(Data("\(id)[\(piece)] ".utf8))
                    }
                    if !delta.isEmpty { stdout.write(Data(delta.utf8)) }
                case .tail(let tail):
                    stdout.write(Data(tail.utf8))
                }
            }

        if ProcessInfo.processInfo.environment["FQ_PROMPT_IDS"] != nil {
            stdout.write(Data(("prefill: " + promptIds.map(String.init)
                .joined(separator: " ") + "\n").utf8))
            stdout.write(Data(("pieces: " + promptIds.map {
                "\($0)=[\(tokenizer.decode([$0], skipSpecialTokens: false))]"
            }.joined(separator: " ") + "\n").utf8))
        }
        if !args.quiet {
            let tokensPerSecond = stats.decodeSeconds > 0
                ? Double(stats.newTokens) / stats.decodeSeconds
                : 0
            let prefillPerSecond = stats.prefillSeconds > 0
                ? Double(stats.prefillTokens) / stats.prefillSeconds
                : 0
            let prefillText = stats.prefillSeconds > 0
                ? String(format: " prefill=%.2fs (%@tok/s)",
                         stats.prefillSeconds,
                         String(format: "%.1f", prefillPerSecond))
                : ""
            let footer = "\n[stop=\(String(describing: stats.reason)) prefill=\(stats.prefillTokens)tok new=\(stats.newTokens)tok decode=\(String(format: "%.2f", stats.decodeSeconds))s tok/s=\(String(format: "%.3f", tokensPerSecond))\(prefillText)]\n"
            stderr.write(Data(footer.utf8))
        }
        if args.counters {
            // A separate line, on its own gate: the footer above is greppable
            // by `docs/COMMUNITY_BENCHMARKS.md` and stays byte-identical.
            let now = RunnerCounterValues(runner)
            let values = countersAtDecodeStart.map { now.delta(from: $0) } ?? now
            let line = RunnerCounters.line(
                values,
                expertStride: model.routedExpertStrideBytes(),
                slots: runtime.expertCacheSlots,
                scope: countersAtDecodeStart == nil ? "whole-run" : "decode")
            stderr.write(Data((line + "\n").utf8))
            // The prefill's own breakdown, from the snapshot taken at the
            // prefill/decode boundary. Without it a prefill-dominated run
            // reports only the decode delta -- all zeros when there is little
            // decode -- and "how much of the prefill is GPU, how much is I/O,
            // how much is the host" is exactly the question the chunk-size and
            // tile-vs-whole-layer work turns on.
            if let atDecodeStart = countersAtDecodeStart {
                let prefillLine = RunnerCounters.line(
                    atDecodeStart,
                    expertStride: model.routedExpertStrideBytes(),
                    slots: runtime.expertCacheSlots,
                    scope: "prefill")
                stderr.write(Data((prefillLine + "\n").utf8))
            }
        }
        // Prefill-side I/O, on its own opt-in gate rather than appended to the
        // documented `--counters` schema: the chunk-size question is "does the
        // expert read volume fall as the chunk grows", and that is a different
        // measurement from the decode breakdown.
        if environment["FQ_PREFILL_COUNTERS"] != nil {
            let stride = UInt64(model.routedExpertStrideBytes())
            let bytes = runner.totalPrefillExpertMisses &* stride
            let chunks = runner.totalPrefillChunks
            let perChunkMB = chunks > 0
                ? Double(bytes) / Double(chunks) / 1_048_576.0
                : 0
            let line = String(
                format: "prefill_counters chunks=%llu tiles=%llu expert_misses=%llu bytes=%.2fGB MB_per_chunk=%.1f",
                chunks,
                runner.totalPrefillTiles,
                runner.totalPrefillExpertMisses,
                Double(bytes) / 1_073_741_824.0,
                perChunkMB)
            stderr.write(Data((line + "\n").utf8))
        }
        // The engine's own pread sequence, for the offline replay. Written on
        // its own gate and after the footer, because it is not a measurement
        // but an input to one: the offline harness prices the drive on the
        // offsets the engine actually issued rather than on a synthetic draw.
        if let tracePath = ProcessInfo.processInfo.environment["FQ_EXPERT_TRACE"] {
            try runner.writeExpertTrace(to: tracePath)
        }
        return RunResult(exitCode: 0)
    } catch is CancellationError {
        stdout.write(Data("\n".utf8))
        return RunResult(exitCode: 130)
    } catch {
        return errored(stderr, "\(error)", 1)
    }
}

private func errored(_ stderr: FileHandle, _ message: String, _ code: Int32) -> RunResult {
    stderr.write(Data("error: \(message)\n".utf8))
    return RunResult(exitCode: code)
}

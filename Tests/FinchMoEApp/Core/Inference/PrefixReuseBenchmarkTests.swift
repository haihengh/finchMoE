import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

/// Cold-vs-hot PP/TG benchmark on the shipping installs.
///
/// Replays the frozen `real-generation-v1` cases — the same three the
/// published host rows use — through the shipping client at the same sampling
/// settings (app defaults, 128-token cap, 4K context). Per case: a discarded
/// warmup cold run, a measured cold run, and hot follow-ups that resume from
/// the measured run's KV. Install-gated; run explicitly:
///
///     swift test --filter PrefixReuseBenchmarkTests
///
/// Not part of the deterministic suite: it needs a real `.finch` install and
/// minutes of wall time. Numbers print as `[BENCH …]` lines.
@Suite struct PrefixReuseBenchmarkTests {
    struct Model {
        let label: String
        let path: String
        var installExists: Bool {
            FileManager.default.fileExists(atPath: path + "/manifest.json")
        }
    }

    static let models = [
        Model(label: "3.6-35B-A3B",
              path: "/Volumes/samsung 2t/code/finchMoE/models/Qwen3.6-35B-A3B-4bit.finch"),
        Model(label: "3.8-125B-ple4bit",
              path: "/Volumes/samsung 2t/code/finchMoE/models/Qwen3.8-Flash-Next-125B-ple4bit.finch"),
    ]
    static let caseIDs = ["short-explanation", "medium-review", "long-synthesis"]

    /// `FQ_BENCH_CASES=short-explanation,medium-review` narrows the sweep so a
    /// single case can be run on its own (the long 3.8 case is minutes of
    /// prefill by itself).
    static var selectedCases: [String] {
        guard let raw = ProcessInfo.processInfo.environment["FQ_BENCH_CASES"],
              !raw.isEmpty else { return caseIDs }
        return raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    /// `FQ_BENCH_NO_WARMUP=1` skips the discarded warmup for a case that would
    /// otherwise not fit a short measurement window; use it only when the page
    /// cache is already warm.
    static var warmupEnabled: Bool {
        ProcessInfo.processInfo.environment["FQ_BENCH_NO_WARMUP"] == nil
    }

    struct Turn {
        var text = ""
        var diagnostics: AppDiagnostics?
    }

    static func repoRoot() -> URL {
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
    }

    static func casePrompt(_ id: String) throws -> String {
        let url = repoRoot()
            .appendingPathComponent("docs/benchmark-prompts/real-generation-v1/\(id).json")
        let data = try Data(contentsOf: url)
        guard let messages = try JSONSerialization.jsonObject(with: data) as? [[String: Any]],
              let content = messages.first?["content"] as? String else {
            throw AppInferenceError.invalidRequest("cannot read benchmark case \(id)")
        }
        return content
    }

    static func runTurn(_ client: RealInferenceClient,
                        request: AppGenerationRequest) async throws -> Turn {
        var turn = Turn()
        for try await event in client.generate(request) {
            switch event {
            case .token(let token):
                turn.text += token.textDelta
            case .finished(let diagnostics), .cancelled(let diagnostics):
                turn.diagnostics = diagnostics
            case .failed(let error, let partial):
                turn.diagnostics = partial
                throw error
            case .prefillProgress:
                break
            }
        }
        return turn
    }

    static func request(model: Model,
                        prompt: String,
                        history: [AppChatTurn] = []) -> AppGenerationRequest {
        // The published protocol's sampling, which is also the app's default:
        // temperature 0.2, top-k 64, top-p 0.95, 128-token cap, 4K context.
        AppGenerationRequest(modelDirectory: URL(fileURLWithPath: model.path),
                             prompt: prompt,
                             history: history,
                             maxNewTokens: 128,
                             maxContextTokens: 4_096,
                             temperature: 0.2,
                             topK: 64,
                             topP: 0.95,
                             repetitionPenalty: 1)
    }

    static func rate(_ tokens: Int?, _ seconds: Double?) -> String {
        guard let tokens, let seconds, seconds > 0 else { return "n/a" }
        return String(format: "%.1f tok/s", Double(tokens) / seconds)
    }

    /// Results also go to a file, flushed per line: a run killed mid-sweep
    /// (these measurements are disk-bound and long) keeps every row it had
    /// already produced.
    static let resultsURL = repoRoot()
        .appendingPathComponent("benchmark-results/prefix-cold-hot.log")

    static func log(_ line: String) {
        print(line)
        let data = Data((line + "\n").utf8)
        if let handle = try? FileHandle(forWritingTo: resultsURL) {
            defer { try? handle.close() }
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        } else {
            try? FileManager.default.createDirectory(
                at: resultsURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try? data.write(to: resultsURL)
        }
    }

    static func coldRow(case id: String, turn: Turn) -> String {
        let d = turn.diagnostics
        return "[BENCH \(id)] cold pp: prompt=\(d?.promptTokenCount ?? -1)tok "
            + "prefill=\(String(format: "%.2f", d?.prefillSeconds ?? -1))s "
            + "(\(rate(d?.promptTokenCount, d?.prefillSeconds))) | "
            + "tg: new=\(d?.generatedTokens ?? -1)tok "
            + "\(rate(d?.generatedTokens, d?.decodeSeconds)) "
            + "decode=\(String(format: "%.2f", d?.decodeSeconds ?? -1))s"
    }

    static func hotRow(case id: String, turn: Turn, note: String = "") -> String {
        let d = turn.diagnostics
        let computed = d?.computedPromptTokenCount ?? -1
        return "[BENCH \(id)] hot\(note) pp: prompt=\(d?.promptTokenCount ?? -1)tok "
            + "cached=\(d?.cachedPromptTokens ?? -1) computed=\(computed) "
            + "prefill=\(String(format: "%.2f", d?.prefillSeconds ?? -1))s "
            + "(\(rate(computed, d?.prefillSeconds)) residual) | "
            + "tg: new=\(d?.generatedTokens ?? -1)tok "
            + "\(rate(d?.generatedTokens, d?.decodeSeconds)) "
            + "decode=\(String(format: "%.2f", d?.decodeSeconds ?? -1))s"
    }

    static func sweep(_ model: Model) async throws {
        let client = RealInferenceClient(promptReuseEnabled: true)
        // The published protocol samples (temperature 0.2), and a sampled
        // request needs the logits head — the load key must say so or the
        // first generation is refused as a settings change.
        try await client.ensureLoaded(
            modelDirectory: URL(fileURLWithPath: model.path),
            maxContextTokens: 4_096,
            options: AppRuntimeOptions(),
            forceLogitsHead: true) { _ in }

        // Awaited, not a detached task: the two model sweeps run back to back
        // and a 16 GB box must never hold both models at once.
        do {
            for id in Self.selectedCases {
                let prompt = try Self.casePrompt(id)

                // Discard a warmup cold run so the measured run is not paying
                // first-touch page-cache costs (the published protocol does the
                // same per case).
                if Self.warmupEnabled {
                    _ = try await Self.runTurn(
                        client, request: Self.request(model: model, prompt: prompt))
                }

                let cold = try await Self.runTurn(
                    client, request: Self.request(model: model, prompt: prompt))
                Self.log(Self.coldRow(case: id, turn: cold))

                // The hot follow-up is the 426-token medium prompt, so the
                // residual prefill rate is measured on real compute rather
                // than on fixed overhead, and the decode gets a full sample.
                let followUp = try Self.casePrompt("medium-review")
                let hot = try await Self.runTurn(
                    client,
                    request: Self.request(
                        model: model, prompt: followUp,
                        history: [
                            AppChatTurn(role: .user, text: prompt),
                            AppChatTurn(role: .assistant, text: cold.text),
                        ]))
                Self.log(Self.hotRow(case: id, turn: hot, note: "(+426tok follow-up)"))
            }
        } catch {
            await client.unload()
            throw error
        }
        await client.unload()
    }

    @Test(.enabled(if: models[0].installExists), .timeLimit(.minutes(60)))
    func qwen36ColdAndHot() async throws {
        Self.log("[BENCH model=\(Self.models[0].label)]")
        try await Self.sweep(Self.models[0])
    }

    @Test(.enabled(if: models[1].installExists), .timeLimit(.minutes(120)))
    func qwen38ColdAndHot() async throws {
        Self.log("[BENCH model=\(Self.models[1].label)]")
        try await Self.sweep(Self.models[1])
    }
}

import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

/// Heavy, install-gated: drives a real conversation through the resident
/// session and checks the second turn resumed from the first turn's KV, and
/// that resuming cost less and read the same as a full re-prefill of the
/// same conversation. Skipped unless the Qwen 3.6 install is on this machine
/// (shared runners never have it), like the engine's other install-gated
/// suites.
@Suite struct PrefixReuseInstallTests {
    static let installPath =
        "/Volumes/samsung 2t/code/finchMoE/models/Qwen3.6-35B-A3B-4bit.finch"

    static var installExists: Bool {
        FileManager.default.fileExists(atPath: installPath + "/manifest.json")
    }

    static let filler = ("The following is a reference document about storage "
        + ("A write-ahead log records every mutation before it reaches the "
           + "main data structure, so recovery can replay an incomplete tail. ")
            .repeated(times: 20)
        + "End of reference document.")

    struct Turn {
        var text = ""
        var diagnostics: AppDiagnostics?
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

    static func request(prompt: String,
                        history: [AppChatTurn],
                        maxNewTokens: Int = 32) -> AppGenerationRequest {
        AppGenerationRequest(modelDirectory: URL(fileURLWithPath: installPath),
                             prompt: prompt,
                             history: history,
                             maxNewTokens: maxNewTokens,
                             maxContextTokens: 4_096,
                             temperature: 0,
                             topK: nil,
                             topP: nil,
                             repetitionPenalty: 1)
    }

    static func load(_ client: RealInferenceClient) async throws {
        try await client.ensureLoaded(
            modelDirectory: URL(fileURLWithPath: installPath),
            maxContextTokens: 4_096,
            options: AppRuntimeOptions(),
            forceLogitsHead: false) { _ in }
    }

    @Test(.enabled(if: installExists), .timeLimit(.minutes(30)))
    func aFollowUpTurnReusesAndMatchesAFullPrefill() async throws {
        let firstPrompt = Self.filler + "\n\nReply with the single word: ready"
        let secondPrompt = "Reply with the single word: ok"

        // Arm A — reuse on: cold first turn, then a resumed follow-up.
        let reused = RealInferenceClient(promptReuseEnabled: true)
        try await Self.load(reused)
        let a1 = try await Self.runTurn(
            reused, request: Self.request(prompt: firstPrompt, history: []))
        let a1Diagnostics = try #require(a1.diagnostics)
        let aHistory = [
            AppChatTurn(role: .user, text: firstPrompt),
            AppChatTurn(role: .assistant, text: a1.text),
        ]
        let a2 = try await Self.runTurn(
            reused, request: Self.request(prompt: secondPrompt, history: aHistory))
        let a2Diagnostics = try #require(a2.diagnostics)
        print("reuse on : turn1 prefill=\(a1Diagnostics.prefillSeconds ?? -1)s "
              + "| turn2 prompt=\(a2Diagnostics.promptTokenCount ?? -1) "
              + "cached=\(a2Diagnostics.cachedPromptTokens ?? -1) "
              + "prefill=\(a2Diagnostics.prefillSeconds ?? -1)s "
              + "text=\(a2.text.prefix(40).debugDescription)")

        // The maxTokens bridge: a turn cut by the cap continues by prepending
        // the sampled boundary token before the new user turn.
        let cutPrompt = "Count: one two three"
        let cut = try await Self.runTurn(
            reused,
            request: Self.request(prompt: cutPrompt, history: [], maxNewTokens: 2))
        let cutDiagnostics = try #require(cut.diagnostics)
        #expect(cutDiagnostics.stopReason == .maxTokens)
        let cutFollowUp = try await Self.runTurn(
            reused,
            request: Self.request(
                prompt: "Reply with the single word: done",
                history: [
                    AppChatTurn(role: .user, text: cutPrompt),
                    AppChatTurn(role: .assistant, text: cut.text),
                ]))
        let cutFollowUpDiagnostics = try #require(cutFollowUp.diagnostics)
        print("reuse on : maxTokens-cut follow-up "
              + "cached=\(cutFollowUpDiagnostics.cachedPromptTokens ?? -1) "
              + "text=\(cutFollowUp.text.prefix(40).debugDescription)")
        #expect((cutFollowUpDiagnostics.cachedPromptTokens ?? 0) > 0)

        // The edited-history arm: an edit to the first turn must fall back to
        // a full prefill rather than resume from a prefix that no longer
        // exists.
        let editedFirst = Self.filler.replacingOccurrences(
            of: "storage", with: "storage-engine")
            + "\n\nReply with the single word: ready"
        let a3 = try await Self.runTurn(
            reused,
            request: Self.request(prompt: secondPrompt, history: [
                AppChatTurn(role: .user, text: editedFirst),
                AppChatTurn(role: .assistant, text: a1.text),
            ]))
        let a3Diagnostics = try #require(a3.diagnostics)
        print("reuse on : turn3 (edited history) "
              + "cached=\(a3Diagnostics.cachedPromptTokens ?? -1)")
        await reused.unload()

        // Arm B — reuse off: the same conversation, fully re-prefilled.
        let control = RealInferenceClient(promptReuseEnabled: false)
        try await Self.load(control)
        let b1 = try await Self.runTurn(
            control, request: Self.request(prompt: firstPrompt, history: []))
        let bHistory = [
            AppChatTurn(role: .user, text: firstPrompt),
            AppChatTurn(role: .assistant, text: b1.text),
        ]
        let b2 = try await Self.runTurn(
            control, request: Self.request(prompt: secondPrompt, history: bHistory))
        let b2Diagnostics = try #require(b2.diagnostics)
        print("reuse off: turn1 prefill=\(b1.diagnostics?.prefillSeconds ?? -1)s "
              + "| turn2 prompt=\(b2Diagnostics.promptTokenCount ?? -1) "
              + "cached=\(b2Diagnostics.cachedPromptTokens ?? -1) "
              + "prefill=\(b2Diagnostics.prefillSeconds ?? -1)s "
              + "text=\(b2.text.prefix(40).debugDescription)")
        await control.unload()

        #expect(a1Diagnostics.cachedPromptTokens == 0)
        #expect(!a1.text.isEmpty)

        // The reuse itself: most of the second prompt came back from the KV.
        let cached = try #require(a2Diagnostics.cachedPromptTokens)
        #expect(cached > 0)
        #expect(cached < (a2Diagnostics.promptTokenCount ?? 0))
        #expect(!a2.text.isEmpty)

        // Quality: the resumed turn reads exactly like the fully re-prefilled
        // one, and cost strictly less prefill.
        #expect(a2.text == b2.text)
        let reusedPrefill = try #require(a2Diagnostics.prefillSeconds)
        let controlPrefill = try #require(b2Diagnostics.prefillSeconds)
        #expect(reusedPrefill * 2 < controlPrefill)

        // Safety: the edited history did not resume from the stale prefix.
        #expect(a3Diagnostics.cachedPromptTokens == 0)
    }
}

private extension String {
    func repeated(times: Int) -> String {
        String(repeating: self, count: times)
    }
}

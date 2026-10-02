import Foundation
import Testing
@testable import FinchMoE
@testable import FinchMoEAppCore

@Suite struct SessionPromptCacheTests {
    private func key(maxContext: Int = 4_096,
                     forceLogitsHead: Bool = false) -> SessionLoadKey {
        SessionLoadKey(directory: URL(fileURLWithPath: "/models/qwen"),
                       maxContext: maxContext,
                       options: AppRuntimeOptions(),
                       forceLogitsHead: forceLogitsHead)
    }

    private func result(kvPosition: Int,
                        kvBackedCount: Int? = nil,
                        boundary: [Int32] = [99],
                        reason: StopReason = .endOfTurn) -> RawDecodeResult {
        let backedCount = kvBackedCount ?? kvPosition
        return RawDecodeResult(
            prefillTokens: kvPosition,
            cachedPromptTokens: 0,
            computedPrefillTokens: kvPosition,
            prefillSeconds: 1,
            newTokens: 1,
            decodeSeconds: 1,
            reason: reason,
            kvPosition: kvPosition,
            kvBackedTokenIDs: Array(0..<max(backedCount, 0)).map(Int32.init),
            uncommittedBoundaryTokenIDs: boundary)
    }

    private let firstQuestion = GFTokenizer.Message(role: .user, content: "First?")
    private let firstAnswer = GFTokenizer.Message(role: .assistant, content: "First answer")

    private func published(kvPosition: Int = 5,
                           reason: StopReason = .endOfTurn) -> SessionPromptCache {
        var cache = SessionPromptCache()
        cache.publish(sessionKey: key(),
                      messages: [firstQuestion],
                      assistantText: "First answer",
                      result: result(kvPosition: kvPosition, reason: reason))
        return cache
    }

    @Test func aStrictExtensionHitsTheFastPath() {
        let cache = published()

        #expect(cache.match(sessionKey: key(),
                            messages: [],
                            renderedPromptIDs: [0, 1, 2, 3, 4, 7, 8])
                == .prefixHit(cachedPromptTokens: 5))
    }

    @Test func aRenderedPrefixThatDivergesIsNotAFastPathHit() {
        let cache = published()

        #expect(cache.match(sessionKey: key(),
                            messages: [],
                            renderedPromptIDs: [0, 1, 2, 9, 4, 7]) == .miss)
    }

    @Test func theAssistantTurnPlusOneUserMessageContinues() {
        let cache = published()

        let outcome = cache.match(
            sessionKey: key(),
            messages: [firstQuestion, firstAnswer,
                       GFTokenizer.Message(role: .user, content: "Second?")],
            renderedPromptIDs: [0, 1, 2, 3, 4])
        #expect(outcome == .continuation(cachedPromptTokens: 5,
                                         kvBackedTokenIDs: [0, 1, 2, 3, 4],
                                         boundaryToken: 99,
                                         stopReason: .endOfTurn,
                                         userContent: "Second?"))
    }

    @Test func aMaxTokensCutCarriesItsStopReason() {
        let cache = published(kvPosition: 5, reason: .maxTokens)

        let outcome = cache.match(
            sessionKey: key(),
            messages: [firstQuestion, firstAnswer,
                       GFTokenizer.Message(role: .user, content: "Second?")],
            renderedPromptIDs: [0, 1, 2, 3, 4])
        guard case .continuation(let cached, _, _, let reason, let content) = outcome else {
            Issue.record("expected a continuation, got \(outcome)")
            return
        }
        #expect(cached == 5)
        #expect(reason == .maxTokens)
        #expect(content == "Second?")
    }

    @Test func anEditedAssistantTurnMisses() {
        let cache = published()

        #expect(cache.match(
            sessionKey: key(),
            messages: [firstQuestion,
                       GFTokenizer.Message(role: .assistant, content: "Edited answer"),
                       GFTokenizer.Message(role: .user, content: "Second?")],
            renderedPromptIDs: [0, 1, 2, 3, 4]) == .miss)
    }

    @Test func anEditedEarlierTurnMisses() {
        let cache = published()

        #expect(cache.match(
            sessionKey: key(),
            messages: [GFTokenizer.Message(role: .user, content: "Edited question"),
                       firstAnswer,
                       GFTokenizer.Message(role: .user, content: "Second?")],
            renderedPromptIDs: [0, 1, 2, 3, 4]) == .miss)
    }

    @Test func aContinuationThatIsNotOneUserMessageMisses() {
        let cache = published()

        // Nothing appended.
        #expect(cache.match(sessionKey: key(),
                            messages: [firstQuestion, firstAnswer],
                            renderedPromptIDs: [0, 1, 2, 3, 4]) == .miss)
        // More than one appended message.
        #expect(cache.match(
            sessionKey: key(),
            messages: [firstQuestion, firstAnswer,
                       GFTokenizer.Message(role: .user, content: "Second?"),
                       GFTokenizer.Message(role: .assistant, content: "Answer")],
            renderedPromptIDs: [0, 1, 2, 3, 4]) == .miss)
        // The appended message is not the user's.
        #expect(cache.match(
            sessionKey: key(),
            messages: [firstQuestion, firstAnswer,
                       GFTokenizer.Message(role: .assistant, content: "Second?")],
            renderedPromptIDs: [0, 1, 2, 3, 4]) == .miss)
    }

    @Test func aDifferentSessionKeyMisses() {
        let cache = published()

        #expect(cache.match(sessionKey: key(maxContext: 8_192),
                            messages: [],
                            renderedPromptIDs: [0, 1, 2, 3, 4, 7]) == .miss)
        #expect(cache.match(sessionKey: key(forceLogitsHead: true),
                            messages: [],
                            renderedPromptIDs: [0, 1, 2, 3, 4, 7]) == .miss)
    }

    @Test func publicationRefusesEndStatesItCannotVouchFor() {
        var cache = SessionPromptCache()

        // A stop-string cut cannot be replayed from the visible text.
        cache.publish(sessionKey: key(), messages: [firstQuestion],
                      assistantText: "First answer",
                      result: result(kvPosition: 5, reason: .stopString))
        #expect(cache.entry == nil)

        // The KV position disagrees with the tokens the KV is said to back.
        cache.publish(sessionKey: key(), messages: [firstQuestion],
                      assistantText: "First answer",
                      result: result(kvPosition: 5, kvBackedCount: 3))
        #expect(cache.entry == nil)

        // More than one token was sampled past the committed KV.
        cache.publish(sessionKey: key(), messages: [firstQuestion],
                      assistantText: "First answer",
                      result: result(kvPosition: 5, boundary: [99, 98]))
        #expect(cache.entry == nil)

        // Nothing was generated.
        cache.publish(sessionKey: key(), messages: [firstQuestion],
                      assistantText: "First answer",
                      result: result(kvPosition: 0))
        #expect(cache.entry == nil)
    }

    @Test func invalidateDropsTheEntry() {
        var cache = published()
        #expect(cache.entry != nil)

        cache.invalidate()
        #expect(cache.entry == nil)
        #expect(cache.match(sessionKey: key(), messages: [],
                            renderedPromptIDs: [0, 1, 2, 3, 4, 7]) == .miss)
    }
}

@Suite struct AppDiagnosticsPrefillAccountingTests {
    private func diagnostics(promptTokens: Int?,
                             cachedTokens: Int?,
                             prefillSeconds: Double?) -> AppDiagnostics {
        AppDiagnostics(generatedTokens: 3,
                       stopReason: .endOfTurn,
                       promptTokenCount: promptTokens,
                       cachedPromptTokens: cachedTokens,
                       prefillSeconds: prefillSeconds,
                       timeToFirstTokenSeconds: 0.1,
                       decodeSeconds: 1,
                       tokensPerSecond: 3,
                       peakMemoryBytes: nil,
                       runtimeOptions: AppRuntimeOptions())
    }

    @Test func computedTokensSubtractTheCachedPrefix() {
        let value = diagnostics(promptTokens: 1_004, cachedTokens: 964,
                                prefillSeconds: 2)
        #expect(value.computedPromptTokenCount == 40)
        #expect(value.prefillTokensPerSecond == 20)
    }

    @Test func withoutCachingTheAccountingIsUnchanged() {
        let value = diagnostics(promptTokens: 500, cachedTokens: nil,
                                prefillSeconds: 10)
        #expect(value.computedPromptTokenCount == 500)
        #expect(value.prefillTokensPerSecond == 50)
    }

    @Test func aFullyReusedPromptHasNoPrefillRate() {
        let value = diagnostics(promptTokens: 500, cachedTokens: 500,
                                prefillSeconds: 0.5)
        #expect(value.computedPromptTokenCount == 0)
        #expect(value.prefillTokensPerSecond == nil)
    }
}

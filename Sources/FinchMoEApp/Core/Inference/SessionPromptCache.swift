import Foundation
import FinchMoE

/// One-turn KV reuse for the app's resident session.
///
/// The app replays the whole conversation every turn, so without this every
/// message re-prefills the entire history — O(n²) prompt processing across a
/// chat, which on the shipping installs is tens of seconds per turn by the
/// middle of a conversation. The HTTP server has had the fix for a while
/// (`ServerPromptCache`, `single-prefix` mode); this is the same idea kept
/// where the app's generation actually runs: `RealInferenceSession`.
///
/// What is reused is what decode actually wrote. After a generation returns,
/// the entry records the conversation as it was rendered, the visible
/// assistant turn, the KV position and the token IDs the KV is known to back.
/// The next request resumes in one of two ways:
///
/// - **Prefix hit** — the newly rendered prompt strictly extends the cached
///   tokens. Cheap to check and the only path that works when the history
///   re-renders token-identically.
/// - **Continuation** — the conversation grew by exactly the cached assistant
///   turn plus one user message, but the re-rendered prompt cannot reproduce
///   the KV suffix, because the generation prompt ends with template
///   artifacts (Qwen's empty `<think>` block) that a re-rendered history does
///   not contain. As on the server, the prompt is then built from the *true*
///   token stream: the cached tokens, the uncommitted boundary token, and the
///   tokenizer's continuation for the new user message.
///
/// Everything else is a miss, and a miss costs only the reuse, never
/// correctness: the caller resets and prefills from scratch.
struct SessionPromptCache {
    struct Entry: Equatable {
        var sessionKey: SessionLoadKey
        /// The conversation as rendered for the generation, including the
        /// final user turn — what a continuation request must reproduce.
        var inputMessages: [GFTokenizer.Message]
        /// The assistant turn decode wrote, as the user saw it.
        var assistantTurn: GFTokenizer.Message
        var stopReason: StopReason
        var kvPosition: Int
        var kvBackedTokenIDs: [Int32]
        var uncommittedBoundaryTokenIDs: [Int32]
    }

    enum Outcome: Equatable {
        case miss
        /// The rendered prompt already extends the cached tokens; the caller
        /// prefills the rendered suffix.
        case prefixHit(cachedPromptTokens: Int)
        /// The caller must rebuild the prompt from the true token stream: the
        /// cached tokens followed by the bridge for `userContent` (the
        /// boundary token is prepended when the previous turn was cut by
        /// `maxTokens`; otherwise it is prepended by the bridge itself and the
        /// caller checks it agrees with `boundaryToken`).
        case continuation(cachedPromptTokens: Int,
                          kvBackedTokenIDs: [Int32],
                          boundaryToken: Int32,
                          stopReason: StopReason,
                          userContent: String)
    }

    /// The last generation's end state, while the session still holds it.
    private(set) var entry: Entry?

    mutating func invalidate() {
        entry = nil
    }

    /// Record the end state of a generation that finished cleanly.
    ///
    /// The guards mirror the server's `publish`: the KV must be exactly as
    /// long as the tokens it is known to back, nothing may have been sampled
    /// past the last committed token except the single boundary token, and the
    /// generation must have ended in a way whose text can be replayed — a
    /// stop-string cut cannot, which is why it is refused outright there.
    mutating func publish(sessionKey: SessionLoadKey,
                          messages: [GFTokenizer.Message],
                          assistantText: String,
                          result: RawDecodeResult) {
        guard result.kvPosition > 0,
              result.kvPosition == result.kvBackedTokenIDs.count,
              !result.kvBackedTokenIDs.isEmpty,
              result.uncommittedBoundaryTokenIDs.count == 1,
              result.reason == .endOfTurn
                || result.reason == .maxTokens else {
            entry = nil
            return
        }
        entry = Entry(
            sessionKey: sessionKey,
            inputMessages: messages,
            assistantTurn: GFTokenizer.Message(role: .assistant,
                                               content: assistantText),
            stopReason: result.reason,
            kvPosition: result.kvPosition,
            kvBackedTokenIDs: result.kvBackedTokenIDs,
            uncommittedBoundaryTokenIDs: result.uncommittedBoundaryTokenIDs)
    }

    /// How the next request should start.
    func match(sessionKey: SessionLoadKey,
               messages: [GFTokenizer.Message],
               renderedPromptIDs: [Int32]) -> Outcome {
        guard let entry,
              entry.sessionKey == sessionKey,
              entry.kvPosition == entry.kvBackedTokenIDs.count,
              entry.kvPosition > 0,
              entry.uncommittedBoundaryTokenIDs.count == 1 else {
            return .miss
        }

        if entry.kvPosition < renderedPromptIDs.count,
           renderedPromptIDs.prefix(entry.kvPosition)
            .elementsEqual(entry.kvBackedTokenIDs) {
            return .prefixHit(cachedPromptTokens: entry.kvPosition)
        }

        let inputCount = entry.inputMessages.count
        guard messages.count == inputCount + 2,
              messages.prefix(inputCount).elementsEqual(entry.inputMessages),
              assistantMatches(messages[inputCount], entry.assistantTurn),
              entry.stopReason == .endOfTurn || entry.stopReason == .maxTokens,
              let boundaryToken = entry.uncommittedBoundaryTokenIDs.first else {
            return .miss
        }
        let continuation = messages[inputCount + 1]
        guard continuation.role == .user,
              continuation.toolCalls.isEmpty,
              continuation.toolCallID == nil,
              let userContent = continuation.content else {
            return .miss
        }

        return .continuation(cachedPromptTokens: entry.kvPosition,
                             kvBackedTokenIDs: entry.kvBackedTokenIDs,
                             boundaryToken: boundaryToken,
                             stopReason: entry.stopReason,
                             userContent: userContent)
    }

    private func assistantMatches(_ incoming: GFTokenizer.Message,
                                  _ cached: GFTokenizer.Message) -> Bool {
        incoming.role == .assistant
            && cached.role == .assistant
            && incoming.toolCalls.isEmpty
            && cached.toolCalls.isEmpty
            && incoming.content == cached.content
    }
}

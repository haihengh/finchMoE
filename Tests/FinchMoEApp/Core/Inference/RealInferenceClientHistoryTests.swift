import Foundation
import Testing
import FinchMoE
import FinchMoEDecodeProtocol
@testable import FinchMoEAppCore

@Suite struct RealInferenceClientHistoryTests {
    @Test func chatMessagesPutHistoryBeforeTheNewQuestion() {
        let messages = RealInferenceSession.chatMessages(
            prompt: "Follow-up",
            history: [
                AppChatTurn(role: .user, text: "First"),
                AppChatTurn(role: .assistant, text: "First answer"),
            ])

        #expect(messages.count == 3)
        #expect(messages.map(\.role) == [.user, .assistant, .user])
        #expect(messages.map(\.content) == ["First", "First answer", "Follow-up"])
    }

    @Test func anEmptyHistoryStillSendsTheQuestion() {
        let messages = RealInferenceSession.chatMessages(prompt: "Only", history: [])

        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "Only")
    }

    @Test func trimmingKeepsEverythingWhenItAlreadyFits() {
        let history = [
            AppChatTurn(role: .user, text: "First"),
            AppChatTurn(role: .assistant, text: "First answer"),
            AppChatTurn(role: .user, text: "Second"),
            AppChatTurn(role: .assistant, text: "Second answer"),
        ]

        let kept = RealInferenceSession.trimmedHistory(history) { _ in true }

        #expect(kept == history)
    }

    @Test func trimmingDropsOldestExchangesUntilThePromptFits() {
        let history = [
            AppChatTurn(role: .user, text: "First"),
            AppChatTurn(role: .assistant, text: "First answer"),
            AppChatTurn(role: .user, text: "Second"),
            AppChatTurn(role: .assistant, text: "Second answer"),
            AppChatTurn(role: .user, text: "Third"),
            AppChatTurn(role: .assistant, text: "Third answer"),
        ]

        // Only the newest exchange plus the question fits.
        let kept = RealInferenceSession.trimmedHistory(history) { $0.count <= 2 }

        #expect(kept.map(\.text) == ["Third", "Third answer"])
    }

    @Test func trimmingRemovesWholeExchangesRatherThanLoneReplies() {
        let history = [
            AppChatTurn(role: .user, text: "First"),
            AppChatTurn(role: .assistant, text: "First answer"),
            AppChatTurn(role: .user, text: "Second"),
        ]

        // Nothing fits: the trim must not leave an assistant reply with the
        // question that produced it removed.
        let kept = RealInferenceSession.trimmedHistory(history) { $0.count <= 0 }

        #expect(kept.isEmpty)
    }

    @Test func aPromptWithHistoryDecodesFromAFrameWrittenWithoutIt() throws {
        // An older build's request has no `history` key; the service must still
        // accept it rather than failing the whole frame.
        let legacy = Data("""
        {
          "prompt": "Only the question",
          "maxNewTokens": 128,
          "maxContextTokens": 4096,
          "temperature": 0.2,
          "repetitionPenalty": 1,
          "runtimeOptions": {
            "expertCacheSlots": 16,
            "expertCachePolicy": "lfu",
            "prefillEnabled": true,
            "prefillChunkTokens": 512,
            "rdadvisePolicy": "off",
            "modelVerification": "full-sha256"
          },
          "generationID": "\(UUID().uuidString)"
        }
        """.utf8)

        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self,
            from: legacy)

        #expect(decoded.prompt == "Only the question")
        #expect(decoded.history.isEmpty)
    }

    @Test func historySurvivesAFrameRoundTrip() throws {
        let request = DecodeGenerationRequest(
            prompt: "Follow-up",
            history: [
                DecodeChatTurn(role: "user", text: "First"),
                DecodeChatTurn(role: "assistant", text: "First answer"),
            ],
            maxNewTokens: 128,
            maxContextTokens: 4_096,
            temperature: 0.2)

        let decoded = try JSONDecoder().decode(
            DecodeGenerationRequest.self,
            from: JSONEncoder().encode(request))

        #expect(decoded.history == request.history)
    }
}

import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct ChatSessionTests {
    @Test func derivedTitleUsesTheFirstUserLineCollapsed() {
        let session = ChatSession(messages: [
            ChatMessage(role: .user, text: "  Explain   the\nMoE router  "),
            ChatMessage(role: .assistant, text: "Sure."),
        ])

        #expect(session.isUntitled)
        #expect(session.displayTitle == "Explain the")
    }

    @Test func derivedTitleFallsBackToThePlaceholderWithoutAQuestion() {
        #expect(ChatSession().displayTitle == ChatSession.untitledTitle)
        #expect(ChatSession(messages: [
            ChatMessage(role: .assistant, text: "An answer with no question."),
        ]).displayTitle == ChatSession.untitledTitle)
    }

    @Test func derivedTitleIsClippedToTheCharacterLimit() {
        let long = String(repeating: "word ", count: 40)
        let session = ChatSession(messages: [
            ChatMessage(role: .user, text: long),
        ])

        #expect(session.displayTitle.count <= ChatSession.titleCharacterLimit + 1)
        #expect(session.displayTitle.hasSuffix("\u{2026}"))
    }

    @Test func aUserSetTitleWinsOverTheDerivedOne() {
        var session = ChatSession(messages: [
            ChatMessage(role: .user, text: "First question"),
        ])
        session.title = "  Renamed chat  "

        #expect(!session.isUntitled)
        #expect(session.displayTitle == "  Renamed chat  ")
    }

    @Test func onlyCompletedHealthyTurnsAreReplayable() {
        let session = ChatSession(messages: [
            ChatMessage(role: .user, text: "First"),
            ChatMessage(role: .assistant, text: "First answer"),
            ChatMessage(role: .user, text: "Second"),
            ChatMessage(role: .assistant, text: "half written", isComplete: false),
            ChatMessage(role: .assistant, text: "", isComplete: true),
            ChatMessage(role: .assistant, text: "partial", failureText: "boom"),
        ])

        // The three dropped shapes: still streaming, finished empty, and
        // finished in failure. The bare user turn stays: it is a question the
        // model really was asked, whatever happened to its answer.
        #expect(session.replayableMessages.map(\.text)
            == ["First", "First answer", "Second"])
    }

    @Test func plainTextTranscriptLabelsBothSpeakers() {
        let session = ChatSession(messages: [
            ChatMessage(role: .user, text: "Question"),
            ChatMessage(role: .assistant, text: "Answer"),
        ])

        #expect(session.plainTextTranscript == "You:\nQuestion\n\nFinchMoE:\nAnswer")
    }

    @Test func appendingMovesTheUpdatedStamp() {
        var session = ChatSession()
        let stamp = Date(timeIntervalSince1970: 1_000)
        session.append(ChatMessage(
            role: .user,
            text: "hi",
            createdAt: stamp))

        #expect(session.updatedAt == stamp)
        #expect(session.hasMessages)
    }
}

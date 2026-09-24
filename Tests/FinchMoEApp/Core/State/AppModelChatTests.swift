import Foundation
import Testing
@testable import FinchMoEAppCore

@MainActor
@Suite struct AppModelChatTests {
    @Test func aFreshModelStartsWithOneEmptySession() {
        let model = AppModel()

        #expect(model.sessions.count == 1)
        #expect(model.activeSession.messages.isEmpty)
        #expect(model.activeSessionID == model.sessions[0].id)
    }

    @Test func runAppendsTheQuestionAndAStreamingReply() {
        let model = readyModel(client: MockInferenceClient(
            response: "an answer",
            tokenDelayNanos: 20_000_000))
        model.promptText = "First question"
        model.maxNewTokensOverride = 1

        model.run()

        #expect(model.isRunning)
        #expect(model.activeSession.messages.count == 2)
        #expect(model.activeSession.messages[0].role == .user)
        #expect(model.activeSession.messages[0].text == "First question")
        #expect(model.activeSession.messages[1].role == .assistant)
        #expect(!model.activeSession.messages[1].isComplete)
        #expect(model.isStreamingIntoActiveSession)
        model.cancel()
    }

    @Test func aFinishedRunCompletesTheReply() async {
        let model = readyModel(client: MockInferenceClient(
            response: "an answer",
            tokenDelayNanos: 1))
        model.promptText = "First question"
        model.maxNewTokensOverride = 2

        model.run()
        await waitForIdle(model)

        #expect(model.activeSession.messages.count == 2)
        let reply = model.activeSession.messages[1]
        #expect(reply.isComplete)
        #expect(reply.failureText == nil)
        #expect(reply.text.contains("an answer"))
        #expect(!model.isStreamingIntoActiveSession)
    }

    @Test func theNextRequestCarriesCompletedTurnsAsHistory() async throws {
        let model = readyModel(client: MockInferenceClient(
            response: "first answer",
            tokenDelayNanos: 1))
        model.promptText = "First question"
        model.maxNewTokensOverride = 2

        model.run()
        await waitForIdle(model)

        // A streaming or failed reply is not a turn the model produced, so it
        // must not be on the wire.
        model.activeSession.messages.forEach { #expect($0.isReplayable) }

        model.promptText = "Follow-up question"
        let request = try model.makeRequest()

        #expect(request.prompt == "Follow-up question")
        #expect(request.history.map(\.text) == ["First question", model.activeSession.messages[1].text])
        #expect(request.history.map(\.role) == [.user, .assistant])
    }

    @Test func aFailedRunNotesTheFailureOnTheReply() async {
        let model = readyModel(client: MockInferenceClient(
            tokenDelayNanos: 1,
            failureMessage: "synthetic failure"))
        model.promptText = "Doomed question"

        model.run()
        await waitForIdle(model)

        let reply = model.activeSession.messages.last
        #expect(reply?.role == .assistant)
        #expect(reply?.isComplete == true)
        #expect(reply?.failureText != nil)
        // The failed turn is kept on screen but kept out of the next prompt,
        // while the question that produced it stays a real turn.
        #expect(model.activeSession.replayableMessages.map(\.role) == [.user])
    }

    @Test func aCancelledRunKeepsThePartialReplyWithoutAFailureNote() async {
        let model = readyModel(client: MockInferenceClient(
            response: "one two three four",
            tokenDelayNanos: 20_000_000))
        model.promptText = "Stop this"
        model.maxNewTokensOverride = 10

        model.run()
        // Let at least one token land so there is a partial reply to keep.
        for _ in 0..<200 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        model.cancel()
        await waitForIdle(model)

        let reply = model.activeSession.messages.last
        #expect(reply?.isComplete == true)
        #expect(reply?.failureText == nil)
    }

    @Test func regeneratingReplacesTheLastReply() async {
        let client = MockInferenceClient(response: "an answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.promptText = "First question"
        model.maxNewTokensOverride = 2

        model.run()
        await waitForIdle(model)
        let firstReplyID = model.activeSession.messages.last?.id

        #expect(model.canRegenerateLastReply)
        model.regenerateLastReply()

        #expect(model.activeSession.messages.count == 2)
        #expect(model.activeSession.messages[0].text == "First question")
        #expect(model.activeSession.messages.last?.id != firstReplyID)
        await waitForIdle(model)
        #expect(model.activeSession.messages.last?.isComplete == true)
    }

    @Test func deletingSessionsAlwaysLeavesOneChat() async {
        let model = readyModel(client: MockInferenceClient(response: "a", tokenDelayNanos: 1))
        model.promptText = "First question"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)
        let first = model.activeSessionID

        model.newSession()
        #expect(model.sessions.count == 2)

        model.deleteSession(first)
        #expect(model.sessions.count == 1)
        #expect(model.activeSession.messages.isEmpty)

        model.deleteSession(model.activeSessionID)
        #expect(model.sessions.count == 1)
    }

    @Test func selectingASessionRestoresTheTranscriptMirrors() async {
        let model = readyModel(client: MockInferenceClient(response: "a", tokenDelayNanos: 1))
        model.promptText = "First question"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)
        let first = model.activeSessionID

        model.newSession()
        #expect(model.outputPromptText.isEmpty)
        #expect(!model.hasOutputTranscript)

        model.selectSession(first)
        #expect(model.outputPromptText == "First question")
        #expect(model.hasOutputTranscript)
    }

    @Test func clearingEmptiesTheActiveConversation() async {
        let model = readyModel(client: MockInferenceClient(response: "a", tokenDelayNanos: 1))
        model.promptText = "First question"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)

        model.clearOutput()

        #expect(model.activeSession.messages.isEmpty)
        #expect(!model.hasOutputTranscript)
        #expect(model.sessions.count == 1)
    }

    @Test func newChatIsIgnoredWhileTheActiveChatIsEmpty() {
        let model = AppModel()

        model.newSession()

        #expect(model.sessions.count == 1)
    }

    @Test func chatHistoryAndTheOpenChatSurviveARelaunch() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)
        let client = MockInferenceClient(response: "an answer", tokenDelayNanos: 1)

        let first = readyModel(client: client, chatStore: store)
        first.promptText = "Persisted question"
        first.maxNewTokensOverride = 2
        first.run()
        await waitForIdle(first)

        let relaunched = readyModel(client: client, chatStore: store)

        #expect(relaunched.sessions.count == 1)
        #expect(relaunched.activeSession.messages.count == 2)
        #expect(relaunched.activeSession.messages[0].text == "Persisted question")
        #expect(relaunched.activeSession.displayTitle == "Persisted question")
        #expect(relaunched.outputPromptText == "Persisted question")
    }

    @Test func aModelWithoutAChatStoreNeverTouchesTheDisk() async {
        let model = readyModel(client: MockInferenceClient(response: "a", tokenDelayNanos: 1))
        model.promptText = "In memory only"
        model.maxNewTokensOverride = 1
        model.run()
        await waitForIdle(model)

        #expect(model.activeSession.messages.count == 2)
    }

    private func readyModel(
        client: MockInferenceClient,
        chatStore: ChatSessionFileStore? = nil
    ) -> AppModel {
        let model = AppModel(client: client, chatStore: chatStore)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(
            modelDirectory: FileManager.default.temporaryDirectory,
            loadSeconds: 1)
        return model
    }

    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("AppModelChatTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        return directory
    }
}

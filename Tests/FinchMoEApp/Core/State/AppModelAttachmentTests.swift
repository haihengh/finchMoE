import Foundation
import Testing
@testable import FinchMoEAppCore

@MainActor
@Suite struct AppModelAttachmentTests {

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("Attachments-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: url,
                                                withIntermediateDirectories: true)
        return url
    }

    @Test func imagesPersistAndReplay() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)
        let model = AppModel(client: MockInferenceClient(), chatStore: store)
        let image = AppImageAttachment(mediaType: "image/jpeg",
                                       data: Data([0x01, 0x02, 0x03]),
                                       pixelWidth: 10,
                                       pixelHeight: 20)

        model.beginChatExchange(prompt: "", images: [image])
        let user = try #require(model.activeSession.messages.first)
        #expect(user.role == .user)
        #expect(user.attachments?.count == 1)
        let stored = try #require(user.attachments?.first)
        #expect(store.attachmentData(relativePath: stored.fileName) == image.data)

        let turn = model.chatTurn(for: user)
        #expect(turn.text.isEmpty)
        #expect(turn.images.first?.data == image.data)
        #expect(turn.images.first?.pixelWidth == 10)

        // The archive decodes in a fresh model and the turn is replayable.
        model.persistChatSessions()
        let archive = try #require(store.load())
        let reloaded = try #require(archive.sessions.first?.messages.first)
        #expect(reloaded.attachments?.count == 1)
        #expect(reloaded.isReplayable)
    }

    @Test func deletingASessionRemovesItsAttachmentFiles() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)
        let model = AppModel(client: MockInferenceClient(), chatStore: store)
        model.newSession() // make the first conversation non-empty
        let image = AppImageAttachment(mediaType: "image/jpeg",
                                       data: Data([0xAA]))
        model.promptText = "keep"
        model.beginChatExchange(prompt: "keep", images: [image])
        let stored = try #require(model.activeSession.messages.first?.attachments?.first)
        let url = store.attachmentsDirectory
            .appendingPathComponent(stored.fileName)
        #expect(FileManager.default.fileExists(atPath: url.path))

        model.deleteSession(model.activeSessionID)
        #expect(!FileManager.default.fileExists(atPath: url.path))
    }

    @Test func imageOnlyTurnsAreReplayableAndTitled() {
        let message = ChatMessage(
            role: .user,
            text: "",
            attachments: [ChatImageAttachment(fileName: "a/b.jpg",
                                             mediaType: "image/jpeg")])
        #expect(message.isReplayable)
        #expect(ChatSession.derivedTitle(from: [message]) == "Image")
    }
}

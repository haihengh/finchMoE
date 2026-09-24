import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct ChatSessionFileStoreTests {
    @Test func roundTripsSessionsAndTheActiveSelection() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)

        let older = ChatSession(
            title: "Renamed",
            createdAt: Date(timeIntervalSince1970: 100),
            updatedAt: Date(timeIntervalSince1970: 200),
            messages: [
                ChatMessage(
                    role: .user,
                    text: "What is a mixture of experts?",
                    createdAt: Date(timeIntervalSince1970: 150)),
            ])
        let archive = ChatSessionArchive(
            sessions: [older],
            activeSessionID: older.id)

        try store.save(archive)

        #expect(store.load() == archive)
    }

    @Test func aMissingFileLoadsNothingRatherThanFailing() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        #expect(ChatSessionFileStore(directory: directory).load() == nil)
    }

    @Test func aCorruptFileIsReportedAsEmptyButLeftOnDisk() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)
        let contents = Data("{ this is not the archive".utf8)
        try contents.write(to: store.fileURL)

        #expect(store.load() == nil)
        // Chat history is user data: a file this build cannot read must not be
        // deleted the way an unreadable settings file is.
        #expect(try Data(contentsOf: store.fileURL) == contents)
    }

    @Test func anArchiveFromAnotherVersionIsNotLoaded() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = ChatSessionFileStore(directory: directory)
        try store.save(ChatSessionArchive(
            version: ChatSessionArchive.currentVersion + 1,
            sessions: [ChatSession()]))

        #expect(store.load() == nil)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ChatSessionFileStoreTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true)
        return directory
    }
}

import Foundation

/// What one `chat-sessions.json` holds.
public struct ChatSessionArchive: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public var version: Int
    public var sessions: [ChatSession]
    /// Remembered so relaunching the app reopens the conversation the user
    /// was last in, rather than always jumping to the newest one.
    public var activeSessionID: UUID?

    public init(version: Int = currentVersion,
                sessions: [ChatSession] = [],
                activeSessionID: UUID? = nil) {
        self.version = version
        self.sessions = sessions
        self.activeSessionID = activeSessionID
    }
}

/// Reads and writes the chat history file.
///
/// Kept separate from `MacAppSettingsFileStore` on purpose: settings are
/// per-model and cheap to rebuild, while chat history is user data that must
/// survive a corrupt or partially written file rather than being recreated.
/// A file this store cannot decode is therefore left untouched and reported as
/// "nothing loaded" instead of being deleted.
public struct ChatSessionFileStore: Sendable {
    public static let fileName = "chat-sessions.json"

    public let fileURL: URL

    public init(fileURL: URL) {
        self.fileURL = fileURL
    }

    public init(directory: URL) {
        self.init(fileURL: directory.appendingPathComponent(
            Self.fileName,
            isDirectory: false))
    }

    public static func defaultFileURL(
        fileManager: FileManager = .default
    ) -> URL {
        let base = fileManager.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask).first
            ?? fileManager.finchHomeDirectory
                .appendingPathComponent("Library/Application Support",
                                        isDirectory: true)
        return base
            .appendingPathComponent("FinchMoE", isDirectory: true)
            .appendingPathComponent(fileName, isDirectory: false)
    }

    /// Images live beside the archive, never inside its JSON.
    public var attachmentsDirectory: URL {
        fileURL.deletingLastPathComponent()
            .appendingPathComponent("chat-attachments", isDirectory: true)
    }

    public func attachmentData(relativePath: String) -> Data? {
        guard let url = attachmentURL(relativePath) else { return nil }
        return try? Data(contentsOf: url)
    }

    public func saveAttachment(_ data: Data, relativePath: String) throws {
        guard let url = attachmentURL(relativePath) else {
            throw CocoaError(.fileWriteInvalidFileName)
        }
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }

    public func removeAttachment(relativePath: String) {
        guard let url = attachmentURL(relativePath) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    /// Removes the whole attachment subtree for one message or session.
    public func removeAttachmentDirectory(relativePath: String) {
        guard let url = attachmentURL(relativePath) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    private func attachmentURL(_ relativePath: String) -> URL? {
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: true)
        guard !components.isEmpty,
              !relativePath.hasPrefix("/"),
              !components.contains(where: { $0 == ".." || $0 == "." }) else {
            return nil
        }
        var url = attachmentsDirectory
        for component in components {
            url.appendPathComponent(String(component), isDirectory: false)
        }
        return url
    }

    public func load(fileManager: FileManager = .default) -> ChatSessionArchive? {
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        guard let archive = try? JSONDecoder().decode(
            ChatSessionArchive.self,
            from: data) else { return nil }
        guard archive.version == ChatSessionArchive.currentVersion else { return nil }
        return archive
    }

    public func save(_ archive: ChatSessionArchive,
                     fileManager: FileManager = .default) throws {
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(archive)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }
}

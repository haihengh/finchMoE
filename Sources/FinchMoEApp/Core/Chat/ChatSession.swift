import Foundation

/// One stored conversation: the messages, plus the bookkeeping the sidebar
/// needs to show and sort it.
///
/// A session is a value type so the app can hand SwiftUI an immutable snapshot
/// per render, and `AppModel` can persist one without worrying about the view
/// holding a live reference.
public struct ChatSession: Identifiable, Codable, Equatable, Sendable {
    public static let untitledTitle = "New Chat"
    public static let titleCharacterLimit = 44

    public var id: UUID
    /// Empty until the user renames the session; see `displayTitle`.
    public var title: String
    public var createdAt: Date
    public var updatedAt: Date
    public var messages: [ChatMessage]

    public init(id: UUID = UUID(),
                title: String = "",
                createdAt: Date = Date(),
                updatedAt: Date = Date(),
                messages: [ChatMessage] = []) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.messages = messages
    }

    public var hasMessages: Bool { !messages.isEmpty }

    public var isUntitled: Bool {
        title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// What the sidebar shows: the user's own title when they set one,
    /// otherwise a title derived from the first thing they asked.
    public var displayTitle: String {
        isUntitled ? Self.derivedTitle(from: messages) : title
    }

    public var lastUserMessage: ChatMessage? {
        messages.last { $0.role == .user }
    }

    public var lastAssistantMessage: ChatMessage? {
        messages.last { $0.role == .assistant }
    }

    /// True when the newest assistant reply finished empty, which is the only
    /// state where offering "Regenerate" beats offering "Copy".
    public var isAwaitingReply: Bool {
        guard let last = messages.last else { return false }
        return last.role == .user
    }

    /// The turns to replay as prompt context, newest last.
    public var replayableMessages: [ChatMessage] {
        messages.filter(\.isReplayable)
    }

    /// A plain-text rendering of the whole conversation, for "Copy
    /// conversation" and for the clipboard from the sidebar.
    public var plainTextTranscript: String {
        messages.compactMap { message -> String? in
            guard !message.text.isEmpty else { return nil }
            let speaker = message.role == .user ? "You" : "FinchMoE"
            return "\(speaker):\n\(message.text)"
        }
        .joined(separator: "\n\n")
    }

    public mutating func append(_ message: ChatMessage) {
        messages.append(message)
        updatedAt = message.createdAt
    }

    public mutating func removeAllMessages() {
        messages.removeAll()
        updatedAt = Date()
    }

    /// The sidebar label for a session the user has not named: the first line
    /// of the first thing they asked, collapsed and clipped.
    public static func derivedTitle(from messages: [ChatMessage]) -> String {
        guard let first = messages.first(where: { $0.role == .user }),
              let line = first.text
                .split(whereSeparator: \.isNewline)
                .map({ $0.trimmingCharacters(in: .whitespaces) })
                .first(where: { !$0.isEmpty })
        else {
            return untitledTitle
        }
        let collapsed = line.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        guard collapsed.count > titleCharacterLimit else { return collapsed }
        let clipped = collapsed.prefix(titleCharacterLimit)
            .trimmingCharacters(in: .whitespaces)
        return clipped + "\u{2026}"
    }
}

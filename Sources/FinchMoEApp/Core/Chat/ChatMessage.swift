import Foundation

/// Who authored one message in a chat session.
///
/// Deliberately narrower than the tokenizer's role set: the Mac app's
/// conversation only ever holds turns the user or the model produced, and a
/// system prompt has no UI to edit it yet.
public enum ChatRole: String, Codable, Sendable, Equatable, CaseIterable {
    case user
    case assistant

    public var isUser: Bool { self == .user }
}

/// One message in a chat session.
///
/// `isComplete` is false only for the assistant message currently being
/// generated. That is what separates "still streaming" from "finished with an
/// empty answer" once the run ends, and it is what keeps a half-written reply
/// out of the next turn's history.
public struct ChatMessage: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var role: ChatRole
    public var text: String
    public var createdAt: Date
    public var isComplete: Bool
    /// Set when generation ended in an error. The partial `text`, if any, is
    /// kept so the user still sees how far the model got.
    public var failureText: String?

    public init(id: UUID = UUID(),
                role: ChatRole,
                text: String,
                createdAt: Date = Date(),
                isComplete: Bool = true,
                failureText: String? = nil) {
        self.id = id
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.isComplete = isComplete
        self.failureText = failureText
    }

    /// True while this message is the streaming placeholder of a running
    /// generation and no text has arrived yet.
    public var isAwaitingFirstToken: Bool {
        !isComplete && text.isEmpty
    }

    /// Whether this message can be replayed to the model as context.
    ///
    /// A failed or still-streaming reply is not a turn the model ever
    /// produced in full, so replaying it would feed the model its own
    /// truncated output.
    public var isReplayable: Bool {
        isComplete && failureText == nil && !text.isEmpty
    }
}

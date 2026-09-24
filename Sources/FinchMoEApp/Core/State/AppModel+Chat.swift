import Foundation

/// Chat sessions and turns, split out of `AppModel.swift` because they are a
/// different concern from model loading and installation: this file owns the
/// conversation, that file owns the runtime.
extension AppModel {
    // MARK: - Reading

    /// The conversation on screen. Falls back to a session carrying the
    /// current id so the value is stable even if the list is momentarily
    /// inconsistent; `sessions` is never left empty by the mutators below.
    public var activeSession: ChatSession {
        sessions.first { $0.id == activeSessionID }
            ?? sessions.first
            ?? ChatSession(id: activeSessionID)
    }

    public var activeMessages: [ChatMessage] { activeSession.messages }

    /// True while a reply is streaming into the active conversation, which is
    /// what the transcript uses to decide whether the last bubble is still
    /// being written.
    public var isStreamingIntoActiveSession: Bool {
        guard let id = streamingAssistantMessageID else { return false }
        return activeSession.messages.contains { $0.id == id }
    }

    public var canStartNewSession: Bool { !isRunning }

    /// Regenerating re-asks the last question, so it needs a completed
    /// exchange to replace and a model that is ready to answer.
    public var canRegenerateLastReply: Bool {
        !isRunning && isModelAvailable && !loadState.isLoading && !hasStaleLoadedRuntime
            && activeSession.lastAssistantMessage != nil
            && activeSession.lastUserMessage != nil
    }

    // MARK: - Session lifecycle

    /// Starts a new conversation. A no-op when the conversation on screen is
    /// already empty, so repeatedly pressing New Chat does not pile up blank
    /// sessions in the sidebar.
    public func newSession() {
        guard canStartNewSession else { return }
        guard activeSession.hasMessages else { return }
        let session = ChatSession()
        sessions.insert(session, at: 0)
        activeSessionID = session.id
        syncLegacyTranscriptFromActiveSession()
        persistChatSessions()
    }

    public func selectSession(_ id: UUID) {
        guard canStartNewSession, id != activeSessionID else { return }
        guard sessions.contains(where: { $0.id == id }) else { return }
        activeSessionID = id
        syncLegacyTranscriptFromActiveSession()
        persistChatSessions()
    }

    public func deleteSession(_ id: UUID) {
        guard canStartNewSession else { return }
        sessions.removeAll { $0.id == id }
        if sessions.isEmpty { sessions = [ChatSession()] }
        if activeSessionID == id { activeSessionID = sessions[0].id }
        syncLegacyTranscriptFromActiveSession()
        persistChatSessions()
    }

    public func deleteAllSessions() {
        guard canStartNewSession else { return }
        let session = ChatSession()
        sessions = [session]
        activeSessionID = session.id
        syncLegacyTranscriptFromActiveSession()
        persistChatSessions()
    }

    public func renameSession(_ id: UUID, to title: String) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard sessions[index].title != trimmed else { return }
        sessions[index].title = trimmed
        persistChatSessions()
    }

    /// Drops every message from the conversation on screen but keeps the
    /// session itself, so "Clear" reads as clearing the chat rather than
    /// silently starting a new one.
    func clearActiveSessionMessages() {
        var session = activeSession
        guard session.hasMessages else {
            persistChatSessions()
            return
        }
        session.removeAllMessages()
        replaceActiveSession(session)
        persistChatSessions()
    }

    /// Asks the last question again, dropping the reply being replaced so the
    /// model does not see its own previous answer as context.
    public func regenerateLastReply() {
        guard canRegenerateLastReply,
              let question = activeSession.lastUserMessage?.text else { return }
        var session = activeSession
        if session.messages.last?.role == .assistant {
            session.messages.removeLast()
        }
        if session.messages.last?.role == .user {
            session.messages.removeLast()
        }
        replaceActiveSession(session)
        promptText = question
        run()
    }

    // MARK: - Turns

    /// Records the user's question and the assistant bubble that the running
    /// generation will fill in.
    func beginChatExchange(prompt: String) {
        var session = activeSession
        let now = Date()
        session.append(ChatMessage(role: .user, text: prompt, createdAt: now))
        let reply = ChatMessage(
            role: .assistant,
            text: "",
            createdAt: now,
            isComplete: false)
        streamingAssistantMessageID = reply.id
        session.append(reply)
        session.updatedAt = now
        replaceActiveSession(session, moveToFront: true)
    }

    /// Streams text into the bubble being written.
    func updateStreamingAssistantMessage(with text: String) {
        guard let id = streamingAssistantMessageID else { return }
        var session = activeSession
        guard let index = session.messages.firstIndex(where: { $0.id == id }),
              session.messages[index].text != text else { return }
        session.messages[index].text = text
        replaceActiveSession(session)
    }

    /// Closes the exchange once the run reaches a terminal state, keeping
    /// whatever was streamed and noting the error beside it.
    func endChatExchange(failureText: String?) {
        guard let id = streamingAssistantMessageID else { return }
        streamingAssistantMessageID = nil

        var session = activeSession
        guard let index = session.messages.firstIndex(where: { $0.id == id }) else {
            persistChatSessions()
            return
        }
        var reply = session.messages[index]
        if reply.text.isEmpty {
            reply.text = Self.longestStreamedText(
                generationTranscriptMailbox?.completeText ?? "",
                outputText)
        }
        reply.isComplete = true
        reply.failureText = failureText
        session.messages[index] = reply
        session.updatedAt = Date()
        replaceActiveSession(session)
        persistChatSessions()
    }

    /// A user-stopped generation is not a failure: the partial reply is still
    /// the model's answer, so it gets no error note.
    func chatFailureText(for error: AppInferenceError?) -> String? {
        guard let error else { return nil }
        if case .cancelled = error { return nil }
        return error.userMessage
    }

    // MARK: - Storage

    /// Rebuilds the session list from disk, or starts one empty conversation
    /// when there is nothing stored yet.
    func restoreChatSessions() {
        if let archive = chatStore?.load(), !archive.sessions.isEmpty {
            sessions = archive.sessions.sorted { $0.updatedAt > $1.updatedAt }
            if let stored = archive.activeSessionID,
               sessions.contains(where: { $0.id == stored }) {
                activeSessionID = stored
            } else {
                activeSessionID = sessions[0].id
            }
        } else {
            let session = ChatSession()
            sessions = [session]
            activeSessionID = session.id
        }
        syncLegacyTranscriptFromActiveSession()
    }

    func persistChatSessions() {
        guard let chatStore else { return }
        let archive = ChatSessionArchive(
            sessions: sessions,
            activeSessionID: activeSessionID)
        try? chatStore.save(archive)
    }

    /// Points the legacy single-exchange transcript mirrors at the active
    /// conversation, so the status HUD and the copy actions keep describing
    /// what is actually on screen after a session switch.
    func syncLegacyTranscriptFromActiveSession() {
        let session = activeSession
        outputPromptText = session.lastUserMessage?.text ?? ""
        outputText = session.lastAssistantMessage?.text ?? ""
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
    }

    // MARK: - Helpers

    private func replaceActiveSession(
        _ session: ChatSession,
        moveToFront: Bool = false
    ) {
        if let index = sessions.firstIndex(where: { $0.id == session.id }) {
            sessions[index] = session
            if moveToFront, index != 0 {
                sessions.remove(at: index)
                sessions.insert(session, at: 0)
            }
        } else {
            sessions.insert(session, at: 0)
        }
    }

    /// The mailbox is the lossless channel and `outputText` the delta-merged
    /// one; whichever saw more of the reply is the one to keep.
    static func longestStreamedText(_ lhs: String, _ rhs: String) -> String {
        lhs.count >= rhs.count ? lhs : rhs
    }
}

#if DEBUG
extension AppModel {
    /// Development hook: installs a conversation without loading a model, so
    /// previews and the offscreen snapshot renderer can show the chat UI with
    /// real content.
    public func debugReplaceChatSessions(
        _ sessions: [ChatSession],
        activeSessionID: UUID? = nil
    ) {
        self.sessions = sessions.isEmpty ? [ChatSession()] : sessions
        self.activeSessionID = activeSessionID ?? self.sessions[0].id
        syncLegacyTranscriptFromActiveSession()
    }
}
#endif

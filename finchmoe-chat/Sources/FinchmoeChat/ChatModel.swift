import Foundation
import SwiftUI

struct Message: Identifiable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String = ""
    var isStreaming = false
    var usage: ChatUsage?

    // The think block (everything between the first <think> and </think>),
    // collapsed in the UI like a disclosure group.
    var thinkPart: String? {
        guard let start = text.range(of: "<think>"),
              let end = text.range(of: "</think>", range: start.upperBound..<text.endIndex) else {
            return text.contains("<think>") ? String(text.dropFirst(7)) : nil
        }
        return String(text[start.upperBound..<end.lowerBound])
    }
    var visiblePart: String {
        guard let thinkPart else { return text }
        if let end = text.range(of: "</think>") {
            return String(text[end.upperBound...])
        }
        return ""  // still inside the think — nothing visible yet
    }
}

struct Conversation: Identifiable {
    let id = UUID()          // doubles as the server session_id
    var title: String = "New chat"
    var messages: [Message] = []
}

@MainActor
final class ChatModel: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var selectedID: UUID?
    @Published var serverHost: String {
        didSet { UserDefaults.standard.set(serverHost, forKey: "serverHost") }
    }
    @Published var serverPort: String {
        didSet { UserDefaults.standard.set(serverPort, forKey: "serverPort") }
    }
    @Published var temperature: Double {
        didSet { UserDefaults.standard.set(temperature, forKey: "temperature") }
    }
    @Published var maxTokens: Int {
        didSet { UserDefaults.standard.set(maxTokens, forKey: "maxTokens") }
    }
    @Published var isGenerating = false
    @Published var errorText: String?

    private let client = SSEClient()

    init() {
        serverHost = UserDefaults.standard.string(forKey: "serverHost") ?? "localhost"
        serverPort = UserDefaults.standard.string(forKey: "serverPort") ?? "9000"
        var t = UserDefaults.standard.double(forKey: "temperature")
        if t == 0 { t = 0.7 }  // default 0.7
        temperature = t
        var m = UserDefaults.standard.integer(forKey: "maxTokens")
        if m == 0 { m = 512 }
        maxTokens = m

        client.onDelta = { [weak self] delta in
            Task { @MainActor in self?.appendDelta(delta) }
        }
        client.onFinish = { [weak self] usage in
            Task { @MainActor in self?.finishGeneration(usage: usage) }
        }
        client.onError = { [weak self] message in
            Task { @MainActor in self?.failGeneration(message) }
        }

        if conversations.isEmpty { newConversation() }
    }

    var selected: Conversation? {
        conversations.first { $0.id == selectedID }
    }

    func newConversation() {
        let c = Conversation()
        conversations.insert(c, at: 0)
        selectedID = c.id
    }

    func select(_ id: UUID) { selectedID = id }

    func send(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating, let index = conversations.firstIndex(where: { $0.id == selectedID }) else { return }

        if conversations[index].title == "New chat" {
            conversations[index].title = String(trimmed.prefix(40))
        }
        conversations[index].messages.append(Message(role: .user, text: trimmed))
        conversations[index].messages.append(Message(role: .assistant, isStreaming: true))

        let history = conversations[index].messages.dropLast().map { msg in
            ["role": msg.role == .user ? "user" : "assistant", "content": msg.text]
        }
        isGenerating = true
        errorText = nil

        let base = URL(string: "http://\(serverHost):\(serverPort)")!
        client.streamChat(baseURL: base, messages: history, sessionID: selectedID!.uuidString,
                          temperature: temperature, maxTokens: maxTokens)
    }

    func stop() {
        // The engine stops when the connection drops — cancel the task via a
        // new empty request is overkill; mark done on error instead.
    }

    private func appendDelta(_ delta: String) {
        guard let index = conversations.firstIndex(where: { $0.id == selectedID }),
              let last = conversations[index].messages.last,
              last.isStreaming else { return }
        conversations[index].messages[conversations[index].messages.count - 1].text += delta
    }

    private func finishGeneration(usage: ChatUsage?) {
        guard let index = conversations.firstIndex(where: { $0.id == selectedID }),
              let last = conversations[index].messages.last else { return }
        conversations[index].messages[conversations[index].messages.count - 1].isStreaming = false
        if let usage {
            conversations[index].messages[conversations[index].messages.count - 1].usage = usage
        }
        isGenerating = false
    }

    private func failGeneration(_ message: String) {
        isGenerating = false
        errorText = "Server error: \(message)"
        if let index = conversations.firstIndex(where: { $0.id == selectedID }),
           conversations[index].messages.last?.isStreaming == true {
            conversations[index].messages[conversations[index].messages.count - 1].isStreaming = false
        }
    }
}

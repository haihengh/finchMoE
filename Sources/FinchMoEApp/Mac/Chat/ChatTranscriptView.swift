import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

/// The scrolling conversation.
///
/// The reply being written is polled out of the generation mailbox at ~11 Hz
/// rather than driven by the model's token events: those are throttled for the
/// metrics HUD, while the mailbox is the lossless text channel, so this is what
/// keeps streaming smooth without making every token re-render the whole app.
struct ChatTranscriptView: View {
    let model: AppModel

    @State private var liveReplyText = ""
    @State private var isPinnedToBottom = true

    private var streamingMessageID: UUID? {
        guard model.isRunning, let last = model.activeMessages.last,
              last.role == .assistant else { return nil }
        return last.id
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // A plain VStack, not a LazyVStack: laziness is worth little at
                // conversation sizes, and a lazy container does not always
                // propose the full width to its rows, which is what breaks
                // left/right bubble alignment. Finished answers are parsed
                // through `MarkdownDocumentCache`, so re-rendering the whole
                // stack while a reply streams stays cheap.
                VStack(alignment: .leading, spacing: 18) {
                    if model.activeMessages.isEmpty {
                        ChatEmptyStateView(model: model)
                            .padding(.top, 40)
                    } else {
                        ForEach(model.activeMessages) { message in
                            ChatMessageBubble(
                                message: message,
                                streamingText: message.id == streamingMessageID
                                    ? liveReplyText
                                    : nil,
                                canRegenerate: canRegenerate(message),
                                onRegenerate: model.regenerateLastReply)
                                .id(message.id)
                        }
                    }
                }
                .padding(.horizontal, 28)
                .padding(.vertical, 24)
                .frame(maxWidth: 820, alignment: .leading)
                .frame(maxWidth: .infinity)
            }
            .textSelection(.enabled)
            .defaultScrollAnchor(.bottom)
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentSize.height
                    - geometry.contentOffset.y
                    - geometry.containerSize.height < 120
            } action: { _, pinned in
                isPinnedToBottom = pinned
            }
            .onChange(of: model.activeMessages.count) { _, _ in
                scrollToBottom(proxy, animated: true)
            }
            .onChange(of: liveReplyText) { _, _ in
                guard isPinnedToBottom else { return }
                scrollToBottom(proxy, animated: false)
            }
            .onChange(of: model.activeSessionID) { _, _ in
                isPinnedToBottom = true
                scrollToBottom(proxy, animated: false)
            }
        }
        .task(id: streamPollKey) {
            liveReplyText = ""
            guard model.isRunning else { return }
            while !Task.isCancelled {
                let text = model.generationTranscriptMailbox?.completeText
                    ?? model.outputText
                if text != liveReplyText { liveReplyText = text }
                try? await Task.sleep(for: .milliseconds(90))
            }
        }
    }

    /// Restarts the poll when a run starts, stops, or the user switches chats.
    private var streamPollKey: String {
        "\(model.activeSessionID.uuidString)-\(model.isRunning)"
    }

    private func canRegenerate(_ message: ChatMessage) -> Bool {
        model.canRegenerateLastReply
            && message.role == .assistant
            && message.id == model.activeMessages.last?.id
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy, animated: Bool) {
        guard let last = model.activeMessages.last else { return }
        if animated {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(last.id, anchor: .bottom)
            }
        } else {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }
}

/// Shown when the active conversation has no messages yet.
private struct ChatEmptyStateView: View {
    let model: AppModel

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 30))
                .foregroundStyle(.quaternary)
                .padding(.bottom, 2)
                .accessibilityHidden(true)

            Text("Start a conversation")
                .font(.title3.weight(.semibold))

            Text("Ask a question, or describe what you want written, explained, or planned.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)

            if let hint = statusHint {
                Text(hint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
                    .padding(.top, 2)
            }
            if let detail = model.presentation.detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? .red : .secondary)
                    .multilineTextAlignment(.center)
            }
            if model.canLoadModel || model.canReloadModel {
                Button(model.canReloadModel ? "Reload Model" : "Load Model") {
                    if model.canReloadModel {
                        model.reloadModel()
                    } else {
                        model.loadModel()
                    }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
                .padding(.top, 6)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var statusHint: String? {
        if case .loading = model.loadState { return "Loading the model" }
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        if !model.loadState.isReady { return "Load the model to begin" }
        return nil
    }
}

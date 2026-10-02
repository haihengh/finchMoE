import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

/// One message in the transcript.
///
/// User turns are the filled bubble on the right, assistant turns the wide
/// panel on the left, which is what makes the conversation scannable the way a
/// normal chat app is. A finished answer is rendered as markdown blocks; a
/// still-streaming one stays plain text, because re-parsing markdown on every
/// token costs more than the formatting is worth mid-flight.
///
/// A bubble can be copied two ways: the buttons that fade in under it on
/// hover, and the right-click menu, which is where the second kind of copy —
/// the answer as markdown rather than as it reads — lives.
struct ChatMessageBubble: View {
    let message: ChatMessage
    /// Non-nil only for the reply currently being written.
    var streamingText: String?
    var canRegenerate: Bool
    var onRegenerate: () -> Void

    init(
        message: ChatMessage,
        streamingText: String? = nil,
        canRegenerate: Bool = false,
        onRegenerate: @escaping () -> Void = {}
    ) {
        self.message = message
        self.streamingText = streamingText
        self.canRegenerate = canRegenerate
        self.onRegenerate = onRegenerate
    }

    @State private var isHovering = false
    @State private var copyFeedback = false

    private var isUser: Bool { message.role == .user }

    var body: some View {
        // A message with nothing to copy and nothing to regenerate offers no
        // menu at all, rather than an empty one.
        if hasMenuItems {
            row.contextMenu { menuItems }
        } else {
            row
        }
    }

    private var row: some View {
        // The row is a single full-width view aligned to the correct edge
        // rather than an HStack of content plus Spacer: a Spacer only pushes
        // when the container proposes the full width, which a lazy stack does
        // not always do, and then every bubble ends up hugging the left.
        VStack(alignment: isUser ? .trailing : .leading, spacing: 6) {
            bubble
            if isHovering && !isStreaming {
                actions
            }
        }
        .frame(maxWidth: .infinity, alignment: isUser ? .trailing : .leading)
        .onHover { hovering in
            withAnimation(.easeOut(duration: 0.12)) { isHovering = hovering }
        }
        .task(id: copyFeedback) {
            guard copyFeedback else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            withAnimation(.easeOut(duration: 0.15)) { copyFeedback = false }
        }
    }

    private var isStreaming: Bool { streamingText != nil }

    /// What the bubble is showing, which while a reply streams is the partial
    /// answer rather than the message it is filling in.
    private var visibleText: String { streamingText ?? message.text }

    @ViewBuilder
    private var bubble: some View {
        if isUser {
            Text(message.text)
                .font(.body)
                .foregroundStyle(.white)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(FinchMoEMacTheme.accentColor, in: .rect(cornerRadius: 18))
                // The cap goes on the content so the text wraps short of the
                // cap; the background above has already hugged what is left.
                .frame(maxWidth: 560, alignment: .trailing)
        } else {
            assistantContent
                .padding(.horizontal, 16)
                .padding(.vertical, 14)
                .background {
                    RoundedRectangle(cornerRadius: 16)
                        .fill(FinchPlatformColors.controlBackground)
                        .overlay {
                            RoundedRectangle(cornerRadius: 16)
                                .stroke(.separator.opacity(0.7), lineWidth: 0.6)
                        }
                }
                .frame(maxWidth: 680, alignment: .leading)
        }
    }

    @ViewBuilder
    private var assistantContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let streamingText {
                if streamingText.isEmpty {
                    TypingIndicator()
                } else {
                    HStack(alignment: .bottom, spacing: 3) {
                        Text(streamingText)
                            .font(.body)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        StreamingCaret()
                    }
                }
            } else if message.text.isEmpty {
                Text("No response was produced.")
                    .font(.body)
                    .foregroundStyle(.secondary)
            } else {
                MarkdownBlockView(
                    document: MarkdownDocumentCache.document(for: message.text))
            }

            if let failure = message.failureText {
                Divider().padding(.vertical, 2)
                Label(failure, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Copying

    /// True when there is something for "Copy" to put on the clipboard.
    private var canCopy: Bool { !visibleText.isEmpty }

    private var hasMenuItems: Bool { canCopy || canRegenerate }

    /// The right-click menu.
    ///
    /// The answer is parsed here rather than in `body` so that the transcript
    /// does not rebuild every answer's plain text on every render — the
    /// menu's contents are only ever wanted once it is open.
    @ViewBuilder
    private var menuItems: some View {
        if canCopy {
            Button("Copy", action: copyPlainText)
            // A streaming reply is plain text on screen and a user's own turn
            // is not markdown, so neither has a second form to offer.
            if !isUser, !isStreaming {
                Button("Copy as Markdown") { copyText(message.text) }
                codeItems
            }
        }
        if canRegenerate {
            if canCopy { Divider() }
            Button("Regenerate Reply", action: onRegenerate)
        }
    }

    /// One code block copies on click; several need a submenu, because a
    /// single "Copy Code" would have to guess which one is meant.
    @ViewBuilder
    private var codeItems: some View {
        let blocks = MessageCopyFormatter.codeBlocks(
            of: MarkdownDocumentCache.document(for: message.text))
        if blocks.count == 1 {
            Button("Copy Code") { copyText(blocks[0]) }
        } else if !blocks.isEmpty {
            Menu("Copy Code") {
                ForEach(Array(blocks.enumerated()), id: \.offset) { index, code in
                    Button(MessageCopyFormatter.codeBlockLabel(for: code, index: index)) {
                        copyText(code)
                    }
                }
            }
        }
    }

    /// What "Copy" means: the answer as it reads. A finished reply is written
    /// in markdown, so copying its source would paste the `**` and the `#`
    /// along with it.
    private func copyPlainText() {
        if isUser || isStreaming {
            copyText(visibleText)
        } else {
            copyText(MessageCopyFormatter.plainText(
                of: MarkdownDocumentCache.document(for: message.text)))
        }
    }

    private func copyText(_ text: String) {
        Clipboard.copy(text)
        withAnimation(.easeIn(duration: 0.12)) { copyFeedback = true }
    }

    private var actions: some View {
        HStack(spacing: 4) {
            actionButton(
                systemName: copyFeedback ? "checkmark.circle.fill" : "doc.on.doc",
                label: copyFeedback ? "Copied" : "Copy",
                tint: copyFeedback ? FinchMoEMacTheme.accentColor : .secondary,
                action: copyPlainText)
            if canRegenerate {
                actionButton(
                    systemName: "arrow.clockwise",
                    label: "Regenerate",
                    tint: .secondary,
                    action: onRegenerate)
            }
        }
        .padding(.leading, isUser ? 0 : 4)
    }

    private func actionButton(
        systemName: String,
        label: String,
        tint: Color,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.caption)
                .foregroundStyle(tint)
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.borderless)
        .help(label)
        .accessibilityLabel(label)
    }
}

/// Three dots that rise in turn while the model is still prefilling.
private struct TypingIndicator: View {
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(.secondary)
                    .frame(width: 6, height: 6)
                    .phaseAnimator([0.25, 1.0]) { dot, opacity in
                        dot.opacity(opacity)
                    } animation: { _ in
                        .easeInOut(duration: 0.55).delay(Double(index) * 0.15)
                    }
            }
        }
        .frame(height: 18)
        .accessibilityLabel("Waiting for the first token")
    }
}

/// A cursor at the end of the reply being written.
private struct StreamingCaret: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 1)
            .fill(FinchMoEMacTheme.accentColor)
            .frame(width: 2.5, height: 14)
            .phaseAnimator([1.0, 0.15]) { caret, opacity in
                caret.opacity(opacity)
            } animation: { _ in
                .easeInOut(duration: 0.6)
            }
            .accessibilityHidden(true)
    }
}

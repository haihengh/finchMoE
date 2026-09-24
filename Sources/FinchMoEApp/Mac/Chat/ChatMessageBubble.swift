import AppKit
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
struct ChatMessageBubble: View {
    let message: ChatMessage
    /// Non-nil only for the reply currently being written.
    var streamingText: String?
    var canRegenerate = false
    var onRegenerate: () -> Void = {}

    @State private var isHovering = false
    @State private var copyFeedback = false

    private var isUser: Bool { message.role == .user }

    var body: some View {
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
                        .fill(Color(nsColor: .controlBackgroundColor))
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

    private var actions: some View {
        HStack(spacing: 4) {
            actionButton(
                systemName: copyFeedback ? "checkmark.circle.fill" : "doc.on.doc",
                label: copyFeedback ? "Copied" : "Copy",
                tint: copyFeedback ? FinchMoEMacTheme.accentColor : .secondary
            ) {
                copyToPasteboard(message.text)
                withAnimation(.easeIn(duration: 0.12)) { copyFeedback = true }
            }
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

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
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

import AppKit
import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

/// The conversation list down the left edge: every chat the user has had, with
/// the newest at the top, plus the actions that manage them.
struct ChatSessionSidebar: View {
    let model: AppModel

    @State private var renameTarget: ChatSession?
    @State private var renameDraft = ""
    @State private var isConfirmingDeleteAll = false

    private let rowWidth: CGFloat = 236

    var body: some View {
        VStack(spacing: 0) {
            titleBarDragArea
            header
            Divider()
            list
            Divider()
            footer
        }
        .frame(width: rowWidth)
        .background(Color(nsColor: .underPageBackgroundColor))
        .alert("Rename Chat", isPresented: renameBinding) {
            TextField("Name", text: $renameDraft)
            Button("Cancel", role: .cancel) { renameTarget = nil }
            Button("Rename") {
                if let target = renameTarget {
                    model.renameSession(target.id, to: renameDraft)
                }
                renameTarget = nil
            }
        }
        .confirmationDialog(
            "Delete every chat?",
            isPresented: $isConfirmingDeleteAll,
            titleVisibility: .visible
        ) {
            Button("Delete All Chats", role: .destructive) {
                model.deleteAllSessions()
            }
        } message: {
            Text("This removes the whole chat history and cannot be undone.")
        }
    }

    /// The traffic lights float over the top-left of the window, so the list
    /// starts below them and this strip keeps the window draggable.
    private var titleBarDragArea: some View {
        Color.clear
            .frame(height: 38)
            .contentShape(Rectangle())
            .gesture(WindowDragGesture())
    }

    private var header: some View {
        HStack(spacing: 8) {
            Text("Chats")
                .font(.headline)
            Spacer()
            Button(action: model.newSession) {
                Label("New Chat", systemImage: "square.and.pencil")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .disabled(!model.canStartNewSession)
            .help("New chat (\u{2318}N)")
            .accessibilityLabel("New chat")
        }
        .padding(.horizontal, 14)
        .padding(.bottom, 10)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(spacing: 2) {
                ForEach(model.sessions) { session in
                    row(for: session)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
        }
    }

    private func row(for session: ChatSession) -> some View {
        let isActive = session.id == model.activeSessionID
        return Button {
            model.selectSession(session.id)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isActive ? "bubble.left.fill" : "bubble.left")
                    .font(.caption)
                    .foregroundStyle(isActive
                                     ? FinchMoEMacTheme.accentColor
                                     : Color.secondary)
                    .frame(width: 14)
                VStack(alignment: .leading, spacing: 1) {
                    Text(session.displayTitle)
                        .font(.callout)
                        .lineLimit(1)
                        .foregroundStyle(.primary)
                    Text(subtitle(for: session))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 7)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(.rect(cornerRadius: 8))
        }
        .buttonStyle(.plain)
        .background {
            RoundedRectangle(cornerRadius: 8)
                .fill(isActive
                      ? FinchMoEMacTheme.accentColor.opacity(0.16)
                      : Color.clear)
        }
        .disabled(!model.canStartNewSession)
        .contextMenu {
            Button("Rename\u{2026}") {
                renameDraft = session.title
                renameTarget = session
            }
            Button("Copy Conversation") {
                copyToPasteboard(session.plainTextTranscript)
            }
            .disabled(session.messages.isEmpty)
            Divider()
            Button("Delete", role: .destructive) {
                model.deleteSession(session.id)
            }
        }
        .accessibilityLabel(session.displayTitle)
        .accessibilityValue(isActive ? "Current chat" : "")
    }

    private func subtitle(for session: ChatSession) -> String {
        let turns = session.messages.count
        let stamp = session.updatedAt.formatted(.relative(presentation: .numeric))
        guard turns > 0 else { return stamp }
        return "\(turns) message\(turns == 1 ? "" : "s") \u{00B7} \(stamp)"
    }

    private var footer: some View {
        HStack(spacing: 8) {
            Button("Delete All\u{2026}") {
                isConfirmingDeleteAll = true
            }
            .buttonStyle(.borderless)
            .font(.caption)
            .foregroundStyle(.secondary)
            .disabled(!model.canStartNewSession || model.sessions.count <= 1)
            Spacer()
            Text("\(model.sessions.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }

    private var renameBinding: Binding<Bool> {
        Binding {
            renameTarget != nil
        } set: { isPresented in
            if !isPresented { renameTarget = nil }
        }
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

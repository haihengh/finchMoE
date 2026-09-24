import AppKit
import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI

/// The message box at the bottom of the chat.
struct PromptComposerView: View {
    @Bindable var model: AppModel
    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    /// Height the draft actually needs once it wraps, measured off a hidden
    /// copy of the same text. A `TextEditor` does not size itself.
    @State private var measuredTextHeight: CGFloat = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            editor
            footer
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .animation(.easeOut(duration: 0.12), value: editorHeight)
        .onAppear { promptFocused = true }
    }

    private var editor: some View {
        TextEditor(text: $model.promptText)
            .accessibilityLabel("Message")
            .font(.body)
            .scrollContentBackground(.hidden)
            .focused($promptFocused)
            .onKeyPress(.return, phases: [.down, .repeat]) { keyPress in
                switch PromptSubmissionPolicy.decision(
                    newlineShortcut: model.newlineShortcut,
                    modifiers: keyPress.modifiers,
                    canRun: model.canRun,
                    hasMarkedText: promptHasMarkedText,
                    isRepeat: keyPress.phase.contains(.repeat)) {
                case .submit:
                    model.run()
                    return .handled
                case .consume:
                    return .handled
                case .deferToEditor:
                    return .ignored
                }
            }
            .frame(height: editorHeight)
            .overlay(alignment: .topLeading) {
                if model.promptText.isEmpty {
                    // Matches the NSTextView text origin: 5pt line fragment
                    // padding, no vertical inset.
                    Text(placeholder)
                        .font(.body)
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 5)
                        .allowsHitTesting(false)
                }
            }
            .background(alignment: .topLeading) {
                // A hidden copy of the draft, wrapped at the same width, is
                // how the box knows how tall to be: a TextEditor keeps its
                // own height, so without this a long message scrolls inside a
                // one-line box instead of pushing the composer taller.
                Text(measurementText)
                    .font(.body)
                    .padding(.leading, 5)
                    .fixedSize(horizontal: false, vertical: true)
                    .hidden()
                    .onGeometryChange(for: CGFloat.self) { proxy in
                        proxy.size.height
                    } action: { height in
                        measuredTextHeight = height
                    }
            }
    }

    private var measurementText: String {
        model.promptText.isEmpty ? " " : model.promptText + "\n"
    }

    private var placeholder: String {
        model.loadState.isReady ? "Message FinchMoE" : "Load the model to start chatting"
    }

    private var promptHasMarkedText: Bool {
        (NSApp.keyWindow?.firstResponder as? NSTextView)?.hasMarkedText() == true
    }

    /// One comfortable line at rest, growing as the draft wraps up to about
    /// six lines and then scrolling inside the box.
    private var editorHeight: CGFloat {
        min(max(measuredTextHeight + 4, 28), 132)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            promptTips
            Spacer()
            secondaryAction
            GenerateControl(model: model)
        }
    }

    private var promptTips: some View {
        Button {
            showingPromptTips.toggle()
        } label: {
            Label("Prompt tips", systemImage: "questionmark.circle")
                .labelStyle(.iconOnly)
                .frame(width: 26, height: 26)
                .contentShape(Circle())
        }
        .buttonStyle(.borderless)
        .foregroundStyle(.secondary)
        .help("Prompt tips")
        .popover(isPresented: $showingPromptTips,
                 attachmentAnchor: .point(.top),
                 arrowEdge: .top) {
            promptGuide
        }
    }

    private var promptGuide: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Prompting this model")
                .font(.headline)

            tipSection("Ask for a clear task",
                       "Say what you want the model to create, explain, plan, or transform. Put the essential context in the same prompt.")
            tipSection("Shape the answer",
                       "Specify a useful length, sections, tone, or output format. Concrete constraints work better than a long list of vague preferences.")
            tipSection("Follow up in the same chat",
                       "Earlier turns of this conversation are sent again with each new message, so you can refine the answer instead of restating it.")
            tipSection("Anchor important facts",
                       "Include facts the answer must preserve and say what should be checked. Generated factual claims can still be wrong or outdated.")
            tipSection("For code and calculations",
                       "Provide types, dimensions, interfaces, edge cases, or a small scaffold. Compile or run the result before relying on it.")
        }
        .font(.callout)
        .frame(width: 390, alignment: .leading)
        .padding(18)
    }

    private func tipSection(_ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .fontWeight(.semibold)
            Text(detail)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Clearing acts on the whole conversation, which is why it only appears
    /// once there is one; otherwise it clears the unsent draft.
    @ViewBuilder
    private var secondaryAction: some View {
        if !model.isRunning && model.activeSession.hasMessages {
            Button {
                model.clearOutput()
            } label: {
                Label("Clear chat", systemImage: "trash")
                    .labelStyle(.iconOnly)
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Clear chat")
        } else if !model.isRunning && !model.promptText.isEmpty {
            Button {
                model.promptText = ""
                promptFocused = true
            } label: {
                Label("Clear message", systemImage: "xmark.circle.fill")
                    .labelStyle(.iconOnly)
                    .symbolRenderingMode(.hierarchical)
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Clear message")
        }
    }
}

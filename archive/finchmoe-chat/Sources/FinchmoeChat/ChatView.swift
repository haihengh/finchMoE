import SwiftUI

struct ChatView: View {
    @EnvironmentObject var model: ChatModel

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            conversation
            Divider()
            inspector
        }
        .frame(minWidth: 900, minHeight: 560)
    }

    // MARK: Sidebar

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("FinchMoE").font(.headline).foregroundStyle(.tint)
                Spacer()
                Button { model.newConversation() } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.borderless)
                .help("New chat")
            }
            .padding(.horizontal, 12)
            .padding(.top, 12)

            List(selection: $model.selectedID) {
                ForEach(model.conversations) { c in
                    Text(c.title)
                        .lineLimit(1)
                        .tag(c.id)
                        .contextMenu {
                            Button("Delete", role: .destructive) {
                                if let i = model.conversations.firstIndex(where: { $0.id == c.id }) {
                                    model.conversations.remove(at: i)
                                    if model.selectedID == c.id { model.selectedID = model.conversations.first?.id }
                                }
                            }
                        }
                }
            }
            .listStyle(.sidebar)
        }
        .frame(width: 220)
    }

    // MARK: Conversation

    private var conversation: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 14) {
                        ForEach(model.selected?.messages ?? []) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                    .padding(16)
                }
                .onChange(of: model.selected?.messages.last?.text) { _ in
                    if let last = model.selected?.messages.last {
                        withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            if let err = model.errorText {
                Text(err).font(.caption).foregroundStyle(.red).padding(.horizontal)
            }

            InputBar()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: Inspector

    private var inspector: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Server").font(.headline)
            TextField("Host", text: $model.serverHost)
            TextField("Port", text: $model.serverPort)
            Divider()

            Text("Sampling").font(.headline)
            HStack {
                Text("Temperature")
                Spacer()
                Text(String(format: "%.1f", model.temperature)).monospacedDigit()
            }
            Slider(value: $model.temperature, in: 0.0...1.5, step: 0.1)
            HStack {
                Text("Max tokens")
                Spacer()
                TextField("", value: $model.maxTokens, format: .number)
                    .frame(width: 70)
                    .textFieldStyle(.roundedBorder)
                    .multilineTextAlignment(.trailing)
            }

            Divider()

            if let usage = model.selected?.messages.last(where: { $0.usage != nil })?.usage {
                Text("Last response").font(.headline)
                if let tps = usage.tokens_per_second {
                    Label(String(format: "%.1f tok/s", tps), systemImage: "speedometer")
                }
                if let c = usage.completion_tokens { Text("\(c) tokens").font(.caption) }
            }

            Spacer()
        }
        .padding(14)
        .frame(width: 260)
    }
}

struct MessageBubble: View {
    let message: Message
    @State private var showThink = false

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if message.role == .assistant {
                Image(systemName: "sparkles")
                    .foregroundStyle(.tint)
                    .padding(.top, 3)
            } else {
                Spacer(minLength: 60)
            }

            VStack(alignment: .leading, spacing: 6) {
                if let think = message.thinkPart, !think.isEmpty {
                    DisclosureGroup(isExpanded: $showThink) {
                        Text(think)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "brain")
                            Text(message.isStreaming && message.visiblePart.isEmpty
                                 ? "Thinking…" : "Thought process")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }
                }

                if !message.visiblePart.isEmpty {
                    Text(message.visiblePart)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(message.role == .user ? Color.accentColor.opacity(0.15) : Color(nsColor: .windowBackgroundColor))
            )

            if message.role == .user {
                Spacer(minLength: 60)
            }
        }
    }
}

struct InputBar: View {
    @EnvironmentObject var model: ChatModel
    @State private var text = ""

    var body: some View {
        HStack(alignment: .bottom, spacing: 8) {
            TextField("Message FinchMoE…", text: $text, axis: .vertical)
                .lineLimit(1...6)
                .textFieldStyle(.plain)
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 12).fill(Color(nsColor: .windowBackgroundColor)))
                .onSubmit { submit() }
            Button { submit() } label: {
                Image(systemName: model.isGenerating ? "stop.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 24))
            }
            .buttonStyle(.plain)
            .disabled(!model.isGenerating && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(12)
    }

    private func submit() {
        let t = text
        text = ""
        model.send(t)
    }
}

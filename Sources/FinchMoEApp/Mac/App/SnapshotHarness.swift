#if DEBUG
import AppKit
import FinchMoEAppCore
import SwiftUI

/// Renders the whole window offscreen to a PNG.
///
/// The chat layout is the kind of thing that either reads right or does not,
/// and this box has no screen-recording permission for `screencapture`, so the
/// only way to look at a change here is to draw it ourselves:
///
///     .build/debug/FinchMoEMac --snapshot /tmp/chat.png
enum SnapshotHarness {
    @MainActor
    static func runIfRequested(
        arguments: [String] = CommandLine.arguments
    ) {
        if let index = arguments.firstIndex(of: "--snapshot"),
           arguments.indices.contains(index + 1) {
            render(to: arguments[index + 1])
        }
        if let index = arguments.firstIndex(of: "--snapshot-messages"),
           arguments.indices.contains(index + 1) {
            renderMessages(to: arguments[index + 1])
        }
    }

    /// Renders just the message rows, with the exact container modifiers
    /// `ChatTranscriptView` uses. The window render cannot show them: a
    /// `LazyVStack` inside a `ScrollView` lays out nothing offscreen.
    @MainActor
    static func renderMessages(
        to path: String,
        width: CGFloat = 820,
        height: CGFloat = 1800
    ) {
        let model = makeModel()
        let content = VStack(alignment: .leading, spacing: 18) {
            ForEach(model.activeMessages) { message in
                ChatMessageBubble(
                    message: message,
                    streamingText: nil,
                    canRegenerate: false,
                    onRegenerate: {})
            }
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 24)
        .frame(maxWidth: 820, alignment: .leading)
        .frame(maxWidth: .infinity)
        .frame(width: width, height: height, alignment: .top)
        .background(Color(nsColor: .windowBackgroundColor))

        write(content, to: path, size: CGSize(width: width, height: height))
    }

    @MainActor
    static func render(
        to path: String,
        size: CGSize = CGSize(width: 1240, height: 800)
    ) {
        let model = makeModel()
        write(
            RootView(model: model)
                .frame(width: size.width, height: size.height),
            to: path,
            size: size)
    }

    /// Draws the view through a real (offscreen) window rather than through
    /// `ImageRenderer`.
    ///
    /// `ImageRenderer` draws a `ScrollView`'s frame and nothing inside it — an
    /// answer's code panels and tables are scrollable, so they came out as
    /// empty boxes. A hosted window lays the view out for real, so what lands
    /// in the PNG is what the window shows.
    @MainActor
    private static func write<V: View>(
        _ content: V,
        to path: String,
        size: CGSize
    ) {
        let hosting = NSHostingView(
            rootView: content.frame(width: size.width, height: size.height))
        hosting.frame = CGRect(origin: .zero, size: size)

        let window = NSWindow(
            contentRect: hosting.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false)
        window.contentView = hosting
        window.layoutIfNeeded()
        hosting.layoutSubtreeIfNeeded()

        let scale: CGFloat = 2
        guard let bitmap = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: Int(size.width * scale),
            pixelsHigh: Int(size.height * scale),
            bitsPerSample: 8,
            samplesPerPixel: 4,
            hasAlpha: true,
            isPlanar: false,
            colorSpaceName: .deviceRGB,
            bytesPerRow: 0,
            bitsPerPixel: 0) else {
            FileHandle.standardError.write(Data("snapshot: no bitmap\n".utf8))
            exit(1)
        }
        bitmap.size = size
        hosting.cacheDisplay(in: hosting.bounds, to: bitmap)

        guard let png = bitmap.representation(using: .png, properties: [:]) else {
            FileHandle.standardError.write(Data("snapshot: render failed\n".utf8))
            exit(1)
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
        } catch {
            FileHandle.standardError.write(
                Data("snapshot: \(error)\n".utf8))
            exit(1)
        }
        print("snapshot written to \(path)")
        exit(0)
    }

    @MainActor
    private static func makeModel() -> AppModel {
        let model = AppModel()
        let directory = URL(fileURLWithPath: model.modelPathText)
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 1.4)
        let sessions = debugSessions()
        model.debugReplaceChatSessions(sessions, activeSessionID: sessions[0].id)
        return model
    }

    private static func debugSessions() -> [ChatSession] {
        let now = Date()
        let conversation = ChatSession(
            title: "",
            createdAt: now.addingTimeInterval(-3_600),
            updatedAt: now,
            messages: [
                ChatMessage(
                    role: .user,
                    text: "Why does a mixture-of-experts layer keep only a few experts hot, and what does that mean for memory?",
                    createdAt: now.addingTimeInterval(-120)),
                ChatMessage(
                    role: .assistant,
                    text: """
                    A sparse MoE layer routes each token to a **small subset** of \
                    experts, so decode touches far fewer weights than the parameter \
                    count suggests.

                    ## What stays resident

                    1. The router and shared expert, always.
                    2. Only the experts the token selected.

                    ```swift
                    let selected = router.topK(token, k: 8)
                    for expert in selected { expert.apply(token) }
                    ```

                    > Memory is therefore set by the cache, not by the checkpoint.

                    - 4-bit experts stream in from disk
                    - the LRU cache decides how often

                    ---

                    That trade is why expert-cache slots matter more than raw RAM.
                    """,
                    createdAt: now.addingTimeInterval(-110)),
                ChatMessage(
                    role: .user,
                    text: "Compare the quantization options, with numbers.",
                    createdAt: now.addingTimeInterval(-60)),
                // The rendering regression case: a title, a markdown table, a
                // box-drawing table, an ASCII chart in a fence, generics in
                // prose, and an unterminated fence. Every one of these used to
                // take the whole answer down to plain text.
                ChatMessage(
                    role: .assistant,
                    text: """
                    ## Where the 4-bit install lands

                    | Precision | Size | Speed | Notes |
                    |:----------|-----:|------:|:------|
                    | 4-bit     | 1.95 GB | 11.4 tok/s | the default |
                    | 3-bit     | 1.4 GB | 12.4 tok/s | quality cliff |
                    | GGUF Q4_K | 20 GB | 1.7 tok/s | streaming |

                    The same numbers as a box table, which the model draws by hand:

                    ┌───────────┬────────┬────────────┐
                    │ Precision │  Size  │   Speed    │
                    ├───────────┼────────┼────────────┤
                    │ 4-bit     │ 1.95GB │ 11.4 tok/s │
                    │ 3-bit     │ 1.40GB │ 12.4 tok/s │
                    └───────────┴────────┴────────────┘

                    And the decode rate as a chart:

                    ```text
                    tok/s
                     14 |        ███
                     12 |  ███   ███
                     10 |  ███   ███
                      8 |  ███   ███   ███
                        +------------------
                          4bit  3bit  GGUF
                    ```

                    1. The router picks experts per token.
                       - `Array<Int>` indices, k = 8
                       - the rest stay on disk
                    2. Only the hot set stays resident.

                    ```swift
                    let experts: [Int] = router.topK(token, k: 8)
                    ```
                    """,
                    createdAt: now.addingTimeInterval(-50)),
            ])
        let earlier = ChatSession(
            title: "",
            createdAt: now.addingTimeInterval(-86_400),
            updatedAt: now.addingTimeInterval(-3_600),
            messages: [
                ChatMessage(
                    role: .user,
                    text: "Explain chunked prefill",
                    createdAt: now.addingTimeInterval(-3_700)),
                ChatMessage(
                    role: .assistant,
                    text: "It bounds peak memory by processing the prompt in chunks.",
                    createdAt: now.addingTimeInterval(-3_600)),
            ])
        let renamed = ChatSession(
            title: "Metal kernel notes",
            createdAt: now.addingTimeInterval(-172_800),
            updatedAt: now.addingTimeInterval(-90_000),
            messages: [
                ChatMessage(
                    role: .user,
                    text: "How do I size threadgroups for the int4 GEMV?",
                    createdAt: now.addingTimeInterval(-90_100)),
            ])
        return [conversation, earlier, renamed]
    }
}
#endif

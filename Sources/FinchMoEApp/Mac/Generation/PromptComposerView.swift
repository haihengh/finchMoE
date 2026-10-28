import FinchMoEAppCore
import FinchMoEMacPresentation
import PhotosUI
import SwiftUI
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#endif

/// The message box at the bottom of the chat.
struct PromptComposerView: View {
    @Bindable var model: AppModel

    init(model: AppModel) {
        self._model = Bindable(model)
    }

    @FocusState private var promptFocused: Bool
    @State private var showingPromptTips = false
    @State private var showingPhotoPicker = false
    @State private var showingFileImporter = false
    @State private var showingCamera = false
    @State private var photoItem: PhotosPickerItem?
    /// Height the draft actually needs once it wraps, measured off a hidden
    /// copy of the same text. A `TextEditor` does not size itself.
    @State private var measuredTextHeight: CGFloat = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.pendingImages.isEmpty {
                attachmentStrip
            }
            editor
            footer
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background {
            RoundedRectangle(cornerRadius: 22)
                .fill(FinchPlatformColors.controlBackground)
                .overlay {
                    RoundedRectangle(cornerRadius: 22)
                        .stroke(.separator.opacity(0.5), lineWidth: 0.5)
                }
        }
        .animation(.easeOut(duration: 0.12), value: editorHeight)
        .animation(.easeOut(duration: 0.15), value: model.pendingImages.count)
        .photosPicker(isPresented: $showingPhotoPicker,
                      selection: $photoItem,
                      matching: .images)
        .fileImporter(isPresented: $showingFileImporter,
                      allowedContentTypes: [.image],
                      allowsMultipleSelection: true) { result in
            importFiles(result)
        }
        #if canImport(UIKit)
        .fullScreenCover(isPresented: $showingCamera) {
            CameraPicker(
                isPresented: $showingCamera,
                maxDimension: CGFloat(AppImagePreparation.recommendedPixelDimension(
                    for: model.installDescriptor.architecture))) { data in
                addImage(data: data)
            }
            .ignoresSafeArea()
        }
        #endif
        .onChange(of: photoItem) { _, item in
            guard let item else { return }
            Task { @MainActor in
                defer { photoItem = nil }
                guard let data = try? await item.loadTransferable(type: Data.self)
                else {
                    model.error = .invalidRequest(
                        "That photo could not be loaded.")
                    return
                }
                addImage(data: data)
            }
        }
        #if canImport(AppKit)
        .onAppear { promptFocused = true }
        #endif
        // The iOS shell pins the interface orientation while the keyboard is
        // up: UIKit 27 crashes in the candidate bar when the interface rotates
        // under it. Not raising the keyboard on launch (iOS) keeps rotation
        // available until the user actually starts typing.
        .onChange(of: promptFocused) { _, focused in
            FinchPlatformInput.setTextInputFocused(focused)
        }
        .onDisappear { FinchPlatformInput.setTextInputFocused(false) }
    }

    private var attachmentStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(model.pendingImages) { image in
                    AttachmentThumbnail(data: image.data, size: 58)
                        .overlay(alignment: .topTrailing) {
                            Button {
                                model.removePendingImage(id: image.id)
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.system(size: 16, weight: .semibold))
                                    .symbolRenderingMode(.palette)
                                    .foregroundStyle(.white, .black.opacity(0.62))
                            }
                            .buttonStyle(.plain)
                            .offset(x: 5, y: -5)
                            .accessibilityLabel("Remove image")
                        }
                }
            }
            .padding(.top, 4)
            .padding(.trailing, 6)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func importFiles(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            for url in urls {
                let scoped = url.startAccessingSecurityScopedResource()
                defer {
                    if scoped { url.stopAccessingSecurityScopedResource() }
                }
                guard let data = try? Data(contentsOf: url) else {
                    model.error = .invalidRequest(
                        "\(url.lastPathComponent) could not be read.")
                    continue
                }
                addImage(data: data)
            }
        case .failure(let error):
            model.error = .invalidRequest(error.localizedDescription)
        }
    }

    private func addImage(data: Data) {
        // The import budget is model/platform-specific: a 3.8 image on a
        // phone is 1024 px, not the tower's desktop 1536 px ceiling.
        let maxDimension = AppImagePreparation.recommendedPixelDimension(
            for: model.installDescriptor.architecture)
        Task { @MainActor in
            do {
                let prepared = try await Task.detached(priority: .userInitiated) {
                    try AppImagePreparation.prepare(data: data,
                                                    maxDimension: maxDimension)
                }.value
                model.addPendingImages([AppPendingImage(
                    data: prepared.data,
                    mediaType: prepared.mediaType,
                    pixelWidth: prepared.pixelWidth,
                    pixelHeight: prepared.pixelHeight)])
            } catch {
                model.error = .invalidRequest("\(error)")
            }
        }
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
        FinchPlatformInput.isComposingMarkedText
    }

    /// One comfortable line at rest, growing as the draft wraps up to about
    /// six lines and then scrolling inside the box.
    private var editorHeight: CGFloat {
        min(max(measuredTextHeight + 4, 28), 132)
    }

    private var footer: some View {
        HStack(spacing: 10) {
            HStack(spacing: 4) {
                attachControl
                cameraControl
            }
            promptTips
            Spacer()
            secondaryAction
            GenerateControl(model: model)
        }
    }

    /// iOS only. The Mac shell has no camera; its file importer covers images.
    @ViewBuilder
    private var cameraControl: some View {
        #if canImport(UIKit)
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            Button {
                showingCamera = true
            } label: {
                Image(systemName: "camera")
                    .font(.system(size: 14, weight: .medium))
                    .frame(width: 26, height: 26)
                    .contentShape(Circle())
            }
            .buttonStyle(.borderless)
            .foregroundStyle(model.canAttachImage ? .secondary : .tertiary)
            .disabled(!model.canAttachImage)
            .help("Take a photo")
            .accessibilityLabel("Take photo")
        }
        #endif
    }

    private var attachControl: some View {
        Menu {
            #if canImport(UIKit)
            Button {
                showingPhotoPicker = true
            } label: {
                Label("Photo Library", systemImage: "photo.on.rectangle")
            }
            #endif
            Button {
                showingFileImporter = true
            } label: {
                Label("Choose File…", systemImage: "folder")
            }
        } label: {
            Image(systemName: "paperclip")
                .font(.system(size: 14, weight: .medium))
                .frame(width: 26, height: 26)
                .contentShape(Circle())
        }
        #if canImport(AppKit)
        .menuStyle(.borderlessButton)
        #endif
        .fixedSize()
        .foregroundStyle(model.canAttachImage ? .secondary : .tertiary)
        .disabled(!model.canAttachImage)
        .help(model.supportsVision
              ? "Attach a photo or image file"
              : "Attach images (requires a Qwen Vision model)")
        .accessibilityLabel("Attach image")
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
        } else if !model.isRunning && (!model.promptText.isEmpty
                                       || !model.pendingImages.isEmpty) {
            Button {
                model.promptText = ""
                model.clearPendingImages()
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

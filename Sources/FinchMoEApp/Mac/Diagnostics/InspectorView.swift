import FinchMoEAppCore
import FinchMoEMacPresentation
import SwiftUI
import UniformTypeIdentifiers
#if canImport(AppKit)
import AppKit
#endif

struct InspectorView: View {
    @Bindable var model: AppModel
    #if !canImport(AppKit)
    @State private var showsFolderImporter = false
    @AppStorage(FinchAccentTheme.storageKey)
    private var accentThemeRawValue = FinchAccentTheme.default.rawValue
    #endif

    init(model: AppModel) {
        self._model = Bindable(model)
    }

    var body: some View {
        form
        #if !canImport(AppKit)
        // iOS has no NSOpenPanel; the folder picker is SwiftUI's fileImporter.
        .fileImporter(
            isPresented: $showsFolderImporter,
            allowedContentTypes: [.folder]
        ) { result in
            switch result {
            case .success(let url):
                _ = url.startAccessingSecurityScopedResource()
                model.setModelURL(url)
            case .failure(let error):
                model.reportModelSelectionError(error.localizedDescription)
            }
        }
        #endif
    }

    private var form: some View {
        Form {
            #if !canImport(AppKit)
            appearanceSection
            #endif
            modelSection
            #if canImport(AppKit)
            // The local HTTP server is a sibling process, a mechanism that only
            // exists on macOS. The iOS app talks to the engine in-process, so
            // there is nothing to host — the section is not shown at all there.
            serverSection
            #endif
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(FinchPlatformColors.windowBackground)
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        Clipboard.copy(model.modelPathText)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            } label: {
                settingLabel("Path", """
                The directory this preset installs to and loads from. On iPhone \
                and iPad it lives in the app's Documents folder, visible in \
                Files under On My iPhone > FinchMoE > Models, so other tools can \
                choose it. Change it with Preset, or point at an existing \
                install with Choose Local Directory.
                """)
            }
            LabeledContent {
                Picker("Preset", selection: modelChoiceBinding) {
                    ForEach(model.modelChoices) { choice in
                        Text(choice.label).tag(Optional(choice))
                    }
                    Text("Local directory").tag(Optional<AppModelChoice>.none)
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .accessibilityIdentifier("model-preset-picker")
            } label: {
                settingLabel("Preset", """
                Which model build the app runs. Each row installs into its own \
                folder. Qwen 3.6 3-bit is the lightest default; 3.8 models are \
                larger and need more RAM. Abliterated variants remove refusal \
                behavior and otherwise match their base model.
                """)
            }
            HStack(spacing: 4) {
                Button {
                    chooseLocalModelDirectory()
                } label: {
                    Label("Choose Local Directory", systemImage: "folder")
                }
                .disabled(model.isRunning || model.isInstallingModel)
                SettingInfoButton(title: "Local directory", detail: """
                Point the app at an existing .finch install already on this \
                device instead of downloading a published build.
                """)
                Spacer()
            }
            if let message = model.modelSelectionError {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            } else if case .partial(let message) = model.installationStatus {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            if model.canUnloadModel {
                Button("Unload Model", action: model.unloadModel)
            }
            LabeledContent("State") {
                Text(model.presentation.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if model.requiresModelInstallation {
                LabeledContent("Download") {
                    Text(MetricFormat.storage(model.installDescriptor.approximateDownloadBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(model.installDescriptor.installedBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let requirement = model.installRequirement {
                    LabeledContent("Available") {
                        Text(MetricFormat.storage(requirement.availableBytes))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                if !model.installDescriptor.supportsRemoteInstall {
                    Text("This preset loads a completed local .finch directory. Use Choose Local Directory for an existing install, or repack/download it outside the app.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    #if !canImport(AppKit)
    /// App-wide accent selection. iOS-only: the picker lives in the Settings
    /// sheet, and the Mac app keeps its fixed green accent.
    private var appearanceSection: some View {
        Section {
            LazyVGrid(
                columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 4),
                spacing: 12
            ) {
                ForEach(FinchAccentTheme.allCases) { theme in
                    let isSelected = theme == selectedAccentTheme
                    Button {
                        accentThemeRawValue = theme.rawValue
                    } label: {
                        VStack(spacing: 6) {
                            Circle()
                                .fill(theme.color)
                                .frame(width: 32, height: 32)
                                .overlay {
                                    if isSelected {
                                        Image(systemName: "checkmark")
                                            .font(.system(size: 13, weight: .bold))
                                            .foregroundStyle(.white)
                                    }
                                }
                                .overlay {
                                    Circle().stroke(.separator.opacity(0.6),
                                                    lineWidth: 0.5)
                                }
                            Text(theme.displayName)
                                .font(.caption2.weight(isSelected ? .semibold : .regular))
                                .foregroundStyle(isSelected ? .primary : .secondary)
                                .lineLimit(1)
                                .minimumScaleFactor(0.8)
                        }
                        .frame(maxWidth: .infinity)
                        .contentShape(.rect)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(theme.displayName) accent")
                    .accessibilityAddTraits(isSelected ? .isSelected : [])
                }
            }
            .padding(.vertical, 4)
            Text("Colors the send button, user messages, selected chats, and other controls. The choice applies immediately and is remembered across launches.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } header: {
            HStack(spacing: 4) {
                Text("Appearance")
                SettingInfoButton(title: "Appearance", detail: """
                The accent color used for the send button, user messages, \
                selected chats, and status indicators. The choice applies \
                immediately and is remembered across launches.
                """)
            }
        }
    }

    private var selectedAccentTheme: FinchAccentTheme {
        FinchAccentTheme(rawValue: accentThemeRawValue) ?? .default
    }
    #endif

    #if canImport(AppKit)
    private var serverSection: some View {
        Section("Local server") {
            LabeledContent("Port") {
                Stepper(value: $model.localServerPort, in: 1...65_535, step: 1) {
                    Text("\(model.localServerPort)").monospacedDigit()
                }
                .fixedSize()
            }
            LabeledContent("Model ID") {
                TextField("Model ID", text: $model.localServerModelID)
                    .textFieldStyle(.roundedBorder)
            }
            switch model.localServerState {
            case .stopped:
                Text("Stopped")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            case .starting:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Starting")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            case .running(let url):
                LabeledContent("URL") {
                    Text(url.absoluteString)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            case .failed(let message):
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            HStack {
                Button("Start", action: model.startLocalServer)
                    .disabled(!model.canStartLocalServer)
                Button("Stop", action: model.stopLocalServer)
                    .disabled(!model.canStopLocalServer)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }
    #endif

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent {
                Picker("Context", selection: contextSelection) {
                    ForEach(model.allowedContextLengths) { option in
                        Text(option.menuLabel(architecture: model.installDescriptor.architecture))
                            .tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("Context", """
                Maximum number of tokens the model can see and keep in memory: \
                the prompt plus the reply. Larger contexts cost more KV-cache \
                RAM; 4K is a safe default on iPhone and iPad.
                """)
            }
            LabeledContent {
                Picker("KV cache", selection: $model.runtimeOptions.kvCacheMode) {
                    ForEach(AppKVCacheMode.allCases) { mode in
                        Text(mode.label)
                            .tag(mode)
                            .disabled(!mode.isAvailable(
                                for: model.installDescriptor.architecture))
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("KV cache", """
                How attention K/V is stored. FP16 is the safe default. Int8 \
                halves KV memory on Qwen 3.6 only; Qwen 3.8 and Gemma must stay \
                on FP16. Changing it takes effect after a model reload.
                """)
            }
            .onChange(of: model.runtimeOptions.kvCacheMode) { _, newValue in
                guard newValue.isAvailable(for: model.installDescriptor.architecture) else {
                    model.runtimeOptions.kvCacheMode = .fp16
                    return
                }
            }
            Text(kvCacheDetail)
                .font(.caption)
                .foregroundStyle(.secondary)
            if !AppKVCacheMode.int8.isAvailable(
                for: model.installDescriptor.architecture) {
                Text("Int8 KV is wired for the Qwen 3.6 body only; this model must stay on FP16.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            LabeledContent {
                Picker("Slots", selection: slotsSelection) {
                    ForEach(model.allowedExpertCacheSlots, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("Slots", """
                How many routed-expert tiles stay cached in RAM. More slots can \
                speed up decoding but use more memory; 16 is the default. Qwen \
                3.8 routes to 10 experts per token, so 10 is its minimum. \
                Applies after reloading the model.
                """)
            }
            LabeledContent {
                Picker("Eviction", selection: $model.runtimeOptions.expertCachePolicy) {
                    ForEach(AppExpertCachePolicy.allCases) { policy in
                        Text(policy == .lfu ? "LFU (default)" : policy.label).tag(policy)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("Eviction", """
                Which expert to drop when the cache is full. LFU evicts the \
                least frequently used; LRU evicts the one unused for the \
                longest time.
                """)
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if let cap = model.memoryCap {
                Text("On this device, Qwen 3.8 Vision runs with at most \(cap.expertCacheSlots) expert slots, \(cap.contextTokens / 1_024)K context and \(cap.prefillChunkTokens)-token prefill chunks so the vision tower fits under the app memory limit. Your choices above are kept for other models.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            } label: {
                settingLabel("Temperature", """
                Controls sampling randomness. 0 is deterministic greedy \
                decoding; higher values make answers more varied and less \
                predictable.
                """)
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            settingToggle("Top-K", detail: """
            Limits each token choice to the K most likely candidates. Lower K \
            is more focused; higher K is more diverse.
            """, isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                } label: {
                    settingLabel("K value", "The number of candidates retained by Top-K.")
                }
            }
            settingToggle("Top-P", detail: """
            Nucleus sampling: keeps the smallest set of tokens whose combined \
            probability reaches P. Lower P is more focused; higher P is more \
            diverse.
            """, isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                } label: {
                    settingLabel("P value", "The probability mass retained by Top-P.")
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            settingToggle("Prefill", detail: """
            Processes the prompt in chunks before generating. This lowers peak \
            memory and keeps the app responsive on long prompts.
            """, isOn: $model.runtimeOptions.prefillEnabled)
            if model.runtimeOptions.prefillEnabled {
                LabeledContent {
                    Picker("Prefill chunk", selection: prefillChunkSelection) {
                        ForEach(model.allowedPrefillChunkTokens, id: \.self) { tokens in
                            Text(tokens == AppRuntimeOptions.defaultPrefillChunkTokens
                                    ? "\(tokens) (default)" : "\(tokens)")
                                .tag(tokens)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                } label: {
                    settingLabel("Prefill chunk", """
                    Tokens processed per prefill chunk. Smaller chunks use less \
                    memory; larger chunks can finish prefill faster.
                    """)
                }
            }
            settingToggle("Reuse prompt cache", detail: """
            Keeps the previous turn's KV cache and continues from it, so a \
            follow-up only processes the new text instead of the whole \
            conversation. Applies from the next message; no reload needed.
            """, isOn: $model.promptReuseEnabled)
            Text("Keeps the previous turn's cache and continues from it, so a follow-up only prefills what is new instead of re-processing the whole conversation. Applies from the next message; no reload needed.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent {
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("RDADVISE", """
                Experimental file-cache advice for streamed expert reads. It can \
                help short decodes but hurt long ones; Off is the safe choice.
                """)
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent {
                Picker("Model verification", selection: $model.runtimeOptions.modelVerification) {
                    ForEach(AppModelVerification.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            } label: {
                settingLabel("Verification", """
                How the install is checked before loading. Automatic uses a \
                valid trusted receipt and falls back to hashing. Full SHA-256 \
                always hashes every file. Trusted receipt accepts a valid \
                receipt without re-hashing.
                """)
            }
            Text(model.runtimeOptions.modelVerification.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
            if model.hasStaleLoadedRuntime {
                Text("Reload required")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    /// What the selected KV format actually costs and where it is wired.
    private var kvCacheDetail: String {
        switch model.runtimeOptions.kvCacheMode {
        case .fp16:
            return "FP16 K/V on the full-attention layers. Applies after a reload."
        case .int8:
            return "Int8 K/V with one FP16 scale per 64-element block on the full-attention layers: about 48% less K/V while decoding, and a quantize step on each write. Qwen 3.6 only — other models refuse to load with it. Applies after a reload."
        case .turbo4bit:
            return "Turbo 4-bit is a planned format with no kernel behind it and stays disabled."
        }
    }

    /// A settings-row label with the popover that explains the control.
    private func settingLabel(_ title: String, _ detail: String) -> some View {
        HStack(spacing: 4) {
            Text(title)
            SettingInfoButton(title: title, detail: detail)
        }
    }

    /// A switch row with its explanation button at the trailing edge.
    private func settingToggle(_ title: String,
                               detail: String,
                               isOn: Binding<Bool>) -> some View {
        HStack(spacing: 4) {
            Toggle(title, isOn: isOn)
            SettingInfoButton(title: title, detail: detail)
        }
    }

    /// The pickers show the values the current model will actually load. On
    /// a phone the 3.8 vision row is capped (see `AppModelMemoryBudget`); the
    /// user's persisted choice stays in `maxContextTokens`/`runtimeOptions`
    /// for the other models, while the setter edits the persisted value.
    private var contextSelection: Binding<Int> {
        Binding {
            model.effectiveMaxContextTokens
        } set: { tokens in
            model.maxContextTokens = tokens
        }
    }

    private var slotsSelection: Binding<Int> {
        Binding {
            model.effectiveRuntimeOptions.expertCacheSlots
        } set: { slots in
            model.runtimeOptions.expertCacheSlots = slots
        }
    }

    private var prefillChunkSelection: Binding<Int> {
        Binding {
            model.effectiveRuntimeOptions.prefillChunkTokens
        } set: { tokens in
            model.runtimeOptions.prefillChunkTokens = tokens
        }
    }

    private var modelChoiceBinding: Binding<AppModelChoice?> {
        Binding {
            model.selectedModelChoice
        } set: { choice in
            guard let choice else { return }
            model.setModelChoice(choice)
        }
    }

    private func chooseLocalModelDirectory() {
        #if canImport(AppKit)
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            model.setModelURL(url)
        }
        #else
        showsFolderImporter = true
        #endif
    }

}

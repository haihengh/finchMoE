import AppKit
import FinchMoEAppCore
import SwiftUI

struct InspectorView: View {
    @Bindable var model: AppModel

    var body: some View {
        Form {
            modelSection
            serverSection
            memorySection
            generationSection
            runtimeSection
            RunnerDiagnosticsSection(diagnostics: model.diagnostics)
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modelSection: some View {
        Section("Model") {
            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .font(.caption)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                }
            }
            LabeledContent("Preset") {
                Picker("Preset", selection: modelChoiceBinding) {
                    ForEach(model.modelChoices) { choice in
                        Text(choice.label).tag(Optional(choice))
                    }
                    Text("Local directory").tag(Optional<AppModelChoice>.none)
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            HStack {
                Button {
                    chooseLocalModelDirectory()
                } label: {
                    Label("Choose Local Directory", systemImage: "folder")
                }
                .disabled(model.isRunning || model.isInstallingModel)
                Spacer()
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

    private var memorySection: some View {
        Section("Memory") {
            LabeledContent("Context") {
                Picker("Context", selection: $model.maxContextTokens) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.menuLabel(architecture: model.installDescriptor.architecture))
                            .tag(option.tokens)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            LabeledContent("KV cache") {
                Picker("KV cache", selection: $model.runtimeOptions.kvCacheMode) {
                    ForEach(AppKVCacheMode.allCases) { mode in
                        Text(mode.label)
                            .tag(mode)
                            .disabled(!mode.isAvailable)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            .onChange(of: model.runtimeOptions.kvCacheMode) { _, newValue in
                guard newValue.isAvailable else {
                    model.runtimeOptions.kvCacheMode = .fp16
                    return
                }
            }
            Text(kvCacheDetail)
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            Text("More slots can improve decode speed by keeping more experts in memory, but they also use more RAM. Changes are compared with 4K context and 16 slots and apply after reloading the model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var generationSection: some View {
        Section("Generation") {
            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }
            Text("0 uses deterministic greedy decoding. Higher values make sampling more varied.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                }
            }
            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
                .disabled(!model.topKEnabled)
            if model.topKEnabled && model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }
        }
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var runtimeSection: some View {
        Section("Runtime") {
            Toggle("Prefill", isOn: $model.runtimeOptions.prefillEnabled)
            LabeledContent("RDADVISE") {
                Picker("RDADVISE", selection: $model.runtimeOptions.rdadvisePolicy) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            Text("RDADVISE is experimental. It may speed up short decodes but slow down long decodes.")
                .font(.caption)
                .foregroundStyle(.secondary)
            LabeledContent("Verification") {
                Picker("Model verification", selection: $model.runtimeOptions.modelVerification) {
                    ForEach(AppModelVerification.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
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

    private var modelChoiceBinding: Binding<AppModelChoice?> {
        Binding {
            model.selectedModelChoice
        } set: { choice in
            guard let choice else { return }
            model.setModelChoice(choice)
        }
    }

    private func chooseLocalModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            model.setModelURL(url)
        }
    }

}

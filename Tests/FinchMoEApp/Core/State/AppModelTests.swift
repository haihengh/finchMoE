import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct AppModelTests {
    @MainActor
    @Test func defaultsUseSampledRequest() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        let request = try model.makeRequest()
        #expect(request.temperature == 0.2)
        #expect(request.topK == 64)
        #expect(request.topP == 0.95)
        #expect(request.maxNewTokens == 4_096)
        #expect(request.repetitionPenalty == 1)
        #expect(!request.isPureGreedy)
        #expect(request.runtimeOptions.expertCacheSlots == 16)
        #expect(request.runtimeOptions.expertCachePolicy == .lfu)
        #expect(request.runtimeOptions.rdadvisePolicy == .off)
        #expect(request.runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func runDisabledWhenPromptEmpty() {
        let model = AppModel()
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "   "
        #expect(!model.canRun)
    }

    @MainActor
    @Test func runDisabledUntilModelReady() {
        let model = AppModel()
        model.promptText = "go"
        #expect(!model.canRun)
    }

    @MainActor
    @Test func disablingTopKNeutralizesBothTruncationControls() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.topKEnabled = false
        model.topPEnabled = true

        let request = try model.makeRequest()
        #expect(request.topK == nil)
        #expect(request.topP == nil)
    }

    @MainActor
    @Test func prefillToggleSurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"

        model.runtimeOptions.prefillEnabled = false
        #expect(try !model.makeRequest().runtimeOptions.prefillEnabled)

        model.runtimeOptions.prefillEnabled = true
        #expect(try model.makeRequest().runtimeOptions.prefillEnabled)
    }

    @MainActor
    @Test func adaptiveRDAdvicePolicySurvivesRequestCreation() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.runtimeOptions.rdadvisePolicy = .adaptive

        let request = try model.makeRequest()
        #expect(request.runtimeOptions.rdadvisePolicy == .adaptive)
    }

    @MainActor
    @Test func loadAffectingRuntimeChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func contextChangeMarksReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(!model.hasStaleLoadedRuntime)
        model.maxContextTokens = AppContextLengthOption.eightK.tokens
        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func appResponseLimitUsesSelectedContext() throws {
        let model = AppModel()
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.promptText = "go"
        model.maxContextTokens = AppContextLengthOption.sixtyFourK.tokens

        #expect(try model.makeRequest().maxNewTokens == AppContextLengthOption.sixtyFourK.tokens)
    }

    @MainActor
    @Test func requestTimePrefillChangeMarksReadySessionStale() {
        // The engine builds chunked-prefill scratch from the chunk size when
        // the runner is loaded and its session guard rejects any runtime
        // option change afterwards (`RealInferenceClient`'s
        // `loadedKey == requestKey`), so a prefill change on a ready session
        // cannot be served without a reload. Tracking it here is what makes
        // the Inspector show "Reload required" instead of letting the next
        // generation fail late with `reloadRequired`.
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.runtimeOptions.prefillEnabled = false

        #expect(model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func newlineShortcutDoesNotMarkReadySessionStale() {
        let model = AppModel(client: MockLifecycleInferenceClient())
        let directory = FileManager.default.temporaryDirectory
        model.modelPathText = directory.path
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.setNewlineShortcut(.shiftReturn)

        #expect(model.newlineShortcut == .shiftReturn)
        #expect(!model.hasStaleLoadedRuntime)
    }

    @MainActor
    @Test func mockRunUpdatesOutputAndDiagnostics() async throws {
        let client = MockInferenceClient(response: "alpha beta", tokenDelayNanos: 1)
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        model.promptText = "go"
        model.maxNewTokensOverride = 4
        model.run()

        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(!model.isRunning)
        #expect(model.outputText.contains("alpha beta"))
        #expect(model.diagnostics != nil)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func runSnapshotsPromptIntoOutputTranscript() async throws {
        let client = MockInferenceClient(response: "answer", tokenDelayNanos: 1)
        let model = readyModel(client: client)
        model.setSentPromptBehavior(.keep)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1
        model.run()

        #expect(model.outputPromptText == "original prompt")
        #expect(model.promptText == "original prompt")
        #expect(model.hasOutputTranscript)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\noriginal prompt")

        model.promptText = "edited prompt"
        await waitForIdle(model)

        #expect(model.outputPromptText == "original prompt")
        #expect(model.outputResponsePlainText == "answer")
        #expect(model.outputConversationPlainText
            == "You:\noriginal prompt\n\nAnswer:\nanswer")
        #expect(!model.outputConversationPlainText.contains("edited prompt"))
    }

    @MainActor
    @Test func clearAfterSendingPreservesTranscriptAndNextDraft() async throws {
        let client = MockInferenceClient(
            response: "answer",
            tokenDelayNanos: 20_000_000)
        let model = readyModel(client: client)
        model.setSentPromptBehavior(.clear)
        model.promptText = "original prompt"
        model.maxNewTokensOverride = 1

        model.run()

        #expect(model.isRunning)
        #expect(model.outputPromptText == "original prompt")
        #expect(model.promptText.isEmpty)

        model.promptText = "next draft"
        await waitForIdle(model)

        #expect(model.promptText == "next draft")
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\noriginal prompt\n\nAnswer:\n"))
    }

    @MainActor
    @Test func failedValidationDoesNotClearPrompt() {
        let model = readyModel(client: MockInferenceClient(response: "answer"))
        model.setSentPromptBehavior(.clear)
        model.promptText = "keep invalid prompt"
        model.maxNewTokensOverride = 0

        model.run()

        #expect(!model.isRunning)
        #expect(model.promptText == "keep invalid prompt")
        #expect(model.outputPromptText.isEmpty)
        #expect(model.error != nil)
    }

    @MainActor
    @Test func staleReadySessionDisablesGenerationUntilReload() throws {
        let client = MockLifecycleInferenceClient()
        let directory = try makeCompleteModelInstall("stale-runtime")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "go"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        #expect(model.canRun)
        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.hasStaleLoadedRuntime)
        #expect(!model.canRun)
        #expect(model.canReloadModel)
        #expect(client.ensureLoadedCallCount() == 0)
    }

    @MainActor
    @Test func cancelAfterPartialOutputCanBeCleared() async throws {
        let client = MockInferenceClient(response: "one two three four five", tokenDelayNanos: 20_000_000)
        client.prefillSteps = 0
        let model = readyModel(client: client)
        model.promptText = "stop after token"
        model.maxNewTokensOverride = 10
        model.run()

        for _ in 0..<200 where model.liveTokenCount == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.liveTokenCount > 0)
        model.cancel()
        #expect(model.isCancellationPending)
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(!model.isCancellationPending)
        #expect(model.error == .cancelled)
        #expect(model.hasOutputTranscript)
        #expect(!model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.hasPrefix(
            "You:\nstop after token\n\nAnswer:\n"))

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
        #expect(model.outputPromptText.isEmpty)
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText.isEmpty)
        #expect(model.error == nil)
    }

    @MainActor
    @Test func cancelDuringPrefillKeepsPromptSnapshotUntilClear() async throws {
        let client = MockInferenceClient(response: "unused", tokenDelayNanos: 1_000_000)
        client.prefillSteps = 20
        let model = readyModel(client: client)
        model.promptText = "prefill prompt"
        model.run()

        for _ in 0..<200 where model.livePrefillDone == 0 {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }

        #expect(model.outputPromptText == "prefill prompt")
        model.cancel()
        await waitForIdle(model)

        #expect(!model.isRunning)
        #expect(model.outputPromptText == "prefill prompt")
        #expect(model.outputText.isEmpty)
        #expect(model.outputResponsePlainText.isEmpty)
        #expect(model.outputConversationPlainText == "You:\nprefill prompt")
        #expect(model.hasOutputTranscript)

        model.clearOutput()
        #expect(!model.hasOutputTranscript)
    }

    @MainActor
    @Test func failedEventThenThrownErrorKeepsFirstTerminalState() async throws {
        let client = MockInferenceClient(tokenDelayNanos: 1, failureMessage: "synthetic failure")
        let model = readyModel(client: client)
        model.promptText = "fail"

        model.run()
        await waitForIdle(model)

        #expect(model.error?.userMessage == "synthetic failure")
        #expect(model.diagnostics?.stopReason == .failed)
    }

    @MainActor
    @Test func changingModelPathInvalidatesLoadedStateAndDiagnostics() {
        // Mock installer keeps install readiness deterministic: the label after
        // the path change must not depend on the volume's real free space.
        let model = AppModel(
            client: MockInferenceClient(),
            installer: MockModelInstallerClient())
        let oldURL = FileManager.default.temporaryDirectory.appendingPathComponent("old.finch")
        let newURL = FileManager.default.temporaryDirectory.appendingPathComponent(
            "new-\(UUID().uuidString).finch", isDirectory: true)
        try? FileManager.default.createDirectory(at: newURL, withIntermediateDirectories: true)
        try? Data("{}".utf8).write(to: newURL.appendingPathComponent("manifest.json"))
        defer { try? FileManager.default.removeItem(at: newURL) }
        model.modelPathText = oldURL.path
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)
        model.diagnostics = AppDiagnostics(
            generatedTokens: 1,
            stopReason: .eos,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 1,
            tokensPerSecond: 1,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
        model.error = .unknown("old error")

        model.setModelURL(newURL)

        #expect(model.modelPathText == newURL.standardizedFileURL.path)
        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
        #expect(model.diagnostics == nil)
        #expect(model.error == nil)
        #expect(model.presentation.label == "Model required")
        #expect(!model.canRun)
    }

    /// The user-facing form of the stale-receipt bug: a model directory
    /// copied or migrated to a new path must become loadable when selected,
    /// instead of showing the install screen with Download disabled and no
    /// Load button.
    @MainActor
    @Test func selectingAMovedInstallOffersLoad() throws {
        let url = try makeCompleteModelInstall("moved-selection")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(
            with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/somewhere/else/model.finch"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
            .write(to: receiptURL)

        let model = AppModel(
            client: MockInferenceClient(),
            installer: MockModelInstallerClient())
        model.setModelURL(url)

        #expect(model.installationStatus == .complete)
        #expect(model.canLoadModel)
        #expect(model.modelSelectionError == nil)
    }

    /// A folder that is not a model must not be treated as one. Before this
    /// guard, picking the wrong folder resolved to the default (Gemma)
    /// descriptor, so the install screen silently offered a Gemma download.
    @MainActor
    @Test func selectingAFolderWithoutAManifestKeepsTheCurrentSelection() throws {
        let model = AppModel(
            client: MockInferenceClient(),
            installer: MockModelInstallerClient())
        let originalPath = model.modelPathText
        let bogus = FileManager.default.temporaryDirectory.appendingPathComponent(
            "not-a-model-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: bogus, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: bogus) }

        model.setModelURL(bogus)

        #expect(model.modelPathText == originalPath)
        #expect(model.modelSelectionError?.contains("manifest.json") == true)
    }

    /// Every picker row must land on its checkout's `models/<install-name>`
    /// directory. This is the macOS half of the contract; the no-checkout
    /// half is covered at the location layer
    /// (`installURLOutsideACheckoutUsesTheAppRoot`) because `swift test` always
    /// runs inside a package checkout.
    @MainActor
    @Test func setModelChoiceInCheckoutUsesRowDirectory() throws {
        let model = AppModel(
            modelDirectory: FileManager.default.temporaryDirectory,
            client: MockInferenceClient(),
            installer: MockModelInstallerClient())
        model.applyLoadState(.ready(modelDirectory: FileManager.default.temporaryDirectory,
                                    loadSeconds: 1))

        let choice = AppModelChoice.qwen3_6_3bit
        let root = try #require(model.packageRootURL)
        model.setModelChoice(choice)

        #expect(model.modelPathText == choice.defaultURL(packageRoot: root).path)
        #expect(model.modelPathText.hasSuffix("/models/Qwen3.6-35B-A3B-3bit.finch"))
        #expect(model.selectedModelChoice == choice)
        #expect(model.installDescriptor == choice.descriptor)
        #expect(model.loadState == .notLoaded)
        #expect(model.installState == .idle)
    }

    /// A saved int8 choice must not survive into a Qwen 3.8 load: the runner
    /// refuses that combination, and an app-side crash is not a valid way to
    /// report it. Launching the install clamps the option to FP16.
    @MainActor
    @Test func persistedInt8KVFallsBackToFP16OnQwen38() throws {
        let directory = try makeCompleteModelInstall(
            "kv-int8-qwen38",
            arch: .qwen3_8_flashNext_125B,
            descriptor: .qwen3_8_2bit,
            routedExpertBits: 2)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MacAppSettingsFileStore.save(
            MacAppSettings(kvCacheMode: .int8),
            forModelDirectory: directory)

        let model = AppModel(modelDirectory: directory,
                             settingsPersistenceEnabled: true)

        #expect(model.installDescriptor.architecture.isQwen3_8)
        #expect(model.runtimeOptions.kvCacheMode == .fp16)
        model.promptText = "go"
        #expect(try model.makeRequest().runtimeOptions.kvCacheMode == .fp16)
    }

    @MainActor
    private func readyModel(client: MockInferenceClient) -> AppModel {
        let model = AppModel(client: client)
        model.modelPathText = FileManager.default.temporaryDirectory.path
        model.loadState = .ready(modelDirectory: FileManager.default.temporaryDirectory, loadSeconds: 1)
        return model
    }

    @MainActor
    private func waitForIdle(_ model: AppModel) async {
        for _ in 0..<200 where model.isRunning {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }
}

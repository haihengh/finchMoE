import Foundation
import FinchMoERepackCore
import Observation

@MainActor
@Observable
public final class AppModel {
    public enum RunState: Equatable {
        case idle
        case running
    }

    public enum LocalServerState: Equatable {
        case stopped
        case starting
        case running(URL)
        case failed(String)

        public var isRunning: Bool {
            if case .running = self { return true }
            return false
        }

        public var isActive: Bool {
            switch self {
            case .starting, .running: return true
            case .stopped, .failed: return false
            }
        }
    }

    public var modelPathText: String
    public private(set) var selectedModelChoice: AppModelChoice?
    public var promptText: String = ""
    /// Images attached to the draft. Cleared when the message is sent (with
    /// the prompt) or when the model changes.
    public var pendingImages: [AppPendingImage] = []
    // `internal(set)` rather than `private(set)`: the chat layer lives in
    // `AppModel+Chat.swift`, and `private` setters are only visible inside the
    // file that declares them. Still hidden from the SwiftUI target, which is
    // a separate module.
    public internal(set) var outputPromptText: String = ""
    public var outputText: String = ""
    public var runState: RunState = .idle
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens: Int = 4096
    public var temperature: Double = 0.2
    public var topKEnabled: Bool = true
    public var topK: Int = 64
    public var topPEnabled: Bool = true
    public var topP: Double = 0.95
    /// Whether a chat turn may resume from the previous turn's retained KV
    /// prefix. Live: applied per request, no model reload on change.
    public var promptReuseEnabled: Bool = true
    public private(set) var newlineShortcut: AppNewlineShortcut = .return
    public private(set) var sentPromptBehavior: AppSentPromptBehavior = .clear
    /// Every conversation, newest first. Never empty: an empty list is
    /// normalized back to one fresh session so the chat UI always has a
    /// session to show.
    public internal(set) var sessions: [ChatSession] = []
    public internal(set) var activeSessionID: UUID = UUID()
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var localServerPort: Int = 8080
    public var localServerModelID: String = "finchmoe-local"
    public private(set) var localServerState: LocalServerState = .stopped
    public private(set) var localServerLog: String = ""
    public var installState: AppModelInstallState = .idle
    public private(set) var installETAPresentation: DownloadETAPresentation = .hidden
    public private(set) var installETAText: String?
    public private(set) var installReadiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus
    /// Why the last local-folder selection was refused, if it was. Kept apart
    /// from `error` (a generation error) so the install screen can explain a
    /// bad picker choice without looking like a failed run.
    public private(set) var modelSelectionError: String?

    public var loadState: AppModelLoadState = .notLoaded
    public private(set) var loadedRuntimeKey: AppLoadedRuntimeKey?
    public private(set) var phase: AppGenerationPhase = .idle
    public private(set) var liveTokenCount: Int = 0
    public private(set) var liveElapsedDecodeSeconds: Double = 0
    public private(set) var livePrefillDone: Int = 0
    public private(set) var livePrefillTotal: Int = 0
    public private(set) var livePrefillCached: Int = 0
    public private(set) var liveMemoryBytes: UInt64?
    public private(set) var isCancellationPending: Bool = false

    private let client: any AppInferenceClient
    private var installer: any AppModelInstallerClient
    private var runTask: Task<Void, Never>?
    private var loadTask: Task<Void, Never>?
    private var installTask: Task<Void, Never>?
    #if canImport(AppKit)
    private var localServerProcess: Process?
    #endif
    private var unloadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var unloadGeneration: UInt64 = 0
    private var installGeneration: UInt64 = 0
    private var pendingExplicitLoadRuntimeKey: AppLoadedRuntimeKey?
    private var activeRunRuntimeKey: AppLoadedRuntimeKey?
    private var hasHandledTerminalEvent = false
    private let memorySampler: AppMemorySampler
    private let settingsPersistenceEnabled: Bool
    /// nil disables chat persistence, which is what the tests want: a unit
    /// test must not read or write the developer's real chat history.
    let chatStore: ChatSessionFileStore?
    /// The assistant message the running generation is filling in. Held as an
    /// id rather than an index so a session switch (or a trim) cannot make the
    /// stream write into the wrong bubble.
    var streamingAssistantMessageID: UUID?
    private let installETAClock: SuspendingClock
    private let installETAOrigin: SuspendingClock.Instant
    private var installETAEstimator = DownloadETAEstimator()

    public init(modelDirectory: URL? = nil,
                client: any AppInferenceClient = RealInferenceClient(),
                installer: (any AppModelInstallerClient)? = nil,
                memorySampler: AppMemorySampler = AppMemorySampler(),
                settingsPersistenceEnabled: Bool = false,
                chatStore: ChatSessionFileStore? = nil) {
        #if canImport(UIKit)
        // iOS keeps app-owned models in Documents so the Files app can see
        // them; move installs made before that change before resolving paths.
        AppModelLocation.prepareAppModelStorage()
        #endif
        let directory = (modelDirectory ?? AppModelLocation.defaultURL()).standardizedFileURL
        // The descriptor must key off the directory's checkpoint (Qwen or
        // Gemma), because it drives the probe, the install UI, and the
        // installer. A fresh app-owned destination has no manifest yet, so it
        // takes the app's default row (Qwen 3.6 3-bit) instead of Gemma; every
        // other destination keeps the probe's own fallback.
        let descriptor = AppModelInstallationProbe.matchingDescriptor(
            at: directory,
            missingManifestFallback: AppModelLocation.defaultDescriptor(for: directory)
                ?? .default)
        let installETAClock = SuspendingClock()
        let settings = settingsPersistenceEnabled
            ? MacAppSettingsFileStore.loadOrCreate(forModelDirectory: directory)
            : MacAppSettings()
        self.modelPathText = directory.path
        self.selectedModelChoice = Self.modelChoice(for: directory, descriptor: descriptor)
        // A setting saved before int8 KV became model-gated (or copied from a
        // 3.6 install) must not reach the runner on a model that cannot serve
        // it: the runner traps on int8 + Qwen 3.8, which leaves FP16 as the
        // only valid choice for that body.
        var runtimeOptions = AppRuntimeOptions(persisted: settings)
        if !runtimeOptions.kvCacheMode.isAvailable(for: descriptor.architecture) {
            runtimeOptions.kvCacheMode = .fp16
        }
        // Qwen 3.8 routes to ten experts per token. A persisted 8-slot choice
        // from a 3.6 layout would otherwise trap the first decode layer.
        let minimumSlots = descriptor.architecture.topKExperts
        if runtimeOptions.expertCacheSlots < minimumSlots,
           let replacement = AppRuntimeOptions.allowedSlotCounts.first(
               where: { $0 >= minimumSlots }) {
            runtimeOptions.expertCacheSlots = replacement
        }
        self.runtimeOptions = runtimeOptions
        self.maxContextTokens = settings.contextTokens
        self.temperature = settings.temperature
        self.topKEnabled = settings.topKEnabled
        self.topK = settings.topK
        self.topPEnabled = settings.topPEnabled
        self.topP = settings.topP
        self.newlineShortcut = settings.newlineShortcut
        self.sentPromptBehavior = settings.sentPromptBehavior
        self.promptReuseEnabled = settings.promptReuseEnabled
        self.installationStatus = AppModelInstallationProbe.status(at: directory,
                                                                    descriptor: descriptor)
        self.client = client
        self.installer = installer ?? Self.makeInstaller(descriptor: descriptor)
        self.memorySampler = memorySampler
        self.settingsPersistenceEnabled = settingsPersistenceEnabled
        self.chatStore = chatStore
        self.installETAClock = installETAClock
        self.installETAOrigin = installETAClock.now
        restoreChatSessions()
        refreshInstallReadiness()
    }

    public var isRunning: Bool { runState == .running }

    public var isModelAvailable: Bool { loadState.isReady }

    public var hasStaleLoadedRuntime: Bool {
        guard loadState.isReady, let loadedRuntimeKey else { return false }
        return loadedRuntimeKey != currentRuntimeKey
    }

    public var canLoadModel: Bool {
        isModelInstalled && !isRunning && (loadState == .notLoaded || loadState.isFailed)
    }

    public var canCancelLoad: Bool {
        if case .loading = loadState { return loadTask != nil }
        return false
    }

    public var canReloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady && hasStaleLoadedRuntime
    }

    public var canUnloadModel: Bool {
        isModelInstalled && !isRunning && loadState.isReady
    }

    public var isModelInstalled: Bool { installationStatus == .complete }

    public var requiresModelInstallation: Bool { !isModelInstalled }

    public var installDescriptor: AppModelInstallDescriptor { installer.descriptor }

    /// The phone-only budget for the largest vision model, or nil when the
    /// current model fits the persisted settings as-is.
    public var memoryCap: AppModelMemoryBudget.Cap? {
        AppModelMemoryBudget.cap(for: installDescriptor)
    }

    /// The context actually loaded and requested. The picker keeps the user's
    /// persisted choice; this is the value the runner is built with.
    public var effectiveMaxContextTokens: Int {
        min(maxContextTokens, memoryCap?.contextTokens ?? Int.max)
    }

    /// The cache sizes the current model can actually use. Qwen 3.8 routes
    /// to ten experts per token, and a layer's cache must hold at least
    /// top-k; offering the 8-slot choice there would trap the decoder.
    public var allowedExpertCacheSlots: [Int] {
        var slots = AppRuntimeOptions.allowedSlotCounts.filter {
            $0 >= installDescriptor.architecture.topKExperts
        }
        if let memoryCap {
            slots = slots.filter { $0 <= memoryCap.expertCacheSlots }
        }
        return slots
    }

    /// Context choices that fit the current model on this device.
    public var allowedContextLengths: [AppContextLengthOption] {
        let maximum = memoryCap?.contextTokens ?? Int.max
        return AppContextLengthOption.allCases.filter { $0.tokens <= maximum }
    }

    public var allowedPrefillChunkTokens: [Int] {
        let maximum = memoryCap?.prefillChunkTokens ?? Int.max
        return AppRuntimeOptions.allowedPrefillChunkTokens.filter { $0 <= maximum }
    }

    /// The options actually loaded and requested. Settings are shared across
    /// models in one folder, so the cap is applied here rather than written
    /// back into `runtimeOptions`: switching to a smaller model must not
    /// inherit the phone's 3.8-vision limits.
    public var effectiveRuntimeOptions: AppRuntimeOptions {
        var options = runtimeOptions
        if let minimumSlots = AppRuntimeOptions.allowedSlotCounts.first(
            where: { $0 >= installDescriptor.architecture.topKExperts }) {
            options.expertCacheSlots = max(options.expertCacheSlots, minimumSlots)
        }
        if let memoryCap {
            options.expertCacheSlots = min(options.expertCacheSlots,
                                           memoryCap.expertCacheSlots)
            options.prefillChunkTokens = min(options.prefillChunkTokens,
                                             memoryCap.prefillChunkTokens)
        }
        return options
    }

    public var modelChoices: [AppModelChoice] { AppModelChoice.allCases }

    public var packageRootURL: URL? { AppModelLocation.packageRootURL() }

    public var installRequirement: AppModelInstallRequirement? {
        installReadiness.requirement
    }

    public var isInstallingModel: Bool { installState.isInstalling }

    public var canInstallModel: Bool {
        guard case .ready = installReadiness else { return false }
        return !isRunning && !loadState.isLoading && !isInstallingModel
            && requiresModelInstallation
            && installDescriptor.supportsRemoteInstall
    }

    public var canCancelInstall: Bool { installState.canCancel }

    public var installDownloadedBytes: UInt64? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        return min(addition.overflow ? UInt64.max : addition.partialValue, total)
    }

    public var installTotalBytes: UInt64? {
        guard case .copyingPayload(_, _, let total) = installState else {
            return nil
        }
        return total
    }

    public var installReusedBytes: UInt64? {
        guard case .copyingPayload(let reused, _, _) = installState else {
            return nil
        }
        return reused
    }

    public var installDownloadedThisRunBytes: UInt64? {
        guard case .copyingPayload(_, let downloaded, _) = installState else {
            return nil
        }
        return downloaded
    }

    public var installProgressFraction: Double? {
        guard case .copyingPayload(let reused, let downloaded, let total) = installState,
              total > 0 else {
            return nil
        }
        let addition = reused.addingReportingOverflow(downloaded)
        let done = addition.overflow ? UInt64.max : addition.partialValue
        return min(max(Double(done) / Double(total), 0), 1)
    }

    public var installPhaseLabel: String {
        switch installState {
        case .idle: return "Model required"
        case .checking: return "Checking installation"
        case .downloadingMetadata: return "Downloading metadata"
        case .planning: return "Planning installation"
        case .reservingOutput: return "Reserving storage"
        case .copyingPayload: return "Downloading model"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing installation"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding download"
        case .cancelled: return "Download paused"
        case .recoverable: return "Saved download needs attention"
        case .installed: return "Model installed"
        case .failed: return "Installation failed"
        }
    }

    public var canRun: Bool {
        let hasText = !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let hasImages = !pendingImages.isEmpty
        return !isRunning && isModelAvailable && !loadState.isLoading
            && !hasStaleLoadedRuntime
            && (hasText || hasImages)
            && (!hasImages || supportsVision)
    }

    /// The selected install is one of the vision builds and is complete.
    public var supportsVision: Bool {
        isModelInstalled
            && installDescriptor.requiresVision
            && (installDescriptor.architecture.isQwen3_6
                || installDescriptor.architecture.isQwen3_8)
    }

    /// Attaching is offered only where the runtime can consume the image.
    public var canAttachImage: Bool {
        supportsVision && !isRunning && !loadState.isLoading
    }

    public func addPendingImages(_ images: [AppPendingImage]) {
        guard canAttachImage else { return }
        let room = AppImageAttachment.maximumImagesPerTurn - pendingImages.count
        guard room > 0 else { return }
        pendingImages.append(contentsOf: images.prefix(room))
    }

    public func removePendingImage(id: UUID) {
        pendingImages.removeAll { $0.id == id }
    }

    public func clearPendingImages() {
        pendingImages.removeAll()
    }

    public var canCancel: Bool { isRunning && !isCancellationPending }

    /// The local OpenAI-compatible server is a sibling process the app spawns,
    /// which is a mechanism that exists on macOS only. Every platform can ask;
    /// only macOS can start it.
    public static var localServerIsSupported: Bool {
        #if canImport(AppKit)
        true
        #else
        false
        #endif
    }

    public var canStartLocalServer: Bool {
        Self.localServerIsSupported
            && isModelInstalled && !isRunning && !loadState.isLoading && !localServerState.isActive
            && (1...65_535).contains(localServerPort)
    }

    public var canStopLocalServer: Bool { localServerState.isActive }

    public var hasOutputTranscript: Bool {
        !outputPromptText.isEmpty || !outputText.isEmpty
    }

    public var outputResponsePlainText: String {
        generationTranscriptMailbox?.completeText ?? outputText
    }

    public var outputConversationPlainText: String {
        let response = outputResponsePlainText
        switch (outputPromptText.isEmpty, response.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return "You:\n\(outputPromptText)"
        case (true, false):
            return "Answer:\n\(response)"
        case (false, false):
            return "You:\n\(outputPromptText)\n\nAnswer:\n\(response)"
        }
    }

    public var liveTokensPerSecond: Double {
        liveElapsedDecodeSeconds > 0 ? Double(liveTokenCount) / liveElapsedDecodeSeconds : 0
    }

    public var presentation: AppPresentationState {
        AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: requiresModelInstallation,
            installState: installState,
            installReadiness: installReadiness,
            loadState: loadState,
            hasStaleRuntime: hasStaleLoadedRuntime,
            isRunning: isRunning,
            isGenerationCancellationPending: isCancellationPending,
            generationPhase: phase,
            livePrefillDone: livePrefillDone,
            livePrefillTotal: livePrefillTotal,
            livePrefillCached: livePrefillCached,
            lastStopReason: diagnostics?.stopReason))
    }

    public var currentProcessMemoryBytes: UInt64? {
        guard loadState.isReady || isRunning else { return nil }
        if let reporter = client as? any AppInferenceMemoryReporting,
           let bytes = reporter.currentInferenceMemoryBytes {
            return bytes
        }
        return memorySampler.sample()
    }

    /// The jetsam ceiling for this process, when the platform reports one.
    ///
    /// Shown beside the live footprint so the cap in
    /// `AppModelMemoryBudget` can be checked against the machine it is
    /// running on instead of the iPhone figure it was priced for. nil on
    /// macOS, where no such limit exists.
    public var processMemoryLimitBytes: UInt64? {
        AppMemorySampler.memoryLimitBytes
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox? {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
    }

    private var currentRuntimeKey: AppLoadedRuntimeKey {
        AppLoadedRuntimeKey(modelDirectory: URL(fileURLWithPath: modelPathText),
                            maxContextTokens: effectiveMaxContextTokens,
                            options: effectiveRuntimeOptions,
                            forceLogitsHead: currentForceLogitsHead)
    }

    private var currentForceLogitsHead: Bool {
        temperature != 0
    }

    public func setModelChoice(_ choice: AppModelChoice) {
        // A checkout resolves rows against `models/`; an app bundle (iOS) has
        // no checkout, so the row resolves to the app-owned Models directory.
        // Gating on `packageRootURL` here is what made every iOS selection a
        // silent no-op.
        setModelURL(
            AppModelLocation.installURL(for: choice, packageRoot: packageRootURL),
            descriptor: choice.descriptor)
    }

    /// Points the app at an existing `.finch` directory chosen by the user.
    ///
    /// A directory with no `manifest.json` is not a model, and selecting one
    /// must not silently fall back to the default descriptor — that turned
    /// "I picked the wrong folder" into "the app switched to Gemma and
    /// offered a download". The error is surfaced by the install screen and
    /// the current selection is left alone.
    public func setModelURL(_ url: URL) {
        let manifest = url.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            modelSelectionError = url.lastPathComponent
                + " is not a .finch model folder: manifest.json was not found. "
                + "Choose the model directory itself, not its parent."
            return
        }
        modelSelectionError = nil
        setModelURL(url, descriptor: AppModelInstallationProbe.matchingDescriptor(at: url))
    }

    /// Reports a picker failure (the file importer's own error) in the same
    /// place as a rejected selection.
    public func reportModelSelectionError(_ message: String) {
        modelSelectionError = message
    }

    private func setModelURL(_ url: URL, descriptor: AppModelInstallDescriptor) {
        guard !isRunning else { return }
        modelSelectionError = nil
        let path = url.standardizedFileURL.path
        let newChoice = Self.modelChoice(for: url, descriptor: descriptor)
        guard path != modelPathText || descriptor != installer.descriptor else { return }

        stopLocalServer()
        pendingImages.removeAll()
        modelPathText = path
        selectedModelChoice = newChoice
        installer.cancel()
        installer = Self.makeInstaller(descriptor: descriptor)
        applyPersistedSettings(
            forModelDirectory: URL(fileURLWithPath: path, isDirectory: true))
        // The persistence store may be off (tests) or the loaded settings may
        // predate the model the row now selects; the current descriptor is the
        // authority either way.
        sanitizeRuntimeOptionsForCurrentModel()
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        installGeneration &+= 1
        installTask?.cancel()
        installTask = nil
        resetInstallETA()
        installState = .idle
        pendingExplicitLoadRuntimeKey = nil
        activeRunRuntimeKey = nil
        loadedRuntimeKey = nil
        loadState = .notLoaded
        diagnostics = nil
        error = nil
        phase = .idle
        installationStatus = AppModelInstallationProbe.status(
            at: URL(fileURLWithPath: path), descriptor: descriptor)
        refreshInstallReadiness()

        if let lifecycle = client as? AppModelLifecycleClient {
            unloadGeneration &+= 1
            let generation = unloadGeneration
            let task = Task { [weak self, lifecycle] in
                await lifecycle.unload()
                self?.clearUnloadTask(generation: generation)
            }
            unloadTask = task
        }
    }

    public func loadModel() {
        guard canLoadModel else { return }
        beginLoad()
    }

    public func perform(_ action: AppModelAction) {
        switch action {
        case .install: installModel()
        case .cancelInstall: cancelInstall()
        case .load, .retryLoad: loadModel()
        case .cancelLoad: cancelLoad()
        case .reload: reloadModel()
        case .unload: unloadModel()
        }
    }

    public func setNewlineShortcut(_ shortcut: AppNewlineShortcut) {
        guard newlineShortcut != shortcut else { return }
        newlineShortcut = shortcut
        persistSettings()
    }

    public func setSentPromptBehavior(_ behavior: AppSentPromptBehavior) {
        guard sentPromptBehavior != behavior else { return }
        sentPromptBehavior = behavior
        persistSettings()
    }

    public func reloadModel() {
        guard canReloadModel else { return }
        beginLoad()
    }

    private func beginLoad() {
        guard let lifecycle = client as? AppModelLifecycleClient else {
            loadState = .failed(.modelLoadFailed("This client has no model load lifecycle."))
            return
        }
        let directory = URL(fileURLWithPath: modelPathText)
        let maxContext = effectiveMaxContextTokens
        let options = effectiveRuntimeOptions
        let forceLogitsHead = currentForceLogitsHead
        let runtimeKey = AppLoadedRuntimeKey(modelDirectory: directory,
                                             maxContextTokens: maxContext,
                                             options: options,
                                             forceLogitsHead: forceLogitsHead)
        let pendingUnload = unloadTask
        loadGeneration &+= 1
        let generation = loadGeneration
        pendingExplicitLoadRuntimeKey = runtimeKey
        error = nil
        loadState = .loading(.validatingDirectory)
        loadTask = Task.detached { [weak self, lifecycle, pendingUnload] in
            do {
                await pendingUnload?.value
                try Task.checkCancellation()
                try await lifecycle.ensureLoaded(modelDirectory: directory,
                                                 maxContextTokens: maxContext,
                                                 options: options,
                                                 forceLogitsHead: forceLogitsHead) { [weak self] state in
                    Task { @MainActor in
                        self?.applyLoadState(state, generation: generation)
                    }
                }
            } catch is CancellationError {
            } catch let appError as AppInferenceError {
                await self?.applyLoadState(.failed(appError), generation: generation)
            } catch {
                await self?.applyLoadState(
                    .failed(.modelLoadFailed("\(error)")),
                    generation: generation)
            }
            await self?.clearLoadTask(generation: generation)
        }
    }

    public func cancelLoad() {
        guard canCancelLoad, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .cancelling
        loadGeneration &+= 1
        loadTask?.cancel()
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func unloadModel() {
        guard canUnloadModel, let lifecycle = client as? AppModelLifecycleClient else { return }
        loadState = .unloading
        unloadGeneration &+= 1
        let generation = unloadGeneration
        unloadTask = Task { [weak self, lifecycle] in
            await lifecycle.unload()
            guard let self, generation == self.unloadGeneration else { return }
            self.loadedRuntimeKey = nil
            self.liveMemoryBytes = nil
            self.loadState = .notLoaded
            self.clearUnloadTask(generation: generation)
        }
    }

    public func startLocalServer() {
        guard canStartLocalServer else { return }
        stopLocalServer()
        localServerState = .starting
        localServerLog = "Starting server..."

        #if canImport(AppKit)
        guard let executable = Self.localServerExecutableURL() else {
            localServerState = .failed("FinchMoEServer executable was not found. Build the FinchMoEServer product first.")
            return
        }

        let process = Process()
        let outputPipe = Pipe()
        process.executableURL = executable
        process.arguments = [
            "--model", URL(fileURLWithPath: modelPathText).standardizedFileURL.path,
            "--port", "\(localServerPort)",
            "--model-id", localServerModelID.isEmpty ? installDescriptor.shortName : localServerModelID,
            "--max-context", "\(effectiveMaxContextTokens)",
            "--verify", effectiveRuntimeOptions.modelVerification.rawValue,
        ]
        process.standardOutput = outputPipe
        process.standardError = outputPipe
        process.terminationHandler = { [weak self] process in
            Task { @MainActor in
                guard let self, self.localServerProcess === process else { return }
                self.localServerProcess = nil
                if process.terminationStatus == 0 {
                    self.localServerState = .stopped
                } else {
                    self.localServerState = .failed("FinchMoEServer exited with status \(process.terminationStatus).")
                }
            }
        }
        outputPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                guard let self, self.localServerProcess === process else { return }
                self.localServerLog.append(text)
                if text.contains("FinchMoEServer ready") {
                    self.localServerState = .running(
                        URL(string: "http://127.0.0.1:\(self.localServerPort)")!)
                }
            }
        }

        do {
            try process.run()
            localServerProcess = process
        } catch {
            outputPipe.fileHandleForReading.readabilityHandler = nil
            localServerState = .failed("Failed to start FinchMoEServer: \(error)")
        }
        #else
        localServerState = .failed(
            "The local HTTP server runs as a sibling process, which is available in the Mac app only.")
        #endif
    }

    public func stopLocalServer() {
        #if canImport(AppKit)
        guard let process = localServerProcess else {
            localServerState = .stopped
            localServerLog = ""
            return
        }
        process.terminate()
        localServerProcess = nil
        localServerState = .stopped
        #else
        localServerState = .stopped
        localServerLog = ""
        #endif
    }

    public func installModel() {
        guard !isRunning, !loadState.isLoading, !isInstallingModel,
              requiresModelInstallation else {
            return
        }
        refreshInstallReadiness()
        guard canInstallModel else { return }
        installTask?.cancel()
        installer.cancel()
        resetInstallETA()
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .checking
        installTask = Task { [weak self, installer] in
            do {
                for try await event in installer.installDefaultModel(outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.applyInstallEvent(event, generation: generation)
                }
                self?.finishInstallStream(generation: generation)
            } catch is CancellationError {
                self?.finishInstallCancellation(generation: generation)
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func cancelInstall() {
        guard canCancelInstall else { return }
        installState = .cancelling
        installer.cancel()
    }

    public var hasPartialModelDownload: Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: modelPathText) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscardModelDownload: Bool {
        hasPartialModelDownload && !isInstallingModel && !isRunning
    }

    public func discardModelDownload() {
        guard canDiscardModelDownload else { return }
        let outputDirectory = URL(fileURLWithPath: modelPathText)
        installGeneration &+= 1
        let generation = installGeneration
        installState = .discarding
        installTask = Task { [weak self, installer] in
            do {
                try await installer.discardPartialInstall(
                    outputDirectory: outputDirectory)
                guard let self, generation == self.installGeneration else { return }
                self.installTask = nil
                self.installState = .idle
                self.refreshInstallReadiness()
            } catch {
                self?.finishInstallFailure(error, generation: generation)
            }
        }
    }

    public func refreshInstallReadiness() {
        refreshInstallReadiness(
            at: URL(fileURLWithPath: modelPathText, isDirectory: true).standardizedFileURL)
    }

    public func recheckModelAtCurrentLocation() {
        let directory = URL(fileURLWithPath: modelPathText, isDirectory: true)
            .standardizedFileURL
        modelPathText = directory.path
        refreshInstallReadiness(at: directory)
    }

    private func refreshInstallReadiness(at outputDirectory: URL) {
        installationStatus = AppModelInstallationProbe.status(
            at: outputDirectory,
            descriptor: installer.descriptor)
        guard !isModelInstalled else { return }
        installReadiness = .checking
        do {
            let requirement = try installer.checkInstallRequirement(
                outputDirectory: outputDirectory)
            installReadiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            installReadiness = .failed("\(error)")
        }
    }

    private func applyInstallEvent(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == installGeneration else { return }
        switch event {
        case .checking:
            resetInstallETA()
            installState = .checking
        case .downloadingMetadata:
            resetInstallETA()
            installState = .downloadingMetadata
        case .planning:
            resetInstallETA()
            installState = .planning
        case .reservingOutput:
            resetInstallETA()
            installState = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            installState = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateInstallETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetInstallETA()
            installState = .hashingOutput(file)
        case .finalizing:
            resetInstallETA()
            installState = .finalizing
        case .installed(let directory):
            resetInstallETA()
            let directory = directory.standardizedFileURL
            installationStatus = AppModelInstallationProbe.status(
                at: directory,
                descriptor: installer.descriptor)
            guard installationStatus == .complete else {
                finishInstallFailure(
                    RepackError.configurationInvalid(detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            installState = .installed(modelDirectory: directory)
            installTask = nil
            modelPathText = directory.path
            // The installer promotes into a `realpath`-resolved directory,
            // which can differ from the row's logical path (the checkout's
            // `models/` is a symlink; on iOS `/var` is). `modelChoice`
            // canonicalizes both sides, so this cannot clear a correct
            // highlight.
            selectedModelChoice = Self.modelChoice(for: directory,
                                                   descriptor: installer.descriptor)
            loadState = .notLoaded
        }
    }

    private func finishInstallStream(generation: UInt64) {
        guard generation == installGeneration, installTask != nil else { return }
        if installState == .cancelling {
            finishInstallCancellation(generation: generation)
        } else if !isModelInstalled {
            finishInstallFailure(
                RepackError.configurationInvalid(detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishInstallCancellation(generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        installState = .cancelled
        resetInstallETA()
        refreshInstallReadiness()
    }

    private func updateInstallETA(
        reusedBytes: UInt64,
        downloadedThisRunBytes: UInt64,
        totalBytes: UInt64
    ) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        let timestamp = installETATimestamp
        setInstallETAPresentation(
            installETAEstimator.update(observation, timestamp: timestamp))
    }

    private var installETATimestamp: Double {
        let components = installETAOrigin.duration(to: installETAClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetInstallETA() {
        installETAEstimator.reset()
        installETAPresentation = .hidden
        installETAText = nil
    }

    private func setInstallETAPresentation(
        _ presentation: DownloadETAPresentation
    ) {
        installETAPresentation = presentation
        installETAText = DownloadETAFormatter.string(for: presentation)
    }

    private func applyPersistedSettings(forModelDirectory modelDirectory: URL) {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        runtimeOptions = AppRuntimeOptions(persisted: settings)
        sanitizeRuntimeOptionsForCurrentModel()
        maxContextTokens = settings.contextTokens
        temperature = settings.temperature
        topKEnabled = settings.topKEnabled
        topK = settings.topK
        topPEnabled = settings.topPEnabled
        topP = settings.topP
        newlineShortcut = settings.newlineShortcut
        sentPromptBehavior = settings.sentPromptBehavior
        promptReuseEnabled = settings.promptReuseEnabled
    }

    /// Keeps a load-time option inside what the selected install can actually
    /// run. Int8 KV is Qwen 3.6 only; the runner traps rather than misread a
    /// 3.8 timeline, so the app must fall back before asking it to load.
    private func sanitizeRuntimeOptionsForCurrentModel() {
        if !runtimeOptions.kvCacheMode.isAvailable(
            for: installDescriptor.architecture) {
            runtimeOptions.kvCacheMode = .fp16
        }
        // 3.8 routes to ten experts per token; 8 slots cannot hold one
        // layer's working set and `makeExpertCachePlan` traps on it.
        let minimumSlots = installDescriptor.architecture.topKExperts
        if runtimeOptions.expertCacheSlots < minimumSlots,
           let replacement = AppRuntimeOptions.allowedSlotCounts.first(
               where: { $0 >= minimumSlots }) {
            runtimeOptions.expertCacheSlots = replacement
        }
    }

    private func persistSettings() {
        guard settingsPersistenceEnabled else { return }
        let settings = MacAppSettings(
            contextTokens: maxContextTokens,
            expertCacheSlots: runtimeOptions.expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: runtimeOptions.prefillEnabled,
            prefillChunkTokens: runtimeOptions.prefillChunkTokens,
            expertCachePolicy: runtimeOptions.expertCachePolicy,
            promptReuseEnabled: promptReuseEnabled,
            modelVerification: runtimeOptions.modelVerification,
            newlineShortcut: newlineShortcut,
            sentPromptBehavior: sentPromptBehavior,
            kvCacheMode: runtimeOptions.kvCacheMode)
        let modelDirectory = URL(fileURLWithPath: modelPathText, isDirectory: true)
        try? MacAppSettingsFileStore.save(
            settings,
            forModelDirectory: modelDirectory)
    }

    private func finishInstallFailure(_ error: Error, generation: UInt64) {
        guard generation == installGeneration else { return }
        installTask = nil
        resetInstallETA()
        let hasSavedDownload = hasPartialModelDownload
        installState = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            let requirement = AppModelInstallRequirement(probePath: path,
                                                          requiredBytes: required,
                                                          availableBytes: available)
            installReadiness = .insufficientSpace(requirement)
        } else {
            refreshInstallReadiness()
            if hasSavedDownload {
                installState = .recoverable("\(error)")
            }
        }
    }

    func applyLoadState(_ state: AppModelLoadState) {
        applyLoadState(state, generation: loadGeneration)
    }

    private func applyLoadState(_ state: AppModelLoadState, generation: UInt64) {
        guard generation == loadGeneration else { return }
        if case .ready(let directory, _) = state,
           directory.standardizedFileURL.path
            != URL(fileURLWithPath: modelPathText).standardizedFileURL.path {
            return
        }
        loadState = state
        switch state {
        case .notLoaded:
            loadedRuntimeKey = nil
        case .loading, .cancelling, .unloading:
            break
        case .ready(_, let seconds):
            loadedRuntimeKey = pendingExplicitLoadRuntimeKey
                ?? activeRunRuntimeKey
                ?? currentRuntimeKey
            pendingExplicitLoadRuntimeKey = nil
            _ = seconds
        case .failed(let loadError):
            pendingExplicitLoadRuntimeKey = nil
            error = loadError
        }
    }

    public func clearOutput() {
        guard !isRunning else { return }
        outputPromptText = ""
        outputText = ""
        pendingImages.removeAll()
        generationTranscriptMailbox?.reset()
        diagnostics = nil
        error = nil
        clearActiveSessionMessages()
    }

    public func run() {
        guard canRun else { return }
        let request: AppGenerationRequest
        do {
            request = try makeRequest()
        } catch let appError as AppInferenceError {
            error = appError
            return
        } catch {
            let appError = AppInferenceError.unknown("\(error)")
            self.error = appError
            return
        }
        persistSettings()

        generationTranscriptMailbox?.reset()
        outputPromptText = request.prompt
        outputText = ""
        beginChatExchange(prompt: request.prompt, images: request.images)
        diagnostics = nil
        error = nil
        hasHandledTerminalEvent = false
        activeRunRuntimeKey = AppLoadedRuntimeKey(
            modelDirectory: request.modelDirectory,
            maxContextTokens: request.maxContextTokens,
            options: request.runtimeOptions,
            forceLogitsHead: !request.isPureGreedy)
        isCancellationPending = false
        liveTokenCount = 0
        liveElapsedDecodeSeconds = 0
        livePrefillDone = 0
        livePrefillTotal = 0
        livePrefillCached = 0
        liveMemoryBytes = nil
        // An image turn spends its first seconds in the vision tower, before
        // the text engine has any prefill counts to report. Naming that phase
        // separately is what stops a phone from showing "Prefill" for a
        // picture that is still being encoded.
        phase = request.hasImageContent ? .processingImage : .prefill
        runState = .running
        if sentPromptBehavior == .clear {
            promptText = ""
            pendingImages.removeAll()
        }

        runTask = Task.detached { [weak self, client, request] in
            guard let self else { return }
            do {
                for try await event in client.generate(request) {
                    await self.apply(event)
                }
            } catch let appError as AppInferenceError {
                await self.finishStreamFailure(appError)
            } catch {
                await self.finishStreamFailure(.unknown("\(error)"))
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        isCancellationPending = true
        client.cancel()
    }

    public func makeRequest() throws -> AppGenerationRequest {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: modelPathText),
            prompt: promptText,
            images: pendingImages.map(\.attachment),
            history: activeSession.replayableMessages.map { message in
                chatTurn(for: message)
            },
            maxNewTokens: maxNewTokensOverride ?? effectiveMaxContextTokens,
            maxContextTokens: effectiveMaxContextTokens,
            temperature: Float(temperature),
            topK: topKEnabled ? topK : nil,
            topP: topKEnabled && topPEnabled ? Float(topP) : nil,
            repetitionPenalty: 1.0,
            runtimeOptions: effectiveRuntimeOptions,
            promptReuseEnabled: promptReuseEnabled)
        try request.validate(requireModelDirectory: true)
        return request
    }

    func apply(_ event: AppInferenceEvent) {
        switch event {
        case .prefillProgress(let done, let total, let cached):
            phase = .prefill
            livePrefillDone = done
            livePrefillTotal = total
            livePrefillCached = cached
        case .token(let token):
            phase = .decode
            liveTokenCount = token.index + 1
            liveElapsedDecodeSeconds = token.elapsedDecodeSeconds
            if let reporter = client as? any AppInferenceMemoryReporting {
                liveMemoryBytes = reporter.currentInferenceMemoryBytes
            } else {
                liveMemoryBytes = memorySampler.sample()
            }
            if !token.textDelta.isEmpty {
                outputText += token.textDelta
            }
            updateStreamingAssistantMessage(
                with: generationTranscriptMailbox?.completeText ?? outputText)
        case .finished(let diagnostics):
            finishSuccessfully(diagnostics)
        case .cancelled(let diagnostics):
            finishCancelled(diagnostics)
        case .failed(let appError, let partial):
            diagnostics = partial
            materializeServiceTranscript()
            finishWithError(appError)
        }
    }

    private func finishSuccessfully(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        finishTerminalRun()
    }

    private func finishCancelled(_ diagnostics: AppDiagnostics) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        materializeServiceTranscript()
        self.diagnostics = diagnostics
        error = .cancelled
        finishTerminalRun()
    }

    private func materializeServiceTranscript() {
        guard let reporter = client as? any AppInferenceTranscriptReporting else { return }
        outputText = reporter.generationTranscriptMailbox.completeText
    }

    private func finishWithError(_ appError: AppInferenceError) {
        guard !hasHandledTerminalEvent else { return }
        hasHandledTerminalEvent = true
        error = appError
        finishTerminalRun()
    }

    private func finishStreamFailure(_ appError: AppInferenceError) {
        materializeServiceTranscript()
        finishWithError(appError)
    }

    private func finishTerminalRun() {
        phase = .idle
        runState = .idle
        isCancellationPending = false
        activeRunRuntimeKey = nil
        runTask = nil
        endChatExchange(failureText: chatFailureText(for: error))
    }

    private func clearLoadTask(generation: UInt64) {
        guard generation == loadGeneration else { return }
        loadTask = nil
        pendingExplicitLoadRuntimeKey = nil
    }

    private func clearUnloadTask(generation: UInt64) {
        guard generation == unloadGeneration else { return }
        unloadTask = nil
    }

    /// Picks the installer for a descriptor's route. There are two remote
    /// routes and they are not interchangeable: a published `.finch` has
    /// nothing to repack, and a safetensors checkpoint has no finished layout
    /// to fetch. A descriptor with neither falls back to the repack client,
    /// which is what the probe-only entries have always used and which reports
    /// `supportsRemoteInstall == false` for them either way.
    private static func makeInstaller(
        descriptor: AppModelInstallDescriptor
    ) -> any AppModelInstallerClient {
        if let distribution = FinchDistributionInstallerClient(descriptor: descriptor) {
            return distribution
        }
        return RepackModelInstallerClient(descriptor: descriptor)
    }

    /// A row's destination is checkout-relative on the Mac and app-owned inside
    /// a bundle; both sides are canonicalized before comparing, because
    /// installers hand back `realpath`-resolved directories (the checkout's
    /// `models/` is a symlink here, and `/var` is one on iOS) while the row
    /// path is the logical one.
    private static func modelChoice(for directory: URL,
                                    descriptor: AppModelInstallDescriptor) -> AppModelChoice? {
        let directory = canonicalURL(directory)
        let candidates = AppModelChoice.allCases.filter { $0.descriptor == descriptor }
        if let root = AppModelLocation.packageRootURL() {
            return candidates.first {
                canonicalURL($0.defaultURL(packageRoot: root)) == directory
            }
        }
        return candidates.first {
            canonicalURL(AppModelLocation.appInstallURL(for: $0)) == directory
        }
    }

    private static func canonicalURL(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath().standardizedFileURL
    }

    #if canImport(AppKit)
    /// The sibling `FinchMoEServer` binary, next to this executable or in the
    /// checkout's build directory. macOS-only, like the server itself.
    private static func localServerExecutableURL() -> URL? {
        let fileManager = FileManager.default
        let executableName = "FinchMoEServer"
        var candidates: [URL] = []
        if let executableURL = Bundle.main.executableURL {
            candidates.append(executableURL.deletingLastPathComponent()
                .appendingPathComponent(executableName, isDirectory: false))
        }
        if let root = AppModelLocation.packageRootURL() {
            candidates.append(root.appendingPathComponent(".build/release/\(executableName)", isDirectory: false))
            candidates.append(root.appendingPathComponent(".build/debug/\(executableName)", isDirectory: false))
        }
        return candidates.first { fileManager.isExecutableFile(atPath: $0.path) }
    }
    #endif
}

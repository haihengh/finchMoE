import Foundation
import Testing
@testable import FinchMoEAppCore

@Suite struct MacAppSettingsTests {
    @Test func settingsFileLivesBesideModelDirectory() {
        let model = URL(fileURLWithPath: "/tmp/FinchMoE/gemma4.finch",
                        isDirectory: true)
        #expect(MacAppSettingsFileStore.fileURL(forModelDirectory: model).path
            == "/tmp/FinchMoE/mac-app-settings.json")
    }

    @Test func missingFileCreatesReadableDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.finch", isDirectory: true)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        #expect(FileManager.default.fileExists(atPath: fileURL.path))
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func malformedFileIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try Data("not json".utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(contentsOf: fileURL))
        #expect(decoded == MacAppSettings())
    }

    @Test func invalidValuesAreReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let invalid = MacAppSettings(contextTokens: 123)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        try JSONEncoder().encode(invalid).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
    }

    @Test func legacySettingsGetChatComposerDefaultsWithoutLosingValues() throws {
        let data = Data("""
        {
          "version": 1,
          "contextTokens": 8192,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": false,
          "topK": 32,
          "topPEnabled": false,
          "topP": 0.8,
          "prefillEnabled": false
        }
        """.utf8)

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)

        #expect(settings.contextTokens == 8_192)
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(!settings.topKEnabled)
        #expect(settings.topK == 32)
        #expect(!settings.topPEnabled)
        #expect(settings.topP == 0.8)
        #expect(!settings.prefillEnabled)
        #expect(settings.modelVerification == .automatic)
        // No migration marker: a file from the single-shot app is moved onto
        // the chat conventions.
        #expect(settings.newlineShortcut == .shiftReturn)
        #expect(settings.sentPromptBehavior == .clear)
        #expect(settings.chatComposerMigrated)
    }

    @Test func theComposerMigrationRunsOnceAndKeepsLaterChoices() throws {
        let beforeChat = Data("""
        {
          "version": 1,
          "contextTokens": 4096,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "newlineShortcut": "return",
          "sentPromptBehavior": "keep"
        }
        """.utf8)

        let migrated = try JSONDecoder().decode(MacAppSettings.self, from: beforeChat)
        #expect(migrated.newlineShortcut == .shiftReturn)
        #expect(migrated.sentPromptBehavior == .clear)

        // Saving writes the marker, so the user's next choice sticks.
        let chosen = MacAppSettings(
            newlineShortcut: .return,
            sentPromptBehavior: .keep)
        let roundTripped = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(chosen))
        #expect(roundTripped.newlineShortcut == .return)
        #expect(roundTripped.sentPromptBehavior == .keep)
    }

    @Test(arguments: AppNewlineShortcut.allCases)
    func newlineShortcutRoundTrips(_ shortcut: AppNewlineShortcut) throws {
        let initial = MacAppSettings(newlineShortcut: shortcut)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func sendMessageOptionsUseUserFacingOrderAndLabels() {
        #expect(AppNewlineShortcut.sendMessageOptions == [.shiftReturn, .return])
        #expect(AppNewlineShortcut.shiftReturn.sendMessageLabel == "Return")
        #expect(AppNewlineShortcut.return.sendMessageLabel == "Command-Return")
    }

    @Test(arguments: AppModelVerification.allCases)
    func modelVerificationRoundTrips(_ mode: AppModelVerification) throws {
        let initial = MacAppSettings(modelVerification: mode)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func unknownPersistedVerificationDoesNotWipeTheOtherSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        // Written by a future build, read back by this one. `loadOrCreate`
        // answers *any* decode throw by deleting the file, so decoding the raw
        // string rather than the enum is what keeps one unknown mode from
        // resetting slots, temperature and the shortcut along with it.
        try Data("""
        {
          "version": 1,
          "contextTokens": 8192,
          "expertCacheSlots": 24,
          "temperature": 0.4,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "modelVerification": "quantum-notarized",
          "newlineShortcut": "shift-return"
        }
        """.utf8).write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings.modelVerification == .automatic)
        #expect(settings.contextTokens == 8_192)
        #expect(settings.expertCacheSlots == 24)
        #expect(settings.temperature == 0.4)
        #expect(settings.newlineShortcut == .shiftReturn)
    }

    @Test(arguments: AppSentPromptBehavior.allCases)
    func sentPromptBehaviorRoundTrips(_ behavior: AppSentPromptBehavior) throws {
        let initial = MacAppSettings(sentPromptBehavior: behavior)
        let decoded = try JSONDecoder().decode(
            MacAppSettings.self,
            from: JSONEncoder().encode(initial))

        #expect(decoded == initial)
    }

    @Test func invalidNewlineShortcutIsReplacedWithDefaults() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let fileURL = MacAppSettingsFileStore.fileURL(forModelDirectory: model)
        let invalid = Data("""
        {
          "version": 1,
          "contextTokens": 4096,
          "expertCacheSlots": 16,
          "temperature": 0.2,
          "topKEnabled": true,
          "topK": 64,
          "topPEnabled": true,
          "topP": 0.95,
          "prefillEnabled": true,
          "newlineShortcut": "invalid"
        }
        """.utf8)
        try invalid.write(to: fileURL)

        let settings = MacAppSettingsFileStore.loadOrCreate(forModelDirectory: model)

        #expect(settings == MacAppSettings())
    }

    @MainActor
    @Test func appModelLoadsAndSavesPersistedSettings() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true)
        let initial = MacAppSettings(
            contextTokens: 8_192,
            expertCacheSlots: 24,
            temperature: 0.4,
            topKEnabled: false,
            topK: 32,
            topPEnabled: false,
            topP: 0.8,
            prefillEnabled: false,
            modelVerification: .trustedInstall,
            newlineShortcut: .shiftReturn,
            sentPromptBehavior: .clear)
        try MacAppSettingsFileStore.save(initial, forModelDirectory: modelDirectory)

        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)
        #expect(model.maxContextTokens == 8_192)
        #expect(model.runtimeOptions.expertCacheSlots == 24)
        #expect(model.temperature == 0.4)
        #expect(!model.topKEnabled)
        #expect(model.topK == 32)
        #expect(!model.topPEnabled)
        #expect(model.topP == 0.8)
        #expect(!model.runtimeOptions.prefillEnabled)
        #expect(model.runtimeOptions.modelVerification == .trustedInstall)
        #expect(model.newlineShortcut == .shiftReturn)
        #expect(model.sentPromptBehavior == .clear)

        model.temperature = 0.6
        model.runtimeOptions.expertCacheSlots = 32
        model.runtimeOptions.prefillEnabled = true
        model.runtimeOptions.modelVerification = .fullSha256
        let beforeGenerate = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(beforeGenerate == initial)

        model.loadState = .ready(modelDirectory: modelDirectory, loadSeconds: 0)
        model.promptText = "Save these settings"
        model.run()
        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.temperature == 0.6)
        #expect(saved.expertCacheSlots == 32)
        #expect(saved.prefillEnabled)
        #expect(saved.modelVerification == .fullSha256)
        #expect(saved.newlineShortcut == .shiftReturn)
        #expect(saved.sentPromptBehavior == .clear)
        model.cancel()
    }

    @MainActor
    @Test func newlineShortcutPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)

        model.setNewlineShortcut(.shiftReturn)

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.newlineShortcut == .shiftReturn)
    }

    @MainActor
    @Test func sentPromptBehaviorPersistsImmediately() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let modelDirectory = root.appendingPathComponent("gemma4.finch", isDirectory: true)
        let model = AppModel(
            modelDirectory: modelDirectory,
            settingsPersistenceEnabled: true)

        model.setSentPromptBehavior(.clear)

        let saved = MacAppSettingsFileStore.loadOrCreate(
            forModelDirectory: modelDirectory)
        #expect(saved.sentPromptBehavior == .clear)
    }

    @MainActor
    @Test func changingModelDirectoryLoadsItsNewlineShortcut() throws {
        let root = try makeTemporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first/model.finch", isDirectory: true)
        let second = root.appendingPathComponent("second/model.finch", isDirectory: true)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .return),
            forModelDirectory: first)
        try MacAppSettingsFileStore.save(
            MacAppSettings(newlineShortcut: .shiftReturn),
            forModelDirectory: second)
        let model = AppModel(modelDirectory: first, settingsPersistenceEnabled: true)
        #expect(model.newlineShortcut == .return)

        model.setModelURL(second)

        #expect(model.newlineShortcut == .shiftReturn)
    }

    private func makeTemporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("MacAppSettingsTests-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true)
        return root
    }
}

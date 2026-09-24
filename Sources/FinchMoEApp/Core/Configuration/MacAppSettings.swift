import Foundation

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 1

    var version: Int = currentVersion
    var contextTokens: Int = AppContextLengthOption.fourK.tokens
    var expertCacheSlots: Int = 16
    var temperature: Double = 0.2
    var topKEnabled: Bool = true
    var topK: Int = 64
    var topPEnabled: Bool = true
    var topP: Double = 0.95
    var prefillEnabled: Bool = true
    var modelVerification: AppModelVerification = .automatic
    // Chat-window conventions: Return sends, Shift-Return makes a new line,
    // and the box empties once the message is in the transcript. Files written
    // before the chat UI are moved onto these once, see `init(from:)`.
    var newlineShortcut: AppNewlineShortcut = .shiftReturn
    var sentPromptBehavior: AppSentPromptBehavior = .clear
    /// Full-attention K/V storage. Load-time: the app's reload key carries it.
    var kvCacheMode: AppKVCacheMode = .fp16
    /// True once a file has been through that move. Written as `true` by any
    /// settings this build saves, so the migration runs exactly once.
    var chatComposerMigrated: Bool = true

    private enum CodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case modelVerification
        case newlineShortcut
        case sentPromptBehavior
        case kvCacheMode
        case chatComposerMigrated
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.fourK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         modelVerification: AppModelVerification = .automatic,
         newlineShortcut: AppNewlineShortcut = .shiftReturn,
         sentPromptBehavior: AppSentPromptBehavior = .clear,
         kvCacheMode: AppKVCacheMode = .fp16,
         chatComposerMigrated: Bool = true) {
        self.version = version
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.modelVerification = modelVerification
        self.newlineShortcut = newlineShortcut
        self.sentPromptBehavior = sentPromptBehavior
        self.kvCacheMode = kvCacheMode
        self.chatComposerMigrated = chatComposerMigrated
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = try container.decode(Int.self, forKey: .version)
        contextTokens = try container.decode(Int.self, forKey: .contextTokens)
        expertCacheSlots = try container.decode(Int.self, forKey: .expertCacheSlots)
        temperature = try container.decode(Double.self, forKey: .temperature)
        topKEnabled = try container.decode(Bool.self, forKey: .topKEnabled)
        topK = try container.decode(Int.self, forKey: .topK)
        topPEnabled = try container.decode(Bool.self, forKey: .topPEnabled)
        topP = try container.decode(Double.self, forKey: .topP)
        prefillEnabled = try container.decode(Bool.self, forKey: .prefillEnabled)
        // Decoded as a raw `String`, not through the enum: `decode` of a
        // RawRepresentable throws `dataCorrupted` on a value it does not know,
        // and `loadOrCreate` answers *any* throw by deleting the file -- so one
        // unrecognized verification mode would silently reset every other
        // setting the user has. `decodeIfPresent` also covers a file written
        // before this field existed, which is why `currentVersion` stays 1.
        modelVerification = (try container.decodeIfPresent(
            String.self,
            forKey: .modelVerification))
            .flatMap(AppModelVerification.init(rawValue:)) ?? .automatic
        newlineShortcut = try container.decodeIfPresent(
            AppNewlineShortcut.self,
            forKey: .newlineShortcut) ?? .shiftReturn
        sentPromptBehavior = try container.decodeIfPresent(
            AppSentPromptBehavior.self,
            forKey: .sentPromptBehavior) ?? .clear
        kvCacheMode = try container.decodeIfPresent(
            AppKVCacheMode.self, forKey: .kvCacheMode) ?? .fp16

        // A file with no marker was written before the window became a chat:
        // Return was the newline key and the composer deliberately kept the
        // prompt for another run. In a chat window that reads as "Enter does
        // nothing and the box never clears", so those two values are moved to
        // the chat conventions once. Everything else in the file is kept, and
        // both remain user-selectable in the Settings menu.
        let alreadyMigrated = try container.decodeIfPresent(
            Bool.self,
            forKey: .chatComposerMigrated) ?? false
        if !alreadyMigrated {
            newlineShortcut = .shiftReturn
            sentPromptBehavior = .clear
        }
        chatComposerMigrated = true
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                let settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                guard settings.isValid() else { throw InvalidSettings() }
                return settings
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let settings = MacAppSettings()
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}

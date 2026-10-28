import Foundation
import FinchMoE

public enum AppExpertCachePolicy: String, CaseIterable, Codable, Sendable, Identifiable {
    case lfu
    case lru

    public var id: String { rawValue }
    public var label: String { rawValue.uppercased() }
}

public enum AppRDAdvicePolicy: String, CaseIterable, Sendable, Identifiable {
    case off
    case `default`
    case bounded
    case adaptive

    public var id: String { rawValue }
    public var label: String { rawValue.capitalized }

    var runtimeValue: RDAdvicePolicyMode {
        switch self {
        case .off: return .off
        case .default: return .default
        case .bounded: return .bounded
        case .adaptive: return .adaptive
        }
    }
}

public enum AppModelVerification: String, CaseIterable, Codable, Sendable, Identifiable {
    case automatic = "auto"
    case fullSha256 = "full-sha256"
    case trustedInstall = "trusted-install"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .automatic: return "Automatic"
        case .fullSha256: return "Full SHA-256"
        case .trustedInstall: return "Trust verified install"
        }
    }

    /// The line under the picker. The three modes are not variations on one
    /// cost: `Automatic` is the only one whose work depends on the install, and
    /// the only one that can absorb a broken receipt instead of failing.
    public var detail: String {
        switch self {
        case .automatic:
            return "Use verified-install.json when it is present and valid, and hash otherwise."
        case .fullSha256:
            return "Always hash every layer and PLE part on first use. Slowest, and independent of the receipt."
        case .trustedInstall:
            return "Require verified-install.json and size-check against it. Fails when the receipt is missing."
        }
    }

    /// The raw values are the wire format for the decode service, which rejects
    /// anything it does not know (`Entry.swift:161`) -- so an app built from
    /// this source needs a matching service, not an older one.
    var runtimeValue: ModelIntegrityPreference {
        switch self {
        case .automatic: return .automatic
        case .fullSha256: return .fullSha256
        case .trustedInstall: return .sizeCheckTrustedReceipt
        }
    }
}

public enum AppKVCacheMode: String, CaseIterable, Codable, Sendable, Identifiable {
    case fp16
    case int8
    case turbo4bit = "turbo-4bit"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fp16: return "FP16 (16-bit)"
        case .int8: return "Int8 (block scale)"
        case .turbo4bit: return "Turbo 4-bit (Planned)"
        }
    }

    /// int8 ships for the Qwen 3.6 body only: only its prefill and decode
    /// bodies quantize on write and bind the int8 timelines on read, and the
    /// runner refuses a Qwen 3.8 install rather than read a timeline nobody
    /// writes. turbo-4bit is still a planned format with no kernel behind it,
    /// so both are disabled where they cannot load rather than crashing later.
    public func isAvailable(for architecture: ArchConfig) -> Bool {
        switch self {
        case .fp16: return true
        case .int8: return architecture.isQwen3_6
        case .turbo4bit: return false
        }
    }

    /// Compact form for the settings summary line.
    public var shortLabel: String {
        switch self {
        case .fp16: return "FP16 KV"
        case .int8: return "int8 KV"
        case .turbo4bit: return "4-bit KV"
        }
    }

    public var kvStorageMode: KVStorageMode {
        switch self {
        case .int8: return .int8
        case .fp16, .turbo4bit: return .fp16
        }
    }
}

public struct AppRuntimeOptions: Equatable, Sendable {
    public static let allowedSlotCounts = RuntimeConfiguration.allowedExpertCacheSlots
    public static let allowedPrefillChunkTokens = RuntimeConfiguration.allowedPrefillChunkTokens
    /// The chunk the app starts a fresh install on, which the settings picker
    /// marks "(default)". Kept pointed at the engine's constant so the label
    /// cannot drift when the default moves — it did: the picker hardcoded 512
    /// long after the engine default became 1,024.
    public static let defaultPrefillChunkTokens = RuntimeConfiguration.defaultPrefillChunkTokens

    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    /// How full-attention K/V bytes are stored. Load-time: changing it forces a
    /// reload, which is why it lives in the runtime key as well.
    public var kvCacheMode: AppKVCacheMode

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: AppExpertCachePolicy = .lfu,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = RuntimeConfiguration.defaultPrefillChunkTokens,
                rdadvisePolicy: AppRDAdvicePolicy = .off,
                modelVerification: AppModelVerification = .automatic,
                kvCacheMode: AppKVCacheMode = .fp16) {
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.prefillEnabled = prefillEnabled
        self.prefillChunkTokens = prefillChunkTokens
        self.rdadvisePolicy = rdadvisePolicy
        self.modelVerification = modelVerification
        self.kvCacheMode = kvCacheMode
    }

    /// The one way persisted settings become runtime options. There are two
    /// callers -- `AppModel.init` and `applyPersistedSettings`, for a launch and
    /// a model switch -- and while they each built the options themselves they
    /// drifted: a newly persisted field reached one and not the other, so the
    /// setting survived changing models but not relaunching the app.
    init(persisted settings: MacAppSettings) {
        self.init(expertCacheSlots: settings.expertCacheSlots,
                  expertCachePolicy: settings.expertCachePolicy,
                  prefillEnabled: settings.prefillEnabled,
                  prefillChunkTokens: settings.prefillChunkTokens,
                  modelVerification: settings.modelVerification,
                  kvCacheMode: settings.kvCacheMode)
    }

    public func validate() throws {
        guard Self.allowedSlotCounts.contains(expertCacheSlots) else {
            throw AppInferenceError.invalidRequest(
                "expert cache slots must be one of \(Self.allowedSlotCounts)")
        }
        guard Self.allowedPrefillChunkTokens.contains(prefillChunkTokens) else {
            throw AppInferenceError.invalidRequest(
                "prefill chunk size must be one of \(Self.allowedPrefillChunkTokens)")
        }
    }

    public var prefillConfig: PrefillRuntimeConfig {
        prefillEnabled ? .production(chunkTokens: prefillChunkTokens) : .off
    }

    public var resultSummary: String {
        let prefill = prefillEnabled ? "prefill \(prefillChunkTokens)" : "prefill off"
        // Exhaustive on purpose: as a ternary this labelled every mode that was
        // not `.fullSha256` as "trusted receipt", which is the one thing
        // `automatic` is not -- it takes the receipt only when the receipt is
        // good. The summary is what the diagnostics pane reports as the settings
        // a run used, so a wrong label there is a wrong answer to "did I get the
        // fast path?".
        let verification: String
        switch modelVerification {
        case .automatic: verification = "auto verification"
        case .fullSha256: verification = "full SHA-256"
        case .trustedInstall: verification = "trusted receipt"
        }
        return "Cache \(expertCacheSlots) \(expertCachePolicy.label), \(prefill), \(kvCacheMode.shortLabel), RDADVISE \(rdadvisePolicy.label.lowercased()), \(verification)"
    }

    public static func slotsLabel(for slots: Int) -> String {
        switch slots {
        case 8: "8, -0.8 GB"
        case 10: "10, Low"
        case 16: "16, Default"
        case 24: "24, +0.8 GB"
        case 32: "32, +1.61 GB"
        default: "\(slots)"
        }
    }

    /// The prefill pipeline keeps `(depth + 1)` tiles in flight, and the
    /// engine refuses a tile plan that does not fit the slot pool. The app
    /// exposes the slot count but not the tile geometry, so derive the tile
    /// width from the budget: the low-memory 8-slot setting must use 4
    /// experts per tile at the default depth 1, or the engine rejects the
    /// first prefill after the user picked that setting to save memory.
    static func prefillTileExperts(forSlots slots: Int) -> Int {
        max(1, min(8, slots / 2))
    }

    public func resolvedRuntimeConfiguration(forceLogitsHead: Bool) throws -> RuntimeConfiguration {
        try validate()
        return RuntimeConfiguration(
            expertCacheSlots: expertCacheSlots,
            expertCachePolicy: expertCachePolicy == .lru ? .lru : .lfu,
            rdadvisePolicy: rdadvisePolicy.runtimeValue,
            prefillEnabled: prefillEnabled,
            prefillChunkTokens: prefillChunkTokens,
            forceLogitsHead: forceLogitsHead,
            prefillTileExperts: Self.prefillTileExperts(forSlots: expertCacheSlots),
            kvStorageMode: kvCacheMode.kvStorageMode)
    }
}

public struct AppLoadedRuntimeKey: Equatable, Sendable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var expertCacheSlots: Int
    public var expertCachePolicy: AppExpertCachePolicy
    public var rdadvisePolicy: AppRDAdvicePolicy
    public var modelVerification: AppModelVerification
    public var kvCacheMode: AppKVCacheMode
    /// The engine builds its prefill path from these, so a session loaded
    /// with one chunk size cannot serve a request built for another — the
    /// regenerate must be flagged here or the next run fails late instead.
    public var prefillEnabled: Bool
    public var prefillChunkTokens: Int
    public var forceLogitsHead: Bool

    public init(modelDirectory: URL,
                maxContextTokens: Int,
                options: AppRuntimeOptions,
                forceLogitsHead: Bool = false) {
        self.modelDirectory = modelDirectory.standardizedFileURL
        self.maxContextTokens = maxContextTokens
        self.expertCacheSlots = options.expertCacheSlots
        self.expertCachePolicy = options.expertCachePolicy
        self.rdadvisePolicy = options.rdadvisePolicy
        self.modelVerification = options.modelVerification
        self.kvCacheMode = options.kvCacheMode
        self.prefillEnabled = options.prefillEnabled
        self.prefillChunkTokens = options.prefillChunkTokens
        self.forceLogitsHead = forceLogitsHead
    }
}

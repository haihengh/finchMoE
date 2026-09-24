import Foundation
import FinchMoE

public enum AppExpertCachePolicy: String, CaseIterable, Sendable, Identifiable {
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

    /// int8 ships; turbo-4bit is still a planned format with no kernel behind
    /// it, so it stays disabled rather than pretending to load.
    public var isAvailable: Bool { self == .fp16 || self == .int8 }

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
                prefillChunkTokens: Int = 512,
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
                  prefillEnabled: settings.prefillEnabled,
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
        return "Cache \(expertCacheSlots) \(expertCachePolicy.label), \(prefill), FP16 KV, RDADVISE \(rdadvisePolicy.label.lowercased()), \(verification)"
    }

    public static func slotsLabel(for slots: Int) -> String {
        switch slots {
        case 8: "8, -0.8 GB"
        case 16: "16, Default"
        case 24: "24, +0.8 GB"
        case 32: "32, +1.61 GB"
        default: "\(slots)"
        }
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
        self.forceLogitsHead = forceLogitsHead
    }
}

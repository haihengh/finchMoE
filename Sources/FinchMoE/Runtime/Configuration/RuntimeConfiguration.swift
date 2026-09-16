public enum RuntimeHeadPath: String, Codable, Sendable {
    case fusedRows = "fused-rows"
    case logits
}

public enum RuntimePrefillPolicy: String, Codable, Sendable {
    case off
    case chunked
}

public enum RuntimePrefillAttentionPath: String, Codable, Sendable {
    case causalTiled = "causal-tiled"
    case fullTensorOps2DPreferred = "full-tensorops-2d-preferred"
    case fullTensorOps2DValidityV2 = "full-tensorops-2d-validity-v2"
}

public enum RuntimeExpertCachePolicy: String, Codable, Sendable {
    case lfu
    case lru
}

public struct RuntimeConfiguration: Sendable, Equatable {
    public static let allowedExpertCacheSlots = [8, 16, 24, 32]
    // Prefill cost is dominated by re-reading a layer's routed-expert pool once
    // per chunk (the LFU cache holds singles of experts, not the pool), so the
    // chunk size is the prefill I/O divisor. The ceiling was 128 — the upstream
    // Gemma install's tuning point — and was never swept on Qwen, whose 256
    // experts make the union saturate far earlier. Prefill scratch is ~154 KiB
    // per chunk token for Qwen 3.6, so 512 costs ~79 MiB and 1024 ~158 MiB
    // against a ~1.1 GiB resident budget.
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024]

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    public let rdadvisePolicy: RDAdvicePolicyMode
    public let prefillPolicy: RuntimePrefillPolicy
    public let prefillChunkTokens: Int
    public let prefillAttentionPath: RuntimePrefillAttentionPath
    public let headPath: RuntimeHeadPath
    /// When false, no QSA sparse-block selector is built and a Qwen 3.8
    /// full-attention layer takes the dense path for every position — the state
    /// the documented `FQ_QSA_OFF=1` control selects.
    ///
    /// It is settable here rather than only from the environment because
    /// **nothing else can reach that state**: `validateQwen38Layers` requires
    /// the indexer tensors on every full layer with no `indexerNumHeads > 0`
    /// gate, so an install built without them throws `tensorNotFound` before a
    /// runner exists. A test that must cover the selector-less path therefore
    /// has to inject it.
    public let qsaIndexerEnabled: Bool

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = 512,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                qsaIndexerEnabled: Bool = true) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.qsaIndexerEnabled = qsaIndexerEnabled
    }

    public static var production: RuntimeConfiguration {
        RuntimeConfiguration()
    }

    public var fp16RingEnabled: Bool { true }
    public var rdadviseEnabled: Bool { rdadvisePolicy != .off }
    public var prefillConfig: PrefillRuntimeConfig {
        switch prefillPolicy {
        case .off:
            return .off
        case .chunked:
            return .production(chunkTokens: prefillChunkTokens)
        }
    }
    public var modelExpertCachePolicy: ExpertCachePolicy {
        expertCachePolicy == .lru ? .lru : .lfu
    }
}

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
    /// How many routed-expert tiles the prefill may keep in flight, and how many
    /// experts a tile holds.
    ///
    /// These two together set the prefill's I/O overlap. A tile is the unit the
    /// streamed MoE loop reads and then computes on, so while one tile computes
    /// the drive can only be working if the *next* one was already issued:
    /// `maxPendingDepth` is how many may be outstanding. The default depth of 1
    /// allows a single tile of lookahead, which caps the drive's duty cycle at
    /// roughly (one tile's read time) / (one tile's compute time) however fast
    /// the drive is — and on the 125B that measures about 1.1 GB/s against a
    /// drive that serves 3.4 GB/s in decode.
    ///
    /// The pair must fit the slot budget: the scheduler requires
    /// `(depth + 1) * tileExperts <= expertCacheSlots`, so raising the depth
    /// means raising the slots with it (32 slots allow depth 3 at 8 experts, or
    /// depth 7 at 4). See `PrefillRoutedTileSchedulerConfig.fitsSlotBudget`.
    public let prefillTileDepth: Int
    public let prefillTileExperts: Int
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
                qsaIndexerEnabled: Bool = true,
                prefillTileDepth: Int = 1,
                prefillTileExperts: Int = 8) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition(prefillTileDepth >= 1, "prefill tile depth must be at least 1")
        precondition((1...16).contains(prefillTileExperts),
                     "prefill tiles hold 1 to 16 experts")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.qsaIndexerEnabled = qsaIndexerEnabled
        self.prefillTileDepth = prefillTileDepth
        self.prefillTileExperts = prefillTileExperts
    }

    /// Whether the tile pair fits the expert-cache slot budget the prefill
    /// streamer needs, i.e. `(depth + 1) * tileExperts <= slots`.
    public func prefillTilesFitSlots(_ slots: Int) -> Bool {
        (prefillTileDepth + 1) * prefillTileExperts <= slots
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

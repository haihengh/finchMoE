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
    /// A model's top-k is a second, per-model floor: a layer's expert cache
    /// must hold every routed expert at once (`PreadExpertStreamer` traps
    /// below that), so 3.8 (top-k 10) cannot use the 8-slot setting. The app
    /// filters its picker and clamps its effective options accordingly.
    public static let allowedExpertCacheSlots = [8, 10, 16, 24, 32]
    // Prefill cost is dominated by re-reading a layer's routed-expert pool once
    // per chunk (the LFU cache holds singles of experts, not the pool), so the
    // chunk size is the prefill I/O divisor. The ceiling was 128 — the upstream
    // Gemma install's tuning point — and was never swept on Qwen, whose 256
    // experts make the union saturate far earlier. Prefill scratch is ~154 KiB
    // per chunk token for Qwen 3.6, so 512 costs ~79 MiB and 1024 ~158 MiB
    // against a ~1.1 GiB resident budget.
    public static let allowedPrefillChunkTokens = [32, 64, 128, 256, 512, 1024]
    /// The shipping chunk, and the top of the list above. Measured on the 3.8
    /// 125B at 2,907 tokens: 512 prefills in 141.6 s having read 127.7 GB at
    /// 4.91 GB peak RSS; 1,024 in 111.3 s having read 72.1 GB at 4.84 GB. GPU
    /// time is the same at both (48.06 s vs 48.13 s), so the whole −21 % is the
    /// halved re-read, and the doubled scratch costs no extra peak RSS here.
    /// Seeded runs at both chunks are byte-identical. The old 512 default came
    /// from a sweep that read 1024 as "only a little more"; it is +27 % pp.
    /// See `STRATA_PP_KV_ANALYSIS.md` §4.1.
    public static let defaultPrefillChunkTokens = 1024

    public let expertCacheSlots: Int
    public let expertCachePolicy: RuntimeExpertCachePolicy
    /// Decode reads every routed expert on every step and does not retain
    /// expert contents between steps. Prefill keeps the cached path. This is
    /// the old engine's `cache_entries = 0` mode: it lets the OS page cache,
    /// rather than the per-layer LFU slots, decide what stays hot.
    public let directExpertCache: Bool
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
    /// How full-attention K/V bytes are stored. `fp16` is the shipping default;
    /// `int8` keeps an fp16 scale per 64-element block beside an int8 row. Only
    /// the Qwen 3.6 body quantizes on write and binds the int8 timelines on
    /// read, and the runner refuses other models rather than read a timeline
    /// nothing writes (`KVStorageMode` in `KVCacheManager.swift`).
    public let kvStorageMode: KVStorageMode

    public init(expertCacheSlots: Int = 16,
                expertCachePolicy: RuntimeExpertCachePolicy = .lfu,
                directExpertCache: Bool = false,
                rdadvisePolicy: RDAdvicePolicyMode = .off,
                prefillEnabled: Bool = true,
                prefillChunkTokens: Int = RuntimeConfiguration.defaultPrefillChunkTokens,
                prefillAttentionPath: RuntimePrefillAttentionPath = .fullTensorOps2DPreferred,
                forceLogitsHead: Bool = false,
                qsaIndexerEnabled: Bool = true,
                prefillTileDepth: Int = 1,
                prefillTileExperts: Int = 8,
                kvStorageMode: KVStorageMode = .fp16) {
        precondition(Self.allowedExpertCacheSlots.contains(expertCacheSlots),
                     "unsupported expert-cache slot count")
        precondition(Self.allowedPrefillChunkTokens.contains(prefillChunkTokens),
                     "unsupported prefill chunk size")
        precondition(prefillTileDepth >= 1, "prefill tile depth must be at least 1")
        precondition((1...16).contains(prefillTileExperts),
                     "prefill tiles hold 1 to 16 experts")
        self.expertCacheSlots = expertCacheSlots
        self.expertCachePolicy = expertCachePolicy
        self.directExpertCache = directExpertCache
        self.rdadvisePolicy = rdadvisePolicy
        self.prefillPolicy = prefillEnabled ? .chunked : .off
        self.prefillChunkTokens = prefillChunkTokens
        self.prefillAttentionPath = prefillAttentionPath
        self.headPath = forceLogitsHead ? .logits : .fusedRows
        self.qsaIndexerEnabled = qsaIndexerEnabled
        self.kvStorageMode = kvStorageMode
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

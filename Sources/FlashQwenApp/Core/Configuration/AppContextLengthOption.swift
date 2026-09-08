import FlashQwen

public enum AppContextLengthOption: Int, CaseIterable, Identifiable, Sendable {
    case fourK = 4_096
    case eightK = 8_192
    case sixteenK = 16_384
    case thirtyTwoK = 32_768
    case sixtyFourK = 65_536

    public var id: Int { rawValue }
    public var tokens: Int { rawValue }

    public var shortLabel: String {
        "\(tokens / 1_024)K"
    }

    public var fp16KVBytes: UInt64 {
        Self.fp16KVBytes(tokens: tokens, architecture: .gemma4_26B_A4B)
    }

    /// FP16 K+V cache bytes for `tokens` context positions under the given
    /// architecture. Qwen 3.6 has no sliding-window layers (its linear layers
    /// hold no KV at all) — its cache is the 10 full-attention layers only
    /// (2 KV heads × 256, fp16 K+V).
    public static func fp16KVBytes(tokens: Int, architecture: ArchConfig) -> UInt64 {
        let fullLayers = architecture.fullAttentionLayerMask.reduce(0) {
            $0 + ($1 == 0 ? 0 : 1)
        }
        // slidingWindow == 0 means the non-full layers are linear-attention
        // (Qwen) — no KV rows to count there.
        let slidingLayers = architecture.slidingWindow > 0
            ? architecture.numLayers - fullLayers
            : 0
        let fp16Bytes = 2
        let keyAndValue = 2
        let slidingRows = min(
            tokens,
            architecture.slidingWindow + PrefillRuntimeConfig.defaultChunked.chunkTokens)
        let slidingBytesPerRow = architecture.numKVHeads
            * architecture.headDim * keyAndValue * fp16Bytes
        let fullBytesPerRow = architecture.numFullKVHeads
            * architecture.fullHeadDim * keyAndValue * fp16Bytes
        return UInt64(slidingLayers * slidingRows * slidingBytesPerRow)
            + UInt64(fullLayers * tokens * fullBytesPerRow)
    }

    public var menuLabel: String {
        switch self {
        case .fourK: "4K, Default"
        case .eightK: "8K, +85 MB"
        case .sixteenK: "16K, +250 MB"
        case .thirtyTwoK: "32K, +590 MB"
        case .sixtyFourK: "64K, +1.26 GB"
        }
    }
}

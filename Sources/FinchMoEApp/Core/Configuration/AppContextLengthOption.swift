import Foundation
import FinchMoE

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

    /// FP16 K+V cache bytes this context length allocates under `architecture`.
    /// The architecture is required rather than defaulted — a default here is
    /// what previously reported Gemma's sizes against a Qwen install.
    public func fp16KVBytes(architecture: ArchConfig) -> UInt64 {
        Self.fp16KVBytes(tokens: tokens, architecture: architecture)
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

    /// Menu text for this option under `architecture`: the label plus the KV
    /// bytes it adds over the 4K default. Computed rather than hardcoded — the
    /// previous literals were sized for 3.6's 10 full-attention layers, so 3.8
    /// (12 layers) displayed 1.26 GB where it actually allocates 1.51 GB.
    public func menuLabel(architecture: ArchConfig) -> String {
        guard self != .fourK else { return "\(shortLabel), Default" }
        let baseline = Self.fourK.fp16KVBytes(architecture: architecture)
        let value = fp16KVBytes(architecture: architecture)
        let delta = value > baseline ? value - baseline : 0
        return "\(shortLabel), +\(Self.byteLabel(delta))"
    }

    static func byteLabel(_ bytes: UInt64) -> String {
        let megabytes = Double(bytes) / 1_000_000
        guard megabytes < 1_000 else {
            return String(format: "%.2f GB", megabytes / 1_000)
        }
        return "\(Int(megabytes.rounded())) MB"
    }
}

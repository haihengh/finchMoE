import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536])
    }

    @Test func optionsReportProductionFP16KVAllocation() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes / 1_048_576
        }
        #expect(mebibytes == [305, 385, 545, 865, 1_505])
        #expect(AppContextLengthOption.allCases.map(\.menuLabel) == [
            "4K, Default",
            "8K, +85 MB",
            "16K, +250 MB",
            "32K, +590 MB",
            "64K, +1.26 GB",
        ])
    }

    /// Qwen 3.6 has no sliding-window layers: 10 full-attention layers ×
    /// 2 KV heads × 256 head dim × (K+V) × fp16 per position.
    @Test func qwenFP16KVAllocationIsFullAttentionOnly() {
        let bytes = { tokens in
            AppContextLengthOption.fp16KVBytes(
                tokens: tokens, architecture: .qwen3_6_35B_A3B)
        }
        // Per position: 10 layers × 2 × 256 × 2 × 2 = 20480 B = 20 KiB.
        #expect(bytes(4_096) == 10 * 2 * 256 * 2 * 2 * 4_096)
        #expect(bytes(65_536) == 10 * 2 * 256 * 2 * 2 * 65_536)
    }
}

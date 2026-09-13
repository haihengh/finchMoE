import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536])
    }

    /// Gemma has a sliding window, so only its full-attention layers grow with
    /// context; the sliding layers are capped at window + chunk.
    @Test func gemmaFP16KVAllocationMatchesProduction() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes(architecture: .gemma4_26B_A4B) / 1_048_576
        }
        #expect(mebibytes == [305, 385, 545, 865, 1_505])
    }

    /// Qwen 3.6 has no sliding-window layers: 10 full-attention layers ×
    /// 2 KV heads × 256 head dim × (K+V) × fp16 per position.
    @Test func qwen36FP16KVAllocationIsFullAttentionOnly() {
        let bytes = { tokens in
            AppContextLengthOption.fp16KVBytes(
                tokens: tokens, architecture: .qwen3_6_35B_A3B)
        }
        // Per position: 10 layers × 2 × 256 × 2 × 2 = 20480 B = 20 KiB.
        #expect(bytes(4_096) == 10 * 2 * 256 * 2 * 2 * 4_096)
        #expect(bytes(65_536) == 10 * 2 * 256 * 2 * 2 * 65_536)
    }

    /// Qwen 3.8 has 12 full-attention layers rather than 10, so its cache is
    /// 1.2× 3.6's at every context length — 1.61 GB at 64K, not 1.34 GB.
    @Test func qwen38FP16KVAllocationCountsTwelveFullLayers() {
        let bytes = { tokens in
            AppContextLengthOption.fp16KVBytes(
                tokens: tokens, architecture: .qwen3_8_flashNext_125B)
        }
        #expect(bytes(65_536) == 12 * 2 * 256 * 2 * 2 * 65_536)
    }

    /// The menu figure is the delta over the 4K default, computed from the
    /// loaded architecture. 3.8's 64K delta is 1.51 GB — the old hardcoded
    /// literals showed 1.26 GB (3.6's figure) for every model.
    @Test func menuLabelsAreComputedPerArchitecture() {
        #expect(AppContextLengthOption.allCases.map {
            $0.menuLabel(architecture: .qwen3_6_35B_A3B)
        } == [
            "4K, Default",
            "8K, +84 MB",
            "16K, +252 MB",
            "32K, +587 MB",
            "64K, +1.26 GB",
        ])
        #expect(AppContextLengthOption.allCases.map {
            $0.menuLabel(architecture: .qwen3_8_flashNext_125B)
        } == [
            "4K, Default",
            "8K, +101 MB",
            "16K, +302 MB",
            "32K, +705 MB",
            "64K, +1.51 GB",
        ])
    }

    /// Regression guard for the bug this replaced: every installable
    /// descriptor must carry its own architecture, because the menu sizes its
    /// figures off it and a fallback silently reported Gemma's numbers.
    @Test func installableDescriptorsCarryTheirOwnArchitecture() {
        #expect(AppModelInstallDescriptor.qwen3_6.architecture == .qwen3_6_35B_A3B)
        #expect(AppModelInstallDescriptor.qwen3_8.architecture == .qwen3_8_flashNext_125B)
        #expect(AppModelInstallDescriptor.default.architecture == .gemma4_26B_A4B)
    }
}

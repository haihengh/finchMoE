import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppContextLengthOptionTests {
    @Test func optionsUseSupportedContextLengthsInAscendingOrder() {
        #expect(AppContextLengthOption.allCases.map(\.tokens)
            == [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144])
    }

    /// Gemma has a sliding window, so only its full-attention layers grow with
    /// context; the sliding layers are capped at window + chunk.
    ///
    /// These numbers moved +75 MiB each when the prefill chunk default went
    /// 128 → 512 (`docs/OPTIMIZATION_PLAN.md` 1.1): the sliding layers must hold
    /// one chunk's rows beyond the window, so 384 more tokens per sliding layer
    /// is 384 × 10 layers × 2 heads × 256 dim × 2 (K+V) × 2 B. Qwen is
    /// unaffected — its `slidingWindow` is 0, so it has no sliding layers.
    @Test func gemmaFP16KVAllocationMatchesProduction() {
        let mebibytes = AppContextLengthOption.allCases.map {
            $0.fp16KVBytes(architecture: .gemma4_26B_A4B) / 1_048_576
        }
        #expect(mebibytes == [380, 460, 620, 940, 1_580, 2_860, 5_420])
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
            "128K, +2.60 GB",
            "256K, +5.28 GB",
        ])
        #expect(AppContextLengthOption.allCases.map {
            $0.menuLabel(architecture: .qwen3_8_flashNext_125B)
        } == [
            "4K, Default",
            "8K, +101 MB",
            "16K, +302 MB",
            "32K, +705 MB",
            "64K, +1.51 GB",
            "128K, +3.12 GB",
            "256K, +6.34 GB",
        ])
    }

    /// Regression guard for the bug this replaced: every installable
    /// descriptor must carry its own architecture, because the menu sizes its
    /// figures off it and a fallback silently reported Gemma's numbers.
    ///
    /// The table is asserted against `installable` rather than listing
    /// descriptors by hand, and that coverage assertion is the part that earns
    /// its keep: this test used to name three descriptors explicitly, so the
    /// abliterated pair would have been added with no KV figures checked at all
    /// and nothing would have failed. Now a descriptor added to the scan
    /// without an expectation here breaks the coverage line.
    ///
    /// Note an abliterated entry deliberately shares its base's architecture —
    /// same layers, same heads — so the expectations below matching pairwise is
    /// correct, not a copy-paste slip.
    @Test func installableDescriptorsCarryTheirOwnArchitecture() {
        let expected: [(AppModelInstallDescriptor, ArchConfig)] = [
            (.qwen3_6, .qwen3_6_35B_A3B),
            (.qwen3_6_abliterated, .qwen3_6_35B_A3B),
            (.qwen3_8, .qwen3_8_flashNext_125B),
            (.qwen3_8_abliterated, .qwen3_8_flashNext_125B),
        ]
        // Compared order-insensitively: reordering `installable` is legitimate
        // (it only steers the fallback scan), but dropping or adding an entry
        // without an expectation here is not.
        #expect(expected.map(\.0.displayName).sorted()
                    == AppModelInstallDescriptor.installable.map(\.displayName).sorted(),
                "the architecture expectations must cover exactly the installable set")
        for (descriptor, architecture) in expected {
            #expect(descriptor.architecture == architecture,
                    "\(descriptor.displayName) carries the wrong architecture")
        }
        #expect(AppModelInstallDescriptor.default.architecture == .gemma4_26B_A4B)
    }
}

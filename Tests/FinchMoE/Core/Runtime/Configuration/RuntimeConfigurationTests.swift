import Testing
@testable import FinchMoE

@Suite struct RuntimeConfigurationTests {
    @Test func productionDefaultsAreStable() {
        let runtime = RuntimeConfiguration.production
        #expect(runtime.fp16RingEnabled)
        #expect(runtime.expertCacheSlots == 16)
        #expect(runtime.expertCachePolicy == .lfu)
        #expect(runtime.rdadvisePolicy == .off)
        #expect(!runtime.rdadviseEnabled)
        #expect(runtime.prefillPolicy == .chunked)
        #expect(runtime.prefillChunkTokens == 512)
        #expect(runtime.prefillAttentionPath == .fullTensorOps2DPreferred)
        #expect(runtime.headPath == .fusedRows)
    }

    @Test func retainedControlsReachTypedRuntime() {
        let runtime = RuntimeConfiguration(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            rdadvisePolicy: .adaptive,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            prefillAttentionPath: .causalTiled,
            forceLogitsHead: true)
        #expect(runtime.expertCacheSlots == 32)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.rdadviseEnabled)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.prefillAttentionPath == .causalTiled)
        #expect(runtime.headPath == .logits)
    }

    @Test(arguments: [32, 64, 128])
    func productionPrefillSupportsPublicChunkSizes(_ chunkTokens: Int) {
        let runtime = RuntimeConfiguration(prefillChunkTokens: chunkTokens)
        #expect(runtime.prefillConfig.mode == .chunked)
        #expect(runtime.prefillConfig.chunkTokens == chunkTokens)
    }

    /// `.fusedRows` is the default, but it is not available on Qwen 3.8: that
    /// head folds RMSNorm into one kernel, and 3.8's head input is the root HC
    /// mixer collapse, which the kernel cannot express.
    ///
    /// This is a regression test for a silent, total failure. The kernel-site
    /// check knew about 3.8 but the three sites that *read* the buffer it would
    /// have written did not, so a 3.8 model at temperature 0 read zero-filled
    /// memory and emitted token 0 forever. Anything that reads `greedyTokenBuf`
    /// has to agree with the site that decides whether it gets written, which is
    /// why the rule now lives on the flag itself and is asserted here — a pure
    /// function, so it is checked on every run rather than only when a 174 GB
    /// install happens to be present.
    @Test func fusedGreedyHeadIsUnavailableOnQwen38() {
        #expect(
            !RealForwardRunner.fusedGreedyHeadEnabled(
                headPath: .fusedRows, config: .qwen3_8_flashNext_125B))
        #expect(
            RealForwardRunner.fusedGreedyHeadEnabled(
                headPath: .fusedRows, config: .gemma4Toy()))
        #expect(
            !RealForwardRunner.fusedGreedyHeadEnabled(
                headPath: .logits, config: .qwen3_8_flashNext_125B))
        #expect(
            !RealForwardRunner.fusedGreedyHeadEnabled(
                headPath: .logits, config: .gemma4Toy()))
    }

    /// The prefill tile pipeline's budget rule, as a pure function.
    ///
    /// `(depth + 1) * tileExperts <= slots` is what the streamer enforces at
    /// runtime, and it is the reason the two knobs cannot be set
    /// independently: depth is how many tiles may be in flight, and each one
    /// needs its own slots. The measured arms of the sweep are checked here so
    /// a future default change has to confront the pairing.
    @Test func prefillTileBudgetPairsDepthWithSlots() {
        #expect(RuntimeConfiguration().prefillTileDepth == 1)
        #expect(RuntimeConfiguration().prefillTileExperts == 8)

        let baseline = RuntimeConfiguration(prefillTileDepth: 1, prefillTileExperts: 8)
        #expect(baseline.prefillTilesFitSlots(16))
        #expect(!baseline.prefillTilesFitSlots(8))

        let deep = RuntimeConfiguration(prefillTileDepth: 3, prefillTileExperts: 8)
        #expect(deep.prefillTilesFitSlots(32))
        #expect(!deep.prefillTilesFitSlots(24))
        #expect(!deep.prefillTilesFitSlots(16))

        // Same in-flight bytes, fewer experts per tile: the shape the sweep's
        // `{16,3,4}` arm used, which fits a 16-slot budget.
        let narrow = RuntimeConfiguration(prefillTileDepth: 3, prefillTileExperts: 4)
        #expect(narrow.prefillTilesFitSlots(16))

        let widest = RuntimeConfiguration(prefillTileDepth: 7, prefillTileExperts: 8)
        #expect(widest.prefillTilesFitSlots(64))
        #expect(!widest.prefillTilesFitSlots(32))
    }
}

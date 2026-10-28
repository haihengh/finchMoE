import Testing
@testable import FinchMoEAppCore

@Suite struct AppModelMemoryBudgetTests {
    /// The single cap covers iPhone *and* iPad — `underJetsam:` is the fact
    /// that matters, not which of the two it is, because the iPad's ceiling
    /// has not been measured from inside the app yet. Raising it for pad is a
    /// pricing decision to make from `AppMemorySampler.memoryLimitBytes`
    /// readings, not a branch to add here.
    @Test func deviceCapsTheLargestVisionModel() throws {
        let cap = try #require(AppModelMemoryBudget.cap(
            for: .qwen3_8_abliterated_2bit_vision, underJetsam: true))
        #expect(cap.expertCacheSlots == 10)
        #expect(cap.contextTokens == 4_096)
        #expect(cap.prefillChunkTokens == 128)
        // The cap is only useful if each value is one the engine accepts.
        #expect(AppRuntimeOptions.allowedSlotCounts.contains(cap.expertCacheSlots))
        #expect(AppRuntimeOptions.allowedPrefillChunkTokens.contains(
            cap.prefillChunkTokens))
        #expect(AppContextLengthOption.allCases.contains {
            $0.tokens == cap.contextTokens
        })
        let runtime = try AppRuntimeOptions(expertCacheSlots: cap.expertCacheSlots)
            .resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime.prefillTilesFitSlots(cap.expertCacheSlots))
    }

    @Test func macAndOtherModelsKeepTheirSettings() {
        #expect(AppModelMemoryBudget.cap(
            for: .qwen3_8_abliterated_2bit_vision, underJetsam: false) == nil)
        // 3.8 text has no tower to pay for; 3.6 vision is small enough. The
        // second one matters on iPad too: 3.6 is the default model there, so
        // it keeps the desktop chunk default and gets the prefill gains.
        #expect(AppModelMemoryBudget.cap(
            for: .qwen3_8_abliterated_2bit, underJetsam: true) == nil)
        #expect(AppModelMemoryBudget.cap(
            for: .qwen3_6_3bit_vision, underJetsam: true) == nil)
    }
}

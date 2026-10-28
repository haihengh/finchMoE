import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppRuntimeOptionsTests {
    @Test func defaultsMatchProduction() throws {
        let options = AppRuntimeOptions()
        #expect(options.expertCacheSlots == 16)
        #expect(options.expertCachePolicy == .lfu)
        #expect(options.prefillEnabled)
        #expect(options.prefillChunkTokens == 1024)
        #expect(options.rdadvisePolicy == .off)
        #expect(options.modelVerification == .automatic)

        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == .production)
        #expect(options.resultSummary ==
            "Cache 16 LFU, prefill 1024, FP16 KV, RDADVISE off, auto verification")
    }

    @Test func prefillChunkDefaultIsOfferedByThePicker() {
        // The settings picker marks `defaultPrefillChunkTokens` "(default)". A
        // default outside the offered set would render no marker at all, and
        // the picker had hardcoded 512 as its default long after 1,024 became
        // the real one — this is what keeps the two from drifting apart again.
        #expect(AppRuntimeOptions.defaultPrefillChunkTokens == 1024)
        #expect(AppRuntimeOptions.allowedPrefillChunkTokens
            .contains(AppRuntimeOptions.defaultPrefillChunkTokens))
        #expect(AppRuntimeOptions().prefillChunkTokens
            == AppRuntimeOptions.defaultPrefillChunkTokens)
    }

    @Test func persistedSettingsCarryTheCacheChoices() {
        let settings = MacAppSettings(prefillChunkTokens: 1024,
                                      expertCachePolicy: .lru)
        let options = AppRuntimeOptions(persisted: settings)
        #expect(options.prefillChunkTokens == 1024)
        #expect(options.expertCachePolicy == .lru)
    }

    @Test func summaryNamesTheSelectedCacheOptions() {
        // The diagnostics pane reads this line back; the KV label and the
        // chunk size it reports must be the ones actually in use.
        let int8 = AppRuntimeOptions(prefillChunkTokens: 1024, kvCacheMode: .int8)
        #expect(int8.resultSummary.contains("int8 KV"))
        #expect(int8.resultSummary.contains("prefill 1024"))
        #expect(AppRuntimeOptions().resultSummary.contains("FP16 KV"))
    }

    /// Int8 KV has kernels only on the Qwen 3.6 body. Offering it on a 3.8
    /// install used to reach the runner's precondition and crash the app on
    /// Load; the picker and the saved-setting clamp both consult this.
    @Test func int8KVIsQwen36Only() {
        #expect(AppKVCacheMode.fp16.isAvailable(for: .gemma4_26B_A4B))
        #expect(AppKVCacheMode.fp16.isAvailable(for: .qwen3_8_flashNext_125B))
        #expect(AppKVCacheMode.int8.isAvailable(for: .qwen3_6_35B_A3B))
        #expect(!AppKVCacheMode.int8.isAvailable(for: .qwen3_8_flashNext_125B))
        #expect(!AppKVCacheMode.int8.isAvailable(for: .gemma4_26B_A4B))
        for architecture in [ArchConfig.gemma4_26B_A4B,
                             .qwen3_6_35B_A3B,
                             .qwen3_8_flashNext_125B] {
            #expect(!AppKVCacheMode.turbo4bit.isAvailable(for: architecture))
        }
    }

    @Test func verificationModesAreDistinctlyLabelled() {
        // The picker renders `label` and the diagnostics pane renders
        // `resultSummary`. Three modes that summarised alike would leave that
        // pane unable to answer the one question it exists for -- whether this
        // run took the receipt -- so distinctness is the property under test,
        // not the exact wording.
        let summaryOptions = AppModelVerification.allCases.map {
            AppRuntimeOptions(modelVerification: $0).resultSummary
        }
        #expect(Set(summaryOptions).count == AppModelVerification.allCases.count)
        #expect(Set(AppModelVerification.allCases.map(\.label)).count
            == AppModelVerification.allCases.count)
        #expect(AppModelVerification.allCases.allSatisfy { !$0.detail.isEmpty })
        // First, because the segmented picker renders `allCases` in order.
        #expect(AppModelVerification.allCases.first == .automatic)

        #expect(AppModelVerification.automatic.runtimeValue == .automatic)
        #expect(AppModelVerification.fullSha256.runtimeValue == .fullSha256)
        #expect(AppModelVerification.trustedInstall.runtimeValue
            == .sizeCheckTrustedReceipt)
    }

    @Test func everyPublicChoiceMapsToRuntime() throws {
        for slots in AppRuntimeOptions.allowedSlotCounts {
            let runtime = try AppRuntimeOptions(expertCacheSlots: slots)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.expertCacheSlots == slots)
            // Every offered slot count must be usable with prefill: the
            // engine rejects a tile plan that needs more slots than exist,
            // so the low-memory 8-slot choice has to shrink the tile.
            #expect(runtime.prefillTilesFitSlots(slots))
        }
        #expect(AppRuntimeOptions.prefillTileExperts(forSlots: 8) == 4)
        #expect(AppRuntimeOptions.prefillTileExperts(forSlots: 10) == 5)
        #expect(AppRuntimeOptions.prefillTileExperts(forSlots: 16) == 8)
        for chunk in AppRuntimeOptions.allowedPrefillChunkTokens {
            let runtime = try AppRuntimeOptions(prefillChunkTokens: chunk)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.prefillConfig.chunkTokens == chunk)
        }
        for policy in AppRDAdvicePolicy.allCases {
            let runtime = try AppRuntimeOptions(rdadvisePolicy: policy)
                .resolvedRuntimeConfiguration(forceLogitsHead: false)
            #expect(runtime.rdadvisePolicy == policy.runtimeValue)
        }
    }

    @Test func runtimeAndTrustChoicesAreExplicit() throws {
        let options = AppRuntimeOptions(
            expertCacheSlots: 32,
            expertCachePolicy: .lru,
            prefillEnabled: false,
            prefillChunkTokens: 64,
            rdadvisePolicy: .adaptive,
            modelVerification: .trustedInstall)
        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: true)
        #expect(runtime.modelExpertCachePolicy == .lru)
        #expect(runtime.prefillConfig == .off)
        #expect(runtime.rdadvisePolicy == .adaptive)
        #expect(runtime.headPath == .logits)
        #expect(options.modelVerification.runtimeValue == .sizeCheckTrustedReceipt)
    }

    @Test func validationRejectsValuesOutsideClosedSets() {
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(expertCacheSlots: 12).validate()
        }
        #expect(throws: AppInferenceError.self) {
            try AppRuntimeOptions(prefillChunkTokens: 96).validate()
        }
    }

    @Test func loadedRuntimeKeyTracksOnlyLoadTimeChoices() {
        let directory = URL(fileURLWithPath: "/tmp/model.finch")
        let base = AppRuntimeOptions()
        let baseline = AppLoadedRuntimeKey(
            modelDirectory: directory, maxContextTokens: 4096, options: base)

        var variants: [AppRuntimeOptions] = []
        var value = base
        value.expertCacheSlots = 24; variants.append(value)
        value = base; value.expertCachePolicy = .lru; variants.append(value)
        value = base; value.rdadvisePolicy = .bounded; variants.append(value)
        value = base; value.modelVerification = .trustedInstall; variants.append(value)
        value = base; value.prefillEnabled = false; variants.append(value)
        value = base; value.prefillChunkTokens = 512; variants.append(value)

        for variant in variants {
            #expect(AppLoadedRuntimeKey(
                modelDirectory: directory,
                maxContextTokens: 4096,
                options: variant) != baseline)
        }
        #expect(AppLoadedRuntimeKey(
            modelDirectory: directory,
            maxContextTokens: 4096,
            options: base,
            forceLogitsHead: true) != baseline)

        // Prefill belongs in this set: the engine sizes its chunked-prefill
        // scratch from `prefillChunkTokens` when the runner is built and
        // rejects larger chunks afterwards (`RealForwardRunner` guards
        // against `scratch.layout.chunkTokens`), so a changed chunk size or a
        // flipped prefill toggle cannot ride a loaded session. Tracking it
        // here is what makes the UI say "Reload required" instead of letting
        // the next generation fail late with `reloadRequired`.
    }
}

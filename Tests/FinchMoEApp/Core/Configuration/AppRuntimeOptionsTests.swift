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
        #expect(options.prefillChunkTokens == 512)
        #expect(options.rdadvisePolicy == .off)
        #expect(options.modelVerification == .automatic)

        let runtime = try options.resolvedRuntimeConfiguration(forceLogitsHead: false)
        #expect(runtime == .production)
        #expect(options.resultSummary ==
            "Cache 16 LFU, prefill 512, FP16 KV, RDADVISE off, auto verification")
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
        }
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
        value = base; value.prefillChunkTokens = 1024; variants.append(value)

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

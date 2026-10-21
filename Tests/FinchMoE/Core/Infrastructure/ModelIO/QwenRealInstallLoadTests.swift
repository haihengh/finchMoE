import Testing
import Foundation
import Metal
@testable import FinchMoE

/// Loads the REAL repacked Qwen 3.6 35B-A3B install
/// (`models/Qwen3.6-35B-A3B-4bit.finch`, gitignored) with the real preset —
/// the same gate the toy repack test exercises, at production shapes:
/// validateArch against `ArchConfig.qwen3_6_35B_A3B`, then
/// validateRuntimeSchema over all 613 resident entries, plus accessor
/// spot-checks. Skipped when the install is not present.
@Suite struct QwenRealInstallLoadTests {

    private static let installPath =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-4bit.finch"

    private static var installExists: Bool {
        FileManager.default.fileExists(
            atPath: installPath + "/manifest.json")
    }

    private static let visionInstallPath =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-3bit-vision.finch"

    private static var visionInstallExists: Bool {
        FileManager.default.fileExists(atPath: visionInstallPath + "/manifest.json")
    }

    /// The published vision-bearing install is still a text install: the four
    /// extra `vision/` files are ordinary manifest entries the loader never
    /// opens, and every text entry resolves exactly as it does without them.
    @Test(.enabled(if: visionInstallExists))
    func visionInstallLoadsTextOnly() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.visionInstallPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        #expect(model.routedExpertWeightBits == 3)
        #expect(model.embedding.shape == (248_320, 2_048, 0, 0))
        #expect(model.packedExpertsLayout.expertsPerLayer == 256)
        #expect(try model.router(layer: 0).shape == (256, 2_048, 0, 0))
        // The app's memory readout uses the device allocation ledger because
        // process footprint accounting can leave the clean mmap pages out.
        // The resident buffer alone must already be counted here.
        #expect(ctx.device.currentAllocatedSize >= 1_894_721_536)
    }

    @Test(.enabled(if: installExists))
    func realInstallLoadsWithQwenPreset() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)

        // Global entries.
        #expect(model.embedding.shape == (248_320, 2_048, 0, 0))
        #expect(model.lmHead.shape == (248_320, 2_048, 0, 0))
        let finalNorm = try #require(model.finalNorm)
        #expect(finalNorm.shape == (2_048, 0, 0, 0))

        // GDN layer 0 (mask[0] == 0).
        #expect(try model.gdnInProjQKV(layer: 0).shape == (8_192, 2_048, 0, 0))
        #expect(try model.gdnInProjZ(layer: 0).shape == (4_096, 2_048, 0, 0))
        #expect(try model.gdnInProjA(layer: 0).shape == (32, 2_048, 0, 0))
        #expect(try model.gdnInProjB(layer: 0).shape == (32, 2_048, 0, 0))
        #expect(try model.gdnOutProj(layer: 0).shape == (2_048, 4_096, 0, 0))
        #expect(try model.gdnNormWeight(layer: 0).shape == (128, 0, 0, 0))
        #expect(try model.gdnALog(layer: 0).shape == (32, 0, 0, 0))
        #expect(try model.gdnDtBias(layer: 0).shape == (32, 0, 0, 0))
        #expect(try model.gdnConv1D(layer: 0).shape == (32_768, 0, 0, 0))

        // Full-attention layer 3 (mask[3] == 1): doubled q_proj.
        #expect(try model.qProj(layer: 3).shape == (8_192, 2_048, 0, 0))
        #expect(try model.kProj(layer: 3).shape == (512, 2_048, 0, 0))
        #expect(try model.vProj(layer: 3).shape == (512, 2_048, 0, 0))
        #expect(try model.oProj(layer: 3).shape == (2_048, 4_096, 0, 0))
        #expect(try model.qNorm(layer: 3).shape == (256, 0, 0, 0))

        // MoE entries (both layer types).
        #expect(try model.router(layer: 0).shape == (256, 2_048, 0, 0))
        #expect(try model.sharedExpertGate(layer: 0).shape == (512, 2_048, 0, 0))
        #expect(try model.sharedExpertGateProj(layer: 0).shape == (1, 2_048, 0, 0))
        #expect(model.packedExpertsLayout.expertsPerLayer == 256)
    }

    /// A 3.6 install must be able to build the production runner.
    ///
    /// `RealForwardRunner.init` walks the 3.8-only surface — hyper-connection
    /// scratch, the QSA indexer state, the PLE hash constants — and every one
    /// of those accessors raises `tensorNotFound` on a manifest that has no
    /// such tensor. An ungated call there is therefore not a subtle numeric
    /// drift: it makes the entire 3.6 engine unconstructible, and because the
    /// only tests that build a non-3.8 runner are the `QwenLayer0DebugTests`
    /// probes, the symptom is a pile of unrelated-looking diagnostics rather
    /// than a load failure. Exactly that happened with a stray unconditional
    /// `model.pleHashConstants()`; this is the gate that names it.
    @Test(.enabled(if: installExists))
    func realInstallBuildsTheProductionRunner() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_6_35B_A3B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        // Wired, not merely allocated: a fresh runner sits at position 0 and
        // carries every per-layer state its mask calls for.
        #expect(runner.continuationPosition == 0)
    }
}

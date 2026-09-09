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
        #expect(model.finalNorm.shape == (2_048, 0, 0, 0))

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
}

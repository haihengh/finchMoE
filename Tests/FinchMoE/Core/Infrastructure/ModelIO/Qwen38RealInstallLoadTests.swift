import Testing
import Foundation
import Metal
@testable import FinchMoE

/// Loads the REAL repacked Qwen 3.8 Flash-Next 125B install
/// (`models/Qwen3.8-Flash-Next-125B.finch`, gitignored) with the real preset,
/// at production shapes. Skipped when the install is not present.
///
/// The 3.6 counterpart is `QwenRealInstallLoadTests`; this is the 3.8 one. It
/// exists because a real install is the only thing that can check the *frozen
/// geometry* — every other test builds a synthetic fixture from the same
/// `ArchConfig` the engine reads, so a config value that is wrong, or a
/// weight file laid out for a different value, agrees with itself and passes.
/// That is not hypothetical: the 3.8 soup came from a GDN key-head *ratio* the
/// fixtures all shared with a hardcoded kernel constant
/// (`qwen38-m4-gdn-gqa-divisor-bug`). Nothing here recomputes a shape from the
/// config to compare against the same config — the expected numbers are
/// literals derived from `Qwen3.8-Flash-Next-bf16/config.json`, so a config
/// edit that changes the engine's behaviour fails here.
@Suite struct Qwen38RealInstallLoadTests {

    private static let installPath =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.8-Flash-Next-125B.finch"

    private static var installExists: Bool {
        FileManager.default.fileExists(atPath: installPath + "/manifest.json")
    }

    /// Frozen literals: hidden 2560, 24 Q heads, 2 KV heads, full head_dim
    /// 256, GDN 16 key / 48 value heads of 128, conv kernel 4, 512 experts,
    /// shared-expert width 640, vocab 248320.
    @Test(.enabled(if: installExists))
    func realInstall38LoadsWithQwenPreset() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_8_flashNext_125B)

        let cfg = model.config
        #expect(cfg.numLayers == 48)
        #expect(cfg.hiddenSize == 2560)
        #expect(cfg.numExperts == 512)
        #expect(cfg.topKExperts == 10)

        // Global entries. There is no `model.norm` on this family — the root
        // hyper-connection mixer is the terminal norm — so a non-nil
        // `finalNorm` would mean the install is carrying a 3.6-shaped manifest.
        #expect(model.embedding.shape == (248_320, 2_560, 0, 0))
        #expect(model.lmHead.shape == (248_320, 2_560, 0, 0))
        #expect(cfg.tieWordEmbeddings == false)
        #expect(model.finalNorm == nil)

        // GDN layer 0. qkv is q(16·128) + k(16·128) + v(48·128) = 10240 wide,
        // and `z` is the value width alone. A checkpoint laid out for the 3.6
        // ratio (32 value heads) would be 8192 / 4096 here.
        #expect(try model.gdnInProjQKV(layer: 0).shape == (10_240, 2_560, 0, 0))
        #expect(try model.gdnInProjZ(layer: 0).shape == (6_144, 2_560, 0, 0))
        #expect(try model.gdnInProjA(layer: 0).shape == (48, 2_560, 0, 0))
        #expect(try model.gdnInProjB(layer: 0).shape == (48, 2_560, 0, 0))
        #expect(try model.gdnOutProj(layer: 0).shape == (2_560, 6_144, 0, 0))
        #expect(try model.gdnNormWeight(layer: 0).shape == (128, 0, 0, 0))
        #expect(try model.gdnALog(layer: 0).shape == (48, 0, 0, 0))
        #expect(try model.gdnDtBias(layer: 0).shape == (48, 0, 0, 0))
        #expect(try model.gdnConv1D(layer: 0).shape == (40_960, 0, 0, 0))

        // Full-attention layer 3 (0-based; every 4th layer). The `attnOutputGate`
        // folds a second half into q_proj, so q is 2·24·256.
        #expect(try model.qProj(layer: 3).shape == (12_288, 2_560, 0, 0))
        #expect(try model.kProj(layer: 3).shape == (512, 2_560, 0, 0))
        #expect(try model.vProj(layer: 3).shape == (512, 2_560, 0, 0))
        #expect(try model.oProj(layer: 3).shape == (2_560, 6_144, 0, 0))
        #expect(try model.qNorm(layer: 3).shape == (256, 0, 0, 0))
        #expect(try model.kNorm(layer: 3).shape == (256, 0, 0, 0))

        // The QSA indexer rides only the full layers; a GDN layer has none.
        #expect(try model.indexerQKProj(layer: 3).shape.0 > 0)
        #expect(try model.indexerQLayernorm(layer: 3).shape == (128, 0, 0, 0))
        #expect(try model.indexerKLayernorm(layer: 3).shape == (128, 0, 0, 0))
        #expect(throws: ModelError.self) { _ = try model.indexerQKProj(layer: 0) }

        // MoE, on both layer types.
        #expect(try model.router(layer: 0).shape == (512, 2_560, 0, 0))
        #expect(try model.sharedExpertGate(layer: 0).shape == (640, 2_560, 0, 0))
        #expect(try model.sharedExpertGateProj(layer: 0).shape == (1, 2_560, 0, 0))
        #expect(model.packedExpertsLayout.expertsPerLayer == 512)

        // The hyper-connection backbone: one mixer per branch per layer, plus
        // the terminal one. These are the tensors that carry the `(1 + w)`
        // gamma fold, and they are the reason this family has no `input_norm` /
        // `post_attention_norm` to check instead.
        #expect(try model.attnHyperConnection(layer: 0).hcNorm.shape == (10_240, 0, 0, 0))
        #expect(try model.mlpHyperConnection(layer: 0).hcNorm.shape == (10_240, 0, 0, 0))
        // `.shape.0` rather than the whole tuple: the root gate is the one
        // mixer whose width is not pinned by a sibling, so a mismatch here
        // should print the width it actually found.
        #expect(try model.hyperConnectionMixer().hcNorm.shape.0 == 10_240)
        #expect(throws: ModelError.self) { _ = try model.inputNorm(layer: 0) }
        #expect(throws: ModelError.self) { _ = try model.postAttnNorm(layer: 0) }

        // PLE n-gram block: layer index 1 only, and its hash metadata is
        // readable — one multiplier per gram position (ngram size 3), and one
        // (offset, vocabSize) pair per head, where the head count is
        // (ngramSize − 1) · headsPerNgram = 16. `pleHashConstants` validates
        // the I64 vectors' length and alignment on the way out, so this is a
        // real read of the resident bytes rather than a shape echo.
        #expect(model.pleLayerIndex == 1)
        #expect(try model.pleConv1D().shape.0 > 0)
        #expect(try model.pleKeyProj().shape.0 > 0)
        #expect(try model.pleValueProj().shape.0 > 0)
        let ple = try model.pleHashConstants()
        #expect(ple.multipliers.count == 3)
        #expect(ple.headOffsets.count == 16)
        #expect(ple.headVocabSizes.count == 16)
    }

    /// The config-driven half of the GDN mapping — the actual root cause of the
    /// 3.8 soup. The kernel's divisor is `numValueHeads / numKeyHeads`, so this
    /// pins the two numbers it divides, and the division itself: 3 here, 2 on
    /// 3.6. A regression to a hardcoded 2 is invisible without this.
    @Test(.enabled(if: installExists))
    func realInstall38CarriesTheThreeToOneGQARatio() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_8_flashNext_125B)
        let cfg = model.config
        #expect(cfg.linearNumKeyHeads == 16)
        #expect(cfg.linearNumValueHeads == 48)
        #expect(GDN.valueHeadsPerKeyHead(numValueHeads: cfg.linearNumValueHeads,
                                        numKeyHeads: cfg.linearNumKeyHeads) == 3)
    }

    /// A 3.8 install must build the production runner — the mirror of the 3.6
    /// test, and the gate on this family's own conditional surface: the PLE
    /// streamer, the QSA indexer state, the hyper-connection scratch.
    @Test(.enabled(if: installExists))
    func realInstall38BuildsTheProductionRunner() throws {
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: Self.installPath),
            device: ctx.device,
            expecting: .qwen3_8_flashNext_125B)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 256,
                                           runtimeConfiguration: .production)
        #expect(runner.continuationPosition == 0)
    }
}

import Testing
import Foundation
@testable import FinchMoE

/// M0 gate tests for the Qwen3.8-Flash-Next arch model: the built-in preset
/// must match the BF16 snapshot's config.json field-by-field, family dispatch
/// must run through the constants/helpers, and no new bare family-literal
/// comparisons may appear outside `ModelTypes.swift`.
@Suite struct Qwen38ArchModelTests {

    @Test func qwen38PresetMatchesSnapshotConfigFieldByField() {
        // Ground truth: models/Qwen3.8-Flash-Next-bf16/config.json
        // (text_config of the qwen4_exp root; model_type qwen4_exp_text).
        let a = ArchConfig.qwen3_8_flashNext_125B
        #expect(a.hiddenSize == 2560)
        #expect(a.intermediateSize == 640)        // shared_expert_intermediate_size
        #expect(a.moeIntermediateSize == 640)     // moe_intermediate_size
        #expect(a.numHeads == 24)                 // num_attention_heads
        #expect(a.numKVHeads == 2)                // num_key_value_heads
        #expect(a.numFullKVHeads == 2)            // (no num_global_key_value_heads)
        #expect(a.headDim == 128)                 // linear_key_head_dim (nominal)
        #expect(a.fullHeadDim == 256)             // head_dim
        #expect(a.vocabSize == 248320)
        #expect(a.slidingWindow == 0)
        #expect(a.finalLogitSoftcap == 0.0)       // no final_logit_softcapping key
        // rope_parameters is flat for qwen4_exp_text: theta 1e7 for the whole
        // model (no full/sliding split), partial_rotary_factor 0.25.
        #expect(a.ropeTheta == 10_000_000.0)
        #expect(a.fullRopeTheta == 10_000_000.0)
        #expect(a.partialRotaryFactor == 0.25)
        #expect(a.numLayers == 48)
        #expect(a.numExperts == 512)
        #expect(a.topKExperts == 10)              // num_experts_per_tok
        #expect(a.tieWordEmbeddings == false)
        #expect(a.attentionKEqV == false)
        #expect(a.hiddenActivation == "silu")     // hidden_act
        // Gated DeltaNet (shared with 3.6; V=48 vs 32).
        #expect(a.linearNumKeyHeads == 16)
        #expect(a.linearNumValueHeads == 48)
        #expect(a.linearKeyHeadDim == 128)
        #expect(a.linearValueHeadDim == 128)
        #expect(a.linearConvKernelDim == 4)
        #expect(a.attnOutputGate == true)         // q_proj carries a gate half
        // Hyper-connections (hc_count / hc_lowrank).
        #expect(a.hyperConnectionCount == 4)
        #expect(a.hyperConnectionLowrank == 320)
        #expect(a.hyperConnectionDim == 4 * 2560) // wide stream 10240
        // QSA indexer.
        #expect(a.indexerNumHeads == 4)
        #expect(a.indexerKVHeads == 1)
        #expect(a.indexerHeadDim == 128)
        #expect(a.indexerBudget == 2048)
        #expect(a.indexerCompressRatio == 4)
        // PLE n-gram: ngram_size 3 / heads_per_ngram 8 / split_ngram_parts 128.
        // Row geometry (160 cols, 2_500_012 rows per part) is frozen from the
        // snapshot tensor shape, not the config.
        #expect(a.ngramSize == 3)
        #expect(a.headsPerNgram == 8)
        #expect(a.ngramRowDim == 160)
        #expect(a.ngramPartCount == 128)
        #expect(a.ngramPartRows == 2_500_012)
        #expect(a.ngramTotalRows == 128 * 2_500_012)
        // ple_layer_ids [2] in the config is 1-based; the engine indexes from
        // 0 (snapshot census: PLE tensors live under layers.1).
        #expect(a.pleLayerIndexes == [1])
        #expect(a.pleConvKernelSize == 4)
        #expect(a.pleHeadCount == (3 - 1) * 8)    // 16 heads
    }

    @Test func qwen38MaskFlagsEveryFourthLayerFromThree() {
        let a = ArchConfig.qwen3_8_flashNext_125B
        #expect(a.fullAttentionLayerMask.count == 48)
        let fulls = a.fullAttentionLayerMask.indices
            .filter { a.fullAttentionLayerMask[$0] != 0 }
        // layer_types [L,L,L,F] x12 -> full at 0-based 3,7,...,47.
        #expect(fulls == Array(stride(from: 3, through: 47, by: 4)))
        #expect(fulls.count == 12)
    }

    @Test func qwen38FamilyDispatchRunsThroughConstantsAndPresets() {
        // Distinct family strings; no aliasing between 3.6 / 3.8 / gemma.
        #expect(ArchConfig.gemma4Family != ArchConfig.qwen3_6Family)
        #expect(ArchConfig.qwen3_6Family != ArchConfig.qwen3_8Family)
        #expect(ArchConfig.qwen3_8Family != ArchConfig.gemma4Family)

        #expect(ArchConfig.preset(forModelFamily: ArchConfig.qwen3_6Family)
                == .qwen3_6_35B_A3B)
        #expect(ArchConfig.preset(forModelFamily: ArchConfig.qwen3_8Family)
                == .qwen3_8_flashNext_125B)
        #expect(ArchConfig.preset(forModelFamily: ArchConfig.gemma4Family)
                == .gemma4_26B_A4B)
        #expect(ArchConfig.preset(forModelFamily: nil) == .gemma4_26B_A4B)

        let g38 = ArchConfig.qwen3_8_flashNext_125B
        #expect(g38.isQwen3_8)
        #expect(g38.isQwenHybrid)
        #expect(!g38.isQwen3_6)

        let g36 = ArchConfig.qwen3_6_35B_A3B
        #expect(g36.isQwen3_6)
        #expect(g36.isQwenHybrid)
        #expect(!g36.isQwen3_8)
        // 3.6 has no hyper-connections / n-gram: geometry collapses to zero.
        #expect(g36.hyperConnectionDim == 0)
        #expect(g36.ngramTotalRows == 0)
        #expect(g36.pleHeadCount == 0)

        let gemma = ArchConfig.gemma4_26B_A4B
        #expect(!gemma.isQwenHybrid)
        #expect(gemma.hyperConnectionDim == 0)
        #expect(gemma.pleHeadCount == 0)
    }

    /// Regression gate: family decisions must read through
    /// `ArchConfig.qwen*Family` / `isQwen*` helpers. A bare comparison
    /// against the literal family string outside ModelTypes.swift is a
    /// dispatch-site smell that the dual-family (3.6 + 3.8) work forbids —
    /// new sites would silently serve only one family.
    @Test func noBareFamilyLiteralComparisonsOutsideModelTypes() throws {
        let root = try #require(try repoRoot())
        let forbidden = try literalComparisonLines(root: root)
        if !forbidden.isEmpty {
            let message = "bare family-literal comparisons outside ModelTypes.swift:\n"
                + forbidden.joined(separator: "\n")
            Issue.record(Comment(rawValue: message))
        }
    }

    // MARK: - Filesystem scan helpers

    private func repoRoot() throws -> URL? {
        // Tests/FinchMoE/Core/Infrastructure/ModelIO/ -> walk up to Package.swift.
        var dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        for _ in 0..<8 {
            if FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("Package.swift").path) {
                return dir
            }
            dir = dir.deletingLastPathComponent()
        }
        return nil
    }

    private func literalComparisonLines(root: URL) throws -> [String] {
        let sources = ["Sources/FinchMoE", "Sources/FinchMoERepack",
                       "Sources/FinchMoEFormat"]
        var violations: [String] = []
        for sub in sources {
            let base = root.appendingPathComponent(sub)
            guard let enumerator = FileManager.default.enumerator(
                at: base, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]) else { continue }
            for case let url as URL in enumerator where url.pathExtension == "swift" {
                // ModelTypes.swift is the one allowed home of the constants
                // (they must equal the literal family strings).
                if url.lastPathComponent == "ModelTypes.swift" { continue }
                let lines = try String(contentsOf: url, encoding: .utf8)
                    .components(separatedBy: .newlines)
                for (i, line) in lines.enumerated() where !line.trimmingCharacters(in: .whitespaces).hasPrefix("//") {
                    let trimmed = line.replacingOccurrences(of: "//.*", with: "",
                                                            options: .regularExpression)
                    if trimmed.contains("==") || trimmed.contains("!=") {
                        for fam in ["qwen3_6", "qwen3_8", "gemma4"] {
                            if trimmed.contains("\"\(fam)\"") {
                                violations.append("\(url.path):\(i + 1): \(line)")
                                break
                            }
                        }
                    }
                }
            }
        }
        return violations
    }
}

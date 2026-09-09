import Testing
import Foundation
@testable import FinchMoERepackCore

/// M0 repack-side gates: `ArchInfo.family(from:)` must map the Qwen3.8
/// model types (root `qwen4_exp`, text `qwen4_exp_text`) to the qwen3_8
/// family — ahead of the GDN-field heuristic that would misclassify them as
/// qwen3_6 — and `ArchInfo.load` must parse the hyper-connection / indexer /
/// n-gram / PLE fields with the config's 1-based ple_layer_ids converted to
/// 0-based engine layer indexes.
@Suite struct ArchInfoTests {

    /// Minimal text_config mirroring the real Qwen3.8-Flash-Next config.json
    /// key subset that ArchInfo reads (values verbatim from the BF16
    /// snapshot's text_config).
    private static func qwen38ConfigJSON(
        modelType: String = "qwen4_exp_text",
        linearHeads: Int = 16,
        pleLayerIDs: [Int] = [2]
    ) throws -> Data {
        var layerTypes: [String] = []
        for i in 0..<48 { layerTypes.append(i % 4 == 3 ? "full_attention" : "linear_attention") }
        let config: [String: Any] = [
            "model_type": modelType,
            "hidden_size": 2560,
            "shared_expert_intermediate_size": 640,
            "moe_intermediate_size": 640,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": linearHeads,
            "linear_num_value_heads": 48,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 248320,
            "hidden_act": "silu",
            "num_hidden_layers": 48,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "tie_word_embeddings": false,
            "layer_types": layerTypes,
            "rope_parameters": [
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
            ],
            "output_gate_type": "sigmoid",
            "hc_count": 4,
            "hc_lowrank": 320,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "ngram_size": 3,
            "heads_per_ngram": 8,
            "split_ngram_parts": 128,
            "ple_conv_kernel_size": 4,
            "ple_layer_ids": pleLayerIDs,
        ]
        return try JSONSerialization.data(withJSONObject: config)
    }

    private static func load(_ data: Data) throws -> ArchInfo {
        let path = NSTemporaryDirectory() + "archinfo-\(UUID().uuidString).json"
        try data.write(to: URL(fileURLWithPath: path))
        defer { try? FileManager.default.removeItem(atPath: path) }
        return try ArchInfo.load(configPath: path)
    }

    @Test func qwen4ExpTextMapsToQwen38Family() throws {
        let config = try Self.qwen38ConfigJSON()
        let a = try Self.load(config)
        #expect(a.modelFamily == "qwen3_8")
    }

    @Test func qwen4ExpRootModelTypeAlsoMapsToQwen38() throws {
        // The hybrid checkpoint's root config says model_type qwen4_exp and
        // wraps the text config under text_config.
        let text = try JSONSerialization.jsonObject(
            with: Self.qwen38ConfigJSON(modelType: "qwen4_exp")) as! [String: Any]
        let root: [String: Any] = ["model_type": "qwen4_exp", "text_config": text]
        let a = try Self.load(try JSONSerialization.data(withJSONObject: root))
        #expect(a.modelFamily == "qwen3_8")
    }

    @Test func qwen38FieldsParseFromConfig() throws {
        let a = try Self.load(try Self.qwen38ConfigJSON())
        // Hyper-connections / QSA indexer / PLE n-gram.
        #expect(a.hyperConnectionCount == 4)
        #expect(a.hyperConnectionLowrank == 320)
        #expect(a.indexerNumHeads == 4)
        #expect(a.indexerKVHeads == 1)
        #expect(a.indexerHeadDim == 128)
        #expect(a.indexerBudget == 2048)
        #expect(a.indexerCompressRatio == 4)
        #expect(a.ngramSize == 3)
        #expect(a.headsPerNgram == 8)
        #expect(a.ngramPartCount == 128)
        #expect(a.pleConvKernelSize == 4)
        #expect(a.attnOutputGate == true)  // output_gate_type "sigmoid"
        // Row geometry frozen from the snapshot tensor shape.
        #expect(a.ngramRowDim == 160)
        #expect(a.ngramPartRows == 2_500_012)
        // ple_layer_ids [2] is 1-based in the HF config -> engine layer 1.
        #expect(a.pleLayerIndexes == [1])
        // The doubled-q + GDN reads that 3.8 shares with 3.6.
        #expect(a.linearNumKeyHeads == 16)
        #expect(a.linearNumValueHeads == 48)
        #expect(a.linearConvKernelDim == 4)
        #expect(a.fullAttentionLayerMask.count == 48)
        #expect(a.fullAttentionLayerMask[3] == 1)
        #expect(a.fullAttentionLayerMask[47] == 1)
        #expect(a.fullAttentionLayerMask[0] == 0)
        #expect(a.hiddenActivation == "silu")
        #expect(a.ropeTheta == 10_000_000)
        #expect(a.fullRopeTheta == 10_000_000)
    }

    @Test func qwen36ConfigStillMapsToQwen36AndOmitsQwen38Fields() throws {
        var text = try JSONSerialization.jsonObject(
            with: Self.qwen38ConfigJSON()) as! [String: Any]
        text["model_type"] = "qwen3_5_moe"
        text["num_hidden_layers"] = 40
        // qwen3_5_moe signals the gate with attn_output_gate, not output_gate_type.
        text["attn_output_gate"] = true
        text.removeValue(forKey: "output_gate_type")
        // layer_types has only 40 entries now; rebuild via a re-serialized dict
        // is handled below by trimming the mask keys.
        var layerTypes: [String] = []
        for i in 0..<40 { layerTypes.append(i % 4 == 3 ? "full_attention" : "linear_attention") }
        text["layer_types"] = layerTypes
        // 3.8-only keys removed (the 3.6 config carries none of them).
        for key in ["hc_count", "hc_lowrank", "indexer_n_heads",
                    "indexer_kv_heads", "indexer_head_dim", "indexer_budget",
                    "indexer_compress_ratio", "ngram_size", "heads_per_ngram",
                    "split_ngram_parts", "ple_conv_kernel_size", "ple_layer_ids"] {
            text.removeValue(forKey: key)
        }
        let a = try Self.load(try JSONSerialization.data(withJSONObject: text))
        #expect(a.modelFamily == "qwen3_6")
        #expect(a.attnOutputGate == true)
        #expect(a.hyperConnectionCount == nil)
        #expect(a.ngramRowDim == nil)
        #expect(a.pleLayerIndexes == nil)
    }
}

import Foundation
import FinchMoEFormat
@testable import FinchMoERepackCore

/// Builds a tiny synthetic Qwen 3.6 bf16 safetensors snapshot on disk so the
/// quantizing repack can be exercised end to end without the real 70 GB
/// checkpoint. Mirrors the real tensor inventory: `model.language_model.*`
/// nesting, fused `mlp.experts.gate_up_proj`, untied `lm_head.weight`, vision
/// tensors to exclude.
enum SyntheticQwenSnapshot {

    enum Toy {
        static let D = 64
        static let intermediate = 64
        static let moeIntermediate = 64
        static let numHeads = 4
        static let numKVHeads = 2
        static let numFullKVHeads = 2
        static let linearKeyHeads = 4
        static let linearValueHeads = 8
        static let linearKeyHeadDim = 32
        static let linearValueHeadDim = 32
        static let fullHeadDim = 32
        static let vocab = 256
        static let numLayers = 4
        static let experts = 4
        static let topK = 2
        static let convKernel = 4
        /// [L, L, L, F]
        static let fullMask: [Int] = [0, 0, 0, 1]
        static let keyDim = linearKeyHeads * linearKeyHeadDim              // 128
        static let valueDim = linearValueHeads * linearValueHeadDim        // 256
        static let qkvDim = 2 * keyDim + valueDim                          // 512

        static func layerTypes() -> [String] {
            fullMask.map { $0 == 1 ? "full_attention" : "linear_attention" }
        }

        static func configJSON() -> [String: Any] {
            [
                "model_type": "qwen3_5_moe",
                "text_config": [
                    "model_type": "qwen3_5_moe_text",
                    "hidden_size": D,
                    "num_hidden_layers": numLayers,
                    "num_attention_heads": numHeads,
                    "num_key_value_heads": numKVHeads,
                    "head_dim": fullHeadDim,
                    "linear_num_key_heads": linearKeyHeads,
                    "linear_num_value_heads": linearValueHeads,
                    "linear_key_head_dim": linearKeyHeadDim,
                    "linear_value_head_dim": linearValueHeadDim,
                    "linear_conv_kernel_dim": convKernel,
                    "vocab_size": vocab,
                    "num_experts": experts,
                    "num_experts_per_tok": topK,
                    "moe_intermediate_size": moeIntermediate,
                    "shared_expert_intermediate_size": intermediate,
                    "tie_word_embeddings": false,
                    "attention_k_eq_v": false,
                    "hidden_act": "silu",
                    "attn_output_gate": true,
                    "layer_types": layerTypes(),
                    "rope_parameters": [
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 0.25,
                    ],
                    "sliding_window": 0,
                    "final_logit_softcapping": 0.0,
                ],
            ]
        }

        /// All checkpoint tensor names → shapes, in deterministic order.
        static func tensorShapes() -> [(name: String, shape: [Int])] {
            var out: [(String, [Int])] = [
                ("model.language_model.embed_tokens.weight", [vocab, D]),
                ("model.language_model.norm.weight", [D]),
                ("lm_head.weight", [vocab, D]),
                ("model.visual.blocks.0.attn.proj.weight", [D, D]),
            ]
            for L in 0..<numLayers {
                let p = "model.language_model.layers.\(L)"
                out.append((p + ".input_layernorm.weight", [D]))
                out.append((p + ".post_attention_layernorm.weight", [D]))
                if fullMask[L] == 1 {
                    out.append((p + ".self_attn.q_proj.weight", [2 * numHeads * fullHeadDim, D]))
                    out.append((p + ".self_attn.k_proj.weight", [numFullKVHeads * fullHeadDim, D]))
                    out.append((p + ".self_attn.v_proj.weight", [numFullKVHeads * fullHeadDim, D]))
                    out.append((p + ".self_attn.o_proj.weight", [D, numHeads * fullHeadDim]))
                    out.append((p + ".self_attn.q_norm.weight", [fullHeadDim]))
                    out.append((p + ".self_attn.k_norm.weight", [fullHeadDim]))
                } else {
                    out.append((p + ".linear_attn.in_proj_qkv.weight", [qkvDim, D]))
                    out.append((p + ".linear_attn.in_proj_z.weight", [valueDim, D]))
                    out.append((p + ".linear_attn.in_proj_a.weight", [linearValueHeads, D]))
                    out.append((p + ".linear_attn.in_proj_b.weight", [linearValueHeads, D]))
                    out.append((p + ".linear_attn.out_proj.weight", [D, valueDim]))
                    out.append((p + ".linear_attn.norm.weight", [linearValueHeadDim]))
                    out.append((p + ".linear_attn.conv1d.weight", [qkvDim, 1, convKernel]))
                    out.append((p + ".linear_attn.A_log", [linearValueHeads]))
                    out.append((p + ".linear_attn.dt_bias", [linearValueHeads]))
                }
                out.append((p + ".mlp.gate.weight", [experts, D]))
                out.append((p + ".mlp.shared_expert.gate_proj.weight", [intermediate, D]))
                out.append((p + ".mlp.shared_expert.up_proj.weight", [intermediate, D]))
                out.append((p + ".mlp.shared_expert.down_proj.weight", [D, intermediate]))
                out.append((p + ".mlp.shared_expert_gate.weight", [1, D]))
                out.append((p + ".mlp.experts.gate_up_proj", [experts, 2 * moeIntermediate, D]))
                out.append((p + ".mlp.experts.down_proj", [experts, D, moeIntermediate]))
            }
            return out
        }

        static func residentEntryCount() -> Int {
            // 3 global + 3 GDN layers × 16 + 1 full layer × 13
            3 + 3 * 16 + 13
        }
    }

    /// Deterministic bf16 bits for `count` elements, in a realistic weight
    /// range (roughly ±2). Raw random 16-bit patterns would span ±3e38 and
    /// overflow the quantizer's per-group Float32 range arithmetic.
    static func bf16Bits(count: Int, seed: UInt64) -> [UInt16] {
        var state = seed
        var out = [UInt16](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
            let value = -2.0 + 4.0 * fraction
            out[i] = FinchTurboQuantization.bf16Bits(value)
        }
        return out
    }

    /// Writes a single-shard snapshot into `dir`: config.json,
    /// model.safetensors.index.json, model-00001-of-00001.safetensors, and
    /// dummy tokenizer files. Returns the snapshot directory path.
    @discardableResult
    static func write(into dir: String, seed: UInt64 = 0x51A7) throws -> String {
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: dir), withIntermediateDirectories: true)

        let configData = try JSONSerialization.data(
            withJSONObject: Toy.configJSON(), options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: dir).appendingPathComponent("config.json"))

        let shardName = "model-00001-of-00001.safetensors"
        let shapes = Toy.tensorShapes()
        var header: [String: Any] = [:]
        var weightMap: [String: String] = [:]
        var payload = Data()
        var cursor = 0
        for (name, shape) in shapes {
            let elements = shape.reduce(1, *)
            let bits = bf16Bits(count: elements, seed: seed &+ UInt64(name.count) &* 7919)
            var bytes = Data(capacity: bits.count * 2)
            for b in bits {
                bytes.append(UInt8(truncatingIfNeeded: b & 0xFF))
                bytes.append(UInt8(truncatingIfNeeded: b >> 8))
            }
            let start = cursor
            cursor += bytes.count
            header[name] = [
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [start, cursor],
            ]
            weightMap[name] = shardName
            payload.append(bytes)
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var file = Data(capacity: 8 + headerData.count + payload.count)
        var headerLen = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
        file.append(headerData)
        file.append(payload)
        try file.write(to: URL(fileURLWithPath: dir).appendingPathComponent(shardName))

        let index: [String: Any] = ["weight_map": weightMap]
        let indexData = try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
        try indexData.write(
            to: URL(fileURLWithPath: dir).appendingPathComponent("model.safetensors.index.json"))

        for tokenizer in ["tokenizer.json", "tokenizer_config.json"] {
            try Data("{}".utf8).write(
                to: URL(fileURLWithPath: dir).appendingPathComponent(tokenizer))
        }
        return dir
    }
}

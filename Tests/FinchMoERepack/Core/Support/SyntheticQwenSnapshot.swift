import Foundation
import FinchMoEFormat
@testable import FinchMoERepackCore

/// Builds tiny synthetic Qwen bf16 safetensors snapshots on disk so the
/// quantizing repack can be exercised end to end without the real 70 GB /
/// 352 GB checkpoints. Mirrors the real tensor inventory for both families:
/// `model.language_model.*` nesting, fused `mlp.experts.gate_up_proj`, untied
/// `lm_head.weight`, vision / MTP tensors to exclude. Qwen3.8 additionally
/// carries the hyper-connection mixers (per-layer + root, no final norm), the
/// QSA indexer on full layers, and the layer-1 PLE block — I64 hash metadata
/// and raw-BF16 `shard_N.weight` part tensors.
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

    /// Qwen3.8-Flash-Next toy. Geometry mirrors the real model's derived
    /// relationships: the hyper-connection plane is 4×D wide (hc_count 4);
    /// PLE gathers one 160-wide row of the frozen n-gram table per head,
    /// `pleHeads = (ngram_size − 1) × heads_per_ngram` in total (4 × 160 =
    /// 640 — the real model's 16 = 2 × 8), and `ngram_heads_offsets` /
    /// `ngram_heads_vocab_sizes` carry one I64 entry per head; conv1d and the
    /// PLE norms run at plane width; the full layers carry the doubled-q
    /// self_attn plus the QSA indexer (`(q+kv) heads × headDim` rows). PLE
    /// sits on layer 1 (GDN, like the real checkpoint) and parts are
    /// 4 × 32 × 160 — the config's `split_ngram_parts` says 4, and the
    /// planner census-corrects the arch's part rows (32) from the live
    /// headers. Note `heads_per_ngram` is per ORDER, not total: the engine's
    /// runtime schema derives the gathered width and I64 counts from
    /// (ngramSize − 1) × headsPerNgram, so the toy config must agree with
    /// its own tensor shapes under that formula.
    enum Toy38 {
        static let D = 64
        static let plane = 4 * D                        // hc_count 4
        static let hcLowrank = 64
        static let intermediate = 64
        static let moeIntermediate = 64
        static let numHeads = 4
        static let numKVHeads = 2                       // full layers' GQA (and GDN kv)
        static let fullHeadDim = 32
        static let linearKeyHeads = 4
        static let linearValueHeads = 8
        static let linearKeyHeadDim = 32
        static let linearValueHeadDim = 32
        static let convKernel = 4
        static let vocab = 256
        static let numLayers = 4
        static let experts = 4
        static let topK = 2
        static let fullMask: [Int] = [0, 0, 0, 1]       // full at layer 3
        static let keyDim = linearKeyHeads * linearKeyHeadDim
        static let valueDim = linearValueHeads * linearValueHeadDim
        static let qkvDim = 2 * keyDim + valueDim

        // QSA indexer: (n q-heads + kv) × head_dim rows of index_qk_proj.
        static let indexerNumHeads = 2
        static let indexerKVHeads = 1
        static let indexerHeadDim = 32
        static let indexerBudget = 512
        static let indexerCompressRatio = 4

        // PLE n-gram block (layer 1 only). ngramRowDim is frozen at 160 in
        // ArchInfo for qwen3_8; the part rows (32) are census-corrected.
        static let ngramSize = 3
        /// Per-gram-order heads (2) — the real model's 8 with ngram_size 3;
        /// total gathered heads = (ngramSize − 1) × headsPerNgram = 4.
        static let headsPerNgram = 2
        static let ngramRowDim = 160
        static let ngramPartCount = 4
        static let ngramPartRows = 32
        static let pleConvKernel = 4
        /// Config ple_layer_ids is 1-based; engine layer = id - 1.
        static let pleLayerIDs = [2]
        /// Total PLE heads: one 160-wide table row gathered per head.
        static var pleHeads: Int { (ngramSize - 1) * headsPerNgram }   // 4
        static var ngramWidth: Int { pleHeads * ngramRowDim }          // 640

        static func layerTypes() -> [String] {
            fullMask.map { $0 == 1 ? "full_attention" : "linear_attention" }
        }

        static func configJSON() -> [String: Any] {
            [
                "model_type": "qwen4_exp",
                "text_config": [
                    "model_type": "qwen4_exp_text",
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
                    "hidden_act": "silu",
                    // 3.8 GDN output gate is sigmoid (3.6's is silu); the
                    // doubled-q gate half is present either way.
                    "output_gate_type": "sigmoid",
                    "layer_types": layerTypes(),
                    "rope_parameters": [
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 0.25,
                    ],
                    "sliding_window": 0,
                    "final_logit_softcapping": 0.0,
                    // Hyper-connection.
                    "hc_count": 4,
                    "hc_lowrank": hcLowrank,
                    // QSA indexer.
                    "indexer_n_heads": indexerNumHeads,
                    "indexer_kv_heads": indexerKVHeads,
                    "indexer_head_dim": indexerHeadDim,
                    "indexer_budget": indexerBudget,
                    "indexer_compress_ratio": indexerCompressRatio,
                    // PLE n-gram.
                    "ngram_size": ngramSize,
                    "heads_per_ngram": headsPerNgram,
                    "split_ngram_parts": ngramPartCount,
                    "ple_conv_kernel_size": pleConvKernel,
                    "ple_layer_ids": pleLayerIDs,
                ],
            ]
        }

        /// All checkpoint tensor names → shapes → dtypes, deterministic order.
        static func tensorShapes() -> [(name: String, shape: [Int], dtype: String)] {
            var out: [(String, [Int], String)] = [
                ("model.language_model.embed_tokens.weight", [vocab, D], "BF16"),
                ("lm_head.weight", [vocab, D], "BF16"),
                // Root hyper-connection mixer: no block_inject_weight at root,
                // and there is no model.norm anywhere in this family.
                ("model.language_model.hyper_connection_mixer.hc_norm.weight", [plane], "BF16"),
                ("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                 [hcLowrank, plane], "BF16"),
                ("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                 [plane, hcLowrank], "BF16"),
                // Decoys the planner must drop: vision tower + MTP next-token
                // head (mtp carries a `.layers.0.` stage and hyper-connection
                // names that must never reach routing / transform).
                ("model.visual.blocks.0.attn.qkv.weight", [D, D], "BF16"),
                ("model.visual.blocks.0.attn.proj.weight", [D, D], "BF16"),
                ("mtp.layers.0.self_attn.q_proj.weight", [2 * numHeads * fullHeadDim, D], "BF16"),
                ("mtp.hyper_connection_mixer.hc_norm.weight", [plane], "BF16"),
            ]
            for L in 0..<numLayers {
                let p = "model.language_model.layers.\(L)"
                // Per-layer hyper-connection bundles (all 48 real layers have
                // them; block_inject_weight only exists here, not at root).
                out.append((p + ".attn_hyper_connection.hc_norm.weight", [plane], "BF16"))
                out.append((p + ".attn_hyper_connection.input_mix_weight_down.weight",
                            [hcLowrank, plane], "BF16"))
                out.append((p + ".attn_hyper_connection.input_mix_weight_up.weight",
                            [plane, hcLowrank], "BF16"))
                out.append((p + ".attn_hyper_connection.block_inject_weight.weight",
                            [4, plane], "BF16"))
                out.append((p + ".mlp_hyper_connection.hc_norm.weight", [plane], "BF16"))
                out.append((p + ".mlp_hyper_connection.input_mix_weight_down.weight",
                            [hcLowrank, plane], "BF16"))
                out.append((p + ".mlp_hyper_connection.input_mix_weight_up.weight",
                            [plane, hcLowrank], "BF16"))
                out.append((p + ".mlp_hyper_connection.block_inject_weight.weight",
                            [4, plane], "BF16"))
                if fullMask[L] == 1 {
                    out.append((p + ".self_attn.q_proj.weight",
                                [2 * numHeads * fullHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.k_proj.weight",
                                [numKVHeads * fullHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.v_proj.weight",
                                [numKVHeads * fullHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.o_proj.weight",
                                [D, numHeads * fullHeadDim], "BF16"))
                    out.append((p + ".self_attn.q_norm.weight", [fullHeadDim], "BF16"))
                    out.append((p + ".self_attn.k_norm.weight", [fullHeadDim], "BF16"))
                    out.append((p + ".self_attn.indexer.index_qk_proj.weight",
                                [(indexerNumHeads + indexerKVHeads) * indexerHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.indexer.q_layernorm.weight",
                                [indexerHeadDim], "BF16"))
                    out.append((p + ".self_attn.indexer.k_layernorm.weight",
                                [indexerHeadDim], "BF16"))
                } else {
                    out.append((p + ".linear_attn.in_proj_qkv.weight", [qkvDim, D], "BF16"))
                    out.append((p + ".linear_attn.in_proj_z.weight", [valueDim, D], "BF16"))
                    out.append((p + ".linear_attn.in_proj_a.weight",
                                [linearValueHeads, D], "BF16"))
                    out.append((p + ".linear_attn.in_proj_b.weight",
                                [linearValueHeads, D], "BF16"))
                    out.append((p + ".linear_attn.out_proj.weight", [D, valueDim], "BF16"))
                    out.append((p + ".linear_attn.norm.weight", [linearValueHeadDim], "BF16"))
                    out.append((p + ".linear_attn.conv1d.weight", [qkvDim, 1, convKernel], "BF16"))
                    out.append((p + ".linear_attn.A_log", [linearValueHeads], "BF16"))
                    out.append((p + ".linear_attn.dt_bias", [linearValueHeads], "BF16"))
                }
                if pleLayerIDs.contains(L + 1) {
                    // PLE head projections + norms at plane/hidden widths, and
                    // the I64 hash metadata (multipliers per gram position,
                    // per-head offsets/vocab sizes into the combined table).
                    out.append((p + ".ple.conv1d.weight", [plane, 1, pleConvKernel], "BF16"))
                    out.append((p + ".ple.key_proj.weight", [plane, ngramWidth], "BF16"))
                    out.append((p + ".ple.value_proj.weight", [D, ngramWidth], "BF16"))
                    out.append((p + ".ple.norm_query.weight", [plane], "BF16"))
                    out.append((p + ".ple.norm_key.weight", [plane], "BF16"))
                    out.append((p + ".ple.norm_conv.weight", [plane], "BF16"))
                    out.append((p + ".ple.ple_embedding.layer_multipliers",
                                [ngramSize], "I64"))
                    out.append((p + ".ple.ple_embedding.ngram_heads_offsets",
                                [pleHeads], "I64"))
                    out.append((p + ".ple.ple_embedding.ngram_heads_vocab_sizes",
                                [pleHeads], "I64"))
                    // The n-gram table parts ride their own files. Names are
                    // lexical in the real snapshot (shard_10 < shard_2); the
                    // toy keeps <10 parts so any ordering bug stays invisible
                    // here and is covered by the real-part tests in M4.
                    for i in 0..<ngramPartCount {
                        out.append((p + ".ple.ple_embedding.ngram_embedding.shard_\(i).weight",
                                    [ngramPartRows, ngramRowDim], "BF16"))
                    }
                }
                out.append((p + ".mlp.gate.weight", [experts, D], "BF16"))
                out.append((p + ".mlp.shared_expert.gate_proj.weight",
                            [intermediate, D], "BF16"))
                out.append((p + ".mlp.shared_expert.up_proj.weight",
                            [intermediate, D], "BF16"))
                out.append((p + ".mlp.shared_expert.down_proj.weight",
                            [D, intermediate], "BF16"))
                out.append((p + ".mlp.shared_expert_gate.weight", [1, D], "BF16"))
                out.append((p + ".mlp.experts.gate_up_proj",
                            [experts, 2 * moeIntermediate, D], "BF16"))
                out.append((p + ".mlp.experts.down_proj", [experts, D, moeIntermediate], "BF16"))
            }
            return out
        }

        /// Resident entries: embed + lm_head + root mixer (3) = 5 global;
        /// GDN layers carry 9 linear_attn + 5 mlp + 8 hyper-connection = 22,
        /// full layers 6 self_attn + 5 mlp + 8 HC + 3 indexer = 22, and the
        /// layer-1 PLE block adds 6 head BF16 + 3 I64 metadata = 9.
        static var residentEntryCount: Int {
            5 + (numLayers - 1) * 22 + 22 + 9
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
            out[i] = FinchQuantization.bf16Bits(value)
        }
        return out
    }

    /// Deterministic raw little-endian int64 bytes (I64 PLE metadata tensors).
    private static func i64Bytes(count: Int, seed: UInt64) -> Data {
        var state = seed
        var out = Data(capacity: count * 8)
        for _ in 0..<count {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            var value = Int64(bitPattern: state)
            withUnsafeBytes(of: &value) { out.append(contentsOf: $0) }
        }
        return out
    }

    private static func bf16Bytes(count: Int, seed: UInt64) -> Data {
        let bits = bf16Bits(count: count, seed: seed)
        var bytes = Data(capacity: bits.count * 2)
        for b in bits {
            bytes.append(UInt8(truncatingIfNeeded: b & 0xFF))
            bytes.append(UInt8(truncatingIfNeeded: b >> 8))
        }
        return bytes
    }

    /// Writes the single-shard snapshot files (config.json,
    /// model.safetensors.index.json, model-00001-of-00001.safetensors,
    /// dummy tokenizers) for an `entries` inventory, per-tensor dtype-aware.
    private static func writeFiles(into dir: String,
                                   config: [String: Any],
                                   entries: [(name: String, shape: [Int], dtype: String)],
                                   seed: UInt64) throws -> String {
        try FileManager.default.createDirectory(
            at: URL(fileURLWithPath: dir), withIntermediateDirectories: true)

        let configData = try JSONSerialization.data(
            withJSONObject: config, options: [.sortedKeys])
        try configData.write(to: URL(fileURLWithPath: dir).appendingPathComponent("config.json"))

        let shardName = "model-00001-of-00001.safetensors"
        var header: [String: Any] = [:]
        var weightMap: [String: String] = [:]
        var payload = Data()
        var cursor = 0
        for (name, shape, dtype) in entries {
            let elements = shape.reduce(1, *)
            let bytes: Data
            if dtype == "I64" {
                bytes = i64Bytes(count: elements, seed: seed &+ UInt64(name.count) &* 104_729)
            } else {
                bytes = bf16Bytes(count: elements, seed: seed &+ UInt64(name.count) &* 7919)
            }
            let start = cursor
            cursor += bytes.count
            header[name] = [
                "dtype": dtype,
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

    /// Writes a single-shard snapshot into `dir`: config.json,
    /// model.safetensors.index.json, model-00001-of-00001.safetensors, and
    /// dummy tokenizer files. Returns the snapshot directory path.
    @discardableResult
    static func write(into dir: String, seed: UInt64 = 0x51A7) throws -> String {
        try writeFiles(into: dir,
                       config: Toy.configJSON(),
                       entries: Toy.tensorShapes().map {
                           (name: $0.name, shape: $0.shape, dtype: "BF16")
                       },
                       seed: seed)
    }

    /// The qwen3_8-Flash-Next toy (see `Toy38`): mixed BF16/I64 shard with
    /// hyper-connection, indexer, and PLE tensors plus vision/MTP decoys.
    @discardableResult
    static func write38(into dir: String, seed: UInt64 = 0x51A8) throws -> String {
        try writeFiles(into: dir,
                       config: Toy38.configJSON(),
                       entries: Toy38.tensorShapes(),
                       seed: seed)
    }
}

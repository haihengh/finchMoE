import Testing
import Foundation
import Metal
import FinchMoERepackCore
import FinchMoEFormat
import FinchMoEValidationSupport
@testable import FinchMoE

/// End-to-end proof of the Qwen3.8-Flash-Next load path: a tiny synthetic
/// bf16 checkpoint with the full 3.8 tensor inventory (hyper-connection
/// mixers, QSA indexer, PLE n-gram head + raw part files, root mixer, no
/// `model.norm`) is repacked into `.finch`, then the real runtime
/// `Model.load(expecting:)` passes `validateRuntimeSchema` +
/// `validateArch` and every family-branching accessor resolves. This is the
/// gate the real 125B install must pass before M3 can run on it.
///
/// The toy geometry follows the REAL derived relationships (mirroring the
/// repack support's `SyntheticQwenSnapshot.Toy38`): total PLE heads =
/// (ngramSize − 1) × headsPerNgram, so key/value projections are
/// [·, pleHeads × 160] and the I64 offsets/vocab arrays hold pleHeads
/// entries — the engine schema derives all three from the config.
@Suite struct Qwen38EngineLoadTests {

    // MARK: - Toy model (mirrors SyntheticQwenSnapshot.Toy38, engine-valid)

    // Internal, not fileprivate: `Qwen38ToyReplayTests` (a separate file)
    // composes an fp32 replay against this same toy geometry.
    enum Toy38 {
        static let D = 64
        static let plane = 4 * D                        // hc_count 4
        static let hcLowrank = 64
        static let intermediate = 64
        static let moeIntermediate = 64
        static let numHeads = 4
        static let numKVHeads = 2
        static let numFullKVHeads = 2                  // k/v heads on full layers
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
        static let fullMask: [UInt8] = [0, 0, 0, 1]     // full at layer 3
        static let keyDim = linearKeyHeads * linearKeyHeadDim
        static let valueDim = linearValueHeads * linearValueHeadDim
        static let qkvDim = 2 * keyDim + valueDim

        // QSA indexer (full layers only).
        //
        // The budget is deliberately tiny: the selection width is
        // `budget + r − 1` (11 here), so a decode of a dozen-odd tokens
        // outgrows it and the full layer must rank blocks instead of reading
        // the whole timeline. A budget at or above the context — the real
        // model's 2048 against a 256-cell toy — would leave the selector
        // permanently in its dense regime, never exercised.
        static let indexerNumHeads = 2
        static let indexerKVHeads = 1
        static let indexerHeadDim = 32
        static let indexerBudget = 8
        static let indexerCompressRatio = 4

        // PLE n-gram block (layer 1 only). pleHeads total heads =
        // (ngramSize − 1) × headsPerNgram — the real model's 16 = 2 × 8.
        static let ngramSize = 3
        static let headsPerNgram = 2
        static let ngramRowDim = 160
        static let ngramPartCount = 4
        static let ngramPartRows = 32
        static let pleConvKernel = 4
        /// The hash's cut token (`eos_token_id`). Any nonzero in-vocab value:
        /// 0 would make every window read as "the previous token was an EOS",
        /// which the engine refuses rather than hashing wrong.
        static let pleEosTokenId = 200
        /// Config ple_layer_ids is 1-based; the engine arch stores id − 1.
        static let pleLayerIDs = [2]
        static var pleHeads: Int { (ngramSize - 1) * headsPerNgram }   // 4
        static var ngramWidth: Int { pleHeads * ngramRowDim }          // 640

        static let arch = ArchConfig(
            hiddenSize: D,
            intermediateSize: intermediate,
            moeIntermediateSize: moeIntermediate,
            numHeads: numHeads,
            numKVHeads: numKVHeads,
            numFullKVHeads: numFullKVHeads,
            headDim: linearKeyHeadDim,
            fullHeadDim: fullHeadDim,
            vocabSize: vocab,
            slidingWindow: 0,
            finalLogitSoftcap: 0,
            ropeTheta: 1_000_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: numLayers,
            numExperts: experts,
            topKExperts: topK,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: fullMask,
            hiddenActivation: "silu",
            modelFamily: "qwen3_8",
            attnOutputGate: true,
            linearNumKeyHeads: linearKeyHeads,
            linearNumValueHeads: linearValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            linearConvKernelDim: convKernel,
            hyperConnectionCount: 4,
            hyperConnectionLowrank: hcLowrank,
            indexerNumHeads: indexerNumHeads,
            indexerKVHeads: indexerKVHeads,
            indexerHeadDim: indexerHeadDim,
            indexerBudget: indexerBudget,
            indexerCompressRatio: indexerCompressRatio,
            ngramSize: ngramSize,
            headsPerNgram: headsPerNgram,
            ngramRowDim: ngramRowDim,
            ngramPartCount: ngramPartCount,
            ngramPartRows: ngramPartRows,
            pleLayerIndexes: pleLayerIDs.map { $0 - 1 },
            pleConvKernelSize: pleConvKernel,
            pleEosTokenId: pleEosTokenId)

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
                    "output_gate_type": "sigmoid",
                    "layer_types": fullMask.map { $0 == 1 ? "full_attention" : "linear_attention" },
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
                    "eos_token_id": pleEosTokenId,
                ],
            ]
        }

        /// All checkpoint tensor names → shapes → dtypes, deterministic order.
        /// No visual/MTP decoys: the planner exclusion of those is asserted
        /// repack-side; this inventory is exactly what the engine schema
        /// requires, so a spurious requirement surfaces as a missing tensor.
        static func tensorShapes() -> [(name: String, shape: [Int], dtype: String)] {
            var out: [(String, [Int], String)] = [
                ("model.language_model.embed_tokens.weight", [vocab, D], "BF16"),
                ("lm_head.weight", [vocab, D], "BF16"),
                // Root hyper-connection mixer (no block_inject at the root,
                // and no model.norm anywhere in this family).
                ("model.language_model.hyper_connection_mixer.hc_norm.weight", [plane], "BF16"),
                ("model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                 [hcLowrank, plane], "BF16"),
                ("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                 [plane, hcLowrank], "BF16"),
            ]
            for L in 0..<numLayers {
                let p = "model.language_model.layers.\(L)"
                for bundle in ["attn_hyper_connection", "mlp_hyper_connection"] {
                    out.append((p + ".\(bundle).hc_norm.weight", [plane], "BF16"))
                    out.append((p + ".\(bundle).input_mix_weight_down.weight",
                                [hcLowrank, plane], "BF16"))
                    out.append((p + ".\(bundle).input_mix_weight_up.weight",
                                [plane, hcLowrank], "BF16"))
                    out.append((p + ".\(bundle).block_inject_weight.weight",
                                [4, plane], "BF16"))
                }
                if fullMask[L] == 1 {
                    out.append((p + ".self_attn.q_proj.weight",
                                [2 * numHeads * fullHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.k_proj.weight",
                                [numFullKVHeads * fullHeadDim, D], "BF16"))
                    out.append((p + ".self_attn.v_proj.weight",
                                [numFullKVHeads * fullHeadDim, D], "BF16"))
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
                    out.append((p + ".linear_attn.conv1d.weight",
                                [qkvDim, 1, convKernel], "BF16"))
                    out.append((p + ".linear_attn.A_log", [linearValueHeads], "BF16"))
                    out.append((p + ".linear_attn.dt_bias", [linearValueHeads], "BF16"))
                }
                if pleLayerIDs.contains(L + 1) {
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
                out.append((p + ".mlp.experts.down_proj",
                            [experts, D, moeIntermediate], "BF16"))
            }
            return out
        }

        /// Deterministic raw little-endian int64 bytes (I64 PLE metadata).
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

        /// Real values for the three PLE hash vectors, or nil for anything
        /// else. These cannot be noise like the weights: the engine checks
        /// every head's reach against the table's `partCount × partRows` rows
        /// when it builds the host router (llama's own load-time bound,
        /// `qwen4exp.cpp:135`), and random int64s trip it. Odd coprime
        /// multipliers; offsets that put heads 2 and 3 in the back half of
        /// the table, so a gather crosses a part boundary.
        private static func pleHashMeta(for name: String, count: Int) -> Data? {
            let half = Int64(ngramPartCount * ngramPartRows / 2)      // 64 rows
            let values: [Int64]
            if name.hasSuffix(".ple_embedding.layer_multipliers") {
                values = (0..<count).map { Int64(2 * $0 + 3) }        // 3, 5, 7
            } else if name.hasSuffix(".ple_embedding.ngram_heads_offsets") {
                values = (0..<count).map { Int64($0 / headsPerNgram) * half }
            } else if name.hasSuffix(".ple_embedding.ngram_heads_vocab_sizes") {
                values = Array(repeating: half, count: count)
            } else {
                return nil
            }
            var out = Data(capacity: count * 8)
            for var value in values {
                withUnsafeBytes(of: &value) { out.append(contentsOf: $0) }
            }
            return out
        }

        /// Deterministic BF16 bytes in a realistic weight range (roughly ±2),
        /// seeded per-tensor the same way the repack-side snapshot does, so
        /// the repacked install reproduces the source byte-for-byte.
        static func bf16Bytes(count: Int, seed: UInt64) -> Data {
            var state = seed
            var out = Data(capacity: count * 2)
            for _ in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
                let value = -2.0 + 4.0 * fraction
                let bits = FinchQuantization.bf16Bits(value)
                out.append(UInt8(truncatingIfNeeded: bits & 0xFF))
                out.append(UInt8(truncatingIfNeeded: bits >> 8))
            }
            return out
        }

        /// Writes the single-shard snapshot into `dir` (config.json, index,
        /// shard, dummy tokenizers). Returns the directory.
        static func writeSnapshot(into dir: String, seed: UInt64 = 0x51A9) throws {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: dir), withIntermediateDirectories: true)
            let configData = try JSONSerialization.data(
                withJSONObject: configJSON(), options: [.sortedKeys])
            try configData.write(
                to: URL(fileURLWithPath: dir).appendingPathComponent("config.json"))

            let shardName = "model-00001-of-00001.safetensors"
            var header: [String: Any] = [:]
            var weightMap: [String: String] = [:]
            var payload = Data()
            var cursor = 0
            for (name, shape, dtype) in tensorShapes() {
                let elements = shape.reduce(1, *)
                let bytes: Data
                if dtype == "I64" {
                    bytes = pleHashMeta(for: name, count: elements)
                        ?? i64Bytes(count: elements,
                                    seed: seed &+ UInt64(name.count) &* 104_729)
                } else {
                    bytes = bf16Bytes(count: elements, seed: seed &+ UInt64(name.count) &* 7919)
                }
                let start = cursor
                cursor += bytes.count
                header[name] = ["dtype": dtype, "shape": shape,
                                "data_offsets": [start, cursor]]
                weightMap[name] = shardName
                payload.append(bytes)
            }
            var headerData = try JSONSerialization.data(withJSONObject: header,
                                                        options: [.sortedKeys])
            let pad = (8 - headerData.count % 8) % 8
            headerData.append(Data(repeating: 0x20, count: pad))
            var file = Data(capacity: 8 + headerData.count + payload.count)
            var headerLen = UInt64(headerData.count).littleEndian
            withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
            file.append(headerData)
            file.append(payload)
            try file.write(to: URL(fileURLWithPath: dir).appendingPathComponent(shardName))

            let index: [String: Any] = ["weight_map": weightMap]
            let indexData = try JSONSerialization.data(withJSONObject: index,
                                                       options: [.sortedKeys])
            try indexData.write(to: URL(fileURLWithPath: dir)
                .appendingPathComponent("model.safetensors.index.json"))
            for tokenizer in ["tokenizer.json", "tokenizer_config.json"] {
                try Data("{}".utf8).write(
                    to: URL(fileURLWithPath: dir).appendingPathComponent(tokenizer))
            }
        }
    }

    // MARK: - Harness

    /// Repacks a fresh toy snapshot and returns the install directory.
    static func makeInstall() async throws -> String {
        let base = NSTemporaryDirectory() + "qwen38-engine-\(UUID().uuidString)"
        let src = base + "-src"
        let out = base + "-out"
        try Toy38.writeSnapshot(into: src)
        defer { try? FileManager.default.removeItem(atPath: src) }
        try? FileManager.default.removeItem(atPath: out)
        let options = LocalQwenRepackOptions(
            snapshotDir: src, outputDir: out, minFreeReserveBytes: 0)
        _ = try await LocalQwenRepacker(options: options).run()
        return out
    }

    static func loadToy38() async throws -> Model {
        let directory = try await Self.makeInstall()
        let device = try #require(MTLCreateSystemDefaultDevice())
        return try Model.load(directoryURL: URL(fileURLWithPath: directory),
                              device: device,
                              expecting: Toy38.arch)
    }

    // MARK: - Load smoke + accessors

    @Test func toy38InstallLoadsAndValidates() async throws {
        let out = try await Self.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }

        // The manifest family selects the BUILT-IN preset (production dims).
        let detected = try ManifestReader.detectPreset(
            directoryURL: URL(fileURLWithPath: out))
        #expect(detected.modelFamily == ArchConfig.qwen3_8Family)

        // Full runtime load: manifest + resident decode + validateArch
        // against the toy arch + validateQwen38Layers.
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: URL(fileURLWithPath: out),
                                   device: device,
                                   expecting: Toy38.arch)

        // Global tensors + the missing final norm (3.8: no model.norm).
        #expect(model.embedding.shape == (UInt32(Toy38.vocab), UInt32(Toy38.D), 0, 0))
        #expect(model.lmHead.shape == (UInt32(Toy38.vocab), UInt32(Toy38.D), 0, 0))
        #expect(model.finalNorm == nil)
        #expect(model.pleLayerIndex == 1)

        // Hyper-connection mixers: per-layer 4-tuples + root 3-tuple.
        for L in 0..<Toy38.numLayers {
            let attn = try model.attnHyperConnection(layer: L)
            #expect(attn.hcNorm.shape == (UInt32(Toy38.plane), 0, 0, 0))
            #expect(attn.mixDown.shape == (UInt32(Toy38.hcLowrank), UInt32(Toy38.plane), 0, 0))
            #expect(attn.mixUp.shape == (UInt32(Toy38.plane), UInt32(Toy38.hcLowrank), 0, 0))
            #expect(attn.blockInject.shape == (4, UInt32(Toy38.plane), 0, 0))
            let mlp = try model.mlpHyperConnection(layer: L)
            #expect(mlp.hcNorm.shape == (UInt32(Toy38.plane), 0, 0, 0))
            #expect(mlp.blockInject.shape == (4, UInt32(Toy38.plane), 0, 0))
        }
        let root = try model.hyperConnectionMixer()
        #expect(root.hcNorm.shape == (UInt32(Toy38.plane), 0, 0, 0))
        #expect(root.mixDown.shape == (UInt32(Toy38.hcLowrank), UInt32(Toy38.plane), 0, 0))
        #expect(root.mixUp.shape == (UInt32(Toy38.plane), UInt32(Toy38.hcLowrank), 0, 0))

        // GDN layer 0 (linear attention) — the 3.6-shaped nine on the shallow
        // prefix.
        #expect(try model.gdnInProjQKV(layer: 0).shape == (UInt32(Toy38.qkvDim), UInt32(Toy38.D), 0, 0))
        #expect(try model.gdnInProjZ(layer: 0).shape == (UInt32(Toy38.valueDim), UInt32(Toy38.D), 0, 0))
        #expect(try model.gdnInProjA(layer: 0).shape == (UInt32(Toy38.linearValueHeads), UInt32(Toy38.D), 0, 0))
        #expect(try model.gdnOutProj(layer: 0).shape == (UInt32(Toy38.D), UInt32(Toy38.valueDim), 0, 0))
        #expect(try model.gdnNormWeight(layer: 0).shape == (UInt32(Toy38.linearValueHeadDim), 0, 0, 0))
        #expect(try model.gdnConv1D(layer: 0).shape == (UInt32(Toy38.qkvDim * Toy38.convKernel), 0, 0, 0))
        #expect(try model.gdnALog(layer: 0).shape == (UInt32(Toy38.linearValueHeads), 0, 0, 0))
        #expect(try model.gdnDtBias(layer: 0).shape == (UInt32(Toy38.linearValueHeads), 0, 0, 0))

        // Full layer 3: doubled-q self_attn + per-head norms + QSA indexer.
        #expect(try model.qProj(layer: 3).shape == (UInt32(2 * Toy38.numHeads * Toy38.fullHeadDim), UInt32(Toy38.D), 0, 0))
        #expect(try model.kProj(layer: 3).shape == (UInt32(Toy38.numFullKVHeads * Toy38.fullHeadDim), UInt32(Toy38.D), 0, 0))
        #expect(try model.oProj(layer: 3).shape == (UInt32(Toy38.D), UInt32(Toy38.numHeads * Toy38.fullHeadDim), 0, 0))
        #expect(try model.qNorm(layer: 3).shape == (UInt32(Toy38.fullHeadDim), 0, 0, 0))
        #expect(try model.kNorm(layer: 3).shape == (UInt32(Toy38.fullHeadDim), 0, 0, 0))
        let indexRows = (Toy38.indexerNumHeads + Toy38.indexerKVHeads) * Toy38.indexerHeadDim
        #expect(try model.indexerQKProj(layer: 3).shape == (UInt32(indexRows), UInt32(Toy38.D), 0, 0))
        #expect(try model.indexerQLayernorm(layer: 3).shape == (UInt32(Toy38.indexerHeadDim), 0, 0, 0))
        #expect(try model.indexerKLayernorm(layer: 3).shape == (UInt32(Toy38.indexerHeadDim), 0, 0, 0))

        // The 3.8 accessors guard: block norms and GDN lookups have no
        // entries under this family / on these layer kinds.
        #expect(throws: ModelError.self) { try model.inputNorm(layer: 0) }
        #expect(throws: ModelError.self) { try model.indexerQKProj(layer: 0) }   // GDN layer

        // PLE head (layer 1): head projections, grouped norms, squeezed fp16
        // conv, and the I64 hash metadata.
        #expect(try model.pleConv1D().shape == (UInt32(Toy38.plane * Toy38.pleConvKernel), 0, 0, 0))
        #expect(try model.pleKeyProj().shape == (UInt32(Toy38.plane), UInt32(Toy38.ngramWidth), 0, 0))
        #expect(try model.pleValueProj().shape == (UInt32(Toy38.D), UInt32(Toy38.ngramWidth), 0, 0))
        #expect(try model.pleNormQuery().shape == (UInt32(Toy38.plane), 0, 0, 0))
        #expect(try model.pleNormKey().shape == (UInt32(Toy38.plane), 0, 0, 0))
        #expect(try model.pleNormConv().shape == (UInt32(Toy38.plane), 0, 0, 0))
        #expect(try model.pleLayerMultipliers().shape == (UInt32(Toy38.ngramSize), 0, 0, 0))
        #expect(try model.pleHeadsOffsets().shape == (UInt32(Toy38.pleHeads), 0, 0, 0))
        #expect(try model.pleHeadsVocabSizes().shape == (UInt32(Toy38.pleHeads), 0, 0, 0))

        // Shared-expert + router + routed layout (family-independent).
        #expect(try model.sharedExpertGate(layer: 0).shape == (UInt32(Toy38.intermediate), UInt32(Toy38.D), 0, 0))
        #expect(try model.sharedExpertGateProj(layer: 0).shape == (1, UInt32(Toy38.D), 0, 0))
        #expect(try model.router(layer: 0).shape == (UInt32(Toy38.experts), UInt32(Toy38.D), 0, 0))
        #expect(model.packedExpertsLayout.expertsPerLayer == Toy38.experts)
        #expect(model.plePartOpenCount() == 0)      // nothing opened yet
    }

    // MARK: - PLE part streaming

    @Test func plePartsStreamVerifiedRows() async throws {
        let model = try await Self.loadToy38()

        let part = try model.openPLEPart(2)
        #expect(part.partIndex == 2)
        #expect(part.rows == Toy38.ngramPartRows)
        #expect(part.columns == Toy38.ngramRowDim)
        #expect(part.byteStride == Toy38.ngramRowDim * 2)
        #expect(part.sizeBytes == UInt64(Toy38.ngramPartRows * Toy38.ngramRowDim * 2))
        #expect(model.plePartOpenCount() == 1)

        // Cached: a second open returns the same streamer (no re-open).
        #expect(try model.openPLEPart(2) === part)
        #expect(model.plePartOpenCount() == 1)

        // Whole-part read is byte-exact against the checkpoint slice: the
        // repack copies parts raw, and the streamer must reproduce them.
        let all = try part.readRows(0..<Toy38.ngramPartRows)
        #expect(all.count == Toy38.ngramPartRows * part.byteStride)
        let source = Toy38.bf16Bytes(
            count: Toy38.ngramPartRows * Toy38.ngramRowDim,
            seed: 0x51A9 &+ UInt64(
                "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight"
                    .count) &* 7919)
        #expect(all == source)

        // Range reads are contiguous slices of the whole-part buffer.
        let head = try part.readRows(0..<17)
        let tail = try part.readRows(17..<Toy38.ngramPartRows)
        #expect(head + tail == all)
        let middle = try part.readRows(5..<9)
        let expected = all.subdata(in: (5 * part.byteStride)..<(9 * part.byteStride))
        #expect(middle == expected)
        let lastRow = try part.readRows((Toy38.ngramPartRows - 1)..<Toy38.ngramPartRows)
        #expect(lastRow.count == part.byteStride)
        #expect(try part.readRows(3..<3).isEmpty)

        // Opening the other parts lazily works and counts them.
        _ = try model.openPLEPart(0)
        _ = try model.openPLEPart(3)
        #expect(model.plePartOpenCount() == 3)
    }

    @Test func plePartsRejectInvalidRanges() async throws {
        let model = try await Self.loadToy38()
        let part = try model.openPLEPart(1)

        // Row ranges outside 0..<rows throw the streamer's range error.
        #expect(throws: StreamerError.self) { try part.readRows(31..<33) }
        #expect(throws: StreamerError.self) { try part.readRows(-1..<2) }

        // Part indexes outside the shard set throw ModelError at the model.
        #expect(throws: ModelError.self) { try model.openPLEPart(4) }
        #expect(throws: ModelError.self) { try model.openPLEPart(-1) }
        #expect(model.plePartOpenCount() == 1)
    }

    /// The hash constants come back from the resident tensors byte-exact, and
    /// the host router turns *this install's* constants into *this install's*
    /// rows.
    ///
    /// Two things the other PLE tests cannot see. The constants are int64 read
    /// out of a raw resident buffer: a byte-swapped or misaligned read still
    /// lands in range and still routes *some* row, so every downstream check
    /// would pass while the engine read a different table — the round-trip is
    /// only provable against the values the checkpoint wrote. And the rows are
    /// checked against `PLERef` (the locked `qwen4exp.cpp` oracle) fed from the
    /// manifest rather than from the test's own copy of the fixture, which is
    /// what ties the gather to the bytes on disk at those rows.
    @Test func pleHashMetadataAndGatherMatchTheCheckpoint() async throws {
        let directory = try await Self.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: directory) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: URL(fileURLWithPath: directory),
                                   device: device,
                                   expecting: Toy38.arch)

        // The three int64 vectors, exactly as `Toy38.pleHashMeta` wrote them.
        let half = UInt64(Toy38.ngramPartCount * Toy38.ngramPartRows / 2)
        let wantMultipliers: [UInt64] = (0..<Toy38.ngramSize).map { UInt64(2 * $0 + 3) }
        let wantOffsets: [UInt64] = [0, 0, half, half]
        let wantVocabs: [UInt64] = Array(repeating: half, count: Toy38.pleHeads)
        let meta = try model.pleHashConstants()
        #expect(meta.multipliers == wantMultipliers)
        #expect(meta.headOffsets == wantOffsets)
        #expect(meta.headVocabSizes == wantVocabs)

        // Position 0 has no predecessors, so its window is the token plus a
        // sticky EOS on both sides.
        let host = try #require(PLEHost(config: Toy38.arch,
                                        multipliers: meta.multipliers,
                                        headOffsets: meta.headOffsets,
                                        headVocabSizes: meta.headVocabSizes))
        host.record(position: 0, token: 7)
        let eos = Int32(Toy38.pleEosTokenId)
        let rows = host.rowIndices(atPosition: 0)
        let wantRows = PLERef.rowIndices(context: [7, eos, eos],
                                         multipliers: meta.multipliers,
                                         vocabSizes: meta.headVocabSizes,
                                         offsets: meta.headOffsets,
                                         headsPerNGram: Toy38.headsPerNgram)
        #expect(rows == wantRows, "rows \(rows) vs \(wantRows)")

        // Every gathered value is the byte pair on disk at that row: the part
        // split, the row offset and the BF16→FP16 widening, all at once.
        let gathered = try host.gather(atPosition: 0) { try model.openPLEPart($0) }
        #expect(gathered.count == Toy38.ngramWidth)

        var expected = [Float16](repeating: 0, count: gathered.count)
        let rowDim = Toy38.ngramRowDim
        for (h, row) in rows.enumerated() {
            let (part, rowInPart) = host.location(ofRow: row)
            let path = directory + String(format: "/ple_shards/shard_%03d.bin", part)
            let raw = try Data(contentsOf: URL(fileURLWithPath: path))
            let base = rowInPart * rowDim * 2
            for d in 0..<rowDim {
                let lo = UInt16(raw[base + 2 * d])
                let hi = UInt16(raw[base + 2 * d + 1])
                let value = Float16(Quantization.bf16ToFloat(lo | (hi << 8)))
                expected[h * rowDim + d] = value
            }
        }

        var mismatch = -1
        for i in 0..<expected.count where gathered[i] != expected[i] {
            mismatch = i
            break
        }
        if mismatch >= 0 {
            let head = mismatch / rowDim
            let dim = mismatch % rowDim
            Issue.record("gathered head \(head) dim \(dim) = \(gathered[mismatch]), but row \(rows[head]) holds \(expected[mismatch])")
        }
    }

    // MARK: - Load/stream negatives against the install on disk

    @Test func missingPartFileSurvivesLoadButFailsFirstOpen() async throws {
        let out = try await Self.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        try FileManager.default.removeItem(
            atPath: out + "/ple_shards/shard_000.bin")

        // Schema validation is metadata-only: load still succeeds.
        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: URL(fileURLWithPath: out),
                                   device: device,
                                   expecting: Toy38.arch)
        #expect(model.plePartOpenCount() == 0)
        // The physical absence surfaces at first touch.
        #expect(throws: ModelError.self) { try model.openPLEPart(0) }
    }

    @Test func truncatedPartFileFailsOnFirstOpen() async throws {
        let out = try await Self.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        // One byte short of the manifest size; the schema (metadata) passes.
        let partPath = out + "/ple_shards/shard_001.bin"
        let data = try Data(contentsOf: URL(fileURLWithPath: partPath))
        try data.prefix(data.count - 1).write(to: URL(fileURLWithPath: partPath))

        let device = try #require(MTLCreateSystemDefaultDevice())
        let model = try Model.load(directoryURL: URL(fileURLWithPath: out),
                                   device: device,
                                   expecting: Toy38.arch)
        #expect(throws: ModelError.self) { try model.openPLEPart(1) }
        #expect(model.plePartOpenCount() == 0)
    }

    @Test func corruptedPartManifestSizeRejectedAtLoad() async throws {
        let out = try await Self.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        // Patch the manifest's recorded size for one shard; validateRuntimeSchema
        // checks every ple_shards entry against ngramPartRows × 160 × 2.
        let manifestPath = out + "/manifest.json"
        var json = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: manifestPath))) as! [String: Any]
        var files = json["files"] as! [String: Any]
        var entry = files["ple_shards/shard_002.bin"] as! [String: Any]
        entry["size"] = (entry["size"] as! UInt64) + 1
        files["ple_shards/shard_002.bin"] = entry
        json["files"] = files
        let patched = try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
        try patched.write(to: URL(fileURLWithPath: manifestPath))

        let device = try #require(MTLCreateSystemDefaultDevice())
        #expect(throws: ModelError.self) {
            try Model.load(directoryURL: URL(fileURLWithPath: out),
                           device: device,
                           expecting: Toy38.arch)
        }
    }

    // MARK: - Runtime-schema negatives (validateQwen38Layers paths)

    @Test func rejectsMissingRootMixerTensor() async throws {
        let model = try await Self.loadToy38()
        var entries = model.residentIndex.entries
        entries.removeValue(forKey: "language_model.hyper_connection_mixer.hc_norm.weight")
        let index = ResidentIndex(header: model.residentIndex.header, entries: entries)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: Toy38.arch)
        }
    }

    @Test func rejectsCorruptPleMetadataAndAffineEntries() async throws {
        let model = try await Self.loadToy38()
        let entries = model.residentIndex.entries

        // I64 hash metadata: wrong count, wrong dtype, and misaligned sizes
        // must all fail the requireInt64 path.
        let offsets = try #require(entries["language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        let invalid = [
            copy(offsets, sizeBytes: offsets.sizeBytes + 8),        // count+1
            copy(offsets, dtype: FinchFormatV1.DType.bf16.rawValue), // wrong dtype
            copy(offsets, shape: (UInt32(Toy38.pleHeads - 1), 0, 0, 0)),
            copy(offsets, scaleSize: 2),                            // aux must be absent
        ]
        for entry in invalid {
            var mutated = entries
            mutated["language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"] = entry
            let index = ResidentIndex(header: model.residentIndex.header, entries: mutated)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: index,
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: Toy38.arch)
            }
        }

        // Per-layer hyper-connection BF16 gate: wrong dtype / wrong count.
        let hcNorm = try #require(entries["language_model.layers.0.attn_hyper_connection.hc_norm.weight"])
        for entry in [
            copy(hcNorm, dtype: FinchFormatV1.DType.fp32.rawValue),
            copy(hcNorm, sizeBytes: hcNorm.sizeBytes + 2),
            copy(hcNorm, shape: (UInt32(Toy38.plane - 1), 0, 0, 0)),
        ] {
            var mutated = entries
            mutated["language_model.layers.0.attn_hyper_connection.hc_norm.weight"] = entry
            let index = ResidentIndex(header: model.residentIndex.header, entries: mutated)
            #expect(throws: ModelError.self) {
                try Model.validateRuntimeSchema(
                    residentIndex: index,
                    layout: model.packedExpertsLayout,
                    manifest: model.manifest,
                    config: Toy38.arch)
            }
        }

        // PLE head projection affine: missing scale/bias must fail.
        let keyProj = try #require(entries["language_model.layers.1.ple.key_proj.weight"])
        var mutated = entries
        mutated["language_model.layers.1.ple.key_proj.weight"] = copy(keyProj, scaleSize: 0)
        let index = ResidentIndex(header: model.residentIndex.header, entries: mutated)
        #expect(throws: ModelError.self) {
            try Model.validateRuntimeSchema(
                residentIndex: index,
                layout: model.packedExpertsLayout,
                manifest: model.manifest,
                config: Toy38.arch)
        }
    }

    private func copy(
        _ entry: ResidentIndexEntry,
        dtype: UInt8? = nil,
        sizeBytes: UInt64? = nil,
        shape: (UInt32, UInt32, UInt32, UInt32)? = nil,
        scaleSize: UInt64? = nil
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: entry.name,
            dtype: dtype ?? entry.dtype,
            fileOffset: entry.fileOffset,
            sizeBytes: sizeBytes ?? entry.sizeBytes,
            shape: shape ?? entry.shape,
            scaleOffset: entry.scaleOffset,
            scaleSize: scaleSize ?? entry.scaleSize,
            biasOffset: entry.biasOffset,
            biasSize: entry.biasSize)
    }
}

/// M3.1c decode-wiring gate: runs one decode step (position 0 — no prefill is
/// required for the first token) through the REAL hybrid decode dispatch on
/// the toy 3.8 install with the production kernels and the `qwenLayerDebugHook`,
/// then checks the hyper-connection mechanics end to end:
///   * plane init — hc.pre at layer 0 is hc exact copies of the embedding;
///   * attn/ffn mixer stages — hc.mid/attnBlockOut/ffnBlockIn snapshots land
///     and the attn combine mutated the plane (hc.mid ≠ hc.pre);
///   * cross-layer persistence — hc.pre of layer L+1 == hc.post of layer L
///     (the deferred-tail ffn combine is the only plane writer between);
///   * the GDN sigmoid-gated body (layers 0-2) and the full-attention body
///     (layer 3) both stay finite through the routed tail;
///   * the head collapses the root hyper_connection_mixer over the plane and
///     lm_head produces finite logits.
/// A second test runs the M3.4 chunked prefill path and holds it against the
/// same tokens decoded one step at a time.
@Suite struct Qwen38DecodeWiringTests {

    private static let hc = 4

    static func makeRunner(_ model: Model) throws -> RealForwardRunner {
        let context = try MetalContext()
        return try RealForwardRunner(model: model, context: context,
                                     maxContext: 256)
    }

    @Test func decodeStepWiring() async throws {
        let model = try await Qwen38EngineLoadTests.loadToy38()
        let D = Qwen38EngineLoadTests.Toy38.D
        let hcDim = D * Self.hc
        let vocab = Qwen38EngineLoadTests.Toy38.vocab
        let valueDim = Qwen38EngineLoadTests.Toy38.valueDim   // 256
        let qkvDim = Qwen38EngineLoadTests.Toy38.qkvDim       // 512
        let numV = Qwen38EngineLoadTests.Toy38.linearValueHeads

        let runner = try Self.makeRunner(model)
        var captured: [String: [Float16]] = [:]
        runner.qwenLayerDebugHook = { L, name, values in
            captured["\(L)|\(name)"] = values
        }
        defer { runner.qwenLayerDebugHook = nil }

        guard let logits = model.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            Issue.record("logits alloc failed"); return
        }
        try await runner.produce(token: 7, position: 0, into: logits)

        func snap(_ layer: Int, _ name: String) -> [Float16]? {
            captured["\(layer)|\(name)"]
        }

        // --- Layer 0 (GDN): plane init == hc copies of the embedding. ---
        let preL0 = try #require(snap(0, "preLayer"), "L0 preLayer hook")
        let planeL0 = try #require(snap(0, "hc.pre"), "L0 hc.pre hook")
        #expect(preL0.count == D && planeL0.count == hcDim,
                "L0 snapshot sizes: preLayer \(preL0.count) plane \(planeL0.count)")
        var planeIsReplicatedEmbedding = true
        for c in 0..<Self.hc where planeIsReplicatedEmbedding {
            for i in 0..<D where planeIsReplicatedEmbedding {
                if planeL0[c * D + i] != preL0[i] {
                    planeIsReplicatedEmbedding = false
                }
            }
        }
        #expect(planeIsReplicatedEmbedding,
                "hc.pre(L0) must be hc exact copies of the embedding")

        // --- Mixer stages land with the right widths. ---
        for (name, count) in [("attnBlockIn", D), ("attnBlockOut", D),
                              ("ffnBlockIn", D)] {
            #expect(snap(0, name)?.count == count, "L0 \(name) snapshot")
        }
        #expect(snap(0, "hc.mid")?.count == hcDim, "L0 hc.mid snapshot")
        #expect(snap(0, "qkvConv")?.count == qkvDim, "L0 qkvConv snapshot")
        #expect(snap(0, "recurrentOut")?.count == valueDim,
                "L0 recurrentOut snapshot")
        #expect(snap(0, "gFloat")?.count == 2 * numV, "L0 gFloat snapshot")
        #expect(snap(0, "recState")?.count
                == numV * Qwen38EngineLoadTests.Toy38.linearValueHeadDim
                * Qwen38EngineLoadTests.Toy38.linearValueHeadDim,
                "L0 recState snapshot")

        // The attn combine scattered oOut into the plane (hc.mid reads the
        // plane between the two combines of layer 0).
        let midL0 = try #require(snap(0, "hc.mid"), "L0 hc.mid hook")
        #expect(zip(planeL0, midL0).contains(where: { $0 != $1 }),
                "attn combine must mutate the plane (hc.mid ≠ hc.pre)")

        // --- Cross-layer plane persistence: tail(L) == pre(L+1), exactly. ---
        for L in 0..<3 {
            let post = try #require(snap(L, "hc.post"), "L\(L) hc.post hook")
            let nextPre = try #require(snap(L + 1, "hc.pre"),
                                       "L\(L+1) hc.pre hook")
            #expect(post == nextPre,
                    "hc.pre(L+1) must equal hc.post(L): plane persists across the deferred tail")
        }

        // --- Full layer 3: dense body + per-head snapshots stay finite. ---
        let fullLayer = 3
        let qRows = Qwen38EngineLoadTests.Toy38.numHeads
            * Qwen38EngineLoadTests.Toy38.fullHeadDim
        for (name, count) in [("attnBlockIn", D), ("attnBlockOut", D),
                              ("ffnBlockIn", D)] {
            #expect(snap(fullLayer, name)?.count == count,
                    "L\(fullLayer) \(name) snapshot")
        }
        #expect(snap(fullLayer, "qOutF")?.count == qRows,
                "L3 qOutF snapshot")
        #expect(snap(fullLayer, "gateF")?.count == qRows,
                "L3 gateF snapshot")
        // kF/vF land when an FP16 KV cache exists for the hybrid decode.

        // --- Everything the layer loop and the routed tail wrote is finite. ---
        var finiteCount = 0
        for (key, values) in captured where !values.isEmpty {
            if values.allSatisfy({ $0.isFinite }) { finiteCount += 1 } else {
                Issue.record("non-finite snapshot at \(key)")
            }
        }
        #expect(finiteCount == captured.count,
                "all \(captured.count) snapshots must be finite")

        // --- Root-mixer head: logits finite, argmax in range. ---
        let lPtr = logits.contents().bindMemory(to: Float16.self,
                                                capacity: vocab)
        let lLogits = Array(UnsafeBufferPointer(start: lPtr, count: vocab))
        #expect(lLogits.allSatisfy { $0.isFinite }, "head logits finite")
        guard let best = lLogits.indices.max(by: { lLogits[$0] < lLogits[$1] })
        else {
            Issue.record("logits empty"); return
        }
        #expect((0..<vocab).contains(best), "argmax \(best) in vocab range")
    }

    /// M3.2d: once the context outgrows the indexer's selection width the
    /// full layer must attend through the indexer's cell list rather than the
    /// whole timeline, and that list must be a causal, ascending, exactly-
    /// `capacity`-sized subset of what the dense path would have read.
    ///
    /// This is the only test that runs the selector through the real runner —
    /// the kernel suites feed it buffers directly, and the toy's budget is
    /// chosen small precisely so a short decode crosses the width.
    @Test func sparseDecodeSelectsIndexerCells() async throws {
        let model = try await Qwen38EngineLoadTests.loadToy38()
        let runner = try Self.makeRunner(model)
        let vocab = Qwen38EngineLoadTests.Toy38.vocab
        let fullLayer = 3, r = Qwen38EngineLoadTests.Toy38.indexerCompressRatio
        let budget = Qwen38EngineLoadTests.Toy38.indexerBudget
        // The runner's maxContext caps the width the same way the state does.
        let capacity = min(256, budget + r - 1)

        guard let logits = model.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            Issue.record("logits alloc failed"); return
        }

        let steps = capacity + 6
        for p in 0..<steps {
            try await runner.produce(token: Int32(1 + p % 7), position: p, into: logits)
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocab)
            let row = Array(UnsafeBufferPointer(start: ptr, count: vocab))
            #expect(row.allSatisfy { $0.isFinite }, "step \(p): logits finite")
        }

        let pos = steps - 1
        let sel = try #require(runner.qsaSelection(layer: fullLayer),
                               "the toy's full layer must carry an indexer")
        // Block `b` is pooled on the step that fills it, so after `pos` the
        // timeline holds every block that is complete.
        #expect(sel.pooledBlocks == (pos + 1) / r,
                "pooled \(sel.pooledBlocks) blocks at pos \(pos), expected \((pos + 1) / r)")

        // Past the width the selection fills its capacity exactly.
        #expect(sel.cells.count == capacity,
                "selected \(sel.cells.count) cells, expected the full width \(capacity)")
        #expect(sel.cells == sel.cells.sorted(), "the cell list must ascend")
        #expect(Set(sel.cells).count == sel.cells.count, "no cell selected twice")
        #expect(sel.cells.allSatisfy { Int($0) <= pos },
                "selection must stay causal — nothing past \(pos)")
        // This step ends inside an incomplete tail block (steps − 1 is not a
        // multiple of r), so the query's own cell is force-visible and is the
        // largest one selected. That is the bias arm doing its job.
        #expect(pos % r != r - 1, "case built around the query NOT ending a block")
        #expect(sel.cells.last == UInt32(pos),
                "the query's own cell must survive the selection")
    }

    /// M3.4: the chunked prefill path, held against decode over the same
    /// tokens — and against itself across chunk boundaries.
    ///
    /// Twelve tokens is the shortest chunk the runner can be asked for and the
    /// most demanding one: below 32 tokens every projection in the chunk takes
    /// the repeated-GEMV path, which *is* decode's `int4.encode`, so a chunk
    /// and twelve sequential decode steps run the same kernels over the same
    /// values. That is what makes this a wiring test rather than a tolerance
    /// test — a wrong buffer, offset or stage order moves the logits far beyond
    /// the fp16 rounding the two paths share. It has already earned its keep:
    /// the head passed its `planeOffset` in elements where `setBuffer(_:offset:)`
    /// wants bytes, so every prefill with `t > 1` read the wrong plane row. The
    /// tell was not noise but structure — the logits changed with the boundary.
    ///
    /// Twelve also straddles the toy's indexer capacity (11): rows 0…10 attend
    /// the whole timeline through `encodeFull`, row 11 attends through the
    /// indexer's selected cell list. Both branches of the 3.8 attention inside
    /// one chunk, each against the matching decode step.
    ///
    /// Three tiers, because bitwise logits are not on offer here:
    ///
    ///  1. **Chunking invariance** — one 12-token span, two 6-token spans and
    ///     twelve 1-token spans must land on identical stages and identical
    ///     logits. The strongest statement available, and it needs no
    ///     reference: the chunk path is a pure function of the tokens and the
    ///     state they build, so the boundary must not be observable.
    ///  2. **Shared arithmetic** — at layer 0, every stage the chunk and a
    ///     decode step compute the same way must match bit for bit.
    ///  3. **Whole-model agreement** — the logits against twelve decode steps,
    ///     via cosine. Exact equality is not achievable: the chunk's routed
    ///     reduce takes fp16 `routePartials` and adds the shared residual in a
    ///     separate fp16 rounding, where decode's `moe_phase2_down_reduce_k8`
    ///     keeps each per-slot value fp32, seeds the accumulator with the
    ///     residual and rounds once. That is structural, not a wiring bug, and
    ///     at layer 0 it costs 1 fp16 ulp of the row maximum (`mlpBlockIn`
    ///     nDiff 28/64); four layers of this toy amplify it to ~6%.
    ///
    /// For calibration, in the final logits: correct is ≈ 0.98 cosine, and both
    /// a chunking-only perturbation and the mis-wired head sat at ≈ 0. Nothing
    /// in the logits separates chaos from a wiring bug — which is why tier 1
    /// carries the weight.
    @Test func prefillChunkMatchesDecodeSteps() async throws {
        let model = try await Qwen38EngineLoadTests.loadToy38()
        let vocab = Qwen38EngineLoadTests.Toy38.vocab
        let fullLayer = 3
        let r = Qwen38EngineLoadTests.Toy38.indexerCompressRatio
        let budget = Qwen38EngineLoadTests.Toy38.indexerBudget
        // The runner's maxContext caps the width the same way the state does.
        let capacity = min(256, budget + r - 1)
        let chunk = capacity + 1
        let tokens = (0..<chunk).map { Int32(3 + $0) }

        let read: (MTLBuffer) -> [Float16] = { buf in
            let ptr = buf.contents().bindMemory(to: Float16.self, capacity: vocab)
            return Array(UnsafeBufferPointer(start: ptr, count: vocab))
        }

        /// One scenario: a fresh runner carrying the stage hook and a logits
        /// buffer of its own, returned with everything it recorded.
        func scenario(_ body: (RealForwardRunner, MTLBuffer) async throws -> Int)
            async throws -> (runner: RealForwardRunner, stages: [String: [Float16]],
                             logits: [Float16], newPosition: Int) {
            let runner = try Self.makeRunner(model)
            var stages: [String: [Float16]] = [:]
            runner.qwenLayerDebugHook = { L, name, values in
                stages["\(L)|\(name)"] = values
            }
            let logits = try #require(model.device.makeBuffer(
                length: vocab * MemoryLayout<Float16>.size,
                options: .storageModeShared))
            let newPosition = try await body(runner, logits)
            runner.qwenLayerDebugHook = nil
            return (runner, stages, read(logits), newPosition)
        }

        /// The chunked path over `spans`, in order.
        func prefill(_ spans: [[Int32]]) async throws
            -> (runner: RealForwardRunner, stages: [String: [Float16]],
                logits: [Float16], newPosition: Int) {
            try await scenario { runner, logits in
                var position = 0
                var advanced = 0
                for span in spans {
                    let result = try await runner.prefillChunked(
                        tokens: span[...], startPosition: position,
                        outputMode: .logits, config: .defaultChunked,
                        into: logits, onProgress: { _ in })
                    position += span.count
                    advanced = result.newPosition
                }
                return advanced
            }
        }

        /// The first element two stage sets disagree on, or nil.
        func firstDivergence(_ a: [String: [Float16]],
                             _ b: [String: [Float16]]) -> String? {
            guard a.count == b.count else { return "\(a.count) stages vs \(b.count)" }
            for key in a.keys.sorted() {
                guard let x = a[key], let y = b[key] else { return "\(key) on one side only" }
                guard x.count == y.count else {
                    return "\(key): \(x.count) elements vs \(y.count)"
                }
                for i in 0..<x.count where x[i] != y[i] {
                    return "\(key)[\(i)]: \(x[i]) vs \(y[i])"
                }
            }
            return nil
        }

        /// How far `b` is from `a`, in fp16 ulps of `a`'s largest magnitude.
        func worst(_ a: [Float16], _ b: [Float16])
            -> (diff: Float, maxAbs: Float, ulps: Float) {
            var diff: Float = 0
            var maxAbs: Float = 0
            for i in 0..<min(a.count, b.count) {
                diff = max(diff, abs(Float(a[i]) - Float(b[i])))
                maxAbs = max(maxAbs, abs(Float(a[i])))
            }
            let ulp = maxAbs > 0 ? Float(Float16(maxAbs).ulp) : 1
            return (diff, maxAbs, diff / ulp)
        }
        func describe(_ a: [Float16], _ b: [Float16]) -> String {
            let w = worst(a, b)
            return "\(w.diff) apart at max |x| = \(w.maxAbs) (\(w.ulps) fp16 ulps)"
        }

        // The reference: the same tokens one step at a time, on a runner of its
        // own so both paths start from the same empty state.
        let decode = try await scenario { runner, logits in
            for p in 0..<chunk {
                try await runner.produce(token: tokens[p], position: p, into: logits)
            }
            return chunk
        }

        // The whole run as a single span. `tokens.count` sets the chunk width
        // here, not the config's 32-token ceiling — which is what keeps this
        // on the GEMV path and comparable to decode.
        let whole = try await prefill([tokens])
        #expect(whole.newPosition == chunk,
                "prefill advanced to \(whole.newPosition), expected \(chunk)")

        // MARK: Tier 1 — the chunk path cannot see where the spans fell

        let halves = try await prefill([Array(tokens[0..<(chunk / 2)]),
                                        Array(tokens[(chunk / 2)...])])
        let singles = try await prefill(tokens.map { [$0] })
        for (label, split) in [("6+6", halves), ("12×1", singles)] {
            let stages = firstDivergence(whole.stages, split.stages)
            #expect(stages == nil,
                    "\(label) must leave every stage bit-identical to one \(chunk)-token chunk; first divergence: \(stages ?? "")")
            let logits = worst(whole.logits, split.logits)
            #expect(whole.logits == split.logits,
                    "\(label) logits differ from one \(chunk)-token chunk by \(logits.diff) at max |x| = \(logits.maxAbs)")
        }

        // The same span once more, on a runner of its own — an independent
        // context with its own buffers must reproduce `whole` exactly. This
        // is the guard against a chunk that reads state it never wrote: an
        // unwritten row, or a scratch region that survives from an earlier
        // allocation, surfaces here as a rerun that disagrees with itself
        // instead of as a mystery cosine four tiers down.
        let rerun = try await prefill([tokens])
        let rerunStages = firstDivergence(whole.stages, rerun.stages)
        #expect(rerunStages == nil,
                "a rerun of the same span must be bit-identical; first divergence: \(rerunStages ?? "")")
        #expect(whole.logits == rerun.logits,
                "a rerun of the same span must produce the same logits: \(worst(whole.logits, rerun.logits))")

        // MARK: Tier 2 — the arithmetic the two paths share

        // Layer 0 is the only layer still comparable: every stage below it is
        // the same kernel over the same values, and every stage above it has
        // the routed reduce's one ulp baked in. `qkvProjected` is absent from
        // the decode side — the chunk snapshots the fused pre-conv projection,
        // which decode never materializes.
        let shared = ["hc.pre", "attnBlockIn", "qkvConv", "recurrentOut",
                      "recState", "attnBlockOut", "hc.mid", "ffnBlockIn", "sharedOut"]
        for stage in shared {
            let key = "0|\(stage)"
            let want = try #require(decode.stages[key], "decode never reached \(key)")
            let have = try #require(whole.stages[key], "chunk never reached \(key)")
            #expect(want.count == have.count,
                    "\(key): decode \(want.count) elements, chunk \(have.count)")
            #expect(want == have, "\(key) must be bit-exact: \(describe(want, have))")
        }

        // `gFloat` is the one shared stage the two paths snapshot at different
        // widths: decode records the fused pair, `[g | beta]` at 2·numV, and
        // the chunk records `g` alone at numV. Every element the chunk writes
        // is decode's elementwise twin, so the chunk's whole run must be
        // bit-exact as decode's prefix.
        do {
            let key = "0|gFloat"
            let want = try #require(decode.stages[key], "decode never reached \(key)")
            let have = try #require(whole.stages[key], "chunk never reached \(key)")
            let lead = Array(want.prefix(have.count))
            #expect(have.count <= want.count,
                    "\(key): the chunk recorded \(have.count) elements, past decode's \(want.count)")
            #expect(lead == have,
                    "\(key) must be bit-exact over the chunk's \(have.count): \(describe(lead, have))")
        }

        // The routed reduce moves exactly two layer-0 stages — `mlpBlockIn` is
        // shared + routed, `hc.post` is the plane after it lands — and it moves
        // them by one ulp of the row maximum. A mis-routed expert, a wrong tile
        // offset or a dropped term is O(1) relative, so a four-ulp bound of the
        // row maximum still fails every wiring defect while admitting the fp16
        // partials the chunk's reduce is built on.
        for stage in ["mlpBlockIn", "hc.post"] {
            let key = "0|\(stage)"
            let want = try #require(decode.stages[key], "decode never reached \(key)")
            let have = try #require(whole.stages[key], "chunk never reached \(key)")
            let w = worst(want, have)
            #expect(w.ulps <= 4,
                    "\(key) is \(w.ulps) fp16 ulps from decode — \(describe(want, have))")
        }

        // MARK: Tier 3 — the whole model against decode

        #expect(whole.logits.allSatisfy { $0.isFinite }, "chunked logits finite")
        #expect(decode.logits.allSatisfy { $0.isFinite }, "decode logits finite")

        var maxDiff: Float = 0
        var argmaxGot = 0
        var argmaxRef = 0
        var dot = 0.0
        var normGot = 0.0
        var normRef = 0.0
        for i in 0..<vocab {
            let g = Double(whole.logits[i]), d = Double(decode.logits[i])
            maxDiff = max(maxDiff, abs(Float(g - d)))
            dot += g * d
            normGot += g * g
            normRef += d * d
            if whole.logits[i] > whole.logits[argmaxGot] { argmaxGot = i }
            if decode.logits[i] > decode.logits[argmaxRef] { argmaxRef = i }
        }
        let cosine = dot / (normGot.squareRoot() * normRef.squareRoot())
        let rmsRef = Float((normRef / Double(vocab)).squareRoot())
        #expect(cosine > 0.9,
                "chunked prefill is \(cosine) cosine from \(chunk) decode steps (max |Δ| = \(maxDiff), rms \(rmsRef))")
        #expect(argmaxGot == argmaxRef,
                "chunk argmax \(argmaxGot) vs decode argmax \(argmaxRef) at cosine \(cosine)")

        // MARK: Pooling — the one piece of the chunk's work that outlives it

        // The chunk's batched pool must leave the indexer exactly where twelve
        // steps of per-block pooling leave it.
        let chunkSel = try #require(whole.runner.qsaSelection(layer: fullLayer),
                                    "the toy's full layer must carry an indexer")
        let decodeSel = try #require(decode.runner.qsaSelection(layer: fullLayer),
                                     "the toy's full layer must carry an indexer")
        #expect(chunkSel.pooledBlocks == chunk / r,
                "chunk pooled \(chunkSel.pooledBlocks) blocks, expected \(chunk / r)")
        #expect(chunkSel.pooledBlocks == decodeSel.pooledBlocks,
                "chunk pooled \(chunkSel.pooledBlocks) blocks, decode pooled \(decodeSel.pooledBlocks)")
    }

    // MARK: - PLE decode wiring (M3.3)

    /// Decode token 7 at position 0 on two installs repacked from the same
    /// seed — byte-identical weights, byte-identical hash metadata — with one
    /// install's PLE parts overwritten by zeros. The table is then the only
    /// thing that differs between the runs, and the snapshots say exactly what
    /// it moved.
    ///
    /// The zeroed table is the isolation trick and it is what makes the two
    /// halves of this test work together. It is a provable no-op: every row
    /// reads as zero, so `key` and `value` are zero, the gate multiplies a
    /// zero vector, `silu(0)` is zero, and the plane gains exactly nothing —
    /// which means anything that differs between the runs differs *because of
    /// the table* and not because the PLE block ran at all. So the two
    /// assertions bracket the insertion point from both sides:
    ///   * strictly upstream (`hc.pre` is snapshotted at the top of the layer,
    ///     before anything is encoded) — bit-identical, including every
    ///     snapshot of layer 0;
    ///   * from the write itself (`hc.mid` is read after CB1 completes, so it
    ///     holds the PLE's plane add *and* the attn combine) — must differ.
    ///
    /// The second half is what stops a silently-unwired PLE from passing: if
    /// the block were skipped, both runs would be identical everywhere and
    /// only the "must differ" assertions would fail.
    ///
    /// Pooled layer 1 is a GDN layer here, so this exercises the linear-
    /// attention branch's call site; the full-attention branch has the
    /// identical `gPLE?(cb)` before its `gAttnMix`. One decode step at
    /// position 0 is enough — the hash window is `[7, EOS, EOS]` there, and
    /// the ring's deeper history is covered by `PLEHostTests`.
    @Test func pleTermsLandInTheLayerOnePlane() async throws {
        let real = try await Qwen38EngineLoadTests.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: real) }
        let zeroed = try await Qwen38EngineLoadTests.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: zeroed) }
        try Self.zeroPleShards(in: zeroed)

        let device = try #require(MTLCreateSystemDefaultDevice())
        let load: (String) throws -> Model = { directory in
            try Model.load(directoryURL: URL(fileURLWithPath: directory),
                           device: device,
                           expecting: Qwen38EngineLoadTests.Toy38.arch)
        }
        let modelReal = try load(real)
        let modelZeroed = try load(zeroed)
        let runnerReal = try Self.makeRunner(modelReal)
        let runnerZeroed = try Self.makeRunner(modelZeroed)

        var realSnaps: [String: [Float16]] = [:]
        var zeroSnaps: [String: [Float16]] = [:]
        runnerReal.qwenLayerDebugHook = { L, name, values in
            realSnaps["\(L)|\(name)"] = values
        }
        runnerZeroed.qwenLayerDebugHook = { L, name, values in
            zeroSnaps["\(L)|\(name)"] = values
        }
        defer {
            runnerReal.qwenLayerDebugHook = nil
            runnerZeroed.qwenLayerDebugHook = nil
        }

        let vocab = Qwen38EngineLoadTests.Toy38.vocab
        guard let logits = device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            Issue.record("logits alloc failed"); return
        }
        try await runnerReal.produce(token: 7, position: 0, into: logits)
        try await runnerZeroed.produce(token: 7, position: 0, into: logits)

        #expect(Set(realSnaps.keys) == Set(zeroSnaps.keys),
                "both runs must reach the same snapshots")
        #expect(realSnaps.values.allSatisfy { !$0.isEmpty },
                "no empty snapshots — an empty one would hide a difference")

        // The zeroed table's premise: the PLE contributes nothing. If a
        // grouped RMS of zeros were to divide by zero and poison the plane,
        // this is where it shows up, rather than as a fake "the table changed
        // something" pass downstream.
        for (key, values) in zeroSnaps {
            #expect(values.allSatisfy { $0.isFinite }, "zero-table snapshot \(key)")
        }

        // Upstream of the PLE, the two runs must agree bit for bit — the whole
        // of layer 0, and layer 1 up to the plane as it arrived.
        let upstream = ["0|preLayer", "0|hc.pre", "0|hc.mid", "0|hc.post",
                        "0|qkvConv", "0|recurrentOut", "0|gFloat",
                        "1|preLayer", "1|hc.pre"]
        for key in upstream {
            #expect(realSnaps[key] == zeroSnaps[key],
                    "\(key) is upstream of the PLE and must be bit-identical")
        }

        // From the plane write onward the table is what changed: the attn mix
        // reads the plane the PLE just added to, so `hc.mid`, the block input
        // derived from it, and everything after all move.
        let downstream = ["1|hc.mid", "1|attnBlockIn", "1|hc.post",
                          "2|hc.pre", "2|hc.mid"]
        for key in downstream {
            #expect(realSnaps[key] != zeroSnaps[key],
                    "\(key) is downstream of the PLE and must differ — identical here means the PLE never ran")
        }
    }

    /// Overwrite every PLE part file with zeros and re-point the manifest's
    /// SHA-256 at the new bytes. Sizes are unchanged, so the install still
    /// passes every structural check; the hash has to be updated because
    /// `openPLEPart` verifies a part the first time it is touched, and a
    /// zeroed table with a stale hash is rejected at load.
    private static func zeroPleShards(in directory: String) throws {
        let shardDir = directory + "/ple_shards"
        let expected = Qwen38EngineLoadTests.Toy38.ngramPartCount
        let shards = try FileManager.default.contentsOfDirectory(atPath: shardDir)
        guard shards.count == expected else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [
                NSLocalizedDescriptionKey:
                    "expected \(expected) PLE part files, found \(shards.count)"])
        }

        var digest = ""
        for name in shards.sorted() {
            let url = URL(fileURLWithPath: shardDir + "/" + name)
            let size = try Data(contentsOf: url).count
            let zeros = Data(count: size)
            try zeros.write(to: url)
            digest = Sha256Verifier.hashData(zeros)
        }

        let manifestURL = URL(fileURLWithPath: directory + "/manifest.json")
        guard var json = try JSONSerialization.jsonObject(
                with: Data(contentsOf: manifestURL)) as? [String: Any],
              var files = json["files"] as? [String: Any] else {
            throw CocoaError(.fileReadCorruptFile, userInfo: [
                NSLocalizedDescriptionKey: "manifest.json has no files map"])
        }
        for name in shards {
            let key = "ple_shards/" + name
            guard var entry = files[key] as? [String: Any] else {
                throw CocoaError(.fileReadCorruptFile, userInfo: [
                    NSLocalizedDescriptionKey: "manifest.json has no entry for \(key)"])
            }
            entry["sha256"] = digest
            files[key] = entry
        }
        json["files"] = files
        try JSONSerialization.data(withJSONObject: json, options: [.sortedKeys])
            .write(to: manifestURL)
    }
}

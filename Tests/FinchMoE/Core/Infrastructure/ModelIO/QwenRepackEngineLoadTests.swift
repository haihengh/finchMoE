import Testing
import Foundation
import Metal
import FinchMoERepackCore
import FinchMoEFormat
@testable import FinchMoE

/// End-to-end proof of the Qwen quantizing repack: a tiny synthetic bf16
/// checkpoint is repacked into `.finchturbo`, then the real runtime
/// `Model.load(expecting:)` loads it and `validateRuntimeSchema` +
/// `validateArch` pass against the matching toy preset. This is the gate the
/// real 35B install must pass before it can run.
@Suite struct QwenRepackEngineLoadTests {

    // MARK: - Toy model (mirrors the repack test's SyntheticQwenSnapshot)

    private enum Toy {
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
        static let fullMask: [UInt8] = [0, 0, 0, 1]
        static let keyDim = linearKeyHeads * linearKeyHeadDim
        static let valueDim = linearValueHeads * linearValueHeadDim
        static let qkvDim = 2 * keyDim + valueDim

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
            modelFamily: "qwen3_6",
            attnOutputGate: true,
            linearNumKeyHeads: linearKeyHeads,
            linearNumValueHeads: linearValueHeads,
            linearKeyHeadDim: linearKeyHeadDim,
            linearValueHeadDim: linearValueHeadDim,
            linearConvKernelDim: convKernel)

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
                    "layer_types": fullMask.map { $0 == 1 ? "full_attention" : "linear_attention" },
                    "rope_parameters": [
                        "rope_theta": 1_000_000.0,
                        "partial_rotary_factor": 0.25,
                    ],
                    "sliding_window": 0,
                    "final_logit_softcapping": 0.0,
                ],
            ]
        }

        static func tensorShapes() -> [(name: String, shape: [Int])] {
            var out: [(String, [Int])] = [
                ("model.language_model.embed_tokens.weight", [vocab, D]),
                ("model.language_model.norm.weight", [D]),
                ("lm_head.weight", [vocab, D]),
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

        static func writeSnapshot(into dir: String, seed: UInt64) throws {
            try FileManager.default.createDirectory(
                at: URL(fileURLWithPath: dir), withIntermediateDirectories: true)
            let configData = try JSONSerialization.data(
                withJSONObject: configJSON(), options: [.sortedKeys])
            try configData.write(to: URL(fileURLWithPath: dir).appendingPathComponent("config.json"))

            let shardName = "model-00001-of-00001.safetensors"
            var header: [String: Any] = [:]
            var weightMap: [String: String] = [:]
            var payload = Data()
            var cursor = 0
            var state = seed
            for (name, shape) in tensorShapes() {
                let elements = shape.reduce(1, *)
                var bytes = Data(capacity: elements * 2)
                for _ in 0..<elements {
                    state = state &* 6364136223846793005 &+ 1442695040888963407
                    let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
                    let value = -2.0 + 4.0 * fraction
                    let bits = FinchTurboQuantization.bf16Bits(value)
                    bytes.append(UInt8(truncatingIfNeeded: bits & 0xFF))
                    bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                }
                let start = cursor
                cursor += bytes.count
                header[name] = ["dtype": "BF16", "shape": shape,
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

    // MARK: - Tests

    @Test func quantizedToyInstallLoadsAndValidates() async throws {
        let base = NSTemporaryDirectory() + "qwen-engine-\(UUID().uuidString)"
        let src = base + "-src"
        try Toy.writeSnapshot(into: src, seed: 0x71EA)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let out = base + "-out"
        let options = LocalQwenRepackOptions(
            snapshotDir: src, outputDir: out, minFreeReserveBytes: 0)
        _ = try await LocalQwenRepacker(options: options).run()
        defer { try? FileManager.default.removeItem(atPath: out) }

        let ctx = try MetalContext()
        // The preset auto-detection: the manifest's qwen3_6 family picks the
        // built-in Qwen preset.
        let detected = try ManifestReader.detectPreset(
            directoryURL: URL(fileURLWithPath: out))
        // The manifest family selects the BUILT-IN preset (production dims),
        // not the toy dims.
        #expect(detected == .qwen3_6_35B_A3B)

        // The full runtime load: manifest + resident index decode, arch
        // cross-check against the toy preset, and validateRuntimeSchema.
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: out),
            device: ctx.device,
            expecting: Toy.arch)

        // Family-branching accessors resolve the repacked entries.
        let embed = model.embedding
        #expect(embed.shape == (UInt32(Toy.vocab), UInt32(Toy.D), 0, 0))
        #expect(model.lmHead.shape == (UInt32(Toy.vocab), UInt32(Toy.D), 0, 0))
        #expect(model.finalNorm.shape == (UInt32(Toy.D), 0, 0, 0))

        let qkv = try model.gdnInProjQKV(layer: 0)
        #expect(qkv.shape == (UInt32(Toy.qkvDim), UInt32(Toy.D), 0, 0))
        #expect(try model.gdnInProjZ(layer: 0).shape == (UInt32(Toy.valueDim), UInt32(Toy.D), 0, 0))
        #expect(try model.gdnInProjA(layer: 0).shape == (UInt32(Toy.linearValueHeads), UInt32(Toy.D), 0, 0))
        #expect(try model.gdnOutProj(layer: 0).shape == (UInt32(Toy.D), UInt32(Toy.valueDim), 0, 0))
        #expect(try model.gdnALog(layer: 0).shape == (UInt32(Toy.linearValueHeads), 0, 0, 0))
        #expect(try model.gdnDtBias(layer: 0).shape == (UInt32(Toy.linearValueHeads), 0, 0, 0))
        #expect(try model.gdnNormWeight(layer: 0).shape == (UInt32(Toy.linearValueHeadDim), 0, 0, 0))
        #expect(try model.gdnConv1D(layer: 0).shape == (UInt32(Toy.qkvDim * Toy.convKernel), 0, 0, 0))
        #expect(try model.sharedExpertGateProj(layer: 0).shape == (1, UInt32(Toy.D), 0, 0))

        #expect(try model.qProj(layer: 3).shape == (UInt32(2 * Toy.numHeads * Toy.fullHeadDim), UInt32(Toy.D), 0, 0))
        #expect(try model.kProj(layer: 3).shape == (UInt32(Toy.numFullKVHeads * Toy.fullHeadDim), UInt32(Toy.D), 0, 0))
        #expect(try model.qNorm(layer: 3).shape == (UInt32(Toy.fullHeadDim), 0, 0, 0))

        #expect(try model.router(layer: 0).shape == (UInt32(Toy.experts), UInt32(Toy.D), 0, 0))
        #expect(try model.sharedExpertGate(layer: 0).shape == (UInt32(Toy.intermediate), UInt32(Toy.D), 0, 0))
        #expect(try model.inputNorm(layer: 0).shape == (UInt32(Toy.D), 0, 0, 0))
        #expect(try model.postAttnNorm(layer: 0).shape == (UInt32(Toy.D), 0, 0, 0))

        // Routed experts: layout.json metadata validated at load; the blobs
        // open lazily via the streamer path.
        #expect(model.packedExpertsLayout.expertsPerLayer == Toy.experts)
    }
}

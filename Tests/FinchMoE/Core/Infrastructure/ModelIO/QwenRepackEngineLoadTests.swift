import Testing
import Foundation
import Metal
import FinchMoERepackCore
import FinchMoEFormat
import FinchMoECLICore
@testable import FinchMoE

/// End-to-end proof of the Qwen quantizing repack: a tiny synthetic bf16
/// checkpoint is repacked into `.finch`, then the real runtime
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

        /// The same toy `model.visual.*` tower the repack fixture writes:
        /// patch embed, position table, 12 tensors per block, merger. Only
        /// used by the vision-install load test; a text-only fixture keeps
        /// its original tensor stream byte for byte.
        static func visionTensorShapes() -> [(name: String, shape: [Int])] {
            let hidden = 32
            let layers = 2
            let mlp = 64
            var out: [(String, [Int])] = [
                ("model.visual.patch_embed.proj.weight", [hidden, 3, 2, 2, 2]),
                ("model.visual.patch_embed.proj.bias", [hidden]),
                ("model.visual.pos_embed.weight", [4, hidden]),
            ]
            for block in 0..<layers {
                let p = "model.visual.blocks.\(block)"
                out.append((p + ".norm1.weight", [hidden]))
                out.append((p + ".norm1.bias", [hidden]))
                out.append((p + ".norm2.weight", [hidden]))
                out.append((p + ".norm2.bias", [hidden]))
                out.append((p + ".attn.qkv.weight", [3 * hidden, hidden]))
                out.append((p + ".attn.qkv.bias", [3 * hidden]))
                out.append((p + ".attn.proj.weight", [hidden, hidden]))
                out.append((p + ".attn.proj.bias", [hidden]))
                out.append((p + ".mlp.linear_fc1.weight", [mlp, hidden]))
                out.append((p + ".mlp.linear_fc1.bias", [mlp]))
                out.append((p + ".mlp.linear_fc2.weight", [hidden, mlp]))
                out.append((p + ".mlp.linear_fc2.bias", [hidden]))
            }
            out.append(("model.visual.merger.norm.weight", [hidden]))
            out.append(("model.visual.merger.norm.bias", [hidden]))
            out.append(("model.visual.merger.linear_fc1.weight", [2 * hidden, 2 * hidden]))
            out.append(("model.visual.merger.linear_fc1.bias", [2 * hidden]))
            out.append(("model.visual.merger.linear_fc2.weight", [D, 2 * hidden]))
            out.append(("model.visual.merger.linear_fc2.bias", [D]))
            return out
        }

        static func tensorShapes(includeVision: Bool = false)
            -> [(name: String, shape: [Int])] {
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
            if includeVision {
                out.append(contentsOf: visionTensorShapes())
            }
            return out
        }

        static func writeSnapshot(into dir: String, seed: UInt64,
                                  includeVision: Bool = false) throws {
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
            for (name, shape) in tensorShapes(includeVision: includeVision) {
                let elements = shape.reduce(1, *)
                var bytes = Data(capacity: elements * 2)
                for _ in 0..<elements {
                    state = state &* 6364136223846793005 &+ 1442695040888963407
                    let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
                    let value = -2.0 + 4.0 * fraction
                    let bits = FinchQuantization.bf16Bits(value)
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
            if includeVision {
                try Data(#"{"patch_size":2,"temporal_patch_size":2,"merge_size":2}"#.utf8)
                    .write(to: URL(fileURLWithPath: dir)
                        .appendingPathComponent("preprocessor_config.json"))
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
        let finalNorm = try #require(model.finalNorm)   // qwen3_6: always present
        #expect(finalNorm.shape == (UInt32(Toy.D), 0, 0, 0))

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

    /// A vision-bearing install is still a valid text install: the extra
    /// `vision/` files are declared in the manifest and verified, but the
    /// loader opens only the text file set, and every text byte is identical
    /// to the text-only repack of the same snapshot. This is the compatibility
    /// claim the published `-vision` artifacts rest on until the runtime
    /// consumes the tower.
    @Test func visionToyInstallLoadsTextOnly() async throws {
        let base = NSTemporaryDirectory() + "qwen-engine-vision-\(UUID().uuidString)"
        let src = base + "-src"
        try Toy.writeSnapshot(into: src, seed: 0x71ED, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let textOut = base + "-text"
        _ = try await LocalQwenRepacker(
            options: LocalQwenRepackOptions(snapshotDir: src, outputDir: textOut,
                                            minFreeReserveBytes: 0)).run()
        defer { try? FileManager.default.removeItem(atPath: textOut) }

        let visionOut = base + "-vision"
        _ = try await LocalQwenRepacker(
            options: LocalQwenRepackOptions(snapshotDir: src, outputDir: visionOut,
                                            minFreeReserveBytes: 0,
                                            includeVision: true)).run()
        defer { try? FileManager.default.removeItem(atPath: visionOut) }

        let fm = FileManager.default
        #expect(fm.fileExists(atPath: visionOut + "/vision/vision.safetensors"))
        #expect(fm.fileExists(atPath: visionOut + "/vision/model_config.json"))
        #expect(fm.fileExists(atPath: visionOut + "/vision/preprocessor_config.json"))
        #expect(!fm.fileExists(atPath: textOut + "/vision"))

        // The text payloads are the same bytes: including the tower changes
        // no engine-visible file.
        for relative in ["model_weights.bin", "packed_experts/layer_00.bin"] {
            let text = try Data(contentsOf: URL(fileURLWithPath: textOut + "/" + relative))
            let vision = try Data(contentsOf: URL(fileURLWithPath: visionOut + "/" + relative))
            #expect(text == vision, "\(relative) differs when vision is included")
        }

        // Manifest + receipt cover the extras, and the verifier accepts them.
        let manifestData = try Data(contentsOf: URL(
            fileURLWithPath: visionOut + "/manifest.json"))
        let manifest = try FinchManifestCodec.decode(manifestData)
        #expect(manifest.files["vision/vision.safetensors"] != nil)
        #expect(manifest.files["vision/model_config.json"] != nil)
        #expect(manifest.files["vision/preprocessor_config.json"] != nil)
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputFinch: visionOut))
        #expect(verify.unexpectedEntries.isEmpty)

        // The real runtime load ignores the extras and resolves the text
        // entries exactly as it does for the text-only install.
        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: visionOut),
            device: ctx.device,
            expecting: Toy.arch)
        #expect(model.embedding.shape == (UInt32(Toy.vocab), UInt32(Toy.D), 0, 0))
        #expect(model.packedExpertsLayout.expertsPerLayer == Toy.experts)
    }

    /// The 3-bit twin of `quantizedToyInstallLoadsAndValidates`: repack the
    /// toy snapshot with `--routed-expert-bits 3`, then run the full runtime
    /// load. This is the round-trip that matters for the experiment — the
    /// manifest's `routedExpert.weightBits` and the layout's per-subtensor
    /// sizes/bits must agree with the 24-bit-triplet blobs the writer emitted
    /// (a mismatch here is how a 3-bit install would fail to load at all).
    @Test func threeBitToyInstallLoadsAndValidates() async throws {
        let base = NSTemporaryDirectory() + "qwen-engine-3bit-\(UUID().uuidString)"
        let src = base + "-src"
        try Toy.writeSnapshot(into: src, seed: 0x71EB)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let out = base + "-out"
        let options = LocalQwenRepackOptions(
            snapshotDir: src, outputDir: out,
            minFreeReserveBytes: 0, routedExpertBits: 3)
        _ = try await LocalQwenRepacker(options: options).run()
        defer { try? FileManager.default.removeItem(atPath: out) }

        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: out),
            device: ctx.device,
            expecting: Toy.arch)

        #expect(model.routedExpertWeightBits == 3)
        #expect(model.packedExpertsLayout.expertsPerLayer == Toy.experts)

        let expert = model.packedExpertsLayout.expert(layer: 0, expert: 0)
        let gate = try #require(expert.subTensors["gate"])
        #expect(gate.bits == 3)
        #expect(gate.dtype == "U32")
        #expect(gate.size == UInt64(Toy.moeIntermediate * Toy.D * 3 / 8))
        let down = try #require(expert.subTensors["down"])
        #expect(down.bits == 3)
        #expect(down.size == UInt64(Toy.D * Toy.moeIntermediate * 3 / 8))
        // Aux scales/biases keep the int4 layout (BF16 per group of 64).
        let gateScales = try #require(expert.subTensors["gate_scales"])
        #expect(gateScales.size
            == UInt64(Toy.moeIntermediate * (Toy.D / 64) * 2))
    }

    /// 2-bit twin of `threeBitToyInstallLoadsAndValidates`, plus a
    /// byte-exact check of a written expert blob: the packed int2 bytes,
    /// scales and biases of expert 0's gate must equal
    /// `Quantization.quantizeInt2Affine` applied to the toy source rows.
    /// This is the writer-side stride guard for the 2-bit width (the same
    /// class of bug as the int8 row-stride regression, which loads fine and
    /// computes garbage).
    @Test func twoBitToyInstallLoadsAndBlobBytesMatchQuantizer() async throws {
        let base = NSTemporaryDirectory() + "qwen-engine-2bit-\(UUID().uuidString)"
        let src = base + "-src"
        let seed: UInt64 = 0x71EC
        try Toy.writeSnapshot(into: src, seed: seed)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let out = base + "-out"
        _ = try await LocalQwenRepacker(
            options: LocalQwenRepackOptions(snapshotDir: src, outputDir: out,
                                            minFreeReserveBytes: 0,
                                            routedExpertBits: 2)).run()
        defer { try? FileManager.default.removeItem(atPath: out) }

        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: out),
            device: ctx.device,
            expecting: Toy.arch)
        #expect(model.routedExpertWeightBits == 2)

        // Regenerate the toy source stream for the layer-0 expert tensor.
        var state = seed
        var expertValues: [Float] = []
        let wantName = "model.language_model.layers.0.mlp.experts.gate_up_proj"
        for (name, shape) in Toy.tensorShapes() {
            let elements = shape.reduce(1, *)
            let keep = name == wantName
            for _ in 0..<elements {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                if keep {
                    let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
                    let bits = Quantization.bf16Bits(-2.0 + 4.0 * fraction)
                    expertValues.append(Quantization.bf16ToFloat(bits))
                }
            }
        }
        #expect(expertValues.count == Toy.experts * 2 * Toy.moeIntermediate * Toy.D)

        let layer0 = model.packedExpertsLayout.layers[0]
        let expert0 = layer0.experts[0]
        let gate = try #require(expert0.subTensors["gate"])
        let gateScales = try #require(expert0.subTensors["gate_scales"])
        let gateBiases = try #require(expert0.subTensors["gate_biases"])
        let rows = Toy.moeIntermediate
        let cols = Toy.D
        #expect(gate.bits == 2)
        #expect(gate.size == UInt64(rows * cols / 4))

        let blob = try Data(contentsOf: URL(fileURLWithPath:
            out + "/packed_experts/\(layer0.file)"))
        let blobBase = Int(expert0.offset)
        for r in 0..<rows {
            let row = Array(expertValues[r * cols..<(r + 1) * cols])
            let expected = Quantization.quantizeInt2Affine(row)
            let packedBase = blobBase + Int(gate.offset) + r * (cols / 4)
            #expect(Array(blob[packedBase..<(packedBase + cols / 4)])
                == expected.packed, "gate row \(r) packed")
            let groups = cols / 64
            var scaleBytes = [UInt8]()
            for v in expected.scales {
                scaleBytes.append(UInt8(truncatingIfNeeded: v))
                scaleBytes.append(UInt8(truncatingIfNeeded: v >> 8))
            }
            let sBase = blobBase + Int(gateScales.offset) + r * groups * 2
            #expect(Array(blob[sBase..<(sBase + groups * 2)]) == scaleBytes,
                    "gate row \(r) scales")
            var biasBytes = [UInt8]()
            for v in expected.biases {
                biasBytes.append(UInt8(truncatingIfNeeded: v))
                biasBytes.append(UInt8(truncatingIfNeeded: v >> 8))
            }
            let bBase = blobBase + Int(gateBiases.offset) + r * groups * 2
            #expect(Array(blob[bBase..<(bBase + groups * 2)]) == biasBytes,
                    "gate row \(r) biases")
        }
    }

    /// The writer's reserved entries must be byte-exact for every width a
    /// single install mixes: int4 (embeddings/attention/shared), int8 (GDN
    /// projections, router). Size validation alone cannot see a row-stride
    /// bug — an install with int8 entries written at the wrong stride still
    /// loads and still passes `validateRuntimeSchema`, then computes NaN.
    /// This regenerates the toy source (deterministic LCG) and compares the
    /// resident bytes row by row against `Quantization.quantize*Affine`.
    @Test func residentEntriesMatchWriterQuantizationByteForByte() async throws {
        let base = NSTemporaryDirectory() + "qwen-engine-bytes-\(UUID().uuidString)"
        let src = base + "-src"
        let seed: UInt64 = 0x71EA
        try Toy.writeSnapshot(into: src, seed: seed)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let out = base + "-out"
        _ = try await LocalQwenRepacker(
            options: LocalQwenRepackOptions(snapshotDir: src, outputDir: out,
                                            minFreeReserveBytes: 0)).run()
        defer { try? FileManager.default.removeItem(atPath: out) }

        let interesting: [String: (resident: String, bits: Int)] = [
            "model.language_model.embed_tokens.weight":
                ("language_model.model.embed_tokens.weight", 4),
            // Layer 0 is a linear-attention layer in the toy's fullMask.
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight":
                ("language_model.model.layers.0.linear_attn.in_proj_qkv.weight", 8),
        ]

        // Replay the source stream exactly as Toy.writeSnapshot wrote it.
        var state = seed
        var sourceRows: [String: (rows: Int, cols: Int, values: [Float])] = [:]
        for (name, shape) in Toy.tensorShapes() {
            let elements = shape.reduce(1, *)
            let keep = interesting[name] != nil
            var values: [Float] = []
            if keep { values.reserveCapacity(elements) }
            for _ in 0..<elements {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                if keep {
                    let fraction = Float(state >> 40) / Float(UInt64(1) << 24)
                    let bits = Quantization.bf16Bits(-2.0 + 4.0 * fraction)
                    values.append(Quantization.bf16ToFloat(bits))
                }
            }
            if keep { sourceRows[name] = (shape[0], shape[1], values) }
        }

        let ctx = try MetalContext()
        let model = try Model.load(
            directoryURL: URL(fileURLWithPath: out),
            device: ctx.device,
            expecting: Toy.arch)
        let file = try Data(contentsOf: URL(fileURLWithPath: out + "/model_weights.bin"))

        for (sourceName, spec) in interesting {
            let source = try #require(sourceRows[sourceName])
            let entry = try #require(model.residentIndex.entries[spec.resident])
            let rows = source.rows
            let cols = source.cols
            let groups = cols / Quantization.groupSize
            let rowBytes = cols * spec.bits / 8
            #expect(entry.sizeBytes == UInt64(rows * rowBytes))
            for r in 0..<rows {
                let row = Array(source.values[r * cols..<(r + 1) * cols])
                let packed: [UInt8]
                let scales: [UInt16]
                let biases: [UInt16]
                if spec.bits == 8 {
                    let q = Quantization.quantizeInt8Affine(row)
                    packed = q.packed
                    scales = q.scales
                    biases = q.biases
                } else {
                    let q = Quantization.quantizeInt4Affine(row)
                    packed = q.packed
                    scales = q.scales
                    biases = q.biases
                }
                let packedBase = Int(entry.fileOffset) + r * rowBytes
                #expect(Array(file[packedBase..<(packedBase + rowBytes)]) == packed,
                        "\(sourceName) row \(r) packed bytes")
                for (base, values, label) in [
                    (Int(entry.scaleOffset), scales, "scales"),
                    (Int(entry.biasOffset), biases, "biases"),
                ] {
                    var expected = [UInt8]()
                    for v in values {
                        expected.append(UInt8(truncatingIfNeeded: v))
                        expected.append(UInt8(truncatingIfNeeded: v >> 8))
                    }
                    let range = (base + r * groups * 2)..<(base + (r + 1) * groups * 2)
                    #expect(Array(file[range]) == expected,
                            "\(sourceName) row \(r) \(label) bytes")
                }
            }
        }
    }

    /// The prefill path must attribute its own GPU time and its own expert
    /// bytes, on the real encoder and not in a unit test of the fields.
    ///
    /// Before those calls were added, every `gpu_*` and every `io_mb` on a
    /// `scope=prefill` counters line was structurally zero: the counters
    /// existed, the 3.6 prefill body simply never wrote them. A field that can
    /// only ever print zero looks exactly like a measurement of zero, so the
    /// prefill ranking in `docs/STRATA_PP_KV_ANALYSIS.md` rested on a number
    /// that was never taken. `RunnerCountersTests` cannot see that -- it tests
    /// the arithmetic on numbers a hand-built snapshot supplies. Only running
    /// the encoder can.
    ///
    /// This toy is the smallest thing that reaches every site on that path.
    /// `fullMask = [0, 0, 0, 1]` gives three GDN layers and one full-attention
    /// layer, so both layer-kind stacks are non-zero and their sum is exactly
    /// the layer total; and 4 tokens x 4 layers x topK 2 routes enough experts
    /// through the tile prefetch for the prefill miss count to be non-zero.
    @Test
    func prefillAttributesGpuTimeAndExpertBytesToItself() async throws {
        let base = NSTemporaryDirectory() + "qwen-pp-\(UUID().uuidString)"
        let src = base + "-src"
        try Toy.writeSnapshot(into: src, seed: 0x51EE)
        defer { try? FileManager.default.removeItem(atPath: src) }

        let out = base + "-out"
        _ = try await LocalQwenRepacker(options: LocalQwenRepackOptions(
            snapshotDir: src, outputDir: out, minFreeReserveBytes: 0)).run()
        defer { try? FileManager.default.removeItem(atPath: out) }

        let dest = URL(fileURLWithPath: out)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dest, device: ctx.device,
                                   expecting: Toy.arch)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: 8,
                                           runtimeConfiguration: .production)
        // Wide enough for either head path, matching the CLI's own
        // `logitBytes` calculation.
        let logits = try #require(ctx.device.makeBuffer(
            length: model.config.vocabSize * MemoryLayout<Int32>.stride,
            options: .storageModeShared))
        let wallStart = ContinuousClock.now
        _ = try await runner.prefillChunked(
            tokens: [1, 2, 3, 4][0...], startPosition: 0,
            outputMode: .greedyIfAvailable, config: .defaultChunked,
            into: logits, onProgress: { _ in })
        let wallNanos = wallStart.duration(to: .now).components.attoseconds / 1_000_000_000

        // The two layer-kind stacks must tile the layer total exactly, so
        // `gpu_cb1` cannot be silently dropping or double-counting a kind.
        #expect(runner.totalGpuCb1GdnNanos > 0)
        #expect(runner.totalGpuCb1FullAttnNanos > 0)
        #expect(runner.totalGpuCb1GdnNanos + runner.totalGpuCb1FullAttnNanos
                == runner.totalGpuCb1Nanos)
        // The routed stack sits beside those rather than inside them: `cb1` is
        // the layer's own forward, `routed` is that separate per-layer buffer.
        #expect(runner.totalGpuRoutedNanos > 0)
        // Bytes, not only time. The chunked prefill streams tiles through its
        // own `fetch` bindings and never reaches the decode-side miss counter,
        // which is why this needs its own total.
        #expect(runner.totalPrefillExpertMisses > 0)
        // The coverage gate the decode counters document: a buffer still in
        // flight reports no timestamps, so `gpu_samples` against the buffer
        // count says how much of the total was actually measured. Zero samples
        // would mean the driver publishes nothing and every assertion above is
        // vacuous.
        #expect(runner.totalGpuSamples > 0)
        #expect(runner.totalPrefillCommandBuffers > 0)
        #expect(runner.totalGpuSamples
                >= runner.totalPrefillCommandBuffers / 2)
        // Plausibility, the same gate the decode counters state: every recorded
        // buffer ran on this one queue inside this call, so the two stacks
        // cannot sum to more than the call took. The 4x slack is for timer
        // granularity and the host work between buffers -- a total much larger
        // than the wall clock means the timestamps are not usable, and that is
        // the finding worth failing on.
        let measuredNanos = runner.totalGpuCb1Nanos + runner.totalGpuRoutedNanos
        #expect(wallNanos > 0)
        #expect(measuredNanos <= wallNanos * 4)

        // The point of all of the above is one line a human ranks work by, so
        // render it from the live counters rather than only checking the fields.
        // A prefill-scoped line folded the prefill miss total into `misses` and
        // multiplied it by the expert stride for `io_mb`; both must be non-zero
        // here, because a zero in either is indistinguishable from a real
        // measurement of zero in the logs this ranking was built from.
        let line = RunnerCounters.line(RunnerCounterValues(runner),
                                       expertStride: model.routedExpertStrideBytes(),
                                       scope: "prefill")
        func field(_ key: String) -> String? {
            for token in line.split(separator: " ") {
                let trimmed = token.hasSuffix("]") ? token.dropLast() : token[...]
                let parts = trimmed.split(separator: "=", maxSplits: 1)
                if parts.count == 2, parts[0] == Substring(key) { return String(parts[1]) }
            }
            return nil
        }
        #expect(field("misses") != "0" && field("misses") != nil)
        #expect(field("io_mb").map { Double($0) ?? 0 } != 0)
        #expect(field("gpu_cb1_wall_ms").map { Double($0) ?? 0 } != 0)
        #expect(field("gpu_cb1_gdn_wall_ms").map { Double($0) ?? 0 } != 0)
        #expect(field("gpu_cb1_fullattn_wall_ms").map { Double($0) ?? 0 } != 0)
        #expect(field("gpu_routed_wall_ms").map { Double($0) ?? 0 } != 0)
    }
}

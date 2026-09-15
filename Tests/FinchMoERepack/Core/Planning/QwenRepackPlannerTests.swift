import Testing
import Foundation
import FinchMoEFormat
@testable import FinchMoERepackCore

@Suite struct QwenRepackPlannerTests {

    private static func makeSnapshot() throws -> String {
        let dir = NSTemporaryDirectory() + "qwen-repack-plan-\(UUID().uuidString)"
        return try SyntheticQwenSnapshot.write(into: dir)
    }

    @Test func checkpointNamesRemapToEngineNames() {
        #expect(QwenRepackPlanner.residentName(
            for: "model.language_model.embed_tokens.weight")
            == "language_model.model.embed_tokens.weight")
        #expect(QwenRepackPlanner.residentName(
            for: "model.language_model.layers.3.linear_attn.A_log")
            == "language_model.model.layers.3.linear_attn.A_log")
        #expect(QwenRepackPlanner.residentName(for: "lm_head.weight") == "lm_head.weight")
        #expect(QwenRepackPlanner.residentName(
            for: "model.visual.blocks.0.attn.proj.weight") == nil)
    }

    @Test func planCoversEveryEntryWithCorrectDtypesAndShapes() throws {
        let dir = try Self.makeSnapshot()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        let plan = try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                              arch: snapshot.arch,
                                              shardHeaders: snapshot.shardHeaders,
                                              outputDir: dir + "/out")

        #expect(plan.arch.modelFamily == "qwen3_6")
        #expect(plan.excludedTensorNames == ["model.visual.blocks.0.attn.proj.weight"])
        #expect(plan.resident.entries.count == SyntheticQwenSnapshot.Toy.residentEntryCount())
        #expect(plan.layers.count == SyntheticQwenSnapshot.Toy.numLayers)

        let byName = Dictionary(uniqueKeysWithValues: plan.resident.entries.map { ($0.name, $0) })
        let D = SyntheticQwenSnapshot.Toy.D
        let vd = SyntheticQwenSnapshot.Toy.valueDim

        func affineSizes(_ rows: Int, _ cols: Int, _ bits: Int) -> (UInt64, UInt64) {
            (UInt64(rows) * UInt64(cols) / UInt64(8 / bits),
             UInt64(rows) * UInt64(cols / 64) * 2)
        }

        // Global entries.
        let embed = try #require(byName["language_model.model.embed_tokens.weight"])
        let (ew, ea) = affineSizes(SyntheticQwenSnapshot.Toy.vocab, D, 4)
        #expect(embed.dtype == FinchFormatV1.DType.u32.rawValue)
        #expect(embed.sizeBytes == ew)
        #expect(embed.scaleSize == ea && embed.biasSize == ea)
        #expect(embed.logicalShape4 == [256, 64, 0, 0])

        let lmHead = try #require(byName["lm_head.weight"])
        #expect(lmHead.dtype == FinchFormatV1.DType.u32.rawValue)

        let finalNorm = try #require(byName["language_model.model.norm.weight"])
        #expect(finalNorm.dtype == FinchFormatV1.DType.bf16.rawValue)
        #expect(finalNorm.sizeBytes == UInt64(D * 2))
        #expect(finalNorm.scaleSize == 0 && finalNorm.biasSize == 0)

        // GDN layer 0 (projections int8: int4 noise on the recurrent rows
        // amplifies ~16x — see QwenRepackPlanner linear_attn policy).
        let qkv = try #require(byName["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"])
        let (qw, qa) = affineSizes(SyntheticQwenSnapshot.Toy.qkvDim, D, 8)
        #expect(qkv.sizeBytes == qw && qkv.scaleSize == qa)

        let conv = try #require(byName["language_model.model.layers.0.linear_attn.conv1d.weight"])
        #expect(conv.dtype == FinchFormatV1.DType.fp16.rawValue)
        #expect(conv.sizeBytes == UInt64(SyntheticQwenSnapshot.Toy.qkvDim * 4 * 2))
        #expect(conv.logicalShape4 == [2048, 0, 0, 0])

        let aLog = try #require(byName["language_model.model.layers.0.linear_attn.A_log"])
        #expect(aLog.dtype == FinchFormatV1.DType.fp32.rawValue)
        #expect(aLog.sizeBytes == UInt64(SyntheticQwenSnapshot.Toy.linearValueHeads * 4))
        #expect(aLog.logicalShape4 == [8, 0, 0, 0])

        let inA = try #require(byName["language_model.model.layers.0.linear_attn.in_proj_a.weight"])
        let (aw, aa) = affineSizes(SyntheticQwenSnapshot.Toy.linearValueHeads, D, 8)
        #expect(inA.sizeBytes == aw && inA.scaleSize == aa)

        // Router is int8.
        let router = try #require(byName["language_model.model.layers.0.mlp.gate.weight"])
        let (rw, ra) = affineSizes(SyntheticQwenSnapshot.Toy.experts, D, 8)
        #expect(router.sizeBytes == rw && router.scaleSize == ra)

        // Full layer 3: doubled q_proj.
        let qP = try #require(byName["language_model.model.layers.3.self_attn.q_proj.weight"])
        let qRows = 2 * SyntheticQwenSnapshot.Toy.numHeads * SyntheticQwenSnapshot.Toy.fullHeadDim
        let (pqw, _) = affineSizes(qRows, D, 4)
        #expect(qP.sizeBytes == pqw)
        #expect(qP.logicalShape4 == [UInt32(qRows), UInt32(D), 0, 0])

        let outP = try #require(byName["language_model.model.layers.0.linear_attn.out_proj.weight"])
        #expect(outP.logicalShape4 == [UInt32(D), UInt32(vd), 0, 0])

        // Payload layout: all offsets page-aligned beyond the index, sorted,
        // non-overlapping, exactly filling the resident region.
        #expect(plan.resident.indexSize % 16_384 == 0)
        var cursor = plan.resident.indexSize
        let sorted = plan.resident.entries.sorted { $0.fileOffset < $1.fileOffset }
        for entry in sorted {
            #expect(entry.fileOffset >= cursor)
            #expect(entry.fileOffset % 4 == 0)
            #expect(entry.scaleOffset % 2 == 0)
            #expect(entry.biasOffset % 2 == 0)
            cursor = entry.fileOffset + entry.sizeBytes
            if entry.scaleSize > 0 { cursor = max(cursor, entry.scaleOffset + entry.scaleSize) }
            if entry.biasSize > 0 { cursor = max(cursor, entry.biasOffset + entry.biasSize) }
        }
        #expect(cursor == plan.resident.indexSize + plan.resident.residentSize)
    }

    @Test func expertLayersSplitFusedGateUp() throws {
        let dir = try Self.makeSnapshot()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        let plan = try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                              arch: snapshot.arch,
                                              shardHeaders: snapshot.shardHeaders,
                                              outputDir: dir + "/out")

        let D = SyntheticQwenSnapshot.Toy.D
        let f = SyntheticQwenSnapshot.Toy.moeIntermediate
        for layer in plan.layers {
            #expect(layer.expertsPerLayer == SyntheticQwenSnapshot.Toy.experts)
            #expect(layer.expertStride % 16_384 == 0)
            let roles = layer.subTensors.map { "\($0.role).\($0.component)" }
            let expected = ["gate.weights", "gate.scales", "gate.biases",
                            "up.weights", "up.scales", "up.biases",
                            "down.weights", "down.scales", "down.biases"]
            #expect(roles == expected)

            let gateW = layer.subTensors[0]
            let upW = layer.subTensors[3]
            let downW = layer.subTensors[6]
            #expect(gateW.logicalShape == [UInt64(f), UInt64(D)])
            #expect(upW.logicalShape == [UInt64(f), UInt64(D)])
            #expect(downW.logicalShape == [UInt64(D), UInt64(f)])
            #expect(gateW.bitsForWeights == 4)
            // The up half sits f*d*2 bytes into each expert's fused block.
            #expect(upW.sourceBaseOffset == UInt64(f * D * 2))
            #expect(gateW.sourceBaseOffset == 0)
            #expect(gateW.sizeInExpertBlob == UInt64(f * D / 2))
            #expect(upW.offsetInExpertBlob
                == gateW.offsetInExpertBlob + gateW.sizeInExpertBlob
                    + layer.subTensors[1].sizeInExpertBlob
                    + layer.subTensors[2].sizeInExpertBlob)
        }
    }

    @Test func nonLanguageTensorsAreExcludedNotRejected() throws {
        let dir = try Self.makeSnapshot()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        // The toy snapshot carries one vision tensor; plan must not throw.
        let plan = try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                              arch: snapshot.arch,
                                              shardHeaders: snapshot.shardHeaders,
                                              outputDir: dir + "/out")
        #expect(plan.excludedTensorNames.count == 1)
    }

    // MARK: - Qwen3.8-Flash-Next

    private static func makeSnapshot38() throws -> String {
        let dir = NSTemporaryDirectory() + "qwen38-repack-plan-\(UUID().uuidString)"
        return try SyntheticQwenSnapshot.write38(into: dir)
    }

    private static func plan38(snapshotDir: String) throws -> QwenRepackPlan {
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: snapshotDir)
        return try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                          arch: snapshot.arch,
                                          shardHeaders: snapshot.shardHeaders,
                                          outputDir: snapshotDir + "/out")
    }

    @Test func checkpointNamesRemap38ToShallowLanguageModel() {
        let f = ArchInfo.qwen38Family
        #expect(QwenRepackPlanner.residentName(
            for: "model.language_model.embed_tokens.weight", family: f)
            == "language_model.embed_tokens.weight")
        #expect(QwenRepackPlanner.residentName(
            for: "model.language_model.layers.1.ple.conv1d.weight", family: f)
            == "language_model.layers.1.ple.conv1d.weight")
        // There is no inner `model.` stage and no model.norm in 3.8.
        #expect(QwenRepackPlanner.residentName(for: "lm_head.weight", family: f)
            == "lm_head.weight")
        #expect(QwenRepackPlanner.residentName(
            for: "mtp.layers.0.self_attn.q_proj.weight", family: f) == nil)
        #expect(QwenRepackPlanner.residentName(
            for: "model.visual.blocks.0.attn.qkv.weight", family: f) == nil)
        // The 3.6 mapping is untouched by the family parameter default.
        #expect(QwenRepackPlanner.residentName(
            for: "model.language_model.layers.3.linear_attn.A_log")
            == "language_model.model.layers.3.linear_attn.A_log")
    }

    @Test func plan38CoversEveryEntryWithExpectedTransformsAndParts() throws {
        let dir = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let plan = try Self.plan38(snapshotDir: dir)
        let t = SyntheticQwenSnapshot.Toy38.self

        #expect(plan.arch.modelFamily == "qwen3_8")
        // Vision tower + MTP head (incl. mtp's own `.layers.0.` full layer and
        // hyper-connection bundle) are dropped, never routed or planned.
        #expect(plan.excludedTensorNames == [
            "model.visual.blocks.0.attn.proj.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
        ])
        #expect(plan.resident.entries.count == t.residentEntryCount)
        #expect(plan.layers.count == t.numLayers)
        // Every layer carries the routed-expert bundle (real checkpoint shape).
        #expect(plan.layers.allSatisfy { $0.expertsPerLayer == t.experts })

        // PLE parts: 4 × [32, 160], int4-affine quantized (Phase 3),
        // contiguous naming, in-plan rows.
        let groupSize = FinchQuantization.pleGroupSize
        for (i, part) in plan.pleParts.enumerated() {
            #expect(part.partIndex == i)
            #expect(part.relativePath == String(format: "ple_shards/shard_%03d.bin", i))
            #expect(part.path.hasSuffix("/ple_shards/shard_\(String(format: "%03d", i)).bin"))
            #expect(part.rows == t.ngramPartRows && part.cols == t.ngramRowDim)
            // The transform metadata the writer + size accounting derive from:
            // the Phase-1 group size, and the fixed per-row stride / total size
            // for the [packed][scale][bias] layout.
            #expect(part.groupSize == groupSize)
            let nGroups = t.ngramRowDim / groupSize
            #expect(part.rowByteStride == t.ngramRowDim / 2 + 2 * nGroups * 2)
            #expect(part.quantizedByteCount == UInt64(t.ngramPartRows) * UInt64(part.rowByteStride))
        }
        // Census correction: manifest arch reflects the live part geometry.
        #expect(plan.arch.ngramPartCount == t.ngramPartCount)
        #expect(plan.arch.ngramPartRows == t.ngramPartRows)
        #expect(plan.arch.pleLayerIndexes == [1])   // config ple_layer_ids [2], 1-based

        let byName = Dictionary(uniqueKeysWithValues: plan.resident.entries.map { ($0.name, $0) })
        func affineSizes(_ rows: Int, _ cols: Int, _ bits: Int) -> (UInt64, UInt64) {
            (UInt64(rows) * UInt64(cols) / UInt64(8 / bits),
             UInt64(rows) * UInt64(cols / 64) * 2)
        }

        // Root hyper-connection mixer: raw-BF16 grouped-RMS gate + int4 mixes,
        // resident AFTER lm_head (there is no final norm to hold (3,1,0)).
        let rootNorm = try #require(byName["language_model.hyper_connection_mixer.hc_norm.weight"])
        #expect(rootNorm.dtype == FinchFormatV1.DType.bf16.rawValue)
        #expect(rootNorm.sizeBytes == UInt64(t.plane * 2))
        let rootDown = try #require(
            byName["language_model.hyper_connection_mixer.input_mix_weight_down.weight"])
        let (rdw, rda) = affineSizes(t.hcLowrank, t.plane, 4)
        #expect(rootDown.dtype == FinchFormatV1.DType.u32.rawValue)
        #expect(rootDown.sizeBytes == rdw && rootDown.scaleSize == rda)
        let rootUp = try #require(
            byName["language_model.hyper_connection_mixer.input_mix_weight_up.weight"])
        let (ruw, rua) = affineSizes(t.plane, t.hcLowrank, 4)
        #expect(rootUp.sizeBytes == ruw && rootUp.scaleSize == rua)

        // Per-layer hyper-connection bundle (block_inject_weight only here).
        let hcNorm = try #require(
            byName["language_model.layers.0.attn_hyper_connection.hc_norm.weight"])
        #expect(hcNorm.dtype == FinchFormatV1.DType.bf16.rawValue)
        #expect(hcNorm.scaleSize == 0 && hcNorm.biasSize == 0)
        let hcInject = try #require(
            byName["language_model.layers.0.mlp_hyper_connection.block_inject_weight.weight"])
        let (iw, ia) = affineSizes(4, t.plane, 4)
        #expect(hcInject.sizeBytes == iw && hcInject.scaleSize == ia)

        // QSA indexer on the full layer (3): int4 index_qk_proj, 1+w norms.
        let indexQK = try #require(
            byName["language_model.layers.3.self_attn.indexer.index_qk_proj.weight"])
        let indexRows = (t.indexerNumHeads + t.indexerKVHeads) * t.indexerHeadDim
        let (qxw, qxa) = affineSizes(indexRows, t.D, 4)
        #expect(indexQK.sizeBytes == qxw && indexQK.scaleSize == qxa)
        let qLN = try #require(
            byName["language_model.layers.3.self_attn.indexer.q_layernorm.weight"])
        #expect(qLN.dtype == FinchFormatV1.DType.bf16.rawValue)
        #expect(qLN.sizeBytes == UInt64(t.indexerHeadDim * 2))

        // PLE head: fp16 squeezed conv, int8 key/value projections, 1+w norms.
        let pleConv = try #require(byName["language_model.layers.1.ple.conv1d.weight"])
        #expect(pleConv.dtype == FinchFormatV1.DType.fp16.rawValue)
        #expect(pleConv.sizeBytes == UInt64(t.plane * t.pleConvKernel * 2))
        #expect(pleConv.logicalShape4 == [UInt32(t.plane * t.pleConvKernel), 0, 0, 0])
        let keyProj = try #require(byName["language_model.layers.1.ple.key_proj.weight"])
        let (kw, ka) = affineSizes(t.plane, t.ngramWidth, 8)
        #expect(keyProj.sizeBytes == kw && keyProj.scaleSize == ka)
        let valueProj = try #require(byName["language_model.layers.1.ple.value_proj.weight"])
        let (vw, va) = affineSizes(t.D, t.ngramWidth, 8)
        #expect(valueProj.sizeBytes == vw && valueProj.scaleSize == va)
        let normConv = try #require(byName["language_model.layers.1.ple.norm_conv.weight"])
        #expect(normConv.dtype == FinchFormatV1.DType.bf16.rawValue)
        #expect(normConv.sizeBytes == UInt64(t.plane * 2))

        // I64 hash metadata rides the resident file raw (dtype byte 4).
        let multipliers = try #require(
            byName["language_model.layers.1.ple.ple_embedding.layer_multipliers"])
        #expect(multipliers.dtype == FinchFormatV1.DType.i64.rawValue)
        #expect(multipliers.sizeBytes == UInt64(t.ngramSize * 8))
        #expect(multipliers.logicalShape4 == [UInt32(t.ngramSize), 0, 0, 0])
        for suffix in ["ngram_heads_offsets", "ngram_heads_vocab_sizes"] {
            let entry = try #require(byName[
                "language_model.layers.1.ple.ple_embedding.\(suffix)"])
            #expect(entry.dtype == FinchFormatV1.DType.i64.rawValue)
            #expect(entry.sizeBytes == UInt64(t.pleHeads * 8))
        }

        // GDN projections stay int8; full-attention projections int4.
        let qkv = try #require(byName["language_model.layers.0.linear_attn.in_proj_qkv.weight"])
        let (qw, qa) = affineSizes(t.qkvDim, t.D, 8)
        #expect(qkv.sizeBytes == qw && qkv.scaleSize == qa)
        let qP = try #require(byName["language_model.layers.3.self_attn.q_proj.weight"])
        let (pqw, _) = affineSizes(2 * t.numHeads * t.fullHeadDim, t.D, 4)
        #expect(qP.sizeBytes == pqw)

        // No (1+w) bakes that do not exist in this family: there is no final
        // norm, no input/post-attention norms, and no inner `model.` stage.
        #expect(byName["language_model.model.norm.weight"] == nil)
        #expect(plan.resident.entries.allSatisfy {
            !$0.name.hasSuffix(".input_layernorm.weight")
                && !$0.name.hasSuffix(".post_attention_layernorm.weight")
                && !$0.name.contains(".model.norm")
        })

        // Payload layout invariants (same walk as the 3.6 test).
        #expect(plan.resident.indexSize % 16_384 == 0)
        var cursor = plan.resident.indexSize
        for entry in plan.resident.entries.sorted(by: { $0.fileOffset < $1.fileOffset }) {
            #expect(entry.fileOffset >= cursor)
            cursor = entry.fileOffset + entry.sizeBytes
            if entry.scaleSize > 0 { cursor = max(cursor, entry.scaleOffset + entry.scaleSize) }
            if entry.biasSize > 0 { cursor = max(cursor, entry.biasOffset + entry.biasSize) }
        }
        #expect(cursor == plan.resident.indexSize + plan.resident.residentSize)
    }

    @Test func plan38ResidentOrderHasNoFinalNormAndPlacesRootMixerLast() throws {
        let dir = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let plan = try Self.plan38(snapshotDir: dir)
        let names = plan.resident.entries.map(\.name)
        func idx(_ suffix: String) -> Int {
            names.firstIndex { $0.hasSuffix(suffix) } ?? -1
        }

        // Embedding first; lm_head then the root mixer hold the tail slots
        // (the qwen3_6 final-norm slot (3,1,0) is taken by the root mixer).
        #expect(names[0] == "language_model.embed_tokens.weight")
        #expect(idx("language_model.lm_head.weight") < idx(
            "hyper_connection_mixer.hc_norm.weight"))
        // Root mixer is the last resident group, exactly where model.norm sat.
        let lastLayerName = names.last ?? ""
        #expect(lastLayerName.hasPrefix("language_model.hyper_connection_mixer."))

        // Per-layer ordering: GDN projections → shared expert → router → HC
        // bundles; layer 1 adds the PLE block after the mixers; the full
        // layer's indexer trails its own mixers.
        func order(_ a: String, _ b: String) -> Bool { idx(a) < idx(b) }
        #expect(order("layers.0.linear_attn.in_proj_qkv.weight",
                      "layers.0.mlp.shared_expert.gate_proj.weight"))
        #expect(order("layers.0.mlp.shared_expert.down_proj.weight",
                      "layers.0.mlp.gate.weight"))
        #expect(order("layers.0.mlp.gate.weight",
                      "layers.0.attn_hyper_connection.hc_norm.weight"))
        #expect(order("layers.0.attn_hyper_connection.block_inject_weight.weight",
                      "layers.0.mlp_hyper_connection.hc_norm.weight"))
        #expect(order("layers.1.mlp_hyper_connection.block_inject_weight.weight",
                      "layers.1.ple.conv1d.weight"))
        #expect(order("layers.1.ple.norm_conv.weight",
                      "layers.1.ple.ple_embedding.layer_multipliers"))
        #expect(order("layers.3.mlp_hyper_connection.block_inject_weight.weight",
                      "layers.3.self_attn.indexer.index_qk_proj.weight"))
        #expect(order("layers.3.self_attn.indexer.k_layernorm.weight",
                      "layers.3.self_attn.q_proj.weight") == false)

        // Layers run strictly in order with no interleaving.
        let starts = (0..<SyntheticQwenSnapshot.Toy38.numLayers).map { l in
            names.firstIndex { $0.contains("layers.\(l).") } ?? -1
        }
        let ends = (0..<SyntheticQwenSnapshot.Toy38.numLayers).map { l in
            names.lastIndex { $0.contains("layers.\(l).") } ?? -1
        }
        for l in 0..<(starts.count - 1) {
            #expect(ends[l] < starts[l + 1])
        }
    }

    @Test func plan38RejectsConfigPartCountMismatch() throws {
        let dir = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        // Break the config↔checkpoint agreement: 3 configured parts vs 4 live.
        let configPath = URL(fileURLWithPath: dir + "/config.json")
        var config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: configPath)) as! [String: Any]
        var tc = config["text_config"] as! [String: Any]
        tc["split_ngram_parts"] = 3
        config["text_config"] = tc
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: configPath)

        do {
            _ = try Self.plan38(snapshotDir: dir)
            Issue.record("expected config split_ngram_parts mismatch to throw")
        } catch RepackError.configurationInvalid(let detail) {
            #expect(detail.contains("split_ngram_parts"))
        } catch {
            Issue.record("unexpected error: \(error)")
        }
    }
}

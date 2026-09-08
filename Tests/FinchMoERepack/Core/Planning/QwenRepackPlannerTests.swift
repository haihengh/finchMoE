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
        #expect(embed.dtype == FinchTurboFormatV1.DType.u32.rawValue)
        #expect(embed.sizeBytes == ew)
        #expect(embed.scaleSize == ea && embed.biasSize == ea)
        #expect(embed.logicalShape4 == [256, 64, 0, 0])

        let lmHead = try #require(byName["lm_head.weight"])
        #expect(lmHead.dtype == FinchTurboFormatV1.DType.u32.rawValue)

        let finalNorm = try #require(byName["language_model.model.norm.weight"])
        #expect(finalNorm.dtype == FinchTurboFormatV1.DType.bf16.rawValue)
        #expect(finalNorm.sizeBytes == UInt64(D * 2))
        #expect(finalNorm.scaleSize == 0 && finalNorm.biasSize == 0)

        // GDN layer 0.
        let qkv = try #require(byName["language_model.model.layers.0.linear_attn.in_proj_qkv.weight"])
        let (qw, qa) = affineSizes(SyntheticQwenSnapshot.Toy.qkvDim, D, 4)
        #expect(qkv.sizeBytes == qw && qkv.scaleSize == qa)

        let conv = try #require(byName["language_model.model.layers.0.linear_attn.conv1d.weight"])
        #expect(conv.dtype == FinchTurboFormatV1.DType.fp16.rawValue)
        #expect(conv.sizeBytes == UInt64(SyntheticQwenSnapshot.Toy.qkvDim * 4 * 2))
        #expect(conv.logicalShape4 == [2048, 0, 0, 0])

        let aLog = try #require(byName["language_model.model.layers.0.linear_attn.A_log"])
        #expect(aLog.dtype == FinchTurboFormatV1.DType.fp32.rawValue)
        #expect(aLog.sizeBytes == UInt64(SyntheticQwenSnapshot.Toy.linearValueHeads * 4))
        #expect(aLog.logicalShape4 == [8, 0, 0, 0])

        let inA = try #require(byName["language_model.model.layers.0.linear_attn.in_proj_a.weight"])
        let (aw, aa) = affineSizes(SyntheticQwenSnapshot.Toy.linearValueHeads, D, 4)
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
}

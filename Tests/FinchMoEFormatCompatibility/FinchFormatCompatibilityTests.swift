import Foundation
import Testing
@testable import FinchMoE
@testable import FinchMoEFormat
@testable import FinchMoERepackCore

@Suite struct FinchFormatCompatibilityTests {
    @Test func productionWritersMatchPreRefactorV1Fixtures() throws {
        let fixture = makeFixture()
        let manifestData = try FinchJSON.encodeManifest(
            plan: fixture.plan,
            modelID: "fixture/model",
            sourceSnapshotHash: "fixture-snapshot",
            files: fixture.files,
            expertsPerLayer: fixture.config.numExperts,
            numLayers: fixture.config.numLayers,
            expertStride: fixture.expertStride,
            bitWidths: FinchJSON.QuantBitWidths(
                embedding: 4,
                attention: 4,
                linearAttention: 4,
                router: 8,
                sharedExpert: 4,
                routedExpert: 4))
        let layoutData = try FinchJSON.encodeLayout(
            plan: fixture.plan, expertStride: fixture.expertStride)
        let indexData = try ResidentWriter.encodeIndex(plan: fixture.plan.resident)

        let frozenManifest = try frozenFixture("manifest.json")

        let frozenLayout = try frozenFixture("layout.json")
        let frozenIndex = try frozenFixture("resident-index.bin")
        #expect(manifestData == frozenManifest)
        #expect(layoutData == frozenLayout)
        #expect(indexData == frozenIndex)

        let manifest = try ManifestReader.decode(
            data: frozenManifest,
            expecting: fixture.config)
        let layout = try PackedExpertsLayoutReader.decode(
            data: frozenLayout, manifest: manifest)
        #expect(layout.expert(layer: 0, expert: 1).offset == fixture.expertStride)
        let manifestRoot = try #require(
            JSONSerialization.jsonObject(with: frozenManifest) as? [String: Any])
        #expect(manifestRoot["bitWidthOverridesHonored"] as? Int == 120)

        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("finch-index-fixture-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: temp) }
        try frozenIndex.write(to: temp)
        let index = try ResidentIndexReader.load(fileURL: temp)
        #expect(index.header.indexSize == fixture.plan.resident.indexSize)
        #expect(index.entries[fixture.plan.resident.entries[0].name]?.dtype
                == FinchFormatV1.DType.u32.rawValue)

        let hashes = [
            hash(frozenManifest),
            hash(frozenLayout),
            hash(frozenIndex),
        ]
        #expect(hashes == [
            // Manifest hash changed when the linearAttention quant slot
            // became a required wire key (fixture regenerated).
            "1de997f6e819687c4951d415579575890f3b3a520c7d0909f27a0e6498ab8e02",
            "acf57a355128d1d8afb1d09b75dc1aa7fef98687871b1ff9f738dd511cb6e341",
            "aa705246112c17d4b60422a2705bda53e96553f7ff96e09b00e6fbb5a4ffa594",
        ], "fixture hashes: \(hashes)")
    }

    @Test func sharedConstantsMatchExistingFacades() {
        #expect(FinchJSON.magic == FinchFormatV1.magic)
        #expect(FinchJSON.versionMajor == FinchFormatV1.versionMajor)
        #expect(FinchBinary.indexHeaderBytes == FinchFormatV1.residentHeaderBytes)
        #expect(FinchBinary.indexEntryBytes == FinchFormatV1.residentEntryBytes)
    }

    private func makeFixture() -> (
        config: ArchConfig,
        plan: RepackPlan,
        files: [(relativePath: String, info: FinchJSON.FileEntry)],
        expertStride: UInt64
    ) {
        let config = ArchConfig(
            hiddenSize: 64, intermediateSize: 128, moeIntermediateSize: 32,
            numHeads: 4, numKVHeads: 2, numFullKVHeads: 1,
            headDim: 16, fullHeadDim: 32, vocabSize: 1024,
            slidingWindow: 128, finalLogitSoftcap: 30,
            ropeTheta: 10_000, fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25, numLayers: 1, numExperts: 2,
            topKExperts: 1, tieWordEmbeddings: true, attentionKEqV: true,
            fullAttentionLayerMask: [0], hiddenActivation: "gelu_pytorch_tanh",
            modelFamily: "gemma4", attnOutputGate: false,
            linearNumKeyHeads: 0, linearNumValueHeads: 0,
            linearKeyHeadDim: 0, linearValueHeadDim: 0, linearConvKernelDim: 0)
        let arch = ArchInfo(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            moeIntermediateSize: config.moeIntermediateSize,
            numHeads: config.numHeads,
            numKVHeads: config.numKVHeads,
            numFullKVHeads: config.numFullKVHeads,
            headDim: config.headDim,
            fullHeadDim: config.fullHeadDim,
            vocabSize: config.vocabSize,
            slidingWindow: config.slidingWindow,
            finalLogitSoftcap: config.finalLogitSoftcap,
            ropeTheta: config.ropeTheta,
            fullRopeTheta: config.fullRopeTheta,
            partialRotaryFactor: config.partialRotaryFactor,
            numLayers: config.numLayers,
            numExperts: config.numExperts,
            topKExperts: config.topKExperts,
            tieWordEmbeddings: config.tieWordEmbeddings,
            attentionKEqV: config.attentionKEqV,
            fullAttentionLayerMask: config.fullAttentionLayerMask,
            hiddenActivation: config.hiddenActivation,
            modelFamily: nil,
            attnOutputGate: nil,
            linearNumKeyHeads: nil,
            linearNumValueHeads: nil,
            linearKeyHeadDim: nil,
            linearValueHeadDim: nil,
            linearConvKernelDim: nil)
        let source = SourceTensor(
            name: "fixture.weight", shardPath: "/dev/null", dtype: .u32,
            shape: [1, 1], absoluteOffset: 0, sizeBytes: 16)
        let indexSize = FinchFormatV1.alignmentBytes
        let residentEntry = ResidentEntry(
            name: "language_model.model.embed_tokens.weight",
            dtype: FinchFormatV1.DType.u32.rawValue,
            logicalShape4: [1, 1, 0, 0],
            fileOffset: indexSize, sizeBytes: 16,
            scaleOffset: indexSize + 16, scaleSize: 8,
            biasOffset: indexSize + 24, biasSize: 8,
            quantSpec: QuantSpec(bits: 4),
            sourceWeight: source, sourceScales: source, sourceBiases: source)
        let nameBytes = Array(residentEntry.name.utf8)
        let resident = ResidentFilePlan(
            path: "/fixture/model_weights.bin",
            entries: [residentEntry],
            stringTable: nameBytes,
            stringTableOffsets: [0],
            indexSize: indexSize,
            residentSize: 32)
        let expertStride = FinchFormatV1.alignmentBytes
        let slice = PerExpertTensorSlice(
            role: "gate", component: "weights",
            dtype: FinchFormatV1.DType.u32.rawValue,
            logicalShape: [8, 8], offsetInExpertBlob: 0,
            sizeInExpertBlob: 32, sourceOffsetPerExpert: 32,
            sourceTensor: source, bitsForWeights: 4)
        let layer = LayerFilePlan(
            layerIndex: 0, path: "/fixture/packed_experts/layer_00.bin",
            expertsPerLayer: 2, expertStride: expertStride,
            subTensors: [slice])
        let plan = RepackPlan(
            arch: arch, baseMode: "affine", baseGroupSize: 64,
            bitsOverrideCount: 120, resident: resident, layers: [layer],
            matchedModelID: nil, excludedMultimodalTensorNames: [])
        let zeroSHA = String(repeating: "0", count: 64)
        let files: [(relativePath: String, info: FinchJSON.FileEntry)] = [
            ("model_weights.bin", .init(size: resident.totalSize, sha256: zeroSHA)),
            ("packed_experts/layout.json", .init(size: 1, sha256: zeroSHA)),
            ("packed_experts/layer_00.bin", .init(
                size: 2 * expertStride, sha256: zeroSHA)),
        ]
        return (config, plan, files, expertStride)
    }

    private func hash(_ data: Data) -> String {
        var stream = Sha256Stream()
        data.withUnsafeBytes { stream.update($0) }
        return stream.finalizeHexString()
    }

    private func frozenFixture(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(
            forResource: name, withExtension: "base64", subdirectory: "Fixtures/v1"))
        let encoded = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        return try #require(Data(base64Encoded: encoded, options: .ignoreUnknownCharacters))
    }
}

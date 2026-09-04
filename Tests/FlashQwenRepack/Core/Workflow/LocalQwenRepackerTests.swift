import Testing
import Foundation
import FlashQwenFormat
@testable import FlashQwenRepackCore

@Suite struct LocalQwenRepackerTests {

    private static func makeOutputDir() -> String {
        NSTemporaryDirectory() + "qwen-fqturbo-\(UUID().uuidString)"
    }

    private static func runRepack(snapshotDir: String) async throws -> LocalQwenRepackResult {
        let out = makeOutputDir()
        let options = LocalQwenRepackOptions(
            snapshotDir: snapshotDir, outputDir: out,
            minFreeReserveBytes: 0)
        return try await LocalQwenRepacker(options: options).run()
    }

    @Test func rowScratchAcceptsWidestRealRow() throws {
        // The real model's widest row is the GDN out_proj: 4096 columns.
        // (Caught by the first real dry run — the scratch cap was 2048.)
        var scratch = QwenQuantizedWriter.QwenRowScratch()
        let bytes = UnsafeMutableRawBufferPointer.allocate(
            byteCount: 4096 * 2, alignment: 8)
        defer { bytes.deallocate() }
        bytes.initializeMemory(as: UInt8.self, repeating: 0)
        let floats = try scratch.decodeBf16Row(
            UnsafeRawBufferPointer(bytes), count: 4096)
        #expect(floats.count == 4096)
        let q = FQTurboQuantization.quantizeInt4Affine(floats)
        #expect(q.packed.count == 2048)
    }

    @Test func repackProducesLoadableFormatFiles() async throws {
        let dir = NSTemporaryDirectory() + "qwen-repack-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: dir)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack(snapshotDir: dir)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }
        #expect(result.excludedTensorCount == 1)

        let fm = FileManager.default
        let out = result.outputDir
        #expect(fm.fileExists(atPath: out + "/model_weights.bin"))
        #expect(fm.fileExists(atPath: out + "/manifest.json"))
        #expect(fm.fileExists(atPath: out + "/verified-install.json"))
        #expect(fm.fileExists(atPath: out + "/packed_experts/layout.json"))
        for L in 0..<SyntheticQwenSnapshot.Toy.numLayers {
            #expect(fm.fileExists(atPath: out + String(format: "/packed_experts/layer_%02d.bin", L)))
        }
        #expect(fm.fileExists(atPath: out + "/tokenizer/config.json"))
        #expect(fm.fileExists(atPath: out + "/tokenizer/tokenizer.json"))

        // Manifest: qwen3_6 arch + locked quant slots.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: out + "/manifest.json"))
        let manifest = try FQTurboManifestCodec.decode(manifestData)
        #expect(manifest.modelID == "local/Qwen3.6-35B-A3B")
        #expect(manifest.arch.modelFamily == "qwen3_6")
        #expect(manifest.arch.attnOutputGate == true)
        #expect(manifest.arch.linearNumValueHeads == 8)
        #expect(manifest.arch.fullAttentionLayerMask == [0, 0, 0, 1])
        #expect(manifest.numLayers == 4 && manifest.expertsPerLayer == 4)
        #expect(manifest.expertStride % 16_384 == 0)
        let quant = try #require(manifest.quant)
        #expect(quant.embedding.weightBits == 4)
        #expect(quant.attention.weightBits == 4)
        #expect(quant.router.weightBits == 8)
        #expect(quant.sharedExpert.weightBits == 4)
        #expect(quant.routedExpert.weightBits == 4)
        #expect(quant.embedding.groupSize == 64 && quant.embedding.scheme == "affine")

        // Resident index: every engine-expected entry present with the right
        // dtype and shape.
        let weightsURL = URL(fileURLWithPath: out + "/model_weights.bin")
        let weightsData = try Data(contentsOf: weightsURL)
        let (header, entries) = try weightsData.withUnsafeBytes { raw -> (
            FQTurboResidentIndexHeaderV1, [FQTurboResidentIndexEntryV1]
        ) in
            let h = try FQTurboResidentIndexCodec.decodeHeader(raw)
            let e = try FQTurboResidentIndexCodec.decodeRegion(raw, header: h)
            return (h, e)
        }
        #expect(entries.count == SyntheticQwenSnapshot.Toy.residentEntryCount())
        #expect(header.indexSize + header.residentSize == UInt64(weightsData.count))
        let byName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })

        let embed = try #require(byName["language_model.model.embed_tokens.weight"])
        #expect(embed.dtype == 0 && embed.shape == [256, 64, 0, 0])
        let router = try #require(byName["language_model.model.layers.0.mlp.gate.weight"])
        #expect(router.dtype == 0 && router.shape == [4, 64, 0, 0])
        #expect(router.sizeBytes == UInt64(4 * 64))   // int8: one byte per weight
        let conv = try #require(byName["language_model.model.layers.0.linear_attn.conv1d.weight"])
        #expect(conv.dtype == 2 && conv.shape == [2048, 0, 0, 0])
        let aLog = try #require(byName["language_model.model.layers.0.linear_attn.A_log"])
        #expect(aLog.dtype == 3 && aLog.shape == [8, 0, 0, 0])
        let norm = try #require(byName["language_model.model.layers.0.input_layernorm.weight"])
        #expect(norm.dtype == 1 && norm.shape == [64, 0, 0, 0])
        #expect(byName["lm_head.weight"] != nil)
        #expect(byName["language_model.model.layers.0.mlp.shared_expert_gate.weight"] != nil)
        #expect(byName["language_model.model.layers.3.self_attn.q_proj.weight"]?.shape
            == [256, 64, 0, 0])

        // Layout.json decodes with 9 subtensors per expert.
        let layoutData = try Data(contentsOf: URL(fileURLWithPath: out + "/packed_experts/layout.json"))
        let layout = try FQTurboPackedExpertsLayoutCodec.decode(layoutData)
        #expect(layout.layers.count == 4)
        for layer in layout.layers {
            #expect(layer.experts.count == 4)
            for expert in layer.experts {
                #expect(expert.tensors.keys.sorted() == [
                    "down", "down_biases", "down_scales",
                    "gate", "gate_biases", "gate_scales",
                    "up", "up_biases", "up_scales",
                ])
            }
        }
    }

    @Test func quantizedPayloadBytesMatchTheCanonicalQuantizer() async throws {
        let dir = NSTemporaryDirectory() + "qwen-repack-qsrc-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: dir)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack(snapshotDir: dir)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }

        // Spot-check one int4 entry end to end: the source rows quantized by
        // FQTurboQuantization must byte-match the packed payload.
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        let srcTensor = try #require(snapshot.shardHeaders.flatMap(\.tensors)
            .first { $0.name == "model.language_model.layers.0.linear_attn.in_proj_a.weight" })
        let rows = Int(srcTensor.shape[0])
        let cols = Int(srcTensor.shape[1])
        let srcPath = srcTensor.shardPath
        let srcFD = try Posix.openRead(srcPath)
        defer { close(srcFD) }
        let srcBytes = UnsafeMutableRawBufferPointer.allocate(
            byteCount: rows * cols * 2, alignment: 8)
        defer { srcBytes.deallocate() }
        try Posix.preadAll(fd: srcFD, path: srcPath,
                           buf: srcBytes.baseAddress!, count: rows * cols * 2,
                           offset: srcTensor.absoluteOffset)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: result.outputDir + "/model_weights.bin"))
        let (_, entries) = try weightsData.withUnsafeBytes { raw -> (
            FQTurboResidentIndexHeaderV1, [FQTurboResidentIndexEntryV1]
        ) in
            let h = try FQTurboResidentIndexCodec.decodeHeader(raw)
            return (h, try FQTurboResidentIndexCodec.decodeRegion(raw, header: h))
        }
        let entry = try #require(entries.first {
            $0.name == "language_model.model.layers.0.linear_attn.in_proj_a.weight"
        })

        var packedPayload = [UInt8]()
        var scalesPayload = [UInt16]()
        var biasesPayload = [UInt16]()
        for row in 0..<rows {
            let bits16 = (0..<cols).map { i -> UInt16 in
                UInt16(srcBytes[row * cols * 2 + 2 * i])
                    | UInt16(srcBytes[row * cols * 2 + 2 * i + 1]) << 8
            }
            let floats = bits16.map { FQTurboQuantization.bf16ToFloat($0) }
            let q = FQTurboQuantization.quantizeInt4Affine(floats)
            packedPayload.append(contentsOf: q.packed)
            scalesPayload.append(contentsOf: q.scales)
            biasesPayload.append(contentsOf: q.biases)
        }

        let fileRange = entry.fileOffset..<(entry.fileOffset + entry.sizeBytes)
        #expect(Array(weightsData.subdata(in: Data.Index(fileRange.lowerBound)..<Data.Index(fileRange.upperBound))) == packedPayload)
        let scaleRange = entry.scaleOffset..<(entry.scaleOffset + entry.scaleSize)
        let scaleBytes = weightsData.subdata(in: Data.Index(scaleRange.lowerBound)..<Data.Index(scaleRange.upperBound))
        #expect(scaleBytes.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self)) == scalesPayload
        })
        let biasRange = entry.biasOffset..<(entry.biasOffset + entry.biasSize)
        let biasBytes = weightsData.subdata(in: Data.Index(biasRange.lowerBound)..<Data.Index(biasRange.upperBound))
        #expect(biasBytes.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: UInt16.self)) == biasesPayload
        })
    }

    @Test func normWeightsBakeOnePlusW() async throws {
        let dir = NSTemporaryDirectory() + "qwen-repack-nsrc-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: dir)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack(snapshotDir: dir)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }

        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        let srcTensor = try #require(snapshot.shardHeaders.flatMap(\.tensors)
            .first { $0.name == "model.language_model.norm.weight" })
        let srcPath = srcTensor.shardPath
        let srcFD = try Posix.openRead(srcPath)
        defer { close(srcFD) }
        let srcBytes = UnsafeMutableRawBufferPointer.allocate(
            byteCount: Int(srcTensor.sizeBytes), alignment: 8)
        defer { srcBytes.deallocate() }
        try Posix.preadAll(fd: srcFD, path: srcPath,
                           buf: srcBytes.baseAddress!, count: Int(srcTensor.sizeBytes),
                           offset: srcTensor.absoluteOffset)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: result.outputDir + "/model_weights.bin"))
        let (_, entries) = try weightsData.withUnsafeBytes { raw -> (
            FQTurboResidentIndexHeaderV1, [FQTurboResidentIndexEntryV1]
        ) in
            let h = try FQTurboResidentIndexCodec.decodeHeader(raw)
            return (h, try FQTurboResidentIndexCodec.decodeRegion(raw, header: h))
        }
        let entry = try #require(entries.first { $0.name == "language_model.model.norm.weight" })

        let n = Int(srcTensor.shape[0])
        let payload = weightsData.subdata(
            in: Data.Index(entry.fileOffset)..<Data.Index(entry.fileOffset + entry.sizeBytes))
        let written = payload.withUnsafeBytes { raw -> [UInt16] in
            Array(raw.bindMemory(to: UInt16.self))
        }
        for i in 0..<n {
            let srcBits = UInt16(srcBytes[2 * i]) | UInt16(srcBytes[2 * i + 1]) << 8
            let src = FQTurboQuantization.bf16ToFloat(srcBits)
            let expected = FQTurboQuantization.bf16Bits(1.0 + src)
            #expect(written[i] == expected)
        }
    }
}

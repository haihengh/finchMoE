import Testing
import Foundation
import FinchMoEFormat
@testable import FinchMoERepackCore

@Suite struct LocalQwenRepackerTests {

    private static func makeOutputDir() -> String {
        NSTemporaryDirectory() + "qwen-finch-\(UUID().uuidString)"
    }

    private static func runRepack(snapshotDir: String,
                                  includeVision: Bool = false) async throws
        -> LocalQwenRepackResult {
        let out = makeOutputDir()
        let options = LocalQwenRepackOptions(
            snapshotDir: snapshotDir, outputDir: out,
            minFreeReserveBytes: 0,
            includeVision: includeVision)
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
        let q = FinchQuantization.quantizeInt4Affine(floats)
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
        // Text-only remains text-only: no vision payload unless asked for.
        #expect(!fm.fileExists(atPath: out + "/vision"))

        // Manifest: qwen3_6 arch + locked quant slots.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: out + "/manifest.json"))
        let manifest = try FinchManifestCodec.decode(manifestData)
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
            FinchResidentIndexHeaderV1, [FinchResidentIndexEntryV1]
        ) in
            let h = try FinchResidentIndexCodec.decodeHeader(raw)
            let e = try FinchResidentIndexCodec.decodeRegion(raw, header: h)
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
        let layout = try FinchPackedExpertsLayoutCodec.decode(layoutData)
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
        // FinchQuantization must byte-match the packed payload.
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        let srcTensor = try #require(snapshot.shardHeaders.flatMap(\.tensors)
            .first { $0.name == "model.language_model.embed_tokens.weight" })
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
            FinchResidentIndexHeaderV1, [FinchResidentIndexEntryV1]
        ) in
            let h = try FinchResidentIndexCodec.decodeHeader(raw)
            return (h, try FinchResidentIndexCodec.decodeRegion(raw, header: h))
        }
        let entry = try #require(entries.first {
            $0.name == "language_model.model.embed_tokens.weight"
        })

        var packedPayload = [UInt8]()
        var scalesPayload = [UInt16]()
        var biasesPayload = [UInt16]()
        for row in 0..<rows {
            let bits16 = (0..<cols).map { i -> UInt16 in
                UInt16(srcBytes[row * cols * 2 + 2 * i])
                    | UInt16(srcBytes[row * cols * 2 + 2 * i + 1]) << 8
            }
            let floats = bits16.map { FinchQuantization.bf16ToFloat($0) }
            let q = FinchQuantization.quantizeInt4Affine(floats)
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
            FinchResidentIndexHeaderV1, [FinchResidentIndexEntryV1]
        ) in
            let h = try FinchResidentIndexCodec.decodeHeader(raw)
            return (h, try FinchResidentIndexCodec.decodeRegion(raw, header: h))
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
            let src = FinchQuantization.bf16ToFloat(srcBits)
            let expected = FinchQuantization.bf16Bits(1.0 + src)
            #expect(written[i] == expected)
        }
    }

    // MARK: - Vision payload

    /// The vision repack carries the source `model.visual.*` tower in one
    /// safetensors file plus its config sidecars, and declares all of them in
    /// the manifest so the download and verification paths cover them.
    @Test func visionRepackWritesTowerAndSidecars() async throws {
        let dir = NSTemporaryDirectory() + "qwen-vision-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: dir, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack(snapshotDir: dir, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }
        let out = result.outputDir
        let fm = FileManager.default

        // The toy source has no MTP head, so including vision leaves nothing
        // dropped — and the tower must not be counted as dropped either.
        #expect(result.excludedTensorCount == 0)

        let payloadPath = out + "/vision/vision.safetensors"
        #expect(fm.fileExists(atPath: payloadPath))
        #expect(fm.fileExists(atPath: out + "/vision/model_config.json"))
        #expect(fm.fileExists(atPath: out + "/vision/preprocessor_config.json"))
        // The real 3.6 snapshot has no video preprocessor config.
        #expect(!fm.fileExists(atPath: out + "/vision/video_preprocessor_config.json"))

        let manifestData = try Data(contentsOf: URL(fileURLWithPath: out + "/manifest.json"))
        let manifest = try FinchManifestCodec.decode(manifestData)
        for path in ["vision/vision.safetensors",
                     "vision/model_config.json",
                     "vision/preprocessor_config.json"] {
            let entry = try #require(manifest.files[path])
            let size = try #require(fm.attributesOfItem(
                atPath: out + "/" + path)[.size] as? NSNumber)
            #expect(entry.size == size.uint64Value)
        }
        #expect(manifest.files["vision/video_preprocessor_config.json"] == nil)

        // The receipt covers the same file set, and a full verify accepts it.
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputFinch: out))
        #expect(verify.unexpectedEntries.isEmpty)
        #expect(verify.fileCount == manifest.files.count + 1)   // + manifest.json

        // The payload is a standard safetensors file holding exactly the
        // source tower, byte for byte.
        let payload = try Data(contentsOf: URL(fileURLWithPath: payloadPath))
        var headerLength: UInt64 = 0
        for i in 0..<8 {
            headerLength |= UInt64(payload[i]) << UInt64(i * 8)
        }
        let headerData = payload.subdata(in: 8..<(8 + Int(headerLength)))
        let header = try Safetensors.parseHeaderBytes(
            path: payloadPath, fileSize: UInt64(payload.count), headerBytes: headerData)
        let source = try QwenLocalSnapshot.load(snapshotDir: dir)
        let expected = source.shardHeaders.flatMap(\.tensors)
            .filter { $0.name.hasPrefix("model.visual.") }
        let t = SyntheticQwenSnapshot.Toy.self
        #expect(expected.count == t.visionLayers * 12 + 9)
        #expect(header.tensors.count == expected.count)
        #expect(Set(header.tensors.map(\.name)) == Set(expected.map(\.name)))

        let name = "model.visual.blocks.1.attn.qkv.weight"
        let placed = try #require(header.tensors.first { $0.name == name })
        let copied = payload.subdata(
            in: Int(placed.absoluteOffset)..<Int(placed.absoluteOffset + placed.sizeBytes))
        #expect(copied == (try Self.readSourceTensor(source, name: name)))

        // The sidecar is a verbatim copy of the source config.
        let sourceConfig = try Data(contentsOf: URL(
            fileURLWithPath: dir + "/config.json"))
        let copiedConfig = try Data(contentsOf: URL(
            fileURLWithPath: out + "/vision/model_config.json"))
        #expect(copiedConfig == sourceConfig)
    }

    /// Without the preprocessor config there is no way for a vision runtime to
    /// reproduce the source's image preprocessing, so the repack refuses
    /// instead of shipping a payload with missing semantics.
    @Test func visionRepackRequiresThePreprocessorConfig() async throws {
        let dir = NSTemporaryDirectory() + "qwen-vision-missing-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: dir, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        try FileManager.default.removeItem(
            atPath: dir + "/preprocessor_config.json")

        do {
            _ = try await Self.runRepack(snapshotDir: dir, includeVision: true)
            Issue.record("repacked a vision payload with no preprocessor config")
        } catch let error as RepackError {
            guard case .configurationInvalid(let detail) = error else {
                Issue.record("expected configurationInvalid, got \(error)")
                return
            }
            #expect(detail.contains("preprocessor_config.json"))
        }
    }

    // MARK: - Qwen3.8-Flash-Next

    private static func runRepack38(snapshotDir: String,
                                    includeVision: Bool = false) async throws
        -> LocalQwenRepackResult {
        let out = makeOutputDir()
        let options = LocalQwenRepackOptions(
            snapshotDir: snapshotDir, outputDir: out,
            minFreeReserveBytes: 0,
            includeVision: includeVision)
        return try await LocalQwenRepacker(options: options).run()
    }

    /// Re-quantize a PLE part's raw BF16 source bytes row by row with the
    /// canonical Phase-2 codec, laying each row out as
    /// [packed nibbles: cols/2][scale BF16 × nGroups][bias BF16 × nGroups]
    /// — exactly the stride `QwenQuantizedWriter.writePLEPart` writes. A
    /// byte-compare of this against the written part proves the repack is the
    /// canonical transform, not a verbatim copy.
    private static func quantizedPLEBytes(_ source: Data, cols: Int, groupSize: Int) -> Data {
        let rows = source.count / (cols * 2)
        let nGroups = cols / groupSize
        let rowStride = cols / 2 + 2 * nGroups * 2
        var out = [UInt8](repeating: 0, count: rows * rowStride)
        for r in 0..<rows {
            var floats = [Float](repeating: 0, count: cols)
            for k in 0..<cols {
                let b = UInt16(source[r * cols * 2 + 2 * k])
                    | UInt16(source[r * cols * 2 + 2 * k + 1]) << 8
                floats[k] = FinchQuantization.bf16ToFloat(b)
            }
            let q = FinchQuantization.quantizeInt4AffinePLE(floats, groupSize: groupSize)
            var o = r * rowStride
            out.replaceSubrange(o..<(o + q.packed.count), with: q.packed)
            o += q.packed.count
            q.scales.withUnsafeBytes { memcpy(&out[o], $0.baseAddress!, $0.count) }; o += q.scales.count * 2
            q.biases.withUnsafeBytes { memcpy(&out[o], $0.baseAddress!, $0.count) }
        }
        return Data(out)
    }

    /// Raw source bytes of one checkpoint tensor (from the snapshot load).
    private static func readSourceTensor(
        _ snapshot: QwenLocalSnapshot.Snapshot, name: String
    ) throws -> Data {
        let srcTensor = try #require(snapshot.shardHeaders.flatMap(\.tensors)
            .first { $0.name == name })
        let srcFD = try Posix.openRead(srcTensor.shardPath)
        defer { close(srcFD) }
        var bytes = Data(count: Int(srcTensor.sizeBytes))
        try bytes.withUnsafeMutableBytes { buf in
            try Posix.preadAll(fd: srcFD, path: srcTensor.shardPath,
                               buf: buf.baseAddress!, count: Int(srcTensor.sizeBytes),
                               offset: srcTensor.absoluteOffset)
        }
        return bytes
    }

    @Test func qwen38RepackWritesPLEPartsAndFamilyManifest() async throws {
        let dir = NSTemporaryDirectory() + "qwen38-repack-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write38(into: dir)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack38(snapshotDir: dir)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }
        let t = SyntheticQwenSnapshot.Toy38.self
        let fm = FileManager.default
        let out = result.outputDir

        // Expected file list for a qwen3_8 install: resident + manifest +
        // receipt + packed experts (one layer file per layer) + the 4 PLE
        // shard files + tokenizer sidecars.
        #expect(fm.fileExists(atPath: out + "/model_weights.bin"))
        #expect(fm.fileExists(atPath: out + "/manifest.json"))
        #expect(fm.fileExists(atPath: out + "/verified-install.json"))
        for L in 0..<t.numLayers {
            #expect(fm.fileExists(atPath: out + String(format: "/packed_experts/layer_%02d.bin", L)))
        }
        // PLE parts are int4-affine quantized (Phase 3), not raw BF16:
        // per-row [packed nibbles: cols/2][scale BF16 × nGroups][bias BF16 × nGroups].
        let cols = t.ngramRowDim
        let groupSize = FinchQuantization.pleGroupSize
        let nGroups = cols / groupSize
        let rowStride = cols / 2 + 2 * nGroups * 2
        let partBytes = t.ngramPartRows * rowStride
        for i in 0..<t.ngramPartCount {
            let p = out + String(format: "/ple_shards/shard_%03d.bin", i)
            #expect(fm.fileExists(atPath: p))
            #expect((try? fm.attributesOfItem(atPath: p)[.size] as? Int) ?? 0 == partBytes)
        }
        #expect(fm.fileExists(atPath: out + "/tokenizer/config.json"))

        // Parts are int4-affine quantizations of their source BF16 rows —
        // re-quantize the source with the canonical codec and byte-compare.
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)
        for i in [0, 3] {
            let source = try Self.readSourceTensor(
                snapshot,
                name: "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(i).weight")
            let written = try Data(contentsOf: URL(fileURLWithPath:
                out + String(format: "/ple_shards/shard_%03d.bin", i)))
            #expect(written == Self.quantizedPLEBytes(source, cols: cols, groupSize: groupSize))
        }

        // Manifest: qwen3_8 identity, census-corrected PLE geometry, parts in
        // manifest.files, locked quant slots unchanged from qwen3_6.
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: out + "/manifest.json"))
        let manifest = try FinchManifestCodec.decode(manifestData)
        #expect(manifest.modelID == "local/Qwen3.8-Flash-Next-125B")
        #expect(manifest.arch.modelFamily == "qwen3_8")
        #expect(manifest.arch.hyperConnectionCount == 4)
        #expect(manifest.arch.hyperConnectionLowrank == t.hcLowrank)
        #expect(manifest.arch.indexerNumHeads == t.indexerNumHeads)
        #expect(manifest.arch.indexerBudget == t.indexerBudget)
        #expect(manifest.arch.ngramSize == t.ngramSize)
        #expect(manifest.arch.pleConvKernelSize == t.pleConvKernel)
        #expect(manifest.arch.pleLayerIndexes == [1])
        #expect(manifest.arch.ngramPartCount == t.ngramPartCount)
        #expect(manifest.arch.ngramPartRows == t.ngramPartRows)
        #expect(manifest.arch.fullAttentionLayerMask == [0, 0, 0, 1])
        for i in 0..<t.ngramPartCount {
            let entry = manifest.files[String(format: "ple_shards/shard_%03d.bin", i)]
            #expect(entry?.size == UInt64(partBytes))
        }
        let quant = try #require(manifest.quant)
        #expect(quant.embedding.weightBits == 4 && quant.router.weightBits == 8)
        // The new additive PLE n-gram slot is present for a qwen3_8 install,
        // locked to int4 / the Phase-1 group size / affine.
        let pleQuant = try #require(quant.pleNgram)
        #expect(pleQuant.weightBits == 4)
        #expect(pleQuant.groupSize == FinchQuantization.pleGroupSize)
        #expect(pleQuant.scheme == "affine")

        // Resident index: family identity + the I64 PLE metadata rides raw.
        let weightsData = try Data(contentsOf: URL(fileURLWithPath: out + "/model_weights.bin"))
        let (_, entries) = try weightsData.withUnsafeBytes { raw -> (
            FinchResidentIndexHeaderV1, [FinchResidentIndexEntryV1]
        ) in
            let h = try FinchResidentIndexCodec.decodeHeader(raw)
            return (h, try FinchResidentIndexCodec.decodeRegion(raw, header: h))
        }
        #expect(entries.count == t.residentEntryCount)
        let byName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })
        let hcNorm = try #require(
            byName["language_model.layers.1.attn_hyper_connection.hc_norm.weight"])
        #expect(hcNorm.dtype == 1 && hcNorm.shape == [UInt32(t.plane), 0, 0, 0])
        let i64Offsets = try #require(
            byName["language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        #expect(i64Offsets.dtype == 4)          // DType.i64 raw byte copy
        #expect(i64Offsets.shape == [UInt32(t.pleHeads), 0, 0, 0])
        #expect(i64Offsets.sizeBytes == UInt64(t.pleHeads * 8))
        let conv = try #require(byName["language_model.layers.1.ple.conv1d.weight"])
        #expect(conv.dtype == 2)                // fp16 squeezed
        #expect(conv.shape == [UInt32(t.plane * t.pleConvKernel), 0, 0, 0])
        let indexQK = try #require(
            byName["language_model.layers.3.self_attn.indexer.index_qk_proj.weight"])
        let indexRows = (t.indexerNumHeads + t.indexerKVHeads) * t.indexerHeadDim
        #expect(indexQK.dtype == 0 && indexQK.shape == [UInt32(indexRows), UInt32(t.D), 0, 0])
        #expect(byName["language_model.hyper_connection_mixer.hc_norm.weight"] != nil)
        // The 3.8 family has no model.norm / no inner model. stage.
        #expect(byName["language_model.model.norm.weight"] == nil)
    }

    /// Qwen 3.8 sources also carry a video preprocessor config; it is optional
    /// in the payload contract, and when present it ships with the install.
    @Test func qwen38VisionRepackWritesBothPreprocessorConfigs() async throws {
        let dir = NSTemporaryDirectory() + "qwen38-vision-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write38(into: dir, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack38(snapshotDir: dir, includeVision: true)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }
        let out = result.outputDir
        let fm = FileManager.default

        // Only the two MTP decoys remain excluded; the visual tensors are now
        // part of the install.
        #expect(result.excludedTensorCount == 2)
        #expect(fm.fileExists(atPath: out + "/vision/vision.safetensors"))
        #expect(fm.fileExists(atPath: out + "/vision/model_config.json"))
        #expect(fm.fileExists(atPath: out + "/vision/preprocessor_config.json"))
        #expect(fm.fileExists(atPath: out + "/vision/video_preprocessor_config.json"))

        let manifestData = try Data(contentsOf: URL(fileURLWithPath: out + "/manifest.json"))
        let manifest = try FinchManifestCodec.decode(manifestData)
        for path in ["vision/vision.safetensors",
                     "vision/model_config.json",
                     "vision/preprocessor_config.json",
                     "vision/video_preprocessor_config.json"] {
            #expect(manifest.files[path] != nil)
        }
        let verify = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputFinch: out))
        #expect(verify.unexpectedEntries.isEmpty)
    }

    @Test func qwen38RawNormMetadataAndBakedNormPayloads() async throws {
        let dir = NSTemporaryDirectory() + "qwen38-repack-bytes-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write38(into: dir)
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let result = try await Self.runRepack38(snapshotDir: dir)
        defer { try? FileManager.default.removeItem(atPath: result.outputDir) }
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: dir)

        let weightsData = try Data(contentsOf: URL(fileURLWithPath: result.outputDir + "/model_weights.bin"))
        let (_, entries) = try weightsData.withUnsafeBytes { raw -> (
            FinchResidentIndexHeaderV1, [FinchResidentIndexEntryV1]
        ) in
            let h = try FinchResidentIndexCodec.decodeHeader(raw)
            return (h, try FinchResidentIndexCodec.decodeRegion(raw, header: h))
        }
        let byName = Dictionary(uniqueKeysWithValues: entries.map { ($0.name, $0) })
        func payloadBytes(_ entry: FinchResidentIndexEntryV1) -> Data {
            weightsData.subdata(
                in: Data.Index(entry.fileOffset)
                    ..< Data.Index(entry.fileOffset + entry.sizeBytes))
        }

        // hc_norm.weight is a grouped-RMS gate the family's blanket
        // `norm.weight` rule covers: the gate multiplies (1 + w), so the
        // payload must be the source bf16 values with 1.0 added — NOT the
        // raw bytes. (Only linear_attn.norm.weight is multiplied raw.)
        let hcSource = try Self.readSourceTensor(
            snapshot, name: "model.language_model.hyper_connection_mixer.hc_norm.weight")
        let hcEntry = try #require(
            byName["language_model.hyper_connection_mixer.hc_norm.weight"])
        #expect(hcEntry.sizeBytes == UInt64(hcSource.count))
        let hcWritten = payloadBytes(hcEntry).withUnsafeBytes { raw -> [UInt16] in
            Array(raw.bindMemory(to: UInt16.self))
        }
        for i in 0..<(Int(hcSource.count) / 2) {
            let srcBits = UInt16(hcSource[2 * i]) | UInt16(hcSource[2 * i + 1]) << 8
            #expect(hcWritten[i] == FinchQuantization.bf16Bits(
                1.0 + FinchQuantization.bf16ToFloat(srcBits)))
        }

        // Indexer layernorms ARE (1 + w) baked (Qwen RMSNorm form).
        let lnSource = try Self.readSourceTensor(
            snapshot, name: "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight")
        let lnEntry = try #require(
            byName["language_model.layers.3.self_attn.indexer.k_layernorm.weight"])
        let written = payloadBytes(lnEntry).withUnsafeBytes { raw -> [UInt16] in
            Array(raw.bindMemory(to: UInt16.self))
        }
        for i in 0..<(Int(lnSource.count) / 2) {
            let srcBits = UInt16(lnSource[2 * i]) | UInt16(lnSource[2 * i + 1]) << 8
            #expect(written[i] == FinchQuantization.bf16Bits(
                1.0 + FinchQuantization.bf16ToFloat(srcBits)))
        }

        // PLE I64 hash metadata round-trips byte-exact (dtype 4, raw copy).
        let i64Source = try Self.readSourceTensor(
            snapshot,
            name: "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets")
        let i64Entry = try #require(
            byName["language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"])
        #expect(i64Entry.dtype == 4)
        #expect(payloadBytes(i64Entry) == i64Source)
    }
}

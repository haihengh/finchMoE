import CoreGraphics
import Foundation
import ImageIO
import Metal
import Testing
@testable import FinchMoE

/// The vision payload and config are new file formats, so their failure modes
/// are tested directly: a wrong header length, a non-BF16 tensor, a malformed
/// offset, and the exact geometry of the shipped tower.
@Suite struct VisionPayloadTests {

    // MARK: - Fixtures

    static func writeVisionConfig(into directory: URL,
                                  outHidden: Int = 2_048,
                                  textHidden: Int = 2_048,
                                  mropeSection: [Int] = [11, 11, 10]) throws {
        let visionDir = directory.appendingPathComponent("vision", isDirectory: true)
        try FileManager.default.createDirectory(at: visionDir,
                                                withIntermediateDirectories: true)
        let config: [String: Any] = [
            "image_token_id": 248_056,
            "video_token_id": 248_057,
            "vision_start_token_id": 248_053,
            "vision_end_token_id": 248_054,
            "vision_config": [
                "depth": 27,
                "hidden_size": 1_152,
                "intermediate_size": 4_304,
                "num_heads": 16,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "num_position_embeddings": 2_304,
                "out_hidden_size": outHidden,
                "in_channels": 3,
            ],
            "text_config": [
                "hidden_size": textHidden,
                "head_dim": 256,
                "rope_parameters": [
                    "mrope_interleaved": true,
                    "mrope_section": mropeSection,
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10_000_000,
                ],
            ],
        ]
        let preprocessor: [String: Any] = [
            "size": ["shortest_edge": 65_536, "longest_edge": 16_777_216],
            "patch_size": 16,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys])
            .write(to: visionDir.appendingPathComponent("model_config.json"))
        try JSONSerialization.data(withJSONObject: preprocessor, options: [.sortedKeys])
            .write(to: visionDir.appendingPathComponent("preprocessor_config.json"))
    }

    static func writePayload(into directory: URL,
                             header: [String: Any],
                             payload: Data) throws {
        let visionDir = directory.appendingPathComponent("vision", isDirectory: true)
        try FileManager.default.createDirectory(at: visionDir,
                                                withIntermediateDirectories: true)
        var data = Data()
        var length = UInt64(
            try JSONSerialization.data(withJSONObject: header).count).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        let headerData = try JSONSerialization.data(withJSONObject: header,
                                                    options: [.sortedKeys])
        // The writer pads the header to 8 bytes; the parser does not require
        // it, but matching the producer keeps this fixture realistic.
        var padded = headerData
        while (8 + padded.count) % 8 != 0 { padded.append(0x20) }
        // Recompute the length after padding.
        data.removeAll(keepingCapacity: true)
        var paddedLength = UInt64(padded.count).littleEndian
        withUnsafeBytes(of: &paddedLength) { data.append(contentsOf: $0) }
        data.append(padded)
        data.append(payload)
        try data.write(to: visionDir.appendingPathComponent("vision.safetensors"))
    }

    private func makeDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finch-vision-\(UUID().uuidString)",
                                    isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    // MARK: - Config

    @Test func configParsesTheShippedGeometry() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeVisionConfig(into: dir)
        let config = try VisionConfig.load(
            directory: FinchModelDirectory(rootURL: dir))
        #expect(config.geometry.depth == 27)
        #expect(config.geometry.hiddenSize == 1_152)
        #expect(config.geometry.headDim == 72)
        #expect(config.geometry.gridSide == 48)
        #expect(config.textHiddenSize == 2_048)
        #expect(config.imageTokenId == 248_056)
        #expect(config.visionStartTokenId == 248_053)
        #expect(config.textRotaryDim == 64)
        #expect(config.mropeSection == [11, 11, 10])
    }

    @Test func configRejectsAMismatchedMergerWidth() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeVisionConfig(into: dir, outHidden: 2_560)
        // The text hidden size in the fixture stays 2048; a 2560 merger is the
        // 3.8 geometry and must not be accepted as the 3.6 one.
        do {
            _ = try VisionConfig.load(directory: FinchModelDirectory(rootURL: dir))
            Issue.record("accepted a merger width that disagrees with the text model")
        } catch let error as VisionError {
            #expect("\(error)".contains("out_hidden_size"))
        }
    }

    @Test func configRejectsAnIncompleteMropeSection() throws {
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writeVisionConfig(into: dir, mropeSection: [11, 11, 9])
        do {
            _ = try VisionConfig.load(directory: FinchModelDirectory(rootURL: dir))
            Issue.record("accepted an mRoPE section that does not cover 32 pairs")
        } catch let error as VisionError {
            #expect("\(error)".contains("mRoPE"))
        }
    }

    // MARK: - Safetensors

    @Test func payloadMapsTensorsAndTheirBytes() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let payload = Data([0x00, 0x3C, 0x00, 0x40, 0x00, 0x3C, 0x00, 0x40])
        try Self.writePayload(into: dir, header: [
            "model.visual.pos_embed.weight": [
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offsets": [0, 8],
            ],
        ], payload: payload)

        let weights = try VisionWeights(
            directory: FinchModelDirectory(rootURL: dir), device: device)
        let tensor = try weights.tensor("model.visual.pos_embed.weight")
        #expect(tensor.shape == [2, 2])
        #expect(tensor.offset == 0)
        #expect(tensor.sizeBytes == 8)
        let bytes = weights.buffer.contents()
            .advanced(by: tensor.offset)
            .bindMemory(to: UInt8.self, capacity: 8)
        #expect((0..<8).map { bytes[$0] } == Array(payload))
    }

    @Test func payloadRejectsANonBF16Tensor() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePayload(into: dir, header: [
            "bad": ["dtype": "F32", "shape": [2], "data_offsets": [0, 8]],
        ], payload: Data(count: 8))
        #expect(throws: VisionError.self) {
            _ = try VisionWeights(
                directory: FinchModelDirectory(rootURL: dir), device: device)
        }
    }

    @Test func payloadRejectsAShapeThatDoesNotFitItsOffsets() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Self.writePayload(into: dir, header: [
            "bad": ["dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 8]],
        ], payload: Data(count: 8))
        #expect(throws: VisionError.self) {
            _ = try VisionWeights(
                directory: FinchModelDirectory(rootURL: dir), device: device)
        }
    }

    @Test func payloadRejectsAHeaderBeyondTheFile() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let dir = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        var data = Data()
        var length = UInt64(1 << 20).littleEndian
        withUnsafeBytes(of: &length) { data.append(contentsOf: $0) }
        data.append(Data(count: 16))
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("vision"), withIntermediateDirectories: true)
        try data.write(to: dir.appendingPathComponent("vision/vision.safetensors"))
        #expect(throws: VisionError.self) {
            _ = try VisionWeights(
                directory: FinchModelDirectory(rootURL: dir), device: device)
        }
    }
}

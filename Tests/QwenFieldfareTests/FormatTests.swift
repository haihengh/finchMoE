import XCTest
@testable import QwenFieldfareFormat
@testable import QwenFieldfareRepack

final class FormatTests: XCTestCase {

    func testMagicBytes() {
        XCTAssertEqual(QTurboFormatV1.magic.count, 8)
        XCTAssertEqual(QTurboFormatV1.magic, Array("QTURBO1\0".utf8))
        XCTAssertEqual(QTurboFormatV1.pageSize, 16384)
        XCTAssertEqual(QTurboFormatV1.expertBlobAlignment, 16384)
    }

    func testAlignment() {
        XCTAssertEqual(QTurboFormatV1.align(0, to: 16384), 0)
        XCTAssertEqual(QTurboFormatV1.align(1, to: 16384), 16384)
        XCTAssertEqual(QTurboFormatV1.align(16384, to: 16384), 16384)
        XCTAssertEqual(QTurboFormatV1.align(16385, to: 16384), 32768)
        XCTAssertEqual(QTurboFormatV1.align(100, to: 64), 128)
    }

    func testPackedExpertFilename() {
        XCTAssertEqual(QTurboFormatV1.packedExpertFilename(layer: 0), "layer_00.bin")
        XCTAssertEqual(QTurboFormatV1.packedExpertFilename(layer: 3), "layer_03.bin")
        XCTAssertEqual(QTurboFormatV1.packedExpertFilename(layer: 47), "layer_47.bin")
    }

    func testDefaultConfig() {
        let c = QTurboModelConfig()
        XCTAssertEqual(c.hiddenSize, 2048)
        XCTAssertEqual(c.numHiddenLayers, 48)
        XCTAssertEqual(c.numExperts, 128)
        XCTAssertEqual(c.numExpertsPerTok, 8)
        XCTAssertEqual(c.qDim, 4096)          // 32 * 128
        XCTAssertEqual(c.kvDim, 512)          // 4 * 128
        XCTAssertEqual(c.gqaGroupSize, 8)     // 32 / 4
        XCTAssertEqual(c.vocabSize, 151936)
        XCTAssertEqual(c.ropeTheta, 1_000_000.0)
    }

    func testExpertLayout() {
        let c = QTurboModelConfig()
        let layout = QTurboRepackPlanner.expertLayout(config: c)

        // 9 sub-tensors: gate/up/down × weight/scales/biases
        XCTAssertEqual(layout.subTensors.count, 9)

        // gate_proj.weight: [768, 2048] int4 → 768*2048/2 = 786432 bytes
        let gw = layout.subTensor("gate_proj.weight")!
        XCTAssertEqual(gw.length, 768 * 2048 / 2)
        XCTAssertEqual(gw.offset, 0)

        // gate_proj.scales: [768, 2048/64=32] fp16 → 768*32*2 = 49152
        let gs = layout.subTensor("gate_proj.scales")!
        XCTAssertEqual(gs.length, 768 * 32 * 2)

        // down_proj.weight: [2048, 768] int4 → 2048*768/2 = 786432
        let dw = layout.subTensor("down_proj.weight")!
        XCTAssertEqual(dw.length, 2048 * 768 / 2)

        // down_proj.scales: [2048, 768/64=12] fp16 → 2048*12*2 = 49152
        let ds = layout.subTensor("down_proj.scales")!
        XCTAssertEqual(ds.length, 2048 * 12 * 2)

        // Blob size and aligned stride.
        XCTAssertEqual(layout.blobSize, gw.length * 2 /*gate,up weight*/ + gs.length * 4 /*gate,up scales+biases*/
                       + dw.length + ds.length * 2)
        XCTAssertEqual(layout.alignedStride % QTurboFormatV1.expertBlobAlignment, 0)
        XCTAssertGreaterThanOrEqual(layout.alignedStride, layout.blobSize)
    }

    func testTensorClassification() {
        // Routed expert.
        let e = QTurboRepackPlanner.classify("model.layers.5.mlp.experts.42.gate_proj.weight")
        if case let .expert(layer, expert, kind) = e {
            XCTAssertEqual(layer, 5)
            XCTAssertEqual(expert, 42)
            XCTAssertEqual(kind, "gate_proj.weight")
        } else {
            XCTFail("expected expert classification")
        }

        // Resident tensors.
        XCTAssertEqual(QTurboRepackPlanner.classify("model.embed_tokens.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("lm_head.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("model.norm.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("model.layers.0.self_attn.q_proj.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("model.layers.0.mlp.shared_expert.gate_proj.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("model.layers.0.mlp.gate.weight"), .resident)
        XCTAssertEqual(QTurboRepackPlanner.classify("model.layers.0.input_layernorm.weight"), .resident)
    }

    func testManifestRoundTrip() throws {
        let c = QTurboModelConfig()
        let layout = QTurboRepackPlanner.expertLayout(config: c)
        let manifest = QTurboManifestV1(
            config: c,
            tensors: [QTurboTensorEntry(name: "model.norm.weight", dtype: .fp16,
                                        shape: [2048], offset: 64, length: 4096)],
            residentBlobSize: 4160,
            expertLayout: layout,
            expertFiles: [QTurboExpertFileInfo(layer: 0, filename: "layer_00.bin",
                                               numExperts: 128, fileSize: layout.alignedStride * 128)]
        )
        let data = try manifest.encoded()
        let decoded = try QTurboManifestV1.decode(from: data)
        XCTAssertEqual(decoded, manifest)
        XCTAssertNotNil(decoded.tensor(named: "model.norm.weight"))
    }
}

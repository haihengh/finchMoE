import FinchMoE
import FinchMoEVision
import Foundation
import Metal
import Testing
@testable import FinchMoEAppCore
@testable import FinchMoEVision

/// The prompt builder's mRoPE bookkeeping, independent of the tower: an image
/// advances the timeline by the merged grid width, not by its placeholder
/// count, and each image token carries its own (t, h, w) triple.
@Suite struct VisionPromptBuilderTests {

    private static var modelPath: String {
        ProcessInfo.processInfo.environment["FINCH_VISION_MODEL"]
            ?? FileManager.default.currentDirectoryPath
                + "/models/Qwen3.6-35B-A3B-3bit-vision.finch"
    }

    private static var modelExists: Bool {
        FileManager.default.fileExists(atPath: modelPath + "/manifest.json")
    }

    @Test(.enabled(if: modelExists))
    func imagePositionsUseTheMergedGridTimeline() async throws {
        let modelURL = URL(fileURLWithPath: Self.modelPath)
        let device = try #require(MTLCreateSystemDefaultDevice())
        let queue = try #require(device.makeCommandQueue())
        let tokenizer = try await GFTokenizer.load(forModelDirectory: modelURL)
        let directory = try FinchModelDirectory(rootURL: modelURL)
        let config = try VisionConfig.load(directory: directory)

        let hidden = config.textHiddenSize
        let tokenCount = 4
        let buffer = try #require(device.makeBuffer(
            length: tokenCount * hidden * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let attachment = AppImageAttachment(mediaType: "image/jpeg",
                                            data: Data([0x01]))
        let embedding = VisionEngine.ImageEmbedding(
            embeddings: buffer,
            tokenCount: tokenCount,
            gridH: 4, gridW: 4,          // merged grid is 2×2
            pixelWidth: 64, pixelHeight: 64,
            textHiddenSize: hidden,
            mropeSection: config.mropeSection)

        let plan = try AppVisionPromptBuilder.build(
            history: [],
            prompt: "Describe.",
            images: [attachment],
            tokenCounts: [attachment.id: tokenCount],
            embeddings: [attachment.id: embedding],
            tokenizer: tokenizer,
            imageTokenID: Int32(config.imageTokenId),
            device: device,
            queue: queue)

        let runs = AppVisionPromptBuilder.imageRuns(
            in: plan.promptIDs, tokenID: Int32(config.imageTokenId))
        #expect(runs.count == 1)
        let run = try #require(runs.first)
        #expect(run.count == tokenCount)
        #expect(plan.multimodal.positions.count == plan.promptIDs.count)
        #expect(plan.multimodal.hiddenRowPositions.count == tokenCount)

        let base = plan.multimodal.positions[run.lowerBound].temporal
        #expect(plan.multimodal.positions[run.lowerBound].height == base)
        #expect(plan.multimodal.positions[run.lowerBound].width == base)
        #expect(plan.multimodal.positions[run.lowerBound + 1].height == base)
        #expect(plan.multimodal.positions[run.lowerBound + 1].width == base + 1)
        #expect(plan.multimodal.positions[run.lowerBound + 2].height == base + 1)
        #expect(plan.multimodal.positions[run.lowerBound + 2].width == base)

        // The turn continues after max(2, 2) positions, not after four tokens.
        if run.upperBound < plan.multimodal.positions.count {
            #expect(plan.multimodal.positions[run.upperBound].temporal == base + 2)
        }
        let last = plan.multimodal.positions.last?.temporal ?? 0
        #expect(plan.multimodal.decodePositionOffset
                == Int(last + 1) - plan.promptIDs.count)

        // And the placeholder rows point at the image's absolute positions.
        #expect(plan.multimodal.hiddenRowPositions
            == (0..<tokenCount).map { Int32(run.lowerBound + $0) })
    }
}

import FinchMoE
import Foundation
import Metal

/// Public facade over the vision tower for the app layers.
///
/// `FinchMoEAppCore` owns prompt construction and the text runtime; this type
/// keeps the tower, preprocessing, and payload details on the vision side.
public final class VisionEngine {
    public struct ImageEmbedding {
        /// FP16 `[tokenCount, textHiddenSize]`, merge-block order.
        public let embeddings: MTLBuffer
        public let tokenCount: Int
        /// Patch grid before the 2×2 merge.
        public let gridH: Int
        public let gridW: Int
        public let pixelWidth: Int
        public let pixelHeight: Int
        /// Text-model hidden width of each row.
        public let textHiddenSize: Int
        /// Interleaved mRoPE sections for this checkpoint.
        public let mropeSection: [Int]

        public var mergedGridH: Int { gridH / 2 }
        public var mergedGridW: Int { gridW / 2 }
    }

    public let imageTokenID: Int32
    public let visionStartTokenID: Int32
    public let visionEndTokenID: Int32
    /// Interleaved mRoPE sections, e.g. `[11, 11, 10]`.
    public let mropeSection: [Int]
    public let textHiddenSize: Int

    private let tower: VisionTower

    /// Whether `directory` carries the vision payload the engine needs.
    public static func isAvailable(modelDirectory: URL,
                                   fileManager: FileManager = .default) -> Bool {
        fileManager.fileExists(atPath: modelDirectory
            .appendingPathComponent("vision/vision.safetensors").path)
            && fileManager.fileExists(atPath: modelDirectory
                .appendingPathComponent("vision/model_config.json").path)
            && fileManager.fileExists(atPath: modelDirectory
                .appendingPathComponent("vision/preprocessor_config.json").path)
    }

    public init(modelDirectory: URL,
                device: MTLDevice,
                queue: MTLCommandQueue) throws {
        let directory = try FinchModelDirectory(rootURL: modelDirectory)
        self.tower = try VisionTower(directory: directory,
                                     device: device,
                                     queue: queue)
        self.imageTokenID = Int32(tower.config.imageTokenId)
        self.visionStartTokenID = Int32(tower.config.visionStartTokenId)
        self.visionEndTokenID = Int32(tower.config.visionEndTokenId)
        self.mropeSection = tower.config.mropeSection
        self.textHiddenSize = tower.config.textHiddenSize
    }

    /// Preprocess and encode one image. The returned buffer stays valid as long
    /// as the caller (or an in-flight generation) retains it.
    public func encode(imageData: Data,
                       maxPixels: Int = 1_536 * 1_536) throws -> ImageEmbedding {
        let run = try tower.encode(imageData: imageData, maxPixels: maxPixels)
        return ImageEmbedding(embeddings: run.image.embeddings,
                              tokenCount: run.image.tokenCount,
                              gridH: run.image.gridH,
                              gridW: run.image.gridW,
                              pixelWidth: run.image.pixelWidth,
                              pixelHeight: run.image.pixelHeight,
                              textHiddenSize: tower.config.textHiddenSize,
                              mropeSection: tower.config.mropeSection)
    }
}

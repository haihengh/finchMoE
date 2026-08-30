import Foundation

/// Model hyper-parameters for Qwen3-30B-A3B (mirrors the relevant fields of
/// HuggingFace `config.json`).
public struct QTurboModelConfig: Codable, Sendable, Equatable {
    public var hiddenSize: Int
    public var headDim: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var numHiddenLayers: Int
    public var numExperts: Int
    public var numExpertsPerTok: Int
    public var moeIntermediateSize: Int
    public var sharedExpertIntermediateSize: Int
    public var vocabSize: Int
    public var ropeTheta: Float
    public var rmsNormEps: Float
    public var maxPositionEmbeddings: Int
    public var normTopkProb: Bool
    public var quantGroupSize: Int
    public var quantBits: Int
    public var bosTokenId: Int
    public var eosTokenId: Int

    public init(
        hiddenSize: Int = 2048,
        headDim: Int = 128,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 4,
        numHiddenLayers: Int = 48,
        numExperts: Int = 128,
        numExpertsPerTok: Int = 8,
        moeIntermediateSize: Int = 768,
        sharedExpertIntermediateSize: Int = 6144,
        vocabSize: Int = 151936,
        ropeTheta: Float = 1_000_000.0,
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 40960,
        normTopkProb: Bool = true,
        quantGroupSize: Int = 64,
        quantBits: Int = 4,
        bosTokenId: Int = 151643,
        eosTokenId: Int = 151645
    ) {
        self.hiddenSize = hiddenSize
        self.headDim = headDim
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numHiddenLayers = numHiddenLayers
        self.numExperts = numExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.vocabSize = vocabSize
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.normTopkProb = normTopkProb
        self.quantGroupSize = quantGroupSize
        self.quantBits = quantBits
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
    }

    /// Q projection output dimension = numAttentionHeads * headDim.
    public var qDim: Int { numAttentionHeads * headDim }
    /// K/V projection output dimension = numKeyValueHeads * headDim.
    public var kvDim: Int { numKeyValueHeads * headDim }
    /// Number of Q heads served by each KV head (GQA group size).
    public var gqaGroupSize: Int { numAttentionHeads / numKeyValueHeads }
}

/// Describes a single resident tensor stored in `model_weights.bin`.
public struct QTurboTensorEntry: Codable, Sendable, Equatable {
    public var name: String
    public var dtype: QTurboDType
    public var shape: [Int]
    /// Byte offset of the tensor payload within `model_weights.bin`.
    public var offset: Int
    /// Byte length of the tensor payload.
    public var length: Int

    public init(name: String, dtype: QTurboDType, shape: [Int], offset: Int, length: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.length = length
    }
}

/// Describes the sub-tensors that make up a single expert within a packed
/// per-layer file. Offsets are relative to the start of the expert blob.
public struct QTurboExpertSubTensor: Codable, Sendable, Equatable {
    public var kind: String   // e.g. "gate_proj.weight", "down_proj.scales"
    public var dtype: QTurboDType
    public var shape: [Int]
    public var offset: Int    // relative to expert blob start
    public var length: Int

    public init(kind: String, dtype: QTurboDType, shape: [Int], offset: Int, length: Int) {
        self.kind = kind
        self.dtype = dtype
        self.shape = shape
        self.offset = offset
        self.length = length
    }
}

/// Layout for a single expert blob (shared by every expert in every layer,
/// since all experts have identical shapes).
public struct QTurboExpertLayout: Codable, Sendable, Equatable {
    /// Ordered list of sub-tensors within one expert blob.
    public var subTensors: [QTurboExpertSubTensor]
    /// Total (unpadded) size of one expert blob in bytes.
    public var blobSize: Int
    /// Page-aligned stride between consecutive experts in a packed file.
    public var alignedStride: Int

    public init(subTensors: [QTurboExpertSubTensor], blobSize: Int, alignedStride: Int) {
        self.subTensors = subTensors
        self.blobSize = blobSize
        self.alignedStride = alignedStride
    }

    /// Returns the sub-tensor for a given kind, or nil.
    public func subTensor(_ kind: String) -> QTurboExpertSubTensor? {
        subTensors.first { $0.kind == kind }
    }
}

/// Per-layer packed expert file metadata.
public struct QTurboExpertFileInfo: Codable, Sendable, Equatable {
    public var layer: Int
    public var filename: String
    public var numExperts: Int
    public var fileSize: Int

    public init(layer: Int, filename: String, numExperts: Int, fileSize: Int) {
        self.layer = layer
        self.filename = filename
        self.numExperts = numExperts
        self.fileSize = fileSize
    }
}

/// Top-level `.qturbo` manifest, serialized as `manifest.json`.
public struct QTurboManifestV1: Codable, Sendable, Equatable {
    public var formatVersion: UInt32
    public var config: QTurboModelConfig
    /// Resident tensor index (name → placement in model_weights.bin).
    public var tensors: [QTurboTensorEntry]
    /// Size of model_weights.bin (including header/magic) in bytes.
    public var residentBlobSize: Int
    /// Shared layout for every expert blob.
    public var expertLayout: QTurboExpertLayout
    /// Per-layer expert file info.
    public var expertFiles: [QTurboExpertFileInfo]

    public init(
        formatVersion: UInt32 = QTurboFormatV1.formatVersion,
        config: QTurboModelConfig,
        tensors: [QTurboTensorEntry],
        residentBlobSize: Int,
        expertLayout: QTurboExpertLayout,
        expertFiles: [QTurboExpertFileInfo]
    ) {
        self.formatVersion = formatVersion
        self.config = config
        self.tensors = tensors
        self.residentBlobSize = residentBlobSize
        self.expertLayout = expertLayout
        self.expertFiles = expertFiles
    }

    // MARK: - Lookups

    /// Fast tensor lookup by name.
    public func tensor(named name: String) -> QTurboTensorEntry? {
        tensors.first { $0.name == name }
    }

    // MARK: - Serialization

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(self)
    }

    public static func decode(from data: Data) throws -> QTurboManifestV1 {
        try JSONDecoder().decode(QTurboManifestV1.self, from: data)
    }

    public func write(to url: URL) throws {
        try encoded().write(to: url, options: .atomic)
    }

    public static func load(from url: URL) throws -> QTurboManifestV1 {
        try decode(from: Data(contentsOf: url))
    }
}

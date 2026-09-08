import Foundation
import Metal
import Darwin
import FinchMoEFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.fqturbo/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }
    /// GDN linear-attention projection width (`linear_attn.in_proj_qkv/z/a/b`,
    /// `out_proj`). 8 on the production build; raw/bf16 installs have no quant
    /// manifest and never consult this (default 4 is inert there).
    public var linearAttentionWeightBits: Int { manifest.quant?.linearAttention.weightBits ?? 4 }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL
    let modelDirectory: FQTurboModelDirectory

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: FQTurboModelDirectory) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "finchmoe.expert-streamers")
    }

    // MARK: - Resident accessors

    public var embedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens.weight")
    }

    /// Gemma 4 ties lm_head to the embedding (the transpose for the GEMV path
    /// is the kernel's job). Qwen 3.6 has an untied `lm_head.weight`.
    public var lmHead: TensorView {
        switch config.modelFamily {
        case "qwen3_6": return try! resident(name: "lm_head.weight")
        default:        return embedding
        }
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.weight")
    }
    /// Router weight. Gemma writer emits `.router.proj.weight` (no `.mlp.`
    /// segment); Qwen 3.6 uses `.mlp.gate.weight`.
    public func router(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case "qwen3_6":
            return try resident(name: "language_model.model.layers.\(L).mlp.gate.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).router.proj.weight")
        }
    }
    /// Shared-expert FFN. Gemma writer emits `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment; Qwen 3.6 keeps the full
    /// `.mlp.shared_expert.{gate,up,down}_proj.weight` names.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case "qwen3_6":
            return try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.gate_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.gate_proj.weight")
        }
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case "qwen3_6":
            return try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.up_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.up_proj.weight")
        }
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case "qwen3_6":
            return try resident(name: "language_model.model.layers.\(L).mlp.shared_expert.down_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.down_proj.weight")
        }
    }

    // MARK: - Qwen 3.6 GDN (linear-attention) accessors
    //
    // GDN layers replace `self_attn.*` with `linear_attn.*`; the shared
    // expert gate is the sigmoid scalar `mlp.shared_expert_gate.weight` [1, D].
    // All are Qwen-only — touching them on a Gemma install throws
    // `tensorNotFound`.

    private func qwenResident(_ suffix: String, layer L: Int) throws -> TensorView {
        guard config.modelFamily == "qwen3_6" else {
            throw ModelError.tensorNotFound(name: "language_model.model.layers.\(L).\(suffix) (qwen3_6-only)")
        }
        return try resident(name: "language_model.model.layers.\(L).\(suffix)")
    }

    public func gdnInProjQKV(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_qkv.weight", layer: L)
    }
    public func gdnInProjZ(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_z.weight", layer: L)
    }
    public func gdnInProjA(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_a.weight", layer: L)
    }
    public func gdnInProjB(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_b.weight", layer: L)
    }
    public func gdnOutProj(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.out_proj.weight", layer: L)
    }
    public func gdnALog(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.A_log", layer: L)
    }
    public func gdnDtBias(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.dt_bias", layer: L)
    }
    public func gdnNormWeight(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.norm.weight", layer: L)
    }
    public func gdnConv1D(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.conv1d.weight", layer: L)
    }
    public func sharedExpertGateProj(layer L: Int) throws -> TensorView {
        try qwenResident("mlp.shared_expert_gate.weight", layer: L)
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).input_layernorm.weight")
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_attention_layernorm.weight")
    }
    public var finalNorm: TensorView {
        try! resident(name: "language_model.model.norm.weight")
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        func checkedRelativeOffset(_ absolute: UInt64,
                                   size: UInt64,
                                   field: String) throws -> UInt64 {
            if size == 0 {
                guard absolute == 0 else {
                    throw ModelError.indexCorrupt(detail: "\(name).\(field) has an absent nonzero offset")
                }
                return 0
            }
            guard absolute >= residentFileOffset else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) precedes the resident payload")
            }
            let relative = absolute - residentFileOffset
            guard relative <= residentIndex.header.residentSize,
                  size <= residentIndex.header.residentSize - relative else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) exceeds the resident payload")
            }
            return relative
        }
        let relativeOffset = try checkedRelativeOffset(
            entry.fileOffset, size: entry.sizeBytes, field: "weights")
        let scaleRel = try checkedRelativeOffset(
            entry.scaleOffset, size: entry.scaleSize, field: "scales")
        let biasRel = try checkedRelativeOffset(
            entry.biasOffset, size: entry.biasSize, field: "biases")
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: FQTurboFormatV1.DType.u32.rawValue)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: layerFD,
                                              named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy,
            fileDescriptor: layerFD)
        streamersBox.layerVerified[L] = true
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.fqturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig = .gemma4_26B_A4B,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256
        let modelDirectory = try FQTurboModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do { manifestFD = try modelDirectory.openFile("manifest.json") }
        catch ModelError.missingFile { throw ModelError.partialInstall(path: directoryURL.path) }
        defer { close(manifestFD) }
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let receiptFD: Int32
            do {
                receiptFD = try modelDirectory.openFile(VerifiedInstallReceiptReader.fileName)
            } catch ModelError.missingFile {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName) is missing")
            }
            defer { close(receiptFD) }
            let receiptData = try modelDirectory.readMetadata(
                fileDescriptor: receiptFD,
                relativePath: VerifiedInstallReceiptReader.fileName,
                maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
            let loadedReceipt = try VerifiedInstallReceiptReader.decode(data: receiptData)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
        defer { close(layoutFD) }
        let layoutData = try modelDirectory.readMetadata(
            fileDescriptor: layoutFD, relativePath: "packed_experts/layout.json",
            maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
        guard UInt64(layoutData.count) == layoutEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "packed_experts/layout.json",
                expected: layoutEntry.size,
                actual: UInt64(layoutData.count))
        }
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(fileDescriptor: weightsFD,
                                      named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        guard Sha256Verifier.hashData(layoutData).lowercased()
                == layoutEntry.sha256.lowercased() else {
            throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
        }
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let layout = try PackedExpertsLayoutReader.decode(data: layoutData,
                                                          manifest: manifest)
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(modelDirectory: modelDirectory,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  manifest: manifest,
                                  config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(fileSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: weightsFD)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    private static func validateTrustedReceiptLayerLayout(modelDirectory: FQTurboModelDirectory,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
            }
            return value
        }

        func dimensions(_ rows: Int, _ columns: Int, field: String) throws -> (UInt32, UInt32) {
            guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                  r > 0, c > 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
            }
            return (r, c)
        }

        func requireBF16(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == FQTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required BF16 schema")
            }
        }

        /// Raw (unquantized) 1-D resident tensor of a fixed dtype — used for
        /// the Qwen GDN scalars (`A_log`/`dt_bias` fp32) and the conv1d weight
        /// (fp16; the writer converts bf16 → fp16 at emit).
        func requireRaw(_ name: String, count: Int, dtype: FQTurboFormatV1.DType) throws {
            precondition(dtype == .fp16 || dtype == .fp32,
                         "requireRaw supports only fp16/fp32")
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let stride = dtype == .fp32 ? MemoryLayout<Float>.size : MemoryLayout<Float16>.size
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(stride), field: name)
            guard entry.dtype == dtype.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(stride) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required raw \(dtype) schema")
            }
        }

        func affineSizes(rows: Int,
                         columns: Int,
                         slot: ManifestQuantSlot,
                         field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
            let shape = try dimensions(rows, columns, field: field)
            guard slot.weightBits == 4 || slot.weightBits == 8,
                  slot.groupSize > 0,
                  columns % slot.groupSize == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
            }
            let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
            let bitCount = try checkedMultiply(elements, UInt64(slot.weightBits), field: field)
            guard bitCount % 8 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
            }
            let groups = UInt64(columns / slot.groupSize)
            let auxElements = try checkedMultiply(UInt64(shape.0), groups, field: field)
            let auxBytes = try checkedMultiply(
                auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
            return (shape, bitCount / 8, auxBytes)
        }

        func requireAffine(_ name: String,
                           rows: Int,
                           columns: Int,
                           slot: ManifestQuantSlot) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let expected = try affineSizes(
                rows: rows, columns: columns, slot: slot, field: name)
            let primaryAlignment: UInt64 = slot.weightBits == 4
                ? UInt64(MemoryLayout<UInt16>.alignment)
                : 1
            guard entry.dtype == FQTurboFormatV1.DType.u32.rawValue,
                  entry.shape.0 == expected.shape.0,
                  entry.shape.1 == expected.shape.1,
                  entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expected.weight,
                  entry.scaleSize == expected.aux,
                  entry.biasSize == expected.aux,
                  entry.fileOffset % primaryAlignment == 0,
                  entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                  entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
            }
        }

        try requireAffine(
            "language_model.model.embed_tokens.weight",
            rows: config.vocabSize,
            columns: config.hiddenSize,
            slot: quant.embedding)
        try requireBF16("language_model.model.norm.weight", count: config.hiddenSize)

        // Per-layer tensor sets diverge by family: Gemma 4 has the
        // q/k/v sandwich norms + router auxiliaries; Qwen 3.6 has GDN
        // (linear_attn.*) on the non-full layers, a doubled q_proj + output
        // gate on the full layers, and a sigmoid-gated shared expert. The
        // routed-expert packed layout below is family-independent.
        switch config.modelFamily {
        case "qwen3_6":
            try validateQwen36Layers(config: config, quant: quant,
                                     requireBF16: requireBF16,
                                     requireAffine: requireAffine,
                                     requireRaw: requireRaw,
                                     checkedIntMultiply: checkedIntMultiply)
        default:
            try validateGemma4Layers(config: config, quant: quant,
                                     requireBF16: requireBF16,
                                     requireAffine: requireAffine,
                                     checkedIntMultiply: checkedIntMultiply)
        }

        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }

    // MARK: - Per-layer schema validators (split by model family)

    private static func validateGemma4Layers(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, "layer \(layer) key/value")

            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                try requireBF16("\(prefix).\(name)", config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", headDimension)
            try requireBF16("\(prefix).self_attn.k_norm.weight", headDimension)
            try requireBF16("\(prefix).router.per_expert_scale", config.numExperts)
            try requireBF16("\(prefix).layer_scalar", 1)

            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              queryDimension, config.hiddenSize,
                              quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              kvDimension, config.hiddenSize,
                              quant.attention)
            if !isFull {
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  kvDimension, config.hiddenSize,
                                  quant.attention)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension,
                              quant.attention)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              config.hiddenSize, config.intermediateSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).router.proj.weight",
                              config.numExperts, config.hiddenSize,
                              quant.router)
        }
    }

    private static func validateQwen36Layers(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        requireRaw: (String, Int, FQTurboFormatV1.DType) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let keyDim = try checkedIntMultiply(
            config.linearNumKeyHeads, config.linearKeyHeadDim,
            "GDN key dim")
        let valueDim = try checkedIntMultiply(
            config.linearNumValueHeads, config.linearValueHeadDim,
            "GDN value dim")
        let qkvDim = try checkedIntMultiply(keyDim, 2, "GDN q+k dim")
            + valueDim
        let convCount = try checkedIntMultiply(
            qkvDim, config.linearConvKernelDim, "GDN conv weight")

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0

            try requireBF16("\(prefix).input_layernorm.weight", config.hiddenSize)
            try requireBF16("\(prefix).post_attention_layernorm.weight", config.hiddenSize)

            if isFull {
                // q_proj is doubled for attn_output_gate: [2*Q*head_dim, D].
                let queryRows = try checkedIntMultiply(
                    config.numHeads, config.fullHeadDim, "layer \(layer) query")
                let doubledQuery = try checkedIntMultiply(
                    queryRows, 2, "layer \(layer) doubled query")
                let kvRows = try checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim,
                    "layer \(layer) key/value")

                try requireAffine("\(prefix).self_attn.q_proj.weight",
                                  doubledQuery, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  kvRows, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  kvRows, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.o_proj.weight",
                                  config.hiddenSize, queryRows,
                                  quant.attention)
                try requireBF16("\(prefix).self_attn.q_norm.weight", config.fullHeadDim)
                try requireBF16("\(prefix).self_attn.k_norm.weight", config.fullHeadDim)
            } else {
                // GDN (linear-attention) layer. The five projections ride the
                // dedicated linearAttention slot (8-bit on the production
                // build; the recurrent state amplifies their quant noise).
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  qkvDim, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  valueDim, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  config.linearNumValueHeads, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  config.linearNumValueHeads, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  config.hiddenSize, valueDim,
                                  quant.linearAttention)
                try requireBF16("\(prefix).linear_attn.norm.weight",
                                config.linearValueHeadDim)
                try requireRaw("\(prefix).linear_attn.A_log",
                               config.linearNumValueHeads, .fp32)
                try requireRaw("\(prefix).linear_attn.dt_bias",
                               config.linearNumValueHeads, .fp32)
                // conv1d.weight is [qkvDim, 1, kernel] in the checkpoint; the
                // writer emits the squeezed [qkvDim, kernel] rows as raw FP16
                // (bf16 → fp16 conversion at emit).
                try requireRaw("\(prefix).linear_attn.conv1d.weight",
                               convCount, .fp16)
            }

            // Shared expert + sigmoid gate + router (both layer types).
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              config.hiddenSize, config.intermediateSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                              1, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.gate.weight",
                              config.numExperts, config.hiddenSize,
                              quant.router)
        }
    }

}

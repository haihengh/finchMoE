import Foundation
import Metal
import QwenFieldfareFormat

/// Loads a `.qturbo` model: reads the manifest, memory-maps `model_weights.bin`
/// and wraps it in a zero-copy `MTLBuffer`, builds the expert streamer and the
/// KV cache, and compiles the Metal pipelines.
public final class Model {

    public enum ModelError: Error, CustomStringConvertible {
        case manifestMissing(String)
        case residentMissing(String)
        case mmapFailed(String)
        case bufferCreationFailed
        case badMagic
        case tensorMissing(String)

        public var description: String {
            switch self {
            case .manifestMissing(let s): return "Model: manifest not found at \(s)"
            case .residentMissing(let s): return "Model: \(s) not found"
            case .mmapFailed(let s): return "Model: mmap failed for \(s)"
            case .bufferCreationFailed: return "Model: could not create Metal buffer from mmap"
            case .badMagic: return "Model: model_weights.bin has bad magic"
            case .tensorMissing(let s): return "Model: resident tensor missing — \(s)"
            }
        }
    }

    public let modelDir: URL
    public let manifest: QTurboManifestV1
    public var config: QTurboModelConfig { manifest.config }

    public let metal: MetalContext
    public let pipelines: KernelPipelines
    public let kvCache: KVCacheManager
    public let streamer: PreadExpertStreamer

    /// Zero-copy Metal buffer backing the entire model_weights.bin file.
    public let residentBuffer: MTLBuffer

    private let mmapBase: UnsafeMutableRawPointer
    private let mmapLength: Int
    private let mappedFD: Int32

    /// Fast tensor-name → entry lookup.
    private let tensorIndex: [String: QTurboTensorEntry]

    public init(modelDir: URL,
                metal: MetalContext,
                maxSeqLen: Int = 32768,
                expertCacheSlots: Int = 16) throws {
        self.modelDir = modelDir
        self.metal = metal

        // 1. Manifest.
        let manifestURL = modelDir.appendingPathComponent(QTurboFormatV1.manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw ModelError.manifestMissing(manifestURL.path)
        }
        let manifest = try QTurboManifestV1.load(from: manifestURL)
        self.manifest = manifest

        var index: [String: QTurboTensorEntry] = [:]
        for t in manifest.tensors { index[t.name] = t }
        self.tensorIndex = index

        // 2. mmap model_weights.bin.
        let residentURL = modelDir.appendingPathComponent(QTurboFormatV1.residentBlobFilename)
        guard FileManager.default.fileExists(atPath: residentURL.path) else {
            throw ModelError.residentMissing(residentURL.path)
        }
        let fd = open(residentURL.path, O_RDONLY)
        guard fd >= 0 else { throw ModelError.mmapFailed(residentURL.path) }
        self.mappedFD = fd

        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw ModelError.mmapFailed(residentURL.path) }
        let fileSize = Int(st.st_size)
        // Round length up to page size for bytesNoCopy compatibility.
        let pageSize = Int(getpagesize())
        let mapLen = QTurboFormatV1.align(fileSize, to: pageSize)

        guard let base = mmap(nil, mapLen, PROT_READ, MAP_PRIVATE, fd, 0), base != MAP_FAILED else {
            close(fd); throw ModelError.mmapFailed(residentURL.path)
        }
        self.mmapBase = base
        self.mmapLength = mapLen
        madvise(base, mapLen, MADV_RANDOM)

        // 3. Validate magic.
        let magicOK = QTurboFormatV1.magic.withUnsafeBytes { m -> Bool in
            memcmp(base, m.baseAddress!, QTurboFormatV1.magicLength) == 0
        }
        guard magicOK else {
            munmap(base, mapLen); close(fd); throw ModelError.badMagic
        }

        // 4. Zero-copy MTLBuffer over the whole file.
        guard let buf = metal.device.makeBuffer(bytesNoCopy: base, length: mapLen,
                                                 options: .storageModeShared,
                                                 deallocator: nil) else {
            munmap(base, mapLen); close(fd); throw ModelError.bufferCreationFailed
        }
        buf.label = "resident.model_weights"
        self.residentBuffer = buf

        // 5. Pipelines, KV cache, streamer.
        self.pipelines = try KernelPipelines(context: metal)
        self.kvCache = try KVCacheManager(device: metal.device, config: manifest.config, maxSeqLen: maxSeqLen)
        self.streamer = try PreadExpertStreamer(modelDir: modelDir,
                                                layout: manifest.expertLayout,
                                                numLayers: manifest.config.numHiddenLayers,
                                                capacity: expertCacheSlots)
    }

    deinit {
        munmap(mmapBase, mmapLength)
        close(mappedFD)
    }

    // MARK: - Resident tensor access

    public func entry(_ name: String) -> QTurboTensorEntry? { tensorIndex[name] }

    /// Byte offset of a resident tensor within `residentBuffer`.
    public func offset(_ name: String) throws -> Int {
        guard let e = tensorIndex[name] else { throw ModelError.tensorMissing(name) }
        return e.offset
    }

    /// Returns (offset, length) for a resident tensor.
    public func region(_ name: String) throws -> (offset: Int, length: Int) {
        guard let e = tensorIndex[name] else { throw ModelError.tensorMissing(name) }
        return (e.offset, e.length)
    }

    // MARK: - Convenience name builders (Qwen3)

    public func layerPrefix(_ l: Int) -> String { "model.layers.\(l)" }

    // MARK: - Raw pointer into resident mmap (CPU side, e.g. embedding lookup)

    public func residentPointer(offset: Int) -> UnsafeRawPointer {
        UnsafeRawPointer(mmapBase.advanced(by: offset))
    }
}

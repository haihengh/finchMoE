import Foundation
import QwenFieldfareFormat

/// Resolves tensor names across one or more safetensors shards, keeping the
/// readers alive so their mmaps stay valid.
public final class MultiShardTensorSource {
    private let readers: [SafetensorsReader]
    private var nameToReader: [String: SafetensorsReader] = [:]

    public init(shardURLs: [URL]) throws {
        var rs: [SafetensorsReader] = []
        for u in shardURLs {
            let r = try SafetensorsReader(url: u)
            rs.append(r)
            for n in r.tensorNames { nameToReader[n] = r }
        }
        self.readers = rs
    }

    public var allNames: [String] { Array(nameToReader.keys).sorted() }

    public func contains(_ name: String) -> Bool { nameToReader[name] != nil }

    public func tensor(_ name: String) throws -> (dtype: QTurboDType, shape: [Int], buffer: UnsafeRawBufferPointer) {
        guard let r = nameToReader[name] else {
            throw SafetensorsReader.ReaderError.tensorNotFound(name)
        }
        return try r.tensor(name: name)
    }

    public func byteLength(_ name: String) -> Int? {
        nameToReader[name]?.byteLength(name: name)
    }
}

/// Packs routed expert tensors into per-layer `packed_experts/layer_{NN}.bin`
/// files, each expert page-aligned to 16 KiB, matching the shared expert
/// layout from `QTurboRepackPlanner`.
public final class ExpertPacker {

    public enum PackError: Error, CustomStringConvertible {
        case missingTensor(String)
        case sizeMismatch(String, expected: Int, got: Int)
        case ioError(String)

        public var description: String {
            switch self {
            case .missingTensor(let s): return "ExpertPacker: missing tensor \(s)"
            case .sizeMismatch(let s, let e, let g):
                return "ExpertPacker: size mismatch for \(s) expected \(e) got \(g)"
            case .ioError(let s): return "ExpertPacker: IO error \(s)"
            }
        }
    }

    private let source: MultiShardTensorSource
    private let config: QTurboModelConfig
    private let layout: QTurboExpertLayout

    public init(source: MultiShardTensorSource, config: QTurboModelConfig) {
        self.source = source
        self.config = config
        self.layout = QTurboRepackPlanner.expertLayout(config: config)
    }

    public var expertLayout: QTurboExpertLayout { layout }

    /// Packs a single layer. Returns file info.
    @discardableResult
    public func packLayer(_ layer: Int,
                          outputDir: URL,
                          onExpert: ((Int, Int) -> Void)? = nil) throws -> QTurboExpertFileInfo {
        let packedDir = outputDir.appendingPathComponent(QTurboFormatV1.packedExpertsDir)
        try FileManager.default.createDirectory(at: packedDir, withIntermediateDirectories: true)
        let filename = QTurboFormatV1.packedExpertFilename(layer: layer)
        let fileURL = packedDir.appendingPathComponent(filename)

        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: fileURL) else {
            throw PackError.ioError("cannot open \(fileURL.path)")
        }
        defer { try? handle.close() }

        let stride = layout.alignedStride
        let padTemplate = [UInt8](repeating: 0, count: stride) // reused pad buffer

        for e in 0..<config.numExperts {
            var blob = Data(capacity: stride)
            for kind in QTurboRepackPlanner.expertKinds {
                let name = "model.layers.\(layer).mlp.experts.\(e).\(kind)"
                guard source.contains(name) else {
                    throw PackError.missingTensor(name)
                }
                let (_, _, buf) = try source.tensor(name)
                guard let sub = layout.subTensor(kind) else {
                    throw PackError.missingTensor("layout \(kind)")
                }
                // Validate length matches expected sub-tensor length.
                if buf.count != sub.length {
                    throw PackError.sizeMismatch(name, expected: sub.length, got: buf.count)
                }
                blob.append(contentsOf: UnsafeRawBufferPointer(start: buf.baseAddress, count: buf.count))
            }
            // Pad to aligned stride.
            if blob.count < stride {
                blob.append(contentsOf: padTemplate[0..<(stride - blob.count)])
            }
            try handle.write(contentsOf: blob)
            onExpert?(layer, e)
        }

        let fileSize = stride * config.numExperts
        return QTurboExpertFileInfo(layer: layer, filename: filename,
                                    numExperts: config.numExperts, fileSize: fileSize)
    }

    /// Packs all layers, returning per-layer file info.
    public func packAllLayers(outputDir: URL,
                              onProgress: ((Int, Int) -> Void)? = nil) throws -> [QTurboExpertFileInfo] {
        var infos: [QTurboExpertFileInfo] = []
        for layer in 0..<config.numHiddenLayers {
            let info = try packLayer(layer, outputDir: outputDir) { l, e in
                onProgress?(l, e)
            }
            infos.append(info)
        }
        return infos
    }
}

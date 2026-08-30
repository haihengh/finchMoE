import Foundation
import QwenFieldfareFormat

/// Orchestrates the full repack: (optionally download) → parse config →
/// write `model_weights.bin` (resident tensors) → pack experts → write
/// `manifest.json`.
public struct RepackCommand {

    public enum RepackError: Error, CustomStringConvertible {
        case noShardsFound(String)
        case ioError(String)

        public var description: String {
            switch self {
            case .noShardsFound(let s): return "Repack: no safetensors shards found in \(s)"
            case .ioError(let s): return "Repack: IO error \(s)"
            }
        }
    }

    public var sourceDir: URL       // HF cache dir or download target (contains safetensors + config.json)
    public var outputDir: URL       // .qturbo output directory
    public var download: Bool       // download from HF if shards missing
    public var repo: String

    public init(sourceDir: URL, outputDir: URL, download: Bool = false,
                repo: String = "mlx-community/Qwen3-30B-A3B-4bit") {
        self.sourceDir = sourceDir
        self.outputDir = outputDir
        self.download = download
        self.repo = repo
    }

    public func run() async throws {
        let fm = FileManager.default
        try fm.createDirectory(at: outputDir, withIntermediateDirectories: true)

        // 1. Ensure shards present (download if requested).
        var shards = discoverShards(in: sourceDir)
        if shards.isEmpty && download {
            log("No shards found locally — downloading from \(repo) …")
            let dl = HFDownloader(repo: repo)
            _ = try await dl.downloadAll(into: sourceDir)
            shards = discoverShards(in: sourceDir)
        }
        guard !shards.isEmpty else { throw RepackError.noShardsFound(sourceDir.path) }
        log("Found \(shards.count) safetensors shard(s).")

        // 2. Load config.
        let config = try loadConfig()
        log("Config: hidden=\(config.hiddenSize) layers=\(config.numHiddenLayers) experts=\(config.numExperts)")

        // 3. Open tensor source across shards.
        let source = try MultiShardTensorSource(shardURLs: shards)

        // 4. Write model_weights.bin (resident tensors) + collect index.
        log("Writing resident weights → \(QTurboFormatV1.residentBlobFilename) …")
        let (tensorEntries, residentSize) = try writeResidentBlob(source: source)

        // 5. Pack experts.
        log("Packing experts (\(config.numHiddenLayers) layers × \(config.numExperts) experts) …")
        let packer = ExpertPacker(source: source, config: config)
        let expertFiles = try packer.packAllLayers(outputDir: outputDir) { layer, expert in
            if expert == 0 {
                self.log("  layer \(String(format: "%02d", layer)) …")
            }
        }

        // 6. Write manifest.
        let manifest = QTurboManifestV1(
            config: config,
            tensors: tensorEntries,
            residentBlobSize: residentSize,
            expertLayout: packer.expertLayout,
            expertFiles: expertFiles
        )
        let manifestURL = outputDir.appendingPathComponent(QTurboFormatV1.manifestFilename)
        try manifest.write(to: manifestURL)
        log("Wrote manifest → \(manifestURL.path)")
        log("Repack complete: \(outputDir.path)")
    }

    // MARK: - Resident blob

    /// Writes all resident tensors into model_weights.bin with an 8-byte magic +
    /// 4-byte version header, each tensor aligned to residentTensorAlignment.
    private func writeResidentBlob(source: MultiShardTensorSource) throws
        -> (entries: [QTurboTensorEntry], size: Int) {
        let fileURL = outputDir.appendingPathComponent(QTurboFormatV1.residentBlobFilename)
        FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        guard let handle = try? FileHandle(forWritingTo: fileURL) else {
            throw RepackError.ioError("cannot open \(fileURL.path)")
        }
        defer { try? handle.close() }

        // Header: magic(8) + version(4 LE) + padding to alignment.
        var header = Data()
        header.append(contentsOf: QTurboFormatV1.magic)
        var ver = QTurboFormatV1.formatVersion.littleEndian
        withUnsafeBytes(of: &ver) { header.append(contentsOf: $0) }
        try handle.write(contentsOf: header)
        var offset = header.count

        func padTo(_ alignment: Int) throws {
            let aligned = QTurboFormatV1.align(offset, to: alignment)
            if aligned > offset {
                let pad = Data(count: aligned - offset)
                try handle.write(contentsOf: pad)
                offset = aligned
            }
        }

        // Collect resident tensor names from the source, classified.
        var residentNames: [String] = []
        for name in source.allNames {
            if case .resident = QTurboRepackPlanner.classify(name) {
                residentNames.append(name)
            }
        }
        // Stable ordering for reproducibility.
        residentNames.sort()

        var entries: [QTurboTensorEntry] = []
        for name in residentNames {
            try padTo(QTurboFormatV1.residentTensorAlignment)
            let (dtype, shape, buf) = try source.tensor(name)
            let data = Data(bytes: buf.baseAddress!, count: buf.count)
            try handle.write(contentsOf: data)
            entries.append(QTurboTensorEntry(name: name, dtype: dtype, shape: shape,
                                             offset: offset, length: buf.count))
            offset += buf.count
        }

        return (entries, offset)
    }

    // MARK: - Config

    private func loadConfig() throws -> QTurboModelConfig {
        let configURL = sourceDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            log("config.json not found — using Qwen3-30B-A3B defaults.")
            return QTurboModelConfig()
        }
        func i(_ k: String, _ d: Int) -> Int { (obj[k] as? NSNumber)?.intValue ?? d }
        func f(_ k: String, _ d: Float) -> Float { (obj[k] as? NSNumber)?.floatValue ?? d }
        func b(_ k: String, _ d: Bool) -> Bool { (obj[k] as? NSNumber)?.boolValue ?? d }

        // MLX quantization block.
        var groupSize = 64
        var bits = 4
        if let q = obj["quantization"] as? [String: Any] {
            groupSize = (q["group_size"] as? NSNumber)?.intValue ?? 64
            bits = (q["bits"] as? NSNumber)?.intValue ?? 4
        }

        let defaults = QTurboModelConfig()
        return QTurboModelConfig(
            hiddenSize: i("hidden_size", defaults.hiddenSize),
            headDim: i("head_dim", defaults.headDim),
            numAttentionHeads: i("num_attention_heads", defaults.numAttentionHeads),
            numKeyValueHeads: i("num_key_value_heads", defaults.numKeyValueHeads),
            numHiddenLayers: i("num_hidden_layers", defaults.numHiddenLayers),
            numExperts: i("num_experts", defaults.numExperts),
            numExpertsPerTok: i("num_experts_per_tok", defaults.numExpertsPerTok),
            moeIntermediateSize: i("moe_intermediate_size", defaults.moeIntermediateSize),
            sharedExpertIntermediateSize: i("intermediate_size", defaults.sharedExpertIntermediateSize),
            vocabSize: i("vocab_size", defaults.vocabSize),
            ropeTheta: f("rope_theta", defaults.ropeTheta),
            rmsNormEps: f("rms_norm_eps", defaults.rmsNormEps),
            maxPositionEmbeddings: i("max_position_embeddings", defaults.maxPositionEmbeddings),
            normTopkProb: b("norm_topk_prob", defaults.normTopkProb),
            quantGroupSize: groupSize,
            quantBits: bits,
            bosTokenId: i("bos_token_id", defaults.bosTokenId),
            eosTokenId: i("eos_token_id", defaults.eosTokenId)
        )
    }

    // MARK: - Helpers

    private func discoverShards(in dir: URL) -> [URL] {
        guard let items = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil) else { return [] }
        return items
            .filter { $0.lastPathComponent.hasSuffix(".safetensors") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private func log(_ s: String) {
        FileHandle.standardError.write(Data((s + "\n").utf8))
    }
}

import Foundation

/// Options for the local Qwen 3.6 quantizing repack: reads a bf16 safetensors
/// snapshot from disk and writes the `.finch` install (int4-affine weights,
/// int8 router, raw fp16/fp32 GDN entries). The Gemma remote-streaming path
/// is untouched.
public struct LocalQwenRepackOptions: Sendable {
    public let snapshotDir: String
    public let outputDir: String
    public let overwrite: Bool
    public let copyAuditPath: String?
    public let minFreeReserveBytes: UInt64

    public init(snapshotDir: String,
                outputDir: String,
                overwrite: Bool = false,
                copyAuditPath: String? = nil,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) {
        self.snapshotDir = snapshotDir
        self.outputDir = outputDir
        self.overwrite = overwrite
        self.copyAuditPath = copyAuditPath
        self.minFreeReserveBytes = minFreeReserveBytes
    }
}

public struct LocalQwenRepackResult: Sendable {
    public let outputDir: String
    public let outputBytes: UInt64
    public let excludedTensorCount: Int
}

/// Orchestrates the Qwen quantizing repack end to end: snapshot load →
/// plan → quantizing writes → layout.json → tokenizer sidecars →
/// manifest.json → verified-install receipt → atomic rename.
public final class LocalQwenRepacker {
    private let options: LocalQwenRepackOptions
    private let audit: RepackAudit
    private let startTime = Date()

    public init(options: LocalQwenRepackOptions,
                audit: RepackAudit = RepackAudit()) {
        self.options = options
        self.audit = audit
    }

    public func run(progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in }) async throws
        -> LocalQwenRepackResult {
        try validateOptions()
        let installLock = try InstallLock.acquire(outputDirectory: options.outputDir)
        defer { withExtendedLifetime(installLock) {} }
        let paths = installLock.paths
        if try Posix.entryKind(paths.finalDirectory) == .directory, !options.overwrite {
            throw RepackError.configurationInvalid(detail:
                "output directory already exists: \(paths.finalDirectory)")
        }
        let hasPartial = try Posix.entryKind(paths.partialDirectory) == .directory
        if hasPartial {
            throw RepackError.installStateCorrupt(
                path: paths.partialDirectory,
                detail: "stale partial directory from a previous run; remove it first")
        }

        do {
            return try await runPrepared(paths: paths, progress: progress)
        } catch {
            // No resume support on the local path — a failed run leaves no
            // reusable state, so drop the partial directory.
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> LocalQwenRepackResult {
        progress(.downloadingMetadata)
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: options.snapshotDir)
        try Task.checkCancellation()

        let plan = try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                              arch: snapshot.arch,
                                              shardHeaders: snapshot.shardHeaders,
                                              outputDir: paths.partialDirectory)
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
        progress(.planning(downloadBytes: 0, outputBytes: outputBytes))

        let diskRequirement = try DiskSpaceChecker.requireAvailable(
            path: paths.parentDirectory,
            bytes: outputBytes,
            reserveBytes: options.minFreeReserveBytes)
        progress(.checkingDisk(diskRequirement))
        try Task.checkCancellation()

        audit.remoteRepoID = nil
        audit.sourceSnapshotSha256 = snapshot.metadata.indexSha256Hex
        audit.bitWidthOverridesHonored = 0
        audit.tensorsDroppedMultimodal = plan.excludedTensorNames
        audit.packedExpertLayoutMode = "identity"

        progress(.reservingOutput(bytes: outputBytes))
        try Posix.mkdirP(paths.partialDirectory)
        try Posix.mkdirP((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts"))

        progress(.copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: outputBytes))
        try QwenQuantizedWriter.writeResident(
            plan: plan.resident,
            audit: audit,
            cancellationCheck: Task.checkCancellation)
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            progress(.hashingOutput("packed_experts/" + (layer.path as NSString).lastPathComponent))
            _ = try QwenQuantizedWriter.writeLayer(
                plan: layer,
                audit: audit,
                cancellationCheck: Task.checkCancellation)
        }

        try Task.checkCancellation()
        let layoutPath = ((paths.partialDirectory as NSString)
            .appendingPathComponent("packed_experts") as NSString)
            .appendingPathComponent("layout.json")
        let expertStride = plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertStride ?? 0
        let layoutData = try FinchJSON.encodeLayout(
            layers: plan.layers,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride)
        try writeSmall(path: layoutPath, data: layoutData)
        try FinchLayoutValidator.validate(path: layoutPath, layers: plan.layers)
        try recordOutputFile(relativePath: "packed_experts/layout.json",
                             path: layoutPath,
                             progress: progress)

        try Task.checkCancellation()
        try copyTokenizers(snapshotDir: options.snapshotDir,
                           partialDir: paths.partialDirectory,
                           progress: progress)

        progress(.finalizing)
        try Task.checkCancellation()
        try writeManifest(plan: plan,
                          partialDir: paths.partialDirectory,
                          metadata: snapshot.metadata,
                          expertStride: expertStride)

        try Task.checkCancellation()
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }

        audit.wallTimeSeconds = Date().timeIntervalSince(startTime)
        audit.wholeFileHeapBuffers = false
        if let auditPath = options.copyAuditPath {
            let data = try audit.toJSONData(outputDir: options.outputDir)
            try Posix.mkdirP((auditPath as NSString).deletingLastPathComponent)
            try data.write(to: URL(fileURLWithPath: auditPath))
        }

        return LocalQwenRepackResult(outputDir: options.outputDir,
                                     outputBytes: outputBytes,
                                     excludedTensorCount: plan.excludedTensorNames.count)
    }

    private func validateOptions() throws {
        let index = (options.snapshotDir as NSString)
            .appendingPathComponent("model.safetensors.index.json")
        guard try Posix.entryKind(index) == .regular else {
            throw RepackError.indexJsonInvalid(path: index, detail: "snapshot has no index.json")
        }
        let config = (options.snapshotDir as NSString)
            .appendingPathComponent("config.json")
        guard try Posix.entryKind(config) == .regular else {
            throw RepackError.configJsonInvalid(path: config, detail: "snapshot has no config.json")
        }
    }

    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        audit.outputFiles.append(.init(relativePath: relativePath, size: size, sha256: sha))
    }

    private func writeSmall(path: String, data: Data) throws {
        try Posix.mkdirP((path as NSString).deletingLastPathComponent)
        try data.write(to: URL(fileURLWithPath: path), options: [.atomic])
        audit.recordWrite(bytes: data.count)
    }

    private func copyTokenizers(snapshotDir: String,
                                partialDir: String,
                                progress: @Sendable (ModelInstallProgress) -> Void) throws {
        let tokenizerDir = (partialDir as NSString).appendingPathComponent("tokenizer")
        let sidecars: [(name: String, required: Bool)] = [
            ("config.json", true),
            ("tokenizer.json", true),
            ("tokenizer_config.json", true),
            ("special_tokens_map.json", false),
            ("chat_template.jinja", false),
            ("chat_template.json", false),
            ("vocab.json", false),
            ("merges.txt", false),
        ]
        for file in sidecars {
            try Task.checkCancellation()
            let src = (snapshotDir as NSString).appendingPathComponent(file.name)
            guard try Posix.entryKind(src) == .regular else {
                if file.required {
                    throw RepackError.missingTensor(name: file.name)
                }
                continue
            }
            try Posix.mkdirP(tokenizerDir)
            let dst = (tokenizerDir as NSString).appendingPathComponent(file.name)
            if try Posix.entryKind(dst) == .regular {
                try FileManager.default.removeItem(atPath: dst)
            }
            try FileManager.default.copyItem(atPath: src, toPath: dst)
            try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                 path: dst,
                                 progress: progress)
        }
    }

    private func writeManifest(plan: QwenRepackPlan,
                               partialDir: String,
                               metadata: QwenLocalSnapshot.SourceMetadata,
                               expertStride: UInt64) throws {
        // Locked writer decisions (docs/QWEN36_PORT.md): embedding/attention/
        // shared/routed int4 affine, router int8 affine, group 64. The GDN
        // linear_attn projections are int8 — int4 noise on them amplifies
        // through the recurrent state and drowns the final logits.
        let bits = FinchJSON.QuantBitWidths(
            embedding: 4, attention: 4, linearAttention: 8, router: 8,
            sharedExpert: 4, routedExpert: 4)
        let files = audit.outputFiles.map {
            ($0.relativePath, FinchJSON.FileEntry(size: $0.size, sha256: $0.sha256))
        }
        let data = try FinchJSON.encodeManifest(
            arch: plan.arch,
            baseMode: "affine",
            baseGroupSize: 64,
            bitsOverrideCount: 0,
            modelID: "local/Qwen3.6-35B-A3B",
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits)
        let tmp = (partialDir as NSString).appendingPathComponent("manifest.json.tmp")
        let final = (partialDir as NSString).appendingPathComponent("manifest.json")
        try writeSmall(path: tmp, data: data)
        try Posix.rename(from: tmp, to: final)
        let manifestSha = try Sha256Stream.hashFile(path: final)
        let receipt = try VerifiedInstallReceiptWriter.encode(
            outputDir: options.outputDir,
            manifestSha256: manifestSha,
            manifestSize: UInt64(data.count),
            sourceRepoID: "local:" + (options.snapshotDir as NSString).lastPathComponent,
            sourceRevision: metadata.indexSha256Hex,
            files: audit.outputFiles)
        let receiptPath = (partialDir as NSString)
            .appendingPathComponent(VerifiedInstallReceiptWriter.fileName)
        try writeSmall(path: receiptPath, data: receipt)
    }
}

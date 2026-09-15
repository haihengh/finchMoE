import Foundation

/// Options for the local Qwen 3.6 / 3.8 quantizing repack: reads a bf16
/// safetensors snapshot from disk and writes the `.finch` install (int4-affine
/// weights, int8 router, raw fp16/fp32 GDN entries). Qwen3.8 additionally
/// emits the PLE n-gram table as raw-BF16 part files under `ple_shards/`. The
/// Gemma remote-streaming path is untouched.
public struct LocalQwenRepackOptions: Sendable {
    public let snapshotDir: String
    public let outputDir: String
    public let overwrite: Bool
    /// Continue a partial directory left by an interrupted run instead of
    /// refusing it. Output files the journal records as complete — and that
    /// still hash to their recorded digest — are reused; everything else is
    /// rewritten. Without this flag a stale partial is refused exactly as it
    /// was before resume existed.
    public let resume: Bool
    public let copyAuditPath: String?
    public let minFreeReserveBytes: UInt64

    public init(snapshotDir: String,
                outputDir: String,
                overwrite: Bool = false,
                resume: Bool = false,
                copyAuditPath: String? = nil,
                minFreeReserveBytes: UInt64 = 1 * 1024 * 1024 * 1024) {
        self.snapshotDir = snapshotDir
        self.outputDir = outputDir
        self.overwrite = overwrite
        self.resume = resume
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
    /// Resume state, established by `runPrepared` before the first output file
    /// is written. Nil for any run that never reached that point.
    private var journal: LocalRepackJournal?
    private var journalPath: String?

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

        // Everything that decides whether an existing partial directory may be
        // written into is settled here, before a byte is written or deleted.
        var resumeJournal: LocalRepackJournal? = nil
        if try Posix.entryKind(paths.partialDirectory) == .directory {
            guard options.resume else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "stale partial directory from a previous run; "
                        + "pass --resume to continue it, or remove it first")
            }
            let journalPath = LocalRepackJournal.path(inPartialDirectory: paths.partialDirectory)
            // The journal is written before the first payload byte, so its
            // absence means this partial carries no record of what is finished
            // — an older run's leftover, or a directory that lost its state.
            // Resuming onto it would mean guessing, so refuse instead.
            guard try Posix.entryKind(journalPath) == .regular else {
                throw RepackError.installStateCorrupt(
                    path: paths.partialDirectory,
                    detail: "partial directory has no \(LocalRepackJournal.fileName), "
                        + "so none of its files are known complete; remove it first")
            }
            resumeJournal = try LocalRepackJournal.load(from: journalPath)
        }

        do {
            return try await runPrepared(paths: paths,
                                         resumeJournal: resumeJournal,
                                         progress: progress)
        } catch {
            // A resumed run keeps its partial directory — that state is the
            // whole point of --resume, and the journal lets the next attempt
            // pick up where this one stopped. Without --resume the behaviour
            // is unchanged: a failed run leaves nothing behind.
            if !options.resume {
                try? FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            throw error
        }
    }

    private func runPrepared(paths: RemoteInstallPaths,
                             resumeJournal: LocalRepackJournal?,
                             progress: @escaping @Sendable (ModelInstallProgress) -> Void) async throws
        -> LocalQwenRepackResult {
        progress(.downloadingMetadata)
        let snapshot = try QwenLocalSnapshot.load(snapshotDir: options.snapshotDir)
        try Task.checkCancellation()

        let fingerprint = LocalRepackJournal.Fingerprint(
            sourceDirectory: URL(fileURLWithPath: options.snapshotDir).path,
            sourceIndexSha256: snapshot.metadata.indexSha256Hex,
            outputDirectory: URL(fileURLWithPath: options.outputDir).path,
            modelFamily: snapshot.arch.modelFamily,
            numLayers: snapshot.arch.numLayers)
        if let resumeJournal, resumeJournal.fingerprint != fingerprint {
            // This is the property the old unconditional refusal protected: a
            // partial is never written into by a run that would produce
            // different bytes, so a resumed install can never be a mixture of
            // two models.
            let previous = resumeJournal.fingerprint
            let indexPrefix = previous.sourceIndexSha256.prefix(12)
            let family = previous.modelFamily ?? "no family"
            throw RepackError.installStateIncompatible(
                detail: "partial directory was written from a different source: "
                    + "\(previous.sourceDirectory), index \(indexPrefix)…, "
                    + "\(family), \(previous.numLayers) layers; "
                    + "remove it first")
        }

        let plan = try QwenRepackPlanner.plan(meta: snapshot.metadata,
                                              arch: snapshot.arch,
                                              shardHeaders: snapshot.shardHeaders,
                                              outputDir: paths.partialDirectory)
        let pleBytes = plan.pleParts.reduce(UInt64(0)) {
            $0 + $1.quantizedByteCount
        }
        let outputBytes = plan.resident.totalSize
            + plan.layers.reduce(UInt64(0)) { $0 + $1.fileSize }
            + pleBytes
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

        // The journal exists before the first output file does, so any kill
        // from here on leaves a partial that can be resumed rather than
        // discarded. A resumed run adopts the journal it already has.
        let journalPath = LocalRepackJournal.path(inPartialDirectory: paths.partialDirectory)
        if let resumeJournal {
            self.journal = resumeJournal
        } else {
            let fresh = LocalRepackJournal(fingerprint: fingerprint)
            try fresh.write(to: journalPath)
            self.journal = fresh
        }
        self.journalPath = journalPath

        progress(.copyingPayload(
            reusedBytes: 0, downloadedThisRunBytes: 0, totalBytes: outputBytes))
        _ = try produce(relativePath: plan.resident.relativePath,
                        path: plan.resident.path) {
            try QwenQuantizedWriter.writeResident(
                plan: plan.resident,
                audit: audit,
                cancellationCheck: Task.checkCancellation)
        }
        for layer in plan.layers where layer.expertsPerLayer > 0 {
            try Task.checkCancellation()
            progress(.hashingOutput(layer.relativePath))
            _ = try produce(relativePath: layer.relativePath,
                            path: layer.path) {
                try QwenQuantizedWriter.writeLayer(
                    plan: layer,
                    audit: audit,
                    cancellationCheck: Task.checkCancellation)
            }
        }
        for part in plan.pleParts {
            try Task.checkCancellation()
            progress(.hashingOutput(part.relativePath))
            _ = try produce(relativePath: part.relativePath,
                            path: part.path) {
                try QwenQuantizedWriter.writePLEPart(
                    plan: part,
                    audit: audit,
                    cancellationCheck: Task.checkCancellation)
            }
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
        // The validator runs before the file is hashed, so a journal entry for
        // this path is proof the layout validated: a resumed run reuses an
        // artifact whose digest was recorded only after that check passed, and
        // need not repeat it.
        _ = try produce(relativePath: "packed_experts/layout.json",
                        path: layoutPath) {
            try writeSmall(path: layoutPath, data: layoutData)
            try FinchLayoutValidator.validate(path: layoutPath, layers: plan.layers)
            return try recordOutputFile(relativePath: "packed_experts/layout.json",
                                        path: layoutPath,
                                        progress: progress)
        }

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
        // The journal is run bookkeeping, not install content. Drop it before
        // the partial directory is promoted, or the install would carry a file
        // the manifest does not list. The cost of dying inside this window is
        // a partial that must be discarded rather than resumed — one syscall
        // wide, and it fails loudly rather than silently.
        try? FileManager.default.removeItem(atPath: journalPath)
        try? Posix.fsyncDirectory(paths.partialDirectory)
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

    /// Writes one output file — or, on `--resume`, reuses the one a previous
    /// run already wrote and journaled.
    ///
    /// `relativePath` is the `manifest.files` key, so the journal, the audit
    /// and the manifest all name the same file the same way.
    ///
    /// Ordering is the correctness argument. `write` returns only after the
    /// file is complete *and* hashed, and the journal entry is appended after
    /// that, so an entry means "finished", never "started". Durability is not
    /// assumed of it: a couple of these writers (`writeSmall`, the tokenizer
    /// copies) never `fsync` at all, and on macOS an `fsync` does not promise
    /// the platter caught up either. Which is why reuse re-reads the bytes
    /// instead of trusting the entry — see `reuseIfIntact`.
    private func produce(relativePath: String,
                         path: String,
                         write: () throws -> RepackAudit.OutputFile) throws
        -> RepackAudit.OutputFile {
        guard let journalPath, var journal = self.journal else {
            // Unreachable in practice: `runPrepared` establishes the journal
            // before the first write.
            return try write()
        }
        if let entry = journal.entry(for: relativePath),
           let reused = try reuseIfIntact(entry: entry, path: path) {
            audit.outputFiles.append(reused)
            return reused
        }
        let written = try write()
        journal.record(LocalRepackJournal.Entry(relativePath: written.relativePath,
                                                size: written.size,
                                                sha256: written.sha256))
        try journal.write(to: journalPath)
        self.journal = journal
        return written
    }

    /// Returns the recorded file when the bytes on disk still match the
    /// journal, or nil when the file has to be written again.
    ///
    /// Cheapest check first. Existence and size settle most cases; the digest
    /// is what settles the one that matters, because every writer here
    /// `ftruncate`s to the final size before filling, so a file interrupted
    /// mid-write is exactly the right length and wrong in its tail.
    private func reuseIfIntact(entry: LocalRepackJournal.Entry,
                               path: String) throws -> RepackAudit.OutputFile? {
        guard try Posix.entryKind(path) == .regular else { return nil }
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        guard size == entry.size else { return nil }
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        guard sha == entry.sha256 else { return nil }
        return RepackAudit.OutputFile(relativePath: entry.relativePath,
                                      size: size, sha256: sha)
    }

    @discardableResult
    private func recordOutputFile(relativePath: String,
                                  path: String,
                                  progress: @Sendable (ModelInstallProgress) -> Void) throws
        -> RepackAudit.OutputFile {
        progress(.hashingOutput(relativePath))
        try Task.checkCancellation()
        let fd = try Posix.openRead(path)
        defer { close(fd) }
        let size = try Posix.fileSize(fd: fd, path: path)
        let sha = try WriterCore.hashEntireFile(path: path,
                                                size: size,
                                                audit: audit,
                                                cancellationCheck: Task.checkCancellation)
        let file = RepackAudit.OutputFile(relativePath: relativePath, size: size, sha256: sha)
        audit.outputFiles.append(file)
        return file
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
            _ = try produce(relativePath: "tokenizer/\(file.name)", path: dst) {
                // `copyItem` refuses an existing destination, and an
                // interrupted run leaves one behind.
                if try Posix.entryKind(dst) == .regular {
                    try FileManager.default.removeItem(atPath: dst)
                }
                try FileManager.default.copyItem(atPath: src, toPath: dst)
                return try recordOutputFile(relativePath: "tokenizer/\(file.name)",
                                            path: dst,
                                            progress: progress)
            }
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
        let modelID = plan.arch.modelFamily == ArchInfo.qwen38Family
            ? "local/Qwen3.8-Flash-Next-125B"
            : "local/Qwen3.6-35B-A3B"
        // PLE n-gram quantization slot: present (int4, the plan's group size)
        // for a qwen3_8 install that has PLE parts, absent otherwise — this
        // is the additive manifest slot Phase 3 defines, and its absence is
        // how old raw-BF16 installs stay loadable (Phase 4 backward-compat).
        let pleNgram: FinchJSON.PleNgramQuant? = plan.pleParts.first.map {
            FinchJSON.PleNgramQuant(weightBits: 4, scheme: "affine",
                                    groupSize: $0.groupSize)
        }
        let data = try FinchJSON.encodeManifest(
            arch: plan.arch,
            baseMode: "affine",
            baseGroupSize: 64,
            bitsOverrideCount: 0,
            modelID: modelID,
            sourceSnapshotHash: "sha256:" + metadata.indexSha256Hex,
            files: files,
            expertsPerLayer: plan.layers.first(where: { $0.expertsPerLayer > 0 })?.expertsPerLayer ?? 0,
            numLayers: plan.arch.numLayers,
            expertStride: expertStride,
            bitWidths: bits,
            pleNgram: pleNgram)
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

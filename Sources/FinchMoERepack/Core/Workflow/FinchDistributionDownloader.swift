import Foundation
import FinchMoEFormat

/// Installs a published `.finch` distribution from Hugging Face.
///
/// This is the read-side twin of `RemoteStreamingRepacker`. That one pulls an
/// upstream safetensors checkpoint and *builds* the `.finch` layout, which is
/// why it needs a shard index, a range plan, a fingerprint allowlist and a
/// byte-range checkpoint. Here the layout already exists on the remote, so
/// there is no plan: `manifest.json` names every file with its size and digest,
/// and the download is "fetch exactly those, then prove each one".
///
/// The digest check is the whole integrity story, and it is a strong one — the
/// manifest that names a file also names what that file must hash to.
public enum FinchDistributionDownloader {

    /// Files in flight. Hugging Face rate-limits, and the retry policy absorbs
    /// 429s with `Retry-After`, but there is no reason to invite them: the link
    /// saturates well before this on the installs published so far.
    public static let defaultConcurrency = 6

    /// Reserve held back on top of the download so a finished install is not
    /// immediately under storage pressure.
    public static let defaultReserveBytes: UInt64 = 1 << 30

    public static func run(source: FinchDistribution,
                           outputDirectory: String,
                           token: String? = nil,
                           concurrency: Int = defaultConcurrency,
                           reserveBytes: UInt64 = defaultReserveBytes,
                           audit: RepackAudit? = nil,
                           progress: @escaping @Sendable (ModelInstallProgress) -> Void = { _ in })
        async throws -> VerifyInstallResult {
        let lock = try InstallLock.acquire(outputDirectory: outputDirectory)
        let paths = lock.paths
        try Task.checkCancellation()

        // 1. Manifest first: it is small, and it decides everything after it.
        //
        // The staging directory is inside `.partial`, not `NSTemporaryDirectory`.
        // `fetchSmallFile` stages its download and then `rename(2)`s it into
        // place, and rename cannot cross filesystems — on a machine whose boot
        // volume and model volume differ (the normal case here: the model lives
        // on an external SSD) a system-temp staging file fails with EXDEV.
        progress(.downloadingMetadata)
        try Posix.mkdirP(paths.partialDirectory)
        let source0 = HuggingFaceRemoteSource(repoID: source.repoID,
                                              requestedRevision: source.revision,
                                              token: token,
                                              tempDirectory: paths.partialDirectory)
        let manifestInfo = try await source0.resolveFileInfo(filename: manifestName,
                                                             audit: audit)
        guard manifestInfo.size <= VerifiedInstallTool.manifestMaxBytes else {
            throw RepackError.remoteFileTooLarge(path: manifestName,
                                                 size: manifestInfo.size,
                                                 cap: VerifiedInstallTool.manifestMaxBytes)
        }
        let manifestPath = (paths.partialDirectory as NSString)
            .appendingPathComponent(manifestName)
        try await source0.fetchSmallFile(filename: manifestName,
                                         info: manifestInfo,
                                         capBytes: VerifiedInstallTool.manifestMaxBytes,
                                         outputPath: manifestPath,
                                         audit: audit)

        let manifestData = try Posix.readBoundedData(
            manifestPath, maximumBytes: VerifiedInstallTool.manifestMaxBytes)
        let manifest = try decodeManifest(manifestData)
        let manifestSha = hash(manifestData)

        let remote = source0.pinned(commit: manifestInfo.resolvedCommit)
        let files = orderedFiles(of: manifest)
        let totalBytes = try total(of: manifest)

        // 2. Resume state. A checkpoint that does not describe this exact
        //    download is refused rather than reused.
        let fingerprint = FinchDistributionCheckpoint.Fingerprint(
            repoID: source.repoID,
            requestedRevision: source.revision,
            resolvedCommit: manifestInfo.resolvedCommit,
            manifestSha256: manifestSha)
        var checkpoint = try loadCheckpoint(paths: paths, fingerprint: fingerprint)
        var reusedBytes = try validateReusedFiles(expected: manifest.files,
                                                  files: files,
                                                  partialDirectory: paths.partialDirectory,
                                                  checkpoint: &checkpoint)
        try checkpoint.write(to: paths.checkpointFile)

        progress(.planning(downloadBytes: totalBytes, outputBytes: totalBytes))

        // 3. Disk, before any payload byte moves.
        let requirement = try DiskSpaceChecker.assess(
            path: paths.parentDirectory,
            bytes: totalBytes &- min(totalBytes, reusedBytes),
            reserveBytes: reserveBytes)
        progress(.checkingDisk(requirement))
        guard requirement.canInstall else {
            throw RepackError.diskSpaceInsufficient(
                path: paths.parentDirectory,
                required: requirement.requiredBytes,
                available: requirement.availableBytes)
        }

        // 4. Payload.
        let counters = DownloadCounters(reusedBytes: reusedBytes,
                                        totalBytes: totalBytes,
                                        progress: progress)
        progress(.copyingPayload(reusedBytes: reusedBytes,
                                 downloadedThisRunBytes: 0,
                                 totalBytes: totalBytes))
        try await downloadAll(remote: remote,
                              manifest: manifest,
                              files: files,
                              partialDirectory: paths.partialDirectory,
                              checkpointPath: paths.checkpointFile,
                              concurrency: max(1, concurrency),
                              counters: counters,
                              checkpoint: &checkpoint)

        try Task.checkCancellation()

        // 5. Verify every byte, then write the receipt naming the *final*
        //    location — the bytes are still in `.partial`.
        //
        // Staging files are cleared first: a transfer killed mid-flight can
        // leave one behind, and verification would report it as an entry the
        // manifest never declared.
        removeStagingFiles(inPartialDirectory: paths.partialDirectory)
        let result = try VerifiedInstallTool.verify(
            root: URL(fileURLWithPath: paths.partialDirectory),
            recordedOutputDirectory: paths.finalDirectory,
            sourceRepoID: source.repoID,
            sourceRevision: manifestInfo.resolvedCommit,
            toolVersion: "FinchMoERepack download-finch",
            onHashingFile: { progress(.hashingOutput($0)) })

        // 6. Promote.
        progress(.finalizing)
        try Task.checkCancellation()
        try promote(paths: paths)
        try? FileManager.default.removeItem(atPath: paths.checkpointFile)
        return result
    }

    public static let manifestName = "manifest.json"

    /// Reads the resume checkpoint for an interrupted download, or nil when
    /// there is none. Throws for a checkpoint that exists but cannot be
    /// trusted — "unreadable" must never be mistaken for "absent".
    public static func inspectPersistentInstall(outputDirectory: String)
        throws -> FinchDistributionCheckpoint? {
        let paths = try RemoteInstallPaths(outputDirectory: outputDirectory)
        guard try Posix.entryKind(paths.checkpointFile) == .regular else { return nil }
        return try FinchDistributionCheckpoint.load(from: paths.checkpointFile)
    }

    /// Removes the partial directory, the checkpoint and the staging file.
    public static func discardPartial(outputDirectory: String) throws {
        let paths = try RemoteInstallPaths(outputDirectory: outputDirectory)
        guard try Posix.entryKind(paths.finalDirectory) == .absent else {
            throw RepackError.installPathUnsafe(
                path: paths.finalDirectory,
                detail: "an installed model already exists at this path")
        }
        if try Posix.entryKind(paths.partialDirectory) == .directory {
            try FileManager.default.removeItem(atPath: paths.partialDirectory)
        }
        if try Posix.entryKind(paths.checkpointFile) == .regular {
            try FileManager.default.removeItem(atPath: paths.checkpointFile)
        }
        try Posix.fsyncDirectory(paths.parentDirectory)
    }

    // MARK: - Download

    private static func downloadAll(remote: HuggingFaceRemoteSource,
                                    manifest: FinchManifestV1,
                                    files: [String],
                                    partialDirectory: String,
                                    checkpointPath: String,
                                    concurrency: Int,
                                    counters: DownloadCounters,
                                    checkpoint: inout FinchDistributionCheckpoint)
        async throws {
        let pending = files.filter { checkpoint.entry(for: $0) == nil }
        guard !pending.isEmpty else { return }

        let batch = Batch(pending: pending,
                          manifest: manifest,
                          resolvedCommit: remote.resolvedCommit ?? remote.requestedRevision,
                          partialDirectory: partialDirectory,
                          counters: counters,
                          remote: remote)

        // Each transfer carries its own retry envelope, so one file can fail and
        // recover without disturbing the others. The group is refilled as tasks
        // land, which keeps at most `concurrency` transfers in flight without a
        // separate worker-pool abstraction.
        try await withThrowingTaskGroup(of: FinchDistributionCheckpoint.Entry.self) { group in
            let total = batch.pending.count
            var next = 0
            for _ in 0..<min(concurrency, total) {
                let relativePath = batch.pending[next]
                next += 1
                group.addTask { try await batch.fetch(relativePath: relativePath) }
            }
            while let entry = try await group.next() {
                checkpoint.record(entry)
                try checkpoint.write(to: checkpointPath)
                if next < total, !Task.isCancelled {
                    let relativePath = batch.pending[next]
                    next += 1
                    group.addTask { try await batch.fetch(relativePath: relativePath) }
                }
            }
        }
    }

    /// The per-download state a worker needs. A reference type because the task
    /// group closure is concurrent and captures it by reference.
    private final class Batch: @unchecked Sendable {
        let pending: [String]
        let manifest: FinchManifestV1
        let resolvedCommit: String
        let partialDirectory: String
        let counters: DownloadCounters
        let remote: HuggingFaceRemoteSource

        init(pending: [String],
             manifest: FinchManifestV1,
             resolvedCommit: String,
             partialDirectory: String,
             counters: DownloadCounters,
             remote: HuggingFaceRemoteSource) {
            self.pending = pending
            self.manifest = manifest
            self.resolvedCommit = resolvedCommit
            self.partialDirectory = partialDirectory
            self.counters = counters
            self.remote = remote
        }

        /// Transfers one file and returns the checkpoint entry it earned.
        ///
        /// No per-file HEAD. The transfer's own expectation machinery already
        /// pins the identity: `RemoteRangeTransfer` demands a 206 whose
        /// `Content-Range` matches `bytes 0-(size-1)/size` exactly, so a remote
        /// file of a different size, or a server that ignored the range, fails
        /// before the bytes are kept. The manifest supplies the size the range
        /// is built from, and the digest check below is the final word — so a
        /// HEAD would add a round trip per file and prove nothing further.
        ///
        /// The digest is computed from the bytes that actually landed, never
        /// copied from the manifest, so nothing can record a digest it does not
        /// have.
        func fetch(relativePath: String) async throws -> FinchDistributionCheckpoint.Entry {
            guard let expected = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(
                    detail: "manifest has no entry for \(relativePath)")
            }
            let destination = (partialDirectory as NSString)
                .appendingPathComponent(relativePath)
            try Posix.mkdirP((destination as NSString).deletingLastPathComponent)
            try? FileManager.default.removeItem(atPath: destination)

            let info = RemoteFileInfo(filename: relativePath,
                                      resolvedCommit: resolvedCommit,
                                      size: expected.size,
                                      etag: nil,
                                      xetHash: nil,
                                      acceptsRanges: true)
            let transferred = try await remote.downloadRangeToTempFile(
                filename: relativePath,
                info: info,
                offset: 0,
                length: Int(expected.size),
                targetPath: destination,
                progress: { [counters] cumulative in
                    counters.update(relativePath, cumulative: cumulative)
                })

            let digest = try Sha256Stream.hashFile(path: transferred.path)
            guard digest.lowercased() == expected.sha256.lowercased() else {
                try? FileManager.default.removeItem(atPath: destination)
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) SHA mismatch after download "
                        + "(expected \(expected.sha256), got \(digest))")
            }
            return FinchDistributionCheckpoint.Entry(relativePath: relativePath,
                                                     size: expected.size,
                                                     sha256: digest)
        }
    }

    /// Thread-safe progress accounting shared by the download workers.
    /// Internal rather than private so its accounting can be tested: the
    /// transfer callback reports a per-file cumulative, and reading it as a
    /// delta overcounts progress by whatever factor the retries and callbacks
    /// happen to produce.
    final class DownloadCounters: @unchecked Sendable {
        private let lock = NSLock()
        /// Bytes reported so far per file. The transfer's callback reports a
        /// *cumulative* count for its own file, not a delta — so the aggregate
        /// has to difference each file against its own previous reading, and a
        /// transfer that restarts (the retry envelope re-reports from zero)
        /// correctly moves its contribution back down.
        private var perFile: [String: UInt64] = [:]
        private var downloaded: UInt64 = 0
        private let reusedBytes: UInt64
        private let totalBytes: UInt64
        private let progress: @Sendable (ModelInstallProgress) -> Void

        init(reusedBytes: UInt64,
             totalBytes: UInt64,
             progress: @escaping @Sendable (ModelInstallProgress) -> Void) {
            self.reusedBytes = reusedBytes
            self.totalBytes = totalBytes
            self.progress = progress
        }

        func update(_ relativePath: String, cumulative: UInt64) {
            lock.lock()
            let previous = perFile[relativePath] ?? 0
            perFile[relativePath] = cumulative
            if cumulative >= previous {
                downloaded += cumulative - previous
            } else {
                downloaded -= min(previous - cumulative, downloaded)
            }
            let snapshot = downloaded
            lock.unlock()
            // Outside the lock: the consumer is a UI stream, and it must never
            // be able to block a transfer.
            progress(.copyingPayload(reusedBytes: reusedBytes,
                                     downloadedThisRunBytes: snapshot,
                                     totalBytes: totalBytes))
        }
    }

    // MARK: - Resume

    /// Re-hashes every file a previous run claimed to have finished, and drops
    /// the ones that no longer match.
    ///
    /// A checkpoint entry can outlive its bytes: the process can die between a
    /// file's `fsync` and the checkpoint's, and on macOS `fsync` does not
    /// promise the data reached the platter. Size alone cannot separate a
    /// finished file from a torn one, so anything whose size matches is read
    /// back and hashed; anything whose size does not match is re-fetched without
    /// being read.
    /// Internal rather than private so the resume guarantee can be tested
    /// directly: it is the one function whose failure mode is silent
    /// corruption rather than a visible error.
    static func validateReusedFiles(expected: [String: FinchManifestFileV1],
                                    files: [String],
                                    partialDirectory: String,
                                    checkpoint: inout FinchDistributionCheckpoint)
        throws -> UInt64 {
        var reused: UInt64 = 0
        for relativePath in files {
            guard checkpoint.entry(for: relativePath) != nil,
                  let expected = expected[relativePath] else { continue }
            let path = (partialDirectory as NSString).appendingPathComponent(relativePath)
            guard try Posix.entryKind(path) == .regular else {
                checkpoint.forget(relativePath: relativePath)
                continue
            }
            guard let size = try fileSize(path), size == expected.size else {
                checkpoint.forget(relativePath: relativePath)
                continue
            }
            guard let digest = try? Sha256Stream.hashFile(path: path),
                  digest.lowercased() == expected.sha256.lowercased() else {
                checkpoint.forget(relativePath: relativePath)
                continue
            }
            reused = try VerifiedInstallTool.addingVerifiedBytes(reused, expected.size)
        }
        return reused
    }

    /// Removes `finchmoe-range-*.tmp` staging files left by an interrupted
    /// transfer. Only that prefix is touched, and only inside `.partial`.
    private static func removeStagingFiles(inPartialDirectory directory: String) {
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory)
        else { return }
        for name in names where name.hasPrefix("finchmoe-range-") && name.hasSuffix(".tmp") {
            try? FileManager.default.removeItem(
                atPath: (directory as NSString).appendingPathComponent(name))
        }
    }

    private static func fileSize(_ path: String) throws -> UInt64? {
        var info = stat()
        guard stat(path, &info) == 0 else {
            throw RepackError.fileStatFailed(path: path, errno: errno)
        }
        return UInt64(info.st_size)
    }

    private static func loadCheckpoint(paths: RemoteInstallPaths,
                                       fingerprint: FinchDistributionCheckpoint.Fingerprint)
        throws -> FinchDistributionCheckpoint {
        guard try Posix.entryKind(paths.checkpointFile) == .regular else {
            // No checkpoint. Any existing partial belongs to a run we cannot
            // identify, so it is cleared rather than mixed with this one.
            if try Posix.entryKind(paths.partialDirectory) == .directory {
                try FileManager.default.removeItem(atPath: paths.partialDirectory)
            }
            try Posix.mkdirP(paths.partialDirectory)
            return FinchDistributionCheckpoint(fingerprint: fingerprint)
        }
        let existing = try FinchDistributionCheckpoint.load(from: paths.checkpointFile)
        guard existing.fingerprint == fingerprint else {
            throw RepackError.installStateIncompatible(
                detail: "saved download belongs to \(existing.fingerprint.repoID) "
                    + "at \(existing.fingerprint.resolvedCommit), not "
                    + "\(fingerprint.repoID) at \(fingerprint.resolvedCommit)")
        }
        return existing
    }

    // MARK: - Helpers

    /// Smallest first, ties broken by path so the order is deterministic.
    ///
    /// Alphabetical order starts the workers on `model_weights.bin` and the
    /// expert layers, which are hundreds of MB each — nothing completes for
    /// several minutes, so a download cancelled early leaves the checkpoint
    /// empty and the next run restarts from nothing. Taking the small files
    /// first fills the checkpoint within seconds, which is what makes an
    /// interrupted download actually resumable. It also gives the UI a
    /// non-zero reused-bytes figure to show on the next run.
    private static func orderedFiles(of manifest: FinchManifestV1) -> [String] {
        manifest.files.keys.sorted {
            let left = manifest.files[$0]?.size ?? 0
            let right = manifest.files[$1]?.size ?? 0
            return left == right ? $0 < $1 : left < right
        }
    }

    private static func decodeManifest(_ data: Data) throws -> FinchManifestV1 {
        do {
            return try FinchManifestCodec.decode(data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func hash(_ data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }

    private static func total(of manifest: FinchManifestV1) throws -> UInt64 {
        var total: UInt64 = 0
        for entry in manifest.files.values {
            total = try VerifiedInstallTool.addingVerifiedBytes(total, entry.size)
        }
        return total
    }

    private static func promote(paths: RemoteInstallPaths) throws {
        if try Posix.entryKind(paths.finalDirectory) == .directory {
            try Posix.renameSwap(paths.partialDirectory, paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
            try? FileManager.default.removeItem(atPath: paths.partialDirectory)
        } else {
            try Posix.rename(from: paths.partialDirectory, to: paths.finalDirectory)
            try Posix.fsyncDirectory(paths.parentDirectory)
        }
    }
}

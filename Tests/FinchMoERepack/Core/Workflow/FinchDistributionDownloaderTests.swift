import Foundation
import Testing

import FinchMoEFormat
@testable import FinchMoERepackCore

/// Covers the two pieces of the distribution download whose failure mode is
/// silent: the resume check, which decides whether bytes already on disk may be
/// trusted, and the progress accounting, which the UI reads.
@Suite
struct FinchDistributionDownloaderTests {

    // MARK: - Checkpoint

    @Test func checkpointRoundTrips() throws {
        let root = tmpDirForRemote("dist-checkpoint")
        let path = (root as NSString).appendingPathComponent("resume.json")
        defer { cleanUpDownload(root) }
        try Posix.mkdirP(root)

        var checkpoint = FinchDistributionCheckpoint(fingerprint: sampleFingerprint())
        checkpoint.record(FinchDistributionCheckpoint.Entry(
            relativePath: "packed_experts/layer_00.bin",
            size: 432,
            sha256: String(repeating: "a", count: 64)))

        try checkpoint.write(to: path)

        #expect(try FinchDistributionCheckpoint.load(from: path) == checkpoint)
        #expect(try Data(contentsOf: URL(fileURLWithPath: path)).count < 4096)
    }

    @Test func oversizedCheckpointIsRejectedBeforeDecode() throws {
        let root = tmpDirForRemote("dist-checkpoint-large")
        let path = (root as NSString).appendingPathComponent("resume.json")
        defer { cleanUpDownload(root) }
        try Posix.mkdirP(root)
        let descriptor = try Posix.openCreateRW(path)
        try Posix.ftruncate(descriptor,
                            path: path,
                            size: FinchDistributionCheckpoint.maximumBytes + 1)
        close(descriptor)

        #expect(throws: RepackError.self) {
            _ = try FinchDistributionCheckpoint.load(from: path)
        }
    }

    /// A download is only resumable when it would produce the same bytes, and
    /// the manifest digest alone pins the file set and every expected digest.
    @Test func fingerprintDistinguishesRepoCommitAndManifest() {
        let base = sampleFingerprint()
        #expect(base == sampleFingerprint())
        #expect(base != FinchDistributionCheckpoint.Fingerprint(
            repoID: "other/model",
            requestedRevision: base.requestedRevision,
            resolvedCommit: base.resolvedCommit,
            manifestSha256: base.manifestSha256))
        #expect(base != FinchDistributionCheckpoint.Fingerprint(
            repoID: base.repoID,
            requestedRevision: base.requestedRevision,
            resolvedCommit: String(repeating: "f", count: 40),
            manifestSha256: base.manifestSha256))
        #expect(base != FinchDistributionCheckpoint.Fingerprint(
            repoID: base.repoID,
            requestedRevision: base.requestedRevision,
            resolvedCommit: base.resolvedCommit,
            manifestSha256: String(repeating: "e", count: 64)))
    }

    @Test func recordingTheSamePathReplacesRatherThanDuplicates() {
        var checkpoint = FinchDistributionCheckpoint(fingerprint: sampleFingerprint())
        checkpoint.record(.init(relativePath: "a.bin", size: 1, sha256: "one"))
        checkpoint.record(.init(relativePath: "b.bin", size: 1, sha256: "two"))
        checkpoint.record(.init(relativePath: "a.bin", size: 2, sha256: "three"))

        #expect(checkpoint.entries.count == 2)
        #expect(checkpoint.entry(for: "a.bin")?.sha256 == "three")
    }

    // MARK: - Resume validation

    /// The whole point of the re-hash: a file of exactly the right size whose
    /// contents are wrong must not be adopted. Size alone cannot tell a finished
    /// file from a torn one, so trusting the checkpoint here would produce an
    /// install whose receipt was computed from the corruption.
    @Test func resumeAdoptsOnlyFilesThatStillHashToTheManifestDigest() throws {
        let root = tmpDirForRemote("dist-resume")
        defer { cleanUpDownload(root) }
        try Posix.mkdirP(root)

        let good = Data(repeating: 0x41, count: 512)
        let tornOnDisk = Data(repeating: 0x42, count: 512)   // right size, wrong bytes
        let shortOnDisk = Data(repeating: 0x43, count: 100)  // wrong size
        let promised = Data(repeating: 0x44, count: 512)     // what the manifest asks for

        try good.write(to: URL(fileURLWithPath: path(root, "good.bin")))
        try tornOnDisk.write(to: URL(fileURLWithPath: path(root, "torn.bin")))
        try shortOnDisk.write(to: URL(fileURLWithPath: path(root, "short.bin")))
        // "missing.bin" is never written.

        // The manifest digests describe what these files *should* be. `torn.bin`
        // is deliberately the size the manifest expects while holding different
        // bytes — that is the case size checking alone cannot catch.
        #expect(tornOnDisk.count == promised.count)
        let expected = [
            "good.bin": manifestFile(good),
            "torn.bin": manifestFile(promised),
            "short.bin": manifestFile(promised),
            "missing.bin": manifestFile(promised)
        ]
        let names = expected.keys.sorted()

        var checkpoint = FinchDistributionCheckpoint(fingerprint: sampleFingerprint())
        for name in names {
            checkpoint.record(.init(relativePath: name, size: 0, sha256: "claimed"))
        }

        let reused = try FinchDistributionDownloader.validateReusedFiles(
            expected: expected,
            files: names,
            partialDirectory: root,
            checkpoint: &checkpoint)

        #expect(reused == UInt64(good.count))
        #expect(checkpoint.entry(for: "good.bin") != nil)
        #expect(checkpoint.entry(for: "torn.bin") == nil)
        #expect(checkpoint.entry(for: "short.bin") == nil)
        #expect(checkpoint.entry(for: "missing.bin") == nil)
    }

    @Test func resumeIgnoresFilesWithNoCheckpointEntry() throws {
        let root = tmpDirForRemote("dist-resume-none")
        defer { cleanUpDownload(root) }
        try Posix.mkdirP(root)
        let data = Data(repeating: 0x7, count: 64)
        try data.write(to: URL(fileURLWithPath: path(root, "present.bin")))

        var checkpoint = FinchDistributionCheckpoint(fingerprint: sampleFingerprint())
        let reused = try FinchDistributionDownloader.validateReusedFiles(
            expected: ["present.bin": manifestFile(data)],
            files: ["present.bin"],
            partialDirectory: root,
            checkpoint: &checkpoint)

        // Present and matching, but nothing claimed it was finished.
        #expect(reused == 0)
        #expect(checkpoint.entries.isEmpty)
    }

    // MARK: - Progress accounting

    /// `RemoteRangeTransfer` reports a *cumulative* byte count for its own
    /// transfer, and the retry envelope re-reports from zero when it restarts.
    /// Summing those readings as if they were deltas overcounts by whatever the
    /// callback and retry cadence happens to produce.
    @Test func cumulativeReadingsAreDifferencedNotSummed() {
        let reported = ReportedBytes()
        let counters = FinchDistributionDownloader.DownloadCounters(
            reusedBytes: 100,
            totalBytes: 1000,
            progress: { reported.record($0) })

        counters.update("a.bin", cumulative: 100)
        counters.update("a.bin", cumulative: 250)
        counters.update("b.bin", cumulative: 50)
        counters.update("a.bin", cumulative: 250)

        #expect(reported.values == [100, 250, 300, 300])
    }

    @Test func restartedTransferMovesItsContributionBackDown() {
        let reported = ReportedBytes()
        let counters = FinchDistributionDownloader.DownloadCounters(
            reusedBytes: 0,
            totalBytes: 1000,
            progress: { reported.record($0) })

        counters.update("a.bin", cumulative: 400)
        counters.update("b.bin", cumulative: 100)
        counters.update("a.bin", cumulative: 0)     // a.bin's transfer restarted

        #expect(reported.values == [400, 500, 100])
    }

    /// The progress callback is `@Sendable`, so the readings it collects live in
    /// a reference type rather than a captured `var`.
    private final class ReportedBytes: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [UInt64] = []

        var values: [UInt64] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }

        func record(_ event: ModelInstallProgress) {
            guard case .copyingPayload(_, let downloaded, _) = event else { return }
            lock.lock()
            storage.append(downloaded)
            lock.unlock()
        }
    }

    // MARK: - Helpers

    private func path(_ root: String, _ name: String) -> String {
        (root as NSString).appendingPathComponent(name)
    }

    private func manifestFile(_ data: Data) -> FinchManifestFileV1 {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return FinchManifestFileV1(size: UInt64(data.count),
                                   sha256: hasher.finalizeHexString())
    }

    private func sampleFingerprint() -> FinchDistributionCheckpoint.Fingerprint {
        FinchDistributionCheckpoint.Fingerprint(
            repoID: "haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated",
            requestedRevision: "main",
            resolvedCommit: String(repeating: "a", count: 40),
            manifestSha256: String(repeating: "b", count: 64))
    }

    private func cleanUpDownload(_ root: String) {
        cleanUpRemote([root])
        try? FileManager.default.removeItem(atPath: root + ".resume.json")
    }
}

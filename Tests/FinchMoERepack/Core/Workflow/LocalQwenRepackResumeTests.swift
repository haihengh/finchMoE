import Testing
import Foundation
import FinchMoEFormat
@testable import FinchMoERepackCore

/// `--resume` on the local snapshot path.
///
/// Every interesting failure here is about trusting the wrong thing: a file
/// the journal calls complete that is torn, a partial directory built from a
/// different source, a partial with no journal at all. Each is a way to end up
/// with an install that looks whole and is not, so each has a case that
/// asserts the refusal or the rewrite — and the success cases assert a resumed
/// install is byte-identical to a clean one, which is the only claim that
/// matters.
///
/// The fixture is the toy Qwen 3.8 snapshot (4 layers, 4 PLE parts, 4
/// experts), driven through the real `LocalQwenRepacker`.
@Suite struct LocalQwenRepackResumeTests {

    // MARK: - Fixtures

    private static func makeSnapshot38() throws -> String {
        let dir = NSTemporaryDirectory() + "qwen38-resume-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write38(into: dir)
        return dir
    }

    private static func makeOutput() -> String {
        NSTemporaryDirectory() + "qwen38-resume-out-\(UUID().uuidString)"
    }

    private static func repack(snapshot: String,
                               output: String,
                               resume: Bool,
                               audit: RepackAudit = RepackAudit()) async throws
        -> LocalQwenRepackResult {
        let options = LocalQwenRepackOptions(snapshotDir: snapshot,
                                             outputDir: output,
                                             resume: resume,
                                             minFreeReserveBytes: 0)
        return try await LocalQwenRepacker(options: options, audit: audit).run()
    }

    private static func manifestFiles(_ installDir: String) throws
        -> [String: FinchManifestFileV1] {
        let data = try Data(contentsOf: URL(fileURLWithPath: installDir + "/manifest.json"))
        return try FinchManifestCodec.decode(data).files
    }

    /// The fingerprint `runPrepared` builds for this snapshot/output pair,
    /// recomputed here from the same inputs deliberately: if the repacker ever
    /// starts pinning something else, these fixtures stop matching and the
    /// resume tests fail loudly instead of quietly resuming onto a stale
    /// partial.
    private static func fingerprint(snapshot: String,
                                    output: String) throws -> LocalRepackJournal.Fingerprint {
        let loaded = try QwenLocalSnapshot.load(snapshotDir: snapshot)
        return LocalRepackJournal.Fingerprint(
            sourceDirectory: URL(fileURLWithPath: snapshot).path,
            sourceIndexSha256: loaded.metadata.indexSha256Hex,
            outputDirectory: URL(fileURLWithPath: output).path,
            modelFamily: loaded.arch.modelFamily,
            numLayers: loaded.arch.numLayers)
    }

    /// Stages the state a run killed *after* finishing every file would leave:
    /// a complete install sitting in the `.partial` directory with a journal
    /// recording each of its files, digest and all. Returns the manifest of the
    /// clean run it was built from, which is what a resumed run must reproduce.
    ///
    /// `torn` names one file to corrupt in place — rewritten at its recorded
    /// size but with the wrong bytes, which is exactly what a kill mid-write
    /// leaves behind, since every writer here `ftruncate`s to the final size
    /// before filling.
    @discardableResult
    private static func stagePartial(fromCleanRunOf snapshot: String,
                                     output: String,
                                     torn: String? = nil,
                                     dropJournal: Bool = false) async throws
        -> [String: FinchManifestFileV1] {
        let clean = try await repack(snapshot: snapshot, output: output, resume: false)
        let files = try manifestFiles(clean.outputDir)

        let partial = output + ".partial"
        try FileManager.default.moveItem(atPath: clean.outputDir, toPath: partial)

        if let torn {
            let path = (partial as NSString).appendingPathComponent(torn)
            let size = try #require(files[torn]).size
            // Right length, wrong content: a size check alone cannot see this.
            try Data(count: Int(size)).write(to: URL(fileURLWithPath: path))
        }
        guard !dropJournal else { return files }

        var journal = LocalRepackJournal(fingerprint: try fingerprint(
            snapshot: snapshot, output: output))
        for (relativePath, entry) in files {
            journal.record(LocalRepackJournal.Entry(relativePath: relativePath,
                                                    size: entry.size,
                                                    sha256: entry.sha256))
        }
        try journal.write(to: LocalRepackJournal.path(inPartialDirectory: partial))
        return files
    }

    // MARK: - The happy path

    /// A resumed run reuses every journaled file and rebuilds the same install.
    @Test func resumeRebuildsTheCleanInstallByteForByte() async throws {
        let snapshot = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: snapshot) }

        let reference = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: reference) }
        let referenceRun = try await Self.repack(snapshot: snapshot, output: reference, resume: false)
        let referenceManifest = try Data(contentsOf: URL(
            fileURLWithPath: referenceRun.outputDir + "/manifest.json"))

        let output = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: output) }
        let staged = try await Self.stagePartial(fromCleanRunOf: snapshot, output: output)

        let reusedAudit = RepackAudit()
        let resumed = try await Self.repack(snapshot: snapshot, output: output,
                                            resume: true, audit: reusedAudit)

        // The resumed install *is* the clean install: same manifest bytes, and
        // therefore the same digest for every file it lists.
        let resumedManifest = try Data(contentsOf: URL(
            fileURLWithPath: resumed.outputDir + "/manifest.json"))
        #expect(resumedManifest == referenceManifest)
        #expect(try Self.manifestFiles(resumed.outputDir) == staged)

        // The journal is run bookkeeping, not install content.
        #expect(!FileManager.default.fileExists(
            atPath: resumed.outputDir + "/" + LocalRepackJournal.fileName))
        #expect(!FileManager.default.fileExists(atPath: output + ".partial"))

        // Reuse, not just correctness. `sourceBytesRead` counts the tensor
        // bytes the writers pulled out of the snapshot, so a fully reused run
        // reads none of them — it never had to look at the source to know the
        // file on disk was right. `outputBytesWritten` counts what actually
        // reached disk, which for a reused run is the manifest and receipt and
        // nothing else; the bound is the resident file's own size, so the
        // assertion scales with the fixture rather than pinning a byte count.
        #expect(reusedAudit.sourceBytesRead == 0)
        let residentSize = try #require(staged["model_weights.bin"]).size
        #expect(reusedAudit.outputBytesWritten < residentSize)
    }

    // MARK: - Refusals

    /// The guard the old unconditional refusal provided, kept: a partial is
    /// never written into by a run that would produce different bytes.
    @Test func resumeRefusesAPartialWrittenFromADifferentSource() async throws {
        let snapshot38 = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: snapshot38) }
        let output = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: output) }
        try await Self.stagePartial(fromCleanRunOf: snapshot38, output: output)

        // A different family with different tensors: nothing about this run
        // would produce the bytes already sitting in the partial.
        let snapshot36 = NSTemporaryDirectory() + "qwen36-resume-src-\(UUID().uuidString)"
        try SyntheticQwenSnapshot.write(into: snapshot36)
        defer { try? FileManager.default.removeItem(atPath: snapshot36) }

        do {
            _ = try await Self.repack(snapshot: snapshot36, output: output, resume: true)
            Issue.record("resumed from a partial belonging to a different source")
        } catch let error as RepackError {
            guard case .installStateIncompatible(let detail) = error else {
                Issue.record("expected installStateIncompatible, got \(error)")
                return
            }
            #expect(detail.contains("different source"))
        }
    }

    /// A partial with no journal records nothing as complete, so there is
    /// nothing to build on and no way to tell what is stale.
    @Test func resumeRefusesAPartialWithNoJournal() async throws {
        let snapshot = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: snapshot) }
        let output = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: output) }
        try await Self.stagePartial(fromCleanRunOf: snapshot, output: output, dropJournal: true)

        do {
            _ = try await Self.repack(snapshot: snapshot, output: output, resume: true)
            Issue.record("resumed from a partial with no journal")
        } catch let error as RepackError {
            guard case .installStateCorrupt(_, let detail) = error else {
                Issue.record("expected installStateCorrupt, got \(error)")
                return
            }
            #expect(detail.contains(LocalRepackJournal.fileName))
        }
    }

    /// Without `--resume` a stale partial is refused exactly as it was before
    /// resume existed, journal or not.
    @Test func aStalePartialIsStillRefusedWithoutResume() async throws {
        let snapshot = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: snapshot) }
        let output = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: output) }
        try await Self.stagePartial(fromCleanRunOf: snapshot, output: output)

        do {
            _ = try await Self.repack(snapshot: snapshot, output: output, resume: false)
            Issue.record("overwrote a stale partial without --resume")
        } catch let error as RepackError {
            guard case .installStateCorrupt(_, let detail) = error else {
                Issue.record("expected installStateCorrupt, got \(error)")
                return
            }
            #expect(detail.contains("--resume"))
        }
    }

    // MARK: - A file the journal is wrong about

    /// The journal says a file is complete; the disk disagrees. The digest is
    /// the only thing that can catch it — the file is the recorded length, so
    /// the size check passes — and the file must be rewritten, not trusted.
    @Test func aTornFileIsRewrittenRatherThanTrusted() async throws {
        let snapshot = try Self.makeSnapshot38()
        defer { try? FileManager.default.removeItem(atPath: snapshot) }

        let reference = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: reference) }
        let referenceRun = try await Self.repack(snapshot: snapshot, output: reference, resume: false)
        let referenceManifest = try Data(contentsOf: URL(
            fileURLWithPath: referenceRun.outputDir + "/manifest.json"))

        let output = Self.makeOutput()
        defer { try? FileManager.default.removeItem(atPath: output) }
        let tornPath = "ple_shards/shard_000.bin"
        let staged = try await Self.stagePartial(fromCleanRunOf: snapshot,
                                                 output: output, torn: tornPath)

        // The torn file still has the recorded size — this is the case a size
        // check alone would wave through.
        let onDisk = try #require(staged[tornPath]).size
        let attrs = try FileManager.default.attributesOfItem(
            atPath: (output + ".partial" as NSString).appendingPathComponent(tornPath))
        #expect((attrs[.size] as? NSNumber)?.uint64Value == onDisk)

        _ = try await Self.repack(snapshot: snapshot, output: output, resume: true)

        // Rewritten, byte for byte. Asserting on the file itself rather than on
        // the manifest matters: a size-only implementation would have reused
        // the zeros and then written the *journal's* digest into the manifest,
        // so the manifest would look perfect while the install was wrong.
        let manifest = try Data(contentsOf: URL(
            fileURLWithPath: output + "/manifest.json"))
        #expect(manifest == referenceManifest)
        let original = try Data(contentsOf: URL(fileURLWithPath:
            (referenceRun.outputDir as NSString).appendingPathComponent(tornPath)))
        let rewritten = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent(tornPath)))
        #expect(rewritten == original)
    }
}

import Foundation

/// A published, already-repacked `.finch` install hosted on Hugging Face.
///
/// This is the engine's second install route, and it is the mirror image of
/// `SupportedModelSource`. That one describes a *repack* source — an upstream
/// safetensors checkpoint whose bytes have to be rewritten into the `.finch`
/// layout on the way down, which is why it needs a shard index, a range plan,
/// and a fingerprint allowlist. A distribution is the opposite: the `.finch`
/// layout is already on the remote, and the `manifest.json` that ships beside
/// it declares every file with its size and digest.
///
/// That property is what makes this path cheap to verify: **the manifest is
/// both the download set and the integrity check**. There is no planning step
/// and nothing to fingerprint — `FinchDistributionDownloader` fetches exactly
/// `manifest.files` and re-hashes each one against the digest that named it.
///
/// `revision` is a pinned commit, not a branch. It is recorded in the install
/// checkpoint and in the receipt, so a resumed download can prove it is
/// finishing the same install it started.
public struct FinchDistribution: Sendable, Equatable {
    public let repoID: String
    public let revision: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64

    public init(repoID: String,
                revision: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64) {
        self.repoID = repoID
        self.revision = revision
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
    }
}

/// Resume state for a distribution download: which files are finished.
///
/// Deliberately not `RemoteInstallCheckpoint`. That one tracks byte *ranges*
/// within a source shard, keyed by a range-plan fingerprint that only exists
/// because the repacker computed a plan. A distribution download has no plan —
/// the files are the unit — so the schema is file-level.
///
/// **Why a completed file is re-hashed on resume.** A torn `.range.tmp` is not
/// the hazard here (the transfer removes its target on failure); the hazard is
/// that a checkpoint entry can outlive the bytes it describes when the process
/// dies between the file's `fsync` and the checkpoint's, or when an unclean
/// shutdown loses data that `fsync` had not yet forced to the platter on macOS
/// (`Posix.fsync` issues `fsync`, not `F_FULLFSYNC`). Size alone cannot tell a
/// finished file from a torn one. Re-reading what is already on disk costs one
/// sequential pass per resume; trusting a torn file costs a silently corrupt
/// 97 GiB install whose digests are recomputed *from the corruption*, so
/// nothing downstream could catch it. Same reasoning, and the same vocabulary,
/// as `LocalRepackJournal`.
public struct FinchDistributionCheckpoint: Codable, Equatable, Sendable {

    /// Bumped when the on-disk shape changes; an older checkpoint is refused
    /// rather than misread.
    public static let currentVersion = 1

    /// The largest install is Qwen 3.8 at 189 files; the cap is headroom.
    public static let maximumBytes: UInt64 = 8 << 20

    /// What a partial directory must match to be resumable. A download that
    /// would produce the same bytes is one whose repo, commit and manifest
    /// digest all agree — the manifest digest alone pins the file set and every
    /// file's expected digest, so agreeing on it means agreeing on everything
    /// the download will write.
    public struct Fingerprint: Codable, Equatable, Sendable {
        public let repoID: String
        public let requestedRevision: String
        public let resolvedCommit: String
        public let manifestSha256: String

        public init(repoID: String,
                    requestedRevision: String,
                    resolvedCommit: String,
                    manifestSha256: String) {
            self.repoID = repoID
            self.requestedRevision = requestedRevision
            self.resolvedCommit = resolvedCommit
            self.manifestSha256 = manifestSha256
        }
    }

    /// One finished file. `sha256` is the digest computed from the bytes on
    /// disk, so it is exactly the digest the manifest asked for.
    public struct Entry: Codable, Equatable, Sendable {
        public let relativePath: String
        public let size: UInt64
        public let sha256: String

        public init(relativePath: String, size: UInt64, sha256: String) {
            self.relativePath = relativePath
            self.size = size
            self.sha256 = sha256
        }
    }

    public let version: Int
    public let fingerprint: Fingerprint
    public var entries: [Entry]

    public init(fingerprint: Fingerprint) {
        self.version = Self.currentVersion
        self.fingerprint = fingerprint
        self.entries = []
    }

    /// Reads a checkpoint. Throws `installStateCorrupt` for anything present but
    /// unreadable — a checkpoint that cannot be trusted is never treated as
    /// absent, because "absent" would silently mean "start over" while the
    /// stale partial directory is still there.
    public static func load(from path: String) throws -> FinchDistributionCheckpoint {
        let data: Data
        do {
            data = try Posix.readBoundedData(path, maximumBytes: maximumBytes)
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
        let checkpoint: FinchDistributionCheckpoint
        do {
            checkpoint = try JSONDecoder().decode(FinchDistributionCheckpoint.self, from: data)
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
        guard checkpoint.version == currentVersion else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "checkpoint version \(checkpoint.version) is not \(currentVersion)")
        }
        return checkpoint
    }

    /// Writes the whole checkpoint durably: temp -> fsync -> rename -> fsync dir.
    /// It is a few tens of KB at full length, so rewriting it per completed file
    /// costs far less than a partial rewrite that could tear.
    public func write(to path: String) throws {
        let directory = (path as NSString).deletingLastPathComponent
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data: Data
        do {
            data = try encoder.encode(self)
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
        try Posix.atomicWrite(data, to: path, durableIn: directory)
    }

    public func entry(for relativePath: String) -> Entry? {
        entries.first { $0.relativePath == relativePath }
    }

    /// Records a finished file. A repeated path replaces the earlier entry: the
    /// later write is the one on disk.
    public mutating func record(_ entry: Entry) {
        if let index = entries.firstIndex(where: { $0.relativePath == entry.relativePath }) {
            entries[index] = entry
        } else {
            entries.append(entry)
        }
    }

    public mutating func forget(relativePath: String) {
        entries.removeAll { $0.relativePath == relativePath }
    }
}

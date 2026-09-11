import Foundation

/// Completion journal for the local snapshot repack — the state that makes
/// `--resume` safe there. It lives inside the `.partial` directory, so it
/// disappears with the partial: `--discard-partial` removes both, and a clean
/// run drops it before promoting the directory.
///
/// It is deliberately not the remote path's `RemoteInstallCheckpoint`: that
/// one tracks which byte ranges of a *download* arrived, while this one tracks
/// which local *output files* are finished. The two share nothing but the
/// `installState*` error vocabulary.
///
/// **Why a file present in the journal is still re-hashed on resume.** Every
/// writer in this repack `ftruncate`s its output to the final size *before*
/// filling it (`QwenQuantizedWriter.swift:29,347,410`), so a run killed
/// mid-file leaves a file of exactly the right size whose tail is a hole —
/// size alone cannot tell a finished file from a torn one. And on macOS an
/// `fsync` (which is what `Posix.fsync` issues, not `F_FULLFSYNC`) does not
/// promise the data reached the platter, so a journal entry that survived an
/// unclean shutdown can outlive the bytes it describes. Re-reading what is
/// already on disk costs one sequential pass, and it happens once per resume;
/// trusting a torn file costs a silently corrupt 145 GB install whose digests
/// are recomputed *from the corruption*, so nothing downstream could catch it.
struct LocalRepackJournal: Codable {

    static let fileName = ".repack-journal.json"

    /// Bumped when the on-disk shape changes; an older journal is refused
    /// rather than misread.
    static let currentVersion = 1

    /// A real Qwen 3.8 install journals 179 entries — 1 resident, 48 expert
    /// layers, 128 PLE parts, `layout.json`, and the tokenizer sidecars. The
    /// cap is generous headroom, not a target.
    static let maximumBytes: UInt64 = 8 << 20

    /// What a partial directory has to match to be resumable: a run that would
    /// produce the same bytes. The snapshot index digest pins the weights, the
    /// family and depth pin the plan, and the output path is recorded verbatim
    /// in the manifest and the receipt — so resuming across any of them would
    /// write a file that describes a different install than the one it is
    /// inside of.
    struct Fingerprint: Codable, Equatable {
        let sourceDirectory: String
        let sourceIndexSha256: String
        let outputDirectory: String
        /// Optional because `ArchInfo` carries it optional; a snapshot with no
        /// recognised family fingerprints as such rather than as the empty
        /// string, which a real family could never be.
        let modelFamily: String?
        let numLayers: Int
    }

    /// One finished output file. `sha256` is the digest the writer computed
    /// after its `fsync`, so it is exactly the digest the manifest will carry.
    struct Entry: Codable {
        let relativePath: String
        let size: UInt64
        let sha256: String
    }

    let version: Int
    let fingerprint: Fingerprint
    var entries: [Entry]

    init(fingerprint: Fingerprint) {
        self.version = Self.currentVersion
        self.fingerprint = fingerprint
        self.entries = []
    }

    static func path(inPartialDirectory directory: String) -> String {
        (directory as NSString).appendingPathComponent(fileName)
    }

    /// Reads a journal. Throws `installStateCorrupt` for anything present but
    /// unreadable — a journal that cannot be trusted is never treated as
    /// absent, because "absent" would silently mean "start over" while the
    /// stale partial directory is still there.
    static func load(from path: String) throws -> LocalRepackJournal {
        let data: Data
        do {
            data = try Posix.readBoundedData(path, maximumBytes: maximumBytes)
        } catch let error as RepackError {
            throw error
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
        let journal: LocalRepackJournal
        do {
            journal = try JSONDecoder().decode(LocalRepackJournal.self, from: data)
        } catch {
            throw RepackError.installStateCorrupt(path: path, detail: "\(error)")
        }
        guard journal.version == currentVersion else {
            throw RepackError.installStateCorrupt(
                path: path,
                detail: "journal version \(journal.version) is not \(currentVersion)")
        }
        return journal
    }

    /// Writes the whole journal durably: temp → fsync → rename → fsync dir.
    /// The file is a few tens of KB even at full length, so rewriting it per
    /// completed file costs far less than a partial rewrite that could tear.
    func write(to path: String) throws {
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

    func entry(for relativePath: String) -> Entry? {
        entries.first { $0.relativePath == relativePath }
    }

    /// Records a finished file. A repeated path replaces the earlier entry:
    /// the later write is the one on disk.
    mutating func record(_ entry: Entry) {
        if let index = entries.firstIndex(where: { $0.relativePath == entry.relativePath }) {
            entries[index] = entry
        } else {
            entries.append(entry)
        }
    }
}

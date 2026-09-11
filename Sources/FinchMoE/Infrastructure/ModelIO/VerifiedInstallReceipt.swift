import Foundation

public enum ModelIntegrityPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt
}

/// What the caller *asks* for. Distinct from `ModelIntegrityPolicy`, which is
/// what the loader *resolved to*: `automatic` is not a policy, it is a rule for
/// choosing one, and the choice depends on state only the loader can see (whether
/// a usable receipt exists on disk).
///
/// Keeping `automatic` out of `ModelIntegrityPolicy` is deliberate. Both lazy
/// verification gates (`Model.swift` layer + PLE part) switch exhaustively over
/// the policy, so a third case would force a third arm at each — and the obvious
/// thing to write there (`case .automatic: break`) is precisely the bug:
/// "unresolved" would then mean "skip the hash".
public enum ModelIntegrityPreference: Sendable, Equatable {
    /// Use the trusted-install receipt when one is present and valid; otherwise
    /// fall back to hashing. Never verifies less than `.fullSha256` — the
    /// fallback hashes, and the receipt path only skips hashing that a
    /// validated receipt independently covers.
    case automatic
    case fullSha256
    case sizeCheckTrustedReceipt
}

/// What actually happened, for callers that want to report it. A `Model` whose
/// `integrityPolicy == .fullSha256` may have reached that policy by explicit
/// request or by falling back, and only this tells them apart.
public enum ModelIntegrityOutcome: Sendable, Equatable {
    case explicitFullSha256
    case explicitTrustedReceipt
    case automaticUsedReceipt
    case automaticFellBackAbsent
    case automaticFellBackInvalid(detail: String)

    /// The message to emit, or `nil` when the caller should stay quiet.
    ///
    /// Only a *present but unusable* receipt is a signal: an absent one is the
    /// normal state of an install that was never `--verify-install`ed, and
    /// warning on it would be noise on every fresh checkout. Keeping the
    /// silent-vs-warn rule here is what stops the CLI, the server and the app
    /// from disagreeing about which cases are worth surfacing.
    public var warningMessage: String? {
        guard case .automaticFellBackInvalid(let detail) = self else { return nil }
        return "\(VerifiedInstallReceiptReader.fileName) is present but unusable "
            + "(\(detail)); verified with full SHA-256 instead"
    }

    public var isWarning: Bool { warningMessage != nil }
}

public struct VerifiedInstallReceipt: Codable, Equatable, Sendable {
    public struct FileEntry: Codable, Equatable, Sendable {
        public let size: UInt64
        public let sha256: String

        public init(size: UInt64, sha256: String) {
            self.size = size
            self.sha256 = sha256
        }
    }

    public let schemaVersion: Int
    public let manifestSha256: String
    public let modelDirectoryPath: String
    public let sourceRepoID: String?
    public let sourceRevision: String?
    public let verificationTimestamp: String
    public let toolVersion: String
    public let files: [String: FileEntry]

    public init(schemaVersion: Int = 1,
                manifestSha256: String,
                modelDirectoryPath: String,
                sourceRepoID: String? = nil,
                sourceRevision: String? = nil,
                verificationTimestamp: String,
                toolVersion: String,
                files: [String: FileEntry]) {
        self.schemaVersion = schemaVersion
        self.manifestSha256 = manifestSha256
        self.modelDirectoryPath = modelDirectoryPath
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.verificationTimestamp = verificationTimestamp
        self.toolVersion = toolVersion
        self.files = files
    }
}

public enum VerifiedInstallReceiptReader {
    public static let fileName = "verified-install.json"
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    public static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> VerifiedInstallReceipt {
        do {
            let directory = try FinchModelDirectory(rootURL: directoryURL)
            let data = try directory.readMetadata(fileName, maxBytes: maxBytes)
            return try decode(data: data)
        } catch ModelError.missingFile {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName) is missing")
        } catch let error as ModelError {
            if case .trustedReceiptInvalid = error { throw error }
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        } catch {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        }
    }

    /// Is a receipt file there at all?
    ///
    /// `load` collapses every failure into `.trustedReceiptInvalid`, so the error
    /// alone cannot tell "no receipt in this install" from "a receipt that is
    /// broken, unreadable, or not a regular file" — and those want opposite
    /// treatment (silence vs. a warning). Rather than string-matching `detail`,
    /// ask the filesystem. Only ENOENT counts as absent; a symlink, FIFO,
    /// directory or unreadable file is a *presence* signal, because something
    /// deliberately put something at that path.
    ///
    /// Inherits `openFile`'s `O_NOFOLLOW`, so a symlinked receipt reports present
    /// (and then fails the load) instead of silently following it.
    package static func isPresent(directoryURL: URL) -> Bool {
        do {
            let directory = try FinchModelDirectory(rootURL: directoryURL)
            let fd = try directory.openFile(fileName)
            close(fd)
            return true
        } catch ModelError.missingFile {
            return false
        } catch {
            return true
        }
    }

    package static func decode(data: Data) throws -> VerifiedInstallReceipt {
        do { return try JSONDecoder().decode(VerifiedInstallReceipt.self, from: data) }
        catch { throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)") }
    }

    public static func validate(_ receipt: VerifiedInstallReceipt,
                                directoryURL: URL,
                                manifest: Manifest,
                                manifestSha256: String,
                                manifestSize: UInt64) throws {
        try validateManifestBinding(receipt,
                                    directoryURL: directoryURL,
                                    manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }

        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
        }
    }

    public static func validateManifestBinding(_ receipt: VerifiedInstallReceipt,
                                               directoryURL: URL,
                                               manifestSha256: String) throws {
        guard receipt.schemaVersion == 1 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "unsupported schemaVersion \(receipt.schemaVersion)")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")
        }

        // The binding is physical-path based: a checkout whose models/ is a
        // symlink to the payload home must still match a receipt recorded
        // through the real directory (and vice versa). standardizedFileURL
        // keeps symlinks, so resolve them on both sides before comparing.
        let actualPath = directoryURL.resolvingSymlinksInPath().path
        let receiptPath = URL(fileURLWithPath: receipt.modelDirectoryPath)
            .resolvingSymlinksInPath().path
        guard receiptPath == actualPath else {
            throw ModelError.trustedReceiptInvalid(detail: "model directory mismatch")
        }
    }
}

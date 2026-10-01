import Foundation
import Darwin
import FinchMoEFormat

public struct VerifyInstallOptions: Sendable {
    public let inputFinch: String

    public init(inputFinch: String) {
        self.inputFinch = inputFinch
    }
}

public struct VerifyInstallResult: Sendable {
    public let receiptPath: String
    public let fileCount: Int
    public let bytesVerified: UInt64
    public let unexpectedEntries: [String]
}

public enum VerifiedInstallTool {
    public static let metadataMaxBytes: UInt64 = 16 * 1024 * 1024
    public static let manifestMaxBytes: UInt64 = 4 * 1024 * 1024
    // Qwen 3.6 (256 experts x 40 layers) produces a ~22 MB layout.json and
    // Qwen 3.8 (512 experts x 48 layers) near 55 MB; keep this in sync with
    // PackedExpertsLayoutReader.defaultMaxBytes.
    public static let layoutMaxBytes: UInt64 = 128 * 1024 * 1024

    /// Verifies an install in place and refreshes its receipt. The
    /// `sourceRevision` falls back to the manifest's own `sourceSnapshotHash`,
    /// which is what the CLI has always recorded.
    public static func run(options: VerifyInstallOptions) throws -> VerifyInstallResult {
        let root = URL(fileURLWithPath: options.inputFinch).standardizedFileURL
        return try verify(root: root,
                          recordedOutputDirectory: root.path,
                          sourceRepoID: nil,
                          sourceRevision: nil,
                          toolVersion: "FinchMoERepack verify-install")
    }

    /// The shared verification core, also used by the distribution downloader.
    ///
    /// `root` is the directory whose bytes are hashed, and the receipt is
    /// written into it. `recordedOutputDirectory` is the path recorded *inside*
    /// that receipt, and mid-install the two differ: the bytes are still in the
    /// `.partial` directory, but the receipt must name the final location,
    /// because `VerifiedInstallReceiptReader.validateManifestBinding` compares
    /// `modelDirectoryPath` against the directory the install is later opened
    /// from. The streaming repacker has always done this — it writes the receipt
    /// into `.partial` while encoding the final output path — and a download
    /// that recorded its own `.partial` path would fail binding the moment it
    /// was promoted.
    ///
    /// When `sourceRevision` is nil the manifest's `sourceSnapshotHash` is
    /// recorded instead, which is what the CLI has always written.
    public static func verify(root: URL,
                              recordedOutputDirectory: String,
                              sourceRepoID: String?,
                              sourceRevision: String?,
                              toolVersion: String,
                              onHashingFile: ((String) -> Void)? = nil)
        throws -> VerifyInstallResult {
        let access = try FinchDirectoryAccess(rootPath: root.path)
        let manifestFD = try access.openFile("manifest.json")
        defer { close(manifestFD) }
        _ = fcntl(manifestFD, F_NOCACHE, 1)
        let manifestData = try access.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: manifestMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestSha = hashMetadata(manifestData)
        let manifest = try loadManifest(data: manifestData)

        let layoutRelativePath = "packed_experts/layout.json"
        guard let layoutManifestEntry = manifest.files[layoutRelativePath] else {
            throw RepackError.configurationInvalid(
                detail: "manifest missing \(layoutRelativePath)")
        }
        let layoutFD = try access.openFile(layoutRelativePath)
        defer { close(layoutFD) }
        _ = fcntl(layoutFD, F_NOCACHE, 1)
        let layoutData = try access.readMetadata(
            fileDescriptor: layoutFD, relativePath: layoutRelativePath,
            maxBytes: layoutMaxBytes)
        let layoutSize = UInt64(layoutData.count)
        let layoutSha = hashMetadata(layoutData)
        let layout = try loadLayout(data: layoutData)
        try validatePackedExpertLayout(manifest: manifest, layout: layout)
        guard layoutSize == layoutManifestEntry.size else {
            throw RepackError.configurationInvalid(
                detail: "\(layoutRelativePath) size \(layoutSize) != manifest \(layoutManifestEntry.size)")
        }
        guard layoutSha.lowercased() == layoutManifestEntry.sha256.lowercased() else {
            throw RepackError.configurationInvalid(detail: "\(layoutRelativePath) SHA mismatch")
        }

        var files: [RepackAudit.OutputFile] = []
        files.reserveCapacity(manifest.files.count)
        var bytesVerified = manifestSize
        for relativePath in manifest.files.keys.sorted() {
            guard let entry = manifest.files[relativePath] else { continue }
            onHashingFile?(relativePath)
            let actualSize: UInt64
            let actualSha: String
            if relativePath == layoutRelativePath {
                actualSize = layoutSize
                actualSha = layoutSha
            } else {
                (actualSize, actualSha) = try inspectFile(
                    access: access, relativePath: relativePath)
            }
            guard actualSize == entry.size else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) size \(actualSize) != manifest \(entry.size)")
            }
            guard actualSha.lowercased() == entry.sha256.lowercased() else {
                throw RepackError.configurationInvalid(detail: "\(relativePath) SHA mismatch")
            }
            bytesVerified = try addingVerifiedBytes(bytesVerified, actualSize)
            files.append(RepackAudit.OutputFile(relativePath: relativePath,
                                                size: actualSize,
                                                sha256: actualSha))
        }
        let unexpectedEntries = try findUnexpectedEntries(access: access, manifest: manifest)

        let receiptData = try VerifiedInstallReceiptWriter.encode(
            outputDir: recordedOutputDirectory,
            manifestSha256: manifestSha,
            manifestSize: manifestSize,
            sourceRepoID: sourceRepoID,
            sourceRevision: sourceRevision ?? manifest.sourceSnapshotHash,
            toolVersion: toolVersion,
            files: files)
        let receiptPath = root.appendingPathComponent(VerifiedInstallReceiptWriter.fileName).path
        try access.atomicWrite(receiptData, to: VerifiedInstallReceiptWriter.fileName)
        return VerifyInstallResult(receiptPath: receiptPath,
                                   fileCount: files.count + 1,
                                   bytesVerified: bytesVerified,
                                   unexpectedEntries: unexpectedEntries)
    }

    private static func hashMetadata(_ data: Data) -> String {
        var hasher = Sha256Stream()
        data.withUnsafeBytes { hasher.update($0) }
        return hasher.finalizeHexString()
    }

    private static func inspectFile(access: FinchDirectoryAccess,
                                    relativePath: String) throws -> (UInt64, String) {
        let fd = try access.openFile(relativePath)
        defer { close(fd) }
        let size = try access.fileSize(
            fileDescriptor: fd, relativePath: relativePath)
        let sha = try access.hash(
            fileDescriptor: fd, relativePath: relativePath, noCache: true)
        return (size, sha)
    }

    package static func addingVerifiedBytes(_ current: UInt64,
                                            _ next: UInt64) throws -> UInt64 {
        let (result, overflow) = current.addingReportingOverflow(next)
        guard !overflow else {
            throw RepackError.configurationInvalid(
                detail: "verified byte count exceeds UInt64")
        }
        return result
    }

    private static func loadManifest(data: Data) throws -> FinchManifestV1 {
        do {
            return try FinchManifestCodec.decode(data)
        } catch {
            throw RepackError.configurationInvalid(detail: "manifest.json invalid: \(error)")
        }
    }

    private static func loadLayout(data: Data) throws -> FinchPackedExpertsLayoutV1 {
        do {
            return try FinchPackedExpertsLayoutCodec.decode(data)
        } catch {
            throw RepackError.configurationInvalid(detail: "packed_experts/layout.json invalid: \(error)")
        }
    }

    private static func validatePackedExpertLayout(manifest: FinchManifestV1,
                                                   layout: FinchPackedExpertsLayoutV1) throws {
        do { try FinchV1StructuralValidator.crossValidate(manifest: manifest, layout: layout) }
        catch {
            throw RepackError.configurationInvalid(
                detail: "packed expert layout does not match manifest: \(error)")
        }
        let expectedLayerSize = UInt64(layout.expertsPerLayer) * layout.expertStride
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw RepackError.configurationInvalid(detail: "manifest missing \(relativePath)")
            }
            guard manifestEntry.size == expectedLayerSize else {
                throw RepackError.configurationInvalid(
                    detail: "\(relativePath) manifest size \(manifestEntry.size) != \(expectedLayerSize)")
            }
            for (index, expert) in layer.experts.enumerated() {
                let expertID = expert.expert ?? index
                guard expert.offset <= expectedLayerSize,
                      expert.size <= expectedLayerSize - expert.offset else {
                    throw RepackError.configurationInvalid(
                        detail: "\(relativePath) expert \(expertID) range exceeds file size")
                }
            }
        }
    }

    private static func findUnexpectedEntries(access: FinchDirectoryAccess,
                                              manifest: FinchManifestV1) throws -> [String] {
        let declaredFiles = Set(manifest.files.keys)
            .union(["manifest.json", VerifiedInstallReceiptWriter.fileName])
        var allowed = declaredFiles
        for path in declaredFiles {
            var parts = path.split(separator: "/").map(String.init)
            while parts.count > 1 {
                _ = parts.removeLast()
                allowed.insert(parts.joined(separator: "/"))
            }
        }
        allowed.insert("tokenizer")
        let scanDepth = max(16, declaredFiles.lazy.map {
            $0.split(separator: "/", omittingEmptySubsequences: false).count
        }.max() ?? 1)

        var unexpected: [String] = []
        for rel in try access.relativeEntries(maxDepth: scanDepth) {
            if rel == ".DS_Store" { continue }
            if rel == "tokenizer" || rel.hasPrefix("tokenizer/") { continue }
            if !allowed.contains(rel) {
                unexpected.append(rel)
            }
        }
        return unexpected.sorted()
    }
}

import Foundation
import FinchMoE

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    /// The descriptor whose checkpoint a directory holds, keyed on the
    /// manifest's `sourceSnapshotHash`. A directory the app cannot identify
    /// (no manifest, unreadable, unknown hash) resolves to `.default`, which
    /// also keeps the not-yet-installed download flow on the Gemma install.
    /// Note: a Qwen directory whose receipt is corrupt/missing still resolves
    /// to its own descriptor and surfaces `.partial` in the install UI — Qwen
    /// is never remotely installable (repack-made installs only); repair is
    /// re-running the repack's receipt.
    ///
    /// The scan is over `AppModelInstallDescriptor.installable` in order rather
    /// than a chain of `if`s, so adding a family is one entry in one list. A
    /// family missing from that list is not a loud failure: the directory
    /// resolves to the Gemma default and the UI offers to download a model that
    /// is already there.
    public static func matchingDescriptor(at directory: URL) -> AppModelInstallDescriptor {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .default
        }
        do {
            let manifest = try ManifestReader.load(
                directoryURL: directory,
                expecting: try ManifestReader.detectPreset(directoryURL: directory))
            return AppModelInstallDescriptor.installable.first {
                manifest.sourceSnapshotHash == "sha256:" + $0.sourceIndexSHA256
            } ?? .default
        } catch {
            return .default
        }
    }

    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor = .default
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: try ManifestReader.detectPreset(directoryURL: directory))
            let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(descriptor.displayName)")
            }
            let layout = directory.appendingPathComponent("packed_experts/layout.json")
            guard FileManager.default.fileExists(atPath: layout.path) else {
                return .partial("packed_experts/layout.json is missing")
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}

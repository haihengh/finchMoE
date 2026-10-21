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
    ///
    /// Two axes separate entries that share a `sourceSnapshotHash`: the
    /// resident-weights digest (base vs abliterated) and the routed-expert
    /// width (2/3/4-bit builds of one checkpoint, which differ only under
    /// `packed_experts/`). The width comes from the manifest's
    /// `quant.routedExpert.weightBits` with the same `?? 4` default the runtime
    /// applies (`Model.routedExpertWeightBits`), so the probe and the loader
    /// agree on a manifest that declares nothing.
    public static func matchingDescriptor(
        at directory: URL,
        missingManifestFallback: AppModelInstallDescriptor = .default
    ) -> AppModelInstallDescriptor {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return missingManifestFallback
        }
        do {
            let manifest = try ManifestReader.load(
                directoryURL: directory,
                expecting: try ManifestReader.detectPreset(directoryURL: directory))
            let declaredBits = manifest.quant?.routedExpert.weightBits ?? 4
            let hasVision = manifest.files["vision/vision.safetensors"] != nil
            // Weights first. A base install and its abliterated twin carry the
            // *same* source snapshot hash — it covers the tensor index, which
            // abliteration does not change — so the snapshot scan alone cannot
            // separate them and would hand both directories the entry that
            // happens to come first. The resident-weights digest is what
            // actually differs; the width filter is what keeps a 4-bit entry
            // from claiming a 3-bit directory, since the widths of one
            // checkpoint share even the weights digest. The vision filter is
            // the same story one axis over: the text and vision siblings share
            // both the weights digest and the width.
            if let weights = manifest.files["model_weights.bin"]?.sha256,
               let match = AppModelInstallDescriptor.installable.first(where: {
                   $0.weightsSHA256 == weights
                       && $0.accepts(routedExpertBits: declaredBits)
                       && $0.accepts(visionPayloadPresent: hasVision)
               }) {
                return match
            }
            // The fallback is load-bearing, not decorative: an install whose
            // weights digest is missing or unrecognised (an older repack, a
            // future layout that renames the resident file, a descriptor that
            // declines to pin weights) still resolves exactly as it did before
            // this existed, rather than falling through to Gemma. It too
            // prefers a width-consistent entry, and only when no entry's width
            // matches either does it reproduce the pre-width answer verbatim.
            return AppModelInstallDescriptor.installable.first {
                manifest.sourceSnapshotHash == "sha256:" + $0.sourceIndexSHA256
                    && $0.accepts(routedExpertBits: declaredBits)
                    && $0.accepts(visionPayloadPresent: hasVision)
            } ?? AppModelInstallDescriptor.installable.first {
                manifest.sourceSnapshotHash == "sha256:" + $0.sourceIndexSHA256
                    && $0.accepts(visionPayloadPresent: hasVision)
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

        let manifest: Manifest
        do {
            manifest = try ManifestReader.load(
                directoryURL: directory,
                expecting: try ManifestReader.detectPreset(directoryURL: directory))
        } catch {
            return .partial("\(error)")
        }
        let expectedSource = "sha256:" + descriptor.sourceIndexSHA256
        guard manifest.sourceSnapshotHash == expectedSource else {
            return .partial("installed checkpoint does not match \(descriptor.displayName)")
        }
        // The source hash cannot see the expert width (the widths of one
        // checkpoint share it), so the completion check reads the manifest
        // too: without this, a 3-bit directory completes under the 4-bit
        // descriptor and the install path has no way to notice that the
        // bytes that landed are not the row's build.
        let declaredBits = manifest.quant?.routedExpert.weightBits ?? 4
        guard descriptor.accepts(routedExpertBits: declaredBits) else {
            return .partial("installed checkpoint does not match \(descriptor.displayName)")
        }
        let hasVision = manifest.files["vision/vision.safetensors"] != nil
        guard descriptor.accepts(visionPayloadPresent: hasVision) else {
            return .partial("installed checkpoint does not match \(descriptor.displayName)")
        }
        if descriptor.requiresVision {
            // A vision row is only complete when the payload and both config
            // sidecars are declared. The size-check fallback below would catch
            // them once the receipt is stale, but a freshly installed receipt
            // is valid and would otherwise let a text-only directory pass.
            for required in ["vision/vision.safetensors",
                             "vision/model_config.json",
                             "vision/preprocessor_config.json"]
            where manifest.files[required] == nil {
                return .partial("\(required) is missing")
            }
        }
        let layout = directory.appendingPathComponent("packed_experts/layout.json")
        guard FileManager.default.fileExists(atPath: layout.path) else {
            return .partial("packed_experts/layout.json is missing")
        }
        do {
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            // A directory that was moved or copied — the Documents migration,
            // the Files app, a Hugging Face download unpacked elsewhere —
            // carries a receipt bound to the old physical path. Re-hashing
            // every file here would make the install screen wait minutes, but
            // refusing outright leaves the user with a greyed-out Download
            // and no Load button at all.
            //
            // Size-check the declared files instead:
            //  * every file present at its declared size means the directory
            //    is shaped like an install, so it can be offered to the
            //    loader (this is a presence check, not a trust decision);
            //  * the engine's `.automatic` integrity policy rejects the
            //    unusable receipt and falls back to hashing the bytes before
            //    use, so correctness still rests on the manifest's digests.
            if let problem = firstIncompleteFile(manifest: manifest,
                                                 directory: directory) {
                return .partial(problem)
            }
            return .complete
        }
    }

    /// The first manifest-declared file that is missing or not its declared
    /// size, or nil when the install's file set is complete. Used only after
    /// the receipt path binding fails; it answers "are the bytes here?", not
    /// "are they the right bytes?".
    private static func firstIncompleteFile(manifest: Manifest,
                                            directory: URL) -> String? {
        let fileManager = FileManager.default
        for (relativePath, entry) in manifest.files.sorted(by: { $0.key < $1.key }) {
            let path = directory.appendingPathComponent(relativePath).path
            guard let attributes = try? fileManager.attributesOfItem(atPath: path),
                  attributes[.type] as? FileAttributeType == .typeRegular,
                  let size = (attributes[.size] as? NSNumber)?.uint64Value else {
                return "\(relativePath) is missing"
            }
            guard size == entry.size else {
                return "\(relativePath) is \(size) bytes; the manifest declares \(entry.size)"
            }
        }
        return nil
    }
}

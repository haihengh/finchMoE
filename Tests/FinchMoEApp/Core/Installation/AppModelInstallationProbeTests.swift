import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-missing-\(UUID().uuidString).finch")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-partial-\(UUID().uuidString).finch")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("{}".utf8).write(to: url.appendingPathComponent("manifest.json"))
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func validBoundedMetadataIsComplete() throws {
        let url = try makeCompleteModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    @Test func receiptBoundToDifferentPathIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.finch"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        guard case .partial = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
    }

    @Test func differentCheckpointIsPartial() throws {
        let url = try makeCompleteModelInstall("wrong-checkpoint")
        defer { try? FileManager.default.removeItem(at: url) }
        let descriptor = AppModelInstallDescriptor(
            displayName: "different",
            repoID: "example/different",
            revision: "revision",
            sourceIndexSHA256: String(repeating: "f", count: 64),
            approximateDownloadBytes: 1,
            installedBytes: 1,
            rangeStagingBytes: 1,
            reserveBytes: 1,
            architecture: .gemma4_26B_A4B)
        guard case .partial = AppModelInstallationProbe.status(at: url, descriptor: descriptor) else {
            Issue.record("expected checkpoint mismatch to be partial")
            return
        }
    }

    @Test func qwenInstallIsCompleteUnderQwenDescriptor() throws {
        let url = try makeCompleteModelInstall(
            "qwen",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(AppModelInstallationProbe.status(at: url, descriptor: .qwen3_6) == .complete)
    }

    @Test func qwenInstallIsPartialUnderDefaultDescriptor() throws {
        let url = try makeCompleteModelInstall(
            "qwen-default-probe",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: url) }
        guard case .partial(let message) = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected Qwen install probed as Gemma to be partial")
            return
        }
        // The probe names the expected (Gemma) checkpoint in its mismatch text.
        #expect(message.contains("does not match Gemma 4 26B-A4B IT 4-bit"))
    }

    @Test func matchingDescriptorReadsManifestHash() throws {
        let qwen = try makeCompleteModelInstall(
            "match-qwen",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: qwen) }
        #expect(AppModelInstallationProbe.matchingDescriptor(at: qwen) == .qwen3_6)

        let gemma = try makeCompleteModelInstall("match-gemma")
        defer { try? FileManager.default.removeItem(at: gemma) }
        #expect(AppModelInstallationProbe.matchingDescriptor(at: gemma) == .default)
    }

    /// 3.8 is a second family in the scan, not a special case: the descriptor
    /// has to be *found* by its hash and it must not shadow 3.6. A scan that
    /// returned the first entry regardless of hash would pass a single-family
    /// test and mislabel every directory once there are two.
    @Test func matchingDescriptorFindsTheSecondFamilyToo() throws {
        let qwen38 = try makeCompleteModelInstall(
            "match-qwen38",
            arch: ArchConfig.qwen3_8_flashNext_125B,
            modelID: "local/Qwen3.8-Flash-Next-125B",
            descriptor: .qwen3_8)
        defer { try? FileManager.default.removeItem(at: qwen38) }
        #expect(AppModelInstallationProbe.matchingDescriptor(at: qwen38) == .qwen3_8)

        let qwen36 = try makeCompleteModelInstall(
            "match-qwen36-both",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: qwen36) }
        #expect(AppModelInstallationProbe.matchingDescriptor(at: qwen36) == .qwen3_6)
    }

    /// Every descriptor the app can recognise must be reachable from the
    /// scan — an entry that exists but is not in `installable` is dead code
    /// that reads as "unknown checkpoint" at runtime.
    ///
    /// Uniqueness is asserted on `weightsSHA256`, **not** `sourceIndexSHA256`.
    /// That used to be the right field and no longer is: a base install and its
    /// abliterated twin genuinely share a source snapshot hash, because that
    /// hash covers the tensor index and abliteration does not change any name,
    /// shape or offset. Two entries colliding there is the expected state, not
    /// the failure it would be for the weights digest — which is exactly why
    /// `matchingDescriptor` consults the weights digest first. The invariant the
    /// original assertion protected ("the scan can tell them apart") is
    /// preserved here, on the field that can actually carry it.
    @Test func everyDescriptorResolvesFromItsOwnHash() {
        for descriptor in AppModelInstallDescriptor.installable {
            #expect(!descriptor.sourceIndexSHA256.isEmpty,
                    "\(descriptor.displayName) has no source hash to match on")
            #expect(descriptor.weightsSHA256?.isEmpty == false,
                    "\(descriptor.displayName) pins no weights digest; the scan cannot tell it apart from an architecture twin")
        }
        #expect(Set(AppModelInstallDescriptor.installable.compactMap(\.weightsSHA256)).count
                    == AppModelInstallDescriptor.installable.count,
                "two descriptors share a weights digest; the scan cannot tell them apart")
        #expect(AppModelInstallDescriptor.installable.contains(.qwen3_6))
        #expect(AppModelInstallDescriptor.installable.contains(.qwen3_6_abliterated))
        #expect(AppModelInstallDescriptor.installable.contains(.qwen3_8))
        #expect(AppModelInstallDescriptor.installable.contains(.qwen3_8_abliterated))
    }

    /// The regression guard for the abliterated entries. The two installs below
    /// are built from descriptors that agree on *everything* the scan used to
    /// look at — same `sourceIndexSHA256`, same architecture, same model id, so
    /// the fixture writes the same `sourceSnapshotHash` — and differ only in the
    /// resident-weights digest. Before `weightsSHA256` existed both directories
    /// resolved to whichever entry came first, which is how a repacked
    /// abliterated install ended up labelled as the base model.
    @Test func baseAndAbliteratedTwinsResolveApart() throws {
        #expect(AppModelInstallDescriptor.qwen3_6.sourceIndexSHA256
                    == AppModelInstallDescriptor.qwen3_6_abliterated.sourceIndexSHA256,
                "premise of this test: the twin descriptors share a source hash")

        let base = try makeCompleteModelInstall(
            "twin-base",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: base) }

        let abliterated = try makeCompleteModelInstall(
            "twin-abliterated",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6_abliterated)
        defer { try? FileManager.default.removeItem(at: abliterated) }

        #expect(AppModelInstallationProbe.matchingDescriptor(at: base) == .qwen3_6)
        #expect(AppModelInstallationProbe.matchingDescriptor(at: abliterated) == .qwen3_6_abliterated)
        #expect(AppModelInstallationProbe.matchingDescriptor(at: base)
                    != AppModelInstallationProbe.matchingDescriptor(at: abliterated))
    }

    /// The weights digest is consulted first, but it must not become a hard
    /// requirement: an install whose digest no descriptor recognises still
    /// resolves by its source snapshot rather than collapsing to Gemma.
    @Test func unrecognisedWeightsDigestFallsBackToSourceSnapshot() throws {
        let install = try makeCompleteModelInstall(
            "unknown-weights",
            arch: ArchConfig.qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6)
        defer { try? FileManager.default.removeItem(at: install) }

        // Rewrite the manifest with a digest no descriptor pins, keeping the
        // source snapshot hash intact.
        let manifestURL = install.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        var files = manifest["files"] as? [String: Any] ?? [:]
        files["model_weights.bin"] = ["size": 0,
                                      "sha256": String(repeating: "a", count: 64)]
        manifest["files"] = files
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL)

        #expect(AppModelInstallationProbe.matchingDescriptor(at: install) == .qwen3_6)
    }

    @Test func matchingDescriptorWithoutManifestFallsBackToDefault() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-match-missing-\(UUID().uuidString).finch")
        #expect(AppModelInstallationProbe.matchingDescriptor(at: url) == .default)
    }
}

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

    /// A receipt records the physical directory it was made for. Moving or
    /// copying an install — the Documents migration, the Files app, unpacking
    /// a Hugging Face download — leaves that path stale. The directory is
    /// still usable: the file set is complete, and the engine falls back to
    /// hashing the bytes because the receipt no longer binds. Refusing it is
    /// what left the install screen with a disabled Download and no Load.
    @Test func receiptBoundToDifferentPathIsAdoptedWhenFilesArePresent() throws {
        let url = try makeCompleteModelInstall("wrong-path")
        defer { try? FileManager.default.removeItem(at: url) }
        let receiptURL = url.appendingPathComponent("verified-install.json")
        var receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as! [String: Any]
        receipt["modelDirectoryPath"] = "/different/model.finch"
        try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys]).write(to: receiptURL)
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    @Test func missingReceiptIsAdoptedWhenFilesArePresent() throws {
        let url = try makeCompleteModelInstall("no-receipt")
        defer { try? FileManager.default.removeItem(at: url) }
        try FileManager.default.removeItem(
            at: url.appendingPathComponent("verified-install.json"))
        #expect(AppModelInstallationProbe.status(at: url) == .complete)
    }

    /// The fallback is a presence check, not a trust decision: a declared
    /// file that is gone (or the wrong length) still reports why.
    @Test func incompleteFileSetIsPartialEvenWithoutAReceipt() throws {
        let url = try makeCompleteModelInstall("short-file")
        defer { try? FileManager.default.removeItem(at: url) }
        try FileManager.default.removeItem(
            at: url.appendingPathComponent("verified-install.json"))
        // The fixture declares layer_00 as zero bytes; writing any content
        // makes the on-disk size disagree with the manifest.
        let layer = url.appendingPathComponent("packed_experts/layer_00.bin")
        try Data(count: 7).write(to: layer)
        guard case .partial(let message) = AppModelInstallationProbe.status(at: url) else {
            Issue.record("expected partial status")
            return
        }
        #expect(message.contains("layer_00.bin"))
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

    // MARK: - Vision capability

    /// The text and vision siblings share resident weights and routed bits, so
    /// only the manifest's `vision/vision.safetensors` entry can tell them
    /// apart. A vision directory must resolve to the vision row and must not
    /// satisfy its text sibling (or vice versa).
    @Test func visionInstallResolvesToTheVisionRow() throws {
        let url = try makeCompleteModelInstall(
            "vision-36",
            arch: .qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6_3bit_vision)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(AppModelInstallationProbe.matchingDescriptor(at: url)
            == .qwen3_6_3bit_vision)
        #expect(AppModelInstallationProbe.status(
            at: url, descriptor: .qwen3_6_3bit_vision) == .complete)
        guard case .partial = AppModelInstallationProbe.status(
            at: url, descriptor: .qwen3_6_3bit) else {
            Issue.record("a vision install satisfied its text sibling")
            return
        }
    }

    @Test func textInstallDoesNotSatisfyTheVisionRow() throws {
        let url = try makeCompleteModelInstall(
            "text-36",
            arch: .qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6_3bit)
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(AppModelInstallationProbe.matchingDescriptor(at: url) == .qwen3_6_3bit)
        guard case .partial = AppModelInstallationProbe.status(
            at: url, descriptor: .qwen3_6_3bit_vision) else {
            Issue.record("a text install satisfied the vision row")
            return
        }
    }

    @Test func visionInstallWithAMissingPayloadIsPartial() throws {
        let url = try makeCompleteModelInstall(
            "vision-missing",
            arch: .qwen3_6_35B_A3B,
            modelID: "local/Qwen3.6-35B-A3B",
            descriptor: .qwen3_6_3bit_vision)
        defer { try? FileManager.default.removeItem(at: url) }
        // Drop the receipt so the probe's size-check fallback runs, then remove
        // the payload it should notice.
        try FileManager.default.removeItem(
            at: url.appendingPathComponent("verified-install.json"))
        try FileManager.default.removeItem(
            at: url.appendingPathComponent("vision/vision.safetensors"))

        guard case .partial(let message) = AppModelInstallationProbe.status(
            at: url, descriptor: .qwen3_6_3bit_vision) else {
            Issue.record("missing vision payload was not reported")
            return
        }
        #expect(message.contains("vision/vision.safetensors"))
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
    /// Uniqueness is asserted on the **(weights digest, expert width, vision
    /// payload) triple**, not on any field alone. `sourceIndexSHA256` cannot
    /// carry it because a base install and its abliterated twin genuinely share
    /// a source snapshot hash (it covers the tensor index, which abliteration
    /// does not change). `weightsSHA256` cannot carry it either: the 2-, 3- and
    /// 4-bit builds of one checkpoint share the resident digest, and a vision
    /// sibling shares both the digest and the width — only the `vision/`
    /// file set differs. What the scan needs is that triple, which is exactly
    /// what `matchingDescriptor` reads.
    @Test func everyDescriptorResolvesFromItsOwnHash() {
        for descriptor in AppModelInstallDescriptor.installable {
            #expect(!descriptor.sourceIndexSHA256.isEmpty,
                    "\(descriptor.displayName) has no source hash to match on")
            #expect(descriptor.weightsSHA256?.isEmpty == false,
                    "\(descriptor.displayName) pins no weights digest; the scan cannot tell it apart from an architecture twin")
            #expect(descriptor.routedExpertBits != nil,
                    "\(descriptor.displayName) pins no expert width; the scan cannot tell it apart from a width twin")
        }
        let identities = AppModelInstallDescriptor.installable.map {
            "\($0.weightsSHA256 ?? "none")"
                + "|\($0.routedExpertBits.map(String.init) ?? "any")"
                + "|\($0.requiresVision)"
        }
        #expect(Set(identities).count == identities.count,
                "two descriptors share a weights digest, width and vision capability; the scan cannot tell them apart")
        for expected: AppModelInstallDescriptor in [
            .qwen3_6, .qwen3_6_3bit, .qwen3_6_3bit_vision,
            .qwen3_6_abliterated, .qwen3_6_abliterated_vision,
            .qwen3_6_abliterated_3bit, .qwen3_6_abliterated_3bit_vision,
            .qwen3_8, .qwen3_8_vision,
            .qwen3_8_3bit, .qwen3_8_3bit_vision,
            .qwen3_8_2bit, .qwen3_8_2bit_vision,
            .qwen3_8_abliterated, .qwen3_8_abliterated_vision,
            .qwen3_8_abliterated_3bit, .qwen3_8_abliterated_3bit_vision,
            .qwen3_8_abliterated_2bit, .qwen3_8_abliterated_2bit_vision,
        ] {
            #expect(AppModelInstallDescriptor.installable.contains(expected),
                    "\(expected.displayName) is not in `installable`; the scan cannot find it")
        }
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

    /// The width axis, the exact analogue of `baseAndAbliteratedTwinsResolveApart`:
    /// three 3.8 installs whose descriptors agree on source hash, weights digest
    /// and architecture — differing only in the declared routed-expert width —
    /// must each resolve to their own entry.
    @Test func widthsThatShareWeightsDigestResolveApart() throws {
        #expect(AppModelInstallDescriptor.qwen3_8.weightsSHA256
                    == AppModelInstallDescriptor.qwen3_8_3bit.weightsSHA256
                && AppModelInstallDescriptor.qwen3_8.weightsSHA256
                    == AppModelInstallDescriptor.qwen3_8_2bit.weightsSHA256,
                "premise of this test: the width twins share a weights digest")

        for (descriptor, tag) in [(AppModelInstallDescriptor.qwen3_8, "width-4"),
                                  (.qwen3_8_3bit, "width-3"),
                                  (.qwen3_8_2bit, "width-2")] {
            let install = try makeCompleteModelInstall(
                tag,
                arch: ArchConfig.qwen3_8_flashNext_125B,
                modelID: "local/Qwen3.8-Flash-Next-125B",
                descriptor: descriptor)
            defer { try? FileManager.default.removeItem(at: install) }
            #expect(AppModelInstallationProbe.matchingDescriptor(at: install) == descriptor,
                    "\(tag) resolved to the wrong width entry")
        }
    }

    /// Guard for the first-match-wins hazard the width filter closes: before it,
    /// `.qwen3_8` — first in the family — would claim every PLE width.
    @Test func fourBitEntryDoesNotClaimAThreeBitDirectory() throws {
        let install = try makeCompleteModelInstall(
            "width-claim-guard",
            arch: ArchConfig.qwen3_8_flashNext_125B,
            modelID: "local/Qwen3.8-Flash-Next-125B",
            descriptor: .qwen3_8,
            routedExpertBits: 3)
        defer { try? FileManager.default.removeItem(at: install) }
        #expect(AppModelInstallationProbe.matchingDescriptor(at: install) == .qwen3_8_3bit)
        #expect(AppModelInstallationProbe.matchingDescriptor(at: install) != .qwen3_8)
    }

    /// `?? 4` is what keeps a manifest that declares no quant block resolving
    /// exactly as it did before widths existed; the runtime applies the same
    /// default (`Model.routedExpertWeightBits`), so probe and loader agree.
    @Test func manifestWithoutQuantKeepsTheFourBitAnswer() throws {
        let install = try makeCompleteModelInstall(
            "no-quant",
            arch: ArchConfig.qwen3_8_flashNext_125B,
            modelID: "local/Qwen3.8-Flash-Next-125B",
            descriptor: .qwen3_8)
        defer { try? FileManager.default.removeItem(at: install) }
        let manifestURL = install.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        manifest.removeValue(forKey: "quant")
        try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
            .write(to: manifestURL)
        #expect(AppModelInstallationProbe.matchingDescriptor(at: install) == .qwen3_8)
    }

    /// The completion check reads the width too: a 3-bit directory must not
    /// report `.complete` under the 4-bit row, or a finished install can never
    /// notice that the bytes that landed are not the row's build.
    @Test func statusRejectsAWidthMismatch() throws {
        let install = try makeCompleteModelInstall(
            "status-width",
            arch: ArchConfig.qwen3_8_flashNext_125B,
            modelID: "local/Qwen3.8-Flash-Next-125B",
            descriptor: .qwen3_8,
            routedExpertBits: 3)
        defer { try? FileManager.default.removeItem(at: install) }
        #expect(AppModelInstallationProbe.status(at: install, descriptor: .qwen3_8_3bit) == .complete)
        guard case .partial = AppModelInstallationProbe.status(at: install, descriptor: .qwen3_8) else {
            Issue.record("a 3-bit install must not complete under the 4-bit descriptor")
            return
        }
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

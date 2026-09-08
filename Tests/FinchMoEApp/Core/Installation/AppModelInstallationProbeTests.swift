import Foundation
import Testing
import FinchMoE
@testable import FinchMoEAppCore

@Suite struct AppModelInstallationProbeTests {
    @Test func missingDirectoryIsMissing() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-missing-\(UUID().uuidString).finchturbo")
        #expect(AppModelInstallationProbe.status(at: url) == .missing)
    }

    @Test func manifestWithoutFinalMetadataIsPartial() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-partial-\(UUID().uuidString).finchturbo")
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
        receipt["modelDirectoryPath"] = "/different/model.finchturbo"
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
            reserveBytes: 1)
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

    @Test func matchingDescriptorWithoutManifestFallsBackToDefault() {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("finchmoe-match-missing-\(UUID().uuidString).finchturbo")
        #expect(AppModelInstallationProbe.matchingDescriptor(at: url) == .default)
    }
}

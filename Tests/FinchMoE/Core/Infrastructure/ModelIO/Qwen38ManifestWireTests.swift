import Testing
import Foundation
import FinchMoEFormat
@testable import FinchMoE

/// M0 wire gate for the Qwen3.8-Flash-Next arch keys: a qwen3_8 manifest
/// (every 3.8 `arch` field populated — the 14 structural ones plus
/// `pleEosTokenId`, which M3.3 plumbed end to end) encodes at format minor 0
/// with every key present, round-trips, auto-detects the built-in qwen3_8
/// preset, and
/// passes `ManifestReader` arch validation exactly against that preset —
/// while a nil-key manifest stays byte-identical to the pre-3.8 writer
/// (omission is asserted at the JSON level).
@Suite struct Qwen38ManifestWireTests {

    private static let zeroSHA = String(repeating: "0", count: 64)

    /// Wire arch carrying every Qwen3.8-Flash-Next field, mirroring
    /// `ArchConfig.qwen3_8_flashNext_125B` values (1:1 with the preset test).
    private static func qwen38Arch() -> FinchManifestArchV1 {
        FinchManifestArchV1(
            hiddenSize: 2560, ffnIntermediate: 640, moeIntermediateSize: 640,
            numHeads: 24, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 128, fullHeadDim: 256, vocabSize: 248320,
            slidingWindow: 0, finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0, fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 48, numExperts: 512, topKExperts: 10,
            tieWordEmbeddings: false, attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: (0..<48).map { $0 % 4 == 3 ? 1 : 0 },
            modelFamily: "qwen3_8", attnOutputGate: true,
            linearNumKeyHeads: 16, linearNumValueHeads: 48,
            linearKeyHeadDim: 128, linearValueHeadDim: 128,
            linearConvKernelDim: 4,
            hyperConnectionCount: 4, hyperConnectionLowrank: 320,
            indexerNumHeads: 4, indexerKVHeads: 1, indexerHeadDim: 128,
            indexerBudget: 2048, indexerCompressRatio: 4,
            ngramSize: 3, headsPerNgram: 8, ngramRowDim: 160,
            ngramPartCount: 128, ngramPartRows: 2_500_012,
            pleLayerIndexes: [1], pleConvKernelSize: 4, pleEosTokenId: 248044)
    }

    private static func manifest() -> FinchManifestV1 {
        FinchManifestV1(
            flags: ["streamingPresent": true],
            modelID: "Qwen/Qwen3.8-Flash-Next-test",
            sourceSnapshotHash: "snapshot",
            arch: qwen38Arch(),
            quant: nil,
            files: [
                "model_weights.bin": FinchManifestFileV1(
                    size: 16_384, sha256: zeroSHA),
                "packed_experts/layout.json": FinchManifestFileV1(
                    size: 1, sha256: zeroSHA),
            ],
            expertsPerLayer: 512,
            numLayers: 48,
            expertStride: FinchFormatV1.alignmentBytes,
            bitWidthOverridesHonored: nil)
    }

    @Test func qwen38WireRoundTripsWithAllNewKeysPresent() throws {
        let m = Self.manifest()
        let encoded = try FinchManifestCodec.encode(m)
        // Minor stays 0: the new keys are additive (see FinchFormatV1).
        #expect(try FinchManifestCodec.decodeUnchecked(encoded) == m)
        #expect(try FinchManifestCodec.decode(encoded) == m)

        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let arch = try #require(root["arch"] as? [String: Any])
        for key in ["hyperConnectionCount", "hyperConnectionLowrank",
                    "indexerNumHeads", "indexerKVHeads", "indexerHeadDim",
                    "indexerBudget", "indexerCompressRatio",
                    "ngramSize", "headsPerNgram", "ngramRowDim",
                    "ngramPartCount", "ngramPartRows",
                    "pleLayerIndexes", "pleConvKernelSize",
                    "pleEosTokenId"] {
            #expect(arch[key] != nil, "missing wire key \(key)")
        }
        #expect(arch["modelFamily"] as? String == "qwen3_8")
        #expect(arch["pleLayerIndexes"] as? [Int] == [1])
    }

    @Test func nilQwen38KeysAreOmittedFromWire() throws {
        // Same manifest but with every 3.8 key nil (a Gemma / 3.6 writer
        // path): the encoded JSON must not contain the keys at all, keeping
        // pre-3.8 manifests byte-identical.
        let base = Self.manifest()
        let arch = FinchManifestArchV1(
            hiddenSize: 2560, ffnIntermediate: 640, moeIntermediateSize: 640,
            numHeads: 24, numKVHeads: 2, numFullKVHeads: 2,
            headDim: 128, fullHeadDim: 256, vocabSize: 248320,
            slidingWindow: 0, finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0, fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: 48, numExperts: 512, topKExperts: 10,
            tieWordEmbeddings: false, attentionKEqV: false,
            hiddenActivation: "silu",
            fullAttentionLayerMask: (0..<48).map { $0 % 4 == 3 ? 1 : 0 },
            modelFamily: "qwen3_6", attnOutputGate: true,
            linearNumKeyHeads: 16, linearNumValueHeads: 32,
            linearKeyHeadDim: 128, linearValueHeadDim: 128,
            linearConvKernelDim: 4)
        let m = FinchManifestV1(
            flags: base.flags, modelID: base.modelID,
            sourceSnapshotHash: base.sourceSnapshotHash, arch: arch,
            quant: nil, files: base.files,
            expertsPerLayer: base.expertsPerLayer,
            numLayers: base.numLayers, expertStride: base.expertStride,
            bitWidthOverridesHonored: nil)
        let encoded = try FinchManifestCodec.encode(m)
        let root = try #require(
            JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        let wireArch = try #require(root["arch"] as? [String: Any])
        for key in ["hyperConnectionCount", "hyperConnectionLowrank",
                    "indexerNumHeads", "ngramSize", "pleLayerIndexes",
                    "pleConvKernelSize", "pleEosTokenId"] {
            #expect(wireArch[key] == nil, "nil key \(key) leaked into the wire")
        }
    }

    @Test func qwen38ManifestDecodesAgainstPresetAndRejectsOthers() throws {
        let encoded = try FinchManifestCodec.encode(Self.manifest())
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("qwen38-wire-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try encoded.write(to: dir.appendingPathComponent("manifest.json"))

        // Auto-detection picks the built-in qwen3_8 preset from the family.
        let detected = try ManifestReader.detectPreset(directoryURL: dir)
        #expect(detected == .qwen3_8_flashNext_125B)

        // Full decode + arch cross-check passes only against that preset.
        let m = try ManifestReader.decode(
            data: encoded, expecting: .qwen3_8_flashNext_125B)
        #expect(m.arch.modelFamily == "qwen3_8")
        #expect(m.arch.ngramRowDim == 160)

        #expect(throws: ModelError.self) {
            _ = try ManifestReader.decode(
                data: encoded, expecting: .gemma4_26B_A4B)
        }
        #expect(throws: ModelError.self) {
            _ = try ManifestReader.decode(
                data: encoded, expecting: .qwen3_6_35B_A3B)
        }
    }
}

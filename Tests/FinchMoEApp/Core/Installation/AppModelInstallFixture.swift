import Darwin
import Foundation
import FinchMoE
@testable import FinchMoEAppCore

func makeCompleteModelInstall(_ tag: String,
                              arch: ArchConfig = .gemma4_26B_A4B,
                              modelID: String = "test/gemma-4-26b-a4b",
                              descriptor: AppModelInstallDescriptor = .default) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("finchmoe-complete-\(tag)-\(UUID().uuidString).fqturbo")
    let experts = directory.appendingPathComponent("packed_experts", isDirectory: true)
    try FileManager.default.createDirectory(at: experts, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: experts.appendingPathComponent("layout.json"))

    var files: [String: Any] = [
        "model_weights.bin": ["size": 0, "sha256": String(repeating: "0", count: 64)],
        "packed_experts/layout.json": ["size": 2, "sha256": String(repeating: "0", count: 64)],
    ]
    for layer in 0..<arch.numLayers {
        files[String(format: "packed_experts/layer_%02d.bin", layer)] = [
            "size": 0,
            "sha256": String(repeating: "0", count: 64),
        ]
    }
    var archFields: [String: Any] = [
        "modelFamily": arch.modelFamily,
        "hiddenSize": arch.hiddenSize,
        "ffnIntermediate": arch.intermediateSize,
        "moeIntermediateSize": arch.moeIntermediateSize,
        "numHeads": arch.numHeads,
        "numKVHeads": arch.numKVHeads,
        "numFullKVHeads": arch.numFullKVHeads,
        "headDim": arch.headDim,
        "fullHeadDim": arch.fullHeadDim,
        "vocabSize": arch.vocabSize,
        "slidingWindow": arch.slidingWindow,
        "finalLogitSoftcap": arch.finalLogitSoftcap,
        "ropeTheta": arch.ropeTheta,
        "fullRopeTheta": arch.fullRopeTheta,
        "partialRotaryFactor": arch.partialRotaryFactor,
        "numLayers": arch.numLayers,
        "numExperts": arch.numExperts,
        "topKExperts": arch.topKExperts,
        "tieWordEmbeddings": arch.tieWordEmbeddings,
        "attentionKEqV": arch.attentionKEqV,
        "hiddenActivation": arch.hiddenActivation,
        "fullAttentionLayerMask": arch.fullAttentionLayerMask.map(Int.init),
    ]
    // GDN/linear-attention fields are required by the manifest validator only
    // for the Qwen3.6 family (ManifestReader.validateArch).
    if arch.modelFamily == "qwen3_6" {
        archFields["attnOutputGate"] = arch.attnOutputGate
        archFields["linearNumKeyHeads"] = arch.linearNumKeyHeads
        archFields["linearNumValueHeads"] = arch.linearNumValueHeads
        archFields["linearKeyHeadDim"] = arch.linearKeyHeadDim
        archFields["linearValueHeadDim"] = arch.linearValueHeadDim
        archFields["linearConvKernelDim"] = arch.linearConvKernelDim
    }
    let manifest: [String: Any] = [
        "magic": "FQTURBO",
        "versionMajor": 1,
        "versionMinor": 0,
        "flags": ["streamingPresent": true],
        "modelID": modelID,
        "sourceSnapshotHash": "sha256:" + descriptor.sourceIndexSHA256,
        "quant": [
            "embedding": quantSlot(4),
            "attention": quantSlot(4),
            "linearAttention": quantSlot(8),
            "router": quantSlot(8),
            "sharedExpert": quantSlot(4),
            "routedExpert": quantSlot(4),
        ],
        "arch": archFields,
        "files": files,
        "expertsPerLayer": arch.numExperts,
        "numLayers": arch.numLayers,
        "expertStride": UInt64(getpagesize()),
    ]
    let manifestData = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
    let manifestURL = directory.appendingPathComponent("manifest.json")
    try manifestData.write(to: manifestURL)
    let manifestHash = Sha256Verifier.hashData(manifestData)
    let receipt: [String: Any] = [
        "schemaVersion": 1,
        "manifestSha256": manifestHash,
        "modelDirectoryPath": directory.standardizedFileURL.path,
        "verificationTimestamp": "2026-07-11T00:00:00Z",
        "toolVersion": "FinchMoEAppCoreTests",
        "files": [:],
    ]
    let receiptData = try JSONSerialization.data(withJSONObject: receipt, options: [.sortedKeys])
    try receiptData.write(to: directory.appendingPathComponent("verified-install.json"))
    return directory
}

private func quantSlot(_ weightBits: Int) -> [String: Any] {
    [
        "weightBits": weightBits,
        "scheme": "affine",
        "scaleType": "bf16",
        "biasType": "bf16",
        "groupSize": Quantization.groupSize,
    ]
}

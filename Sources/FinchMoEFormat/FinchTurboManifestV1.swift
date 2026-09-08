import Foundation

package struct FinchTurboManifestFileV1: Codable, Equatable, Sendable {
    package let size: UInt64
    package let sha256: String

    package init(size: UInt64, sha256: String) {
        self.size = size
        self.sha256 = sha256
    }
}

package struct FinchTurboManifestArchV1: Codable, Equatable, Sendable {
    package let hiddenSize: Int
    package let ffnIntermediate: Int
    package let moeIntermediateSize: Int
    package let numHeads: Int
    package let numKVHeads: Int
    package let numFullKVHeads: Int
    package let headDim: Int
    package let fullHeadDim: Int
    package let vocabSize: Int
    package let slidingWindow: Int
    package let finalLogitSoftcap: Double
    package let ropeTheta: Double
    package let fullRopeTheta: Double
    package let partialRotaryFactor: Double
    package let numLayers: Int
    package let numExperts: Int
    package let topKExperts: Int
    package let tieWordEmbeddings: Bool
    package let attentionKEqV: Bool
    package let hiddenActivation: String
    package let fullAttentionLayerMask: [Int]
    // Qwen3.6 / Gated DeltaNet fields. Optional so pre-Qwen (Gemma) manifests
    // still decode; the runtime coalesces nil to the "no GDN" default.
    package let modelFamily: String?
    package let attnOutputGate: Bool?
    package let linearNumKeyHeads: Int?
    package let linearNumValueHeads: Int?
    package let linearKeyHeadDim: Int?
    package let linearValueHeadDim: Int?
    package let linearConvKernelDim: Int?

    package init(hiddenSize: Int, ffnIntermediate: Int, moeIntermediateSize: Int,
                 numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
                 headDim: Int, fullHeadDim: Int, vocabSize: Int,
                 slidingWindow: Int, finalLogitSoftcap: Double,
                 ropeTheta: Double, fullRopeTheta: Double,
                 partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
                 topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
                 hiddenActivation: String, fullAttentionLayerMask: [Int],
                 modelFamily: String? = nil, attnOutputGate: Bool? = nil,
                 linearNumKeyHeads: Int? = nil, linearNumValueHeads: Int? = nil,
                 linearKeyHeadDim: Int? = nil, linearValueHeadDim: Int? = nil,
                 linearConvKernelDim: Int? = nil) {
        self.hiddenSize = hiddenSize
        self.ffnIntermediate = ffnIntermediate
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.hiddenActivation = hiddenActivation
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.modelFamily = modelFamily
        self.attnOutputGate = attnOutputGate
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
    }
}

// Encode only non-nil Gated DeltaNet fields so that pre-Qwen (Gemma)
// manifests remain byte-identical to the original V1 wire format; Qwen fields
// appear only when present. Decoding stays synthesized (missing key -> nil).
extension FinchTurboManifestArchV1 {
    package func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(ffnIntermediate, forKey: .ffnIntermediate)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numHeads, forKey: .numHeads)
        try c.encode(numKVHeads, forKey: .numKVHeads)
        try c.encode(numFullKVHeads, forKey: .numFullKVHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(fullHeadDim, forKey: .fullHeadDim)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(finalLogitSoftcap, forKey: .finalLogitSoftcap)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(fullRopeTheta, forKey: .fullRopeTheta)
        try c.encode(partialRotaryFactor, forKey: .partialRotaryFactor)
        try c.encode(numLayers, forKey: .numLayers)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(topKExperts, forKey: .topKExperts)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(attentionKEqV, forKey: .attentionKEqV)
        try c.encode(hiddenActivation, forKey: .hiddenActivation)
        try c.encode(fullAttentionLayerMask, forKey: .fullAttentionLayerMask)
        if let v = modelFamily { try c.encode(v, forKey: .modelFamily) }
        if let v = attnOutputGate { try c.encode(v, forKey: .attnOutputGate) }
        if let v = linearNumKeyHeads { try c.encode(v, forKey: .linearNumKeyHeads) }
        if let v = linearNumValueHeads { try c.encode(v, forKey: .linearNumValueHeads) }
        if let v = linearKeyHeadDim { try c.encode(v, forKey: .linearKeyHeadDim) }
        if let v = linearValueHeadDim { try c.encode(v, forKey: .linearValueHeadDim) }
        if let v = linearConvKernelDim { try c.encode(v, forKey: .linearConvKernelDim) }
    }
}

package struct FinchTurboManifestQuantSlotV1: Codable, Equatable, Sendable {
    package let weightBits: Int
    package let scheme: String
    package let scaleType: String
    package let biasType: String
    package let groupSize: Int

    package init(weightBits: Int, scheme: String, scaleType: String,
                 biasType: String, groupSize: Int) {
        self.weightBits = weightBits
        self.scheme = scheme
        self.scaleType = scaleType
        self.biasType = biasType
        self.groupSize = groupSize
    }
}

package struct FinchTurboManifestQuantV1: Codable, Equatable, Sendable {
    package let embedding: FinchTurboManifestQuantSlotV1
    package let attention: FinchTurboManifestQuantSlotV1
    /// Qwen GDN (`linear_attn.in_proj_qkv/z/a/b`, `out_proj`) — 8-bit on the
    /// production build: int4 noise on these amplifies ~16x through the
    /// recurrent state (see docs/QWEN36_PORT.md), so the slot is distinct
    /// from `attention` (full-attention q/k/v/o stay at 4).
    package let linearAttention: FinchTurboManifestQuantSlotV1
    package let router: FinchTurboManifestQuantSlotV1
    package let sharedExpert: FinchTurboManifestQuantSlotV1
    package let routedExpert: FinchTurboManifestQuantSlotV1

    package init(embedding: FinchTurboManifestQuantSlotV1,
                 attention: FinchTurboManifestQuantSlotV1,
                 linearAttention: FinchTurboManifestQuantSlotV1,
                 router: FinchTurboManifestQuantSlotV1,
                 sharedExpert: FinchTurboManifestQuantSlotV1,
                 routedExpert: FinchTurboManifestQuantSlotV1) {
        self.embedding = embedding
        self.attention = attention
        self.linearAttention = linearAttention
        self.router = router
        self.sharedExpert = sharedExpert
        self.routedExpert = routedExpert
    }
}

package struct FinchTurboManifestV1: Codable, Equatable, Sendable {
    package let magic: String
    package let versionMajor: Int
    package let versionMinor: Int
    package let flags: [String: Bool]
    package let modelID: String
    package let sourceSnapshotHash: String?
    package let arch: FinchTurboManifestArchV1
    package let quant: FinchTurboManifestQuantV1?
    package let files: [String: FinchTurboManifestFileV1]
    package let expertsPerLayer: Int
    package let numLayers: Int
    package let expertStride: UInt64
    package let bitWidthOverridesHonored: Int?

    package init(magic: String = FinchTurboFormatV1.magic,
                 versionMajor: Int = FinchTurboFormatV1.versionMajor,
                 versionMinor: Int = FinchTurboFormatV1.versionMinor,
                 flags: [String: Bool], modelID: String,
                 sourceSnapshotHash: String?, arch: FinchTurboManifestArchV1,
                 quant: FinchTurboManifestQuantV1?,
                 files: [String: FinchTurboManifestFileV1],
                 expertsPerLayer: Int, numLayers: Int, expertStride: UInt64,
                 bitWidthOverridesHonored: Int?) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.flags = flags
        self.modelID = modelID
        self.sourceSnapshotHash = sourceSnapshotHash
        self.arch = arch
        self.quant = quant
        self.files = files
        self.expertsPerLayer = expertsPerLayer
        self.numLayers = numLayers
        self.expertStride = expertStride
        self.bitWidthOverridesHonored = bitWidthOverridesHonored
    }
}

package enum FinchTurboManifestCodec {
    package static func decode(_ data: Data) throws -> FinchTurboManifestV1 {
        let manifest = try decodeUnchecked(data)
        try validate(manifest)
        return manifest
    }

    package static func decodeUnchecked(_ data: Data) throws -> FinchTurboManifestV1 {
        let manifest: FinchTurboManifestV1
        do { manifest = try JSONDecoder().decode(FinchTurboManifestV1.self, from: data) }
        catch { throw FinchTurboFormatError.invalid(field: "manifest.json", reason: "\(error)") }
        return manifest
    }

    package static func encode(_ manifest: FinchTurboManifestV1) throws -> Data {
        try validate(manifest)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(manifest))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw FinchTurboFormatError.invalid(field: "manifest.json", reason: "\(error)")
        }
    }

    package static func validate(_ manifest: FinchTurboManifestV1) throws {
        guard manifest.magic == FinchTurboFormatV1.magic else {
            throw FinchTurboFormatError.invalid(field: "manifest.magic", reason: "expected FINCHTURBO")
        }
        guard manifest.versionMajor == FinchTurboFormatV1.versionMajor,
              manifest.versionMinor >= 0 else {
            throw FinchTurboFormatError.invalid(field: "manifest.version", reason: "unsupported version")
        }
        for flag in manifest.flags.keys where !FinchTurboFormatV1.knownFlags.contains(flag) {
            throw FinchTurboFormatError.invalid(field: "manifest.flags.\(flag)", reason: "unknown v1 flag")
        }
        guard !manifest.modelID.isEmpty,
              manifest.numLayers > 0, manifest.expertsPerLayer > 0,
              manifest.expertStride > 0,
              manifest.expertStride % FinchTurboFormatV1.alignmentBytes == 0 else {
            throw FinchTurboFormatError.invalid(field: "manifest", reason: "invalid dimensions or stride")
        }
        guard manifest.arch.numLayers == manifest.numLayers,
              manifest.arch.numExperts == manifest.expertsPerLayer else {
            throw FinchTurboFormatError.invalid(
                field: "manifest.arch", reason: "dimensions disagree with streaming metadata")
        }
        let arch = manifest.arch
        guard arch.hiddenSize > 0, arch.ffnIntermediate > 0,
              arch.moeIntermediateSize > 0, arch.numHeads > 0,
              arch.numKVHeads > 0, arch.numFullKVHeads > 0,
              arch.headDim > 0, arch.fullHeadDim > 0,
              arch.vocabSize > 0, arch.slidingWindow >= 0,
              arch.topKExperts > 0, arch.topKExperts <= arch.numExperts,
              arch.finalLogitSoftcap.isFinite,
              arch.ropeTheta.isFinite, arch.ropeTheta > 0,
              arch.fullRopeTheta.isFinite, arch.fullRopeTheta > 0,
              arch.partialRotaryFactor.isFinite,
              arch.partialRotaryFactor >= 0, arch.partialRotaryFactor <= 1,
              !arch.hiddenActivation.isEmpty,
              arch.fullAttentionLayerMask.count == arch.numLayers,
              arch.fullAttentionLayerMask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
            throw FinchTurboFormatError.invalid(
                field: "manifest.arch", reason: "invalid architecture values")
        }
        if let quant = manifest.quant {
            for (name, slot) in [
                ("embedding", quant.embedding),
                ("attention", quant.attention),
                ("router", quant.router),
                ("sharedExpert", quant.sharedExpert),
                ("routedExpert", quant.routedExpert),
            ] {
                guard slot.weightBits > 0, slot.weightBits <= 32,
                      slot.groupSize > 0,
                      !slot.scheme.isEmpty, !slot.scaleType.isEmpty,
                      !slot.biasType.isEmpty else {
                    throw FinchTurboFormatError.invalid(
                        field: "manifest.quant.\(name)", reason: "invalid quantization values")
                }
            }
        }
        let reservedFiles: Set<String> = ["manifest.json", "verified-install.json"]
        let filePaths = manifest.files.keys.sorted()
        var canonicalPaths: [String: String] = [:]
        for path in filePaths {
            try FinchTurboPathValidator.validateRelativePath(path, field: "manifest.files.\(path)")
            let key = FinchTurboPathValidator.appleFilesystemKey(path)
            guard canonicalPaths.updateValue(path, forKey: key) == nil else {
                throw FinchTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "filesystem-equivalent duplicate path")
            }
            guard key != "tokenizer",
                  !reservedFiles.contains(key),
                  !reservedFiles.contains(where: { key.hasPrefix("\($0)/") }) else {
                throw FinchTurboFormatError.invalid(
                    field: "manifest.files.\(path)", reason: "reserved artifact filename")
            }
            let entry = manifest.files[path]!
            guard entry.sha256.count == 64,
                  entry.sha256.unicodeScalars.allSatisfy({ scalar in
                      ("0"..."9").contains(Character(String(scalar)))
                          || ("a"..."f").contains(Character(String(scalar)))
                          || ("A"..."F").contains(Character(String(scalar)))
                  }) else {
                throw FinchTurboFormatError.invalid(
                    field: "manifest.files.\(path).sha256", reason: "expected 64 hexadecimal characters")
            }
        }
        for (key, path) in canonicalPaths {
            var components = key.split(separator: "/").map(String.init)
            while components.count > 1 {
                _ = components.removeLast()
                let ancestor = components.joined(separator: "/")
                if canonicalPaths[ancestor] != nil {
                    throw FinchTurboFormatError.invalid(
                        field: "manifest.files.\(path)",
                        reason: "file path collides with a directory prefix")
                }
            }
        }
    }

}

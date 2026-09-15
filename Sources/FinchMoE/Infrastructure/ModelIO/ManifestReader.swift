import Foundation
import FinchMoEFormat

public struct ManifestFileEntry: Decodable, Equatable, Sendable {
    public let size: UInt64
    public let sha256: String
}

public struct ManifestArch: Decodable, Equatable, Sendable {
    public let hiddenSize: Int
    public let ffnIntermediate: Int
    public let moeIntermediateSize: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let hiddenActivation: String
    public let fullAttentionLayerMask: [Int]
    // Qwen3.6 / Gated DeltaNet fields (absent in Gemma manifests).
    public let modelFamily: String?
    public let attnOutputGate: Bool?
    public let linearNumKeyHeads: Int?
    public let linearNumValueHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearValueHeadDim: Int?
    public let linearConvKernelDim: Int?
    // Qwen3.8-Flash-Next fields (additive at minor 0; absent in Gemma / 3.6).
    public let hyperConnectionCount: Int?
    public let hyperConnectionLowrank: Int?
    public let indexerNumHeads: Int?
    public let indexerKVHeads: Int?
    public let indexerHeadDim: Int?
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?
    public let ngramSize: Int?
    public let headsPerNgram: Int?
    public let ngramRowDim: Int?
    public let ngramPartCount: Int?
    public let ngramPartRows: Int?
    public let pleLayerIndexes: [Int]?
    public let pleConvKernelSize: Int?
    public let pleEosTokenId: Int?
}

public struct ManifestQuantSlot: Decodable, Equatable, Sendable {
    public let weightBits: Int
    public let scheme: String
    public let scaleType: String
    public let biasType: String
    public let groupSize: Int
}

public struct ManifestQuant: Decodable, Equatable, Sendable {
    public let embedding: ManifestQuantSlot
    public let attention: ManifestQuantSlot
    public let linearAttention: ManifestQuantSlot
    public let router: ManifestQuantSlot
    public let sharedExpert: ManifestQuantSlot
    public let routedExpert: ManifestQuantSlot
    /// PLE n-gram table quantization slot. `nil` on raw-BF16 PLE installs (which
    /// keep the legacy byte path) and on non-qwen3_8 families; present with
    /// int4 / group 32 / "affine" when the PLE table was quantized.
    public let pleNgram: ManifestQuantSlot?
}

public struct Manifest: Decodable, Equatable, Sendable {
    public let magic: String
    public let versionMajor: Int
    public let versionMinor: Int
    public let flags: [String: Bool]
    public let modelID: String
    public let sourceSnapshotHash: String?
    public let arch: ManifestArch
    public let quant: ManifestQuant?
    public let files: [String: ManifestFileEntry]
    public let expertsPerLayer: Int
    public let numLayers: Int
    public let expertStride: UInt64
}

public enum ManifestReader {
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    /// Recognized flag keys. Anything else in `manifest.flags` is an error.
    public static let knownFlags: Set<String> = FinchFormatV1.knownFlags

    /// Fixed required entries. Packed-layer filenames come from layout.json and
    /// are cross-validated only after that document is decoded.
    public static let requiredFiles: [String] = [
        "model_weights.bin",
        "packed_experts/layout.json",
    ]

    public static func load(directoryURL: URL,
                            expecting: ArchConfig,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> Manifest {
        let directory = try FinchModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        return try decode(data: data, expecting: expecting)
    }

    /// Peeks the installed manifest's arch and returns the built-in preset it
    /// matches, so loaders pick the Qwen or Gemma preset from the model itself
    /// instead of hardcoding one. Unknown/nil families fall back to Gemma.
    ///
    /// `allowManifestArch` (`FINCHMOE_EXPECT_ARCH=1`, set by the toy CLI smoke)
    /// returns the arch the manifest declares about *itself* instead. Without it
    /// a non-production geometry cannot load through the CLI at all: the preset
    /// is a cross-check that the declared dims match the family's shipped shape,
    /// and against a deliberately 64-dim toy install that comparison can only
    /// fail. Understand the trade before setting it — `validateArch` then
    /// compares the manifest against itself, so it still catches an internally
    /// inconsistent manifest but no longer a wrong-but-consistent one. The
    /// `.fullSha256` content hashes are a separate check and still run.
    ///
    /// Default `false`: production behavior is unchanged.
    public static func detectPreset(directoryURL: URL,
                                    maxBytes: UInt64 = defaultMaxBytes,
                                    allowManifestArch: Bool = false) throws -> ArchConfig {
        let directory = try FinchModelDirectory(rootURL: directoryURL)
        let data: Data
        do {
            data = try directory.readMetadata("manifest.json", maxBytes: maxBytes)
        } catch ModelError.missingFile {
            throw ModelError.partialInstall(path: directoryURL.path)
        }
        let wire = try FinchManifestCodec.decodeUnchecked(data)
        if allowManifestArch {
            return ArchConfig(manifestArch: ManifestArch(wire: wire.arch))
        }
        return ArchConfig.preset(forModelFamily: wire.arch.modelFamily)
    }

    package static func decode(data: Data,
                               expecting: ArchConfig) throws -> Manifest {
        let manifest: Manifest
        do {
            let wire = try FinchManifestCodec.decodeUnchecked(data)
            guard wire.magic == FinchFormatV1.magic else {
                throw ModelError.notAFinchDirectory
            }
            guard wire.versionMajor == FinchFormatV1.versionMajor,
                  wire.versionMinor >= 0 else {
                throw ModelError.unsupportedVersion(major: wire.versionMajor,
                                                    minor: wire.versionMinor)
            }
            for key in wire.flags.keys where !FinchFormatV1.knownFlags.contains(key) {
                throw ModelError.unknownFlag(name: key)
            }
            if wire.expertStride % FinchFormatV1.alignmentBytes != 0 {
                throw ModelError.expertStrideNotPageAligned(
                    stride: wire.expertStride,
                    pageSize: Int(FinchFormatV1.alignmentBytes))
            }
            try FinchManifestCodec.validate(wire)
            manifest = Manifest(wire: wire)
        } catch let error as ModelError {
            throw error
        } catch {
            throw ModelError.indexCorrupt(detail: "manifest.json: \(error)")
        }

        try validate(manifest, against: expecting)
        return manifest
    }

    static func validate(_ m: Manifest,
                         against expected: ArchConfig) throws {
        if m.flags["quantKV"] == true {
            throw ModelError.indexCorrupt(
                detail: "manifest requests removed quantized-KV runtime support")
        }
        try validateArch(m.arch, expected: expected)
        if let quant = m.quant {
            try validateQuant(quant)
        } else if expected.numLayers == ArchConfig.gemma4_26B_A4B.numLayers,
                  expected.hiddenSize == ArchConfig.gemma4_26B_A4B.hiddenSize {
            throw ModelError.indexCorrupt(detail: "manifest.quant is required for the production architecture")
        }
        for f in requiredFiles {
            if m.files[f] == nil { throw ModelError.missingFile(name: f) }
        }
    }

    private static func validateQuant(_ quant: ManifestQuant) throws {
        let slots: [(String, ManifestQuantSlot, Set<Int>)] = [
            ("embedding", quant.embedding, [4]),
            ("attention", quant.attention, [4]),
            ("linearAttention", quant.linearAttention, [4, 8]),
            ("router", quant.router, [8]),
            ("sharedExpert", quant.sharedExpert, [4, 8]),
            ("routedExpert", quant.routedExpert, [4]),
        ]
        for (name, slot, allowedBits) in slots {
            guard allowedBits.contains(slot.weightBits),
                  slot.scheme.lowercased() == "affine",
                  slot.scaleType.lowercased() == "bf16",
                  slot.biasType.lowercased() == "bf16",
                  slot.groupSize == Quantization.groupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for \(name)")
            }
        }
        // PLE n-gram slot: optional (raw-BF16 installs omit it). When present it
        // must be the exact int4 / group-32 / affine layout the writer produces;
        // anything else means a different on-disk stride we cannot decode, so
        // reject the load rather than misread bytes (Phase 4 backward-compat).
        if let ple = quant.pleNgram {
            guard ple.weightBits == 4,
                  ple.scheme.lowercased() == "affine",
                  ple.scaleType.lowercased() == "bf16",
                  ple.biasType.lowercased() == "bf16",
                  ple.groupSize == Quantization.pleGroupSize else {
                throw ModelError.indexCorrupt(detail: "unsupported quantization for pleNgram")
            }
        }
    }

    private static func validateArch(_ a: ManifestArch,
                                     expected e: ArchConfig) throws {
        func check<T: Equatable & CustomStringConvertible>(
            _ field: String, _ actual: T, _ expected: T) throws {
            if actual != expected {
                throw ModelError.archMismatch(field: field,
                                              expected: "\(expected)",
                                              actual: "\(actual)")
            }
        }
        try check("hiddenSize",          a.hiddenSize,          e.hiddenSize)
        try check("ffnIntermediate",     a.ffnIntermediate,     e.intermediateSize)
        try check("moeIntermediateSize", a.moeIntermediateSize, e.moeIntermediateSize)
        try check("numHeads",            a.numHeads,            e.numHeads)
        try check("numKVHeads",          a.numKVHeads,          e.numKVHeads)
        try check("numFullKVHeads",      a.numFullKVHeads,      e.numFullKVHeads)
        try check("headDim",             a.headDim,             e.headDim)
        try check("fullHeadDim",         a.fullHeadDim,         e.fullHeadDim)
        try check("vocabSize",           a.vocabSize,           e.vocabSize)
        try check("slidingWindow",       a.slidingWindow,       e.slidingWindow)
        try check("finalLogitSoftcap",   a.finalLogitSoftcap,   e.finalLogitSoftcap)
        try check("ropeTheta",           a.ropeTheta,           e.ropeTheta)
        try check("fullRopeTheta",       a.fullRopeTheta,       e.fullRopeTheta)
        try check("partialRotaryFactor", a.partialRotaryFactor, e.partialRotaryFactor)
        try check("numLayers",           a.numLayers,           e.numLayers)
        try check("numExperts",          a.numExperts,          e.numExperts)
        try check("topKExperts",         a.topKExperts,         e.topKExperts)
        try check("tieWordEmbeddings",   a.tieWordEmbeddings,   e.tieWordEmbeddings)
        try check("attentionKEqV",       a.attentionKEqV,       e.attentionKEqV)
        try check("hiddenActivation",    a.hiddenActivation,    e.hiddenActivation)
        let actualMask = a.fullAttentionLayerMask.map { UInt8($0) }
        try check("fullAttentionLayerMask",
                  actualMask.description,
                  e.fullAttentionLayerMask.description)
        // Gated DeltaNet / linear-attention fields are only present (and only
        // required) for the Qwen hybrid families (3.6 and 3.8-Flash-Next
        // share the `linear_attn.*` machinery).
        if e.isQwenHybrid {
            try check("modelFamily",       a.modelFamily ?? "",              e.modelFamily)
            try check("attnOutputGate",    a.attnOutputGate ?? false,        e.attnOutputGate)
            try check("linearNumKeyHeads", a.linearNumKeyHeads ?? 0,         e.linearNumKeyHeads)
            try check("linearNumValueHeads", a.linearNumValueHeads ?? 0,     e.linearNumValueHeads)
            try check("linearKeyHeadDim",  a.linearKeyHeadDim ?? 0,          e.linearKeyHeadDim)
            try check("linearValueHeadDim", a.linearValueHeadDim ?? 0,       e.linearValueHeadDim)
            try check("linearConvKernelDim", a.linearConvKernelDim ?? 0,     e.linearConvKernelDim)
        }
        // Qwen3.8-Flash-Next fields (hyper-connection / QSA indexer / PLE
        // n-gram): required when (and only when) the family is qwen3_8.
        if e.modelFamily == ArchConfig.qwen3_8Family {
            try check("hyperConnectionCount",  a.hyperConnectionCount ?? 0,  e.hyperConnectionCount)
            try check("hyperConnectionLowrank", a.hyperConnectionLowrank ?? 0, e.hyperConnectionLowrank)
            try check("indexerNumHeads",       a.indexerNumHeads ?? 0,       e.indexerNumHeads)
            try check("indexerKVHeads",        a.indexerKVHeads ?? 0,        e.indexerKVHeads)
            try check("indexerHeadDim",        a.indexerHeadDim ?? 0,        e.indexerHeadDim)
            try check("indexerBudget",         a.indexerBudget ?? 0,         e.indexerBudget)
            try check("indexerCompressRatio",  a.indexerCompressRatio ?? 0,  e.indexerCompressRatio)
            try check("ngramSize",             a.ngramSize ?? 0,             e.ngramSize)
            try check("headsPerNgram",         a.headsPerNgram ?? 0,         e.headsPerNgram)
            try check("ngramRowDim",           a.ngramRowDim ?? 0,           e.ngramRowDim)
            try check("ngramPartCount",        a.ngramPartCount ?? 0,        e.ngramPartCount)
            try check("ngramPartRows",         a.ngramPartRows ?? 0,         e.ngramPartRows)
            try check("pleLayerIndexes",       a.pleLayerIndexes ?? [],      e.pleLayerIndexes)
            try check("pleConvKernelSize",     a.pleConvKernelSize ?? 0,     e.pleConvKernelSize)
            try check("pleEosTokenId",         a.pleEosTokenId ?? 0,         e.pleEosTokenId)
        }
    }
}

private extension ManifestFileEntry {
    init(wire: FinchManifestFileV1) {
        self.init(size: wire.size, sha256: wire.sha256)
    }
}

/// The arch a manifest declares about itself, read back as an `ArchConfig`.
///
/// Only `detectPreset`'s `allowManifestArch` path uses this. The optional
/// Qwen3.6 / 3.8 fields fall back to the same zero defaults the built-in
/// presets use when a family does not carry them, so a Gemma manifest converts
/// to exactly what `preset(forModelFamily:)` would have produced.
private extension ArchConfig {
    init(manifestArch a: ManifestArch) {
        self.init(
            hiddenSize: a.hiddenSize,
            intermediateSize: a.ffnIntermediate,
            moeIntermediateSize: a.moeIntermediateSize,
            numHeads: a.numHeads,
            numKVHeads: a.numKVHeads,
            numFullKVHeads: a.numFullKVHeads,
            headDim: a.headDim,
            fullHeadDim: a.fullHeadDim,
            vocabSize: a.vocabSize,
            slidingWindow: a.slidingWindow,
            finalLogitSoftcap: a.finalLogitSoftcap,
            ropeTheta: a.ropeTheta,
            fullRopeTheta: a.fullRopeTheta,
            partialRotaryFactor: a.partialRotaryFactor,
            numLayers: a.numLayers,
            numExperts: a.numExperts,
            topKExperts: a.topKExperts,
            tieWordEmbeddings: a.tieWordEmbeddings,
            attentionKEqV: a.attentionKEqV,
            fullAttentionLayerMask: a.fullAttentionLayerMask.map { UInt8($0) },
            hiddenActivation: a.hiddenActivation,
            modelFamily: a.modelFamily ?? "",
            attnOutputGate: a.attnOutputGate ?? false,
            linearNumKeyHeads: a.linearNumKeyHeads ?? 0,
            linearNumValueHeads: a.linearNumValueHeads ?? 0,
            linearKeyHeadDim: a.linearKeyHeadDim ?? 0,
            linearValueHeadDim: a.linearValueHeadDim ?? 0,
            linearConvKernelDim: a.linearConvKernelDim ?? 0,
            hyperConnectionCount: a.hyperConnectionCount ?? 0,
            hyperConnectionLowrank: a.hyperConnectionLowrank ?? 0,
            indexerNumHeads: a.indexerNumHeads ?? 0,
            indexerKVHeads: a.indexerKVHeads ?? 0,
            indexerHeadDim: a.indexerHeadDim ?? 0,
            indexerBudget: a.indexerBudget ?? 0,
            indexerCompressRatio: a.indexerCompressRatio ?? 0,
            ngramSize: a.ngramSize ?? 0,
            headsPerNgram: a.headsPerNgram ?? 0,
            ngramRowDim: a.ngramRowDim ?? 0,
            ngramPartCount: a.ngramPartCount ?? 0,
            ngramPartRows: a.ngramPartRows ?? 0,
            pleLayerIndexes: a.pleLayerIndexes ?? [],
            pleConvKernelSize: a.pleConvKernelSize ?? 0,
            pleEosTokenId: a.pleEosTokenId ?? 0)
    }
}

private extension ManifestArch {
    init(wire: FinchManifestArchV1) {
        self.init(hiddenSize: wire.hiddenSize,
                  ffnIntermediate: wire.ffnIntermediate,
                  moeIntermediateSize: wire.moeIntermediateSize,
                  numHeads: wire.numHeads,
                  numKVHeads: wire.numKVHeads,
                  numFullKVHeads: wire.numFullKVHeads,
                  headDim: wire.headDim,
                  fullHeadDim: wire.fullHeadDim,
                  vocabSize: wire.vocabSize,
                  slidingWindow: wire.slidingWindow,
                  finalLogitSoftcap: wire.finalLogitSoftcap,
                  ropeTheta: wire.ropeTheta,
                  fullRopeTheta: wire.fullRopeTheta,
                  partialRotaryFactor: wire.partialRotaryFactor,
                  numLayers: wire.numLayers,
                  numExperts: wire.numExperts,
                  topKExperts: wire.topKExperts,
                  tieWordEmbeddings: wire.tieWordEmbeddings,
                  attentionKEqV: wire.attentionKEqV,
                  hiddenActivation: wire.hiddenActivation,
                  fullAttentionLayerMask: wire.fullAttentionLayerMask,
                  modelFamily: wire.modelFamily,
                  attnOutputGate: wire.attnOutputGate,
                  linearNumKeyHeads: wire.linearNumKeyHeads,
                  linearNumValueHeads: wire.linearNumValueHeads,
                  linearKeyHeadDim: wire.linearKeyHeadDim,
                  linearValueHeadDim: wire.linearValueHeadDim,
                  linearConvKernelDim: wire.linearConvKernelDim,
                  hyperConnectionCount: wire.hyperConnectionCount,
                  hyperConnectionLowrank: wire.hyperConnectionLowrank,
                  indexerNumHeads: wire.indexerNumHeads,
                  indexerKVHeads: wire.indexerKVHeads,
                  indexerHeadDim: wire.indexerHeadDim,
                  indexerBudget: wire.indexerBudget,
                  indexerCompressRatio: wire.indexerCompressRatio,
                  ngramSize: wire.ngramSize,
                  headsPerNgram: wire.headsPerNgram,
                  ngramRowDim: wire.ngramRowDim,
                  ngramPartCount: wire.ngramPartCount,
                  ngramPartRows: wire.ngramPartRows,
                  pleLayerIndexes: wire.pleLayerIndexes,
                  pleConvKernelSize: wire.pleConvKernelSize,
                  pleEosTokenId: wire.pleEosTokenId)
    }
}

private extension ManifestQuantSlot {
    init(wire: FinchManifestQuantSlotV1) {
        self.init(weightBits: wire.weightBits, scheme: wire.scheme,
                  scaleType: wire.scaleType, biasType: wire.biasType,
                  groupSize: wire.groupSize)
    }
}

private extension ManifestQuant {
    init(wire: FinchManifestQuantV1) {
        self.init(embedding: ManifestQuantSlot(wire: wire.embedding),
                  attention: ManifestQuantSlot(wire: wire.attention),
                  linearAttention: ManifestQuantSlot(wire: wire.linearAttention),
                  router: ManifestQuantSlot(wire: wire.router),
                  sharedExpert: ManifestQuantSlot(wire: wire.sharedExpert),
                  routedExpert: ManifestQuantSlot(wire: wire.routedExpert),
                  pleNgram: wire.pleNgram.map(ManifestQuantSlot.init(wire:)))
    }
}

private extension Manifest {
    init(wire: FinchManifestV1) {
        self.init(magic: wire.magic,
                  versionMajor: wire.versionMajor,
                  versionMinor: wire.versionMinor,
                  flags: wire.flags,
                  modelID: wire.modelID,
                  sourceSnapshotHash: wire.sourceSnapshotHash,
                  arch: ManifestArch(wire: wire.arch),
                  quant: wire.quant.map(ManifestQuant.init(wire:)),
                  files: wire.files.mapValues(ManifestFileEntry.init(wire:)),
                  expertsPerLayer: wire.expertsPerLayer,
                  numLayers: wire.numLayers,
                  expertStride: wire.expertStride)
    }
}

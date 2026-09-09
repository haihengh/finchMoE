import Foundation
import FinchMoEFormat

/// JSON encoders for `manifest.json` and `packed_experts/layout.json`. The
/// files are small (kilobytes), so we use Foundation's `JSONSerialization`
/// rather than streaming.
enum FinchJSON {

    static let magic = FinchFormatV1.magic
    static let versionMajor = FinchFormatV1.versionMajor
    static let versionMinor = FinchFormatV1.versionMinor

    struct FileEntry {
        let size: UInt64
        let sha256: String
    }

    struct QuantBitWidths {
        var embedding: Int
        var attention: Int
        var linearAttention: Int
        var router: Int
        var sharedExpert: Int
        var routedExpert: Int
    }

    static func encodeManifest(plan: RepackPlan,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        try encodeManifest(arch: plan.arch,
                           baseMode: plan.baseMode,
                           baseGroupSize: plan.baseGroupSize,
                           bitsOverrideCount: plan.bitsOverrideCount,
                           modelID: modelID,
                           sourceSnapshotHash: sourceSnapshotHash,
                           files: files,
                           expertsPerLayer: expertsPerLayer,
                           numLayers: numLayers,
                           expertStride: expertStride,
                           bitWidths: bitWidths)
    }

    static func encodeManifest(arch: ArchInfo,
                                      baseMode: String,
                                      baseGroupSize: Int,
                                      bitsOverrideCount: Int,
                                      modelID: String,
                                      sourceSnapshotHash: String,
                                      files: [(relativePath: String, info: FileEntry)],
                                      expertsPerLayer: Int,
                                      numLayers: Int,
                                      expertStride: UInt64,
                                      bitWidths: QuantBitWidths) throws -> Data {
        let bitWidthsByQuantSlot = [
            "embedding": bitWidths.embedding,
            "attention": bitWidths.attention,
            "linearAttention": bitWidths.linearAttention,
            "router": bitWidths.router,
            "sharedExpert": bitWidths.sharedExpert,
            "routedExpert": bitWidths.routedExpert,
        ]
        let wireArch = FinchManifestArchV1(
            hiddenSize: arch.hiddenSize,
            ffnIntermediate: arch.intermediateSize,
            moeIntermediateSize: arch.moeIntermediateSize,
            numHeads: arch.numHeads,
            numKVHeads: arch.numKVHeads,
            numFullKVHeads: arch.numFullKVHeads,
            headDim: arch.headDim,
            fullHeadDim: arch.fullHeadDim,
            vocabSize: arch.vocabSize,
            slidingWindow: arch.slidingWindow,
            finalLogitSoftcap: arch.finalLogitSoftcap,
            ropeTheta: arch.ropeTheta,
            fullRopeTheta: arch.fullRopeTheta,
            partialRotaryFactor: arch.partialRotaryFactor,
            numLayers: arch.numLayers,
            numExperts: arch.numExperts,
            topKExperts: arch.topKExperts,
            tieWordEmbeddings: arch.tieWordEmbeddings,
            attentionKEqV: arch.attentionKEqV,
            hiddenActivation: arch.hiddenActivation,
            fullAttentionLayerMask: arch.fullAttentionLayerMask.map(Int.init),
            modelFamily: arch.modelFamily,
            attnOutputGate: arch.attnOutputGate,
            linearNumKeyHeads: arch.linearNumKeyHeads,
            linearNumValueHeads: arch.linearNumValueHeads,
            linearKeyHeadDim: arch.linearKeyHeadDim,
            linearValueHeadDim: arch.linearValueHeadDim,
            linearConvKernelDim: arch.linearConvKernelDim,
            hyperConnectionCount: arch.hyperConnectionCount,
            hyperConnectionLowrank: arch.hyperConnectionLowrank,
            indexerNumHeads: arch.indexerNumHeads,
            indexerKVHeads: arch.indexerKVHeads,
            indexerHeadDim: arch.indexerHeadDim,
            indexerBudget: arch.indexerBudget,
            indexerCompressRatio: arch.indexerCompressRatio,
            ngramSize: arch.ngramSize,
            headsPerNgram: arch.headsPerNgram,
            ngramRowDim: arch.ngramRowDim,
            ngramPartCount: arch.ngramPartCount,
            ngramPartRows: arch.ngramPartRows,
            pleLayerIndexes: arch.pleLayerIndexes,
            pleConvKernelSize: arch.pleConvKernelSize)
        func slot(_ name: String) throws -> FinchManifestQuantSlotV1 {
            guard let weightBits = bitWidthsByQuantSlot[name] else {
                throw RepackError.configurationInvalid(
                    detail: "missing manifest quant slot bit width for \(name)")
            }
            return FinchManifestQuantSlotV1(
                weightBits: weightBits,
                scheme: baseMode,
                scaleType: "BF16",
                biasType: "BF16",
                groupSize: baseGroupSize)
        }
        let quant = FinchManifestQuantV1(
            embedding: try slot("embedding"),
            attention: try slot("attention"),
            linearAttention: try slot("linearAttention"),
            router: try slot("router"),
            sharedExpert: try slot("sharedExpert"),
            routedExpert: try slot("routedExpert"))
        var wireFiles: [String: FinchManifestFileV1] = [:]
        wireFiles.reserveCapacity(files.count)
        for file in files {
            guard wireFiles.updateValue(
                FinchManifestFileV1(size: file.info.size, sha256: file.info.sha256),
                forKey: file.relativePath) == nil else {
                throw RepackError.configurationInvalid(
                    detail: "duplicate manifest file entry \(file.relativePath)")
            }
        }
        return try FinchManifestCodec.encode(FinchManifestV1(
            flags: [
                "streamingPresent": true,
                "quantKV": false,
                "aneSharedExpert": false,
            ],
            modelID: modelID,
            sourceSnapshotHash: sourceSnapshotHash,
            arch: wireArch,
            quant: quant,
            files: wireFiles,
            expertsPerLayer: expertsPerLayer,
            numLayers: numLayers,
            expertStride: expertStride,
            bitWidthOverridesHonored: bitsOverrideCount))
    }

    static func encodeLayout(plan: RepackPlan,
                                    expertStride: UInt64) throws -> Data {
        try encodeLayout(layers: plan.layers,
                         numLayers: plan.arch.numLayers,
                         expertStride: expertStride)
    }

    static func encodeLayout(layers: [LayerFilePlan],
                                    numLayers: Int,
                                    expertStride: UInt64) throws -> Data {
        var wireLayers: [FinchLayerV1] = []
        wireLayers.reserveCapacity(layers.count)
        for lp in layers {
            let layerFile = (lp.path as NSString).lastPathComponent
            var experts: [FinchExpertV1] = []
            experts.reserveCapacity(lp.expertsPerLayer)
            for e in 0..<lp.expertsPerLayer {
                let physicalRank = lp.physicalRank(for: e)
                let base = UInt64(physicalRank) * lp.expertStride
                var tensors: [String: FinchSubTensorV1] = [:]
                for slice in lp.subTensors {
                    let key: String
                    switch slice.component {
                    case "weights": key = slice.role
                    case "scales":  key = slice.role + "_scales"
                    case "biases":  key = slice.role + "_biases"
                    default:        key = slice.role + "_" + slice.component
                    }
                    guard slice.dtype == FinchFormatV1.DType.u32.rawValue
                            || slice.dtype == FinchFormatV1.DType.bf16.rawValue else {
                        throw RepackError.configurationInvalid(
                            detail: "unsupported packed expert dtype \(slice.dtype) for \(key)")
                    }
                    let shape = try slice.logicalShape.enumerated().map { index, value in
                        guard value <= UInt64(UInt32.max) else {
                            throw RepackError.configurationInvalid(
                                detail: "packed expert shape[\(index)] exceeds UInt32")
                        }
                        return UInt32(value)
                    }
                    let previous = tensors.updateValue(FinchSubTensorV1(
                        offset: slice.offsetInExpertBlob,
                        size: slice.sizeInExpertBlob,
                        dtype: slice.dtype == FinchFormatV1.DType.u32.rawValue ? "U32" : "BF16",
                        shape: shape,
                        bits: slice.bitsForWeights), forKey: key)
                    guard previous == nil else {
                        throw RepackError.configurationInvalid(
                            detail: "duplicate packed expert tensor key \(key)")
                    }
                }
                experts.append(FinchExpertV1(
                    expert: e,
                    physicalRank: nil,
                    offset: base,
                    size: lp.expertStride,
                    tensors: tensors))
            }
            wireLayers.append(FinchLayerV1(layer: lp.layerIndex,
                                        file: layerFile,
                                        experts: experts))
        }
        return try FinchPackedExpertsLayoutCodec.encode(
            FinchPackedExpertsLayoutV1(
                expertStride: expertStride,
                numLayers: numLayers,
                expertsPerLayer: layers.first?.expertsPerLayer ?? 0,
                layers: wireLayers))
    }
}

import Foundation

package struct FQTurboSubTensorV1: Codable, Equatable, Sendable {
    package let offset: UInt64
    package let size: UInt64
    package let dtype: String
    package let shape: [UInt32]
    package let bits: Int?

    package init(offset: UInt64, size: UInt64, dtype: String,
                 shape: [UInt32], bits: Int?) {
        self.offset = offset
        self.size = size
        self.dtype = dtype
        self.shape = shape
        self.bits = bits
    }
}

package struct FQTurboExpertV1: Codable, Equatable, Sendable {
    package let expert: Int?
    package let physicalRank: Int?
    package let offset: UInt64
    package let size: UInt64
    package let tensors: [String: FQTurboSubTensorV1]

    package init(expert: Int?, physicalRank: Int?, offset: UInt64, size: UInt64,
                 tensors: [String: FQTurboSubTensorV1]) {
        self.expert = expert
        self.physicalRank = physicalRank
        self.offset = offset
        self.size = size
        self.tensors = tensors
    }
}

package struct FQTurboLayerV1: Codable, Equatable, Sendable {
    package let layer: Int
    package let file: String
    package let experts: [FQTurboExpertV1]

    package init(layer: Int, file: String, experts: [FQTurboExpertV1]) {
        self.layer = layer
        self.file = file
        self.experts = experts
    }
}

package struct FQTurboPackedExpertsLayoutV1: Codable, Equatable, Sendable {
    package let expertStride: UInt64
    package let numLayers: Int
    package let expertsPerLayer: Int
    package let layers: [FQTurboLayerV1]

    package init(expertStride: UInt64, numLayers: Int, expertsPerLayer: Int,
                 layers: [FQTurboLayerV1]) {
        self.expertStride = expertStride
        self.numLayers = numLayers
        self.expertsPerLayer = expertsPerLayer
        self.layers = layers
    }
}

package enum FQTurboPackedExpertsLayoutCodec {
    package static func decode(_ data: Data) throws -> FQTurboPackedExpertsLayoutV1 {
        let layout: FQTurboPackedExpertsLayoutV1
        do { layout = try JSONDecoder().decode(FQTurboPackedExpertsLayoutV1.self, from: data) }
        catch { throw FQTurboFormatError.invalid(field: "packed_experts/layout.json", reason: "\(error)") }
        try FQTurboV1StructuralValidator.validate(layout)
        return layout
    }

    package static func encode(_ layout: FQTurboPackedExpertsLayoutV1) throws -> Data {
        try FQTurboV1StructuralValidator.validate(layout)
        let encoder = JSONEncoder()
        do {
            let object = try JSONSerialization.jsonObject(with: encoder.encode(layout))
            return try JSONSerialization.data(
                withJSONObject: object,
                options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        } catch {
            throw FQTurboFormatError.invalid(
                field: "packed_experts/layout.json", reason: "\(error)")
        }
    }
}

package enum FQTurboV1StructuralValidator {
    package static func validate(_ layout: FQTurboPackedExpertsLayoutV1) throws {
        guard layout.numLayers > 0, layout.expertsPerLayer > 0,
              layout.expertStride > 0,
              layout.expertStride % FQTurboFormatV1.alignmentBytes == 0,
              layout.layers.count == layout.numLayers else {
            throw FQTurboFormatError.invalid(field: "layout", reason: "invalid dimensions or stride")
        }
        var layerIDs = Set<Int>()
        var layerFiles = Set<String>()
        for layer in layout.layers {
            guard layer.layer >= 0, layer.layer < layout.numLayers,
                  layerIDs.insert(layer.layer).inserted else {
                throw FQTurboFormatError.invalid(field: "layout.layers", reason: "duplicate or invalid layer")
            }
            try FQTurboPathValidator.validateBasename(layer.file,
                                                      field: "layout.layers[\(layer.layer)].file")
            let fileKey = FQTurboPathValidator.appleFilesystemKey(layer.file)
            guard fileKey != "layout.json" else {
                throw FQTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "reserved packed-expert filename")
            }
            guard layerFiles.insert(fileKey).inserted else {
                throw FQTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].file",
                    reason: "duplicate layer filename")
            }
            guard layer.experts.count == layout.expertsPerLayer else {
                throw FQTurboFormatError.invalid(field: "layout.layers[\(layer.layer)].experts",
                                                reason: "wrong expert count")
            }
            var logicalIDs = Set<Int>()
            var physicalRanks = Set<Int>()
            var offsets = Set<UInt64>()
            let hasExplicitLogicalIDs = layer.experts.map(\.expert)
            guard hasExplicitLogicalIDs.allSatisfy({ $0 == nil })
                    || hasExplicitLogicalIDs.allSatisfy({ $0 != nil }) else {
                throw FQTurboFormatError.invalid(
                    field: "layout.layers[\(layer.layer)].experts",
                    reason: "expert ids must be either all explicit or all positional")
            }
            for (position, expert) in layer.experts.enumerated() {
                let logical = expert.expert ?? position
                let physical = expert.physicalRank ?? logical
                guard logical >= 0, logical < layout.expertsPerLayer,
                      physical >= 0, physical < layout.expertsPerLayer,
                      logicalIDs.insert(logical).inserted,
                      physicalRanks.insert(physical).inserted,
                      offsets.insert(expert.offset).inserted else {
                    throw FQTurboFormatError.invalid(field: "layout.layers[\(layer.layer)].experts",
                                                    reason: "duplicate or invalid expert mapping")
                }
                let expectedOffset = try fqturboCheckedMultiply(UInt64(physical), layout.expertStride,
                                                               field: "expert.offset")
                guard expert.offset == expectedOffset, expert.size == layout.expertStride else {
                    throw FQTurboFormatError.invalid(field: "expert[\(logical)]",
                                                    reason: "offset or size does not match physical rank")
                }
                var tensorRanges: [(start: UInt64, end: UInt64, name: String)] = []
                for (name, tensor) in expert.tensors {
                    guard tensor.dtype == "U32" || tensor.dtype == "BF16",
                          !name.isEmpty, tensor.size > 0,
                          !tensor.shape.isEmpty,
                          tensor.shape.allSatisfy({ $0 > 0 }),
                          tensor.bits.map({ $0 > 0 && $0 <= 32 }) ?? true else {
                        throw FQTurboFormatError.invalid(field: "expert[\(logical)].tensors.\(name)",
                                                        reason: "invalid dtype or shape")
                    }
                    let end = try fqturboCheckedAdd(tensor.offset, tensor.size,
                                                   field: "tensor.\(name).range")
                    guard end <= expert.size else {
                        throw FQTurboFormatError.invalid(field: "expert[\(logical)].tensors.\(name)",
                                                        reason: "range exceeds expert blob")
                    }
                    tensorRanges.append((tensor.offset, end, name))
                }
                let sortedRanges = tensorRanges.sorted {
                    $0.start == $1.start ? $0.end < $1.end : $0.start < $1.start
                }
                for pair in zip(sortedRanges, sortedRanges.dropFirst())
                    where pair.0.end > pair.1.start {
                    throw FQTurboFormatError.invalid(
                        field: "expert[\(logical)].tensors",
                        reason: "overlapping ranges \(pair.0.name) and \(pair.1.name)")
                }
            }
        }
    }

    package static func crossValidate(manifest: FQTurboManifestV1,
                                      layout: FQTurboPackedExpertsLayoutV1) throws {
        try crossValidate(
            manifestNumLayers: manifest.numLayers,
            manifestExpertsPerLayer: manifest.expertsPerLayer,
            manifestExpertStride: manifest.expertStride,
            manifestFileSizes: manifest.files.mapValues(\.size),
            layout: layout)
    }

    package static func crossValidate(
        manifestNumLayers: Int,
        manifestExpertsPerLayer: Int,
        manifestExpertStride: UInt64,
        manifestFileSizes: [String: UInt64],
        layout: FQTurboPackedExpertsLayoutV1
    ) throws {
        guard manifestNumLayers == layout.numLayers,
              manifestExpertsPerLayer == layout.expertsPerLayer,
              manifestExpertStride == layout.expertStride else {
            throw FQTurboFormatError.invalid(field: "manifest/layout",
                                            reason: "dimension mismatch")
        }
        let expectedLayerSize = try fqturboCheckedMultiply(UInt64(layout.expertsPerLayer),
                                                          layout.expertStride,
                                                          field: "layout.layerSize")
        for layer in layout.layers {
            let path = "packed_experts/\(layer.file)"
            guard manifestFileSizes[path] == expectedLayerSize else {
                throw FQTurboFormatError.invalid(field: "manifest.files.\(path)",
                                                reason: "missing or wrong layer size")
            }
        }
    }
}

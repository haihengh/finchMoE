import FinchMoE
import Foundation
import Metal

/// One image encoded by the tower, in the text model's hidden width.
struct VisionImageEmbedding {
    /// `[tokenCount, textHiddenSize]` FP16, merge-block order.
    let embeddings: MTLBuffer
    let tokenCount: Int
    let gridT: Int
    let gridH: Int
    let gridW: Int
    let pixelWidth: Int
    let pixelHeight: Int
}

/// Trace of one tower run. The capture buffers are only allocated when the
/// caller asks for them (tests do; the app never does).
struct VisionTowerRun {
    let image: VisionImageEmbedding
    let patchEmbed: MTLBuffer?
    let blockOutputs: [Int: MTLBuffer]
}

/// Qwen3-VL BF16 vision tower.
///
/// The tower reads its weights straight out of the mapped
/// `vision/vision.safetensors` payload; only the image activation buffers are
/// allocated per run. It is deliberately separate from `FinchMoE`: a text-only
/// application links only the engine and never compiles these shaders or opens
/// the payload.
final class VisionTower {
    let config: VisionConfig
    let kernels: VisionTowerKernels
    private let weights: VisionWeights
    /// `pos_embed.weight` converted once, because the bilinear interpolation is
    /// cheap on the CPU and the table is only 2304×1152.
    private let positionTable: [Float]

    init(directory: FinchModelDirectory,
         device: MTLDevice,
         queue: MTLCommandQueue) throws {
        self.config = try VisionConfig.load(directory: directory)
        self.kernels = try VisionTowerKernels(device: device, queue: queue)
        self.weights = try VisionWeights(directory: directory, device: device)

        let table = try weights.tensor("model.visual.pos_embed.weight")
        let side = config.geometry.gridSide
        let hidden = config.geometry.hiddenSize
        guard table.shape == [side * side, hidden] else {
            throw VisionError.invalidPayload(
                "pos_embed.weight has shape \(table.shape), expected "
                + "\([side * side, hidden])")
        }
        let pointer = weights.buffer.contents()
            .advanced(by: table.offset)
            .assumingMemoryBound(to: UInt16.self)
        var converted = [Float](repeating: 0, count: side * side * hidden)
        for index in converted.indices {
            converted[index] = Self.bfloatToFloat(pointer[index])
        }
        self.positionTable = converted

        // Fail at load time, not on the first image, if a tensor is missing.
        try validateWeights()
    }

    // MARK: - Entry points

    func encode(imageData: Data,
                maxPixels: Int = 1_536 * 1_536,
                captureBlocks: Set<Int> = []) throws -> VisionTowerRun {
        let prepared = try VisionPreprocessor.prepare(
            imageData: imageData,
            config: config,
            device: kernels.device,
            maxPixels: maxPixels)
        return try encode(prepared: prepared, captureBlocks: captureBlocks)
    }

    func encode(prepared: VisionPreparedImage,
                captureBlocks: Set<Int> = []) throws -> VisionTowerRun {
        let geometry = config.geometry
        let patches = prepared.patchCount
        let hidden = geometry.hiddenSize
        guard patches > 0, patches % (geometry.spatialMergeSize * geometry.spatialMergeSize) == 0 else {
            throw VisionError.unsupportedImage("patch count \(patches) is not merge-aligned")
        }
        guard let commandBuffer = kernels.queue.makeCommandBuffer() else {
            throw VisionError.encodingFailed("could not create a command buffer")
        }

        let hiddenBytes = patches * hidden * 2
        let hiddenA = try kernels.makeBuffer(length: hiddenBytes)
        let hiddenB = try kernels.makeBuffer(length: hiddenBytes)
        let normed = try kernels.makeBuffer(length: hiddenBytes)
        let qkv = try kernels.makeBuffer(length: patches * hidden * 3 * 2)
        let attention = try kernels.makeBuffer(length: hiddenBytes)
        let mlp = try kernels.makeBuffer(length: patches * geometry.intermediateSize * 2)

        let position = try Self.makeHalfBuffer(
            kernels: kernels,
            values: interpolatedPositionEmbedding(prepared: prepared))
        let rope = makeRopeTables(prepared: prepared)
        let cosine = try Self.makeFloatBuffer(kernels: kernels, values: rope.cosine)
        let sine = try Self.makeFloatBuffer(kernels: kernels, values: rope.sine)

        var captures: [Int: MTLBuffer] = [:]
        func capture(_ source: MTLBuffer, key: Int, bytes: Int) throws {
            let destination = try kernels.makeBuffer(length: bytes)
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw VisionError.encodingFailed("could not create a blit encoder")
            }
            blit.copy(from: source, sourceOffset: 0,
                      to: destination, destinationOffset: 0,
                      size: bytes)
            blit.endEncoding()
            captures[key] = destination
        }

        // Patch embed: pixels [patches, 1536] × weight [1152, 1536].
        let patchWeight = try matrix("model.visual.patch_embed.proj.weight",
                                     rows: hidden,
                                     cols: geometry.inChannels
                                         * geometry.temporalPatchSize
                                         * geometry.patchSize
                                         * geometry.patchSize)
        let patchBias = try vector("model.visual.patch_embed.proj.bias", count: hidden)
        kernels.encodeGEMM(commandBuffer: commandBuffer,
                           a: prepared.pixels, m: patches,
                           k: patchWeight.cols,
                           weights: patchWeight,
                           bias: patchBias,
                           residual: nil,
                           c: hiddenA,
                           flags: VisionTowerKernels.gemmBias)
        if captureBlocks.contains(-1) {
            try capture(hiddenA, key: -1, bytes: hiddenBytes)
        }
        kernels.encodeAdd(commandBuffer: commandBuffer,
                          a: hiddenA, b: position, out: hiddenA,
                          count: patches * hidden)

        let source = hiddenA
        let destination = hiddenB
        for layer in 0..<geometry.depth {
            let prefix = "model.visual.blocks.\(layer)"
            kernels.encodeLayerNorm(
                commandBuffer: commandBuffer,
                x: source,
                weight: try vector("\(prefix).norm1.weight", count: hidden),
                bias: try vector("\(prefix).norm1.bias", count: hidden),
                rows: patches, columns: hidden, eps: geometry.layerNormEps,
                out: normed)
            if layer == 0, captureBlocks.contains(-2) {
                try capture(normed, key: -2, bytes: hiddenBytes)
            }

            let qkvWeight = try matrix("\(prefix).attn.qkv.weight",
                                       rows: hidden * 3, cols: hidden)
            kernels.encodeGEMM(commandBuffer: commandBuffer,
                               a: normed, m: patches, k: hidden,
                               weights: qkvWeight,
                               bias: try vector("\(prefix).attn.qkv.bias",
                                                count: hidden * 3),
                               residual: nil,
                               c: qkv,
                               flags: VisionTowerKernels.gemmBias)
            kernels.encodeRope2D(commandBuffer: commandBuffer,
                                 qkv: qkv,
                                 cos: cosine, sin: sine,
                                 tokens: patches)
            if layer == 0, captureBlocks.contains(-3) {
                try capture(qkv, key: -3, bytes: patches * hidden * 3 * 2)
            }
            kernels.encodeAttention(commandBuffer: commandBuffer,
                                    qkv: qkv, out: attention,
                                    tokens: patches)
            if layer == 0, captureBlocks.contains(-4) {
                try capture(attention, key: -4, bytes: hiddenBytes)
            }

            let projWeight = try matrix("\(prefix).attn.proj.weight",
                                        rows: hidden, cols: hidden)
            kernels.encodeGEMM(commandBuffer: commandBuffer,
                               a: attention, m: patches, k: hidden,
                               weights: projWeight,
                               bias: try vector("\(prefix).attn.proj.bias",
                                                count: hidden),
                               residual: source,
                               c: destination,
                               flags: VisionTowerKernels.gemmBias
                                   | VisionTowerKernels.gemmResidual)
            if layer == 0, captureBlocks.contains(-5) {
                try capture(destination, key: -5, bytes: hiddenBytes)
            }

            kernels.encodeLayerNorm(
                commandBuffer: commandBuffer,
                x: destination,
                weight: try vector("\(prefix).norm2.weight", count: hidden),
                bias: try vector("\(prefix).norm2.bias", count: hidden),
                rows: patches, columns: hidden, eps: geometry.layerNormEps,
                out: normed)
            if layer == 0, captureBlocks.contains(-7) {
                try capture(normed, key: -7, bytes: hiddenBytes)
            }

            let fc1Weight = try matrix("\(prefix).mlp.linear_fc1.weight",
                                       rows: geometry.intermediateSize,
                                       cols: hidden)
            kernels.encodeGEMM(commandBuffer: commandBuffer,
                               a: normed, m: patches, k: hidden,
                               weights: fc1Weight,
                               bias: try vector("\(prefix).mlp.linear_fc1.bias",
                                                count: geometry.intermediateSize),
                               residual: nil,
                               c: mlp,
                               flags: VisionTowerKernels.gemmBias
                                   | VisionTowerKernels.gemmGeluTanh)
            if layer == 0, captureBlocks.contains(-6) {
                try capture(mlp, key: -6,
                            bytes: patches * geometry.intermediateSize * 2)
            }

            let fc2Weight = try matrix("\(prefix).mlp.linear_fc2.weight",
                                       rows: hidden,
                                       cols: geometry.intermediateSize)
            kernels.encodeGEMM(commandBuffer: commandBuffer,
                               a: mlp, m: patches, k: geometry.intermediateSize,
                               weights: fc2Weight,
                               bias: try vector("\(prefix).mlp.linear_fc2.bias",
                                                count: hidden),
                               residual: destination,
                               c: source,
                               flags: VisionTowerKernels.gemmBias
                                   | VisionTowerKernels.gemmResidual)

            if captureBlocks.contains(layer) {
                try capture(source, key: layer, bytes: hiddenBytes)
            }
        }

        // Merger: per-patch LayerNorm, 2×2 grouping, then two 4608-wide linears.
        let mergeSize = geometry.spatialMergeSize
        let mergedWidth = hidden * mergeSize * mergeSize
        let groups = patches / (mergeSize * mergeSize)
        kernels.encodeLayerNorm(
            commandBuffer: commandBuffer,
            x: source,
            weight: try vector("model.visual.merger.norm.weight", count: hidden),
            bias: try vector("model.visual.merger.norm.bias", count: hidden),
            rows: patches, columns: hidden, eps: geometry.layerNormEps,
            out: normed)

        let merged = try kernels.makeBuffer(length: groups * mergedWidth * 2)
        kernels.encodeConcatMerge(commandBuffer: commandBuffer,
                                  input: normed, output: merged,
                                  patches: patches, width: hidden)

        let mergerFC1 = try matrix("model.visual.merger.linear_fc1.weight",
                                   rows: mergedWidth, cols: mergedWidth)
        let mergerMid = try kernels.makeBuffer(length: groups * mergedWidth * 2)
        kernels.encodeGEMM(commandBuffer: commandBuffer,
                           a: merged, m: groups, k: mergedWidth,
                           weights: mergerFC1,
                           bias: try vector("model.visual.merger.linear_fc1.bias",
                                            count: mergedWidth),
                           residual: nil,
                           c: mergerMid,
                           flags: VisionTowerKernels.gemmBias
                               | VisionTowerKernels.gemmGeluExact)

        let outputWidth = geometry.outHiddenSize
        let mergerFC2 = try matrix("model.visual.merger.linear_fc2.weight",
                                   rows: outputWidth, cols: mergedWidth)
        let embeddings = try kernels.makeBuffer(length: groups * outputWidth * 2)
        kernels.encodeGEMM(commandBuffer: commandBuffer,
                           a: mergerMid, m: groups, k: mergedWidth,
                           weights: mergerFC2,
                           bias: try vector("model.visual.merger.linear_fc2.bias",
                                            count: outputWidth),
                           residual: nil,
                           c: embeddings,
                           flags: VisionTowerKernels.gemmBias)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw VisionError.encodingFailed("vision tower command buffer: \(error)")
        }

        let image = VisionImageEmbedding(
            embeddings: embeddings,
            tokenCount: groups,
            gridT: prepared.gridT,
            gridH: prepared.gridH,
            gridW: prepared.gridW,
            pixelWidth: prepared.pixelWidth,
            pixelHeight: prepared.pixelHeight)
        return VisionTowerRun(image: image,
                              patchEmbed: captures[-1],
                              blockOutputs: captures.filter { $0.key != -1 })
    }

    // MARK: - CPU-side tables

    private func interpolatedPositionEmbedding(prepared: VisionPreparedImage) -> [Float16] {
        let side = config.geometry.gridSide
        let hidden = config.geometry.hiddenSize
        let merge = config.geometry.spatialMergeSize
        let gridH = prepared.gridH
        let gridW = prepared.gridW

        func linspace(_ count: Int) -> [Float] {
            guard count > 1 else { return [0] }
            let step = Double(side - 1) / Double(count - 1)
            return (0..<count).map { Float(Double($0) * step) }
        }
        let hGrid = linspace(gridH)
        let wGrid = linspace(gridW)

        var output = [Float16](repeating: 0, count: gridH * gridW * hidden)
        var token = 0
        for blockY in stride(from: 0, to: gridH, by: merge) {
            for blockX in stride(from: 0, to: gridW, by: merge) {
                for intraY in 0..<merge {
                    for intraX in 0..<merge {
                        let h = blockY + intraY
                        let w = blockX + intraX
                        let hValue = hGrid[h]
                        let wValue = wGrid[w]
                        let h0 = Int(hValue)
                        let w0 = Int(wValue)
                        let h1 = min(h0 + 1, side - 1)
                        let w1 = min(w0 + 1, side - 1)
                        let hf = hValue - Float(h0)
                        let wf = wValue - Float(w0)
                        let c00 = (1 - hf) * (1 - wf)
                        let c01 = (1 - hf) * wf
                        let c10 = hf * (1 - wf)
                        let c11 = hf * wf
                        let i00 = (h0 * side + w0) * hidden
                        let i01 = (h0 * side + w1) * hidden
                        let i10 = (h1 * side + w0) * hidden
                        let i11 = (h1 * side + w1) * hidden
                        let base = token * hidden
                        for d in 0..<hidden {
                            output[base + d] = Float16(
                                c00 * positionTable[i00 + d]
                                + c01 * positionTable[i01 + d]
                                + c10 * positionTable[i10 + d]
                                + c11 * positionTable[i11 + d])
                        }
                        token += 1
                    }
                }
            }
        }
        return output
    }

    private func makeRopeTables(prepared: VisionPreparedImage)
        -> (cosine: [Float], sine: [Float]) {
        let merge = config.geometry.spatialMergeSize
        let gridH = prepared.gridH
        let gridW = prepared.gridW
        let pairs = 18
        let invFrequency = (0..<pairs).map { index in
            1.0 / pow(10_000.0, Double(2 * index) / 36.0)
        }
        var cosine = [Float](repeating: 0, count: gridH * gridW * 36)
        var sine = [Float](repeating: 0, count: gridH * gridW * 36)
        var token = 0
        for blockY in stride(from: 0, to: gridH, by: merge) {
            for blockX in stride(from: 0, to: gridW, by: merge) {
                for intraY in 0..<merge {
                    for intraX in 0..<merge {
                        let h = blockY + intraY
                        let w = blockX + intraX
                        for index in 0..<pairs {
                            let hAngle = Float(h) * Float(invFrequency[index])
                            let wAngle = Float(w) * Float(invFrequency[index])
                            cosine[token * 36 + index] = cos(hAngle)
                            sine[token * 36 + index] = sin(hAngle)
                            cosine[token * 36 + pairs + index] = cos(wAngle)
                            sine[token * 36 + pairs + index] = sin(wAngle)
                        }
                        token += 1
                    }
                }
            }
        }
        return (cosine, sine)
    }

    // MARK: - Weights

    private func validateWeights() throws {
        let hidden = config.geometry.hiddenSize
        let intermediate = config.geometry.intermediateSize
        let merged = hidden * config.geometry.spatialMergeSize * config.geometry.spatialMergeSize
        _ = try matrix("model.visual.patch_embed.proj.weight",
                       rows: hidden,
                       cols: config.geometry.inChannels
                           * config.geometry.temporalPatchSize
                           * config.geometry.patchSize
                           * config.geometry.patchSize)
        _ = try vector("model.visual.patch_embed.proj.bias", count: hidden)
        for layer in 0..<config.geometry.depth {
            let prefix = "model.visual.blocks.\(layer)"
            _ = try vector("\(prefix).norm1.weight", count: hidden)
            _ = try vector("\(prefix).norm1.bias", count: hidden)
            _ = try vector("\(prefix).norm2.weight", count: hidden)
            _ = try vector("\(prefix).norm2.bias", count: hidden)
            _ = try matrix("\(prefix).attn.qkv.weight", rows: hidden * 3, cols: hidden)
            _ = try vector("\(prefix).attn.qkv.bias", count: hidden * 3)
            _ = try matrix("\(prefix).attn.proj.weight", rows: hidden, cols: hidden)
            _ = try vector("\(prefix).attn.proj.bias", count: hidden)
            _ = try matrix("\(prefix).mlp.linear_fc1.weight",
                           rows: intermediate, cols: hidden)
            _ = try vector("\(prefix).mlp.linear_fc1.bias", count: intermediate)
            _ = try matrix("\(prefix).mlp.linear_fc2.weight",
                           rows: hidden, cols: intermediate)
            _ = try vector("\(prefix).mlp.linear_fc2.bias", count: hidden)
        }
        _ = try vector("model.visual.merger.norm.weight", count: hidden)
        _ = try vector("model.visual.merger.norm.bias", count: hidden)
        _ = try matrix("model.visual.merger.linear_fc1.weight",
                       rows: merged, cols: merged)
        _ = try vector("model.visual.merger.linear_fc1.bias", count: merged)
        _ = try matrix("model.visual.merger.linear_fc2.weight",
                       rows: config.geometry.outHiddenSize, cols: merged)
        _ = try vector("model.visual.merger.linear_fc2.bias",
                       count: config.geometry.outHiddenSize)
    }

    private func matrix(_ name: String, rows: Int, cols: Int) throws -> VisionBFloatMatrix {
        let tensor = try weights.tensor(name)
        guard tensor.shape.reduce(1, *) == rows * cols,
              tensor.sizeBytes == rows * cols * 2 else {
            throw VisionError.invalidPayload(
                "\(name) has shape \(tensor.shape), expected \(rows)×\(cols) BF16")
        }
        return VisionBFloatMatrix(buffer: weights.buffer,
                                  offset: tensor.offset,
                                  rows: rows, cols: cols)
    }

    private func vector(_ name: String, count: Int) throws -> VisionBFloatVector {
        let tensor = try weights.tensor(name)
        guard tensor.shape.reduce(1, *) == count,
              tensor.sizeBytes == count * 2 else {
            throw VisionError.invalidPayload(
                "\(name) has shape \(tensor.shape), expected \(count) BF16")
        }
        return VisionBFloatVector(buffer: weights.buffer,
                                  offset: tensor.offset,
                                  count: count)
    }

    // MARK: - Helpers

    private static func bfloatToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    private static func makeHalfBuffer(kernels: VisionTowerKernels,
                                       values: [Float16]) throws -> MTLBuffer {
        try values.withUnsafeBytes { raw in
            try kernels.makeBuffer(bytes: raw.baseAddress!, length: raw.count)
        }
    }

    private static func makeFloatBuffer(kernels: VisionTowerKernels,
                                        values: [Float]) throws -> MTLBuffer {
        try values.withUnsafeBytes { raw in
            try kernels.makeBuffer(bytes: raw.baseAddress!, length: raw.count)
        }
    }
}

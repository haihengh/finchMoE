import Foundation
import Metal

/// A BF16 tensor inside the mapped vision payload. `rows` is the linear
/// layer's output width and `cols` its input width, so the GEMM computes
/// `y = x * weight^T` directly against the safetensors layout.
struct VisionBFloatMatrix {
    let buffer: MTLBuffer
    let offset: Int
    let rows: Int
    let cols: Int
}

struct VisionBFloatVector {
    let buffer: MTLBuffer
    let offset: Int
    let count: Int
}

/// Compute pipelines and dispatch helpers for the vision tower. The shaders
/// live in this target (not FinchMoE) so a text-only application never compiles
/// them, and they are compiled at runtime like the engine's own Metal sources.
final class VisionTowerKernels {
    static let gemmBias: UInt32 = 1 << 0
    static let gemmResidual: UInt32 = 1 << 1
    static let gemmGeluTanh: UInt32 = 1 << 2
    static let gemmGeluExact: UInt32 = 1 << 3

    let device: MTLDevice
    let queue: MTLCommandQueue
    private let gemmPipeline: MTLComputePipelineState
    private let layernormPipeline: MTLComputePipelineState
    private let ropePipeline: MTLComputePipelineState
    private let attentionPipeline: MTLComputePipelineState
    private let concatPipeline: MTLComputePipelineState
    private let addPipeline: MTLComputePipelineState
    private let zeroBuffer: MTLBuffer

    init(device: MTLDevice, queue: MTLCommandQueue) throws {
        self.device = device
        self.queue = queue

        let source: String
        if let url = Bundle.module.url(forResource: "vision_tower",
                                       withExtension: "metal",
                                       subdirectory: "Metal")
            ?? Bundle.module.url(forResource: "vision_tower",
                                 withExtension: "metal") {
            source = try String(contentsOf: url, encoding: .utf8)
        } else {
            throw VisionError.encodingFailed(
                "vision_tower.metal is missing from the FinchMoEVision bundle")
        }
        let options = MTLCompileOptions()
        options.languageVersion = .version4_0
        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: options)
        } catch {
            throw VisionError.encodingFailed("could not compile the vision shaders: \(error)")
        }
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw VisionError.encodingFailed("missing vision shader \(name)")
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw VisionError.encodingFailed("could not build \(name): \(error)")
            }
        }
        self.gemmPipeline = try pipeline("vision_gemm_bf16w")
        self.layernormPipeline = try pipeline("vision_layernorm_bf16w")
        self.ropePipeline = try pipeline("vision_rope_2d")
        self.attentionPipeline = try pipeline("vision_attention")
        self.concatPipeline = try pipeline("vision_concat_merge")
        self.addPipeline = try pipeline("vision_add_f16")
        guard let zero = device.makeBuffer(length: 16, options: .storageModeShared) else {
            throw VisionError.encodingFailed("could not allocate the zero buffer")
        }
        self.zeroBuffer = zero
    }

    func makeBuffer(length: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: max(length, 16),
                                             options: .storageModeShared) else {
            throw VisionError.encodingFailed("could not allocate a \(length)-byte buffer")
        }
        return buffer
    }

    func makeBuffer(bytes: UnsafeRawPointer, length: Int) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(bytes: bytes,
                                             length: max(length, 16),
                                             options: .storageModeShared) else {
            throw VisionError.encodingFailed("could not upload a \(length)-byte buffer")
        }
        return buffer
    }

    // MARK: - GEMM

    func encodeGEMM(commandBuffer: MTLCommandBuffer,
                    a: MTLBuffer,
                    aOffset: Int = 0,
                    m: Int,
                    k: Int,
                    weights: VisionBFloatMatrix,
                    bias: VisionBFloatVector?,
                    residual: MTLBuffer?,
                    residualOffset: Int = 0,
                    c: MTLBuffer,
                    cOffset: Int = 0,
                    flags: UInt32) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(gemmPipeline)
        encoder.setBuffer(a, offset: aOffset, index: 0)
        encoder.setBuffer(weights.buffer, offset: weights.offset, index: 1)
        encoder.setBuffer(bias?.buffer ?? zeroBuffer, offset: bias?.offset ?? 0, index: 2)
        encoder.setBuffer(residual ?? c,
                          offset: residual != nil ? residualOffset : cOffset,
                          index: 3)
        encoder.setBuffer(c, offset: cOffset, index: 4)
        var mValue = UInt32(m)
        var nValue = UInt32(weights.rows)
        var kValue = UInt32(k)
        var flagValue = flags
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&flagValue, length: MemoryLayout<UInt32>.size, index: 8)
        let groups = MTLSize(width: (weights.rows + 63) / 64,
                             height: (m + 15) / 16,
                             depth: 1)
        encoder.dispatchThreadgroups(groups,
                                     threadsPerThreadgroup: MTLSize(width: 256,
                                                                    height: 1,
                                                                    depth: 1))
        encoder.endEncoding()
    }

    // MARK: - LayerNorm

    func encodeLayerNorm(commandBuffer: MTLCommandBuffer,
                         x: MTLBuffer,
                         xOffset: Int = 0,
                         weight: VisionBFloatVector,
                         bias: VisionBFloatVector,
                         rows: Int,
                         columns: Int,
                         eps: Float,
                         out: MTLBuffer,
                         outOffset: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(layernormPipeline)
        encoder.setBuffer(x, offset: xOffset, index: 0)
        encoder.setBuffer(weight.buffer, offset: weight.offset, index: 1)
        encoder.setBuffer(bias.buffer, offset: bias.offset, index: 2)
        encoder.setBuffer(out, offset: outOffset, index: 3)
        var rowsValue = UInt32(rows)
        var columnsValue = UInt32(columns)
        var epsValue = eps
        encoder.setBytes(&rowsValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&columnsValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&epsValue, length: MemoryLayout<Float>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - RoPE

    func encodeRope2D(commandBuffer: MTLCommandBuffer,
                      qkv: MTLBuffer,
                      cos: MTLBuffer,
                      sin: MTLBuffer,
                      tokens: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(ropePipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(cos, offset: 0, index: 1)
        encoder.setBuffer(sin, offset: 0, index: 2)
        var tokensValue = UInt32(tokens)
        encoder.setBytes(&tokensValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: 36, height: 16, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: 36, height: 16, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Attention

    func encodeAttention(commandBuffer: MTLCommandBuffer,
                         qkv: MTLBuffer,
                         out: MTLBuffer,
                         tokens: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(attentionPipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(out, offset: 0, index: 1)
        var tokensValue = UInt32(tokens)
        var scale = Float(1.0 / 72.0).squareRoot()
        encoder.setBytes(&tokensValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: (tokens + 15) / 16, height: 16, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Elementwise add

    func encodeAdd(commandBuffer: MTLCommandBuffer,
                   a: MTLBuffer,
                   b: MTLBuffer,
                   out: MTLBuffer,
                   count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(a, offset: 0, index: 0)
        encoder.setBuffer(b, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var countValue = UInt32(count)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    // MARK: - Merger concat

    func encodeConcatMerge(commandBuffer: MTLCommandBuffer,
                           input: MTLBuffer,
                           output: MTLBuffer,
                           patches: Int,
                           width: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(concatPipeline)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var patchesValue = UInt32(patches)
        var widthValue = UInt32(width)
        encoder.setBytes(&patchesValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&widthValue, length: MemoryLayout<UInt32>.size, index: 3)
        let total = (patches / 4) * width * 4
        encoder.dispatchThreads(
            MTLSize(width: total, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }
}

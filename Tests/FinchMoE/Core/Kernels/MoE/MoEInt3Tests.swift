import Foundation
import Metal
import Testing
@testable import FinchMoE
import FinchMoEValidationSupport

/// 3-bit routed experts (the `3bit-experiments` branch): the int3 decode
/// pipeline must reproduce an independent float reference built from the
/// *same* quantized rows — the reference dequantizes with
/// FinchQuantization and runs a naive FP32 FFN, so a bit-layout slip in the
/// Metal unpacking (triplet order, lane mapping, group indexing) shows up as
/// a mismatch rather than a plausible-looking number.
@Suite struct MoEInt3Tests {
    private static let dimension = 128
    private static let intermediate = 64
    private static let topK = 8

    private struct RoutedBlob {
        let bytes: [UInt8]
        let offsets: MoEExpertOffsets
    }

    private static func denseGEMV(_ rows: [[Float]], _ x: [Float]) -> [Float] {
        rows.map { row in
            var acc: Float = 0
            for i in 0..<x.count { acc += row[i] * x[i] }
            return acc
        }
    }

    private static func silu(_ x: Float) -> Float {
        x / (1 + Foundation.exp(-x))
    }

    private static func makeBlob(gate: [[Float]],
                                 up: [[Float]],
                                 down: [[Float]]) -> RoutedBlob {
        func packed(_ rows: [[Float]])
            -> (weights: [UInt8], scales: [UInt16], biases: [UInt16]) {
            let quantized = rows.map { Quantization.quantizeInt3Affine($0) }
            return (quantized.flatMap(\.packed),
                    quantized.flatMap(\.scales),
                    quantized.flatMap(\.biases))
        }
        var bytes = [UInt8]()
        func append(_ values: [UInt8]) { bytes.append(contentsOf: values) }
        func append(_ values: [UInt16]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }
        let gateValues = packed(gate)
        let upValues = packed(up)
        let downValues = packed(down)
        let gateW = UInt32(bytes.count); append(gateValues.weights)
        let gateS = UInt32(bytes.count); append(gateValues.scales)
        let gateB = UInt32(bytes.count); append(gateValues.biases)
        let upW = UInt32(bytes.count); append(upValues.weights)
        let upS = UInt32(bytes.count); append(upValues.scales)
        let upB = UInt32(bytes.count); append(upValues.biases)
        let downW = UInt32(bytes.count); append(downValues.weights)
        let downS = UInt32(bytes.count); append(downValues.scales)
        let downB = UInt32(bytes.count); append(downValues.biases)
        return RoutedBlob(
            bytes: bytes,
            offsets: MoEExpertOffsets(
                gateWOff: gateW, gateSOff: gateS, gateBOff: gateB,
                upWOff: upW, upSOff: upS, upBOff: upB,
                downWOff: downW, downSOff: downS, downBOff: downB))
    }

    @Test func routedPipelineInt3SiluMatchesReference() throws {
        try Self.runInt3Pipeline(dimension: Self.dimension,
                                 intermediate: Self.intermediate,
                                 topK: Self.topK,
                                 seed: 0x3A1)
    }

    /// Production-shaped rows: many groups per row, so the kernel's four-group
    /// block path (and not just its single-group tail) actually executes.
    /// The 2026-10-04 eval crash (garbage output on the real 3-bit install)
    /// lived in exactly this gap — every small-dims test used one group.
    @Test func routedPipelineInt3RealisticDimsMatchesReference() throws {
        try Self.runInt3Pipeline(dimension: 2048,
                                 intermediate: 512,
                                 topK: 2,
                                 seed: 0x3A2)
    }

    private static func runInt3Pipeline(dimension: Int,
                                        intermediate: Int,
                                        topK: Int,
                                        seed: UInt64) throws {
        var rng = SeedTree(seed).key("int3-routed-moe")
        func matrix(rows: Int, columns: Int) -> [[Float]] {
            (0..<rows).map { _ in
                (0..<columns).map { _ in rng.uniform(-0.4, 0.4) }
            }
        }

        var gates = [[[Float]]]()
        var ups = [[[Float]]]()
        var downs = [[[Float]]]()
        for _ in 0..<topK {
            gates.append(matrix(rows: intermediate, columns: dimension))
            ups.append(matrix(rows: intermediate, columns: dimension))
            downs.append(matrix(rows: dimension, columns: intermediate))
        }
        let x = (0..<dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let residual = (0..<dimension).map { _ in
            Float(Float16(rng.uniform(-0.5, 0.5)))
        }
        let routingWeights = (0..<topK).map {
            Float(Float16(0.04 + Float($0) * 0.015))
        }

        // Reference: the same int3 rows, dequantized, through a naive FP32 FFN.
        var expected = residual
        for slot in 0..<topK {
            let gateDeq = gates[slot].map {
                Quantization.dequantizeInt3Affine(
                    Quantization.quantizeInt3Affine($0), n: dimension)
            }
            let upDeq = ups[slot].map {
                Quantization.dequantizeInt3Affine(
                    Quantization.quantizeInt3Affine($0), n: dimension)
            }
            let downDeq = downs[slot].map {
                Quantization.dequantizeInt3Affine(
                    Quantization.quantizeInt3Affine($0), n: intermediate)
            }
            let gateOut = Self.denseGEMV(gateDeq, x)
            let upOut = Self.denseGEMV(upDeq, x)
            let act = zip(gateOut, upOut).map { Self.silu($0) * $1 }
            let downOut = Self.denseGEMV(downDeq, act)
            expected = zip(expected, downOut).map {
                $0 + routingWeights[slot] * $1
            }
        }

        let blobs = (0..<topK).map {
            Self.makeBlob(gate: gates[$0], up: ups[$0], down: downs[$0])
        }

        let context = try MetalContext()
        let kernel = try MoE(context: context)
        let routedBuffers = blobs.compactMap {
            context.device.makeBuffer(bytes: $0.bytes,
                                      length: $0.bytes.count,
                                      options: .storageModeShared)
        }
        guard routedBuffers.count == topK,
              let xBuffer = Fp16Buffer.make(context.device, values: x),
              let residualBuffer = Fp16Buffer.make(context.device, values: residual),
              let routingBuffer = Fp16Buffer.make(context.device, values: routingWeights),
              let acts = Fp16Buffer.make(
                context.device, count: topK * intermediate),
              let output = Fp16Buffer.make(context.device, count: dimension),
              let argumentBuffer = kernel.makeRoutedArgumentBuffer(
                routedBlobs: routedBuffers,
                topK: UInt32(topK)) else {
            Issue.record("buffer allocation failed")
            return
        }

        let command = context.queue.makeCommandBuffer()!
        kernel.encodeRoutedPersistentPhase1U16Load(
            commandBuffer: command,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            x: xBuffer,
            acts: acts,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            activation: .silu,
            expertBits: 3)
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: command,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: acts,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: output,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            expertBits: 3)
        // The hit-split arm runs the subset kernel for two halves of the
        // slots; it must produce the same acts/output as the full kernel.
        // (The real decode path uses this whenever the prefetch plan splits.)
        let splitOutput = try #require(Fp16Buffer.make(context.device, count: dimension))
        let splitActs = try #require(Fp16Buffer.make(context.device, count: topK * intermediate))
        guard topK >= 2 else { return }
        let half = topK / 2
        let split = context.queue.makeCommandBuffer()!
        for slots in [Array(UInt32(0)..<UInt32(half)),
                      Array(UInt32(half)..<UInt32(topK))] {
            let slotBuffer = try #require(context.device.makeBuffer(
                bytes: slots,
                length: slots.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared))
            kernel.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: split,
                routedArgBuffer: argumentBuffer,
                routedBlobs: routedBuffers,
                routedOffsets: blobs[0].offsets,
                x: xBuffer,
                acts: splitActs,
                activeSlots: slotBuffer,
                activeSlotIndices: slots,
                activeCount: UInt32(slots.count),
                d: UInt32(dimension),
                f: UInt32(intermediate),
                topK: UInt32(topK),
                activation: .silu,
                expertBits: 3)
        }
        kernel.encodeRoutedPersistentPhase2Reduce(
            commandBuffer: split,
            routedArgBuffer: argumentBuffer,
            routedBlobs: routedBuffers,
            routedOffsets: blobs[0].offsets,
            acts: splitActs,
            routingWeights: routingBuffer,
            residual: residualBuffer,
            y: splitOutput,
            d: UInt32(dimension),
            f: UInt32(intermediate),
            topK: UInt32(topK),
            expertBits: 3)

        command.commit()
        command.waitUntilCompleted()
        #expect(command.error == nil)
        split.commit()
        split.waitUntilCompleted()
        #expect(split.error == nil)

        let actual = Fp16Buffer.read(output, count: dimension)
        #expect(RelError.compute(actual: actual, reference: expected)
            < Tolerance.fp16ChainedReduction)
        let splitActual = Fp16Buffer.read(splitOutput, count: dimension)
        #expect(splitActual == actual)
    }
}

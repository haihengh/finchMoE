import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// The int8 KV writer: `kv_quantize_int8_rows`.
///
/// The contract the attention readers depend on is narrow and worth pinning:
/// every element dequantizes to within half a step of the block's own scale,
/// the scale is the block absmax / 127, and an all-zero block yields scale 0
/// rather than a NaN from dividing by it.
@Suite struct KVQuantizeTests {
    private static let blockElements = 64

    /// Quantizes on the GPU, then dequantizes on the CPU exactly the way the
    /// attention kernels do (`float(char) * float(scale)`).
    private func roundTrip(rows: [[Float]]) throws -> (dequantized: [Float], scales: [Float]) {
        let context = try MetalContext()
        let quantize = try KVQuantize(context: context)
        let rowDim: Int = try #require(rows.first?.count)
        let blocksPerRow: Int = rowDim / Self.blockElements
        let flat: [Float] = rows.flatMap { $0 }

        let halves: [Float16] = flat.map { Float16($0) }
        let source = try #require(context.device.makeBuffer(
            bytes: halves,
            length: halves.count * MemoryLayout<Float16>.size,
            options: .storageModeShared))
        let destination = try #require(context.device.makeBuffer(
            length: flat.count,
            options: .storageModeShared))
        let scaleBytes: Int = rows.count * blocksPerRow * MemoryLayout<Float16>.size
        let scales = try #require(context.device.makeBuffer(
            length: scaleBytes,
            options: .storageModeShared))

        let cb = try #require(context.queue.makeCommandBuffer())
        quantize.encode(commandBuffer: cb,
                        source: source, sourceOffset: 0,
                        rows: rows.count,
                        destination: destination, destinationOffset: 0,
                        scales: scales, scalesOffset: 0,
                        rowDim: rowDim)
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)

        let quantized = destination.contents().bindMemory(to: Int8.self, capacity: flat.count)
        let scaleValues = scales.contents().bindMemory(to: Float16.self,
                                                       capacity: rows.count * blocksPerRow)
        var step = [Float](repeating: 0, count: rows.count * blocksPerRow)
        var dequantized = [Float](repeating: 0, count: flat.count)
        for index in 0..<flat.count {
            let block: Int = index / Self.blockElements
            let scale: Float = Float(scaleValues[block])
            step[block] = scale
            dequantized[index] = Float(quantized[index]) * scale
        }
        return (dequantized, step)
    }

    @Test(arguments: [512, 1024])
    func rowsQuantizeWithinHalfAStepOfTheirBlockScale(rowDim: Int) throws {
        var rng = SeedTree(0x8B).key("row-\(rowDim)")
        let rows: [[Float]] = (0..<3).map { _ in
            (0..<rowDim).map { _ in rng.uniform(-0.4, 0.4) }
        }

        let stored: [[Float]] = rows.map { $0.map { Float(Float16($0)) } }
        let result = try roundTrip(rows: rows)
        let blockCount: Int = rowDim / Self.blockElements

        for row in 0..<rows.count {
            for block in 0..<blockCount {
                let rowBase: Int = row * rowDim
                let start: Int = rowBase + block * Self.blockElements
                let end: Int = start + Self.blockElements
                var absmax: Float = 0
                for index in start..<end {
                    absmax = max(absmax, abs(stored[row][index - rowBase]))
                }
                let scale: Float = result.scales[row * blockCount + block]
                let expected: Float = absmax / 127
                #expect(abs(scale - expected) <= expected * 0.01 + 1e-7)

                // Half a step from rounding to int8, plus the fp16 scale's own
                // relative error applied to the magnitude being scaled.
                for index in start..<end {
                    let value: Float = stored[row][index - rowBase]
                    let tolerance: Float = scale / 2 + abs(value) * 0.002 + 1e-6
                    let error: Float = abs(result.dequantized[index] - value)
                    #expect(error <= tolerance)
                }
            }
        }
    }

    @Test func anAllZeroBlockStoresAZeroScaleAndReadsBackAsZero() throws {
        let rowDim = 512
        var row = [Float](repeating: 0, count: rowDim)
        row[0] = 0.25
        row[rowDim - 1] = -0.125

        let result = try roundTrip(rows: [row])

        #expect(result.scales[1] == 0)
        #expect(result.dequantized[rowDim / 2] == 0)
        let firstBlockError: Float = abs(result.dequantized[0] - Float(Float16(0.25)))
        #expect(firstBlockError <= result.scales[0] / 2 + 1e-6)
    }

    @Test func aQuietBlockKeepsItsResolutionBesideALoudOne() throws {
        // Two blocks whose magnitudes differ by 1000x: the quiet one must not
        // be crushed to zero, which is the whole reason the store carries a
        // scale per 64 elements rather than one per row.
        let rowDim = 512
        var row = [Float](repeating: 0, count: rowDim)
        for i in 0..<64 { row[i] = Float(Float16(0.001 * Float(i + 1))) }
        for i in 64..<128 { row[i] = 1.0 }

        let result = try roundTrip(rows: [row])

        var maxRelativeError: Float = 0
        for i in 0..<64 {
            let expected: Float = 0.001 * Float(i + 1)
            let error: Float = abs(result.dequantized[i] - expected) / expected
            maxRelativeError = max(maxRelativeError, error)
        }
        #expect(maxRelativeError < 0.01)
    }
}

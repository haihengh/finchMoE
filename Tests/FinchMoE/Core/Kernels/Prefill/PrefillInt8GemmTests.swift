import Foundation
import Metal
import Testing

@testable import FinchMoE

/// The batched int8 projection replaces a per-token GEMV that had a numerical
/// oracle behind it. This is that oracle again: the same affine dequantization
/// (`w = q * scale + bias`, one pair per row per 64-wide K group) accumulated in
/// the same order the GEMV used, so any difference is the kernel's and not the
/// math's. The tolerance is the reassociation slack — the batched kernel sums
/// the K axis per element with fp32 accumulation, while the GEMV sums in fp32
/// after a per-group split, so the two differ in the last bits of a
/// two-thousand-term sum, not in its value.
@Suite struct PrefillInt8GemmTests {
    static let groupSize = 64

    /// Deterministic pseudo-random bytes, so a failure is reproducible.
    static func fill(_ buf: MTLBuffer, count: Int, seed: UInt64,
                     asInt8: Bool = false, range: Range<Float>? = nil) {
        let ptr = buf.contents().bindMemory(to: UInt8.self, capacity: count)
        var s = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        for i in 0..<count {
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let r = UInt8((s >> 33) & 0xFF)
            if asInt8 { ptr[i] = r }
            else { ptr[i] = r }
        }
    }

    static func halfBuffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        let buf = try #require(device.makeBuffer(length: values.count * 2,
                                                 options: .storageModeShared))
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: values.count)
        for (i, v) in values.enumerated() { ptr[i] = Float16(v) }
        return buf
    }

    static func bf16Buffer(_ device: MTLDevice, _ values: [Float]) throws -> MTLBuffer {
        let buf = try #require(device.makeBuffer(length: values.count * 2,
                                                 options: .storageModeShared))
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: values.count)
        for (i, v) in values.enumerated() {
            // bf16 is the top half of an fp32, rounded to nearest even.
            let bits = v.bitPattern
            let rounded = bits &+ 0x7FFF &+ ((bits >> 16) & 1)
            ptr[i] = UInt16(rounded >> 16)
        }
        return buf
    }

    /// Reads bf16 values back the way the kernel sees them.
    ///
    /// A bfloat is the *top* half of an fp32, not a Float16: reconstructing it
    /// as one produces values off by a factor of thousands, which reads exactly
    /// like a kernel that computed nonsense. It cost a round of that here.
    static func readBF16(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: UInt16.self, capacity: count)
        return (0..<count).map { Float(bitPattern: UInt32(ptr[$0]) << 16) }
    }

    /// And fp16 values, likewise: the reference has to start from the same
    /// bytes, or the comparison charges the kernel for the storage formats.
    static func readFP16(_ buf: MTLBuffer, _ count: Int) -> [Float] {
        let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
        return (0..<count).map { Float(ptr[$0]) }
    }

    /// Reproduces `dequant_int8_gemv_simd`'s arithmetic on the host: per group,
    /// `acc += s * (q . x) + b * sum(x)`, accumulated across groups in fp32.
    static func gemvReference(weights: [UInt8], scales: [Float], biases: [Float],
                              x: [Float], rows: Int, columns: Int) -> [Float] {
        let groups = columns / groupSize
        var out = [Float](repeating: 0, count: rows)
        for m in 0..<rows {
            var acc: Float = 0
            for g in 0..<groups {
                var dot: Float = 0
                var sumx: Float = 0
                for i in 0..<groupSize {
                    let k = g * groupSize + i
                    let q = Float(weights[m * columns + k])
                    let xv = x[k]
                    dot += q * xv
                    sumx += xv
                }
                acc += scales[m * groups + g] * dot
                acc += biases[m * groups + g] * sumx
            }
            out[m] = acc
        }
        return out
    }

    @Test func batchedMatchesTheGemvsArithmetic() throws {
        let ctx = try MetalContext()
        let gemm = try PrefillInt8Gemm(context: ctx)
        let rows = 64, columns = 128, tokens = 32
        let groups = columns / Self.groupSize

        var w = [UInt8](repeating: 0, count: rows * columns)
        var s = [Float](repeating: 0, count: rows * groups)
        var b = [Float](repeating: 0, count: rows * groups)
        var x = [Float](repeating: 0, count: tokens * columns)
        var rs: UInt64 = 12_345
        func next() -> Float {
            rs = rs &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float((rs >> 40) & 0x3FF) / 512.0 - 1.0
        }
        for i in 0..<w.count { w[i] = UInt8(Int((next() + 1.0) * 60.0)) }
        for i in 0..<s.count { s[i] = next() * 0.4 }
        for i in 0..<b.count { b[i] = next() * 0.4 }
        for i in 0..<x.count { x[i] = next() }

        let wBuf = try #require(ctx.device.makeBuffer(length: w.count,
                                                      options: .storageModeShared))
        memcpy(wBuf.contents(), w, w.count)
        let sBuf = try Self.bf16Buffer(ctx.device, s)
        let bBuf = try Self.bf16Buffer(ctx.device, b)
        let xBuf = try Self.halfBuffer(ctx.device, x)
        let yBuf = try #require(ctx.device.makeBuffer(length: tokens * rows * 2,
                                                      options: .storageModeShared))
        memset(yBuf.contents(), 0, tokens * rows * 2)

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("no command buffer"); return
        }
        gemm.encode(commandBuffer: cb,
                    weights: wBuf, weightsOffset: 0,
                    scales: sBuf, scalesOffset: 0,
                    biases: bBuf, biasesOffset: 0,
                    x: xBuf, xOffset: 0, y: yBuf, yOffset: 0,
                    tokens: tokens, rows: rows, columns: columns)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { Issue.record("kernel error: \(error)"); return }

        let y = yBuf.contents().bindMemory(to: Float16.self, capacity: tokens * rows)
        // The reference starts from the bytes the kernel reads, so what is left
        // is the accumulation order and the single fp16 store: an fp32 sum over
        // 128 terms in a different order, then rounded once. One fp16 ulp is
        // 2^-11 relative, so the band is that plus slack.
        let sRead = Self.readBF16(sBuf, rows * groups)
        let bRead = Self.readBF16(bBuf, rows * groups)
        let xRead = Self.readFP16(xBuf, tokens * columns)
        var worst: Float = 0
        var worstAt = (0, 0)
        var bad = 0
        var samples: [String] = []
        for t in 0..<tokens {
            let row = Array(xRead[(t * columns)..<((t + 1) * columns)])
            let want = Self.gemvReference(weights: w, scales: sRead, biases: bRead,
                                          x: row, rows: rows, columns: columns)
            for n in 0..<rows {
                let got = Float(y[t * rows + n])
                let delta = abs(got - want[n])
                let tol = max(1e-4, abs(want[n]) * 2e-3)
                if delta > tol {
                    bad += 1
                    if samples.count < 12 {
                        samples.append("t\(t) n\(n): got \(got) want \(want[n])")
                    }
                    if delta > worst { worst = delta; worstAt = (t, n) }
                }
            }
        }
        // The coordinates are the diagnosis: which tokens and rows move says
        // whether the tile mapping, the shared-memory stride or the guards are
        // wrong.
        #expect(bad == 0, "worst \(worst) at token \(worstAt.0) row \(worstAt.1); \(bad) bad. First: \(samples.joined(separator: " | "))")
    }

    /// The tile edges are masked in-kernel, so shapes that do not divide the
    /// tile must still be correct — and unwritten output must stay untouched.
    @Test func raggedShapesAreCorrectAndDoNotOverwrite() throws {
        let ctx = try MetalContext()
        let gemm = try PrefillInt8Gemm(context: ctx)
        let rows = 100, columns = 64, tokens = 40   // neither divides its tile
        let groups = columns / Self.groupSize

        var w = [UInt8](repeating: 0, count: rows * columns)
        var s = [Float](repeating: 0, count: rows * groups)
        var b = [Float](repeating: 0, count: rows * groups)
        var x = [Float](repeating: 0, count: tokens * columns)
        var rs: UInt64 = 999
        for i in 0..<w.count { rs = rs &* 6364136223846793005 &+ 1442695040888963407
                              w[i] = UInt8((rs >> 30) & 0x7F) }
        for i in 0..<s.count { s[i] = 0.25 }
        for i in 0..<b.count { b[i] = 0.125 }
        for i in 0..<x.count { rs = rs &* 6364136223846793005 &+ 1442695040888963407
                               x[i] = Float((rs >> 40) & 0xFF) / 255.0 }

        let wBuf = try #require(ctx.device.makeBuffer(length: w.count,
                                                      options: .storageModeShared))
        memcpy(wBuf.contents(), w, w.count)
        let sBuf = try Self.bf16Buffer(ctx.device, s)
        let bBuf = try Self.bf16Buffer(ctx.device, b)
        let xBuf = try Self.halfBuffer(ctx.device, x)
        // Sentinel-filled, and deliberately longer than the output: a kernel
        // that ignored T or M would write past it and this would catch that.
        let yCount = tokens * rows + 256
        let yBuf = try #require(ctx.device.makeBuffer(length: yCount * 2,
                                                      options: .storageModeShared))
        let yPtr = yBuf.contents().bindMemory(to: Float16.self, capacity: yCount)
        for i in 0..<yCount { yPtr[i] = Float16(-999) }

        guard let cb = ctx.queue.makeCommandBuffer() else {
            Issue.record("no command buffer"); return
        }
        gemm.encode(commandBuffer: cb,
                    weights: wBuf, weightsOffset: 0, scales: sBuf, scalesOffset: 0,
                    biases: bBuf, biasesOffset: 0, x: xBuf, xOffset: 0,
                    y: yBuf, yOffset: 0,
                    tokens: tokens, rows: rows, columns: columns)
        cb.commit()
        cb.waitUntilCompleted()
        if let error = cb.error { Issue.record("kernel error: \(error)"); return }

        let sRead = Self.readBF16(sBuf, rows * groups)
        let bRead = Self.readBF16(bBuf, rows * groups)
        let xRead = Self.readFP16(xBuf, tokens * columns)
        for t in 0..<tokens {
            let row = Array(xRead[(t * columns)..<((t + 1) * columns)])
            let want = Self.gemvReference(weights: w, scales: sRead, biases: bRead,
                                          x: row, rows: rows, columns: columns)
            for n in 0..<rows {
                let got = Float(yPtr[t * rows + n])
                let tol = max(1e-4, abs(want[n]) * 2e-3)
                #expect(abs(got - want[n]) <= tol,
                        "token \(t) row \(n): \(got) vs \(want[n])")
            }
        }
        for i in (tokens * rows)..<yCount {
            #expect(Float(yPtr[i]) == -999, "wrote past the output at \(i)")
        }
    }
}

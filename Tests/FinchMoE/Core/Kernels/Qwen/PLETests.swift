import Testing
import Foundation
import Metal
@testable import FinchMoE
import FinchMoEValidationSupport

/// Compares the Qwen 3.8 PLE kernels (`Metal/Qwen/ple.metal`) against the fp32
/// `PLERef` committed with M3.3a, plus the hand-computable properties that
/// reference cannot state (a gate of exactly 0.5, the history roll).
///
/// The discipline of the other kernel suites: every input is rounded the way
/// the real path would round it (activations FP16, norm gammas BF16) and the
/// *rounded* values feed the reference, so the comparison isolates the kernel
/// math rather than re-measuring the storage format. The one place the real
/// path also rounds and this cannot is the chained intermediate `gated` — the
/// GPU norm reads an FP16-rounded copy — which is why the end-to-end cases
/// carry `fp16ChainedReduction` rather than `fp16Reduction`.
///
/// Geometry note: the projections (`lc_key`/`lc_value`) and the three grouped
/// norms are *not* these kernels — the projections are the engine's int8 GEMV
/// path and the norms are `hc_grouped_rms`. The end-to-end case therefore sets
/// the projections to identity so the CPU side reproduces them exactly, and
/// drives the GPU from the same vectors the reference derives them from. That
/// leaves the four PLE kernels plus the shared norm as the only machinery
/// under test.
@Suite struct PLETests {

    private static let eps: Float = 1e-6

    // MARK: - Helpers

    private static func bf16(_ rng: inout SplitMix64, _ n: Int,
                             _ lo: Float, _ hi: Float) -> [Float] {
        (0..<n).map { _ in
            Quantization.bf16ToFloat(Quantization.bf16Bits(rng.uniform(lo, hi)))
        }
    }

    private static func bf16Buffer(_ device: MTLDevice, _ values: [Float]) -> MTLBuffer? {
        let bits = values.map { Quantization.bf16Bits($0) }
        guard let buf = device.makeBuffer(length: bits.count * 2,
                                          options: .storageModeShared) else { return nil }
        let p = buf.contents().bindMemory(to: UInt16.self, capacity: bits.count)
        for i in 0..<bits.count { p[i] = bits[i] }
        return buf
    }

    private static func f32Buffer(_ device: MTLDevice, count: Int) -> MTLBuffer? {
        device.makeBuffer(length: count * MemoryLayout<Float>.size,
                          options: .storageModeShared)
    }

    private static func readF32(_ buf: MTLBuffer, count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float.self, capacity: count)
        return (0..<count).map { p[$0] }
    }

    private static func writeF32(_ buf: MTLBuffer, _ values: [Float]) {
        let p = buf.contents().bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
    }

    /// The exact identity matrix the end-to-end case uses for both
    /// projections: `keyProj` is `[n, n]`, `valueProj` is `[m, n]` with a 1.0
    /// on the diagonal and nothing else, so `proj · x` returns `x` (and its
    /// first `m` entries) with no rounding at all on the CPU side.
    private static func identityRows(rows: Int, cols: Int) -> [Float] {
        var m = [Float](repeating: 0, count: rows * cols)
        for r in 0..<min(rows, cols) { m[r * cols + r] = 1 }
        return m
    }

    private static func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }

    // MARK: - End to end

    /// Runs the four PLE kernels (and `hc_grouped_rms` for the three norms) as
    /// the decode step will: `key`/`query` norms → gate → gated value → conv
    /// norm → dilated conv → add into the plane.
    ///
    /// Returns the plane and the new history, both FP16-rounded as stored.
    private static func runChain(
        context: MetalContext, ple: PLE, hc: HyperConnection,
        emb: [Float], plane: [Float],
        keyProj: [Float], valueProj: [Float],
        normKey: [Float], normQuery: [Float], normConv: [Float],
        convWeight: [Float],
        state: [Float],
        streamCount: Int, convKernel: Int, dilation: UInt32
    ) throws -> (plane: [Float], state: [Float]) {
        // The projections, on the CPU. Exact for the identity used here — and
        // in the engine this is the int8 GEMV path, not a PLE kernel.
        let hcDim = plane.count
        let nEmbd = hcDim / streamCount
        let nEmb = emb.count
        var key = [Float](repeating: 0, count: hcDim)
        for c in 0..<hcDim {
            var acc: Float = 0
            for i in 0..<nEmb { acc += keyProj[c * nEmb + i] * emb[i] }
            key[c] = acc
        }
        var value = [Float](repeating: 0, count: nEmbd)
        for d in 0..<nEmbd {
            var acc: Float = 0
            for i in 0..<nEmb { acc += valueProj[d * nEmb + i] * emb[i] }
            value[d] = acc
        }

        let hist = (convKernel - 1) * Int(dilation)
        let dev = context.device
        guard let keyBuf    = Fp16Buffer.make(dev, values: key),
              let planeBuf  = Fp16Buffer.make(dev, values: plane),
              let gKey      = Self.bf16Buffer(dev, normKey),
              let gQuery    = Self.bf16Buffer(dev, normQuery),
              let gConv     = Self.bf16Buffer(dev, normConv),
              let normKeyBuf    = Fp16Buffer.make(dev, count: hcDim),
              let normQueryBuf  = Fp16Buffer.make(dev, count: hcDim),
              let gateBuf       = Self.f32Buffer(dev, count: streamCount),
              let valueBuf      = Fp16Buffer.make(dev, values: value),
              let gatedBuf      = Fp16Buffer.make(dev, count: hcDim),
              let normConvBuf   = Fp16Buffer.make(dev, count: hcDim),
              let weightBuf = Fp16Buffer.make(dev, values: convWeight),
              let stateBuf  = Fp16Buffer.make(dev, values: state),
              let convOut   = Fp16Buffer.make(dev, count: hcDim),
              let newState  = Fp16Buffer.make(dev, count: hist * hcDim)
        else { throw ChainError.allocation }

        guard let cb = context.queue.makeCommandBuffer() else { throw ChainError.allocation }
        hc.encodeGroupedRMS(commandBuffer: cb, x: keyBuf, gamma: gKey,
                            out: normKeyBuf, d: UInt32(nEmbd), hc: UInt32(streamCount), eps: eps)
        hc.encodeGroupedRMS(commandBuffer: cb, x: planeBuf, gamma: gQuery,
                            out: normQueryBuf, d: UInt32(nEmbd), hc: UInt32(streamCount), eps: eps)
        ple.encodeGate(commandBuffer: cb,
                       key: normKeyBuf, query: normQueryBuf, gate: gateBuf,
                       d: UInt32(nEmbd), invSqrtD: 1.0 / Float(nEmbd).squareRoot(),
                       hc: UInt32(streamCount))
        ple.encodeGatedValue(commandBuffer: cb, value: valueBuf, gate: gateBuf,
                             gated: gatedBuf, d: UInt32(nEmbd), hc: UInt32(streamCount))
        hc.encodeGroupedRMS(commandBuffer: cb, x: gatedBuf, gamma: gConv,
                            out: normConvBuf, d: UInt32(nEmbd), hc: UInt32(streamCount), eps: eps)
        ple.encodeConvUpdate(commandBuffer: cb, weight: weightBuf,
                             state: stateBuf, x: normConvBuf,
                             out: convOut, newState: newState,
                             c: UInt32(hcDim), kernel: UInt32(convKernel), dilation: dilation)
        ple.encodePlaneAdd(commandBuffer: cb, plane: planeBuf,
                           gated: gatedBuf, conv: convOut, n: UInt32(hcDim))
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { throw ChainError.commandBuffer(err) }

        return (Fp16Buffer.read(planeBuf, count: hcDim),
                Fp16Buffer.read(newState, count: hist * hcDim))
    }

    private enum ChainError: Error {
        case allocation
        case commandBuffer(Error)
    }

    /// Round-trips every input through the reference the same way the kernel
    /// path does, so the two sides see identical numbers.
    private static func referenceChain(
        emb: [Float], plane: [Float],
        keyProj: [Float], valueProj: [Float],
        normKey: [Float], normQuery: [Float], normConv: [Float],
        convWeight: [Float], state: [Float],
        streamCount: Int, convKernel: Int, dilation: Int
    ) -> (plane: [Float], history: [Float]) {
        PLERef.forward(embedding: emb, plane: plane,
                       keyProj: keyProj, valueProj: valueProj,
                       normKey: normKey, normQuery: normQuery, normConv: normConv,
                       convWeight: convWeight, convHistory: state,
                       streamCount: streamCount,
                       convKernel: convKernel, dilation: dilation,
                       eps: Self.eps)
    }

    @Test("all four kernels reproduce PLERef end to end")
    func chain_matchesRef() throws {
        // hc 4 × 64 = 256 channels keeps the identity projections (256² and
        // 64×256) trivial to build while still exercising multi-threadgroup
        // dispatch on every kernel.
        let streamCount = 4, nEmbd = 64, hcDim = streamCount * nEmbd
        let nEmb = hcDim
        let kernel = 4, dilation = 3
        let hist = (kernel - 1) * dilation

        var rng = SeedTree(0xF01).key("ple-chain")
        let emb = (0..<nEmb).map { _ in Float(Float16(rng.uniform(-2, 2))) }
        let plane = (0..<hcDim).map { _ in Float(Float16(rng.uniform(-2, 2))) }
        let normKey = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normQuery = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normConv = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let convWeight = (0..<hcDim * kernel).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let keyProj = Self.identityRows(rows: hcDim, cols: nEmb)
        let valueProj = Self.identityRows(rows: nEmbd, cols: nEmb)

        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let hc = try HyperConnection(context: ctx)

        // A mid-sequence token: the history is full and nonzero, so every conv
        // tap — including the dilated ones — carries signal.
        let state = (0..<hist * hcDim).map { _ in Float(Float16(rng.uniform(-2, 2))) }

        let got = try Self.runChain(
            context: ctx, ple: ple, hc: hc,
            emb: emb, plane: plane, keyProj: keyProj, valueProj: valueProj,
            normKey: normKey, normQuery: normQuery, normConv: normConv,
            convWeight: convWeight, state: state,
            streamCount: streamCount, convKernel: kernel, dilation: UInt32(dilation))
        let want = Self.referenceChain(
            emb: emb, plane: plane, keyProj: keyProj, valueProj: valueProj,
            normKey: normKey, normQuery: normQuery, normConv: normConv,
            convWeight: convWeight, state: state,
            streamCount: streamCount, convKernel: kernel, dilation: dilation)

        let planeErr = RelError.compute(actual: got.plane, reference: want.plane)
        #expect(planeErr < Tolerance.fp16ChainedReduction,
                "plane rel err \(planeErr) over \(hcDim) channels")
        let stateErr = RelError.compute(actual: got.state, reference: want.history)
        #expect(stateErr < Tolerance.fp16ChainedReduction,
                "rolled history rel err \(stateErr)")
    }

    @Test("streaming the chain token by token stays on the reference")
    func chain_streamingMatchesPerTokenRef() throws {
        // The state carried between dispatches is the whole risk here: a roll
        // that is off by one row still produces a *plausible* plane for this
        // token and only corrupts the next one. Twelve tokens give the shift
        // enough runway to show up.
        let streamCount = 2, nEmbd = 32, hcDim = streamCount * nEmbd
        let nEmb = hcDim
        let kernel = 4, dilation = 3
        let hist = (kernel - 1) * dilation
        let tokens = 12

        var rng = SeedTree(0xF02).key("ple-stream")
        let normKey = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normQuery = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normConv = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let convWeight = (0..<hcDim * kernel).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let keyProj = Self.identityRows(rows: hcDim, cols: nEmb)
        let valueProj = Self.identityRows(rows: nEmbd, cols: nEmb)
        let embs = (0..<tokens).map { _ in
            (0..<nEmb).map { _ in Float(Float16(rng.uniform(-2, 2))) } }
        let planes = (0..<tokens).map { _ in
            (0..<hcDim).map { _ in Float(Float16(rng.uniform(-2, 2))) } }

        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let hc = try HyperConnection(context: ctx)

        // A sequence start: the recurrent row is zero, not partly filled.
        var gpuState = [Float](repeating: 0, count: hist * hcDim)
        var refState = gpuState

        for t in 0..<tokens {
            let got = try Self.runChain(
                context: ctx, ple: ple, hc: hc,
                emb: embs[t], plane: planes[t], keyProj: keyProj, valueProj: valueProj,
                normKey: normKey, normQuery: normQuery, normConv: normConv,
                convWeight: convWeight, state: gpuState,
                streamCount: streamCount, convKernel: kernel, dilation: UInt32(dilation))
            let want = Self.referenceChain(
                emb: embs[t], plane: planes[t], keyProj: keyProj, valueProj: valueProj,
                normKey: normKey, normQuery: normQuery, normConv: normConv,
                convWeight: convWeight, state: refState,
                streamCount: streamCount, convKernel: kernel, dilation: dilation)

            let planeErr = RelError.compute(actual: got.plane, reference: want.plane)
            #expect(planeErr < Tolerance.fp16ChainedReduction,
                    "t=\(t) plane rel err \(planeErr)")
            // The states are compared against the reference's *own* roll, so a
            // shift error is caught here rather than being fed forward.
            let stateErr = RelError.compute(actual: got.state, reference: want.history)
            #expect(stateErr < Tolerance.fp16ChainedReduction,
                    "t=\(t) history rel err \(stateErr)")

            gpuState = got.state
            refState = want.history
        }
    }

    // MARK: - The gate

    @Test("the gate is sigmoid of the signed square root of the scaled dot")
    func gate_signedSquareRoot() throws {
        // Constant inputs make the normed vectors exactly ±gamma: with
        // gamma = 1 and a constant c ≠ 0, rms = |c| and every normed element is
        // sign(c). So key = all +1 against query = all ±1 gives a raw dot of
        // exactly ±d, and a scaled dot of ±√d — with d = 64, ±8.
        //
        // The square root is what this case is for: the dot here is far from
        // the clamp floor, so the gate is sigmoid(±√8) = 0.94419 / 0.05581 and
        // a kernel that fed `s` straight into the sigmoid would land on
        // sigmoid(±8) = 0.99966 / 0.00034. The zero-dot case cannot see the
        // difference — √0 is 0 either way — so this is the only test that pins
        // it.
        let streamCount = 4, d = 64, hcDim = streamCount * d
        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let hc = try HyperConnection(context: ctx)

        let ones = [Float](repeating: 1, count: hcDim)
        let g = Self.bf16Buffer(ctx.device, ones)!

        for querySign in [Float(1), Float(-1)] {
            let keyB   = Fp16Buffer.make(ctx.device, values: ones)!
            let queryB = Fp16Buffer.make(ctx.device, values: [Float](repeating: querySign, count: hcDim))!
            let nk = Fp16Buffer.make(ctx.device, count: hcDim)!
            let nq = Fp16Buffer.make(ctx.device, count: hcDim)!
            let gate = Self.f32Buffer(ctx.device, count: streamCount)!

            let cb = ctx.queue.makeCommandBuffer()!
            hc.encodeGroupedRMS(commandBuffer: cb, x: keyB, gamma: g, out: nk,
                                d: UInt32(d), hc: UInt32(streamCount), eps: Self.eps)
            hc.encodeGroupedRMS(commandBuffer: cb, x: queryB, gamma: g, out: nq,
                                d: UInt32(d), hc: UInt32(streamCount), eps: Self.eps)
            ple.encodeGate(commandBuffer: cb, key: nk, query: nq, gate: gate,
                           d: UInt32(d), invSqrtD: 1.0 / Float(d).squareRoot(),
                           hc: UInt32(streamCount))
            cb.commit()
            cb.waitUntilCompleted()

            // The 1e-6 eps in the RMS shifts the scaled dot off ±8 by ~4e-6,
            // which moves the sigmoid by ~4e-7 — far below the tolerance.
            let scaledDot = querySign * Float(d).squareRoot()          // ±√d
            let want = Self.sigmoid(querySign * scaledDot.magnitude.squareRoot())
            let gates = Self.readF32(gate, count: streamCount)
            for c in 0..<streamCount {
                #expect(abs(gates[c] - want) < 1e-5,
                        "sign \(querySign) stream \(c): \(gates[c]) vs \(want)")
            }
        }
    }

    @Test("a zero dot gives a gate of exactly 0.5, not something near it")
    func gate_zeroDotIsExactlyHalf() throws {
        // The clamp floor keeps √0 finite but must never reach the sigmoid:
        // sgn(0)·√1e-6 is exactly 0, so the gate is exactly 1/(1+exp(0)).
        // This is the property a `max(|s|, 1e-6)`-before-the-sign version
        // would silently lose.
        let streamCount = 4, d = 64, hcDim = streamCount * d
        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let hc = try HyperConnection(context: ctx)

        let g = Self.bf16Buffer(ctx.device, [Float](repeating: 1, count: hcDim))!
        let zeros = [Float](repeating: 0, count: hcDim)
        let query = (0..<hcDim).map { i in Float(Float16(0.3 + Float(i % 7) * 0.1)) }

        let keyB = Fp16Buffer.make(ctx.device, values: zeros)!
        let queryB = Fp16Buffer.make(ctx.device, values: query)!
        let nk = Fp16Buffer.make(ctx.device, count: hcDim)!
        let nq = Fp16Buffer.make(ctx.device, count: hcDim)!
        let gate = Self.f32Buffer(ctx.device, count: streamCount)!

        let cb = ctx.queue.makeCommandBuffer()!
        hc.encodeGroupedRMS(commandBuffer: cb, x: keyB, gamma: g, out: nk,
                            d: UInt32(d), hc: UInt32(streamCount), eps: Self.eps)
        hc.encodeGroupedRMS(commandBuffer: cb, x: queryB, gamma: g, out: nq,
                            d: UInt32(d), hc: UInt32(streamCount), eps: Self.eps)
        ple.encodeGate(commandBuffer: cb, key: nk, query: nq, gate: gate,
                       d: UInt32(d), invSqrtD: 1.0 / Float(d).squareRoot(),
                       hc: UInt32(streamCount))
        cb.commit()
        cb.waitUntilCompleted()

        let gates = Self.readF32(gate, count: streamCount)
        for c in 0..<streamCount {
            #expect(gates[c] == 0.5, "stream \(c) gate is not exactly 0.5")
        }
    }

    // MARK: - The conv

    @Test("the conv reads only the dilated taps and rolls history by one row")
    func conv_dilatedTapsAndRoll() throws {
        // The dilation makes 6 of the 9 history rows invisible to this token's
        // output. A dilation of 1 — the obvious misreading, since every other
        // depthwise conv in this engine has one — would light up all 9, so the
        // per-row sweep is the discriminator.
        let c = 256, kernel = 4, dilation = 3
        let hist = (kernel - 1) * dilation
        var rng = SeedTree(0xF21).key("ple-conv-taps")
        let weight = (0..<c * kernel).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let x = (0..<c).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)

        let weightB = Fp16Buffer.make(ctx.device, values: weight)!
        let xB = Fp16Buffer.make(ctx.device, values: x)!
        let histLen = hist * c

        /// One conv dispatch from `state`; returns (out, newState).
        func run(_ state: [Float]) throws -> ([Float], [Float]) {
            let stateB = Fp16Buffer.make(ctx.device, values: state)!
            let outB = Fp16Buffer.make(ctx.device, count: c)!
            let newB = Fp16Buffer.make(ctx.device, count: histLen)!
            let cb = ctx.queue.makeCommandBuffer()!
            ple.encodeConvUpdate(commandBuffer: cb, weight: weightB, state: stateB, x: xB,
                                 out: outB, newState: newB,
                                 c: UInt32(c), kernel: UInt32(kernel), dilation: UInt32(dilation))
            cb.commit()
            cb.waitUntilCompleted()
            return (Fp16Buffer.read(outB, count: c), Fp16Buffer.read(newB, count: histLen))
        }

        let zeroState = [Float](repeating: 0, count: histLen)
        let baseline = try run(zeroState).0

        for row in 0..<hist {
            var s = zeroState
            let back = hist - row                       // the tap distance
            for ch in 0..<c { s[row * c + ch] = Float(ch % 5) + 1 }
            let out = try run(s).0
            let moved = zip(out, baseline).contains { abs($0 - $1) > 1e-6 }
            let inField = back % dilation == 0 && back < kernel * dilation
            #expect(moved == inField,
                    "history row \(row) (t−\(back)) moved=\(moved), expected \(inField)")
        }

        // The roll: every row shifts up, the tail is this token's normed value,
        // and the row that falls off the end is gone.
        let (_, rolled) = try run(zeroState)
        for ch in 0..<c {
            for row in 0..<(hist - 1) {
                #expect(rolled[row * c + ch] == 0,
                        "row \(row) channel \(ch) should have shifted to zero")
            }
            #expect(rolled[(hist - 1) * c + ch] == x[ch],
                    "the tail row must be this token's normed value")
        }

        // A full state shifts by exactly one row — the case the zero state
        // above cannot distinguish from "write zeros everywhere".
        let full = (0..<histLen).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let (_, shifted) = try run(full)
        for row in 0..<(hist - 1) {
            for ch in 0..<c {
                #expect(shifted[row * c + ch] == full[(row + 1) * c + ch],
                        "row \(row) channel \(ch) did not shift from row \(row + 1)")
            }
        }
    }

    @Test("a channel count that is not a multiple of the threadgroup is exact")
    func conv_partialThreadgroup() throws {
        // `c >= C` returns early, so trailing threads in the last threadgroup
        // do no work. Nothing in this kernel has a barrier after that point —
        // a barrier there would be divergent control flow and undefined.
        let c = 200, kernel = 4, dilation = 3
        let hist = (kernel - 1) * dilation
        var rng = SeedTree(0xF22).key("ple-partial")
        let weight = (0..<c * kernel).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let x = (0..<c).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let state = (0..<hist * c).map { _ in Float(Float16(rng.uniform(-1, 1))) }

        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let weightB = Fp16Buffer.make(ctx.device, values: weight)!
        let stateB = Fp16Buffer.make(ctx.device, values: state)!
        let xB = Fp16Buffer.make(ctx.device, values: x)!
        let outB = Fp16Buffer.make(ctx.device, count: c)!
        let newB = Fp16Buffer.make(ctx.device, count: hist * c)!

        let cb = ctx.queue.makeCommandBuffer()!
        ple.encodeConvUpdate(commandBuffer: cb, weight: weightB, state: stateB, x: xB,
                             out: outB, newState: newB,
                             c: UInt32(c), kernel: UInt32(kernel), dilation: UInt32(dilation))
        cb.commit()
        cb.waitUntilCompleted()

        let got = Fp16Buffer.read(outB, count: c)
        // Direct, from the reference's own tap rule.
        for ch in 0..<c {
            var acc: Float = 0
            for k in 0..<kernel {
                let back = (kernel - 1 - k) * dilation
                let tap = back == 0 ? x[ch] : state[(hist - back) * c + ch]
                acc += weight[ch * kernel + k] * tap
            }
            let want = acc / (1 + exp(-acc))
            #expect(abs(got[ch] - want) < 1e-3, "channel \(ch): \(got[ch]) vs \(want)")
        }
    }

    // MARK: - Real geometry

    @Test("the real decode geometry runs and stays finite")
    func realGeometry_finite() throws {
        // 4 streams × 2560 = 10240 channels, kernel 4 dilation 3 (the config's
        // ngram size), which is the shape every decode step will use. This is
        // a smoke case for the dispatch arithmetic — C/256 lands exactly (40
        // groups) and the buffers are the real sizes.
        let streamCount = 4, nEmbd = 2560, hcDim = streamCount * nEmbd
        let kernel = 4, dilation = 3
        let hist = (kernel - 1) * dilation
        var rng = SeedTree(0xF31).key("ple-real")

        let ctx = try MetalContext()
        let ple = try PLE(context: ctx)
        let hc = try HyperConnection(context: ctx)

        let key = (0..<hcDim).map { _ in Float(Float16(rng.uniform(-2, 2))) }
        let query = (0..<hcDim).map { _ in Float(Float16(rng.uniform(-2, 2))) }
        let normKey = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normQuery = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let normConv = Self.bf16(&rng, hcDim, 0.5, 1.5)
        let convWeight = (0..<hcDim * kernel).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let value = (0..<nEmbd).map { _ in Float(Float16(rng.uniform(-1, 1))) }
        let state = (0..<hist * hcDim).map { _ in Float(Float16(rng.uniform(-1, 1))) }

        let keyB = Fp16Buffer.make(ctx.device, values: key)!
        let queryB = Fp16Buffer.make(ctx.device, values: query)!
        let gK = Self.bf16Buffer(ctx.device, normKey)!
        let gQ = Self.bf16Buffer(ctx.device, normQuery)!
        let gC = Self.bf16Buffer(ctx.device, normConv)!
        let nk = Fp16Buffer.make(ctx.device, count: hcDim)!
        let nq = Fp16Buffer.make(ctx.device, count: hcDim)!
        let gate = Self.f32Buffer(ctx.device, count: streamCount)!
        let valueB = Fp16Buffer.make(ctx.device, values: value)!
        let gatedB = Fp16Buffer.make(ctx.device, count: hcDim)!
        let nc = Fp16Buffer.make(ctx.device, count: hcDim)!
        let weightB = Fp16Buffer.make(ctx.device, values: convWeight)!
        let stateB = Fp16Buffer.make(ctx.device, values: state)!
        let outB = Fp16Buffer.make(ctx.device, count: hcDim)!
        let newB = Fp16Buffer.make(ctx.device, count: hist * hcDim)!
        let planeB = Fp16Buffer.make(ctx.device, values: query)!

        let cb = ctx.queue.makeCommandBuffer()!
        hc.encodeGroupedRMS(commandBuffer: cb, x: keyB, gamma: gK, out: nk,
                            d: UInt32(nEmbd), hc: UInt32(streamCount), eps: Self.eps)
        hc.encodeGroupedRMS(commandBuffer: cb, x: queryB, gamma: gQ, out: nq,
                            d: UInt32(nEmbd), hc: UInt32(streamCount), eps: Self.eps)
        ple.encodeGate(commandBuffer: cb, key: nk, query: nq, gate: gate,
                       d: UInt32(nEmbd), invSqrtD: 1.0 / Float(nEmbd).squareRoot(),
                       hc: UInt32(streamCount))
        ple.encodeGatedValue(commandBuffer: cb, value: valueB, gate: gate,
                             gated: gatedB, d: UInt32(nEmbd), hc: UInt32(streamCount))
        hc.encodeGroupedRMS(commandBuffer: cb, x: gatedB, gamma: gC, out: nc,
                            d: UInt32(nEmbd), hc: UInt32(streamCount), eps: Self.eps)
        ple.encodeConvUpdate(commandBuffer: cb, weight: weightB, state: stateB, x: nc,
                             out: outB, newState: newB,
                             c: UInt32(hcDim), kernel: UInt32(kernel), dilation: UInt32(dilation))
        ple.encodePlaneAdd(commandBuffer: cb, plane: planeB, gated: gatedB,
                           conv: outB, n: UInt32(hcDim))
        cb.commit()
        cb.waitUntilCompleted()
        if let err = cb.error { Issue.record("command buffer failed: \(err)") }

        let gates = Self.readF32(gate, count: streamCount)
        for c in 0..<streamCount {
            #expect(gates[c].isFinite && gates[c] > 0 && gates[c] < 1,
                    "stream \(c) gate out of range: \(gates[c])")
        }
        // The gate is the only place an FP16 overflow could hide (a huge dot
        // would saturate before the sigmoid), so pin the plane as finite too.
        for v in Fp16Buffer.read(planeB, count: hcDim) {
            #expect(v.isFinite, "plane went non-finite")
        }
    }
}

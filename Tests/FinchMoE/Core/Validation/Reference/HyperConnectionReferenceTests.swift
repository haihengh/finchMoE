import Testing
import Foundation
import FinchMoEValidationSupport

/// Cross-validates `HyperConnectionRef` (the Qwen 3.8 hyper-connection
/// grouped RMS / mixer / combine locked in docs/QWEN38_PORT.md
/// "Hyper-connection math (locked)" and `archive/llama.cpp/
/// src/models/qwen4exp.cpp` `build_hc_mix`/`build_hc_combine`) against a
/// second formulation written from scratch as scalar loops. The point isn't
/// to prove the kernel right — it's to prove the *reference* right, so the
/// kernel tests below have a trustworthy comparator.
///
/// The naive formulation shares no code path with the reference and sums
/// every dot in the opposite order (and silu as x·sigmoid(x) instead of
/// x/(1+e^-x)), so agreement is a real check on the math, not on shared
/// code. Both must agree to `Tolerance.identity`.
@Suite struct HyperConnectionReferenceTests {

    /// Independent scalar grouped RMS: sum-of-squares accumulated even/odd
    /// then merged (different tree than the reference's ascending pass), and
    /// the gamma applied as `x · (inv · gamma)` instead of `(x · inv) · gamma`.
    private static func naiveGroupedRMS(
        x: [Float], gamma: [Float], streamCount: Int, eps: Float
    ) -> [Float] {
        precondition(x.count == gamma.count)
        let d = x.count / streamCount
        var y = x
        for c in 0..<streamCount {
            let base = c * d
            var even: Float = 0
            var odd: Float = 0
            for i in 0..<d {
                let v = x[base + i]
                if i & 1 == 0 { even += v * v } else { odd += v * v }
            }
            let inv = 1.0 / ((even + odd) / Float(d) + eps).squareRoot()
            for i in 0..<d {
                y[base + i] = x[base + i] * (inv * gamma[base + i])
            }
        }
        return y
    }

    /// Independent scalar transcription of the doc mix. Same stages, but the
    /// lo/gate/inject dots run in **descending** k order and the mean
    /// accumulates streams in descending order, so any rounding that depends
    /// on summation order would show up between the two implementations.
    private static func naiveMix(
        plane: [Float],
        gamma: [Float],
        down: [Float],
        up: [Float],
        blockInject: [Float]? = nil,
        streamCount: Int,
        lowrank: Int,
        eps: Float
    ) -> (blockInput: [Float], inject: [Float]) {
        let hcDim = plane.count
        let d = hcDim / streamCount
        let xn = naiveGroupedRMS(x: plane, gamma: gamma, streamCount: streamCount, eps: eps)
        let invHc = 1.0 / Float(streamCount)

        var lo = [Float](repeating: 0, count: lowrank)
        for r in 0..<lowrank {
            let row = r * hcDim
            var acc: Float = 0
            for k in (0..<hcDim).reversed() {
                acc += down[row + k] * xn[k]
            }
            let x = acc * invHc                       // ÷ hc BEFORE silu
            lo[r] = x * (1.0 / (1.0 + expf(-x)))      // silu as x·sigmoid(x)
        }

        var gate = [Float](repeating: 0, count: hcDim)
        for i in 0..<hcDim {
            let row = i * lowrank
            var acc: Float = 0
            for r in (0..<lowrank).reversed() {
                acc += up[row + r] * lo[r]
            }
            gate[i] = 1.0 / (1.0 + expf(-acc))
        }

        var blockInput = [Float](repeating: 0, count: d)
        for c in (0..<streamCount).reversed() {
            let base = c * d
            for i in 0..<d {
                blockInput[i] += xn[base + i] * gate[base + i]
            }
        }
        for i in 0..<d {
            blockInput[i] *= invHc
        }

        var inject: [Float] = []
        if let w = blockInject {
            inject = [Float](repeating: 0, count: streamCount)
            for c in (0..<streamCount).reversed() {
                let row = c * hcDim
                var acc: Float = 0
                for k in (0..<hcDim).reversed() {
                    acc += w[row + k] * xn[k]
                }
                inject[c] = acc
            }
        }
        return (blockInput, inject)
    }

    /// Independent scalar combine: doc formula `plane[c·D+d] += b[d]·w[c]`,
    /// written out directly.
    private static func naiveCombine(
        plane: [Float], blockOut: [Float], inject: [Float], streamCount: Int
    ) -> [Float] {
        let d = plane.count / streamCount
        var y = plane
        let invHc = 1.0 / Float(streamCount)
        for c in 0..<streamCount {
            let base = c * d
            let w = 2.0 * (1.0 / (1.0 + expf(-inject[c] * invHc)))
            for i in 0..<d {
                y[base + i] += blockOut[i] * w
            }
        }
        return y
    }

    private static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
    private static func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }

    // Geometries exercised in the parameterized tests: tiny (hand-checkable),
    // the toy-install shape (hc 4 × 256, lowrank 64), and the real model
    // shape (hc 4 × 2560 = 10240 plane, lowrank 320).
    private static func seededMixData(
        hc: Int, d: Int, lr: Int, seed: UInt64
    ) -> (plane: [Float], gamma: [Float], down: [Float], up: [Float], inject: [Float]) {
        var rng = SeedTree(seed).key("hyper-connection-ref")
        let hcDim = hc * d
        let plane = (0..<hcDim).map { _ in rng.uniform(-1.0, 1.0) }
        let gamma = (0..<hcDim).map { _ in rng.uniform(0.5, 1.5) }
        let down = (0..<(lr * hcDim)).map { _ in rng.uniform(-0.25, 0.25) }
        let up = (0..<(hcDim * lr)).map { _ in rng.uniform(-0.25, 0.25) }
        let inject = (0..<(hc * hcDim)).map { _ in rng.uniform(-0.25, 0.25) }
        return (plane, gamma, down, up, inject)
    }

    // MARK: - grouped RMS

    @Test("grouped RMS reduces over each stream separately")
    func groupedRmsIsPerStream() {
        // Two streams: [1, 1] and [100, 100]. A whole-plane reduction would
        // scale both to ~0.0141; per-stream reduction scales each to ~1.
        let x: [Float] = [1, 1, 100, 100]
        let gamma = [Float](repeating: 1, count: 4)
        let eps: Float = 1e-6
        let y = HyperConnectionRef.groupedRMS(x: x, gamma: gamma, streamCount: 2, eps: eps)

        let stream0Factor = 1.0 / (1.0 + eps).squareRoot()    // rms([1,1]) → ~1
        #expect(abs(y[0] - stream0Factor) < 1e-6)
        #expect(abs(y[1] - stream0Factor) < 1e-6)
        #expect(abs(y[2] - 1.0) < 1e-6, "stream [100,100] must self-normalise")
        #expect(abs(y[3] - 1.0) < 1e-6)
    }

    @Test("grouped RMS matches an independent scalar formulation", arguments: [
        (2, 4, UInt64(0xB1)),       // tiny
        (4, 256, UInt64(0xB2)),     // toy geometry
        (4, 2560, UInt64(0xB3)),    // real geometry
    ])
    func groupedRmsMatchesNaive(hc: Int, d: Int, seed: UInt64) {
        var rng = SeedTree(seed).key("grouped-rms")
        let x = (0..<(hc * d)).map { _ in rng.uniform(-1.0, 1.0) }
        let gamma = (0..<(hc * d)).map { _ in rng.uniform(0.5, 1.5) }
        let eps: Float = 1e-6

        let ref = HyperConnectionRef.groupedRMS(x: x, gamma: gamma, streamCount: hc, eps: eps)
        let naive = Self.naiveGroupedRMS(x: x, gamma: gamma, streamCount: hc, eps: eps)
        let relErr = RelError.compute(actual: ref, reference: naive)
        #expect(relErr < Tolerance.identity, "hc=\(hc) d=\(d): relErr=\(relErr)")
    }

    // MARK: - mix + combine

    @Test("mix + combine match a doc-faithful scalar transcription", arguments: [
        (2, 4, 3, UInt64(0xA1)),       // tiny
        (4, 256, 64, UInt64(0xA2)),    // toy-install geometry
        (4, 2560, 320, UInt64(0xA3)),  // real-model geometry
    ])
    func mixAndCombineMatchNaive(hc: Int, d: Int, lr: Int, seed: UInt64) {
        let data = Self.seededMixData(hc: hc, d: d, lr: lr, seed: seed)
        let eps: Float = HyperConnectionRef.rmsEps

        let (blockInput, inject) = HyperConnectionRef.mix(
            plane: data.plane, gamma: data.gamma, down: data.down, up: data.up,
            blockInject: data.inject, streamCount: hc, lowrank: lr, eps: eps)
        let (nBlock, nInject) = Self.naiveMix(
            plane: data.plane, gamma: data.gamma, down: data.down, up: data.up,
            blockInject: data.inject, streamCount: hc, lowrank: lr, eps: eps)

        let blockRel = RelError.compute(actual: blockInput, reference: nBlock)
        let injectRel = RelError.compute(actual: inject, reference: nInject)
        #expect(blockRel < Tolerance.identity,
                "hc=\(hc) d=\(d): blockInput relErr=\(blockRel)")
        #expect(injectRel < Tolerance.identity,
                "hc=\(hc) d=\(d): inject relErr=\(injectRel)")

        var rng = SeedTree(seed).key("combine")
        let blockOut = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let combined = HyperConnectionRef.combine(
            plane: data.plane, blockOut: blockOut, inject: inject, streamCount: hc)
        let nCombined = Self.naiveCombine(
            plane: data.plane, blockOut: blockOut, inject: nInject, streamCount: hc)
        let combRel = RelError.compute(actual: combined, reference: nCombined)
        #expect(combRel < Tolerance.identity, "hc=\(hc) d=\(d): combine relErr=\(combRel)")
    }

    @Test("zero inject is a plain residual add (2·sigmoid(0) = 1)")
    func zeroInjectIsPlainAdd() {
        let hc = 4, d = 256
        var rng = SeedTree(0xC1).key("plain-add")
        let plane = (0..<(hc * d)).map { _ in rng.uniform(-1.0, 1.0) }
        let blockOut = (0..<d).map { _ in rng.uniform(-1.0, 1.0) }
        let zero = [Float](repeating: 0, count: hc)

        let y = HyperConnectionRef.combine(
            plane: plane, blockOut: blockOut, inject: zero, streamCount: hc)

        // Per-stream scatter weight w[c] = 2·sigmoid(0/hc) = 1 exactly, so
        // y[c·D+d] == plane[c·D+d] + blockOut[d] exactly.
        var expected = plane
        for c in 0..<hc {
            let base = c * d
            for i in 0..<d {
                expected[base + i] += blockOut[i]
            }
        }
        let relErr = RelError.compute(actual: y, reference: expected)
        #expect(relErr < 1e-6, "relErr=\(relErr)")
    }

    @Test("block input never sees the inject weights; the root mixer has none")
    func blockInputIgnoresInject() {
        let hc = 4, d = 256, lr = 64
        let data = Self.seededMixData(hc: hc, d: d, lr: lr, seed: 0xC2)
        let eps: Float = HyperConnectionRef.rmsEps

        let withInject = HyperConnectionRef.mix(
            plane: data.plane, gamma: data.gamma, down: data.down, up: data.up,
            blockInject: data.inject, streamCount: hc, lowrank: lr, eps: eps)
        let root = HyperConnectionRef.mix(
            plane: data.plane, gamma: data.gamma, down: data.down, up: data.up,
            blockInject: nil, streamCount: hc, lowrank: lr, eps: eps)

        // Root hyper_connection_mixer = the same mixer without inject: the
        // collapsed block input feeds lm_head directly.
        #expect(root.inject.isEmpty)
        #expect(withInject.inject.count == hc)
        let relErr = RelError.compute(actual: root.blockInput, reference: withInject.blockInput)
        #expect(relErr < Tolerance.identity, "blockInput must not depend on block_inject: \(relErr)")
    }

    // MARK: - stage-order locks (doc transcription on hand-written numbers)

    @Test("÷hc precedes silu; the gate is a sigmoid; inject carries no activation")
    func docStageOrderLocked() {
        // hc = 2, stream width 2, lowrank 1: the whole mixer collapses to
        // scalar gates. Plane streams [1, 1] and [1, 64] (unit gamma).
        let hc = 2, d = 2, lr = 1
        let plane: [Float] = [1, 1, 1, 64]
        let gamma = [Float](repeating: 1, count: 4)
        let down: [Float] = [0, 0, 0, 1]        // lo = xn[3]
        let up: [Float] = [0.25, 0.5, 1.0, 2.0] // gate z_i = up[i]·lo
        // [streamCount, hcDim] = [2, 4] row-major: row 0 picks xn[0] and
        // xn[2]; row 1 is all zero (root-like, exercises the nil-absent dot).
        let injectW: [Float] = [0.25, 0, 0.25, 0, 0, 0, 0, 0]
        let eps: Float = 1e-6

        // xn transcribed from the doc: per-stream rms over [1,1] and [1,64].
        let inv0 = 1.0 / (((1.0 * 1.0 + 1.0 * 1.0) / 2.0) + eps).squareRoot()
        let inv1 = 1.0 / (((1.0 + 64.0 * 64.0) / 2.0) + eps).squareRoot()
        let xn: [Float] = [inv0, inv0, inv1, 64 * inv1]

        // A doc-faithful mini-mixer: activateLo sees the ÷hc already applied,
        // gate activation applied to up·lo, mean over the hc streams.
        func miniMix(
            loActivate: (Float) -> Float,
            gateActivate: (Float) -> Float
        ) -> [Float] {
            let loAct = loActivate(xn[3])               // lo = down·xn = xn[3]
            var gate = [Float](repeating: 0, count: 4)
            for i in 0..<4 {
                gate[i] = gateActivate(up[i] * loAct)
            }
            var out = [Float](repeating: 0, count: d)
            for c in 0..<hc {
                for i in 0..<d {
                    out[i] += xn[c * d + i] * gate[c * d + i]
                }
            }
            let invHc = 1.0 / Float(hc)
            for i in 0..<d {
                out[i] *= invHc
            }
            return out
        }

        let docOrder = miniMix(loActivate: { Self.silu($0 / 2) }, gateActivate: Self.sigmoid)
        let (blockInput, inject) = HyperConnectionRef.mix(
            plane: plane, gamma: gamma, down: down, up: up,
            blockInject: injectW, streamCount: hc, lowrank: lr, eps: eps)

        let docRel = RelError.compute(actual: blockInput, reference: docOrder)
        #expect(docRel < Tolerance.identity, "doc transcription mismatch: \(docRel)")

        // Inject transcribed from the doc: raw row dot with xn (no activation).
        let expectedInject: [Float] = [
            0.25 * xn[0] + 0.25 * xn[2],   // row 0 of injectW
            0,                             // row 1 is all zero
        ]
        let injRel = RelError.compute(actual: inject, reference: expectedInject)
        #expect(injRel < Tolerance.identity, "inject transcription mismatch: \(injRel)")

        // Wrong orderings must NOT match: each is a plausible bug, and each
        // lands > 1e-2 away from the reference while the correct order is
        // within Tolerance.identity.
        let siluBeforeDivide = miniMix(loActivate: { Self.silu($0) / 2 }, gateActivate: Self.sigmoid)
        let siluGate = miniMix(loActivate: { Self.silu($0 / 2) }, gateActivate: Self.silu)

        let wrongDivide = RelError.compute(actual: blockInput, reference: siluBeforeDivide)
        let wrongGate = RelError.compute(actual: blockInput, reference: siluGate)
        #expect(wrongDivide > 1e-2, "trap: ÷hc must sit BEFORE the silu (relErr=\(wrongDivide))")
        #expect(wrongGate > 1e-2, "trap: the read gate must be a sigmoid (relErr=\(wrongGate))")

        // An activated inject would move the scatter weight w[c] away from
        // the raw-dot value — lock inject as a raw dot.
        let wrongInject = 2.0 * Self.silu(expectedInject[0] / 2)
        let wRel = abs(wrongInject - expectedInject[0]) / max(abs(expectedInject[0]), 1e-6)
        #expect(wRel > 1e-2, "trap: inject must carry no activation (relErr=\(wRel))")
    }
}

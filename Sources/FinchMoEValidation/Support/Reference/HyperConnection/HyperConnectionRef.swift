import Foundation

/// FP32 reference for the Qwen 3.8 Flash-Next hyper-connection mixer,
/// grounded in `archive/llama.cpp/src/models/qwen4exp.cpp` `build_hc_mix`
/// (:218-264), `build_hc_combine` (:266-286) and the terminal root mixer
/// (:380-393). The locked math lives in `docs/QWEN38_PORT.md`
/// "Hyper-connection math (locked)".
///
/// The residual is `streamCount` parallel planes of width
/// `x.count / streamCount` laid out contiguously — one 10240-wide plane per
/// token when streamCount = 4 and the hidden width is 2560. Per mixer:
///
/// ```text
/// xn          = grouped RMS over each stream, scaled by gamma (plain scale)
/// lo          = silu( down · xn · 1/streamCount )   // ÷ hc BEFORE silu
/// gate        = sigmoid( up · lo )
/// blockInput  = mean over streams of (xn · gate)
/// inject[c]   = blockInject[c] · xn                 // no activation
/// ```
///
/// and the combine re-scatters the block output into the plane:
/// `plane[c·D + d] += blockOut[d] · 2·sigmoid(inject[c] · 1/streamCount)`
/// (2·sigmoid centres the scatter weights on 1, `:274-275`). The root
/// `hyper_connection_mixer` is the same mix **without** blockInject: its
/// collapsed blockInput feeds lm_head directly — there is no `model.norm`.
///
/// Every step is pure fp32 and written as an explicit scalar loop — a
/// deliberately different op-tree from the Metal kernels (no SIMD group
/// reductions, no vectorized dot products), which is what makes the
/// kernel-vs-reference comparison meaningful. `down`/`up`/`blockInject`
/// arrive row-major ([out, in], the HF checkpoint orientation the repack
/// keeps: down [lowrank, hcDim], up [hcDim, lowrank],
/// block_inject [streamCount, hcDim]).
public enum HyperConnectionRef {
    /// Model RMS epsilon — the same hyperparameter the GDN norms use
    /// (`f_norm_rms_eps`, qwen4exp.cpp:232; the real config is 1e-6).
    public static let rmsEps: Float = 1e-6

    /// Grouped RMSNorm: each of `streamCount` consecutive planes of width
    /// `x.count / streamCount` is normalized by its own mean square, then
    /// every element is scaled by `gamma` (length == `x.count`). The gamma
    /// is a plain multiplier — the HC norms are never 1+w-baked (only the
    /// five zero-centred gammas are, `qwen4exp.py:134-136`); a baked gamma
    /// would arrive here pre-summed the same way.
    public static func groupedRMS(
        x: [Float], gamma: [Float], streamCount: Int, eps: Float
    ) -> [Float] {
        precondition(x.count == gamma.count, "x and gamma must match length")
        precondition(streamCount > 0 && x.count % streamCount == 0,
                     "plane must split into whole streams")
        let d = x.count / streamCount
        var y = [Float](repeating: 0, count: x.count)
        for c in 0..<streamCount {
            let base = c * d
            var sumSq: Float = 0
            for i in 0..<d {
                let v = x[base + i]
                sumSq += v * v
            }
            let inv = 1.0 / (sumSq / Float(d) + eps).squareRoot()
            for i in 0..<d {
                y[base + i] = x[base + i] * inv * gamma[base + i]
            }
        }
        return y
    }

    /// One hyper-connection mix (`build_hc_mix`, qwen4exp.cpp:218-264):
    /// block input = mean over the streams of the gated, gamma-scaled plane;
    /// `inject[c]` = row c of `blockInject` dotted with `xn` when present,
    /// empty at the root (`w_inject` nullptr, `:382`).
    public static func mix(
        plane: [Float],
        gamma: [Float],
        down: [Float],
        up: [Float],
        blockInject: [Float]? = nil,
        streamCount: Int,
        lowrank: Int,
        eps: Float = HyperConnectionRef.rmsEps
    ) -> (blockInput: [Float], inject: [Float]) {
        let hcDim = plane.count
        precondition(streamCount > 0 && hcDim % streamCount == 0,
                     "plane must split into whole streams")
        let d = hcDim / streamCount
        precondition(gamma.count == hcDim, "gamma must be one entry per plane element")
        precondition(down.count == lowrank * hcDim, "down must be [lowrank, hcDim] row-major")
        precondition(up.count == hcDim * lowrank, "up must be [hcDim, lowrank] row-major")
        if let w = blockInject {
            precondition(w.count == streamCount * hcDim,
                         "block_inject must be [streamCount, hcDim] row-major")
        }

        // xn: grouped RMS over each stream, scaled by the [hcDim] gamma
        // (plain scale: rms per stream of width d, then gamma over the whole
        // plane — qwen4exp.cpp:230-234).
        let xn = groupedRMS(x: plane, gamma: gamma, streamCount: streamCount, eps: eps)

        // lo = silu(down · xn · 1/hc): the ÷hc sits BEFORE the silu (:238).
        var lo = [Float](repeating: 0, count: lowrank)
        let invHc = 1.0 / Float(streamCount)
        for r in 0..<lowrank {
            let row = r * hcDim
            var acc: Float = 0
            for k in 0..<hcDim {
                acc += down[row + k] * xn[k]
            }
            lo[r] = silu(acc * invHc)
        }

        // gate = sigmoid(up · lo) (:239), then gated = xn · gate (:242).
        var gate = [Float](repeating: 0, count: hcDim)
        for i in 0..<hcDim {
            let row = i * lowrank
            var acc: Float = 0
            for r in 0..<lowrank {
                acc += up[row + r] * lo[r]
            }
            gate[i] = sigmoid(acc)
        }

        // blockInput[d] = mean over the streams of gated (:242-255).
        // Accumulated stream-by-stream, then scaled by 1/hc (:255).
        var blockInput = [Float](repeating: 0, count: d)
        for c in 0..<streamCount {
            let base = c * d
            for i in 0..<d {
                blockInput[i] += xn[base + i] * gate[base + i]
            }
        }
        for i in 0..<d {
            blockInput[i] *= invHc
        }

        // inject[c] = row c of block_inject dotted with xn, no activation
        // (:258-260). Absent at the root mixer.
        var inject: [Float] = []
        if let w = blockInject {
            inject = [Float](repeating: 0, count: streamCount)
            for c in 0..<streamCount {
                let row = c * hcDim
                var acc: Float = 0
                for k in 0..<hcDim {
                    acc += w[row + k] * xn[k]
                }
                inject[c] = acc
            }
        }
        return (blockInput, inject)
    }

    /// One hyper-connection combine (`build_hc_combine`, qwen4exp.cpp:266-286):
    /// `plane[c·D + d] += blockOut[d] · 2·sigmoid(inject[c] · 1/hc)`. The
    /// coefficient is per stream, so a zero injection is a plain residual add.
    public static func combine(
        plane: [Float],
        blockOut: [Float],
        inject: [Float],
        streamCount: Int
    ) -> [Float] {
        precondition(streamCount > 0 && plane.count % streamCount == 0,
                     "plane must split into whole streams")
        let d = plane.count / streamCount
        precondition(blockOut.count == d, "blockOut must be one stream wide")
        precondition(inject.count == streamCount, "one scatter weight per stream")

        var y = plane
        let invHc = 1.0 / Float(streamCount)
        for c in 0..<streamCount {
            let base = c * d
            let w = 2.0 * sigmoid(inject[c] * invHc)   // sigmoid then ·2 (:275-276)
            for i in 0..<d {
                y[base + i] += blockOut[i] * w
            }
        }
        return y
    }

    /// `x / (1 + e^-x)`, the same expression GDNRef uses for the GDN silu.
    static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }

    /// `1 / (1 + e^-x)`, the GDNRef gate β form.
    static func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }
}

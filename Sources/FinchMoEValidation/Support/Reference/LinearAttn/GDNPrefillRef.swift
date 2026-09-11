import Foundation
import FinchMoE

/// fp32 CPU reference for the GDN (gated-delta-net) prefill unit — the same
/// sequential per-token recurrence as `GDNRef`, batched over the chunk's T
/// tokens. Grounded in `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:
/// `torch_causal_conv1d_fn` (prefill branch), `l2norm`,
/// `torch_recurrent_gated_delta_rule`, and `Qwen3_5MoeRMSNormGated`.
///
/// Layouts mirror the Metal kernels:
///   conv input/output: flat `[T][C]` (the fused [q|k|v] block per token)
///   conv w `[C,4]`, conv state `[C,3]`
///   ab `[T][2V]` (a rows then b rows per token), g/beta `[T][V]` fp32
///   out `[T][V][D]`
///
/// Every step is a naive fp32 loop (or reuses `GDNRef`'s naive loops) — a
/// deliberately different op-tree from the Metal two-stage block reductions.
public enum GDNPrefillRef {

    /// `torch_causal_conv1d_fn` at prefill (kernel 4): for each (t, c),
    /// `out[t][c] = silu(w0*x[t-3][c] + w1*x[t-2][c] + w2*x[t-1][c] + w3*x[t][c])`,
    /// with taps before the chunk seeded from `state` (`x[idx] = state[3+idx]`
    /// for idx < 0). The post-chunk state is the last 3 raw inputs:
    /// `newState[j] = x[T-3+j]`, negative indices seeded from `state` — valid
    /// for every T (1, 2, or larger). `x` is flat `[T][C]`.
    public static func convChunk(
        w: [Float], state: [Float], x: [Float], channels: Int, tokens: Int
    ) -> (out: [Float], newState: [Float]) {
        var out = [Float](repeating: 0, count: tokens * channels)
        for t in 0..<tokens {
            for c in 0..<channels {
                var acc: Float = 0
                for tap in 0..<4 {
                    let idx = t - 3 + tap
                    let val: Float
                    if idx < 0 {
                        // state[c*3 + j] = x[j-3]; slot for x[idx] is c*3+3+idx.
                        val = state[c * 3 + (3 + idx)]
                    } else {
                        val = x[idx * channels + c]
                    }
                    acc += w[4 * c + tap] * val
                }
                out[t * channels + c] = GDNRef.silu(acc)
            }
        }
        var ns = [Float](repeating: 0, count: 3 * channels)
        for c in 0..<channels {
            for j in 0..<3 {
                let idx = tokens - 3 + j
                ns[c * 3 + j] = idx >= 0
                    ? x[idx * channels + c]
                    : state[c * 3 + (3 + idx)]
            }
        }
        return (out, ns)
    }

    /// Batched gate: for each (t, v),
    ///   `beta = sigmoid(b)`, `g = -exp(A_log) * softplus(a + dt_bias)`
    /// from the fp16-rounded `ab` ([T][2V], a rows then b rows — the kernel
    /// reads the QMM's fp16 output, so the reference works on the rounded
    /// values the test supplies).
    public static func gateBatch(
        ab: [Float], A_log: [Float], dt_bias: [Float],
        numValueHeads: Int, tokens: Int
    ) -> (g: [Float], beta: [Float]) {
        var g = [Float](repeating: 0, count: tokens * numValueHeads)
        var beta = [Float](repeating: 0, count: tokens * numValueHeads)
        for t in 0..<tokens {
            for v in 0..<numValueHeads {
                let av = ab[t * (2 * numValueHeads) + v]
                let bv = ab[t * (2 * numValueHeads) + numValueHeads + v]
                g[t * numValueHeads + v] = -expf(A_log[v])
                    * GDNRef.softplus(av + dt_bias[v])
                beta[t * numValueHeads + v] = 1.0 / (1.0 + expf(-bv))
            }
        }
        return (g, beta)
    }

    /// The gated-delta-rule recurrence applied sequentially to each of the
    /// chunk's T tokens, per value head (state is in/out, v-major
    /// `[V][D][D]`). `conv` is the fused flat `[T][C]` conv output with the
    /// k/v blocks at element offsets `kOffset`/`vOffset`; `g`/`beta` are
    /// [T][V] fp32. Returns `out` [T][V][D]. Reuses `GDNRef.recurrentStep`
    /// (the validated decode-step reference) — the recurrence IS the decode
    /// step, just unrolled.
    public static func recurrentChunk(
        state: inout [Float],
        conv: [Float],
        g: [Float],
        beta: [Float],
        channels: Int,
        kOffset: Int,
        vOffset: Int,
        numValueHeads: Int,
        numKeyHeads: Int,
        headDim: Int,
        tokens: Int,
        scale: Float
    ) -> [Float] {
        // Grouped (repeat_interleave) pairing: value head `hv` reads key head
        // `hv / (V/K)`. Qwen 3.6 is 32/16 and 3.8 is 48/16, so the divisor has
        // to come from the geometry — a literal 2 is wrong on 3.8.
        let vPerK = numValueHeads / numKeyHeads
        var out = [Float](repeating: 0, count: tokens * numValueHeads * headDim)
        for t in 0..<tokens {
            for hv in 0..<numValueHeads {
                let kh = hv / vPerK
                let q = Array(conv[(t * channels + kh * headDim)..<(t * channels + kh * headDim + headDim)])
                let k = Array(conv[(t * channels + kOffset + kh * headDim)..<(t * channels + kOffset + kh * headDim + headDim)])
                let v = Array(conv[(t * channels + vOffset + hv * headDim)..<(t * channels + vOffset + hv * headDim + headDim)])
                let sBase = hv * headDim * headDim
                var slice = Array(state[sBase..<(sBase + headDim * headDim)])
                let o = GDNRef.recurrentStep(
                    state: &slice, q: q, k: k, v: v,
                    g: g[t * numValueHeads + hv], beta: beta[t * numValueHeads + hv],
                    scale: scale)
                for (i, val) in o.enumerated() {
                    out[(t * numValueHeads + hv) * headDim + i] = val
                }
                state.replaceSubrange(sBase..<(sBase + headDim * headDim), with: slice)
            }
        }
        return out
    }

    /// Batched `Qwen3_5MoeRMSNormGated` over each (t, hv) vector:
    /// `x * rsqrt(mean(x^2) + eps) * weight * silu(z)`; `weight` is [headDim]
    /// shared across heads.
    public static func rmsNormGatedBatch(
        x: [Float], z: [Float], weight: [Float],
        numValueHeads: Int, headDim: Int, tokens: Int
    ) -> [Float] {
        var out = [Float](repeating: 0, count: tokens * numValueHeads * headDim)
        for t in 0..<tokens {
            for hv in 0..<numValueHeads {
                let base = (t * numValueHeads + hv) * headDim
                let xs = Array(x[base..<(base + headDim)])
                let zs = Array(z[base..<(base + headDim)])
                let y = GDNRef.rmsNormGated(xs, weight: weight, z: zs)
                out.replaceSubrange(base..<(base + headDim), with: y)
            }
        }
        return out
    }
}

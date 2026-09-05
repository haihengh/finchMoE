import Foundation
import FlashQwen

/// fp32 CPU reference for one GDN (gated-delta-net) decode step, grounded in
/// `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`:
/// `torch_causal_conv1d_update`, `l2norm`, `torch_recurrent_gated_delta_rule`,
/// and `Qwen3_5MoeRMSNormGated`.
///
/// The state convention mirrors the Metal kernel: the per-head recurrent state
/// is stored **v-major** (`S[v][k]`), so a matvec `sum_k S[v,k]*x[k]` and the
/// rank-1 write `S[v,k] += k[k]*delta[v]` are both contiguous over `k`.
///
/// Every step is pure fp32 and written as an explicit loop (a deliberately
/// different op-tree from the Metal two-stage block reduction) so that the
/// kernel-vs-reference comparison is meaningful.
public enum GDNRef {
    public static let l2NormEps: Float = 1e-6
    public static let rmsEps: Float = 1e-6

    /// `torch_causal_conv1d_update` at decode (seq_len 1, kernel 4):
    /// `out = silu(w0*s0 + w1*s1 + w2*s2 + w3*x)`, new state = `[s1, s2, x]`.
    /// `w` has 4 elements per channel, `state` 3, `x` 1; each is per-channel.
    public static func causalConvUpdate(
        w: [Float], state: [Float], x: [Float]
    ) -> (out: [Float], newState: [Float]) {
        let channels = x.count
        var out = [Float](repeating: 0, count: channels)
        var ns = [Float](repeating: 0, count: 3 * channels)
        for c in 0..<channels {
            let acc = w[4*c + 0] * state[3*c + 0]
                   + w[4*c + 1] * state[3*c + 1]
                   + w[4*c + 2] * state[3*c + 2]
                   + w[4*c + 3] * x[c]
            out[c] = silu(acc)
            ns[3*c + 0] = state[3*c + 1]
            ns[3*c + 1] = state[3*c + 2]
            ns[3*c + 2] = x[c]
        }
        return (out, ns)
    }

    /// Per-value-head gate, from `Qwen3_5MoeGatedDeltaNet.forward`:
    ///   `beta = sigmoid(b)`
    ///   `g    = -exp(A_log) * softplus(a + dt_bias)`
    /// `a/b/A_log/dt_bias` are per-head (length `numValueHeads`) fp32.
    public static func gate(
        a: [Float], b: [Float], A_log: [Float], dt_bias: [Float]
    ) -> (g: [Float], beta: [Float]) {
        var g = [Float](repeating: 0, count: a.count)
        var beta = [Float](repeating: 0, count: a.count)
        for i in 0..<a.count {
            g[i]    = -expf(A_log[i]) * softplus(a[i] + dt_bias[i])
            beta[i] = 1.0 / (1.0 + expf(-b[i]))
        }
        return (g, beta)
    }

    /// FP32 reference for the fused `gdn_gate_gemv` kernel: int4-affine GEMVs
    /// of in_proj_a/in_proj_b (`aRows` then `bRows`, each [V, N]) followed by
    /// `gate`. The GEMVs go through `DequantInt4GemvRef` (bulk-dequant +
    /// `vDSP_dotpr`), a different op-tree than the kernel's per-group scalar
    /// loop.
    public static func gateGEMV(
        aRows: [Quantization.Int4AffineRow],
        bRows: [Quantization.Int4AffineRow],
        x: [Float],
        A_log: [Float],
        dt_bias: [Float]
    ) -> (g: [Float], beta: [Float]) {
        let a = DequantInt4GemvRef.apply(weightRows: aRows, x: x, n: x.count)
        let b = DequantInt4GemvRef.apply(weightRows: bRows, x: x, n: x.count)
        return gate(a: a, b: b, A_log: A_log, dt_bias: dt_bias)
    }

    /// `softplus(x) = log(1 + exp(x))`, stable for x > 0 via `x + log(1 + exp(-x))`.
    /// Uses `log1p` (a different op-tree than the kernel's `log(1 + ·)` form) so
    /// a matching result within tolerance is a real check, not an identical path.
    public static func softplus(_ x: Float) -> Float {
        if x > 0 { return x + log1p(expf(-x)) }
        return log1p(expf(x))
    }

    /// `l2norm(x, dim=-1, eps=1e-6)`: `x * rsqrt(sum(x^2) + eps)` (sum-based).
    public static func l2Norm(_ x: [Float]) -> [Float] {
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1.0 / sqrtf(ss + l2NormEps)
        return x.map { $0 * inv }
    }

    /// `Qwen3_5MoeRMSNormGated`: `x * rsqrt(mean(x^2) + eps) * weight * silu(z)`.
    public static func rmsNormGated(
        _ x: [Float], weight: [Float], z: [Float]
    ) -> [Float] {
        let n = x.count
        var ss: Float = 0
        for v in x { ss += v * v }
        let inv = 1.0 / sqrtf(ss / Float(n) + rmsEps)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            out[i] = x[i] * inv * weight[i] * silu(z[i])
        }
        return out
    }

    /// Recurrent gated-delta-rule decode step for one value head.
    ///
    /// - `state`: in/out, v-major `[headDim][headDim]` (v rows, k columns), fp32.
    /// - `q`, `k`, `v`: raw per-head vectors (already conv-updated), length `headDim`.
    /// - `g`: `g = -exp(A_log) * softplus(a + dt_bias)` for this head (precomputed).
    /// - `beta`: `sigmoid(b)` for this head (precomputed).
    /// - `scale`: `1/sqrt(headDim)`; applied to `q` only (AFTER l2norm — torch
    ///   `torch_recurrent_gated_delta_rule` does `query = l2norm(query, eps=1e-6)`
    ///   then `query = query * scale`; scaling INSIDE the norm would cancel).
    ///
    /// Order (from `torch_recurrent_gated_delta_rule`, decode branch):
    /// decay → residual read → rank-1 write → output read from the UPDATED state.
    /// Returns the readout vector `o[v]` (length `headDim`).
    public static func recurrentStep(
        state: inout [Float], q: [Float], k: [Float], v: [Float],
        g: Float, beta: Float, scale: Float
    ) -> [Float] {
        let d = v.count
        let decay = expf(g)

        let qn = l2Norm(q).map { $0 * scale }
        let kn = l2Norm(k)

        // S = S * decay (elementwise over [v, k])
        for i in 0..<(d * d) { state[i] *= decay }

        // r[v] = sum_k S[v,k] * kn[k]
        var r = [Float](repeating: 0, count: d)
        for vIdx in 0..<d {
            var acc: Float = 0
            let row = vIdx * d
            for kIdx in 0..<d { acc += state[row + kIdx] * kn[kIdx] }
            r[vIdx] = acc
        }

        // delta = beta * (v - r); write S[v,k] += kn[k] * delta[v]
        for vIdx in 0..<d {
            let delta = beta * (v[vIdx] - r[vIdx])
            var acc: Float = 0
            let row = vIdx * d
            for kIdx in 0..<d {
                state[row + kIdx] += kn[kIdx] * delta
                acc += state[row + kIdx] * qn[kIdx]
            }
            r[vIdx] = acc  // reuse `r` as the readout scratch
        }
        return r
    }

    static func silu(_ x: Float) -> Float { x / (1 + expf(-x)) }
}

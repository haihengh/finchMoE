import Foundation
import FlashQwen

/// fp32 CPU references for the Qwen 3.6 decode-layer fusions in
/// `Metal/Qwen/qwen_decode.metal`, grounded in `qwen3_5_moe`:
///   * `Qwen3_5MoeDecoderLayer`  — `hidden += attn`, `post_attention_layernorm`
///   * `Qwen3_5MoeAttention`     — `attn_output_gate`, q_norm/k_norm, RoPE
///   * `Qwen3_5MoeSparseMoeBlock` — `shared_expert_gate`
///
/// The norms reuse `RmsNormRef` (Accelerate pipeline) and the rotation reuses
/// `RopeRef.apply` (bulk vForce trig, paired convention) — deliberately
/// different op-trees from the kernel's two-stage block reductions and
/// per-pair `cosf`/`sinf`.
public enum QwenDecodeRef {
    public static let rmsEps: Float = 1e-6

    /// `hidden[i] += attn[i]` (half-rounded in place) then
    /// `out = rmsnorm(hidden) * weight`.
    public static func postAttn(
        hidden: [Float], attn: [Float], weight: [Float]
    ) -> (hidden: [Float], out: [Float]) {
        let h = zip(hidden, attn).map { a, b in Float(Float16(a + b)) }
        return (h, RmsNormRef.apply(x: h, weight: weight, eps: rmsEps))
    }

    /// Elementwise `a += b` (half-rounded).
    public static func vecAdd(_ a: [Float], _ b: [Float]) -> [Float] {
        return zip(a, b).map { x, y in Float(Float16(x + y)) }
    }

    /// `attn[i] *= sigmoid(gate[i])` (half-rounded).
    public static func attnOutputGate(attn: [Float], gate: [Float]) -> [Float] {
        return zip(attn, gate).map { a, g in
            let s = 1.0 / (1.0 + expf(-g))
            return Float(Float16(a * s))
        }
    }

    /// `h1[i] *= sigmoid(dot(W, x))` where W is a single int4-affine row of
    /// length N (dot via `DequantInt4GemvRef`, h1 scaling half-rounded).
    public static func sharedGate(
        weightRow: Quantization.Int4AffineRow,
        x: [Float],
        h1: [Float]
    ) -> [Float] {
        let dot = DequantInt4GemvRef.apply(weightRows: [weightRow], x: x, n: x.count)[0]
        let gate = 1.0 / (1.0 + expf(-dot))
        return h1.map { Float(Float16($0 * gate)) }
    }

    /// Qwen full-attention epilogue. `qProj` is [numQHeads, 2*headDim]
    /// (per-head `[q_256 | gate_256]` pairs); returns the normalized+rotated
    /// q/k and the raw gate halves. Rotation via `RopeRef.apply` (paired
    /// convention, rotary-dim frequency denominator — the Qwen text RoPE).
    public static func fullAttnEpilogue(
        qProj: [Float],
        kIn: [Float],
        qWeight: [Float],
        kWeight: [Float],
        headDim: Int,
        numQHeads: Int,
        numKVHeads: Int,
        rotaryDim: Int,
        position: Int,
        theta: Float
    ) -> (qOut: [Float], gateOut: [Float], kOut: [Float]) {
        var qSlices: [Float] = []
        var gateOut = [Float](repeating: 0, count: numQHeads * headDim)
        for h in 0..<numQHeads {
            let base = h * 2 * headDim
            let qSlice = Array(qProj[base..<(base + headDim)])
            qSlices += RmsNormRef.apply(x: qSlice, weight: qWeight, eps: rmsEps)
            for i in 0..<headDim {
                gateOut[h * headDim + i] = qProj[base + headDim + i]
            }
        }
        let qRot = RopeRef.apply(input: qSlices, numTokens: 1, numHeads: numQHeads,
                                 headDim: headDim, rotaryDim: rotaryDim,
                                 position: position, theta: theta)
        var kNorm: [Float] = []
        for h in 0..<numKVHeads {
            let base = h * headDim
            let kSlice = Array(kIn[base..<(base + headDim)])
            kNorm += RmsNormRef.apply(x: kSlice, weight: kWeight, eps: rmsEps)
        }
        let kRot = RopeRef.apply(input: kNorm, numTokens: 1, numHeads: numKVHeads,
                                 headDim: headDim, rotaryDim: rotaryDim,
                                 position: position, theta: theta)
        return (qRot, gateOut, kRot)
    }
}

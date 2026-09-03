import Foundation

/// FP32 reference for the elementwise `silu_mul_fp16` kernel:
/// `y[i] = silu(gate[i]) * up[i]`, computed in fp32 from fp16-rounded inputs
/// (the test rounds inputs to FP16 first and feeds the rounded values here).
public enum SiluMulRef {
    public static func apply(gate: [Float], up: [Float]) -> [Float] {
        precondition(gate.count == up.count, "gate and up must match length")
        return zip(gate, up).map { g, u in
            let silu = g / (1.0 + exp(-g))
            return silu * u
        }
    }
}

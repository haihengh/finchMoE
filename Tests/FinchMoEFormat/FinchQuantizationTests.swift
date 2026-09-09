import Foundation
import Testing
import FinchMoEFormat

/// Regression guard for the denormal-scale quantization trap: Qwen 3.6 bf16
/// exports carry rows of subnormal-range residue (e.g. layer 0
/// `linear_attn.in_proj_qkv` row 143 is alternating ±2^-123 = BF16 0x0200 /
/// 0x8200). The old `q = (w - bias) * (1 / scale)` overflowed the FP32
/// reciprocal to inf when the BF16-rounded group scale is subnormal, then
/// produced NaN at the Int() conversion and crashed the repack.
@Suite struct FinchQuantizationTests {

    /// Alternating ±2^-123 BF16 values, exactly as found in the checkpoint.
    private static func subnormalResidueRow() -> [Float] {
        (0..<FinchQuantization.groupSize).map { k in
            FinchQuantization.bf16ToFloat(k % 2 == 0 ? 0x0200 : 0x8200)
        }
    }

    @Test func int8SubnormalResidueRowQuantizesWithoutTrap() {
        let row = Self.subnormalResidueRow()
        let q = row.withUnsafeBufferPointer {
            FinchQuantization.quantizeInt8Affine($0, count: row.count)
        }
        let decoded = FinchQuantization.dequantizeInt8Affine(q, n: row.count)
        // Bias-only reconstruction carries the group's error; the row spans
        // ±1.175e-37 so tolerance is in that ballpark.
        for (a, b) in zip(decoded, row) {
            #expect(abs(a - b) < 1e-36)
        }
    }

    @Test func int4SubnormalResidueRowQuantizesWithoutTrap() {
        let row = Self.subnormalResidueRow()
        let q = row.withUnsafeBufferPointer {
            FinchQuantization.quantizeInt4Affine($0, count: row.count)
        }
        let decoded = FinchQuantization.dequantizeInt4Affine(q, n: row.count)
        for (a, b) in zip(decoded, row) {
            #expect(abs(a - b) < 1e-36)
        }
    }

    @Test func ordinaryRangeStillRoundTripsWithinAffineError() {
        // Sanity that the division-based codec kept normal-range accuracy:
        // a ±1 row at int8 keeps error well under one scale step (2/255).
        let row = (0..<FinchQuantization.groupSize).map {
            Float($0 % 3) - 1.0   // -1, 0, 1, -1, ...
        }
        let q = row.withUnsafeBufferPointer {
            FinchQuantization.quantizeInt8Affine($0, count: row.count)
        }
        let decoded = FinchQuantization.dequantizeInt8Affine(q, n: row.count)
        for (a, b) in zip(decoded, row) {
            #expect(abs(a - b) < 0.02)
        }
    }
}

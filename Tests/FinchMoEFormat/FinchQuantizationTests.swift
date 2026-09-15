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

    // MARK: - PLE table (160-wide, group 32, per-row fixed stride)

    /// A 160-wide row in the PLE value profile: small, roughly zero-mean,
    /// bounded a few steps below the int4 floor. Deterministic, no RNG.
    private static func pleNormalRow(cols: Int) -> [Float] {
        (0..<cols).map { i in
            let g = (i * 37 % 13) - 6      // -6...6, varies within and across groups
            return Float(g) * 0.003 + 0.0007 * Float(i % 5 - 2)
        }
    }

    @Test func pleRoundTrip160WithinAffineError() {
        let cols = 160
        let row = Self.pleNormalRow(cols: cols)
        let q = FinchQuantization.quantizeInt4AffinePLE(row)   // group 32
        let decoded = FinchQuantization.dequantizeInt4AffinePLE(q, n: cols)
        // Per-group span is the natural affine bound; allow one full step
        // (span/15) plus headroom for BF16 scale/bias rounding. A wrong codec
        // would emit errors of ~span or larger, far beyond this.
        let nGroups = cols / FinchQuantization.pleGroupSize
        for g in 0..<nGroups {
            let lo = g * FinchQuantization.pleGroupSize
            var span: Float = 0
            for k in lo..<(lo + FinchQuantization.pleGroupSize) { span = max(span, abs(row[k])) }
            span *= 2  // bound the group's max-min by 2*max|w|
            for k in lo..<(lo + FinchQuantization.pleGroupSize) {
                #expect(abs(decoded[k] - row[k]) < span / 15.0 * 1.5,
                        "k=\(k) decoded=\(decoded[k]) row=\(row[k])")
            }
        }
    }

    @Test func pleRowHasFixedStrideLayout() {
        // Design decision 3: per-row [packed: cols/2][scales: nG][biases: nG].
        // 160 cols / group 32 → 5 groups; 80 + 5*2 + 5*2 = 100 bytes/row.
        let cols = 160
        let row = Self.pleNormalRow(cols: cols)
        let q = FinchQuantization.quantizeInt4AffinePLE(row)
        #expect(q.packed.count == cols / 2)
        #expect(q.scales.count == cols / FinchQuantization.pleGroupSize)
        #expect(q.biases.count == cols / FinchQuantization.pleGroupSize)
        let onDiskBytes = q.packed.count + q.scales.count * 2 + q.biases.count * 2
        #expect(onDiskBytes == 100)
    }

    @Test func pleConstantGroupReconstructsExactly() {
        // Constant group → scale=1, bias=value, q=0 → exact round trip.
        let cols = 160
        let v: Float = 0.5  // exactly representable in BF16
        let row = [Float](repeating: v, count: cols)
        let q = FinchQuantization.quantizeInt4AffinePLE(row)
        let decoded = FinchQuantization.dequantizeInt4AffinePLE(q, n: cols)
        for a in decoded { #expect(a == v) }
    }

    @Test func pleSubnormalResidueRowQuantizesWithoutTrap() {
        // Same denormal-scale trap the group-64 codec guards against, now at
        // group 32: alternating ±2^-123 per group. Must not NaN/trap.
        let cols = 160
        let row = (0..<cols).map { k in
            FinchQuantization.bf16ToFloat(k % 2 == 0 ? 0x0200 : 0x8200)
        }
        let q = FinchQuantization.quantizeInt4AffinePLE(row)
        let decoded = FinchQuantization.dequantizeInt4AffinePLE(q, n: cols)
        for (a, b) in zip(decoded, row) {
            #expect(a.isFinite)
            #expect(abs(a - b) < 1e-36)
        }
    }

    @Test func pleDequantDecodeIsSelfDescribingAndIndependent() {
        // Decode must not depend on the quantizer: hand-build a row with all
        // nibbles = 3, scale = 2.0, bias = 1.0 → every element = 3*2+1 = 7.
        let cols = 160
        let nG = cols / FinchQuantization.pleGroupSize   // 5
        let row = FinchQuantization.Int4AffinePLERow(
            packed: [UInt8](repeating: 0x33, count: cols / 2),
            scales: [UInt16](repeating: FinchQuantization.bf16Bits(2.0), count: nG),
            biases: [UInt16](repeating: FinchQuantization.bf16Bits(1.0), count: nG))
        let decoded = FinchQuantization.dequantizeInt4AffinePLE(row, n: cols)
        #expect(decoded.count == cols)
        for a in decoded { #expect(a == 7.0) }
    }

    @Test func pleDequantSupportsOtherGroupSizes() {
        // The decoder derives group size from the row (n / nGroups), so a
        // 4-group (group-40) row of the same width must also decode: all
        // nibbles = 2, scale = 0.5, bias = 0 → every element = 2*0.5 = 1.
        let cols = 160
        let row = FinchQuantization.Int4AffinePLERow(
            packed: [UInt8](repeating: 0x22, count: cols / 2),
            scales: [UInt16](repeating: FinchQuantization.bf16Bits(0.5), count: 4),
            biases: [UInt16](repeating: FinchQuantization.bf16Bits(0.0), count: 4))
        let decoded = FinchQuantization.dequantizeInt4AffinePLE(row, n: cols)
        for a in decoded { #expect(a == 1.0) }
    }
}

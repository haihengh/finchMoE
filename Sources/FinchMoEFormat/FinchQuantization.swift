import Foundation

/// Canonical affine int4/int8 quantization for the `.finch` on-disk format.
///
/// This is the single source of truth shared by every side that touches the
/// packed weights: the engine's decode reference, the Metal kernels' expected
/// layout, and the repack writer. Keeping the math here — rather than in the
/// engine's `Quantization` facade — is what lets the repack tool (which can only
/// import `FinchMoEFormat`, not the engine) emit byte-identical int4 from a
/// bf16 checkpoint.
///
/// Bit layout: unsigned packed nibbles (low nibble = even index, high = odd),
/// with BF16 `scale` + `bias` per group of `groupSize`. Decode is
/// `w ≈ q * scale + bias`. Scale and bias are stored as raw BF16 bits
/// (`UInt16`) so the same buffer uploads verbatim to a Metal
/// `device const bfloat*`.
public enum FinchQuantization {

    public static let groupSize: Int = 64

    // MARK: - BF16 helpers
    //
    // BF16 = top 16 bits of FP32 (8-bit exponent, 7-bit mantissa). Stored on
    // disk and on the GPU as the native `bfloat` type; in Swift we carry the
    // raw bits as `UInt16` and convert via this pair. Round-half-to-even on
    // encode matches the IEEE-754 default.

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        let bits = x.bitPattern
        let lsb  = (bits >> 16) & 1
        let roundingBias: UInt32 = 0x7FFF &+ lsb
        let rounded = (bits &+ roundingBias) >> 16
        return UInt16(truncatingIfNeeded: rounded)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    // MARK: - INT4 affine

    /// MLX `affine` 4-bit row. Packed unsigned nibbles, BF16 scale + bias per
    /// group of 64. `scales` and `biases` carry BF16 bit patterns as `UInt16`
    /// so the same buffer can be uploaded to a Metal `device const bfloat*`.
    public struct Int4AffineRow {
        public let packed: [UInt8]   // N / 2 bytes; low nibble = even index, high = odd
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// Affine 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`.
    /// Scale and bias are computed from per-group min/max, then rounded to BF16.
    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        quantizeInt4Affine(row, count: row.count)
    }

    /// Buffer form: quantizes the first `count` elements of `buffer` (a
    /// reusable scratch buffer the writer owns), so the repack can stream
    /// rows without allocating an input array per row.
    public static func quantizeInt4Affine(_ buffer: UnsafeBufferPointer<Float>,
                                          count: Int) -> Int4AffineRow {
        precondition(count % groupSize == 0,
                     "row length \(count) is not a multiple of \(groupSize)")

        let nGroups = count / groupSize
        var packed = [UInt8](repeating: 0, count: count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the writer stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            // Quantize against the BF16-rounded scale directly. A reciprocal
            // would overflow FP32 to inf when the group's range is so small
            // that the rounded scale is subnormal (checkpoints carry
            // denormal-range residue rows), turning finite values into
            // NaN/inf at the Int() conversion below.
            let effectiveScale = scale == 0 ? Float(1) : scale

            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                let qv = scale == 0 ? Float(0) : (w - bias) / effectiveScale
                var q = Int(qv.rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffineRow(packed: packed, scales: scales, biases: biases)
    }

    private static func quantizeInt4Affine(_ row: [Float], count: Int) -> Int4AffineRow {
        row.withUnsafeBufferPointer { quantizeInt4Affine($0, count: count) }
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let byteIdx = g * (groupSize / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * groupSize + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT4 affine (PLE table)

    /// Group size for the PLE n-gram embedding table. Measured 2026-09-11
    /// (Phase 1): the finest group whose int4 error stays within the
    /// rest-of-model group-64 floor — 160-wide rows split into 5 groups.
    public static let pleGroupSize: Int = 32

    /// PLE-table 4-bit row. Unlike `Int4AffineRow` (group of 64, written as
    /// per-file packed/scale/bias regions), the PLE table is 160 wide, uses
    /// group `pleGroupSize`, and carries a fixed per-row on-disk layout (plan
    /// design decision 3): `[packed nibbles: cols/2][scale BF16 × nGroups]`
    /// `[bias BF16 × nGroups]`. The three arrays serialize contiguously per
    /// row, so one row is `cols/2 + 2 * nGroups * 2` bytes (100 for 160 cols /
    /// group 32) — a fixed stride with no per-row variable metadata.
    public struct Int4AffinePLERow {
        public let packed: [UInt8]   // cols / 2 bytes; low nibble = even index
        public let scales: [UInt16]  // cols / groupSize BF16 bits
        public let biases: [UInt16]  // cols / groupSize BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    /// PLE 4-bit quantize: `q ∈ [0..15]`, `w ≈ q * scale + bias`, per-group
    /// scale/bias from min/max rounded through BF16. Identical affine math and
    /// edge-case handling to `quantizeInt4Affine`; only the group size (32
    /// vs 64) and the row struct differ.
    public static func quantizeInt4AffinePLE(_ row: [Float],
                                             groupSize: Int = pleGroupSize) -> Int4AffinePLERow {
        row.withUnsafeBufferPointer {
            quantizeInt4AffinePLE($0, count: row.count, groupSize: groupSize)
        }
    }

    /// Buffer form (see `quantizeInt4Affine(_:count:)`).
    public static func quantizeInt4AffinePLE(_ buffer: UnsafeBufferPointer<Float>,
                                             count: Int,
                                             groupSize: Int = pleGroupSize) -> Int4AffinePLERow {
        precondition(count % groupSize == 0,
                     "PLE row length \(count) is not a multiple of \(groupSize)")

        let nGroups = count / groupSize
        var packed = [UInt8](repeating: 0, count: count / 2)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            // Constant group: scale=1, bias=value preserves exact reconstruction.
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 15.0
                biasF  = wmin
            }
            // Round through BF16 first, then quantize against the rounded
            // values so the runtime decode (which reads BF16) reproduces the
            // same q the writer stored.
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            // Quantize against the BF16-rounded scale directly — a reciprocal
            // would overflow FP32 to inf for a subnormal rounded scale.
            let effectiveScale = scale == 0 ? Float(1) : scale

            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                let qv = scale == 0 ? Float(0) : (w - bias) / effectiveScale
                var q = Int(qv.rounded())
                q = max(0, min(15, q))
                let nibble = UInt8(q) & 0x0F
                let byteIdx = g * (groupSize / 2) + (k / 2)
                if (k & 1) == 0 {
                    packed[byteIdx] = (packed[byteIdx] & 0xF0) | nibble
                } else {
                    packed[byteIdx] = (packed[byteIdx] & 0x0F) | (nibble << 4)
                }
            }
        }
        return Int4AffinePLERow(packed: packed, scales: scales, biases: biases)
    }

    /// Decode a PLE 4-bit row. Group size is derived from the row itself
    /// (`n / nGroups`), so the row is self-describing — no separate group-size
    /// field to keep in sync. For a 160-wide row with group 32 that is 5
    /// groups.
    public static func dequantizeInt4AffinePLE(_ r: Int4AffinePLERow, n: Int) -> [Float] {
        precondition(n == r.packed.count * 2)
        let nGroups = r.scales.count
        precondition(nGroups > 0 && r.biases.count == nGroups && n % nGroups == 0,
                     "malformed PLE row: n=\(n), packed=\(r.packed.count), " +
                     "scales=\(r.scales.count), biases=\(r.biases.count)")
        let groupSize = n / nGroups
        var out = [Float](repeating: 0, count: n)
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                let byteIdx = g * (groupSize / 2) + (k / 2)
                let b = r.packed[byteIdx]
                let nibble: Int = (k & 1) == 0 ? Int(b & 0x0F) : Int(b >> 4)
                out[g * groupSize + k] = Float(nibble) * scale + bias
            }
        }
        return out
    }

    // MARK: - INT8 affine

    public struct Int8AffineRow {
        public let packed: [UInt8]   // N unsigned bytes
        public let scales: [UInt16]  // N / 64 BF16 bits
        public let biases: [UInt16]  // N / 64 BF16 bits

        public init(packed: [UInt8], scales: [UInt16], biases: [UInt16]) {
            self.packed = packed
            self.scales = scales
            self.biases = biases
        }
    }

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        row.withUnsafeBufferPointer { quantizeInt8Affine($0, count: row.count) }
    }

    /// Buffer form (see `quantizeInt4Affine(_:count:)`).
    public static func quantizeInt8Affine(_ buffer: UnsafeBufferPointer<Float>,
                                          count: Int) -> Int8AffineRow {
        precondition(count % groupSize == 0,
                     "row length \(count) is not a multiple of \(groupSize)")

        let nGroups = count / groupSize
        var packed = [UInt8](repeating: 0, count: count)
        var scales = [UInt16](repeating: 0, count: nGroups)
        var biases = [UInt16](repeating: 0, count: nGroups)

        for g in 0..<nGroups {
            var wmin: Float =  .infinity
            var wmax: Float = -.infinity
            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                if w < wmin { wmin = w }
                if w > wmax { wmax = w }
            }
            let scaleF: Float
            let biasF:  Float
            if wmax == wmin {
                scaleF = 1
                biasF  = wmin
            } else {
                scaleF = (wmax - wmin) / 255.0
                biasF  = wmin
            }
            let sBits = bf16Bits(scaleF)
            let bBits = bf16Bits(biasF)
            scales[g] = sBits
            biases[g] = bBits
            let scale = bf16ToFloat(sBits)
            let bias  = bf16ToFloat(bBits)
            // Quantize against the BF16-rounded scale directly. A reciprocal
            // would overflow FP32 to inf when the group's range is so small
            // that the rounded scale is subnormal (checkpoints carry
            // denormal-range residue rows), turning finite values into
            // NaN/inf at the Int() conversion below.
            let effectiveScale = scale == 0 ? Float(1) : scale

            for k in 0..<groupSize {
                let w = buffer[g * groupSize + k]
                let qv = scale == 0 ? Float(0) : (w - bias) / effectiveScale
                var q = Int(qv.rounded())
                q = max(0, min(255, q))
                packed[g * groupSize + k] = UInt8(q)
            }
        }
        return Int8AffineRow(packed: packed, scales: scales, biases: biases)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        precondition(n == r.packed.count)
        var out = [Float](repeating: 0, count: n)
        let nGroups = n / groupSize
        for g in 0..<nGroups {
            let scale = bf16ToFloat(r.scales[g])
            let bias  = bf16ToFloat(r.biases[g])
            for k in 0..<groupSize {
                out[g * groupSize + k] = Float(r.packed[g * groupSize + k]) * scale + bias
            }
        }
        return out
    }
}

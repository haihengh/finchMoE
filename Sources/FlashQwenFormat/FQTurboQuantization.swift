import Foundation

/// Canonical affine int4/int8 quantization for the `.fqturbo` on-disk format.
///
/// This is the single source of truth shared by every side that touches the
/// packed weights: the engine's decode reference, the Metal kernels' expected
/// layout, and the repack writer. Keeping the math here — rather than in the
/// engine's `Quantization` facade — is what lets the repack tool (which can only
/// import `FlashQwenFormat`, not the engine) emit byte-identical int4 from a
/// bf16 checkpoint.
///
/// Bit layout: unsigned packed nibbles (low nibble = even index, high = odd),
/// with BF16 `scale` + `bias` per group of `groupSize`. Decode is
/// `w ≈ q * scale + bias`. Scale and bias are stored as raw BF16 bits
/// (`UInt16`) so the same buffer uploads verbatim to a Metal
/// `device const bfloat*`.
public enum FQTurboQuantization {

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

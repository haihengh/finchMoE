import Foundation
import FinchMoEFormat

/// Engine-facing quantization API. The math itself is canonical in
/// `FinchMoEFormat.FinchQuantization` (the shared `.finch` format module),
/// which both the engine and the repack tool import. This enum is a thin
/// forwarding facade so the existing `Quantization.*` call sites in kernels,
/// the reference runner, and the test suite keep compiling unchanged.
public enum Quantization {

    public typealias Int4AffineRow = FinchQuantization.Int4AffineRow
    public typealias Int8AffineRow = FinchQuantization.Int8AffineRow
    public typealias Int4AffinePLERow = FinchQuantization.Int4AffinePLERow

    public static let groupSize: Int = FinchQuantization.groupSize
    /// Group width for the PLE n-gram table (32, measured 2026-09-11).
    public static let pleGroupSize: Int = FinchQuantization.pleGroupSize

    // MARK: - BF16 helpers

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        FinchQuantization.bf16Bits(x)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        FinchQuantization.bf16ToFloat(bits)
    }

    // MARK: - INT4 affine

    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        FinchQuantization.quantizeInt4Affine(row)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        FinchQuantization.dequantizeInt4Affine(r, n: n)
    }

    // MARK: - INT4 affine (PLE table)

    public static func dequantizeInt4AffinePLE(_ r: Int4AffinePLERow, n: Int) -> [Float] {
        FinchQuantization.dequantizeInt4AffinePLE(r, n: n)
    }

    // MARK: - INT8 affine

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        FinchQuantization.quantizeInt8Affine(row)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        FinchQuantization.dequantizeInt8Affine(r, n: n)
    }
}

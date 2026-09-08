import Foundation
import FinchMoEFormat

/// Engine-facing quantization API. The math itself is canonical in
/// `FinchMoEFormat.FinchTurboQuantization` (the shared `.finchturbo` format module),
/// which both the engine and the repack tool import. This enum is a thin
/// forwarding facade so the existing `Quantization.*` call sites in kernels,
/// the reference runner, and the test suite keep compiling unchanged.
public enum Quantization {

    public typealias Int4AffineRow = FinchTurboQuantization.Int4AffineRow
    public typealias Int8AffineRow = FinchTurboQuantization.Int8AffineRow

    public static let groupSize: Int = FinchTurboQuantization.groupSize

    // MARK: - BF16 helpers

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        FinchTurboQuantization.bf16Bits(x)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        FinchTurboQuantization.bf16ToFloat(bits)
    }

    // MARK: - INT4 affine

    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        FinchTurboQuantization.quantizeInt4Affine(row)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        FinchTurboQuantization.dequantizeInt4Affine(r, n: n)
    }

    // MARK: - INT8 affine

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        FinchTurboQuantization.quantizeInt8Affine(row)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        FinchTurboQuantization.dequantizeInt8Affine(r, n: n)
    }
}

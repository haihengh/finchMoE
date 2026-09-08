import Foundation
import FinchMoEFormat

/// Engine-facing quantization API. The math itself is canonical in
/// `FinchMoEFormat.FQTurboQuantization` (the shared `.fqturbo` format module),
/// which both the engine and the repack tool import. This enum is a thin
/// forwarding facade so the existing `Quantization.*` call sites in kernels,
/// the reference runner, and the test suite keep compiling unchanged.
public enum Quantization {

    public typealias Int4AffineRow = FQTurboQuantization.Int4AffineRow
    public typealias Int8AffineRow = FQTurboQuantization.Int8AffineRow

    public static let groupSize: Int = FQTurboQuantization.groupSize

    // MARK: - BF16 helpers

    @inline(__always)
    public static func bf16Bits(_ x: Float) -> UInt16 {
        FQTurboQuantization.bf16Bits(x)
    }

    @inline(__always)
    public static func bf16ToFloat(_ bits: UInt16) -> Float {
        FQTurboQuantization.bf16ToFloat(bits)
    }

    // MARK: - INT4 affine

    public static func quantizeInt4Affine(_ row: [Float]) -> Int4AffineRow {
        FQTurboQuantization.quantizeInt4Affine(row)
    }

    public static func dequantizeInt4Affine(_ r: Int4AffineRow, n: Int) -> [Float] {
        FQTurboQuantization.dequantizeInt4Affine(r, n: n)
    }

    // MARK: - INT8 affine

    public static func quantizeInt8Affine(_ row: [Float]) -> Int8AffineRow {
        FQTurboQuantization.quantizeInt8Affine(row)
    }

    public static func dequantizeInt8Affine(_ r: Int8AffineRow, n: Int) -> [Float] {
        FQTurboQuantization.dequantizeInt8Affine(r, n: n)
    }
}

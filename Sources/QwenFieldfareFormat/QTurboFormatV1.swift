import Foundation

/// Constants describing the on-disk `.qturbo` v1 container format used by
/// QwenFieldfare. The format stores all resident (always-in-RAM) tensors in a
/// single `model_weights.bin` blob, and streams routed MoE experts from
/// per-layer files under `packed_experts/`.
public enum QTurboFormatV1 {

    /// 8-byte magic prefix at the start of `model_weights.bin`: `QTURBO1\0`.
    public static let magic: [UInt8] = [
        0x51, 0x54, 0x55, 0x52, 0x42, 0x4F, 0x31, 0x00 // "QTURBO1\0"
    ]

    /// Number of magic bytes.
    public static let magicLength: Int = 8

    /// Container format version.
    public static let formatVersion: UInt32 = 1

    /// Page size used for aligning expert blobs so that `pread` + page cache
    /// operate on aligned boundaries. 16 KiB.
    public static let pageSize: Int = 16384

    /// Alignment (bytes) for each expert blob within a per-layer packed file.
    /// Equal to the page size.
    public static let expertBlobAlignment: Int = 16384

    /// Alignment (bytes) for each resident tensor within `model_weights.bin`.
    /// Aligned to 64 bytes to keep GPU loads well-aligned.
    public static let residentTensorAlignment: Int = 64

    /// Standard manifest filename.
    public static let manifestFilename: String = "manifest.json"

    /// Standard resident-weights blob filename.
    public static let residentBlobFilename: String = "model_weights.bin"

    /// Directory (relative to the model root) holding per-layer expert files.
    public static let packedExpertsDir: String = "packed_experts"

    /// Returns the packed expert filename for a given layer index, e.g.
    /// `layer_03.bin`.
    public static func packedExpertFilename(layer: Int) -> String {
        return String(format: "layer_%02d.bin", layer)
    }

    /// Rounds `value` up to the next multiple of `alignment`.
    @inlinable
    public static func align(_ value: Int, to alignment: Int) -> Int {
        precondition(alignment > 0)
        let rem = value % alignment
        return rem == 0 ? value : value + (alignment - rem)
    }
}

/// Data types recognized by the format layer. The runtime only consumes
/// `fp16` (for norms, scales, biases, embeddings) and `int4Packed` (for
/// quantized projection / expert weights).
public enum QTurboDType: String, Codable, Sendable {
    case fp16
    case fp32
    case int4Packed  // MLX affine 4-bit, 8 nibbles per u32, group_size=64
    case uint32
    case bf16

    /// Size in bytes of a single element (for packed int4, the size of the
    /// packing unit — one u32 holds 8 nibbles).
    public var elementStride: Int {
        switch self {
        case .fp16, .bf16: return 2
        case .fp32, .uint32, .int4Packed: return 4
        }
    }
}

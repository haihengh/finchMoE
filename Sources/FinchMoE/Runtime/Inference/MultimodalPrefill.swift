import Metal

/// One token's mRoPE position. Text tokens carry the same value on all three
/// axes; image tokens carry the (temporal, height, width) grid coordinate.
public struct MRoPEPosition: Sendable, Equatable {
    public var temporal: Int32
    public var height: Int32
    public var width: Int32

    public init(temporal: Int32, height: Int32, width: Int32) {
        self.temporal = temporal
        self.height = height
        self.width = width
    }

    public init(_ value: Int32) {
        self.init(temporal: value, height: value, width: value)
    }
}

/// Multimodal prompt data for one prefill.
///
/// The type is deliberately vision-agnostic: it carries positions and a
/// precomputed FP16 embedding table, so `FinchMoE` never learns about image
/// formats, the vision tower, or the payload. `nil` everywhere means the
/// existing text-only path runs bit-identically.
public struct MultimodalPrefillInput: @unchecked Sendable {
    /// One entry per token in the `prefillChunked` slice, in the same order.
    public var positions: [MRoPEPosition]
    /// FP16 `[hiddenRowPositions.count, hiddenSize]` rows, laid out
    /// contiguously, that replace the embedding lookup at those positions.
    public var hiddenRows: MTLBuffer
    public var hiddenRowByteOffset: Int
    /// Absolute KV positions of those rows, ascending.
    public var hiddenRowPositions: [Int32]
    /// Interleaved mRoPE sections, e.g. `[11, 11, 10]`.
    public var mropeSection: [Int]
    /// RoPE position for decode token `p` is `p + decodePositionOffset`.
    /// Images advance the mRoPE timeline by the merged grid width, not by the
    /// placeholder token count, so this is usually negative.
    public var decodePositionOffset: Int

    public init(positions: [MRoPEPosition],
                hiddenRows: MTLBuffer,
                hiddenRowByteOffset: Int = 0,
                hiddenRowPositions: [Int32],
                mropeSection: [Int],
                decodePositionOffset: Int) {
        self.positions = positions
        self.hiddenRows = hiddenRows
        self.hiddenRowByteOffset = hiddenRowByteOffset
        self.hiddenRowPositions = hiddenRowPositions
        self.mropeSection = mropeSection
        self.decodePositionOffset = decodePositionOffset
    }
}

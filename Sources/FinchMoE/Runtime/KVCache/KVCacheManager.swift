import Foundation
import Darwin
import Metal

/// Which attention variant a layer runs. Gemma 4 interleaves 25 sliding-window
/// layers with 5 full-attention layers (the latter carry the K=V shared-tensor
/// quirk). Qwen 3.6 adds `.linear` — gated-delta-net layers keep no KV at all
/// (their recurrent state lives in the runner). Sourced from
/// `ArchConfig.fullAttentionLayerMask`.
public enum LayerKind: Sendable { case swa, full, linear }

/// How K/V bytes are stored for a layer.
///
/// `fp16` is the shipping default and the only mode the sliding-window ring
/// supports. `int8` keeps one fp16 scale per 64-element block beside an int8
/// row, which halves the steady-state KV footprint at the cost of a quantize
/// dispatch on the write path and a multiply on the read path. It applies to
/// **full-attention layers only**: those are the ones that grow with context,
/// and the SWA ring's wrap logic is deliberately left on fp16.
public enum KVStorageMode: String, Sendable, Equatable, CaseIterable {
    case fp16
    case int8

    public var label: String {
        switch self {
        case .fp16: return "FP16 (16-bit)"
        case .int8: return "Int8 (8-bit + block scale)"
        }
    }
}

/// A read view the fp16 attention kernels bind. `offset` stays 0; ring-enabled
/// SWA layers expose the physical start slot for diagnostics while kernels map
/// logical positions with the supplied ring capacity.
public struct KVView: @unchecked Sendable {
    public let buffer: MTLBuffer
    /// Byte offset of logical position 0. Always 0 under linear storage.
    public let offset: Int
    /// Bytes per token (numKVHeads * headDim * sizeof(FP16)).
    public let stride: Int
    /// Number of valid positions written so far (== `position`). Attention reads
    /// `[0, validTokenCount]` (inclusive of the just-written token).
    public let validTokenCount: Int
    /// Ring start slot. 0 under linear storage; the hook for the wrap-aware path.
    public let startSlot: Int
}

/// A read view for an int8-stored K/V timeline: int8 rows plus their fp16 block
/// scales. Offsets are always 0 — the kernels walk logical positions from the
/// buffer base and map them through `rowStride`.
public struct KVInt8View: @unchecked Sendable {
    public let values: MTLBuffer
    public let scales: MTLBuffer
    /// Bytes per token of int8 values (== kvDim).
    public let rowStride: Int
    /// Bytes per token of fp16 scales (== blocksPerRow * 2).
    public let scaleStride: Int
    public let blocksPerRow: Int
    public let validTokenCount: Int
}

/// The write target for one token's int8 K or V row.
public struct KVInt8Target: @unchecked Sendable {
    public let values: MTLBuffer
    public let valuesOffset: Int
    public let scales: MTLBuffer
    public let scalesOffset: Int
}

/// Per-layer K/V storage for the decode loop.
///
/// One K buffer and one V buffer per layer, allocated once in `init` — the
/// decode hot path never allocates. Linear storage sizes every layer for
/// `maxContext`; FP16 ring storage caps SWA layers to their physical capacity
/// while full-attention layers remain linear.
///
/// Under `int8` the full-attention layers swap that linear FP16 timeline for an
/// int8 one plus fp16 block scales, and keep a small FP16 **staging** area that
/// `kSlot`/`vSlot` hand to the projection GEMV and the norm/RoPE epilogue. The
/// caller quantizes the staged row into the int8 timeline afterwards, so the
/// attention kernels only ever read int8 for those layers.
///
/// Gemma 4 full-attention layers carry the `attention_k_eq_v` quirk: K and V
/// share the `k_proj` weight, so a single 4-bit dequant + GEMV produces the
/// raw projection. But after that the K-slot runs `k_norm` (per-head, with
/// scale) + RoPE while the V-slot runs `v_norm` (per-head, no scale) and
/// skips RoPE — they diverge before entering attention. So the cache buffers
/// must be separate; aliasing them would smash the V values with K's normed,
/// rotated bytes (gemma4-block.md §2.2). We always allocate K and V slots,
/// independent of `attentionKEqV`.
///
/// The K/V projection GEMV writes straight into the slot returned by
/// `kSlot`/`vSlot` (no separate `kv_write` kernel); the runner then norms +
/// optionally RoPE's each slot in place. `advance()` bumps the cursor once
/// both are written.
///
/// 8 GB rule: storage is bounded by per-layer physical capacity, allocated
/// once. `reset()` returns physical pages to the OS via `MADV_DONTNEED` so a
/// finished generation does not keep its KV resident into the next turn.
public final class KVCacheManager {
    public let config: ArchConfig
    public let maxContext: Int
    public let fp16RingEnabled: Bool
    public let storageMode: KVStorageMode

    private let kBuffers: [MTLBuffer]
    private let vBuffers: [MTLBuffer]
    private let strides:  [Int]         // fp16 bytes per token, per layer
    private let kinds:    [LayerKind]
    private let capacityTokens: [Int]

    /// int8 timelines and their scales, per layer. nil for fp16 layers.
    private let kInt8Buffers: [MTLBuffer?]
    private let vInt8Buffers: [MTLBuffer?]
    private let kScaleBuffers: [MTLBuffer?]
    private let vScaleBuffers: [MTLBuffer?]
    /// FP16 staging rows the projection/epilogue write into before quantizing.
    private let kStageBuffers: [MTLBuffer?]
    private let vStageBuffers: [MTLBuffer?]
    private let stageRows: Int
    private let int8LayerMask: [Bool]

    /// Elements per int8 row (`numFullKVHeads * fullHeadDim`), 0 when the mode
    /// or the shape does not support int8.
    private let int8RowElements: Int
    private let int8BlocksPerRow: Int

    public private(set) var position: Int = 0

    private static let fp16Size = 2

    public init(device: MTLDevice,
                config: ArchConfig,
                maxContext: Int,
                fp16RingEnabled: Bool = false,
                slidingWindow: Int? = nil,
                maxPrefillChunkTokens: Int = 128,
                fp16RingCapacityOverride: Int? = nil,
                storageMode: KVStorageMode = .fp16) throws {
        precondition(maxContext > 0, "maxContext must be positive")
        precondition(maxPrefillChunkTokens > 0, "maxPrefillChunkTokens must be positive")
        self.config = config
        self.maxContext = maxContext
        let ringEnabled = fp16RingEnabled
        self.fp16RingEnabled = ringEnabled
        self.storageMode = storageMode

        let swaStride  = config.numKVHeads     * config.headDim     * Self.fp16Size
        let fullStride = config.numFullKVHeads * config.fullHeadDim  * Self.fp16Size
        let swaCapacity = min(maxContext,
                              max(1, fp16RingCapacityOverride
                                  ?? ((slidingWindow ?? config.slidingWindow) + maxPrefillChunkTokens)))

        // int8 is offered only where the quantizer's eight-block layout matches
        // the row exactly; anything else keeps fp16 rather than silently
        // quantizing a shape the kernel cannot address.
        let fullRowElements = config.numFullKVHeads * config.fullHeadDim
        let int8Supported = storageMode == .int8
            && fullRowElements > 0
            && fullRowElements % KVQuantize.blockElements == 0
        let blocksPerRow = int8Supported ? fullRowElements / KVQuantize.blockElements : 0
        let int8Stride = int8Supported ? fullRowElements : 0
        let scaleStride = int8Supported ? blocksPerRow * Self.fp16Size : 0
        self.int8RowElements = int8Stride
        self.int8BlocksPerRow = blocksPerRow

        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        var st: [Int] = []
        var kd: [LayerKind] = []
        var caps: [Int] = []
        var k8: [MTLBuffer?] = []
        var v8: [MTLBuffer?] = []
        var ks8: [MTLBuffer?] = []
        var vs8: [MTLBuffer?] = []
        var kStage: [MTLBuffer?] = []
        var vStage: [MTLBuffer?] = []
        var int8Rows: [Bool] = []
        ks.reserveCapacity(config.numLayers)
        vs.reserveCapacity(config.numLayers)
        st.reserveCapacity(config.numLayers)
        kd.reserveCapacity(config.numLayers)
        caps.reserveCapacity(config.numLayers)
        k8.reserveCapacity(config.numLayers)
        v8.reserveCapacity(config.numLayers)
        ks8.reserveCapacity(config.numLayers)
        vs8.reserveCapacity(config.numLayers)
        kStage.reserveCapacity(config.numLayers)
        vStage.reserveCapacity(config.numLayers)
        int8Rows.reserveCapacity(config.numLayers)

        let stagingRows = int8Supported ? min(maxContext, maxPrefillChunkTokens) : 0
        self.stageRows = stagingRows

        for layer in 0..<config.numLayers {
            let isFull = config.fullAttentionLayerMask[layer] != 0
            // Qwen hybrid (3.6/3.8) gated-delta-net layers store no KV:
            // their recurrent state is runner-side. Skip the (large)
            // allocation entirely.
            let isLinear = !isFull && config.isQwenHybrid
            let stride = isLinear ? 0 : (isFull ? fullStride : swaStride)
            let capacity = isLinear ? 0 : (ringEnabled && !isFull ? swaCapacity : maxContext)
            let length = max(1, capacity * stride)

            guard let kBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            kBuf.label = "kv.K.layer\(layer)"
            ks.append(kBuf)

            guard let vBuf = device.makeBuffer(length: length, options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            vBuf.label = "kv.V.layer\(layer)"
            vs.append(vBuf)

            st.append(stride)
            kd.append(isLinear ? .linear : (isFull ? .full : .swa))
            caps.append(capacity)

            // int8 timelines exist only for full-attention layers. The fp16
            // buffers above stay allocated for those layers as the staging
            // area; see `kSlot`/`vSlot`.
            let layerInt8 = int8Supported && isFull
            int8Rows.append(layerInt8)
            if layerInt8 {
                let rows = max(1, capacity)
                guard let k8Buf = device.makeBuffer(length: rows * int8Stride,
                                                    options: .storageModeShared),
                      let v8Buf = device.makeBuffer(length: rows * int8Stride,
                                                    options: .storageModeShared),
                      let k8Scale = device.makeBuffer(length: rows * scaleStride,
                                                      options: .storageModeShared),
                      let v8Scale = device.makeBuffer(length: rows * scaleStride,
                                                      options: .storageModeShared),
                      let kStageBuf = device.makeBuffer(length: max(1, stagingRows * stride),
                                                        options: .storageModeShared),
                      let vStageBuf = device.makeBuffer(length: max(1, stagingRows * stride),
                                                        options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                k8Buf.label = "kv.K8.layer\(layer)"
                v8Buf.label = "kv.V8.layer\(layer)"
                k8Scale.label = "kv.Kscale.layer\(layer)"
                v8Scale.label = "kv.Vscale.layer\(layer)"
                kStageBuf.label = "kv.Kstage.layer\(layer)"
                vStageBuf.label = "kv.Vstage.layer\(layer)"
                k8.append(k8Buf)
                v8.append(v8Buf)
                ks8.append(k8Scale)
                vs8.append(v8Scale)
                kStage.append(kStageBuf)
                vStage.append(vStageBuf)
            } else {
                k8.append(nil)
                v8.append(nil)
                ks8.append(nil)
                vs8.append(nil)
                kStage.append(nil)
                vStage.append(nil)
            }
        }

        self.kBuffers = ks
        self.vBuffers = vs
        self.strides  = st
        self.kinds    = kd
        self.capacityTokens = caps
        self.kInt8Buffers = k8
        self.vInt8Buffers = v8
        self.kScaleBuffers = ks8
        self.vScaleBuffers = vs8
        self.kStageBuffers = kStage
        self.vStageBuffers = vStage
        self.int8LayerMask = int8Rows
    }

    public func layerKind(_ layer: Int) -> LayerKind { kinds[layer] }

    /// Bytes per token for `layer` (K and V share the same stride).
    public func stride(layer: Int) -> Int { strides[layer] }

    /// Physical token capacity for `layer`. Ring-enabled SWA layers can be
    /// smaller than `maxContext`; full layers and ring-off storage stay linear.
    public func capacity(layer: Int) -> Int { capacityTokens[layer] }

    public func ringCapacity(layer: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        return capacityTokens[layer]
    }

    /// True when this layer's K/V live in the int8 timeline.
    public func usesInt8Storage(layer: Int) -> Bool { int8LayerMask[layer] }

    /// Elements per int8 row, or 0 when the mode is inactive.
    public func int8RowElementCount() -> Int { int8RowElements }

    /// fp16 scales per int8 row, or 0 when the mode is inactive.
    public func int8BlocksPerRowCount() -> Int { int8BlocksPerRow }

    /// Tokens the per-layer fp16 staging area holds. int8 writes land here
    /// first and are quantized into the timeline afterwards.
    public func stagingCapacity() -> Int { stageRows }

    /// Total bytes of the K and V storage for `layer`, as allocated.
    public func bufferLength(layer: Int) -> Int {
        if usesInt8Storage(layer: layer) {
            let rows = max(1, capacityTokens[layer])
            let values = rows * int8RowElements * 2
            let scales = rows * int8BlocksPerRow * Self.fp16Size * 2
            let staging = max(1, stageRows * strides[layer]) * 2
            return values + scales + staging
        }
        return capacityTokens[layer] * strides[layer]
    }

    /// Write target for this layer's K projection at `position`.
    ///
    /// int8 layers hand back the fp16 staging row: the GEMV and the norm/RoPE
    /// epilogue run exactly as they do on fp16 storage, and the caller
    /// quantizes the result with `int8Target` in the same command buffer.
    public func kSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        if usesInt8Storage(layer: layer) {
            let stage = position % max(1, stageRows)
            return (kStageBuffers[layer]!, stage * strides[layer])
        }
        return (kBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// Write target for this layer's V projection at `position`. Always
    /// distinct from `kSlot` — full layers no longer alias K and V (Gemma 4
    /// applies different per-head norms + RoPE to K vs V; gemma4-block.md §2.2).
    public func vSlot(layer: Int, position: Int) -> (buffer: MTLBuffer, offset: Int) {
        validateRange(start: position, count: 1)
        if usesInt8Storage(layer: layer) {
            let stage = position % max(1, stageRows)
            return (vStageBuffers[layer]!, stage * strides[layer])
        }
        return (vBuffers[layer], physicalSlot(layer: layer, position: position) * strides[layer])
    }

    /// The int8 destination for one staged position.
    public func int8Target(layer: Int, isKey: Bool, position: Int) -> KVInt8Target {
        precondition(usesInt8Storage(layer: layer),
                     "layer \(layer) is not stored as int8")
        validateRange(start: position, count: 1)
        let slot = physicalSlot(layer: layer, position: position)
        if isKey {
            return KVInt8Target(values: kInt8Buffers[layer]!,
                                valuesOffset: slot * int8RowElements,
                                scales: kScaleBuffers[layer]!,
                                scalesOffset: slot * int8BlocksPerRow * Self.fp16Size)
        }
        return KVInt8Target(values: vInt8Buffers[layer]!,
                            valuesOffset: slot * int8RowElements,
                            scales: vScaleBuffers[layer]!,
                            scalesOffset: slot * int8BlocksPerRow * Self.fp16Size)
    }

    /// int8 read view for a layer's key timeline.
    public func keyInt8View(layer: Int, validTokenCount: Int) -> KVInt8View {
        int8View(layer: layer, isKey: true, validTokenCount: validTokenCount)
    }

    /// int8 read view for a layer's value timeline.
    public func valueInt8View(layer: Int, validTokenCount: Int) -> KVInt8View {
        int8View(layer: layer, isKey: false, validTokenCount: validTokenCount)
    }

    private func int8View(layer: Int, isKey: Bool, validTokenCount: Int) -> KVInt8View {
        precondition(usesInt8Storage(layer: layer),
                     "layer \(layer) is not stored as int8")
        validateValidTokenCount(validTokenCount)
        return KVInt8View(
            values: isKey ? kInt8Buffers[layer]! : vInt8Buffers[layer]!,
            scales: isKey ? kScaleBuffers[layer]! : vScaleBuffers[layer]!,
            rowStride: int8RowElements,
            scaleStride: int8BlocksPerRow * Self.fp16Size,
            blocksPerRow: int8BlocksPerRow,
            validTokenCount: validTokenCount)
    }

    public func kRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        if usesInt8Storage(layer: layer) {
            // Staging is what the caller reads back for the quantize step.
            precondition(count <= max(1, stageRows),
                         "staging holds \(stageRows) tokens, asked for \(count)")
            let stage = start % max(1, stageRows)
            precondition(stage + count <= max(1, stageRows),
                         "staging range \(start)..<\(start + count) wraps")
            return (kStageBuffers[layer]!, stage * strides[layer], strides[layer])
        }
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (kBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func vRange(layer: Int, start: Int, count: Int) -> (buffer: MTLBuffer, offset: Int, stride: Int) {
        validateRange(start: start, count: count)
        if usesInt8Storage(layer: layer) {
            precondition(count <= max(1, stageRows),
                         "staging holds \(stageRows) tokens, asked for \(count)")
            let stage = start % max(1, stageRows)
            precondition(stage + count <= max(1, stageRows),
                         "staging range \(start)..<\(start + count) wraps")
            return (vStageBuffers[layer]!, stage * strides[layer], strides[layer])
        }
        validateContiguousPhysicalRange(layer: layer, start: start, count: count)
        return (vBuffers[layer], physicalSlot(layer: layer, position: start) * strides[layer], strides[layer])
    }

    public func keyView(layer: Int) -> KVView {
        keyView(layer: layer, validTokenCount: position)
    }

    public func keyView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: kBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    public func valueView(layer: Int) -> KVView {
        valueView(layer: layer, validTokenCount: position)
    }

    func keyBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        keyView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    func valueBuffer(layer: Int, validTokenCount: Int) -> MTLBuffer {
        valueView(layer: layer, validTokenCount: validTokenCount).buffer
    }

    public func valueView(layer: Int, validTokenCount: Int) -> KVView {
        validateValidTokenCount(validTokenCount)
        return KVView(buffer: vBuffers[layer], offset: 0, stride: strides[layer],
                      validTokenCount: validTokenCount, startSlot: ringStartSlot(layer: layer,
                                                                                 validTokenCount: validTokenCount))
    }

    /// Advance the position cursor once the current token's K/V are written
    /// across all layers.
    public func advance() { advance(by: 1) }

    public func advance(by count: Int) {
        precondition(count >= 0, "advance count must be non-negative")
        precondition(position + count <= maxContext, "advance would exceed maxContext")
        position += count
    }

    /// Drop all cached positions and return physical pages to the OS.
    ///
    /// No buffer zeroing — the attention kernels read only `[0, validTokenCount]`,
    /// and `validTokenCount` is now 0. `MADV_DONTNEED` on the page-aligned span
    /// releases resident memory between turns; pages fault back in on next write.
    public func reset() {
        position = 0
        let pageSize = Int(getpagesize())
        var advised = Set<ObjectIdentifier>()
        for layer in 0..<config.numLayers {
            advise(kBuffers[layer], pageSize: pageSize, seen: &advised)
            advise(vBuffers[layer], pageSize: pageSize, seen: &advised)
            if let buffer = kInt8Buffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
            if let buffer = vInt8Buffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
            if let buffer = kScaleBuffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
            if let buffer = vScaleBuffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
            if let buffer = kStageBuffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
            if let buffer = vStageBuffers[layer] { advise(buffer, pageSize: pageSize, seen: &advised) }
        }
    }

    private func validateRange(start: Int, count: Int) {
        precondition(count >= 0, "count must be non-negative")
        precondition(start >= 0, "start must be non-negative")
        precondition(start + count <= maxContext,
                     "range \(start)..<\(start + count) exceeds maxContext \(maxContext)")
    }

    private func validateValidTokenCount(_ count: Int) {
        precondition(count >= 0, "validTokenCount must be non-negative")
        precondition(count <= maxContext,
                     "validTokenCount \(count) exceeds maxContext \(maxContext)")
    }

    private func physicalSlot(layer: Int, position: Int) -> Int {
        let capacity = capacityTokens[layer]
        guard capacity > 0 else { return 0 }   // linear-attention layer: no storage
        return position % capacity
    }

    private func ringStartSlot(layer: Int, validTokenCount: Int) -> Int {
        guard fp16RingEnabled, kinds[layer] == .swa else { return 0 }
        let capacity = capacityTokens[layer]
        guard validTokenCount > capacity else { return 0 }
        return validTokenCount % capacity
    }

    private func validateContiguousPhysicalRange(layer: Int, start: Int, count: Int) {
        guard count > 0, fp16RingEnabled, kinds[layer] == .swa else { return }
        let capacity = capacityTokens[layer]
        let physicalStart = start % capacity
        precondition(physicalStart + count <= capacity,
                     "range \(start)..<\(start + count) wraps FP16 KV ring capacity \(capacity)")
    }

    private func advise(_ buffer: MTLBuffer, pageSize: Int, seen: inout Set<ObjectIdentifier>) {
        let id = ObjectIdentifier(buffer)
        if seen.contains(id) { return }
        seen.insert(id)
        // MTLBuffer allocations are page-aligned; round the length down to a
        // whole number of pages so we never hand madvise a partial tail page.
        let len = (buffer.length / pageSize) * pageSize
        if len > 0 {
            _ = posix_madvise(buffer.contents(), len, POSIX_MADV_DONTNEED)
        }
    }
}

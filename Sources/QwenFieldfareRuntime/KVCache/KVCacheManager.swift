import Foundation
import Metal
import QwenFieldfareFormat

/// Linear (non-ring) KV cache for Qwen3. Every layer uses full causal
/// attention, so storage is a simple `[maxSeqLen, numKVHeads, headDim]` fp16
/// buffer per layer for K and for V.
///
/// Memory at 32K context: 48 layers × 32768 tokens × (4 heads × 128 dim × 2 B)
/// × 2 (K+V) ≈ 3.22 GB. Buffers are allocated lazily via `.storageModeShared`
/// so CPU-side `append` and GPU-side attention both see the same memory.
public final class KVCacheManager {

    public let device: MTLDevice
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let maxSeqLen: Int

    /// Bytes per token per layer = numKVHeads * headDim * sizeof(fp16).
    public let tokenStride: Int

    private var kBuffers: [MTLBuffer]
    private var vBuffers: [MTLBuffer]

    /// Current number of valid tokens (positions 0..<count).
    public private(set) var count: Int = 0

    public init(device: MTLDevice,
                config: QTurboModelConfig,
                maxSeqLen: Int = 32768) throws {
        self.device = device
        self.numLayers = config.numHiddenLayers
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.maxSeqLen = maxSeqLen
        self.tokenStride = config.numKeyValueHeads * config.headDim * MemoryLayout<UInt16>.size

        let perLayerBytes = tokenStride * maxSeqLen
        var ks: [MTLBuffer] = []
        var vs: [MTLBuffer] = []
        ks.reserveCapacity(numLayers)
        vs.reserveCapacity(numLayers)
        for l in 0..<numLayers {
            guard let k = device.makeBuffer(length: perLayerBytes, options: .storageModeShared),
                  let v = device.makeBuffer(length: perLayerBytes, options: .storageModeShared) else {
                throw KVError.allocationFailed(layer: l, bytes: perLayerBytes)
            }
            k.label = "kv.K.layer\(l)"
            v.label = "kv.V.layer\(l)"
            ks.append(k)
            vs.append(v)
        }
        self.kBuffers = ks
        self.vBuffers = vs
    }

    public enum KVError: Error, CustomStringConvertible {
        case allocationFailed(layer: Int, bytes: Int)
        case outOfRange(position: Int, max: Int)

        public var description: String {
            switch self {
            case .allocationFailed(let l, let b):
                return "KVCache: allocation failed for layer \(l) (\(b) bytes)"
            case .outOfRange(let p, let m):
                return "KVCache: position \(p) out of range (max \(m))"
            }
        }
    }

    /// Resets the cache for a new sequence (does not free memory).
    public func reset() { count = 0 }

    /// Advances the logical token count after all layers have appended a token
    /// at `position`. Call once per generated token.
    public func advance(to position: Int) {
        count = max(count, position + 1)
    }

    public func kBuffer(layer: Int) -> MTLBuffer { kBuffers[layer] }
    public func vBuffer(layer: Int) -> MTLBuffer { vBuffers[layer] }

    /// Byte offset of a given token position within a layer's K/V buffer.
    public func byteOffset(position: Int) -> Int { position * tokenStride }

    /// Writes `keys` and `values` (each numKVHeads*headDim fp16 elements) at
    /// `position` for `layer`. Pointers must reference fp16 data.
    public func append(layer: Int,
                       position: Int,
                       keys: UnsafeRawPointer,
                       values: UnsafeRawPointer) throws {
        guard position >= 0 && position < maxSeqLen else {
            throw KVError.outOfRange(position: position, max: maxSeqLen)
        }
        let off = byteOffset(position: position)
        memcpy(kBuffers[layer].contents().advanced(by: off), keys, tokenStride)
        memcpy(vBuffers[layer].contents().advanced(by: off), values, tokenStride)
    }

    /// Returns the K and V buffers plus the valid length for attention over the
    /// range `[0, upTo)`. The buffers are the full per-layer buffers; the caller
    /// uses `length` to bound the causal attention scan.
    public func slice(layer: Int, upTo length: Int) -> (k: MTLBuffer, v: MTLBuffer, length: Int) {
        let n = min(length, maxSeqLen)
        return (kBuffers[layer], vBuffers[layer], n)
    }

    /// Total bytes currently allocated across all layers (K + V).
    public var allocatedBytes: Int {
        tokenStride * maxSeqLen * numLayers * 2
    }
}

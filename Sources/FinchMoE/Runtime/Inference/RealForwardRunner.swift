import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.finch` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only post-FFN norm; nil on Qwen 3.6 (no sandwich).
        let postF1: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail
    // Qwen 3.6 kernels (pipelines are in the shared library for both families).
    private let gdn: GDN
    private let gdnPrefill: GDNPrefill
    private let qwenFusions: QwenDecodeFusions
    // Qwen 3.8 Flash-Next hyper-connection (mix/combine/plane-init kernels).
    private let hyperConnection: HyperConnection

    // Qwen 3.8 Flash-Next PLE n-gram head — layer `pleLayerIndex` only, and
    // nil on every other family. The four device kernels live in `ple`; the
    // routing that decides *which* 16 rows of the 102.4 GB table this token
    // reads is host-side (`pleHost`), because the hash is 64-bit wrap
    // arithmetic and the table is pread'd, not resident.
    private let ple: PLE?
    private let pleHost: PLEHost?
    private let pleGathered: MTLBuffer      // fp16 [gatheredWidth]
    private let pleKey: MTLBuffer           // fp16 [hc·D]  = key_proj · gathered
    private let pleKeyNormed: MTLBuffer     // fp16 [hc·D]
    private let pleQueryNormed: MTLBuffer   // fp16 [hc·D]  = normed copy of the plane
    private let pleValue: MTLBuffer         // fp16 [D]     = value_proj · gathered
    private let pleGate: MTLBuffer          // fp32 [hc]    — one gate per stream
    private let pleGated: MTLBuffer         // fp16 [hc·D]
    private let pleConvIn: MTLBuffer        // fp16 [hc·D]  = normed gated value
    private let pleConvOut: MTLBuffer       // fp16 [hc·D]
    private let pleConvState: MTLBuffer     // fp16 [(kern-1)·dil, hc·D]
    private let pleConvStateNext: MTLBuffer // fp16 [same]  — the roll's destination

    // GDN linear-attention projections (in_proj_qkv/z/a/b, out_proj): the
    // manifest linearAttention slot is 4 on Gemma / legacy installs and 8 on
    // the production Qwen build. At 8 the decode gate decomposes onto two
    // int8 GEMVs + the batch-gate kernel at T=1, and prefill runs repeated
    // per-token int8 GEMVs (no batched int8 QMM exists yet).
    private let linearAttnBits: Int
    private let int8GEMV: DequantInt8GEMV?
    private let gateAB: MTLBuffer?   // fp16 [2V] in_proj_a|b decode scratch

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    // Qwen 3.6 GDN state + scratch (all empty/1-element for Gemma installs).
    /// Per-GDN-layer recurrent state, fp32 [V][headDim][headDim], v-major —
    /// 2 MiB per layer, zeroed in `reset()`. Indexed via `gdnStateIndexByLayer`.
    private let gdnRecurrentState: [MTLBuffer]
    /// Per-GDN-layer causal-conv state, fp16 [qkvDim, 3], updated in place.
    private let gdnConvState: [MTLBuffer]
    /// Fused in_proj_a|in_proj_b int4-affine weights assembled at init
    /// (a-rows then b-rows) + scales/biases — the kernel reads one block.
    private let gdnGateWeights: [(weights: MTLBuffer, scales: MTLBuffer, biases: MTLBuffer)]
    /// Layer → index into the three arrays above.
    private let gdnStateIndexByLayer: [Int]
    private let qkvConv: MTLBuffer              // [qkvDim] fp16 conv in/out
    private let zBuf: MTLBuffer                 // [V*headDim] fp16 in_proj_z out
    private let gBeta: MTLBuffer                // [2V] fp32 g | beta
    private let qGateBuf: MTLBuffer             // [2*Q*fullHeadDim] fp16 q|gate
    private let gateBuf: MTLBuffer              // [Q*fullHeadDim] fp16 gate half
    /// BF16 ones buffers: Qwen's router has no router.scale / per_expert_scale.
    private let qwenOnesEffectiveScale: MTLBuffer?
    private let qwenOnesPerExpertScale: MTLBuffer?
    /// Qwen 3.8 hyper-connection plane + mixer scratch (empty for other
    /// families). The plane `hcPlane` ([hc·D] fp16, 4 streams of 2560) is the
    /// cross-layer residual: rebuilt from the embedding every decode step and
    /// updated in place by each layer's two combines. `hcXn`/`hcGateRaw`/
    /// `hcGated` ([hc·D]) and `hcLo` ([lowrank]) are the grouped-RMS → silu →
    /// sigmoid-gate mixer stages; `hcInject` ([hc]) carries the current
    /// mixer's raw block_inject row dot into the combine. One mixer's scratch
    /// is reused by the next — the serial kernel order inside a CB is the only
    /// ordering the stages need.
    private let hcPlane: MTLBuffer
    private let hcXn: MTLBuffer
    private let hcGateRaw: MTLBuffer
    private let hcGated: MTLBuffer
    private let hcLo: MTLBuffer
    private let hcInject: MTLBuffer
    /// Qwen 3.8 Flash-Next QSA indexer: the kernels and the per-full-layer
    /// key timelines they keep. Nil for every other family and for a Qwen 3.8
    /// install whose manifest carries no indexer, in which case the full
    /// layers keep the dense `encodeFull` path.
    private let qsaIndexer: QSAIndexer?
    private let qsaState: QSAIndexerState?
    /// Internal test hook: receives (layer, phase, values) snapshots —
    /// "preLayer" (the layer input hidden, before any compute), "postAttn"
    /// (the post_attention_layernorm output, pre-MoE) after the layer-head CB
    /// completes, and "postLayer" (the final hidden) after the tail. Nil in
    /// production — the copies only run when a hook is installed.
    internal var qwenLayerDebugHook: ((Int, String, [Float16]) -> Void)? = nil

    /// `FQ_QSA_DUMP=<path>`: append one line per step and full-attention layer
    /// describing what the QSA ranking selected — the cell count, the pooled
    /// block count, and a hash of the selected cell indices.
    ///
    /// This exists because the ranking path is where the engine stops being
    /// bit-reproducible (past `indexerBudget + r − 1` = 2051 tokens on this
    /// model), and the aggregate is the wrong instrument for finding it: two
    /// runs differ in ~99% of their logits, which says the divergence happened
    /// *somewhere earlier*. A fingerprint per step and layer turns that into
    /// the first step and layer that actually moved, which is what a kernel
    /// investigation needs.
    ///
    /// Two phases are dumped. `decode` records come from the end of
    /// `produceToken`, after its wait, which is the only point where the
    /// selection is stable. `prefill` records come from the chunk loop in
    /// `prefillChunked`, after a drain, and describe each full layer's
    /// selection for that chunk's *last* row — the prefill has no finer safe
    /// point, since its work is otherwise in flight. The decode records placed
    /// the divergence in the prefill (it is already present at the first decode
    /// step), so the prefill records are the ones that can narrow it further.
    private var qsaDumpPath: String? = nil

    // MARK: Row fingerprints (FQ_ROW_HASH)

    /// The row-fingerprint instrument, off unless `FQ_ROW_HASH` names a path.
    /// `rowHashLayout` sizes its side buffer; both are nil when it is off, and
    /// every use site guards on them, so the default path pays nothing.
    private var rowHash: PrefillRowHash? = nil
    private var rowHashBuffer: MTLBuffer? = nil
    internal private(set) var rowHashLayout: RowHashLayout? = nil
    private var rowHashDumpPath: String? = nil
    /// `FQ_GDN_SPLIT=1`: commit the GDN sub-stages as separate command buffers
    /// so each reports its own GPU time. See `totalGpuGdnProjNanos`.
    private var gdnSubStageSplit = false
    /// Batch the int8 projections the GDN layers spend their time in, instead
    /// of re-walking the weight matrix once per token. On by default;
    /// `FQ_INT8_GEMM=0` restores the per-token GEMV. See `PrefillInt8Gemm`.
    private var int8ProjectionGemm = true
    private var prefillInt8Gemm: PrefillInt8Gemm?
    /// The last prefill's row count, for the dump's shape.
    private var rowHashRowCount: Int = 0

    /// Fingerprint the whole residual plane, one hash per row, at one stage of
    /// one layer. One small dispatch per (layer, stage) — no wait, no readback,
    /// nothing that changes when the work lands, because the timing of the run
    /// is itself the variable under investigation.
    /// One row-fingerprint dispatch over an arbitrary buffer. `rowBytes` is what
    /// is hashed from each row and `rowStrideBytes` what separates them, so a
    /// caller can fingerprint a plane row (hcDim halves), an attention output
    /// (qDim) or the QSA selection (capacity UInt32s) with the same call.
    private func encodeRowHash(_ src: MTLBuffer,
                               layer L: Int, stage: Int,
                               rowCount: Int, rowBase: Int,
                               rowStrideBytes: Int,
                               srcRowBase: Int = 0,
                               rowBytes: Int? = nil,
                               into target: MTLCommandBuffer) {
        guard let rowHash, let dst = rowHashBuffer, let layout = rowHashLayout,
              rowCount > 0 else { return }
        rowHash.encode(commandBuffer: target,
                       src: src,
                       dst: dst,
                       dstOffsetBytes: layout.offsetBytes(layer: L, stage: stage),
                       dstRowBase: rowBase,
                       srcRowBase: srcRowBase,
                       rowCount: UInt32(rowCount),
                       rowStrideBytes: UInt32(rowStrideBytes),
                       rowBytes: UInt32(rowBytes ?? rowStrideBytes))
    }

    private func encodePlaneRowHash(_ scratch: PrefillChunkScratchBuffers,
                                    layer L: Int, stage: Int, tokenCount: Int,
                                    rowBase: Int,
                                    into target: MTLCommandBuffer) {
        encodeRowHash(scratch.qwen38Plane, layer: L, stage: stage,
                      rowCount: tokenCount, rowBase: rowBase,
                      rowStrideBytes: cfg.hyperConnectionDim * MemoryLayout<Float16>.stride,
                      into: target)
    }

    /// Write the side buffer as raw little-endian `UInt64`s, laid out
    /// `[layer][stage][row]` (see `RowHashLayout`), preceded by a 16-byte
    /// header of `[magic, layerCount, stageCount, rowCount]` as `UInt32`s.
    ///
    /// Best-effort, like the other diagnostics: a failure here must not break
    /// the run it is diagnosing.
    ///
    /// The name matches `LogitProducer.dumpRowHashes` deliberately. It was
    /// `writeRowHashDump` once, and the protocol's default no-op implementation
    /// satisfied the call site silently — the instrument ran, hashed, and
    /// dumped nothing, with no error anywhere.
    /// Whether the instrument is on. `RawCompletion` needs this to know whether
    /// to drain at the prefill/decode boundary when no logits dump was asked
    /// for -- the row fingerprints live on the GPU and are read on the host.
    public var wantsRowHashesDump: Bool { rowHashDumpPath != nil }

    public func dumpRowHashes() {
        guard let path = rowHashDumpPath, let dst = rowHashBuffer,
              let layout = rowHashLayout, rowHashRowCount > 0 else { return }
        let stride = MemoryLayout<UInt64>.stride
        var data = Data(capacity: 16 + layout.totalBytes)
        for value in [UInt32(0x5248_4831), UInt32(layout.layerCount),
                      UInt32(PrefillRowHash.stageCount), UInt32(rowHashRowCount)] {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        let ptr = dst.contents().bindMemory(to: UInt64.self,
                                            capacity: layout.totalBytes / stride)
        for L in 0..<layout.layerCount {
            for stage in 0..<PrefillRowHash.stageCount {
                for row in 0..<rowHashRowCount {
                    let value = ptr[layout.elementIndex(layer: L, stage: stage, row: row)]
                    withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
                }
            }
        }
        do {
            try data.write(to: URL(fileURLWithPath: path))
            // One line a reader can diff before opening the file: a digest over
            // every hash. Two runs that agree here agree everywhere in it.
            var digest: UInt64 = 0xcbf2_9ce4_8422_2325
            for L in 0..<layout.layerCount {
                for stage in 0..<PrefillRowHash.stageCount {
                    for row in 0..<rowHashRowCount {
                        digest = (digest ^ ptr[layout.elementIndex(layer: L, stage: stage,
                                                                   row: row)])
                            &* 0x0000_0100_0000_01b3
                    }
                }
            }
            let line = "row_hash: layers=\(layout.layerCount)"
                + " stages=\(PrefillRowHash.stageCount) rows=\(rowHashRowCount)"
                + " digest=\(String(digest, radix: 16)) path=\(path)\n"
            FileHandle.standardError.write(Data(line.utf8))
        } catch {
            let line = "row_hash: failed to write \(path): \(error)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    /// Best-effort; a diagnostic must never break the run it is diagnosing.
    private func dumpQSASelection(position: Int, phase: String) {
        guard let path = qsaDumpPath, let st = qsaState else { return }
        var lines = ""
        for L in 0..<cfg.numLayers {
            guard let li = st.index(ofLayer: L) else { continue }
            let lay = st.layers[li]
            let written = Int(lay.cellCount.contents()
                .bindMemory(to: UInt32.self, capacity: 1).pointee)
            let count = min(written, st.capacity)
            let ptr = lay.cells.contents()
                .bindMemory(to: UInt32.self, capacity: max(st.capacity, 1))
            // FNV-1a over the selected indices: order-sensitive, and the list
            // is ascending by construction, so a single swapped cell shows.
            var h: UInt64 = 0xcbf2_9ce4_8422_2325
            for i in 0..<count {
                h = (h ^ UInt64(ptr[i])) &* 0x0000_0100_0000_01b3
            }
            lines += "\(phase) step=\(position) layer=\(L) cells=\(count)"
                + " pooled=\(lay.pooledBlocks) hash=\(String(h, radix: 16))\n"
        }
        guard !lines.isEmpty, let data = lines.data(using: .utf8) else { return }
        if let fh = FileHandle(forWritingAtPath: path) {
            fh.seekToEndOfFile()
            fh.write(data)
            try? fh.close()
        } else {
            try? data.write(to: URL(fileURLWithPath: path))
        }
    }

    /// Test/debug: the QSA cells a full layer's attention read on the last
    /// decode step, and how many blocks of its indexer timeline are pooled.
    /// Nil when the layer carries no indexer. Valid once the step's command
    /// buffer has completed, which `produce` guarantees on return.
    ///
    /// An empty list is not an error: while the context is inside the
    /// selection width the ranking never runs and attention takes the dense
    /// path, reading every causally-visible cell — which is exactly what a
    /// zero `cellCount` records.
    internal func qsaSelection(layer L: Int) -> (cells: [UInt32], pooledBlocks: Int)? {
        guard let st = qsaState, let li = st.index(ofLayer: L) else { return nil }
        let lay = st.layers[li]
        let written = Int(lay.cellCount.contents()
            .bindMemory(to: UInt32.self, capacity: 1).pointee)
        let count = min(written, st.capacity)
        let ptr = lay.cells.contents()
            .bindMemory(to: UInt32.self, capacity: max(st.capacity, 1))
        return (Array(UnsafeBufferPointer(start: ptr, count: count)),
                lay.pooledBlocks)
    }
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    /// The prefill's routed-expert tile pipeline. Instance state, not a static,
    /// because the depth and tile width are runtime settings: they set how much
    /// of the prefill's I/O can overlap its compute, and the default single tile
    /// of lookahead is what caps the drive's duty cycle (see
    /// `RuntimeConfiguration.prefillTileDepth`).
    private let prefillRoutedTileSchedulerConfig: PrefillRoutedTileSchedulerConfig

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) folds
    /// RMSNorm + lm_head into one kernel that skips the 512 KB logits write and
    /// leaves a greedy argmax in `lastGreedyToken`; callers that sample from the
    /// logits buffer (non-greedy configs) must pass `forceLogitsHead: true` or
    /// they read a never-written buffer.
    ///
    /// **Unavailable on Qwen 3.8**, whose head input is the root HC mixer
    /// collapse — a data-dependent gate, not a plain norm — so the kernel cannot
    /// express it and the logits path (`prefillFinalRowHead.encodeLogits` /
    /// `gFinalNorm` + `gLmHead`) runs instead. The arch check lives *here*, on
    /// the flag, rather than at the kernel-selection site alone: `greedyTokenBuf`
    /// is written only by the fused kernel, so every site that reads it has to
    /// agree with the site that decides whether to run it. They did not, and
    /// Qwen 3.8 at temperature 0 read a never-written buffer — zero-filled —
    /// and emitted token 0 forever. Keep this the single source of truth.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0

    /// The rule behind `useFusedGreedyHead`, split out so it can be tested
    /// without building a 125B install — the mistake it guards against was
    /// invisible for exactly that reason.
    static func fusedGreedyHeadEnabled(headPath: RuntimeHeadPath,
                                       config: ArchConfig) -> Bool {
        headPath == .fusedRows && !config.isQwen3_8
    }

    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        self.model = model
        self.ctx = context
        self.cfg = model.config
        self.maxContext = maxContext
        self.useFusedGreedyHead = Self.fusedGreedyHeadEnabled(
            headPath: runtimeConfiguration.headPath, config: model.config)
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        self.prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig(
            maxPendingDepth: runtimeConfiguration.prefillTileDepth,
            tileExperts: runtimeConfiguration.prefillTileExperts)
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(context: context)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits)
        self.moe       = try MoE(context: context)
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.gdn = try GDN(context: context)
        self.gdnPrefill = try GDNPrefill(context: context)
        self.qwenFusions = try QwenDecodeFusions(context: context)
        self.hyperConnection = try HyperConnection(context: context)
        self.linearAttnBits = model.linearAttentionWeightBits
        self.int8GEMV = model.linearAttentionWeightBits == 8
            ? try DequantInt8GEMV(context: context) : nil
        // gateAB is allocated with the Qwen state below (linearAttn == 8).
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        let device = context.device
        let D = cfg.hiddenSize
        let F = cfg.intermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok

        func sharedProj(_ view: TensorView, rows: UInt32, cols: UInt32) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: rows,
                                 cols: cols)
        }
        // Qwen hybrid (3.6 + 3.8): no post-FFN1 norm tensor, the router
        // 1/sqrt(D) fold already happened at repack, and GDN recurrent state
        // is runner-side. Gemma paths leave all of these empty / 1.0.
        let isQwen = cfg.isQwenHybrid
        var sharedViews: [LayerSharedExpertProjections] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate, rows: UInt32(F), cols: UInt32(D)),
                up: sharedProj(up, rows: UInt32(F), cols: UInt32(D)),
                down: sharedProj(down, rows: UInt32(D), cols: UInt32(F)),
                postF1: isQwen ? nil : try model.postFFN1(layer: L)))
        }
        self.sharedExpertProjections = sharedViews

        // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets its
        // own BF16 [D] buffer — the kernel reads `effective_scale[i]` and we
        // pay for the multiply once per generation, not per token.
        // Qwen's router applies no input scale (plain softmax): every layer
        // shares one BF16-ones buffer.
        var perLayer: [MTLBuffer] = []
        perLayer.reserveCapacity(cfg.numLayers)
        let invSqrtD = Float(1.0) / Float(D).squareRoot()
        let dInts = D
        for L in 0..<cfg.numLayers {
            guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            if isQwen {
                for i in 0..<dInts { dst[i] = Quantization.bf16Bits(1.0) }
            } else {
                let scaleView = try model.routerScale(layer: L)
                let src = scaleView.buffer.contents()
                    .advanced(by: Int(scaleView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<dInts {
                    let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                    dst[i] = Quantization.bf16Bits(v)
                }
            }
            buf.label = "effective_scale.L\(L)"
            perLayer.append(buf)
        }
        self.effectiveScaleBuffers = perLayer

        // MARK: Qwen 3.6 state + scratch (all Gemma paths leave these empty).
        if isQwen {
            let v = cfg.linearNumValueHeads
            let headDim = cfg.linearValueHeadDim
            let qkvDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim * 2
                + v * headDim
            let fullQ = cfg.numHeads * cfg.fullHeadDim
            let groupCount = Quantization.groupSize

            func zeros(_ count: Int, _ stride: Int, label: String) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                                options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                memset(b.contents(), 0, b.length)
                b.label = label
                return b
            }
            self.qkvConv  = try buf(max(qkvDim, 1))
            self.zBuf     = try buf(max(v * headDim, 1))
            self.qGateBuf = try buf(max(2 * fullQ, 1))
            self.gateBuf  = try buf(max(fullQ, 1))
            self.gBeta    = try zeros(2 * v, MemoryLayout<Float>.size,
                                      label: "qwen.g_beta")
            self.gateAB   = linearAttnBits == 8
                ? try buf(max(2 * v, 1)) : nil

            var recState: [MTLBuffer] = []
            var convState: [MTLBuffer] = []
            var gateWeights: [(MTLBuffer, MTLBuffer, MTLBuffer)] = []
            var stateIndex: [Int] = Array(repeating: -1, count: cfg.numLayers)
            for L in 0..<cfg.numLayers where cfg.fullAttentionLayerMask[L] == 0 {
                stateIndex[L] = recState.count
                recState.append(try zeros(
                    v * headDim * headDim, MemoryLayout<Float>.size,
                    label: "gdn_recurrent_state.L\(L)"))
                convState.append(try zeros(
                    qkvDim * 3, MemoryLayout<Float16>.size,
                    label: "gdn_conv_state.L\(L)"))

                // The fused in_proj_a|in_proj_b int4-affine block (a-rows
                // then b-rows) is only assembled for the 4-bit decode gate
                // kernel; at 8 bits in_proj_a/b stay separate resident
                // tensors and the decode gate decomposes onto two int8 GEMVs
                // + the batch-gate kernel (T=1).
                if linearAttnBits == 4 {
                    // TensorView offsets are buffer-relative; copy the ranges
                    // once at init.
                    let aView = try model.gdnInProjA(layer: L)
                    let bView = try model.gdnInProjB(layer: L)
                    let rowBytes = D / 2
                    let groups = D / groupCount
                    let auxBytes = v * groups * MemoryLayout<UInt16>.size
                    guard let wBuf = device.makeBuffer(
                            length: 2 * v * rowBytes,
                            options: .storageModeShared),
                          let sBuf = device.makeBuffer(
                            length: 2 * auxBytes,
                            options: .storageModeShared),
                          let bBuf = device.makeBuffer(
                            length: 2 * auxBytes,
                            options: .storageModeShared) else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    func residentBytes(_ view: TensorView, _ offset: UInt64) -> UnsafeMutableRawPointer {
                        view.buffer.contents().advanced(by: Int(offset))
                    }
                    func copy(_ view: TensorView, _ offset: UInt64, to dst: UnsafeMutableRawPointer, len: Int) {
                        memcpy(dst, residentBytes(view, offset), len)
                    }
                    let wDst = wBuf.contents()
                    copy(aView, aView.offset, to: wDst, len: v * rowBytes)
                    copy(bView, bView.offset,
                         to: wDst.advanced(by: v * rowBytes), len: v * rowBytes)
                    let sDst = sBuf.contents()
                    copy(aView, aView.scaleOffset, to: sDst, len: auxBytes)
                    copy(bView, bView.scaleOffset,
                         to: sDst.advanced(by: auxBytes), len: auxBytes)
                    let bDst = bBuf.contents()
                    copy(aView, aView.biasOffset, to: bDst, len: auxBytes)
                    copy(bView, bView.biasOffset,
                         to: bDst.advanced(by: auxBytes), len: auxBytes)
                    gateWeights.append((wBuf, sBuf, bBuf))
                }
            }
            self.gdnRecurrentState = recState
            self.gdnConvState = convState
            self.gdnGateWeights = gateWeights
            self.gdnStateIndexByLayer = stateIndex

            func ones(_ count: Int, label: String) throws -> MTLBuffer {
                guard let b = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                                options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                let ptr = b.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count { ptr[i] = Quantization.bf16Bits(1.0) }
                b.label = label
                return b
            }
            self.qwenOnesEffectiveScale = try ones(D, label: "qwen.ones_effective_scale")
            self.qwenOnesPerExpertScale = try ones(cfg.numExperts, label: "qwen.ones_per_expert_scale")

            // Qwen 3.8 Flash-Next: the wide hyper-connection residual plane and
            // its per-mixer scratch. The plane is rebuilt from the embedding
            // every decode step (`hc_plane_init` in produceToken), so nothing
            // here needs zeroing in reset() — unlike the GDN states above.
            if cfg.isQwen3_8 {
                let hcDim = cfg.hyperConnectionDim       // hc·D = 10240
                let hcCount = cfg.hyperConnectionCount   // 4 streams
                let lowrank = cfg.hyperConnectionLowrank // 320
                self.hcPlane   = try zeros(hcDim, 2, label: "qwen38.hc_plane")
                self.hcXn      = try zeros(hcDim, 2, label: "qwen38.hc_xn")
                self.hcGateRaw = try zeros(hcDim, 2, label: "qwen38.hc_gate_raw")
                self.hcGated   = try zeros(hcDim, 2, label: "qwen38.hc_gated")
                self.hcLo      = try zeros(max(lowrank, 1), 2, label: "qwen38.hc_lo")
                self.hcInject  = try zeros(max(hcCount, 1), 2,
                                           label: "qwen38.hc_inject")
            } else {
                self.hcPlane   = try buf(1)
                self.hcXn      = try buf(1)
                self.hcGateRaw = try buf(1)
                self.hcGated   = try buf(1)
                self.hcLo      = try buf(1)
                self.hcInject  = try buf(1)
            }

            // Qwen 3.8 Flash-Next: the QSA indexer's per-full-layer key
            // timelines. Only the raw and pooled keys persist across steps —
            // they are the indexer's own cache — so this is the one Qwen 3.8
            // structure `reset()` has to clear.
            //
            // `FQ_QSA_OFF=1` builds the runner without the sparse-block
            // selector, keeping the dense attention path. It is the mechanism
            // that isolates the ranking dispatches from plain context length
            // when a long run turns out not to be bit-reproducible — and, since
            // `validateQwen38Layers` requires the indexer tensors on every full
            // layer unconditionally, it is the *only* way to reach a runner
            // that has full layers but no indexer state. (An earlier comment
            // here claimed a 3.8 repack without indexer tensors loads this way.
            // It does not: that install fails validation with `tensorNotFound`
            // before any runner exists.)
            self.qsaDumpPath = ProcessInfo.processInfo
                .environment["FQ_QSA_DUMP"]
            // The row-fingerprint instrument. Rows are positions in the
            // sequence, not chunk slots (each chunk reuses the same plane rows,
            // so a per-chunk index would let the last chunk erase the rest), so
            // the side buffer is sized by context: 8 bytes x 3 stages x 48
            // layers is 1.1 KiB per position, about 9 MiB at 8k and 75 MiB at
            // 64k. It is opt-in, and the dump reports the cost at startup.
            self.gdnSubStageSplit = ProcessInfo.processInfo
                .environment["FQ_GDN_SPLIT"] == "1"
            // On by default: the batched kernel replaces one GEMV dispatch per
            // token with a tiled one that reuses the weight tile across the
            // token dimension, worth 82.6 s -> 22.3 s of projection work on a
            // 2940-token prefill. `FQ_INT8_GEMM=0` restores the per-token GEMV,
            // which is also what a shape the tile cannot cover falls back to.
            self.int8ProjectionGemm = ProcessInfo.processInfo
                .environment["FQ_INT8_GEMM"] != "0"
            self.prefillInt8Gemm = int8ProjectionGemm
                ? try PrefillInt8Gemm(context: ctx) : nil
            self.rowHashDumpPath = ProcessInfo.processInfo
                .environment["FQ_ROW_HASH"]
            if let path = rowHashDumpPath, cfg.isQwen3_8 {
                let layout = RowHashLayout(maxRows: maxContext,
                                           layerCount: cfg.numLayers)
                self.rowHashLayout = layout
                self.rowHash = try PrefillRowHash(context: ctx)
                self.rowHashBuffer = try ctx.device.makeBuffer(
                    length: layout.totalBytes, options: .storageModeShared)
                if let buf = rowHashBuffer {
                    // Shared storage, so the dump is a host read of bytes the
                    // GPU wrote; the caller drains before reading (see
                    // `dumpRowHashes`).
                    memset(buf.contents(), 0, layout.totalBytes)
                }
                FileHandle.standardError.write(Data((
                    "row_hash: instrument on, \(layout.layerCount) layers x "
                    + "\(PrefillRowHash.stageCount) stages x \(layout.maxRows) rows = "
                    + "\(layout.totalBytes) bytes -> \(path)\n").utf8))
            }
            // Two ways in, one state: the documented `FQ_QSA_OFF=1` control,
            // and the injectable `qsaIndexerEnabled: false` a test uses (see
            // its doc comment — no install can express this).
            let qsaDisabled = !runtimeConfiguration.qsaIndexerEnabled
                || ProcessInfo.processInfo.environment["FQ_QSA_OFF"] == "1"
            if !qsaDisabled,
               let state = try QSAIndexerState(device: device, config: cfg,
                                               maxContext: maxContext) {
                self.qsaIndexer = try QSAIndexer(context: ctx)
                self.qsaState = state
            } else {
                self.qsaIndexer = nil
                self.qsaState = nil
            }

            // Qwen 3.8 Flash-Next PLE n-gram head. The host hash needs its
            // three I64 constants up front; a 3.8 install whose PLE is
            // malformed traps in the initializer rather than routing every
            // token off the end of the table. Fetching them is family-gated:
            // the accessors behind `pleHashConstants()` are themselves
            // family-guarded and raise `tensorNotFound` on a model that has
            // no PLE at all, so a 3.6 install must never reach them.
            var pleConstants: (multipliers: [UInt64], headOffsets: [UInt64],
                               headVocabSizes: [UInt64])? = nil
            if cfg.isQwen3_8 { pleConstants = try model.pleHashConstants() }
            // `FQ_PLE_QUANT_SIM=<groupSize>`: decode every raw-BF16 PLE row as
            // if the table had been quantized to that group size, with the
            // install left alone. Phase 5.1's isolation knob — one variable
            // between two otherwise identical runs.
            let pleQuantSimulation = ProcessInfo.processInfo
                .environment["FQ_PLE_QUANT_SIM"].flatMap(Int.init)
            if let c = pleConstants,
               let host = try PLEHost(config: cfg,
                                      multipliers: c.multipliers,
                                      headOffsets: c.headOffsets,
                                      headVocabSizes: c.headVocabSizes,
                                      quantizationSimulation: pleQuantSimulation) {
                self.pleHost = host
                self.ple = try PLE(context: ctx)
                let hcDim = cfg.hyperConnectionDim
                let hcCount = cfg.hyperConnectionCount
                let hist = (cfg.pleConvKernelSize - 1) * cfg.ngramSize
                let gathered = cfg.ngramRowDim
                    * (cfg.ngramSize - 1) * cfg.headsPerNgram
                self.pleGathered     = try zeros(max(gathered, 1), 2, label: "qwen38.ple_gathered")
                self.pleKey          = try zeros(hcDim, 2, label: "qwen38.ple_key")
                self.pleKeyNormed    = try zeros(hcDim, 2, label: "qwen38.ple_key_normed")
                self.pleQueryNormed  = try zeros(hcDim, 2, label: "qwen38.ple_query_normed")
                self.pleValue        = try zeros(D, 2, label: "qwen38.ple_value")
                self.pleGate         = try zeros(max(hcCount, 1), 4, label: "qwen38.ple_gate")
                self.pleGated        = try zeros(hcDim, 2, label: "qwen38.ple_gated")
                self.pleConvIn       = try zeros(hcDim, 2, label: "qwen38.ple_conv_in")
                self.pleConvOut      = try zeros(hcDim, 2, label: "qwen38.ple_conv_out")
                // The conv history IS persistent state (like the GDN conv
                // states): `reset()` clears it, `resetTransientState()` must
                // not. Zero at a sequence start is what makes the first
                // `hist` tokens read zeros rather than wrap.
                self.pleConvState     = try zeros(max(hist, 1) * hcDim, 2,
                                                  label: "qwen38.ple_conv_state")
                self.pleConvStateNext = try zeros(max(hist, 1) * hcDim, 2,
                                                  label: "qwen38.ple_conv_state_next")
            } else {
                self.pleHost = nil
                self.ple = nil
                self.pleGathered     = try buf(1)
                self.pleKey          = try buf(1)
                self.pleKeyNormed    = try buf(1)
                self.pleQueryNormed  = try buf(1)
                self.pleValue        = try buf(1)
                self.pleGate         = try buf(1, MemoryLayout<Float>.size)
                self.pleGated        = try buf(1)
                self.pleConvIn       = try buf(1)
                self.pleConvOut      = try buf(1)
                self.pleConvState     = try buf(1)
                self.pleConvStateNext = try buf(1)
            }
        } else {
            self.qkvConv  = try buf(1)
            self.zBuf     = try buf(1)
            self.qGateBuf = try buf(1)
            self.gateBuf  = try buf(1)
            self.gBeta    = try buf(1, MemoryLayout<Float>.size)
            self.gateAB   = nil
            self.gdnRecurrentState = []
            self.gdnConvState = []
            self.gdnGateWeights = []
            self.gdnStateIndexByLayer = []
            self.qwenOnesEffectiveScale = nil
            self.qwenOnesPerExpertScale = nil
            self.hcPlane   = try buf(1)
            self.hcXn      = try buf(1)
            self.hcGateRaw = try buf(1)
            self.hcGated   = try buf(1)
            self.hcLo      = try buf(1)
            self.hcInject  = try buf(1)
            self.qsaIndexer = nil
            self.qsaState   = nil
            self.ple = nil
            self.pleHost = nil
            self.pleGathered      = try buf(1)
            self.pleKey           = try buf(1)
            self.pleKeyNormed     = try buf(1)
            self.pleQueryNormed   = try buf(1)
            self.pleValue         = try buf(1)
            self.pleGate          = try buf(1, MemoryLayout<Float>.size)
            self.pleGated         = try buf(1)
            self.pleConvIn        = try buf(1)
            self.pleConvOut       = try buf(1)
            self.pleConvState     = try buf(1)
            self.pleConvStateNext = try buf(1)
        }
    }

    public func reset() {
        kv?.reset()
        // The GDN recurrent/conv states are part of the KV-cache-style
        // persistent state — zeroed here, NOT in resetTransientState(), which
        // also runs on continuation and must keep the state intact.
        for s in gdnRecurrentState { memset(s.contents(), 0, s.length) }
        for c in gdnConvState { memset(c.contents(), 0, c.length) }
        // Same lifetime: the QSA indexer's key timeline is positional state,
        // and `pooledBlocks` in particular must return to 0 or the next run
        // would skip pooling the blocks it refills.
        qsaState?.reset()
        // And the PLE's: the window of recent tokens and the dilated-conv
        // history are both positional, and a stale window would hash the new
        // sequence's first tokens against the old one's last.
        pleHost?.reset()
        memset(pleConvState.contents(), 0, pleConvState.length)
        memset(pleConvStateNext.contents(), 0, pleConvStateNext.length)
        // The row fingerprints describe one prefill, so the extent resets with
        // it; a later generation must not report rows it never hashed.
        rowHashRowCount = 0
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    // Expert selection, which is CPU planning on the critical path. It sits
    // *outside* the `io` window -- `tIoStart` is read after this returns -- and
    // inside no `cb1` bucket either, so before this counter it landed in the
    // serial sum's unexplained remainder rather than in any measured span.
    public private(set) var totalIoPlanNanos: UInt64 = 0
    // The `io` window's parts, held on the model because two of the three are
    // only visible inside the fetch. `io - plan` is not the identity here:
    // `plan` is outside the window, and the three parts below do not tile it --
    // what remains after subtracting them is the continuation hops, the
    // `streamersQueue.sync` and the `ensureLayerOpened` check, all of which are
    // the per-layer fixed cost this split exists to find.
    public var totalIoDispatchNanos: UInt64 { model.routedIoDispatchNanos() }
    public var totalIoReadNanos: UInt64 { model.routedIoReadNanos() }
    public var totalIoTailNanos: UInt64 { model.routedIoTailNanos() }
    // `ioReadNanos` split, and this one *is* an exact tiling: `fanout + span +
    // drain == read`, by construction in the streamer rather than by assertion
    // here. `ioThreadNanos` is not a fourth part -- it is the summed thread time
    // inside the span, so it is normally larger than the span and its ratio to
    // it is the achieved parallelism. Reported together so a reader cannot
    // mistake it for a part and get a total that doubles the window.
    public var totalIoFanoutNanos: UInt64 { model.routedIoFanoutNanos() }
    public var totalIoSpanNanos: UInt64 { model.routedIoSpanNanos() }
    public var totalIoDrainNanos: UInt64 { model.routedIoDrainNanos() }
    public var totalIoThreadNanos: UInt64 { model.routedIoThreadNanos() }
    public var totalIoPreadNanos: UInt64 { model.routedIoPreadNanos() }
    public var totalIoCopyNanos: UInt64 { model.routedIoCopyNanos() }
    /// Per-read latency, log2-bucketed. The sums above give the mean; this is
    /// what says whether the mean is the reads or a tail among them.
    public var totalIoLatencyHistogram: [UInt64] { model.routedIoLatencyHistogram() }

    // The engine's own pread sequence, so the drive can be priced offline on
    // the real offset pattern rather than a synthetic one: the offline probes
    // bracket this workload at 7.1 GB/s for diverse offsets and 14.4 GB/s for a
    // repeated pool, and the engine's actual sequence is neither. Collected
    // only when `FQ_EXPERT_TRACE` names a path, so the default run allocates
    // nothing and the append costs one branch per miss.
    private let expertTracePath: String? =
        ProcessInfo.processInfo.environment["FQ_EXPERT_TRACE"]
    /// Flat, because the replay harness only ever reads it forward: `[layer,
    /// missCount, expert...]` repeated per layer per step. The expert ids are
    /// the router's, not slot indices, so the harness can rebuild file offsets
    /// from the layout alone.
    public private(set) var expertTrace: [Int32] = []

    /// Writes the collected pread sequence for offline replay, as one integer
    /// per line. Expert ids only: the harness rebuilds byte offsets from the
    /// install's own layout, so nothing here depends on the trace being taken
    /// on the machine that replays it.
    public func writeExpertTrace(to path: String) throws {
        var out = Data()
        out.reserveCapacity(expertTrace.count * 6)
        for value in expertTrace {
            out.append(contentsOf: Array("\(value)\n".utf8))
        }
        try out.write(to: URL(fileURLWithPath: path))
    }
    // Expert-cache hit/miss counts, accumulated once per layer in
    // `encodeRoutedTail`. Deliberately counts rather than bytes: the byte
    // figure is `misses * expertStride` exactly, but obtaining the stride per
    // layer costs a lock in the very window this instrumentation exists to
    // measure, so the multiplication happens once at print time instead.
    public private(set) var totalExpertHits: UInt64 = 0
    public private(set) var totalExpertMisses: UInt64 = 0
    // Prefill-side routed-expert I/O, kept separate from the decode pair above.
    // The chunked prefill loops stream tiles through their own `fetch` bindings
    // and never call `encodeRoutedTail`, so `totalExpertMisses` is structurally
    // decode-only on the chunked path. These make the prefill read volume
    // observable, which is the number the chunk-size question turns on:
    // prefill reads `misses * expertStride` bytes, and a layer's pool is
    // re-read once per chunk.
    public private(set) var totalPrefillExpertMisses: UInt64 = 0
    public private(set) var totalPrefillTiles: UInt64 = 0
    public private(set) var totalPrefillChunks: UInt64 = 0
    // Full-attention layers whose QSA indexer ran the ranked path rather than
    // the dense one. The indexer budget is a position threshold (~2048 on the
    // real 3.8 install), so a soak long enough to cross it shows a step in the
    // attention bucket; without this count that step has no explanation.
    public private(set) var totalIndexerRankedLayers: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    // Sub-buckets of `totalCb1Nanos`, tiled by a cursor so that gaps and
    // double-counts are structurally impossible rather than merely intended:
    // the six spans below sum to `totalCb1Nanos` exactly, and
    // `totalCb1WaitNanos` is the pipeline wait that `totalCb1Nanos` subtracts
    // and therefore excludes. `lapseCb1` is the only way the cursor moves.
    //
    // `totalCb1AttentionNanos` and the three GDN spans are alternatives, not
    // peers: a layer is one kind or the other, so compare them per layer
    // (12:36 on Qwen 3.8, 10:30 on 3.6), never as a ratio of the raw sums.
    //
    // Only the Qwen decode paths are instrumented. A Gemma run reports every
    // bucket here as 0 against a nonzero `cb1` — an unmistakable "not
    // instrumented" signal rather than a measured zero.
    public private(set) var totalCb1OtherNanos: UInt64 = 0
    public private(set) var totalCb1AttentionNanos: UInt64 = 0
    // The GDN core is split three ways because item 2.3's entire action list is
    // to fuse `gdn_gate` into `gdn_recurrent`, and to let `gdn_rmsnorm_gated`
    // read `gdn_recurrent`'s output inside a single encoder. One lumped "GDN"
    // number cannot separate "the in-projections dominate" from "the
    // gate+recurrent dispatch pair dominates", and only the second of those
    // justifies that work — so the split is what makes 2.3 decidable at all.
    public private(set) var totalCb1GdnProjNanos: UInt64 = 0
    public private(set) var totalCb1GdnConvGateNanos: UInt64 = 0
    public private(set) var totalCb1GdnRecurrentNanos: UInt64 = 0
    public private(set) var totalCb1RouterNanos: UInt64 = 0
    public private(set) var totalCb1WaitNanos: UInt64 = 0

    // How many command buffers one decode actually commits — item 2.2's
    // missing number, and the only counter here with a value that can be
    // predicted from the source before the run. Counted at the commit sites
    // themselves rather than globally: the prefill bodies commit through the
    // same `ctx.queue`, so a global count would fold two profiles together.
    public private(set) var totalDecodeCommandBuffers: UInt64 = 0

    // One per `produceToken`, i.e. per decode step. Every other counter here is
    // read against this denominator, and it is counted rather than inferred
    // from the token count because the two are not the same number: a greedy
    // prefill seed produces a token without a forward, and an `.off` prefill
    // runs the decode path once per prompt token.
    public private(set) var totalForwards: UInt64 = 0

    // GPU-side time, the one thing the encode clocks above cannot see. Read
    // from `gpuStartTime`/`gpuEndTime` after completion: no extra command
    // buffer, no extra wait, no change to commit order.
    //
    // Provisional by nature, and the caller must treat it that way. Metal
    // documents these as coarse, and on some Apple GPUs they are synthetic. The
    // validity gate is that the summed GPU time lands at a plausible fraction
    // of token wall time (40-70% here, given the deliberate overlap) — a total
    // of 0, or several times the token time, means the timestamps are not
    // usable and the finding is *that*, not the number.
    //
    // `totalGpuSamples` is the other half of the gate: the routed tail is
    // deliberately never waited on, so a buffer that has not completed yet
    // contributes nothing, and a total built from a handful of samples must not
    // be read as if it covered the run.
    public private(set) var totalGpuCb1Nanos: UInt64 = 0
    // `totalGpuCb1Nanos` split by layer kind. The CPU-side `attention` and
    // `gdn_*` buckets are alternatives — each layer runs one stack or the other
    // — so they are compared by dividing by the layer counts (12 full-attention
    // / 36 GDN on Qwen 3.8; 10 / 30 on 3.6). GPU time has the same shape with
    // larger stakes: the two stacks run different kernels, so one summed figure
    // cannot say which owns it, and the answer decides different work. Read
    // this as a per-layer comparison, not a per-step one.
    //
    // The prefill accumulates into these same three fields (see
    // `recordPrefillLayerGpuTime`), and the split is far sharper there: on a
    // 426-token prefill the GDN stack is 13.07 s against full attention's
    // 0.92 s, i.e. ~363 ms per GDN layer against ~77 ms per full-attention one.
    // The routed and tail buffers go to `totalGpuRoutedNanos` as they do in
    // decode.
    public private(set) var totalGpuCb1FullAttnNanos: UInt64 = 0
    public private(set) var totalGpuCb1GdnNanos: UInt64 = 0
    public private(set) var totalGpuRoutedNanos: UInt64 = 0
    public private(set) var totalGpuSamples: UInt64 = 0

    // The prefill's command buffers, counted at the prefill commit sites for
    // the reason `totalDecodeCommandBuffers` gives above: both profiles commit
    // through the same `ctx.queue`, so one counter would fold them together.
    //
    // It exists so the GPU totals can be checked for coverage. `totalGpuSamples`
    // only counts buffers whose timestamps came back real, and the decode path
    // deliberately leaves its routed tail unwaited, so a decode total built from
    // a handful of samples must not read as a whole one. The prefill waits on
    // every buffer it commits, so here the two numbers should land together --
    // and if they do not, that is the finding.
    public private(set) var totalPrefillCommandBuffers: UInt64 = 0

    // The GDN sub-stage split (`FQ_GDN_SPLIT=1`). `totalGpuCb1GdnNanos` says the
    // linear-attention stack owns 44% of a prefill without saying which part of
    // it does, and the three candidates want different work done to them: the
    // input projections are GEMMs, the conv is a bandwidth-bound stencil, and
    // the chunked recurrent scan is neither.
    //
    // The split is taken by committing the sub-stages as separate command
    // buffers that are *not* waited on where they are committed -- only their
    // submission order matters, so the layer's existing wait completes all of
    // them and then their timestamps can be read. That keeps the sync pattern
    // (and so the overlap) unchanged, which is the point: an instrument that
    // adds waits measures a different prefill. What it does change is command
    // buffer granularity, so the knob's own effect on the total is measured as
    // its control rather than assumed away.
    //
    // Prefill only, and only for the 3.8 body: the decode path does not split.
    //
    // **`totalGpuCb1GdnNanos` means something different while this is on.** The
    // split moves the GDN work into the four stage buffers, so cb1 collapses to
    // the layer's post-GDN remainder (~259 ms on a run whose unsplit GDN total
    // is ~13,150 ms). Read the stages against the *unsplit* total; the stages sum
    // to 95% of it, the rest being the seq mix and the plane combines after the
    // GDN block.
    //
    // What it says on the 125B, 426 tokens at chunk 512: the input projections
    // are 8.66 s of the 12.64 s those four stages cover, the output projection
    // 3.24 s, the chunked recurrent scan 0.72 s, and the conv1d with its gated
    // activation 0.009 s. The GDN stack is a projection story -- 95% GEMMs, 6%
    // scan -- which is the opposite of what the sequential chunked scan's shape
    // suggests, and it means a GDN fusion pass would be folding a 6% term.
    public private(set) var totalGpuGdnProjNanos: UInt64 = 0
    public private(set) var totalGpuGdnConvNanos: UInt64 = 0
    public private(set) var totalGpuGdnScanNanos: UInt64 = 0
    /// The GDN output projection, split out for the reason the split comment
    /// gives: it is a GEMM but it runs last, so it cannot share stage 1.
    public private(set) var totalGpuGdnOutProjNanos: UInt64 = 0

    // The PLE n-gram gather, which is Qwen-3.8-only and appears in no other
    // counter: it runs before the layer loop, once per head, and reads rows
    // straight off disk from a table far too large to cache. Wall clock, not
    // an encode clock — it is a synchronous host-side read, not a dispatch.
    //
    // `totalPlePartOpens` counts `open` calls the gather makes — one per head,
    // each followed by a single-row pread — not *distinct* parts: a cached
    // handle still costs a pread, and the preads are the thing that costs.
    public private(set) var totalPleGatherNanos: UInt64 = 0
    public private(set) var totalPleGathers: UInt64 = 0
    public private(set) var totalPlePartOpens: UInt64 = 0
    /// Bytes the PLE gather has requested, taken from the host after each call
    /// rather than recomputed here, so it follows the install's actual row
    /// stride — the number that prices the table's quantization.
    public private(set) var totalPleRowBytes: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    /// Advances `cursor` to now and returns the elapsed nanoseconds, so that
    /// consecutive calls tile a span with no gap and no overlap.
    ///
    /// The shared cursor is the mechanism, not a convenience. Mixing a cursor
    /// with independently-taken `now` readings is exactly what lets a span go
    /// missing or get counted twice — and either error still prints a
    /// plausible-looking number, which is worse than an obviously broken one.
    ///
    /// The single place the cursor moves without attributing anything is the
    /// pipeline wait: `totalCb1Nanos` subtracts `waitNanos`, so the buckets must
    /// skip precisely that span or the sum comes out `wait` too high.
    private func lapseCb1(_ cursor: inout UInt64) -> UInt64 {
        let now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        defer { cursor = now }
        return now &- cursor
    }

    /// Commit a command buffer and count it. Used at the decode commit sites
    /// only — see `totalDecodeCommandBuffers`.
    private func commitCounting(_ cb: MTLCommandBuffer) {
        totalDecodeCommandBuffers &+= 1
        cb.commit()
    }

    /// GPU time a completed buffer reports, or 0 if the driver has not
    /// published one. Counts a sample only when it returns a real number, so
    /// `totalGpuSamples` can be checked against the buffer count — a partial
    /// total is visible rather than passing for a whole one.
    private func recordGpuTime(_ cb: MTLCommandBuffer) -> UInt64 {
        guard cb.status == .completed else { return 0 }
        let start = cb.gpuStartTime
        let end = cb.gpuEndTime
        guard start.isFinite, end.isFinite, start > 0, end > start else { return 0 }
        totalGpuSamples &+= 1
        return UInt64((end - start) * 1_000_000_000)
    }

    /// Commit a prefill command buffer and count it. See
    /// `totalPrefillCommandBuffers`.
    private func commitCountingPrefill(_ cb: MTLCommandBuffer) {
        totalPrefillCommandBuffers &+= 1
        cb.commit()
    }

    /// The prefill's per-layer GPU time, into the same accumulators the decode
    /// path uses.
    ///
    /// Reusing them is sound because every counter here is read as a *delta*
    /// (`RunnerCounterValues.delta`): a `scope=prefill` line is the snapshot at
    /// the prefill/decode boundary and so holds prefill-only values, and a
    /// `scope=decode` line holds decode-only ones, from one set of fields. That
    /// is also why the split by layer kind carries over cleanly — "cb1" means
    /// the layer's own forward in both profiles, and the full-attention/GDN
    /// division is the same property of the same layer.
    ///
    /// The prefill's totals need their own plausibility gate, and it is the one
    /// the decode comment above states: thousands of samples summing to a
    /// number several times the wall clock means the timestamps are not usable,
    /// and *that* is the finding.
    /// Close the current GDN sub-stage and open the next one.
    ///
    /// The buffer handed in is committed here and recorded for later timing;
    /// what comes back is fresh. Nothing waits, so the GPU still runs these back
    /// to back in submission order and the layer's single existing wait covers
    /// them all -- the only thing that changes is where the buffer boundaries
    /// fall, and boundaries are what a GPU timestamp can report.
    private func splitGdnSubStage(_ cb: MTLCommandBuffer, closing stage: Int?,
                                  into pending: inout [(stage: Int, cb: MTLCommandBuffer)])
        -> MTLCommandBuffer {
        guard gdnSubStageSplit, let next = ctx.queue.makeCommandBuffer() else {
            return cb
        }
        commitCountingPrefill(cb)
        if let stage { pending.append((stage: stage, cb: cb)) }
        return next
    }

    private func recordPrefillLayerGpuTime(_ cb: MTLCommandBuffer, isFull: Bool) {
        let nanos = recordGpuTime(cb)
        totalGpuCb1Nanos &+= nanos
        if isFull {
            totalGpuCb1FullAttnNanos &+= nanos
        } else {
            totalGpuCb1GdnNanos &+= nanos
        }
    }

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }

        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        for (spanIndex, span) in spans.enumerated() {
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
            // The row fingerprints are written by this chunk's command buffers;
            // remember how far they reach so the dump knows how much of the
            // side buffer is live.
            if rowHashLayout != nil {
                rowHashRowCount = max(rowHashRowCount,
                                      span.startPosition + span.tokenCount)
            }
            // `FQ_QSA_DUMP`: the chunk's work is committed by now, so a drain
            // makes the ranking state readable, and it holds each full layer's
            // selection for this chunk's LAST row. That is the prefill-side
            // fingerprint: the decode-side one showed the divergence is already
            // present at the first decode step, which places it in here.
            if qsaDumpPath != nil {
                drainGPU()
                dumpQSASelection(position: span.startPosition + span.tokenCount - 1,
                                 phase: "prefill")
            }
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(config: PrefillRuntimeConfig) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               runtime: config,
                                               maxContext: maxContext)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let q: TensorView
            let k: TensorView
            let v: TensorView
            let o: TensorView
            let postAttention: TensorView
            let preFFN: TensorView
            let preFFN2: TensorView
            let postFFN2: TensorView
            let postFFN: TensorView
            let layerScalar: TensorView
            let qNorm: TensorView
            let kNorm: TensorView
            let router: TensorView
            let routerPerExpertScale: TensorView
        }

        // Gemma-only tensor views; the Qwen branch fetches its own per layer.
        let layerViews: [LayerPrefillQKVViews] = cfg.isQwenHybrid
            ? []
            : try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                q: try model.qProj(layer: L),
                k: try model.kProj(layer: L),
                v: isFull ? (try model.kProj(layer: L)) : (try model.vProj(layer: L)),
                o: try model.oProj(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                preFFN: try model.preFFN(layer: L),
                preFFN2: try model.preFFN2(layer: L),
                postFFN2: try model.postFFN2(layer: L),
                postFFN: try model.postFFN(layer: L),
                layerScalar: try model.layerScalar(layer: L),
                qNorm: try model.qNorm(layer: L),
                kNorm: try model.kNorm(layer: L),
                router: try model.router(layer: L),
                routerPerExpertScale: try model.routerPerExpertScale(layer: L))
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        guard let tokenBuffer = ctx.device.makeBuffer(bytes: tokenIDs,
                                                      length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                                                      options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        // Gemma scales embeddings by sqrt(hidden); the Qwen hybrid families do
        // NOT (qwen3_5_moe and qwen4_exp feed embed_tokens straight into the
        // layers — llama's build_inp_embd applies no scale for QWEN4EXP — the
        // residual would otherwise stay embed-dominated and the layer
        // contributions would be attenuated by 1/sqrt(D) every residual add).
        let sqrtHidden = cfg.isQwenHybrid ? 1.0 : Float(D).squareRoot()
        let t = tokens.count
        let emb = model.embedding

        // PLE routing for the whole chunk, before any layer runs: the rows are
        // a hash of each token and its two predecessors, and the table rows are
        // read straight off disk (16 × 320 B = 5 KB a token — nothing is
        // cached, because the reads are hash-random over a 102.4 GB table).
        // Recording is per token and in position order, exactly as decode does
        // it one token at a time; a chunk is just several of those in a row.
        if let pleHost, cfg.isQwen3_8 {
            // Per-token stride from the layout, never the buffer divided by
            // the run length: the buffer is sized on the config's *maximum*
            // chunk, so for any shorter chunk that division would spread the
            // rows at the wrong stride — and the layer reads them at this one.
            let width = scratch.layout.qwen38PleGatheredRowElements
            let bytes = width * MemoryLayout<Float16>.stride
            precondition(scratch.layout.qwen38PleGatheredElements >= t * width,
                         "PLE gather scratch too small for a \(t)-token chunk")
            for row in 0..<t {
                let position = startPosition + row
                pleHost.record(position: position, token: tokens[tokens.startIndex + row])
                let gathered = try pleHost.gather(atPosition: position) { part in
                    try model.openPLEPart(part)
                }
                // The gather is one token's n-gram row set; it can only ever be
                // this wide, but a short read would be silent garbage.
                precondition(gathered.count >= width,
                             "PLE gather returned \(gathered.count) elements, need \(width)")
                gathered.withUnsafeBytes { src in
                    memcpy(scratch.qwen38PleGathered.contents().advanced(by: row * bytes),
                           src.baseAddress!,
                           bytes)
                }
            }
        }

        totalPrefillChunks &+= 1
        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillEmbed.encode(commandBuffer: cb,
                            table: emb.buffer,
                            tableOffset: Int(emb.offset),
                            scales: emb.buffer,
                            scalesOffset: Int(emb.scaleOffset),
                            biases: emb.buffer,
                            biasesOffset: Int(emb.biasOffset),
                            tokens: tokenBuffer,
                            out: scratch.hidden,
                            t: UInt32(t),
                            d: UInt32(D),
                            outScale: sqrtHidden)
        if cfg.isQwen3_8 {
            // The 3.8 residual is the wide hyper-connection plane, seeded once
            // per chunk with hc copies of each row's embedding (`hc_init`,
            // qwen4exp.cpp :324-331). The whole chunk's rows go in one
            // dispatch — this is the plane's only write that is not a combine.
            // Same command buffer as the embedding, so it must follow it.
            hyperConnection.encodeSeqPlaneInit(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               plane: scratch.qwen38Plane,
                                               d: UInt32(D),
                                               hc: UInt32(cfg.hyperConnectionCount),
                                               tokens: UInt32(t))
        }

        for L in 0..<cfg.numLayers {
            model.beginOpeningRoutedExpertStreamer(layer: L)
            if cfg.isQwen3_6 {
                cb = try await encodeQwenPrefillLayer(L, scratch: scratch,
                                                      startPosition: startPosition,
                                                      tokenCount: t, cb: cb)
                continue
            }
            if cfg.isQwen3_8 {
                // Hyper-connections replace every per-layer norm, so the 3.8
                // body is its own pass — the Gemma path below would read
                // `inputNorm`/`postAttnNorm`/`postFFN` that a 3.8 install does
                // not carry.
                cb = try await encodeQwen38PrefillLayer(L, scratch: scratch,
                                                        startPosition: startPosition,
                                                        tokenCount: t, cb: cb)
                continue
            }
            let views = layerViews[L]
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: views.q,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: qDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.k,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: views.v,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            let rotatedPairs = isFull
                ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                : UInt32(headDim / 2)
            prefillQKVEpilogue.encode(commandBuffer: cb,
                                       q: scratch.q,
                                       k: scratch.kStage,
                                       v: scratch.vStage,
                                       qWeight: views.qNorm.buffer,
                                       qWeightOffset: Int(views.qNorm.offset),
                                       kWeight: views.kNorm.buffer,
                                       kWeightOffset: Int(views.kNorm.offset),
                                       startPosition: UInt32(startPosition),
                                       queryCount: UInt32(t),
                                       headDim: UInt32(headDim),
                                       numQHeads: UInt32(cfg.numHeads),
                                       numKVHeads: UInt32(numKVHeads),
                                       qTokenStrideElements: UInt32(qDim),
                                       kvTokenStrideElements: UInt32(kvDim),
                                       theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                       rotatedPairs: rotatedPairs,
                                       eps: eps)

            if let kv {
                let bytes = t * kvDim * MemoryLayout<Float16>.stride
                try copyPrefillKVToCache(commandBuffer: cb,
                                         kv: kv,
                                         layer: L,
                                         startPosition: startPosition,
                                         tokenCount: t,
                                         keySource: scratch.kStage,
                                         valueSource: scratch.vStage,
                                         bytesPerToken: bytes / t)
            }
            let params = PrefillAttentionParams(
                    startPosition: UInt32(startPosition),
                    queryCount: UInt32(t),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(cfg.numHeads),
                    numKVHeads: UInt32(numKVHeads),
                    kvValidCount: UInt32(startPosition + t),
                    slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                    kvTokenStrideElements: UInt32(kvDim),
                    qTokenStrideElements: UInt32(qDim),
                    oTokenStrideElements: UInt32(qDim),
                    scale: 1.0)
            if let kv {
                    let keyBuffer = kv.keyBuffer(layer: L, validTokenCount: startPosition + t)
                    let valueBuffer = kv.valueBuffer(layer: L, validTokenCount: startPosition + t)
                    let ringCapacity = kv.ringCapacity(layer: L)
                    let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    prefillAttention.encodeCausal(commandBuffer: cb,
                                                  q: scratch.q,
                                                  k: keyBuffer,
                                                  v: valueBuffer,
                                                  out: scratch.attentionOutput,
                                                  params: params,
                                                  kvRingCapacity: activeRingCapacity,
                                                  path: prefillAttentionPath)
            } else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }
            encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.o,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: qDim,
                                     tokenCount: t,
                                     xStrideElements: qDim,
                                     yStrideElements: D)
            prefillPostAttention.encode(commandBuffer: cb,
                                            hidden: scratch.hidden,
                                            attn: scratch.h1,
                                            denseX: scratch.denseX,
                                            routedX: scratch.routedX,
                                            routerX: scratch.routerX,
                                            postAttentionWeight: views.postAttention.buffer,
                                            postAttentionWeightOffset: Int(views.postAttention.offset),
                                            preFFNWeight: views.preFFN.buffer,
                                            preFFNWeightOffset: Int(views.preFFN.offset),
                                            preFFN2Weight: views.preFFN2.buffer,
                                            preFFN2WeightOffset: Int(views.preFFN2.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            attnStrideElements: UInt32(D),
                                            denseStrideElements: UInt32(D),
                                            routedStrideElements: UInt32(D),
                                            routerStrideElements: UInt32(D),
                                            eps: eps)
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router.buffer,
                        weightsOffset: Int(views.router.offset),
                        scales: views.router.buffer,
                        scalesOffset: Int(views.router.scaleOffset),
                        biases: views.router.buffer,
                        biasesOffset: Int(views.router.biasOffset),
                        hidden: scratch.routerX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: views.routerPerExpertScale.buffer,
                        perExpertScaleOffset: Int(views.routerPerExpertScale.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))

                    cb.commit()
                    waitForCompletion(cb)
                    if let error = cb.error {
                        throw error
                    }

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    guard let sharedCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    let sharedProj = sharedExpertProjections[L]
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: scratch.denseX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.intermediateSize,
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    // Gemma-only prefill path: postF1 always present here.
                    let postF1 = sharedProj.postF1!
                    prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                           x: scratch.h1,
                                           weight: postF1.buffer,
                                           weightOffset: Int(postF1.offset),
                                           out: scratch.h1,
                                           t: UInt32(t),
                                           d: UInt32(D),
                                           eps: eps)
                    sharedCB.commit()
                    waitForCompletion(sharedCB)
                    if let error = sharedCB.error {
                        throw error
                    }

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            waitForCompletion(pending.commandBuffer)
                        }
                        if let error = pending.commandBuffer.error {
                            throw error
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        totalPrefillExpertMisses &+= UInt64(fetch.plannedMissSlots.count)
                        totalPrefillTiles &+= 1
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = ctx.queue.makeCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    let scalarBits = views.layerScalar.buffer.contents()
                        .advanced(by: Int(views.layerScalar.offset))
                        .assumingMemoryBound(to: UInt16.self)[0]
                    prefillLayerTail.encode(commandBuffer: tailCB,
                                            h2: scratch.h2,
                                            h1: scratch.h1,
                                            hidden: scratch.hidden,
                                            postFFN2Weight: views.postFFN2.buffer,
                                            postFFN2WeightOffset: Int(views.postFFN2.offset),
                                            postFFNWeight: views.postFFN.buffer,
                                            postFFNWeightOffset: Int(views.postFFN.offset),
                                            queryCount: UInt32(t),
                                            d: UInt32(D),
                                            h2StrideElements: UInt32(D),
                                            h1StrideElements: UInt32(D),
                                            hiddenStrideElements: UInt32(D),
                                            eps: eps,
                                            layerScalar: Quantization.bf16ToFloat(scalarBits))
                    tailCB.commit()
                    withExtendedLifetime(metadata) {
                        waitForCompletion(tailCB)
                    }
                    if let error = tailCB.error {
                        throw error
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = ctx.queue.makeCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        if writeFinalHead {
            let lm = model.lmHead
            guard let finalCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if cfg.isQwen3_8 {
                // No `model.norm` in this family: the head input is the root
                // `hyper_connection_mixer` collapsing the chunk's last plane
                // row to [D] — the same stage every layer's mixer runs, with
                // no block_inject and no inject target (decode's `rootMixer`
                // head; qwen4exp.cpp :380-390). Reached through the chunked
                // twin with `tokens: 1` so the row arithmetic is the layer
                // path's, not a second copy of it.
                let rootMixer = try model.hyperConnectionMixer()
                encodeQwen38SeqMix(commandBuffer: finalCB,
                                   norm: rootMixer.hcNorm,
                                   down: rootMixer.mixDown,
                                   up: rootMixer.mixUp,
                                   blockInject: nil,
                                   plane: scratch.qwen38Plane,
                                   // BYTES: `planeOffset` reaches
                                   // `setBuffer(_:offset:)`. Passing the
                                   // element count here read an arbitrary
                                   // offset — row 0 was right only when t == 1.
                                   planeOffset: (t - 1) * cfg.hyperConnectionDim
                                       * MemoryLayout<Float16>.stride,
                                   blockOut: scratch.normed, blockOutOffset: 0,
                                   inject: nil, injectOffset: 0,
                                   scratch: scratch,
                                   d: UInt32(D),
                                   hc: UInt32(cfg.hyperConnectionCount),
                                   lowrank: UInt32(cfg.hyperConnectionLowrank),
                                   tokens: 1,
                                   invHc: 1.0 / Float(cfg.hyperConnectionCount),
                                   eps: eps)
                int4.encode(commandBuffer: finalCB,
                            weights: lm.buffer, weightsOffset: Int(lm.offset),
                            scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                            biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                            x: scratch.normed, y: logits,
                            m: UInt32(cfg.vocabSize), n: UInt32(D))
                finalCB.commit()
                waitForCompletion(finalCB)
                if let error = finalCB.error {
                    throw error
                }
                kv?.advance(by: tokens.count)
                prefillChunkState.markCommitted()
                return
            }
            guard let finalNorm = model.finalNorm else {
                throw ModelError.tensorNotFound(name: "language_model.model.norm.weight")
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            waitForCompletion(finalCB)
            if let error = finalCB.error {
                throw error
            }
            if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    /// One batched int4-affine projection for prefill: MPP (t >= 32, q/kv/o
    /// families) when available, then the QMM batch kernel, then repeated
    /// decode-style GEMVs. `x` is [tokenCount][columns], `y` is
    /// [tokenCount][rows] with the given element strides.
    private func encodeRepeatedInt8(commandBuffer: MTLCommandBuffer,
                                    weights: TensorView,
                                    x: MTLBuffer,
                                    y: MTLBuffer,
                                    rows: Int,
                                    columns: Int,
                                    tokenCount: Int,
                                    xStrideElements: Int,
                                    yStrideElements: Int,
                                    yBaseElements: Int = 0) {
        guard tokenCount >= 1 else { return }
        for row in 0..<tokenCount {
            int8GEMV!.encode(commandBuffer: commandBuffer,
                             weights: weights.buffer,
                             weightsOffset: Int(weights.offset),
                             scales: weights.buffer,
                             scalesOffset: Int(weights.scaleOffset),
                             biases: weights.buffer,
                             biasesOffset: Int(weights.biasOffset),
                             x: x,
                             xOffset: row * xStrideElements * MemoryLayout<Float16>.size,
                             y: y,
                             yOffset: (yBaseElements + row * yStrideElements)
                                * MemoryLayout<Float16>.size,
                             m: UInt32(rows),
                             n: UInt32(columns))
        }
    }

    /// The batched form of `encodeRepeatedInt8`, for the sites whose output row
    /// stride equals the row count. Falls back to the per-token GEMV loop when
    /// the knob is off, the kernel is absent, or the shape is not one the tile
    /// covers — so the default path is untouched and the A/B is one env var.
    @discardableResult
    private func encodeInt8ProjectionBatched(commandBuffer: MTLCommandBuffer,
                                             weights: TensorView,
                                             x: MTLBuffer,
                                             y: MTLBuffer,
                                             rows: Int,
                                             columns: Int,
                                             tokenCount: Int,
                                             yStrideElements: Int,
                                             yBaseElements: Int = 0) -> Bool {
        guard let gemm = prefillInt8Gemm,
              tokenCount >= PrefillInt8Gemm.tokensPerTile,
              yStrideElements == rows,
              PrefillInt8Gemm.supports(columns: columns) else {
            return false
        }
        gemm.encode(commandBuffer: commandBuffer,
                    weights: weights.buffer, weightsOffset: Int(weights.offset),
                    scales: weights.buffer, scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer, biasesOffset: Int(weights.biasOffset),
                    x: x, xOffset: 0,
                    y: y,
                    yOffset: yBaseElements * MemoryLayout<Float16>.stride,
                    tokens: tokenCount, rows: rows, columns: columns)
        return true
    }

    private func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                      family: PrefillProjectionFamily,
                                      weights: TensorView,
                                      x: MTLBuffer,
                                      xBaseOffset: Int = 0,
                                      y: MTLBuffer,
                                      yBaseOffset: Int = 0,
                                      rows: Int,
                                      columns: Int,
                                      tokenCount: Int,
                                      xStrideElements: Int,
                                      yStrideElements: Int) {
        if tokenCount >= 32,
           family == .q || family == .kv || family == .o,
           let candidate = prefillMPPAffineInt4 {
            let path = candidate.encode(
                commandBuffer: commandBuffer,
                weights: weights.buffer,
                weightsOffset: Int(weights.offset),
                scales: weights.buffer,
                scalesOffset: Int(weights.scaleOffset),
                biases: weights.buffer,
                biasesOffset: Int(weights.biasOffset),
                x: x,
                xOffset: xBaseOffset,
                y: y,
                yOffset: yBaseOffset,
                m: tokenCount,
                n: rows,
                k: columns)
            if path == .affineThreadgroupF16 {
                return
            }
        }
        if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                            chunkTokens: tokenCount) == .qmm {
            prefillQMM.encode(commandBuffer: commandBuffer,
                              weights: weights.buffer,
                              weightsOffset: Int(weights.offset),
                              scales: weights.buffer,
                              scalesOffset: Int(weights.scaleOffset),
                              biases: weights.buffer,
                              biasesOffset: Int(weights.biasOffset),
                              x: x,
                              xOffset: xBaseOffset,
                              y: y,
                              yOffset: yBaseOffset,
                              t: tokenCount,
                              n: rows,
                              k: columns)
            return
        }
        for row in 0..<tokenCount {
            int4.encode(commandBuffer: commandBuffer,
                        weights: weights.buffer,
                        weightsOffset: Int(weights.offset),
                        scales: weights.buffer,
                        scalesOffset: Int(weights.scaleOffset),
                        biases: weights.buffer,
                        biasesOffset: Int(weights.biasOffset),
                        x: x,
                        xOffset: xBaseOffset
                            + row * xStrideElements * MemoryLayout<Float16>.stride,
                        y: y,
                        yOffset: yBaseOffset
                            + row * yStrideElements * MemoryLayout<Float16>.stride,
                        m: UInt32(rows),
                        n: UInt32(columns))
        }
    }

    private func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                               source: MTLBuffer,
                               destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                               sourceTokenOffset: Int,
                               tokenCount: Int,
                               bytesPerToken: Int) throws {
        guard tokenCount > 0 else { return }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw ModelError.residentBufferWrapFailed
        }
        blit.copy(from: source,
                  sourceOffset: sourceTokenOffset * bytesPerToken,
                  to: destination.buffer,
                  destinationOffset: destination.offset,
                  size: tokenCount * bytesPerToken)
        blit.endEncoding()
    }

    private func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                      kv: KVCacheManager,
                                      layer: Int,
                                      startPosition: Int,
                                      tokenCount: Int,
                                      keySource: MTLBuffer,
                                      valueSource: MTLBuffer,
                                      bytesPerToken: Int) throws {
        let capacity = kv.capacity(layer: layer)
        let physicalStart = startPosition % capacity
        let firstSpan = min(tokenCount, capacity - physicalStart)
        let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
        let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keyFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueFirst,
                          sourceTokenOffset: 0,
                          tokenCount: firstSpan,
                          bytesPerToken: bytesPerToken)
        guard firstSpan < tokenCount else { return }

        let secondCount = tokenCount - firstSpan
        let secondStart = startPosition + firstSpan
        let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
        let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: keySource,
                          destination: keySecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
        try copyPrefillKV(commandBuffer: commandBuffer,
                          source: valueSource,
                          destination: valueSecond,
                          sourceTokenOffset: firstSpan,
                          tokenCount: secondCount,
                          bytesPerToken: bytesPerToken)
    }

    /// One Qwen 3.6 prefill layer: the batched chunk form of
    /// `encodeQwenDecodeLayer` — token mixer (GDN or full attention) +
    /// post-attention residual/norm + router on the passed-in CB, then the
    /// silu shared expert, the silu streamed routed tiles, and the
    /// `hidden += h2` combine. Math pinned to `qwen3_5_moe` (see
    /// `docs/QWEN36_PORT.md`). Returns the fresh CB the next layer encodes on.
    private func encodeQwenPrefillLayer(
        _ L: Int,
        scratch: PrefillChunkScratchBuffers,
        startPosition: Int,
        tokenCount: Int,
        cb: MTLCommandBuffer
    ) async throws -> MTLCommandBuffer {
        let t = tokenCount
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let isFull = cfg.fullAttentionLayerMask[L] != 0
        var oSnapBuf: MTLBuffer?    // debug: pre-gated-norm o capture (GDN only)

        let inNorm = try model.inputNorm(layer: L)
        let postAttnNorm = try model.postAttnNorm(layer: L)
        let routerW = try model.router(layer: L)
        guard let onesEffective = qwenOnesEffectiveScale,
              let onesExpert = qwenOnesPerExpertScale else {
            preconditionFailure("Qwen prefill layer on a non-Qwen runner")
        }

        prefillRMS.encodeBF16W(commandBuffer: cb,
                               x: scratch.hidden,
                               weight: inNorm.buffer,
                               weightOffset: Int(inNorm.offset),
                               out: scratch.normed,
                               t: UInt32(t),
                               d: UInt32(D),
                               eps: eps)

        let gPostAttn: (MTLCommandBuffer) -> Void = { [self] cb in
            for row in 0..<t {
                qwenFusions.encodePostAttn(commandBuffer: cb,
                                           hidden: scratch.hidden, hiddenOffset: row * D * 2,
                                           attn: scratch.h1, attnOffset: row * D * 2,
                                           out: scratch.denseX, outOffset: row * D * 2,
                                           weight: postAttnNorm.buffer,
                                           weightOffset: Int(postAttnNorm.offset),
                                           d: UInt32(D), eps: eps)
            }
        }

        if isFull {
            // Full-attention layer: doubled q_proj (per-head q|gate pairs),
            // q/k per-head norms + partial RoPE (the decode epilogue, once per
            // token — its q_out lands packed in the q scratch, gate_out in the
            // dedicated gate scratch), KV cache write, attention with Qwen's
            // scale rsqrt(head_dim), then the output gate.
            let qP = try model.qProj(layer: L)
            let kP = try model.kProj(layer: L)
            let vP = try model.vProj(layer: L)
            let oP = try model.oProj(layer: L)
            let qN = try model.qNorm(layer: L)
            let kN = try model.kNorm(layer: L)
            let headDim = cfg.fullHeadDim
            let numQ = cfg.numHeads
            let numKV = cfg.numFullKVHeads
            let qDim = numQ * headDim
            let kvDim = numKV * headDim
            let rotaryDim = Int(Double(headDim) * cfg.partialRotaryFactor)

            encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: qP,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: 2 * qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: 2 * qDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: kP,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: vP,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            for row in 0..<t {
                qwenFusions.encodeFullAttnEpilogue(
                    commandBuffer: cb,
                    qProj: scratch.q, qProjOffset: row * 2 * qDim * 2,
                    qOut: scratch.q, qOutOffset: row * qDim * 2,
                    gateOut: scratch.qwenQGate, gateOutOffset: row * qDim * 2,
                    k: scratch.kStage, kOffset: row * kvDim * 2,
                    qWeight: qN.buffer, qWeightOffset: Int(qN.offset),
                    kWeight: kN.buffer, kWeightOffset: Int(kN.offset),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(numQ),
                    numKVHeads: UInt32(numKV),
                    position: UInt32(startPosition + row),
                    theta: Float(cfg.fullRopeTheta),
                    rotaryDim: UInt32(rotaryDim),
                    eps: eps)
            }

            if let kv {
                let bytes = t * kvDim * MemoryLayout<Float16>.stride
                try copyPrefillKVToCache(commandBuffer: cb,
                                         kv: kv,
                                         layer: L,
                                         startPosition: startPosition,
                                         tokenCount: t,
                                         keySource: scratch.kStage,
                                         valueSource: scratch.vStage,
                                         bytesPerToken: bytes / t)
            } else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }

            let params = PrefillAttentionParams(
                startPosition: UInt32(startPosition),
                queryCount: UInt32(t),
                headDim: UInt32(headDim),
                numQHeads: UInt32(numQ),
                numKVHeads: UInt32(numKV),
                kvValidCount: UInt32(startPosition + t),
                slidingWindow: UInt32(startPosition + t),
                kvTokenStrideElements: UInt32(kvDim),
                qTokenStrideElements: UInt32(qDim),
                oTokenStrideElements: UInt32(qDim),
                scale: Float(1.0 / Double(headDim).squareRoot()))
            if let kv {
                prefillAttention.encodeCausal(
                    commandBuffer: cb,
                    q: scratch.q,
                    k: kv.keyBuffer(layer: L, validTokenCount: startPosition + t),
                    v: kv.valueBuffer(layer: L, validTokenCount: startPosition + t),
                    out: scratch.attentionOutput,
                    params: params,
                    kvRingCapacity: 0,
                    path: prefillAttentionPath)
            }

            for row in 0..<t {
                qwenFusions.encodeAttnOutputGate(commandBuffer: cb,
                                                 attn: scratch.attentionOutput,
                                                 attnOffset: row * qDim * 2,
                                                 gate: scratch.qwenQGate,
                                                 gateOffset: row * qDim * 2,
                                                 n: UInt32(qDim))
            }
            encodeInt4Projection(commandBuffer: cb,
                                 family: .o,
                                 weights: oP,
                                 x: scratch.attentionOutput,
                                 y: scratch.h1,
                                 rows: D,
                                 columns: qDim,
                                 tokenCount: t,
                                 xStrideElements: qDim,
                                 yStrideElements: D)
            gPostAttn(cb)
        } else {
            // GDN (linear-attention) layer: in_proj_qkv → batched silu causal
            // conv (separate in/out — the batched form races in place) →
            // fused a|b QMM + batched gate → sequential recurrent step →
            // batched gated RMSNorm → out_proj. q/k/v read from the conv
            // output at offsets 0 / keyDim / 2*keyDim — no split copies.
            let si = gdnStateIndexByLayer[L]
            precondition(si >= 0, "GDN layer \(L) without state")
            let qkvP = try model.gdnInProjQKV(layer: L)
            let zP = try model.gdnInProjZ(layer: L)
            let outP = try model.gdnOutProj(layer: L)
            let convW = try model.gdnConv1D(layer: L)
            let aLog = try model.gdnALog(layer: L)
            let dt = try model.gdnDtBias(layer: L)
            let normW = try model.gdnNormWeight(layer: L)
            let recState = gdnRecurrentState[si]
            let convState = gdnConvState[si]
            let aP = linearAttnBits == 8 ? try model.gdnInProjA(layer: L) : nil
            let bP = linearAttnBits == 8 ? try model.gdnInProjB(layer: L) : nil
            let keyDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim
            let valueDim = cfg.linearNumValueHeads * cfg.linearValueHeadDim
            let qkvDim = 2 * keyDim + valueDim
            let numV = cfg.linearNumValueHeads
            let headDim = cfg.linearValueHeadDim
            let scale = 1.0 / Float(cfg.linearKeyHeadDim).squareRoot()
            let betaByteOffset = t * numV * MemoryLayout<Float>.size

            if linearAttnBits == 8 {
                // int8 linear_attn projections: no batched int8 QMM exists,
                // so each chunk token runs one decode-style int8 GEMV per
                // projection (the 4-bit fused a|b block is not assembled).
                let qkvBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: qkvP, x: scratch.normed,
                    y: scratch.qwenQKVProj, rows: qkvDim, columns: D,
                    tokenCount: t, yStrideElements: qkvDim)
                if !qkvBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: qkvP,
                                       x: scratch.normed,
                                       y: scratch.qwenQKVProj,
                                       rows: qkvDim,
                                       columns: D,
                                       tokenCount: t,
                                       xStrideElements: D,
                                       yStrideElements: qkvDim)
                }
                let zBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: zP, x: scratch.normed,
                    y: scratch.qwenZ, rows: valueDim, columns: D,
                    tokenCount: t, yStrideElements: valueDim)
                if !zBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: zP,
                                       x: scratch.normed,
                                       y: scratch.qwenZ,
                                       rows: valueDim,
                                       columns: D,
                                       tokenCount: t,
                                       xStrideElements: D,
                                       yStrideElements: valueDim)
                }
                // a-rows then b-rows into the [T][2V] ab scratch the batched
                // gate kernel reads (same layout the fused QMM wrote).
                encodeRepeatedInt8(commandBuffer: cb,
                                   weights: aP!,
                                   x: scratch.normed,
                                   y: scratch.qwenAB,
                                   rows: numV,
                                   columns: D,
                                   tokenCount: t,
                                   xStrideElements: D,
                                   yStrideElements: 2 * numV,
                                   yBaseElements: 0)
                encodeRepeatedInt8(commandBuffer: cb,
                                   weights: bP!,
                                   x: scratch.normed,
                                   y: scratch.qwenAB,
                                   rows: numV,
                                   columns: D,
                                   tokenCount: t,
                                   xStrideElements: D,
                                   yStrideElements: 2 * numV,
                                   yBaseElements: numV)
            } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: qkvP,
                                     x: scratch.normed,
                                     y: scratch.qwenQKVProj,
                                     rows: qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: zP,
                                     x: scratch.normed,
                                     y: scratch.qwenZ,
                                     rows: valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: valueDim)
                // in_proj_a|in_proj_b as one [2V, D] QMM over the
                // init-assembled fused block (the same buffer the decode
                // gate GEMV reads).
                let gateW = gdnGateWeights[si]
                prefillQMM.encode(commandBuffer: cb,
                                  weights: gateW.weights,
                                  scales: gateW.scales,
                                  biases: gateW.biases,
                                  x: scratch.normed,
                                  y: scratch.qwenAB,
                                  t: t,
                                  n: 2 * numV,
                                  k: D)
            }
            gdnPrefill.encodeGateBatch(commandBuffer: cb,
                                       ab: scratch.qwenAB,
                                       A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                       dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                       g: scratch.qwenGBeta,
                                       beta: scratch.qwenGBeta, betaOffset: betaByteOffset,
                                       numValueHeads: numV,
                                       tokens: t)
            gdnPrefill.encodeConvChunk(commandBuffer: cb,
                                       w: convW.buffer, wOffset: Int(convW.offset),
                                       state: convState,
                                       x: scratch.qwenQKVProj,
                                       out: scratch.qwenQKVConvOut,
                                       newState: scratch.qwenConvNewState,
                                       channels: qkvDim,
                                       tokens: t)
            gdnPrefill.encodeRecurrentSeq(commandBuffer: cb,
                                          state: recState,
                                          conv: scratch.qwenQKVConvOut,
                                          g: scratch.qwenGBeta,
                                          beta: scratch.qwenGBeta, betaOffset: betaByteOffset,
                                          out: scratch.qwenRecOut,
                                          headDim: UInt32(headDim),
                                          channels: UInt32(qkvDim),
                                          kOffset: UInt32(keyDim),
                                          vOffset: UInt32(2 * keyDim),
                                          numValueHeads: numV,
                                          numKeyHeads: cfg.linearNumKeyHeads,
                                          tokens: t,
                                          scale: scale,
                                          l2eps: eps)
            if qwenLayerDebugHook != nil {
                // Debug: the gated RMSNorm below overwrites qwenRecOut in
                // place, so capture the pre-norm recurrent output o on this CB
                // now (blit encode order = execution order).
                let oBytes = t * valueDim * MemoryLayout<Float16>.stride
                oSnapBuf = ctx.device.makeBuffer(length: oBytes,
                                                 options: .storageModeShared)
                if let blit = cb.makeBlitCommandEncoder() {
                    blit.copy(from: scratch.qwenRecOut, sourceOffset: 0,
                              to: oSnapBuf!, destinationOffset: 0,
                              size: oBytes)
                    blit.endEncoding()
                }
            }
            gdnPrefill.encodeRMSNormGatedBatch(commandBuffer: cb,
                                               x: scratch.qwenRecOut,
                                               z: scratch.qwenZ,
                                               weight: normW.buffer, weightOffset: Int(normW.offset),
                                               out: scratch.qwenRecOut,
                                               headDim: UInt32(headDim),
                                               numValueHeads: numV,
                                               tokens: t,
                                               eps: eps)
            if linearAttnBits == 8 {
                let outProjBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: outP, x: scratch.qwenRecOut,
                    y: scratch.h1, rows: D, columns: valueDim,
                    tokenCount: t, yStrideElements: D)
                if !outProjBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: outP,
                                       x: scratch.qwenRecOut,
                                       y: scratch.h1,
                                       rows: D,
                                       columns: valueDim,
                                       tokenCount: t,
                                       xStrideElements: valueDim,
                                       yStrideElements: D)
                }
            } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: outP,
                                     x: scratch.qwenRecOut,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: valueDim,
                                     tokenCount: t,
                                     xStrideElements: valueDim,
                                     yStrideElements: D)
            }
            gPostAttn(cb)
        }

        // Router (both layer types): plain softmax over all experts, top-8
        // renormalized — identical math to the kernel's top-8 softmax, so the
        // Gemma block is reused with ones-filled scales (as in decode).
        prefillRouter.encodeGemma4Block(
            commandBuffer: cb,
            weights: routerW.buffer,
            weightsOffset: Int(routerW.offset),
            scales: routerW.buffer,
            scalesOffset: Int(routerW.scaleOffset),
            biases: routerW.buffer,
            biasesOffset: Int(routerW.biasOffset),
            hidden: scratch.denseX,
            effectiveScale: onesEffective,
            perExpertScale: onesExpert,
            perExpertScaleOffset: 0,
            outIndices: scratch.routeIDs,
            outWeights: scratch.routeWeights,
            queryCount: UInt32(t),
            numExperts: UInt32(cfg.numExperts),
            d: UInt32(D),
            topK: UInt32(cfg.topKExperts),
            hiddenStrideElements: UInt32(D))
        cb.commit()
        waitForCompletion(cb)
        if let error = cb.error {
            throw error
        }

        if let hook = qwenLayerDebugHook, !isFull {
            // The mixer CB has completed and nothing has reused h1 yet: for a
            // GDN layer it holds the pre-residual branch output (xa = the
            // out_proj result, one contiguous D-wide row per token) — the
            // shared-expert tail overwrites h1 below, so snapshot now. Torch
            // side: `layer.linear_attn(layer.input_layernorm(x))`.
            let bytes = t * D * MemoryLayout<Float16>.stride
            if let tmp = ctx.device.makeBuffer(length: bytes,
                                               options: .storageModeShared),
               let snapCB = ctx.queue.makeCommandBuffer(),
               let blit = snapCB.makeBlitCommandEncoder() {
                blit.copy(from: scratch.h1, sourceOffset: 0,
                          to: tmp, destinationOffset: 0, size: bytes)
                blit.endEncoding()
                snapCB.commit()
                waitForCompletion(snapCB)
                let ptr = tmp.contents().bindMemory(to: Float16.self,
                                                    capacity: t * D)
                for row in 0..<t {
                    hook(L, "prefillXA.\(row)",
                         Array(UnsafeBufferPointer(start: ptr.advanced(by: row * D),
                                                   count: D)))
                }
            }
            // Per-stage snapshots for the branch-fidelity drill: normed (the
            // mixer input), post-conv qkv, post-gated-rmsnorm h, and z — the
            // pre-out_proj stage h is the last stage before xa, so the first
            // stage that diverges against torch localizes the fault.
            let keyDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim
            let valueDim = cfg.linearNumValueHeads * cfg.linearValueHeadDim
            let qkvDim = 2 * keyDim + valueDim
            func stage(_ name: String, _ buf: MTLBuffer, _ elements: Int) {
                guard let tmp = ctx.device.makeBuffer(
                        length: elements * MemoryLayout<Float16>.stride,
                        options: .storageModeShared),
                      let snapCB = ctx.queue.makeCommandBuffer(),
                      let blit = snapCB.makeBlitCommandEncoder() else { return }
                blit.copy(from: buf, sourceOffset: 0, to: tmp,
                          destinationOffset: 0,
                          size: elements * MemoryLayout<Float16>.stride)
                blit.endEncoding()
                snapCB.commit()
                waitForCompletion(snapCB)
                let ptr = tmp.contents().bindMemory(to: Float16.self,
                                                    capacity: elements)
                hook(L, name, Array(UnsafeBufferPointer(start: ptr,
                                                        count: elements)))
            }
            stage("pfGdn.normed", scratch.normed, t * D)
            stage("pfGdn.conv", scratch.qwenQKVConvOut, t * qkvDim)
            stage("pfGdn.o", oSnapBuf!, t * valueDim)
            stage("pfGdn.h", scratch.qwenRecOut, t * valueDim)
            stage("pfGdn.z", scratch.qwenZ, t * valueDim)
            // g/beta are fp32 [T][V] (g at 0, beta at t*V floats); hook them
            // fp16-rounded so the stage drill can compare against torch.
            func gateStage(_ name: String, _ buf: MTLBuffer,
                           _ offsetBytes: Int, _ elements: Int) {
                guard let tmp = ctx.device.makeBuffer(
                        length: elements * MemoryLayout<Float>.stride,
                        options: .storageModeShared),
                      let snapCB = ctx.queue.makeCommandBuffer(),
                      let blit = snapCB.makeBlitCommandEncoder() else { return }
                blit.copy(from: buf, sourceOffset: offsetBytes, to: tmp,
                          destinationOffset: 0,
                          size: elements * MemoryLayout<Float>.stride)
                blit.endEncoding()
                snapCB.commit()
                waitForCompletion(snapCB)
                let ptr = tmp.contents().bindMemory(to: Float.self,
                                                    capacity: elements)
                hook(L, name,
                     (0..<elements).map { Float16(ptr[$0]) })
            }
            let numV2 = cfg.linearNumValueHeads
            gateStage("pfGdn.g", scratch.qwenGBeta, 0, t * numV2)
            gateStage("pfGdn.beta", scratch.qwenGBeta,
                      t * numV2 * MemoryLayout<Float>.size, t * numV2)
        }

        // CPU readback of the router indices → expert grouping, as in decode.
        let routeCount = t * cfg.topKExperts
        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        var routeIDs = [UInt32]()
        routeIDs.reserveCapacity(routeCount)
        var routeWeights = [Float16]()
        routeWeights.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeights.append(weightPtr[i])
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                       weights: routeWeights,
                                                       queryCount: t,
                                                       topK: cfg.topKExperts)
        let schedulerConfig = prefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
            guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
            }
            routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
        } else {
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

        // Shared expert (silu) on an early-committed CB, then the Qwen post
        // stage: sigmoid(shared_expert_gate · x) scales h1 per token.
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                            x: scratch.denseX,
                                            y: scratch.h1,
                                            gate: sharedProj.gate,
                                            up: sharedProj.up,
                                            down: sharedProj.down,
                                            scratchGate: scratch.sharedGateScratch,
                                            scratchUp: scratch.sharedUpScratch,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t,
                                            d: D,
                                            intermediate: cfg.intermediateSize,
                                            xStrideElements: D,
                                            yStrideElements: D,
                                            activation: .silu)
        let sharedGate = try model.sharedExpertGateProj(layer: L)
        for row in 0..<t {
            qwenFusions.encodeSharedGate(commandBuffer: sharedCB,
                                         weights: sharedGate.buffer,
                                         weightsOffset: Int(sharedGate.offset),
                                         scales: sharedGate.buffer,
                                         scalesOffset: Int(sharedGate.scaleOffset),
                                         biases: sharedGate.buffer,
                                         biasesOffset: Int(sharedGate.biasOffset),
                                         x: scratch.denseX, xOffset: row * D * 2,
                                         h1: scratch.h1, h1Offset: row * D * 2,
                                         n: UInt32(D), d: UInt32(D))
        }
        sharedCB.commit()
        waitForCompletion(sharedCB)
        if let error = sharedCB.error {
            throw error
        }

        // Streamed routed-expert tiles (silu), mirroring the Gemma tile loop.
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device,
            routes: routes)
        let routedOffsets = model.routedExpertOffsets(layer: L)
        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()
        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                waitForCompletion(pending.commandBuffer)
            }
            if let error = pending.commandBuffer.error {
                throw error
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex,
                routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L,
                        experts: expertIDs,
                        avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending:
                    break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail: "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: L,
                tileIndex: tileIndex,
                routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tile.pairStart),
                                                  pairCount: Int(tile.pairCount))
            totalPrefillExpertMisses &+= UInt64(fetch.plannedMissSlots.count)
            totalPrefillTiles &+= 1
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tileIndex,
                                       plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device,
                binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(D),
                binding: fetch.binding,
                offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            _ = prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.denseX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: streamedParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows,
                activation: .silu)
            tileCB.commit()
            pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                   commandBuffer: tileCB,
                                                   fetch: fetch,
                                                   argumentBuffer: argumentBuffer))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }

        // Tail: h2 = routed reduce; Qwen then combines h2 += h1 (the shared
        // expert, already gate-scaled — the decode path's phase-2 residual)
        // and hidden += h2. No layer_scalar, no sandwich norms.
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                          routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights,
                                          h2: scratch.h2,
                                          queryCount: UInt32(t),
                                          topK: UInt32(cfg.topKExperts),
                                          d: UInt32(D))
        for row in 0..<t {
            qwenFusions.encodeVecAdd(commandBuffer: tailCB,
                                     a: scratch.h2, aOffset: row * D * 2,
                                     b: scratch.h1, bOffset: row * D * 2,
                                     d: UInt32(D))
            qwenFusions.encodeVecAdd(commandBuffer: tailCB,
                                     a: scratch.hidden, aOffset: row * D * 2,
                                     b: scratch.h2, bOffset: row * D * 2,
                                     d: UInt32(D))
        }
        tailCB.commit()
        withExtendedLifetime(metadata) {
            waitForCompletion(tailCB)
        }
        if let error = tailCB.error {
            throw error
        }

        if let hook = qwenLayerDebugHook {
            // The prefill scratch is private-storage; blit the row to a
            // shared snapshot buffer for the diagnostic hook.
            func row(_ name: String, _ buf: MTLBuffer, _ stride: Int,
                     sourceRow: Int = 0) {
                guard let tmp = ctx.device.makeBuffer(length: stride * 2,
                                                      options: .storageModeShared),
                      let snapCB = ctx.queue.makeCommandBuffer(),
                      let blit = snapCB.makeBlitCommandEncoder() else { return }
                blit.copy(from: buf, sourceOffset: sourceRow * stride * 2,
                          to: tmp, destinationOffset: 0, size: stride * 2)
                blit.endEncoding()
                snapCB.commit()
                snapCB.waitUntilCompleted()
                let ptr = tmp.contents().bindMemory(to: Float16.self,
                                                    capacity: stride)
                hook(L, name, Array(UnsafeBufferPointer(start: ptr,
                                                        count: stride)))
            }
            row("prefillHidden0", scratch.hidden, cfg.hiddenSize)
            row("prefillDense0", scratch.denseX, cfg.hiddenSize)
            // Per-token rows: the mixers and the MoE tail write denseX /
            // hidden row-strided — snapshots of every row let the diagnostic
            // replay find the first diverging layer/token.
            for t in 0..<tokenCount {
                row("prefillHidden.\(t)", scratch.hidden, cfg.hiddenSize,
                    sourceRow: t)
                row("prefillDense.\(t)", scratch.denseX, cfg.hiddenSize,
                    sourceRow: t)
            }
            // Router readback for the last token (shared-storage scratch):
            // the engine's expert ids/weights for the MoE-tail comparison.
            do {
                let topK = cfg.topKExperts
                let idPtr = scratch.routeIDs.contents()
                    .bindMemory(to: UInt32.self, capacity: tokenCount * topK)
                let wPtr = scratch.routeWeights.contents()
                    .bindMemory(to: Float16.self, capacity: tokenCount * topK)
                let last = tokenCount - 1
                hook(L, "pfRouteIDs", (0..<topK).map { idPtr[last * topK + $0] }
                    .map { Float16($0) })
                hook(L, "pfRouteW", (0..<topK).map { wPtr[last * topK + $0] }
                    .map { Float16($0) })
            }
            let si = gdnStateIndexByLayer[L]
            if si >= 0 {
                let n = cfg.linearNumValueHeads * cfg.linearValueHeadDim
                    * cfg.linearValueHeadDim
                let ptr = gdnRecurrentState[si].contents()
                    .bindMemory(to: Float.self, capacity: n)
                hook(L, "pfState", Array(UnsafeBufferPointer(start: ptr, count: n))
                    .map { Float16($0) })
                let cn = (cfg.linearNumKeyHeads * cfg.linearKeyHeadDim * 2
                    + cfg.linearNumValueHeads * cfg.linearValueHeadDim) * 3
                let cPtr = gdnConvState[si].contents()
                    .bindMemory(to: Float16.self, capacity: cn)
                hook(L, "pfConvState", Array(UnsafeBufferPointer(start: cPtr,
                                                                 count: cn)))
            } else if let kv {
                // Full-attention layer: snapshot every KV row the prefill
                // wrote (already normalized + rotated by the epilogue).
                func slot(_ name: String,
                          _ s: (buffer: MTLBuffer, offset: Int)) {
                    let n = cfg.numFullKVHeads * cfg.fullHeadDim
                    let ptr = s.buffer.contents().advanced(by: s.offset)
                        .bindMemory(to: Float16.self, capacity: n)
                    hook(L, name, Array(UnsafeBufferPointer(start: ptr,
                                                            count: n)))
                }
                for p in 0..<tokenCount {
                    slot("pfK.\(p)", kv.kSlot(layer: L, position: startPosition + p))
                    slot("pfV.\(p)", kv.vSlot(layer: L, position: startPosition + p))
                }
            }
        }

        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            return nextCB
        }
        return cb
    }

    /// One layer's deferred routed command-buffer bundle: the routed CB itself,
    /// the early-committed shared-expert CB it depends on, and the optional
    /// phase-1-hit-split CB. The next layer drains it before queueing its own.
    private struct PendingRoutedCommand {
        let cb: MTLCommandBuffer
        let sharedCB: MTLCommandBuffer?
        let phase1HitCB: MTLCommandBuffer?
        let encodeAndCommitNanos: UInt64
    }

    private func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                            waitIfNeeded: Bool) {
        if waitIfNeeded {
            func wait(_ cb: MTLCommandBuffer) {
                waitForCompletion(cb)
            }
            if let sharedCB = pending.sharedCB {
                wait(sharedCB)
            }
            if let phase1HitCB = pending.phase1HitCB {
                wait(phase1HitCB)
            }
            wait(pending.cb)
        } else if let err = pending.cb.error {
            print("CB error: \(err)")
        }
        if let sharedCB = pending.sharedCB {
            if let err = sharedCB.error {
                print("CB error: \(err)")
            }
        }
        if let phase1HitCB = pending.phase1HitCB,
           let err = phase1HitCB.error {
            print("CB error: \(err)")
        }
        // The routed tail is deliberately not waited on, so whether these
        // buffers have completed by the time they are retired is a race with
        // the GPU — on the decode path a whole layer's CPU encode runs between
        // one layer's commit and its drain, which is usually (not always)
        // enough. `recordGpuTime` returns 0 for the ones still in flight and
        // `totalGpuSamples` shows how partial the total is.
        totalGpuRoutedNanos &+= recordGpuTime(pending.cb)
        if let sharedCB = pending.sharedCB {
            totalGpuRoutedNanos &+= recordGpuTime(sharedCB)
        }
        if let phase1HitCB = pending.phase1HitCB {
            totalGpuRoutedNanos &+= recordGpuTime(phase1HitCB)
        }
        totalCb2Nanos &+= pending.encodeAndCommitNanos
    }

    private func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
        for i in 0..<slots.count { ptr[i] = slots[i] }
    }

    /// The routed-expert tail shared by both model families: CPU readback of
    /// router indices → plan/fetch/advise the expert blobs → phase-1 (with the
    /// family activation) → phase-2 reduce into `routedResidual` (Gemma's
    /// `zeroResidual`, or Qwen's `h1Buf` so the combine adds the shared
    /// expert) → family tail (`fused_layer_tail` / `vec_add`). The shared
    /// expert runs on an early-committed CB overlapping the expert pread;
    /// `sharedPostEncoder` is the family-specific post stage after the FFN
    /// (Gemma post-FFN norm / Qwen sigmoid gate).
    private func encodeRoutedTail(
        layer L: Int,
        position: Int,
        routedX: MTLBuffer,
        denseX: MTLBuffer,
        sharedProj: LayerSharedExpertProjections,
        activation: SharedExpertActivation,
        routedResidual: MTLBuffer,
        sharedPostEncoder: (MTLCommandBuffer) -> Void,
        tailEncoder: (MTLCommandBuffer) -> Void,
        pending: inout PendingRoutedCommand?
    ) async throws {
        let D = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)

        // CPU readback to fetch routed-expert blobs from disk.
        let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                      capacity: cfg.topKExperts)
        var experts = [Int](repeating: 0, count: cfg.topKExperts)
        for i in 0..<cfg.topKExperts {
            experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
        }

        let routedOffsets = model.routedExpertOffsets(layer: L)
        let topK = UInt32(cfg.topKExperts)
        let canPlanPhase1HitSplit =
            cfg.topKExperts <= MoE.maxStreamedExperts
        // Expert selection runs on the critical path between the router readback
        // above and the preads below, and it is inside neither the `cb1` buckets
        // nor the `io` window (`tIoStart` is read after this returns). Before
        // this timer it fell into the serial sum's unexplained remainder.
        let tPlan = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let plannedFetch = canPlanPhase1HitSplit
            ? try model.planRoutedExperts(layer: L, experts: experts)
            : nil
        totalIoPlanNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tPlan
        // Expert-cache accounting for this layer, counted here and nowhere
        // else: this is the one and only `planRoutedExperts` call per layer, and
        // the plan below is consumed by several branches. Counting inside any of
        // them would multiply every total by the number of consumers.
        //
        // Never call `planRoutedExperts` a second time for accounting either —
        // `makeExpertCachePlan` mutates `useClock`, `expertUseCount`,
        // `slotLastUse` and `slotExpert`, so a second call would double every
        // expert's use count, change eviction order, and move the very hit rate
        // it was measuring.
        if let plan = plannedFetch {
            totalExpertHits &+= UInt64(plan.hits)
            totalExpertMisses &+= UInt64(plan.misses.count)
            if expertTracePath != nil {
                // Layer, miss count, then the expert ids: enough for the
                // harness to rebuild every pread offset in order without
                // reproducing the cache.
                expertTrace.append(Int32(L))
                expertTrace.append(Int32(plan.misses.count))
                for index in plan.misses {
                    expertTrace.append(Int32(experts[index]))
                }
            }
        } else {
            // Unreachable while the router's `topK <= maxStreamedExperts`
            // precondition holds, but instrumentation must not be the thing
            // that traps a decode: count an unplannable layer the way the fetch
            // path treats it, as all-miss.
            totalExpertMisses &+= UInt64(experts.count)
        }
        var phase1HitCB: MTLCommandBuffer?
        var phase1HitSplitArgBuf: MTLBuffer?
        var phase1HitSplitRoutedBufs: [MTLBuffer] = []
        var phase1HitSlots: [UInt32] = []
        var phase1MissSlots: [UInt32] = []

        if let plan = plannedFetch {
            let missSet = Set(plan.misses)
            phase1HitSlots = (0..<cfg.topKExperts)
                .filter { !missSet.contains($0) }
                .map { UInt32($0) }
            phase1MissSlots = plan.misses.map { UInt32($0) }
        }
        func encodeRoutedPhase1Full(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer]
        ) {
            moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                    routedArgBuffer: argBuf,
                                                    routedBlobs: routedBufs,
                                                    routedOffsets: routedOffsets,
                                                    x: routedX,
                                                    acts: moeActs,
                                                    d: D,
                                                    f: FmoE,
                                                    topK: topK,
                                                    activation: activation)
        }

        func encodeRoutedPhase1Subset(
            _ cb: MTLCommandBuffer,
            argBuf: MTLBuffer,
            routedBufs: [MTLBuffer],
            activeSlots: MTLBuffer,
            activeSlotIndices: [UInt32],
            activeCount: UInt32
        ) {
            moe.encodeRoutedPersistentPhase1SubsetU16Load(
                commandBuffer: cb,
                routedArgBuffer: argBuf,
                routedBlobs: routedBufs,
                routedOffsets: routedOffsets,
                x: routedX,
                acts: moeActs,
                activeSlots: activeSlots,
                activeSlotIndices: activeSlotIndices,
                activeCount: activeCount,
                d: D,
                f: FmoE,
                topK: topK,
                activation: activation)
        }

        if let plan = plannedFetch,
           plan.hits > 0,
           !plan.misses.isEmpty {
            let plannedBlobs = try model.routedExpertBuffers(for: plan)
            phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
            phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                routedBlobs: phase1HitSplitRoutedBufs,
                topK: topK)
            if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                let cb = ctx.queue.makeCommandBuffer()!
                encodeRoutedPhase1Subset(
                    cb,
                    argBuf: argBuf,
                    routedBufs: phase1HitSplitRoutedBufs,
                    activeSlots: moeHitActiveSlots,
                    activeSlotIndices: phase1HitSlots,
                    activeCount: UInt32(phase1HitSlots.count))
                phase1HitCB = cb
            }
        }

        // The shared dense MLP depends only on denseX, not on the routed
        // experts. Commit it without waiting so its GPU work overlaps the
        // routed-expert pread. The routed CB follows it on the same queue,
        // so the combine sees h1Buf.
        let gSharedFFN: (MTLCommandBuffer) -> Void = { [self] cb in
            try! shared.encode(commandBuffer: cb,
                               x: denseX,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct,
                               activation: activation)
        }
        let sharedCB = ctx.queue.makeCommandBuffer()!
        gSharedFFN(sharedCB)
        sharedPostEncoder(sharedCB)
        commitCounting(sharedCB)
        if let cb = phase1HitCB {
            commitCounting(cb)
        }
        if rdadviseEnabled && rdadvisePolicyMode != .off {
            let requestedMisses = plannedFetch?.misses.count ?? experts.count
            let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                layer: L,
                missCount: requestedMisses)
            if let skipped = shouldSkipRDAdvice(position: position,
                                                requestedMisses: requestedMisses,
                                                estimatedBytes: estimatedAdviceBytes,
                                                canOverlapUsefulGPUWork: true) {
                recordRDAdvice(skipped, wallNanos: 0)
            } else {
                let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                let result: ExpertIOAdviceResult
                if let plannedFetch {
                    result = try model.adviseRoutedExperts(plan: plannedFetch)
                } else {
                    result = try model.adviseRoutedExperts(layer: L, experts: experts)
                }
                let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                recordRDAdvice(result, wallNanos: wallNanos)
                updateRDAdvicePolicy(after: result, position: position)
            }
        }

        // Routed-expert pread — overlaps the shared MLP GPU work above.
        let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs: [TensorView]
        if let plannedFetch {
            blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
        } else {
            blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
        }
        let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
        totalIoNanos &+= layerIo
        let routedBufs = blobs.map { $0.buffer }
        let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        let routedCB = ctx.queue.makeCommandBuffer()!
        let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
            ? phase1HitSplitArgBuf
            : nil
        let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
            routedBlobs: routedBufs,
            topK: topK)
        if splitArgBuf != nil {
            writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
            encodeRoutedPhase1Subset(
                routedCB,
                argBuf: argBuf,
                routedBufs: routedBufs,
                activeSlots: moeMissActiveSlots,
                activeSlotIndices: phase1MissSlots,
                activeCount: UInt32(phase1MissSlots.count))
        } else {
            encodeRoutedPhase1Full(routedCB,
                                   argBuf: argBuf,
                                   routedBufs: routedBufs)
        }
        moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                               routedArgBuffer: argBuf,
                                               routedBlobs: routedBufs,
                                               routedOffsets: routedOffsets,
                                               acts: moeActs,
                                               routingWeights: outWeights,
                                               residual: routedResidual,
                                               y: h2Buf,
                                               d: D,
                                               f: FmoE,
                                               topK: topK)
        tailEncoder(routedCB)
        commitCounting(routedCB)
        precondition(pending == nil,
                     "routed command-buffer pipeline drained before queuing the next layer")
        pending = PendingRoutedCommand(
            cb: routedCB,
            sharedCB: sharedCB,
            phase1HitCB: phase1HitCB,
            encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
    }

    /// One Qwen 3.6 decode layer: CB1 runs the token mixer (GDN or full
    /// attention) + post-attention residual/norm + the router, then the
    /// shared `encodeRoutedTail` schedules the silu experts, the sigmoid-gated
    /// shared expert, and the `hidden += h2` combine. Math pinned to
    /// `qwen3_5_moe` (see `docs/QWEN36_PORT.md`).
    private func encodeQwenDecodeLayer(
        _ L: Int,
        position: Int,
        pending: inout PendingRoutedCommand?
    ) async throws {
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let isFull = cfg.fullAttentionLayerMask[L] != 0
        let seqLen = UInt32(position + 1)

        let inNorm = try model.inputNorm(layer: L)
        let postAttnNorm = try model.postAttnNorm(layer: L)
        let routerW = try model.router(layer: L)
        let sharedProj = sharedExpertProjections[L]
        let sharedGate = try model.sharedExpertGateProj(layer: L)
        guard let onesEffective = qwenOnesEffectiveScale,
              let onesExpert = qwenOnesPerExpertScale else {
            preconditionFailure("Qwen decode layer on a non-Qwen runner")
        }

        let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Every cb1 sub-bucket is measured by advancing this one cursor, so the
        // spans tile the layer's cb1 exactly. See `lapseCb1`.
        var cb1Cursor = tCb1Start
        let cb = ctx.queue.makeCommandBuffer()!

        if let hook = qwenLayerDebugHook {
            let ptr = hidden.contents().bindMemory(to: Float16.self,
                                                   capacity: cfg.hiddenSize)
            hook(L, "preLayer", Array(UnsafeBufferPointer(start: ptr,
                                                          count: cfg.hiddenSize)))
        }

        let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)
        }
        let gPostAttn: (MTLCommandBuffer) -> Void = { [self] cb in
            qwenFusions.encodePostAttn(commandBuffer: cb,
                                       hidden: hidden,
                                       attn: oOut,
                                       out: denseX,
                                       weight: postAttnNorm.buffer,
                                       weightOffset: Int(postAttnNorm.offset),
                                       d: D, eps: eps)
        }

        if isFull {
            // Full-attention layer: q_proj is doubled (per-head q|gate pairs),
            // q/k per-head norms, partial RoPE (rotary dim = 0.25 * head_dim),
            // attention scale rsqrt(head_dim), then the output gate.
            let qP = try model.qProj(layer: L)
            let kP = try model.kProj(layer: L)
            let vP = try model.vProj(layer: L)
            let oP = try model.oProj(layer: L)
            let qN = try model.qNorm(layer: L)
            let kN = try model.kNorm(layer: L)
            let kSlot = kv?.kSlot(layer: L, position: position)
                ?? (buffer: kStage, offset: 0)
            let vSlot = kv?.vSlot(layer: L, position: position)
                ?? (buffer: vStage, offset: 0)
            let headDim = UInt32(cfg.fullHeadDim)
            let numQ = UInt32(cfg.numHeads)
            let numKV = UInt32(cfg.numFullKVHeads)
            let qRows = numQ * headDim
            let rotaryDim = UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)

            let gProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: qP.buffer, weightsOffset: Int(qP.offset),
                            scales: qP.buffer, scalesOffset: Int(qP.scaleOffset),
                            biases: qP.buffer, biasesOffset: Int(qP.biasOffset),
                            x: normed,
                            y: qGateBuf,
                            m: 2 * qRows, n: D)
                int4.encode(commandBuffer: cb,
                            weights: kP.buffer, weightsOffset: Int(kP.offset),
                            scales: kP.buffer, scalesOffset: Int(kP.scaleOffset),
                            biases: kP.buffer, biasesOffset: Int(kP.biasOffset),
                            x: normed,
                            y: kSlot.buffer, yOffset: kSlot.offset,
                            m: numKV * headDim, n: D)
                int4.encode(commandBuffer: cb,
                            weights: vP.buffer, weightsOffset: Int(vP.offset),
                            scales: vP.buffer, scalesOffset: Int(vP.scaleOffset),
                            biases: vP.buffer, biasesOffset: Int(vP.biasOffset),
                            x: normed,
                            y: vSlot.buffer, yOffset: vSlot.offset,
                            m: numKV * headDim, n: D)
            }
            let gEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                qwenFusions.encodeFullAttnEpilogue(
                    commandBuffer: cb,
                    qProj: qGateBuf,
                    qOut: qScratch,
                    gateOut: gateBuf,
                    k: kSlot.buffer, kOffset: kSlot.offset,
                    qWeight: qN.buffer, qWeightOffset: Int(qN.offset),
                    kWeight: kN.buffer, kWeightOffset: Int(kN.offset),
                    headDim: headDim,
                    numQHeads: numQ,
                    numKVHeads: numKV,
                    position: UInt32(position),
                    theta: Float(cfg.fullRopeTheta),
                    rotaryDim: rotaryDim,
                    eps: eps)
            }
            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                attention.encodeFull(commandBuffer: cb,
                                     q: qScratch,
                                     // The kernel walks positions [0, seqLen)
                                     // from the buffer start (the cache base),
                                     // NOT from the current token's slot —
                                     // k/v offsets are always 0 on this path.
                                     k: kSlot.buffer, kOffset: 0,
                                     v: vSlot.buffer, vOffset: 0,
                                     out: attnOut,
                                     headDim: headDim,
                                     numQHeads: numQ,
                                     numKVHeads: numKV,
                                     seqLen: seqLen,
                                     scale: nil)   // rsqrt(head_dim) — Qwen's scaling
            }
            let gGate: (MTLCommandBuffer) -> Void = { [self] cb in
                qwenFusions.encodeAttnOutputGate(commandBuffer: cb,
                                                 attn: attnOut,
                                                 gate: gateBuf,
                                                 n: qRows)
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: oP.buffer, weightsOffset: Int(oP.offset),
                            scales: oP.buffer, scalesOffset: Int(oP.scaleOffset),
                            biases: oP.buffer, biasesOffset: Int(oP.biasOffset),
                            x: attnOut,
                            y: oOut,
                            m: D, n: qRows)
            }
            gInputNorm(cb)
            totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
            gProj(cb)
            gEpilogue(cb)
            gAttention(cb)
            gGate(cb)
            gOProj(cb)
            totalCb1AttentionNanos &+= lapseCb1(&cb1Cursor)
            gPostAttn(cb)
        } else {
            // GDN (linear-attention) layer: in_proj_qkv → silu causal conv →
            // gate (fused a/b GEMV) → recurrent step → gated RMSNorm →
            // out_proj. q/k/v read from the conv output at byte offsets
            // 0 / keyDim*2 / keyDim*4 (the conv output IS the [q,k,v] block).
            let si = gdnStateIndexByLayer[L]
            precondition(si >= 0, "GDN layer \(L) without state")
            let qkvP = try model.gdnInProjQKV(layer: L)
            let zP = try model.gdnInProjZ(layer: L)
            let outP = try model.gdnOutProj(layer: L)
            let convW = try model.gdnConv1D(layer: L)
            let aLog = try model.gdnALog(layer: L)
            let dt = try model.gdnDtBias(layer: L)
            let normW = try model.gdnNormWeight(layer: L)
            let recState = gdnRecurrentState[si]
            let convState = gdnConvState[si]
            let aP = linearAttnBits == 8 ? try model.gdnInProjA(layer: L) : nil
            let bP = linearAttnBits == 8 ? try model.gdnInProjB(layer: L) : nil
            let keyDim = UInt32(cfg.linearNumKeyHeads * cfg.linearKeyHeadDim)
            let valueDim = UInt32(cfg.linearNumValueHeads * cfg.linearValueHeadDim)
            let qkvDim = 2 * keyDim + valueDim
            let numV = cfg.linearNumValueHeads
            let headDim = UInt32(cfg.linearValueHeadDim)
            let scale = 1.0 / Float(cfg.linearKeyHeadDim).squareRoot()
            let betaByteOffset = numV * MemoryLayout<Float>.size

            let gProj: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: qkvP.buffer, weightsOffset: Int(qkvP.offset),
                                     scales: qkvP.buffer, scalesOffset: Int(qkvP.scaleOffset),
                                     biases: qkvP.buffer, biasesOffset: Int(qkvP.biasOffset),
                                     x: normed,
                                     y: qkvConv,
                                     m: qkvDim, n: D)
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: zP.buffer, weightsOffset: Int(zP.offset),
                                     scales: zP.buffer, scalesOffset: Int(zP.scaleOffset),
                                     biases: zP.buffer, biasesOffset: Int(zP.biasOffset),
                                     x: normed,
                                     y: zBuf,
                                     m: valueDim, n: D)
                } else {
                    int4.encode(commandBuffer: cb,
                                weights: qkvP.buffer, weightsOffset: Int(qkvP.offset),
                                scales: qkvP.buffer, scalesOffset: Int(qkvP.scaleOffset),
                                biases: qkvP.buffer, biasesOffset: Int(qkvP.biasOffset),
                                x: normed,
                                y: qkvConv,
                                m: qkvDim, n: D)
                    int4.encode(commandBuffer: cb,
                                weights: zP.buffer, weightsOffset: Int(zP.offset),
                                scales: zP.buffer, scalesOffset: Int(zP.scaleOffset),
                                biases: zP.buffer, biasesOffset: Int(zP.biasOffset),
                                x: normed,
                                y: zBuf,
                                m: valueDim, n: D)
                }
            }
            let gConv: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeCausalConvUpdate(commandBuffer: cb,
                                           w: convW.buffer, wOffset: Int(convW.offset),
                                           state: convState,
                                           x: qkvConv,
                                           out: qkvConv,
                                           newState: convState,
                                           channels: Int(qkvDim))
            }
            let gGateGEMV: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    // Two int8 GEMVs (in_proj_a then in_proj_b) into the fp16
                    // [2V] ab scratch, then the batched gate formula at T=1 —
                    // g at offset 0, beta at +V floats, the same layout and
                    // math as the 4-bit fused kernel.
                    guard let gemv = int8GEMV, let ab = gateAB,
                          let aView = aP, let bView = bP else { return }
                    gemv.encode(commandBuffer: cb,
                                weights: aView.buffer, weightsOffset: Int(aView.offset),
                                scales: aView.buffer, scalesOffset: Int(aView.scaleOffset),
                                biases: aView.buffer, biasesOffset: Int(aView.biasOffset),
                                x: normed,
                                y: ab,
                                m: UInt32(numV), n: D)
                    gemv.encode(commandBuffer: cb,
                                weights: bView.buffer, weightsOffset: Int(bView.offset),
                                scales: bView.buffer, scalesOffset: Int(bView.scaleOffset),
                                biases: bView.buffer, biasesOffset: Int(bView.biasOffset),
                                x: normed,
                                y: ab,
                                yOffset: numV * MemoryLayout<Float16>.size,
                                m: UInt32(numV), n: D)
                    gdnPrefill.encodeGateBatch(commandBuffer: cb,
                                               ab: ab,
                                               A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                               dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                               g: gBeta,
                                               beta: gBeta, betaOffset: betaByteOffset,
                                               numValueHeads: numV,
                                               tokens: 1)
                } else {
                    let gateW = gdnGateWeights[si]
                    gdn.encodeGateGEMV(commandBuffer: cb,
                                       weights: gateW.weights,
                                       scales: gateW.scales,
                                       biases: gateW.biases,
                                       x: normed,
                                       A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                       dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                       g: gBeta,
                                       beta: gBeta, betaOffset: betaByteOffset,
                                       numValueHeads: numV,
                                       n: D)
                }
            }
            let gRecurrent: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeRecurrent(commandBuffer: cb,
                                    state: recState,
                                    q: qkvConv,
                                    k: qkvConv, kOffset: Int(keyDim) * 2,
                                    v: qkvConv, vOffset: Int(keyDim) * 4,
                                    g: gBeta,
                                    beta: gBeta, betaOffset: betaByteOffset,
                                    out: attnOut,
                                    numValueHeads: numV,
                                    numKeyHeads: cfg.linearNumKeyHeads,
                                    headDim: headDim,
                                    scale: scale,
                                    l2eps: 1e-6)
            }
            let gNormGated: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeRMSNormGated(commandBuffer: cb,
                                       x: attnOut,
                                       z: zBuf,
                                       weight: normW.buffer, weightOffset: Int(normW.offset),
                                       out: attnOut,
                                       numValueHeads: numV,
                                       headDim: headDim,
                                       eps: eps)
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: outP.buffer, weightsOffset: Int(outP.offset),
                                     scales: outP.buffer, scalesOffset: Int(outP.scaleOffset),
                                     biases: outP.buffer, biasesOffset: Int(outP.biasOffset),
                                     x: attnOut,
                                     y: oOut,
                                     m: D, n: valueDim)
                } else {
                    int4.encode(commandBuffer: cb,
                                weights: outP.buffer, weightsOffset: Int(outP.offset),
                                scales: outP.buffer, scalesOffset: Int(outP.scaleOffset),
                                biases: outP.buffer, biasesOffset: Int(outP.biasOffset),
                                x: attnOut,
                                y: oOut,
                                m: D, n: valueDim)
                }
            }
            gInputNorm(cb)
            totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
            gProj(cb)
            totalCb1GdnProjNanos &+= lapseCb1(&cb1Cursor)
            gConv(cb)
            gGateGEMV(cb)
            totalCb1GdnConvGateNanos &+= lapseCb1(&cb1Cursor)
            gRecurrent(cb)
            gNormGated(cb)
            gOProj(cb)
            totalCb1GdnRecurrentNanos &+= lapseCb1(&cb1Cursor)
            gPostAttn(cb)
        }

        // Router (both layer types): plain softmax over all experts, top-8
        // renormalized — mathematically identical to the kernel's top-8
        // softmax, so the Gemma kernel is reused with ones-filled scales.
        // The router is the last work encoded into this layer's cb1, so its
        // span runs through the commit call inclusive.
        totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
        moe.encodeRouterGemma4(commandBuffer: cb,
                               weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                               scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                               biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                               hidden: denseX,
                               effectiveScale: onesEffective,
                               perExpertScale: onesExpert,
                               outIndices: outIndices, outWeights: outWeights,
                               numExperts: UInt32(cfg.numExperts), d: D,
                               topK: UInt32(cfg.topKExperts))
        commitCounting(cb)
        totalCb1RouterNanos &+= lapseCb1(&cb1Cursor)
        let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        waitForCompletion(cb)
        let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
        // Read once: `recordGpuTime` counts a sample only when it returns a
        // real number, so a second call would double `totalGpuSamples`.
        let cb1GpuNanos = recordGpuTime(cb)
        totalGpuCb1Nanos &+= cb1GpuNanos
        if isFull {
            totalGpuCb1FullAttnNanos &+= cb1GpuNanos
        } else {
            totalGpuCb1GdnNanos &+= cb1GpuNanos
        }
        // cb1 subtracts this wait, so the buckets must *skip* exactly this span
        // rather than attribute it. Attributing it would put the tile sum
        // `wait` above cb1 — and that sum still looks like a plausible number,
        // which is why this has to be structural rather than remembered.
        cb1Cursor &+= waitNanos
        if let previous = pending {
            // The debug hook snapshots read shared buffers on the CPU — wait
            // out the deferred tail so the snapshots are race-free. Without a
            // hook, the tail completes in CB order before the next layer.
            finishPendingRoutedCommand(previous,
                                       waitIfNeeded: qwenLayerDebugHook != nil)
            pending = nil
        }
        // Final lapse, then cb1 from that same reading, so
        // `other + attention + gdn + router == cb1` holds by construction
        // rather than to within the gap between two separate clock reads.
        totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
        totalCb1Nanos &+= cb1Cursor &- tCb1Start &- waitNanos
        totalCb1WaitNanos &+= waitNanos

        if let hook = qwenLayerDebugHook {
            func snap(_ name: String, _ buf: MTLBuffer, _ count: Int) {
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
                hook(L, name, Array(UnsafeBufferPointer(start: ptr, count: count)))
            }
            snap("postAttn", denseX, cfg.hiddenSize)
            snap("normed", normed, cfg.hiddenSize)
            snap("qkvConv", qkvConv, 8192)
            snap("recurrentOut", attnOut, 4096)
            snap("oOut", oOut, cfg.hiddenSize)
            do {
                let n = 2 * cfg.linearNumValueHeads
                let ptr = gBeta.contents().bindMemory(to: Float.self, capacity: n)
                hook(L, "gFloat", Array(UnsafeBufferPointer(start: ptr, count: n))
                    .map { Float16($0) })
            }
            if isFull, let kv {
                let n = cfg.numHeads * cfg.fullHeadDim
                let qPtr = qScratch.contents().bindMemory(to: Float16.self, capacity: n)
                hook(L, "qOutF", Array(UnsafeBufferPointer(start: qPtr, count: n)))
                let gPtr = gateBuf.contents().bindMemory(to: Float16.self, capacity: n)
                hook(L, "gateF", Array(UnsafeBufferPointer(start: gPtr, count: n)))
                let slot = kv.kSlot(layer: L, position: position)
                let kn = cfg.numFullKVHeads * cfg.fullHeadDim
                let kPtr = slot.buffer.contents().advanced(by: slot.offset)
                    .bindMemory(to: Float16.self, capacity: kn)
                hook(L, "kF", Array(UnsafeBufferPointer(start: kPtr, count: kn)))
                let vSlot = kv.vSlot(layer: L, position: position)
                let vPtr = vSlot.buffer.contents().advanced(by: vSlot.offset)
                    .bindMemory(to: Float16.self, capacity: kn)
                hook(L, "vF", Array(UnsafeBufferPointer(start: vPtr, count: kn)))
            }
            let si = gdnStateIndexByLayer[L]
            if si >= 0 {
                let n = cfg.linearNumValueHeads * cfg.linearValueHeadDim
                    * cfg.linearValueHeadDim
                let ptr = gdnRecurrentState[si].contents()
                    .bindMemory(to: Float.self, capacity: n)
                hook(L, "recState", Array(UnsafeBufferPointer(start: ptr, count: n))
                    .map { Float16($0) })
            }
        }

        // Shared expert post stage: sigmoid(shared_expert_gate · x) scales h1.
        let sharedPost: (MTLCommandBuffer) -> Void = { [self] cb in
            qwenFusions.encodeSharedGate(commandBuffer: cb,
                                         weights: sharedGate.buffer,
                                         weightsOffset: Int(sharedGate.offset),
                                         scales: sharedGate.buffer,
                                         scalesOffset: Int(sharedGate.scaleOffset),
                                         biases: sharedGate.buffer,
                                         biasesOffset: Int(sharedGate.biasOffset),
                                         x: denseX,
                                         h1: h1Buf,
                                         n: D, d: D)
        }
        // Qwen tail: phase-2 writes h2 = shared + routed (residual = h1Buf);
        // the layer closes with hidden += h2. No layer_scalar, no sandwich.
        let tail: (MTLCommandBuffer) -> Void = { [self] cb in
            qwenFusions.encodeVecAdd(commandBuffer: cb,
                                     a: hidden, b: h2Buf,
                                     d: D)
        }
        try await encodeRoutedTail(
            layer: L,
            position: position,
            routedX: denseX,
            denseX: denseX,
            sharedProj: sharedProj,
            activation: .silu,
            routedResidual: h1Buf,
            sharedPostEncoder: sharedPost,
            tailEncoder: tail,
            pending: &pending)

        if let hook = qwenLayerDebugHook {
            // The just-committed tail (hidden += h2) is in flight — wait it
            // out so the post-layer snapshot is race-free.
            if let p = pending {
                finishPendingRoutedCommand(p, waitIfNeeded: true)
                pending = nil
            }
            let ptr = hidden.contents().bindMemory(to: Float16.self,
                                                   capacity: cfg.hiddenSize)
            hook(L, "postLayer", Array(UnsafeBufferPointer(start: ptr,
                                                           count: cfg.hiddenSize)))
        }
    }

    /// Qwen 3.8 Flash-Next hyper-connection mix over a chunk — the
    /// `[tokens]`-wide twin of `encodeQwen38Mix`, same stage order and the same
    /// math (`build_hc_mix`, qwen4exp.cpp :226-262), with the token row moved.
    ///
    /// Three of the five stages are the *decode* kernels reached with a flat
    /// `tokens·width` element count — `hc_silu_scale` and `hc_gate_mul` are
    /// already elementwise over N, so they have no `_seq` twin and must not
    /// grow one. The grouped RMS, the stream mean and the int4 projections are
    /// the chunked forms.
    ///
    /// Every stage encodes serially into `cb` and reads only what an earlier
    /// stage in the same CB wrote, so one shared scratch set serves the
    /// layer's second mixer exactly as it does in decode.
    private func encodeQwen38SeqMix(
        commandBuffer cb: MTLCommandBuffer,
        norm: TensorView,
        down: TensorView,
        up: TensorView,
        blockInject: TensorView?,
        plane: MTLBuffer, planeOffset: Int,
        blockOut: MTLBuffer, blockOutOffset: Int,
        inject: MTLBuffer?, injectOffset: Int,
        scratch: PrefillChunkScratchBuffers,
        d: UInt32,
        hc: UInt32,
        lowrank: UInt32,
        tokens: UInt32,
        invHc: Float,
        eps: Float
    ) {
        precondition((blockInject == nil) == (inject == nil),
                     "block_inject and inject buffers come as a pair")
        let hcDim = Int(d) * Int(hc)

        // xn = per-stream RMS of the plane, scaled by the [hc·D] BF16 gamma.
        hyperConnection.encodeSeqGroupedRMS(commandBuffer: cb,
                                            x: plane, xOffset: planeOffset,
                                            gamma: norm.buffer,
                                            gammaOffset: Int(norm.offset),
                                            out: scratch.qwen38Xn,
                                            d: d, hc: hc, tokens: tokens, eps: eps)
        // lo = silu(down · xn · 1/hc) — the ÷hc precedes the silu.
        encodeInt4Projection(commandBuffer: cb,
                             family: .kv,
                             weights: down,
                             x: scratch.qwen38Xn,
                             y: scratch.qwen38Lo,
                             rows: Int(lowrank),
                             columns: hcDim,
                             tokenCount: Int(tokens),
                             xStrideElements: hcDim,
                             yStrideElements: Int(lowrank))
        hyperConnection.encodeSiluScale(commandBuffer: cb,
                                        z: scratch.qwen38Lo,
                                        out: scratch.qwen38Lo,
                                        n: tokens * lowrank,
                                        invHc: invHc)
        // Raw read-gate dot over the whole plane: z = up · lo.
        encodeInt4Projection(commandBuffer: cb,
                             family: .kv,
                             weights: up,
                             x: scratch.qwen38Lo,
                             y: scratch.qwen38GateRaw,
                             rows: hcDim,
                             columns: Int(lowrank),
                             tokenCount: Int(tokens),
                             xStrideElements: Int(lowrank),
                             yStrideElements: hcDim)
        if let w = blockInject, let inj = inject {
            // inject = block_inject row c · xn — a raw dot, no activation (the
            // combine's 2·sigmoid(·/hc) is applied on read).
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: w,
                                 x: scratch.qwen38Xn,
                                 y: inj, yBaseOffset: injectOffset,
                                 rows: Int(hc),
                                 columns: hcDim,
                                 tokenCount: Int(tokens),
                                 xStrideElements: hcDim,
                                 yStrideElements: Int(hc))
        }
        // gated = xn · sigmoid(z), collapsed per token to the [D] block input.
        hyperConnection.encodeGateMul(commandBuffer: cb,
                                      xn: scratch.qwen38Xn, z: scratch.qwen38GateRaw,
                                      out: scratch.qwen38Gated, n: tokens * UInt32(hcDim))
        hyperConnection.encodeSeqStreamMean(commandBuffer: cb,
                                            gated: scratch.qwen38Gated,
                                            out: blockOut, outOffset: blockOutOffset,
                                            d: d, hc: hc, tokens: tokens, invHc: invHc)
    }

    /// One Qwen 3.8 Flash-Next prefill layer: the chunked form of
    /// `encodeQwen38DecodeLayer` (qwen4exp.cpp decode :329-378). Per layer:
    ///   1. the PLE n-gram block on `pleLayerIndex`, reading the plane *as the
    ///      layer received it* and adding both of its terms into that same
    ///      plane before the mixer normalizes it;
    ///   2. attn mix — plane → xn → silu(÷hc) lowrank → sigmoid read gate →
    ///      stream mean = the [D] `normed` block input, plus the raw
    ///      block_inject dot (`qwen38Inject`);
    ///   3. attention body (full or GDN) at [D] width, the GDN body's output
    ///      norm taking a SIGMOID z-gate (the family's one GDN delta);
    ///   4. attn combine — plane += oOut · 2·sigmoid(inject/hc);
    ///   5. ffn mix on the updated plane → `denseX` + a fresh `qwen38Inject`;
    ///   6. router, shared expert, streamed routed tiles, then h2 = shared +
    ///      routed and plane += h2 · 2·sigmoid(inject/hc).
    /// There is no `hidden` write in a 3.8 layer — the plane *is* the residual,
    /// rebuilt from the embedding once per chunk.
    ///
    /// `docs/QWEN38_PORT.md` pins the math; the M3.4 parity test runs this
    /// against T decode steps of the same tokens.
    private func encodeQwen38PrefillLayer(
        _ L: Int,
        scratch: PrefillChunkScratchBuffers,
        startPosition: Int,
        tokenCount: Int,
        cb inbound: MTLCommandBuffer
    ) async throws -> MTLCommandBuffer {
        // `cb` is a local so the GDN sub-stage split can move the buffer
        // boundary mid-layer (see `splitGdnSubStage`); every encode site in this
        // body keeps saying `commandBuffer: cb` and follows it.
        var cb = inbound
        // The buffers the split closed, timed together after this layer’s one
        // existing wait. Empty unless `FQ_GDN_SPLIT=1`.
        var gdnSubStages: [(stage: Int, cb: MTLCommandBuffer)] = []
        let t = tokenCount
        let D = cfg.hiddenSize
        let hc = cfg.hyperConnectionCount
        let hcDim = cfg.hyperConnectionDim
        let lowrank = cfg.hyperConnectionLowrank
        let invHc: Float = 1.0 / Float(hc)
        let eps: Float = 1e-6
        let isFull = cfg.fullAttentionLayerMask[L] != 0
        let tokens32 = UInt32(t)
        let planeN = UInt32(hcDim)
        // The row every snapshot below is taken from: the chunk's last. After
        // the layer it holds exactly what T sequential decode steps leave in
        // the plane, which is what makes the M3.4 parity test a comparison of
        // like with like.
        let snapRow = t - 1

        let attnMixW = try model.attnHyperConnection(layer: L)
        let ffnMixW = try model.mlpHyperConnection(layer: L)
        let routerW = try model.router(layer: L)
        guard let onesEffective = qwenOnesEffectiveScale,
              let onesExpert = qwenOnesPerExpertScale else {
            preconditionFailure("Qwen prefill layer on a non-Qwen runner")
        }

        // The plane is private storage, so every debug snapshot is a blit of
        // one row to a shared buffer, encoded at the point in the layer where
        // that row is live and read after the CB carrying the blit completes.
        var snapshots: [(name: String, buffer: MTLBuffer, isFloat32: Bool)] = []
        func snapRowValue(_ name: String, _ src: MTLBuffer,
                          _ offsetElements: Int, into target: MTLCommandBuffer,
                          elements: Int? = nil,
                          stride: Int = MemoryLayout<Float16>.stride) {
            let bytes = (elements ?? hcDim) * stride
            guard qwenLayerDebugHook != nil,
                  let tmp = ctx.device.makeBuffer(length: bytes,
                                                  options: .storageModeShared),
                  let blit = target.makeBlitCommandEncoder() else { return }
            blit.copy(from: src, sourceOffset: offsetElements * stride,
                      to: tmp, destinationOffset: 0, size: bytes)
            blit.endEncoding()
            snapshots.append((name: name, buffer: tmp,
                              isFloat32: stride == MemoryLayout<Float>.stride))
        }
        func drainSnapshots() {
            guard let hook = qwenLayerDebugHook, !snapshots.isEmpty else { return }
            for snap in snapshots {
                if snap.isFloat32 {
                    let count = snap.buffer.length / MemoryLayout<Float>.stride
                    let ptr = snap.buffer.contents().bindMemory(to: Float.self,
                                                                capacity: count)
                    hook(L, snap.name,
                         Array(UnsafeBufferPointer(start: ptr, count: count)).map(Float16.init))
                } else {
                    let count = snap.buffer.length / MemoryLayout<Float16>.stride
                    let ptr = snap.buffer.contents().bindMemory(to: Float16.self,
                                                                capacity: count)
                    hook(L, snap.name, Array(UnsafeBufferPointer(start: ptr, count: count)))
                }
            }
            snapshots.removeAll()
        }
        snapRowValue("hc.pre", scratch.qwen38Plane, snapRow * hcDim, into: cb)
        // Layer entry: the previous layer's output, for every row. Agreement
        // here and disagreement at the next layer's entry puts the divergence
        // inside this layer.
        encodePlaneRowHash(scratch, layer: L, stage: 0, tokenCount: t,
                           rowBase: startPosition, into: cb)

        // --- PLE n-gram block (before the mixer normalizes the plane) --------
        if let ple, L == model.pleLayerIndex {
            // The projections are validated against the manifest's
            // `linearAttention` slot at load, so a 3.8 install whose PLE loads
            // at all has them int8 — the same kernel, and the same reason, as
            // the GDN projections.
            guard int8GEMV != nil else {
                preconditionFailure("Qwen 3.8 PLE requires the int8 GEMV path")
            }
            let keyP   = try model.pleKeyProj()
            let valueP = try model.pleValueProj()
            let normQ  = try model.pleNormQuery()
            let normK  = try model.pleNormKey()
            let normC  = try model.pleNormConv()
            let convW  = try model.pleConv1D()
            let gathered = cfg.ngramRowDim * (cfg.ngramSize - 1) * cfg.headsPerNgram
            // One stream's width: the gate's dot and both norm reductions run
            // over a single D-wide stream, not the whole plane.
            let invSqrtD = 1.0 / Float(D).squareRoot()
            let kern = UInt32(cfg.pleConvKernelSize)
            let dil  = UInt32(cfg.ngramSize)
            // No batched int8 QMM exists — one decode-style GEMV per projection
            // per token, exactly as the 3.6 GDN body does it.
            encodeRepeatedInt8(commandBuffer: cb,
                               weights: keyP,
                               x: scratch.qwen38PleGathered,
                               y: scratch.qwen38PleKey,
                               rows: hcDim,
                               columns: gathered,
                               tokenCount: t,
                               xStrideElements: gathered,
                               yStrideElements: hcDim)
            encodeRepeatedInt8(commandBuffer: cb,
                               weights: valueP,
                               x: scratch.qwen38PleGathered,
                               y: scratch.qwen38PleValue,
                               rows: D,
                               columns: gathered,
                               tokenCount: t,
                               xStrideElements: gathered,
                               yStrideElements: D)
            // Normed over one stream under a whole-plane gamma — the same
            // operator, on the same layout, as the HC mixers' norms.
            hyperConnection.encodeSeqGroupedRMS(commandBuffer: cb,
                                                x: scratch.qwen38PleKey,
                                                gamma: normK.buffer,
                                                gammaOffset: Int(normK.offset),
                                                out: scratch.qwen38PleKeyNormed,
                                                d: UInt32(D), hc: UInt32(hc),
                                                tokens: tokens32, eps: eps)
            hyperConnection.encodeSeqGroupedRMS(commandBuffer: cb,
                                                x: scratch.qwen38Plane,
                                                gamma: normQ.buffer,
                                                gammaOffset: Int(normQ.offset),
                                                out: scratch.qwen38PleQueryNormed,
                                                d: UInt32(D), hc: UInt32(hc),
                                                tokens: tokens32, eps: eps)
            ple.encodeSeqGate(commandBuffer: cb,
                              key: scratch.qwen38PleKeyNormed,
                              query: scratch.qwen38PleQueryNormed,
                              gate: scratch.qwen38PleGate,
                              d: UInt32(D), invSqrtD: invSqrtD,
                              hc: UInt32(hc), tokens: tokens32)
            ple.encodeSeqGatedValue(commandBuffer: cb,
                                    value: scratch.qwen38PleValue,
                                    gate: scratch.qwen38PleGate,
                                    gated: scratch.qwen38PleGated,
                                    d: UInt32(D), hc: UInt32(hc), tokens: tokens32)
            hyperConnection.encodeSeqGroupedRMS(commandBuffer: cb,
                                                x: scratch.qwen38PleGated,
                                                gamma: normC.buffer,
                                                gammaOffset: Int(normC.offset),
                                                out: scratch.qwen38PleConvIn,
                                                d: UInt32(D), hc: UInt32(hc),
                                                tokens: tokens32, eps: eps)
            // The conv history is persistent runner state (like the GDN conv
            // states): a chunk shorter than the receptive field takes its taps
            // out of the history, and the roll leaves the chunk's tail behind
            // for the next chunk. `newState` must not alias `state`.
            ple.encodeSeqConv(commandBuffer: cb,
                              weight: convW.buffer, weightOffset: Int(convW.offset),
                              state: pleConvState,
                              x: scratch.qwen38PleConvIn,
                              out: scratch.qwen38PleConvOut,
                              newState: scratch.qwen38PleConvNewState,
                              c: planeN, kernel: kern, dilation: dil,
                              tokens: tokens32)
            // Both PLE terms land in the plane the layer's mixer is about to
            // normalize: `query` above saw the plane *without* them.
            ple.encodePlaneAdd(commandBuffer: cb,
                               plane: scratch.qwen38Plane,
                               gated: scratch.qwen38PleGated,
                               conv: scratch.qwen38PleConvOut,
                               n: tokens32 * planeN)
        }

        // --- attn mix ---------------------------------------------------------
        encodeQwen38SeqMix(commandBuffer: cb,
                           norm: attnMixW.hcNorm, down: attnMixW.mixDown,
                           up: attnMixW.mixUp, blockInject: attnMixW.blockInject,
                           plane: scratch.qwen38Plane, planeOffset: 0,
                           blockOut: scratch.normed, blockOutOffset: 0,
                           inject: scratch.qwen38Inject, injectOffset: 0,
                           scratch: scratch,
                           d: UInt32(D), hc: UInt32(hc), lowrank: UInt32(lowrank),
                           tokens: tokens32, invHc: invHc, eps: eps)
        snapRowValue("attnBlockIn", scratch.normed, snapRow * D, into: cb, elements: D)

        if isFull {
            let qP = try model.qProj(layer: L)
            let kP = try model.kProj(layer: L)
            let vP = try model.vProj(layer: L)
            let oP = try model.oProj(layer: L)
            let qN = try model.qNorm(layer: L)
            let kN = try model.kNorm(layer: L)
            let headDim = cfg.fullHeadDim
            let numQ = cfg.numHeads
            let numKV = cfg.numFullKVHeads
            let qDim = numQ * headDim
            let kvDim = numKV * headDim
            let rotaryDim = Int(Double(headDim) * cfg.partialRotaryFactor)

            encodeInt4Projection(commandBuffer: cb,
                                 family: .q,
                                 weights: qP,
                                 x: scratch.normed,
                                 y: scratch.q,
                                 rows: 2 * qDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: 2 * qDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: kP,
                                 x: scratch.normed,
                                 y: scratch.kStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)
            encodeInt4Projection(commandBuffer: cb,
                                 family: .kv,
                                 weights: vP,
                                 x: scratch.normed,
                                 y: scratch.vStage,
                                 rows: kvDim,
                                 columns: D,
                                 tokenCount: t,
                                 xStrideElements: D,
                                 yStrideElements: kvDim)

            for row in 0..<t {
                qwenFusions.encodeFullAttnEpilogue(
                    commandBuffer: cb,
                    qProj: scratch.q, qProjOffset: row * 2 * qDim * 2,
                    qOut: scratch.q, qOutOffset: row * qDim * 2,
                    gateOut: scratch.qwenQGate, gateOutOffset: row * qDim * 2,
                    k: scratch.kStage, kOffset: row * kvDim * 2,
                    qWeight: qN.buffer, qWeightOffset: Int(qN.offset),
                    kWeight: kN.buffer, kWeightOffset: Int(kN.offset),
                    headDim: UInt32(headDim),
                    numQHeads: UInt32(numQ),
                    numKVHeads: UInt32(numKV),
                    position: UInt32(startPosition + row),
                    theta: Float(cfg.fullRopeTheta),
                    rotaryDim: UInt32(rotaryDim),
                    eps: eps)
            }

            // Hand the chunk's K/V to the cache before anything reads it. The
            // epilogue wrote `kStage` in this same CB, this blit is encoded
            // after it, and the per-row attention further down is encoded
            // after the blit — one CB, so submission order makes that exact.
            // `vStage` is copied raw: values take no rope.
            if let kv {
                try copyPrefillKVToCache(
                    commandBuffer: cb,
                    kv: kv,
                    layer: L,
                    startPosition: startPosition,
                    tokenCount: t,
                    keySource: scratch.kStage,
                    valueSource: scratch.vStage,
                    bytesPerToken: kvDim * MemoryLayout<Float16>.stride)
            }

            let qsaLayer = qsaState.flatMap { st in
                st.index(ofLayer: L).map { (state: st, index: $0) }
            }
            // No indexer means no selection: `Int.max` says "every position is
            // inside the selection width", so the dense path is taken
            // everywhere. The decode path already spells it this way
            // (`idxDense`); here a `?? 0` said the opposite — the width was 0,
            // so every position was outside it, and the prefill dispatched the
            // cells path with `nCells: 0`, which traps on
            // `precondition(nCells > 0)`. That made any Qwen 3.8 install
            // without indexer tensors — a configuration this file documents as
            // supported — crash in chunked prefill.
            let idxCapacity = qsaLayer?.state.capacity ?? Int.max
            // The position one past the chunk — both the KV length the
            // per-row attention sees and the end of the poolable block range.
            let endPosition = startPosition + t

            // A0: the q/k/v projections after the RoPE and norm epilogue. Against
            // `idxcells` this separates "the projections moved" from "the selection
            // moved" — the two readings the hunt could not tell apart from the plane.
            // The rotated K and V beside the queries: A0 alone covers only `q`, and
            // "the projections matched" is a claim about all three.
            encodeRowHash(scratch.kStage, layer: L, stage: 7,
                          rowCount: t, rowBase: startPosition,
                          rowStrideBytes: kvDim * MemoryLayout<Float16>.stride,
                          into: cb)
            encodeRowHash(scratch.vStage, layer: L, stage: 8,
                          rowCount: t, rowBase: startPosition,
                          rowStrideBytes: kvDim * MemoryLayout<Float16>.stride,
                          into: cb)
            encodeRowHash(scratch.q, layer: L, stage: 3,
                          rowCount: t, rowBase: startPosition,
                          rowStrideBytes: qDim * MemoryLayout<Float16>.stride,
                          into: cb)
            // MARK: QSA indexer (M3.4 chunk form)
            //
            // The chunk's indexer timeline in three sweeps, because the
            // per-token steps do not commute: every row's query projection and
            // key hand-off first (one batched QMM, then T `encodeQKPost`s), then
            // the blocks this chunk completes — a block is complete when the
            // chunk's last cell fills it, so the pool is one contiguous range,
            // not a per-row branch — then each row's score/select pair. The
            // four kernels are the M3.2d-validated ones verbatim, reached with
            // offsets.
            //
            // Nothing here is zero-filled. A block is pooled only once every
            // one of its cells has been written by `encodeQKPost`, and both
            // sub-chunk boundaries are accounted for: the range's first block
            // (`startPosition / r`) keeps the cells an earlier chunk wrote
            // before it, and the last complete block end is `endPosition / r`.
            // The incomplete tail block is scored but never pooled — the
            // kernel's +1e9 bias forces it visible, which is what the decode
            // path relies on too.
            if let qsaIndexer, let qsaLayer {
                let idxQK = try model.indexerQKProj(layer: L)
                let idxQGamma = try model.indexerQLayernorm(layer: L)
                let idxKGamma = try model.indexerKLayernorm(layer: L)
                let st = qsaLayer.state
                let li = qsaLayer.index
                let lay = st.layers[li]
                let r = st.r
                let idxDim = st.idxDim
                let nHeads = st.numQHeads
                let nKVHeads = st.numKVHeads
                let projDim = (nHeads + nKVHeads) * idxDim

                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: idxQK,
                                     x: scratch.normed,
                                     y: scratch.qwen38IdxQKProj,
                                     rows: projDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: projDim)
                for row in 0..<t {
                    let pos = startPosition + row
                    qsaIndexer.encodeQKPost(
                        commandBuffer: cb,
                        qk: scratch.qwen38IdxQKProj,
                        qkOffset: row * projDim * MemoryLayout<Float16>.stride,
                        qGamma: idxQGamma.buffer,
                        qGammaOffset: Int(idxQGamma.offset),
                        qOut: scratch.qwen38IdxQ,
                        qOutOffset: row * nHeads * idxDim * MemoryLayout<Float16>.stride,
                        kRaw: lay.rawKeys,
                        pos: UInt32(pos),
                        nHeads: UInt32(nHeads),
                        idxDim: UInt32(idxDim),
                        nRot: UInt32(st.nRot),
                        theta: st.theta,
                        eps: st.eps)
                }
                // The indexer's own q and its raw key timeline, after the
                // projection and the layer norms. The keys are the interesting
                // one: they live in a persistent per-layer timeline that this
                // chunk writes in place and a later chunk pools in completed
                // blocks, so a difference here is the indexer moving rather than
                // anything the attention or the planes did.
                encodeRowHash(scratch.qwen38IdxQ, layer: L, stage: 9,
                              rowCount: t, rowBase: startPosition,
                              rowStrideBytes: nHeads * idxDim
                                  * MemoryLayout<Float16>.stride,
                              into: cb)
                encodeRowHash(lay.rawKeys, layer: L, stage: 10,
                              rowCount: t, rowBase: startPosition,
                              rowStrideBytes: idxDim * MemoryLayout<Float16>.stride,
                              srcRowBase: startPosition,
                              into: cb)
                // Blocks whose every cell this chunk has written. `bFirst` is
                // the block holding `startPosition`, which an earlier chunk
                // may already have partly filled; `endPosition / r` blocks are
                // complete below the chunk's end.
                let bFirst = startPosition / r
                var poolCount = 0
                while (bFirst + poolCount) * r + r - 1 < endPosition {
                    poolCount += 1
                }
                if poolCount > 0 {
                    qsaIndexer.encodeBlockPoolNormRope(
                        commandBuffer: cb,
                        kRaw: lay.rawKeys,
                        kGamma: idxKGamma.buffer,
                        kGammaOffset: Int(idxKGamma.offset),
                        pooled: lay.pooled,
                        firstBlock: UInt32(bFirst),
                        blockCount: UInt32(poolCount),
                        r: UInt32(r),
                        idxDim: UInt32(idxDim),
                        nRot: UInt32(st.nRot),
                        theta: st.theta,
                        eps: st.eps)
                    st.advancePooledBlocks(li, by: poolCount)
                    // The pooled keys, immediately after the pool writes them and
                    // before any score reads them. This is the checkpoint the
                    // pool-vs-selection question needs: a difference here is the
                    // aggregation, and no difference here with the cells
                    // differing is the scoring and the radix select. Rows are
                    // *blocks* for this stage, indexed absolutely, so the source
                    // and the destination both start at the first new block.
                    encodeRowHash(lay.pooled, layer: L, stage: 11,
                                  rowCount: poolCount, rowBase: bFirst,
                                  rowStrideBytes: idxDim * MemoryLayout<Float16>.stride,
                                  srcRowBase: bFirst,
                                  into: cb)
                }
                for row in 0..<t {
                    let pos = startPosition + row
                    guard idxCapacity < pos + 1 else { continue }
                    qsaIndexer.encodeBlockScores(commandBuffer: cb,
                                                 q: scratch.qwen38IdxQ,
                                                 qOffset: row * nHeads * idxDim
                                                    * MemoryLayout<Float16>.stride,
                                                 pooled: lay.pooled,
                                                 scores: scratch.qwen38IdxScore,
                                                 nHeads: UInt32(nHeads),
                                                 idxDim: UInt32(idxDim),
                                                 r: UInt32(r),
                                                 nKv: UInt32(pos + 1),
                                                 pos: UInt32(pos))
                    qsaIndexer.encodeSelectCells(
                        commandBuffer: cb,
                        scores: scratch.qwen38IdxScore,
                        cells: scratch.qwen38Cells,
                        cellsOffset: row * idxCapacity * MemoryLayout<UInt32>.stride,
                        count: scratch.qwen38CellCount,
                        pos: UInt32(pos),
                        nKv: UInt32(pos + 1),
                        r: UInt32(r),
                        budget: UInt32(st.budget))
                }
            }

            // Per row, exactly as decode: the dense rows take the whole
            // timeline, the rows past the selection width take the indexer's
            // cell list. Both are the decode kernels — `encodeFull` builds its
            // causal mask from `seqLen`, `encodeFullCells` reads the cells the
            // select just wrote — so a chunk row and a decode step are the same
            // arithmetic. The batched `encodeCausal` path is deliberately NOT
            // used here: it is a different kernel, and a 3.8 chunk would then
            // attend two ways at once.
            guard let kv else {
                throw PrefillError.chunkedUnsupported(
                    "chunked prefill attention requires FP16 KV")
            }
            let kBuf = kv.keyBuffer(layer: L, validTokenCount: endPosition)
            let vBuf = kv.valueBuffer(layer: L, validTokenCount: endPosition)
            for row in 0..<t {
                let pos = startPosition + row
                if idxCapacity >= pos + 1 {
                    attention.encodeFull(commandBuffer: cb,
                                         q: scratch.q, qOffset: row * qDim * 2,
                                         k: kBuf, kOffset: 0,
                                         v: vBuf, vOffset: 0,
                                         out: scratch.attentionOutput,
                                         outOffset: row * qDim * 2,
                                         headDim: UInt32(headDim),
                                         numQHeads: UInt32(numQ),
                                         numKVHeads: UInt32(numKV),
                                         seqLen: UInt32(pos + 1),
                                         scale: nil)   // rsqrt(head_dim)
                } else {
                    // The selection fills its width exactly, so the cell count
                    // is `capacity` without reading back the kernel's counter.
                    attention.encodeFullCells(
                        commandBuffer: cb,
                        q: scratch.q, qOffset: row * qDim * 2,
                        k: kBuf, kOffset: 0,
                        v: vBuf, vOffset: 0,
                        cells: scratch.qwen38Cells,
                        cellsOffset: row * idxCapacity * MemoryLayout<UInt32>.stride,
                        out: scratch.attentionOutput, outOffset: row * qDim * 2,
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(numQ),
                        numKVHeads: UInt32(numKV),
                        nCells: UInt32(idxCapacity),
                        scale: nil)   // rsqrt(head_dim)
                }
            }

            for row in 0..<t {
            // A1: the QSA selection for the whole chunk, one row per position, at the
            // selection width. Hashed here — after the row loop that writes it, not
            // inside it — because a partial selection would fingerprint as a
            // divergence of its own. This stage decides whether the ranking is where
            // a divergence starts.
            if idxCapacity >= 1 && idxCapacity <= 65536 {
                encodeRowHash(scratch.qwen38Cells, layer: L, stage: 4,
                              rowCount: t, rowBase: startPosition,
                              rowStrideBytes: idxCapacity * MemoryLayout<UInt32>.stride,
                              into: cb)
            }
            // A2: the attention output, before the gate. Differing with `qkv` equal
            // is the attention itself.
            encodeRowHash(scratch.attentionOutput, layer: L, stage: 5,
                          rowCount: t, rowBase: startPosition,
                          rowStrideBytes: qDim * MemoryLayout<Float16>.stride,
                          into: cb)
                qwenFusions.encodeAttnOutputGate(commandBuffer: cb,
                                                 attn: scratch.attentionOutput,
                                                 attnOffset: row * qDim * 2,
                                                 gate: scratch.qwenQGate,
                                                 gateOffset: row * qDim * 2,
                                                 n: UInt32(qDim))
            }
            encodeInt4Projection(commandBuffer: cb,
                                 family: .o,
                                 weights: oP,
                                 x: scratch.attentionOutput,
                                 y: scratch.qwen38OOut,
                                 rows: D,
                                 columns: qDim,
                                 tokenCount: t,
                                 xStrideElements: qDim,
                                 yStrideElements: D)
            // A3: the block output, after the gate and o_proj.
            encodeRowHash(scratch.qwen38OOut, layer: L, stage: 6,
                          rowCount: t, rowBase: startPosition,
                          rowStrideBytes: D * MemoryLayout<Float16>.stride,
                          into: cb)
            snapRowValue("attnBlockOut", scratch.qwen38OOut, snapRow * D, into: cb, elements: D)
        } else {
            // GDN (linear-attention) layer: identical to the Qwen 3.6 body
            // except the output norm's z-gate — sigmoid here, silu there
            // (qwen4exp `build_norm_gated` :411-421) — and the out_proj target,
            // which lands in the plane-visible `qwen38OOut` rather than the
            // [D] `h1` the 3.6 tail adds to.
            let si = gdnStateIndexByLayer[L]
            precondition(si >= 0, "GDN layer \(L) without state")
            let qkvP = try model.gdnInProjQKV(layer: L)
            let zP = try model.gdnInProjZ(layer: L)
            let outP = try model.gdnOutProj(layer: L)
            let convW = try model.gdnConv1D(layer: L)
            let aLog = try model.gdnALog(layer: L)
            let dt = try model.gdnDtBias(layer: L)
            let normW = try model.gdnNormWeight(layer: L)
            let recState = gdnRecurrentState[si]
            let convState = gdnConvState[si]
            let aP = linearAttnBits == 8 ? try model.gdnInProjA(layer: L) : nil
            let bP = linearAttnBits == 8 ? try model.gdnInProjB(layer: L) : nil
            let keyDim = cfg.linearNumKeyHeads * cfg.linearKeyHeadDim
            let valueDim = cfg.linearNumValueHeads * cfg.linearValueHeadDim
            let qkvDim = 2 * keyDim + valueDim
            let numV = cfg.linearNumValueHeads
            let headDim = cfg.linearValueHeadDim
            let scale = 1.0 / Float(cfg.linearKeyHeadDim).squareRoot()
            let betaByteOffset = t * numV * MemoryLayout<Float>.size

            // Stage 1 (the input projections: qkv, z, and the gate) opens here.
            // Everything before it — the hyper-connection norm into `normed` —
            // belongs to the layer’s own buffer.
            cb = splitGdnSubStage(cb, closing: nil, into: &gdnSubStages)
            if linearAttnBits == 8 {
                let qkvBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: qkvP, x: scratch.normed,
                    y: scratch.qwenQKVProj, rows: qkvDim, columns: D,
                    tokenCount: t, yStrideElements: qkvDim)
                if !qkvBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: qkvP,
                                       x: scratch.normed,
                                       y: scratch.qwenQKVProj,
                                       rows: qkvDim,
                                       columns: D,
                                       tokenCount: t,
                                       xStrideElements: D,
                                       yStrideElements: qkvDim)
                }
                let zBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: zP, x: scratch.normed,
                    y: scratch.qwenZ, rows: valueDim, columns: D,
                    tokenCount: t, yStrideElements: valueDim)
                if !zBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: zP,
                                       x: scratch.normed,
                                       y: scratch.qwenZ,
                                       rows: valueDim,
                                       columns: D,
                                       tokenCount: t,
                                       xStrideElements: D,
                                       yStrideElements: valueDim)
                }
                encodeRepeatedInt8(commandBuffer: cb,
                                   weights: aP!,
                                   x: scratch.normed,
                                   y: scratch.qwenAB,
                                   rows: numV,
                                   columns: D,
                                   tokenCount: t,
                                   xStrideElements: D,
                                   yStrideElements: 2 * numV,
                                   yBaseElements: 0)
                encodeRepeatedInt8(commandBuffer: cb,
                                   weights: bP!,
                                   x: scratch.normed,
                                   y: scratch.qwenAB,
                                   rows: numV,
                                   columns: D,
                                   tokenCount: t,
                                   xStrideElements: D,
                                   yStrideElements: 2 * numV,
                                   yBaseElements: numV)
            } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: qkvP,
                                     x: scratch.normed,
                                     y: scratch.qwenQKVProj,
                                     rows: qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: zP,
                                     x: scratch.normed,
                                     y: scratch.qwenZ,
                                     rows: valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: valueDim)
                let gateW = gdnGateWeights[si]
                prefillQMM.encode(commandBuffer: cb,
                                  weights: gateW.weights,
                                  scales: gateW.scales,
                                  biases: gateW.biases,
                                  x: scratch.normed,
                                  y: scratch.qwenAB,
                                  t: t,
                                  n: 2 * numV,
                                  k: D)
            }
            // Stage 2 (A_log/dt_bias gating and the conv1d) opens here.
            cb = splitGdnSubStage(cb, closing: 0, into: &gdnSubStages)
            gdnPrefill.encodeGateBatch(commandBuffer: cb,
                                       ab: scratch.qwenAB,
                                       A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                       dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                       g: scratch.qwenGBeta,
                                       beta: scratch.qwenGBeta, betaOffset: betaByteOffset,
                                       numValueHeads: numV,
                                       tokens: t)
            gdnPrefill.encodeConvChunk(commandBuffer: cb,
                                       w: convW.buffer, wOffset: Int(convW.offset),
                                       state: convState,
                                       x: scratch.qwenQKVProj,
                                       out: scratch.qwenQKVConvOut,
                                       newState: scratch.qwenConvNewState,
                                       channels: qkvDim,
                                       tokens: t)
            snapRowValue("qkvProjected", scratch.qwenQKVProj, snapRow * qkvDim,
                         into: cb, elements: qkvDim)
            snapRowValue("qkvConv", scratch.qwenQKVConvOut, snapRow * qkvDim,
                         into: cb, elements: qkvDim)
            snapRowValue("gFloat", scratch.qwenGBeta, snapRow * numV, into: cb,
                         elements: numV, stride: MemoryLayout<Float>.stride)
            // Stage 3 (the chunked recurrent scan) opens here and carries the
            // gated RMSNorm that closes it.
            cb = splitGdnSubStage(cb, closing: 1, into: &gdnSubStages)
            gdnPrefill.encodeRecurrentSeq(commandBuffer: cb,
                                          state: recState,
                                          conv: scratch.qwenQKVConvOut,
                                          g: scratch.qwenGBeta,
                                          beta: scratch.qwenGBeta, betaOffset: betaByteOffset,
                                          out: scratch.qwenRecOut,
                                          headDim: UInt32(headDim),
                                          channels: UInt32(qkvDim),
                                          kOffset: UInt32(keyDim),
                                          vOffset: UInt32(2 * keyDim),
                                          numValueHeads: numV,
                                          numKeyHeads: cfg.linearNumKeyHeads,
                                          tokens: t,
                                          scale: scale,
                                          l2eps: eps)
            snapRowValue("recState", recState, 0, into: cb,
                         elements: numV * headDim * headDim,
                         stride: MemoryLayout<Float>.stride)
            gdnPrefill.encodeRMSNormGatedBatch(commandBuffer: cb,
                                               x: scratch.qwenRecOut,
                                               z: scratch.qwenZ,
                                               weight: normW.buffer, weightOffset: Int(normW.offset),
                                               out: scratch.qwenRecOut,
                                               headDim: UInt32(headDim),
                                               numValueHeads: numV,
                                               tokens: t,
                                               eps: eps,
                                               activation: .sigmoid)
            // Snapshotted AFTER the gated norm, matching decode: its
            // `recurrentOut` hook reads `attnOut` after `gNormGated` wrote it
            // back in place, so the two stages must be compared at the same
            // point in the pipeline.
            snapRowValue("recurrentOut", scratch.qwenRecOut, snapRow * valueDim,
                         into: cb, elements: valueDim)
            if linearAttnBits == 8 {
            // Stage 4, the output projection. A GEMM like stage 1, but it runs
            // after the scan, so it cannot share stage 1’s buffer — which is why
            // four stages, not three, are what sum against the layer’s GDN time.
            cb = splitGdnSubStage(cb, closing: 2, into: &gdnSubStages)
                let outProjBatched = encodeInt8ProjectionBatched(
                    commandBuffer: cb, weights: outP, x: scratch.qwenRecOut,
                    y: scratch.qwen38OOut, rows: D, columns: valueDim,
                    tokenCount: t, yStrideElements: D)
                if !outProjBatched {
                    encodeRepeatedInt8(commandBuffer: cb,
                                       weights: outP,
                                       x: scratch.qwenRecOut,
                                       y: scratch.qwen38OOut,
                                       rows: D,
                                       columns: valueDim,
                                       tokenCount: t,
                                       xStrideElements: valueDim,
                                       yStrideElements: D)
                }
            } else {
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: outP,
                                     x: scratch.qwenRecOut,
                                     y: scratch.qwen38OOut,
                                     rows: D,
                                     columns: valueDim,
                                     tokenCount: t,
                                     xStrideElements: valueDim,
                                     yStrideElements: D)
            }
            snapRowValue("attnBlockOut", scratch.qwen38OOut, snapRow * D, into: cb, elements: D)
            cb = splitGdnSubStage(cb, closing: 3, into: &gdnSubStages)
        }

        // --- attn combine, then the ffn mix on the updated plane ------------
        hyperConnection.encodeSeqCombine(commandBuffer: cb,
                                         plane: scratch.qwen38Plane,
                                         blockOut: scratch.qwen38OOut,
                                         inject: scratch.qwen38Inject,
                                         d: UInt32(D), hc: UInt32(hc),
                                         tokens: tokens32, invHc: invHc)
        snapRowValue("hc.mid", scratch.qwen38Plane, snapRow * hcDim, into: cb)
        // After the attention block's write-back, before the routed tail.
        encodePlaneRowHash(scratch, layer: L, stage: 1, tokenCount: t,
                           rowBase: startPosition, into: cb)
        encodeQwen38SeqMix(commandBuffer: cb,
                           norm: ffnMixW.hcNorm, down: ffnMixW.mixDown,
                           up: ffnMixW.mixUp, blockInject: ffnMixW.blockInject,
                           plane: scratch.qwen38Plane, planeOffset: 0,
                           blockOut: scratch.denseX, blockOutOffset: 0,
                           inject: scratch.qwen38Inject, injectOffset: 0,
                           scratch: scratch,
                           d: UInt32(D), hc: UInt32(hc), lowrank: UInt32(lowrank),
                           tokens: tokens32, invHc: invHc, eps: eps)
        snapRowValue("ffnBlockIn", scratch.denseX, snapRow * D, into: cb, elements: D)

        // Router (both layer types): plain softmax over all experts, top-8
        // renormalized — identical math to the kernel's top-8 softmax, so the
        // Gemma block is reused with ones-filled scales (as in decode).
        prefillRouter.encodeGemma4Block(
            commandBuffer: cb,
            weights: routerW.buffer,
            weightsOffset: Int(routerW.offset),
            scales: routerW.buffer,
            scalesOffset: Int(routerW.scaleOffset),
            biases: routerW.buffer,
            biasesOffset: Int(routerW.biasOffset),
            hidden: scratch.denseX,
            effectiveScale: onesEffective,
            perExpertScale: onesExpert,
            perExpertScaleOffset: 0,
            outIndices: scratch.routeIDs,
            outWeights: scratch.routeWeights,
            queryCount: UInt32(t),
            numExperts: UInt32(cfg.numExperts),
            d: UInt32(D),
            topK: UInt32(cfg.topKExperts),
            hiddenStrideElements: UInt32(D))
        commitCountingPrefill(cb)
        waitForCompletion(cb)
        if let error = cb.error {
            throw error
        }
        // The layer’s own forward: attention or GDN, plus its projections.
        // The same bucket the decode path calls cb1, and the same split by
        // layer kind.
        recordPrefillLayerGpuTime(cb, isFull: isFull)
        // Every split buffer was committed before this one and the queue is
        // ordered, so the wait above completed them all and their timestamps
        // are now readable. Nothing was waited on where it was committed.
        for sub in gdnSubStages {
            guard sub.cb.status == .completed else { continue }
            switch sub.stage {
            case 0: totalGpuGdnProjNanos &+= recordGpuTime(sub.cb)
            case 1: totalGpuGdnConvNanos &+= recordGpuTime(sub.cb)
            case 2: totalGpuGdnScanNanos &+= recordGpuTime(sub.cb)
            default: totalGpuGdnOutProjNanos &+= recordGpuTime(sub.cb)
            }
        }
        gdnSubStages.removeAll()
        drainSnapshots()

        // CPU readback of the router indices → expert grouping, as in decode.
        let routeCount = t * cfg.topKExperts
        let idPtr = scratch.routeIDs.contents()
            .bindMemory(to: UInt32.self, capacity: routeCount)
        let weightPtr = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: routeCount)
        var routeIDs = [UInt32]()
        routeIDs.reserveCapacity(routeCount)
        var routeWeights = [Float16]()
        routeWeights.reserveCapacity(routeCount)
        for i in 0..<routeCount {
            routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
            routeWeights.append(weightPtr[i])
        }
        let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                       weights: routeWeights,
                                                       queryCount: t,
                                                       topK: cfg.topKExperts)
        let schedulerConfig = prefillRoutedTileSchedulerConfig
        let routeTileExpertCount: Int
        if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
            guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                throw PrefillError.chunkedUnsupported(
                    "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
            }
            routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
        } else {
            routeTileExpertCount = schedulerConfig.tileExperts
        }
        let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
            pairs,
            queryCount: t,
            topK: cfg.topKExperts,
            numExperts: cfg.numExperts,
            tileExpertCount: routeTileExpertCount,
            expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

        // Shared expert (silu) on an early-committed CB, then the Qwen post
        // stage: sigmoid(shared_expert_gate · x) scales h1 per token.
        guard let sharedCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        let sharedProj = sharedExpertProjections[L]
        try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                            x: scratch.denseX,
                                            y: scratch.h1,
                                            gate: sharedProj.gate,
                                            up: sharedProj.up,
                                            down: sharedProj.down,
                                            scratchGate: scratch.sharedGateScratch,
                                            scratchUp: scratch.sharedUpScratch,
                                            scratchAct: scratch.sharedActScratch,
                                            queryCount: t,
                                            d: D,
                                            intermediate: cfg.intermediateSize,
                                            xStrideElements: D,
                                            yStrideElements: D,
                                            activation: .silu)
        let sharedGate = try model.sharedExpertGateProj(layer: L)
        for row in 0..<t {
            qwenFusions.encodeSharedGate(commandBuffer: sharedCB,
                                         weights: sharedGate.buffer,
                                         weightsOffset: Int(sharedGate.offset),
                                         scales: sharedGate.buffer,
                                         scalesOffset: Int(sharedGate.scaleOffset),
                                         biases: sharedGate.buffer,
                                         biasesOffset: Int(sharedGate.biasOffset),
                                         x: scratch.denseX, xOffset: row * D * 2,
                                         h1: scratch.h1, h1Offset: row * D * 2,
                                         n: UInt32(D), d: UInt32(D))
        }
        commitCountingPrefill(sharedCB)
        waitForCompletion(sharedCB)
        if let error = sharedCB.error {
            throw error
        }
        totalGpuRoutedNanos &+= recordGpuTime(sharedCB)

        // Streamed routed-expert tiles (silu), mirroring the Gemma tile loop.
        let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
            device: ctx.device,
            routes: routes)
        let routedOffsets = model.routedExpertOffsets(layer: L)
        struct PendingPrefillTile {
            let tileIndex: Int
            let commandBuffer: MTLCommandBuffer
            let fetch: PrefillStreamedTileFetchResult
            let argumentBuffer: PrefillStreamedTileArgumentBuffer
        }
        var pendingTiles: [PendingPrefillTile] = []
        var tileLifetime = PrefillStreamedTileSlotLifetime()
        func drainOldestPendingTile() throws {
            guard !pendingTiles.isEmpty else { return }
            let pending = pendingTiles.removeFirst()
            withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                waitForCompletion(pending.commandBuffer)
            totalGpuRoutedNanos &+= recordGpuTime(pending.commandBuffer)
            }
            if let error = pending.commandBuffer.error {
                throw error
            }
            if !pending.fetch.plannedMissSlots.isEmpty {
                try tileLifetime.complete(tileIndex: pending.tileIndex)
            }
        }

        let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
        for (tileIndex, tile) in routes.tiles.enumerated() {
            let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                forTile: tileIndex,
                routes: routes)
            var plannedFetch: RoutedExpertFetchPlan?
            if !pendingTiles.isEmpty {
                let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                if !pendingAssignedSlots.isEmpty {
                    let pendingSlots = Set(pendingAssignedSlots)
                    let plan = try model.planRoutedExpertsIfPossible(
                        layer: L,
                        experts: expertIDs,
                        avoidingSlots: pendingSlots)
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: pendingAssignedSlots,
                            avoidingSlotPlanAvailable: plan != nil))
                    switch decision {
                    case .prefetchNext:
                        guard let plan else {
                            throw ModelError.indexCorrupt(
                                detail: "routed tile scheduler requested missing plan")
                        }
                        plannedFetch = plan
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler ignored pending tile")
                    }
                } else {
                    let decision = routedTileScheduler.decide(
                        PrefillRoutedTileSchedulerInput(
                            hasPendingTile: true,
                            pendingDepth: pendingTiles.count,
                            pendingAssignedSlots: [],
                            avoidingSlotPlanAvailable: false))
                    switch decision {
                    case .drainBeforeIssue:
                        try drainOldestPendingTile()
                    case .issueWithoutPending, .prefetchNext:
                        throw ModelError.indexCorrupt(
                            detail: "routed tile scheduler failed to drain empty-slot pending tile")
                    }
                }
            } else {
                let decision = routedTileScheduler.decide(
                    PrefillRoutedTileSchedulerInput(
                        hasPendingTile: false,
                        pendingAssignedSlots: [],
                        avoidingSlotPlanAvailable: false))
                switch decision {
                case .issueWithoutPending:
                    break
                case .prefetchNext, .drainBeforeIssue:
                    throw ModelError.indexCorrupt(
                        detail: "routed tile scheduler requested pending action without pending tile")
                }
            }
            let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                model: model,
                layer: L,
                tileIndex: tileIndex,
                routes: routes,
                plannedFetch: plannedFetch,
                avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
            try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                  pairStart: Int(tile.pairStart),
                                                  pairCount: Int(tile.pairCount))
            totalPrefillExpertMisses &+= UInt64(fetch.plannedMissSlots.count)
            totalPrefillTiles &+= 1
            if !fetch.plannedMissSlots.isEmpty {
                try tileLifetime.begin(tileIndex: tileIndex,
                                       plannedSlots: fetch.plannedMissSlots)
            }
            let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                device: ctx.device,
                binding: fetch.binding)
            let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                pairStart: tile.pairStart,
                pairCount: tile.pairCount,
                d: UInt32(D),
                routedIntermediate: UInt32(cfg.moeIntermediateSize),
                topK: UInt32(cfg.topKExperts),
                hiddenStrideElements: UInt32(D),
                binding: fetch.binding,
                offsets: routedOffsets)
            guard let tileCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            _ = prefillGroupedMoE.encodeStreamedBatched(
                commandBuffer: tileCB,
                hidden: scratch.denseX,
                sortedPairs: metadata.sortedPairs,
                routePartials: scratch.routePartials,
                gateUpActScratch: scratch.routedGateUpActScratch,
                downScratch: scratch.routedDownScratch,
                argumentBuffer: argumentBuffer,
                binding: fetch.binding,
                params: streamedParams,
                pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows,
                activation: .silu)
            commitCountingPrefill(tileCB)
            pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                   commandBuffer: tileCB,
                                                   fetch: fetch,
                                                   argumentBuffer: argumentBuffer))
            while pendingTiles.count > schedulerConfig.maxPendingDepth {
                try drainOldestPendingTile()
            }
        }
        while !pendingTiles.isEmpty {
            try drainOldestPendingTile()
        }

        // Tail: h2 = routed reduce, h2 += h1 (the shared expert, already
        // gate-scaled — the decode path's phase-2 residual), then the MLP
        // combine scatters h2 into the plane. 3.8 has no `hidden` write, no
        // layer_scalar and no sandwich norms.
        guard let tailCB = ctx.queue.makeCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                          routePartials: scratch.routePartials,
                                          routeWeights: scratch.routeWeights,
                                          h2: scratch.h2,
                                          queryCount: UInt32(t),
                                          topK: UInt32(cfg.topKExperts),
                                          d: UInt32(D))
        for row in 0..<t {
            qwenFusions.encodeVecAdd(commandBuffer: tailCB,
                                     a: scratch.h2, aOffset: row * D * 2,
                                     b: scratch.h1, bOffset: row * D * 2,
                                     d: UInt32(D))
        }
        snapRowValue("mlpBlockIn", scratch.h2, snapRow * D, into: tailCB,
                     elements: D)
        snapRowValue("sharedOut", scratch.h1, snapRow * D, into: tailCB,
                     elements: D)
        hyperConnection.encodeSeqCombine(commandBuffer: tailCB,
                                         plane: scratch.qwen38Plane,
                                         blockOut: scratch.h2,
                                         inject: scratch.qwen38Inject,
                                         d: UInt32(D), hc: UInt32(hc),
                                         tokens: tokens32, invHc: invHc)
        snapRowValue("hc.post", scratch.qwen38Plane, snapRow * hcDim, into: tailCB)
        // After the routed-expert tail. Encode into `tailCB` and not `cb`: this
        // CB is the one committed here, so the hash lands after the tail that
        // produced the plane.
        encodePlaneRowHash(scratch, layer: L, stage: 2, tokenCount: t,
                           rowBase: startPosition, into: tailCB)
        commitCountingPrefill(tailCB)
        withExtendedLifetime(metadata) {
            waitForCompletion(tailCB)
        }
        if let error = tailCB.error {
            throw error
        totalGpuRoutedNanos &+= recordGpuTime(tailCB)
        }
        drainSnapshots()

        if L + 1 < cfg.numLayers {
            guard let nextCB = ctx.queue.makeCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            return nextCB
        }
        return cb
    }

    /// Qwen 3.8 Flash-Next hyper-connection mix stage — llama `build_hc_mix`
    /// (`archive/llama.cpp/src/models/qwen4exp.cpp` :226-262): grouped RMS
    /// over the [hc·D] plane scaled by the raw-BF16 [hc·D] gamma → int4 down
    /// [lowrank × hc·D] → silu(·÷hc) → int4 up [hc·D × lowrank] → sigmoid
    /// read gate × xn → mean over the hc streams into the [D] `blockOut`. The
    /// per-stream scatter weights (raw `block_inject` row dot over xn, no
    /// activation) land in `inject` [hc]; the root mixer passes nil for both
    /// `blockInject` and `inject`. All stages encode serially into `cb` — each
    /// reads only what an earlier stage in the same CB wrote, so one shared
    /// scratch set (`hcXn`/`hcLo`/`hcGateRaw`/`hcGated`) is reused by the
    /// layer's second mixer.
    private func encodeQwen38Mix(
        commandBuffer cb: MTLCommandBuffer,
        norm: TensorView,
        down: TensorView,
        up: TensorView,
        blockInject: TensorView?,
        blockOut: MTLBuffer,
        inject: MTLBuffer?,
        d: UInt32,
        hc: UInt32,
        lowrank: UInt32,
        invHc: Float,
        eps: Float
    ) {
        precondition((blockInject == nil) == (inject == nil),
                     "block_inject and inject buffers come as a pair")
        let hcDim = UInt32(Int(d) * Int(hc))

        // xn = per-stream RMS of the plane, scaled by the [hc·D] BF16 gamma.
        hyperConnection.encodeGroupedRMS(commandBuffer: cb,
                                         x: hcPlane,
                                         gamma: norm.buffer,
                                         gammaOffset: Int(norm.offset),
                                         out: hcXn,
                                         d: d, hc: hc, eps: eps)
        // lo = silu(down · xn · 1/hc) — the ÷hc precedes the silu.
        int4.encode(commandBuffer: cb,
                    weights: down.buffer, weightsOffset: Int(down.offset),
                    scales: down.buffer, scalesOffset: Int(down.scaleOffset),
                    biases: down.buffer, biasesOffset: Int(down.biasOffset),
                    x: hcXn, y: hcLo, m: lowrank, n: hcDim)
        hyperConnection.encodeSiluScale(commandBuffer: cb,
                                        z: hcLo, out: hcLo,
                                        n: lowrank, invHc: invHc)
        // Raw read-gate dot over the whole plane: z = up · lo.
        int4.encode(commandBuffer: cb,
                    weights: up.buffer, weightsOffset: Int(up.offset),
                    scales: up.buffer, scalesOffset: Int(up.scaleOffset),
                    biases: up.buffer, biasesOffset: Int(up.biasOffset),
                    x: hcLo, y: hcGateRaw, m: hcDim, n: lowrank)
        if let w = blockInject, let inj = inject {
            // inject[c] = block_inject row c · xn — a raw dot, no activation
            // (the combine's 2·sigmoid(·/hc) is applied on read).
            int4.encode(commandBuffer: cb,
                        weights: w.buffer, weightsOffset: Int(w.offset),
                        scales: w.buffer, scalesOffset: Int(w.scaleOffset),
                        biases: w.buffer, biasesOffset: Int(w.biasOffset),
                        x: hcXn, y: inj, m: hc, n: hcDim)
        }
        // gated = xn · sigmoid(z), collapsed to the [D] block input.
        hyperConnection.encodeGateMul(commandBuffer: cb,
                                      xn: hcXn, z: hcGateRaw,
                                      out: hcGated, n: hcDim)
        hyperConnection.encodeStreamMean(commandBuffer: cb,
                                         gated: hcGated,
                                         out: blockOut,
                                         d: d, hc: hc, invHc: invHc)
    }

    /// One Qwen 3.8 Flash-Next decode layer: the hyper-connection mixers
    /// replace both per-layer norms (llama `qwen4exp.cpp` decode :329-378).
    /// Per layer:
    ///   1. attn mix — plane → xn → silu(÷hc) lowrank → sigmoid read gate →
    ///      stream mean = the [D] attn block input (`normed`), plus the raw
    ///      block_inject dot (`hcInject`).
    ///   2. attention body (full or GDN) at [D] width — the same bodies as
    ///      Qwen 3.6, except the GDN output norm's z-gate is a SIGMOID
    ///      (`qwen4exp` `build_norm_gated` :411-421, the family's sole GDN
    ///      numerical delta).
    ///   3. attn combine — plane += oOut · 2·sigmoid(inject/hc).
    ///   4. ffn mix on the updated plane → `denseX` + a fresh `hcInject`.
    ///   5. router on `denseX`; the deferred tail runs the experts and closes
    ///      with plane += h2 · 2·sigmoid(inject/hc).
    /// `hidden` is untouched by the layer — the plane is the residual (it is
    /// rebuilt from the embedding each step in `produceToken`).
    ///
    /// Debug-hook snapshots: "hc.pre"/"hc.post" bracket the layer's plane
    /// writes (post is llama's `l_last`), with "attnBlockIn"/"attnBlockOut"/
    /// "hc.mid"/"ffnBlockIn" between the stages.
    private func encodeQwen38DecodeLayer(
        _ L: Int,
        position: Int,
        pending: inout PendingRoutedCommand?
    ) async throws {
        let D = UInt32(cfg.hiddenSize)
        let hc = UInt32(cfg.hyperConnectionCount)
        let lowrank = UInt32(cfg.hyperConnectionLowrank)
        let invHc: Float = 1.0 / Float(cfg.hyperConnectionCount)
        let eps: Float = 1e-6
        let isFull = cfg.fullAttentionLayerMask[L] != 0
        let seqLen = UInt32(position + 1)
        let hcDim = cfg.hyperConnectionDim

        let attnMixW = try model.attnHyperConnection(layer: L)
        let ffnMixW = try model.mlpHyperConnection(layer: L)
        let routerW = try model.router(layer: L)
        let sharedProj = sharedExpertProjections[L]
        let sharedGate = try model.sharedExpertGateProj(layer: L)
        guard let onesEffective = qwenOnesEffectiveScale,
              let onesExpert = qwenOnesPerExpertScale else {
            preconditionFailure("Qwen decode layer on a non-Qwen runner")
        }

        let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // Every cb1 sub-bucket is measured by advancing this one cursor, so the
        // spans tile the layer's cb1 exactly. See `lapseCb1`.
        var cb1Cursor = tCb1Start
        let cb = ctx.queue.makeCommandBuffer()!

        if let hook = qwenLayerDebugHook {
            func snap(_ name: String, _ buf: MTLBuffer, _ count: Int) {
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
                hook(L, name, Array(UnsafeBufferPointer(start: ptr, count: count)))
            }
            snap("preLayer", hidden, cfg.hiddenSize)
            // hc.pre: the wide plane entering the layer (layer 0: the
            // embedding copies from plane init; later layers: the previous
            // layer's l_last). The plane is only written by the deferred
            // tail, which a hook run already waited out at the previous
            // layer's hc.post snapshot, so this read is race-free.
            snap("hc.pre", hcPlane, hcDim)
        }

        // The two mixers of the layer and the attention scatter — all encode
        // serially into CB1. The ffn mix reuses the same scratch after the
        // attn combine consumed it.
        let gAttnMix: (MTLCommandBuffer) -> Void = { [self] cb in
            encodeQwen38Mix(commandBuffer: cb,
                            norm: attnMixW.hcNorm, down: attnMixW.mixDown,
                            up: attnMixW.mixUp, blockInject: attnMixW.blockInject,
                            blockOut: normed, inject: hcInject,
                            d: D, hc: hc, lowrank: lowrank, invHc: invHc,
                            eps: eps)
        }
        let gAttnCombine: (MTLCommandBuffer) -> Void = { [self] cb in
            hyperConnection.encodeCombine(commandBuffer: cb,
                                          plane: hcPlane,
                                          blockOut: oOut,
                                          inject: hcInject,
                                          d: D, hc: hc, invHc: invHc)
        }
        let gFfnMix: (MTLCommandBuffer) -> Void = { [self] cb in
            encodeQwen38Mix(commandBuffer: cb,
                            norm: ffnMixW.hcNorm, down: ffnMixW.mixDown,
                            up: ffnMixW.mixUp, blockInject: ffnMixW.blockInject,
                            blockOut: denseX, inject: hcInject,
                            d: D, hc: hc, lowrank: lowrank, invHc: invHc,
                            eps: eps)
        }
        // The MLP combine runs in the deferred tail CB, after h2Buf lands.
        let gMlpCombine: (MTLCommandBuffer) -> Void = { [self] cb in
            hyperConnection.encodeCombine(commandBuffer: cb,
                                          plane: hcPlane,
                                          blockOut: h2Buf,
                                          inject: hcInject,
                                          d: D, hc: hc, invHc: invHc)
        }

        // The PLE n-gram block, on `pleLayerIndex` only (layer 1 — before that
        // layer's attention mixer). Everything it touches is `hcPlane`: the
        // query norm is taken over the plane *as the layer received it*, and
        // both PLE terms are added into it afterwards, so this has to encode
        // before `gAttnMix` normalizes the plane (`qwen4exp.cpp:332-334`).
        let gPLE: ((MTLCommandBuffer) -> Void)?
        if let ple, L == model.pleLayerIndex {
            // The projections are validated against the manifest's
            // `linearAttention` slot at load, so a 3.8 install whose PLE loads
            // at all has them int8 — the same kernel, and the same reason, as
            // the GDN projections (dequant noise here amplifies downstream).
            guard let gemv = int8GEMV else {
                preconditionFailure("Qwen 3.8 PLE requires the int8 GEMV path")
            }
            let keyP   = try model.pleKeyProj()
            let valueP = try model.pleValueProj()
            let normQ  = try model.pleNormQuery()
            let normK  = try model.pleNormKey()
            let normC  = try model.pleNormConv()
            let convW  = try model.pleConv1D()
            let gathered = UInt32(cfg.ngramRowDim * (cfg.ngramSize - 1)
                                    * cfg.headsPerNgram)
            let planeN = UInt32(hcDim)
            // One stream's width: the gate's dot and both norm reductions run
            // over a single 2560-wide stream, not the whole 10240 plane.
            let invSqrtD = 1.0 / Float(D).squareRoot()
            let kern = UInt32(cfg.pleConvKernelSize)
            let dil  = UInt32(cfg.ngramSize)
            gPLE = { [self] cb in
                gemv.encode(commandBuffer: cb,
                            weights: keyP.buffer, weightsOffset: Int(keyP.offset),
                            scales: keyP.buffer, scalesOffset: Int(keyP.scaleOffset),
                            biases: keyP.buffer, biasesOffset: Int(keyP.biasOffset),
                            x: pleGathered, y: pleKey, m: planeN, n: gathered)
                gemv.encode(commandBuffer: cb,
                            weights: valueP.buffer, weightsOffset: Int(valueP.offset),
                            scales: valueP.buffer, scalesOffset: Int(valueP.scaleOffset),
                            biases: valueP.buffer, biasesOffset: Int(valueP.biasOffset),
                            x: pleGathered, y: pleValue, m: D, n: gathered)
                // Normed over one stream under a whole-plane gamma — the same
                // operator, on the same layout, as the HC mixers' norms.
                hyperConnection.encodeGroupedRMS(commandBuffer: cb,
                                                 x: pleKey,
                                                 gamma: normK.buffer,
                                                 gammaOffset: Int(normK.offset),
                                                 out: pleKeyNormed,
                                                 d: D, hc: hc, eps: eps)
                hyperConnection.encodeGroupedRMS(commandBuffer: cb,
                                                 x: hcPlane,
                                                 gamma: normQ.buffer,
                                                 gammaOffset: Int(normQ.offset),
                                                 out: pleQueryNormed,
                                                 d: D, hc: hc, eps: eps)
                ple.encodeGate(commandBuffer: cb,
                               key: pleKeyNormed, query: pleQueryNormed,
                               gate: pleGate,
                               d: D, invSqrtD: invSqrtD, hc: hc)
                ple.encodeGatedValue(commandBuffer: cb,
                                     value: pleValue, gate: pleGate,
                                     gated: pleGated,
                                     d: D, hc: hc)
                hyperConnection.encodeGroupedRMS(commandBuffer: cb,
                                                 x: pleGated,
                                                 gamma: normC.buffer,
                                                 gammaOffset: Int(normC.offset),
                                                 out: pleConvIn,
                                                 d: D, hc: hc, eps: eps)
                // A distinct `newState` buffer rather than an alias: the roll
                // is only alias-safe when every thread reaches it, which the
                // divisible real geometry satisfies but the kernel does not
                // promise. The second buffer costs 180 KB.
                ple.encodeConvUpdate(commandBuffer: cb,
                                     weight: convW.buffer,
                                     weightOffset: Int(convW.offset),
                                     state: pleConvState,
                                     x: pleConvIn,
                                     out: pleConvOut,
                                     newState: pleConvStateNext,
                                     c: planeN, kernel: kern, dilation: dil)
                ple.encodePlaneAdd(commandBuffer: cb,
                                   plane: hcPlane,
                                   gated: pleGated,
                                   conv: pleConvOut,
                                   n: planeN)
            }
        } else {
            gPLE = nil
        }

        if isFull {
            // Full-attention layer: q_proj is doubled (per-head q|gate pairs),
            // q/k per-head norms, partial RoPE (rotary dim = 0.25 * head_dim),
            // attention scale rsqrt(head_dim), then the output gate. (The
            // QSA indexer mask rides on top of this dense body — M3.2.)
            let qP = try model.qProj(layer: L)
            let kP = try model.kProj(layer: L)
            let vP = try model.vProj(layer: L)
            let oP = try model.oProj(layer: L)
            let qN = try model.qNorm(layer: L)
            let kN = try model.kNorm(layer: L)
            let kSlot = kv?.kSlot(layer: L, position: position)
                ?? (buffer: kStage, offset: 0)
            let vSlot = kv?.vSlot(layer: L, position: position)
                ?? (buffer: vStage, offset: 0)
            let headDim = UInt32(cfg.fullHeadDim)
            let numQ = UInt32(cfg.numHeads)
            let numKV = UInt32(cfg.numFullKVHeads)
            let qRows = numQ * headDim
            let rotaryDim = UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor)

            let gProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: qP.buffer, weightsOffset: Int(qP.offset),
                            scales: qP.buffer, scalesOffset: Int(qP.scaleOffset),
                            biases: qP.buffer, biasesOffset: Int(qP.biasOffset),
                            x: normed,
                            y: qGateBuf,
                            m: 2 * qRows, n: D)
                int4.encode(commandBuffer: cb,
                            weights: kP.buffer, weightsOffset: Int(kP.offset),
                            scales: kP.buffer, scalesOffset: Int(kP.scaleOffset),
                            biases: kP.buffer, biasesOffset: Int(kP.biasOffset),
                            x: normed,
                            y: kSlot.buffer, yOffset: kSlot.offset,
                            m: numKV * headDim, n: D)
                int4.encode(commandBuffer: cb,
                            weights: vP.buffer, weightsOffset: Int(vP.offset),
                            scales: vP.buffer, scalesOffset: Int(vP.scaleOffset),
                            biases: vP.buffer, biasesOffset: Int(vP.biasOffset),
                            x: normed,
                            y: vSlot.buffer, yOffset: vSlot.offset,
                            m: numKV * headDim, n: D)
            }
            let gEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                qwenFusions.encodeFullAttnEpilogue(
                    commandBuffer: cb,
                    qProj: qGateBuf,
                    qOut: qScratch,
                    gateOut: gateBuf,
                    k: kSlot.buffer, kOffset: kSlot.offset,
                    qWeight: qN.buffer, qWeightOffset: Int(qN.offset),
                    kWeight: kN.buffer, kWeightOffset: Int(kN.offset),
                    headDim: headDim,
                    numQHeads: numQ,
                    numKVHeads: numKV,
                    position: UInt32(position),
                    theta: Float(cfg.fullRopeTheta),
                    rotaryDim: rotaryDim,
                    eps: eps)
            }
            // MARK: QSA indexer (M3.2d)
            //
            // The sparse-block selector that masks this layer's attention.
            // Nil when the install carries no indexer — every other family,
            // and a Qwen 3.8 repack built without the indexer tensors — in
            // which case attention keeps the dense path it has always had.
            //
            // It keeps its own key timeline rather than reusing the KV cache:
            // `rawKeys` gains this token's key head every step, unnormed and
            // unrotated, and the block timeline gains a pooled key on the step
            // that completes each block. Both are per-layer state, because a
            // layer's indexer chain and its attention share one command buffer
            // while the previous layer's may still be in flight.
            let qsaLayer: (state: QSAIndexerState, index: Int)? = qsaState.flatMap { st in
                st.index(ofLayer: L).map { (state: st, index: $0) }
            }
            // True while every causally-visible cell is selected, i.e. the
            // indexer has nothing left to drop: `ggml_top_k` returns exactly
            // `min(n_kv, budget + r − 1)` cells, which for n_kv below the
            // selection width is n_kv itself — every visible cell, all of them
            // causal. The mask then degenerates to the plain causal one
            // `encodeFull` builds from `seqLen`, so the ranking dispatches are
            // skipped and attention takes its dense path. The timeline above
            // still runs every step, so the pooled blocks are there on the
            // step this stops being true.
            let idxDense = (qsaLayer?.state.capacity ?? Int.max) >= position + 1
            let gIndexer: ((MTLCommandBuffer) -> Void)?
            if let qsaIndexer, let qsaLayer {
                let idxQK = try model.indexerQKProj(layer: L)
                let idxQGamma = try model.indexerQLayernorm(layer: L)
                let idxKGamma = try model.indexerKLayernorm(layer: L)
                let st = qsaLayer.state
                let li = qsaLayer.index
                let lay = st.layers[li]          // buffer references — stable across the step
                let r = st.r
                let pos = UInt32(position)
                let nvis = UInt32(position + 1)
                let idxDim = UInt32(st.idxDim)
                let nHeads = UInt32(st.numQHeads)
                let nRot = UInt32(st.nRot)
                let budget = UInt32(st.budget)
                // Block `position / r` is complete exactly when this cell ends
                // it, which is why pooling is a per-step branch and not a loop.
                let completesPool = (position % r) == (r - 1)
                let poolBlock = UInt32(position / r)

                gIndexer = { [self] cb in
                    int4.encode(commandBuffer: cb,
                                weights: idxQK.buffer, weightsOffset: Int(idxQK.offset),
                                scales: idxQK.buffer, scalesOffset: Int(idxQK.scaleOffset),
                                biases: idxQK.buffer, biasesOffset: Int(idxQK.biasOffset),
                                x: normed,
                                y: lay.qkProj,
                                m: (nHeads + UInt32(st.numKVHeads)) * idxDim, n: D)
                    // Query heads normed and partially roped at this position;
                    // the key head copied verbatim into the timeline.
                    qsaIndexer.encodeQKPost(commandBuffer: cb,
                                            qk: lay.qkProj,
                                            qGamma: idxQGamma.buffer,
                                            qGammaOffset: Int(idxQGamma.offset),
                                            qOut: lay.qIdx,
                                            kRaw: lay.rawKeys,
                                            pos: pos,
                                            nHeads: nHeads,
                                            idxDim: idxDim,
                                            nRot: nRot,
                                            theta: st.theta,
                                            eps: st.eps)
                    if completesPool {
                        qsaIndexer.encodeBlockPoolNormRope(commandBuffer: cb,
                                                           kRaw: lay.rawKeys,
                                                           kGamma: idxKGamma.buffer,
                                                           kGammaOffset: Int(idxKGamma.offset),
                                                           pooled: lay.pooled,
                                                           firstBlock: poolBlock,
                                                           blockCount: 1,
                                                           r: UInt32(r),
                                                           idxDim: idxDim,
                                                           nRot: nRot,
                                                           theta: st.theta,
                                                           eps: st.eps)
                        st.advancePooledBlocks(li)
                    }
                    guard !idxDense else { return }
                    // Score every block of the timeline against this query —
                    // including the incomplete tail, which the bias makes
                    // force-visible — then reduce the scores to the cells the
                    // attention may read.
                    qsaIndexer.encodeBlockScores(commandBuffer: cb,
                                                 q: lay.qIdx,
                                                 pooled: lay.pooled,
                                                 scores: lay.scores,
                                                 nHeads: nHeads,
                                                 idxDim: idxDim,
                                                 r: UInt32(r),
                                                 nKv: nvis,
                                                 pos: pos)
                    qsaIndexer.encodeSelectCells(commandBuffer: cb,
                                                 scores: lay.scores,
                                                 cells: lay.cells,
                                                 count: lay.cellCount,
                                                 pos: pos,
                                                 nKv: nvis,
                                                 r: UInt32(r),
                                                 budget: budget)
                }
            } else {
                gIndexer = nil
            }
            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                if let qsaLayer, !idxDense {
                    // The selection fills its width exactly, so the cell count
                    // is `capacity` here without reading back the kernel's
                    // counter — which would stall the command buffer for a
                    // number the host can already prove.
                    attention.encodeFullCells(commandBuffer: cb,
                                              q: qScratch,
                                              k: kSlot.buffer, kOffset: 0,
                                              v: vSlot.buffer, vOffset: 0,
                                              cells: qsaLayer.state.layers[qsaLayer.index].cells,
                                              out: attnOut,
                                              headDim: headDim,
                                              numQHeads: numQ,
                                              numKVHeads: numKV,
                                              nCells: UInt32(qsaLayer.state.capacity),
                                              scale: nil)   // rsqrt(head_dim)
                } else {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: headDim,
                                         numQHeads: numQ,
                                         numKVHeads: numKV,
                                         seqLen: seqLen,
                                         scale: nil)   // rsqrt(head_dim) — Qwen's scaling
                }
            }
            let gGate: (MTLCommandBuffer) -> Void = { [self] cb in
                qwenFusions.encodeAttnOutputGate(commandBuffer: cb,
                                                 attn: attnOut,
                                                 gate: gateBuf,
                                                 n: qRows)
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: oP.buffer, weightsOffset: Int(oP.offset),
                            scales: oP.buffer, scalesOffset: Int(oP.scaleOffset),
                            biases: oP.buffer, biasesOffset: Int(oP.biasOffset),
                            x: attnOut,
                            y: oOut,
                            m: D, n: qRows)
            }
            gPLE?(cb)
            gAttnMix(cb)
            // Anchor on gProj … gOProj only. `gAttnMix` above is the *second*
            // call in this list, not the last: a lapse placed "after gAttnMix"
            // would land before gProj and silently fold the q/k/v in-projection,
            // the QSA indexer, attention itself and o_proj into `other`.
            totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
            gProj(cb)
            if gIndexer != nil, !idxDense {
                // Ranking — `encodeBlockScores` + `encodeSelectCells` — only
                // runs once the timeline outgrows the QSA state's capacity;
                // the closure returns early below that (the `guard` inside it).
                // On the 3.8 install that is a *step* partway through a long
                // soak: attention changes character mid-run, and without this
                // count the step has no explanation.
                totalIndexerRankedLayers &+= 1
            }
            gIndexer?(cb)
            gEpilogue(cb)
            gAttention(cb)
            gGate(cb)
            gOProj(cb)
            totalCb1AttentionNanos &+= lapseCb1(&cb1Cursor)
            gAttnCombine(cb)
            gFfnMix(cb)
        } else {
            // GDN (linear-attention) layer: identical to the Qwen 3.6 body
            // except the output norm's z-gate — sigmoid here, silu there
            // (qwen4exp `build_norm_gated` :411-421).
            let si = gdnStateIndexByLayer[L]
            precondition(si >= 0, "GDN layer \(L) without state")
            let qkvP = try model.gdnInProjQKV(layer: L)
            let zP = try model.gdnInProjZ(layer: L)
            let outP = try model.gdnOutProj(layer: L)
            let convW = try model.gdnConv1D(layer: L)
            let aLog = try model.gdnALog(layer: L)
            let dt = try model.gdnDtBias(layer: L)
            let normW = try model.gdnNormWeight(layer: L)
            let recState = gdnRecurrentState[si]
            let convState = gdnConvState[si]
            let aP = linearAttnBits == 8 ? try model.gdnInProjA(layer: L) : nil
            let bP = linearAttnBits == 8 ? try model.gdnInProjB(layer: L) : nil
            let keyDim = UInt32(cfg.linearNumKeyHeads * cfg.linearKeyHeadDim)
            let valueDim = UInt32(cfg.linearNumValueHeads * cfg.linearValueHeadDim)
            let qkvDim = 2 * keyDim + valueDim
            let numV = cfg.linearNumValueHeads
            let headDim = UInt32(cfg.linearValueHeadDim)
            let scale = 1.0 / Float(cfg.linearKeyHeadDim).squareRoot()
            let betaByteOffset = numV * MemoryLayout<Float>.size

            let gProj: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: qkvP.buffer, weightsOffset: Int(qkvP.offset),
                                     scales: qkvP.buffer, scalesOffset: Int(qkvP.scaleOffset),
                                     biases: qkvP.buffer, biasesOffset: Int(qkvP.biasOffset),
                                     x: normed,
                                     y: qkvConv,
                                     m: qkvDim, n: D)
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: zP.buffer, weightsOffset: Int(zP.offset),
                                     scales: zP.buffer, scalesOffset: Int(zP.scaleOffset),
                                     biases: zP.buffer, biasesOffset: Int(zP.biasOffset),
                                     x: normed,
                                     y: zBuf,
                                     m: valueDim, n: D)
                } else {
                    int4.encode(commandBuffer: cb,
                                weights: qkvP.buffer, weightsOffset: Int(qkvP.offset),
                                scales: qkvP.buffer, scalesOffset: Int(qkvP.scaleOffset),
                                biases: qkvP.buffer, biasesOffset: Int(qkvP.biasOffset),
                                x: normed,
                                y: qkvConv,
                                m: qkvDim, n: D)
                    int4.encode(commandBuffer: cb,
                                weights: zP.buffer, weightsOffset: Int(zP.offset),
                                scales: zP.buffer, scalesOffset: Int(zP.scaleOffset),
                                biases: zP.buffer, biasesOffset: Int(zP.biasOffset),
                                x: normed,
                                y: zBuf,
                                m: valueDim, n: D)
                }
            }
            let gConv: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeCausalConvUpdate(commandBuffer: cb,
                                           w: convW.buffer, wOffset: Int(convW.offset),
                                           state: convState,
                                           x: qkvConv,
                                           out: qkvConv,
                                           newState: convState,
                                           channels: Int(qkvDim))
            }
            let gGateGEMV: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    // Two int8 GEMVs (in_proj_a then in_proj_b) into the fp16
                    // [2V] ab scratch, then the batched gate formula at T=1 —
                    // g at offset 0, beta at +V floats, the same layout and
                    // math as the 4-bit fused kernel.
                    guard let gemv = int8GEMV, let ab = gateAB,
                          let aView = aP, let bView = bP else { return }
                    gemv.encode(commandBuffer: cb,
                                weights: aView.buffer, weightsOffset: Int(aView.offset),
                                scales: aView.buffer, scalesOffset: Int(aView.scaleOffset),
                                biases: aView.buffer, biasesOffset: Int(aView.biasOffset),
                                x: normed,
                                y: ab,
                                m: UInt32(numV), n: D)
                    gemv.encode(commandBuffer: cb,
                                weights: bView.buffer, weightsOffset: Int(bView.offset),
                                scales: bView.buffer, scalesOffset: Int(bView.scaleOffset),
                                biases: bView.buffer, biasesOffset: Int(bView.biasOffset),
                                x: normed,
                                y: ab,
                                yOffset: numV * MemoryLayout<Float16>.size,
                                m: UInt32(numV), n: D)
                    gdnPrefill.encodeGateBatch(commandBuffer: cb,
                                               ab: ab,
                                               A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                               dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                               g: gBeta,
                                               beta: gBeta, betaOffset: betaByteOffset,
                                               numValueHeads: numV,
                                               tokens: 1)
                } else {
                    let gateW = gdnGateWeights[si]
                    gdn.encodeGateGEMV(commandBuffer: cb,
                                       weights: gateW.weights,
                                       scales: gateW.scales,
                                       biases: gateW.biases,
                                       x: normed,
                                       A_log: aLog.buffer, A_logOffset: Int(aLog.offset),
                                       dt_bias: dt.buffer, dt_biasOffset: Int(dt.offset),
                                       g: gBeta,
                                       beta: gBeta, betaOffset: betaByteOffset,
                                       numValueHeads: numV,
                                       n: D)
                }
            }
            let gRecurrent: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeRecurrent(commandBuffer: cb,
                                    state: recState,
                                    q: qkvConv,
                                    k: qkvConv, kOffset: Int(keyDim) * 2,
                                    v: qkvConv, vOffset: Int(keyDim) * 4,
                                    g: gBeta,
                                    beta: gBeta, betaOffset: betaByteOffset,
                                    out: attnOut,
                                    numValueHeads: numV,
                                    numKeyHeads: cfg.linearNumKeyHeads,
                                    headDim: headDim,
                                    scale: scale,
                                    l2eps: 1e-6)
            }
            let gNormGated: (MTLCommandBuffer) -> Void = { [self] cb in
                gdn.encodeRMSNormGated(commandBuffer: cb,
                                       x: attnOut,
                                       z: zBuf,
                                       weight: normW.buffer, weightOffset: Int(normW.offset),
                                       out: attnOut,
                                       numValueHeads: numV,
                                       headDim: headDim,
                                       eps: eps,
                                       activation: .sigmoid)
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                if linearAttnBits == 8 {
                    int8GEMV!.encode(commandBuffer: cb,
                                     weights: outP.buffer, weightsOffset: Int(outP.offset),
                                     scales: outP.buffer, scalesOffset: Int(outP.scaleOffset),
                                     biases: outP.buffer, biasesOffset: Int(outP.biasOffset),
                                     x: attnOut,
                                     y: oOut,
                                     m: D, n: valueDim)
                } else {
                    int4.encode(commandBuffer: cb,
                                weights: outP.buffer, weightsOffset: Int(outP.offset),
                                scales: outP.buffer, scalesOffset: Int(outP.scaleOffset),
                                biases: outP.buffer, biasesOffset: Int(outP.biasOffset),
                                x: attnOut,
                                y: oOut,
                                m: D, n: valueDim)
                }
            }
            gPLE?(cb)
            gAttnMix(cb)
            totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
            gProj(cb)
            totalCb1GdnProjNanos &+= lapseCb1(&cb1Cursor)
            gConv(cb)
            gGateGEMV(cb)
            totalCb1GdnConvGateNanos &+= lapseCb1(&cb1Cursor)
            gRecurrent(cb)
            gNormGated(cb)
            gOProj(cb)
            totalCb1GdnRecurrentNanos &+= lapseCb1(&cb1Cursor)
            gAttnCombine(cb)
            gFfnMix(cb)
        }

        // Router (both layer types): plain softmax over all experts, top-8
        // renormalized — mathematically identical to the kernel's top-8
        // softmax, so the Gemma kernel is reused with ones-filled scales.
        // The router is the last work encoded into this layer's cb1, so its
        // span runs through the commit call inclusive.
        totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
        moe.encodeRouterGemma4(commandBuffer: cb,
                               weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                               scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                               biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                               hidden: denseX,
                               effectiveScale: onesEffective,
                               perExpertScale: onesExpert,
                               outIndices: outIndices, outWeights: outWeights,
                               numExperts: UInt32(cfg.numExperts), d: D,
                               topK: UInt32(cfg.topKExperts))
        commitCounting(cb)
        totalCb1RouterNanos &+= lapseCb1(&cb1Cursor)
        let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        waitForCompletion(cb)
        let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
        // Read once: `recordGpuTime` counts a sample only when it returns a
        // real number, so a second call would double `totalGpuSamples`.
        let cb1GpuNanos = recordGpuTime(cb)
        totalGpuCb1Nanos &+= cb1GpuNanos
        if isFull {
            totalGpuCb1FullAttnNanos &+= cb1GpuNanos
        } else {
            totalGpuCb1GdnNanos &+= cb1GpuNanos
        }
        // cb1 subtracts this wait, so the buckets must *skip* exactly this span
        // rather than attribute it. Attributing it would put the tile sum
        // `wait` above cb1 — and that sum still looks like a plausible number,
        // which is why this has to be structural rather than remembered.
        cb1Cursor &+= waitNanos
        if let previous = pending {
            // The debug hook snapshots read shared buffers on the CPU — wait
            // out the deferred tail so the snapshots are race-free. Without a
            // hook, the tail completes in CB order before the next layer.
            finishPendingRoutedCommand(previous,
                                       waitIfNeeded: qwenLayerDebugHook != nil)
            pending = nil
        }
        // Final lapse, then cb1 from that same reading, so
        // `other + attention + gdn + router == cb1` holds by construction
        // rather than to within the gap between two separate clock reads.
        totalCb1OtherNanos &+= lapseCb1(&cb1Cursor)
        totalCb1Nanos &+= cb1Cursor &- tCb1Start &- waitNanos
        totalCb1WaitNanos &+= waitNanos

        if let hook = qwenLayerDebugHook {
            func snap(_ name: String, _ buf: MTLBuffer, _ count: Int) {
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: count)
                hook(L, name, Array(UnsafeBufferPointer(start: ptr, count: count)))
            }
            snap("attnBlockIn", normed, cfg.hiddenSize)
            snap("attnBlockOut", oOut, cfg.hiddenSize)
            // hc.mid: the plane after the attn combine, before the ffn mix.
            snap("hc.mid", hcPlane, hcDim)
            snap("ffnBlockIn", denseX, cfg.hiddenSize)
            snap("qkvConv", qkvConv, 2 * Int(cfg.linearNumKeyHeads * cfg.linearKeyHeadDim)
                + cfg.linearNumValueHeads * cfg.linearValueHeadDim)
            snap("recurrentOut", attnOut,
                 cfg.linearNumValueHeads * cfg.linearValueHeadDim)
            do {
                let n = 2 * cfg.linearNumValueHeads
                let ptr = gBeta.contents().bindMemory(to: Float.self, capacity: n)
                hook(L, "gFloat", Array(UnsafeBufferPointer(start: ptr, count: n))
                    .map { Float16($0) })
            }
            if isFull, let kv {
                let n = cfg.numHeads * cfg.fullHeadDim
                let qPtr = qScratch.contents().bindMemory(to: Float16.self, capacity: n)
                hook(L, "qOutF", Array(UnsafeBufferPointer(start: qPtr, count: n)))
                let gPtr = gateBuf.contents().bindMemory(to: Float16.self, capacity: n)
                hook(L, "gateF", Array(UnsafeBufferPointer(start: gPtr, count: n)))
                let slot = kv.kSlot(layer: L, position: position)
                let kn = cfg.numFullKVHeads * cfg.fullHeadDim
                let kPtr = slot.buffer.contents().advanced(by: slot.offset)
                    .bindMemory(to: Float16.self, capacity: kn)
                hook(L, "kF", Array(UnsafeBufferPointer(start: kPtr, count: kn)))
                let vSlot = kv.vSlot(layer: L, position: position)
                let vPtr = vSlot.buffer.contents().advanced(by: vSlot.offset)
                    .bindMemory(to: Float16.self, capacity: kn)
                hook(L, "vF", Array(UnsafeBufferPointer(start: vPtr, count: kn)))
            }
            let si = gdnStateIndexByLayer[L]
            if si >= 0 {
                let n = cfg.linearNumValueHeads * cfg.linearValueHeadDim
                    * cfg.linearValueHeadDim
                let ptr = gdnRecurrentState[si].contents()
                    .bindMemory(to: Float.self, capacity: n)
                hook(L, "recState", Array(UnsafeBufferPointer(start: ptr, count: n))
                    .map { Float16($0) })
            }
        }

        // Shared expert post stage: sigmoid(shared_expert_gate · x) scales h1.
        let sharedPost: (MTLCommandBuffer) -> Void = { [self] cb in
            qwenFusions.encodeSharedGate(commandBuffer: cb,
                                         weights: sharedGate.buffer,
                                         weightsOffset: Int(sharedGate.offset),
                                         scales: sharedGate.buffer,
                                         scalesOffset: Int(sharedGate.scaleOffset),
                                         biases: sharedGate.buffer,
                                         biasesOffset: Int(sharedGate.biasOffset),
                                         x: denseX,
                                         h1: h1Buf,
                                         n: D, d: D)
        }
        // Qwen 3.8 tail: phase-2 writes h2 = shared + routed (residual =
        // h1Buf); the layer closes with the MLP combine scattering h2 into the
        // wide plane. No hidden write, no layer_scalar, no sandwich.
        let tail = gMlpCombine
        try await encodeRoutedTail(
            layer: L,
            position: position,
            routedX: denseX,
            denseX: denseX,
            sharedProj: sharedProj,
            activation: .silu,
            routedResidual: h1Buf,
            sharedPostEncoder: sharedPost,
            tailEncoder: tail,
            pending: &pending)

        if let hook = qwenLayerDebugHook {
            // The just-committed tail (plane += h2·w) is in flight — wait it
            // out so the hc.post snapshot is race-free.
            if let p = pending {
                finishPendingRoutedCommand(p, waitIfNeeded: true)
                pending = nil
            }
            let ptr = hcPlane.contents().bindMemory(to: Float16.self,
                                                    capacity: hcDim)
            hook(L, "hc.post", Array(UnsafeBufferPointer(start: ptr,
                                                         count: hcDim)))
            // The MLP combine's block input (shared + routed, residual-added):
            // splits the MoE block from the combine that scatters it.
            let h2Ptr = h2Buf.contents().bindMemory(to: Float16.self,
                                                    capacity: cfg.hiddenSize)
            hook(L, "mlpBlockIn", Array(UnsafeBufferPointer(start: h2Ptr,
                                                            count: cfg.hiddenSize)))
            // The shared-expert term alone: h2 = shared + routed, so a
            // matching shared leaves the routed path as the difference.
            let h1Ptr = h1Buf.contents().bindMemory(to: Float16.self,
                                                    capacity: cfg.hiddenSize)
            hook(L, "sharedOut", Array(UnsafeBufferPointer(start: h1Ptr,
                                                           count: cfg.hiddenSize)))
        }
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        totalForwards &+= 1
        // Qwen hybrid families do not scale embeddings (see prefillChunked).
        let sqrtHidden = cfg.isQwenHybrid ? 1.0 : Float(cfg.hiddenSize).squareRoot()
        var pendingRoutedCommand: PendingRoutedCommand?

        // Embed lookup + sqrt(H) fused.
        let emb = model.embedding
        do {
            runSync { cb in
                embedInt4.encode(commandBuffer: cb,
                                 table:  emb.buffer, tableOffset:  Int(emb.offset),
                                 scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                                 biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                                 out: hidden,
                                 tokenId: UInt32(bitPattern: token),
                                 d: D,
                                 outScale: sqrtHidden)
                if cfg.isQwen3_8 {
                    // Qwen 3.8: the decode-step residual is the wide
                    // hyper-connection plane, seeded with hc copies of the
                    // embedding (llama `hc_init`, qwen4exp.cpp :329-331).
                    // Same CB — no extra wait; the first layer's attn mix
                    // reads the plane from the next command buffer.
                    hyperConnection.encodePlaneInit(commandBuffer: cb,
                                                    hidden: hidden,
                                                    plane: hcPlane,
                                                    d: D,
                                                    hc: UInt32(cfg.hyperConnectionCount))
                }
            }
        }

        // PLE routing for this token, before any layer runs: the rows are a
        // hash of the token being produced and its two predecessors, and the
        // table rows are read straight off disk (16 × 320 B = 5 KB — nothing
        // is cached, because the reads are hash-random and the table is
        // 102.4 GB). Record first, exactly as llama's `set_input` runs after
        // `apply_ubatch` has already stored the ubatch.
        if let pleHost {
            pleHost.record(position: position, token: token)
            // Timed as a wall clock, unlike the cb1 buckets: this is a
            // synchronous host-side read that blocks the decode, not an
            // encode-and-commit. Under `full-sha256` the first touch of each
            // shard also hashes 800 MB *here*, inside a decode step, so a cold
            // run and a warm one do not measure the same thing.
            let tPle = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // The host's byte total is cumulative over prefill *and* decode, so
            // take the delta rather than assigning it: this counter is reported
            // beside `pleGathers`, which is decode-only, and a per-token figure
            // built from a whole-run numerator and a decode denominator is
            // meaningless (it was 3.7x too high before this).
            let pleBytesBefore = pleHost.totalRowBytesRead
            let gathered = try pleHost.gather(atPosition: position) { part in
                totalPlePartOpens &+= 1
                return try model.openPLEPart(part)
            }
            totalPleRowBytes &+= pleHost.totalRowBytesRead &- pleBytesBefore
            totalPleGatherNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) &- tPle
            totalPleGathers &+= 1
            gathered.withUnsafeBytes { src in
                memcpy(pleGathered.contents(), src.baseAddress!, src.count)
            }
        }

        for L in 0..<cfg.numLayers {
            if cfg.isQwenHybrid {
                if cfg.isQwen3_6 {
                    try await encodeQwenDecodeLayer(L, position: position,
                                                    pending: &pendingRoutedCommand)
                } else {
                    try await encodeQwen38DecodeLayer(L, position: position,
                                                      pending: &pendingRoutedCommand)
                }
                continue
            }
            let isFull = cfg.fullAttentionLayerMask[L] != 0
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let kSlot    = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
            let vSlot    = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let q        = try model.qProj(layer: L)
            let k        = try model.kProj(layer: L)
            // v_proj only exists on SWA layers; full layers reuse k_proj.
            let vProj    = isFull ? k : (try model.vProj(layer: L))
            let o        = try model.oProj(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let qNorm    = try model.qNorm(layer: L)
            let kNorm    = try model.kNorm(layer: L)
            let preFFN   = try model.preFFN(layer: L)
            let preFFN2  = try model.preFFN2(layer: L)
            let sharedProj = sharedExpertProjections[L]
            let postF2   = try model.postFFN2(layer: L)
            let postF    = try model.postFFN(layer: L)
            let routerW  = try model.router(layer: L)
            let perExpertScale = try model.routerPerExpertScale(layer: L)
            let layerScalarView = try model.layerScalar(layer: L)

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let gInputNorm: (MTLCommandBuffer) -> Void = { [self] cb in
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                                out: normed,
                                d: D, eps: eps)
            }

            let gQKV: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)
            }

            let gQKVEpilogue: (MTLCommandBuffer) -> Void = { [self] cb in
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
            }

            let gAttention: (MTLCommandBuffer) -> Void = { [self] cb in
                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: 1.0)
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: 1.0,
                                        ringCapacity: activeRingCapacity)
                }
            }
            let gOProj: (MTLCommandBuffer) -> Void = { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            let gPostAttnSetup: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            }

            let gRouter: (MTLCommandBuffer) -> Void = { [self] cb in
                moe.encodeRouterGemma4(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                    biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                    hidden: routerInput,
                    effectiveScale: effectiveScaleBuffers[L],
                    perExpertScale: perExpertScale.buffer,
                    perExpertScaleOffset: Int(perExpertScale.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }

            let cb = ctx.queue.makeCommandBuffer()!
            gInputNorm(cb)
            gQKV(cb)
            gQKVEpilogue(cb)
            gAttention(cb)
            gOProj(cb)
            gPostAttnSetup(cb)
            gRouter(cb)
            commitCounting(cb)
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitForCompletion(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            totalGpuCb1Nanos &+= recordGpuTime(cb)
            if let pending = pendingRoutedCommand {
                finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            let scalarPtr = layerScalarView.buffer.contents()
                .advanced(by: Int(layerScalarView.offset))
                .assumingMemoryBound(to: UInt16.self)
            let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])

            // Gemma shared-expert post stage: post_feedforward_layernorm_1 on
            // h1. (Qwen replaces this with the sigmoid shared-expert gate.)
            let sharedPost: (MTLCommandBuffer) -> Void = { [self] cb in
                let postF1 = sharedProj.postF1!
                rms.encodeBF16W(commandBuffer: cb, x: h1Buf,
                                weight: postF1.buffer,
                                weightOffset: Int(postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            }
            let tail: (MTLCommandBuffer) -> Void = { [self] cb in
                fusedTail.encode(commandBuffer: cb,
                                 h2: h2Buf,
                                 h1: h1Buf,
                                 hidden: hidden,
                                 postFFN2Weight: postF2.buffer,
                                 postFFN2WeightOffset: Int(postF2.offset),
                                 postFFNWeight: postF.buffer,
                                 postFFNWeightOffset: Int(postF.offset),
                                 d: D,
                                 eps: eps,
                                 layerScalar: layerScalar)
            }
            try await encodeRoutedTail(
                layer: L,
                position: position,
                routedX: routedX,
                denseX: denseX,
                sharedProj: sharedProj,
                activation: .gelu,
                routedResidual: zeroResidual,
                sharedPostEncoder: sharedPost,
                tailEncoder: tail,
                pending: &pendingRoutedCommand)
            continue
        }
        if let pending = pendingRoutedCommand {
            finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        //
        // Qwen 3.8: there is no model.norm. The terminal is the root
        // hyper_connection_mixer — the same mix stages as a layer mixer with
        // no block_inject — collapsing the wide plane into the [D] lm_head
        // input (llama `result_norm` callback, qwen4exp.cpp :380-390).
        let rootMixer = cfg.isQwen3_8 ? try model.hyperConnectionMixer() : nil
        let fNorm     = model.finalNorm     // nil on Qwen 3.8 by design
        guard cfg.isQwen3_8 || fNorm != nil else {
            throw ModelError.tensorNotFound(name: "language_model.model.norm.weight")
        }
        let lm    = model.lmHead   // untied lm_head for Qwen 3.6; tied for Gemma 4
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            if let root = rootMixer {
                self.encodeQwen38Mix(commandBuffer: cb,
                                     norm: root.hcNorm,
                                     down: root.mixDown,
                                     up: root.mixUp,
                                     blockInject: nil,
                                     blockOut: self.normed,
                                     inject: nil,
                                     d: D,
                                     hc: UInt32(self.cfg.hyperConnectionCount),
                                     lowrank: UInt32(self.cfg.hyperConnectionLowrank),
                                     invHc: 1.0 / Float(self.cfg.hyperConnectionCount),
                                     eps: eps)
            } else if let f = fNorm {
                self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                     weight: f.buffer, weightOffset: Int(f.offset),
                                     out: self.normed, d: D, eps: eps)
            }
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            // Only reachable with a fused head — i.e. never on Qwen 3.8 (the
            // gate above excludes it), where fNorm is nil by design.
            guard let f = fNorm else { return }
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: f.buffer, normOffset: Int(f.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        if emitHead {
            // `useFusedGreedyHead` already excludes Qwen 3.8; see its doc.
            let useFusedHeadForThisToken = useFusedGreedyHead
                && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            if useFusedHeadForThisToken {
                runSync(gFusionHead)
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                runSync { cb in
                    gFinalNorm(cb)
                    gLmHead(cb)
                }
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        }

        dumpQSASelection(position: position, phase: "decode")
        kv?.advance()
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        // Every `runSync` call site is a decode site (embed, and the two head
        // variants), so this counts toward the decode total.
        commitCounting(cb)
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
        if let err = cb.error {
            print("CB error: \(err)")
        }
    }

    /// Drain the queue: an empty command buffer committed after everything
    /// else completes only once all of it has, because a `MTLCommandQueue`
    /// starts its buffers in submission order. Cheaper and more local than
    /// threading a command buffer out of `prefillChunked` for the caller to
    /// wait on, and it cannot be skipped by a caller that forgets.
    public func drainGPU() {
        guard let cb = ctx.queue.makeCommandBuffer() else { return }
        commitCounting(cb)
        waitForCompletion(cb)
    }

}

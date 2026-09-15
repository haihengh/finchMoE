import Metal

struct PrefillChunkScratchLayout: Sendable, Equatable {
    let chunkTokens: Int
    let hiddenSize: Int
    let maxQElementsPerToken: Int
    let maxKVElementsPerToken: Int
    let sharedIntermediate: Int
    let routedIntermediate: Int
    let topK: Int
    let routedPairMicrobatchRows: Int
    // Qwen 3.6 prefill extras (sized 1 for Gemma installs).
    /// Full-attention output-gate scratch: T * numHeads * fullHeadDim fp16.
    let qwenQGateElements: Int
    /// GDN in_proj_qkv output / conv input: T * qkvDim fp16. The batched conv
    /// reads raw inputs from this buffer while writing its output elsewhere,
    /// so the conv output has its own buffer.
    let qwenQKVProjElements: Int
    /// GDN conv output (= recurrent q|k|v block): T * qkvDim fp16.
    let qwenQKVConvOutElements: Int
    /// GDN in_proj_z output: T * valueDim fp16 (= [T][V][headDim]).
    let qwenZElements: Int
    /// GDN in_proj_a|b QMM output: T * 2V fp16 (a rows then b rows).
    let qwenABElements: Int
    /// GDN gate g|beta: T * 2V fp32 (g rows then beta rows).
    let qwenGBetaElements: Int
    /// GDN recurrent output / gated-rmsnorm in-out: T * valueDim fp16.
    let qwenRecOutElements: Int
    /// GDN post-chunk conv state scratch: qkvDim * 3 fp16 (blit-carried into
    /// the persistent state).
    let qwenConvNewStateElements: Int
    // Qwen 3.8 Flash-Next prefill extras (sized 1 for other installs).
    /// The hyper-connection residual plane for the whole chunk: T * hc·D fp16.
    /// This is what `hidden` is to the other families — the layer residual —
    /// and it is per-token wide (`hc` copies of a [D] token), so it is the one
    /// buffer with no 3.6 analogue.
    let qwen38PlaneElements: Int
    /// Mixer lowrank stage: T * hc_lowrank fp16. The other three mixer stages
    /// (xn, the raw up dot, and the gated plane) are all plane-wide and reuse
    /// `qwen38PlaneElements` buffers.
    let qwen38LowrankElements: Int
    /// Per-token scatter weights from `block_inject`: T * hc fp16.
    let qwen38InjectElements: Int
    /// Attention output projection: T * D fp16. The attention bodies are
    /// shared with 3.6, so this is the only [T][D] buffer 3.8 adds to the
    /// attention path — and it has to outlive the attention block, because the
    /// combine that scatters it into the plane runs after the ffn mix's read
    /// of the same plane.
    let qwen38OOutElements: Int
    /// Gathered n-gram rows: T * (ngramRowDim · (ngramSize−1) · headsPerNgram)
    /// fp16. Each token's rows come from a host-side hash + pread.
    let qwen38PleGatheredElements: Int
    /// One token's worth of that gather — the row stride the host memcpy walks
    /// with. Carried explicitly because the buffer is sized on the config's
    /// *maximum* chunk, so dividing it by the chunk actually being run gives
    /// the wrong stride for every chunk shorter than the maximum.
    let qwen38PleGatheredRowElements: Int
    /// PLE value projection output: T * D fp16.
    let qwen38PleValueElements: Int
    /// PLE per-(token, stream) gate: T * hc FP32 (the decode gate is FP32 too).
    let qwen38PleGateElements: Int
    /// PLE conv history scratch: (kernel−1)·ngramSize · hc·D fp16, blit-carried
    /// into the persistent per-runner state exactly as the decode path's roll
    /// destination is.
    let qwen38PleConvStateElements: Int
    /// QSA prefill cell lists: T * capacity UInt32 — one selected-cell list per
    /// chunk row, the same shape `QSAIndexerState` holds for a single step.
    let qwen38CellsElements: Int
    /// Indexer `index_qk_proj` output for the chunk: T · (nHeads + kvHeads) ·
    /// idxDim fp16. Only the key rows are persistent (they are copied into the
    /// layer's raw timeline); the query rows are consumed by the same row's
    /// `encodeQKPost`.
    let qwen38IdxQKProjElements: Int
    /// Indexer post-processed queries: T · nHeads · idxDim fp16.
    let qwen38IdxQElements: Int
    /// One row's worth of biased block scores, `maxBlocks` FP32. A single row
    /// rather than `T` of them: each row's score dispatch is followed by that
    /// row's select in the same command buffer, so the score array is dead the
    /// moment the select has read it. `maxBlocks` is the whole timeline
    /// (`ceil(maxContext / r)`) because every complete block is scored on every
    /// row and the selection is what discards them.
    let qwen38IdxScoreElements: Int

    /// `maxContext` is the runner's own context budget, not the config's — the
    /// only consumer is the QSA cell-list sizing, which is capped by it.
    init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32,
                maxContext: Int = Int.max) {
        self.chunkTokens = max(1, min(chunkTokens, PrefillRuntimeConfig.maxChunkTokens))
        self.hiddenSize = config.hiddenSize
        // Doubled q_proj (per-head q|gate pairs) + GDN recurrent scratch are
        // shared by both Qwen hybrid families; Gemma leaves them empty.
        let isQwen = config.isQwenHybrid
        // Qwen's q_proj is doubled (per-head q|gate pairs), so the q scratch
        // holds 2 * numHeads * fullHeadDim per token.
        self.maxQElementsPerToken = isQwen
            ? config.numHeads * 2 * config.fullHeadDim
            : config.numHeads * max(config.headDim, config.fullHeadDim)
        self.maxKVElementsPerToken = max(config.numKVHeads * config.headDim,
                                         config.numFullKVHeads * config.fullHeadDim)
        self.sharedIntermediate = config.intermediateSize
        self.routedIntermediate = config.moeIntermediateSize
        self.topK = config.topKExperts
        self.routedPairMicrobatchRows = max(1, min(routedPairMicrobatchRows, 128))
        if isQwen {
            let qkvDim = config.linearNumKeyHeads * config.linearKeyHeadDim * 2
                + config.linearNumValueHeads * config.linearValueHeadDim
            let valueDim = config.linearNumValueHeads * config.linearValueHeadDim
            self.qwenQGateElements = self.chunkTokens * config.numHeads * config.fullHeadDim
            self.qwenQKVProjElements = self.chunkTokens * qkvDim
            self.qwenQKVConvOutElements = self.chunkTokens * qkvDim
            self.qwenZElements = self.chunkTokens * valueDim
            self.qwenABElements = self.chunkTokens * 2 * config.linearNumValueHeads
            self.qwenGBetaElements = self.chunkTokens * 2 * config.linearNumValueHeads
            self.qwenRecOutElements = self.chunkTokens * valueDim
            self.qwenConvNewStateElements = qkvDim * 3
        } else {
            self.qwenQGateElements = 1
            self.qwenQKVProjElements = 1
            self.qwenQKVConvOutElements = 1
            self.qwenZElements = 1
            self.qwenABElements = 1
            self.qwenGBetaElements = 1
            self.qwenRecOutElements = 1
            self.qwenConvNewStateElements = 1
        }
        if config.isQwen3_8 {
            let hcDim = config.hyperConnectionDim
            // The indexer's selection width, the same expression
            // `QSAIndexerState.capacity` uses: `min(maxContext, budget + r − 1)`.
            // A 3.8 install whose indexer tensors are missing still allocates
            // this (the indexer is optional; the plane and PLE are not).
            let capacity = min(maxContext,
                               config.indexerBudget + config.indexerCompressRatio - 1)
            let gathered = config.ngramRowDim * max(config.ngramSize - 1, 0)
                * config.headsPerNgram
            self.qwen38PlaneElements = self.chunkTokens * hcDim
            self.qwen38LowrankElements = self.chunkTokens * config.hyperConnectionLowrank
            self.qwen38InjectElements = self.chunkTokens * config.hyperConnectionCount
            self.qwen38OOutElements = self.chunkTokens * config.hiddenSize
            self.qwen38PleGatheredRowElements = gathered
            self.qwen38PleGatheredElements = self.chunkTokens * gathered
            self.qwen38PleValueElements = self.chunkTokens * config.hiddenSize
            self.qwen38PleGateElements = self.chunkTokens * config.hyperConnectionCount
            self.qwen38PleConvStateElements = max(config.pleConvKernelSize - 1, 0)
                * config.ngramSize * hcDim
            self.qwen38CellsElements = self.chunkTokens * max(capacity, 1)
            // The indexer's own scratch is sized off `maxContext` rather than
            // `capacity` — `maxBlocks` is the whole block timeline, and a 3.8
            // install with no indexer tensors still allocates it (the indexer
            // is optional; the plane and PLE are not).
            let idxDim = config.indexerHeadDim
            self.qwen38IdxQKProjElements = self.chunkTokens
                * (config.indexerNumHeads + config.indexerKVHeads) * idxDim
            self.qwen38IdxQElements = self.chunkTokens * config.indexerNumHeads * idxDim
            self.qwen38IdxScoreElements = max((maxContext + config.indexerCompressRatio - 1)
                                                / max(config.indexerCompressRatio, 1), 1)
        } else {
            self.qwen38PlaneElements = 1
            self.qwen38LowrankElements = 1
            self.qwen38InjectElements = 1
            self.qwen38OOutElements = 1
            self.qwen38PleGatheredElements = 1
            self.qwen38PleGatheredRowElements = 1
            self.qwen38PleValueElements = 1
            self.qwen38PleGateElements = 1
            self.qwen38PleConvStateElements = 1
            self.qwen38CellsElements = 1
            self.qwen38IdxQKProjElements = 1
            self.qwen38IdxQElements = 1
            self.qwen38IdxScoreElements = 1
        }
    }

    init(config: ArchConfig, runtime: PrefillRuntimeConfig, maxContext: Int = Int.max) {
        self.init(config: config,
                  chunkTokens: runtime.chunkTokens,
                  maxContext: maxContext)
    }

    var hiddenElements: Int { chunkTokens * hiddenSize }
    var normedElements: Int { hiddenElements }
    var qElements: Int { chunkTokens * maxQElementsPerToken }
    var kStageElements: Int { chunkTokens * maxKVElementsPerToken }
    var vStageElements: Int { kStageElements }
    var attentionOutputElements: Int { qElements }
    var denseXElements: Int { hiddenElements }
    var routedXElements: Int { hiddenElements }
    var routerXElements: Int { hiddenElements }
    var h1Elements: Int { hiddenElements }
    var h2Elements: Int { hiddenElements }
    var routePartialElements: Int { chunkTokens * topK * hiddenSize }
    var routeIDElements: Int { chunkTokens * topK }
    var routeWeightElements: Int { routeIDElements }
    var sharedExpertScratchElements: Int { sharedIntermediate }
    var routedGateUpActElements: Int { 3 * routedPairMicrobatchRows * routedIntermediate }
    var routedDownOutputElements: Int { routedPairMicrobatchRows * hiddenSize }
    // The 3.8 mixer's four plane-wide stages and the three PLE plane-wide
    // stages are all one `[T][hc·D]` buffer per stage, so they alias the plane
    // count rather than carrying eight more `1`-valued stored twins for other
    // installs. Each is still a *distinct* buffer (see `PrefillChunkScratchBuffers`).
    var qwen38XnElements: Int { qwen38PlaneElements }
    var qwen38GateRawElements: Int { qwen38PlaneElements }
    var qwen38GatedElements: Int { qwen38PlaneElements }
    var qwen38PleKeyElements: Int { qwen38PlaneElements }
    var qwen38PleKeyNormedElements: Int { qwen38PlaneElements }
    var qwen38PleQueryNormedElements: Int { qwen38PlaneElements }
    var qwen38PleGatedElements: Int { qwen38PlaneElements }
    var qwen38PleConvInElements: Int { qwen38PlaneElements }
    var qwen38PleConvOutElements: Int { qwen38PlaneElements }

    var devicePrivateBytes: Int {
        let fp16Elements = hiddenElements
            + normedElements
            + qElements
            + kStageElements
            + vStageElements
            + attentionOutputElements
            + denseXElements
            + routedXElements
            + routerXElements
            + h1Elements
            + h2Elements
            + routePartialElements
            + 3 * sharedExpertScratchElements
            + routedGateUpActElements
            + routedDownOutputElements
            + qwenQGateElements
            + qwenQKVProjElements
            + qwenQKVConvOutElements
            + qwenZElements
            + qwenABElements
            + qwenRecOutElements
            + qwen38PlaneElements
            + qwen38XnElements
            + qwen38GateRawElements
            + qwen38GatedElements
            + qwen38LowrankElements
            + qwen38InjectElements
            + qwen38OOutElements
            + qwen38PleGatheredElements
            + qwen38PleKeyElements
            + qwen38PleKeyNormedElements
            + qwen38PleQueryNormedElements
            + qwen38PleValueElements
            + qwen38PleGatedElements
            + qwen38PleConvInElements
            + qwen38PleConvOutElements
            + qwen38IdxQKProjElements
            + qwen38IdxQElements
        return fp16Elements * MemoryLayout<Float16>.stride
            + qwenGBetaElements * MemoryLayout<Float>.stride
            + qwen38PleGateElements * MemoryLayout<Float>.stride
            + qwen38IdxScoreElements * MemoryLayout<Float>.stride
    }

    var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
            + qwenConvNewStateElements * MemoryLayout<Float16>.stride
            + qwen38PleConvStateElements * MemoryLayout<Float16>.stride
            + qwen38CellsElements * MemoryLayout<UInt32>.stride
            + MemoryLayout<UInt32>.stride                 // qwen38CellCount
    }

    var totalPersistentBytes: Int {
        devicePrivateBytes + sharedMetadataBytes
    }
}

struct PrefillChunkScratchBuffers {
    let layout: PrefillChunkScratchLayout
    let hidden: MTLBuffer
    let normed: MTLBuffer
    let q: MTLBuffer
    let kStage: MTLBuffer
    let vStage: MTLBuffer
    let attentionOutput: MTLBuffer
    let denseX: MTLBuffer
    let routedX: MTLBuffer
    let routerX: MTLBuffer
    let h1: MTLBuffer
    let h2: MTLBuffer
    let routePartials: MTLBuffer
    let routeIDs: MTLBuffer
    let routeWeights: MTLBuffer
    let sharedGateScratch: MTLBuffer
    let sharedUpScratch: MTLBuffer
    let sharedActScratch: MTLBuffer
    let routedGateUpActScratch: MTLBuffer
    let routedDownScratch: MTLBuffer
    // Qwen 3.6 prefill extras (1-element buffers for Gemma installs).
    let qwenQGate: MTLBuffer              // [T][qDim] fp16 output-gate scratch
    let qwenQKVProj: MTLBuffer            // [T][qkvDim] fp16 proj out / conv in
    let qwenQKVConvOut: MTLBuffer         // [T][qkvDim] fp16 conv out
    let qwenZ: MTLBuffer                  // [T][valueDim] fp16
    let qwenAB: MTLBuffer                 // [T][2V] fp16
    let qwenGBeta: MTLBuffer              // [T][2V] fp32 (g rows then beta rows)
    let qwenRecOut: MTLBuffer             // [T][valueDim] fp16
    let qwenConvNewState: MTLBuffer       // [qkvDim][3] fp16 (shared)
    // Qwen 3.8 Flash-Next prefill extras (1-element buffers for other
    // installs). The mixer stages are all plane-wide except `qwen38Lo`, and
    // each is distinct rather than aliased: a mix reads the plane while a
    // later stage of the same mix writes its own buffer, and the debug-hook
    // snapshots take the stages one at a time.
    let qwen38Plane: MTLBuffer            // [T][hc·D] fp16 residual plane
    let qwen38Xn: MTLBuffer               // [T][hc·D] fp16 normed mask plane
    let qwen38GateRaw: MTLBuffer          // [T][hc·D] fp16 up·lo raw dot
    let qwen38Gated: MTLBuffer            // [T][hc·D] fp16 xn·sigmoid(z)
    let qwen38Lo: MTLBuffer               // [T][hc_lowrank] fp16
    let qwen38Inject: MTLBuffer           // [T][hc] fp16 scatter weights
    let qwen38OOut: MTLBuffer             // [T][D] fp16 attn output projection
    let qwen38PleGathered: MTLBuffer      // [T][gathered] fp16
    let qwen38PleKey: MTLBuffer           // [T][hc·D] fp16 key projection
    let qwen38PleKeyNormed: MTLBuffer     // [T][hc·D] fp16
    let qwen38PleQueryNormed: MTLBuffer   // [T][hc·D] fp16
    let qwen38PleValue: MTLBuffer         // [T][D] fp16
    let qwen38PleGated: MTLBuffer         // [T][hc·D] fp16
    let qwen38PleConvIn: MTLBuffer        // [T][hc·D] fp16 normed gated value
    let qwen38PleConvOut: MTLBuffer       // [T][hc·D] fp16 conv output
    let qwen38PleGate: MTLBuffer          // [T][hc] fp32
    let qwen38PleConvNewState: MTLBuffer  // [hist][hc·D] fp16 (shared)
    let qwen38Cells: MTLBuffer            // [T][capacity] UInt32 (shared)
    let qwen38CellCount: MTLBuffer        // [1] UInt32 (shared, write-only)
    let qwen38IdxQKProj: MTLBuffer        // [T][(nHeads+nKV)·idxDim] fp16
    let qwen38IdxQ: MTLBuffer             // [T][nHeads·idxDim] fp16
    let qwen38IdxScore: MTLBuffer         // [maxBlocks] fp32, one row at a time

    static func allocate(device: MTLDevice,
                         layout: PrefillChunkScratchLayout) throws -> PrefillChunkScratchBuffers {
        func privateBuffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float16>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func privateFp32Buffer(_ elements: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(
                length: max(elements, 1) * MemoryLayout<Float>.stride,
                options: .storageModePrivate)
            else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        func sharedBuffer(_ bytes: Int, label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(bytes, 1),
                                                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }

        return PrefillChunkScratchBuffers(
            layout: layout,
            hidden: try privateBuffer(layout.hiddenElements, label: "prefill.hidden"),
            normed: try privateBuffer(layout.normedElements, label: "prefill.normed"),
            q: try privateBuffer(layout.qElements, label: "prefill.q"),
            kStage: try privateBuffer(layout.kStageElements, label: "prefill.kStage"),
            vStage: try privateBuffer(layout.vStageElements, label: "prefill.vStage"),
            attentionOutput: try privateBuffer(layout.attentionOutputElements, label: "prefill.attnOut"),
            denseX: try privateBuffer(layout.denseXElements, label: "prefill.denseX"),
            routedX: try privateBuffer(layout.routedXElements, label: "prefill.routedX"),
            routerX: try privateBuffer(layout.routerXElements, label: "prefill.routerX"),
            h1: try privateBuffer(layout.h1Elements, label: "prefill.h1"),
            h2: try privateBuffer(layout.h2Elements, label: "prefill.h2"),
            routePartials: try privateBuffer(layout.routePartialElements, label: "prefill.routePartials"),
            routeIDs: try sharedBuffer(layout.routeIDElements * MemoryLayout<UInt32>.stride,
                                       label: "prefill.routeIDs"),
            routeWeights: try sharedBuffer(layout.routeWeightElements * MemoryLayout<Float16>.stride,
                                           label: "prefill.routeWeights"),
            sharedGateScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                 label: "prefill.sharedGateScratch"),
            sharedUpScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                               label: "prefill.sharedUpScratch"),
            sharedActScratch: try privateBuffer(layout.sharedExpertScratchElements,
                                                label: "prefill.sharedActScratch"),
            routedGateUpActScratch: try privateBuffer(layout.routedGateUpActElements,
                                                      label: "prefill.routedGateUpActScratch"),
            routedDownScratch: try privateBuffer(layout.routedDownOutputElements,
                                                 label: "prefill.routedDownScratch"),
            qwenQGate: try privateBuffer(layout.qwenQGateElements, label: "prefill.qwenQGate"),
            qwenQKVProj: try privateBuffer(layout.qwenQKVProjElements, label: "prefill.qwenQKVProj"),
            qwenQKVConvOut: try privateBuffer(layout.qwenQKVConvOutElements, label: "prefill.qwenQKVConvOut"),
            qwenZ: try privateBuffer(layout.qwenZElements, label: "prefill.qwenZ"),
            qwenAB: try privateBuffer(layout.qwenABElements, label: "prefill.qwenAB"),
            qwenGBeta: try privateFp32Buffer(layout.qwenGBetaElements, label: "prefill.qwenGBeta"),
            qwenRecOut: try privateBuffer(layout.qwenRecOutElements, label: "prefill.qwenRecOut"),
            qwenConvNewState: try sharedBuffer(
                layout.qwenConvNewStateElements * MemoryLayout<Float16>.stride,
                label: "prefill.qwenConvNewState"),
            qwen38Plane: try privateBuffer(layout.qwen38PlaneElements, label: "prefill.qwen38Plane"),
            qwen38Xn: try privateBuffer(layout.qwen38XnElements, label: "prefill.qwen38Xn"),
            qwen38GateRaw: try privateBuffer(layout.qwen38GateRawElements,
                                             label: "prefill.qwen38GateRaw"),
            qwen38Gated: try privateBuffer(layout.qwen38GatedElements, label: "prefill.qwen38Gated"),
            qwen38Lo: try privateBuffer(layout.qwen38LowrankElements, label: "prefill.qwen38Lo"),
            qwen38Inject: try privateBuffer(layout.qwen38InjectElements,
                                            label: "prefill.qwen38Inject"),
            qwen38OOut: try privateBuffer(layout.qwen38OOutElements, label: "prefill.qwen38OOut"),
            // Shared, not private: the n-gram gather is written by the host
            // (a hash plus a pread per row) before the chunk's command buffer
            // exists, and `contents()` on a private buffer is NULL.
            qwen38PleGathered: try sharedBuffer(
                layout.qwen38PleGatheredElements * MemoryLayout<Float16>.stride,
                label: "prefill.qwen38PleGathered"),
            qwen38PleKey: try privateBuffer(layout.qwen38PleKeyElements,
                                            label: "prefill.qwen38PleKey"),
            qwen38PleKeyNormed: try privateBuffer(layout.qwen38PleKeyNormedElements,
                                                  label: "prefill.qwen38PleKeyNormed"),
            qwen38PleQueryNormed: try privateBuffer(layout.qwen38PleQueryNormedElements,
                                                    label: "prefill.qwen38PleQueryNormed"),
            qwen38PleValue: try privateBuffer(layout.qwen38PleValueElements,
                                              label: "prefill.qwen38PleValue"),
            qwen38PleGated: try privateBuffer(layout.qwen38PleGatedElements,
                                              label: "prefill.qwen38PleGated"),
            qwen38PleConvIn: try privateBuffer(layout.qwen38PleConvInElements,
                                               label: "prefill.qwen38PleConvIn"),
            qwen38PleConvOut: try privateBuffer(layout.qwen38PleConvOutElements,
                                                label: "prefill.qwen38PleConvOut"),
            qwen38PleGate: try privateFp32Buffer(layout.qwen38PleGateElements,
                                                 label: "prefill.qwen38PleGate"),
            qwen38PleConvNewState: try privateBuffer(layout.qwen38PleConvStateElements,
                                                     label: "prefill.qwen38PleConvNewState"),
            qwen38Cells: try sharedBuffer(
                layout.qwen38CellsElements * MemoryLayout<UInt32>.stride,
                label: "prefill.qwen38Cells"),
            qwen38CellCount: try sharedBuffer(MemoryLayout<UInt32>.stride,
                                              label: "prefill.qwen38CellCount"),
            qwen38IdxQKProj: try privateBuffer(layout.qwen38IdxQKProjElements,
                                               label: "prefill.qwen38IdxQKProj"),
            qwen38IdxQ: try privateBuffer(layout.qwen38IdxQElements,
                                          label: "prefill.qwen38IdxQ"),
            qwen38IdxScore: try privateFp32Buffer(layout.qwen38IdxScoreElements,
                                                  label: "prefill.qwen38IdxScore"))
    }
}

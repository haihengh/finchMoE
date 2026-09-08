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

    init(config: ArchConfig,
                chunkTokens: Int,
                routedPairMicrobatchRows: Int = 32) {
        self.chunkTokens = max(1, min(chunkTokens, 128))
        self.hiddenSize = config.hiddenSize
        let isQwen = config.modelFamily == "qwen3_6"
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
    }

    init(config: ArchConfig, runtime: PrefillRuntimeConfig) {
        self.init(config: config,
                  chunkTokens: runtime.chunkTokens)
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
        return fp16Elements * MemoryLayout<Float16>.stride
            + qwenGBetaElements * MemoryLayout<Float>.stride
    }

    var sharedMetadataBytes: Int {
        routeIDElements * MemoryLayout<UInt32>.stride
            + routeWeightElements * MemoryLayout<Float16>.stride
            + qwenConvNewStateElements * MemoryLayout<Float16>.stride
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
                label: "prefill.qwenConvNewState"))
    }
}

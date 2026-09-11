import Foundation
import Metal
import Darwin
import FinchMoEFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.finch/` model. Resident weights live behind one mmap'd
/// `MTLBuffer`; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    /// The policy the loader *resolved to*. Under
    /// `ModelIntegrityPreference.automatic` this is the answer, not the request;
    /// see `integrityOutcome` for which it was.
    public let integrityPolicy: ModelIntegrityPolicy
    /// How `integrityPolicy` was chosen. Carried on the model rather than in
    /// `ModelLoadStats` because every production call site passes no
    /// `loadStats`, so a stats field would be unreadable in practice.
    public let integrityOutcome: ModelIntegrityOutcome
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }
    /// GDN linear-attention projection width (`linear_attn.in_proj_qkv/z/a/b`,
    /// `out_proj`). 8 on the production build; raw/bf16 installs have no quant
    /// manifest and never consult this (default 4 is inert there).
    public var linearAttentionWeightBits: Int { manifest.quant?.linearAttention.weightBits ?? 4 }

    let residentBuffer: ResidentBuffer
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL
    let modelDirectory: FinchModelDirectory

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue
    /// Lazy PLE n-gram part-file state (qwen3_8 only; `count == 0` for the
    /// other families). Mirrors the layer streamers: one cached streamer per
    /// part file, opened + SHA-verified on first touch.
    let plePartsBox: PLEPartsBox
    let plePartsQueue: DispatchQueue

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        /// Where the `io` wall clock goes, accumulated here rather than on the
        /// runner because two of the three spans are only visible inside the
        /// fetch. Written from the single `DispatchQueue.global` worker that
        /// serialises expert fetches -- the decode loop awaits each fetch
        /// before issuing the next, so there is never more than one writer --
        /// and read only at the two snapshot points, which are outside any
        /// fetch. A lock would be safe and would also be inside the window
        /// these numbers exist to price.
        var ioDispatchNanos: UInt64 = 0
        var ioReadNanos: UInt64 = 0
        var ioTailNanos: UInt64 = 0
        /// `ioReadNanos` split four ways by the streamer that produced it. These
        /// tile `ioReadNanos` exactly -- `ioFanoutNanos + ioSpanNanos +
        /// ioDrainNanos` is that window, and `ioThreadNanos` is summed thread
        /// time *inside* the span, so it exceeds the span rather than adding to
        /// it. Its ratio to the span is the achieved read parallelism.
        var ioFanoutNanos: UInt64 = 0
        var ioSpanNanos: UInt64 = 0
        var ioDrainNanos: UInt64 = 0
        var ioThreadNanos: UInt64 = 0
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    final class PLEPartsBox: @unchecked Sendable {
        var streamers: [PLEPartStreamer?]
        var verified: [Bool]
        init(count: Int) {
            self.streamers = Array(repeating: nil, count: count)
            self.verified = Array(repeating: false, count: count)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         integrityOutcome: ModelIntegrityOutcome,
         residentBuffer: ResidentBuffer,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: FinchModelDirectory) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.integrityOutcome = integrityOutcome
        self.residentBuffer = residentBuffer
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "finchmoe.expert-streamers")
        self.plePartsBox = PLEPartsBox(count: config.ngramPartCount)
        self.plePartsQueue = DispatchQueue(label: "finchmoe.ple-parts")
    }

    // MARK: - Resident accessors

    public var embedding: TensorView {
        // Manifest names drop the HF root "model." prefix. Qwen3.6 nests the
        // transformer under `model.language_model.model.*` (hence
        // "language_model.model.embed_tokens.weight"); Qwen3.8-Flash-Next
        // hangs the layer stack directly off the language model
        // ("language_model.embed_tokens.weight").
        try! resident(name: config.isQwen3_8
            ? "language_model.embed_tokens.weight"
            : "language_model.model.embed_tokens.weight")
    }

    /// Gemma 4 ties lm_head to the embedding (the transpose for the GEMV path
    /// is the kernel's job). Qwen 3.6/3.8 have an untied root `lm_head.weight`.
    public var lmHead: TensorView {
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family, ArchConfig.qwen3_8Family:
            return try! resident(name: "lm_head.weight")
        default: return embedding
        }
    }

    /// Resolve one tensor under a transformer layer, on the family prefix
    /// (3.8 shallow `language_model.layers.<L>`, Gemma/Qwen3.6 deep
    /// `language_model.model.layers.<L>`). The per-family tensor SETS still
    /// differ (a 3.8 layer has no `input_layernorm.weight` and a Gemma layer
    /// has no `linear_attn.*`); the name lookup throws `tensorNotFound` when
    /// the family does not carry the requested tensor.
    private func residentLayer(_ suffix: String, layer L: Int) throws -> TensorView {
        try resident(name: "\(layerPrefix).\(L).\(suffix)")
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.q_proj.weight", layer: L)
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.k_proj.weight", layer: L)
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.v_proj.weight", layer: L)
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.o_proj.weight", layer: L)
    }
    /// Router weight. Gemma writer emits `.router.proj.weight` (no `.mlp.`
    /// segment); Qwen 3.6/3.8 use `.mlp.gate.weight` on the family layer
    /// prefix.
    public func router(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family, ArchConfig.qwen3_8Family:
            return try resident(name: "\(layerPrefix).\(L).mlp.gate.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).router.proj.weight")
        }
    }
    /// Shared-expert FFN. Gemma writer emits `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment; Qwen 3.6/3.8 keep the full
    /// `.mlp.shared_expert.{gate,up,down}_proj.weight` names.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family, ArchConfig.qwen3_8Family:
            return try resident(name: "\(layerPrefix).\(L).mlp.shared_expert.gate_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.gate_proj.weight")
        }
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family, ArchConfig.qwen3_8Family:
            return try resident(name: "\(layerPrefix).\(L).mlp.shared_expert.up_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.up_proj.weight")
        }
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family, ArchConfig.qwen3_8Family:
            return try resident(name: "\(layerPrefix).\(L).mlp.shared_expert.down_proj.weight")
        default:
            return try resident(name: "language_model.model.layers.\(L).mlp.down_proj.weight")
        }
    }

    // MARK: - Qwen hybrid GDN (linear-attention) accessors
    //
    // GDN layers replace `self_attn.*` with `linear_attn.*`; the shared
    // expert gate is the sigmoid scalar `mlp.shared_expert_gate.weight` [1, D].
    // All are Qwen-only — touching them on a Gemma install throws
    // `tensorNotFound`. Qwen3.8-Flash-Next keeps the 3.6 tensor names on a
    // shallower prefix (`language_model.layers` vs `language_model.model.layers`).

    /// Manifest prefix for one transformer layer's tensors, family-dependent
    /// (see `embedding`). Shared by the GDN/self-attn/mlp accessors.
    private var layerPrefix: String {
        config.isQwen3_8 ? "language_model.layers" : "language_model.model.layers"
    }

    private func qwenResident(_ suffix: String, layer L: Int) throws -> TensorView {
        guard config.isQwenHybrid else {
            throw ModelError.tensorNotFound(name: "\(layerPrefix).\(L).\(suffix) (qwen-only)")
        }
        return try resident(name: "\(layerPrefix).\(L).\(suffix)")
    }

    public func gdnInProjQKV(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_qkv.weight", layer: L)
    }
    public func gdnInProjZ(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_z.weight", layer: L)
    }
    public func gdnInProjA(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_a.weight", layer: L)
    }
    public func gdnInProjB(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.in_proj_b.weight", layer: L)
    }
    public func gdnOutProj(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.out_proj.weight", layer: L)
    }
    public func gdnALog(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.A_log", layer: L)
    }
    public func gdnDtBias(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.dt_bias", layer: L)
    }
    public func gdnNormWeight(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.norm.weight", layer: L)
    }
    public func gdnConv1D(layer L: Int) throws -> TensorView {
        try qwenResident("linear_attn.conv1d.weight", layer: L)
    }
    public func sharedExpertGateProj(layer L: Int) throws -> TensorView {
        try qwenResident("mlp.shared_expert_gate.weight", layer: L)
    }
    /// Block input norm (RMSNorm). Qwen3.8-Flash-Next replaces both block
    /// norms with hyper-connections — these lookups throw `tensorNotFound`
    /// there (the M3.1 3.8 layer bodies consume the HC mixers instead).
    public func inputNorm(layer L: Int) throws -> TensorView {
        try residentLayer("input_layernorm.weight", layer: L)
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try residentLayer("post_attention_layernorm.weight", layer: L)
    }
    /// Final RMSNorm before lm_head. Qwen3.8-Flash-Next has no `model.norm`:
    /// the root `hyper_connection_mixer` collapses the 4-stream plane into the
    /// lm_head input (see `hyperConnectionMixer()`), so this is nil there.
    /// Gemma and Qwen3.6 keep the shared `language_model.model.norm.weight`.
    public var finalNorm: TensorView? {
        guard !config.isQwen3_8 else { return nil }
        return try! resident(name: "language_model.model.norm.weight")
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.q_norm.weight", layer: L)
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try residentLayer("self_attn.k_norm.weight", layer: L)
    }

    // MARK: - Qwen3.8-Flash-Next: hyper-connections
    //
    // Flash-Next replaces every block RMSNorm (and the final `model.norm`)
    // with hyper-connections: 4 parallel residual streams fused into one
    // `hyperConnectionDim`-wide plane. Each per-layer mixer reads its 4-tuple
    // (grouped-RMS gate + int4 down/up mix + per-stream block injection);
    // the root mixer (3 tensors, no inject) collapses the plane at the head.
    // All are 3.8-only — the guard throws `tensorNotFound` otherwise.
    // Math locked in docs/QWEN38_PORT.md §hyper-connection.

    private func qwen38Resident(_ name: String) throws -> TensorView {
        guard config.isQwen3_8 else {
            throw ModelError.tensorNotFound(name: name)
        }
        return try resident(name: name)
    }

    /// The per-layer attention-branch mixer bundle (`attn_hyper_connection`):
    /// `hc_norm` raw-BF16 grouped-RMS gate, int4 `input_mix_weight_down` /
    /// `input_mix_weight_up` mix, and the per-stream `block_inject_weight`.
    public func attnHyperConnection(layer L: Int) throws
        -> (hcNorm: TensorView, mixDown: TensorView, mixUp: TensorView,
            blockInject: TensorView) {
        let p = "\(layerPrefix).\(L).attn_hyper_connection."
        return (
            hcNorm:      try qwen38Resident(p + "hc_norm.weight"),
            mixDown:     try qwen38Resident(p + "input_mix_weight_down.weight"),
            mixUp:       try qwen38Resident(p + "input_mix_weight_up.weight"),
            blockInject: try qwen38Resident(p + "block_inject_weight.weight"))
    }

    /// The per-layer MLP-branch mixer bundle (`mlp_hyper_connection`), same
    /// shape as `attnHyperConnection`.
    public func mlpHyperConnection(layer L: Int) throws
        -> (hcNorm: TensorView, mixDown: TensorView, mixUp: TensorView,
            blockInject: TensorView) {
        let p = "\(layerPrefix).\(L).mlp_hyper_connection."
        return (
            hcNorm:      try qwen38Resident(p + "hc_norm.weight"),
            mixDown:     try qwen38Resident(p + "input_mix_weight_down.weight"),
            mixUp:       try qwen38Resident(p + "input_mix_weight_up.weight"),
            blockInject: try qwen38Resident(p + "block_inject_weight.weight"))
    }

    /// The root `hyper_connection_mixer` (3 tensors — no `block_inject`): the
    /// terminal collapse from the wide plane to the lm_head input.
    public func hyperConnectionMixer() throws
        -> (hcNorm: TensorView, mixDown: TensorView, mixUp: TensorView) {
        let p = "language_model.hyper_connection_mixer."
        return (
            hcNorm:  try qwen38Resident(p + "hc_norm.weight"),
            mixDown: try qwen38Resident(p + "input_mix_weight_down.weight"),
            mixUp:   try qwen38Resident(p + "input_mix_weight_up.weight"))
    }

    // MARK: - Qwen3.8-Flash-Next: QSA indexer (full layers only)

    /// Sparse-block indexer projections on a full layer:
    /// `index_qk_proj` (int4) + the 1+w-baked per-head `q_layernorm` /
    /// `k_layernorm` RMS scales. GDN layers carry no indexer — the lookup
    /// throws `tensorNotFound` there.
    public func indexerQKProj(layer L: Int) throws -> TensorView {
        try qwen38Resident("\(layerPrefix).\(L).self_attn.indexer.index_qk_proj.weight")
    }
    public func indexerQLayernorm(layer L: Int) throws -> TensorView {
        try qwen38Resident("\(layerPrefix).\(L).self_attn.indexer.q_layernorm.weight")
    }
    public func indexerKLayernorm(layer L: Int) throws -> TensorView {
        try qwen38Resident("\(layerPrefix).\(L).self_attn.indexer.k_layernorm.weight")
    }

    // MARK: - Qwen3.8-Flash-Next: PLE n-gram head (pleLayerIndexes only)

    /// The 0-based layer hosting the PLE n-gram block (`ple_layer_ids`,
    /// config-1-based → minus 1). nil when the family carries no PLE.
    public var pleLayerIndex: Int? {
        config.isQwen3_8 ? config.pleLayerIndexes.first : nil
    }

    private func pleResident(_ suffix: String) throws -> TensorView {
        guard config.isQwen3_8, let L = config.pleLayerIndexes.first else {
            throw ModelError.tensorNotFound(name: "language_model.layers.?.ple.\(suffix)")
        }
        return try resident(name: "\(layerPrefix).\(L).ple.\(suffix)")
    }

    /// PLE depthwise conv1d — checkpoint [plane, 1, kernel], emitted squeezed
    /// [plane * kernel] as raw FP16 (GDN conv policy).
    public func pleConv1D() throws -> TensorView {
        try pleResident("conv1d.weight")
    }
    /// PLE key/value projections from the gathered n-gram rows (int8).
    public func pleKeyProj() throws -> TensorView {
        try pleResident("key_proj.weight")
    }
    public func pleValueProj() throws -> TensorView {
        try pleResident("value_proj.weight")
    }
    /// PLE grouped-RMS norms (1+w baked).
    public func pleNormQuery() throws -> TensorView {
        try pleResident("norm_query.weight")
    }
    public func pleNormKey() throws -> TensorView {
        try pleResident("norm_key.weight")
    }
    public func pleNormConv() throws -> TensorView {
        try pleResident("norm_conv.weight")
    }
    /// PLE host-hash metadata, raw I64 resident tensors: per-gram-position
    /// multipliers, and per-head vocab offsets / vocab sizes (all
    /// byte-exact — consumed CPU-side by the host hash, see docs).
    public func pleLayerMultipliers() throws -> TensorView {
        try pleResident("ple_embedding.layer_multipliers")
    }
    public func pleHeadsOffsets() throws -> TensorView {
        try pleResident("ple_embedding.ngram_heads_offsets")
    }
    public func pleHeadsVocabSizes() throws -> TensorView {
        try pleResident("ple_embedding.ngram_heads_vocab_sizes")
    }

    /// The three PLE hash-metadata vectors as host-side `UInt64`. They are raw
    /// I64 in the resident file precisely so they round-trip byte-exact —
    /// multipliers run up to 45 bits and are *not* representable as floats —
    /// so this is a straight read of the loaded bytes, validated for length and
    /// dtype at load (`requireInt64`). One multiplier per gram position, and
    /// one `(offset, vocabSize)` pair per head.
    public func pleHashConstants() throws -> (multipliers: [UInt64],
                                              headOffsets: [UInt64],
                                              headVocabSizes: [UInt64]) {
        let heads = config.ngramSize > 1
            ? (config.ngramSize - 1) * config.headsPerNgram : 0
        func read(_ view: TensorView, _ count: Int, _ name: String) throws -> [UInt64] {
            guard count > 0 else { return [] }
            let base = UnsafeRawPointer(view.buffer.contents())
                .advanced(by: Int(view.offset))
            guard Int(view.length) >= count * MemoryLayout<Int64>.size,
                  UInt(bitPattern: base) % UInt(MemoryLayout<Int64>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) is not a readable I64 vector of \(count)")
            }
            let p = base.assumingMemoryBound(to: Int64.self)
            return (0..<count).map { UInt64(bitPattern: p[$0]) }
        }
        return (try read(pleLayerMultipliers(), config.ngramSize, "layer_multipliers"),
                try read(pleHeadsOffsets(), heads, "ngram_heads_offsets"),
                try read(pleHeadsVocabSizes(), heads, "ngram_heads_vocab_sizes"))
    }

    // MARK: - Qwen3.8-Flash-Next: PLE n-gram part files (lazy)

    /// First touch of part file `part` opens it + verifies SHA-256; the
    /// returned streamer is cached for the model lifetime (mirrors the
    /// per-layer routed-expert streamers). Parts are raw BF16 row-major
    /// `[ngramPartRows, ngramRowDim]`; row addressing/hash layout is the M3.3
    /// PLE gather's job.
    public func openPLEPart(_ part: Int) throws -> PLEPartStreamer {
        try plePartsQueue.sync {
            try openPLEPartLocked(part)
        }
    }

    /// Best-effort overlap hook (mirrors `beginOpeningRoutedExpertStreamer`).
    public func beginOpeningPLEPart(_ part: Int) {
        nonisolated(unsafe) let model = self
        plePartsQueue.async {
            _ = try? model.openPLEPartLocked(part)
        }
    }

    /// Test hook: how many part files have been opened so far.
    public func plePartOpenCount() -> Int {
        plePartsQueue.sync { plePartsBox.streamers.compactMap { $0 }.count }
    }

    private func openPLEPartLocked(_ part: Int) throws -> PLEPartStreamer {
        guard config.isQwen3_8, part >= 0, part < plePartsBox.streamers.count else {
            throw ModelError.indexCorrupt(
                detail: "PLE part \(part) out of 0..<\(plePartsBox.streamers.count) (qwen3_8-only)")
        }
        if let existing = plePartsBox.streamers[part] {
            return existing
        }
        let name = String(format: "ple_shards/shard_%03d.bin", part)
        let partFD = try modelDirectory.openFile(name)
        defer { close(partFD) }
        guard let entry = manifest.files[name] else {
            throw ModelError.missingFile(name: name)
        }
        let actualSize = try modelDirectory.fileSize(
            fileDescriptor: partFD, relativePath: name)
        guard actualSize == entry.size else {
            throw ModelError.tensorSizeMismatch(
                name: name, expected: entry.size, actual: actualSize)
        }
        if !plePartsBox.verified[part] {
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: partFD,
                                              named: name,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
        }
        let streamer = try PLEPartStreamer(
            partIndex: part,
            rows: config.ngramPartRows,
            columns: config.ngramRowDim,
            fileDescriptor: partFD)
        plePartsBox.streamers[part] = streamer
        plePartsBox.verified[part] = true
        return streamer
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    /// Resolve a tensor name to a `TensorView` against the resident buffer.
    /// `fileOffset` (absolute) is converted to a buffer-relative offset by
    /// subtracting the resident region's file offset (which equals
    /// `header.indexSize`).
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let residentFileOffset = residentIndex.header.indexSize
        func checkedRelativeOffset(_ absolute: UInt64,
                                   size: UInt64,
                                   field: String) throws -> UInt64 {
            if size == 0 {
                guard absolute == 0 else {
                    throw ModelError.indexCorrupt(detail: "\(name).\(field) has an absent nonzero offset")
                }
                return 0
            }
            guard absolute >= residentFileOffset else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) precedes the resident payload")
            }
            let relative = absolute - residentFileOffset
            guard relative <= residentIndex.header.residentSize,
                  size <= residentIndex.header.residentSize - relative else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) exceeds the resident payload")
            }
            return relative
        }
        let relativeOffset = try checkedRelativeOffset(
            entry.fileOffset, size: entry.sizeBytes, field: "weights")
        let scaleRel = try checkedRelativeOffset(
            entry.scaleOffset, size: entry.scaleSize, field: "scales")
        let biasRel = try checkedRelativeOffset(
            entry.biasOffset, size: entry.biasSize, field: "biases")
        return TensorView(
            buffer: residentBuffer.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: FinchFormatV1.DType.u32.rawValue)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: layerFD,
                                              named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy,
            fileDescriptor: layerFD)
        streamersBox.layerVerified[L] = true
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.finch/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig = .gemma4_26B_A4B,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPreference = .automatic,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let modelDirectory = try FinchModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do { manifestFD = try modelDirectory.openFile("manifest.json") }
        catch ModelError.missingFile { throw ModelError.partialInstall(path: directoryURL.path) }
        defer { close(manifestFD) }
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let resolution = try Self.resolveIntegrity(
            preference: integrityPolicy,
            directoryURL: directoryURL,
            manifestSha256: manifestSha,
            stats: &stats)
        let effectivePolicy = resolution.policy
        let receipt = resolution.receipt

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if effectivePolicy == .sizeCheckTrustedReceipt, let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
            throw ModelError.missingFile(name: "packed_experts/layout.json")
        }
        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
        defer { close(layoutFD) }
        let layoutData = try modelDirectory.readMetadata(
            fileDescriptor: layoutFD, relativePath: "packed_experts/layout.json",
            maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
        guard UInt64(layoutData.count) == layoutEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "packed_experts/layout.json",
                expected: layoutEntry.size,
                actual: UInt64(layoutData.count))
        }
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(fileDescriptor: weightsFD,
                                      named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        guard Sha256Verifier.hashData(layoutData).lowercased()
                == layoutEntry.sha256.lowercased() else {
            throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
        }
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        let layout = try PackedExpertsLayoutReader.decode(data: layoutData,
                                                          manifest: manifest)
        if effectivePolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(modelDirectory: modelDirectory,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  manifest: manifest,
                                  config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(fileSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        let residentBuffer = try ResidentBuffer(
            fileURL: weightsURL,
            fileOffset: residentIndex.header.indexSize,
            residentSize: residentIndex.header.residentSize,
            device: device,
            fileDescriptor: weightsFD)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: effectivePolicy,
            integrityOutcome: resolution.outcome,
            residentBuffer: residentBuffer,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    /// The policy a load actually runs under, paired with the receipt that
    /// justified it.
    ///
    /// These two must never be chosen independently. The receipt-validate gate
    /// keys off `receipt`, the lazy layer/PLE gates and the stored
    /// `Model.integrityPolicy` key off the policy — so a fallback that cleared
    /// the receipt but left the policy at `.sizeCheckTrustedReceipt` would skip
    /// the lazy hashes *and* validate no receipt, verifying less than either
    /// mode alone. Producing both from one `let` makes that state
    /// unrepresentable instead of merely commented against.
    private struct IntegrityResolution {
        let policy: ModelIntegrityPolicy
        let receipt: VerifiedInstallReceipt?
        let outcome: ModelIntegrityOutcome
    }

    /// Resolve `preference` against what is actually on disk.
    ///
    /// The `do` body touches nothing but the receipt, so everything it can throw
    /// is receipt-scoped — which is what licenses catching it and continuing.
    /// Anything else (I/O on the model directory, a decoding failure that
    /// escapes the reader's wrap) propagates rather than being downgraded into a
    /// silent "assume the receipt is bad".
    private static func resolveIntegrity(preference: ModelIntegrityPreference,
                                         directoryURL: URL,
                                         manifestSha256: String,
                                         stats: inout ModelLoadStats) throws -> IntegrityResolution {
        switch preference {
        case .fullSha256:
            return IntegrityResolution(policy: .fullSha256,
                                       receipt: nil,
                                       outcome: .explicitFullSha256)
        case .sizeCheckTrustedReceipt:
            // Explicit stays strict: no receipt is an error, as before.
            let receipt = try Self.loadReceipt(directoryURL: directoryURL,
                                               manifestSha256: manifestSha256,
                                               stats: &stats)
            return IntegrityResolution(policy: .sizeCheckTrustedReceipt,
                                       receipt: receipt,
                                       outcome: .explicitTrustedReceipt)
        case .automatic:
            do {
                let receipt = try Self.loadReceipt(directoryURL: directoryURL,
                                                   manifestSha256: manifestSha256,
                                                   stats: &stats)
                return IntegrityResolution(policy: .sizeCheckTrustedReceipt,
                                           receipt: receipt,
                                           outcome: .automaticUsedReceipt)
            } catch ModelError.trustedReceiptInvalid(let detail) {
                // Falling back hashes *more*, never less, so this is safe to do
                // silently; `isPresent` decides whether it is worth a warning.
                return IntegrityResolution(
                    policy: .fullSha256,
                    receipt: nil,
                    outcome: VerifiedInstallReceiptReader.isPresent(directoryURL: directoryURL)
                        ? .automaticFellBackInvalid(detail: detail)
                        : .automaticFellBackAbsent)
            }
        }
    }

    private static func loadReceipt(directoryURL: URL,
                                    manifestSha256: String,
                                    stats: inout ModelLoadStats) throws -> VerifiedInstallReceipt {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // `defer` so a failed attempt is still counted as receipt-validation
        // time rather than vanishing from the stats.
        defer { stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start }
        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directoryURL)
        try VerifiedInstallReceiptReader.validateManifestBinding(
            receipt, directoryURL: directoryURL, manifestSha256: manifestSha256)
        return receipt
    }

    private static func validateTrustedReceiptLayerLayout(modelDirectory: FinchModelDirectory,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
            }
            return value
        }

        func dimensions(_ rows: Int, _ columns: Int, field: String) throws -> (UInt32, UInt32) {
            guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                  r > 0, c > 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
            }
            return (r, c)
        }

        func requireBF16(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == FinchFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required BF16 schema")
            }
        }

        /// Raw (unquantized) 1-D resident tensor of a fixed dtype — used for
        /// the Qwen GDN scalars (`A_log`/`dt_bias` fp32) and the conv1d weight
        /// (fp16; the writer converts bf16 → fp16 at emit).
        func requireRaw(_ name: String, count: Int, dtype: FinchFormatV1.DType) throws {
            precondition(dtype == .fp16 || dtype == .fp32,
                         "requireRaw supports only fp16/fp32")
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let stride = dtype == .fp32 ? MemoryLayout<Float>.size : MemoryLayout<Float16>.size
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(stride), field: name)
            guard entry.dtype == dtype.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(stride) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required raw \(dtype) schema")
            }
        }

        /// Raw (unquantized) 1-D resident tensor at fixed I64 — the Qwen3.8
        /// PLE hash metadata (per-gram-position layer multipliers + per-head
        /// vocab offsets/sizes, up to 45 bits) must round-trip byte-exact, so
        /// it rides the resident file raw (dtype byte 4) rather than through a
        /// float transform.
        func requireInt64(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<Int64>.size), field: name)
            guard entry.dtype == FinchFormatV1.DType.i64.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<Int64>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required I64 schema")
            }
        }

        func affineSizes(rows: Int,
                         columns: Int,
                         slot: ManifestQuantSlot,
                         field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
            let shape = try dimensions(rows, columns, field: field)
            guard slot.weightBits == 4 || slot.weightBits == 8,
                  slot.groupSize > 0,
                  columns % slot.groupSize == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
            }
            let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
            let bitCount = try checkedMultiply(elements, UInt64(slot.weightBits), field: field)
            guard bitCount % 8 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
            }
            let groups = UInt64(columns / slot.groupSize)
            let auxElements = try checkedMultiply(UInt64(shape.0), groups, field: field)
            let auxBytes = try checkedMultiply(
                auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
            return (shape, bitCount / 8, auxBytes)
        }

        func requireAffine(_ name: String,
                           rows: Int,
                           columns: Int,
                           slot: ManifestQuantSlot) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let expected = try affineSizes(
                rows: rows, columns: columns, slot: slot, field: name)
            let primaryAlignment: UInt64 = slot.weightBits == 4
                ? UInt64(MemoryLayout<UInt16>.alignment)
                : 1
            guard entry.dtype == FinchFormatV1.DType.u32.rawValue,
                  entry.shape.0 == expected.shape.0,
                  entry.shape.1 == expected.shape.1,
                  entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expected.weight,
                  entry.scaleSize == expected.aux,
                  entry.biasSize == expected.aux,
                  entry.fileOffset % primaryAlignment == 0,
                  entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                  entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
            }
        }

        try requireAffine(
            config.isQwen3_8
                ? "language_model.embed_tokens.weight"
                : "language_model.model.embed_tokens.weight",
            rows: config.vocabSize,
            columns: config.hiddenSize,
            slot: quant.embedding)
        // Qwen3.8-Flash-Next has no `model.norm`: the shared final
        // `hyper_connection_mixer` replaces the last RMSNorm.
        if !config.isQwen3_8 {
            try requireBF16("language_model.model.norm.weight", count: config.hiddenSize)
        }

        // Per-layer tensor sets diverge by family: Gemma 4 has the
        // q/k/v sandwich norms + router auxiliaries; Qwen hybrid has GDN
        // (linear_attn.*) on the non-full layers, a doubled q_proj + output
        // gate on the full layers, and a sigmoid-gated shared expert. The
        // routed-expert packed layout below is family-independent.
        switch config.modelFamily {
        case ArchConfig.qwen3_6Family:
            try validateQwen36Layers(config: config, quant: quant,
                                     requireBF16: requireBF16,
                                     requireAffine: requireAffine,
                                     requireRaw: requireRaw,
                                     checkedIntMultiply: checkedIntMultiply)
        case ArchConfig.qwen3_8Family:
            try validateQwen38Layers(config: config, quant: quant,
                                     requireBF16: requireBF16,
                                     requireAffine: requireAffine,
                                     requireRaw: requireRaw,
                                     requireInt64: requireInt64,
                                     checkedIntMultiply: checkedIntMultiply)
        default:
            try validateGemma4Layers(config: config, quant: quant,
                                     requireBF16: requireBF16,
                                     requireAffine: requireAffine,
                                     checkedIntMultiply: checkedIntMultiply)
        }

        // PLE n-gram part files (qwen3_8): every `ple_shards/shard_%03d.bin`
        // must be a manifest.files entry of exactly [ngramPartRows, 160] raw
        // BF16. The parts have no resident-index slots — they stream from
        // their own files via `openPLEPart` (schema documented in
        // `validateQwen38Layers`).
        if config.isQwen3_8 {
            let partRows = config.ngramPartRows
            let partColumns = config.ngramRowDim
            guard partRows > 0, partColumns > 0 else {
                throw ModelError.indexCorrupt(
                    detail: "qwen3_8 preset must set the n-gram part geometry")
            }
            let expectedPartBytes = try checkedMultiply(
                UInt64(partRows),
                UInt64(partColumns * MemoryLayout<UInt16>.size),
                field: "PLE part file")
            for part in 0..<config.ngramPartCount {
                let name = String(format: "ple_shards/shard_%03d.bin", part)
                guard let entry = manifest.files[name] else {
                    throw ModelError.missingFile(name: name)
                }
                guard entry.size == expectedPartBytes else {
                    throw ModelError.tensorSizeMismatch(
                        name: name, expected: expectedPartBytes, actual: entry.size)
                }
            }
        }

        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }

    // MARK: - Per-layer schema validators (split by model family)

    private static func validateGemma4Layers(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, "layer \(layer) key/value")

            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                try requireBF16("\(prefix).\(name)", config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", headDimension)
            try requireBF16("\(prefix).self_attn.k_norm.weight", headDimension)
            try requireBF16("\(prefix).router.per_expert_scale", config.numExperts)
            try requireBF16("\(prefix).layer_scalar", 1)

            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              queryDimension, config.hiddenSize,
                              quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              kvDimension, config.hiddenSize,
                              quant.attention)
            if !isFull {
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  kvDimension, config.hiddenSize,
                                  quant.attention)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension,
                              quant.attention)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              config.hiddenSize, config.intermediateSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).router.proj.weight",
                              config.numExperts, config.hiddenSize,
                              quant.router)
        }
    }

    private static func validateQwen36Layers(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        requireRaw: (String, Int, FinchFormatV1.DType) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let keyDim = try checkedIntMultiply(
            config.linearNumKeyHeads, config.linearKeyHeadDim,
            "GDN key dim")
        let valueDim = try checkedIntMultiply(
            config.linearNumValueHeads, config.linearValueHeadDim,
            "GDN value dim")
        let qkvDim = try checkedIntMultiply(keyDim, 2, "GDN q+k dim")
            + valueDim
        let convCount = try checkedIntMultiply(
            qkvDim, config.linearConvKernelDim, "GDN conv weight")

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0

            try requireBF16("\(prefix).input_layernorm.weight", config.hiddenSize)
            try requireBF16("\(prefix).post_attention_layernorm.weight", config.hiddenSize)

            if isFull {
                // q_proj is doubled for attn_output_gate: [2*Q*head_dim, D].
                let queryRows = try checkedIntMultiply(
                    config.numHeads, config.fullHeadDim, "layer \(layer) query")
                let doubledQuery = try checkedIntMultiply(
                    queryRows, 2, "layer \(layer) doubled query")
                let kvRows = try checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim,
                    "layer \(layer) key/value")

                try requireAffine("\(prefix).self_attn.q_proj.weight",
                                  doubledQuery, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  kvRows, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  kvRows, config.hiddenSize,
                                  quant.attention)
                try requireAffine("\(prefix).self_attn.o_proj.weight",
                                  config.hiddenSize, queryRows,
                                  quant.attention)
                try requireBF16("\(prefix).self_attn.q_norm.weight", config.fullHeadDim)
                try requireBF16("\(prefix).self_attn.k_norm.weight", config.fullHeadDim)
            } else {
                // GDN (linear-attention) layer. The five projections ride the
                // dedicated linearAttention slot (8-bit on the production
                // build; the recurrent state amplifies their quant noise).
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  qkvDim, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  valueDim, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  config.linearNumValueHeads, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  config.linearNumValueHeads, config.hiddenSize,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  config.hiddenSize, valueDim,
                                  quant.linearAttention)
                try requireBF16("\(prefix).linear_attn.norm.weight",
                                config.linearValueHeadDim)
                try requireRaw("\(prefix).linear_attn.A_log",
                               config.linearNumValueHeads, .fp32)
                try requireRaw("\(prefix).linear_attn.dt_bias",
                               config.linearNumValueHeads, .fp32)
                // conv1d.weight is [qkvDim, 1, kernel] in the checkpoint; the
                // writer emits the squeezed [qkvDim, kernel] rows as raw FP16
                // (bf16 → fp16 conversion at emit).
                try requireRaw("\(prefix).linear_attn.conv1d.weight",
                               convCount, .fp16)
            }

            // Shared expert + sigmoid gate + router (both layer types).
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              config.intermediateSize, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              config.hiddenSize, config.intermediateSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                              1, config.hiddenSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.gate.weight",
                              config.numExperts, config.hiddenSize,
                              quant.router)
        }
    }

    /// Qwen3.8-Flash-Next resident schema — hyper-connection mixers in every
    /// layer (they replace the block norms AND `model.norm`), full layers
    /// (3,7,…) additionally carrying the QSA indexer, and layer 1 (0-based,
    /// `pleLayerIndexes`) the PLE n-gram head. Names live on the shallow
    /// `language_model.layers.<L>` prefix; every formula below mirrors the M1
    /// repack transforms (docs/QWEN38_PORT.md + QwenRepackPlannerTests):
    ///
    ///   - `hc_norm` (per-layer + root mixer): raw BF16, `hyperConnectionDim`.
    ///   - down/up mix + `block_inject`: int4 affine (`quant.attention`),
    ///     [lowrank, plane] / [plane, lowrank] / [hc_count, plane].
    ///   - full self-attn: doubled q [2·heads·headDim], k/v [kv·headDim], o
    ///     [D, heads·headDim], q/k_norm [headDim] (1+w-baked payload, BF16
    ///     wire) — the 3.6 full-layer set on the 3.8 prefix.
    ///   - indexer: `index_qk_proj` int4 [(n+kv)·headDim, D], q/k_layernorm
    ///     BF16 [headDim] (1+w-baked).
    ///   - GDN: the 3.6-shaped nine (int8 five-projection set, raw fp32
    ///     A_log/dt_bias, fp16 squeezed conv).
    ///   - PLE head: key/value_proj int8 [plane|D, totalHeads·160], grouped
    ///     norms BF16 [plane], conv fp16 squeezed [plane·kernel], I64
    ///     multipliers [ngramSize] + offsets/vocab sizes [totalHeads] — plus
    ///     the `ple_shards/shard_%03d.bin` manifest entries (raw BF16 parts,
    ///     no resident slots).
    private static func validateQwen38Layers(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        requireRaw: (String, Int, FinchFormatV1.DType) throws -> Void,
        requireInt64: (String, Int) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let D = config.hiddenSize
        let plane = config.hyperConnectionDim
        let lowrank = config.hyperConnectionLowrank
        let streamCount = config.hyperConnectionCount
        guard plane > 0, lowrank > 0, streamCount > 0 else {
            throw ModelError.indexCorrupt(
                detail: "qwen3_8 preset must set the hyper-connection geometry")
        }
        let keyDim = try checkedIntMultiply(
            config.linearNumKeyHeads, config.linearKeyHeadDim, "GDN key dim")
        let valueDim = try checkedIntMultiply(
            config.linearNumValueHeads, config.linearValueHeadDim, "GDN value dim")
        let qkvDim = try checkedIntMultiply(keyDim, 2, "GDN q+k dim")
            + valueDim
        let convCount = try checkedIntMultiply(
            qkvDim, config.linearConvKernelDim, "GDN conv weight")
        // PLE gathered width: one 160-wide row per head, all heads:
        // (ngramSize − 1) orders × headsPerNgram (16 × 160 = 2560 real).
        let pleHeads = try checkedIntMultiply(
            config.ngramSize - 1, config.headsPerNgram, "PLE head count")
        let gatheredWidth = try checkedIntMultiply(
            pleHeads, config.ngramRowDim, "PLE gathered width")

        for layer in 0..<config.numLayers {
            let prefix = "language_model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0

            // Hyper-connection mixers (attn + mlp branches) on every layer.
            for bundle in ["attn_hyper_connection", "mlp_hyper_connection"] {
                try requireBF16("\(prefix).\(bundle).hc_norm.weight", plane)
                try requireAffine("\(prefix).\(bundle).input_mix_weight_down.weight",
                                  lowrank, plane, quant.attention)
                try requireAffine("\(prefix).\(bundle).input_mix_weight_up.weight",
                                  plane, lowrank, quant.attention)
                try requireAffine("\(prefix).\(bundle).block_inject_weight.weight",
                                  streamCount, plane, quant.attention)
            }

            if isFull {
                // Full attention: doubled q (output gate), shared per-head
                // q/k_norm scales [fullHeadDim] — the 3.6 full-layer set.
                let queryRows = try checkedIntMultiply(
                    config.numHeads, config.fullHeadDim, "layer \(layer) query")
                let doubledQuery = try checkedIntMultiply(
                    queryRows, 2, "layer \(layer) doubled query")
                let kvRows = try checkedIntMultiply(
                    config.numFullKVHeads, config.fullHeadDim,
                    "layer \(layer) key/value")
                let indexRows = try checkedIntMultiply(
                    config.indexerNumHeads + config.indexerKVHeads,
                    config.indexerHeadDim, "layer \(layer) indexer")

                try requireAffine("\(prefix).self_attn.q_proj.weight",
                                  doubledQuery, D, quant.attention)
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  kvRows, D, quant.attention)
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  kvRows, D, quant.attention)
                try requireAffine("\(prefix).self_attn.o_proj.weight",
                                  D, queryRows, quant.attention)
                try requireBF16("\(prefix).self_attn.q_norm.weight", config.fullHeadDim)
                try requireBF16("\(prefix).self_attn.k_norm.weight", config.fullHeadDim)

                // QSA indexer (full layers only).
                try requireAffine("\(prefix).self_attn.indexer.index_qk_proj.weight",
                                  indexRows, D, quant.attention)
                try requireBF16("\(prefix).self_attn.indexer.q_layernorm.weight",
                                config.indexerHeadDim)
                try requireBF16("\(prefix).self_attn.indexer.k_layernorm.weight",
                                config.indexerHeadDim)
            } else {
                // GDN (linear-attention) layer, the 3.6-shaped nine.
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  qkvDim, D, quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  valueDim, D, quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  config.linearNumValueHeads, D,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  config.linearNumValueHeads, D,
                                  quant.linearAttention)
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  D, valueDim, quant.linearAttention)
                try requireBF16("\(prefix).linear_attn.norm.weight",
                                config.linearValueHeadDim)
                try requireRaw("\(prefix).linear_attn.A_log",
                               config.linearNumValueHeads, .fp32)
                try requireRaw("\(prefix).linear_attn.dt_bias",
                               config.linearNumValueHeads, .fp32)
                try requireRaw("\(prefix).linear_attn.conv1d.weight",
                               convCount, .fp16)
            }

            // Shared expert + sigmoid gate + router (both layer types).
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              config.intermediateSize, D,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              config.intermediateSize, D,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              D, config.intermediateSize,
                              quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                              1, D, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.gate.weight",
                              config.numExperts, D, quant.router)
        }

        // Root `hyper_connection_mixer` (no block_inject at the root — the
        // terminal collapse feeds lm_head directly; there is no model.norm).
        let root = "language_model.hyper_connection_mixer."
        try requireBF16("\(root)hc_norm.weight", plane)
        try requireAffine("\(root)input_mix_weight_down.weight",
                          lowrank, plane, quant.attention)
        try requireAffine("\(root)input_mix_weight_up.weight",
                          plane, lowrank, quant.attention)

        // PLE n-gram block on `pleLayerIndexes` only.
        let pleConvElements = try checkedIntMultiply(
            plane, config.pleConvKernelSize, "PLE conv weight")
        for L in config.pleLayerIndexes {
            guard L >= 0, L < config.numLayers else {
                throw ModelError.indexCorrupt(
                    detail: "pleLayerIndexes \(L) is outside 0..<\(config.numLayers)")
            }
            let prefix = "language_model.layers.\(L).ple."
            try requireAffine("\(prefix)key_proj.weight",
                              plane, gatheredWidth, quant.linearAttention)
            try requireAffine("\(prefix)value_proj.weight",
                              D, gatheredWidth, quant.linearAttention)
            try requireBF16("\(prefix)norm_query.weight", plane)
            try requireBF16("\(prefix)norm_key.weight", plane)
            try requireBF16("\(prefix)norm_conv.weight", plane)
            try requireRaw("\(prefix)conv1d.weight", pleConvElements, .fp16)
            try requireInt64("\(prefix)ple_embedding.layer_multipliers",
                             config.ngramSize)
            try requireInt64("\(prefix)ple_embedding.ngram_heads_offsets",
                             pleHeads)
            try requireInt64("\(prefix)ple_embedding.ngram_heads_vocab_sizes",
                             pleHeads)
        }

    }

}

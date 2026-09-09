import Foundation
import Metal
import FinchMoEFormat

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String
    // MARK: Qwen3.6 / Gated DeltaNet (linear-attention) fields.
    /// Discriminator so the runtime can pick the right kernel family without
    /// hard-coding field presence. "gemma4" vs "qwen3_6".
    public let modelFamily: String
    /// Full-attention output gate (Qwen `attn_output_gate`): q_proj width is
    /// doubled and the second half gates the attention output via sigmoid.
    public let attnOutputGate: Bool
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int
    // MARK: Qwen3.8 / Flash-Next (hyper-connection + QSA indexer + PLE
    // n-gram) fields. Zero/nil-ish defaults so the Gemma and Qwen3.6 presets
    // are untouched; only the qwen3_8 preset sets them. A field's exact
    // meaning is documented in docs/QWEN38_PORT.md (math locked against
    // llama.cpp qwen4exp.cpp — no transformers qwen4_exp module exists).
    /// Wide-residual stream count (`hc_count`, 4 on Flash-Next): the block
    /// input is the mean over streams and block outputs are injected back
    /// per-stream. 0 = plain RMSNorm blocks (Gemma / Qwen3.6).
    public let hyperConnectionCount: Int
    /// Mix down/up low rank (`hc_lowrank`, 320).
    public let hyperConnectionLowrank: Int
    /// QSA sparse-attention indexer query heads (`indexer_n_heads`, 4).
    public let indexerNumHeads: Int
    /// QSA indexer key heads (`indexer_kv_heads`, 1).
    public let indexerKVHeads: Int
    /// QSA indexer head width (`indexer_head_dim`, 128).
    public let indexerHeadDim: Int
    /// QSA token budget (`indexer_budget`, 2048): at n_kv above
    /// budget + compress_ratio − 1 the layer attends to the top-k selected
    /// cells only; below that the selection is the full set (dense).
    public let indexerBudget: Int
    /// QSA block size (`indexer_compress_ratio`, 4): keys are pooled into
    /// blocks of this many consecutive tokens before scoring.
    public let indexerCompressRatio: Int
    /// PLE n-gram order (`ngram_size`, 3: bigram + trigram heads).
    public let ngramSize: Int
    /// PLE heads per n-gram order (`heads_per_ngram`, 8); total heads =
    /// (ngramSize − 1) × headsPerNgram.
    public let headsPerNgram: Int
    /// Raw n-gram embedding row width (160 columns on this snapshot); the
    /// disk codec pads rows to a multiple of the quantization group size.
    public let ngramRowDim: Int
    /// N-gram embedding part count (`split_ngram_parts`, 128 shard files).
    public let ngramPartCount: Int
    /// Rows in one n-gram part shard (2,500,012 on this snapshot; the repack
    /// freezes it from the source tensor shape — config only gives the base
    /// vocab 20,000,000, not the final padded row count).
    public let ngramPartRows: Int
    /// 0-based layer indexes hosting the PLE block (config `ple_layer_ids`
    /// is 1-based → [2] becomes [1]).
    public let pleLayerIndexes: [Int]
    /// PLE causal convolution kernel width (`ple_conv_kernel_size`, 4).
    public let pleConvKernelSize: Int

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        modelFamily: String,
        attnOutputGate: Bool,
        linearNumKeyHeads: Int,
        linearNumValueHeads: Int,
        linearKeyHeadDim: Int,
        linearValueHeadDim: Int,
        linearConvKernelDim: Int,
        hyperConnectionCount: Int = 0,
        hyperConnectionLowrank: Int = 0,
        indexerNumHeads: Int = 0,
        indexerKVHeads: Int = 0,
        indexerHeadDim: Int = 0,
        indexerBudget: Int = 0,
        indexerCompressRatio: Int = 0,
        ngramSize: Int = 0,
        headsPerNgram: Int = 0,
        ngramRowDim: Int = 0,
        ngramPartCount: Int = 0,
        ngramPartRows: Int = 0,
        pleLayerIndexes: [Int] = [],
        pleConvKernelSize: Int = 0
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.modelFamily = modelFamily
        self.attnOutputGate = attnOutputGate
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearNumValueHeads = linearNumValueHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
        self.hyperConnectionCount = hyperConnectionCount
        self.hyperConnectionLowrank = hyperConnectionLowrank
        self.indexerNumHeads = indexerNumHeads
        self.indexerKVHeads = indexerKVHeads
        self.indexerHeadDim = indexerHeadDim
        self.indexerBudget = indexerBudget
        self.indexerCompressRatio = indexerCompressRatio
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.ngramRowDim = ngramRowDim
        self.ngramPartCount = ngramPartCount
        self.ngramPartRows = ngramPartRows
        self.pleLayerIndexes = pleLayerIndexes
        self.pleConvKernelSize = pleConvKernelSize
    }

    /// Canonical family strings written to `manifest.json -> arch.modelFamily`
    /// and compared at dispatch sites. All family decisions go through these
    /// constants or `isQwen3_6`/`isQwen3_8`/`isQwenHybrid`; never compare
    /// against a bare literal outside this file.
    /// Values single-source from `FQTurboFormatV1` (shared with the repacker).
    public static let gemma4Family = FQTurboFormatV1.gemma4Family
    public static let qwen3_6Family = FQTurboFormatV1.qwen36Family
    public static let qwen3_8Family = FQTurboFormatV1.qwen38Family

    public var isQwen3_6: Bool { modelFamily == Self.qwen3_6Family }
    public var isQwen3_8: Bool { modelFamily == Self.qwen3_8Family }
    /// Qwen hybrid layers (Gated DeltaNet + full-attention) share the
    /// `linear_attn.*` / doubled-q-proj machinery; a site that is GDN-family
    /// rather than 3.6-specific should test this.
    public var isQwenHybrid: Bool { isQwen3_6 || isQwen3_8 }

    /// Hyper-connection wide-stream width (hc_count × hidden). 0 when the
    /// family has no hyper-connections (Gemma / Qwen3.6 use plain RMSNorm).
    public var hyperConnectionDim: Int { hyperConnectionCount * hiddenSize }

    /// Total PLE n-gram embedding rows (all 128 part shards).
    public var ngramTotalRows: Int { ngramPartCount * ngramPartRows }

    /// PLE attention heads: (ngram_size − 1) orders × heads per ngram.
    public var pleHeadCount: Int { ngramSize > 1 ? (ngramSize - 1) * headsPerNgram : 0 }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    /// The built-in preset for a manifest-declared model family. Loaders use
    /// this (via `ManifestReader.detectPreset`) instead of hardcoding Gemma.
    public static func preset(forModelFamily family: String?) -> ArchConfig {
        switch family {
        case Self.qwen3_6Family: return .qwen3_6_35B_A3B
        case Self.qwen3_8Family: return .qwen3_8_flashNext_125B
        default:                 return .gemma4_26B_A4B
        }
    }

    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        modelFamily: "gemma4",
        attnOutputGate: false,
        linearNumKeyHeads: 0,
        linearNumValueHeads: 0,
        linearKeyHeadDim: 0,
        linearValueHeadDim: 0,
        linearConvKernelDim: 0
    )

    /// Qwen3.6-35B-A3B text model: 40 hybrid layers (30 Gated-DeltaNet
    /// linear-attention + 10 full attention, every 4th layer full), 256-expert
    /// MoE with a shared expert in every layer, no tied embeddings, no logit
    /// softcap. Full attention has an output gate and partial rotary (0.25).
    public static let qwen3_6_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,          // shared expert FFN width
        moeIntermediateSize: 512,       // per-expert FFN width
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 128,                   // nominal; GDN layers use linearKey/ValueHeadDim
        fullHeadDim: 256,               // full-attention head_dim
        vocabSize: 248320,
        slidingWindow: 0,               // no sliding window (GDN / full-attention only)
        finalLogitSoftcap: 0.0,         // no softcap
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwenLayerMask(numLayers: 40),
        hiddenActivation: "silu",
        modelFamily: "qwen3_6",
        attnOutputGate: true,
        linearNumKeyHeads: 16,
        linearNumValueHeads: 32,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4
    )

    /// Qwen3.8-Flash-Next text model: 48 hybrid layers (36 Gated-DeltaNet
    /// linear-attention + 12 full attention every 4th layer), where the full
    /// layers are QSA sparse attention with a 4-head × 128 indexer
    /// (budget 2048 tokens / blocks of 4). Hyper-connections (4 streams,
    /// low rank 320) replace every RMSNorm — there is no `model.norm`; a
    /// final shared `hyper_connection_mixer` collapses the wide stream to the
    /// lm_head input. Layer 1 (0-based) additionally hosts the PLE n-gram
    /// block (16 heads over a 20M-base × 16 subrange embedding split into
    /// 128 part files of 2,500,012 × 160). 512 experts top-10 + shared,
    /// GDN 48 value heads, untied lm_head. All values frozen from
    /// `Qwen3.8-Flash-Next-bf16/config.json` + shard-header census
    /// (2026-09-08); math authority is llama.cpp `qwen4exp.cpp`.
    public static let qwen3_8_flashNext_125B = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 640,          // shared expert FFN width
        moeIntermediateSize: 640,       // per-expert FFN width
        numHeads: 24,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 128,                   // nominal; GDN layers use linearKey/ValueHeadDim
        fullHeadDim: 256,               // full-attention (QSA) head_dim
        vocabSize: 248320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 48,
        numExperts: 512,
        topKExperts: 10,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwenLayerMask(numLayers: 48),
        hiddenActivation: "silu",
        modelFamily: "qwen3_8",
        attnOutputGate: true,           // q_proj carries a gate half per head
        linearNumKeyHeads: 16,
        linearNumValueHeads: 48,
        linearKeyHeadDim: 128,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4,
        hyperConnectionCount: 4,
        hyperConnectionLowrank: 320,
        indexerNumHeads: 4,
        indexerKVHeads: 1,
        indexerHeadDim: 128,
        indexerBudget: 2048,
        indexerCompressRatio: 4,
        ngramSize: 3,
        headsPerNgram: 8,
        ngramRowDim: 160,
        ngramPartCount: 128,
        ngramPartRows: 2_500_012,
        pleLayerIndexes: [1],
        pleConvKernelSize: 4
    )

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Qwen hybrid layout: full attention at 0-indexed 3,7,11,... (every
    /// 4th). Shared by 3.6 (40 layers) and 3.8 Flash-Next (48 layers).
    private static func qwenLayerMask(numLayers: Int) -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: numLayers)
        for i in stride(from: 3, to: numLayers, by: 4) { mask[i] = 1 }
        return mask
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAFQTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.fqturbo directory at \(p) is missing manifest.json"
        case .notAFQTurboDirectory:
            return "manifest.json magic does not equal \"FQTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.fqturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}

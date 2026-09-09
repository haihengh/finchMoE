import Foundation
import FinchMoEFormat

/// Architecture facts mirrored into `manifest.json -> arch`. Cross-checked by
/// the runtime loader at startup. Carries enough to describe either the Gemma
/// 4 hybrid-SWA model or the Qwen3.6 Gated-DeltaNet (linear-attention) model;
/// GDN-only fields are nil/0 when the source model has none.
struct ArchInfo: Sendable, Equatable {
    let hiddenSize: Int
    let intermediateSize: Int          // shared expert FFN
    let moeIntermediateSize: Int       // per-expert FFN
    let numHeads: Int
    let numKVHeads: Int
    let numFullKVHeads: Int
    let headDim: Int
    let fullHeadDim: Int
    let vocabSize: Int
    let slidingWindow: Int
    let finalLogitSoftcap: Double
    let ropeTheta: Double
    let fullRopeTheta: Double
    let partialRotaryFactor: Double
    let numLayers: Int
    let numExperts: Int
    let topKExperts: Int
    let tieWordEmbeddings: Bool
    let attentionKEqV: Bool
    /// 1 if `full_attention`, 0 if the other family (Gemma sliding-window or
    /// Qwen linear-attention). Indexed by layer.
    let fullAttentionLayerMask: [UInt8]
    let hiddenActivation: String
    // Qwen3.6 / Gated DeltaNet (linear-attention) fields.
    let modelFamily: String?
    let attnOutputGate: Bool?
    let linearNumKeyHeads: Int?
    let linearNumValueHeads: Int?
    let linearKeyHeadDim: Int?
    let linearValueHeadDim: Int?
    let linearConvKernelDim: Int?
    // Qwen3.8-Flash-Next fields (hyper-connection / QSA indexer / PLE n-gram).
    // Defaulted so direct constructions (tests, fixture builders) compile
    // unchanged; `load` fills them from the config when family == qwen3_8.
    let hyperConnectionCount: Int?
    let hyperConnectionLowrank: Int?
    let indexerNumHeads: Int?
    let indexerKVHeads: Int?
    let indexerHeadDim: Int?
    let indexerBudget: Int?
    let indexerCompressRatio: Int?
    let ngramSize: Int?
    let headsPerNgram: Int?
    let ngramRowDim: Int?
    let ngramPartCount: Int?
    let ngramPartRows: Int?
    let pleLayerIndexes: [Int]?
    let pleConvKernelSize: Int?

    init(hiddenSize: Int, intermediateSize: Int, moeIntermediateSize: Int,
         numHeads: Int, numKVHeads: Int, numFullKVHeads: Int,
         headDim: Int, fullHeadDim: Int, vocabSize: Int,
         slidingWindow: Int, finalLogitSoftcap: Double,
         ropeTheta: Double, fullRopeTheta: Double,
         partialRotaryFactor: Double, numLayers: Int, numExperts: Int,
         topKExperts: Int, tieWordEmbeddings: Bool, attentionKEqV: Bool,
         fullAttentionLayerMask: [UInt8], hiddenActivation: String,
         modelFamily: String? = nil, attnOutputGate: Bool? = nil,
         linearNumKeyHeads: Int? = nil, linearNumValueHeads: Int? = nil,
         linearKeyHeadDim: Int? = nil, linearValueHeadDim: Int? = nil,
         linearConvKernelDim: Int? = nil,
         // Qwen3.8-Flash-Next fields, defaulted so direct constructions
         // compile unchanged; `load` fills them for family == qwen3_8.
         hyperConnectionCount: Int? = nil,
         hyperConnectionLowrank: Int? = nil,
         indexerNumHeads: Int? = nil,
         indexerKVHeads: Int? = nil,
         indexerHeadDim: Int? = nil,
         indexerBudget: Int? = nil,
         indexerCompressRatio: Int? = nil,
         ngramSize: Int? = nil,
         headsPerNgram: Int? = nil,
         ngramRowDim: Int? = nil,
         ngramPartCount: Int? = nil,
         ngramPartRows: Int? = nil,
         pleLayerIndexes: [Int]? = nil,
         pleConvKernelSize: Int? = nil) {
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

    /// Model discriminator derived from the config. Used by the runtime to
    /// pick the right kernel family and by the manifest validator.
    /// Family strings single-source from `FQTurboFormatV1` so the repacker
    /// can never write a family the runtime does not recognize.
    static let qwen36Family = FQTurboFormatV1.qwen36Family
    static let qwen38Family = FQTurboFormatV1.qwen38Family
    static let gemma4Family = FQTurboFormatV1.gemma4Family

    static func family(from tc: [String: Any]) -> String {
        let mt = (tc["model_type"] as? String) ?? ""
        if mt.contains("qwen3_5_moe") || mt.contains("qwen3.6") || mt.contains("qwen3_6") {
            return qwen36Family
        }
        // Qwen3.8-Flash-Next (text config model_type qwen4_exp_text; the root
        // config of the hybrid checkpoint says qwen4_exp). Must come before
        // the GDN-field heuristic below — qwen4_exp_text carries
        // linear_num_key_heads too and would otherwise misclassify as 3.6.
        if mt.contains("qwen4_exp") {
            return qwen38Family
        }
        if mt.contains("gemma") { return gemma4Family }
        // Heuristic: presence of GDN fields means a linear-attention hybrid.
        if tc["linear_num_key_heads"] != nil || tc["linear_attn"] != nil {
            return qwen36Family
        }
        return gemma4Family
    }

    static func load(configPath: String) throws -> ArchInfo {
        let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw RepackError.configJsonInvalid(path: configPath, detail: "not a JSON object")
        }
        // Qwen wraps the text config under "text_config"; Gemma also does.
        let tc = (root["text_config"] as? [String: Any]) ?? root
        let family = family(from: tc)

        func optInt(_ k: String) -> Int? {
            (tc[k] as? Int) ?? (tc[k] as? NSNumber)?.intValue
        }
        func optDouble(_ k: String) -> Double? {
            (tc[k] as? Double) ?? (tc[k] as? NSNumber)?.doubleValue
        }
        func reqInt(_ k: String) throws -> Int {
            guard let n = optInt(k) else {
                throw RepackError.configJsonInvalid(path: configPath, detail: "missing \(k)")
            }
            return n
        }
        func optBool(_ k: String) -> Bool? { (tc[k] as? Bool) }

        // Layer mask: Gemma "full_attention"/"sliding_attention"; Qwen
        // "full_attention"/"linear_attention". Anything not explicitly
        // "full_attention" is the non-full family (mask 0).
        let layerTypes = (tc["layer_types"] as? [String]) ?? []
        let mask = layerTypes.map { UInt8($0 == "full_attention" ? 1 : 0) }

        // RoPE: Gemma nests full/sliding attention; Qwen has a flat
        // rope_parameters with a single theta + partial_rotary_factor.
        let rope = (tc["rope_parameters"] as? [String: Any]) ?? [:]
        let ropeFull = (rope["full_attention"] as? [String: Any]) ?? [:]
        let ropeSWA  = (rope["sliding_attention"] as? [String: Any]) ?? [:]
        let prf = (ropeFull["partial_rotary_factor"] as? NSNumber)?.doubleValue
            ?? (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let fullTheta = (ropeFull["rope_theta"] as? NSNumber)?.doubleValue
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let swaTheta = (ropeSWA["rope_theta"] as? NSNumber)?.doubleValue
            ?? (rope["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0

        let kEqV = optBool("attention_k_eq_v") ?? false
        let tie = optBool("tie_word_embeddings") ?? false
        let act = (tc["hidden_activation"] as? String) ?? (tc["hidden_act"] as? String) ?? "gelu_pytorch_tanh"

        // Qwen-specific optional fields (nil for Gemma).
        let linearKeyHeads = optInt("linear_num_key_heads")
        let linearValHeads = optInt("linear_num_value_heads")
        let linearKeyDim = optInt("linear_key_head_dim")
        let linearValDim = optInt("linear_value_head_dim")
        let linearConvK = optInt("linear_conv_kernel_dim")
        // Qwen3.6 declares the doubled-q gate with attn_output_gate; Qwen3.8
        // expresses it as output_gate_type ("sigmoid" / "silu") instead — both
        // mean the full-attention q_proj carries a per-head gate half.
        let attnGate = optBool("attn_output_gate")
            ?? ((tc["output_gate_type"] as? String) != nil ? true : nil)

        // Qwen3.8-Flash-Next fields; nil for every other family. The n-gram
        // table geometry (row width 160, 2_500_012 rows per shard) does not
        // exist in the config — it is frozen from the BF16 snapshot tensor
        // shape (census of ngram_embedding.shard_*). M5's repack re-verifies
        // against the live source shape before writing.
        let is38 = family == Self.qwen38Family
        let hcCount = optInt("hc_count")
        let hcLowrank = optInt("hc_lowrank")
        let indexerHeads = optInt("indexer_n_heads")
        let indexerKVHeads = optInt("indexer_kv_heads")
        let indexerHeadDim = optInt("indexer_head_dim")
        let indexerBudget = optInt("indexer_budget")
        let indexerCompress = optInt("indexer_compress_ratio")
        let ngramSize = optInt("ngram_size")
        let headsPerNgram = optInt("heads_per_ngram")
        let ngramParts = optInt("split_ngram_parts")
        let pleConvK = optInt("ple_conv_kernel_size")
        // ple_layer_ids in the HF config are 1-based (the snapshot census
        // finds the PLE tensors under layers.1 when the config says [2]);
        // the engine indexes layers from 0.
        let ple1Based = (tc["ple_layer_ids"] as? [Int]) ?? []
        let pleLayers = ple1Based.map { $0 - 1 }

        return ArchInfo(
            hiddenSize: try reqInt("hidden_size"),
            intermediateSize: try (optInt("intermediate_size")
                ?? reqInt("shared_expert_intermediate_size")),
            moeIntermediateSize: try reqInt("moe_intermediate_size"),
            numHeads: try reqInt("num_attention_heads"),
            numKVHeads: try reqInt("num_key_value_heads"),
            numFullKVHeads: try (optInt("num_global_key_value_heads") ?? reqInt("num_key_value_heads")),
            // headDim = the non-full family's head width: GDN key head dim for
            // Qwen, sliding-window head dim for Gemma. fullHeadDim = full
            // attention head dim (Gemma "global_head_dim").
            headDim: optInt("linear_key_head_dim") ?? optInt("head_dim") ?? 128,
            fullHeadDim: optInt("global_head_dim") ?? optInt("head_dim") ?? 256,
            vocabSize: try reqInt("vocab_size"),
            slidingWindow: optInt("sliding_window") ?? 0,
            finalLogitSoftcap: optDouble("final_logit_softcapping") ?? 0.0,
            ropeTheta: swaTheta,
            fullRopeTheta: fullTheta,
            partialRotaryFactor: prf,
            numLayers: try reqInt("num_hidden_layers"),
            numExperts: try reqInt("num_experts"),
            topKExperts: try (optInt("top_k_experts") ?? reqInt("num_experts_per_tok")),
            tieWordEmbeddings: tie,
            attentionKEqV: kEqV,
            fullAttentionLayerMask: mask,
            hiddenActivation: act,
            modelFamily: family,
            attnOutputGate: attnGate,
            linearNumKeyHeads: linearKeyHeads,
            linearNumValueHeads: linearValHeads,
            linearKeyHeadDim: linearKeyDim,
            linearValueHeadDim: linearValDim,
            linearConvKernelDim: linearConvK,
            hyperConnectionCount: is38 ? hcCount : nil,
            hyperConnectionLowrank: is38 ? hcLowrank : nil,
            indexerNumHeads: is38 ? indexerHeads : nil,
            indexerKVHeads: is38 ? indexerKVHeads : nil,
            indexerHeadDim: is38 ? indexerHeadDim : nil,
            indexerBudget: is38 ? indexerBudget : nil,
            indexerCompressRatio: is38 ? indexerCompress : nil,
            ngramSize: is38 ? ngramSize : nil,
            headsPerNgram: is38 ? headsPerNgram : nil,
            ngramRowDim: is38 ? 160 : nil,
            ngramPartCount: is38 ? ngramParts : nil,
            ngramPartRows: is38 ? 2_500_012 : nil,
            pleLayerIndexes: is38 ? (pleLayers.isEmpty ? [1] : pleLayers) : nil,
            pleConvKernelSize: is38 ? pleConvK : nil)
    }
}

import Foundation

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

    /// Model discriminator derived from the config. Used by the runtime to
    /// pick the right kernel family and by the manifest validator.
    static func family(from tc: [String: Any]) -> String {
        let mt = (tc["model_type"] as? String) ?? ""
        if mt.contains("qwen3_5_moe") || mt.contains("qwen3.6") || mt.contains("qwen3_6") {
            return "qwen3_6"
        }
        if mt.contains("gemma") { return "gemma4" }
        // Heuristic: presence of GDN fields means a linear-attention hybrid.
        if tc["linear_num_key_heads"] != nil || tc["linear_attn"] != nil {
            return "qwen3_6"
        }
        return "gemma4"
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
        let attnGate = optBool("attn_output_gate")

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
            linearConvKernelDim: linearConvK)
    }
}

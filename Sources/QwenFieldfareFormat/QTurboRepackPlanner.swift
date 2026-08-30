import Foundation

/// Classifies Qwen3 tensor names into "resident" (kept in model_weights.bin)
/// vs "expert" (streamed from packed per-layer files), and provides the shared
/// expert-blob layout for the repacker and runtime.
public enum QTurboRepackPlanner {

    /// Classification of a tensor name.
    public enum Placement: Equatable {
        case resident
        /// Routed expert sub-tensor: (layer, expert, kind) where kind is e.g.
        /// "gate_proj.weight" or "down_proj.scales".
        case expert(layer: Int, expert: Int, kind: String)
        /// Not part of the model we care about (e.g. rotary inv_freq buffers).
        case ignored
    }

    /// The ordered kinds that make up one expert blob. Order defines on-disk
    /// layout and must match the runtime's expectations.
    public static let expertKinds: [String] = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
        "down_proj.weight", "down_proj.scales", "down_proj.biases",
    ]

    /// Regex-free classifier for a Qwen3 (MLX 4-bit) tensor name.
    public static func classify(_ name: String) -> Placement {
        // Routed experts: model.layers.{L}.mlp.experts.{E}.{proj}.{suffix}
        if let e = parseExpert(name) {
            return .expert(layer: e.layer, expert: e.expert, kind: e.kind)
        }

        // Everything else that belongs to the model stays resident.
        // Resident: embed_tokens, lm_head, norms, attn projections (+scales/biases),
        // shared_expert (+scales/biases), router gate, final norm.
        if name == "model.embed_tokens.weight" { return .resident }
        if name == "lm_head.weight" { return .resident }
        if name == "model.norm.weight" { return .resident }

        if name.hasPrefix("model.layers.") {
            // Ignore rotary/rope inverse-frequency style buffers if present.
            if name.contains("rotary_emb.inv_freq") { return .ignored }
            // Everything else under a layer that is NOT a routed expert is resident:
            // input_layernorm, post_attention_layernorm,
            // self_attn.{q,k,v,o}_proj.{weight,scales,biases},
            // mlp.shared_expert.{gate,up,down}_proj.{weight,scales,biases},
            // mlp.shared_expert_gate.weight (if any), mlp.gate.weight (router).
            return .resident
        }

        return .ignored
    }

    /// Parses a routed-expert tensor name; returns nil if not a routed expert.
    public static func parseExpert(_ name: String) -> (layer: Int, expert: Int, kind: String)? {
        // Expected: model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.{weight|scales|biases}
        let parts = name.split(separator: ".").map(String.init)
        // ["model","layers","L","mlp","experts","E","gate_proj","weight"]
        guard parts.count == 8,
              parts[0] == "model",
              parts[1] == "layers",
              parts[3] == "mlp",
              parts[4] == "experts",
              let layer = Int(parts[2]),
              let expert = Int(parts[5])
        else { return nil }

        let proj = parts[6]  // gate_proj | up_proj | down_proj
        let suffix = parts[7] // weight | scales | biases
        guard proj == "gate_proj" || proj == "up_proj" || proj == "down_proj" else { return nil }
        guard suffix == "weight" || suffix == "scales" || suffix == "biases" else { return nil }

        return (layer, expert, "\(proj).\(suffix)")
    }

    /// Computes the shared expert-blob layout for the given config. All experts
    /// across all layers share identical shapes, so a single layout suffices.
    ///
    /// MLX affine 4-bit layout (group_size = 64, 4 bits):
    /// - weight: int4 packed as u32 (8 nibbles each). For a [rows, cols] logical
    ///   matrix, packed shape is [rows, cols/8] u32 → bytes = rows*cols/2.
    /// - scales/biases: fp16, shape [rows, cols/group_size].
    public static func expertLayout(config: QTurboModelConfig) -> QTurboExpertLayout {
        let hidden = config.hiddenSize                 // 2048
        let moe = config.moeIntermediateSize           // 768
        let group = config.quantGroupSize              // 64

        // gate_proj / up_proj: logical [moe, hidden] = [768, 2048]
        // down_proj:           logical [hidden, moe] = [2048, 768]
        func weightBytes(rows: Int, cols: Int) -> Int { rows * cols / 2 }         // 4-bit
        func scaleBiasBytes(rows: Int, cols: Int) -> Int { rows * (cols / group) * 2 } // fp16

        var subs: [QTurboExpertSubTensor] = []
        var offset = 0

        func addProj(_ proj: String, rows: Int, cols: Int) {
            let wLen = weightBytes(rows: rows, cols: cols)
            subs.append(QTurboExpertSubTensor(
                kind: "\(proj).weight", dtype: .int4Packed,
                shape: [rows, cols / 8], offset: offset, length: wLen))
            offset += wLen

            let sLen = scaleBiasBytes(rows: rows, cols: cols)
            subs.append(QTurboExpertSubTensor(
                kind: "\(proj).scales", dtype: .fp16,
                shape: [rows, cols / group], offset: offset, length: sLen))
            offset += sLen

            let bLen = scaleBiasBytes(rows: rows, cols: cols)
            subs.append(QTurboExpertSubTensor(
                kind: "\(proj).biases", dtype: .fp16,
                shape: [rows, cols / group], offset: offset, length: bLen))
            offset += bLen
        }

        addProj("gate_proj", rows: moe, cols: hidden)
        addProj("up_proj",   rows: moe, cols: hidden)
        addProj("down_proj", rows: hidden, cols: moe)

        let blobSize = offset
        let stride = QTurboFormatV1.align(blobSize, to: QTurboFormatV1.expertBlobAlignment)
        return QTurboExpertLayout(subTensors: subs, blobSize: blobSize, alignedStride: stride)
    }
}

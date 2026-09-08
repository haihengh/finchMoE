import Foundation
import FinchMoEFormat

// MARK: - Qwen repack plan types
//
// The Qwen 3.6 bf16 checkpoint has no pre-quantized tensors and no
// `config.json -> quantization` slot, so the Gemma byte-copy planner cannot
// describe it. This planner instead assigns every source tensor a WRITE
// TRANSFORM (bf16 rows → int4/int8 affine, norm (1+w) baking, bf16 → fp16 /
// fp32 raw conversions) and lays the quantized payloads out in the resident
// file. The expert layers reuse `LayerFilePlan`/`PerExpertTensorSlice` with
// the fused `gate_up_proj` source split into the separate gate/up role
// slices the engine's streamed-expert kernels read.

/// What the writer does with one resident entry's source bytes.
enum QwenWriteTransform: Sendable, Equatable {
    /// bf16 `[rows, cols]` → int4 affine: packed U32 weights + BF16
    /// scales/biases (group 64).
    case int4Affine(rows: Int, cols: Int)
    /// bf16 `[rows, cols]` → int8 affine (the router).
    case int8Affine(rows: Int, cols: Int)
    /// bf16 `[n]` → bf16 with the Qwen `(1 + w)` RMSNorm form baked in, so
    /// the runtime kernels stay weight-direct.
    case normOnePlusW(Int)
    /// bf16 `[n]` → raw bf16. `linear_attn.norm.weight` is a
    /// Qwen3_5MoeRMSNormGated — torch multiplies its weight RAW (init ones),
    /// so the (1 + w) bake would double every GDN gated norm.
    case normRawBf16(Int)
    /// bf16 `[n]` → raw fp16 (linear_attn.conv1d.weight).
    case bf16ToFp16(Int)
    /// bf16 `[n]` → raw fp32 (linear_attn.A_log / dt_bias).
    case bf16ToFp32(Int)

    /// Quantization bits of the emitted weights (nil for raw entries).
    var weightBits: Int? {
        switch self {
        case .int4Affine: 4
        case .int8Affine: 8
        default: nil
        }
    }

    /// Size of the emitted weight payload.
    func weightBytes(rows: Int, cols: Int) -> UInt64 {
        switch self {
        case .int4Affine: UInt64(rows) * UInt64(cols) / 2
        case .int8Affine: UInt64(rows) * UInt64(cols)
        case .normOnePlusW(let n), .normRawBf16(let n), .bf16ToFp16(let n): UInt64(n) * 2
        case .bf16ToFp32(let n): UInt64(n) * 4
        }
    }

    /// Size of the emitted scales (== biases) payload (0 for raw entries).
    func scaleBytes(rows: Int, cols: Int) -> UInt64 {
        guard weightBits != nil else { return 0 }
        return UInt64(rows) * UInt64(cols / FinchQuantization.groupSize) * 2
    }
}

/// One resident entry planned for the Qwen install. Mirrors the fields the
/// resident-index encoder needs, plus the transform the writer applies.
struct QwenResidentEntry: Sendable {
    let name: String
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    let dtype: UInt8
    /// Logical shape after dequant (max rank 4; trailing zeros).
    let logicalShape4: [UInt32]
    let fileOffset: UInt64
    let sizeBytes: UInt64
    let scaleOffset: UInt64
    let scaleSize: UInt64
    let biasOffset: UInt64
    let biasSize: UInt64
    let transform: QwenWriteTransform
    let source: SourceTensor
}

struct QwenResidentFilePlan: Sendable {
    let path: String
    let entries: [QwenResidentEntry]
    let stringTable: [UInt8]
    let stringTableOffsets: [UInt32]
    let indexSize: UInt64
    let residentSize: UInt64
    var totalSize: UInt64 { indexSize + residentSize }
}

struct QwenRepackPlan: Sendable {
    let arch: ArchInfo
    let resident: QwenResidentFilePlan
    let layers: [LayerFilePlan]
    let excludedTensorNames: [String]
}

// MARK: - Planner

enum QwenRepackPlanner {

    /// Remaps a qwen3_5_moe checkpoint name to the resident entry name the
    /// engine resolves. The checkpoint nests the text model as
    /// `model.language_model.*`; the engine (shared with Gemma) expects
    /// `language_model.model.*`. `lm_head.weight` passes through.
    static func residentName(for source: String) -> String? {
        if source == "lm_head.weight" { return source }
        guard source.hasPrefix("model.language_model.") else { return nil }
        return "language_model.model." + source.dropFirst("model.language_model.".count)
    }

    private static func layerIndex(in name: String) -> Int? {
        guard let r = name.range(of: ".layers.") else { return nil }
        let tail = name[r.upperBound...]
        guard let dot = tail.firstIndex(of: ".") else { return nil }
        return Int(tail[tail.startIndex..<dot])
    }

    /// Builds the plan from a local bf16 snapshot (index + shard headers).
    static func plan(meta: QwenLocalSnapshot.SourceMetadata,
                     arch: ArchInfo,
                     shardHeaders: [Safetensors.Header],
                     outputDir: String) throws -> QwenRepackPlan {
        var registry: [String: SourceTensor] = [:]
        registry.reserveCapacity(meta.weightMap.count)
        for h in shardHeaders {
            for t in h.tensors { registry[t.name] = t }
        }

        let fullMask = arch.fullAttentionLayerMask
        var excluded: [String] = []
        var residentSources: [String] = []     // checkpoint names, sorted
        var routedSources: [Int: (gateUp: String, down: String)] = [:]

        for (name, _) in registry {
            guard let mapped = residentName(for: name) else {
                // Vision tower / MTP / other non-text tensors: dropped for
                // the text-only install.
                if !name.hasPrefix("model.language_model.") && name != "lm_head.weight" {
                    excluded.append(name)
                }
                continue
            }
            guard let layer = layerIndex(in: name) else {
                residentSources.append(name)     // embed_tokens, norm, lm_head
                continue
            }
            if name.hasSuffix(".mlp.experts.gate_up_proj") {
                routedSources[layer, default: (gateUp: "", down: "")].gateUp = name
            } else if name.hasSuffix(".mlp.experts.down_proj") {
                routedSources[layer, default: (gateUp: "", down: "")].down = name
            } else {
                residentSources.append(name)
            }
        }
        residentSources.sort(by: qwenResidentOrdering())
        excluded.sort()

        if !routedSources.isEmpty {
            // Every layer has routed experts in the real Qwen checkpoint;
            // synthetic snapshots may omit them entirely.
            for layer in 0..<arch.numLayers {
                guard routedSources[layer] != nil else {
                    throw RepackError.configurationInvalid(
                        detail: "layer \(layer) routed-expert bundle incomplete")
                }
            }
        }

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            sourceNames: residentSources,
                                            registry: registry)

        let layersDir = (outputDir as NSString).appendingPathComponent("packed_experts")
        var layerPlans: [LayerFilePlan] = []
        layerPlans.reserveCapacity(arch.numLayers)
        for layer in 0..<arch.numLayers {
            let path = (layersDir as NSString)
                .appendingPathComponent("layer_\(String(format: "%02d", layer)).bin")
            guard let pair = routedSources[layer] else {
                layerPlans.append(LayerFilePlan(layerIndex: layer, path: path,
                                                expertsPerLayer: 0,
                                                expertStride: 0, subTensors: []))
                continue
            }
            layerPlans.append(try planLayerFile(path: path, layer: layer,
                                                gateUpName: pair.gateUp,
                                                downName: pair.down,
                                                registry: registry, arch: arch))
        }

        return QwenRepackPlan(arch: arch,
                              resident: resident,
                              layers: layerPlans,
                              excludedTensorNames: excluded)
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         sourceNames: [String],
                                         registry: [String: SourceTensor]) throws
                                        -> QwenResidentFilePlan {
        // Pass 1: resolve transforms + string table (no offsets yet).
        struct Proto {
            let source: SourceTensor
            let name: String
            let transform: QwenWriteTransform
            let wSize: UInt64
            let sSize: UInt64
        }
        var protos: [Proto] = []
        protos.reserveCapacity(sourceNames.count)
        var stringTable: [UInt8] = []
        var offsets: [UInt32] = []
        offsets.reserveCapacity(sourceNames.count)
        for source in sourceNames {
            guard let tensor = registry[source] else {
                throw RepackError.missingTensor(name: source)
            }
            guard let name = residentName(for: source) else {
                throw RepackError.unknownTensorPrefix(name: source)
            }
            let transform = try transform(for: source, tensor: tensor)
            let rows: Int
            let cols: Int
            switch transform {
            case .int4Affine(let r, let c), .int8Affine(let r, let c):
                rows = r; cols = c
            case .normOnePlusW(let n), .normRawBf16(let n),
                 .bf16ToFp16(let n), .bf16ToFp32(let n):
                rows = n; cols = 0
            }
            let wSize = transform.weightBytes(rows: rows, cols: cols)
            let sSize = transform.scaleBytes(rows: rows, cols: cols)
            offsets.append(UInt32(stringTable.count))
            stringTable.append(contentsOf: name.utf8)
            protos.append(Proto(source: tensor, name: name, transform: transform,
                                wSize: wSize, sSize: sSize))
        }

        // The index region: header + entries + string table, page-padded.
        let rawIdx = UInt64(FinchBinary.indexHeaderBytes
            + protos.count * FinchBinary.indexEntryBytes
            + stringTable.count)
        let indexSize = roundUpToPage(rawIdx)

        // Pass 2: payload layout, starting after the index region.
        var cursor = indexSize
        var entries: [QwenResidentEntry] = []
        entries.reserveCapacity(protos.count)
        for proto in protos {
            let wOff = cursor
            let sOff = wOff + proto.wSize
            let bOff = sOff + proto.sSize
            cursor = bOff + proto.sSize
            entries.append(QwenResidentEntry(
                name: proto.name,
                dtype: dtypeByte(for: proto.transform),
                logicalShape4: padTo4(logicalShape(for: proto.transform, source: proto.source)),
                fileOffset: wOff, sizeBytes: proto.wSize,
                scaleOffset: proto.sSize > 0 ? sOff : 0,
                scaleSize: proto.sSize,
                biasOffset: proto.sSize > 0 ? bOff : 0,
                biasSize: proto.sSize,
                transform: proto.transform,
                source: proto.source))
        }
        let residentSize = cursor - indexSize

        return QwenResidentFilePlan(path: path,
                                    entries: entries,
                                    stringTable: stringTable,
                                    stringTableOffsets: offsets,
                                    indexSize: indexSize,
                                    residentSize: residentSize)
    }

    /// The transform for one checkpoint tensor, validated against its shape.
    private static func transform(for source: String,
                                  tensor: SourceTensor) throws -> QwenWriteTransform {
        guard tensor.dtype == .bf16 else {
            throw RepackError.dtypeMismatch(name: source,
                detail: "expected BF16 source, got \(tensor.dtype)")
        }
        let shape = tensor.shape
        let isFull = source.contains(".self_attn.")
        if source.hasSuffix(".mlp.gate.weight") {
            guard shape.count == 2, shape[0] > 0, shape[1] % 64 == 0 else {
                throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
            }
            return .int8Affine(rows: Int(shape[0]), cols: Int(shape[1]))
        }
        if source.contains(".experts.") {
            throw RepackError.configurationInvalid(
                detail: "expert tensor \(source) must not reach resident planning")
        }
        if source.hasSuffix(".linear_attn.A_log") || source.hasSuffix(".linear_attn.dt_bias") {
            guard shape.count == 1 else {
                throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
            }
            return .bf16ToFp32(Int(shape[0]))
        }
        if source.hasSuffix(".linear_attn.conv1d.weight") {
            // Checkpoint stores [C, 1, kernel]; the writer emits the squeezed
            // [C * kernel] rows as raw FP16 (bf16 → fp16 conversion at emit).
            guard shape.count == 3, shape[1] == 1 else {
                throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
            }
            return .bf16ToFp16(Int(shape[0] * shape[2]))
        }
        if isFull && (source.hasSuffix(".q_norm.weight") || source.hasSuffix(".k_norm.weight")) {
            return .normOnePlusW(Int(shape[0]))
        }
        if source.hasSuffix(".linear_attn.norm.weight") {
            // Qwen3_5MoeRMSNormGated (inside the Gated DeltaNet) multiplies
            // its weight raw — NOT (1 + w). Must come before the generic
            // `.norm.weight` bake below.
            guard shape.count == 1 else {
                throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
            }
            return .normRawBf16(Int(shape[0]))
        }
        if source.contains(".linear_attn.") {
            // The GatedDeltaNet's five projections (in_proj_qkv / in_proj_z /
            // in_proj_a / in_proj_b / out_proj) are stored at 8-bit: int4
            // dequant noise on these enters the recurrent state and grows
            // ~16x into the deep-layer hidden rows (torch probe isoMax up to
            // 27 at L10), drowning the final logits. Full-attention q/k/v/o
            // and the MoE stay int4 — their noise does not amplify.
            guard source.hasSuffix(".weight"), shape.count == 2,
                  shape[1] % 64 == 0 else {
                throw RepackError.configurationInvalid(
                    detail: "unclassifiable Qwen linear_attn tensor \(source) shape \(shape)")
            }
            return .int8Affine(rows: Int(shape[0]), cols: Int(shape[1]))
        }
        if source.hasSuffix(".norm.weight") || source.hasSuffix(".input_layernorm.weight")
            || source.hasSuffix(".post_attention_layernorm.weight") {
            return .normOnePlusW(Int(shape[0]))
        }
        // Everything else with `.weight` is a 2-D projection → int4 affine.
        guard source.hasSuffix(".weight"), shape.count == 2,
              shape[1] % 64 == 0 else {
            throw RepackError.configurationInvalid(
                detail: "unclassifiable Qwen tensor \(source) shape \(shape)")
        }
        return .int4Affine(rows: Int(shape[0]), cols: Int(shape[1]))
    }

    private static func dtypeByte(for transform: QwenWriteTransform) -> UInt8 {
        switch transform {
        case .int4Affine, .int8Affine: return FinchFormatV1.DType.u32.rawValue
        case .normOnePlusW, .normRawBf16: return FinchFormatV1.DType.bf16.rawValue
        case .bf16ToFp16:              return FinchFormatV1.DType.fp16.rawValue
        case .bf16ToFp32:              return FinchFormatV1.DType.fp32.rawValue
        }
    }

    private static func logicalShape(for transform: QwenWriteTransform,
                                     source: SourceTensor) -> [UInt64] {
        switch transform {
        case .int4Affine(let r, let c), .int8Affine(let r, let c):
            return [UInt64(r), UInt64(c)]
        case .normOnePlusW(let n), .normRawBf16(let n),
             .bf16ToFp16(let n), .bf16ToFp32(let n):
            return [UInt64(n)]
        }
    }

    // MARK: - Layer planning
    //
    // gate_up_proj [E, 2F, D] splits into gate = rows [0, F) and up = rows
    // [F, 2F) of the fused feature dim (transformers `qwen3_5_moe`
    // `modeling_*.py:729,751`); down_proj [E, D, F] is used directly. Each
    // role is int4-quantized per expert into the standard gate/up/down +
    // scales/biases blob layout the engine's streamed kernels read.

    private static func planLayerFile(path: String, layer: Int,
                                      gateUpName: String, downName: String,
                                      registry: [String: SourceTensor],
                                      arch: ArchInfo) throws -> LayerFilePlan {
        let expertCount = arch.numExperts
        let f = arch.moeIntermediateSize      // per-expert intermediate
        let d = arch.hiddenSize
        guard f % 64 == 0, d % 64 == 0 else {
            throw RepackError.configurationInvalid(
                detail: "Qwen expert dims must be group-64 multiples")
        }
        guard let gateUp = registry[gateUpName] else {
            throw RepackError.missingTensor(name: gateUpName)
        }
        guard let down = registry[downName] else {
            throw RepackError.missingTensor(name: downName)
        }
        guard gateUp.shape.count == 3, Int(gateUp.shape[0]) == expertCount,
              Int(gateUp.shape[1]) == 2 * f, Int(gateUp.shape[2]) == d else {
            throw RepackError.shapeMismatch(name: gateUpName,
                detail: "expected [\(expertCount), \(2 * f), \(d)], got \(gateUp.shape)")
        }
        guard down.shape.count == 3, Int(down.shape[0]) == expertCount,
              Int(down.shape[1]) == d, Int(down.shape[2]) == f else {
            throw RepackError.shapeMismatch(name: downName,
                detail: "expected [\(expertCount), \(d), \(f)], got \(down.shape)")
        }
        guard gateUp.dtype == .bf16, down.dtype == .bf16 else {
            throw RepackError.dtypeMismatch(name: gateUpName,
                detail: "expected BF16 routed experts")
        }

        let gateW = UInt64(f) * UInt64(d) / 2
        let gateAux = UInt64(f) * UInt64(d / 64) * 2
        let downW = UInt64(d) * UInt64(f) / 2
        let downAux = UInt64(d) * UInt64(f / 64) * 2
        let roleBytes = gateW + 2 * gateAux       // gate == up sizes
        let blobBytes = 2 * roleBytes + (downW + 2 * downAux)
        let expertStride = roundUpToPage(blobBytes)

        // Per-expert source coordinates: the gate/up halves live at distinct
        // base offsets inside the fused gate_up_proj tensor, so the writer
        // derives the per-role base from the role's slice index (gate = 0,
        // up = f*d*2 bytes into each expert's fused block).
        var subs: [PerExpertTensorSlice] = []
        subs.reserveCapacity(9)

        // gate = fused rows [0, F)
        let gateWSlice = PerExpertTensorSlice(
            role: "gate", component: "weights",
            dtype: FinchFormatV1.DType.u32.rawValue,
            logicalShape: [UInt64(f), UInt64(d)],
            offsetInExpertBlob: 0, sizeInExpertBlob: gateW,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceTensor: gateUp, bitsForWeights: 4)
        let gateSSlice = PerExpertTensorSlice(
            role: "gate", component: "scales",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(f), UInt64(d / 64)],
            offsetInExpertBlob: gateW, sizeInExpertBlob: gateAux,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceTensor: gateUp, bitsForWeights: nil)
        let gateBSlice = PerExpertTensorSlice(
            role: "gate", component: "biases",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(f), UInt64(d / 64)],
            offsetInExpertBlob: gateW + gateAux, sizeInExpertBlob: gateAux,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceTensor: gateUp, bitsForWeights: nil)
        subs.append(contentsOf: [gateWSlice, gateSSlice, gateBSlice])

        // up = fused rows [F, 2F)
        let upWSlice = PerExpertTensorSlice(
            role: "up", component: "weights",
            dtype: FinchFormatV1.DType.u32.rawValue,
            logicalShape: [UInt64(f), UInt64(d)],
            offsetInExpertBlob: roleBytes, sizeInExpertBlob: gateW,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceBaseOffset: UInt64(f * d * 2),
            sourceTensor: gateUp, bitsForWeights: 4)
        let upSSlice = PerExpertTensorSlice(
            role: "up", component: "scales",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(f), UInt64(d / 64)],
            offsetInExpertBlob: roleBytes + gateW, sizeInExpertBlob: gateAux,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceBaseOffset: UInt64(f * d * 2),
            sourceTensor: gateUp, bitsForWeights: nil)
        let upBSlice = PerExpertTensorSlice(
            role: "up", component: "biases",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(f), UInt64(d / 64)],
            offsetInExpertBlob: roleBytes + gateW + gateAux, sizeInExpertBlob: gateAux,
            sourceOffsetPerExpert: UInt64(2 * f * d * 2),
            sourceBaseOffset: UInt64(f * d * 2),
            sourceTensor: gateUp, bitsForWeights: nil)
        subs.append(contentsOf: [upWSlice, upSSlice, upBSlice])

        // down
        let downWSlice = PerExpertTensorSlice(
            role: "down", component: "weights",
            dtype: FinchFormatV1.DType.u32.rawValue,
            logicalShape: [UInt64(d), UInt64(f)],
            offsetInExpertBlob: 2 * roleBytes, sizeInExpertBlob: downW,
            sourceOffsetPerExpert: UInt64(d * f * 2),
            sourceTensor: down, bitsForWeights: 4)
        let downSSlice = PerExpertTensorSlice(
            role: "down", component: "scales",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(d), UInt64(f / 64)],
            offsetInExpertBlob: 2 * roleBytes + downW, sizeInExpertBlob: downAux,
            sourceOffsetPerExpert: UInt64(d * f * 2),
            sourceTensor: down, bitsForWeights: nil)
        let downBSlice = PerExpertTensorSlice(
            role: "down", component: "biases",
            dtype: FinchFormatV1.DType.bf16.rawValue,
            logicalShape: [UInt64(d), UInt64(f / 64)],
            offsetInExpertBlob: 2 * roleBytes + downW + downAux, sizeInExpertBlob: downAux,
            sourceOffsetPerExpert: UInt64(d * f * 2),
            sourceTensor: down, bitsForWeights: nil)
        subs.append(contentsOf: [downWSlice, downSSlice, downBSlice])

        return LayerFilePlan(layerIndex: layer, path: path,
                             expertsPerLayer: expertCount,
                             expertStride: expertStride,
                             subTensors: subs)
    }

    // MARK: - Helpers

    private static func roundUpToPage(_ v: UInt64) -> UInt64 {
        let p = FinchFormatV1.alignmentBytes
        return ((v + p - 1) / p) * p
    }

    private static func padTo4(_ s: [UInt64]) -> [UInt32] {
        var out: [UInt32] = []
        out.reserveCapacity(4)
        for v in s.prefix(4) { out.append(UInt32(v)) }
        while out.count < 4 { out.append(0) }
        return out
    }

    /// Stable resident order: embedding, per-layer groups in layer order
    /// (norms → mixer → shared expert → router), then lm_head + final norm.
    private static func qwenResidentOrdering() -> (String, String) -> Bool {
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "model.language_model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "lm_head.weight"                            { return (3, 0, 0, n) }
            if n == "model.language_model.norm.weight"          { return (3, 1, 0, n) }
            if let li = layerIndex(in: n) {
                return (1, li, slotRank(in: n), n)
            }
            return (2, 0, 0, n)
        }
        return { a, b in
            let ka = key(a), kb = key(b)
            if ka.0 != kb.0 { return ka.0 < kb.0 }
            if ka.1 != kb.1 { return ka.1 < kb.1 }
            if ka.2 != kb.2 { return ka.2 < kb.2 }
            return ka.3 < kb.3
        }
    }

    /// Within-layer slot order (norms, mixer, shared expert, router).
    private static func slotRank(in n: String) -> Int {
        if n.hasSuffix(".input_layernorm.weight")             { return 0 }
        if n.hasSuffix(".post_attention_layernorm.weight")    { return 1 }
        if n.hasSuffix(".self_attn.q_proj.weight")            { return 2 }
        if n.hasSuffix(".self_attn.k_proj.weight")            { return 3 }
        if n.hasSuffix(".self_attn.v_proj.weight")            { return 4 }
        if n.hasSuffix(".self_attn.o_proj.weight")            { return 5 }
        if n.hasSuffix(".self_attn.q_norm.weight")            { return 6 }
        if n.hasSuffix(".self_attn.k_norm.weight")            { return 7 }
        if n.hasSuffix(".linear_attn.in_proj_qkv.weight")     { return 2 }
        if n.hasSuffix(".linear_attn.in_proj_z.weight")       { return 3 }
        if n.hasSuffix(".linear_attn.in_proj_a.weight")       { return 4 }
        if n.hasSuffix(".linear_attn.in_proj_b.weight")       { return 5 }
        if n.hasSuffix(".linear_attn.out_proj.weight")        { return 6 }
        if n.hasSuffix(".linear_attn.norm.weight")            { return 7 }
        if n.hasSuffix(".linear_attn.conv1d.weight")          { return 8 }
        if n.hasSuffix(".linear_attn.A_log")                  { return 9 }
        if n.hasSuffix(".linear_attn.dt_bias")                { return 10 }
        if n.hasSuffix(".mlp.shared_expert.gate_proj.weight") { return 12 }
        if n.hasSuffix(".mlp.shared_expert.up_proj.weight")   { return 13 }
        if n.hasSuffix(".mlp.shared_expert.down_proj.weight") { return 14 }
        if n.hasSuffix(".mlp.shared_expert_gate.weight")      { return 15 }
        if n.hasSuffix(".mlp.gate.weight")                    { return 16 }
        return 100
    }
}

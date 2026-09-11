import Foundation
import FinchMoEFormat

// MARK: - Qwen repack plan types
//
// The Qwen 3.6 / 3.8 bf16 checkpoints have no pre-quantized tensors and no
// `config.json -> quantization` slot, so the Gemma byte-copy planner cannot
// describe them. This planner instead assigns every source tensor a WRITE
// TRANSFORM (bf16 rows → int4/int8 affine, norm (1+w) baking, bf16 → fp16 /
// fp32 raw conversions, I64 byte copy) and lays the quantized payloads out
// in the resident file. The expert layers reuse
// `LayerFilePlan`/`PerExpertTensorSlice` with the fused `gate_up_proj`
// source split into the separate gate/up role slices the engine's streamed
// expert kernels read. Qwen3.8's PLE n-gram table parts (128 × 2.5M × 160
// BF16, layer 1) bypass the resident file entirely: each part tensor is
// copied verbatim to its own `ple_shards/shard_NNN.bin`.

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
    /// int64 `[n]` → raw little-endian byte copy (Qwen3.8 PLE hash metadata:
    /// layer_multipliers / ngram_heads_offsets / ngram_heads_vocab_sizes).
    /// These 45-bit multipliers and 20M-scale offsets must round-trip
    /// exactly — no float representation is lossless here.
    case rawInt64(Int)

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
        case .rawInt64(let n): UInt64(n) * 8
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
    /// dtype byte for IndexEntry: 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32,
    /// 4 = I64 (raw PLE metadata).
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
    /// The `manifest.files` key for this file. Derived here so the writer, the
    /// manifest, and the resume journal cannot disagree about it.
    var relativePath: String { (path as NSString).lastPathComponent }
}

/// One PLE n-gram table part: the part's source tensor copied verbatim
/// (raw BF16, row-major) to its own file. The real model writes 128 parts of
/// 2,500,012 × 160 rows (~800 MB each, 102.4 GB total); synthetic snapshots
/// write fewer, smaller parts and the planner census-corrects
/// `arch.ngramPartCount/ngramPartRows` to match.
struct QwenPLEPartFilePlan: Sendable {
    let partIndex: Int
    /// Absolute write target under the install's `ple_shards/` directory.
    let path: String
    /// `ple_shards/shard_%03d.bin` — the manifest.files relative path.
    let relativePath: String
    let rows: Int
    let cols: Int
    let source: SourceTensor
}

struct QwenRepackPlan: Sendable {
    /// Census-corrected arch: for a qwen3_8 source the PLE part geometry
    /// (part count, rows per part) is taken from the live shard headers, so
    /// the manifest always matches what was actually written.
    let arch: ArchInfo
    let resident: QwenResidentFilePlan
    let layers: [LayerFilePlan]
    /// Empty for every non-qwen3_8 source.
    let pleParts: [QwenPLEPartFilePlan]
    let excludedTensorNames: [String]
}

// MARK: - Planner

enum QwenRepackPlanner {

    /// Remaps a qwen checkpoint name to the resident entry name the engine
    /// resolves. The checkpoint nests the text model as
    /// `model.language_model.*`; the engine (shared with Gemma) expects
    /// `language_model.model.*` for qwen3_6. Qwen3.8 has NO inner `model.`
    /// stage (hyper-connection replaced the final norm, so there is nothing
    /// left under a `model.` grouping): its entries are the shallow
    /// `language_model.*`. `lm_head.weight` passes through.
    static func residentName(for source: String,
                             family: String = ArchInfo.qwen36Family) -> String? {
        if source == "lm_head.weight" { return source }
        guard source.hasPrefix("model.language_model.") else { return nil }
        let tail = source.dropFirst("model.language_model.".count)
        if family == ArchInfo.qwen38Family {
            return "language_model." + tail
        }
        return "language_model.model." + tail
    }

    /// Parses the part index from `...ngram_embedding.shard_<n>.weight`
    /// (lexical order is NOT numeric: `shard_10` < `shard_2`).
    private static func plePartIndex(in name: String) -> Int? {
        guard let r = name.range(of: ".ngram_embedding.shard_") else { return nil }
        let digits = name[r.upperBound...].dropLast(".weight".count)
        guard !digits.isEmpty, digits.allSatisfy(\.isNumber) else { return nil }
        return Int(digits)
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

        // The qwen3_8 family has different resident names (shallow
        // `language_model.*`, no inner `model.`) and extra tensor classes;
        // every family-dependent call goes through this value so qwen3_6
        // planning is bit-for-bit what it was before the family parameter.
        let family = arch.modelFamily ?? ArchInfo.qwen36Family
        var excluded: [String] = []
        var residentSources: [String] = []     // checkpoint names, sorted
        var routedSources: [Int: (gateUp: String, down: String)] = [:]
        // PLE n-gram part tensors keyed by shard index. Checkpoint name order
        // is lexical, NOT numeric (shard_10 < shard_2), so the index is parsed
        // from the suffix. Parts bypass the resident file entirely.
        var plePartsByIndex: [Int: String] = [:]

        for (name, _) in registry {
            if family == ArchInfo.qwen38Family, let partIndex = plePartIndex(in: name) {
                plePartsByIndex[partIndex] = name
                continue
            }
            guard residentName(for: name, family: family) != nil else {
                // Vision tower / MTP / other non-text tensors: dropped for
                // the text-only install.
                if !name.hasPrefix("model.language_model.") && name != "lm_head.weight" {
                    excluded.append(name)
                }
                continue
            }
            guard let layer = layerIndex(in: name) else {
                residentSources.append(name)     // embed_tokens, norm / root
                continue                         // hyper-connection mixer, lm_head
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

        // PLE n-gram part validation + census correction. The part geometry
        // (count, rows per part) is read from the live shard headers: the
        // real snapshot's 128 × 2,500,012 agrees with the config/frozen
        // values, a synthetic snapshot's parts are whatever it wrote, and the
        // manifest arch must match what was actually written in both cases.
        var planArch = arch
        var plePartPlans: [QwenPLEPartFilePlan] = []
        if family == ArchInfo.qwen38Family {
            let count = plePartsByIndex.count
            guard count > 0 else {
                throw RepackError.configurationInvalid(
                    detail: "qwen3_8 snapshot carries no PLE n-gram part tensors "
                        + "(expected \(arch.ngramPartCount ?? 128) parts of "
                        + "\(arch.ngramPartRows ?? 2_500_012) rows)")
            }
            for i in 0..<count where plePartsByIndex[i] == nil {
                throw RepackError.configurationInvalid(
                    detail: "PLE part index gap at \(i) (found \(count) parts, not contiguous)")
            }
            if let configured = arch.ngramPartCount, configured != count {
                throw RepackError.configurationInvalid(
                    detail: "config split_ngram_parts \(configured) != live part count \(count)")
            }
            let cols = arch.ngramRowDim ?? 160
            var rows: Int?
            for i in 0..<count {
                guard let tensor = registry[plePartsByIndex[i]!] else {
                    throw RepackError.missingTensor(name: plePartsByIndex[i]!)
                }
                guard tensor.dtype == .bf16, tensor.shape.count == 2,
                      Int(tensor.shape[1]) == cols else {
                    throw RepackError.shapeMismatch(
                        name: plePartsByIndex[i]!,
                        detail: "expected BF16 [_, \(cols)] n-gram part, got \(tensor.shape)")
                }
                let r = Int(tensor.shape[0])
                if let seen = rows, seen != r {
                    throw RepackError.shapeMismatch(
                        name: plePartsByIndex[i]!,
                        detail: "part rows \(r) differ from other parts (\(seen))")
                }
                rows = r
            }
            guard let uniformRows = rows else {
                throw RepackError.configurationInvalid(detail: "empty PLE part set")
            }
            planArch.ngramPartCount = count
            planArch.ngramPartRows = uniformRows

            let pleDir = (outputDir as NSString).appendingPathComponent("ple_shards")
            plePartPlans.reserveCapacity(count)
            for i in 0..<count {
                let source = registry[plePartsByIndex[i]!]!
                let rel = String(format: "ple_shards/shard_%03d.bin", i)
                plePartPlans.append(QwenPLEPartFilePlan(
                    partIndex: i,
                    path: (pleDir as NSString).appendingPathComponent(
                        String(format: "shard_%03d.bin", i)),
                    relativePath: rel,
                    rows: uniformRows, cols: cols, source: source))
            }
        } else if !plePartsByIndex.isEmpty {
            throw RepackError.configurationInvalid(
                detail: "non-qwen3_8 snapshot carries PLE n-gram part tensors")
        }

        let residentPath = (outputDir as NSString).appendingPathComponent("model_weights.bin")
        let resident = try planResidentFile(path: residentPath,
                                            sourceNames: residentSources,
                                            registry: registry,
                                            family: family)

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

        return QwenRepackPlan(arch: planArch,
                              resident: resident,
                              layers: layerPlans,
                              pleParts: plePartPlans,
                              excludedTensorNames: excluded)
    }

    // MARK: - Resident planning

    private static func planResidentFile(path: String,
                                         sourceNames: [String],
                                         registry: [String: SourceTensor],
                                         family: String = ArchInfo.qwen36Family) throws
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
            guard let name = residentName(for: source, family: family) else {
                throw RepackError.unknownTensorPrefix(name: source)
            }
            let transform = try transform(for: source, tensor: tensor, family: family)
            let rows: Int
            let cols: Int
            switch transform {
            case .int4Affine(let r, let c), .int8Affine(let r, let c):
                rows = r; cols = c
            case .normOnePlusW(let n), .normRawBf16(let n),
                 .bf16ToFp16(let n), .bf16ToFp32(let n), .rawInt64(let n):
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
    /// `family` defaults to qwen3_6 so every legacy call site — and every
    /// qwen3_6 source name, which none of the qwen3_8 clauses below can
    /// match — classifies exactly as before.
    private static func transform(for source: String,
                                  tensor: SourceTensor,
                                  family: String = ArchInfo.qwen36Family) throws -> QwenWriteTransform {
        let shape = tensor.shape

        // Qwen3.8 PLE hash metadata is int64 — admitted before the BF16-only
        // guard. The 45-bit multipliers and 20M-scale per-head vocab offsets
        // must round-trip byte-exact; no float transform is lossless.
        if family == ArchInfo.qwen38Family, tensor.dtype == .i64 {
            guard shape.count == 1,
                  source.hasSuffix(".ple_embedding.layer_multipliers")
                    || source.hasSuffix(".ple_embedding.ngram_heads_offsets")
                    || source.hasSuffix(".ple_embedding.ngram_heads_vocab_sizes") else {
                throw RepackError.shapeMismatch(name: source,
                                                detail: "unexpected I64 tensor \(shape)")
            }
            return .rawInt64(Int(shape[0]))
        }
        guard tensor.dtype == .bf16 else {
            throw RepackError.dtypeMismatch(name: source,
                detail: "expected BF16 source, got \(tensor.dtype)")
        }

        // ---- Qwen3.8-Flash-Next classifications ----
        // Runs before the qwen3_6 chain below, which stays byte-identical
        // (none of these suffixes exist in qwen3_6 snapshots). Without the
        // block, every qwen3_8-only tensor would misclassify: hc_norm and the
        // indexer/PLE norms are 1-D (generic throw / wrong bake), PLE
        // key/value are 8-bit head projections (generic int4), PLE conv1d is
        // 3-D (generic throw).
        if family == ArchInfo.qwen38Family {
            if source.hasSuffix(".hc_norm.weight") {
                // Hyper-connection grouped-RMS gate (per-mixer and the root
                // mixer): the torch module multiplies its weight raw
                // (GatedNorm-style, like the GDN norm) — no (1 + w) bake.
                guard shape.count == 1 else {
                    throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                }
                return .normRawBf16(Int(shape[0]))
            }
            if source.contains(".self_attn.indexer.") {
                if source.hasSuffix(".q_layernorm.weight")
                    || source.hasSuffix(".k_layernorm.weight") {
                    guard shape.count == 1 else {
                        throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                    }
                    return .normOnePlusW(Int(shape[0]))
                }
                guard source.hasSuffix(".index_qk_proj.weight"), shape.count == 2,
                      shape[1] % 64 == 0 else {
                    throw RepackError.configurationInvalid(
                        detail: "unclassifiable Qwen3.8 indexer tensor \(source) shape \(shape)")
                }
                return .int4Affine(rows: Int(shape[0]), cols: Int(shape[1]))
            }
            if source.contains(".ple.") {
                if source.hasSuffix(".conv1d.weight") {
                    // Checkpoint stores [C, 1, kernel]; the writer emits the
                    // squeezed [C * kernel] rows as raw FP16 (GDN conv policy).
                    guard shape.count == 3, shape[1] == 1 else {
                        throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                    }
                    return .bf16ToFp16(Int(shape[0] * shape[2]))
                }
                if source.hasSuffix(".key_proj.weight")
                    || source.hasSuffix(".value_proj.weight") {
                    // PLE head projections feed per-token key/value rows that
                    // persist into the n-gram memory state — int8 like the
                    // GDN linear-attention projections (int4 dequant noise
                    // on those was measured to amplify ~16x downstream).
                    guard shape.count == 2, shape[1] % 64 == 0 else {
                        throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                    }
                    return .int8Affine(rows: Int(shape[0]), cols: Int(shape[1]))
                }
                if source.hasSuffix(".norm_key.weight")
                    || source.hasSuffix(".norm_query.weight")
                    || source.hasSuffix(".norm_conv.weight") {
                    guard shape.count == 1 else {
                        throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                    }
                    return .normOnePlusW(Int(shape[0]))
                }
                // The n-gram part tensors are routed to pleParts in plan();
                // anything else under ple. is unclassifiable.
                throw RepackError.configurationInvalid(
                    detail: "unclassifiable Qwen3.8 ple tensor \(source) shape \(shape)")
            }
            if source.hasSuffix(".input_mix_weight_down.weight")
                || source.hasSuffix(".input_mix_weight_up.weight")
                || source.hasSuffix(".block_inject_weight.weight") {
                // Hyper-connection mix projections (per-mixer bundles and the
                // root mixer) are plain 2-D projections → int4, named
                // explicitly so they never depend on the generic fallthrough.
                guard shape.count == 2, shape[1] % 64 == 0 else {
                    throw RepackError.shapeMismatch(name: source, detail: "\(shape)")
                }
                return .int4Affine(rows: Int(shape[0]), cols: Int(shape[1]))
            }
        }

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
        case .rawInt64:                return FinchFormatV1.DType.i64.rawValue
        }
    }

    private static func logicalShape(for transform: QwenWriteTransform,
                                     source: SourceTensor) -> [UInt64] {
        switch transform {
        case .int4Affine(let r, let c), .int8Affine(let r, let c):
            return [UInt64(r), UInt64(c)]
        case .normOnePlusW(let n), .normRawBf16(let n),
             .bf16ToFp16(let n), .bf16ToFp32(let n), .rawInt64(let n):
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
    /// (norms → mixer → shared expert → router), then lm_head + final norm
    /// (qwen3_6) or the root hyper-connection mixer (qwen3_8, which replaces
    /// the final norm). Only source names are compared — the qwen3_8 mixer
    /// prefix can never match a qwen3_6 name, so qwen3_6 ordering is
    /// unchanged.
    private static func qwenResidentOrdering() -> (String, String) -> Bool {
        func key(_ n: String) -> (Int, Int, Int, String) {
            if n == "model.language_model.embed_tokens.weight" { return (0, 0, 0, n) }
            if n == "lm_head.weight"                            { return (3, 0, 0, n) }
            if n == "model.language_model.norm.weight"
                || n.hasPrefix("model.language_model.hyper_connection_mixer.") {
                return (3, 1, 0, n)
            }
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
        // Qwen3.8-Flash-Next slots (17+). These suffixes cannot appear in a
        // qwen3_6 snapshot, so qwen3_6 ordering is unchanged; they give the
        // qwen3_8 layer files a stable order: attention/HC bodies first, then
        // the per-layer hyper-connection bundles, the QSA indexer (full
        // layers), and the layer-1 PLE block.
        if n.hasSuffix(".attn_hyper_connection.hc_norm.weight")             { return 17 }
        if n.hasSuffix(".attn_hyper_connection.input_mix_weight_down.weight") { return 18 }
        if n.hasSuffix(".attn_hyper_connection.input_mix_weight_up.weight")   { return 19 }
        if n.hasSuffix(".attn_hyper_connection.block_inject_weight.weight")   { return 20 }
        if n.hasSuffix(".mlp_hyper_connection.hc_norm.weight")              { return 21 }
        if n.hasSuffix(".mlp_hyper_connection.input_mix_weight_down.weight") { return 22 }
        if n.hasSuffix(".mlp_hyper_connection.input_mix_weight_up.weight")   { return 23 }
        if n.hasSuffix(".mlp_hyper_connection.block_inject_weight.weight")   { return 24 }
        if n.hasSuffix(".self_attn.indexer.index_qk_proj.weight")           { return 25 }
        if n.hasSuffix(".self_attn.indexer.q_layernorm.weight")             { return 26 }
        if n.hasSuffix(".self_attn.indexer.k_layernorm.weight")             { return 27 }
        if n.hasSuffix(".ple.conv1d.weight")                                { return 28 }
        if n.hasSuffix(".ple.key_proj.weight")                              { return 29 }
        if n.hasSuffix(".ple.value_proj.weight")                            { return 30 }
        if n.hasSuffix(".ple.norm_query.weight")                            { return 31 }
        if n.hasSuffix(".ple.norm_key.weight")                              { return 32 }
        if n.hasSuffix(".ple.norm_conv.weight")                             { return 33 }
        if n.hasSuffix(".ple_embedding.layer_multipliers")                  { return 34 }
        if n.hasSuffix(".ple_embedding.ngram_heads_offsets")                { return 35 }
        if n.hasSuffix(".ple_embedding.ngram_heads_vocab_sizes")            { return 36 }
        return 100
    }
}

import Foundation

/// Errors from the vision payload. Kept separate from `ModelError` so the app
/// can distinguish "this install has no vision runtime" from a corrupt text
/// install, and so the messages name the file or tensor at fault.
enum VisionError: Error, CustomStringConvertible {
    case missingPayload(String)
    case invalidConfig(String)
    case invalidPayload(String)
    case unsupportedImage(String)
    case encodingFailed(String)

    var description: String {
        switch self {
        case .missingPayload(let path):
            return "vision payload is missing: \(path)"
        case .invalidConfig(let detail):
            return "invalid vision config: \(detail)"
        case .invalidPayload(let detail):
            return "invalid vision payload: \(detail)"
        case .unsupportedImage(let detail):
            return "unsupported image: \(detail)"
        case .encodingFailed(let detail):
            return "vision encoding failed: \(detail)"
        }
    }
}

/// The subset of the source config the vision tower needs, validated once at
/// load time so the kernels can assume fixed geometry.
///
/// The source `config.json` is copied into the install verbatim; it is the
/// authority for the tower dimensions, the image/video token ids, and the
/// text-side mRoPE sections. The preprocessor sidecar is the authority for the
/// resize bounds and normalization.
struct VisionConfig: Sendable {
    struct Geometry: Sendable, Equatable {
        let depth: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let numHeads: Int
        let headDim: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let spatialMergeSize: Int
        let numPositionEmbeddings: Int
        let gridSide: Int
        let outHiddenSize: Int
        let inChannels: Int
        let layerNormEps: Float
    }

    let geometry: Geometry
    let imageTokenId: Int
    let visionStartTokenId: Int
    let visionEndTokenId: Int
    let videoTokenId: Int?
    /// Hidden width of the text model the merger projects into.
    let textHiddenSize: Int
    /// Interleaved mRoPE sections; `[11,11,10]` on both shipped checkpoints.
    let mropeSection: [Int]
    let mropeInterleaved: Bool
    /// Text rotary width in elements (head_dim * partial_rotary_factor).
    let textRotaryDim: Int

    let minPixels: Int
    let maxPixels: Int
    let imageMean: [Float]
    let imageStd: [Float]

    /// The largest image the tower will accept for this architecture. The
    /// caller may lower it further for device headroom; this is the hard cap
    /// from the preprocessor config.
    var configuredMaxPixels: Int { min(maxPixels, 16_777_216) }

    static func load(directory: FinchModelDirectory) throws -> VisionConfig {
        let modelData = try directory.readMetadata(
            "vision/model_config.json", maxBytes: 1 << 20)
        let preData = try directory.readMetadata(
            "vision/preprocessor_config.json", maxBytes: 1 << 20)
        let root = try jsonObject(modelData, named: "vision/model_config.json")
        guard let vision = root["vision_config"] as? [String: Any] else {
            throw VisionError.invalidConfig("vision_config is missing")
        }
        let text = root["text_config"] as? [String: Any] ?? [:]

        let depth = try int(vision, "depth")
        let hidden = try int(vision, "hidden_size")
        let intermediate = try int(vision, "intermediate_size")
        let heads = try int(vision, "num_heads")
        let patch = try int(vision, "patch_size")
        let temporal = try int(vision, "temporal_patch_size")
        let merge = try int(vision, "spatial_merge_size")
        let positions = try int(vision, "num_position_embeddings")
        let outHidden = try int(vision, "out_hidden_size")
        let inChannels = try int(vision, "in_channels")
        guard hidden > 0, heads > 0, hidden % heads == 0 else {
            throw VisionError.invalidConfig(
                "hidden_size \(hidden) is not divisible by num_heads \(heads)")
        }
        let headDim = hidden / heads
        let gridSide = Int(Double(positions).squareRoot().rounded())
        guard gridSide * gridSide == positions else {
            throw VisionError.invalidConfig(
                "num_position_embeddings \(positions) is not a perfect square")
        }
        guard patch > 0, temporal > 0, merge > 0, inChannels == 3 else {
            throw VisionError.invalidConfig(
                "patch/temporal/merge/in_channels geometry is unsupported")
        }
        // The first release's kernels are specialized for the 3.6/3.8 tower.
        // Fail loudly rather than silently compute a different model.
        guard depth == 27, hidden == 1152, heads == 16, headDim == 72,
              intermediate == 4304, patch == 16, temporal == 2, merge == 2,
              positions == 2304, inChannels == 3 else {
            throw VisionError.invalidConfig(
                "unsupported tower geometry d=\(depth) h=\(hidden) heads=\(heads) "
                + "inter=\(intermediate) patch=\(patch) temporal=\(temporal) "
                + "merge=\(merge) positions=\(positions)")
        }
        guard let imageToken = intOrNil(root, "image_token_id"),
              let startToken = intOrNil(root, "vision_start_token_id"),
              let endToken = intOrNil(root, "vision_end_token_id") else {
            throw VisionError.invalidConfig(
                "image/vision_start/vision_end token ids are missing")
        }

        let textHidden = intOrNil(text, "hidden_size") ?? 0
        guard outHidden == textHidden, textHidden > 0 else {
            throw VisionError.invalidConfig(
                "merger out_hidden_size \(outHidden) does not match text hidden_size \(textHidden)")
        }
        let textHeadDim = intOrNil(text, "head_dim") ?? 0
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        let partial = (rope["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0
        let rotaryDim = Int((Double(textHeadDim) * partial).rounded())
        let section = (rope["mrope_section"] as? [NSNumber])?.map(\.intValue) ?? []
        let interleaved = (rope["mrope_interleaved"] as? NSNumber)?.boolValue ?? false
        guard textHeadDim > 0, rotaryDim > 0, rotaryDim % 2 == 0,
              section.reduce(0, +) == rotaryDim / 2 else {
            throw VisionError.invalidConfig(
                "mRoPE section \(section) does not cover rotary dim \(rotaryDim) / 2")
        }
        guard interleaved else {
            throw VisionError.invalidConfig("only interleaved mRoPE is supported")
        }

        let pre = try jsonObject(preData, named: "vision/preprocessor_config.json")
        let size = pre["size"] as? [String: Any] ?? [:]
        let minPixels = intOrNil(size, "shortest_edge") ?? 65_536
        let maxPixels = intOrNil(size, "longest_edge") ?? 16_777_216
        let mean = (pre["image_mean"] as? [NSNumber])?.map(\.floatValue)
            ?? [0.5, 0.5, 0.5]
        let std = (pre["image_std"] as? [NSNumber])?.map(\.floatValue)
            ?? [0.5, 0.5, 0.5]
        guard mean.count == 3, std.count == 3, std.allSatisfy({ $0 > 0 }) else {
            throw VisionError.invalidConfig("image_mean/image_std must be three positive values")
        }
        guard (pre["patch_size"] as? NSNumber)?.intValue == patch,
              (pre["temporal_patch_size"] as? NSNumber)?.intValue == temporal,
              (pre["merge_size"] as? NSNumber)?.intValue == merge else {
            throw VisionError.invalidConfig(
                "preprocessor geometry disagrees with vision_config")
        }

        return VisionConfig(
            geometry: Geometry(depth: depth,
                               hiddenSize: hidden,
                               intermediateSize: intermediate,
                               numHeads: heads,
                               headDim: headDim,
                               patchSize: patch,
                               temporalPatchSize: temporal,
                               spatialMergeSize: merge,
                               numPositionEmbeddings: positions,
                               gridSide: gridSide,
                               outHiddenSize: outHidden,
                               inChannels: inChannels,
                               layerNormEps: 1e-6),
            imageTokenId: imageToken,
            visionStartTokenId: startToken,
            visionEndTokenId: endToken,
            videoTokenId: intOrNil(root, "video_token_id"),
            textHiddenSize: textHidden,
            mropeSection: section,
            mropeInterleaved: interleaved,
            textRotaryDim: rotaryDim,
            minPixels: minPixels,
            maxPixels: maxPixels,
            imageMean: mean,
            imageStd: std)
    }

    private static func jsonObject(_ data: Data, named name: String)
        throws -> [String: Any] {
        do {
            guard let object = try JSONSerialization.jsonObject(with: data)
                    as? [String: Any] else {
                throw VisionError.invalidConfig("\(name) is not a JSON object")
            }
            return object
        } catch let error as VisionError {
            throw error
        } catch {
            throw VisionError.invalidConfig("\(name): \(error)")
        }
    }

    private static func int(_ object: [String: Any], _ key: String) throws -> Int {
        guard let value = (object[key] as? NSNumber)?.intValue else {
            throw VisionError.invalidConfig("\(key) is missing")
        }
        return value
    }

    private static func intOrNil(_ object: [String: Any], _ key: String) -> Int? {
        (object[key] as? NSNumber)?.intValue
    }
}

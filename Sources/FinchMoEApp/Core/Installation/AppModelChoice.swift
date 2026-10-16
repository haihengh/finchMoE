import Foundation

/// The entries the app's Preset picker offers.
///
/// Declaration order is picker order (`AppModel.swift`'s `modelChoices` is
/// `allCases`), so each abliterated case sits directly after the base case it is
/// a variant of, and each narrower expert width sits after the wider one.
///
/// One case per install. The widths of a checkpoint share its resident weights
/// and its source index, so the probe tells them apart by the manifest's
/// routed-expert width (`AppModelInstallationProbe`) — which only works because
/// each row here also has its own `defaultURL`: the path that row downloads to
/// and starts from. A row whose distribution were a different width would write
/// misnamed bytes and then flip rows once the install completed.
public enum AppModelChoice: String, CaseIterable, Identifiable, Sendable {
    case gemma4
    case qwen3_6, qwen3_6_3bit
    case qwen3_6_abliterated, qwen3_6_abliterated_3bit
    case qwen3_8, qwen3_8_3bit, qwen3_8_2bit
    case qwen3_8_abliterated, qwen3_8_abliterated_3bit, qwen3_8_abliterated_2bit

    public var id: String { rawValue }

    /// The picker text is the descriptor's own name so the preset list and the
    /// install screen cannot disagree about what a row is.
    public var label: String { descriptor.displayName }

    public var descriptor: AppModelInstallDescriptor {
        switch self {
        case .gemma4: return .default
        case .qwen3_6: return .qwen3_6
        case .qwen3_6_3bit: return .qwen3_6_3bit
        case .qwen3_6_abliterated: return .qwen3_6_abliterated
        case .qwen3_6_abliterated_3bit: return .qwen3_6_abliterated_3bit
        case .qwen3_8: return .qwen3_8
        case .qwen3_8_3bit: return .qwen3_8_3bit
        case .qwen3_8_2bit: return .qwen3_8_2bit
        case .qwen3_8_abliterated: return .qwen3_8_abliterated
        case .qwen3_8_abliterated_3bit: return .qwen3_8_abliterated_3bit
        case .qwen3_8_abliterated_2bit: return .qwen3_8_abliterated_2bit
        }
    }

    public func defaultURL(packageRoot: URL) -> URL {
        switch self {
        case .gemma4:
            return packageRoot.appendingPathComponent("scratch/gemma4.finch", isDirectory: true)
        case .qwen3_6:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-4bit.finch", isDirectory: true)
        case .qwen3_6_3bit:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-3bit.finch", isDirectory: true)
        case .qwen3_6_abliterated:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-abliterated-4bit.finch",
                                                      isDirectory: true)
        case .qwen3_6_abliterated_3bit:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-abliterated-3bit.finch",
                                                      isDirectory: true)
        case .qwen3_8:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-125B-ple4bit.finch", isDirectory: true)
        case .qwen3_8_3bit:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-125B-ple3bit.finch", isDirectory: true)
        case .qwen3_8_2bit:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-125B-ple2bit.finch", isDirectory: true)
        case .qwen3_8_abliterated:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch",
                                                      isDirectory: true)
        case .qwen3_8_abliterated_3bit:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-abliterated-ple3bit.finch",
                                                      isDirectory: true)
        case .qwen3_8_abliterated_2bit:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-abliterated-ple2bit.finch",
                                                      isDirectory: true)
        }
    }
}

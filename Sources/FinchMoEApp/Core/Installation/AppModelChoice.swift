import Foundation

/// The entries the app's Preset picker offers.
///
/// Declaration order is picker order (`AppModel.swift`'s `modelChoices` is
/// `allCases`), so each abliterated case sits directly after the base case it is
/// a variant of rather than being grouped at the end.
public enum AppModelChoice: String, CaseIterable, Identifiable, Sendable {
    case gemma4
    case qwen3_6
    case qwen3_6_abliterated
    case qwen3_8
    case qwen3_8_abliterated

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .gemma4: return "Gemma 4 26B-A4B"
        case .qwen3_6: return "Qwen 3.6 35B-A3B"
        case .qwen3_6_abliterated: return "Qwen 3.6 35B-A3B (Abliterated)"
        case .qwen3_8: return "Qwen 3.8 Flash-Next 125B"
        case .qwen3_8_abliterated: return "Qwen 3.8 Flash-Next 125B (Abliterated)"
        }
    }

    public var descriptor: AppModelInstallDescriptor {
        switch self {
        case .gemma4: return .default
        case .qwen3_6: return .qwen3_6
        case .qwen3_6_abliterated: return .qwen3_6_abliterated
        case .qwen3_8: return .qwen3_8
        case .qwen3_8_abliterated: return .qwen3_8_abliterated
        }
    }

    public func defaultURL(packageRoot: URL) -> URL {
        switch self {
        case .gemma4:
            return packageRoot.appendingPathComponent("scratch/gemma4.finch", isDirectory: true)
        case .qwen3_6:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-4bit.finch", isDirectory: true)
        case .qwen3_6_abliterated:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-abliterated-4bit.finch",
                                                      isDirectory: true)
        case .qwen3_8:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-125B-ple4bit.finch", isDirectory: true)
        case .qwen3_8_abliterated:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-abliterated-ple4bit.finch",
                                                      isDirectory: true)
        }
    }
}
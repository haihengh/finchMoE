import Foundation

public enum AppModelChoice: String, CaseIterable, Identifiable, Sendable {
    case gemma4
    case qwen3_6
    case qwen3_8

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .gemma4: return "Gemma 4 26B-A4B"
        case .qwen3_6: return "Qwen 3.6 35B-A3B"
        case .qwen3_8: return "Qwen 3.8 Flash-Next 125B"
        }
    }

    public var descriptor: AppModelInstallDescriptor {
        switch self {
        case .gemma4: return .default
        case .qwen3_6: return .qwen3_6
        case .qwen3_8: return .qwen3_8
        }
    }

    public func defaultURL(packageRoot: URL) -> URL {
        switch self {
        case .gemma4:
            return packageRoot.appendingPathComponent("scratch/gemma4.finch", isDirectory: true)
        case .qwen3_6:
            return packageRoot.appendingPathComponent("models/Qwen3.6-35B-A3B-4bit.finch", isDirectory: true)
        case .qwen3_8:
            return packageRoot.appendingPathComponent("models/Qwen3.8-Flash-Next-125B.finch", isDirectory: true)
        }
    }
}
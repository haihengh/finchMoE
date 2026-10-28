import Foundation

public enum AppModelLoadPhase: Equatable, Sendable {
    case validatingDirectory
    case tokenizer
    case verifyingWeights
    case preparingRunner

    public var label: String {
        switch self {
        case .validatingDirectory: return "Validating model directory"
        case .tokenizer: return "Loading tokenizer"
        case .verifyingWeights: return "Verifying weights"
        case .preparingRunner: return "Preparing runner"
        }
    }
}

public enum AppModelLoadState: Equatable, Sendable {
    case notLoaded
    case loading(AppModelLoadPhase)
    case cancelling
    case unloading
    case ready(modelDirectory: URL, loadSeconds: Double)
    case failed(AppInferenceError)

    public var isLoading: Bool {
        switch self {
        case .loading, .cancelling, .unloading: return true
        default: return false
        }
    }

    public var isReady: Bool {
        if case .ready = self { return true }
        return false
    }

    public var isFailed: Bool {
        if case .failed = self { return true }
        return false
    }
}

public enum AppGenerationPhase: String, Equatable, Sendable {
    case idle
    /// The vision tower is encoding an attached image (and the multimodal
    /// prompt is being assembled). On a phone this is the long silent stretch
    /// before the first prefill chunk, so it must not read as "Prefill":
    /// there are no prefill counts to show yet and the text engine has not
    /// started.
    case processingImage
    case prefill
    case decode
}

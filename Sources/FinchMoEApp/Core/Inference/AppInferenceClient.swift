import Foundation

public protocol AppInferenceClient: Sendable {
    func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error>
    func cancel()
}

/// A client that owns a loadable model session. Loading is split from
/// generation so the UI can pre-load the ~1.6 GB resident weights once and
/// keep them warm across runs. Generation never loads or replaces a session.
public protocol AppModelLifecycleClient: AnyObject, AppInferenceClient {
    func ensureLoaded(modelDirectory: URL, maxContextTokens: Int,
                      options: AppRuntimeOptions, forceLogitsHead: Bool,
                      onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws
    func unload() async
}

public protocol AppInferenceMemoryReporting: AnyObject {
    var currentInferenceMemoryBytes: UInt64? { get }
}

/// The verification decision the last successful load actually made, rendered
/// as one line (`ModelIntegrityOutcome.logDescription`), or `nil` when nothing
/// is loaded or the client cannot know.
///
/// A capability rather than a field on the load state, for the same reason as
/// the protocols around it: only the in-process and decode-service clients can
/// answer it, and the ~20 existing `AppModelLoadState.ready` construction sites
/// should not have to grow a parameter to say so.
public protocol AppModelIntegrityReporting: AnyObject {
    var modelIntegrityDescription: String? { get }
}

public protocol AppInferenceTranscriptReporting: AnyObject {
    var generationTranscriptMailbox: GenerationTranscriptMailbox { get }
}

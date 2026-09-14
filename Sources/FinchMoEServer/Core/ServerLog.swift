import Foundation
import FinchMoE

enum ServerLog {
    /// The one line here that is not about a request: which verification the
    /// model load actually performed. It is resolved once at startup, and it is
    /// the only record of whether the trusted-install receipt was used or
    /// silently abandoned for hashing — the difference between a 2 s and a 45 s
    /// prefill on the large install.
    static func integrity(_ outcome: ModelIntegrityOutcome) {
        write("model integrity \(outcome.logDescription)")
    }

    static func accepted(id: String, streaming: Bool) {
        write("request \(id) accepted streaming=\(streaming)")
    }

    static func prepared(id: String, promptTokens: Int?) {
        let count = promptTokens.map(String.init) ?? "backend-managed"
        write("request \(id) prepared prompt=\(count)")
    }

    static func queued(id: String) {
        write("request \(id) queued")
    }

    static func generating(id: String) {
        write("request \(id) generating")
    }

    static func completed(id: String,
                          duration: Duration,
                          completion: ServerCompletion) {
        let usage = completion.usage
        write("request \(id) completed in \(format(duration)) "
            + "prompt=\(usage.promptTokens) "
            + "cached=\(usage.promptTokensDetails.cachedTokens) "
            + "completion=\(usage.completionTokens) "
            + "finish=\(completion.finishReason)")
    }

    static func failed(id: String,
                       phase: String,
                       status: UInt,
                       error: Error) {
        write("request \(id) failed phase=\(phase) status=\(status) "
            + "error=\(String(reflecting: error))")
    }

    private static func format(_ duration: Duration) -> String {
        let seconds = Double(duration.components.seconds)
            + Double(duration.components.attoseconds) / 1e18
        return String(format: "%.3fs", seconds)
    }

    private static func write(_ message: String) {
        let line = "[\(Date().formatted(.iso8601))] \(message)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }
}

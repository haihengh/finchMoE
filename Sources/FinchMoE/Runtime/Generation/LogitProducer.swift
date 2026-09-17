import Metal

/// Produces next-token logits for the `Generator`. The production
/// implementation is `RealForwardRunner`; tests use scripted logits so decode
/// behavior stays independent of the kernel stack.
public protocol LogitProducer: AnyObject, Sendable {
    /// Clear any per-generation state, such as KV cache.
    func reset()
    /// Run one token at `position`, leaving FP16 logits in `logits`.
    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws

    /// Block until every command buffer submitted so far has completed, so a
    /// host read of a GPU-written buffer sees the finished bytes.
    ///
    /// The engine does not wait per command buffer — same-queue submission
    /// order is enough for GPU work to be correctly ordered, and the waits cost
    /// more than they buy. `produce` is the exception and guarantees completion
    /// on return (that is what `sampleOnce` and `qsaSelection` rely on);
    /// `prefillChunked` deliberately does not, so anything reading a prefill
    /// result on the CPU — such as the logits dump — must drain first.
    func drainGPU()

    /// Write the prefill row-fingerprint side buffer, if the instrument is on
    /// (`FQ_ROW_HASH=<path>`, see `docs/RUNTIME_CONTROLS.md`).
    ///
    /// Like the logits dump this is only meaningful at the prefill/decode
    /// boundary and only after `drainGPU()`: the plane the hashes describe is
    /// overwritten from the first decode step. A producer that cannot prefill
    /// has nothing to write.
    func dumpRowHashes()

    /// Whether `dumpRowHashes` has anything to write, so a caller can decide
    /// whether to drain when nothing else needs it.
    var wantsRowHashesDump: Bool { get }
}

extension LogitProducer {
    /// Producers with no GPU (the scripted test doubles) have nothing to drain.
    public func drainGPU() {}
    public func dumpRowHashes() {}
    public var wantsRowHashesDump: Bool { false }
}

public protocol ContinuableLogitProducer: LogitProducer {
    var continuationPosition: Int { get }
    func prepareForContinuation(expectedPosition: Int) throws
}

protocol ContextWindowReporting: Sendable {
    var maxContext: Int { get }
}

public enum PrefillOutputMode: Sendable, Equatable {
    case logits
    case greedyIfAvailable
}

public enum PrefillSeed: Sendable, Equatable {
    case logitsWritten
    case greedyToken(UInt32)
}

public struct PrefillResult: Sendable, Equatable {
    public let newPosition: Int
    public let seed: PrefillSeed

    public init(newPosition: Int, seed: PrefillSeed) {
        self.newPosition = newPosition
        self.seed = seed
    }
}

protocol ChunkedPrefillRunner: LogitProducer {
    /// Prefill a prompt slice using the chunked production runtime.
    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode: PrefillOutputMode,
                        config: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult
}

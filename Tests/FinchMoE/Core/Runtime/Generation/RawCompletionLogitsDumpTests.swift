import Testing
import Foundation
import Metal
@testable import FinchMoE

/// The prefill logits dump — the engine side of the llama.cpp oracle.
///
/// The property that makes it usable is *when* it is taken. The logits buffer
/// holds the final prefill row only between the end of prefill and the first
/// decode `produce`; a dump taken after that is a decode row, which is a
/// perfectly well-formed vector and would score as a mysterious mismatch
/// instead of an error. So the timing gets its own case, with a producer whose
/// prefill and decode rows are distinguishable by value rather than merely
/// different.
@Suite struct RawCompletionLogitsDumpTests {

    /// Spikes prefill and decode at different tokens, so a dump taken at the
    /// wrong moment is identifiable on sight.
    final class PrefillVsDecodeProducer: LogitProducer, ChunkedPrefillRunner,
                                        @unchecked Sendable {
        let vocabSize: Int
        private let prefillToken: Int32
        private let decodeToken: Int32

        init(vocabSize: Int, prefillToken: Int32, decodeToken: Int32) {
            self.vocabSize = vocabSize
            self.prefillToken = prefillToken
            self.decodeToken = decodeToken
        }

        func reset() {}

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            spike(decodeToken, into: logits)
        }

        func prefillChunked(tokens: ArraySlice<Int32>,
                            startPosition: Int,
                            outputMode: PrefillOutputMode,
                            config: PrefillRuntimeConfig,
                            into logits: MTLBuffer,
                            onProgress: (Int) -> Void) async throws -> PrefillResult {
            spike(prefillToken, into: logits)
            onProgress(tokens.count)
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .logitsWritten)
        }

        private func spike(_ token: Int32, into logits: MTLBuffer) {
            let ptr = logits.contents().bindMemory(to: Float16.self, capacity: vocabSize)
            for i in 0..<vocabSize { ptr[i] = Float16(-30.0) }
            ptr[Int(token)] = Float16(30.0)
        }
    }

    private static func dumpPath() -> String {
        NSTemporaryDirectory() + "logits-dump-\(UUID().uuidString).bin"
    }

    private static func readFloats(_ path: String) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        #expect(data.count % MemoryLayout<Float>.size == 0)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// The file is raw little-endian f32 widened from the buffer's FP16 — the
    /// exact format the logits-diff scorer reads.
    @Test func theDumpIsWidenedF32() throws {
        let ctx = try MetalContext()
        let vocab = 64
        let scratch = try RawCompletionScratch(context: ctx, vocab: vocab)
        let ptr = scratch.logits.contents().bindMemory(to: Float16.self, capacity: vocab)
        let expected = (0..<vocab).map { Float($0) * 0.5 - 8.0 }
        for i in 0..<vocab { ptr[i] = Float16(expected[i]) }

        let path = Self.dumpPath()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try scratch.writeLogits(to: path)

        let read = try Self.readFloats(path)
        #expect(read.count == vocab)
        // Through FP16, so compare at FP16 precision rather than bitwise.
        #expect(read == expected.map { Float(Float16($0)) })
    }

    /// The dump is the prefill row even though decode ran and overwrote the
    /// buffer afterwards. This is the assertion that fails if the dump ever
    /// moves after the decode loop.
    @Test func theDumpCapturesTheFinalPrefillRow() async throws {
        let ctx = try MetalContext()
        let tok = try await GFTokenizer.load()
        let vocab = tok.vocabSize
        // Top of the vocab: no special token lives there, so neither id can
        // trip a stop and end the generation early.
        let prefillToken = Int32(vocab - 1)
        let decodeToken = Int32(vocab - 2)
        #expect(!tok.stopTokenIDs.contains(prefillToken))
        #expect(!tok.stopTokenIDs.contains(decodeToken))

        let scratch = try RawCompletionScratch(context: ctx, vocab: vocab)
        let promptIds = tok.encode("go", addBOS: true)
        let path = Self.dumpPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        var generated: [Int32] = []
        _ = try await runRawCompletion(
            producer: PrefillVsDecodeProducer(vocabSize: vocab,
                                              prefillToken: prefillToken,
                                              decodeToken: decodeToken),
            tokenizer: tok,
            promptIds: promptIds,
            config: GenerationConfig(maxNewTokens: 3, temperature: 0),
            context: ctx,
            scratch: scratch,
            prefillConfig: .defaultChunked,
            prefillLogitsDumpPath: path) { progress in
                if case .token(_, let id, _) = progress { generated.append(id) }
            }

        // The first token is drawn from the prefill row; every later one is
        // drawn from a decode row. That the loop really did decode is what
        // makes the dump-timing assertion below meaningful rather than vacuous.
        #expect(generated.first == prefillToken)
        #expect(generated.dropFirst().allSatisfy { $0 == decodeToken })
        #expect(generated.count == 3)

        let read = try Self.readFloats(path)
        #expect(read.count == vocab)
        // No `argmax` on Array; the strict comparison keeps the first of any
        // tie, which at one 30 among -30s there is not.
        #expect(read.indices.max { read[$0] < read[$1] } == Int(prefillToken))
        // The decode spike is absent, so this is the prefill row and not the
        // buffer's post-decode contents.
        #expect(read[Int(decodeToken)] == -30.0)
    }
}

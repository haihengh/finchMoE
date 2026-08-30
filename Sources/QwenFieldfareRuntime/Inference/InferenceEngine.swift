import Foundation
import Metal
import QwenFieldfareFormat

/// High-level generation engine tying together `Model`, `ForwardRunner`,
/// `Sampler`, and `Tokenizer`. Used by the CLI and the OpenAI server.
public final class InferenceEngine {

    public let model: Model
    public let runner: ForwardRunner
    public let tokenizer: Tokenizer?

    public struct GenerationOptions: Sendable {
        public var maxTokens: Int
        public var temperature: Float
        public var topP: Float
        public var seed: UInt64?
        public var stopOnEOS: Bool

        public init(maxTokens: Int = 512, temperature: Float = 0.7, topP: Float = 0.9,
                    seed: UInt64? = nil, stopOnEOS: Bool = true) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.seed = seed
            self.stopOnEOS = stopOnEOS
        }
    }

    public init(modelDir: URL, maxSeqLen: Int = 32768, expertCacheSlots: Int = 16) throws {
        let metal = try MetalContext()
        self.model = try Model(modelDir: modelDir, metal: metal,
                               maxSeqLen: maxSeqLen, expertCacheSlots: expertCacheSlots)
        self.runner = try ForwardRunner(model: model)

        // Tokenizer is optional — load if tokenizer.json sits beside the model
        // or in the source directory.
        let tokURL = modelDir.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokURL.path) {
            self.tokenizer = try? Tokenizer(tokenizerJSONURL: tokURL,
                                            bosTokenId: model.config.bosTokenId,
                                            eosTokenId: model.config.eosTokenId)
        } else {
            self.tokenizer = nil
        }
    }

    // MARK: - Token-level generation

    /// Generates from an explicit token prompt, invoking `onToken` for each new
    /// token id. Returns the full list of generated token ids.
    @discardableResult
    public func generate(promptTokens: [Int],
                         options: GenerationOptions,
                         onToken: ((Int) -> Void)? = nil) throws -> [Int] {
        model.kvCache.reset()
        var sampler = Sampler(config: .init(temperature: options.temperature,
                                            topP: options.topP, seed: options.seed))

        var position = 0
        // Prefill: run all prompt tokens; the last step's logits seed the first sample.
        var lastLogits: MTLBuffer? = nil
        for tok in promptTokens {
            lastLogits = try runner.step(token: tok, position: position)
            position += 1
        }
        guard let logitsBuf = lastLogits else { return [] }

        var generated: [Int] = []
        var next = sampler.sample(logitsBuffer: logitsBuf, vocabSize: model.config.vocabSize)

        for _ in 0..<options.maxTokens {
            if options.stopOnEOS && next == model.config.eosTokenId { break }
            generated.append(next)
            onToken?(next)

            let lb = try runner.step(token: next, position: position)
            position += 1
            next = sampler.sample(logitsBuffer: lb, vocabSize: model.config.vocabSize)

            if position >= model.kvCache.maxSeqLen { break }
        }
        return generated
    }

    // MARK: - Text-level generation

    /// Generates text from a prompt string. Requires a tokenizer.
    @discardableResult
    public func generateText(prompt: String,
                             options: GenerationOptions,
                             onPiece: ((String) -> Void)? = nil) throws -> String {
        guard let tokenizer else {
            throw EngineError.noTokenizer
        }
        let promptTokens = tokenizer.encode(prompt)
        var text = ""
        _ = try generate(promptTokens: promptTokens, options: options) { tokenID in
            let piece = tokenizer.decodeToken(tokenID)
            text += piece
            onPiece?(piece)
        }
        return text
    }

    public enum EngineError: Error, CustomStringConvertible {
        case noTokenizer
        public var description: String {
            switch self {
            case .noTokenizer: return "InferenceEngine: no tokenizer.json found next to the model"
            }
        }
    }
}

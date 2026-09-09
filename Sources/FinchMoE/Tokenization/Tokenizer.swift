import Foundation
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        }
    }
}

/// Gemma 4 / Qwen 3.6 tokenizer wrapper.
///
/// Prefers tokenizer sidecars in a completed `.fqturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// The family is detected from the installed `config.json` (`model_type`):
/// Qwen 3.6 has none of the Gemma tool/channel markers (their IDs are -1
/// sentinels on that family) and stops on its own EOS `<|im_end|>` (248046)
/// plus the config's eos_token_id `<|endoftext|>` (248044), matching the
/// checkpoint's generation_config.
///
/// FinchMoE owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
public enum GFTokenizerFamily: String, Sendable, Equatable {
    case gemma4
    case qwen3_6

    /// Detected from a model config's `model_type` (text_config nested).
    static func detect(configJSON: [String: Any]) -> GFTokenizerFamily {
        let root = configJSON
        let tc = (root["text_config"] as? [String: Any]) ?? root
        let mt = (tc["model_type"] as? String) ?? ""
        // Qwen3.8-Flash-Next (qwen4_exp_text) shares the 3.6 tokenizer
        // byte-for-byte, so it lands in the same qwen3_6 handling family.
        if mt.contains("qwen3_5_moe") || mt.contains("qwen3.6") || mt.contains("qwen3_6")
            || mt.contains("qwen4_exp") {
            return .qwen3_6
        }
        return .gemma4
    }
}

public struct GFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v1"

    public let family: GFTokenizerFamily
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    public let channelStartID: Int32
    public let channelEndID: Int32
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["FINCHMOE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(pretrained: modelID)
        return try GFTokenizer(tokenizer: underlying)
    }

    static func loadUncached(from folder: URL) async throws -> GFTokenizer {
        let underlying = try await AutoTokenizer.from(modelFolder: folder)
        var family = GFTokenizerFamily.gemma4
        let configURL = folder.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            family = .detect(configJSON: root)
        }
        return try GFTokenizer(tokenizer: underlying, family: family)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    public init(tokenizer: any Tokenizer,
                family: GFTokenizerFamily = .gemma4) throws {
        self.tokenizer = tokenizer

        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        // bosID/padID are informational — `encode` never prepends on Qwen and
        // the engine never pads.
        let bos: Int
        let pad: Int
        switch family {
        case .gemma4:
            guard let b = tokenizer.bosTokenId else {
                throw GFTokenizerError.missingSpecialToken("<bos>")
            }
            bos = b
            pad = tokenizer.convertTokenToId("<pad>") ?? eos
        case .qwen3_6:
            // Qwen 3.6 has no standalone bos_token; the model config uses
            // <|endoftext|> (248044) as bos/pad/eos_token_id.
            let endoftext = tokenizer.convertTokenToId("<|endoftext|>") ?? eos
            bos = endoftext
            pad = endoftext
        }

        self.family = family
        self.bosID = Int32(bos)
        self.eosID = Int32(eos)
        self.padID = Int32(pad)

        switch family {
        case .gemma4:
            guard let eot = tokenizer.convertTokenToId("<turn|>") else {
                throw GFTokenizerError.missingSpecialToken("<turn|>")
            }
            guard let toolResponse = tokenizer.convertTokenToId("<|tool_response>") else {
                throw GFTokenizerError.missingSpecialToken("<|tool_response>")
            }
            guard let toolCallStart = tokenizer.convertTokenToId("<|tool_call>"),
                  let toolCallEnd = tokenizer.convertTokenToId("<tool_call|>"),
                  let toolResponseEnd = tokenizer.convertTokenToId("<tool_response|>"),
                  let channelStart = tokenizer.convertTokenToId("<|channel>"),
                  let channelEnd = tokenizer.convertTokenToId("<channel|>") else {
                throw GFTokenizerError.missingSpecialToken("Gemma tool/channel markers")
            }
            self.endOfTurnID = Int32(eot)
            self.toolCallStartID = Int32(toolCallStart)
            self.toolCallEndID = Int32(toolCallEnd)
            self.toolResponseID = Int32(toolResponse)
            self.toolResponseEndID = Int32(toolResponseEnd)
            self.channelStartID = Int32(channelStart)
            self.channelEndID = Int32(channelEnd)
            self.stopTokenIDs = [self.eosID, self.endOfTurnID, self.toolResponseID]
            self.vocabSize = 262_144
        case .qwen3_6:
            // Qwen 3.6 special tokens (checkpoint tokenizer): <|endoftext|>
            // = 248044 (the model config's bos/pad/eos_token_id), <|im_start|>
            // = 248045, <|im_end|> = 248046 (the tokenizer config's eos_token).
            // generation_config.json stops on BOTH 248046 and 248044, so the
            // stop set carries both. The Gemma tool/channel markers do not
            // exist — -1 sentinels keep the shared consumers (server tool
            // streaming, structured decode) inert on this family.
            guard let eot = tokenizer.convertTokenToId("<|im_end|>"),
                  let endoftext = tokenizer.convertTokenToId("<|endoftext|>") else {
                throw GFTokenizerError.missingSpecialToken("<|im_end|>/<|endoftext|>")
            }
            self.endOfTurnID = Int32(eot)
            self.toolCallStartID = -1
            self.toolCallEndID = -1
            self.toolResponseID = -1
            self.toolResponseEndID = -1
            self.channelStartID = -1
            self.channelEndID = -1
            self.stopTokenIDs = [self.eosID, self.endOfTurnID, Int32(endoftext)]
            self.vocabSize = 248_320
        }
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        // Qwen has no standalone BOS — its templates frame turns explicitly.
        return addBOS && family == .gemma4 ? [bosID] + base : base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        tokenizer.decode(tokens: ids.map(Int.init), skipSpecialTokens: skipSpecialTokens)
    }

    // MARK: - Chat template

    public enum Role: String, Sendable { case system, developer, user, assistant, tool }
    public struct HistoricalToolCall: Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned IT checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    /// (Gemma framing; the Qwen branch renders the standard
    /// `<|im_start|>role\n…<|im_end|>` ChatML form.)
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            switch family {
            case .gemma4:
                let role = message.role == .assistant ? "model" : message.role.rawValue
                s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
            case .qwen3_6:
                // Qwen role names: user / assistant / system pass through.
                let role = message.role == .developer ? "system" : message.role.rawValue
                s += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n"
            }
        }
        switch family {
        case .gemma4:
            s = Self.bosMark + s + Self.turnOpen + "model\n<|channel>thought\n<channel|>"
        case .qwen3_6:
            // The checkpoint's template opens every assistant turn inside
            // <think> markers (enable_thinking defaults ON). With thinking
            // disabled it renders an EMPTY think block — the model then
            // answers directly. A bare "assistant\n" (no think block) is
            // out of distribution for this model and degenerates.
            s += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        }
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        return [endOfTurnID] + encode(
            "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
            addBOS: false)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools)
        let full = try encodeToolChat(messages: incomingMessages, tools: tools)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        guard matches.count == 1 else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = matches[0] + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .pretrained(let modelID):
                return try await GFTokenizer.loadUncached(pretrained: modelID)
            case .local(let path):
                return try await GFTokenizer.loadUncached(from: URL(fileURLWithPath: path))
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}

import Foundation
import QwenFieldfareRuntime

/// Implements the `/v1/chat/completions` request/response model, the Qwen3 chat
/// template, and JSON (de)serialization. Independent of the transport so it can
/// be unit-tested and reused.
public enum ChatCompletions {

    // MARK: - Request model

    public struct Message: Codable, Sendable {
        public var role: String
        public var content: String
        public init(role: String, content: String) {
            self.role = role
            self.content = content
        }
    }

    public struct Request: Codable, Sendable {
        public var model: String?
        public var messages: [Message]
        public var temperature: Float?
        public var top_p: Float?
        public var max_tokens: Int?
        public var stream: Bool?
        public var seed: UInt64?
    }

    // MARK: - Chat template (Qwen3 im_start/im_end)

    /// Renders the message list into the Qwen3 prompt string, ending with the
    /// assistant open tag so the model continues the assistant turn.
    public static func applyTemplate(_ messages: [Message]) -> String {
        var s = ""
        for m in messages {
            s += "<|im_start|>\(m.role)\n\(m.content)<|im_end|>\n"
        }
        s += "<|im_start|>assistant\n"
        return s
    }

    // MARK: - Response builders

    public static func chatId() -> String {
        "chatcmpl-" + UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24)
    }

    /// Non-streaming completion JSON.
    public static func completionJSON(id: String, model: String, content: String,
                                      promptTokens: Int, completionTokens: Int) -> Data {
        let obj: [String: Any] = [
            "id": id,
            "object": "chat.completion",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [[
                "index": 0,
                "message": ["role": "assistant", "content": content],
                "finish_reason": "stop"
            ]],
            "usage": [
                "prompt_tokens": promptTokens,
                "completion_tokens": completionTokens,
                "total_tokens": promptTokens + completionTokens
            ]
        ]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }

    /// A single SSE streaming chunk (delta).
    public static func streamChunkJSON(id: String, model: String,
                                       delta: String?, finish: String?) -> Data {
        var choice: [String: Any] = ["index": 0]
        if let delta {
            choice["delta"] = ["content": delta]
        } else {
            choice["delta"] = [:]
        }
        if let finish { choice["finish_reason"] = finish }
        let obj: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [choice]
        ]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }

    /// First streaming chunk carrying the assistant role.
    public static func streamRoleChunkJSON(id: String, model: String) -> Data {
        let obj: [String: Any] = [
            "id": id,
            "object": "chat.completion.chunk",
            "created": Int(Date().timeIntervalSince1970),
            "model": model,
            "choices": [["index": 0, "delta": ["role": "assistant"], "finish_reason": NSNull()]]
        ]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }

    /// `GET /v1/models` response.
    public static func modelsJSON(modelId: String) -> Data {
        let obj: [String: Any] = [
            "object": "list",
            "data": [[
                "id": modelId,
                "object": "model",
                "created": Int(Date().timeIntervalSince1970),
                "owned_by": "qwen-fieldfare"
            ]]
        ]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
    }

    public static func decodeRequest(_ data: Data) -> Request? {
        try? JSONDecoder().decode(Request.self, from: data)
    }
}

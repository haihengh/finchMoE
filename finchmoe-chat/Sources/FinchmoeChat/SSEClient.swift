import Foundation

// SSE client for the FinchMoE OpenAI-compatible server (/v1/chat/completions).
// The engine's SSE stream: `data: {"choices":[{"delta":{"content":"..."}}]}` ... `data: [DONE]`.

struct ChatChunk: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable { let content: String? }
        let delta: Delta?
        let finish_reason: String?
    }
    let choices: [Choice]?
}

struct ChatUsage: Decodable {
    let prompt_tokens: Int?
    let completion_tokens: Int?
    let tokens_per_second: Double?
}

final class SSEClient: NSObject, URLSessionDataDelegate {
    private var session: URLSession!
    private var buffer = Data()
    var onDelta: ((String) -> Void)?
    var onFinish: ((ChatUsage?) -> Void)?
    var onError: ((String) -> Void)?

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 3600
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func streamChat(baseURL: URL, messages: [[String: String]], sessionID: String,
                    temperature: Double, maxTokens: Int, isCompletion: Bool = false) {
        var request = URLRequest(url: baseURL.appendingPathComponent(isCompletion ? "/v1/completions" : "/v1/chat/completions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")

        var body: [String: Any] = [
            "messages": messages,
            "session_id": sessionID,
            "max_tokens": maxTokens,
            "temperature": temperature,
        ]
        if isCompletion { body.removeValue(forKey: "messages") }
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        buffer = Data()
        session.dataTask(with: request).resume()
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        buffer.append(data)
        // Parse complete SSE events (separated by \n\n).
        while let range = buffer.range(of: Data("\n\n".utf8)) {
            let event = buffer.subdata(in: buffer.startIndex..<range.lowerBound)
            buffer.removeSubrange(buffer.startIndex..<range.upperBound)
            handleEvent(event)
        }
    }

    private func handleEvent(_ event: Data) {
        guard let text = String(data: event, encoding: .utf8) else { return }
        for line in text.components(separatedBy: "\n") {
            guard line.hasPrefix("data: ") else { continue }
            let payload = String(line.dropFirst(6))
            if payload == "[DONE]" { return }
            guard let data = payload.data(using: .utf8) else { continue }
            if let chunk = try? JSONDecoder().decode(ChatChunk.self, from: data),
               let content = chunk.choices?.first?.delta?.content {
                onDelta?(content)
            }
            // usage rides on the final chunk
            if let full = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let usage = full["usage"] as? [String: Any] {
                let u = ChatUsage(
                    prompt_tokens: usage["prompt_tokens"] as? Int,
                    completion_tokens: usage["completion_tokens"] as? Int,
                    tokens_per_second: usage["tokens_per_second"] as? Double)
                onFinish?(u)
            }
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            onError?(error.localizedDescription)
        } else {
            onFinish?(nil)
        }
    }
}

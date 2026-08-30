import Foundation
import Network
import QwenFieldfareRuntime

/// OpenAI-compatible HTTP/1.1 server built on `Network.framework` (NWListener).
/// Parses requests manually (no external deps) and serves:
///   - `POST /v1/chat/completions` (streaming SSE + non-streaming)
///   - `GET  /v1/models`
///
/// The underlying engine holds shared scratch buffers, so generation is
/// serialized on a dedicated queue; connections are accepted concurrently but
/// only one forward pass runs at a time.
public final class OpenAIServer {

    public enum ServerError: Error, CustomStringConvertible {
        case listenerFailed(String)
        public var description: String {
            switch self { case .listenerFailed(let s): return "OpenAIServer: \(s)" }
        }
    }

    private let engine: InferenceEngine
    private let host: String
    private let port: UInt16
    private let modelId: String

    private var listener: NWListener?
    private let acceptQueue = DispatchQueue(label: "qwen.server.accept")
    private let genQueue = DispatchQueue(label: "qwen.server.generate")

    public init(engine: InferenceEngine, host: String = "127.0.0.1",
                port: UInt16 = 11434, modelId: String = "qwen3-30b-a3b") throws {
        self.engine = engine
        self.host = host
        self.port = port
        self.modelId = modelId
    }

    public func start() throws {
        let params = NWParameters.tcp
        params.allowLocalEndpointReuse = true
        guard let nwPort = NWEndpoint.Port(rawValue: port) else {
            throw ServerError.listenerFailed("invalid port \(port)")
        }
        // Bind to host if provided.
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: nwPort)

        let listener: NWListener
        do {
            listener = try NWListener(using: params)
        } catch {
            throw ServerError.listenerFailed("\(error)")
        }
        self.listener = listener

        listener.newConnectionHandler = { [weak self] conn in
            self?.handle(conn)
        }
        listener.stateUpdateHandler = { state in
            if case .failed(let e) = state {
                FileHandle.standardError.write(Data("Listener failed: \(e)\n".utf8))
            }
        }
        listener.start(queue: acceptQueue)
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func handle(_ conn: NWConnection) {
        conn.start(queue: acceptQueue)
        receiveRequest(conn, accumulated: Data())
    }

    /// Accumulates until headers + body (based on Content-Length) are complete.
    private func receiveRequest(_ conn: NWConnection, accumulated: Data) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 1 << 16) { [weak self] data, _, isComplete, error in
            guard let self else { return }
            var buffer = accumulated
            if let data { buffer.append(data) }

            if let parsed = HTTPRequest.tryParse(buffer) {
                self.route(parsed, on: conn)
                return
            }
            if isComplete || error != nil {
                self.sendSimple(conn, status: "400 Bad Request", body: Data("bad request".utf8))
                return
            }
            self.receiveRequest(conn, accumulated: buffer)
        }
    }

    private func route(_ req: HTTPRequest, on conn: NWConnection) {
        switch (req.method, req.path) {
        case ("GET", "/v1/models"):
            let body = ChatCompletions.modelsJSON(modelId: modelId)
            sendJSON(conn, body: body)

        case ("GET", "/"), ("GET", "/health"):
            sendJSON(conn, body: Data(#"{"status":"ok"}"#.utf8))

        case ("POST", "/v1/chat/completions"):
            handleChat(req, on: conn)

        case ("OPTIONS", _):
            sendSimple(conn, status: "204 No Content", body: Data())

        default:
            sendSimple(conn, status: "404 Not Found", body: Data(#"{"error":"not found"}"#.utf8),
                       contentType: "application/json")
        }
    }

    // MARK: - Chat completions

    private func handleChat(_ req: HTTPRequest, on conn: NWConnection) {
        guard let chatReq = ChatCompletions.decodeRequest(req.body) else {
            sendSimple(conn, status: "400 Bad Request",
                       body: Data(#"{"error":"invalid JSON"}"#.utf8), contentType: "application/json")
            return
        }
        let prompt = ChatCompletions.applyTemplate(chatReq.messages)
        let opts = InferenceEngine.GenerationOptions(
            maxTokens: chatReq.max_tokens ?? 512,
            temperature: chatReq.temperature ?? 0.7,
            topP: chatReq.top_p ?? 0.9,
            seed: chatReq.seed)
        let streaming = chatReq.stream ?? false
        let id = ChatCompletions.chatId()

        if streaming {
            streamChat(prompt: prompt, opts: opts, id: id, on: conn)
        } else {
            blockingChat(prompt: prompt, opts: opts, id: id, on: conn)
        }
    }

    private func blockingChat(prompt: String, opts: InferenceEngine.GenerationOptions,
                              id: String, on conn: NWConnection) {
        genQueue.async { [weak self] in
            guard let self else { return }
            var text = ""
            var completionTokens = 0
            do {
                text = try self.engine.generateText(prompt: prompt, options: opts) { _ in
                    completionTokens += 1
                }
            } catch {
                self.sendSimple(conn, status: "500 Internal Server Error",
                                body: Data("{\"error\":\"\(error)\"}".utf8),
                                contentType: "application/json")
                return
            }
            let promptTokens = self.engine.tokenizer?.encode(prompt).count ?? 0
            let body = ChatCompletions.completionJSON(id: id, model: self.modelId, content: text,
                                                      promptTokens: promptTokens,
                                                      completionTokens: completionTokens)
            self.sendJSON(conn, body: body)
        }
    }

    private func streamChat(prompt: String, opts: InferenceEngine.GenerationOptions,
                            id: String, on conn: NWConnection) {
        // Send SSE headers first.
        let header = """
        HTTP/1.1 200 OK\r
        Content-Type: text/event-stream\r
        Cache-Control: no-cache\r
        Connection: keep-alive\r
        Access-Control-Allow-Origin: *\r
        \r

        """
        conn.send(content: Data(header.utf8), completion: .contentProcessed { _ in })

        // Role chunk.
        sendSSE(conn, data: ChatCompletions.streamRoleChunkJSON(id: id, model: modelId))

        genQueue.async { [weak self] in
            guard let self else { return }
            do {
                _ = try self.engine.generateText(prompt: prompt, options: opts) { piece in
                    let chunk = ChatCompletions.streamChunkJSON(id: id, model: self.modelId,
                                                                delta: piece, finish: nil)
                    self.sendSSE(conn, data: chunk)
                }
            } catch {
                let err = ChatCompletions.streamChunkJSON(id: id, model: self.modelId,
                                                          delta: "\n[error: \(error)]", finish: "stop")
                self.sendSSE(conn, data: err)
            }
            // Final finish chunk + [DONE].
            let done = ChatCompletions.streamChunkJSON(id: id, model: self.modelId, delta: nil, finish: "stop")
            self.sendSSE(conn, data: done)
            conn.send(content: Data("data: [DONE]\r\n\r\n".utf8), completion: .contentProcessed { _ in
                conn.cancel()
            })
        }
    }

    // MARK: - Low-level send helpers

    private func sendSSE(_ conn: NWConnection, data: Data) {
        var payload = Data("data: ".utf8)
        payload.append(data)
        payload.append(Data("\r\n\r\n".utf8))
        conn.send(content: payload, completion: .contentProcessed { _ in })
    }

    private func sendJSON(_ conn: NWConnection, body: Data) {
        sendSimple(conn, status: "200 OK", body: body, contentType: "application/json")
    }

    private func sendSimple(_ conn: NWConnection, status: String, body: Data,
                            contentType: String = "text/plain") {
        var head = "HTTP/1.1 \(status)\r\n"
        head += "Content-Type: \(contentType)\r\n"
        head += "Content-Length: \(body.count)\r\n"
        head += "Access-Control-Allow-Origin: *\r\n"
        head += "Access-Control-Allow-Headers: *\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(body)
        conn.send(content: payload, completion: .contentProcessed { _ in
            conn.cancel()
        })
    }
}

/// Minimal HTTP/1.1 request parser (method, path, headers, body).
struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data

    /// Attempts to parse a full request from `data`. Returns nil if incomplete.
    static func tryParse(_ data: Data) -> HTTPRequest? {
        // Find header/body separator.
        guard let sep = data.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let headerData = data.subdata(in: data.startIndex..<sep.lowerBound)
        guard let headerString = String(data: headerData, encoding: .utf8) else { return nil }

        let lines = headerString.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }
        let comps = requestLine.split(separator: " ")
        guard comps.count >= 2 else { return nil }
        let method = String(comps[0])
        let path = String(comps[1])

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            if let idx = line.firstIndex(of: ":") {
                let key = line[line.startIndex..<idx].trimmingCharacters(in: .whitespaces).lowercased()
                let value = line[line.index(after: idx)...].trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }

        let bodyStart = sep.upperBound
        let available = data.subdata(in: bodyStart..<data.endIndex)
        if let lenStr = headers["content-length"], let len = Int(lenStr) {
            if available.count < len { return nil } // wait for full body
            return HTTPRequest(method: method, path: path, headers: headers,
                               body: available.prefix(len))
        }
        // No body expected (e.g. GET).
        return HTTPRequest(method: method, path: path, headers: headers, body: available)
    }
}

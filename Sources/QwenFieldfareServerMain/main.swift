import Foundation
import QwenFieldfareRuntime
import QwenFieldfareServer

// Standalone OpenAI-compatible server executable.
//
//   qwen-fieldfare-server --model <path.qturbo> [--port 11434] [--host 127.0.0.1]
//                         [--max-seq N] [--cache-slots N]

func value(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
    return args[idx + 1]
}

guard let modelPath = value("--model") else {
    FileHandle.standardError.write(Data("usage: qwen-fieldfare-server --model <path.qturbo> [--port N] [--host H]\n".utf8))
    exit(2)
}

let port = UInt16(value("--port") ?? "11434") ?? 11434
let host = value("--host") ?? "127.0.0.1"
let maxSeq = Int(value("--max-seq") ?? "32768") ?? 32768
let slots = Int(value("--cache-slots") ?? "16") ?? 16

do {
    let engine = try InferenceEngine(modelDir: URL(fileURLWithPath: modelPath),
                                     maxSeqLen: maxSeq, expertCacheSlots: slots)
    let server = try OpenAIServer(engine: engine, host: host, port: port)
    try server.start()
    FileHandle.standardError.write(Data("qwen-fieldfare-server listening on http://\(host):\(port)\n".utf8))
    dispatchMain()
} catch {
    FileHandle.standardError.write(Data("Server failed: \(error)\n".utf8))
    exit(1)
}

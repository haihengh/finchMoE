import Foundation
import QwenFieldfareFormat
import QwenFieldfareRepack
import QwenFieldfareRuntime
import QwenFieldfareServer

// =============================================================================
// qwen-fieldfare — CLI entry point.
//
//   qwen-fieldfare repack --source <hf-cache-or-dir> --output <path> [--download]
//   qwen-fieldfare run    --model <path> --prompt "..." [--max-tokens N]
//                         [--temperature T] [--top-p P] [--raw]
//   qwen-fieldfare serve  --model <path> [--port N] [--host H]
// =============================================================================

struct ArgParser {
    private var args: [String]
    init(_ args: [String]) { self.args = args }

    func value(_ name: String) -> String? {
        guard let idx = args.firstIndex(of: name), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }
    func flag(_ name: String) -> Bool { args.contains(name) }
}

func printUsage() {
    let usage = """
    qwen-fieldfare — Qwen3-30B-A3B inference engine (Swift + Metal)

    USAGE:
      qwen-fieldfare repack --source <dir> --output <path.qturbo> [--download] [--repo <id>]
      qwen-fieldfare run    --model <path.qturbo> --prompt "text"
                            [--max-tokens N] [--temperature T] [--top-p P]
                            [--max-seq N] [--cache-slots N] [--raw] [--seed S]
      qwen-fieldfare serve  --model <path.qturbo> [--port 11434] [--host 127.0.0.1]
                            [--max-seq N] [--cache-slots N]

    COMMANDS:
      repack   Download (optional) and convert MLX 4-bit weights to .qturbo.
      run      Run a single prompt through the model and print the completion.
      serve    Start an OpenAI-compatible HTTP server.
    """
    print(usage)
}

// Qwen3 chat template applied to a single user prompt (used by `run` unless --raw).
func applyChatTemplate(system: String?, user: String) -> String {
    var s = ""
    if let system, !system.isEmpty {
        s += "<|im_start|>system\n\(system)<|im_end|>\n"
    }
    s += "<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    return s
}

func runRepack(_ p: ArgParser) async {
    guard let source = p.value("--source"), let output = p.value("--output") else {
        FileHandle.standardError.write(Data("repack requires --source and --output\n".utf8))
        exit(2)
    }
    let repo = p.value("--repo") ?? "mlx-community/Qwen3-30B-A3B-4bit"
    let cmd = RepackCommand(sourceDir: URL(fileURLWithPath: source),
                            outputDir: URL(fileURLWithPath: output),
                            download: p.flag("--download"),
                            repo: repo)
    do {
        try await cmd.run()
    } catch {
        FileHandle.standardError.write(Data("Repack failed: \(error)\n".utf8))
        exit(1)
    }
}

func runInference(_ p: ArgParser) {
    guard let modelPath = p.value("--model") else {
        FileHandle.standardError.write(Data("run requires --model\n".utf8))
        exit(2)
    }
    let prompt = p.value("--prompt") ?? "Hello!"
    let maxTokens = Int(p.value("--max-tokens") ?? "512") ?? 512
    let temperature = Float(p.value("--temperature") ?? "0.7") ?? 0.7
    let topP = Float(p.value("--top-p") ?? "0.9") ?? 0.9
    let maxSeq = Int(p.value("--max-seq") ?? "32768") ?? 32768
    let slots = Int(p.value("--cache-slots") ?? "16") ?? 16
    let seed = p.value("--seed").flatMap { UInt64($0) }
    let raw = p.flag("--raw")

    do {
        let engine = try InferenceEngine(modelDir: URL(fileURLWithPath: modelPath),
                                         maxSeqLen: maxSeq, expertCacheSlots: slots)
        let finalPrompt = raw ? prompt : applyChatTemplate(system: p.value("--system"), user: prompt)
        let opts = InferenceEngine.GenerationOptions(maxTokens: maxTokens,
                                                     temperature: temperature,
                                                     topP: topP, seed: seed)
        FileHandle.standardError.write(Data("Generating…\n".utf8))
        _ = try engine.generateText(prompt: finalPrompt, options: opts) { piece in
            FileHandle.standardOutput.write(Data(piece.utf8))
        }
        print("")
    } catch {
        FileHandle.standardError.write(Data("Inference failed: \(error)\n".utf8))
        exit(1)
    }
}

func runServe(_ p: ArgParser) async {
    guard let modelPath = p.value("--model") else {
        FileHandle.standardError.write(Data("serve requires --model\n".utf8))
        exit(2)
    }
    let port = UInt16(p.value("--port") ?? "11434") ?? 11434
    let host = p.value("--host") ?? "127.0.0.1"
    let maxSeq = Int(p.value("--max-seq") ?? "32768") ?? 32768
    let slots = Int(p.value("--cache-slots") ?? "16") ?? 16

    do {
        let engine = try InferenceEngine(modelDir: URL(fileURLWithPath: modelPath),
                                         maxSeqLen: maxSeq, expertCacheSlots: slots)
        let server = try OpenAIServer(engine: engine, host: host, port: port)
        try server.start()
        FileHandle.standardError.write(Data("Serving on http://\(host):\(port)\n".utf8))
        // Process is kept alive by the outer dispatchMain() in the command switch.
    } catch {
        FileHandle.standardError.write(Data("Server failed: \(error)\n".utf8))
        exit(1)
    }
}

// MARK: - Dispatch

let allArgs = Array(CommandLine.arguments.dropFirst())
guard let command = allArgs.first else {
    printUsage()
    exit(0)
}
let parser = ArgParser(Array(allArgs.dropFirst()))

switch command {
case "repack":
    let sem = DispatchSemaphore(value: 0)
    Task { await runRepack(parser); sem.signal() }
    sem.wait()
case "run":
    runInference(parser)
case "serve":
    Task { await runServe(parser) }
    dispatchMain()
case "-h", "--help", "help":
    printUsage()
default:
    FileHandle.standardError.write(Data("Unknown command: \(command)\n".utf8))
    printUsage()
    exit(2)
}

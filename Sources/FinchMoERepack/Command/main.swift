import Foundation
import FinchMoERepackCore

private let usage = """
Usage:
  FinchMoERepack --output <model.finch> [--overwrite] [--resume]
  FinchMoERepack --input-snapshot <dir> --output <model.finch> [--overwrite]
  FinchMoERepack --discard-partial --output <model.finch>
  FinchMoERepack --verify-install --input-finch <model.finch>
  FinchMoERepack --help

Without --input-snapshot, the installer streams the supported Gemma 4
checkpoint from Hugging Face and repackages it without materializing the
source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
authentication. A cancelled or interrupted download can be continued with
--resume or removed with --discard-partial.

With --input-snapshot, the installer quantizes a LOCAL bf16 Qwen 3.6 35B-A3B
safetensors snapshot (int4 affine, group 64) into the .finch format.
"""

private struct Arguments {
    var output: String?
    var inputSnapshot: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputFinch: String?

    static func parse(_ values: [String]) throws -> Arguments {
        var parsed = Arguments()
        var index = 1
        while index < values.count {
            let flag = values[index]
            switch flag {
            case "--help":
                throw ParseError.help
            case "--overwrite":
                parsed.overwrite = true
                index += 1
            case "--resume":
                parsed.resume = true
                index += 1
            case "--discard-partial":
                parsed.discardPartial = true
                index += 1
            case "--verify-install":
                parsed.verifyInstall = true
                index += 1
            case "--output", "--input-finch", "--input-snapshot":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                if flag == "--output" {
                    parsed.output = values[index + 1]
                } else if flag == "--input-snapshot" {
                    parsed.inputSnapshot = values[index + 1]
                } else {
                    parsed.inputFinch = values[index + 1]
                }
                index += 2
            default:
                throw ParseError.unknown(flag)
            }
        }

        guard !(parsed.resume && parsed.discardPartial) else {
            throw ParseError.invalidMode("--resume and --discard-partial are mutually exclusive")
        }
        if parsed.discardPartial {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputFinch == nil, !parsed.overwrite, !parsed.verifyInstall,
                  parsed.inputSnapshot == nil else {
                throw ParseError.invalidMode("--discard-partial only accepts --output")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputFinch != nil else {
                throw ParseError.missingRequired("--input-finch")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  parsed.inputSnapshot == nil else {
                throw ParseError.invalidMode("verification accepts only --input-finch")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputFinch == nil else {
                throw ParseError.invalidMode("--input-finch requires --verify-install")
            }
            if let snapshot = parsed.inputSnapshot {
                guard !parsed.resume else {
                    throw ParseError.invalidMode("--resume applies to remote downloads only")
                }
                guard try Posix.entryKind((snapshot as NSString)
                        .appendingPathComponent("model.safetensors.index.json")) == .regular else {
                    throw ParseError.invalidMode("snapshot directory has no model.safetensors.index.json")
                }
            }
        }
        return parsed
    }
}

private enum ParseError: Error, CustomStringConvertible {
    case help
    case unknown(String)
    case missingValue(String)
    case missingRequired(String)
    case invalidMode(String)

    var description: String {
        switch self {
        case .help: return "help"
        case .unknown(let flag): return "unknown argument: \(flag)"
        case .missingValue(let flag): return "missing value for \(flag)"
        case .missingRequired(let flag): return "missing required argument: \(flag)"
        case .invalidMode(let message): return message
        }
    }
}

private func printError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func run(_ values: [String]) async -> Int32 {
    let arguments: Arguments
    do {
        arguments = try Arguments.parse(values)
    } catch ParseError.help {
        print(usage)
        return 0
    } catch {
        printError("error: \(error)\n\n\(usage)")
        return 2
    }

    if arguments.discardPartial, let output = arguments.output {
        do {
            try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            print("Discarded saved download for \(output)")
            return 0
        } catch {
            printError("discard failed: \(error)")
            return 1
        }
    }

    if arguments.verifyInstall, let input = arguments.inputFinch {
        do {
            let result = try VerifiedInstallTool.run(
                options: VerifyInstallOptions(inputFinch: input))
            print("Verified \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            return 0
        } catch {
            printError("verification failed: \(error)")
            return 1
        }
    }

    guard let output = arguments.output else { return 2 }

    if let snapshot = arguments.inputSnapshot {
        let options = LocalQwenRepackOptions(
            snapshotDir: snapshot,
            outputDir: URL(fileURLWithPath: output).path,
            overwrite: arguments.overwrite)
        do {
            let result = try await LocalQwenRepacker(options: options).run()
            print("Repacked Qwen 3.6 35B-A3B bf16 snapshot (\(snapshot))")
            print("Output bytes: \(result.outputBytes)")
            print("Dropped non-text tensors: \(result.excludedTensorCount)")
            print("Model: \(result.outputDir)")
            return 0
        } catch {
            printError("install failed: \(error)")
            return 1
        }
    }

    let options = SupportedModelSource.installOptions(
        outputDirectory: URL(fileURLWithPath: output),
        overwrite: arguments.overwrite,
        token: ProcessInfo.processInfo.environment["HF_TOKEN"],
        resume: arguments.resume)
    do {
        let result = try await RemoteStreamingRepacker(options: options).run()
        print("Installed \(SupportedModelSource.displayName)")
        print("Source revision: \(result.resolvedCommit)")
        print("Model: \(result.outputDir)")
        return 0
    } catch {
        printError("install failed: \(error)")
        return 1
    }
}

exit(await run(CommandLine.arguments))

import Foundation
import FinchMoERepackCore

private let usage = """
Usage:
  FinchMoERepack --output <model.finch> [--overwrite] [--resume]
  FinchMoERepack --input-snapshot <dir> --output <model.finch> [--overwrite]
                 [--routed-expert-bits 2|3|4]
  FinchMoERepack --download-finch <owner/name> --output <model.finch>
                 [--revision <commit>] [--concurrency <n>]
  FinchMoERepack --discard-partial --output <model.finch>
  FinchMoERepack --verify-install --input-finch <model.finch>
  FinchMoERepack --help

Without --input-snapshot, the installer streams the supported Gemma 4
checkpoint from Hugging Face and repackages it without materializing the
source checkpoint on disk. Set HF_TOKEN only if Hugging Face requests
authentication. A cancelled or interrupted download can be continued with
--resume or removed with --discard-partial.

With --input-snapshot, the installer quantizes a LOCAL bf16 Qwen 3.6 35B-A3B
or Qwen 3.8 Flash-Next safetensors snapshot (affine, group 64) into the
.finch format. --routed-expert-bits selects the routed-expert width: 4
(default) or 3 (24-bit triplets, ~25% smaller expert blobs; the runtime
decodes both, and the manifest records which). An interrupted run of either kind can be continued with
--resume, which reuses the output files the partial directory's journal
records as complete and rewrites the rest; a partial whose journal is missing
or describes a different source is refused rather than guessed at.

With --download-finch, the installer fetches an already-repacked .finch
directory published on Hugging Face, such as
haihengh/Qwen3.6-35B-A3B-finchmoe-4bit-abliterated. This is not a repack: the
remote manifest names every file with its size and digest, so the download is
exactly the files it lists, each verified against the digest that named it, and
the receipt is written locally for the path installed here. An interrupted
download resumes automatically when the checkpoint still describes the same
repo, commit and manifest; otherwise it is refused. Use --discard-partial to
remove a partial download instead.
"""

private struct Arguments {
    var output: String?
    var inputSnapshot: String?
    var overwrite = false
    var resume = false
    var discardPartial = false
    var verifyInstall = false
    var inputFinch: String?
    var downloadFinch: String?
    var revision: String?
    var concurrency: Int?
    var routedExpertBits: Int?

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
            case "--output", "--input-finch", "--input-snapshot",
                 "--download-finch", "--revision", "--concurrency",
                 "--routed-expert-bits":
                guard index + 1 < values.count else {
                    throw ParseError.missingValue(flag)
                }
                let value = values[index + 1]
                switch flag {
                case "--output":         parsed.output = value
                case "--input-snapshot": parsed.inputSnapshot = value
                case "--input-finch":    parsed.inputFinch = value
                case "--download-finch": parsed.downloadFinch = value
                case "--revision":       parsed.revision = value
                case "--routed-expert-bits":
                    guard value == "2" || value == "3" || value == "4" else {
                        throw ParseError.invalidMode(
                            "--routed-expert-bits wants 2, 3 or 4, got \(value)")
                    }
                    parsed.routedExpertBits = Int(value)
                default:
                    guard let n = Int(value), n > 0 else {
                        throw ParseError.invalidMode(
                            "--concurrency wants a positive integer, got \(value)")
                    }
                    parsed.concurrency = n
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
            guard parsed.revision == nil, parsed.concurrency == nil else {
                throw ParseError.invalidMode(
                    "--discard-partial does not take --revision or --concurrency")
            }
            return parsed
        }
        if parsed.verifyInstall {
            guard parsed.inputFinch != nil else {
                throw ParseError.missingRequired("--input-finch")
            }
            guard parsed.output == nil, !parsed.overwrite, !parsed.resume,
                  parsed.inputSnapshot == nil, parsed.downloadFinch == nil,
                  parsed.revision == nil, parsed.concurrency == nil else {
                throw ParseError.invalidMode("verification accepts only --input-finch")
            }
        } else {
            guard parsed.output != nil else {
                throw ParseError.missingRequired("--output")
            }
            guard parsed.inputFinch == nil else {
                throw ParseError.invalidMode("--input-finch requires --verify-install")
            }
            guard !(parsed.downloadFinch != nil && parsed.inputSnapshot != nil) else {
                throw ParseError.invalidMode(
                    "--download-finch and --input-snapshot are different install routes")
            }
            guard parsed.downloadFinch != nil || (parsed.revision == nil && parsed.concurrency == nil) else {
                throw ParseError.invalidMode(
                    "--revision and --concurrency require --download-finch")
            }
            if let repo = parsed.downloadFinch {
                let parts = repo.split(separator: "/")
                guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty }) else {
                    throw ParseError.invalidMode(
                        "--download-finch wants a Hugging Face repository id "
                            + "of the form owner/name, got \(repo)")
                }
            }
            if let snapshot = parsed.inputSnapshot {
                guard try Posix.entryKind((snapshot as NSString)
                        .appendingPathComponent("model.safetensors.index.json")) == .regular else {
                    throw ParseError.invalidMode("snapshot directory has no model.safetensors.index.json")
                }
            }
            guard parsed.routedExpertBits == nil || parsed.inputSnapshot != nil else {
                throw ParseError.invalidMode(
                    "--routed-expert-bits requires --input-snapshot")
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

/// Draws install progress as a single updating stderr line.
///
/// A class rather than file-scope state because `copyingPayload` arrives from
/// every download worker at once: the lock serializes the writes, and the
/// throttle keeps a fast link from spending more time formatting than
/// transferring.
private final class ProgressReporter: @unchecked Sendable {
    private let lock = NSLock()
    private var lastDraw = Date.distantPast
    private var lineIsOpen = false

    func report(_ event: ModelInstallProgress) {
        switch event {
        case .downloadingMetadata:
            write(note: "Fetching manifest…")
        case .planning(let downloadBytes, _):
            write(note: String(format: "Downloading %.2f GiB of files",
                               Double(downloadBytes) / 1_073_741_824))
        case .checkingDisk(let requirement):
            write(note: String(format: "Checking disk: %.2f GiB free, %.2f GiB needed",
                               Double(requirement.availableBytes) / 1_073_741_824,
                               Double(requirement.requiredBytes) / 1_073_741_824))
        case .copyingPayload(let reusedBytes, let downloadedThisRunBytes, let totalBytes):
            write(progress: reusedBytes + downloadedThisRunBytes,
                  of: totalBytes,
                  prefix: reusedBytes > 0 ? "Downloading (resumed)" : "Downloading")
        case .hashingOutput:
            break   // 189 individual filenames would drown the line above
        case .reservingOutput, .finalizing:
            write(note: "Verifying and finalizing…")
        }
    }

    /// Ends the updating line so following output starts clean.
    func finish() {
        lock.lock()
        defer { lock.unlock() }
        guard lineIsOpen else { return }
        lineIsOpen = false
        FileHandle.standardError.write(Data("\n".utf8))
    }

    private func write(note: String) {
        lock.lock()
        defer { lock.unlock() }
        endLineLocked()
        FileHandle.standardError.write(Data(("  " + note + "\n").utf8))
    }

    private func write(progress done: UInt64, of total: UInt64, prefix: String) {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        let complete = total > 0 && done >= total
        guard complete || now.timeIntervalSince(lastDraw) >= 0.5 else { return }
        lastDraw = now
        let fraction = total > 0 ? Double(done) / Double(total) : 0
        let line = String(format: "\r  %@ %.2f/%.2f GiB (%.1f%%)   ",
                          prefix,
                          Double(done) / 1_073_741_824,
                          Double(total) / 1_073_741_824,
                          fraction * 100)
        FileHandle.standardError.write(Data(line.utf8))
        lineIsOpen = true
    }

    private func endLineLocked() {
        guard lineIsOpen else { return }
        lineIsOpen = false
        FileHandle.standardError.write(Data("\n".utf8))
    }
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
            if arguments.downloadFinch != nil {
                try FinchDistributionDownloader.discardPartial(outputDirectory: output)
            } else {
                try RemoteStreamingRepacker.discardPartial(outputDirectory: output)
            }
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

    if let repo = arguments.downloadFinch {
        let distribution = FinchDistribution(repoID: repo,
                                             revision: arguments.revision ?? "main",
                                             approximateDownloadBytes: 0,
                                             installedBytes: 0)
        let reporter = ProgressReporter()
        do {
            let result = try await FinchDistributionDownloader.run(
                source: distribution,
                outputDirectory: URL(fileURLWithPath: output).path,
                token: ProcessInfo.processInfo.environment["HF_TOKEN"],
                concurrency: arguments.concurrency ?? FinchDistributionDownloader.defaultConcurrency,
                progress: { reporter.report($0) })
            reporter.finish()
            print("")
            print("Downloaded \(result.fileCount) files (\(result.bytesVerified) bytes)")
            print("Receipt: \(result.receiptPath)")
            if !result.unexpectedEntries.isEmpty {
                print("Unexpected entries: \(result.unexpectedEntries.joined(separator: ", "))")
            }
            print("Model: \(output)")
            return 0
        } catch {
            reporter.finish()
            printError("download failed: \(error)")
            return 1
        }
    }

    if let snapshot = arguments.inputSnapshot {
        let options = LocalQwenRepackOptions(
            snapshotDir: snapshot,
            outputDir: URL(fileURLWithPath: output).path,
            overwrite: arguments.overwrite,
            resume: arguments.resume,
            routedExpertBits: arguments.routedExpertBits ?? 4)
        do {
            let result = try await LocalQwenRepacker(options: options).run()
            print("Repacked bf16 snapshot (\(snapshot))")
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

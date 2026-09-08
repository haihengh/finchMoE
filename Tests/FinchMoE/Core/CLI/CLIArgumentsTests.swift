import Testing
@testable import FinchMoECLICore
@testable import FinchMoE

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.finchturbo", "--prompt", "hi"])
        #expect(arguments.model == "m.finchturbo")
        #expect(arguments.prompt == "hi")
        #expect(arguments.messagesFile == nil)
        #expect(arguments.maxNew == 1_024)
        #expect(arguments.maxContext == 4096)
        #expect(arguments.temperature == 0.2)
        #expect(arguments.topK == 64)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1)
        #expect(arguments.seed == nil)
        #expect(arguments.stops.isEmpty)
        #expect(!arguments.quiet)
        #expect(arguments.verify == .fullSha256)
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.finchturbo", "--prompt", "hi",
            "--max-new", "32", "--max-context", "512",
            "--temperature", "0", "--top-k", "40", "--top-p", "0.95",
            "--repetition-penalty", "1.1", "--seed", "42",
            "--stop", "A", "--stop", "B", "--quiet",
        ])
        #expect(arguments.maxNew == 32)
        #expect(arguments.maxContext == 512)
        #expect(arguments.temperature == 0)
        #expect(arguments.topK == 40)
        #expect(arguments.topP == 0.95)
        #expect(arguments.repetitionPenalty == 1.1)
        #expect(arguments.seed == 42)
        #expect(arguments.stops == ["A", "B"])
        #expect(arguments.quiet)
    }

    @Test func topKZeroRequiresTopPToBeDisabled() throws {
        let disabled = try Args.parse([
            "--model", "m.finchturbo", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.finchturbo", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.finchturbo", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--verify", "--help",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.finchturbo", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.finchturbo"])
        }
    }

    @Test func verifyPolicyParsesAndRejectsUnknownModes() throws {
        let full = try Args.parse(["--model", "m.finchturbo", "--prompt", "hi",
                                    "--verify", "full-sha256"])
        #expect(full.verify == .fullSha256)
        let trusted = try Args.parse(["--model", "m.finchturbo", "--prompt", "hi",
                                      "--verify", "trusted-install"])
        #expect(trusted.verify == .sizeCheckTrustedReceipt)

        #expect(throws: ArgsError.invalidValue(flag: "--verify", value: "size-only")) {
            _ = try Args.parse(["--model", "m.finchturbo", "--prompt", "hi",
                                "--verify", "size-only"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.finchturbo", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.finchturbo", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
}

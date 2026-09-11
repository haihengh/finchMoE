import Testing
@testable import FinchMoECLICore
@testable import FinchMoE

@Suite struct CLIArgumentsTests {
    @Test func defaultsUseProductionGenerationValues() throws {
        let arguments = try Args.parse(["--model", "m.finch", "--prompt", "hi"])
        #expect(arguments.model == "m.finch")
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
        #expect(!arguments.counters)
        #expect(arguments.expertCacheSlots == nil)
        #expect(arguments.verify == .automatic)
    }

    /// All three modes parse, and the two explicit ones keep their meaning.
    /// `auto` being the default is what the test above pins; this pins that it
    /// is reachable by name and that spelling an explicit mode still wins.
    ///
    /// The rejection asserts the exact `ArgsError`, not just that *something*
    /// threw: `--verify` is the one flag whose value the loader resolves into a
    /// security policy, so a typo silently landing on one of the two real modes
    /// is the failure worth pinning.
    @Test func verifyModeParses() throws {
        func parsed(_ mode: String?) throws -> ModelIntegrityPreference {
            var argv = ["--model", "m.finch", "--prompt", "hi"]
            if let mode { argv += ["--verify", mode] }
            return try Args.parse(argv).verify
        }
        #expect(try parsed(nil) == .automatic)
        #expect(try parsed("auto") == .automatic)
        #expect(try parsed("full-sha256") == .fullSha256)
        #expect(try parsed("trusted-install") == .sizeCheckTrustedReceipt)
        #expect(throws: ArgsError.invalidValue(flag: "--verify", value: "size-only")) {
            _ = try parsed("size-only")
        }
    }

    @Test func generationOptionsParseAndStopsRepeat() throws {
        let arguments = try Args.parse([
            "--model", "m.finch", "--prompt", "hi",
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
            "--model", "m.finch", "--prompt", "hi",
            "--top-k", "0", "--top-p", "1",
        ])
        #expect(disabled.topK == nil)
        #expect(disabled.topP == 1)

        #expect(throws: ArgsError.self) {
            _ = try Args.parse([
                "--model", "m.finch", "--prompt", "hi", "--top-k", "0",
            ])
        }
    }

    @Test func topKAboveKernelLimitRejected() {
        #expect(throws: ArgsError.invalidValue(flag: "--top-k", value: "257")) {
            _ = try Args.parse([
                "--model", "m.finch", "--prompt", "hi", "--top-k", "257",
            ])
        }
    }

    @Test func helpListsExactlyThePublicOptions() {
        let expected: Set<String> = [
            "--model", "--prompt", "--messages-file", "--max-new", "--max-context",
            "--temperature", "--top-k", "--top-p", "--repetition-penalty",
            "--seed", "--stop", "--quiet", "--counters", "--expert-cache-slots",
            "--verify", "--help",
        ]
        let words = Args.usage.split { $0.isWhitespace || $0 == "(" || $0 == ")" }
        let options = Set(words.map(String.init).filter { $0.hasPrefix("--") })
        #expect(options == expected)
    }

    @Test func unsupportedSelectorsAreRejected() {
        for flag in ["--runtime-profile", "--experiment-id", "-h"] {
            #expect(throws: ArgsError.unknownFlag(flag)) {
                _ = try Args.parse(["--model", "m.finch", "--prompt", "hi", flag])
            }
        }
    }

    @Test func modelAndPromptAreRequired() {
        #expect(throws: ArgsError.requiredMissing("--model")) {
            _ = try Args.parse(["--prompt", "hi"])
        }
        #expect(throws: ArgsError.modeMissing) {
            _ = try Args.parse(["--model", "m.finch"])
        }
    }

    @Test func messagesFileSelectsChatMode() throws {
        let arguments = try Args.parse([
            "--model", "m.finch", "--messages-file", "chat.json",
        ])
        #expect(arguments.prompt == nil)
        #expect(arguments.messagesFile == "chat.json")
    }

    @Test func expertCacheSlotsAcceptTheRuntimesOwnList() throws {
        for slots in RuntimeConfiguration.allowedExpertCacheSlots {
            let arguments = try Args.parse([
                "--model", "m.finch", "--prompt", "hi",
                "--expert-cache-slots", String(slots),
            ])
            #expect(arguments.expertCacheSlots == slots)
        }
    }

    /// The rejections matter more than the acceptances: `RuntimeConfiguration.init`
    /// `precondition`s on the same list, so anything `parse` lets through that is
    /// not on it would trap the process rather than print an error.
    @Test func expertCacheSlotsRejectAnythingOffTheList() {
        for value in ["7", "12", "0", "-16", "64", "sixteen"] {
            #expect(throws: ArgsError.invalidValue(flag: "--expert-cache-slots",
                                                   value: value)) {
                _ = try Args.parse([
                    "--model", "m.finch", "--prompt", "hi",
                    "--expert-cache-slots", value,
                ])
            }
        }
    }

    @Test func countersIsAFlagWithNoValue() throws {
        let arguments = try Args.parse([
            "--model", "m.finch", "--prompt", "hi", "--counters",
        ])
        #expect(arguments.counters)
        // A flag, so the next argument is not consumed as its value.
        #expect(arguments.prompt == "hi")
        #expect(arguments.model == "m.finch")
    }

    @Test func promptAndMessagesFileAreMutuallyExclusive() {
        #expect(throws: ArgsError.mutuallyExclusive("--prompt", "--messages-file")) {
            _ = try Args.parse([
                "--model", "m.finch", "--prompt", "hi",
                "--messages-file", "chat.json",
            ])
        }
    }
}

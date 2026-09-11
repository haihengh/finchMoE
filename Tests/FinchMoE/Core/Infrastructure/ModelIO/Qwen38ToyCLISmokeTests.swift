import Testing
import Foundation
import FinchMoE
import FinchMoECLICore

/// The M3.5 CLI smoke: drive the real `FinchMoECLI` entry point against a
/// **toy** Qwen 3.8 install, so the whole surface — argument parsing,
/// tokenizer load, engine load, generation, detokenized output — is exercised
/// on the one model family whose weights are small enough to build in a test.
///
/// The engine-level tests prove the arithmetic; this proves the plumbing. It
/// is the cheapest place a family-wide assumption can fail (a manifest slot
/// the CLI reads differently, a tokenizer family the model family does not
/// map onto, a generation config the toy geometry violates) and the only test
/// that would catch it without the 145 GB install M4 builds.
///
/// Tokenizer: the toy snapshot writes `{}` for its tokenizer files, so this
/// puts a real one in the `tokenizer/` sidecar `GFTokenizer` looks for first
/// (`Tokenizer.swift:96-105`). The sidecar needs all three files — see
/// `writeToyTokenizer` for why. The family is chosen from the sidecar's own
/// `config.json`, not the model's, and 3.8 rides the 3.6 tokenizer family
/// (`GFTokenizerFamily.detect`, `Tokenizer.swift:41-52` — `qwen4_exp` maps to
/// `.qwen3_6`), which is why a `qwen4_exp_text` model_type here needs only
/// `<|endoftext|>` and `<|im_end|>` rather than the Gemma tool markers.
///
/// Greedy (`--temperature 0`) so the run is deterministic; the assertion is
/// that it completes and emits text, not what the text says — the toy's
/// weights are seeded noise and are not expected to be coherent.
@Suite struct Qwen38ToyCLISmokeTests {

    @Test func toyInstallDrivesTheCLIEndToEnd() async throws {
        let out = try await Qwen38EngineLoadTests.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        try Self.writeToyTokenizer(into: out)

        // `--verify full-sha256`, explicitly, because that is the coverage this
        // test provides: it hashes the repack output against the manifest
        // `LocalQwenRepacker` just wrote, so a repack that produced a file the
        // manifest does not describe fails here. The default is now `.automatic`,
        // and this install *does* carry a `verified-install.json` — the repacker
        // writes one unconditionally (`LocalQwenRepacker.swift:449-458`), which
        // is what makes the `.automatic` path testable at all — so leaving the
        // flag off would silently trade that content check for a size check.
        //
        // Deliberately *not* `--quiet`: that flag suppresses the `stop=… new=Ntok
        // prefill=Ntok` footer (`Run.swift:114-136`), which is the only account
        // of what the generator actually did, and it goes to stderr — so with
        // `--quiet` a run that generated nothing and a run whose output path
        // broke both present as two empty strings.
        let args = try Args.parse([
            "--model", out,
            "--prompt", "hello world 3",
            "--max-new", "4",
            "--max-context", "64",
            "--temperature", "0",
            "--verify", "full-sha256",
        ])

        // `--verify full-sha256` above, so this leg fails on the flipped byte
        // rather than quietly size-checking past it.
        let (result, text, errors) = await Self.runCLI(args)

        #expect(result.exitCode == 0, "CLI exited \(result.exitCode): \(errors)")

        // The footer is the run's account of itself and carries the weight here.
        // Measured, not assumed: on this fixture the run stops `stop=eos` after
        // `new=1tok`, because the toy's weights are seeded noise and greedy
        // decode's first argmax lands on `<|im_end|>`. EOS streams an empty
        // delta — a special token has no detokenized text — so `stdout` is
        // *legitimately* empty, and asserting on it would be asserting on the
        // toy's random argmax rather than on the plumbing this test exists for.
        //
        // `prefill=13tok` is the whole chain in one number: the sidecar
        // tokenizer loaded and encoded "hello world 3" to exactly the thirteen
        // byte tokens `writeToyTokenizer` guarantees (one per UTF-8 byte), and
        // the engine loaded the toy install and prefilled over them. A break
        // anywhere in arg parsing → tokenizer → model load → prefill moves it.
        #expect(errors.contains("prefill=13tok"),
                "CLI did not prefill the 13-token prompt (stderr: \(errors), stdout: \(text))")
        #expect(errors.contains("stop="),
                "CLI reported no stop reason (stderr: \(errors))")
    }

    /// The default path, end to end — and the one assertion here that *proves*
    /// the default takes the receipt rather than merely surviving without one.
    ///
    /// `packed_experts/layer_00.bin` is size-checked against the receipt under
    /// `.sizeCheckTrustedReceipt` but SHA-256'd under `.fullSha256`
    /// (`Model.swift:632-640`), so a **size-preserving** byte flip in it —
    /// `flipByte`, never truncation, which the size check would catch — is
    /// invisible to one mode and fatal to the other. This leg passes no
    /// `--verify` at all, so it fails the moment the default stops resolving to
    /// the receipt; `prefill=13tok` is what rules out the flip being missed
    /// because layer 0 was never opened.
    ///
    /// The flip lands mid-file on purpose: what this asserts is which
    /// verification path ran, not what the toy's seeded-noise weights then
    /// produced. Nothing reads the flipped expert at a fixed offset, and even if
    /// it did, the run is bounded by `--max-new 4`.
    @Test func defaultVerifyModeTakesTheInstallReceipt() async throws {
        let out = try await Qwen38EngineLoadTests.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        try Self.writeToyTokenizer(into: out)
        let layerURL = URL(fileURLWithPath: out)
            .appendingPathComponent("packed_experts/layer_00.bin")
        let size = try FileManager.default
            .attributesOfItem(atPath: layerURL.path)[.size] as! NSNumber
        try ModelLoaderTests.flipByte(in: layerURL, at: size.uint64Value / 2)

        let args = try Args.parse([
            "--model", out,
            "--prompt", "hello world 3",
            "--max-new", "4",
            "--max-context", "64",
            "--temperature", "0",
        ])
        let (result, text, errors) = await Self.runCLI(args)

        #expect(result.exitCode == 0,
                "the default hashed a layer the receipt already covers — exited \(result.exitCode): \(errors)")
        #expect(errors.contains("prefill=13tok"), "stderr: \(errors), stdout: \(text)")
    }

    /// A receipt that is *present but unusable* is the one case worth
    /// interrupting for, because the caller may have been relying on it. The
    /// warning goes to the **injected** `stderr` (`Run.swift:88-90`) — which is
    /// what makes it observable here at all — and is deliberately not gated on
    /// `--quiet`, a flag that documents itself as suppressing the timing footer
    /// and nothing else.
    ///
    /// The run must still succeed. Falling back to hashing is the whole point of
    /// `.automatic`, and a receipt problem that cost availability would be a
    /// worse bug than the verification tax this mode removes. Zeroing
    /// `manifestSha256` is what makes `validateManifestBinding` reject it: the
    /// receipt still parses, so this exercises the *invalid* arm rather than the
    /// absent one, which stays silent by design.
    @Test func invalidReceiptWarnsOnStderrAndStillRuns() async throws {
        let out = try await Qwen38EngineLoadTests.makeInstall()
        defer { try? FileManager.default.removeItem(atPath: out) }
        try Self.writeToyTokenizer(into: out)
        try ModelLoaderTests.mutateReceipt(
            directoryURL: URL(fileURLWithPath: out)
        ) { root in
            root["manifestSha256"] = String(repeating: "0", count: 64)
        }

        let args = try Args.parse([
            "--model", out,
            "--prompt", "hello world 3",
            "--max-new", "4",
            "--max-context", "64",
            "--temperature", "0",
        ])
        let (result, text, errors) = await Self.runCLI(args)

        #expect(result.exitCode == 0, "CLI exited \(result.exitCode): \(errors)")
        #expect(errors.contains("warning: \(VerifiedInstallReceiptReader.fileName)"
                                + " is present but unusable"),
                "no receipt warning reached stderr: \(errors)")
        #expect(errors.contains("full SHA-256 instead"),
                "the warning does not say what it did instead: \(errors)")
        // A verification change, not a run change.
        #expect(errors.contains("prefill=13tok"), "stderr: \(errors), stdout: \(text)")
    }

    // MARK: - CLI harness

    /// Drives the CLI in-process and captures both streams.
    ///
    /// `FINCHMOE_EXPECT_ARCH` is what makes a toy install loadable at all, and
    /// it is the reason these tests could not be written without touching
    /// `Run.swift`. The CLI resolves the geometry it will validate against from
    /// the *family's built-in preset* (`Run.swift:77` → `ManifestReader.detectPreset`),
    /// so a manifest declaring `qwen3_8` is required to be 2560-wide and the
    /// toy's 64 is rejected outright — `archMismatch(field: "hiddenSize",
    /// expected: "2560", actual: "64")`. That check is right for a real install
    /// and is left exactly as it was; the override only redirects it to the arch
    /// the manifest declares about itself, and only when the variable is set.
    /// Every other toy test sidesteps the same wall by handing `Model.load`
    /// `expecting: Toy38.arch` directly, which the CLI has no way to accept.
    ///
    /// Passed as a parameter rather than via `setenv`: Swift Testing runs cases
    /// in parallel, so mutating the process environment here would race every
    /// other case that reads it.
    private static func runCLI(_ args: Args)
        async -> (result: RunResult, text: String, errors: String) {
        let stdout = Pipe()
        let stderr = Pipe()
        let result = await run(args: args,
                               stdout: stdout.fileHandleForWriting,
                               stderr: stderr.fileHandleForWriting,
                               environment: ["FINCHMOE_EXPECT_ARCH": "1"])
        // Close the write ends, or the reads below block forever: `run` has
        // returned but the pipe still has a live writer.
        try? stdout.fileHandleForWriting.close()
        try? stderr.fileHandleForWriting.close()
        let text = String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(),
                          as: UTF8.self)
        let errors = String(decoding: stderr.fileHandleForReading.readDataToEndOfFile(),
                            as: UTF8.self)
        return (result, text, errors)
    }

    // MARK: - Toy tokenizer

    /// A **byte-level BPE with no merges**, so every byte is exactly one token:
    /// `vocab[bytesToUnicode[b]] = b` for all of `0...255`. The ids therefore
    /// fill the toy's 256-row embedding table exactly, and — the invariant that
    /// makes this fixture safe for a *generation* test rather than just an
    /// encode test — every id the toy's `lm_head` can emit (`0...255`) has a
    /// detokenized form, so the CLI's detokenizer can never fall off the end of
    /// the vocab.
    ///
    /// `WordLevel` was the obvious-looking choice and cannot work at all:
    /// swift-transformers implements no such model class. `TokenizerModel.from`
    /// resolves `tokenizer_config.json`'s `tokenizer_class` against
    /// `knownTokenizers` with `strict: true` (`Tokenizer.swift:158-172`,
    /// `183-199`), so the class has to be a registered one; `GPT2Tokenizer` is,
    /// and it maps to `BPETokenizer`, the byte-level flavor we want. The vocab
    /// keys must then be **byte-level** encoded, not raw characters, because
    /// `ByteLevelPreTokenizer` rewrites each split token through
    /// `byteEncoderTable` *before* `bpe` splits the result into unicode scalars
    /// and looks them up (`PreTokenizer.swift:295-326`,
    /// `BPETokenizer.swift:245-249, 326-340`). So a space is the single scalar
    /// `"Ġ"` (U+0120), not `" "`.
    ///
    /// `merges: []` is required rather than optional: `BPETokenizer.init`
    /// `fatalError`s when the key is *absent*, and only an empty array gets
    /// past `mergesFromConfig` (`BPETokenizer.swift:130-152`).
    ///
    /// The two specials live in `added_tokens` rather than in the vocab, and
    /// are pinned to ids 0 and 1 — they shadow bytes 0 and 1 (NUL and SOH),
    /// control characters the ASCII prompt cannot contain, so nothing is lost.
    ///
    /// The sidecar needs **three** files, which is what the previous fixture
    /// got wrong. `tokenizer.json` alone is not enough: `AutoTokenizer.from`
    /// (`modelFolder:)` reads `tokenizer_config.json` first and throws
    /// `TokenizerError.missingConfig` when it is absent (`Tokenizer.swift:908-925`),
    /// and `config.json` is read separately by `GFTokenizer.loadUncached(from:)`
    /// to pick the family (`Sources/FinchMoE/Tokenization/Tokenizer.swift:111-121`).
    private static func writeToyTokenizer(into installDir: String) throws {
        let fm = FileManager.default
        let dir = URL(fileURLWithPath: installDir)
            .appendingPathComponent("tokenizer", isDirectory: true)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // One token per byte, ids 0...255 — exactly the toy embedding table.
        let table = bytesToUnicode()
        var vocab: [String: Int] = [:]
        for byte in 0...255 { vocab[table[byte]] = byte }

        func addedToken(_ content: String, _ id: Int) -> [String: Any] {
            ["id": id, "content": content, "single_word": false,
             "lstrip": false, "rstrip": false, "normalized": false,
             "special": true]
        }

        let tokenizer: [String: Any] = [
            "version": "1.0",
            "truncation": NSNull(),
            "padding": NSNull(),
            "added_tokens": [addedToken("<|endoftext|>", 0),
                             addedToken("<|im_end|>", 1)],
            "normalizer": NSNull(),
            // `add_prefix_space: false` — Qwen does not lead its text with a
            // space the way GPT-2's tokenizer does. With `use_regex: true` the
            // standard GPT-2 pattern splits the prompt "hello world 3" into
            // ["hello", " world", " 3"], whose bytes are all in 32...119 and so
            // all land inside the embedding table.
            "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false,
                              "use_regex": true, "trim_offsets": true],
            "post_processor": NSNull(),
            "decoder": NSNull(),
            "model": ["type": "BPE", "vocab": vocab, "merges": [String]()],
        ]
        try write(tokenizer, to: dir.appendingPathComponent("tokenizer.json"))

        // `eos_token` is what supplies `Tokenizer.eosTokenId`, which
        // `GFTokenizer.init` refuses to do without (`missingSpecialToken`).
        try write(["tokenizer_class": "GPT2Tokenizer",
                   "eos_token": "<|im_end|>",
                   "model_max_length": 512,
                   "fuse_unk": false],
                  to: dir.appendingPathComponent("tokenizer_config.json"))

        // `model_type` here (not the model dir's config) is what selects the
        // tokenizer family; `qwen4_exp_text` maps to `.qwen3_6`, which needs
        // `<|endoftext|>` and `<|im_end|>` only.
        try write(["eos_token": "<|im_end|>",
                   "model_type": "qwen4_exp_text",
                   "text_config": ["model_type": "qwen4_exp_text"]],
                  to: dir.appendingPathComponent("config.json"))
    }

    /// The canonical GPT-2 byte → unicode map, reimplemented because the real
    /// table is module-internal to `Tokenizers` and unreachable from here.
    /// Mirrors `ByteEncoder.swift:11-296`: bytes `33...126`, `161...172` and
    /// `174...255` stand for themselves; every other byte — `0...32`,
    /// `127...160` and `173`, 68 in total — is assigned `chr(256 + n)` in
    /// ascending byte order. The `173 → "\u{0143}"` (323) entry that
    /// `ByteEncoder.swift:267` pins is the proof of that ordering.
    ///
    /// Returns a 256-element array indexed by byte value.
    private static func bytesToUnicode() -> [String] {
        var table = [String](repeating: "", count: 256)
        var unmapped: [Int] = []
        for byte in 0...255 {
            if (33...126).contains(byte) || (161...172).contains(byte)
                || (174...255).contains(byte) {
                table[byte] = String(UnicodeScalar(UInt8(byte)))
            } else {
                unmapped.append(byte)
            }
        }
        // `unmapped` holds exactly 68 entries, so `256 + n` stays within
        // 256...323 — all of which are valid scalars.
        for (n, byte) in unmapped.enumerated() {
            table[byte] = String(UnicodeScalar(UInt32(256 + n))!)
        }
        return table
    }

    private static func write(_ object: [String: Any], to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object,
                                              options: [.sortedKeys])
        try data.write(to: url)
    }
}

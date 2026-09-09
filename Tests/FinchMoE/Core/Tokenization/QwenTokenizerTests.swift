import Testing
import Foundation
@testable import FinchMoE

/// Qwen 3.6 tokenizer behavior against the REAL install's tokenizer sidecars.
/// Skipped when the install is not present. The family is detected from the
/// installed `config.json` (`model_type: qwen3_5_moe`); these tests pin the
/// family-specific special-token handling: no BOS prepend, the ChatML
/// template, and the generation_config stop set (248046 `<|im_end|>` + 248044
/// `<|endoftext|>`).
@Suite("Qwen tokenizer")
struct QwenTokenizerTests {

    private static let installPath =
        "/Volumes/samsung 2t/code/finchmoe/models/Qwen3.6-35B-A3B-4bit.finch"

    private static var installExists: Bool {
        FileManager.default.fileExists(atPath: installPath + "/manifest.json")
    }

    @Test("Family is detected as qwen3_6 from the install", .enabled(if: installExists))
    func familyDetected() async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        #expect(tok.family == .qwen3_6)
        #expect(tok.vocabSize == 248_320)
    }

    @Test("Stop set covers <|im_end|> and the config eos <|endoftext|>",
          .enabled(if: installExists))
    func stopSet() async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        // generation_config.json stops on BOTH 248046 (<|im_end|>) and 248044
        // (<|endoftext|>, the model config's bos/pad/eos_token_id).
        #expect(tok.stopTokenIDs.contains(248_046))
        #expect(tok.stopTokenIDs.contains(248_044))
        // The Gemma tool markers are inert sentinels on this family.
        #expect(tok.toolResponseID == -1)
        #expect(tok.channelStartID == -1)
    }

    @Test("Encode never prepends BOS on Qwen", .enabled(if: installExists))
    func noBosPrepend() async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let withBOS = tok.encode("Hello", addBOS: true)
        let without = tok.encode("Hello", addBOS: false)
        #expect(withBOS == without)
    }

    @Test("Chat template renders ChatML and round-trips through encode",
          .enabled(if: installExists))
    func chatTemplate() async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let rendered = try tok.applyChatTemplate([
            GFTokenizer.Message(role: .user, content: "Hi")
        ])
        #expect(rendered.contains("<|im_start|>user\nHi<|im_end|>"))
        // Thinking-disabled generation cue (empty think block), per the
        // checkpoint's template with enable_thinking=false.
        #expect(rendered.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        // The rendered template must encode to a non-empty id sequence and
        // decode back with the special tokens stripped.
        let ids = tok.encode(rendered, addBOS: false)
        #expect(!ids.isEmpty)
        let decoded = tok.decode(ids, skipSpecialTokens: true)
        #expect(decoded.contains("Hi"))
    }

    @Test("Special-token IDs are distinct, in-vocab, and match the checkpoint",
          .enabled(if: installExists))
    func specialTokenIDs() async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        // Checkpoint tokenizer: <|endoftext|> = 248044, <|im_start|> = 248045,
        // <|im_end|> = 248046.
        #expect(tok.endOfTurnID == 248_046)
        #expect(tok.eosID == 248_046)      // tokenizer config eos_token
        #expect(tok.bosID == 248_044)      // model config bos_token_id
        #expect(tok.padID == 248_044)
    }
}

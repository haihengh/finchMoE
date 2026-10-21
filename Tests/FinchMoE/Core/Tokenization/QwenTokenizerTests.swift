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

    /// Byte-level BPE (Qwen) has no `<0xXX>` byte-fallback tokens, so a
    /// multi-byte character can be split across ordinary tokens. Decoding the
    /// prefix after the first half yields U+FFFD; the old detokenizer emitted
    /// that replacement character and then dropped the completing token, which
    /// is the diamond-with-question-mark seen in Chinese output. U+4E02 is such
    /// a character in this vocabulary: E4 B8, then 82.
    @Test("Streaming detokenizer assembles a character split across tokens",
          .enabled(if: installExists),
          arguments: ["丂", "丄", "丅", "丆", "丌", "丏", "丒", "丕", "丗", "丠"])
    func streamingSplitCharacter(_ character: String) async throws {
        let tok = try await GFTokenizer.load(forModelDirectory:
            URL(fileURLWithPath: Self.installPath))
        let ids = tok.encode(character, addBOS: false)
        #expect(ids.count > 1, "\(character) is expected to span several tokens")

        var detok = GFDetokenizer(tokenizer: tok)
        var assembled = ""
        for id in ids {
            let delta = detok.push(id)
            #expect(!delta.unicodeScalars.contains("\u{FFFD}"),
                    "stream emitted a replacement character for token \(id)")
            assembled += delta
        }
        assembled += detok.flush()
        #expect(assembled == character)
    }

    /// The byte-level detokenizer's two primitives: the GPT-2 alphabet inverse
    /// and the incomplete-scalar suffix counter that decides what to hold back.
    @Test("Byte-level detokenizer primitives")
    func byteLevelPrimitives() {
        #expect(GFDetokenizer.byteLevelBytes("ä¸") == [0xE4, 0xB8])
        #expect(GFDetokenizer.byteLevelBytes("Ĥ") == [0x82])
        // A scalar outside the byte alphabet is not a byte-level token.
        #expect(GFDetokenizer.byteLevelBytes("中") == nil)

        #expect(GFDetokenizer.incompleteUTF8SuffixCount([]) == 0)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0x41]) == 0)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0xE4]) == 1)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0xE4, 0xB8]) == 2)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0xE4, 0xB8, 0x82]) == 0)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0xF0, 0x9F, 0xA6]) == 3)
        #expect(GFDetokenizer.incompleteUTF8SuffixCount([0xFF]) == 0)
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

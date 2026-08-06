// tokenizer.h
// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer interface and TikTokenizer — a tiktoken-compatible BPE tokenizer
// for Qwen3 that loads its vocabulary from a plain-text .tiktoken file.
//
// Vocabulary file format (.tiktoken)
// ────────────────────────────────────
// The file is a UTF-8 text file generated alongside the .aeromoe model by
// aeromoe_convert.py.  It has two sections separated by the line
// "### special ###":
//
//   # Regular tokens (one per line):
//   <base64_encoded_bytes>\t<integer_rank>
//
//   ### special ###
//   <token_text>\t<integer_id>
//
// Base64-encodes the raw UTF-8 byte sequence of the token.
// Integer ranks are the token IDs (consistent with Qwen3's vocabulary).
//
// Example: the byte sequence [0x48, 0x65, 0x6C, 0x6C, 0x6F] ("Hello") maps to
//   SGVsbG8=\t15043
//
// Path convention
// ───────────────
// For a model at /path/to/model.aeromoe the tokenizer expects a vocabulary at
//   /path/to/model.tiktoken
// This is produced automatically by aeromoe_convert.py and is required for
// ToolEngine to operate.  make_tokenizer() infers the path automatically.
//
// BPE algorithm
// ─────────────
// Follows tiktoken's algorithm exactly:
//   1. Pre-tokenise text with a regex pattern (simplified cl100k_base variant).
//   2. For each piece, iteratively merge the adjacent byte pair with the
//      lowest rank in the vocabulary.
//   3. Look up final segments to obtain token IDs.
//
// Special tokens (e.g. <|im_start|>, <|im_end|>) are matched before BPE so
// they are always encoded as a single ID regardless of their byte content.
// Use encode_with_special() for prompt strings that contain them.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aeromoe {

// Qwen3 well-known special-token IDs.
// These are baked in here so callers need not link the tokenizer just to
// check for the EOS token during generation.
static constexpr uint32_t TOK_ENDOFTEXT  = 151643;   // <|endoftext|>
static constexpr uint32_t TOK_IM_START   = 151644;   // <|im_start|>
static constexpr uint32_t TOK_IM_END     = 151645;   // <|im_end|>
// "Thinking mode" tokens introduced in Qwen3:
static constexpr uint32_t TOK_THINK_START = 151646;  // <think>
static constexpr uint32_t TOK_THINK_END   = 151647;  // </think>

// ── Tokenizer (abstract interface) ───────────────────────────────────────────

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // ── Encoding ──────────────────────────────────────────────────────────────

    // Encode plain text.  Special tokens embedded in `text` are NOT recognised;
    // their byte sequences are BPE-encoded as regular text.
    // Use this only for plain content (e.g. a document to summarise).
    virtual std::vector<uint32_t> encode(const std::string& text) const = 0;

    // Encode text that may contain special-token strings (e.g. "<|im_start|>").
    // Special tokens are matched greedily longest-first and encoded as single
    // IDs.  Remaining spans are BPE-encoded.
    // Always use this for fully-formatted ChatML prompts.
    virtual std::vector<uint32_t> encode_with_special(
                                        const std::string& text) const = 0;

    // ── Decoding ──────────────────────────────────────────────────────────────

    // Decode a sequence of token IDs to UTF-8 text.
    // Special tokens decode to their literal text (e.g. "<|im_end|>").
    virtual std::string decode(const std::vector<uint32_t>& tokens) const = 0;

    // Decode a single token to its UTF-8 byte string.
    // Returns "" for unknown IDs.
    virtual std::string decode_token(uint32_t id) const = 0;

    // ── Introspection ─────────────────────────────────────────────────────────

    // Return the ID of a special token by its string name.
    // Returns UINT32_MAX if the token is not in the vocabulary.
    virtual uint32_t special_token(const std::string& name) const = 0;

    // True if `id` is a special token (id >= first special-token offset).
    virtual bool is_special(uint32_t id) const = 0;

    virtual uint32_t vocab_size() const = 0;
};

// ── TikTokenizer ─────────────────────────────────────────────────────────────

class TikTokenizer final : public Tokenizer {
public:
    TikTokenizer();
    ~TikTokenizer() override;

    // Load vocabulary from a .tiktoken file.
    // Returns true on success, false (+ stderr message) on failure.
    bool load(const std::string& path);

    std::vector<uint32_t> encode(const std::string& text) const override;
    std::vector<uint32_t> encode_with_special(const std::string& text) const override;
    std::string           decode(const std::vector<uint32_t>& tokens) const override;
    std::string           decode_token(uint32_t id) const override;
    uint32_t              special_token(const std::string& name) const override;
    bool                  is_special(uint32_t id) const override;
    uint32_t              vocab_size() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── Factory ───────────────────────────────────────────────────────────────────

// Create a TikTokenizer.  Infers the vocab path from the model path:
//   /path/to/model.aeromoe → /path/to/model.tiktoken
// Returns nullptr if the vocab file cannot be loaded.
std::unique_ptr<Tokenizer> make_tokenizer(const std::string& model_path);

} // namespace aeromoe

// tokenizer.cpp
// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer — tiktoken-compatible BPE tokenizer for Qwen3.
//
// BPE algorithm
// ─────────────
// Follows tiktoken's Rust implementation exactly.  The core insight is that
// tiktoken's vocabulary maps *byte sequences* (not Unicode codepoints) to
// integer ranks, and the BPE algorithm always merges the adjacent pair whose
// merged byte sequence has the lowest rank in the vocabulary.
//
//   bpe_encode_piece(bytes, vocab):
//     parts = [0, 1, 2, …, n]           # part boundaries (indices into bytes)
//     ranks = [vocab.get(bytes[i:i+2])   # rank of each consecutive 2-segment pair
//               for i in range(n-1)]
//     while min(ranks) < ∞:
//       i = argmin(ranks)
//       remove parts[i+1]                # merge parts[i] and parts[i+1]
//       update ranks[i-1] and ranks[i]   # neighbours changed
//     return [vocab[bytes[parts[i]:parts[i+1]]] for each final part]
//
// Pre-tokenisation
// ────────────────
// Text is split into "pieces" before BPE using std::regex.  The pattern is a
// simplified version of tiktoken's cl100k_base/GPT-4 regex:
//
//   (?:'s|'t|'re|'ve|'m|'ll|'d)   English contractions
//   | \w+                          word runs (Unicode-aware in ECMAScript mode)
//   | ?\s                          space+whitespace
//   | ?[^\s\w]+                    punctuation (optional leading space)
//
// This correctly handles ASCII and common Unicode chat text.  For full Unicode
// equivalence with Python's tiktoken you would add ICU \p{L}/\p{N} support.
//
// Special-token handling
// ──────────────────────
// encode_with_special() scans the text for all registered special-token
// strings before BPE.  Matches are sorted longest-first so that e.g.
// "<|im_start|>" wins over a hypothetical shorter prefix.
// ─────────────────────────────────────────────────────────────────────────────
#include "tokenizer.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aeromoe {

// ── Base-64 decode ────────────────────────────────────────────────────────────

static const char B64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Decode a base64 string (with or without padding) to raw bytes.
static std::vector<uint8_t> b64_decode(const std::string& in) {
    static uint8_t lut[256]{};
    static bool inited = false;
    if (!inited) {
        memset(lut, 0xFF, sizeof(lut));
        for (int i = 0; i < 64; ++i)
            lut[static_cast<uint8_t>(B64_ALPHABET[i])] = static_cast<uint8_t>(i);
        inited = true;
    }

    std::vector<uint8_t> out;
    out.reserve(in.size() * 3 / 4 + 2);
    uint32_t accum = 0;
    int      bits  = 0;
    for (unsigned char c : in) {
        if (c == '=' || c == '\n' || c == '\r') continue;
        uint8_t v = lut[c];
        if (v == 0xFF) continue;
        accum = (accum << 6) | v;
        bits  += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accum >> bits) & 0xFF));
        }
    }
    return out;
}

// ── Byte-sequence hash ────────────────────────────────────────────────────────

struct ByteVecHash {
    size_t operator()(const std::vector<uint8_t>& v) const noexcept {
        // FNV-1a over the byte vector
        size_t h = 0xcbf29ce484222325ULL;
        for (uint8_t b : v) { h ^= b; h *= 0x100000001b3ULL; }
        return h;
    }
};

using VocabMap = std::unordered_map<std::vector<uint8_t>, uint32_t, ByteVecHash>;

// ── Pre-tokenisation ──────────────────────────────────────────────────────────

// Split text into pieces that will each be BPE-encoded independently.
// The regex is case-insensitive for contraction matching.
static std::vector<std::string> pretokenize(const std::string& text) {
    // Simplified cl100k / GPT-4 pattern (ECMAScript-compatible).
    // Handles ASCII and Unicode word characters via \w in ECMAScript mode.
    static const std::regex PAT(
        "(?:'s|'t|'re|'ve|'m|'ll|'d)"  // English contractions
        "|\\w+"                          // word runs (incl. Unicode via ECMAScript \w)
        "| ?[^\\s\\w]+"                 // punctuation (optional leading space)
        "|\\s+",                         // whitespace
        std::regex::ECMAScript | std::regex::icase
    );

    std::vector<std::string> pieces;
    auto it  = std::sregex_iterator(text.begin(), text.end(), PAT);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) pieces.push_back((*it).str());
    return pieces;
}

// ── BPE encode a single UTF-8 piece ──────────────────────────────────────────

// Encode a single byte sequence (a pre-tokenisation piece) using the tiktoken
// BPE algorithm.  See module comment for algorithm description.
static std::vector<uint32_t> bpe_encode_piece(
    const std::vector<uint8_t>& piece,
    const VocabMap&              vocab)
{
    const size_t n = piece.size();
    if (n == 0) return {};

    if (n == 1) {
        // Single byte: look up directly (always in vocab for a complete model)
        auto it = vocab.find(piece);
        return it != vocab.end()
               ? std::vector<uint32_t>{it->second}
               : std::vector<uint32_t>{};
    }

    // parts[i] is the start byte index of segment i.
    // parts has n+1 entries: {0, 1, 2, …, n} (sentinel at n).
    std::vector<size_t> parts(n + 1);
    std::iota(parts.begin(), parts.end(), size_t{0});

    // ranks[i] = vocabulary rank of the merged byte sequence
    //            bytes[parts[i] .. parts[i+2])
    // This covers "what rank would we get if we merge segment i with segment i+1?"
    // ranks has n-1 entries (pairs of consecutive segments).
    const uint32_t INF = std::numeric_limits<uint32_t>::max();

    auto get_rank = [&](size_t i) -> uint32_t {
        if (i + 2 >= parts.size()) return INF;
        std::vector<uint8_t> seg(
            piece.begin() + static_cast<ptrdiff_t>(parts[i]),
            piece.begin() + static_cast<ptrdiff_t>(parts[i + 2]));
        auto it = vocab.find(seg);
        return it != vocab.end() ? it->second : INF;
    };

    std::vector<uint32_t> ranks(n - 1);
    for (size_t i = 0; i < ranks.size(); ++i) ranks[i] = get_rank(i);

    // Iteratively merge the lowest-rank pair.
    while (true) {
        uint32_t min_rank = INF;
        size_t   min_i    = SIZE_MAX;
        for (size_t i = 0; i < ranks.size(); ++i) {
            if (ranks[i] < min_rank) { min_rank = ranks[i]; min_i = i; }
        }
        if (min_rank == INF) break;  // no more merges possible

        // Merge: remove the boundary between segment min_i and min_i+1.
        // In parts[] this means erasing parts[min_i + 1].
        parts.erase(parts.begin() + static_cast<ptrdiff_t>(min_i + 1));
        ranks.erase(ranks.begin() + static_cast<ptrdiff_t>(min_i));

        // Update rank at min_i (new pair: merged-segment with its right neighbour)
        if (min_i < ranks.size()) ranks[min_i] = get_rank(min_i);

        // Update rank at min_i-1 (left neighbour now has a longer right partner)
        if (min_i > 0 && min_i - 1 < ranks.size())
            ranks[min_i - 1] = get_rank(min_i - 1);
    }

    // Convert final segments to token IDs.
    std::vector<uint32_t> result;
    result.reserve(parts.size() - 1);
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
        std::vector<uint8_t> seg(
            piece.begin() + static_cast<ptrdiff_t>(parts[i]),
            piece.begin() + static_cast<ptrdiff_t>(parts[i + 1]));
        auto it = vocab.find(seg);
        if (it != vocab.end()) {
            result.push_back(it->second);
        } else {
            // Byte fallback — should not happen with a complete Qwen3 vocab
            // (all single bytes are in the vocabulary).
            for (uint8_t b : seg) {
                std::vector<uint8_t> bv{b};
                auto bit = vocab.find(bv);
                if (bit != vocab.end()) result.push_back(bit->second);
            }
        }
    }
    return result;
}

// ── TikTokenizer::Impl ───────────────────────────────────────────────────────

struct TikTokenizer::Impl {
    VocabMap vocab;                                               // bytes → ID
    std::vector<std::vector<uint8_t>> id_to_bytes;               // ID → bytes

    // Special tokens (indexed two ways for fast lookup in both directions)
    std::unordered_map<std::string,  uint32_t> special_by_name;  // name → ID
    std::unordered_map<uint32_t, std::string>  special_by_id;    // ID → name

    // Pre-sorted for greedy longest-match in encode_with_special.
    // Built once in load().
    struct SpecEntry { std::string name; uint32_t id; };
    std::vector<SpecEntry> special_sorted;  // descending length order

    uint32_t vocab_size     = 0;
    uint32_t special_offset = 151643;  // first special-token ID (Qwen3 default)
};

TikTokenizer::TikTokenizer()  : impl_(std::make_unique<Impl>()) {}
TikTokenizer::~TikTokenizer() = default;

// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer::load
// ─────────────────────────────────────────────────────────────────────────────
bool TikTokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[tokenizer] Cannot open vocab file: %s\n", path.c_str());
        return false;
    }

    auto& D = *impl_;
    bool  in_special = false;
    std::string line;

    while (std::getline(f, line)) {
        // Trim Windows CRLF
        if (!line.empty() && line.back() == '\r') line.pop_back();

        // Skip blank lines and comments
        if (line.empty() || line.front() == '#') continue;

        // Section separator
        if (line == "### special ###") { in_special = true; continue; }

        // Find separator: try tab first, then last space
        size_t sep = line.find('\t');
        if (sep == std::string::npos) sep = line.rfind(' ');
        if (sep == std::string::npos) continue;

        std::string key_str = line.substr(0, sep);
        uint32_t    id      = 0;
        try {
            id = static_cast<uint32_t>(std::stoul(line.substr(sep + 1)));
        } catch (...) { continue; }

        if (in_special) {
            // Special token: key is the literal token text
            D.special_by_name[key_str] = id;
            D.special_by_id[id]        = key_str;
            if (id < D.special_offset) D.special_offset = id;
        } else {
            // Regular token: key is base64-encoded bytes
            auto bytes = b64_decode(key_str);
            if (bytes.empty()) continue;
            D.vocab[bytes] = id;

            // Grow id_to_bytes as needed
            if (id >= static_cast<uint32_t>(D.id_to_bytes.size()))
                D.id_to_bytes.resize(id + 1);
            D.id_to_bytes[id] = bytes;
        }
    }

    // Build sorted special-token list for greedy matching (longest first)
    D.special_sorted.reserve(D.special_by_name.size());
    for (auto& [name, id] : D.special_by_name)
        D.special_sorted.push_back({name, id});
    std::sort(D.special_sorted.begin(), D.special_sorted.end(),
              [](const Impl::SpecEntry& a, const Impl::SpecEntry& b) {
                  return a.name.size() > b.name.size();
              });

    D.vocab_size = static_cast<uint32_t>(D.id_to_bytes.size())
                 + static_cast<uint32_t>(D.special_by_id.size());

    fprintf(stderr, "[tokenizer] Loaded %zu tokens + %zu special tokens from %s\n",
            D.vocab.size(), D.special_by_name.size(), path.c_str());
    return !D.vocab.empty();
}

// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer::encode
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint32_t> TikTokenizer::encode(const std::string& text) const {
    auto& D = *impl_;
    std::vector<uint32_t> result;
    auto pieces = pretokenize(text);
    for (const auto& piece : pieces) {
        std::vector<uint8_t> bytes(piece.begin(), piece.end());
        auto toks = bpe_encode_piece(bytes, D.vocab);
        result.insert(result.end(), toks.begin(), toks.end());
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer::encode_with_special
//
// Scan `text` for special-token strings (greedy longest match).  Spans between
// special tokens are BPE-encoded; special tokens become single IDs.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<uint32_t> TikTokenizer::encode_with_special(const std::string& text) const {
    auto& D = *impl_;
    if (D.special_sorted.empty()) return encode(text);

    std::vector<uint32_t> result;
    size_t pos = 0;
    const size_t len = text.size();

    while (pos < len) {
        // Try to match a special token starting at `pos`
        bool matched = false;
        for (const auto& e : D.special_sorted) {
            const size_t n = e.name.size();
            if (n <= len - pos && text.compare(pos, n, e.name) == 0) {
                result.push_back(e.id);
                pos += n;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        // Find the position of the nearest upcoming special token
        size_t next_special = len;
        for (const auto& e : D.special_sorted) {
            size_t p = text.find(e.name, pos);
            if (p != std::string::npos && p < next_special)
                next_special = p;
        }

        // BPE-encode the span [pos, next_special)
        if (next_special > pos) {
            auto span = text.substr(pos, next_special - pos);
            auto toks = encode(span);
            result.insert(result.end(), toks.begin(), toks.end());
        }
        pos = next_special;
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer::decode / decode_token
// ─────────────────────────────────────────────────────────────────────────────
std::string TikTokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string result;
    result.reserve(tokens.size() * 3);  // rough estimate
    for (uint32_t tok : tokens) result += decode_token(tok);
    return result;
}

std::string TikTokenizer::decode_token(uint32_t id) const {
    auto& D = *impl_;

    // Check special tokens first
    auto sit = D.special_by_id.find(id);
    if (sit != D.special_by_id.end()) return sit->second;

    // Regular token: id indexes into id_to_bytes
    if (id < static_cast<uint32_t>(D.id_to_bytes.size())
        && !D.id_to_bytes[id].empty())
    {
        const auto& bytes = D.id_to_bytes[id];
        return std::string(bytes.begin(), bytes.end());
    }

    return "";  // unknown ID
}

// ─────────────────────────────────────────────────────────────────────────────
// TikTokenizer::special_token / is_special / vocab_size
// ─────────────────────────────────────────────────────────────────────────────
uint32_t TikTokenizer::special_token(const std::string& name) const {
    auto it = impl_->special_by_name.find(name);
    return it != impl_->special_by_name.end() ? it->second : UINT32_MAX;
}

bool TikTokenizer::is_special(uint32_t id) const {
    return impl_->special_by_id.count(id) > 0;
}

uint32_t TikTokenizer::vocab_size() const {
    return impl_->vocab_size;
}

// ─────────────────────────────────────────────────────────────────────────────
// make_tokenizer
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Tokenizer> make_tokenizer(const std::string& model_path) {
    // Replace .aeromoe extension with .tiktoken
    std::string vocab_path = model_path;
    const std::string aeromoe_ext = ".aeromoe";
    size_t ext_pos = vocab_path.rfind(aeromoe_ext);
    if (ext_pos != std::string::npos) {
        vocab_path.replace(ext_pos, aeromoe_ext.size(), ".tiktoken");
    } else {
        vocab_path += ".tiktoken";
    }

    auto tok = std::make_unique<TikTokenizer>();
    if (!tok->load(vocab_path)) return nullptr;
    return tok;
}

} // namespace aeromoe

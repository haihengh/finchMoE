// tool_parser.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementation of ToolParser — parses Qwen3's <tool_call>…</tool_call>
// format into typed ToolCall structs.
//
// No general JSON library is used.  The parser is a hand-written recursive
// descent that handles exactly the Qwen3 tool-call output format:
//
//   <tool_call>
//   {"name": "<string>", "arguments": <json_value>}
//   </tool_call>
//
// Key functions (all static, reentrant):
//   skip_ws()            — advance past whitespace
//   read_json_string()   — read a "…" JSON string with escape handling
//   extract_json_value() — extract any JSON value as a raw substring
//   parse_call_json()    — parse the complete tool-call JSON object
//   parse()              — scan text for blocks and return ToolCall vector
// ─────────────────────────────────────────────────────────────────────────────
#include "tool_parser.h"

#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

namespace finchmoe {

// ── Sentinel tag strings ──────────────────────────────────────────────────────

const std::string ToolParser::kOpenTag  = "<tool_call>";
const std::string ToolParser::kCloseTag = "</tool_call>";

// ── Internal helpers ──────────────────────────────────────────────────────────

// Advance `pos` past ASCII whitespace characters.
static inline size_t skip_ws(const std::string& s, size_t pos) noexcept {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])))
        ++pos;
    return pos;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::read_json_string
//
// Precondition: s[pos] == '"'.
// Reads a JSON double-quoted string, handling \n \r \t \\ \" \/ and \uXXXX
// escapes.  Advances `pos` past the closing '"'.
// Returns false if the string is unterminated or the leading '"' is missing.
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ bool ToolParser::read_json_string(const std::string& s,
                                              size_t&            pos,
                                              std::string&       out) {
    pos = skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;  // consume opening "

    out.clear();
    while (pos < s.size()) {
        unsigned char ch = static_cast<unsigned char>(s[pos]);
        if (ch == '"') { ++pos; return true; }  // closing "

        if (ch != '\\') {
            out += static_cast<char>(ch);
            ++pos;
            continue;
        }

        // Escape sequence
        ++pos;
        if (pos >= s.size()) return false;
        switch (s[pos]) {
            case '"':  out += '"';  break;
            case '\\': out += '\\'; break;
            case '/':  out += '/';  break;
            case 'n':  out += '\n'; break;
            case 'r':  out += '\r'; break;
            case 't':  out += '\t'; break;
            case 'b':  out += '\b'; break;
            case 'f':  out += '\f'; break;
            case 'u': {
                // \uXXXX — decode to UTF-8
                if (pos + 4 >= s.size()) return false;
                char hex[5] = {};
                for (int i = 0; i < 4; ++i) hex[i] = s[pos + 1 + i];
                pos += 4;
                unsigned long code = std::strtoul(hex, nullptr, 16);
                if (code < 0x80) {
                    out += static_cast<char>(code);
                } else if (code < 0x800) {
                    out += static_cast<char>(0xC0 | (code >> 6));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                } else {
                    out += static_cast<char>(0xE0 | (code >> 12));
                    out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                    out += static_cast<char>(0x80 | (code & 0x3F));
                }
                break;
            }
            default:
                out += '\\';
                out += s[pos];
                break;
        }
        ++pos;
    }
    return false;  // unterminated string
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::extract_json_value
//
// Extract any JSON value (object, array, string, number, bool, null) starting
// at `pos` in `s`.  On success:
//   - `end_pos` is set to the first character after the value
//   - `raw_out` is the verbatim substring
// For string values, `raw_out` includes the surrounding double-quotes.
// Returns false if no value could be parsed.
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ bool ToolParser::extract_json_value(const std::string& s,
                                                size_t             pos,
                                                size_t&            end_pos,
                                                std::string&       raw_out) {
    pos = skip_ws(s, pos);
    if (pos >= s.size()) return false;

    const char first = s[pos];

    // ── String ───────────────────────────────────────────────────────────────
    if (first == '"') {
        size_t start = pos;
        ++pos;  // skip opening "
        while (pos < s.size()) {
            if (s[pos] == '\\') { pos += 2; continue; }   // skip escape
            if (s[pos] == '"')  { ++pos; break; }          // closing "
            ++pos;
        }
        raw_out = s.substr(start, pos - start);
        end_pos = pos;
        return true;
    }

    // ── Object or Array ───────────────────────────────────────────────────────
    if (first == '{' || first == '[') {
        const char open  = first;
        const char close = (first == '{') ? '}' : ']';
        int depth = 0;
        size_t start = pos;
        while (pos < s.size()) {
            const char c = s[pos];
            if (c == '"') {
                // Skip string literal — it may contain { } [ ] characters
                ++pos;
                while (pos < s.size()) {
                    if (s[pos] == '\\') { pos += 2; continue; }
                    if (s[pos] == '"')  { ++pos; break; }
                    ++pos;
                }
                continue;
            }
            if (c == open)  { ++depth; ++pos; continue; }
            if (c == close) {
                ++pos;
                if (--depth == 0) {
                    raw_out = s.substr(start, pos - start);
                    end_pos = pos;
                    return true;
                }
            } else {
                ++pos;
            }
        }
        return false;  // unmatched brace / bracket
    }

    // ── Number, boolean, null ─────────────────────────────────────────────────
    size_t start = pos;
    while (pos < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[pos]);
        if (c == ',' || c == '}' || c == ']' || std::isspace(c)) break;
        ++pos;
    }
    if (pos == start) return false;
    raw_out = s.substr(start, pos - start);
    end_pos = pos;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::parse_call_json
//
// Parse exactly the top-level JSON object that Qwen3 emits inside a tool_call
// block.  Expected format:
//
//   { "name": "<fn>", "arguments": <value> }
//
// Both keys are required; "arguments" may be any JSON value (typically an
// object).  Unknown keys are silently consumed.
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ bool ToolParser::parse_call_json(const std::string& json,
                                             ToolCall&          call_out) {
    size_t pos = skip_ws(json, 0);
    if (pos >= json.size() || json[pos] != '{') return false;
    ++pos;  // consume '{'

    bool got_name = false;
    bool got_args = false;

    while (pos < json.size()) {
        pos = skip_ws(json, pos);
        if (pos >= json.size()) return false;
        if (json[pos] == '}') break;  // end of object

        // ── Key ───────────────────────────────────────────────────────────────
        std::string key;
        if (!read_json_string(json, pos, key)) return false;

        // ── Colon ─────────────────────────────────────────────────────────────
        pos = skip_ws(json, pos);
        if (pos >= json.size() || json[pos] != ':') return false;
        ++pos;

        // ── Value ─────────────────────────────────────────────────────────────
        pos = skip_ws(json, pos);

        if (key == "name") {
            if (!read_json_string(json, pos, call_out.name)) return false;
            got_name = true;
        } else if (key == "arguments") {
            size_t end_pos = 0;
            if (!extract_json_value(json, pos, end_pos, call_out.arguments))
                return false;
            pos = end_pos;
            got_args = true;
        } else {
            // Unknown key — skip its value
            size_t end_pos = 0;
            std::string tmp;
            if (!extract_json_value(json, pos, end_pos, tmp)) return false;
            pos = end_pos;
        }

        // Consume optional comma
        pos = skip_ws(json, pos);
        if (pos < json.size() && json[pos] == ',') ++pos;
    }

    // "name" is required; "arguments" defaults to "{}" if absent
    if (!got_args) call_out.arguments = "{}";
    return got_name;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::parse
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ std::vector<ToolCall> ToolParser::parse(const std::string& text) {
    std::vector<ToolCall> result;
    int id_counter = 0;

    size_t pos = 0;
    while (pos < text.size()) {
        // Find the next opening tag
        size_t open_start = text.find(kOpenTag, pos);
        if (open_start == std::string::npos) break;

        size_t json_start = open_start + kOpenTag.size();

        // Find the matching closing tag
        size_t close_start = text.find(kCloseTag, json_start);
        if (close_start == std::string::npos) break;  // unclosed block

        // Extract JSON substring and trim surrounding whitespace
        std::string block = text.substr(json_start, close_start - json_start);
        size_t f = block.find_first_not_of(" \t\n\r");
        size_t l = block.find_last_not_of(" \t\n\r");
        if (f == std::string::npos) {
            pos = close_start + kCloseTag.size();
            continue;
        }
        block = block.substr(f, l - f + 1);

        // Parse the JSON object
        ToolCall tc;
        if (parse_call_json(block, tc)) {
            tc.id = "call_" + std::to_string(++id_counter);
            result.push_back(std::move(tc));
        } else {
            fprintf(stderr, "[tool_parser] Failed to parse tool_call block: %s\n",
                    block.c_str());
        }

        pos = close_start + kCloseTag.size();
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::has_tool_calls
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ bool ToolParser::has_tool_calls(const std::string& text) {
    return text.find(kOpenTag) != std::string::npos;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolParser::strip_tool_calls
//
// Return `text` with every <tool_call>…</tool_call> block removed.
// Content between the last </tool_call> and the end of text is preserved.
// ─────────────────────────────────────────────────────────────────────────────
/*static*/ std::string ToolParser::strip_tool_calls(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    size_t pos = 0;
    while (pos < text.size()) {
        size_t open_start = text.find(kOpenTag, pos);
        if (open_start == std::string::npos) {
            result += text.substr(pos);
            break;
        }
        // Append text before the opening tag (trim trailing whitespace from it)
        std::string before = text.substr(pos, open_start - pos);
        size_t last_non_ws = before.find_last_not_of(" \t\n\r");
        if (last_non_ws != std::string::npos)
            result += before.substr(0, last_non_ws + 1);

        size_t close_start = text.find(kCloseTag, open_start + kOpenTag.size());
        if (close_start == std::string::npos) break;  // unclosed; stop

        pos = close_start + kCloseTag.size();
    }

    return result;
}

} // namespace finchmoe

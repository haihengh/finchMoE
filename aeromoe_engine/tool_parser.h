// tool_parser.h
// ─────────────────────────────────────────────────────────────────────────────
// Parse Qwen3's tool-call output format.
//
// Qwen3 emits tool calls as one or more:
//
//   <tool_call>
//   {"name": "<function_name>", "arguments": {…}}
//   </tool_call>
//
// ToolParser finds every such block in a raw generated-text string, parses
// the embedded JSON, and returns a vector of ToolCall objects.  The parser
// is intentionally minimal: it handles the known Qwen3 output format robustly
// without pulling in a general JSON library.
//
// Parsing errors (malformed JSON, missing required keys) are soft — the block
// is silently skipped so that partial model output doesn't crash the loop.
// Use ToolParser::last_error() to inspect any parse problem.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "tool_schema.h"

#include <string>
#include <vector>

namespace aeromoe {

class ToolParser {
public:
    // ── Main entry points ─────────────────────────────────────────────────────

    // Parse all <tool_call>…</tool_call> blocks in `text`.
    // Each call gets a unique ID ("call_1", "call_2", …).
    // Blocks that fail to parse are skipped; see last_error().
    static std::vector<ToolCall> parse(const std::string& text);

    // Return true if `text` contains at least one <tool_call> opening tag.
    static bool has_tool_calls(const std::string& text);

    // Remove every <tool_call>…</tool_call> block from `text` and return the
    // residual (the "thinking" prefix the model may emit before the first call).
    static std::string strip_tool_calls(const std::string& text);

    // ── Low-level helpers (exposed for tests) ─────────────────────────────────

    // Parse a single raw JSON object string of the form
    //   {"name": "…", "arguments": {…}}
    // and populate `call_out`.  Returns false on failure.
    static bool parse_call_json(const std::string& json, ToolCall& call_out);

    // Extract a balanced JSON value (object, array, string, or primitive)
    // starting at `pos` inside `s`.  On success sets `end_pos` to the first
    // character after the value and sets `raw_out` to the raw substring.
    static bool extract_json_value(const std::string& s, size_t pos,
                                    size_t& end_pos, std::string& raw_out);

    // Read a JSON string (double-quoted) starting at `pos` in `s`.
    // Writes the unescaped value to `out` and advances `pos` past the
    // closing quote.  Returns false on syntax error.
    static bool read_json_string(const std::string& s, size_t& pos,
                                  std::string& out);

private:
    // "<tool_call>" and "</tool_call>" sentinel strings
    static const std::string kOpenTag;
    static const std::string kCloseTag;
};

} // namespace aeromoe

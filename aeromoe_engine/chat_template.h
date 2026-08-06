// chat_template.h
// ─────────────────────────────────────────────────────────────────────────────
// Qwen3 ChatML prompt assembly.
//
// Qwen3 uses the ChatML conversation format:
//
//   <|im_start|>system
//   {system_content}
//   <|im_end|>
//   <|im_start|>user
//   {user_content}
//   <|im_end|>
//   <|im_start|>assistant
//   {assistant_content}
//   <|im_end|>
//
// When tools are provided they are injected into the system prompt as a
// structured JSON Schema block following Qwen3's official tool-calling
// template.  Tool results use the "tool" role:
//
//   <|im_start|>tool
//   <tool_response>
//   {result_content}
//   </tool_response>
//   <|im_end|>
//
// The prompt produced by format_prompt() ends with:
//
//   <|im_start|>assistant
//
// (no closing tag) so the model continues from the next token.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "tool_schema.h"

#include <string>
#include <vector>

namespace aeromoe {

// ── Role ──────────────────────────────────────────────────────────────────────

enum class Role : uint8_t {
    System    = 0,
    User      = 1,
    Assistant = 2,
    Tool      = 3,    // tool-result turn
};

inline const char* role_name(Role r) {
    switch (r) {
        case Role::System:    return "system";
        case Role::User:      return "user";
        case Role::Assistant: return "assistant";
        case Role::Tool:      return "tool";
    }
    return "user";
}

// ── ChatMessage ───────────────────────────────────────────────────────────────

struct ChatMessage {
    Role        role;
    std::string content;

    // Only used when role == Role::Tool:
    // The id of the ToolCall this result responds to.
    std::string tool_call_id;

    // Convenience constructors
    static ChatMessage system   (std::string c)
        { return {Role::System,    std::move(c)}; }
    static ChatMessage user     (std::string c)
        { return {Role::User,      std::move(c)}; }
    static ChatMessage assistant(std::string c)
        { return {Role::Assistant, std::move(c)}; }
    static ChatMessage tool_result(std::string call_id, std::string c)
        { return {Role::Tool, std::move(c), std::move(call_id)}; }
};

// ── JSON Schema serialisation ─────────────────────────────────────────────────

// Serialise a ToolSchema to the JSON object format Qwen3 expects in the
// system prompt:
//
//   {
//     "type": "function",
//     "function": {
//       "name": "...",
//       "description": "...",
//       "parameters": {
//         "type": "object",
//         "properties": { ... },
//         "required": [ ... ]
//       }
//     }
//   }
std::string tool_schema_to_json(const ToolSchema& schema);

// ── Prompt formatting ─────────────────────────────────────────────────────────

// Build the full ChatML prompt string.
//
//   messages : conversation history (system + user/assistant turns + tool results)
//   tools    : registered tool schemas to inject; empty = no tool block
//
// If `messages` contains no system message and `tools` is non-empty, a default
// system message is synthesised.  If a system message exists and tools are
// provided, the tool instructions are appended to that system message.
//
// The returned string is ready to tokenise with
//   tokenizer.encode_with_special(prompt).
std::string format_prompt(const std::vector<ChatMessage>& messages,
                           const std::vector<ToolSchema>&  tools = {});

// Append an open <|im_start|>assistant turn to `prompt` with `partial` as
// the start of the assistant's reply.  Useful for few-shot and constrained
// generation where the caller has already started writing the response.
std::string append_assistant_partial(const std::string& prompt,
                                      const std::string& partial = "");

// Append a tool-result turn to an existing prompt.
// Equivalent to appending ChatMessage::tool_result(...) and re-formatting, but
// avoids having to rebuild the whole history when the caller is streaming.
std::string append_tool_result(const std::string& prompt,
                                const ToolResult&  result);

} // namespace aeromoe

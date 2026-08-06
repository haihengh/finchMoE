// chat_template.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementation of Qwen3 ChatML prompt formatting and tool-schema JSON
// serialisation.
//
// Design notes
// ────────────
// • No external JSON library is used.  Tool schemas are simple objects with
//   known structure, so we build their JSON by string concatenation.  The
//   resulting strings are functionally identical to what Qwen3's Python
//   tokeniser would produce.
//
// • The tool instructions injected into the system prompt are copied verbatim
//   from Qwen3's official tool-calling template (qwen3_template.jinja) with
//   minor formatting adjustments for readability.
//
// • json_escape() ensures that user-supplied strings (names, descriptions)
//   cannot break the JSON structure — all double-quotes and backslashes are
//   escaped before embedding.
// ─────────────────────────────────────────────────────────────────────────────
#include "chat_template.h"

#include <sstream>
#include <string>
#include <vector>

namespace finchmoe {

// ── JSON string helpers ───────────────────────────────────────────────────────

// Escape a string for embedding inside a JSON double-quoted value.
// Only escapes the characters strictly required by the JSON spec.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    // Control character — emit \uXXXX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// tool_schema_to_json
//
// Serialise one ToolSchema to the format Qwen3 expects in the system prompt:
//
//   {
//     "type": "function",
//     "function": {
//       "name": "...",
//       "description": "...",
//       "parameters": {
//         "type": "object",
//         "properties": {
//           "param_name": { "type": "...", "description": "...",
//                           "enum": [...] },
//           ...
//         },
//         "required": ["param1", "param2"]
//       }
//     }
//   }
// ─────────────────────────────────────────────────────────────────────────────
std::string tool_schema_to_json(const ToolSchema& schema) {
    std::ostringstream os;
    os << "{"
       << "\"type\":\"function\","
       << "\"function\":{"
       <<   "\"name\":\""        << json_escape(schema.name)        << "\","
       <<   "\"description\":\"" << json_escape(schema.description) << "\","
       <<   "\"parameters\":{"
       <<     "\"type\":\"object\","
       <<     "\"properties\":{";

    bool first_prop = true;
    std::vector<std::string> required_names;

    for (const auto& p : schema.params) {
        if (!first_prop) os << ',';
        first_prop = false;

        os << "\"" << json_escape(p.name) << "\":{"
           << "\"type\":\""        << param_type_name(p.type)        << "\","
           << "\"description\":\"" << json_escape(p.description)     << "\"";

        if (!p.enum_values.empty()) {
            os << ",\"enum\":[";
            for (size_t i = 0; i < p.enum_values.size(); ++i) {
                if (i > 0) os << ',';
                os << '"' << json_escape(p.enum_values[i]) << '"';
            }
            os << ']';
        }
        os << '}';

        if (p.required) required_names.push_back(p.name);
    }

    os << "},";  // end "properties"

    // "required" array
    os << "\"required\":[";
    for (size_t i = 0; i < required_names.size(); ++i) {
        if (i > 0) os << ',';
        os << '"' << json_escape(required_names[i]) << '"';
    }
    os << "]";

    os << "}"   // end "parameters"
       << "}"   // end "function"
       << "}";  // end outer object

    return os.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool instruction block
//
// This is the Qwen3 official tool-instruction text, injected at the end of the
// system prompt when tools are available.  It mirrors:
//   https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/tokenizer_config.json
// ─────────────────────────────────────────────────────────────────────────────
static std::string build_tool_instructions(const std::vector<ToolSchema>& tools) {
    std::ostringstream os;

    os << "\n\n# Tools\n\n"
       << "You may call one or more functions to assist with the user query.\n\n"
       << "You are provided with function signatures within <tools></tools> XML tags:\n"
       << "<tools>\n";

    for (const auto& schema : tools) {
        os << tool_schema_to_json(schema) << "\n";
    }

    os << "</tools>\n\n"
       << "For each function call, return a json object with function name and "
       << "arguments within <tool_call></tool_call> XML tags:\n"
       << "<tool_call>\n"
       << "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
       << "</tool_call>";

    return os.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// format_prompt
// ─────────────────────────────────────────────────────────────────────────────
std::string format_prompt(const std::vector<ChatMessage>& messages,
                           const std::vector<ToolSchema>&  tools) {
    std::ostringstream os;

    const bool has_tools = !tools.empty();

    // Check whether caller provided an explicit system message.
    bool has_system = false;
    for (const auto& m : messages) {
        if (m.role == Role::System) { has_system = true; break; }
    }

    // If tools are requested but there is no system message, synthesise one.
    // We do this before the main loop so the default system message is always
    // the first turn.
    if (has_tools && !has_system) {
        os << "<|im_start|>system\n"
           << "You are a helpful assistant."
           << build_tool_instructions(tools)
           << "\n<|im_end|>\n";
    }

    // Emit all messages in order.
    for (const auto& msg : messages) {
        switch (msg.role) {
            case Role::System:
                os << "<|im_start|>system\n"
                   << msg.content;
                // Append tool instructions to the (possibly caller-supplied)
                // system message when tools are registered.
                if (has_tools) os << build_tool_instructions(tools);
                os << "\n<|im_end|>\n";
                break;

            case Role::User:
                os << "<|im_start|>user\n"
                   << msg.content
                   << "\n<|im_end|>\n";
                break;

            case Role::Assistant:
                os << "<|im_start|>assistant\n"
                   << msg.content
                   << "\n<|im_end|>\n";
                break;

            case Role::Tool:
                // Tool result: Qwen3 uses <|im_start|>tool with <tool_response> tags.
                os << "<|im_start|>tool\n"
                   << "<tool_response>\n"
                   << msg.content
                   << "\n</tool_response>\n"
                   << "<|im_end|>\n";
                break;
        }
    }

    // Open the next (assistant) turn — model generates from here.
    os << "<|im_start|>assistant\n";

    return os.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// append_assistant_partial
// ─────────────────────────────────────────────────────────────────────────────
std::string append_assistant_partial(const std::string& prompt,
                                      const std::string& partial) {
    // format_prompt() already ends with "<|im_start|>assistant\n".
    // Just append the partial text; the model will continue from there.
    return prompt + partial;
}

// ─────────────────────────────────────────────────────────────────────────────
// append_tool_result
//
// Close the current assistant turn (with the tool-call text the model emitted),
// then append a tool-result turn and re-open the assistant turn.
// ─────────────────────────────────────────────────────────────────────────────
std::string append_tool_result(const std::string& prompt,
                                const ToolResult&  result) {
    std::ostringstream os;
    os << prompt          // already ends at "…assistant\n<generated_text>"
       << "\n<|im_end|>\n"
       << "<|im_start|>tool\n"
       << "<tool_response>\n"
       << result.content
       << "\n</tool_response>\n"
       << "<|im_end|>\n"
       << "<|im_start|>assistant\n";
    return os.str();
}

} // namespace finchmoe

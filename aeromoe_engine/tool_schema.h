// tool_schema.h
// ─────────────────────────────────────────────────────────────────────────────
// Plain-data types that represent the tool-calling contract between the model
// and its environment.
//
//  ToolSchema   — describes one registered function (name, description, params)
//  ToolCall     — a single tool invocation parsed from model output
//  ToolResult   — the value returned to the model after executing a ToolCall
//  ToolHandler  — the C++ callable registered to handle a named tool
//
// All types are pure C++ value types with no Metal or ObjC dependencies so
// they can be used freely in headers that are compiled as plain C++.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aeromoe {

// ── ParamType ─────────────────────────────────────────────────────────────────
// JSON Schema primitive types for tool parameter declarations.

enum class ParamType : uint8_t {
    String  = 0,
    Number  = 1,   // floating-point
    Integer = 2,
    Boolean = 3,
    Array   = 4,
    Object  = 5,
    Null    = 6,
};

inline const char* param_type_name(ParamType t) {
    switch (t) {
        case ParamType::String:  return "string";
        case ParamType::Number:  return "number";
        case ParamType::Integer: return "integer";
        case ParamType::Boolean: return "boolean";
        case ParamType::Array:   return "array";
        case ParamType::Object:  return "object";
        case ParamType::Null:    return "null";
    }
    return "string";
}

// ── ToolParam ─────────────────────────────────────────────────────────────────
// One parameter in a tool's JSON Schema definition.

struct ToolParam {
    std::string              name;
    ParamType                type        = ParamType::String;
    std::string              description;
    bool                     required    = false;
    // Optional enumeration constraint — empty means unconstrained.
    std::vector<std::string> enum_values;
};

// ── ToolSchema ────────────────────────────────────────────────────────────────
// Complete JSON Schema description of one callable function.
// Serialised and injected into the system prompt by chat_template.

struct ToolSchema {
    std::string            name;
    std::string            description;
    std::vector<ToolParam> params;
};

// ── ToolCall ──────────────────────────────────────────────────────────────────
// A single tool invocation parsed from the model's <tool_call>…</tool_call>
// block.  `arguments` is the raw JSON string (which may be "{}" or any valid
// JSON object/value).

struct ToolCall {
    std::string id;          // generated: "call_N" (N = 1-based parse order)
    std::string name;        // function name exactly as the model emitted it
    std::string arguments;   // raw JSON — forward to the registered ToolHandler
};

// ── ToolResult ────────────────────────────────────────────────────────────────
// The value returned from a ToolHandler and formatted back into the prompt.
// `content` may be any string: plain text, JSON, an error message, etc.

struct ToolResult {
    std::string tool_call_id;  // matches ToolCall::id
    std::string content;       // handler return value (plain text or JSON)
};

// ── ToolHandler ───────────────────────────────────────────────────────────────
// Signature for the C++ callable registered for a named tool.
// Input:  raw JSON arguments string from the model (may be "{}" for no-arg tools)
// Output: plain text or JSON response string to feed back to the model

using ToolHandler = std::function<std::string(const std::string& /*json_args*/)>;

} // namespace aeromoe

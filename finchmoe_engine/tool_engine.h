// tool_engine.h
// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine — agentic tool-calling loop built on top of InferenceEngine.
//
// Responsibilities
// ────────────────
// 1. Format ChatML prompts (via chat_template) and tokenise them.
// 2. Run the InferenceEngine's generate() in streaming mode, collecting
//    generated text token by token.
// 3. After generation, scan the output for <tool_call>…</tool_call> blocks.
// 4. Parse each block, dispatch to the registered C++ handler, collect the
//    ToolResult, and inject it as a tool-result turn in the prompt.
// 5. Repeat (up to max_rounds) until the model produces a plain text reply
//    with no tool calls.
// 6. Return the final assistant message as a ChatMessage.
//
// Usage
// ─────
//   auto engine = std::make_unique<InferenceEngine>();
//   engine->init("/path/to/model.finchmoe", cfg);
//
//   auto tokenizer = make_tokenizer("/path/to/model.finchmoe");
//
//   ToolEngine te;
//   te.init(engine.get(), tokenizer.get());
//
//   te.register_tool("get_weather", [](const std::string& args) {
//       // parse args JSON, call weather API, return JSON result
//       return R"({"temperature": 22, "condition": "sunny"})";
//   });
//
//   auto reply = te.chat({
//       ChatMessage::system("You are a helpful assistant."),
//       ChatMessage::user("What is the weather in Paris?"),
//   }, {weather_schema});
//   printf("Assistant: %s\n", reply.content.c_str());
//
// Threading
// ─────────
// ToolEngine shares InferenceEngine's threading model: NOT thread-safe.
// Callers must serialise concurrent chat() invocations.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "inference_engine.h"
#include "tokenizer.h"
#include "tool_schema.h"
#include "chat_template.h"
#include "tool_parser.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace finchmoe {

// ── ToolEngineConfig ──────────────────────────────────────────────────────────

struct ToolEngineConfig {
    // Maximum number of tool-call round-trips before returning the last reply.
    // Prevents infinite loops when the model keeps requesting tools.
    uint32_t max_rounds       = 8;

    // Token budget per generation round (prompt + new tokens).
    uint32_t max_new_tokens   = 1024;

    // Sampler settings (forwarded to GenerateConfig on each round).
    float    temperature      = 0.0f;   // 0 = greedy
    float    top_p            = 1.0f;
    uint32_t top_k            = 0;
    float    repetition_penalty = 1.0f;

    // Print tool invocations and results to stderr for debugging.
    bool     verbose          = false;
};

// ── ToolEngine ────────────────────────────────────────────────────────────────

class ToolEngine {
public:
    ToolEngine()  = default;
    ~ToolEngine() = default;

    // Non-copyable (holds raw pointers to external objects).
    ToolEngine(const ToolEngine&)            = delete;
    ToolEngine& operator=(const ToolEngine&) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Bind to a live InferenceEngine and Tokenizer.
    // Both objects must outlive this ToolEngine.
    // Can be called multiple times to rebind (e.g. for hot-swapping configs).
    void init(InferenceEngine* engine, Tokenizer* tokenizer);

    // ── Tool registration ─────────────────────────────────────────────────────

    // Register a handler for the tool named `name`.
    // The handler receives the raw JSON arguments string from the model and
    // should return a plain text or JSON response.
    // Calling register_tool() with an existing name replaces the old handler.
    void register_tool(const std::string& name, ToolHandler handler);

    // Remove a previously registered handler.
    void unregister_tool(const std::string& name);

    // ── Chat ──────────────────────────────────────────────────────────────────

    // Run the multi-turn agentic loop (blocking).
    //
    //   messages : conversation history (system prompt + prior turns)
    //   tools    : tool schemas to inject into the system prompt;
    //              must include schemas for all registered handlers
    //   cfg      : generation and loop parameters
    //
    // Returns the final ChatMessage::assistant reply (role = Role::Assistant).
    // If the model never produces tool calls the result is the first reply.
    // If max_rounds is exhausted, the last reply is returned as-is.
    ChatMessage chat(const std::vector<ChatMessage>& messages,
                     const std::vector<ToolSchema>&  tools,
                     const ToolEngineConfig&          cfg = {});

    // Streaming variant.
    // `on_text` is called incrementally as each token is decoded to text.
    // It may be called with multi-byte UTF-8 sequences split across token
    // boundaries; accumulate if you need whole characters.
    // Returns the final ChatMessage (same as the non-streaming variant).
    ChatMessage chat_stream(
        const std::vector<ChatMessage>&              messages,
        const std::vector<ToolSchema>&               tools,
        std::function<void(const std::string& text)> on_text,
        const ToolEngineConfig&                      cfg = {});

private:
    // ── Internal generation ───────────────────────────────────────────────────

    // Run one generation round starting from `prompt_text`.
    // On return:
    //   `generated_text_out` — raw model output (may contain tool_call blocks)
    //   `on_text`            — called for each decoded token (may be nullptr)
    Status run_round(const std::string&                           prompt_text,
                     const ToolEngineConfig&                      cfg,
                     std::function<void(const std::string& text)> on_text,
                     std::string&                                  generated_text_out);

    // ── Tool dispatch ─────────────────────────────────────────────────────────

    // Invoke the registered handler for `call`.
    // Returns the ToolResult (including a synthesised error message if no
    // handler is registered or the handler throws).
    ToolResult dispatch(const ToolCall& call);

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Build a GenerateConfig from a ToolEngineConfig round.
    GenerateConfig make_gen_config(const ToolEngineConfig& cfg,
                                   std::function<bool(uint32_t)> on_token_id) const;

    // ── State ─────────────────────────────────────────────────────────────────

    InferenceEngine*              engine_    = nullptr;
    Tokenizer*                    tokenizer_ = nullptr;
    std::map<std::string, ToolHandler> handlers_;
};

} // namespace finchmoe

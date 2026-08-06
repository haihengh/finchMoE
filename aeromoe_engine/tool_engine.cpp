// tool_engine.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Implementation of the ToolEngine multi-turn agentic loop.
//
// Round-trip flow (per generation round)
// ──────────────────────────────────────
//   1. format_prompt(history, tools)  →  prompt_text
//   2. tokenize: encode_with_special(prompt_text)  →  token IDs
//   3. InferenceEngine::generate(token_ids, gen_cfg)
//      - streaming callback: decode each token to text, accumulate
//   4. ToolParser::parse(generated_text)
//      - if empty: done — return the text as the final assistant reply
//   5. For each ToolCall:
//      a. dispatch() → ToolResult
//      b. append_tool_result(prompt, result) — extends prompt_text in-place
//   6. Append the assistant's tool-call text as an assistant turn to history,
//      then append tool-result turns, then format the next round's prompt.
//   7. Go to step 2 (up to max_rounds times).
//
// Prompt accumulation strategy
// ─────────────────────────────
// Rather than re-building the full history from the ChatMessage vector each
// round, we extend the prompt string directly via append_tool_result().
// This avoids quadratic string rebuilding and is safe because the prompt is
// passed to the tokenizer as a new call each round (InferenceEngine does not
// do incremental prefix caching across calls yet).
//
// The ChatMessage history vector IS updated each round so that the caller's
// messages list reflects the full conversation if they keep it.
// ─────────────────────────────────────────────────────────────────────────────
#include "tool_engine.h"

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace aeromoe {

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::init
// ─────────────────────────────────────────────────────────────────────────────
void ToolEngine::init(InferenceEngine* engine, Tokenizer* tokenizer) {
    assert(engine    && "InferenceEngine must not be null");
    assert(tokenizer && "Tokenizer must not be null");
    engine_    = engine;
    tokenizer_ = tokenizer;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::register_tool / unregister_tool
// ─────────────────────────────────────────────────────────────────────────────
void ToolEngine::register_tool(const std::string& name, ToolHandler handler) {
    handlers_[name] = std::move(handler);
}

void ToolEngine::unregister_tool(const std::string& name) {
    handlers_.erase(name);
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::dispatch
//
// Invoke the C++ handler registered for call.name.  Returns a ToolResult with
// either the handler's return value or a human-readable error string.
// ─────────────────────────────────────────────────────────────────────────────
ToolResult ToolEngine::dispatch(const ToolCall& call) {
    ToolResult res;
    res.tool_call_id = call.id;

    auto it = handlers_.find(call.name);
    if (it == handlers_.end()) {
        res.content = "{\"error\": \"No handler registered for tool '" + call.name + "'\"}";
        fprintf(stderr, "[tool_engine] WARNING: no handler for tool '%s'\n",
                call.name.c_str());
        return res;
    }

    try {
        res.content = it->second(call.arguments);
    } catch (const std::exception& e) {
        res.content = std::string("{\"error\": \"Handler threw: ") + e.what() + "\"}";
        fprintf(stderr, "[tool_engine] Handler for '%s' threw: %s\n",
                call.name.c_str(), e.what());
    } catch (...) {
        res.content = "{\"error\": \"Handler threw an unknown exception\"}";
        fprintf(stderr, "[tool_engine] Handler for '%s' threw unknown exception\n",
                call.name.c_str());
    }

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::make_gen_config
// ─────────────────────────────────────────────────────────────────────────────
GenerateConfig ToolEngine::make_gen_config(
    const ToolEngineConfig&     cfg,
    std::function<bool(uint32_t)> on_token_id) const
{
    GenerateConfig gen;
    gen.max_new_tokens      = cfg.max_new_tokens;
    gen.temperature         = cfg.temperature;
    gen.top_p               = cfg.top_p;
    gen.top_k               = cfg.top_k;
    gen.repetition_penalty  = cfg.repetition_penalty;
    gen.stop_token_ids      = {TOK_IM_END};  // stop at <|im_end|>
    gen.on_token            = std::move(on_token_id);
    return gen;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::run_round
//
// Tokenise `prompt_text`, run one generation pass, stream-decode each token,
// and return the full generated text.
// ─────────────────────────────────────────────────────────────────────────────
Status ToolEngine::run_round(
    const std::string&                           prompt_text,
    const ToolEngineConfig&                      cfg,
    std::function<void(const std::string& text)> on_text,
    std::string&                                  generated_text_out)
{
    // Tokenise the prompt (includes special tokens like <|im_start|>)
    auto prompt_ids = tokenizer_->encode_with_special(prompt_text);
    if (prompt_ids.empty()) {
        fprintf(stderr, "[tool_engine] WARNING: empty token sequence after encoding\n");
        return Status::InvalidArg;
    }

    // Accumulate generated text token by token
    generated_text_out.clear();

    // streaming callback: decode each new token and call on_text
    auto on_token_id = [&](uint32_t tok_id) -> bool {
        // Skip special control tokens from the visible output
        if (tokenizer_->is_special(tok_id)) {
            // <|im_end|> means end of turn — stop generation
            if (tok_id == TOK_IM_END) return false;
            return true;  // other special tokens: continue but don't emit
        }
        std::string piece = tokenizer_->decode_token(tok_id);
        generated_text_out += piece;
        if (on_text) on_text(piece);
        return true;
    };

    GenerateConfig gen = make_gen_config(cfg, on_token_id);

    GenerateResult result;
    Status s = engine_->generate(prompt_ids, gen, &result);
    if (!ok(s)) {
        fprintf(stderr, "[tool_engine] generate() failed: %s\n", status_str(s));
        return s;
    }

    return Status::OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::chat
// ─────────────────────────────────────────────────────────────────────────────
ChatMessage ToolEngine::chat(const std::vector<ChatMessage>& messages,
                              const std::vector<ToolSchema>&  tools,
                              const ToolEngineConfig&          cfg) {
    return chat_stream(messages, tools, nullptr, cfg);
}

// ─────────────────────────────────────────────────────────────────────────────
// ToolEngine::chat_stream
//
// The main agentic loop.
// ─────────────────────────────────────────────────────────────────────────────
ChatMessage ToolEngine::chat_stream(
    const std::vector<ChatMessage>&              messages,
    const std::vector<ToolSchema>&               tools,
    std::function<void(const std::string& text)> on_text,
    const ToolEngineConfig&                      cfg)
{
    assert(engine_    && "ToolEngine not initialised — call init() first");
    assert(tokenizer_ && "ToolEngine not initialised — call init() first");

    // ── Build the initial prompt ──────────────────────────────────────────────
    // format_prompt() returns the full ChatML string including the trailing
    // "<|im_start|>assistant\n" that opens the model's reply.
    std::string prompt = format_prompt(messages, tools);

    // We maintain a local mutable history for appending tool results.
    // (A copy of `messages` extended with assistant + tool turns each round.)
    std::vector<ChatMessage> history = messages;

    std::string last_reply;

    for (uint32_t round = 0; round < cfg.max_rounds; ++round) {
        if (cfg.verbose) {
            fprintf(stderr, "[tool_engine] --- Round %u ---\n", round + 1);
        }

        // ── Generate ─────────────────────────────────────────────────────────
        std::string generated;
        Status s = run_round(prompt, cfg, on_text, generated);
        if (!ok(s)) {
            // Return whatever we have so far rather than crashing.
            return ChatMessage::assistant(last_reply.empty() ? "(generation error)" : last_reply);
        }

        if (cfg.verbose) {
            fprintf(stderr, "[tool_engine] Generated (%zu chars): %s\n",
                    generated.size(), generated.c_str());
        }

        // ── Check for tool calls ──────────────────────────────────────────────
        if (!ToolParser::has_tool_calls(generated)) {
            // No tool calls — this is the final reply.
            last_reply = ToolParser::strip_tool_calls(generated);
            break;
        }

        auto calls = ToolParser::parse(generated);
        if (calls.empty()) {
            // Parser found tags but couldn't extract valid calls; treat as final.
            last_reply = ToolParser::strip_tool_calls(generated);
            break;
        }

        // Save this as the last reply in case we hit max_rounds next iteration.
        last_reply = ToolParser::strip_tool_calls(generated);

        // ── Append assistant's tool-call turn to history ──────────────────────
        // The assistant turn includes the tool_call blocks verbatim so the
        // model can see what it requested.
        history.push_back(ChatMessage::assistant(generated));

        // ── Dispatch tool calls and append results ────────────────────────────
        for (const auto& call : calls) {
            if (cfg.verbose) {
                fprintf(stderr, "[tool_engine] Calling tool '%s' args: %s\n",
                        call.name.c_str(), call.arguments.c_str());
            }

            ToolResult res = dispatch(call);

            if (cfg.verbose) {
                fprintf(stderr, "[tool_engine] Tool '%s' returned: %s\n",
                        call.name.c_str(), res.content.c_str());
            }

            // Append to prompt string directly (avoids full re-format)
            // append_tool_result() extends prompt with:
            //   <|im_end|>\n<|im_start|>tool\n<tool_response>\n…\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n
            prompt += "\n<|im_end|>\n";  // close the assistant turn we just opened
            prompt += "<|im_start|>tool\n"
                      "<tool_response>\n"
                    + res.content
                    + "\n</tool_response>\n"
                      "<|im_end|>\n";

            // Also keep history up-to-date for the caller.
            history.push_back(ChatMessage::tool_result(res.tool_call_id, res.content));
        }

        // Re-open the assistant turn for the next round.
        prompt += "<|im_start|>assistant\n";

        // If this was the last allowed round, the next iteration won't run —
        // last_reply already holds the stripped text; we'll return it below.
    }

    return ChatMessage::assistant(last_reply);
}

} // namespace aeromoe

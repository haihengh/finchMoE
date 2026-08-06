// tools/chat_main.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Interactive CLI chat demo for AeroMoE with tool-calling support.
//
// Usage
// ─────
//   ./aeromoe_chat <model.aeromoe> [options]
//
// Options
//   --max-tokens N    max new tokens per generation round (default 512)
//   --temperature F   sampling temperature, 0=greedy (default 0.0)
//   --top-p F         nucleus sampling threshold (default 1.0)
//   --top-k N         top-k sampling (default 0 = off)
//   --rounds N        max tool-call round-trips (default 8)
//   --verbose         print tool dispatch diagnostics
//   --no-tools        disable demo tools (plain chat mode)
//   --system "text"   override the default system prompt
//
// Built-in demo tools
// ───────────────────
//   calculator        evaluate a simple arithmetic expression
//   get_time          return the current UTC date/time
//   echo              echo a message back to the model
//
// Interactive session
// ────────────────────
// Type a message and press Enter.  The model replies, potentially calling
// tools first.  Special commands:
//   /quit  or  /exit  — exit the program
//   /clear            — clear conversation history (keeps system prompt)
//   /stats            — print runtime statistics
//   /tools            — toggle demo tools on/off
//   /system "text"    — replace the system prompt
// ─────────────────────────────────────────────────────────────────────────────

#include "../inference_engine.h"
#include "../tokenizer.h"
#include "../chat_template.h"
#include "../tool_schema.h"
#include "../tool_engine.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace aeromoe;

// ─────────────────────────────────────────────────────────────────────────────
// Demo tool implementations
// ─────────────────────────────────────────────────────────────────────────────

// ── calculator ────────────────────────────────────────────────────────────────
// Evaluates simple arithmetic expressions (+, -, *, /).
// Input JSON: {"expression": "3 + 4 * 2"}
// Output JSON: {"result": 11.0}

namespace {

// Minimal recursive-descent arithmetic evaluator.
struct Calc {
    const char* p;
    double parse_expr();
    double parse_term();
    double parse_factor();
    void skip_ws() { while (*p && (*p == ' ' || *p == '\t')) ++p; }
};

double Calc::parse_factor() {
    skip_ws();
    if (*p == '(') {
        ++p;
        double v = parse_expr();
        skip_ws();
        if (*p == ')') ++p;
        return v;
    }
    if (*p == '-') { ++p; return -parse_factor(); }
    char* end = nullptr;
    double v = std::strtod(p, &end);
    if (end == p) throw std::runtime_error("expected number");
    p = end;
    return v;
}

double Calc::parse_term() {
    double v = parse_factor();
    skip_ws();
    while (*p == '*' || *p == '/') {
        char op = *p++;
        double r = parse_factor();
        v = (op == '*') ? v * r : v / r;
        skip_ws();
    }
    return v;
}

double Calc::parse_expr() {
    double v = parse_term();
    skip_ws();
    while (*p == '+' || *p == '-') {
        char op = *p++;
        double r = parse_term();
        v = (op == '+') ? v + r : v - r;
        skip_ws();
    }
    return v;
}

}  // anonymous namespace

// Extract a JSON string value for `key` from a flat JSON object.
// Returns "" if not found.  Very simple; handles only string values.
static std::string json_str_val(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t p = json.find(pattern);
    if (p == std::string::npos) return "";
    p += pattern.size();
    while (p < json.size() && (json[p] == ' ' || json[p] == ':')) ++p;
    if (p >= json.size() || json[p] != '"') return "";
    ++p;
    std::string val;
    while (p < json.size() && json[p] != '"') {
        if (json[p] == '\\') { ++p; if (p < json.size()) val += json[p]; }
        else val += json[p];
        ++p;
    }
    return val;
}

static std::string tool_calculator(const std::string& args) {
    std::string expr = json_str_val(args, "expression");
    if (expr.empty()) return "{\"error\": \"missing 'expression' field\"}";
    try {
        Calc c{expr.c_str()};
        double result = c.parse_expr();
        char buf[64];
        // Use integer representation if result is whole
        if (result == std::floor(result) && std::abs(result) < 1e15)
            snprintf(buf, sizeof(buf), "{\"result\": %.0f}", result);
        else
            snprintf(buf, sizeof(buf), "{\"result\": %.10g}", result);
        return buf;
    } catch (const std::exception& e) {
        return std::string("{\"error\": \"") + e.what() + "\"}";
    }
}

// ── get_time ──────────────────────────────────────────────────────────────────
// Input JSON: {} (no parameters)
// Output JSON: {"utc": "2025-01-01T12:00:00Z"}

static std::string tool_get_time(const std::string& /*args*/) {
    std::time_t now = std::time(nullptr);
    struct tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string("{\"utc\": \"") + buf + "\"}";
}

// ── echo ──────────────────────────────────────────────────────────────────────
// Echoes a message back to the model — useful for testing the tool loop.
// Input JSON: {"message": "..."}
// Output JSON: {"echo": "..."}

static std::string tool_echo(const std::string& args) {
    std::string msg = json_str_val(args, "message");
    if (msg.empty()) return "{\"echo\": \"\"}";
    // Quick json_escape
    std::string out = "{\"echo\": \"";
    for (char c : msg) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    out += "\"}";
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Demo tool schemas
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<ToolSchema> make_demo_schemas() {
    ToolSchema calc;
    calc.name        = "calculator";
    calc.description = "Evaluate an arithmetic expression and return the numeric result. "
                       "Supports +, -, *, / and parentheses.";
    calc.params.push_back({
        "expression", ParamType::String,
        "The arithmetic expression to evaluate, e.g. \"(3 + 4) * 2\"",
        /*required=*/true, {}
    });

    ToolSchema get_time;
    get_time.name        = "get_time";
    get_time.description = "Return the current UTC date and time in ISO 8601 format.";
    // No parameters needed.

    ToolSchema echo;
    echo.name        = "echo";
    echo.description = "Echo a message back. Useful for testing the tool-calling loop.";
    echo.params.push_back({
        "message", ParamType::String,
        "The message to echo back.",
        /*required=*/true, {}
    });

    return {calc, get_time, echo};
}

// ─────────────────────────────────────────────────────────────────────────────
// Argument parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

static void print_usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s <model.aeromoe> [options]\n"
        "\n"
        "Options:\n"
        "  --max-tokens N    max new tokens per round (default 512)\n"
        "  --temperature F   sampling temperature (default 0.0, greedy)\n"
        "  --top-p F         nucleus threshold (default 1.0)\n"
        "  --top-k N         top-k (default 0 = off)\n"
        "  --rounds N        max tool-call rounds (default 8)\n"
        "  --verbose         print tool dispatch diagnostics\n"
        "  --no-tools        disable demo tools\n"
        "  --system TEXT     override system prompt\n"
        "  --memory-gb F     memory budget in GiB (default 3.5)\n"
        "\n"
        "Interactive commands:\n"
        "  /quit /exit       exit\n"
        "  /clear            clear history\n"
        "  /stats            print runtime stats\n"
        "  /tools            toggle demo tools\n"
        "  /system TEXT      set system prompt\n",
        argv0);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    // ── Parse arguments ───────────────────────────────────────────────────────
    std::string model_path = argv[1];
    ToolEngineConfig te_cfg;
    te_cfg.max_new_tokens = 512;
    te_cfg.max_rounds     = 8;
    te_cfg.temperature    = 0.0f;
    te_cfg.top_p          = 1.0f;
    te_cfg.top_k          = 0;

    bool        use_tools  = true;
    double      memory_gb  = 3.5;
    std::string system_prompt =
        "You are a helpful, knowledgeable assistant. "
        "When the user asks for calculations or the current time, "
        "use the available tools rather than guessing.";

    for (int i = 2; i < argc; ++i) {
        auto arg = std::string(argv[i]);
        if (arg == "--max-tokens" && i + 1 < argc)
            te_cfg.max_new_tokens = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc)
            te_cfg.temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc)
            te_cfg.top_p = std::stof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc)
            te_cfg.top_k = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--rounds" && i + 1 < argc)
            te_cfg.max_rounds = (uint32_t)std::stoul(argv[++i]);
        else if (arg == "--verbose")
            te_cfg.verbose = true;
        else if (arg == "--no-tools")
            use_tools = false;
        else if (arg == "--system" && i + 1 < argc)
            system_prompt = argv[++i];
        else if (arg == "--memory-gb" && i + 1 < argc)
            memory_gb = std::stod(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]); return 0;
        }
    }

    // ── Load model and tokenizer ──────────────────────────────────────────────
    fprintf(stderr, "[chat] Loading model: %s\n", model_path.c_str());

    InferenceConfig ie_cfg;
    ie_cfg.engine.memory_budget_bytes =
        static_cast<size_t>(memory_gb * 1024 * 1024 * 1024);
    ie_cfg.engine.soft_budget_bytes =
        static_cast<size_t>(memory_gb * 0.9 * 1024 * 1024 * 1024);
    ie_cfg.runner.max_seq_len = 4096;

    InferenceEngine engine;
    if (!ok(engine.init(model_path, ie_cfg))) {
        fprintf(stderr, "[chat] Failed to initialise InferenceEngine.\n");
        return 1;
    }
    fprintf(stderr, "[chat] Model loaded.\n");

    auto tokenizer = make_tokenizer(model_path);
    if (!tokenizer) {
        fprintf(stderr, "[chat] Failed to load tokenizer.\n"
                        "       Expected: %s.tiktoken\n"
                        "       (run aeromoe_convert.py to generate it)\n",
                model_path.c_str());
        return 1;
    }
    fprintf(stderr, "[chat] Tokenizer loaded (%u tokens).\n",
            tokenizer->vocab_size());

    // ── Build ToolEngine ──────────────────────────────────────────────────────
    ToolEngine te;
    te.init(&engine, tokenizer.get());

    auto demo_schemas = make_demo_schemas();

    if (use_tools) {
        te.register_tool("calculator", tool_calculator);
        te.register_tool("get_time",   tool_get_time);
        te.register_tool("echo",       tool_echo);
        fprintf(stderr, "[chat] Demo tools registered: calculator, get_time, echo\n");
    }

    // ── Conversation state ────────────────────────────────────────────────────
    std::vector<ChatMessage> history;
    history.push_back(ChatMessage::system(system_prompt));

    fprintf(stderr, "\n=== AeroMoE Interactive Chat ===\n");
    fprintf(stderr, "Type a message and press Enter.  "
                    "/quit to exit, /clear to reset history.\n\n");

    // ── REPL ──────────────────────────────────────────────────────────────────
    while (true) {
        printf("You: ");
        fflush(stdout);

        std::string line;
        if (!std::getline(std::cin, line)) break;  // EOF

        // Trim leading/trailing whitespace
        size_t first = line.find_first_not_of(" \t\r\n");
        size_t last  = line.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) continue;
        line = line.substr(first, last - first + 1);
        if (line.empty()) continue;

        // ── Special commands ──────────────────────────────────────────────────
        if (line == "/quit" || line == "/exit") {
            fprintf(stderr, "\n[chat] Goodbye!\n");
            break;
        }
        if (line == "/clear") {
            // Keep system message, clear the rest
            std::string sys_content = history[0].content;
            history.clear();
            history.push_back(ChatMessage::system(sys_content));
            fprintf(stderr, "[chat] History cleared.\n");
            continue;
        }
        if (line == "/stats") {
            engine.print_stats();
            continue;
        }
        if (line == "/tools") {
            use_tools = !use_tools;
            if (use_tools) {
                te.register_tool("calculator", tool_calculator);
                te.register_tool("get_time",   tool_get_time);
                te.register_tool("echo",       tool_echo);
                fprintf(stderr, "[chat] Demo tools ENABLED.\n");
            } else {
                te.unregister_tool("calculator");
                te.unregister_tool("get_time");
                te.unregister_tool("echo");
                fprintf(stderr, "[chat] Demo tools DISABLED.\n");
            }
            continue;
        }
        if (line.rfind("/system ", 0) == 0) {
            system_prompt = line.substr(8);
            // Replace system message at front of history
            if (!history.empty() && history[0].role == Role::System)
                history[0].content = system_prompt;
            else
                history.insert(history.begin(), ChatMessage::system(system_prompt));
            fprintf(stderr, "[chat] System prompt updated.\n");
            continue;
        }

        // ── Normal user message ───────────────────────────────────────────────
        history.push_back(ChatMessage::user(line));

        const auto& active_schemas =
            use_tools ? demo_schemas : std::vector<ToolSchema>{};

        printf("Assistant: ");
        fflush(stdout);

        // Stream each token to stdout as it is generated
        auto on_text = [](const std::string& text) {
            fputs(text.c_str(), stdout);
            fflush(stdout);
        };

        ChatMessage reply = te.chat_stream(history, active_schemas, on_text, te_cfg);

        printf("\n\n");

        // Append the final assistant reply to history (clean text, no tool tags)
        history.push_back(reply);
    }

    return 0;
}

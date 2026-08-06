// tools/generate_main.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Minimal command-line smoke-test for FinchMoE.
//
// Usage:
//   finchmoe_generate <model.finchmoe> [prompt_token ...] [options]
//
// Options:
//   --max-tokens N      Maximum new tokens to generate (default: 64)
//   --temperature F     Sampling temperature (default: 0.0 = greedy)
//   --top-p F           Nucleus sampling p (default: 1.0)
//   --seed N            RNG seed (default: 42)
//   --verbose           Print engine diagnostics
//
// Tokens are passed as raw uint32 integers on the command line.
// In a real deployment the prompt would first go through a Qwen3 tokenizer
// (e.g. via a Python bridge); here we accept pre-tokenized input for testing.
//
// Build:
//   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -t finchmoe_generate
// ─────────────────────────────────────────────────────────────────────────────

#include "../inference_engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>

using namespace finchmoe;

// ── helpers ───────────────────────────────────────────────────────────────────

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s <model.finchmoe> [--max-tokens N] [--temperature F]\n"
        "          [--top-p F] [--seed N] [--verbose] <token_id> ...\n\n"
        "Example (Qwen3 greeting test):\n"
        "  %s model.finchmoe --max-tokens 32 1 9906 11\n"
        "    → encodes BOS + 'Hello' tokens then generates a continuation\n\n",
        argv0, argv0);
}

static double wall_ms() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()
    ).count() / 1000.0;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 2) { usage(argv[0]); return 1; }

    // ── parse arguments ───────────────────────────────────────────────────────
    std::string model_path;
    std::vector<uint32_t> prompt_tokens;

    InferenceConfig ie_cfg;
    GenerateConfig  gen_cfg;
    gen_cfg.max_new_tokens = 64;
    gen_cfg.temperature    = 0.0f;  // greedy by default

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (model_path.empty() && arg[0] != '-') {
            model_path = arg;
            continue;
        }

        if (strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            gen_cfg.max_new_tokens = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(arg, "--temperature") == 0 && i + 1 < argc) {
            gen_cfg.temperature = (float)atof(argv[++i]);
        } else if (strcmp(arg, "--top-p") == 0 && i + 1 < argc) {
            gen_cfg.top_p = (float)atof(argv[++i]);
        } else if (strcmp(arg, "--seed") == 0 && i + 1 < argc) {
            gen_cfg.rng_seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(arg, "--verbose") == 0) {
            ie_cfg.engine.verbose = true;
        } else if (arg[0] != '-') {
            // Treat as a token ID
            prompt_tokens.push_back((uint32_t)atoi(arg));
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg);
            usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "Error: no model path specified.\n");
        usage(argv[0]);
        return 1;
    }

    if (prompt_tokens.empty()) {
        // Default: BOS token = 1 (Qwen3 uses token ID 1 as BOS)
        fprintf(stderr, "[generate] No tokens provided; using BOS token (1)\n");
        prompt_tokens.push_back(1);
    }

    // ── initialise engine ─────────────────────────────────────────────────────
    fprintf(stderr, "[generate] Loading model from: %s\n", model_path.c_str());
    fprintf(stderr, "[generate] Prompt tokens (%zu): ",
            prompt_tokens.size());
    for (uint32_t t : prompt_tokens) fprintf(stderr, "%u ", t);
    fprintf(stderr, "\n");

    double t_load_start = wall_ms();

    InferenceEngine engine;
    Status s = engine.init(model_path, ie_cfg);
    if (!ok(s)) {
        fprintf(stderr, "[generate] init() failed: %s\n", status_str(s));
        return 2;
    }

    fprintf(stderr, "[generate] Model loaded in %.1f ms\n",
            wall_ms() - t_load_start);

    // ── set up streaming output ────────────────────────────────────────────────
    // In greedy / temperature mode, print each token ID as it is generated.
    gen_cfg.on_token = [](uint32_t tok) -> bool {
        printf("%u\n", tok);
        fflush(stdout);
        return true;  // continue generating
    };

    // ── generate ──────────────────────────────────────────────────────────────
    fprintf(stderr,
            "[generate] Generating up to %u tokens "
            "(temperature=%.2f, top_p=%.2f) ...\n",
            gen_cfg.max_new_tokens, gen_cfg.temperature, gen_cfg.top_p);

    GenerateResult result;
    s = engine.generate(prompt_tokens, gen_cfg, &result);

    if (!ok(s)) {
        fprintf(stderr, "[generate] generate() failed: %s\n", status_str(s));
        return 3;
    }

    // ── report ────────────────────────────────────────────────────────────────
    double total_ms = result.prefill_ms + result.decode_ms;
    double tok_per_sec = (result.output_len > 0)
        ? result.output_len / (result.decode_ms / 1000.0)
        : 0.0;

    fprintf(stderr, "\n[generate] Done.\n");
    fprintf(stderr, "  Prompt tokens : %u\n",   result.prompt_len);
    fprintf(stderr, "  Output tokens : %u\n",   result.output_len);
    fprintf(stderr, "  Prefill time  : %.1f ms (%.1f ms/tok)\n",
            result.prefill_ms,
            result.prompt_len > 0 ? result.prefill_ms / result.prompt_len : 0.0);
    fprintf(stderr, "  Decode time   : %.1f ms (%.1f tok/s)\n",
            result.decode_ms, tok_per_sec);
    fprintf(stderr, "  Total time    : %.1f ms\n", total_ms);

    const char* stop_reason =
        result.stopped_by_eos   ? "EOS token"    :
        result.stopped_by_user  ? "user callback" :
        result.stopped_by_len   ? "max_tokens"    : "unknown";
    fprintf(stderr, "  Stop reason   : %s\n", stop_reason);

    engine.print_stats(stderr);
    engine.shutdown();

    return 0;
}

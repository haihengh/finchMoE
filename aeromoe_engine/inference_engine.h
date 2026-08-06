// inference_engine.h
// ─────────────────────────────────────────────────────────────────────────────
// Public API for AeroMoE end-to-end inference.
//
// InferenceEngine owns the full stack:
//   EngineCore  — file I/O, dense weights, expert cache
//   KVCacheManager — paged KV storage
//   ModelRunner — token-level forward pass
//   Sampler     — next-token selection
//
// Usage sketch
// ─────────────
//   InferenceConfig ie_cfg;
//   ie_cfg.engine.memory_budget_bytes = 3.5 * 1024*1024*1024;
//   ie_cfg.runner.max_seq_len = 2048;
//
//   InferenceEngine ie;
//   ie.init("/path/to/model.aeromoe", ie_cfg);
//
//   GenerateConfig gen;
//   gen.max_new_tokens = 256;
//   gen.temperature    = 0.7f;
//   gen.on_token       = [](uint32_t tok) { printf("%u\n", tok); return true; };
//
//   std::vector<uint32_t> output;
//   ie.generate({1, 9906, 271}, gen, &output);
//
// Threading
// ─────────
//   InferenceEngine is NOT thread-safe.  External callers must serialise
//   concurrent generate() calls.  Multiple independent InferenceEngine
//   instances may run on separate threads (each owns its own Metal command
//   queue; they share the MTLDevice through EngineCore).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "aeromoe_types.h"
#include "engine_core.h"
#include "kv_cache.h"
#include "model_runner.h"
#include "sampler.h"

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <cstdint>

namespace aeromoe {

// ── InferenceConfig ───────────────────────────────────────────────────────────
// Aggregate configuration passed to InferenceEngine::init().
// Individual sub-configs are forwarded to their respective sub-systems.

struct InferenceConfig {
    EngineConfig      engine;   // memory budget, I/O threads, verbosity
    ModelRunnerConfig runner;   // max_seq_len, yarn_scale
};

// ── GenerateConfig ────────────────────────────────────────────────────────────
// Per-request generation parameters.  A new GenerateConfig may be supplied
// on each call to generate() / generate_stream().

struct GenerateConfig {
    // Maximum number of NEW tokens to generate (excluding the prompt).
    uint32_t max_new_tokens         = 256;

    // Sampler parameters
    float    temperature            = 0.0f;  // 0 = greedy
    float    top_p                  = 1.0f;  // nucleus; 1.0 = off
    uint32_t top_k                  = 0;     // 0 = off
    float    repetition_penalty     = 1.0f;  // 1.0 = off

    // Random seed for stochastic sampling (only used when temperature > 0).
    uint64_t rng_seed               = 42;

    // Stop-token IDs.  Generation ends immediately when any of these tokens
    // is sampled (the token is NOT appended to the output).
    // Default: {151645} which is Qwen3's <|im_end|> token.
    std::vector<uint32_t> stop_token_ids = {151645};

    // Streaming callback.  Called once per generated token immediately after
    // sampling.  Return `true` to continue, `false` to abort generation.
    // Set to nullptr to disable streaming (collect all tokens in output vec).
    std::function<bool(uint32_t token_id)> on_token;
};

// ── GenerateResult ────────────────────────────────────────────────────────────

struct GenerateResult {
    std::vector<uint32_t> tokens;       // generated token IDs (excluding prompt)
    uint32_t              prompt_len  = 0;
    uint32_t              output_len  = 0;
    double                prefill_ms  = 0.0;
    double                decode_ms   = 0.0;
    bool                  stopped_by_eos  = false;
    bool                  stopped_by_user = false;  // on_token returned false
    bool                  stopped_by_len  = false;
};

// ── InferenceEngine ───────────────────────────────────────────────────────────

class InferenceEngine {
public:
    InferenceEngine() = default;
    ~InferenceEngine();

    // Non-copyable
    InferenceEngine(const InferenceEngine&)            = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    // ── lifecycle ────────────────────────────────────────────────────────────

    // Open the model file at `model_path`, initialise the Metal device,
    // load all dense weights, and warm up the KV cache and ModelRunner.
    //
    // Returns Status::OK on success.
    // On failure returns an error code; the engine is left in an unusable
    // state and shutdown() should still be called to free partial resources.
    Status init(const std::string& model_path,
                const InferenceConfig& cfg = {});

    // Free all resources.  Called automatically by destructor.
    void shutdown();

    // ── generation ───────────────────────────────────────────────────────────

    // Blocking generation: run the full prefill + decode loop and collect all
    // generated tokens in `result`.
    //
    //   prompt_tokens : tokenised input sequence (BOS token should already be
    //                   prepended by the caller if the model requires it)
    //   gen_cfg       : per-request sampling and stopping parameters
    //   result        : [out] generated token IDs and timing statistics;
    //                   set to nullptr to discard output (useful for benchmarking)
    //
    // Returns Status::OK on success.
    Status generate(const std::vector<uint32_t>& prompt_tokens,
                    const GenerateConfig&         gen_cfg,
                    GenerateResult*               result = nullptr);

    // ── introspection ─────────────────────────────────────────────────────────

    // Access to the underlying subsystems (read-only after init).
    const EngineCore&      engine()  const { return *engine_; }
    const KVCacheManager&  kv_cache()const { return *kv_cache_; }
    const ModelConfig&     config()  const { return engine_->config(); }
    const RuntimeStats&    stats()   const { return engine_->stats(); }

    void print_stats(FILE* out = stderr) const { engine_->print_stats(out); }

private:
    // ── prefill ───────────────────────────────────────────────────────────────
    //
    // Run the model forward on every token in `prompt_tokens`, writing KV
    // entries for each position.  Returns the f32 logits from the LAST token,
    // and the wall-clock time in milliseconds.
    //
    // NOTE: prefill dispatches one forward() call per token (single-token mode).
    //       A future optimisation is batched prefill via GeMM; that requires a
    //       separate gemm_nt kernel and is deferred to a later session.
    Status run_prefill(uint32_t                      seq_id,
                       const std::vector<uint32_t>&  tokens,
                       float**                        last_logits_out,
                       double*                        elapsed_ms_out);

    // ── decode ────────────────────────────────────────────────────────────────
    //
    // Auto-regressive decode loop starting from position `start_pos`.
    // `first_logits` are the logits from the last prefill token; the first
    // sampled token is drawn from them.
    Status run_decode(uint32_t              seq_id,
                      uint32_t              start_pos,
                      float*                first_logits,
                      const GenerateConfig& gen_cfg,
                      GenerateResult*       result);

    // ── helpers ───────────────────────────────────────────────────────────────

    // Return wall-clock time in milliseconds.
    static double wall_ms();

    // Check whether `token_id` is in `stop_token_ids`.
    static bool is_stop_token(uint32_t token_id,
                               const std::vector<uint32_t>& stops);

    // ── state ─────────────────────────────────────────────────────────────────

    std::unique_ptr<EngineCore>      engine_;
    // kv_ledger_ must outlive kv_cache_ — destroyed after it in shutdown().
    std::unique_ptr<MemoryLedger>    kv_ledger_;
    std::unique_ptr<KVCacheManager>  kv_cache_;
    std::unique_ptr<ModelRunner>     runner_;
    std::unique_ptr<Sampler>         sampler_;

    InferenceConfig cfg_;
};

} // namespace aeromoe

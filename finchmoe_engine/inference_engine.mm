// inference_engine.mm
// ─────────────────────────────────────────────────────────────────────────────
// InferenceEngine — top-level generate() loop for FinchMoE.
// Implements inference_engine.h.  ObjC++ (.mm) required for Metal timing.
// ─────────────────────────────────────────────────────────────────────────────

#import "inference_engine.h"

#import <Foundation/Foundation.h>  // NSDate / mach_absolute_time

#include <cstdio>
#include <chrono>
#include <algorithm>
#include <cassert>

namespace finchmoe {

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle
// ─────────────────────────────────────────────────────────────────────────────

InferenceEngine::~InferenceEngine() {
    shutdown();
}

Status InferenceEngine::init(const std::string&     model_path,
                              const InferenceConfig& cfg)
{
    cfg_ = cfg;

    // ── 1. EngineCore: open file, Metal device, dense weights ────────────────
    engine_ = std::make_unique<EngineCore>();
    {
        Status s = engine_->init(model_path, cfg.engine);
        if (!ok(s)) {
            fprintf(stderr, "[InferenceEngine] EngineCore::init failed: %s\n",
                    status_str(s));
            return s;
        }
    }

    const ModelConfig& mc = engine_->config();

    // ── 2. KVCacheManager ────────────────────────────────────────────────────
    // Determine how many physical KV blocks fit in the remaining memory budget
    // after dense weights have been allocated (tracked by MemoryLedger inside
    // EngineCore, but exposed via RuntimeStats.bytes_loaded for now).
    // Use a conservative fraction of the remaining budget.
    //
    // Rough estimate: dense weights for Qwen3.6 ≈ 2.5 GB in bf16.
    // With a 4 GB budget, ~1.5 GB remains for KV cache and activations.
    // We allocate 1 GB for KV cache; the rest is activation buffers.

    constexpr size_t KV_BUDGET = 1ULL * 1024 * 1024 * 1024;  // 1 GiB

    KVCacheConfig kv_cfg;
    kv_cfg.num_layers  = mc.num_hidden_layers;
    kv_cfg.n_kv_heads  = mc.num_key_value_heads;
    kv_cfg.head_dim    = mc.head_dim();
    kv_cfg.dtype       = DType::BF16;
    kv_cfg.num_blocks  = KVCacheConfig::blocks_for_budget(
        KV_BUDGET,
        kv_cfg.num_layers,
        kv_cfg.n_kv_heads,
        kv_cfg.head_dim,
        kv_cfg.dtype
    );

    if (kv_cfg.num_blocks == 0) {
        fprintf(stderr, "[InferenceEngine] KV budget too small for even one block\n");
        return Status::BudgetExceeded;
    }

    if (cfg.engine.verbose) {
        fprintf(stderr, "[InferenceEngine] KV cache: %u blocks × %.1f KB = %.1f MB\n",
                kv_cfg.num_blocks,
                (double)kv_cfg.block_bytes_kv() * 2 / 1024.0,
                (double)kv_cfg.total_bytes() / (1024.0 * 1024.0));
    }

    // KVCacheManager needs a MemoryLedger that outlives it.  We allocate a
    // dedicated ledger for KV cache accounting (separate from the expert-slab
    // ledger owned by EngineCore).  kv_ledger_ is destroyed in shutdown()
    // after kv_cache_ so the reference remains valid for the ledger's lifetime.
    //
    // Soft budget = 75% of hard budget (start signalling pressure at 0.75 × KV_BUDGET).
    kv_ledger_ = std::make_unique<MemoryLedger>(
        KV_BUDGET,
        (KV_BUDGET * 3) / 4
    );

    kv_cache_ = std::make_unique<KVCacheManager>();
    {
        Status s = kv_cache_->init(engine_->device(), *kv_ledger_, kv_cfg);
        if (!ok(s)) {
            fprintf(stderr, "[InferenceEngine] KVCacheManager::init failed: %s\n",
                    status_str(s));
            return s;
        }
    }

    // ── 3. ModelRunner ───────────────────────────────────────────────────────
    runner_ = std::make_unique<ModelRunner>();
    {
        Status s = runner_->init(*engine_, *kv_cache_, cfg.runner);
        if (!ok(s)) {
            fprintf(stderr, "[InferenceEngine] ModelRunner::init failed: %s\n",
                    status_str(s));
            return s;
        }
    }

    // ── 4. Sampler ───────────────────────────────────────────────────────────
    // Constructed with defaults; reconfigured per-request in generate().
    sampler_ = std::make_unique<Sampler>();

    if (cfg.engine.verbose) {
        fprintf(stderr,
            "[InferenceEngine] Ready.  Model: %u layers, "
            "hidden=%u, experts=%u (top-%u), kv_blocks=%u\n",
            mc.num_hidden_layers, mc.hidden_size,
            mc.num_experts, mc.num_experts_per_tok,
            kv_cfg.num_blocks);
    }

    return Status::OK;
}

void InferenceEngine::shutdown() {
    // Destroy in reverse initialisation order to avoid dangling pointers.
    // kv_ledger_ is destroyed AFTER kv_cache_ since kv_cache_ may call
    // kv_ledger_->release() in its destructor.
    sampler_.reset();
    runner_.reset();
    kv_cache_.reset();
    kv_ledger_.reset();
    engine_.reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// generate
// ─────────────────────────────────────────────────────────────────────────────

Status InferenceEngine::generate(const std::vector<uint32_t>& prompt_tokens,
                                  const GenerateConfig&         gen_cfg,
                                  GenerateResult*               result)
{
    assert(engine_  && "InferenceEngine not initialised");
    assert(runner_  && "ModelRunner not initialised");
    assert(sampler_ && "Sampler not initialised");

    if (prompt_tokens.empty()) return Status::InvalidArg;

    // ── configure sampler ─────────────────────────────────────────────────────
    SamplerConfig sc;
    sc.temperature        = gen_cfg.temperature;
    sc.top_p              = gen_cfg.top_p;
    sc.top_k              = gen_cfg.top_k;
    sc.repetition_penalty = gen_cfg.repetition_penalty;
    sc.rng_seed           = gen_cfg.rng_seed;
    sampler_->reconfigure(sc);

    // ── allocate a KV sequence slot ───────────────────────────────────────────
    uint32_t seq_id = kv_cache_->new_sequence();
    if (seq_id == UINT32_MAX) {
        fprintf(stderr, "[InferenceEngine] No free sequence slot\n");
        return Status::BudgetExceeded;
    }

    if (result) {
        result->tokens.clear();
        result->prompt_len       = (uint32_t)prompt_tokens.size();
        result->output_len       = 0;
        result->stopped_by_eos   = false;
        result->stopped_by_user  = false;
        result->stopped_by_len   = false;
    }

    // ── prefill ───────────────────────────────────────────────────────────────
    float*  last_logits  = nullptr;
    double  prefill_ms   = 0.0;

    Status s = run_prefill(seq_id, prompt_tokens, &last_logits, &prefill_ms);
    if (!ok(s)) {
        kv_cache_->free_sequence(seq_id);
        return s;
    }

    if (result) result->prefill_ms = prefill_ms;

    // ── decode ────────────────────────────────────────────────────────────────
    uint32_t decode_start_pos = (uint32_t)prompt_tokens.size();

    s = run_decode(seq_id, decode_start_pos, last_logits, gen_cfg, result);

    // ── cleanup ───────────────────────────────────────────────────────────────
    kv_cache_->free_sequence(seq_id);

    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_prefill
// ─────────────────────────────────────────────────────────────────────────────
Status InferenceEngine::run_prefill(uint32_t                     seq_id,
                                     const std::vector<uint32_t>& tokens,
                                     float**                       last_logits_out,
                                     double*                       elapsed_ms_out)
{
    double t0 = wall_ms();

    float* logits = nullptr;

    for (uint32_t i = 0; i < (uint32_t)tokens.size(); ++i) {
        Status s = runner_->forward(tokens[i], seq_id, /*pos=*/i, &logits);
        if (!ok(s)) return s;
        // Logits from all but the last prefill token are discarded.
        // The KV cache entries are still written so the context is complete.
    }

    if (elapsed_ms_out) *elapsed_ms_out = wall_ms() - t0;
    if (last_logits_out) *last_logits_out = logits;

    return Status::OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_decode
// ─────────────────────────────────────────────────────────────────────────────
Status InferenceEngine::run_decode(uint32_t              seq_id,
                                    uint32_t              start_pos,
                                    float*                first_logits,
                                    const GenerateConfig& gen_cfg,
                                    GenerateResult*       result)
{
    const ModelConfig& mc = engine_->config();
    uint32_t vocab = mc.vocab_size;

    // Context buffer for repetition penalty: holds all tokens seen so far.
    // Starts with the prompt (we don't have the prompt tokens here, but the
    // penalty is applied only to the generated suffix, which is correct for
    // the standard repetition penalty definition).
    //
    // NOTE: For a fully correct repetition penalty over the full prompt, the
    //       caller should pre-populate this via a dedicated API.  Here we track
    //       only the generated tokens for simplicity.
    std::vector<uint32_t> context_tokens;
    context_tokens.reserve(gen_cfg.max_new_tokens + 4);

    double t0      = wall_ms();
    float* logits  = first_logits;
    uint32_t pos   = start_pos;

    for (uint32_t step = 0; step < gen_cfg.max_new_tokens; ++step) {
        // ── apply repetition penalty ─────────────────────────────────────────
        if (gen_cfg.repetition_penalty != 1.0f && !context_tokens.empty()) {
            sampler_->apply_repetition_penalty(
                logits, vocab,
                context_tokens.data(),
                (uint32_t)context_tokens.size()
            );
        }

        // ── sample next token ────────────────────────────────────────────────
        uint32_t token = sampler_->sample(logits, vocab);

        // ── check stop condition ──────────────────────────────────────────────
        if (is_stop_token(token, gen_cfg.stop_token_ids)) {
            if (result) result->stopped_by_eos = true;
            break;
        }

        // ── emit token ────────────────────────────────────────────────────────
        if (result) {
            result->tokens.push_back(token);
            result->output_len++;
        }

        context_tokens.push_back(token);

        // ── streaming callback ────────────────────────────────────────────────
        if (gen_cfg.on_token) {
            bool cont = gen_cfg.on_token(token);
            if (!cont) {
                if (result) result->stopped_by_user = true;
                break;
            }
        }

        // ── forward pass for next token ───────────────────────────────────────
        Status s = runner_->forward(token, seq_id, pos, &logits);
        if (!ok(s)) return s;
        ++pos;
    }

    // If the loop exhausted max_new_tokens without an EOS or user stop:
    if (result && !result->stopped_by_eos && !result->stopped_by_user) {
        result->stopped_by_len = true;
    }

    if (result) result->decode_ms = wall_ms() - t0;

    return Status::OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper utilities
// ─────────────────────────────────────────────────────────────────────────────

double InferenceEngine::wall_ms() {
    using namespace std::chrono;
    auto now = steady_clock::now().time_since_epoch();
    return duration_cast<microseconds>(now).count() / 1000.0;
}

bool InferenceEngine::is_stop_token(uint32_t token_id,
                                     const std::vector<uint32_t>& stops)
{
    for (uint32_t s : stops) {
        if (s == token_id) return true;
    }
    return false;
}

} // namespace finchmoe

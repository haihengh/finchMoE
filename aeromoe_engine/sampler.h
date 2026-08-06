// sampler.h
// ─────────────────────────────────────────────────────────────────────────────
// Next-token sampling from a float32 logits buffer.
//
// Supported modes (all pure-CPU, no Metal dependency):
//   • Greedy    — argmax; temperature == 0 selects this path automatically.
//   • Temperature — scale logits by 1/T before softmax.
//   • Top-k     — keep only the k highest-probability tokens.
//   • Top-p     — nucleus: keep tokens summing to at least p of probability.
//   • Repetition penalty — discount tokens that appear in the context.
//
// All modes compose: if temperature > 0, the order of operations is
//   repetition_penalty → temperature scale → top-k → top-p → categorical.
//
// Thread-safety: NOT thread-safe. Each calling thread should own its own
// Sampler instance with its own RNG state.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <utility>  // std::pair

namespace aeromoe {

// ── SamplerConfig ─────────────────────────────────────────────────────────────

struct SamplerConfig {
    float    temperature        = 0.0f; // 0 = greedy; > 0 stochastic
    float    top_p              = 1.0f; // nucleus probability mass (1 = off)
    uint32_t top_k              = 0;    // 0 = disabled
    float    repetition_penalty = 1.0f; // 1 = off; > 1 penalises repeated tokens
    uint64_t rng_seed           = 42;
};

// ── Sampler ───────────────────────────────────────────────────────────────────

class Sampler {
public:
    explicit Sampler(const SamplerConfig& cfg = {});

    // Replace the sampling configuration and reseed the RNG.
    void reconfigure(const SamplerConfig& cfg);

    // Reseed the RNG without changing the rest of the config.
    void set_seed(uint64_t seed);

    const SamplerConfig& config() const { return cfg_; }

    // ── sampling ─────────────────────────────────────────────────────────────

    // Sample the next token from a raw logits buffer.
    //
    // logits:      [vocab_size] f32.  The buffer is modified in-place by
    //              temperature scaling and optional repetition penalty.
    //              Callers should not reuse it after this call.
    // vocab_size:  Number of elements in logits.
    //
    // Returns the sampled token id in [0, vocab_size).
    uint32_t sample(float* logits, uint32_t vocab_size);

    // Apply repetition penalty to logits in-place, discounting tokens that
    // already appear in [context, context+context_len).
    // Must be called BEFORE sample() if you want the penalty applied.
    void apply_repetition_penalty(
        float*          logits,
        uint32_t        vocab_size,
        const uint32_t* context,
        uint32_t        context_len
    );

    // Greedy argmax — fast CPU scan, no RNG, no allocation.
    // Useful for caller code that does not want to invoke the full sampler.
    static uint32_t argmax(const float* logits, uint32_t n);

private:
    // ── internal sampling helpers ─────────────────────────────────────────────

    // In-place temperature scaling: logits[i] *= (1/temp)
    static void apply_temperature(float* logits, uint32_t n, float temp);

    // Compute softmax-normalised (prob, index) pairs.
    // Clears and fills probs; returns the pair vector sorted descending by prob.
    static void softmax_pairs(const float* logits, uint32_t n,
                              std::vector<std::pair<float, uint32_t>>& out);

    // Truncate sorted probs to the top-k entries and renormalise.
    static void filter_top_k(std::vector<std::pair<float, uint32_t>>& probs,
                              uint32_t k);

    // Truncate sorted probs to a nucleus of cumulative mass >= p, renormalise.
    static void filter_top_p(std::vector<std::pair<float, uint32_t>>& probs,
                              float p);

    // Categorical sample: draw from the distribution defined by probs.
    uint32_t categorical(const std::vector<std::pair<float, uint32_t>>& probs);

    // ── state ─────────────────────────────────────────────────────────────────

    SamplerConfig cfg_;
    std::mt19937  rng_;

    // Reusable heap allocation — avoids per-call vector realloc in the decode loop.
    std::vector<std::pair<float, uint32_t>> work_probs_;
};

} // namespace aeromoe

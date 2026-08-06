// sampler.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Sampler implementation — pure CPU, no Metal dependency.
// ─────────────────────────────────────────────────────────────────────────────

#include "sampler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <limits>

namespace aeromoe {

// ── construction ──────────────────────────────────────────────────────────────

Sampler::Sampler(const SamplerConfig& cfg)
    : cfg_(cfg), rng_(static_cast<uint_fast32_t>(cfg.rng_seed))
{}

void Sampler::reconfigure(const SamplerConfig& cfg) {
    cfg_ = cfg;
    rng_.seed(static_cast<uint_fast32_t>(cfg.rng_seed));
}

void Sampler::set_seed(uint64_t seed) {
    cfg_.rng_seed = seed;
    rng_.seed(static_cast<uint_fast32_t>(seed));
}

// ── public API ────────────────────────────────────────────────────────────────

uint32_t Sampler::sample(float* logits, uint32_t vocab_size) {
    assert(logits && vocab_size > 0);

    // Fast path: greedy
    if (cfg_.temperature <= 0.0f) {
        return argmax(logits, vocab_size);
    }

    // Temperature scale in-place
    apply_temperature(logits, vocab_size, cfg_.temperature);

    // Build (prob, idx) pairs sorted descending
    softmax_pairs(logits, vocab_size, work_probs_);

    // Optional top-k
    if (cfg_.top_k > 0 && cfg_.top_k < vocab_size) {
        filter_top_k(work_probs_, cfg_.top_k);
    }

    // Optional top-p (nucleus)
    if (cfg_.top_p < 1.0f - 1e-6f) {
        filter_top_p(work_probs_, cfg_.top_p);
    }

    return categorical(work_probs_);
}

void Sampler::apply_repetition_penalty(
    float*          logits,
    uint32_t        vocab_size,
    const uint32_t* context,
    uint32_t        context_len)
{
    if (cfg_.repetition_penalty == 1.0f || context_len == 0) return;

    const float penalty = cfg_.repetition_penalty;
    for (uint32_t i = 0; i < context_len; ++i) {
        uint32_t tok = context[i];
        if (tok >= vocab_size) continue;
        // Standard HuggingFace repetition penalty:
        // positive logits are divided; negative logits are multiplied.
        if (logits[tok] >= 0.0f)
            logits[tok] /= penalty;
        else
            logits[tok] *= penalty;
    }
}

// ── static helpers ────────────────────────────────────────────────────────────

/*static*/ uint32_t Sampler::argmax(const float* logits, uint32_t n) {
    uint32_t best = 0;
    float    best_val = logits[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

/*static*/ void Sampler::apply_temperature(float* logits, uint32_t n, float temp) {
    const float inv_temp = 1.0f / temp;
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] *= inv_temp;
    }
}

/*static*/ void Sampler::softmax_pairs(
    const float* logits, uint32_t n,
    std::vector<std::pair<float, uint32_t>>& out)
{
    // Numerically stable softmax: subtract max before exp.
    float max_val = *std::max_element(logits, logits + n);

    out.clear();
    out.reserve(n);

    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float e = std::exp(logits[i] - max_val);
        out.push_back({e, i});
        sum += e;
    }

    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (auto& p : out) p.first *= inv_sum;
    }

    // Sort descending by probability
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });
}

/*static*/ void Sampler::filter_top_k(
    std::vector<std::pair<float, uint32_t>>& probs,
    uint32_t k)
{
    // probs is already sorted descending; just truncate and renormalise.
    if (probs.size() > static_cast<size_t>(k)) {
        probs.resize(k);
    }
    // Renormalise
    float sum = 0.0f;
    for (const auto& p : probs) sum += p.first;
    if (sum > 0.0f) {
        const float inv_sum = 1.0f / sum;
        for (auto& p : probs) p.first *= inv_sum;
    }
}

/*static*/ void Sampler::filter_top_p(
    std::vector<std::pair<float, uint32_t>>& probs,
    float p)
{
    // Nucleus: keep the minimal prefix whose cumulative prob >= p.
    // probs is sorted descending and normalised.
    float cumsum = 0.0f;
    size_t cutoff = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i].first;
        if (cumsum >= p - 1e-7f) {
            cutoff = i + 1;
            break;
        }
    }
    if (cutoff < probs.size()) {
        probs.resize(cutoff);
        // Renormalise
        float sum = 0.0f;
        for (const auto& pr : probs) sum += pr.first;
        if (sum > 0.0f) {
            const float inv_sum = 1.0f / sum;
            for (auto& pr : probs) pr.first *= inv_sum;
        }
    }
}

uint32_t Sampler::categorical(
    const std::vector<std::pair<float, uint32_t>>& probs)
{
    if (probs.empty()) return 0;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng_);

    float cumsum = 0.0f;
    for (const auto& [prob, idx] : probs) {
        cumsum += prob;
        if (r <= cumsum) return idx;
    }
    // Floating-point rounding: return last element if r rounded past 1.0
    return probs.back().second;
}

} // namespace aeromoe

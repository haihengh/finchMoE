// finchmoe_types.h
// ─────────────────────────────────────────────────────────────────────────────
// Shared primitive types, model configuration, and tensor descriptors.
// All structs are plain-old-data so they can be passed safely across the
// Metal ↔ CPU boundary.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <array>

namespace finchmoe {

// ── dtype ─────────────────────────────────────────────────────────────────────

enum class DType : uint32_t {
    BF16   = 0,
    FP16   = 1,
    FP32   = 2,
    FP8    = 3,   // e4m3
};

inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::BF16: return 2;
        case DType::FP16: return 2;
        case DType::FP32: return 4;
        case DType::FP8:  return 1;
    }
    return 2;
}

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::BF16: return "bfloat16";
        case DType::FP16: return "float16";
        case DType::FP32: return "float32";
        case DType::FP8:  return "float8_e4m3";
    }
    return "unknown";
}

// ── model configuration ───────────────────────────────────────────────────────
// Mirrors the header fields in the .finchmoe file.
// Loaded once at startup; immutable thereafter.

struct ModelConfig {
    // Architecture
    uint32_t hidden_size              = 4096;
    uint32_t intermediate_size        = 2048;   // shared/dense expert
    uint32_t moe_intermediate_size    = 768;    // per routed expert
    uint32_t num_hidden_layers        = 94;
    uint32_t num_attention_heads      = 64;
    uint32_t num_key_value_heads      = 4;
    uint32_t vocab_size               = 151936;
    uint32_t max_position_embeddings  = 131072;

    // MoE
    uint32_t num_experts              = 128;    // routed per layer
    uint32_t num_experts_per_tok      = 8;      // top-k active
    uint32_t num_shared_experts       = 1;
    bool     norm_topk_prob           = true;   // renormalize router probs

    // Embedding
    bool     tie_word_embeddings      = false;

    // Norms & positional
    float    rms_norm_eps             = 1e-6f;
    float    rope_theta               = 1'000'000.0f;
    char     rope_scaling_type[8]     = "yarn\0\0\0\0";

    // Storage
    DType    storage_dtype            = DType::BF16;

    // Derived helpers
    uint32_t head_dim() const {
        return hidden_size / num_attention_heads;
    }
    uint32_t kv_dim() const {
        return num_key_value_heads * head_dim();
    }
    size_t expert_slab_bytes() const {
        // gate_proj + up_proj + down_proj, all in storage_dtype
        size_t elem = (size_t)moe_intermediate_size * hidden_size;
        return (2 * elem + (size_t)hidden_size * moe_intermediate_size)
               * dtype_size(storage_dtype);
    }
    // Rough active-param byte budget for one forward pass
    size_t active_bytes_per_token() const {
        size_t attn = (size_t)hidden_size * (
            num_attention_heads * head_dim() +
            num_key_value_heads * head_dim() * 2 +
            hidden_size
        );
        size_t experts = (size_t)num_experts_per_tok
                         * 3 * moe_intermediate_size * hidden_size;
        size_t shared  = (size_t)num_shared_experts
                         * 3 * intermediate_size * hidden_size;
        return (attn + experts + shared)
               * dtype_size(storage_dtype)
               * num_hidden_layers;
    }
};

// ── tensor descriptor ─────────────────────────────────────────────────────────
// Lightweight view into memory — does not own the buffer.

struct TensorView {
    const void*              data    = nullptr;
    std::array<int32_t, 4>   shape   = {0,0,0,0};
    uint32_t                 ndim    = 0;
    DType                    dtype   = DType::BF16;

    size_t numel() const {
        size_t n = 1;
        for (uint32_t i = 0; i < ndim; ++i) n *= (size_t)shape[i];
        return n;
    }
    size_t nbytes() const { return numel() * dtype_size(dtype); }

    int32_t dim(uint32_t i) const { return i < ndim ? shape[i] : 1; }
};

// ── expert key ────────────────────────────────────────────────────────────────

struct ExpertKey {
    uint32_t layer;
    uint32_t expert;

    bool operator==(const ExpertKey& o) const {
        return layer == o.layer && expert == o.expert;
    }
    bool operator<(const ExpertKey& o) const {
        return layer != o.layer ? layer < o.layer : expert < o.expert;
    }
};

// ── memory budget constants ───────────────────────────────────────────────────

constexpr size_t BUDGET_4GB   = 4ULL * 1024 * 1024 * 1024;
constexpr size_t BUDGET_3_5GB = 3ULL * 1024 * 1024 * 1024
                               + 512ULL * 1024 * 1024;

// ── error handling ────────────────────────────────────────────────────────────

enum class Status : int32_t {
    OK              =  0,
    BudgetExceeded  = -1,
    IOError         = -2,
    FormatError     = -3,
    NotFound        = -4,
    MetalError      = -5,
    InvalidArg      = -6,
};

inline bool ok(Status s) { return s == Status::OK; }

inline const char* status_str(Status s) {
    switch (s) {
        case Status::OK:             return "OK";
        case Status::BudgetExceeded: return "BudgetExceeded";
        case Status::IOError:        return "IOError";
        case Status::FormatError:    return "FormatError";
        case Status::NotFound:       return "NotFound";
        case Status::MetalError:     return "MetalError";
        case Status::InvalidArg:     return "InvalidArg";
    }
    return "Unknown";
}

// ── runtime statistics (lock-free via atomics in engine) ──────────────────────

struct RuntimeStats {
    uint64_t cache_hits        = 0;
    uint64_t cache_misses      = 0;
    uint64_t bytes_loaded      = 0;
    uint64_t bytes_evicted     = 0;
    uint64_t tokens_generated  = 0;
    double   mean_load_us      = 0.0; // moving average I/O latency µs
};

} // namespace finchmoe

// model_runner.h
// ─────────────────────────────────────────────────────────────────────────────
// Forward-pass executor for FinchMoE.
//
// ModelRunner owns all activation (scratch) Metal buffers and drives the
// per-token forward pass through all N decoder layers.
//
// Design highlights
// ─────────────────
// • Single-token interface: forward(token_id, seq_id, pos) processes exactly
//   one token.  Prefill loops over tokens in InferenceEngine.
//
// • GPU sync budget: 2 commits per decoder layer.
//   Commit-1  (run_attention):
//       input_norm → Q/K/V GEMVs → QK-norm → RoPE → KV append
//       → GQA decode → O-proj GEMV → residual add → post_norm (pre-MoE norm)
//   Commit-2  (run_moe_layer):
//       zero_accum → [gate_up+down × top_k routed experts]
//       → [gate_up+down × 1 shared expert] → moe_reduce+residual
//
// • CPU gate routing: after Commit-1, `moe_normed_buf_` is coherent (it is a
//   MTLStorageModeShared buffer).  The CPU reads it and dot-products against
//   the bf16 gate matrix (~128 × 4096 MACs ≈ < 1 ms) to select top-k experts
//   before Commit-2 starts.  No additional GPU sync is needed for routing.
//
// • Expert sub-buffers: each expert weight matrix is a sub-range of a larger
//   slab MTLBuffer.  ModelRunner creates zero-copy MTLBuffer views via
//   newBufferWithBytesNoCopy: (safe because Qwen3.6 expert sub-range sizes are
//   multiples of the VM page size) and pools them until after waitUntilCompleted.
//
// • Hidden-state swap: after run_moe_layer the reduced output lives in
//   tok_out_buf_.  We swap hidden_buf_ ↔ tok_out_buf_ pointers so the next
//   layer's attention reads the fresh hidden state at zero copy cost.
//
// Threading: NOT thread-safe.  Own one instance per concurrent sequence
//            (the current InferenceEngine is single-sequence).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"
#include "engine_core.h"
#include "kv_cache.h"
#include "kernels/kernel_dispatch.h"

#include <memory>
#include <vector>
#include <cstdint>
#include <string>

// Forward-declare MTLBuffer for non-ObjC translation units.
#ifdef __OBJC__
@protocol MTLBuffer;
using MTLBufferPtr_mr = id<MTLBuffer>;
#else
using MTLBufferPtr_mr = void*;
#endif

namespace finchmoe {

// ── ModelRunnerConfig ─────────────────────────────────────────────────────────

struct ModelRunnerConfig {
    // Maximum sequence length to pre-allocate RoPE tables for.
    uint32_t max_seq_len  = 4096;

    // YaRN position-interpolation scale (1.0 = standard RoPE, no scaling).
    // Qwen3 uses YaRN with scale factor derived from rope_scaling config.
    float yarn_scale      = 1.0f;
};

// ── LayerMtlBuffers ───────────────────────────────────────────────────────────
// Per-layer cache of MTLBuffer* (as opaque void*) for dense (backbone) weights.
// Built once during init() to avoid string-hash lookups on the hot path.
// All pointers are NON-OWNING; EngineCore retains the underlying buffers.

struct LayerMtlBuffers {
    // Attention projections
    void* q_proj      = nullptr;  // [n_heads * head_dim, hidden] bf16
    void* k_proj      = nullptr;  // [n_kv_heads * head_dim, hidden] bf16
    void* v_proj      = nullptr;  // [n_kv_heads * head_dim, hidden] bf16
    void* o_proj      = nullptr;  // [hidden, n_heads * head_dim] bf16

    // Per-head QK-norms
    void* q_norm      = nullptr;  // [head_dim] bf16
    void* k_norm      = nullptr;  // [head_dim] bf16

    // MoE router gate (used on CPU, kept here for completeness / future GPU routing)
    void* gate_mtl    = nullptr;  // [num_experts, hidden] bf16 — MTLBuffer*

    // Shared expert weights (always active)
    void* sh_gate     = nullptr;  // [inter, hidden] bf16
    void* sh_up       = nullptr;  // [inter, hidden] bf16
    void* sh_down     = nullptr;  // [hidden, inter] bf16

    // Layer norms
    void* input_norm  = nullptr;  // [hidden] bf16
    void* post_norm   = nullptr;  // [hidden] bf16
};

// ── ModelRunner ───────────────────────────────────────────────────────────────

class ModelRunner {
public:
    ModelRunner() = default;
    ~ModelRunner();

    // Non-copyable / non-movable (owns Metal resources).
    ModelRunner(const ModelRunner&)            = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;

    // ── lifecycle ────────────────────────────────────────────────────────────

    // Allocate all activation buffers, build per-layer buffer cache, pre-build
    // RoPE tables, and precompile Metal PSOs.
    // Must be called exactly once before forward().
    Status init(EngineCore&          engine,
                KVCacheManager&      kv_cache,
                const ModelRunnerConfig& cfg = {});

    // Process one token and return a pointer to the [vocab_size] f32 logits.
    //
    //   token_id : vocabulary index of the current token
    //   seq_id   : sequence slot in KVCacheManager (from new_sequence())
    //   pos      : absolute position in the sequence (0-based)
    //   logits   : [out] pointer to the logits buffer inside logits_buf_
    //              Valid until the next call to forward().
    //              The Sampler is permitted to modify it in-place (repetition
    //              penalty, temperature scaling) because forward() overwrites
    //              it unconditionally on re-entry.
    //
    // Returns Status::OK on success, Status::MetalError on GPU error.
    Status forward(uint32_t  token_id,
                   uint32_t  seq_id,
                   uint32_t  pos,
                   float**   logits_out);

    // Release all Metal resources.  Called automatically by destructor.
    void shutdown();

    // ── accessors ────────────────────────────────────────────────────────────

    const ModelRunnerConfig& config() const { return cfg_; }

private:
    // ── forward-pass phases ───────────────────────────────────────────────────

    // 1. Embedding lookup: write hidden_buf_ ← embed_tokens[token_id]
    void phase_embed(uint32_t token_id);

    // 2. Write pos_ids_buf_ ← [pos]  (single uint32 for single-token mode)
    void phase_build_pos_ids(uint32_t pos);

    // 3. Attention sublayer for decoder layer `layer`.
    //    Encodes the full attention pipeline INCLUDING the post-attention norm
    //    (pre-MoE norm) into one command buffer, then commits and waits.
    //    After return, moe_normed_buf_ is coherent on CPU.
    void run_attention(uint32_t layer, uint32_t seq_id, uint32_t pos);

    // 4. MoE sublayer for decoder layer `layer`.
    //    Reads moe_normed_buf_ on CPU for routing, acquires expert slabs,
    //    encodes the expert FFN + shared expert + reduce into one command
    //    buffer, commits and waits.  After return, tok_out_buf_ holds the
    //    new hidden state; the caller swaps it with hidden_buf_.
    void run_moe_layer(uint32_t layer);

    // 5. Final RMSNorm (in-place on hidden_buf_) + GEMV with lm_head.
    //    Writes logits_buf_.
    void phase_lm_head();

    // ── CPU routing helpers ───────────────────────────────────────────────────

    // Expand moe_normed_buf_ (bf16, MTLStorageModeShared) to f32 scratch.
    void expand_moe_normed_to_f32();

    // Compute gate logits[i] = dot(gate_row[i], moe_normed_f32_)
    // for i in [0, num_experts).  Results written to gate_logits_scratch_.
    void cpu_gate_logits(uint32_t layer);

    // Select top-k from gate_logits_scratch_ and optionally renormalise.
    // Populates expert_ids_out and weights_out.
    void cpu_top_k_route(std::vector<uint32_t>& expert_ids_out,
                         std::vector<float>&    weights_out);

    // ── Metal helpers ─────────────────────────────────────────────────────────

    // Create and return a new MTLCommandBuffer from the EngineCore command queue.
    // Returned as void* (id<MTLCommandBuffer>); retained by ORC.
    void* new_cmd();

    // Commit the command buffer, block until complete, then release it.
    void commit_and_wait(void* cmd_buf);

    // Allocate a new MTLBuffer of `bytes` in StorageModeShared.
    // Returns void* (id<MTLBuffer>) owned by self (stored in owned_bufs_).
    // Optionally sets a Metal debug label for GPU frame-capture.
    void* alloc_buf(size_t bytes, const char* label = nullptr);

    // ── Expert sub-buffer helpers ─────────────────────────────────────────────

    // Create a zero-copy MTLBuffer view that wraps [ptr, ptr+length).
    // `ptr` must be page-aligned (4096 B) and `length` must be a multiple
    // of the page size — both conditions hold for Qwen3.6 expert matrices.
    // The returned buffer is added to subbuf_pool_ and released after the
    // MoE command buffer completes.
    // Returns void* (id<MTLBuffer>).
    void* make_subbuf(void* ptr, size_t length);

    // Release all buffers in subbuf_pool_ and clear it.
    void drain_subbuf_pool();

    // ── state ─────────────────────────────────────────────────────────────────

    EngineCore*        engine_   = nullptr;
    KVCacheManager*    kv_cache_ = nullptr;
    ModelRunnerConfig  cfg_;

    // Kernel dispatch (wraps all Metal PSOs)
    std::unique_ptr<kernels::KernelDispatch> dispatch_;

    // Per-layer dense-weight MTLBuffer pointers (non-owning)
    std::vector<LayerMtlBuffers> layer_mtl_;  // size = num_hidden_layers

    // Global dense weight MTLBuffers (non-owning)
    void* embed_mtl_     = nullptr;  // id<MTLBuffer> [vocab, hidden] bf16
    void* final_norm_mtl_= nullptr;  // id<MTLBuffer> [hidden] bf16
    void* lm_head_mtl_   = nullptr;  // id<MTLBuffer> [vocab, hidden] bf16

    // ── Activation buffers (owned, MTLStorageModeShared) ─────────────────────
    //
    // hidden_buf_ / tok_out_buf_ are swapped after every MoE layer so the
    // next attention sees the fresh hidden state without a memcpy.
    //
    void* hidden_buf_       = nullptr;  // [hidden] bf16  — current layer input
    void* tok_out_buf_      = nullptr;  // [hidden] bf16  — MoE layer output (swapped)
    void* normed_buf_       = nullptr;  // [hidden] bf16  — post-input-norm
    void* moe_normed_buf_   = nullptr;  // [hidden] bf16  — post-post-norm (pre-MoE)
    void* q_buf_            = nullptr;  // [n_heads * head_dim] bf16
    void* k_buf_            = nullptr;  // [n_kv_heads * head_dim] bf16
    void* v_buf_            = nullptr;  // [n_kv_heads * head_dim] bf16
    void* attn_out_buf_     = nullptr;  // [n_heads * head_dim] bf16
    void* o_proj_out_buf_   = nullptr;  // [hidden] bf16  — O-proj output
    void* moe_accum_buf_    = nullptr;  // [hidden] f32   — weighted expert accumulator
    void* moe_tmp_buf_      = nullptr;  // [moe_inter] f32 — per-routed-expert temp
    void* shared_tmp_buf_   = nullptr;  // [inter] f32    — shared expert temp
    void* pos_ids_buf_      = nullptr;  // [1] uint32     — current position
    void* cos_table_buf_    = nullptr;  // [max_seq, head_dim/2] f32
    void* sin_table_buf_    = nullptr;  // [max_seq, head_dim/2] f32
    void* logits_buf_       = nullptr;  // [vocab_size] f32  — LM head output

    // All owned MTLBuffers are collected here for bulk release on shutdown.
    std::vector<void*> owned_bufs_;

    // Zero-copy expert sub-buffer pool (drained after each MoE command buffer)
    std::vector<void*> subbuf_pool_;

    // ── CPU scratch ───────────────────────────────────────────────────────────

    // moe_normed_buf_ expanded to f32 for CPU dot products (128 × 4096 MACs)
    std::vector<float> moe_normed_f32_;        // [hidden] f32

    // Gate logits computed on CPU before top-k selection
    std::vector<float> gate_logits_scratch_;   // [num_experts] f32

    // Reusable sorted index buffer for top-k selection (avoids per-layer alloc)
    std::vector<std::pair<float, uint32_t>> topk_sort_scratch_; // [num_experts]

    // ── Dimension cache (from ModelConfig, set in init) ────────────────────────

    uint32_t hidden_;          // hidden_size
    uint32_t head_dim_;        // hidden / n_heads
    uint32_t n_heads_;         // num_attention_heads
    uint32_t n_kv_heads_;      // num_key_value_heads
    uint32_t moe_inter_;       // moe_intermediate_size
    uint32_t inter_;           // intermediate_size  (shared expert)
    uint32_t num_experts_;     // num_experts
    uint32_t top_k_;           // num_experts_per_tok
    uint32_t num_layers_;      // num_hidden_layers
    uint32_t vocab_size_;      // vocab_size
    float    rms_eps_;         // rms_norm_eps
    bool     norm_topk_prob_;  // renormalise router probabilities
};

} // namespace finchmoe

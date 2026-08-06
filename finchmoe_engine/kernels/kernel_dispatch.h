// kernel_dispatch.h
// ─────────────────────────────────────────────────────────────────────────────
// C++ interface to all Metal kernels.
// Wraps pipeline-state caching, threadgroup sizing, and buffer binding so the
// inference loop (Session 4) can call clean C++ functions rather than
// Objective-C Metal API directly.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "../finchmoe_types.h"

// Metal types are opaque void* — only .mm files bridge to real ObjC types.
using MTLDevicePtr        = void*;
using MTLCommandBufferPtr = void*;
using MTLBufferPtr_t      = void*;

#include <cstdint>
#include <string>
#include <unordered_map>

namespace finchmoe {
namespace kernels {

// ── KernelDispatch ────────────────────────────────────────────────────────────
// Singleton per Metal device. Owns the compiled pipeline state objects (PSOs).
// All dispatch*() calls encode a compute command into the provided command buffer.

class KernelDispatch {
public:
    explicit KernelDispatch(MTLDevicePtr device);
    ~KernelDispatch();

    // Non-copyable
    KernelDispatch(const KernelDispatch&) = delete;
    KernelDispatch& operator=(const KernelDispatch&) = delete;

    // ── RMSNorm ───────────────────────────────────────────────────────────────

    // Standard pre/post attention RMSNorm: out = rms_norm(in) * weight
    void dispatch_rms_norm(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t input,           // [rows, hidden]  bf16
        MTLBufferPtr_t weight,          // [hidden]         bf16
        MTLBufferPtr_t output,          // [rows, hidden]  bf16
        uint32_t rows,
        uint32_t hidden,
        float    eps
    );

    // Per-head QK-Norm: in-place, [seq*n_heads, head_dim]
    void dispatch_rms_norm_head(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t tensor,          // [seq * n_heads, head_dim] bf16 (in-place)
        MTLBufferPtr_t weight,          // [head_dim] bf16
        uint32_t seq_times_heads,
        uint32_t head_dim,
        float    eps
    );

    // In-place final norm (model.norm before LM head)
    void dispatch_rms_norm_inplace(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t inout,           // [rows, hidden] bf16 (in-place)
        MTLBufferPtr_t weight,          // [hidden] bf16
        uint32_t rows,
        uint32_t hidden,
        float    eps
    );

    // ── RoPE ─────────────────────────────────────────────────────────────────

    // Build YaRN cos/sin tables (call once at startup)
    void dispatch_build_rope_tables(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t cos_out,         // [max_seq, head_dim/2] f32
        MTLBufferPtr_t sin_out,         // [max_seq, head_dim/2] f32
        uint32_t max_seq_len,
        uint32_t head_dim,
        float rope_theta,
        float yarn_scale
    );

    // Fused Q+K RoPE (single dispatch)
    void dispatch_rope_fused_qk(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t q,               // [seq, n_heads,    head_dim] bf16
        MTLBufferPtr_t k,               // [seq, n_kv_heads, head_dim] bf16
        MTLBufferPtr_t cos_table,       // [max_seq, head_dim/2] f32
        MTLBufferPtr_t sin_table,
        MTLBufferPtr_t pos_ids,         // [seq] uint32
        uint32_t seq_len,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim
    );

    // ── KV cache ──────────────────────────────────────────────────────────────

    void dispatch_kv_append(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t k_in,
        MTLBufferPtr_t v_in,
        MTLBufferPtr_t k_cache,
        MTLBufferPtr_t v_cache,
        MTLBufferPtr_t page_table,
        uint32_t seq_len,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t layer,
        uint32_t seq_start,
        uint32_t max_pages
    );

    // ── Attention ─────────────────────────────────────────────────────────────

    // Single-token decode attention
    void dispatch_gqa_decode(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t q,               // [1, n_heads, head_dim] bf16
        MTLBufferPtr_t k_cache,
        MTLBufferPtr_t v_cache,
        MTLBufferPtr_t page_table,
        MTLBufferPtr_t out,             // [1, n_heads, head_dim] bf16
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t layer,
        uint32_t kv_len,
        uint32_t max_pages
    );

    // Multi-token prefill attention
    void dispatch_gqa_prefill(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t q,               // [q_len, n_heads, head_dim] bf16
        MTLBufferPtr_t k_cache,
        MTLBufferPtr_t v_cache,
        MTLBufferPtr_t page_table,
        MTLBufferPtr_t out,
        uint32_t n_heads,
        uint32_t n_kv_heads,
        uint32_t head_dim,
        uint32_t layer,
        uint32_t kv_len,
        uint32_t q_len,
        uint32_t q_start,
        uint32_t max_pages
    );

    // ── MoE ───────────────────────────────────────────────────────────────────

    // Router: gate logits → top-k indices + renormalized weights
    void dispatch_moe_router(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t gate_logits,     // [seq, num_experts] f32
        MTLBufferPtr_t expert_ids,      // [seq, top_k]       uint32 OUT
        MTLBufferPtr_t expert_weights,  // [seq, top_k]       f32    OUT
        uint32_t seq_len,
        uint32_t num_experts,
        uint32_t top_k,
        bool norm_topk
    );

    // Expert FFN gate+up+SiLU: x → tmp  [moe_inter] f32
    void dispatch_moe_gate_up(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t x,               // [hidden] bf16
        MTLBufferPtr_t gate_w,          // [moe_inter, hidden] bf16
        MTLBufferPtr_t up_w,            // [moe_inter, hidden] bf16
        MTLBufferPtr_t tmp_out,         // [moe_inter] f32
        uint32_t hidden,
        uint32_t moe_inter
    );

    // Expert down projection: tmp → accum (weighted accumulate)
    void dispatch_moe_down(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t tmp,             // [moe_inter] f32
        MTLBufferPtr_t down_w,          // [hidden, moe_inter] bf16
        MTLBufferPtr_t accum,           // [hidden] f32
        uint32_t hidden,
        uint32_t moe_inter,
        float weight
    );

    // Zero accumulator buffer
    void dispatch_moe_zero_accum(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t accum,
        uint32_t hidden
    );

    // Add src accumulator into dst
    void dispatch_moe_add_accum(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t dst,
        MTLBufferPtr_t src,
        uint32_t n
    );

    // Convert f32 accum + residual → bf16 output
    void dispatch_moe_reduce(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t accum,           // [hidden] f32
        MTLBufferPtr_t residual,        // [hidden] bf16
        MTLBufferPtr_t out,             // [hidden] bf16
        uint32_t hidden
    );

    // ── GEMV / projections ────────────────────────────────────────────────────

    void dispatch_gemv_bf16(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t W,               // [out_dim, in_dim] bf16
        MTLBufferPtr_t x,               // [in_dim] bf16
        MTLBufferPtr_t y,               // [out_dim] bf16
        uint32_t out_dim,
        uint32_t in_dim
    );

    // LM head: output is f32 logits
    void dispatch_gemv_lm_head(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t W,               // [vocab, hidden] bf16
        MTLBufferPtr_t x,               // [hidden] bf16
        MTLBufferPtr_t logits,          // [vocab] f32
        uint32_t vocab_size,
        uint32_t hidden
    );

    // Batched GEMV for prefill
    void dispatch_gemv_batched(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t W,
        MTLBufferPtr_t X,               // [in_dim, batch] bf16
        MTLBufferPtr_t Y,               // [out_dim, batch] bf16
        uint32_t out_dim,
        uint32_t in_dim,
        uint32_t batch
    );

    // Elementwise residual add: y += x  (bf16)
    void dispatch_add_residual(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t y,               // in-place
        MTLBufferPtr_t x,
        uint32_t n
    );

    // Greedy argmax over f32 logits
    void dispatch_argmax(
        MTLCommandBufferPtr cmd,
        MTLBufferPtr_t logits,          // [vocab] f32
        MTLBufferPtr_t result,          // [1] uint32
        uint32_t vocab_size
    );

    // ── PSO cache ─────────────────────────────────────────────────────────────

    // Force compile all PSOs upfront (optional; called by EngineCore::init).
    Status precompile_all();

private:
    // Resolve or compile a PSO by kernel function name.
    // Defined in kernel_dispatch.mm
    void* get_pso(const std::string& name);

    MTLDevicePtr device_;

    // name → MTLComputePipelineState* (opaque void* in this header)
    std::unordered_map<std::string, void*> pso_cache_;
};

} // namespace kernels
} // namespace finchmoe

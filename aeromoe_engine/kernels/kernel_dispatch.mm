// kernel_dispatch.mm
// ─────────────────────────────────────────────────────────────────────────────
// KernelDispatch implementation — PSO compilation + command encoding.
// ─────────────────────────────────────────────────────────────────────────────

#include "kernel_dispatch.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cassert>
#include <cstdio>

namespace aeromoe {
namespace kernels {

// ── construction / destruction ────────────────────────────────────────────────

KernelDispatch::KernelDispatch(MTLDevicePtr device)
    : device_(device)
{}

KernelDispatch::~KernelDispatch() {
    for (auto& [name, pso] : pso_cache_) {
        if (pso) CFRelease(pso);
    }
    pso_cache_.clear();
}

// ── PSO resolution ────────────────────────────────────────────────────────────

void* KernelDispatch::get_pso(const std::string& name) {
    auto it = pso_cache_.find(name);
    if (it != pso_cache_.end()) return it->second;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;

    // Compile from default library (all .metal files linked at build time)
    id<MTLLibrary> lib = [dev newDefaultLibrary];
    if (!lib) {
        fprintf(stderr, "[kernel_dispatch] newDefaultLibrary() failed — "
                "make sure Metal shaders are compiled into the app bundle\n");
        return nullptr;
    }

    NSString* fname = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:fname];
    if (!fn) {
        fprintf(stderr, "[kernel_dispatch] Kernel '%s' not found in library\n",
                name.c_str());
        return nullptr;
    }

    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        fprintf(stderr, "[kernel_dispatch] PSO compile failed for '%s': %s\n",
                name.c_str(),
                [[err localizedDescription] UTF8String]);
        return nullptr;
    }

    void* raw = (__bridge_retained void*)pso;
    pso_cache_[name] = raw;
    return raw;
}

Status KernelDispatch::precompile_all() {
    static const char* kernel_names[] = {
        "rms_norm_bf16",
        "rms_norm_head",
        "rms_norm_inplace_bf16",
        "build_yarn_rope_tables",
        "rope_neox_bf16",
        "rope_neox_fused_qk_bf16",
        "kv_cache_append",
        "gqa_attention_decode",
        "gqa_attention_prefill",
        "moe_router_topk",
        "moe_gate_up_silu_bf16",
        "moe_down_proj_bf16",
        "moe_zero_accum",
        "moe_add_accum",
        "moe_reduce_to_bf16",
        "gemv_bf16",
        "gemv_bf16_f32",
        "gemv_bf16_batched",
        "elementwise_add_bf16",
        "argmax_f32",
    };
    for (const char* n : kernel_names) {
        if (!get_pso(n)) return Status::MetalError;
    }
    fprintf(stderr, "[kernel_dispatch] All PSOs compiled OK\n");
    return Status::OK;
}

// ── helper macros ─────────────────────────────────────────────────────────────

#define GET_PSO(name)                                                   \
    id<MTLComputePipelineState> pso =                                   \
        (__bridge id<MTLComputePipelineState>)get_pso(name);            \
    if (!pso) return;                                                    \
    id<MTLCommandBuffer> cmdbuf =                                       \
        (__bridge id<MTLCommandBuffer>)cmd;                             \
    id<MTLComputeCommandEncoder> enc =                                  \
        [cmdbuf computeCommandEncoderWithDispatchType:                  \
            MTLDispatchTypeConcurrent];                                 \
    [enc setComputePipelineState:pso]

#define SET_BUF(idx, buf) \
    [enc setBuffer:(__bridge id<MTLBuffer>)(buf) offset:0 atIndex:(idx)]

#define SET_UINT(idx, val) do { \
    uint32_t _v = (val); \
    [enc setBytes:&_v length:4 atIndex:(idx)]; \
} while(0)

#define SET_FLOAT(idx, val) do { \
    float _v = (val); \
    [enc setBytes:&_v length:4 atIndex:(idx)]; \
} while(0)

#define DISPATCH_1D(groups, tg_sz) do { \
    MTLSize g = MTLSizeMake((groups), 1, 1); \
    MTLSize t = MTLSizeMake((tg_sz),  1, 1); \
    [enc dispatchThreadgroups:g threadsPerThreadgroup:t]; \
    [enc endEncoding]; \
} while(0)

// ── RMSNorm dispatches ────────────────────────────────────────────────────────

void KernelDispatch::dispatch_rms_norm(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t input, MTLBufferPtr_t weight, MTLBufferPtr_t output,
    uint32_t rows, uint32_t hidden, float eps)
{
    GET_PSO("rms_norm_bf16");
    SET_BUF(0, input);
    SET_BUF(1, weight);
    SET_BUF(2, output);
    SET_UINT(3, hidden);
    SET_FLOAT(4, eps);
    // Threadgroup size: min(hidden, 64) rounded to next multiple of 32
    uint32_t tg = ((std::min(hidden, 64u) + 31u) / 32u) * 32u;
    DISPATCH_1D(rows, tg);
}

void KernelDispatch::dispatch_rms_norm_head(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t tensor, MTLBufferPtr_t weight,
    uint32_t seq_times_heads, uint32_t head_dim, float eps)
{
    GET_PSO("rms_norm_head");
    SET_BUF(0, tensor);
    SET_BUF(1, weight);
    SET_BUF(2, tensor);   // in-place: output = input
    SET_UINT(3, head_dim);
    SET_FLOAT(4, eps);
    uint32_t tg = ((std::min(head_dim, 64u) + 31u) / 32u) * 32u;
    DISPATCH_1D(seq_times_heads, tg);
}

void KernelDispatch::dispatch_rms_norm_inplace(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t inout, MTLBufferPtr_t weight,
    uint32_t rows, uint32_t hidden, float eps)
{
    GET_PSO("rms_norm_inplace_bf16");
    SET_BUF(0, inout);
    SET_BUF(1, weight);
    SET_UINT(2, hidden);
    SET_FLOAT(3, eps);
    uint32_t tg = ((std::min(hidden, 64u) + 31u) / 32u) * 32u;
    DISPATCH_1D(rows, tg);
}

// ── RoPE dispatches ───────────────────────────────────────────────────────────

void KernelDispatch::dispatch_build_rope_tables(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t cos_out, MTLBufferPtr_t sin_out,
    uint32_t max_seq_len, uint32_t head_dim,
    float rope_theta, float yarn_scale)
{
    GET_PSO("build_yarn_rope_tables");
    SET_BUF(0, cos_out);
    SET_BUF(1, sin_out);
    SET_FLOAT(2, rope_theta);
    SET_FLOAT(3, yarn_scale);
    SET_UINT(4, head_dim);
    uint32_t tg = head_dim / 2;
    DISPATCH_1D(max_seq_len, tg);
}

void KernelDispatch::dispatch_rope_fused_qk(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t q, MTLBufferPtr_t k,
    MTLBufferPtr_t cos_table, MTLBufferPtr_t sin_table,
    MTLBufferPtr_t pos_ids,
    uint32_t seq_len, uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim)
{
    GET_PSO("rope_neox_fused_qk_bf16");
    SET_BUF(0, q);
    SET_BUF(1, k);
    SET_BUF(2, cos_table);
    SET_BUF(3, sin_table);
    SET_BUF(4, pos_ids);
    SET_UINT(5, n_heads);
    SET_UINT(6, n_kv_heads);
    SET_UINT(7, head_dim);
    uint32_t max_heads = std::max(n_heads, n_kv_heads);
    uint32_t tg        = head_dim / 2;
    DISPATCH_1D(seq_len * max_heads, tg);
}

// ── KV cache ──────────────────────────────────────────────────────────────────

void KernelDispatch::dispatch_kv_append(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t k_in, MTLBufferPtr_t v_in,
    MTLBufferPtr_t k_cache, MTLBufferPtr_t v_cache,
    MTLBufferPtr_t page_table,
    uint32_t seq_len, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t layer, uint32_t seq_start, uint32_t max_pages)
{
    GET_PSO("kv_cache_append");
    SET_BUF(0, k_in);
    SET_BUF(1, v_in);
    SET_BUF(2, k_cache);
    SET_BUF(3, v_cache);
    SET_BUF(4, page_table);
    SET_UINT(5, n_kv_heads);
    SET_UINT(6, head_dim);
    SET_UINT(7, layer);
    SET_UINT(8, seq_start);
    SET_UINT(9, max_pages);
    DISPATCH_1D(seq_len * n_kv_heads, head_dim);
}

// ── Attention ─────────────────────────────────────────────────────────────────

void KernelDispatch::dispatch_gqa_decode(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t q, MTLBufferPtr_t k_cache, MTLBufferPtr_t v_cache,
    MTLBufferPtr_t page_table, MTLBufferPtr_t out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t layer, uint32_t kv_len, uint32_t max_pages)
{
    GET_PSO("gqa_attention_decode");
    SET_BUF(0, q); SET_BUF(1, k_cache); SET_BUF(2, v_cache);
    SET_BUF(3, page_table); SET_BUF(4, out);
    SET_UINT(5, n_heads); SET_UINT(6, n_kv_heads); SET_UINT(7, head_dim);
    SET_UINT(8, layer); SET_UINT(9, kv_len); SET_UINT(10, max_pages);
    DISPATCH_1D(n_heads, head_dim);
}

void KernelDispatch::dispatch_gqa_prefill(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t q, MTLBufferPtr_t k_cache, MTLBufferPtr_t v_cache,
    MTLBufferPtr_t page_table, MTLBufferPtr_t out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t layer, uint32_t kv_len, uint32_t q_len,
    uint32_t q_start, uint32_t max_pages)
{
    GET_PSO("gqa_attention_prefill");
    SET_BUF(0, q); SET_BUF(1, k_cache); SET_BUF(2, v_cache);
    SET_BUF(3, page_table); SET_BUF(4, out);
    SET_UINT(5, n_heads); SET_UINT(6, n_kv_heads); SET_UINT(7, head_dim);
    SET_UINT(8, layer); SET_UINT(9, kv_len); SET_UINT(10, q_start);
    SET_UINT(11, max_pages);
    DISPATCH_1D(q_len * n_heads, head_dim);
}

// ── MoE ───────────────────────────────────────────────────────────────────────

void KernelDispatch::dispatch_moe_router(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t gate_logits, MTLBufferPtr_t expert_ids,
    MTLBufferPtr_t expert_weights,
    uint32_t seq_len, uint32_t num_experts, uint32_t top_k, bool norm_topk)
{
    GET_PSO("moe_router_topk");
    SET_BUF(0, gate_logits); SET_BUF(1, expert_ids); SET_BUF(2, expert_weights);
    SET_UINT(3, num_experts); SET_UINT(4, top_k);
    uint32_t norm = norm_topk ? 1u : 0u;
    SET_UINT(5, norm);
    DISPATCH_1D(seq_len, num_experts);
}

void KernelDispatch::dispatch_moe_gate_up(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t x, MTLBufferPtr_t gate_w, MTLBufferPtr_t up_w,
    MTLBufferPtr_t tmp_out,
    uint32_t hidden, uint32_t moe_inter)
{
    GET_PSO("moe_gate_up_silu_bf16");
    SET_BUF(0, x); SET_BUF(1, gate_w); SET_BUF(2, up_w); SET_BUF(3, tmp_out);
    SET_UINT(4, hidden); SET_UINT(5, moe_inter);
    DISPATCH_1D(moe_inter, 32);
}

void KernelDispatch::dispatch_moe_down(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t tmp, MTLBufferPtr_t down_w, MTLBufferPtr_t accum,
    uint32_t hidden, uint32_t moe_inter, float weight)
{
    GET_PSO("moe_down_proj_bf16");
    SET_BUF(0, tmp); SET_BUF(1, down_w); SET_BUF(2, accum);
    SET_UINT(3, hidden); SET_UINT(4, moe_inter); SET_FLOAT(5, weight);
    DISPATCH_1D(hidden, 32);
}

void KernelDispatch::dispatch_moe_zero_accum(
    MTLCommandBufferPtr cmd, MTLBufferPtr_t accum, uint32_t hidden)
{
    GET_PSO("moe_zero_accum");
    SET_BUF(0, accum); SET_UINT(1, hidden);
    DISPATCH_1D(1, 256);
}

void KernelDispatch::dispatch_moe_add_accum(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t dst, MTLBufferPtr_t src, uint32_t n)
{
    GET_PSO("moe_add_accum");
    SET_BUF(0, dst); SET_BUF(1, src); SET_UINT(2, n);
    DISPATCH_1D(1, 256);
}

void KernelDispatch::dispatch_moe_reduce(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t accum, MTLBufferPtr_t residual, MTLBufferPtr_t out,
    uint32_t hidden)
{
    GET_PSO("moe_reduce_to_bf16");
    SET_BUF(0, accum); SET_BUF(1, residual); SET_BUF(2, out);
    SET_UINT(3, hidden);
    DISPATCH_1D(1, 256);
}

// ── GEMV ──────────────────────────────────────────────────────────────────────

void KernelDispatch::dispatch_gemv_bf16(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t W, MTLBufferPtr_t x, MTLBufferPtr_t y,
    uint32_t out_dim, uint32_t in_dim)
{
    GET_PSO("gemv_bf16");
    SET_BUF(0, W); SET_BUF(1, x); SET_BUF(2, y); SET_UINT(3, in_dim);
    DISPATCH_1D(out_dim, 32);
}

void KernelDispatch::dispatch_gemv_lm_head(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t W, MTLBufferPtr_t x, MTLBufferPtr_t logits,
    uint32_t vocab_size, uint32_t hidden)
{
    GET_PSO("gemv_bf16_f32");
    SET_BUF(0, W); SET_BUF(1, x); SET_BUF(2, logits); SET_UINT(3, hidden);
    DISPATCH_1D(vocab_size, 32);
}

void KernelDispatch::dispatch_gemv_batched(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t W, MTLBufferPtr_t X, MTLBufferPtr_t Y,
    uint32_t out_dim, uint32_t in_dim, uint32_t batch)
{
    GET_PSO("gemv_bf16_batched");
    SET_BUF(0, W); SET_BUF(1, X); SET_BUF(2, Y);
    SET_UINT(3, in_dim); SET_UINT(4, batch);
    DISPATCH_1D(out_dim * batch, 32);
}

void KernelDispatch::dispatch_add_residual(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t y, MTLBufferPtr_t x, uint32_t n)
{
    GET_PSO("elementwise_add_bf16");
    SET_BUF(0, y); SET_BUF(1, x); SET_UINT(2, n);
    uint32_t groups = (n + 255u) / 256u;
    MTLSize g = MTLSizeMake(groups, 1, 1);
    MTLSize t = MTLSizeMake(256, 1, 1);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:t];
    [enc endEncoding];
}

void KernelDispatch::dispatch_argmax(
    MTLCommandBufferPtr cmd,
    MTLBufferPtr_t logits, MTLBufferPtr_t result, uint32_t vocab_size)
{
    GET_PSO("argmax_f32");
    SET_BUF(0, logits); SET_BUF(1, result); SET_UINT(2, vocab_size);
    DISPATCH_1D(1, 256);
}

} // namespace kernels
} // namespace aeromoe

// model_runner.mm
// ─────────────────────────────────────────────────────────────────────────────
// ModelRunner — forward-pass executor for AeroMoE.
// Implements model_runner.h.  ObjC++ (.mm) is required for Metal API calls.
// ─────────────────────────────────────────────────────────────────────────────

#import "model_runner.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>       // memcpy
#include <cmath>         // std::exp
#include <cassert>
#include <algorithm>     // std::partial_sort, std::swap
#include <utility>       // std::pair
#include <cstdio>        // fprintf

namespace aeromoe {

// ── BF16 helpers ─────────────────────────────────────────────────────────────

// Convert a bfloat16 bit pattern (stored as uint16_t) to float32.
// BF16 is the upper 16 bits of an IEEE-754 float32; zero-extending suffices.
static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

// ── ModelRunner ──────────────────────────────────────────────────────────────

ModelRunner::~ModelRunner() {
    shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// init
// ─────────────────────────────────────────────────────────────────────────────
Status ModelRunner::init(EngineCore& engine,
                         KVCacheManager& kv_cache,
                         const ModelRunnerConfig& cfg)
{
    engine_   = &engine;
    kv_cache_ = &kv_cache;
    cfg_      = cfg;

    // ── cache config dimensions ──────────────────────────────────────────────
    const ModelConfig& mc = engine_->config();
    hidden_         = mc.hidden_size;
    head_dim_       = mc.head_dim();
    n_heads_        = mc.num_attention_heads;
    n_kv_heads_     = mc.num_key_value_heads;
    moe_inter_      = mc.moe_intermediate_size;
    inter_          = mc.intermediate_size;
    num_experts_    = mc.num_experts;
    top_k_          = mc.num_experts_per_tok;
    num_layers_     = mc.num_hidden_layers;
    vocab_size_     = mc.vocab_size;
    rms_eps_        = mc.rms_norm_eps;
    norm_topk_prob_ = mc.norm_topk_prob;

    // ── create KernelDispatch ────────────────────────────────────────────────
    dispatch_ = std::make_unique<kernels::KernelDispatch>(engine_->device());

    // Pre-compile all pipeline state objects (PSOs) upfront to avoid
    // first-call latency during inference.
    {
        Status s = dispatch_->precompile_all();
        if (!ok(s)) return s;
    }

    // ── allocate activation (scratch) buffers ───────────────────────────────
    // All buffers use MTLStorageModeShared so the CPU can read them without an
    // explicit blit (needed for embedding lookup, CPU gate routing, logit
    // sampling).

    size_t h2  = (size_t)hidden_ * 2;                      // bf16 hidden row
    size_t hf  = (size_t)hidden_ * 4;                      // f32 hidden row
    size_t q2  = (size_t)n_heads_ * head_dim_ * 2;         // bf16 Q vector
    size_t kv2 = (size_t)n_kv_heads_ * head_dim_ * 2;      // bf16 K or V vector
    size_t mi4 = (size_t)moe_inter_ * 4;                   // f32 MoE inter tmp
    size_t ii4 = (size_t)inter_ * 4;                       // f32 shared expert tmp
    size_t vf  = (size_t)vocab_size_ * 4;                  // f32 logits

    hidden_buf_      = alloc_buf(h2,  "hidden");
    tok_out_buf_     = alloc_buf(h2,  "tok_out");
    normed_buf_      = alloc_buf(h2,  "normed");
    moe_normed_buf_  = alloc_buf(h2,  "moe_normed");
    q_buf_           = alloc_buf(q2,  "q");
    k_buf_           = alloc_buf(kv2, "k");
    v_buf_           = alloc_buf(kv2, "v");
    attn_out_buf_    = alloc_buf(q2,  "attn_out");
    o_proj_out_buf_  = alloc_buf(h2,  "o_proj_out");
    moe_accum_buf_   = alloc_buf(hf,  "moe_accum");
    moe_tmp_buf_     = alloc_buf(mi4, "moe_tmp");
    shared_tmp_buf_  = alloc_buf(ii4, "shared_tmp");
    pos_ids_buf_     = alloc_buf(4,   "pos_ids");       // 1 × uint32
    logits_buf_      = alloc_buf(vf,  "logits");

    // RoPE cosine/sine tables: [max_seq, head_dim/2] f32
    size_t rope_sz = (size_t)cfg_.max_seq_len * (head_dim_ / 2) * 4;
    cos_table_buf_ = alloc_buf(rope_sz, "rope_cos");
    sin_table_buf_ = alloc_buf(rope_sz, "rope_sin");

    // ── CPU scratch vectors ──────────────────────────────────────────────────
    moe_normed_f32_.resize(hidden_, 0.0f);
    gate_logits_scratch_.resize(num_experts_, 0.0f);
    topk_sort_scratch_.resize(num_experts_);

    // ── cache per-layer MTLBuffer pointers ───────────────────────────────────
    // Done once here; avoids string-hash map lookups on the hot 94-layer path.
    layer_mtl_.resize(num_layers_);
    for (uint32_t L = 0; L < num_layers_; ++L) {
        char name[128];
        auto nb = [&](const char* suffix) -> void* {
            snprintf(name, sizeof name, "model.layers.%u.%s", L, suffix);
            return engine_->dense_buf(name);
        };
        LayerMtlBuffers& lm = layer_mtl_[L];
        lm.q_proj     = nb("self_attn.q_proj.weight");
        lm.k_proj     = nb("self_attn.k_proj.weight");
        lm.v_proj     = nb("self_attn.v_proj.weight");
        lm.o_proj     = nb("self_attn.o_proj.weight");
        lm.q_norm     = nb("self_attn.q_norm.weight");
        lm.k_norm     = nb("self_attn.k_norm.weight");
        lm.gate_mtl   = nb("mlp.gate.weight");
        lm.sh_gate    = nb("mlp.shared_expert.gate_proj.weight");
        lm.sh_up      = nb("mlp.shared_expert.up_proj.weight");
        lm.sh_down    = nb("mlp.shared_expert.down_proj.weight");
        lm.input_norm = nb("input_layernorm.weight");
        lm.post_norm  = nb("post_attention_layernorm.weight");

        // Verify all buffers resolved (fatal misconfiguration otherwise)
        assert(lm.q_proj    && "q_proj buffer missing");
        assert(lm.k_proj    && "k_proj buffer missing");
        assert(lm.sh_down   && "sh_down buffer missing");
        assert(lm.input_norm&& "input_norm buffer missing");
    }

    // ── non-layer backbone MTLBuffers ────────────────────────────────────────
    embed_mtl_      = engine_->dense_buf("model.embed_tokens.weight");
    final_norm_mtl_ = engine_->dense_buf("model.norm.weight");
    lm_head_mtl_    = engine_->dense_buf("lm_head.weight");

    // If lm_head is tied to embeddings, fall back.
    if (!lm_head_mtl_) lm_head_mtl_ = embed_mtl_;

    assert(embed_mtl_      && "embed_tokens buffer missing");
    assert(final_norm_mtl_ && "model.norm buffer missing");
    assert(lm_head_mtl_    && "lm_head buffer missing");

    // ── build RoPE tables (one-shot GPU command) ─────────────────────────────
    {
        void* cmd = new_cmd();
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cmd;
        dispatch_->dispatch_build_rope_tables(
            cb,
            (__bridge id<MTLBuffer>)cos_table_buf_,
            (__bridge id<MTLBuffer>)sin_table_buf_,
            cfg_.max_seq_len,
            head_dim_,
            mc.rope_theta,
            cfg_.yarn_scale
        );
        commit_and_wait(cmd);
    }

    return Status::OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// shutdown
// ─────────────────────────────────────────────────────────────────────────────
void ModelRunner::shutdown() {
    // Drain any leftover sub-buffers from a partial run.
    drain_subbuf_pool();

    // Release all owned MTLBuffers (alloc_buf() stored __bridge_retained refs).
    for (void* raw : owned_bufs_) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)raw;
        (void)buf;  // ARC decrements retain count
    }
    owned_bufs_.clear();

    // Null out all buffer pointers (they point into owned_bufs_ storage).
    hidden_buf_ = tok_out_buf_ = normed_buf_ = moe_normed_buf_ = nullptr;
    q_buf_ = k_buf_ = v_buf_ = attn_out_buf_ = o_proj_out_buf_ = nullptr;
    moe_accum_buf_ = moe_tmp_buf_ = shared_tmp_buf_ = nullptr;
    pos_ids_buf_ = cos_table_buf_ = sin_table_buf_ = logits_buf_ = nullptr;

    dispatch_.reset();
    engine_   = nullptr;
    kv_cache_ = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// forward  — main entry point
// ─────────────────────────────────────────────────────────────────────────────
Status ModelRunner::forward(uint32_t token_id,
                            uint32_t seq_id,
                            uint32_t pos,
                            float**  logits_out)
{
    // ── ensure KV pages are allocated and flush page table to GPU ────────────
    if (!kv_cache_->ensure_capacity(seq_id, pos + 1)) {
        fprintf(stderr, "[ModelRunner] KV cache full: seq %u pos %u\n",
                seq_id, pos);
        return Status::BudgetExceeded;
    }
    kv_cache_->flush_page_table();

    // ── 1. Embedding lookup (CPU copy, ~8 KB) ────────────────────────────────
    phase_embed(token_id);

    // ── 2. Write position ID ─────────────────────────────────────────────────
    phase_build_pos_ids(pos);

    // ── 3. Decoder layers ───────────────────────────────────────────────────
    for (uint32_t layer = 0; layer < num_layers_; ++layer) {
        // Attention + post-attention residual + pre-MoE norm (1 GPU sync)
        run_attention(layer, seq_id, pos);

        // CPU gate routing + expert FFN + residual (1 GPU sync)
        run_moe_layer(layer);

        // Swap buffers: tok_out_buf_ (new state) → hidden_buf_ for next layer.
        // This is a pointer swap — zero copy, zero GPU activity.
        std::swap(hidden_buf_, tok_out_buf_);
    }

    // ── 4. Final norm + LM head ──────────────────────────────────────────────
    phase_lm_head();

    // ── 5. Return pointer into logits buffer ─────────────────────────────────
    if (logits_out) {
        id<MTLBuffer> lb = (__bridge id<MTLBuffer>)logits_buf_;
        *logits_out = static_cast<float*>([lb contents]);
    }

    return Status::OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// phase_embed
// ─────────────────────────────────────────────────────────────────────────────
// Copy one embedding row into hidden_buf_ on CPU.
// Both buffers are MTLStorageModeShared so no GPU blit is needed.
void ModelRunner::phase_embed(uint32_t token_id) {
    const DenseBuffers& bb = engine_->backbone();
    assert(bb.embed_tokens && "embed_tokens weight is null");

    size_t row_bytes = (size_t)hidden_ * 2;  // bf16
    const uint8_t* src =
        static_cast<const uint8_t*>(bb.embed_tokens)
        + (size_t)token_id * row_bytes;

    id<MTLBuffer> hbuf = (__bridge id<MTLBuffer>)hidden_buf_;
    memcpy([hbuf contents], src, row_bytes);
}

// ─────────────────────────────────────────────────────────────────────────────
// phase_build_pos_ids
// ─────────────────────────────────────────────────────────────────────────────
void ModelRunner::phase_build_pos_ids(uint32_t pos) {
    id<MTLBuffer> pbuf = (__bridge id<MTLBuffer>)pos_ids_buf_;
    uint32_t* p = static_cast<uint32_t*>([pbuf contents]);
    p[0] = pos;
}

// ─────────────────────────────────────────────────────────────────────────────
// run_attention
// ─────────────────────────────────────────────────────────────────────────────
// Encodes the full attention sublayer plus the post-attention (pre-MoE) norm
// into one command buffer and blocks until the GPU finishes.
//
// Pipeline:
//   input_norm(hidden) → normed
//   Q = q_proj · normed            [n_heads × head_dim] bf16
//   K = k_proj · normed            [n_kv_heads × head_dim] bf16
//   V = v_proj · normed            [n_kv_heads × head_dim] bf16
//   Q = qk_norm_head(Q, q_norm)    [per-head RMSNorm]
//   K = qk_norm_head(K, k_norm)
//   RoPE fused Q,K
//   KV cache append
//   attn_out = GQA_decode(Q, K_cache, V_cache)
//   o_out = o_proj · attn_out
//   hidden += o_out                [attention residual]
//   moe_normed = post_norm(hidden) [pre-MoE norm, read by CPU routing]
//
void ModelRunner::run_attention(uint32_t layer, uint32_t seq_id, uint32_t pos)
{
    const LayerMtlBuffers& lm = layer_mtl_[layer];
    const KVCacheConfig& kvc = kv_cache_->config();

    // Convenience bridged buffer references
    id<MTLBuffer> hidden    = (__bridge id<MTLBuffer>)hidden_buf_;
    id<MTLBuffer> normed    = (__bridge id<MTLBuffer>)normed_buf_;
    id<MTLBuffer> q         = (__bridge id<MTLBuffer>)q_buf_;
    id<MTLBuffer> k         = (__bridge id<MTLBuffer>)k_buf_;
    id<MTLBuffer> v         = (__bridge id<MTLBuffer>)v_buf_;
    id<MTLBuffer> attn_out  = (__bridge id<MTLBuffer>)attn_out_buf_;
    id<MTLBuffer> o_out     = (__bridge id<MTLBuffer>)o_proj_out_buf_;
    id<MTLBuffer> moe_n     = (__bridge id<MTLBuffer>)moe_normed_buf_;
    id<MTLBuffer> pos_ids   = (__bridge id<MTLBuffer>)pos_ids_buf_;
    id<MTLBuffer> cos_tbl   = (__bridge id<MTLBuffer>)cos_table_buf_;
    id<MTLBuffer> sin_tbl   = (__bridge id<MTLBuffer>)sin_table_buf_;

    id<MTLBuffer> q_w  = (__bridge id<MTLBuffer>)lm.q_proj;
    id<MTLBuffer> k_w  = (__bridge id<MTLBuffer>)lm.k_proj;
    id<MTLBuffer> v_w  = (__bridge id<MTLBuffer>)lm.v_proj;
    id<MTLBuffer> o_w  = (__bridge id<MTLBuffer>)lm.o_proj;
    id<MTLBuffer> qn_w = (__bridge id<MTLBuffer>)lm.q_norm;
    id<MTLBuffer> kn_w = (__bridge id<MTLBuffer>)lm.k_norm;
    id<MTLBuffer> in_w = (__bridge id<MTLBuffer>)lm.input_norm;
    id<MTLBuffer> pn_w = (__bridge id<MTLBuffer>)lm.post_norm;

    uint32_t kv_len  = pos + 1;          // total tokens in sequence after append
    uint32_t n_qh    = n_heads_;
    uint32_t n_kvh   = n_kv_heads_;
    uint32_t hdim    = head_dim_;
    uint32_t h       = hidden_;
    float    eps     = rms_eps_;

    void* cmd_raw = new_cmd();
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_raw;

    // 1. Input layernorm: hidden → normed
    dispatch_->dispatch_rms_norm(cmd, hidden, in_w, normed, /*rows=*/1, h, eps);

    // 2. Q, K, V projections
    //    Q: [n_heads * head_dim, hidden] × [hidden] → [n_heads * head_dim]
    dispatch_->dispatch_gemv_bf16(cmd, q_w, normed, q, n_qh * hdim, h);
    //    K: [n_kv_heads * head_dim, hidden] × [hidden] → [n_kv_heads * head_dim]
    dispatch_->dispatch_gemv_bf16(cmd, k_w, normed, k, n_kvh * hdim, h);
    //    V: same shape as K
    dispatch_->dispatch_gemv_bf16(cmd, v_w, normed, v, n_kvh * hdim, h);

    // 3. Per-head QK-norm (in-place)
    //    Qwen3 applies RMSNorm independently to each head's query and key.
    dispatch_->dispatch_rms_norm_head(cmd, q, qn_w, /*seq*heads=*/n_qh,  hdim, eps);
    dispatch_->dispatch_rms_norm_head(cmd, k, kn_w, /*seq*heads=*/n_kvh, hdim, eps);

    // 4. Fused Q,K RoPE (NeoX half-split, YaRN tables pre-built)
    dispatch_->dispatch_rope_fused_qk(cmd, q, k, cos_tbl, sin_tbl, pos_ids,
                                      /*seq_len=*/1, n_qh, n_kvh, hdim);

    // 5. Append K and V into the paged KV cache
    dispatch_->dispatch_kv_append(
        cmd, k, v,
        kv_cache_->k_pool(), kv_cache_->v_pool(), kv_cache_->page_table_buf(),
        /*seq_len=*/1, n_kvh, hdim, layer,
        /*seq_start=*/pos,          // write slot index = current position
        kv_cache_->max_pages()
    );

    // 6. GQA single-token decode attention
    dispatch_->dispatch_gqa_decode(
        cmd, q,
        kv_cache_->k_pool(), kv_cache_->v_pool(), kv_cache_->page_table_buf(),
        attn_out,
        n_qh, n_kvh, hdim, layer, kv_len,
        kv_cache_->max_pages()
    );

    // 7. O-projection: [hidden, n_heads * head_dim] × attn_out → o_out
    dispatch_->dispatch_gemv_bf16(cmd, o_w, attn_out, o_out, h, n_qh * hdim);

    // 8. Post-attention residual: hidden += o_out
    dispatch_->dispatch_add_residual(cmd, hidden, o_out, h);

    // 9. Post-attention layernorm (pre-MoE norm): hidden → moe_normed.
    //    Encoding this at the END of the attention command buffer means we
    //    only need ONE commit+wait before CPU can read moe_normed for routing,
    //    saving one extra GPU sync per layer.
    dispatch_->dispatch_rms_norm(cmd, hidden, pn_w, moe_n, /*rows=*/1, h, eps);

    // ── commit and block ─────────────────────────────────────────────────────
    // After this returns, moe_normed_buf_ is coherent on CPU (MTLStorageModeShared
    // guarantees CPU sees the written data once waitUntilCompleted returns).
    commit_and_wait(cmd_raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// run_moe_layer
// ─────────────────────────────────────────────────────────────────────────────
// CPU gate routing → acquire expert slabs → GPU expert FFN + reduce.
//
// Pipeline:
//   [CPU]  expand moe_normed (bf16→f32)
//   [CPU]  gate_logits[e] = dot(gate_w[e], moe_normed)
//   [CPU]  top-k selection + optional softmax normalisation
//   [GPU]  zero moe_accum
//   [GPU]  for each top-k routed expert:
//              gate_up(moe_normed, gate_w[e], up_w[e]) → moe_tmp
//              down(moe_tmp, down_w[e], weight) → moe_accum  (weighted add)
//   [GPU]  shared expert (always active, weight=1.0):
//              gate_up(moe_normed, sh_gate, sh_up) → shared_tmp
//              down(shared_tmp, sh_down, 1.0) → moe_accum
//   [GPU]  moe_reduce(moe_accum, hidden, tok_out)   [f32 + bf16 → bf16]
//
void ModelRunner::run_moe_layer(uint32_t layer)
{
    // ── CPU phase ─────────────────────────────────────────────────────────────

    // Convert moe_normed_buf_ (bf16) to f32 scratch for dot-product routing.
    expand_moe_normed_to_f32();

    // Compute gate logits (128 × 4096 MACs ≈ < 0.5 ms on Apple Silicon)
    cpu_gate_logits(layer);

    // Select top-k experts and compute normalised weights.
    std::vector<uint32_t> expert_ids;
    std::vector<float>    expert_weights;
    cpu_top_k_route(expert_ids, expert_weights);

    // ── load expert weight slabs (may trigger I/O for cold experts) ───────────
    std::vector<ExpertView> views =
        engine_->acquire_layer_experts(layer, expert_ids);

    // ── GPU phase ─────────────────────────────────────────────────────────────
    const LayerMtlBuffers& lm = layer_mtl_[layer];

    id<MTLBuffer> moe_n    = (__bridge id<MTLBuffer>)moe_normed_buf_;
    id<MTLBuffer> accum    = (__bridge id<MTLBuffer>)moe_accum_buf_;
    id<MTLBuffer> tmp      = (__bridge id<MTLBuffer>)moe_tmp_buf_;
    id<MTLBuffer> sh_tmp   = (__bridge id<MTLBuffer>)shared_tmp_buf_;
    id<MTLBuffer> hidden   = (__bridge id<MTLBuffer>)hidden_buf_;
    id<MTLBuffer> tok_out  = (__bridge id<MTLBuffer>)tok_out_buf_;

    id<MTLBuffer> sh_gate_w = (__bridge id<MTLBuffer>)lm.sh_gate;
    id<MTLBuffer> sh_up_w   = (__bridge id<MTLBuffer>)lm.sh_up;
    id<MTLBuffer> sh_down_w = (__bridge id<MTLBuffer>)lm.sh_down;

    void* cmd_raw = new_cmd();
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_raw;

    // Zero the f32 accumulator before adding any expert outputs.
    dispatch_->dispatch_moe_zero_accum(cmd, accum, hidden_);

    // ── routed experts ────────────────────────────────────────────────────────
    // Expert weight matrices (gate, up, down) live as contiguous sub-slabs
    // inside a single MTLBuffer per expert.  We create zero-copy sub-buffer
    // views so the GPU kernels receive the correct base pointer.
    //
    // For Qwen3.6 (moe_inter=768, hidden=4096):
    //   gate_proj: 768×4096×2 = 6,291,456 B  (6 MiB, multiple of page size)
    //   up_proj:   768×4096×2 = 6,291,456 B
    //   down_proj: 4096×768×2 = 6,291,456 B
    // All sub-ranges are page-aligned → newBufferWithBytesNoCopy: succeeds.

    const size_t gate_sz = (size_t)moe_inter_ * hidden_ * 2;  // bf16
    const size_t down_sz = (size_t)hidden_     * moe_inter_ * 2;

    for (uint32_t i = 0; i < top_k_; ++i) {
        const ExpertView& ev = views[i];

        // Create zero-copy sub-buffer views for this expert's three matrices.
        id<MTLBuffer> gate_sub = (__bridge id<MTLBuffer>)
            make_subbuf((void*)ev.gate_proj.data, gate_sz);
        id<MTLBuffer> up_sub   = (__bridge id<MTLBuffer>)
            make_subbuf((void*)ev.up_proj.data,   gate_sz);
        id<MTLBuffer> down_sub = (__bridge id<MTLBuffer>)
            make_subbuf((void*)ev.down_proj.data,  down_sz);

        // gate_up: x → silu(gate_w·x) * (up_w·x)  → tmp  [moe_inter f32]
        dispatch_->dispatch_moe_gate_up(
            cmd, moe_n, gate_sub, up_sub, tmp,
            hidden_, moe_inter_
        );

        // down: tmp → accum (weighted accumulate with routing weight)
        dispatch_->dispatch_moe_down(
            cmd, tmp, down_sub, accum,
            hidden_, moe_inter_,
            expert_weights[i]
        );
    }

    // ── shared expert (always active, coefficient = 1.0) ─────────────────────
    // Intermediate size for the shared expert is `inter_` (e.g. 2048),
    // larger than the routed `moe_inter_` (e.g. 768).  Uses shared_tmp_buf_.

    dispatch_->dispatch_moe_gate_up(
        cmd, moe_n, sh_gate_w, sh_up_w, sh_tmp,
        hidden_, inter_
    );
    dispatch_->dispatch_moe_down(
        cmd, sh_tmp, sh_down_w, accum,
        hidden_, inter_,
        /*weight=*/1.0f
    );

    // ── reduce: f32 accum + bf16 residual → bf16 tok_out ─────────────────────
    // `hidden` is the pre-MoE hidden state (= post-attention residual).
    // tok_out = accum + hidden  (cast to bf16).
    // After this the caller will std::swap(hidden_buf_, tok_out_buf_).

    dispatch_->dispatch_moe_reduce(cmd, accum, hidden, tok_out, hidden_);

    // ── commit, wait, then release in the right order ────────────────────────
    commit_and_wait(cmd_raw);
    // GPU is done; safe to release resources now.

    // 1. Drain sub-buffers FIRST.  Sub-buffers are zero-copy views into expert
    //    slab memory (newBufferWithBytesNoCopy: / deallocator:nil).  The slab
    //    is still pinned here (pin_count > 0), so its memory is valid.
    //    If we released expert pins first, the cache could evict the slab
    //    immediately, leaving the sub-buffers pointing at freed memory.
    drain_subbuf_pool();

    // 2. Unpin expert slabs.  Now the expert cache is free to evict them
    //    (once all sub-buffers referencing their memory have been released).
    engine_->release_layer_experts(layer, expert_ids);
}

// ─────────────────────────────────────────────────────────────────────────────
// phase_lm_head
// ─────────────────────────────────────────────────────────────────────────────
// Final RMSNorm (in-place on hidden_buf_) followed by the LM-head GEMV.
// Writes vocab_size f32 logits to logits_buf_.
void ModelRunner::phase_lm_head() {
    id<MTLBuffer> hidden   = (__bridge id<MTLBuffer>)hidden_buf_;
    id<MTLBuffer> norm_w   = (__bridge id<MTLBuffer>)final_norm_mtl_;
    id<MTLBuffer> lm_head  = (__bridge id<MTLBuffer>)lm_head_mtl_;
    id<MTLBuffer> logits   = (__bridge id<MTLBuffer>)logits_buf_;

    void* cmd_raw = new_cmd();
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_raw;

    // Final RMSNorm in-place
    dispatch_->dispatch_rms_norm_inplace(
        cmd, hidden, norm_w,
        /*rows=*/1, hidden_, rms_eps_
    );

    // LM head projection: [vocab, hidden] × [hidden] → [vocab] f32 logits
    dispatch_->dispatch_gemv_lm_head(
        cmd, lm_head, hidden, logits,
        vocab_size_, hidden_
    );

    commit_and_wait(cmd_raw);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU gate routing helpers
// ─────────────────────────────────────────────────────────────────────────────

void ModelRunner::expand_moe_normed_to_f32() {
    // moe_normed_buf_ is MTLStorageModeShared; its contents are CPU-coherent
    // after the preceding commit_and_wait in run_attention.
    id<MTLBuffer> mbuf = (__bridge id<MTLBuffer>)moe_normed_buf_;
    const uint16_t* src = static_cast<const uint16_t*>([mbuf contents]);
    float* dst = moe_normed_f32_.data();
    for (uint32_t i = 0; i < hidden_; ++i) {
        dst[i] = bf16_to_f32(src[i]);
    }
}

void ModelRunner::cpu_gate_logits(uint32_t layer) {
    // Gate weight is a [num_experts, hidden] bf16 matrix stored CPU-side
    // (MTLStorageModeShared → backbone().layers[L].gate is [mtlbuf contents]).
    const void* gate_cpu = engine_->backbone().layers[layer].gate;
    assert(gate_cpu && "gate weight pointer is null");

    const uint16_t* gate_bf16 = static_cast<const uint16_t*>(gate_cpu);
    const float*    normed    = moe_normed_f32_.data();
    float*          out       = gate_logits_scratch_.data();

    // Compute one dot product per expert.
    // 128 experts × 4096 hidden = 524 288 multiply-adds.
    // On Apple M-series CPUs this completes in < 0.5 ms.
    for (uint32_t e = 0; e < num_experts_; ++e) {
        const uint16_t* row = gate_bf16 + (size_t)e * hidden_;
        double acc = 0.0;  // double for numerical accuracy over 4096 terms
        for (uint32_t d = 0; d < hidden_; ++d) {
            acc += (double)bf16_to_f32(row[d]) * (double)normed[d];
        }
        out[e] = (float)acc;
    }
}

void ModelRunner::cpu_top_k_route(std::vector<uint32_t>& ids_out,
                                   std::vector<float>&    weights_out)
{
    // Build (logit, expert_id) pairs
    for (uint32_t e = 0; e < num_experts_; ++e) {
        topk_sort_scratch_[e] = {gate_logits_scratch_[e], e};
    }

    // Partial sort: move top-k elements (by descending logit) to front.
    // O(num_experts * log(top_k)) — much faster than full sort.
    std::partial_sort(
        topk_sort_scratch_.begin(),
        topk_sort_scratch_.begin() + top_k_,
        topk_sort_scratch_.end(),
        [](const std::pair<float,uint32_t>& a,
           const std::pair<float,uint32_t>& b) {
            return a.first > b.first;  // descending
        }
    );

    ids_out.resize(top_k_);
    weights_out.resize(top_k_);

    for (uint32_t i = 0; i < top_k_; ++i) {
        ids_out[i]     = topk_sort_scratch_[i].second;
        weights_out[i] = topk_sort_scratch_[i].first;  // raw logit for now
    }

    // Convert logits to normalised routing weights.
    if (norm_topk_prob_) {
        // Qwen3: softmax over the top-k logits (with max-subtraction for
        // numerical stability), then normalise to sum to 1.0.
        float max_v = weights_out[0];  // partial_sort guarantees descending order
        float sum   = 0.0f;
        for (uint32_t i = 0; i < top_k_; ++i) {
            weights_out[i] = std::exp(weights_out[i] - max_v);
            sum += weights_out[i];
        }
        float inv = 1.0f / sum;
        for (uint32_t i = 0; i < top_k_; ++i) {
            weights_out[i] *= inv;
        }
    } else {
        // Alternative: sigmoid on each logit independently (not used for
        // Qwen3.6-35B-A3B, included for compatibility with other variants).
        for (uint32_t i = 0; i < top_k_; ++i) {
            float x = weights_out[i];
            weights_out[i] = 1.0f / (1.0f + std::exp(-x));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metal helpers
// ─────────────────────────────────────────────────────────────────────────────

void* ModelRunner::new_cmd() {
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)engine_->command_queue();
    assert(queue && "Metal command queue is nil");

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    assert(cmd && "Failed to create MTLCommandBuffer");

    // Transfer ownership to C++ (refcount +1); caller must eventually call
    // commit_and_wait() which transfers ownership back to ARC and releases it.
    return (__bridge_retained void*)cmd;
}

void ModelRunner::commit_and_wait(void* cmd_raw) {
    // Transfer ownership back to ARC so the command buffer is released after
    // waitUntilCompleted.
    id<MTLCommandBuffer> cmd =
        (__bridge_transfer id<MTLCommandBuffer>)cmd_raw;

    [cmd commit];
    [cmd waitUntilCompleted];

    // Check for GPU-side errors.
    if (cmd.status == MTLCommandBufferStatusError) {
        NSError* err = cmd.error;
        fprintf(stderr, "[ModelRunner] GPU command buffer error: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(nil)");
        abort();  // unrecoverable; propagate via crash dump
    }
}

void* ModelRunner::alloc_buf(size_t bytes, const char* label) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)engine_->device();
    assert(device && "MTLDevice is nil");

    id<MTLBuffer> buf =
        [device newBufferWithLength:bytes
                            options:MTLResourceStorageModeShared];
    assert(buf && "Failed to allocate MTLBuffer");

    if (label) {
        buf.label = [NSString stringWithUTF8String:label];
    }

    // Keep a retained void* so ARC does not release the buffer prematurely.
    void* retained = (__bridge_retained void*)buf;
    owned_bufs_.push_back(retained);

    // Return a non-owning pointer that callers can cast back to id<MTLBuffer>.
    return (__bridge void*)buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// Expert sub-buffer helpers
// ─────────────────────────────────────────────────────────────────────────────

void* ModelRunner::make_subbuf(void* ptr, size_t length) {
    // ptr must be page-aligned and length must be a multiple of the page size.
    // For Qwen3.6 all expert sub-ranges satisfy these constraints (see header).
    assert(((uintptr_t)ptr & 0xFFF) == 0 &&
           "Expert sub-buffer pointer is not page-aligned");
    assert((length & 0xFFF) == 0 &&
           "Expert sub-buffer length is not a page multiple");

    id<MTLDevice> device = (__bridge id<MTLDevice>)engine_->device();

    // newBufferWithBytesNoCopy: wraps existing CPU memory without copying.
    // deallocator:nil — we do NOT want Metal to free this memory; it is
    // owned by the ExpertCache slab and will be freed by the cache.
    id<MTLBuffer> sub =
        [device newBufferWithBytesNoCopy:ptr
                                  length:length
                                 options:MTLResourceStorageModeShared
                             deallocator:nil];
    assert(sub && "Failed to create zero-copy sub-buffer for expert slab");

    // Retain until drain_subbuf_pool() — must outlive the command buffer.
    subbuf_pool_.push_back((__bridge_retained void*)sub);

    return (__bridge void*)sub;
}

void ModelRunner::drain_subbuf_pool() {
    // Transfer each retained void* back to ARC so the underlying buffers
    // are released.  The expert slab (pointed to by the sub-buffers) is
    // released separately by release_layer_experts().
    for (void* raw : subbuf_pool_) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)raw;
        (void)buf;  // ARC decrements retain count
    }
    subbuf_pool_.clear();
}

} // namespace aeromoe

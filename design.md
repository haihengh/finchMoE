AeroMoE — Design Review & Core Inference Engine Code
Part 1: Design Review
What holds up
Decision	Verdict
Metal-only, no ANE	✅ Correct — dynamic expert binding is impossible to do cleanly on ANE
Layer-sharded expert files, contiguous per-expert slices	✅ Correct — one pread() per miss
Bounded slot cache instead of load/unload per token	✅ Correct — routing locality makes blind unloads waste SSD bandwidth
No speculative prefetch	✅ Correct for v1 — revisit only with measured hit rates
Overlap shared-expert GPU work with SSD reads	✅ Correct, this is the single biggest latency hiding win
Weaknesses to fix before coding
Router→CPU sync point. The GPU must finish router logits before the CPU can plan I/O. That's a per-layer pipeline stall. Fix: split each layer into two command buffers and read back only top_k × (uint16 id, half weight) via a shared-storage-mode buffer — never a full logits copy. On decode (1 token), do top-k on CPU from the raw logits vector (small, e.g. 128 floats); it's cheaper than a GPU sort + readback.
Slot fragmentation. Experts across layers may differ in size. Fix: all slots sized to max_expert_bytes from the manifest. Wastes a few MB, eliminates an allocator.
Eviction during in-flight GPU reads. A slot must not be evicted while a command buffer still references it. Fix: per-slot refcount pinned until command-buffer completion handler fires.
pread into Metal buffers. Use MTLStorageModeShared buffers and pread() directly into buffer.contents — zero-copy on unified memory. Open the expert files with F_NOCACHE off by default; only enable it under memory pressure (watch dispatch_source memory-pressure events).
Budget enforcement is missing a mechanism. Add a central MemoryLedger that every allocation goes through; fail construction if the plan exceeds the ceiling instead of discovering it at runtime.
Part 2: Core Engine Code

Language: C++17 / Objective-C++ (.mm) + Metal Shading Language. Illustrative but structurally complete — tensor shapes come from manifest.json at load time, never hard-coded.

2.1 Manifest & expert index
// aeromoe_format.h
#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace aeromoe {

struct QuantSpec {            // per-tensor-group quantization
    enum Kind : uint8_t { Q4_BLOCK32, Q5_BLOCK32, Q8, F16, F32 };
    Kind kind;
    uint32_t block_size;      // elements per quant block (e.g. 32)
};

struct ModelManifest {
    uint32_t n_layers, d_model, n_heads, n_kv_heads, head_dim;
    uint32_t n_experts, top_k;
    uint32_t d_ff_expert, d_ff_shared;     // shared-expert width; 0 if none
    uint32_t vocab_size, max_context;
    float    rope_theta;
    QuantSpec backbone_q, expert_q, router_q;
    uint64_t max_expert_bytes;             // sizing for uniform slots
    // parsed from manifest.json; validated against tensor coverage
};

// index.bin: one record per (layer, expert), contiguous, mmap-friendly
struct ExpertRecord {
    uint64_t offset;          // into experts/layer_XXX.bin, 64KB aligned
    uint32_t length;          // total bytes: gate+up+down, packed for Metal
    uint32_t gate_up_bytes;   // split point inside the slice
};

struct ExpertIndex {
    std::vector<ExpertRecord> records;     // n_layers * n_experts
    const ExpertRecord& at(uint32_t layer, uint32_t expert) const {
        return records[layer * n_experts_ + expert];
    }
    uint32_t n_experts_;
};

} // namespace aeromoe

2.2 Memory ledger (the 4 GB contract)
// memory_ledger.h
#pragma once
#include <atomic>
#include <cstdint>
#include <stdexcept>

namespace aeromoe {

class MemoryLedger {
public:
    explicit MemoryLedger(uint64_t ceiling_bytes) : ceiling_(ceiling_bytes) {}

    // All engine allocations MUST reserve here first.
    void reserve(uint64_t bytes, const char* tag) {
        uint64_t cur = used_.fetch_add(bytes) + bytes;
        if (cur > ceiling_) {
            used_.fetch_sub(bytes);
            throw std::runtime_error(
                std::string("AeroMoE budget exceeded reserving ") + tag);
        }
    }
    void release(uint64_t bytes) { used_.fetch_sub(bytes); }
    uint64_t used() const { return used_.load(); }
    uint64_t ceiling() const { return ceiling_; }
private:
    std::atomic<uint64_t> used_{0};
    uint64_t ceiling_;
};

} // namespace aeromoe


Construction-time plan check (fail fast, never swap):

// Called once during engine init, before any Metal allocation.
void validate_budget(const ModelManifest& m, const EngineConfig& cfg,
                     MemoryLedger& ledger) {
    ledger.reserve(cfg.backbone_bytes,      "backbone");       // ~1.6–2.1 GB
    ledger.reserve(cfg.kv_budget_bytes,     "kv_cache");       // ~0.6–1.0 GB
    ledger.reserve(cfg.n_slots * m.max_expert_bytes, "expert_slots");
    ledger.reserve(cfg.scratch_bytes,       "metal_scratch");  // fixed arena
    // If we get here, the plan fits under the ceiling (e.g. 3.9 GB).
}

2.3 Expert slot cache (bounded, refcounted, LFU+recency)
// expert_cache.h
#pragma once
#import <Metal/Metal.h>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace aeromoe {

struct SlotKey {
    uint32_t layer, expert;
    bool operator==(const SlotKey& o) const {
        return layer == o.layer && expert == o.expert;
    }
};
struct SlotKeyHash {
    size_t operator()(const SlotKey& k) const {
        return (size_t(k.layer) << 32) ^ k.expert;
    }
};

struct Slot {
    id<MTLBuffer> buffer;        // MTLStorageModeShared, max_expert_bytes
    SlotKey key{UINT32_MAX, UINT32_MAX};
    std::atomic<int> refcount{0};   // pinned while a cmd buffer uses it
    uint32_t freq = 0;              // decayed frequency score
    uint64_t last_use_token = 0;
    bool valid = false;
};

class ExpertCache {
public:
    ExpertCache(id<MTLDevice> dev, uint32_t n_slots, uint64_t slot_bytes,
                MemoryLedger& ledger) : ledger_(ledger) {
        // slot memory was already reserved by validate_budget()
        slots_.resize(n_slots);
        for (auto& s : slots_)
            s.buffer = [dev newBufferWithLength:slot_bytes
                                        options:MTLResourceStorageModeShared];
    }

    // Returns slot if resident (pins it), else nullptr.
    Slot* lookup_and_pin(SlotKey key, uint64_t token_idx) {
        std::lock_guard<std::mutex> g(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        Slot* s = &slots_[it->second];
        s->refcount.fetch_add(1);
        s->freq++; s->last_use_token = token_idx;
        return s;
    }

    // Picks a victim: unpinned, lowest (freq, last_use). Caller fills it.
    Slot* acquire_for_fill(SlotKey key, uint64_t token_idx) {
        std::lock_guard<std::mutex> g(mu_);
        Slot* victim = nullptr;
        for (auto& s : slots_) {
            if (s.refcount.load() != 0) continue;         // pinned: skip
            if (!s.valid) { victim = &s; break; }         // free slot
            if (!victim || std::tie(s.freq, s.last_use_token) <
                           std::tie(victim->freq, victim->last_use_token))
                victim = &s;
        }
        if (!victim) return nullptr;   // all pinned → caller stalls one read
        if (victim->valid) map_.erase(victim->key);
        victim->key = key; victim->valid = false;         // valid after fill
        victim->freq = 1;  victim->last_use_token = token_idx;
        victim->refcount.fetch_add(1);                    // pin for fill+use
        return victim;
    }

    void publish(Slot* s) {           // after pread completes
        std::lock_guard<std::mutex> g(mu_);
        s->valid = true;
        map_[s->key] = uint32_t(s - slots_.data());
    }
    void unpin(Slot* s) { s->refcount.fetch_sub(1); }

    void decay() {                    // call every N tokens
        std::lock_guard<std::mutex> g(mu_);
        for (auto& s : slots_) s.freq >>= 1;
    }
private:
    std::vector<Slot> slots_;
    std::unordered_map<SlotKey, uint32_t, SlotKeyHash> map_;
    std::mutex mu_;
    MemoryLedger& ledger_;
};

} // namespace aeromoe

2.4 I/O planner — bounded parallel pread into Metal buffers
// io_planner.mm
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <unistd.h>
#include "expert_cache.h"
#include "aeromoe_format.h"

namespace aeromoe {

class IOPlanner {
public:
    IOPlanner(const ExpertIndex& idx, std::vector<int> layer_fds,
              ExpertCache& cache, uint32_t max_inflight = 3)
      : idx_(idx), fds_(std::move(layer_fds)), cache_(cache) {
        q_   = dispatch_queue_create("aeromoe.io", DISPATCH_QUEUE_CONCURRENT);
        sem_ = dispatch_semaphore_create(max_inflight);   // autotuned 2–4
        grp_ = dispatch_group_create();
    }

    // For each selected expert: hit → pin; miss → async pread into a slot.
    // Returns pinned slots in expert order; blocks until all fills complete.
    std::vector<Slot*> fetch(uint32_t layer,
                             const std::vector<uint32_t>& experts,
                             uint64_t token_idx) {
        std::vector<Slot*> out(experts.size(), nullptr);
        for (size_t i = 0; i < experts.size(); ++i) {
            SlotKey key{layer, experts[i]};
            if (Slot* hit = cache_.lookup_and_pin(key, token_idx)) {
                out[i] = hit;
                continue;
            }
            Slot* slot = nullptr;
            while (!(slot = cache_.acquire_for_fill(key, token_idx)))
                dispatch_group_wait(grp_, DISPATCH_TIME_FOREVER); // rare
            out[i] = slot;
            const ExpertRecord& rec = idx_.at(layer, experts[i]);
            int fd = fds_[layer];
            dispatch_semaphore_wait(sem_, DISPATCH_TIME_FOREVER);
            dispatch_group_async(grp_, q_, ^{
                // Zero-copy on unified memory: read straight into the
                // MTLBuffer's shared-storage contents.
                void* dst = slot->buffer.contents;
                ssize_t n = pread(fd, dst, rec.length, off_t(rec.offset));
                if (n != (ssize_t)rec.length) abort();  // TODO: error path
                cache_.publish(slot);
                dispatch_semaphore_signal(sem_);
            });
        }
        dispatch_group_wait(grp_, DISPATCH_TIME_FOREVER);
        return out;   // every slot pinned; unpin in cmd-buffer completion
    }
private:
    const ExpertIndex& idx_;
    std::vector<int> fds_;                 // one fd per layer file
    ExpertCache& cache_;
    dispatch_queue_t q_; dispatch_semaphore_t sem_; dispatch_group_t grp_;
};

} // namespace aeromoe

2.5 Metal kernels — Q4 GEMV + fused SwiGLU + weighted combine
// expert_kernels.metal
#include <metal_stdlib>
using namespace metal;

// Q4 block layout (block_size = 32): [half scale][half min][16 packed bytes]
struct Q4Block { half scale; half zmin; uchar q[16]; };

inline float dequant_dot(device const Q4Block* row_blocks,
                         device const half* x, uint n_blocks) {
    float acc = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        device const Q4Block& blk = row_blocks[b];
        float s = float(blk.scale), m = float(blk.zmin);
        device const half* xb = x + b * 32;
        for (uint i = 0; i < 16; ++i) {
            uchar p = blk.q[i];
            acc += (s * float(p & 0xF)  + m) * float(xb[2*i]);
            acc += (s * float(p >> 4)   + m) * float(xb[2*i+1]);
        }
    }
    return acc;
}

// One threadgroup per output row group; simdgroup reduction elided for brevity.
// Fused: y_ff[j] = silu(gate_j · x) * (up_j · x)
kernel void expert_gateup_silu(
    device const Q4Block* gate_w   [[buffer(0)]],  // [d_ff][n_blocks]
    device const Q4Block* up_w     [[buffer(1)]],
    device const half*    x        [[buffer(2)]],  // [d_model]
    device half*          y_ff     [[buffer(3)]],  // [d_ff]
    constant uint&        d_model  [[buffer(4)]],
    constant uint&        d_ff     [[buffer(5)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= d_ff) return;
    uint nb = d_model / 32;
    float g = dequant_dot(gate_w + j * nb, x, nb);
    float u = dequant_dot(up_w   + j * nb, x, nb);
    float silu = g / (1.0f + exp(-g));
    y_ff[j] = half(silu * u);
}

// down projection + weighted accumulate into the shared MoE output.
// Called once per selected expert; routing weight applied here so no
// separate combine pass is needed.
kernel void expert_down_accum(
    device const Q4Block* down_w   [[buffer(0)]],  // [d_model][ff_blocks]
    device const half*    y_ff     [[buffer(1)]],
    device atomic_float*  moe_out  [[buffer(2)]],  // [d_model], fp32 accum
    constant uint&        d_model  [[buffer(3)]],
    constant uint&        d_ff     [[buffer(4)]],
    constant float&       route_w  [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= d_model) return;
    uint nb = d_ff / 32;
    float v = dequant_dot(down_w + i * nb, y_ff, nb);
    atomic_fetch_add_explicit(&moe_out[i], route_w * v,
                              memory_order_relaxed);
}


Production note: replace the scalar inner loop with simdgroup cooperative loads and 4-block unrolling; the structure (fused SiLU, weight-applied accumulate, fp32 accumulator) is the part that matters.

2.6 The per-layer decode loop with I/O overlap
// decode_layer.mm — one MoE layer for one decode token
#include "io_planner.mm"

void DecodeLayer::run(uint32_t layer, id<MTLCommandQueue> queue,
                      LayerWeights& W, TokenState& st, IOPlanner& io,
                      ExpertCache& cache) {
    // ---- Command buffer A: attention + router (backbone, resident) ----
    id<MTLCommandBuffer> cbA = [queue commandBuffer];
    encode_rmsnorm(cbA, st.x, W.attn_norm);
    encode_attention(cbA, st, W);            // QKV, RoPE, paged-KV decode attn
    encode_rmsnorm(cbA, st.x, W.ffn_norm);
    encode_router_logits(cbA, st.x, W.router, st.router_logits); // shared buf
    [cbA commit];
    [cbA waitUntilCompleted];                // unavoidable sync: need routing

    // ---- CPU: top-k from tiny logits vector (n_experts floats) ----
    auto [ids, weights] = topk_cpu(st.router_logits, manifest_.top_k);

    // ---- Overlap: kick SSD reads, run shared expert meanwhile ----
    id<MTLCommandBuffer> cbB = [queue commandBuffer];
    if (W.has_shared_expert)
        encode_shared_expert(cbB, st.x, W.shared, st.moe_out /*fp32 accum*/);
    [cbB commit];                            // runs while pread() proceeds

    std::vector<Slot*> slots = io.fetch(layer, ids, st.token_idx); // blocks

    // ---- Command buffer C: routed experts ----
    id<MTLCommandBuffer> cbC = [queue commandBuffer];
    for (size_t i = 0; i < slots.size(); ++i) {
        const ExpertRecord& rec = index_.at(layer, ids[i]);
        // gate/up live at offset 0, down at gate_up_bytes, inside one slot
        encode_expert_gateup_silu(cbC, slots[i]->buffer, 0, st.x, st.y_ff);
        encode_expert_down_accum (cbC, slots[i]->buffer, rec.gate_up_bytes,
                                  st.y_ff, st.moe_out, weights[i]);
    }
    encode_residual_add(cbC, st.x, st.moe_out);   // fp32→fp16 + residual

    // Unpin slots only after the GPU is done touching them.
    [cbC addCompletedHandler:^(id<MTLCommandBuffer>) {
        for (Slot* s : slots) cache.unpin(s);
    }];
    [cbC commit];
    [cbC waitUntilCompleted];   // v1: sync per layer; v2: pipeline layers
}

2.7 Top-level token loop
int32_t Engine::decode_one_token(TokenState& st) {
    st.x = embed(st.token_id);                      // resident embedding
    for (uint32_t l = 0; l < manifest_.n_layers; ++l)
        layers_[l].run(l, queue_, weights_[l], st, io_, cache_);
    if (st.token_idx % 64 == 0) cache_.decay();     // LFU aging
    return sample(final_norm_lmhead(st.x), st.sampler);
}

Part 3: Notes on what's deliberately omitted (v2 work)
Chunked prefill: reuse IOPlanner::fetch with deduplicated (layer, expert) sets across the 32-token chunk — the cache/pin machinery above already supports it.
Layer pipelining: replace waitUntilCompleted in cbC with events so layer l+1's attention overlaps layer l's expert compute; requires double-buffered TokenState.
Memory-pressure handler: a DISPATCH_SOURCE_TYPE_MEMORYPRESSURE source that shrinks slot count and enables F_NOCACHE on the layer fds.
Error paths: the abort() in the pread callback needs a real cancellation path.

If you'd like this as a downloadable .md file or a PDF, say the word. Otherwise, the natural next step is the converter (safetensors → .aeromoe packer) — that's where the manifest values and ExpertRecord layout get pinned down for the actual Qwen 3.6 35B-A3B checkpoint.




Yes — a Qwen-specific kernel set is unavoidable

TurboFieldfare had to hand-write Gemma-specific kernels because generic kernels don't capture the model family's architectural quirks (Gemma's logit soft-capping, pre/post-norm sandwich, GeGLU). Qwen 3-family MoE has its own set of quirks that make stock kernels either wrong or slow:

Qwen 3-family quirk	Why a generic kernel fails
QK-Norm: per-head RMSNorm applied to Q and K before RoPE	Not present in Llama/Gemma kernels; skipping it produces garbage output
GQA with a specific n_heads : n_kv_heads ratio	Attention kernel must broadcast KV heads; ratio comes from the manifest
Softmax router with renormalized top-k (norm_topk_prob=true)	Routing weights must be re-softmaxed over only the selected k — using raw softmax weights changes numerics
No shared expert in Qwen3 MoE (unlike Qwen2-MoE / DeepSeek)	Changes your overlap plan: cbB in the earlier design has nothing to run — overlap the next layer's norm or KV quantization instead
RMSNorm (with +eps inside sqrt), SwiGLU experts	SwiGLU you already have; RMSNorm epsilon placement must match exactly

⚠️ Verify each of these against the actual "Qwen 3.6 35B-A3B" checkpoint config at conversion time — fine-tuned/uncensored variants occasionally ship altered configs. The converter should hard-fail on unknown fields, exactly like TurboFieldfare's Gemma validator.

Below are the four kernels you must rewrite. Everything else (Q4 GEMV, SwiGLU expert FFN, residual add) carries over from the previous document.

1. Fused RMSNorm (Qwen epsilon placement)
// rmsnorm.metal — y = x / sqrt(mean(x^2) + eps) * w
// One threadgroup per vector; d_model up to 8192.
#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm_f16(
    device const half* x       [[buffer(0)]],
    device const half* w       [[buffer(1)]],
    device half*       y       [[buffer(2)]],
    constant uint&     n       [[buffer(3)]],
    constant float&    eps     [[buffer(4)]],   // from manifest (e.g. 1e-6)
    uint tid  [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]])
{
    threadgroup float partial[32];

    // 1. sum of squares, fp32 accumulation (critical for Q4 backbones)
    float acc = 0.0f;
    for (uint i = tid; i < n; i += tptg) {
        float v = float(x[i]);
        acc += v * v;
    }
    acc = simd_sum(acc);
    if ((tid & 31) == 0) partial[tid >> 5] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        float s = (tid < (tptg + 31) / 32) ? partial[tid] : 0.0f;
        s = simd_sum(s);
        if (tid == 0) partial[0] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. normalize — Qwen: rsqrt(mean + eps), weight multiply (no +1 offset,
    //    unlike Gemma's (1 + w) — this is exactly the kind of trap that
    //    forces per-family kernels)
    float inv = rsqrt(partial[0] / float(n) + eps);
    for (uint i = tid; i < n; i += tptg)
        y[i] = half(float(x[i]) * inv * float(w[i]));
}

2. Fused QK-Norm + RoPE (the Qwen3 signature kernel)

This is the kernel that does not exist in Llama/Gemma engines. Qwen3 applies a per-head RMSNorm to Q and K after projection, before RoPE. Fusing norm + rotation into one pass avoids two extra round trips through unified memory — on bandwidth-starved 8 GB machines this matters.

// qknorm_rope.metal
// Input:  q [n_heads * head_dim], k [n_kv_heads * head_dim]  (post-GEMV)
// Output: in-place normalized + rotated.
// Grid: one threadgroup per head (Q and K dispatched separately or via
// head_offset trick; shown here as one kernel handling both).
#include <metal_stdlib>
using namespace metal;

kernel void qk_norm_rope(
    device half*        vec        [[buffer(0)]],  // q or k, packed heads
    device const half*  norm_w     [[buffer(1)]],  // per-head-dim weights
    constant uint&      head_dim   [[buffer(2)]],
    constant float&     eps        [[buffer(3)]],
    constant float&     rope_theta [[buffer(4)]],  // e.g. 1e6 for Qwen3 long-ctx
    constant uint&      pos        [[buffer(5)]],  // absolute token position
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    device half* h = vec + head * head_dim;
    threadgroup float tg_sum;

    // --- per-head RMSNorm (QK-Norm). head_dim <= 128 → one simdgroup ---
    float v = (tid < head_dim) ? float(h[tid]) : 0.0f;
    float ss = simd_sum(v * v);
    if (tid == 0) tg_sum = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(tg_sum / float(head_dim) + eps);
    float normed = v * inv * float(norm_w[tid]);   // norm_w shared across heads

    // --- RoPE, interleaved-pair convention: pairs (2i, 2i+1) ---
    // NOTE: Qwen uses the "half-split" (GPT-NeoX) layout in HF:
    // pair = (i, i + head_dim/2). Match the converter's layout choice!
    uint half_d = head_dim / 2;
    threadgroup float tmp[128];
    tmp[tid] = normed;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < head_dim) {
        uint  i    = (tid < half_d) ? tid : tid - half_d;
        float freq = pow(rope_theta, -2.0f * float(i) / float(head_dim));
        float ang  = float(pos) * freq;
        float c = cos(ang), s = sin(ang);
        float x0 = tmp[i], x1 = tmp[i + half_d];
        h[tid] = half((tid < half_d) ? x0 * c - x1 * s
                                     : x1 * c + x0 * s);
    }
}


Trap to respect: HuggingFace Qwen checkpoints use the NeoX half-split RoPE pairing, while many GGUF-derived kernels use interleaved pairs. Your .aeromoe converter must pick one convention and the kernel must match it — this is the #1 source of "model loads but outputs gibberish" bugs when porting model families.

3. GQA paged-attention decode kernel

Generic single-query attention exists in many engines, but it must be specialized for: GQA broadcast ratio, your paged-KV block layout, and quantized KV blocks.

// gqa_decode_attn.metal — single-token decode, one threadgroup per Q head.
#include <metal_stdlib>
using namespace metal;

struct KVPageTable {            // matches engine's paged allocator
    constant uint* page_ids;    // logical block -> physical page
    // pages are [page_size][n_kv_heads][head_dim] half (or Q8 in v2)
};

kernel void gqa_decode_attention(
    device const half*  q          [[buffer(0)]],  // [n_heads*head_dim], post-RoPE
    device const half*  kv_pool    [[buffer(1)]],  // physical page pool
    constant uint*      page_ids   [[buffer(2)]],
    device half*        out        [[buffer(3)]],  // [n_heads*head_dim]
    constant uint&      seq_len    [[buffer(4)]],
    constant uint&      page_size  [[buffer(5)]],  // tokens per page, e.g. 64
    constant uint&      n_kv_heads [[buffer(6)]],
    constant uint&      gqa_ratio  [[buffer(7)]],  // n_heads / n_kv_heads
    constant uint&      head_dim   [[buffer(8)]],
    constant uint&      k_pool_off [[buffer(9)]],  // v pool = k pool + offset
    uint h   [[threadgroup_position_in_grid]],     // query head
    uint tid [[thread_position_in_threadgroup]],
    uint tptg[[threads_per_threadgroup]])
{
    const uint kvh   = h / gqa_ratio;              // GQA broadcast
    const float scale = rsqrt(float(head_dim));
    device const half* qh = q + h * head_dim;

    // Online softmax (flash-style, single pass over history)
    float m = -INFINITY, l = 0.0f;
    float acc[128];                                 // head_dim <= 128
    for (uint d = 0; d < head_dim; ++d) acc[d] = 0.0f;

    // Each thread strides over history positions
    for (uint t = tid; t < seq_len; t += tptg) {
        uint page = page_ids[t / page_size];
        uint slot = t % page_size;
        device const half* kvec = kv_pool
            + (size_t(page) * page_size + slot) * n_kv_heads * head_dim
            + kvh * head_dim;
        device const half* vvec = kvec + k_pool_off;

        float s = 0.0f;
        for (uint d = 0; d < head_dim; ++d)
            s += float(qh[d]) * float(kvec[d]);
        s *= scale;

        float m_new = max(m, s);
        float corr  = exp(m - m_new);
        float p     = exp(s - m_new);
        l = l * corr + p;
        for (uint d = 0; d < head_dim; ++d)
            acc[d] = acc[d] * corr + p * float(vvec[d]);
        m = m_new;
    }

    // Cross-thread reduction of (m, l, acc) via threadgroup memory —
    // standard log-sum-exp merge, elided for brevity but REQUIRED:
    //   merged_m = max(m_i); merged_l = Σ l_i * exp(m_i - merged_m);
    //   merged_acc = Σ acc_i * exp(m_i - merged_m);
    // ... reduction code ...

    if (tid == 0)
        for (uint d = 0; d < head_dim; ++d)
            out[h * head_dim + d] = half(acc[d] / l);
}


Production version: vectorize the dot products with half4 loads, keep acc in fp32 registers per simdlane, and add the Q8 KV-block dequant path for older pages.

4. Router: softmax + top-k with renormalization

Small but numerically load-bearing. Qwen3 MoE with norm_topk_prob=true requires the selected weights be renormalized to sum to 1 over the top-k — do it where you already do top-k, on CPU (n_experts floats is tiny):

// router_topk.cpp — CPU side, matches HF Qwen3-MoE semantics
std::pair<std::vector<uint32_t>, std::vector<float>>
topk_renorm(const float* logits, uint32_t n_experts, uint32_t k) {
    // softmax over ALL experts first (fp32)
    float mx = *std::max_element(logits, logits + n_experts);
    std::vector<float> p(n_experts);
    float sum = 0.f;
    for (uint32_t i = 0; i < n_experts; ++i)
        sum += (p[i] = std::exp(logits[i] - mx));
    for (auto& v : p) v /= sum;

    // top-k by probability
    std::vector<uint32_t> ids(n_experts);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                      [&](uint32_t a, uint32_t b){ return p[a] > p[b]; });
    ids.resize(k);

    // renormalize over the selected k (norm_topk_prob=true)
    float ksum = 0.f;
    for (auto i : ids) ksum += p[i];
    std::vector<float> w(k);
    for (uint32_t j = 0; j < k; ++j) w[j] = p[ids[j]] / ksum;
    return {ids, w};
}


If a variant checkpoint sets norm_topk_prob=false, skip the last step — another reason the converter must record this flag in manifest.json rather than assuming.

What you do not need to rewrite
Q4 GEMV / SwiGLU expert kernels from the previous doc — Qwen3 experts are standard SwiGLU, only shapes change (read from manifest).
Residual add, embedding lookup, LM head — generic.
Sampling — generic.
Adjusted overlap plan (no shared expert)

Since Qwen3 MoE has no shared expert, command buffer B from the earlier design is empty. Replace the overlap with: while pread()s are in flight for layer l's experts, encode and commit layer l's KV write-back + old-page KV quantization, and precompute nothing speculative. That keeps the GPU busy during SSD latency without violating the no-prefetch rule.

Want the converter-side counterpart next — i.e., how safetensors → .aeromoe must repack Q/K norm weights, RoPE convention, and per-expert slices so these kernels bind directly?


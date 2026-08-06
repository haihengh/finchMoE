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

// engine_core.mm
// ─────────────────────────────────────────────────────────────────────────────
// EngineCore implementation (Objective-C++ for Metal integration).
// ─────────────────────────────────────────────────────────────────────────────

#include "engine_core.h"
#include "aeromoe_format.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstring>
#include <cassert>
#include <algorithm>

namespace aeromoe {

// ── destructor ────────────────────────────────────────────────────────────────

EngineCore::~EngineCore() {
    shutdown();
}

void EngineCore::shutdown() {
    expert_cache_.reset();
    io_planner_.reset();
    // Dense buffers: release each MTLBuffer
    for (auto& [hash, ptr] : dense_buffers_) {
        if (ptr) CFRelease(ptr);
    }
    dense_buffers_.clear();
    ledger_.reset();
    // Release Metal objects
    if (cmd_queue_) { CFRelease(cmd_queue_); cmd_queue_ = nullptr; }
    if (device_)    { CFRelease(device_);    device_    = nullptr; }
    file_.close();
}

// ── init ─────────────────────────────────────────────────────────────────────

Status EngineCore::init(const std::string& model_path, const EngineConfig& cfg) {
    engine_cfg_ = cfg;

    // 1. Open .aeromoe file
    if (auto s = file_.open(model_path); !ok(s)) return s;

    // 2. Metal device + command queue
    if (auto s = setup_metal(); !ok(s)) return s;

    // 3. Memory ledger
    ledger_ = std::make_unique<MemoryLedger>(
        cfg.memory_budget_bytes, cfg.soft_budget_bytes);

    // 4. IOPlanner
    io_planner_ = std::make_unique<IOPlanner>(file_, cfg.io_threads);

    // 5. ExpertCache wired to IOPlanner's LoadFn
    expert_cache_ = std::make_unique<ExpertCache>(
        device_, *ledger_, io_planner_->make_load_fn());

    // 6. Load all dense backbone weights
    if (auto s = load_dense_weights(); !ok(s)) return s;

    if (cfg.verbose) print_stats();
    return Status::OK;
}

// ── Metal setup ───────────────────────────────────────────────────────────────

Status EngineCore::setup_metal() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        fprintf(stderr, "[engine_core] No Metal device found\n");
        return Status::MetalError;
    }
    device_    = (__bridge_retained void*)dev;
    cmd_queue_ = (__bridge_retained void*)[dev newCommandQueue];
    if (!cmd_queue_) {
        fprintf(stderr, "[engine_core] Failed to create MTLCommandQueue\n");
        return Status::MetalError;
    }
    fprintf(stderr, "[engine_core] Metal device: %s\n",
            [dev.name UTF8String]);
    return Status::OK;
}

// ── dense weight loading ──────────────────────────────────────────────────────

Status EngineCore::load_dense_weights() {
    fprintf(stderr, "[engine_core] Loading dense weights …\n");

    const auto& cfg = file_.config();
    const auto& all = file_.all_dense();
    backbone_.layers.resize(cfg.num_hidden_layers);

    size_t loaded = 0, total = all.size();
    for (const auto& rec : all) {
        void* buf = alloc_and_load_dense(rec);
        if (!buf) {
            fprintf(stderr, "[engine_core] Failed to load dense tensor (hash=%llx)\n",
                    (unsigned long long)rec.name_hash);
            return Status::IOError;
        }
        dense_buffers_[rec.name_hash] = buf;
        ++loaded;
    }

    fprintf(stderr, "[engine_core] Loaded %zu / %zu dense tensors  "
            "(%.2f GB used)\n",
            loaded, total, (double)ledger_->used() / 1e9);

    // Populate backbone_ pointers by resolving well-known names.
    // We use fnv1a_64 lookups for the hot-path helpers.
    using format::fnv1a_64;
    auto get = [&](const std::string& name) -> void* {
        auto it = dense_buffers_.find(fnv1a_64(name));
        if (it == dense_buffers_.end()) return nullptr;
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)it->second;
        return [b contents];
    };

    backbone_.embed_tokens = get("model.embed_tokens.weight");
    backbone_.final_norm   = get("model.norm.weight");
    backbone_.lm_head      = cfg.tie_word_embeddings
                             ? backbone_.embed_tokens
                             : get("lm_head.weight");

    for (uint32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& L  = backbone_.layers[i];
        auto  pf = "model.layers." + std::to_string(i) + ".";

        L.q_proj     = get(pf + "self_attn.q_proj.weight");
        L.k_proj     = get(pf + "self_attn.k_proj.weight");
        L.v_proj     = get(pf + "self_attn.v_proj.weight");
        L.o_proj     = get(pf + "self_attn.o_proj.weight");
        L.q_norm     = get(pf + "self_attn.q_norm.weight");
        L.k_norm     = get(pf + "self_attn.k_norm.weight");
        L.gate       = get(pf + "mlp.gate.weight");
        L.sh_gate    = get(pf + "mlp.shared_expert.gate_proj.weight");
        L.sh_up      = get(pf + "mlp.shared_expert.up_proj.weight");
        L.sh_down    = get(pf + "mlp.shared_expert.down_proj.weight");
        L.input_norm = get(pf + "input_layernorm.weight");
        L.post_norm  = get(pf + "post_attention_layernorm.weight");
    }
    return Status::OK;
}

// Allocate an MTLBuffer, reserve in ledger, pread weight data into it.
void* EngineCore::alloc_and_load_dense(const format::DenseRecord& rec) {
    // Reserve memory
    if (!ok(ledger_->reserve(rec.nbytes))) {
        fprintf(stderr, "[engine_core] Dense weight ledger overflow "
                "(%.2f MB requested, %.2f GB remaining)\n",
                (double)rec.nbytes / 1e6,
                (double)ledger_->remaining_hard() / 1e9);
        return nullptr;
    }

    // Allocate MTLBuffer in shared storage mode (zero-copy)
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    id<MTLBuffer> buf = [dev newBufferWithLength:rec.nbytes
                                        options:MTLResourceStorageModeShared];
    if (!buf) {
        ledger_->release(rec.nbytes);
        return nullptr;
    }

    // pread directly into the buffer's CPU-accessible memory
    void* dst = [buf contents];
    ssize_t n = file_.pread_exact(dst, rec.nbytes, (off_t)rec.offset);
    if (n != (ssize_t)rec.nbytes) {
        ledger_->release(rec.nbytes);
        // buf is ARC-managed if ObjC ARC is on; use CFRelease for explicit
        CFRelease((__bridge CFTypeRef)buf);
        return nullptr;
    }

    return (__bridge_retained void*)buf;
}

// ── dense TensorView ──────────────────────────────────────────────────────────

TensorView EngineCore::dense(const std::string& name) const {
    using format::fnv1a_64;
    auto it = dense_buffers_.find(fnv1a_64(name));
    if (it == dense_buffers_.end()) return {};

    const format::DenseRecord* rec = file_.dense(name);
    if (!rec) return {};

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)it->second;
    TensorView tv;
    tv.data  = [buf contents];
    tv.ndim  = rec->ndim;
    tv.dtype = rec->dtype;
    memcpy(tv.shape.data(), rec->shape, sizeof(rec->shape));
    return tv;
}

// ── expert slab API ───────────────────────────────────────────────────────────

ExpertView EngineCore::acquire_expert(uint32_t layer, uint32_t expert,
                                       Status* status_out) {
    ExpertKey key{layer, expert};
    const CacheEntry* entry = expert_cache_->acquire(key, status_out);
    if (!entry) return {};

    ++stats_.cache_misses;  // updated properly by ExpertCache internally
    return make_expert_view(entry, file_.config(), key);
}

void EngineCore::release_expert(uint32_t layer, uint32_t expert) {
    expert_cache_->release({layer, expert});
}

std::vector<ExpertView> EngineCore::acquire_layer_experts(
    uint32_t layer,
    const std::vector<uint32_t>& expert_ids)
{
    std::vector<ExpertView> views;
    views.reserve(expert_ids.size());
    for (uint32_t eid : expert_ids) {
        Status s;
        ExpertView ev = acquire_expert(layer, eid, &s);
        if (!ok(s)) {
            fprintf(stderr, "[engine_core] Failed to acquire expert (%u,%u): %s\n",
                    layer, eid, status_str(s));
        }
        views.push_back(ev);
    }
    return views;
}

void EngineCore::release_layer_experts(uint32_t layer,
                                        const std::vector<uint32_t>& expert_ids) {
    for (uint32_t eid : expert_ids) {
        release_expert(layer, eid);
    }
}

// ── ExpertView builder ────────────────────────────────────────────────────────

ExpertView EngineCore::make_expert_view(const CacheEntry* entry,
                                         const ModelConfig& cfg,
                                         ExpertKey key) const {
    // Slab layout: gate_proj | up_proj | down_proj (tightly packed)
    const uint8_t* base  = static_cast<const uint8_t*>(entry->data());
    size_t         esz   = dtype_size(cfg.storage_dtype);
    size_t         moe_h = cfg.moe_intermediate_size;
    size_t         hid   = cfg.hidden_size;

    // gate_proj: [moe_h, hid]
    size_t gate_bytes = moe_h * hid * esz;
    // up_proj:   [moe_h, hid]
    size_t up_bytes   = moe_h * hid * esz;
    // down_proj: [hid, moe_h]
    // size_t down_bytes = hid * moe_h * esz;  // (unused but for doc)

    ExpertView ev;
    ev.key = key;

    ev.gate_proj.data    = base;
    ev.gate_proj.dtype   = cfg.storage_dtype;
    ev.gate_proj.ndim    = 2;
    ev.gate_proj.shape   = {(int32_t)moe_h, (int32_t)hid, 0, 0};

    ev.up_proj.data      = base + gate_bytes;
    ev.up_proj.dtype     = cfg.storage_dtype;
    ev.up_proj.ndim      = 2;
    ev.up_proj.shape     = {(int32_t)moe_h, (int32_t)hid, 0, 0};

    ev.down_proj.data    = base + gate_bytes + up_bytes;
    ev.down_proj.dtype   = cfg.storage_dtype;
    ev.down_proj.ndim    = 2;
    ev.down_proj.shape   = {(int32_t)hid, (int32_t)moe_h, 0, 0};

    // Store the raw MTLBuffer pointer so ModelRunner can create page-aligned
    // no-copy sub-buffer views for Metal dispatch without an extra level of
    // indirection.  Non-owning cast: ExpertCache pin_count keeps the buffer
    // alive for the duration of the dispatch.
    ev.mtl_buffer = (__bridge void*)entry->buffer;

    return ev;
}

// ── dense_buf ─────────────────────────────────────────────────────────────────

void* EngineCore::dense_buf(const std::string& name) const {
    using format::fnv1a_64;
    auto it = dense_buffers_.find(fnv1a_64(name));
    // dense_buffers_ stores __bridge_retained void* (i.e. the MTLBuffer object
    // pointer with a +1 retain from when it was originally stored).
    return it != dense_buffers_.end() ? it->second : nullptr;
}

// ── diagnostics ───────────────────────────────────────────────────────────────

void EngineCore::print_stats(FILE* out) const {
    fprintf(out, "\n═══ AeroMoE Engine Stats ════════════════════════════\n");
    ledger_->print_stats(out);
    if (expert_cache_) expert_cache_->print_stats(out);
    if (io_planner_)   io_planner_->print_stats(out);
    fprintf(out,
        "[runtime] tokens=%llu  cache_hits=%llu  cache_misses=%llu\n",
        (unsigned long long)stats_.tokens_generated,
        (unsigned long long)stats_.cache_hits,
        (unsigned long long)stats_.cache_misses);
    fprintf(out, "═════════════════════════════════════════════════════\n\n");
}

void EngineCore::reset_stats() {
    stats_ = RuntimeStats{};
}

} // namespace aeromoe

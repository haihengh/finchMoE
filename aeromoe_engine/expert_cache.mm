// expert_cache.mm
// ─────────────────────────────────────────────────────────────────────────────
// ExpertCache implementation (Objective-C++ for Metal buffer management).
// ─────────────────────────────────────────────────────────────────────────────

#include "expert_cache.h"

#import <Metal/Metal.h>
#include <algorithm>
#include <cstring>

namespace aeromoe {

// ── CacheEntry::data() ────────────────────────────────────────────────────────

void* CacheEntry::data() const {
    if (!buffer) return nullptr;
    return [(__bridge id<MTLBuffer>)buffer contents];
}

// ── ExpertCache ───────────────────────────────────────────────────────────────

ExpertCache::ExpertCache(void* metal_device,
                         MemoryLedger& ledger,
                         LoadFn load_fn)
    : device_(metal_device)
    , ledger_(ledger)
    , load_fn_(std::move(load_fn))
{}

ExpertCache::~ExpertCache() {
    clear_unpinned();
}

// ── alloc / free ─────────────────────────────────────────────────────────────

MTLBufferPtr ExpertCache::alloc_buffer(size_t bytes) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    // MTLStorageModeShared: CPU + GPU share the same physical page — zero copy.
    id<MTLBuffer> buf = [dev newBufferWithLength:bytes
                                        options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[expert_cache] MTLBuffer allocation failed (%zu MB)\n",
                bytes >> 20);
        return nullptr;
    }
    return (__bridge_retained void*)buf;
}

void ExpertCache::free_entry(CacheEntry& e) {
    if (e.buffer) {
        // Release ObjC ownership acquired by __bridge_retained in alloc_buffer
        CFRelease(e.buffer);
        e.buffer = nullptr;
    }
    ledger_.release(e.bytes);
    e.bytes = 0;
}

// ── load_entry ────────────────────────────────────────────────────────────────

Status ExpertCache::load_entry(CacheEntry& e) {
    // e.buffer is already allocated and registered with ledger_ before this call.
    void* dst = [(__bridge id<MTLBuffer>)e.buffer contents];
    return load_fn_(e.key, dst, e.bytes);
}

// ── acquire ───────────────────────────────────────────────────────────────────

const CacheEntry* ExpertCache::acquire(ExpertKey key, Status* status_out) {
    auto set_status = [&](Status s) {
        if (status_out) *status_out = s;
    };

    // Fast path: already resident
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            CacheEntry& e = it->second;
            e.use_count++;
            e.last_tick = next_tick();
            e.pin_count++;
            hits_.fetch_add(1, std::memory_order_relaxed);
            set_status(Status::OK);
            return &e;
        }
    }

    // Miss — need to load
    misses_.fetch_add(1, std::memory_order_relaxed);

    // 1. Ask for expert slab size from load_fn
    //    We call load_fn with dst=nullptr to query size (size encoded in nbytes
    //    field of ExpertRecord; IOPlanner knows it). Here we pre-allocate with
    //    a sentinel size query — but in practice the IOPlanner passes the size
    //    as the nbytes parameter.  We pass SIZE_MAX as a "query" convention.
    //    For cleaner API, we get the size from the record stored in IOPlanner
    //    via the load_fn closure capturing it.
    //
    //    The load_fn is expected to:
    //      - If dst == nullptr && nbytes == SIZE_MAX: fill *reinterpret_cast<size_t*>(...)
    //        Actually simpler: we use a size_query_fn_ separately.
    //
    //    In practice, IOPlanner::make_load_fn captures the ExpertRecord which
    //    has nbytes, so we can query it with a size_t* overload. Since we only
    //    have the one LoadFn signature, we get size from load_fn(key, nullptr, 0)
    //    returning the number of bytes needed as a negative Status trick is messy.
    //
    //    Cleanest approach: IOPlanner registers a size_fn too.
    //    For now we call load_fn with nullptr/0 to get size back via a separate
    //    path — see IOPlanner::slab_bytes(). We do a two-stage load here.

    // The IOPlanner-provided load_fn captures slab sizes.
    // We ask it by calling with (key, nullptr, 0) — convention: returns
    // Status::OK and writes size into a thread_local.
    // See IOPlanner::make_load_fn for the implementation.
    // Here we rely on the returned Status encoding to carry the size.
    // Simple alternative used below: cache already knows slab_size from cfg.

    // ──── ACTUAL IMPLEMENTATION ────
    // We need the slab size. Compute it from the stored model config
    // (passed at construction via the load_fn closure) or just call
    // load_fn(key, nullptr, 0) which returns Status::OK and sets a
    // thread_local slab_bytes; the IOPlanner sets this convention.

    // For a clean implementation we store a size_fn alongside load_fn:
    // This is set up in EngineCore::build_cache() after IOPlanner is ready.
    // Temporarily: ask via a special call.
    size_t slab_bytes = 0;
    {
        // load_fn(key, nullptr, 0) → IOPlanner returns size in a TLS out-param
        Status qs = load_fn_(key, nullptr, 0);
        // IOPlanner sets thread_local g_last_slab_bytes on this path
        extern thread_local size_t g_last_slab_bytes;
        slab_bytes = g_last_slab_bytes;
        if (!ok(qs) && slab_bytes == 0) {
            set_status(Status::NotFound);
            return nullptr;
        }
    }

    // 2. Evict if needed (do outside lock for non-blocking I/O)
    size_t needed = ledger_.soft_evict_needed(slab_bytes);
    if (needed > 0) {
        evict_lfu(needed);
    }

    // 3. Reserve memory
    Status rs = ledger_.reserve(slab_bytes);
    if (!ok(rs)) {
        // Hard eviction round
        evict_lfu(ledger_.evict_needed(slab_bytes));
        rs = ledger_.reserve(slab_bytes);
        if (!ok(rs)) {
            set_status(Status::BudgetExceeded);
            fprintf(stderr, "[expert_cache] Budget exceeded for (%u,%u) — %.2f MB\n",
                    key.layer, key.expert, (double)slab_bytes / 1e6);
            return nullptr;
        }
    }

    // 4. Allocate Metal buffer
    MTLBufferPtr buf = alloc_buffer(slab_bytes);
    if (!buf) {
        ledger_.release(slab_bytes);
        set_status(Status::MetalError);
        return nullptr;
    }

    // 5. Load weights (pread, outside lock)
    CacheEntry tmp;
    tmp.key       = key;
    tmp.buffer    = buf;
    tmp.bytes     = slab_bytes;
    tmp.use_count = 1;
    tmp.last_tick = next_tick();
    tmp.pin_count = 1;

    Status ls = load_entry(tmp);
    if (!ok(ls)) {
        free_entry(tmp);
        set_status(ls);
        return nullptr;
    }

    // 6. Insert into map
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto [it, inserted] = map_.emplace(key, std::move(tmp));
        set_status(Status::OK);
        return &it->second;
    }
}

// ── release ───────────────────────────────────────────────────────────────────

void ExpertCache::release(ExpertKey key) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = map_.find(key);
    if (it != map_.end()) {
        it->second.pin_count = std::max(0, it->second.pin_count - 1);
    }
}

// ── eviction ──────────────────────────────────────────────────────────────────

size_t ExpertCache::evict_lfu(size_t bytes_needed) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (bytes_needed == 0) return 0;

    // Collect eviction candidates (not pinned)
    std::vector<ExpertKey> candidates;
    candidates.reserve(map_.size());
    for (auto& [k, e] : map_) {
        if (e.pin_count == 0) candidates.push_back(k);
    }

    // Sort: lowest use_count first; ties: oldest last_tick first
    std::sort(candidates.begin(), candidates.end(),
        [&](const ExpertKey& a, const ExpertKey& b) {
            const auto& ea = map_.at(a);
            const auto& eb = map_.at(b);
            if (ea.use_count != eb.use_count)
                return ea.use_count < eb.use_count;
            return ea.last_tick < eb.last_tick;
        });

    size_t freed = 0;
    for (const auto& k : candidates) {
        if (freed >= bytes_needed) break;
        auto it = map_.find(k);
        if (it == map_.end()) continue;
        freed += it->second.bytes;
        free_entry(it->second);
        map_.erase(it);
    }
    return freed;
}

void ExpertCache::evict_one(ExpertKey key) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = map_.find(key);
    if (it == map_.end()) return;
    if (it->second.pin_count > 0) return;  // pinned — skip
    free_entry(it->second);
    map_.erase(it);
}

void ExpertCache::clear_unpinned() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto it = map_.begin(); it != map_.end(); ) {
        if (it->second.pin_count == 0) {
            free_entry(it->second);
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}

// ── diagnostics ───────────────────────────────────────────────────────────────

size_t ExpertCache::resident_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return map_.size();
}

size_t ExpertCache::resident_bytes() const {
    std::lock_guard<std::mutex> lk(mutex_);
    size_t total = 0;
    for (auto& [k, e] : map_) total += e.bytes;
    return total;
}

uint64_t ExpertCache::total_hits()   const { return hits_.load(); }
uint64_t ExpertCache::total_misses() const { return misses_.load(); }

void ExpertCache::print_stats(FILE* out) const {
    uint64_t h = total_hits(), m = total_misses();
    double ratio = (h + m) > 0 ? 100.0 * h / (h + m) : 0.0;
    fprintf(out,
        "[expert_cache] resident=%zu  bytes=%.2f GB  "
        "hits=%llu  misses=%llu  hit_rate=%.1f%%\n",
        resident_count(),
        (double)resident_bytes() / 1e9,
        (unsigned long long)h,
        (unsigned long long)m,
        ratio);
}

// ── thread_local slab_bytes (set by IOPlanner, read by acquire) ───────────────
thread_local size_t g_last_slab_bytes = 0;

} // namespace aeromoe

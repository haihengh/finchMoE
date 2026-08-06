// expert_cache.h
// ─────────────────────────────────────────────────────────────────────────────
// ExpertCache — bounded LFU+recency eviction cache for MoE expert slabs.
//
// Each entry is one (layer, expert) slab stored in a MTLBuffer allocated with
// MTLStorageModeShared (zero-copy unified memory on Apple Silicon).
//
// Eviction policy: LFU with recency tie-breaking.
//   - Each entry carries a use_count and a last_used_tick (monotonic counter).
//   - When the MemoryLedger reports pressure, we evict the entry with the
//     lowest use_count; ties broken by oldest last_used_tick.
//   - We never evict an entry that is pinned (currently in active use by
//     a Metal encoder dispatch).
//
// Thread-safety:
//   - A single std::mutex guards the map and eviction logic.
//   - pread() I/O happens outside the lock (in IOPlanner).
//   - Metal buffer allocation/deallocation happens outside the lock
//     (the Objective-C runtime handles its own locking).
//
// Typical slab size for Qwen3.6-35B-A3B (bf16):
//   gate(768×4096) + up(768×4096) + down(4096×768) = 18,874,368 B ≈ 18 MB
// With a 4 GB budget and ~600 MB dense backbone, ~3.4 GB remains for experts
// → fits ~188 expert slabs in cache simultaneously (vs 128 total per layer).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"
#include "memory_ledger.h"

#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <functional>

// Metal buffer type is opaque void* — only .mm files bridge to real ObjC types.
using MTLBufferPtr = void*;

namespace finchmoe {

// ── CacheEntry ────────────────────────────────────────────────────────────────

struct CacheEntry {
    ExpertKey    key;
    MTLBufferPtr buffer   = nullptr; // MTLBuffer* (shared unified memory)
    size_t       bytes    = 0;
    uint64_t     use_count= 0;
    uint64_t     last_tick= 0;
    int32_t      pin_count= 0;       // >0: in active dispatch, must not evict

    // Pointer to the raw weight data (buffer.contents on Apple Silicon)
    void* data() const;              // defined in expert_cache.mm
};

// ── ExpertCache ───────────────────────────────────────────────────────────────

class ExpertCache {
public:
    // load_fn is called by the cache when a slab is not resident.
    // It must fill `dst` (size `nbytes`) with the expert's weight data.
    // Called WITHOUT the cache lock held.
    using LoadFn = std::function<Status(ExpertKey, void* dst, size_t nbytes)>;

    // Called with the ObjC device pointer so entries can allocate MTLBuffers.
    // Pass your MTLDevice* cast to void*.
    explicit ExpertCache(void*        metal_device,
                         MemoryLedger& ledger,
                         LoadFn        load_fn);

    ~ExpertCache();

    // ── main API ──────────────────────────────────────────────────────────────

    // Acquire a slab: loads from disk if not resident, pins it, returns entry.
    // Caller must call release() when the Metal encoder is done with it.
    // Returns nullptr on error (check status_out).
    const CacheEntry* acquire(ExpertKey key, Status* status_out = nullptr);

    // Release a previously acquired entry (decrements pin_count).
    void release(ExpertKey key);

    // Pre-warm: load a list of experts asynchronously.
    // Returns immediately; loads happen in the IOPlanner's thread pool.
    void prefetch(const std::vector<ExpertKey>& keys);

    // ── eviction ──────────────────────────────────────────────────────────────

    // Evict enough entries to free at least `bytes_needed`.
    // Returns total bytes freed (may be 0 if all entries are pinned).
    size_t evict_lfu(size_t bytes_needed);

    // Evict a single specific entry. Fails silently if pinned or not present.
    void evict_one(ExpertKey key);

    // Drop everything that is not pinned (e.g. between inference sessions).
    void clear_unpinned();

    // ── diagnostics ──────────────────────────────────────────────────────────

    size_t   resident_count()  const;
    size_t   resident_bytes()  const;
    uint64_t total_hits()      const;
    uint64_t total_misses()    const;

    void print_stats(FILE* out = stderr) const;

private:
    // ── internal helpers ──────────────────────────────────────────────────────

    // Allocate a new MTLBuffer of `bytes` in shared storage mode.
    // Returns nullptr on allocation failure.
    MTLBufferPtr alloc_buffer(size_t bytes);

    // Free a MTLBuffer and release its memory from the ledger.
    void free_entry(CacheEntry& e);

    // Load data into an already-allocated entry.
    Status load_entry(CacheEntry& e);

    uint64_t next_tick() {
        return tick_.fetch_add(1, std::memory_order_relaxed);
    }

    // ── state ─────────────────────────────────────────────────────────────────

    void*         device_;   // MTLDevice* (opaque)
    MemoryLedger& ledger_;
    LoadFn        load_fn_;

    mutable std::mutex mutex_;

    using KeyHash = struct {
        size_t operator()(ExpertKey k) const noexcept {
            return std::hash<uint64_t>{}(((uint64_t)k.layer << 32) | k.expert);
        }
    };
    std::unordered_map<ExpertKey, CacheEntry, KeyHash> map_;

    std::atomic<uint64_t> tick_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
};

} // namespace finchmoe

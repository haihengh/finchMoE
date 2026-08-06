// kv_cache.h
// ─────────────────────────────────────────────────────────────────────────────
// Paged KV-cache manager for FinchMoE.
//
// Design:
//   Physical storage is a single MTLBuffer pair (k_pool, v_pool) split into
//   fixed-size "blocks". Each block holds PAGE_SIZE tokens for ALL KV heads
//   across ALL layers (one contiguous slab per block).
//
//   A page table maps (sequence_id, page_index) → block_index.
//   The Metal attention kernels receive the page table as a GPU buffer and
//   do their own address arithmetic.
//
// Memory layout per block:
//   k_pool[block, layer, slot, kv_head, head_dim]  bf16
//   v_pool[block, layer, slot, kv_head, head_dim]  bf16
//
//   where slot ∈ [0, PAGE_SIZE), layer ∈ [0, num_layers)
//
//   Stride:
//     element     = dtype_size                         (2 B for bf16)
//     head        = head_dim   * element               (128 B)
//     slot        = n_kv_heads * head                  (512 B)
//     layer       = PAGE_SIZE  * slot                  (8 KB)
//     block       = num_layers * layer                 (752 KB for Qwen3.6)
//
// The page table itself is a CPU-side vector<uint32_t> mirrored to a
// MTLBuffer so the GPU can read it.
//
// Thread-safety: all public methods are guarded by a single mutex.
//   Callers hold pages for the duration of a forward pass; release on eviction
//   or session end.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "finchmoe_types.h"
#include "memory_ledger.h"

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <cassert>

#ifdef __OBJC__
@protocol MTLBuffer;
@protocol MTLDevice;
using MTLBufferPtr_t = id<MTLBuffer>;
using MTLDevicePtr   = id<MTLDevice>;
#else
using MTLBufferPtr_t = void*;
using MTLDevicePtr   = void*;
#endif

namespace finchmoe {

// ── constants ─────────────────────────────────────────────────────────────────

constexpr uint32_t KV_PAGE_SIZE       = 16;    // tokens per block
constexpr uint32_t KV_MAX_SEQUENCES   = 8;     // concurrent sequences
constexpr uint32_t KV_MAX_PAGES_PER_SEQ = 8192; // 8192 * 16 = 131 072 tokens max

// ── KVCacheConfig ─────────────────────────────────────────────────────────────

struct KVCacheConfig {
    uint32_t num_layers;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t num_blocks;   // total physical blocks (set from memory budget)
    DType    dtype = DType::BF16;

    // Derived: bytes per block for K (or V) tensor
    size_t block_bytes_kv() const {
        return (size_t)KV_PAGE_SIZE * num_layers * n_kv_heads * head_dim
               * dtype_size(dtype);
    }
    // Total bytes for both K and V pools
    size_t total_bytes() const {
        return 2ULL * num_blocks * block_bytes_kv();
    }
    // Estimate how many blocks fit in `budget` bytes
    static uint32_t blocks_for_budget(size_t budget,
                                       uint32_t num_layers,
                                       uint32_t n_kv_heads,
                                       uint32_t head_dim,
                                       DType    dtype = DType::BF16) {
        size_t per_block = 2ULL * KV_PAGE_SIZE * num_layers
                           * n_kv_heads * head_dim * dtype_size(dtype);
        return (per_block > 0) ? (uint32_t)(budget / per_block) : 0;
    }
};

// ── SequenceState ─────────────────────────────────────────────────────────────

struct SequenceState {
    uint32_t              seq_id      = 0;
    uint32_t              token_count = 0;  // total tokens written so far
    std::vector<uint32_t> page_list;        // physical block indices (ordered)
};

// ── KVCacheManager ────────────────────────────────────────────────────────────

class KVCacheManager {
public:
    KVCacheManager() = default;
    ~KVCacheManager();

    // Initialize: allocate GPU buffers from device, register with ledger.
    Status init(MTLDevicePtr device, MemoryLedger& ledger,
                const KVCacheConfig& cfg);

    // ── sequence lifecycle ────────────────────────────────────────────────────

    // Allocate a new sequence slot. Returns seq_id or UINT32_MAX on failure.
    uint32_t new_sequence();

    // Ensure enough pages are allocated for `token_count` tokens in sequence.
    // Allocates new blocks as needed. Returns false if out of blocks.
    bool ensure_capacity(uint32_t seq_id, uint32_t token_count);

    // Free all pages for a sequence (e.g. after generation completes).
    void free_sequence(uint32_t seq_id);

    // ── GPU buffer access ─────────────────────────────────────────────────────

    MTLBufferPtr_t k_pool()      const { return k_pool_; }
    MTLBufferPtr_t v_pool()      const { return v_pool_; }
    MTLBufferPtr_t page_table_buf() const { return page_table_buf_; }

    // Flush the CPU page table to the GPU buffer.
    // Must be called before the Metal command buffer that reads the page table.
    void flush_page_table();

    // ── accessors ─────────────────────────────────────────────────────────────

    const KVCacheConfig& config() const { return cfg_; }
    uint32_t token_count(uint32_t seq_id) const;
    uint32_t free_block_count() const;
    uint32_t total_block_count() const { return cfg_.num_blocks; }

    // Max pages per sequence constant (for buffer sizing on GPU)
    uint32_t max_pages() const { return KV_MAX_PAGES_PER_SEQ; }

    void print_stats(FILE* out = stderr) const;

private:
    uint32_t alloc_block();            // pop from free list
    void     free_block(uint32_t blk); // push to free list

    KVCacheConfig cfg_{};

    MTLBufferPtr_t k_pool_         = nullptr;
    MTLBufferPtr_t v_pool_         = nullptr;
    MTLBufferPtr_t page_table_buf_ = nullptr;

    // CPU-side page table: [seq_id * KV_MAX_PAGES_PER_SEQ + page_idx] = block_idx
    // UINT32_MAX = unmapped
    std::vector<uint32_t> page_table_cpu_;

    std::unordered_map<uint32_t, SequenceState> sequences_;
    std::vector<uint32_t>                       free_blocks_;

    mutable std::mutex mutex_;
    uint32_t next_seq_id_ = 1;
};

} // namespace finchmoe

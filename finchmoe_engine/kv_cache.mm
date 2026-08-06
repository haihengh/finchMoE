// kv_cache.mm
// ─────────────────────────────────────────────────────────────────────────────

#include "kv_cache.h"

#import <Metal/Metal.h>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace finchmoe {

KVCacheManager::~KVCacheManager() {
    if (k_pool_)         CFRelease(k_pool_);
    if (v_pool_)         CFRelease(v_pool_);
    if (page_table_buf_) CFRelease(page_table_buf_);
}

Status KVCacheManager::init(MTLDevicePtr device, MemoryLedger& ledger,
                             const KVCacheConfig& cfg) {
    cfg_ = cfg;

    size_t total = cfg.total_bytes();
    // + page table buffer
    size_t pt_bytes = (size_t)KV_MAX_SEQUENCES * KV_MAX_PAGES_PER_SEQ
                      * sizeof(uint32_t);
    size_t grand_total = total + pt_bytes;

    if (!ok(ledger.reserve(grand_total))) {
        fprintf(stderr, "[kv_cache] Cannot allocate %.2f GB for KV cache\n",
                (double)grand_total / 1e9);
        return Status::BudgetExceeded;
    }

    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    size_t pool_bytes = (size_t)cfg.num_blocks * cfg.block_bytes_kv();

    auto alloc = [&](size_t n) -> MTLBufferPtr_t {
        id<MTLBuffer> b = [dev newBufferWithLength:n
                                          options:MTLResourceStorageModeShared];
        return b ? (__bridge_retained void*)b : nullptr;
    };

    k_pool_ = alloc(pool_bytes);
    v_pool_ = alloc(pool_bytes);
    if (!k_pool_ || !v_pool_) return Status::MetalError;

    // Zero-initialize K and V pools (important for first-block reads)
    memset([(__bridge id<MTLBuffer>)k_pool_ contents], 0, pool_bytes);
    memset([(__bridge id<MTLBuffer>)v_pool_ contents], 0, pool_bytes);

    // Page table
    page_table_cpu_.assign(
        KV_MAX_SEQUENCES * KV_MAX_PAGES_PER_SEQ, UINT32_MAX);
    page_table_buf_ = alloc(pt_bytes);
    if (!page_table_buf_) return Status::MetalError;
    flush_page_table();

    // Free block list: all blocks available
    free_blocks_.resize(cfg.num_blocks);
    std::iota(free_blocks_.begin(), free_blocks_.end(), 0u);

    fprintf(stderr, "[kv_cache] Initialized: %u blocks × %.1f KB = %.2f GB\n",
            cfg.num_blocks,
            (double)cfg.block_bytes_kv() / 1024.0,
            (double)grand_total / 1e9);
    return Status::OK;
}

// ── sequence lifecycle ────────────────────────────────────────────────────────

uint32_t KVCacheManager::new_sequence() {
    std::lock_guard<std::mutex> lk(mutex_);
    uint32_t sid = next_seq_id_++;
    sequences_[sid] = SequenceState{sid, 0, {}};
    return sid;
}

bool KVCacheManager::ensure_capacity(uint32_t seq_id, uint32_t token_count) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return false;

    SequenceState& seq = it->second;
    uint32_t pages_needed = (token_count + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;

    while ((uint32_t)seq.page_list.size() < pages_needed) {
        if (free_blocks_.empty()) {
            fprintf(stderr, "[kv_cache] Out of blocks for seq %u\n", seq_id);
            return false;
        }
        uint32_t blk = alloc_block();
        seq.page_list.push_back(blk);
        // Update CPU page table
        uint32_t page_idx = (uint32_t)seq.page_list.size() - 1;
        // Map seq_id slot: use (seq_id - 1) % KV_MAX_SEQUENCES as row
        uint32_t row = (seq_id - 1) % KV_MAX_SEQUENCES;
        page_table_cpu_[row * KV_MAX_PAGES_PER_SEQ + page_idx] = blk;
    }
    seq.token_count = token_count;
    return true;
}

void KVCacheManager::free_sequence(uint32_t seq_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = sequences_.find(seq_id);
    if (it == sequences_.end()) return;

    SequenceState& seq = it->second;
    // Clear page table entries
    uint32_t row = (seq_id - 1) % KV_MAX_SEQUENCES;
    for (uint32_t pi = 0; pi < (uint32_t)seq.page_list.size(); ++pi) {
        page_table_cpu_[row * KV_MAX_PAGES_PER_SEQ + pi] = UINT32_MAX;
        free_block(seq.page_list[pi]);
    }
    sequences_.erase(it);
}

// ── GPU page table sync ───────────────────────────────────────────────────────

void KVCacheManager::flush_page_table() {
    void* dst = [(__bridge id<MTLBuffer>)page_table_buf_ contents];
    size_t bytes = page_table_cpu_.size() * sizeof(uint32_t);
    memcpy(dst, page_table_cpu_.data(), bytes);
}

// ── accessors ─────────────────────────────────────────────────────────────────

uint32_t KVCacheManager::token_count(uint32_t seq_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = sequences_.find(seq_id);
    return it != sequences_.end() ? it->second.token_count : 0;
}

uint32_t KVCacheManager::free_block_count() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return (uint32_t)free_blocks_.size();
}

void KVCacheManager::print_stats(FILE* out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    fprintf(out,
        "[kv_cache] blocks: %u used / %u total  sequences: %zu\n",
        cfg_.num_blocks - (uint32_t)free_blocks_.size(),
        cfg_.num_blocks,
        sequences_.size());
}

// ── internal helpers ──────────────────────────────────────────────────────────

uint32_t KVCacheManager::alloc_block() {
    // Caller holds mutex_
    uint32_t blk = free_blocks_.back();
    free_blocks_.pop_back();
    return blk;
}

void KVCacheManager::free_block(uint32_t blk) {
    free_blocks_.push_back(blk);
}

} // namespace finchmoe

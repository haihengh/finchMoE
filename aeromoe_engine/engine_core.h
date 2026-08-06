// engine_core.h
// ─────────────────────────────────────────────────────────────────────────────
// EngineCore — top-level coordinator that wires the file, ledger, cache,
// and I/O planner together and exposes the API used by the inference loop.
//
// Responsibilities:
//   - Open the .aeromoe file and parse its header + indexes.
//   - Allocate and populate all dense (backbone) tensors at startup.
//   - Serve expert slabs on demand via the ExpertCache.
//   - Expose TensorViews to the Metal dispatch layer.
//   - Maintain aggregate RuntimeStats.
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include "aeromoe_types.h"
#include "aeromoe_format.h"
#include "memory_ledger.h"
#include "expert_cache.h"
#include "io_planner.h"

#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <functional>

// Forward-declare Metal types (ObjC++ only in engine_core.mm)
#ifdef __OBJC__
@protocol MTLDevice;
@protocol MTLBuffer;
@protocol MTLCommandQueue;
using MTLDevicePtr       = id<MTLDevice>;
using MTLCommandQueuePtr = id<MTLCommandQueue>;
#else
using MTLDevicePtr       = void*;
using MTLCommandQueuePtr = void*;
#endif

namespace aeromoe {

// ── EngineConfig — user-visible tuning knobs ──────────────────────────────────

struct EngineConfig {
    size_t memory_budget_bytes  = BUDGET_4GB;
    size_t soft_budget_bytes    = BUDGET_3_5GB;
    int    io_threads           = 4;         // parallel pread() workers
    bool   verbose              = false;     // extra diagnostics
};

// ── DenseBuffers — all resident backbone weights ──────────────────────────────

struct DenseBuffers {
    // Embedding
    void* embed_tokens   = nullptr;  // [vocab_size, hidden_size]

    // Per-layer — indexed by layer id
    struct Layer {
        // Attention
        void* q_proj     = nullptr;  // [n_heads * head_dim, hidden_size]
        void* k_proj     = nullptr;  // [n_kv_heads * head_dim, hidden_size]
        void* v_proj     = nullptr;  // [n_kv_heads * head_dim, hidden_size]
        void* o_proj     = nullptr;  // [hidden_size, n_heads * head_dim]
        void* q_norm     = nullptr;  // [head_dim]   QK-Norm
        void* k_norm     = nullptr;  // [head_dim]
        // MoE router
        void* gate       = nullptr;  // [num_experts, hidden_size]
        // Shared expert (always active)
        void* sh_gate    = nullptr;  // [inter_size, hidden_size]
        void* sh_up      = nullptr;  // [inter_size, hidden_size]
        void* sh_down    = nullptr;  // [hidden_size, inter_size]
        // Norms
        void* input_norm = nullptr;  // [hidden_size]
        void* post_norm  = nullptr;  // [hidden_size]
    };
    std::vector<Layer> layers;

    // Final norm + LM head
    void* final_norm = nullptr;  // [hidden_size]
    void* lm_head    = nullptr;  // [vocab_size, hidden_size] (or ties embed_tokens)
};

// ── ExpertView — what the Metal kernels see for one expert ────────────────────

struct ExpertView {
    TensorView gate_proj;       // [moe_inter, hidden]
    TensorView up_proj;         // [moe_inter, hidden]
    TensorView down_proj;       // [hidden, moe_inter]
    ExpertKey  key;
    void*      mtl_buffer = nullptr; // opaque id<MTLBuffer>* for the whole slab.
                                     // Non-owning; ExpertCache pin keeps it alive.
                                     // Required by ModelRunner to create sub-range
                                     // no-copy buffer views for Metal dispatch.
};

// ── EngineCore ────────────────────────────────────────────────────────────────

class EngineCore {
public:
    EngineCore() = default;
    ~EngineCore();

    // Non-copyable
    EngineCore(const EngineCore&) = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    // ── lifecycle ────────────────────────────────────────────────────────────

    // Initialize: open file, allocate device, load all dense weights.
    // Returns Status::OK on success.
    Status init(const std::string& model_path, const EngineConfig& cfg = {});

    // Free everything. Called automatically by destructor.
    void shutdown();

    // ── dense weight access ──────────────────────────────────────────────────

    // Get a TensorView into a dense weight by name.
    // Returns an empty view (data==nullptr) if not found.
    TensorView dense(const std::string& name) const;

    // Typed helpers for the inference loop (avoids string lookups in hot path)
    const DenseBuffers& backbone() const { return backbone_; }
    const ModelConfig&  config()   const { return file_.config(); }

    // ── expert slab API ──────────────────────────────────────────────────────

    // Acquire an expert for Metal dispatch. Pin is held until release_expert().
    // Thread-safe. May block briefly on first access (I/O load).
    ExpertView acquire_expert(uint32_t layer, uint32_t expert,
                              Status* status_out = nullptr);

    // Release after the Metal command buffer has committed.
    void release_expert(uint32_t layer, uint32_t expert);

    // Acquire a full set of active experts for one layer in parallel.
    // `expert_ids` are the top-k indices from the router.
    std::vector<ExpertView> acquire_layer_experts(
        uint32_t layer,
        const std::vector<uint32_t>& expert_ids);

    void release_layer_experts(uint32_t layer,
                                const std::vector<uint32_t>& expert_ids);

    // ── device / queue ────────────────────────────────────────────────────────

    MTLDevicePtr       device()        const { return device_; }
    MTLCommandQueuePtr command_queue() const { return cmd_queue_; }

    // ── diagnostics ──────────────────────────────────────────────────────────

    const RuntimeStats& stats() const { return stats_; }
    void                print_stats(FILE* out = stderr) const;
    void                reset_stats();

    // ── ModelRunner helpers ───────────────────────────────────────────────────

    // Return the MTLBuffer* (as opaque void*) for a named dense weight.
    // The pointer is non-owning — the EngineCore retains the buffer.
    // Returns nullptr if the name is not found in the dense index.
    // Used by ModelRunner at init time to build its per-layer buffer cache,
    // avoiding repeated string-hash lookups on the hot path.
    void* dense_buf(const std::string& name) const;

private:
    // ── Metal setup ───────────────────────────────────────────────────────────
    Status setup_metal();

    // ── dense loading ─────────────────────────────────────────────────────────
    Status load_dense_weights();
    Status load_one_dense(const std::string& name, void** dst_ptr);
    void*  alloc_and_load_dense(const format::DenseRecord& rec);

    // ── ExpertView builder ────────────────────────────────────────────────────
    ExpertView make_expert_view(const CacheEntry* entry,
                                const ModelConfig& cfg,
                                ExpertKey key) const;

    // ── state ─────────────────────────────────────────────────────────────────

    format::AeroMoEFile file_;
    EngineConfig        engine_cfg_;

    MTLDevicePtr       device_    = nullptr;
    MTLCommandQueuePtr cmd_queue_ = nullptr;

    std::unique_ptr<MemoryLedger> ledger_;
    std::unique_ptr<IOPlanner>    io_planner_;
    std::unique_ptr<ExpertCache>  expert_cache_;

    DenseBuffers backbone_;

    // Dense buffers by FNV-1a hash of name → MTLBuffer*
    std::unordered_map<uint64_t, void*> dense_buffers_;

    mutable RuntimeStats stats_{};
};

} // namespace aeromoe

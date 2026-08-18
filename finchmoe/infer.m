/*
 * infer.m — Complete Qwen3.6-35B-A3B inference engine using Metal
 *
 * Full forward pass: embedding -> 40 transformer layers -> norm -> lm_head -> sample
 * Non-expert weights loaded from model_weights.bin (mmap'd at startup)
 * Expert weights loaded from packed_experts/ per layer per token (pread)
 *
 * Architecture: Qwen3.6-35B-A3B (MoE)
 *   - 40 layers: 30 linear attention (GatedDeltaNet) + 10 full attention
 *   - hidden_size=2048, head_dim=256, num_attention_heads=16, num_kv_heads=2
 *   - 256 experts/layer, 8 active (model trained with 8 experts/token)
 *   - Shared expert per layer (always active)
 *   - Linear attention: conv1d(kernel=4) + gated delta recurrence
 *   - Full attention: standard QKV + scaled dot product + RoPE
 *
 * Command buffer optimization (fused_layer_forward):
 *   Per-layer Metal command buffer structure:
 *     CMD1: attention input projections (3-4 dispatches, 1 commit)
 *     CPU:  attention compute (RoPE/softmax/delta-net)
 *     CMD2: o_proj + residual_add + rms_norm + routing + shared gate/up (8 encoders, 1 commit)
 *           GPU handles residual connection and post-attn norm internally,
 *           eliminating the CPU round-trip that previously split this into 2 cmd buffers.
 *     CPU:  softmax + top-K + pread all K experts (4 pthreads parallel)
 *     CMD3: all K expert forwards + shared SwiGLU + shared down
 *           + GPU-side combine + residual_add + rms_norm -> buf_input (DEFERRED commit)
 *           Batched encoding: 4 encoders for K experts + 2 shared + 3 combine = 9 total
 *   Total: 3 cmd buffers per layer. CMD3 is submitted async (commit without wait).
 *   GPU-side combine in CMD3: for non-last layers, CMD3 also computes:
 *     moe_combine_residual (weighted sum + residual + shared gate -> hidden)
 *     rms_norm (hidden -> buf_input using NEXT layer's input_norm weights)
 *   This allows the next layer's CMD1 to submit immediately without waiting
 *   for CMD3 completion — the GPU queue serializes CMD3(N-1) then CMD1(N).
 *   Saves ~0.83ms/layer deferred_wait + CPU combine + input_norm overhead.
 *   Multi-expert buffers (MAX_K=8 independent slots) allow all K expert
 *   forwards to be encoded into a single command buffer.
 *   Batched encoding: 2 encoders per expert (gate+up fused, SwiGLU+down fused)
 *   + 2 for shared expert = K*2 + 2 total encoders in CMD3.
 *   Double-buffered expert data (buf_multi_expert_data / data_B) for future
 *   async pread overlap with GPU compute.
 *
 * Build:  clang -O2 -Wall -fobjc-arc -framework Metal -framework Foundation -framework Accelerate -lcompression -lpthread infer.m -o infer
 * Run:    ./infer --prompt "Explain relativity" --tokens 50
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include <pthread.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/wait.h>
#include <compression.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/vm_statistics.h>

// ============================================================================
// Memory safety: guard rails to prevent SIGKILL/jetsam on unified-memory
// systems without swap. These are critical on 16GB machines running large
// models (14+ GB) where Metal buffer wrapping can trigger OOM kills.
// ============================================================================

// Safety margin for Metal buffer wrapping: the kernel needs headroom beyond
// the raw weight file size for GPU page tables, IOMMU mappings, and general
// system operation. 2GB is conservative for 16GB machines.
#define METAL_SAFETY_MARGIN_BYTES (256ULL * 1024 * 1024)  // 256MB (was 2GB — far too conservative for 3B active models)

// Returns available memory in bytes (free + inactive + purgeable + speculative).
// Inactive pages are clean file-backed pages the kernel can free instantly.
// Purgeable pages are clean file-backed pages the kernel can discard without
// I/O. Speculative pages are read-ahead pages that haven't been accessed yet.
// On Apple Silicon, these are effectively "available" for new allocations.
static size_t get_available_memory(void) {
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 16384;  // Apple Silicon default
    host_page_size(host, &page_size);

    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;

    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) != KERN_SUCCESS) {
        return 0;  // can't determine — caller should assume unsafe
    }

    size_t free_bytes        = (size_t)vm_stat.free_count * page_size;
    size_t inactive_bytes    = (size_t)vm_stat.inactive_count * page_size;
    size_t purgeable_bytes   = (size_t)vm_stat.purgeable_count * page_size;
    size_t speculative_bytes = (size_t)vm_stat.speculative_count * page_size;

    // Inactive pages are clean file-backed pages the kernel can free instantly
    // without I/O. They are effectively "available" for new allocations.
    return free_bytes + inactive_bytes + purgeable_bytes + speculative_bytes;
}

// Strictly-free bytes (excludes reclaimable cache). The page cache holds the
// expert files that hot-set prefetch re-reads; when free memory collapses the
// kernel evicts that cache and the prefetch turns into net-extra SSD I/O.
static size_t get_free_memory(void) {
    mach_port_t host = mach_host_self();
    vm_size_t page_size = 16384;
    host_page_size(host, &page_size);
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stat, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (size_t)vm_stat.free_count * page_size;
}

// Returns a human-readable memory size string (e.g. "5.52 GB")
// Uses a rotating buffer to allow up to 3 calls in a single printf.
static const char *format_mem_size(size_t bytes) {
    static char buf[3][32];
    static int idx = 0;
    char *b = buf[idx]; idx = (idx + 1) % 3;
    if (bytes >= (1ULL << 30)) {
        snprintf(b, 32, "%.2f GB", (double)bytes / (1ULL << 30));
    } else if (bytes >= (1ULL << 20)) {
        snprintf(b, 32, "%.1f MB", (double)bytes / (1ULL << 20));
    } else {
        snprintf(b, 32, "%.1f KB", (double)bytes / 1024.0);
    }
    return b;
}

// ============================================================================
// Model constants
// ============================================================================

#define HIDDEN_DIM          2048
#define NUM_LAYERS          40
#define NUM_ATTN_HEADS      16
#define NUM_KV_HEADS        2
#define HEAD_DIM            256
#define VOCAB_SIZE          248320
#define RMS_NORM_EPS        1e-6f
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 8
#define MOE_INTERMEDIATE    512
#define SHARED_INTERMEDIATE 512
#define FULL_ATTN_INTERVAL  4
#define GROUP_SIZE          64
#define BITS                4

// Linear attention (GatedDeltaNet) constants
#define LINEAR_NUM_V_HEADS  32
#define LINEAR_NUM_K_HEADS  16
#define LINEAR_KEY_DIM      128   // head_k_dim
#define LINEAR_VALUE_DIM    128   // head_v_dim
#define LINEAR_TOTAL_KEY    (LINEAR_NUM_K_HEADS * LINEAR_KEY_DIM)   // 2048
#define LINEAR_TOTAL_VALUE  (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) // 4096
#define LINEAR_CONV_DIM     (LINEAR_TOTAL_KEY * 2 + LINEAR_TOTAL_VALUE) // 8192
#define CONV_KERNEL_SIZE    4

// Full attention constants
#define ROPE_THETA          10000000.0f
#define PARTIAL_ROTARY      0.25f
#define ROTARY_DIM          (int)(HEAD_DIM * PARTIAL_ROTARY)  // 64

// Expert packed binary layout — 4-bit (from existing code)
#define EXPERT_SIZE_4BIT       1769472
#define GATE_W_OFF_4  0
#define GATE_S_OFF_4  524288
#define GATE_B_OFF_4  557056
#define UP_W_OFF_4    589824
#define UP_S_OFF_4    1114112
#define UP_B_OFF_4    1146880
#define DOWN_W_OFF_4  1179648
#define DOWN_S_OFF_4  1703936
#define DOWN_B_OFF_4  1736704

// Expert packed binary layout — 8-bit (from repack_experts.py --bits 8)
#define EXPERT_SIZE_8BIT       3342336
#define GATE_W_OFF_8  0
#define GATE_S_OFF_8  1048576
#define GATE_B_OFF_8  1081344
#define UP_W_OFF_8    1114112
#define UP_S_OFF_8    2162688
#define UP_B_OFF_8    2195456
#define DOWN_W_OFF_8  2228224
#define DOWN_S_OFF_8  3276800
#define DOWN_B_OFF_8  3309568

// 1-bit expert layout: 32 values/uint32, weights 1/8 of 4-bit, scales/biases same
#define EXPERT_SIZE_1BIT       589824
#define GATE_W_OFF_1  0
#define GATE_S_OFF_1  131072
#define GATE_B_OFF_1  163840
#define UP_W_OFF_1    196608
#define UP_S_OFF_1    327680
#define UP_B_OFF_1    360448
#define DOWN_W_OFF_1  393216
#define DOWN_S_OFF_1  524288
#define DOWN_B_OFF_1  557056

// 2-bit expert layout: 16 values/uint32, weights 1/4 of 4-bit, scales/biases same
// gate/up: [512, 2048] → 512×128 uint32, down: [2048, 512] → 2048×32 uint32
#define EXPERT_SIZE_2BIT       983040
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  262144
#define GATE_B_OFF_2  294912
#define UP_W_OFF_2    327680
#define UP_S_OFF_2    589824
#define UP_B_OFF_2    622592
#define DOWN_W_OFF_2  655360
#define DOWN_S_OFF_2  917504
#define DOWN_B_OFF_2  950272

// 3-bit expert layout: 8 values per 24 bits (3 bytes), group_size=64.
// gate/up: [512, 2048] -> 512*768 bytes, down: [2048, 512] -> 2048*192 bytes
#define EXPERT_SIZE_3BIT       1376256
#define GATE_W_OFF_3  0
#define GATE_S_OFF_3  393216
#define GATE_B_OFF_3  425984
#define UP_W_OFF_3    458752
#define UP_S_OFF_3    851968
#define UP_B_OFF_3    884736
#define DOWN_W_OFF_3  917504
#define DOWN_S_OFF_3  1310720
#define DOWN_B_OFF_3  1343488

// Dynamic offset helpers: pick the right offset based on active format
#define GATE_W_OFF  (g_use_3bit ? GATE_W_OFF_3 : (g_use_2bit ? GATE_W_OFF_2  : (g_use_int8 ? GATE_W_OFF_8  : GATE_W_OFF_4)))
#define GATE_S_OFF  (g_use_3bit ? GATE_S_OFF_3 : (g_use_2bit ? GATE_S_OFF_2  : (g_use_int8 ? GATE_S_OFF_8  : GATE_S_OFF_4)))
#define GATE_B_OFF  (g_use_3bit ? GATE_B_OFF_3 : (g_use_2bit ? GATE_B_OFF_2  : (g_use_int8 ? GATE_B_OFF_8  : GATE_B_OFF_4)))
#define UP_W_OFF    (g_use_3bit ? UP_W_OFF_3   : (g_use_2bit ? UP_W_OFF_2    : (g_use_int8 ? UP_W_OFF_8    : UP_W_OFF_4)))
#define UP_S_OFF    (g_use_3bit ? UP_S_OFF_3   : (g_use_2bit ? UP_S_OFF_2    : (g_use_int8 ? UP_S_OFF_8    : UP_S_OFF_4)))
#define UP_B_OFF    (g_use_3bit ? UP_B_OFF_3   : (g_use_2bit ? UP_B_OFF_2    : (g_use_int8 ? UP_B_OFF_8    : UP_B_OFF_4)))
#define DOWN_W_OFF  (g_use_3bit ? DOWN_W_OFF_3 : (g_use_2bit ? DOWN_W_OFF_2  : (g_use_int8 ? DOWN_W_OFF_8  : DOWN_W_OFF_4)))
#define DOWN_S_OFF  (g_use_3bit ? DOWN_S_OFF_3 : (g_use_2bit ? DOWN_S_OFF_2  : (g_use_int8 ? DOWN_S_OFF_8  : DOWN_S_OFF_4)))
#define DOWN_B_OFF  (g_use_3bit ? DOWN_B_OFF_3 : (g_use_2bit ? DOWN_B_OFF_2  : (g_use_int8 ? DOWN_B_OFF_8  : DOWN_B_OFF_4)))
#define EXPERT_BITS (g_use_1bit ? 1 : (g_use_2bit ? 2 : (g_use_3bit ? 3 : (g_use_int8 ? 8 : 4))))
#define EXPERT_SIZE_MAX 3932160  // max of all expert sizes (8-bit is 3.3MB, rest are smaller)

// KV cache maximum context length — configurable via CLI for agentic workloads
// Qwen 3.6 35B A3B has max_position_embeddings=262144 (256K). This default matches the model.
// The RoPE embeddings are trained for 256K — exceeding this requires YaRN scaling (not implemented).
#define DEFAULT_MAX_SEQ_LEN 262144  // 256K context — matches model's max_position_embeddings
#define DEFAULT_GPU_KV_SEQ  8192     // GPU KV buffer pre-allocation (falls back to CPU past this)
static int g_max_seq_len = DEFAULT_MAX_SEQ_LEN;
static int g_gpu_kv_seq  = DEFAULT_GPU_KV_SEQ;

// Special tokens
#define EOS_TOKEN_1         248046
#define EOS_TOKEN_2         248044
#define THINK_START_TOKEN   248068  // <think>
#define THINK_END_TOKEN     248069  // </think>

#define MODEL_PATH_DEFAULT "../models/Qwen3.6-35B-A3B-4bit-custom"

// ============================================================================
// Timing helper
// ============================================================================

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Per-phase timing accumulators for fused_layer_forward
// Tracks time spent in each pipeline phase across all layers per token.
// Reset at token boundary, printed as summary.
// ============================================================================

typedef struct {
    double deferred_wait;    // waiting for previous CMD3 GPU
    double deferred_cpu;     // CPU readback + combine for deferred experts
    double input_norm;       // CPU RMS norm + CMD1 prep
    double cmd1_submit;      // CMD1 encode + commit
    double cmd1_wait;        // CMD1 waitUntilCompleted
    double cpu_attn;         // CPU attention compute (delta-net or full-attn)
    double cmd2_encode;      // CMD2 encode (o_proj + residual + norm + routing)
    double cmd2_wait;        // CMD2 commit + waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK
    double spec_route;       // speculative early routing (gate matvec + topK)
    double expert_io;        // parallel pread + cache lookup
    double cmd3_encode;      // CMD3 encode experts + submit (deferred)
    double total;            // total per-layer time
    int count;               // number of layers timed
} LayerTimingAccum;

static LayerTimingAccum g_timing = {0};
static int g_timing_enabled = 0;

// Chunked-prefill per-phase timing (env-gated: FINCHMOE_PF_TIMING=1).
// The chunked path never updates g_timing, so it needs its own accumulators.
typedef struct {
    double cmdA_encode;      // Phase A encode + commit
    double cmdA_wait;        // Phase A waitUntilCompleted
    double attn_cpu;         // per-position CPU attention + KV appends
    double cmdB_encode;      // Phase A2 encode + commit (full-attn only)
    double cmdB_wait;        // Phase A2 waitUntilCompleted
    double routing_cpu;      // CPU softmax + topK per position (Phase B)
    double pread_wait;       // expert pread waits (async_pread_wait)
    double cmd3_encode;      // CMD3 encode + deferred commit per position
    double backpressure;     // CMD3(m-1) backpressure waits
    double combine_wait;     // final driver combine wait
    // Phase C S4 perf pass: GGUF chain internals + CMD3 decomposition
    double chain_cpu;        // chain per-position CPU loop (in_proj/conv/norms)
    double delta_wait;       // batched delta_net_step CB commit + wait
    double chain_readback;   // delta out readback + gated norm (CPU)
    double bridge;           // oproj_in -> oproj_in2 CPU bridge memcpy
    double cmd3_wait;        // explicit final-CMD3 wait (FINCHMOE_PF_CMD3WAIT)
    // Phase C S6: GPU wake-latency accounting. g_pf_last_gpu_activity is
    // stamped at every commit AND every wait-return; the gap before a commit
    // is the CPU window in which the GPU had nothing queued — the GPU
    // downclocks after ~1ms idle and the next submission pays the wake tax
    // (measured via FINCHMOE_CBLAT: 0.013ms back-to-back, 0.17ms@0.1ms gap,
    // 1.55ms@1ms gap, 4.4ms@3ms gap).
    double cmdA_gap, delta_gap, cmdB_gap, cmd3_gap;
    double total;            // total across all layer calls
    int layers;              // number of layer calls timed
} ChunkLayerTimingAccum;

static ChunkLayerTimingAccum g_chunk_timing = {0};
static int g_chunk_timing_enabled = 0;  // set from FINCHMOE_PF_TIMING env
static double g_pf_last_gpu_activity = 0;  // S6: last commit or wait-return
static void pf_note_gap(double *bucket) {
    // Call immediately before [cb commit]: bucket += idle window since the
    // last GPU submission or wait-return.
    if (!g_chunk_timing_enabled) return;
    double t = now_ms();
    if (g_pf_last_gpu_activity > 0 && t > g_pf_last_gpu_activity)
        *bucket += t - g_pf_last_gpu_activity;
    g_pf_last_gpu_activity = t;
}
static void pf_note_wait_done(void) {
    // Call immediately after [cb waitUntilCompleted].
    if (g_chunk_timing_enabled) g_pf_last_gpu_activity = now_ms();
}
// Phase C S4 perf pass: per-layer phase attribution. Columns (all ms):
// 0 cmdA_wait, 1 chain_cpu, 2 delta_wait, 3 chain_readback, 4 bridge,
// 5 cmdB_wait, 6 routing_cpu, 7 pread_wait, 8 cmd3_encode, 9 cmd3_wait,
// 10 total.
static double g_pf_per_layer[NUM_LAYERS][11] = {{0}};
static int g_pf_per_layer_count[NUM_LAYERS] = {0};
// Phase C S4 perf pass: GGUF expert pread dedup stats (unique vs total
// slots per layer, accumulated over all layer calls).
static int g_gguf_dedup_unique = 0;
static int g_gguf_dedup_slots = 0;
// Phase C S4 perf pass: per-layer delta recorder (no-op when timing is off).
static inline void pf_per_layer_add(int layer_idx, int col, double ms) {
    if (g_chunk_timing_enabled && layer_idx >= 0 && layer_idx < NUM_LAYERS)
        g_pf_per_layer[layer_idx][col] += ms;
}
// Phase-B trace accumulation (FINCHMOE_DUMP_PHASEB) — written once per prefill
static float *g_pb_acc = NULL;
static size_t g_pb_len = 0;
static int g_debug_layers = 0;  // --debug-layers: print per-layer hidden state stats
static int g_gpu_experts = 0;   // --gpu-experts: force GPU experts (now default)
static int g_cpu_experts = 0;   // --cpu-experts: force CPU experts for debugging
static int g_compare_experts = -1; // --compare-experts N: compare GPU vs CPU expert outputs for layer N

static void debug_print_hidden(const char *tag, int layer_idx, const float *h, int dim) {
    if (!g_debug_layers) return;
    // FINCHMOE_LAYER_DUMP: append the vector at every hook to
    // /tmp/layer_dump.bin for cross-path vector comparison. Each record is
    // prefixed with a tag id (u32) and length (u32) so readers can align.
    if (getenv("FINCHMOE_LAYER_DUMP")) {
        FILE *df = fopen("/tmp/layer_dump.bin", "ab");
        if (df) {
            uint32_t tagid = (uint32_t)(tag[0] * 256 + tag[1]);  // "in", "po", "qv", ...
            uint32_t len = (uint32_t)dim;
            fwrite(&tagid, 4, 1, df);
            fwrite(&len, 4, 1, df);
            fwrite(h, sizeof(float), dim, df);
            fclose(df);
        }
    }
    double sum = 0, sum_sq = 0;
    float minv = h[0], maxv = h[0];
    for (int i = 0; i < dim; i++) {
        sum += h[i];
        sum_sq += (double)h[i] * h[i];
        if (h[i] < minv) minv = h[i];
        if (h[i] > maxv) maxv = h[i];
    }
    double mean = sum / dim;
    double rms = sqrt(sum_sq / dim);
    double std = sqrt(sum_sq / dim - mean * mean);
    fprintf(stderr, "[DEBUG-L%d] %s: mean=%.6f rms=%.6f std=%.6f min=%.6f max=%.6f\n",
            layer_idx, tag, mean, rms, std, minv, maxv);
}

// Temporal prediction pipeline counters (declared early for timing_print access)
static int g_pred_enabled = 0;
static int g_pred_generating = 0;   // only set to 1 after prefill (predictions only help during generation)
static int g_use_mtp = 0;           // --mtp: enable MTP speculative decoding
static uint64_t g_pred_hits = 0;
static uint64_t g_pred_misses = 0;
static uint64_t g_pred_layers = 0;

// Routing data collection for training an expert predictor
// Binary format per sample: int32 layer_idx, int32 K, float32[2048] hidden, int32[K] expert_indices
static FILE *g_routing_log = NULL;
static int g_routing_log_samples = 0;

// LZ4 compressed expert support
// File format: [LZ4IndexEntry × 256] + [compressed blobs]
typedef struct {
    uint64_t offset;
    uint32_t comp_size;
    uint32_t raw_size;
} LZ4IndexEntry;

static LZ4IndexEntry *g_lz4_index[NUM_LAYERS];  // per-layer index (NULL if not using LZ4)
static void *g_lz4_comp_bufs[8];                 // pre-allocated compressed read buffers (MAX_K=8)
static int g_use_lz4 = 0;                        // auto-detected from packed_experts_lz4/

// ============================================================================
// Expert frequency tracking (diagnostic: --freq flag)
// ============================================================================

static int g_expert_freq[NUM_LAYERS][NUM_EXPERTS];  // activation count per (layer, expert)
static int g_freq_tracking = 0;  // enabled by --freq flag
static int g_use_1bit = 0;       // enabled by --1bit flag: use packed_experts_1bit/ + 1-bit kernel
static int g_use_2bit = 0;       // enabled by --2bit flag: use packed_experts_2bit/ + 2-bit kernel
static int g_use_3bit = 1;       // DEFAULT: 3-bit experts (9.1 tok/s, near-4bit quality); --2bit/--4bit/--8bit override
static int g_use_int8 = 0;       // enabled by --int8-experts flag: use 8-bit packed experts
static int g_cache_telemetry_enabled = 0;  // enabled by --cache-telemetry flag
static int g_think_budget = 200;  // max thinking tokens before force-emitting </think> — the model otherwise loops inside the think phase on long-form prompts (2048 was effectively unlimited)
static float g_temperature = 0.7f;  // sampling temperature (0 = greedy argmax); 0.7 ends long gens naturally (bug 15)
static float g_min_p = 0.05f;       // min_p sampling: filter tokens below min_p * top_prob — the
                                    // long-form synonym-drift cure (llama.cpp-style tail filter)
// 0.3 default: T=0.8 amplifies the mild temporal logit drift (Bug 15) into
// merged-word artifacts ("abouta", "roboticton") and mid-block repetition
// loops. 0.1-0.3 is the empirically clean range for this engine; llama.cpp
// Qwen presets behave the same way with top_p/min_p active (which this
// sampler lacks).

// Chunked batched prefill: process prompt tokens in chunks of N through
// batched GPU matmuls instead of one matvec per token. 0 = per-token path
// (baseline). Buffers are sized for PREFILL_CHUNK_MAX positions.
#define PREFILL_CHUNK_MAX 256
#define PF_ATTN_MAX 64     // batched GPU attention cap (positions per dispatch)
static int g_prefill_chunk = 8;  // --prefill-chunk N (0 = per-token path; 8 = pooled+batched-attn sweet spot)
static int g_pf_pool_slots = 0;  // pool-mode expert slots (64 → 32 → 16 → 0 under memory pressure)
static int g_pf_hot_slots = 0;   // prefetch-pool slots per layer (32 → 16 → 0 under memory pressure)
#define PF_HOT_MAX 64
static int g_hot_slot[NUM_LAYERS][NUM_EXPERTS];  // layer → expert → prefetch slot (-1 = not hot)
static int g_hot_expert[NUM_LAYERS][PF_HOT_MAX]; // layer → slot → expert (for prefetch preads)
static int g_hot_loaded = 0;   // hot_sets.bin loaded (prefetch active)
static int g_top_k = 40;            // top-k sampling (1 = greedy)
static int g_no_think = 0;          // 0 = thinking mode on, 1 = skip think block
static int g_low_memory = 0;       // enabled by --low-memory: skip Metal weight wrap, use CPU fallback
static const char *g_dump_logits_path = NULL;  // --dump-logits FILE: save first-token logits for cross-validation

// Tiered I/O: cold fds (F_NOCACHE) for first reads, warm fds (page cached) for repeats
static int *g_layer_fds_cold = NULL;    // [NUM_LAYERS] cold fds (set in main)
static uint8_t g_expert_seen[NUM_LAYERS][NUM_EXPERTS / 8];  // bitset: seen before?

// Async pread state defined after InferPreadTask (see below)

static inline int expert_is_seen(int layer, int expert) {
    return (g_expert_seen[layer][expert >> 3] >> (expert & 7)) & 1;
}
static inline void expert_mark_seen(int layer, int expert) {
    g_expert_seen[layer][expert >> 3] |= (1 << (expert & 7));
}
// Pick fd for expert read. Currently: always use warm fd (OS page cache).
// Tiered I/O (cold F_NOCACHE for first reads) was tested but OS page cache
// without any bypass outperforms all custom caching strategies.
static inline int expert_pick_fd(int layer, int expert, int warm_fd) {
    (void)layer; (void)expert;
    return warm_fd;
}

// Active expert size based on quantization mode
static inline size_t active_expert_size(void) {
    return g_use_1bit ? EXPERT_SIZE_1BIT : (g_use_2bit ? EXPERT_SIZE_2BIT : (g_use_3bit ? EXPERT_SIZE_3BIT : (g_use_int8 ? EXPERT_SIZE_8BIT : EXPERT_SIZE_4BIT)));
}
static int g_freq_total_tokens = 0;  // total tokens processed while tracking

typedef struct {
    uint64_t token_clock;
    uint64_t unique_experts_touched;
    uint64_t cold_misses;
    uint64_t eviction_misses;
    uint64_t evictions;
    uint64_t reuse_le_1;
    uint64_t reuse_le_4;
    uint64_t reuse_le_16;
    uint64_t reuse_le_64;
    uint64_t reuse_gt_64;
    uint64_t reuse_distance_sum;
    uint64_t reuse_distance_samples;
} CacheTelemetry;

static CacheTelemetry g_cache_telemetry = {0};
static uint8_t g_cache_seen[NUM_LAYERS][NUM_EXPERTS];
static uint64_t g_cache_last_touch_token[NUM_LAYERS][NUM_EXPERTS];
static uint64_t g_cache_last_evict_token[NUM_LAYERS][NUM_EXPERTS];

static void cache_telemetry_reset(void) {
    memset(&g_cache_telemetry, 0, sizeof(g_cache_telemetry));
    memset(g_cache_seen, 0, sizeof(g_cache_seen));
    memset(g_cache_last_touch_token, 0, sizeof(g_cache_last_touch_token));
    memset(g_cache_last_evict_token, 0, sizeof(g_cache_last_evict_token));
}

static void cache_telemetry_note_token(void) {
    if (!g_cache_telemetry_enabled) return;
    g_cache_telemetry.token_clock++;
}

static void cache_telemetry_touch(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_miss(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    if (!g_cache_seen[layer_idx][expert_idx]) {
        g_cache_telemetry.cold_misses++;
        g_cache_seen[layer_idx][expert_idx] = 1;
        g_cache_telemetry.unique_experts_touched++;
    } else {
        g_cache_telemetry.eviction_misses++;
        uint64_t dist = 0;
        if (g_cache_last_evict_token[layer_idx][expert_idx] > 0 &&
            g_cache_telemetry.token_clock >= g_cache_last_evict_token[layer_idx][expert_idx]) {
            dist = g_cache_telemetry.token_clock - g_cache_last_evict_token[layer_idx][expert_idx];
        }
        if (dist <= 1) g_cache_telemetry.reuse_le_1++;
        else if (dist <= 4) g_cache_telemetry.reuse_le_4++;
        else if (dist <= 16) g_cache_telemetry.reuse_le_16++;
        else if (dist <= 64) g_cache_telemetry.reuse_le_64++;
        else g_cache_telemetry.reuse_gt_64++;
        g_cache_telemetry.reuse_distance_sum += dist;
        g_cache_telemetry.reuse_distance_samples++;
    }
    g_cache_last_touch_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_evict(int layer_idx, int expert_idx) {
    if (!g_cache_telemetry_enabled) return;
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS || expert_idx < 0 || expert_idx >= NUM_EXPERTS) return;
    g_cache_telemetry.evictions++;
    g_cache_last_evict_token[layer_idx][expert_idx] = g_cache_telemetry.token_clock;
}

static void cache_telemetry_print(uint64_t hits, uint64_t misses) {
    if (!g_cache_telemetry_enabled) return;
    uint64_t total = hits + misses;
    fprintf(stderr, "\n=== Cache Telemetry ===\n");
    fprintf(stderr, "Tokens tracked: %llu\n", g_cache_telemetry.token_clock);
    fprintf(stderr, "Unique experts touched: %llu / %d (%.1f%%)\n",
            g_cache_telemetry.unique_experts_touched,
            NUM_LAYERS * NUM_EXPERTS,
            100.0 * g_cache_telemetry.unique_experts_touched / (NUM_LAYERS * NUM_EXPERTS));
    fprintf(stderr, "Miss breakdown: cold %llu (%.1f%% of misses), eviction %llu (%.1f%% of misses)\n",
            g_cache_telemetry.cold_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.cold_misses / misses : 0.0,
            g_cache_telemetry.eviction_misses,
            misses > 0 ? 100.0 * g_cache_telemetry.eviction_misses / misses : 0.0);
    fprintf(stderr, "Evictions: %llu\n", g_cache_telemetry.evictions);
    fprintf(stderr, "Eviction reuse distance: <=1 tok %llu, <=4 %llu, <=16 %llu, <=64 %llu, >64 %llu",
            g_cache_telemetry.reuse_le_1,
            g_cache_telemetry.reuse_le_4,
            g_cache_telemetry.reuse_le_16,
            g_cache_telemetry.reuse_le_64,
            g_cache_telemetry.reuse_gt_64);
    if (g_cache_telemetry.reuse_distance_samples > 0) {
        fprintf(stderr, " (avg %.1f tok)\n",
                (double)g_cache_telemetry.reuse_distance_sum / g_cache_telemetry.reuse_distance_samples);
    } else {
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "Effective hit rate: %.1f%%\n",
            total > 0 ? 100.0 * hits / total : 0.0);
}

static void timing_reset(void) {
    memset(&g_timing, 0, sizeof(g_timing));
}

static void timing_print(void) {
    if (g_timing.count == 0) return;
    int n = g_timing.count;
    fprintf(stderr, "\n[timing] Per-layer breakdown (avg of %d layers, ms):\n", n);
    fprintf(stderr, "  deferred_wait:  %6.3f\n", g_timing.deferred_wait / n);
    fprintf(stderr, "  deferred_cpu:   %6.3f\n", g_timing.deferred_cpu / n);
    fprintf(stderr, "  input_norm:     %6.3f\n", g_timing.input_norm / n);
    fprintf(stderr, "  cmd1_submit:    %6.3f\n", g_timing.cmd1_submit / n);
    fprintf(stderr, "  cmd1_wait:      %6.3f\n", g_timing.cmd1_wait / n);
    fprintf(stderr, "  spec_route:     %6.3f\n", g_timing.spec_route / n);
    fprintf(stderr, "  cpu_attn:       %6.3f\n", g_timing.cpu_attn / n);
    fprintf(stderr, "  cmd2_encode:    %6.3f\n", g_timing.cmd2_encode / n);
    fprintf(stderr, "  cmd2_wait:      %6.3f\n", g_timing.cmd2_wait / n);
    fprintf(stderr, "  routing_cpu:    %6.3f\n", g_timing.routing_cpu / n);
    fprintf(stderr, "  expert_io:      %6.3f\n", g_timing.expert_io / n);
    fprintf(stderr, "  cmd3_encode:    %6.3f\n", g_timing.cmd3_encode / n);
    fprintf(stderr, "  total_layer:    %6.3f\n", g_timing.total / n);
    fprintf(stderr, "  sum_phases:     %6.3f\n",
            (g_timing.deferred_wait + g_timing.deferred_cpu + g_timing.input_norm +
             g_timing.cmd1_submit + g_timing.cmd1_wait + g_timing.spec_route +
             g_timing.cpu_attn +
             g_timing.cmd2_encode + g_timing.cmd2_wait + g_timing.routing_cpu +
             g_timing.expert_io + g_timing.cmd3_encode) / n);
    fprintf(stderr, "  cmd_buffers:    %d (3 per layer: CMD1+CMD2+CMD3)\n", n * 3);
    fprintf(stderr, "  sync_waits:     %d (2 per layer: CMD1+CMD2, CMD3 deferred)\n", n * 2);
    fprintf(stderr, "  gpu_encoders:   ~%d per layer (CMD1:3-4, CMD2:8-12, CMD3:~10)\n",
            22);  // approximate
    if (g_pred_enabled && g_pred_layers > 0) {
        uint64_t total = g_pred_hits + g_pred_misses;
        double hit_rate = total > 0 ? (double)g_pred_hits / total * 100.0 : 0;
        fprintf(stderr, "  [predict] hits=%llu misses=%llu rate=%.1f%% layers=%llu\n",
                g_pred_hits, g_pred_misses, hit_rate, g_pred_layers);
    }
}

// Chunked-prefill per-phase timing summary (FINCHMOE_PF_TIMING=1).
static double g_pf_enc_micro = 0; static int g_pf_enc_n = 0;   // S8 probe: matvec-encode micro timing
static double g_pf_cbcreate_ms = 0, g_pf_pregap_ms = 0; static int g_pf_cbcreate_n = 0;   // S8 probe: CB create + pre-encode gap
static void chunk_timing_print(void) {
    if (!g_chunk_timing_enabled || g_chunk_timing.layers == 0) return;
    int n = g_chunk_timing.layers;
    if (g_pf_enc_n > 0)
        fprintf(stderr, "  enc_micro (matvecs only): %7.3f (n=%d)\n",
                g_pf_enc_micro / g_pf_enc_n, g_pf_enc_n);
    if (g_pf_cbcreate_n > 0)
        fprintf(stderr, "  cb_create: %7.3f  pre-encode gap: %7.3f (n=%d)\n",
                g_pf_cbcreate_ms / g_pf_cbcreate_n, g_pf_pregap_ms / g_pf_cbcreate_n, g_pf_cbcreate_n);
    fprintf(stderr, "\n[pf-timing] Chunked prefill per-layer breakdown (avg of %d layer calls, ms):\n", n);
    fprintf(stderr, "  cmdA_encode:   %7.3f\n", g_chunk_timing.cmdA_encode / n);
    fprintf(stderr, "  cmdA_wait:     %7.3f\n", g_chunk_timing.cmdA_wait / n);
    fprintf(stderr, "  attn_cpu:      %7.3f\n", g_chunk_timing.attn_cpu / n);
    fprintf(stderr, "  cmdB_encode:   %7.3f\n", g_chunk_timing.cmdB_encode / n);
    fprintf(stderr, "  cmdB_wait:     %7.3f\n", g_chunk_timing.cmdB_wait / n);
    fprintf(stderr, "  routing_cpu:   %7.3f\n", g_chunk_timing.routing_cpu / n);
    fprintf(stderr, "  pread_wait:    %7.3f\n", g_chunk_timing.pread_wait / n);
    fprintf(stderr, "  cmd3_encode:   %7.3f\n", g_chunk_timing.cmd3_encode / n);
    fprintf(stderr, "  backpressure:  %7.3f\n", g_chunk_timing.backpressure / n);
    fprintf(stderr, "  combine_wait:  %7.3f\n", g_chunk_timing.combine_wait / n);
    fprintf(stderr, "  chain_cpu:     %7.3f\n", g_chunk_timing.chain_cpu / n);
    fprintf(stderr, "  delta_wait:    %7.3f\n", g_chunk_timing.delta_wait / n);
    fprintf(stderr, "  chain_rb:      %7.3f\n", g_chunk_timing.chain_readback / n);
    fprintf(stderr, "  bridge:        %7.3f\n", g_chunk_timing.bridge / n);
    fprintf(stderr, "  cmd3_wait:     %7.3f\n", g_chunk_timing.cmd3_wait / n);
    fprintf(stderr, "  -- S6 GPU-idle gaps before each commit (wake tax: ~1.6ms@1ms, ~4.4ms@3ms) --\n");
    fprintf(stderr, "  cmdA_gap:      %7.3f\n", g_chunk_timing.cmdA_gap / n);
    fprintf(stderr, "  delta_gap:     %7.3f\n", g_chunk_timing.delta_gap / n);
    fprintf(stderr, "  cmdB_gap:      %7.3f\n", g_chunk_timing.cmdB_gap / n);
    fprintf(stderr, "  cmd3_gap:      %7.3f\n", g_chunk_timing.cmd3_gap / n);
    if (g_gguf_dedup_slots > 0) {
        fprintf(stderr, "  expert dedup:  %d unique / %d slots (%.1f%%) across all layers\n",
                g_gguf_dedup_unique, g_gguf_dedup_slots,
                100.0 * (double)g_gguf_dedup_unique / (double)g_gguf_dedup_slots);
    }
    fprintf(stderr, "  layer_total:   %7.3f\n", g_chunk_timing.total / n);
    fprintf(stderr, "  total_all:     %7.1f ms across %d layers\n", g_chunk_timing.total, n);
    // Per-layer table (avg across chunk calls): shows which layers carry
    // the deferred CMD3 tail inside cmdA_wait and where preads dominate.
    int shown = 0;
    fprintf(stderr, "  per-layer (avg ms):  lay cmdA_wt chain_cpu dlt_wt chain_rb bridge cmdB_wt route pread_wt c3_enc c3_wt total\n");
    for (int l = 0; l < NUM_LAYERS; l++) {
        if (!g_pf_per_layer_count[l]) continue;
        shown++;
        double d = (double)g_pf_per_layer_count[l];
        fprintf(stderr, "    %3d %6.2f %7.3f %6.3f %7.3f %6.3f %6.3f %5.3f %7.3f %6.3f %6.3f %6.2f\n",
                l, g_pf_per_layer[l][0] / d, g_pf_per_layer[l][1] / d, g_pf_per_layer[l][2] / d,
                g_pf_per_layer[l][3] / d, g_pf_per_layer[l][4] / d, g_pf_per_layer[l][5] / d,
                g_pf_per_layer[l][6] / d, g_pf_per_layer[l][7] / d, g_pf_per_layer[l][8] / d,
                g_pf_per_layer[l][9] / d, g_pf_per_layer[l][10] / d);
    }
    if (!shown) fprintf(stderr, "    (no per-layer records)\n");
}

// ============================================================================
// bf16 <-> f32 conversion (CPU side)
// ============================================================================

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

__attribute__((unused))
static uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

// ============================================================================
// JSON parser (minimal, for model_weights.json)
// ============================================================================

// We use NSJSONSerialization via ObjC since we already link Foundation

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int ndim;
    int shape[4];
    char dtype[8];  // "U32", "BF16", "F32"
    int ggml_type;  // 0 = manifest-native; 12 = Q4_K, 14 = Q6_K (GGUF mode)
} TensorInfo;

typedef struct {
    TensorInfo *tensors;
    int num_tensors;
    int capacity;
    char *model_path;  // from manifest "model" field
} TensorManifest;

static TensorManifest *load_manifest(const char *json_path) {
    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:json_path]];
        if (!data) {
            fprintf(stderr, "ERROR: Cannot read %s\n", json_path);
            return NULL;
        }

        NSError *error = nil;
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data
                                                             options:0
                                                               error:&error];
        if (!root) {
            fprintf(stderr, "ERROR: JSON parse failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return NULL;
        }

        NSDictionary *tensors = root[@"tensors"];
        if (!tensors) {
            fprintf(stderr, "ERROR: No 'tensors' key in manifest\n");
            return NULL;
        }

        TensorManifest *m = calloc(1, sizeof(TensorManifest));
        // Store the model path from the manifest (for expert file auto-detection)
        NSString *model_str = root[@"model"];
        if (model_str) {
            m->model_path = strdup([model_str UTF8String]);
        }
        m->capacity = (int)[tensors count] + 16;
        m->tensors = calloc(m->capacity, sizeof(TensorInfo));
        m->num_tensors = 0;

        for (NSString *key in tensors) {
            NSDictionary *info = tensors[key];
            TensorInfo *t = &m->tensors[m->num_tensors];

            const char *name = [key UTF8String];
            t->name = strdup(name);
            t->offset = [info[@"offset"] unsignedLongLongValue];
            t->size = [info[@"size"] unsignedLongLongValue];

            NSArray *shape = info[@"shape"];
            t->ndim = (int)[shape count];
            for (int i = 0; i < t->ndim && i < 4; i++) {
                t->shape[i] = [shape[i] intValue];
            }

            const char *dtype = [info[@"dtype"] UTF8String];
            strncpy(t->dtype, dtype, 7);

            m->num_tensors++;
        }

        printf("[manifest] Loaded %d tensors from %s\n", m->num_tensors, json_path);
        return m;
    }
}

// Hash table for O(1) tensor lookup (replaces O(N) linear scan).
// FNV-1a hash, open addressing with linear probing.
#define TENSOR_HT_SIZE 8192  // power of 2, > 4x num_tensors (2092)

typedef struct {
    const char *key;     // tensor name (pointer into TensorInfo)
    TensorInfo *value;   // pointer to tensor info
} TensorHTEntry;

static TensorHTEntry tensor_ht[TENSOR_HT_SIZE];
static int tensor_ht_built = 0;

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)*s;
        h *= 16777619u;
    }
    return h;
}

static void build_tensor_ht(TensorManifest *m) {
    if (tensor_ht_built) return;
    memset(tensor_ht, 0, sizeof(tensor_ht));
    for (int i = 0; i < m->num_tensors; i++) {
        uint32_t idx = fnv1a(m->tensors[i].name) & (TENSOR_HT_SIZE - 1);
        while (tensor_ht[idx].key) {
            idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
        }
        tensor_ht[idx].key = m->tensors[i].name;
        tensor_ht[idx].value = &m->tensors[i];
    }
    tensor_ht_built = 1;
}

static TensorInfo *find_tensor(TensorManifest *m, const char *name) {
    if (!tensor_ht_built) build_tensor_ht(m);
    uint32_t idx = fnv1a(name) & (TENSOR_HT_SIZE - 1);
    while (tensor_ht[idx].key) {
        if (strcmp(tensor_ht[idx].key, name) == 0) {
            return tensor_ht[idx].value;
        }
        idx = (idx + 1) & (TENSOR_HT_SIZE - 1);
    }
    return NULL;
}

// ============================================================================
// Weight file: mmap'd binary blob
// ============================================================================

typedef struct {
    void *data;
    size_t size;
    TensorManifest *manifest;
    void *gguf_stage;  // GGUF mode: the staged F32→BF16 conversion buffer
} WeightFile;

static void *g_gguf_stage = NULL;
static void *g_gguf_data_base = NULL;   // the GGUF mmap base (Phase C tensor wraps)
static size_t g_gguf_stage_len = 0;     // Phase C S4: staged-BF16 buffer size (set in open_gguf)
// Phase C S6 probe (FINCHMOE_GGUF_STAGE2): 2MB-aligned anonymous copy of
// the GPU-read QK tensors — file-backed GPU reads pay per-16KB-page DART
// walks (~5-8GB/s effective first-touch); anonymous 2MB pages walk free
// (the pool's ~46GB/s). Scope 1 = cmdA's qkv/z only.
static char *g_gguf_stage2 = NULL;
static size_t g_gguf_stage2_len = 0;
static id<MTLBuffer> g_gguf_stage2_gpu = NULL;
static size_t g_gguf_exp_alloc = 0;     // Phase C S4: GGUF expert pool slot size (4MB)
static int g_pf_pool_slots_gguf = 0;    // Phase C S4: GGUF pool slot count (0 = off)

static WeightFile *open_weights(const char *bin_path, const char *json_path) {
    // mmap the binary file
    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", bin_path, strerror(errno));
        return NULL;
    }

    struct stat st;
    fstat(fd, &st);
    size_t size = st.st_size;

    void *data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap failed: %s\n", strerror(errno));
        return NULL;
    }

    // No madvise: kernel default (demand-paging) is safest.
    // MADV_SEQUENTIAL triggers aggressive readahead — on unified-memory
    // systems without swap, this can cause the kernel to eagerly page in
    // the entire multi-GB file, spiking memory pressure and triggering jetsam
    // (SIGKILL) when the file is later wrapped as a Metal buffer.
    // MADV_RANDOM disables readahead (tested: hurts for matmul patterns).
    // Kernel default demand-paging is the best balance: pages fault in as
    // the GPU accesses them through the unified memory controller.

    TensorManifest *manifest = load_manifest(json_path);
    if (!manifest) {
        munmap(data, size);
        return NULL;
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = size;
    wf->manifest = manifest;

    printf("[weights] mmap'd %.2f GB from %s\n", size / 1e9, bin_path);
    return wf;
}

static int g_missing_tensor_count = 0;
static void *get_tensor_ptr(WeightFile *wf, const char *name) {
    TensorInfo *t = find_tensor(wf->manifest, name);
    if (!t) {
        g_missing_tensor_count++;
        return NULL;
    }
    // GGUF mode: the staged BF16 tensors live in the conversion buffer
    if (wf->gguf_stage && strcmp(t->dtype, "BF16") == 0)
        return (char *)wf->gguf_stage + t->offset;
    return (char *)wf->data + t->offset;
}

static TensorInfo *get_tensor_info(WeightFile *wf, const char *name) {
    return find_tensor(wf->manifest, name);
}

// Detect packing width from manifest shapes (group_size = 64 convention):
//   4-bit: row_u32 == num_groups * 8   (8 values per uint32)
//   8-bit: row_u32 == num_groups * 16  (4 values per uint32)
// Returns 4 when scales are missing (BF16 path, caller must handle) or
// shapes are unexpected.
// ============================================================================
// GGUF importer (--gguf): parse the llama.cpp container and synthesize the
// engine's TensorManifest. The engine consumes tensors via HF-style names;
// a name map translates the GGUF's blk.N.* names. GGUF F32 tensors that the
// engine reads as BF16 (norms, conv1d, dt_bias, routers) are converted into
// a staging buffer at load time; A_log (ssm_a) stays F32.
// Quant types implemented: F32 (0), Q4_K (12), Q6_K (14).
// ============================================================================

// ggml type traits (block size, type size) for the local GGUF's types
static size_t ggml_type_size(int t) {
    switch (t) {
        case 0:  return 4;   // F32
        case 1:  return 2;   // F16
        case 12: return 144; // Q4_K — 256 elems/block
        case 14: return 210; // Q6_K — 256 elems/block
        default: return 0;
    }
}
static int ggml_block_elems(int t) {
    switch (t) {
        case 12: case 14: return 256;
        default: return 1;
    }
}

// GGUF tensor name → the engine's HF-style manifest name.
// Returns 1 and fills out (up to 255 chars) on success; 0 = skip this tensor.
static int gguf_map_name(const char *in, char *out, size_t out_sz) {
    int layer = -1;
    // blk.%d. prefix
    if (sscanf(in, "blk.%d.", &layer) == 1) {
        // skip the MTP block (blk.40 — unused, as today)
        if (layer >= NUM_LAYERS) return 0;
        const char *rest = strchr(in, '.') + 1;   // past "blk."
        rest = strchr(rest, '.') + 1;             // past the layer number
        rest = rest ? rest : in;
        struct { const char *gguf; const char *hf; } map[] = {
            {"attn_norm.weight",              "model.layers.%d.input_layernorm.weight"},
            {"post_attention_norm.weight",    "model.layers.%d.post_attention_layernorm.weight"},
            {"attn_q.weight",                 "model.layers.%d.self_attn.q_proj.weight"},
            {"attn_k.weight",                 "model.layers.%d.self_attn.k_proj.weight"},
            {"attn_v.weight",                 "model.layers.%d.self_attn.v_proj.weight"},
            {"attn_output.weight",            "model.layers.%d.self_attn.o_proj.weight"},
            {"attn_q_norm.weight",            "model.layers.%d.self_attn.q_norm.weight"},
            {"attn_k_norm.weight",            "model.layers.%d.self_attn.k_norm.weight"},
            {"attn_qkv.weight",               "model.layers.%d.linear_attn.in_proj_qkv.weight"},
            {"attn_gate.weight",              "model.layers.%d.linear_attn.in_proj_z.weight"},
            {"ssm_alpha.weight",              "model.layers.%d.linear_attn.in_proj_a.weight"},
            {"ssm_beta.weight",               "model.layers.%d.linear_attn.in_proj_b.weight"},
            {"ssm_conv1d.weight",             "model.layers.%d.linear_attn.conv1d.weight"},
            {"ssm_a",                        "model.layers.%d.linear_attn.A_log"},
            {"ssm_dt.bias",                  "model.layers.%d.linear_attn.dt_bias"},
            {"ssm_norm.weight",               "model.layers.%d.linear_attn.norm.weight"},
            {"ssm_out.weight",                "model.layers.%d.linear_attn.out_proj.weight"},
            {"ffn_gate_inp.weight",           "model.layers.%d.mlp.gate.weight"},
            {"ffn_gate_shexp.weight",         "model.layers.%d.mlp.shared_expert.gate_proj.weight"},
            {"ffn_up_shexp.weight",           "model.layers.%d.mlp.shared_expert.up_proj.weight"},
            {"ffn_down_shexp.weight",         "model.layers.%d.mlp.shared_expert.down_proj.weight"},
            {"ffn_gate_inp_shexp.weight",     "model.layers.%d.mlp.shared_expert_gate.weight"},
        };
        for (size_t i = 0; i < sizeof(map)/sizeof(map[0]); i++) {
            if (strcmp(rest, map[i].gguf) == 0) {
                snprintf(out, out_sz, map[i].hf, layer);
                return 1;
            }
        }
        // the stacked expert tensors + the shared experts are consumed via
        // the expert table, not the manifest
        return 0;
    }
    if (strcmp(in, "token_embd.weight") == 0) { snprintf(out, out_sz, "model.embed_tokens.weight"); return 1; }
    if (strcmp(in, "output.weight") == 0)     { snprintf(out, out_sz, "lm_head.weight"); return 1; }
    if (strcmp(in, "output_norm.weight") == 0){ snprintf(out, out_sz, "model.norm.weight"); return 1; }
    return 0;
}

// The GGUF tensors the engine consumes as BF16 (the engine's norm + router
// paths read uint16_t + bf16_to_f32). F32 → BF16 conversion at load time.
static int gguf_needs_bf16_stage(const char *hf_name) {
    return strstr(hf_name, "norm.weight") != NULL ||
           strstr(hf_name, "conv1d.weight") != NULL ||
           strstr(hf_name, "dt_bias") != NULL ||
           strstr(hf_name, "mlp.gate.weight") != NULL ||
           strstr(hf_name, "shared_expert_gate.weight") != NULL ||
           // the GDN alpha/beta projections: the engine reads these via the
           // BF16 raw path in GGUF mode (tensor_bits() = 0 for F32), so the
           // F32 GGUF data must be staged — otherwise F32 bytes are read as
           // BF16 pairs (alternating garbage).
           strstr(hf_name, "in_proj_a.weight") != NULL ||
           strstr(hf_name, "in_proj_b.weight") != NULL;
}

typedef struct {
    size_t gate_off, up_off, down_off;   // file offsets of the stacked tensors
    size_t gate_slab, up_slab, down_slab; // bytes per expert
    int gate_type, up_type, down_type;    // ggml types (12 = Q4_K, 14 = Q6_K)
} GgufExpertInfo;
static GgufExpertInfo gguf_experts[NUM_LAYERS];
static int g_gguf_fd = -1;   // reopened fd for the expert preads

static WeightFile *open_gguf(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "ERROR: Cannot open %s: %s\n", path, strerror(errno)); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    void *data = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { fprintf(stderr, "ERROR: mmap %s failed\n", path); return NULL; }

    const unsigned char *p = (const unsigned char *)data;
    size_t sz = (size_t)st.st_size;
    #define R32() (*(const uint32_t *)p, p += 4, *((const uint32_t *)(p - 4)))
    #define R64() (*(const uint64_t *)p, p += 8, *((const uint64_t *)(p - 8)))
    #define RS(len) (const char *)(p + 0)
    uint32_t magic = R32();
    if (magic != 0x46554747) { fprintf(stderr, "ERROR: %s is not a GGUF file\n", path); return NULL; }
    uint32_t version = R32();
    uint64_t n_tensors = R64();
    uint64_t n_kv = R64();
    fprintf(stderr, "[gguf] magic OK, version=%u, %llu tensors, %llu kv\n",
            version, (unsigned long long)n_tensors, (unsigned long long)n_kv);

    // KV block (find the alignment; skip the rest)
    uint32_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen = R64();
        char key[256];
        if (klen < sizeof(key)) { memcpy(key, p, klen); p += klen; key[klen] = 0; }
        else { p += klen; key[0] = 0; }
        uint32_t vt = R32();
        switch (vt) {
            case 0: case 1: p += 1; break;                       // u8 / i8
            case 2: case 3: p += 2; break;                       // u16 / i16
            case 4: case 5: p += 4; break;                       // u32 / i32
            case 6: p += 4; break;                               // f32
            case 7: p += 1; break;                               // bool
            case 8: { uint64_t sl = R64(); p += sl; break; }     // string
            case 9: {                                            // array
                uint32_t at = R32();
                uint64_t n = R64();
                if (at == 8) {                                   // array of strings
                    for (uint64_t j = 0; j < n; j++) {
                        uint64_t sl = R64();
                        p += sl;
                    }
                } else {
                    static const size_t esz[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
                    size_t e = at < sizeof(esz)/sizeof(esz[0]) ? esz[at] : 8;
                    p += n * e;
                }
                break;
            }
            case 10: case 11: case 12: p += 8; break;            // u64 / i64 / f64
            default: break;
        }
    }
    // (the alignment key is a u32 in practice; handled by the vt==4 case above
    //  when present — re-scan not needed for this file: alignment = 32 holds)

    // Tensor infos. The data section starts AFTER the infos, aligned — the
    // GGUF toff fields are relative to that aligned start. Pre-scan the infos
    // to locate it (the parse loop below re-walks them).
    const unsigned char *infos_start = p;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen = R64();
        p += nlen;
        if (p > (const unsigned char *)data + sz) { fprintf(stderr, "[gguf] tensor infos overrun — corrupt file?\n"); return NULL; }
        uint32_t nd = R32();
        p += (size_t)nd * 8 + 4 + 8;   // dims + ggml type + toff
    }
    size_t data_off = ((size_t)(p - (const unsigned char *)data) + alignment - 1) & ~(size_t)(alignment - 1);
    p = infos_start;

    TensorManifest *m = calloc(1, sizeof(TensorManifest));
    m->capacity = (int)n_tensors + 16;
    m->tensors = calloc(m->capacity, sizeof(TensorInfo));
    m->num_tensors = 0;
    // the staging buffer for the F32→BF16 conversions (grow as needed)
    static void *stage = NULL;
    static size_t stage_cap = 0;
    g_gguf_stage_len = 0;

    int n_q4 = 0, n_q6 = 0, n_f32 = 0, n_skipped = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t nlen = R64();
        char gname[256];
        if (nlen < sizeof(gname)) { memcpy(gname, p, nlen); p += nlen; gname[nlen] = 0; }
        else { fprintf(stderr, "[gguf] long tensor name — abort\n"); return NULL; }
        uint32_t ndim = R32();
        uint64_t dims[4] = {0};
        for (uint32_t d = 0; d < ndim && d < 4; d++) dims[d] = R64();
        for (uint32_t d = 4; d < ndim; d++) R64();
        uint32_t gtype = R32();
        uint64_t toff = R64();

        // capture the stacked expert tensors (consumed via the pread table)
        {
            int XL = -1;
            int is_expert = 0;
            size_t *offp = NULL, *slabp = NULL; int *typep = NULL;
            // Full-string match required: sscanf("blk.%d.<rest>") returns 1 for
            // ANY "blk.N." prefix (the %d converts, trailing literal mismatch
            // just stops matching), so verify the rest with strcmp via %n.
            int consumed = 0;
            if (sscanf(gname, "blk.%d.%n", &XL, &consumed) == 1 &&
                strcmp(gname + consumed, "ffn_gate_exps.weight") == 0) {
                is_expert = 1; offp = &gguf_experts[XL].gate_off; slabp = &gguf_experts[XL].gate_slab; typep = &gguf_experts[XL].gate_type;
            } else if (sscanf(gname, "blk.%d.%n", &XL, &consumed) == 1 &&
                       strcmp(gname + consumed, "ffn_up_exps.weight") == 0) {
                is_expert = 1; offp = &gguf_experts[XL].up_off; slabp = &gguf_experts[XL].up_slab; typep = &gguf_experts[XL].up_type;
            } else if (sscanf(gname, "blk.%d.%n", &XL, &consumed) == 1 &&
                       strcmp(gname + consumed, "ffn_down_exps.weight") == 0) {
                is_expert = 1; offp = &gguf_experts[XL].down_off; slabp = &gguf_experts[XL].down_slab; typep = &gguf_experts[XL].down_type;
            }
            if (is_expert && XL >= 0 && XL < NUM_LAYERS) {
                // the slab = n_out rows × row_bytes (the dims: [in, out, 256])
                size_t bs = ggml_type_size((int)gtype);
                size_t be = (size_t)ggml_block_elems((int)gtype);
                size_t row_bytes = (dims[0] / be) * bs;   // dims[0] = in_dim
                *offp = data_off + toff;
                *slabp = (size_t)dims[1] * row_bytes;     // dims[1] = out_dim per expert
                *typep = (int)gtype;
                continue;
            }
        }

        char hf_name[256];
        if (!gguf_map_name(gname, hf_name, sizeof(hf_name))) {
            n_skipped++; continue;
        }

        size_t bs = ggml_type_size((int)gtype);
        size_t be = (size_t)ggml_block_elems((int)gtype);
        if (bs == 0) { fprintf(stderr, "[gguf] unsupported type %u for %s\n", gtype, gname); return NULL; }
        uint64_t nelem = 1;
        for (uint32_t d = 0; d < ndim; d++) nelem *= dims[d];
        size_t tsize = (size_t)((nelem + be - 1) / be) * bs;

        TensorInfo *t = &m->tensors[m->num_tensors++];
        t->name = strdup(hf_name);
        t->ndim = (int)ndim;
        for (int d = 0; d < 4; d++) t->shape[d] = (int)dims[d];
        t->ggml_type = (int)gtype;
        if (gtype == 12) { n_q4++; strcpy(t->dtype, "Q4K"); }
        else if (gtype == 14) { n_q6++; strcpy(t->dtype, "Q6K"); }
        else { n_f32++; strcpy(t->dtype, "F32"); }

        if (gtype == 0 && gguf_needs_bf16_stage(hf_name)) {
            // F32 → BF16 staging: the engine's norm/router paths read uint16_t
            const float *src = (const float *)((const unsigned char *)data + data_off + toff);
            size_t nfl = (size_t)nelem;
            size_t need = nfl * 2;
            if (g_gguf_stage_len + need > stage_cap) {
                stage_cap = stage_cap ? stage_cap * 2 : (1 << 20);
                while (stage_cap < g_gguf_stage_len + need) stage_cap *= 2;
                stage = realloc(stage, stage_cap);
            }
            uint16_t *dst = (uint16_t *)((char *)stage + g_gguf_stage_len);
            // NOTE: ssm_conv1d in this GGUF is [4, 8192] with ne0=4 = tap index
            // (fastest) — i.e. [ch0t0, ch0t1, ch0t2, ch0t3, ch1t0, ...], which
            // is already the engine's w[c*K + k] channel-major order. A flat
            // copy is correct; do NOT transpose (verified against HF 2026-08-16).
            for (size_t j = 0; j < nfl; j++) {
                uint32_t b; memcpy(&b, &src[j], 4);
                dst[j] = (uint16_t)(b >> 16);
            }
            t->offset = g_gguf_stage_len;
            t->size = need;
            t->shape[1] = (int)nfl;   // 1-D layout for the staged tensors
            t->ndim = 2;
            strcpy(t->dtype, "BF16");
            g_gguf_stage_len += need;
        } else {
            t->offset = data_off + toff;
            t->size = tsize;
        }
    }
    m->model_path = strdup(".");

    fprintf(stderr, "[gguf] %d tensors mapped (%d Q4_K, %d Q6_K, %d F32; %d skipped), data @ %zu, staged %zu bytes\n",
            m->num_tensors, n_q4, n_q6, n_f32, n_skipped, data_off, g_gguf_stage_len);
    // S6 layout probe: where do the used (GPU-read) tensors live?
    if (getenv("FINCHMOE_GGUF_LAYOUT")) {
        for (int i = 0; i < m->num_tensors; i++) {
            fprintf(stderr, "[gguf-layout] %-48s off=%12zu size=%10zu\n",
                    m->tensors[i].name, m->tensors[i].offset, m->tensors[i].size);
        }
    }
    // Phase C S6: madvise(WILLNEED) on every used tensor — the GPU faults
    // on pages the CPU never touched (the ~2ms/layer single-pass overhead
    // on cmdA/cmdB: KLOOP back-to-back iterations are 1.0-1.35ms but the
    // first pass is ~3.4ms). Kernel readahead makes the pages resident
    // (DART-mapped) before the kernels run; expert slabs are already
    // covered by the pool preads. Escape: FINCHMOE_GGUF_NOWILLNEED.
    if (!getenv("FINCHMOE_GGUF_NOWILLNEED")) {
        double t_w = now_ms();
        size_t wl_total = 0;
        for (int i = 0; i < m->num_tensors; i++) {
            if (m->tensors[i].offset > (size_t)sz) continue;   // staged (heap) markers
            madvise((char *)data + m->tensors[i].offset, m->tensors[i].size, MADV_WILLNEED);
            wl_total += m->tensors[i].size;
        }
        fprintf(stderr, "[gguf] WILLNEED issued for %zu MB of tensor data (%.0f ms)\n",
                wl_total >> 20, now_ms() - t_w);
    }

    WeightFile *wf = calloc(1, sizeof(WeightFile));
    wf->data = data;
    wf->size = sz;
    wf->manifest = m;
    // the staged BF16 copies live OUTSIDE the mmap — the get_tensor_ptr
    // pointer arithmetic assumes the manifest offsets address wf->data.
    // Handle by giving the staged tensors absolute pointer semantics via a
    // special offset marker: we store the stage pointer delta in the offset
    // and translate in get_tensor_ptr.
    g_gguf_stage = stage;
    g_gguf_data_base = data;
    wf->gguf_stage = stage;
    g_gguf_fd = open(path, O_RDONLY);  // for the expert preads
    return wf;
    #undef R32
    #undef R64
}

static int tensor_bits(WeightFile *wf, const char *base_name) {
    // GGUF mode: the manifest records the ggml block type
    if (wf->gguf_stage) {
        char wname[256];
        snprintf(wname, sizeof(wname), "%s.weight", base_name);
        TensorInfo *ti = get_tensor_info(wf, wname);
        if (!ti) return 0;
        if (ti->ggml_type == 12) return 10;   // Q4_K
        if (ti->ggml_type == 14) return 11;   // Q6_K
        return 0;                             // F32/BF16 (staged) → BF16 path
    }
    char wname[256], sname[256];
    snprintf(wname, sizeof(wname), "%s.weight", base_name);
    snprintf(sname, sizeof(sname), "%s.scales", base_name);
    TensorInfo *wi = get_tensor_info(wf, wname);
    TensorInfo *si = get_tensor_info(wf, sname);
    if (!wi || !si) return 4;
    if (wi->shape[1] == si->shape[1] * 16) return 8;
    return 4;
}

// ============================================================================
// Vocabulary for token decoding
// ============================================================================

typedef struct {
    char **tokens;   // token_id -> UTF-8 string
    int *lengths;    // token_id -> byte length
    int num_tokens;
} Vocabulary;

static uint32_t read_u32_vocab(FILE *f) {
    uint32_t v;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}
static uint16_t read_u16_vocab(FILE *f) {
    uint16_t v;
    if (fread(&v, 2, 1, f) != 1) return 0;
    return v;
}

static Vocabulary *load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }

    // Parse BPET format (same as tokenizer.bin):
    //   magic "BPET" (4 bytes)
    //   version    (uint32_t)
    //   vocab_size (uint32_t)
    //   num_merges (uint32_t)
    //   num_added  (uint32_t)
    //   vocab entries: { uint32_t id, uint16_t len, char[len] str }
    //   merge entries: { uint16_t len_a, char[len_a], uint16_t len_b, char[len_b] }
    //   added tokens:  { uint32_t id, uint16_t len, char[len] }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BPET", 4) != 0) {
        fprintf(stderr, "ERROR: %s is not a BPET vocab file (bad magic)\n", path);
        fclose(f);
        return NULL;
    }
    uint32_t version = read_u32_vocab(f);
    if (version != 1) {
        fprintf(stderr, "ERROR: %s BPET version %u, expected 1\n", path, version);
        fclose(f);
        return NULL;
    }
    uint32_t vocab_size  = read_u32_vocab(f);
    uint32_t num_merges  = read_u32_vocab(f);
    uint32_t num_added   = read_u32_vocab(f);

    // Find the maximum token id so we can size the lookup array.
    // We'll read all entries once to find max_id, then again to populate.
    // Actually, read into a temporary array, then build the final lookup.
    Vocabulary *v = calloc(1, sizeof(Vocabulary));
    v->num_tokens = vocab_size + num_added + 256;  // generous: cover all potential ids
    v->tokens = calloc(v->num_tokens, sizeof(char *));
    v->lengths = calloc(v->num_tokens, sizeof(int));

    // Read vocab entries
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t id   = read_u32_vocab(f);
        uint16_t len  = read_u16_vocab(f);
        if (len > 0 && id < v->num_tokens) {
            v->tokens[id] = malloc(len + 1);
            if (fread(v->tokens[id], 1, len, f) != len) break;
            v->tokens[id][len] = '\0';
            v->lengths[id] = len;
        } else if (len > 0) {
            // Token id exceeds allocated range — skip its data
            fseek(f, len, SEEK_CUR);
        }
    }

    // Skip merge entries (2 × uint16_t len + char data per merge)
    for (uint32_t i = 0; i < num_merges; i++) {
        uint16_t len_a = read_u16_vocab(f);
        if (len_a > 0) fseek(f, len_a, SEEK_CUR);
        uint16_t len_b = read_u16_vocab(f);
        if (len_b > 0) fseek(f, len_b, SEEK_CUR);
    }

    // Read added tokens (may have ids beyond vocab_size)
    for (uint32_t i = 0; i < num_added; i++) {
        uint32_t id  = read_u32_vocab(f);
        uint16_t len = read_u16_vocab(f);
        if (len > 0 && id < v->num_tokens && !v->tokens[id]) {
            v->tokens[id] = malloc(len + 1);
            if (fread(v->tokens[id], 1, len, f) != len) break;
            v->tokens[id][len] = '\0';
            v->lengths[id] = len;
        } else if (len > 0) {
            fseek(f, len, SEEK_CUR);
        }
    }

    fclose(f);

    // Qwen byte-fallback recovery: the byte-level BPE vocab stores the raw
    // bytes of multi-byte characters as latin-1 code points U+0080..U+00FF
    // (the HF tokenizers convention — the curly quote ' = U+00E2 U+0080
    // U+0099). The exporter UTF-8-encoded those, so the stored bytes are
    // C2/C3 two-byte sequences. Convert them back to the raw bytes so the
    // decoded text renders correctly (otherwise "’s" shows as mojibake).
    // The Qwen vocab has no precomposed accented tokens (everything
    // non-ASCII routes through the byte fallback), so every C2/C3 sequence
    // is a fallback marker. CJK characters encode in the E4..EF range and
    // pass through untouched.
    for (uint32_t i = 0; i < v->num_tokens; i++) {
        if (!v->tokens[i]) continue;
        unsigned char *s = (unsigned char *)v->tokens[i];
        int has_marker = 0;
        for (int j = 0; s[j]; j++) {
            if ((s[j] == 0xC2 || s[j] == 0xC3) && s[j+1] >= 0x80 && s[j+1] <= 0xBF) {
                has_marker = 1; break;
            }
        }
        if (!has_marker) continue;
        char out[512];
        int o = 0;
        for (int j = 0; s[j] && o < 511; ) {
            if ((s[j] == 0xC2 || s[j] == 0xC3) && s[j+1] >= 0x80 && s[j+1] <= 0x40 + 0xBF - 0xBF) {
                // (unreachable — the real check below)
            }
            if ((s[j] == 0xC2 || s[j] == 0xC3) && s[j+1] >= 0x80 && s[j+1] <= 0xBF) {
                out[o++] = (char)((((unsigned)s[j] & 0x1F) << 6) | ((unsigned)s[j+1] & 0x3F));
                j += 2;
            } else {
                out[o++] = (char)s[j++];
            }
        }
        out[o] = '\0';
        strcpy(v->tokens[i], out);
        v->lengths[i] = o;
    }

    // Find the actual highest token id with a valid string
    uint32_t max_valid = 0;
    for (uint32_t i = 0; i < v->num_tokens; i++) {
        if (v->tokens[i]) max_valid = i;
    }
    printf("[vocab] Loaded %u tokens (max_id=%u) from %s\n", vocab_size + num_added, max_valid, path);
    return v;
}

static const char *decode_token(Vocabulary *v, int token_id) {
    if (!v || token_id < 0 || token_id >= v->num_tokens || !v->tokens[token_id]) {
        return "<unk>";
    }
    const char *raw = v->tokens[token_id];

    // Fast path: if the string is pure ASCII (no UTF-8 multi-byte lead bytes),
    // return it directly — the common case for most tokens.
    int has_utf8 = 0;
    for (const char *p = raw; *p; p++) {
        if ((*p & 0x80)) { has_utf8 = 1; break; }
    }
    if (!has_utf8) return raw;

    // Decode GPT-2 byte-fallback encoding:
    // Bytes 0x00-0xFF are mapped to Unicode U+0100-U+01FF in the vocab.
    // e.g. space (0x20) → 'Ġ' (U+0120), newline (0x0A) → 'Ċ' (U+010A).
    // Convert them back to the original bytes.
    static char decoded[512];
    char *dst = decoded;
    const char *src = raw;
    const char *end = decoded + sizeof(decoded) - 1;
    while (*src && dst < end) {
        // Decode a UTF-8 code point
        unsigned int cp;
        int advance;
        if ((*src & 0x80) == 0) {
            cp = (unsigned char)*src; advance = 1;
        } else if ((*src & 0xE0) == 0xC0 && (src[1] & 0xC0) == 0x80) {
            cp = ((unsigned)(*src & 0x1F) << 6) | (src[1] & 0x3F);
            advance = 2;
        } else if ((*src & 0xF0) == 0xE0 && (src[1] & 0xC0) == 0x80 && (src[2] & 0xC0) == 0x80) {
            cp = ((unsigned)(*src & 0x0F) << 12) | ((unsigned)(src[1] & 0x3F) << 6) | (src[2] & 0x3F);
            advance = 3;
        } else if ((*src & 0xF8) == 0xF0 && (src[1] & 0xC0) == 0x80 && (src[2] & 0xC0) == 0x80 && (src[3] & 0xC0) == 0x80) {
            cp = ((unsigned)(*src & 0x07) << 18) | ((unsigned)(src[1] & 0x3F) << 12) | ((unsigned)(src[2] & 0x3F) << 6) | (src[3] & 0x3F);
            advance = 4;
        } else {
            *dst++ = *src++;  // broken sequence, pass through
            continue;
        }
        // Byte-fallback range: U+0100-U+01FF → byte (cp - 0x100)
        if (cp >= 0x100 && cp <= 0x1FF) {
            *dst++ = (char)(cp - 0x100);
        } else if (cp <= 0x7F) {
            *dst++ = (char)cp;
        } else {
            // Non-byte-fallback multi-byte character (e.g. CJK, emoji) — copy verbatim
            for (int i = 0; i < advance && dst < end; i++)
                *dst++ = src[i];
        }
        src += advance;
    }
    *dst = '\0';
    return decoded;
}

// ============================================================================
// Prompt tokens loader
// ============================================================================

typedef struct {
    uint32_t *ids;
    int count;
} PromptTokens;

static PromptTokens *load_prompt_tokens(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    fread(&pt->count, 4, 1, f);
    pt->ids = malloc(pt->count * sizeof(uint32_t));
    fread(pt->ids, 4, pt->count, f);
    fclose(f);
    return pt;
}

// ============================================================================
// C BPE tokenizer (replaces Python encode_prompt.py)
// ============================================================================
#define TOKENIZER_IMPL
#include "tokenizer.h"

static bpe_tokenizer g_tokenizer;
static int g_tokenizer_loaded = 0;

static void init_tokenizer(void) {
    if (g_tokenizer_loaded) return;
    const char *paths[] = {
        "tokenizer.bin",
        "metal_infer/tokenizer.bin",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        if (access(paths[i], R_OK) == 0) {
            if (bpe_load(&g_tokenizer, paths[i]) == 0) {
                g_tokenizer_loaded = 1;
                return;
            }
        }
    }
    fprintf(stderr, "WARNING: tokenizer.bin not found, tokenization will fail\n");
}

static PromptTokens *encode_prompt_text_to_tokens(const char *text) {
    init_tokenizer();
    if (!g_tokenizer_loaded) return NULL;

    // Allocate output buffer (generous: 4 tokens per character worst case)
    int max_ids = (int)strlen(text) * 4 + 256;
    uint32_t *ids = malloc(max_ids * sizeof(uint32_t));
    if (!ids) return NULL;

    int n = bpe_encode(&g_tokenizer, text, ids, max_ids);
    if (n < 0) { free(ids); return NULL; }

    PromptTokens *pt = calloc(1, sizeof(PromptTokens));
    pt->ids = ids;
    pt->count = n;

    // Cross-reference helper: dump the prompt token IDs in the same format
    // as prompt_tokens_gguf.bin ([n u32][n × u32]) so llama.cpp logit_dump
    // can decode the identical prompt (FINCHMOE_DUMP_PROMPT_TOKENS=file).
    {
        const char *dp = getenv("FINCHMOE_DUMP_PROMPT_TOKENS");
        if (dp) {
            FILE *f = fopen(dp, "wb");
            if (f) {
                uint32_t n32 = (uint32_t)n;
                fwrite(&n32, sizeof(uint32_t), 1, f);
                fwrite(ids, sizeof(uint32_t), (size_t)n, f);
                fclose(f);
                fprintf(stderr, "[tokens] prompt token IDs dumped to %s (%d tokens)\n", dp, n);
            }
        }
    }

    fprintf(stderr, "Tokens (%d): [", n);
    for (int i = 0; i < n && i < 20; i++) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%u", ids[i]);
    }
    if (n > 20) fprintf(stderr, ", ...");
    fprintf(stderr, "]\n");

    return pt;
}

// ============================================================================
// CPU computation kernels
// ============================================================================

// 4-bit dequant matvec: out[out_dim] = W * x[in_dim]
// W is stored as packed uint32 (8 x 4-bit or 4 x 8-bit values per uint32)
// scales/biases are bfloat16 per group
// GGUF block dequant (llama.cpp-reference loops). row = one tensor row
// (n values, ggml_type 12 = Q4_K / 14 = Q6_K), y = the n floats.
static float fp32_from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t fp32_to_bits(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }
// ggml_compute_fp16_to_fp32 (ggml-impl.h): the plain h<<16 bit pattern is only
// the first step — the exponent must be re-normalized and denormals handled.
static float fp16_to_f32(uint16_t h) {
    const uint32_t w = (uint32_t)h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = 1u << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}
// get_scale_min_k4(j, scales, &sc, &m) from llama.cpp's dequantize_row_q4_K
static void get_scale_min_k4(int j, const uint8_t *scales, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = scales[j + 0] & 63;
        *m  = scales[j + 4] & 63;
    } else {
        *sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m  = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4);
    }
}
static void gguf_dequant_row(const void *row, float *y, int n, int ggml_type) {
    const uint8_t *p = (const uint8_t *)row;
    if (ggml_type == 12) {  // Q4_K: 144B blocks of 256
        const int nb = n / 256;
        for (int b = 0; b < nb; b++) {
            uint16_t dh, dm;
            memcpy(&dh, p, 2); memcpy(&dm, p + 2, 2);
            const float d = fp16_to_f32(dh), mmin = fp16_to_f32(dm);
            const uint8_t *scales = p + 4;
            const uint8_t *q = p + 16;
            int is = 0;
            for (int j = 0; j < 256; j += 64) {
                // each 32-byte q chunk = 64 values: low nibbles then the
                // HIGH nibbles of the same bytes (llama.cpp dequantize_row_q4_K)
                uint8_t sc, m;
                get_scale_min_k4(is + 0, scales, &sc, &m);
                const float d1 = d * sc, m1 = mmin * m;
                get_scale_min_k4(is + 1, scales, &sc, &m);
                const float d2 = d * sc, m2 = mmin * m;
                for (int l = 0; l < 32; l++)
                    y[j + l]      = d1 * (q[l] & 0xF) - m1;
                for (int l = 0; l < 32; l++)
                    y[j + l + 32] = d2 * (q[l] >> 4) - m2;
                q += 32; is += 2;
            }
            p += 144; y += 256;
        }
    } else if (ggml_type == 14) {  // Q6_K: 210B blocks of 256
        const int nb = n / 256;
        for (int b = 0; b < nb; b++) {
            // block_q6_K layout: ql(128) + qh(64) + scales(16) + d(2 at +208)
            uint16_t dh; memcpy(&dh, p + 208, 2);
            const float d = fp16_to_f32(dh);
            const uint8_t *ql = p + 0;
            const uint8_t *qh = p + 128;
            const int8_t *sc = (const int8_t *)(p + 192);
            for (int nn = 0; nn < 256; nn += 128) {
                for (int l = 0; l < 32; l++) {
                    const int is = l / 16;
                    const float q1 = (float)((int)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    const float q2 = (float)((int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    const float q3 = (float)((int)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    const float q4 = (float)((int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                    y[l + 0]  = d * sc[is + 0] * q1;
                    y[l + 32] = d * sc[is + 2] * q2;
                    y[l + 64] = d * sc[is + 4] * q3;
                    y[l + 96] = d * sc[is + 6] * q4;
                }
                y += 128; ql += 64; qh += 32; sc += 8;
            }
            p += 210;
        }
    }
}

// GGUF matvec: the row-dequant + dot (the cpu_dequant_matvec bits 10/11 path)
static void gguf_cpu_matvec(const void *W, const float *x, float *out,
                            int out_dim, int in_dim, int ggml_type) {
    const uint8_t *wp = (const uint8_t *)W;
    const size_t row_bytes = (size_t)(in_dim / 256) *
        (ggml_type == 12 ? 144 : 210);
    static float *yb = NULL;
    static int yb_n = 0;
    if (!yb || yb_n < in_dim) { free(yb); yb = malloc(in_dim * sizeof(float)); yb_n = in_dim; }
    for (int r = 0; r < out_dim; r++) {
        gguf_dequant_row(wp + (size_t)r * row_bytes, yb, in_dim, ggml_type);
        float acc = 0.0f;
        for (int i = 0; i < in_dim; i++) acc += yb[i] * x[i];
        out[r] = acc;
    }
}

static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size, int bits
) {
    // GGUF Q4_K / Q6_K (bits 10 / 11) — the llama.cpp block formats.
    // Checked BEFORE the scales==NULL BF16 fallback: Q4_K tensors have no
    // scales, and the raw bytes must never be read as BF16.
    if (bits == 10 || bits == 11) {
        gguf_cpu_matvec(W, x, out, out_dim, in_dim, bits == 10 ? 12 : 14);
        return;
    }

    // BF16 raw path: scales==NULL means unquantized BF16 weight
    if (!scales || !biases) {
        const uint16_t *W_bf16 = (const uint16_t *)W;
        for (int row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            const uint16_t *w_row = W_bf16 + row * in_dim;
            for (int i = 0; i < in_dim; i++) {
                acc += bf16_to_f32(w_row[i]) * x[i];
            }
            out[row] = acc;
        }
        return;
    }

    int num_groups = in_dim / group_size;

    // 3-bit path: 8 values per 24 bits (3 bytes), byte-addressed rows
    if (bits == 3) {
        const uint8_t *W8 = (const uint8_t *)W;
        int triplets = in_dim / 8;
        int triplets_per_g = group_size / 8;
        int row_bytes = in_dim * 3 / 8;
        for (int row = 0; row < out_dim; row++) {
            float acc = 0.0f;
            const uint8_t *w_row = W8 + (size_t)row * row_bytes;
            const uint16_t *s_row = scales + row * num_groups;
            const uint16_t *b_row = biases + row * num_groups;
            for (int t = 0; t < triplets; t++) {
                int g = t / triplets_per_g;
                float scale = bf16_to_f32(s_row[g]);
                float bias = bf16_to_f32(b_row[g]);
                uint32_t packed = (uint32_t)w_row[t * 3] |
                                  ((uint32_t)w_row[t * 3 + 1] << 8) |
                                  ((uint32_t)w_row[t * 3 + 2] << 16);
                int xb = t * 8;
                for (int j = 0; j < 8; j++) {
                    float w_val = (float)((packed >> (3 * j)) & 0x7) * scale + bias;
                    acc += w_val * x[xb + j];
                }
            }
            out[row] = acc;
        }
        return;
    }

    int vals_per_u32 = 32 / bits;  // 32 for 1-bit, 16 for 2-bit, 8 for 4-bit, 4 for 8-bit
    int packed_per_group = group_size / vals_per_u32;
    int packed_cols = in_dim / vals_per_u32;
    int shift_per_val = bits;

    for (int row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = W + row * packed_cols;
        const uint16_t *s_row = scales + row * num_groups;
        const uint16_t *b_row = biases + row * num_groups;

        for (int g = 0; g < num_groups; g++) {
            float scale = bf16_to_f32(s_row[g]);
            float bias = bf16_to_f32(b_row[g]);
            int base_packed = g * packed_per_group;
            int base_x = g * group_size;

            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[base_packed + p];
                int x_base = base_x + p * vals_per_u32;

                for (int n = 0; n < vals_per_u32; n++) {
                    uint32_t val = (packed >> (n * shift_per_val)) & ((1u << bits) - 1);
                    acc += ((float)val * scale + bias) * x[x_base + n];
                }
            }
        }
        out[row] = acc;
    }
}

// RMS normalization: out = x * w / rms(x)
static void cpu_rms_norm(const float *x, const uint16_t *w_bf16, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / dim + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < dim; i++) {
        float weight = bf16_to_f32(w_bf16[i]);
        out[i] = x[i] * inv_rms * weight;
    }
}

// SwiGLU: out = silu(gate) * up
static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

// Sigmoid
static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Softmax over a vector
static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < dim; i++) {
        x[i] *= inv_sum;
    }
}

// Top-K: find K largest indices from scores[dim]
static void cpu_topk(const float *scores, int dim, int K, int *indices, float *values) {
    // Simple selection sort for small K
    // Initialize with -inf
    for (int k = 0; k < K; k++) {
        values[k] = -1e30f;
        indices[k] = 0;
    }

    for (int i = 0; i < dim; i++) {
        // Check if this score beats the smallest in our top-K
        int min_k = 0;
        for (int k = 1; k < K; k++) {
            if (values[k] < values[min_k]) min_k = k;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}

// Normalize top-K weights to sum to 1
static void cpu_normalize_weights(float *weights, int K) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++) sum += weights[k];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int k = 0; k < K; k++) weights[k] *= inv;
    }
}

// Element-wise add: dst += src
__attribute__((unused))
static void cpu_vec_add(float *dst, const float *src, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += src[i];
}

// Element-wise multiply-add: dst += scale * src
static void cpu_vec_madd(float *dst, const float *src, float scale, int dim) {
    for (int i = 0; i < dim; i++) dst[i] += scale * src[i];
}

// Element-wise multiply: dst = a * b
__attribute__((unused))
static void cpu_vec_mul(float *dst, const float *a, const float *b, int dim) {
    for (int i = 0; i < dim; i++) dst[i] = a[i] * b[i];
}

// Copy
static void cpu_vec_copy(float *dst, const float *src, int dim) {
    memcpy(dst, src, dim * sizeof(float));
}

// Zero
__attribute__((unused))
static void cpu_vec_zero(float *dst, int dim) {
    memset(dst, 0, dim * sizeof(float));
}

// Argmax
static int cpu_argmax(const float *x, int dim) {
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < dim; i++) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}

// ============================================================================
// Repetition penalty: prevents greedy sampling from getting stuck in
// high-probability token attractors (e.g. "* * * *" loops at T=0).
// Tracks recently generated tokens in a ring buffer and penalizes their
// logits before sampling.  See Bug 15 (long-generation repetition loop).
// ============================================================================
#define REPETITION_WINDOW 64
static int   g_rep_ring[REPETITION_WINDOW];
static int   g_rep_pos = 0;
static int   g_rep_count = 0;
static float g_rep_penalty = 1.15f;   // 1.15 = mild; 1.0 = disabled

static void rep_penalty_register(int token_id) {
    g_rep_ring[g_rep_pos] = token_id;
    g_rep_pos = (g_rep_pos + 1) % REPETITION_WINDOW;
    if (g_rep_count < REPETITION_WINDOW) g_rep_count++;
}

static void rep_penalty_apply(float *logits, int dim) {
    if (g_rep_penalty <= 1.0f || g_rep_count == 0) return;
    for (int i = 0; i < g_rep_count; i++) {
        int tid = g_rep_ring[i];
        if (tid >= 0 && tid < dim) {
            // Standard formula: push positive logits down, negative up
            if (logits[tid] > 0.0f) {
                logits[tid] /= g_rep_penalty;
            } else {
                logits[tid] *= g_rep_penalty;
            }
        }
    }
}

// Logit diagnostics: periodically dump logit statistics during generation
// to diagnose temporal drift (Bug 15). Enabled via --logit-diag N.
static int g_logit_diag_interval = 0;  // 0 = disabled; N = dump every N tokens

static void logit_diag_dump(const float *logits, int dim, int token_idx, int gen_step) {
    if (g_logit_diag_interval <= 0) return;
    if (gen_step % g_logit_diag_interval != 0 && gen_step != 0) return;

    // Find top 20
    #define DIAG_TOPK 20
    int top_idx[DIAG_TOPK];
    float top_val[DIAG_TOPK];
    for (int i = 0; i < DIAG_TOPK; i++) { top_idx[i] = -1; top_val[i] = -INFINITY; }
    for (int i = 0; i < dim; i++) {
        float v = logits[i];
        // Find min in current top-k
        int min_k = 0;
        for (int k = 1; k < DIAG_TOPK; k++) if (top_val[k] < top_val[min_k]) min_k = k;
        if (v > top_val[min_k]) { top_val[min_k] = v; top_idx[min_k] = i; }
    }
    // Sort descending
    for (int i = 0; i < DIAG_TOPK - 1; i++) {
        for (int j = i + 1; j < DIAG_TOPK; j++) {
            if (top_val[j] > top_val[i]) {
                float tv = top_val[i]; top_val[i] = top_val[j]; top_val[j] = tv;
                int ti = top_idx[i]; top_idx[i] = top_idx[j]; top_idx[j] = ti;
            }
        }
    }

    // Compute softmax + entropy
    float max_val = logits[0];
    for (int i = 1; i < dim; i++) if (logits[i] > max_val) max_val = logits[i];
    double sum_exp = 0.0;
    for (int i = 0; i < dim; i++) sum_exp += expf(logits[i] - max_val);
    double entropy = 0.0;
    for (int i = 0; i < dim; i++) {
        double p = expf(logits[i] - max_val) / sum_exp;
        if (p > 0) entropy -= p * log(p);
    }

    fprintf(stderr, "\n[logit-diag] step=%d token=%d (\"%s\") entropy=%.4f max_logit=%.4f\n",
            gen_step, token_idx,
            decode_token(NULL, token_idx) ? decode_token(NULL, token_idx) : "?",
            entropy, max_val);
    fprintf(stderr, "[logit-diag] top-20: ");
    for (int i = 0; i < DIAG_TOPK; i++) {
        const char *s = decode_token(NULL, top_idx[i]);
        fprintf(stderr, "%s%.4f(%s%s)", i > 0 ? " " : "",
                top_val[i], s ? s : "?", i < DIAG_TOPK - 1 ? "," : "");
    }
    fprintf(stderr, "\n");
    #undef DIAG_TOPK
}

// Temperature sampling with top-k: softmax(logits/T), pick from top K.
// Uses static buffers to avoid per-token malloc (called on every generation step).
// Applies repetition penalty before sampling to prevent attractor loops.
// N-gram repetition blocker state: remembers the last NG_WIN generated
// 2-grams so a candidate completing a 2-gram seen >= 2x recently (or a
// 3-in-a-row run) can be blocked. Guardrail for the temp-sampling path:
// soft rep penalty alone cannot dislodge mid-block repetition loops
// (observed: a ~40-token segment re-emitted 3x at T=0.8).
#define NG_WIN 48
static int ng_last[2] = {-1, -1};
static int ng_win[2][NG_WIN];
static int ng_win_pos = 0;
static int ng_win_fill = 0;

static int ng_blocked(int tok) {
    // 3-in-a-row check
    if (tok == ng_last[0] && tok == ng_last[1]) return 1;
    // 2-gram recurrence within the window
    int dup = 0;
    for (int i = 0; i < ng_win_fill; i++) {
        if (ng_win[0][i] == ng_last[0] && ng_win[1][i] == tok) {
            dup++;
            if (dup >= 2) return 1;
        }
    }
    return 0;
}

static void ng_register(int tok) {
    ng_win[0][ng_win_pos] = ng_last[0];
    ng_win[1][ng_win_pos] = tok;
    ng_win_pos = (ng_win_pos + 1) % NG_WIN;
    if (ng_win_fill < NG_WIN) ng_win_fill++;
    ng_last[1] = ng_last[0];
    ng_last[0] = tok;
}

// Reset all sampler scratch state (n-gram blocker + rep-penalty ring) at the
// start of each generation. Without this, the serve loop leaks one request's
// n-grams into the next: greedy (T=0) sampling is stateful and the same
// prompt gives DIFFERENT completions depending on request order (found via
// the HumanEval harness: identical requests returned 1/20/8 tokens).
static void sampler_state_reset(void) {
    ng_last[0] = ng_last[1] = -1;
    ng_win_pos = 0;
    ng_win_fill = 0;
    g_rep_pos = 0;
    g_rep_count = 0;
}

// Greedy selection with hard n-gram blocking: argmax, and if the winner is
// blocked, -INF it and repeat until an unblocked token is found.
static int sample_greedy_blocked(const float *logits, int dim) {
    for (;;) {
        int chosen = cpu_argmax(logits, dim);
        if (!ng_blocked(chosen)) {
            ng_register(chosen);
            rep_penalty_register(chosen);
            return chosen;
        }
        ((float *)logits)[chosen] = -INFINITY;
    }
}

static int cpu_sample_temp(const float *x, int dim, float temp, int top_k) {
    // Mutable logits buffer (repetition penalty + argmax needs it)
    static float *logits_buf = NULL;
    if (!logits_buf) logits_buf = malloc(VOCAB_SIZE * sizeof(float));
    memcpy(logits_buf, x, dim * sizeof(float));

    // Apply repetition penalty BEFORE argmax/temperature sampling
    rep_penalty_apply(logits_buf, dim);

    if (temp <= 0.0f || top_k <= 1) {
        return sample_greedy_blocked(logits_buf, dim);
    }

    // Static buffers — VOCAB_SIZE is fixed at compile time
    static float *probs = NULL;
    static float *heap = NULL;   // min-heap of top-k probs
    static int   *heap_idx = NULL;
    if (!probs) {
        probs    = malloc(VOCAB_SIZE * sizeof(float));
        heap     = malloc(64 * sizeof(float));  // top_k won't exceed 64
        heap_idx = malloc(64 * sizeof(int));
    }

    // Find max for numerical stability
    float max_val = logits_buf[0];
    for (int i = 1; i < dim; i++) if (logits_buf[i] > max_val) max_val = logits_buf[i];

    // Compute exp(x/T - max/T) and find top-k threshold via min-heap
    float inv_t = 1.0f / temp;
    float sum = 0.0f;
    int heap_n = 0;  // current heap size
    float threshold = 0.0f;

    for (int i = 0; i < dim; i++) {
        float p = expf((logits_buf[i] - max_val) * inv_t);
        probs[i] = p;
        sum += p;

        // Maintain min-heap of top-k probabilities
        if (heap_n < top_k) {
            // Insert: bubble up
            int pos = heap_n++;
            while (pos > 0 && p < heap[(pos-1)/2]) {
                heap[pos] = heap[(pos-1)/2];
                heap_idx[pos] = heap_idx[(pos-1)/2];
                pos = (pos-1)/2;
            }
            heap[pos] = p;
            heap_idx[pos] = i;
        } else if (p > heap[0]) {
            // Replace min: bubble down
            int pos = 0;
            for (;;) {
                int left = 2*pos + 1, right = 2*pos + 2, smallest = pos;
                if (left < heap_n && heap[left] < heap[smallest]) smallest = left;
                if (right < heap_n && heap[right] < heap[smallest]) smallest = right;
                if (smallest == pos) break;
                heap[pos] = heap[smallest];
                heap_idx[pos] = heap_idx[smallest];
                pos = smallest;
            }
            heap[pos] = p;
            heap_idx[pos] = i;
        }
    }

    if (heap_n > 0) threshold = heap[0];  // k-th largest prob

    // N-gram blocker on the top-k candidates only (checking all 248K vocab
    // tokens would cost ~12M ops/step; the loop tokens are always top-k).
    // Blocked candidates get prob 0 and drop out of the distribution.
    for (int k = 0; k < heap_n; k++) {
        if (ng_blocked(heap_idx[k])) {
            probs[heap_idx[k]] = 0.0f;
        }
    }

    // min_p tail filter: zero every token whose probability is below
    // min_p * p_max (the strongest surviving candidate). This removes the
    // low-probability tail that long-form generation drifts into (the
    // synonym-wandering loops) — the llama.cpp-style cure.
    if (g_min_p > 0.0f && g_min_p < 1.0f) {
        float p_max = 0.0f;
        for (int k = 0; k < heap_n; k++) {
            float p = probs[heap_idx[k]];
            if (p > p_max) p_max = p;
        }
        float cutoff = g_min_p * p_max;
        for (int k = 0; k < heap_n; k++) {
            if (probs[heap_idx[k]] < cutoff) probs[heap_idx[k]] = 0.0f;
        }
    }

    // Zero out below threshold, recompute sum
    sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        if (probs[i] < threshold) probs[i] = 0.0f;
        else sum += probs[i];
    }

    // All top-k candidates blocked — fall back to hard-blocked greedy.
    if (sum <= 0.0f) {
        return sample_greedy_blocked(logits_buf, dim);
    }

    // Sample from distribution
    float r = (float)drand48() * sum;
    float cumsum = 0.0f;
    int chosen = 0;
    for (int i = 0; i < dim; i++) {
        cumsum += probs[i];
        if (r <= cumsum) { chosen = i; break; }
    }
    ng_register(chosen);
    rep_penalty_register(chosen);
    return chosen;
}

// SiLU activation
static void cpu_silu(float *x, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// Conv1d depthwise: one step (for incremental inference)
// Input: conv_state[kernel_size-1][channels] + new_input[channels]
// Output: result[channels]
// Weight: [channels, kernel_size, 1] stored as bf16
// This is a depthwise conv1d: each channel is independent
static void cpu_conv1d_step(
    const float *conv_state,    // [(kernel_size-1) * channels] row-major
    const float *new_input,     // [channels]
    const uint16_t *weight_bf16, // [channels * kernel_size] flattened
    float *out,                 // [channels]
    int channels,
    int kernel_size
) {
    // For each channel, compute dot product of [conv_state..., new_input] with weight
    for (int c = 0; c < channels; c++) {
        float acc = 0.0f;
        // Process previous states from conv_state
        for (int k = 0; k < kernel_size - 1; k++) {
            float w = bf16_to_f32(weight_bf16[c * kernel_size + k]);
            acc += conv_state[k * channels + c] * w;
        }
        // Process new input (last position in kernel)
        float w = bf16_to_f32(weight_bf16[c * kernel_size + (kernel_size - 1)]);
        acc += new_input[c] * w;
        out[c] = acc;
    }
    // Apply SiLU
    cpu_silu(out, channels);
}

// ============================================================================
// Metal context for GPU-accelerated matmuls
// ============================================================================

// Maximum number of batched matmul output slots.
// Used for encoding multiple matmuls into one command buffer.
#define MAX_BATCH_SLOTS 8

typedef struct {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLLibrary>              library;
    id<MTLComputePipelineState> matvec_v3;
    id<MTLComputePipelineState> matvec_v5;  // LUT dequant variant
    id<MTLComputePipelineState> matvec_fast;  // for in_dim > 4096
    id<MTLComputePipelineState> gemv_bf16_pipe;    // raw BF16 GEMV (no dequant)
    id<MTLComputePipelineState> gemv_bf16_x2_pipe; // 2 rows/tg (NR0=2), faster for large out_dim
    // Prefill-batched variants (M positions per dispatch, bitwise-identical math)
    id<MTLComputePipelineState> matvec_prefill_4bit;
    id<MTLComputePipelineState> matvec_prefill_8bit;
    id<MTLComputePipelineState> gemv_bf16_prefill;
    id<MTLComputePipelineState> residual_norm_fused_prefill;
    id<MTLComputePipelineState> rms_norm_sum_sq_prefill;
    id<MTLComputePipelineState> rms_norm_apply_bf16_prefill;
    id<MTLComputePipelineState> moe_combine_residual_prefill;
    id<MTLComputePipelineState> matvec_1bit;  // 1-bit expert dequant kernel
    id<MTLComputePipelineState> matvec_2bit;  // 2-bit expert dequant kernel
    id<MTLComputePipelineState> matvec_3bit;  // 3-bit expert dequant kernel
    id<MTLComputePipelineState> matvec_8bit;  // 8-bit expert dequant kernel
    id<MTLComputePipelineState> matvec_qk;    // Phase C: GGUF Q4_K/Q6_K dequant matvec
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pipe;  // Phase C S2: GGUF expert gate+up+SwiGLU
    id<MTLComputePipelineState> matvec_qk_prefill;  // Phase C S3: batched QK prefill matvec
    // Phase C: per-tensor page-aligned zero-copy wraps for GGUF Q4_K/Q6_K
    // tensors. The whole 21.7GB mmap can't be wrapped as one Metal buffer
    // (the driver rejects ~>8GB), and the tensors are only 32-byte aligned —
    // so each wrap starts at the tensor's 16KB page floor and the kernel gets
    // the sub-page offset via setBuffer:offset:.
    #define MAX_GGUF_TBUFS 4096   // tensors (~520) + expert slabs (40 layers × 8 × 3 = 960)
    struct { size_t off; id<MTLBuffer> buf; } gguf_tbufs[MAX_GGUF_TBUFS];
    int gguf_tbuf_count;
    // Phase C S4: chunked GGUF prefill infrastructure
    id<MTLBuffer> gguf_stage_gpu;            // zero-copy wrap of the staged-BF16 heap buffer
    id<MTLBuffer> buf_pool_expert_data_gguf; // GGUF expert-slab pool (g_gguf_exp_alloc slots)
    // Per-position delta-net scratch for the chunked GGUF chain's batched
    // recurrence. [PREFILL_CHUNK_MAX × N] floats (~19 MB at 256 positions).
    id<MTLBuffer> buf_pf_delta_q, buf_pf_delta_k, buf_pf_delta_v;
    id<MTLBuffer> buf_pf_delta_g_decay, buf_pf_delta_beta, buf_pf_delta_out;
    id<MTLComputePipelineState> fused_gate_up_swiglu_pipe;      // 4-bit fused gate+up+swiglu
    id<MTLComputePipelineState> fused_gate_up_swiglu_8bit_pipe; // 8-bit fused gate+up+swiglu
    id<MTLComputePipelineState> fused_gate_up_swiglu_2x_pipe;   // 2-expert fused gate+up+swiglu
    id<MTLComputePipelineState> rms_norm_sum;
    id<MTLComputePipelineState> rms_norm_apply;
    id<MTLComputePipelineState> rms_norm_apply_bf16;
    id<MTLComputePipelineState> residual_add;
    id<MTLComputePipelineState> residual_norm_fused;  // residual_add + rms_norm in one dispatch
    id<MTLComputePipelineState> routing_batch_fused;  // gate+sg+su+seg in one dispatch
    id<MTLComputePipelineState> swiglu;
    // GPU attention pipelines
    id<MTLComputePipelineState> attn_scores_pipe;
    id<MTLComputePipelineState> attn_softmax_pipe;
    id<MTLComputePipelineState> attn_values_pipe;
    id<MTLComputePipelineState> sigmoid_gate_pipe;
    // Reusable buffers for attention matmuls
    id<MTLBuffer> buf_input;     // input vector [HIDDEN_DIM or max projection input]
    id<MTLBuffer> buf_output;    // output vector [max projection output]
    id<MTLBuffer> wf_buf;        // the mmap'd weight file as a Metal buffer
    // Batched matmul output slots (preallocated, reused across dispatches)
    id<MTLBuffer> batch_out[MAX_BATCH_SLOTS];
    // Reusable buffers for expert computation (avoids per-expert alloc)
    // Legacy single-expert buffers (kept for gpu_expert_forward compat)
    id<MTLBuffer> buf_expert_data;   // holds one expert's packed weights (EXPERT_SIZE_MAX bytes)
    id<MTLBuffer> buf_expert_input;  // h_post input [HIDDEN_DIM floats]
    id<MTLBuffer> buf_expert_gate;   // gate_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_up;     // up_proj output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_act;    // SwiGLU output [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_expert_out;    // down_proj output [HIDDEN_DIM floats]
    // Multi-expert buffers: K independent sets so all experts can be encoded
    // into a SINGLE command buffer (no per-expert commit+wait).
    // Each expert k uses slot [k].
    // Double-buffered: set A (data) for GPU compute, set B (data_B) for background pread.
    // Gate/up/act/out only need one set (GPU uses them after pread completes).
    #define MAX_K 8
    id<MTLBuffer> buf_multi_expert_data[MAX_K];   // [EXPERT_SIZE_MAX bytes] each — buffer set A
    id<MTLBuffer> buf_multi_expert_data_B[MAX_K]; // [EXPERT_SIZE_MAX bytes] each — buffer set B (prefetch)
    id<MTLBuffer> buf_multi_expert_gate[MAX_K];   // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_up[MAX_K];     // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_act[MAX_K];    // [MOE_INTERMEDIATE floats]
    id<MTLBuffer> buf_multi_expert_out[MAX_K];    // [HIDDEN_DIM floats]
    id<MTLBuffer> buf_multi_expert_input;         // [HIDDEN_DIM floats] (shared, read-only during dispatch)
    // Shared expert buffers for fused CMD2 (shared gate/up computed in CMD1,
    // SwiGLU + down_proj in CMD2 alongside routed experts)
    id<MTLBuffer> buf_shared_gate;   // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_up;     // [SHARED_INTERMEDIATE floats]
    id<MTLBuffer> buf_shared_act;    // [SHARED_INTERMEDIATE floats] (SwiGLU output)
    id<MTLBuffer> buf_shared_out;    // [HIDDEN_DIM floats] (down_proj output)
    // Fused o_proj+norm+routing buffers (eliminates 1 cmd buffer per layer)
    id<MTLBuffer> buf_residual;     // [HIDDEN_DIM floats] holds residual for GPU add
    id<MTLBuffer> buf_h_mid;        // [HIDDEN_DIM floats] residual+oproj result
    id<MTLBuffer> buf_sum_sq;       // [1 float] for RMS norm reduction
    // GPU attention buffers (for full attention layers)
    #define NUM_FULL_ATTN_LAYERS 10
    id<MTLBuffer> buf_kv_k[NUM_FULL_ATTN_LAYERS];  // K cache per full-attn layer
    id<MTLBuffer> buf_kv_v[NUM_FULL_ATTN_LAYERS];  // V cache per full-attn layer
    id<MTLBuffer> buf_attn_q;       // [NUM_ATTN_HEADS * HEAD_DIM floats] all query heads
    id<MTLBuffer> buf_attn_scores;  // [NUM_ATTN_HEADS * g_gpu_kv_seq floats] all heads' scores
    id<MTLBuffer> buf_attn_out;     // [NUM_ATTN_HEADS * HEAD_DIM floats] full attention output
    id<MTLBuffer> buf_attn_gate;    // [NUM_ATTN_HEADS * HEAD_DIM floats] sigmoid gate
    // Batched-attention prefill pipelines + buffers (M queries per dispatch)
    id<MTLComputePipelineState> attn_scores_prefill_pipe;
    id<MTLComputePipelineState> attn_softmax_prefill_pipe;
    id<MTLComputePipelineState> attn_values_prefill_pipe;
    id<MTLComputePipelineState> sigmoid_gate_prefill_pipe;
    id<MTLBuffer> buf_pf_attn_q;      // [PF_ATTN_MAX, 4096] staged post-norm Q
    id<MTLBuffer> buf_pf_attn_gate;   // [PF_ATTN_MAX, 4096] staged raw q_gate
    id<MTLBuffer> buf_pf_attn_scores; // [PF_ATTN_MAX, 16, g_gpu_kv_seq]
    // CMD3 GPU-side combine buffers (weighted_sum + residual + norm on GPU)
    id<MTLComputePipelineState> moe_combine_residual;  // fused combine kernel
    id<MTLBuffer> buf_moe_hidden;     // [HIDDEN_DIM floats] GPU combine output (hidden state)
    id<MTLBuffer> buf_combine_params; // [10 floats] expert weights[8] + shared_gate_score + padding
    id<MTLBuffer> buf_cmd3_sum_sq;    // [1 float] for RMS norm reduction in CMD3
    // Shared event for CPU-GPU synchronization (async pipeline)
    id<MTLSharedEvent> pipeline_event;   // CPU signals when buf_input is ready
    uint64_t event_value;                // monotonically increasing event counter
    // GPU delta-net (gated_delta_net_step) and conv1d pipelines
    id<MTLComputePipelineState> delta_net_step;  // gated_delta_net_step kernel
    id<MTLComputePipelineState> conv1d_step;     // conv1d_step kernel
    id<MTLComputePipelineState> rms_norm_qk;     // per-head RMS normalize for q and k
    id<MTLComputePipelineState> compute_decay_beta; // g_decay and beta_gate for delta-net
    id<MTLComputePipelineState> gated_rms_norm;  // z-gated output normalization
    id<MTLComputePipelineState> fused_gdn_core;  // fused decay+beta+GDN+gated_norm
    id<MTLComputePipelineState> fused_gdn_full;  // conv1d+qk-norm+GDN+gated_norm, single kernel
    id<MTLComputePipelineState> fused_gdn_batched;  // fused_gdn_full with in-kernel M loop (chunked prefill)
    id<MTLComputePipelineState> fused_gdn_batched_qk;  // Phase C S5: GGUF Q4_K in_proj variant (env-gated)
    // Phase C S4.1: M-batched CMD3 kernels (GGUF group dispatch)
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pool_pipe;
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pool_gateonly_pipe;  // perf probe
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pool_barrier_pipe;   // perf probe
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pool_xstage_pipe;    // perf probe
    id<MTLComputePipelineState> fused_gate_up_swiglu_qk_pool_nox_pipe;       // perf probe
    id<MTLComputePipelineState> matvec_qk_pool_prefill_pipe;
    id<MTLComputePipelineState> ka_nop_pipe;   // S6 keep-alive ticker (GPU clock)
    id<MTLComputePipelineState> swiglu_prefill_batch_pipe;
    id<MTLComputePipelineState> rms_norm_sum_sq_prefill_batch_pipe;
    id<MTLComputePipelineState> rms_norm_apply_bf16_prefill_batch_pipe;
    id<MTLComputePipelineState> moe_combine_residual_prefill_batch_pipe;
    // Persistent GPU state buffers for linear attention layers
    #define NUM_LINEAR_LAYERS 30
    id<MTLBuffer> buf_delta_state[NUM_LINEAR_LAYERS];   // [32*128*128] float per layer
    id<MTLBuffer> buf_conv_state[NUM_LINEAR_LAYERS];     // [3*LINEAR_CONV_DIM] float per layer (v-channels)
    id<MTLBuffer> buf_conv_qk[NUM_LINEAR_LAYERS];       // per-head q/k conv histories [2*32*3*key_dim]
    // Scratch buffers for delta-net inputs/outputs
    id<MTLBuffer> buf_delta_q;        // [2048] float
    id<MTLBuffer> buf_delta_k;        // [2048] float
    id<MTLBuffer> buf_delta_v;        // [8192] float
    id<MTLBuffer> buf_delta_g_decay;  // [64] float
    id<MTLBuffer> buf_delta_beta;     // [64] float
    id<MTLBuffer> buf_delta_output;   // [8192] float
    id<MTLBuffer> buf_conv_input;     // [LINEAR_CONV_DIM] float
    id<MTLBuffer> buf_conv_output;    // [LINEAR_CONV_DIM] float
    // Prefill-batched scratch buffers ([PREFILL_CHUNK_MAX, dim] floats each)
    id<MTLBuffer> buf_pf_input;          // [256, 2048]  normed layer input
    id<MTLBuffer> buf_pf_residual;       // [256, 2048]  layer-0 residual (embed batch)
    id<MTLBuffer> buf_pf_qkv;            // [256, 8192]  linear qkv / full q
    id<MTLBuffer> buf_pf_kv;             // [256, 1024]  full k [..,512) + v [512,..)
    id<MTLBuffer> buf_pf_z;              // [256, 4096]  linear z
    id<MTLBuffer> buf_pf_ba;             // [256, 64]    linear beta [..,32) + alpha [32,..)
    id<MTLBuffer> buf_pf_oproj_in;       // [256, 4096]  attention output
    id<MTLBuffer> buf_pf_oproj_in2;      // CPU-copy bridge: out_proj reads this (no dev-to-dev L2 dependency)
    id<MTLBuffer> buf_pf_oproj;          // [256, 2048]  o_proj output
    id<MTLBuffer> buf_pf_h_mid;          // [256, 2048]  residual + o_proj
    id<MTLBuffer> buf_pf_h_post;         // [256, 2048]  post-norm (= routing input)
    id<MTLBuffer> buf_pf_gate_scores;    // [256, 256]   routing gate scores
    id<MTLBuffer> buf_pf_shared;         // [256, 1024]  shared gate [..,512) + up [512,..)
    id<MTLBuffer> buf_pf_seg;            // [256]        shared expert gate score
    id<MTLBuffer> buf_pf_moe_hidden;     // [256, 2048]  combine output (= next input)
    id<MTLBuffer> buf_pf_combine_params; // [256, 10]    weights[8] + seg + pad
    // Pool-mode expert pread buffers (sized by g_pf_pool_slots; 0 = pool mode
    // disabled). Position m's expert k data lives at pool slot 8m+k.
    id<MTLBuffer> buf_pool_expert_data;   // [P × expert_alloc_size] one buffer
    // Prefill hot-set prefetch pools (2 × g_pf_hot_slots, alternating per
    // layer: hot(L) lives in pool[L%2], prefetched one layer ahead).
    id<MTLBuffer> buf_prefetch_pool[2];
    id<MTLBuffer> buf_pf_expert_input;    // [P, 2048]
    id<MTLBuffer> buf_pf_expert_gate[MAX_K];  // [P, 512]
    id<MTLBuffer> buf_pf_expert_up[MAX_K];    // [P, 512]
    id<MTLBuffer> buf_pf_expert_act[MAX_K];   // [P, 512]
    id<MTLBuffer> buf_pf_shared_gate;     // [P, 512]
    id<MTLBuffer> buf_pf_shared_up;       // [P, 512]
    id<MTLBuffer> buf_pf_shared_act;      // [P, 512]
    id<MTLBuffer> buf_pf_sum_sq;
    // Inter-command-buffer synchronization for fused expert path
    id<MTLFence>      expert_fence;        // MTLFence: encoder-level GPU sync
    id<MTLSharedEvent> expert_sync_event;   // MTLSharedEvent: CB-level GPU sync
    uint64_t           expert_sync_value;   // monotonic counter for expert_sync_event
} MetalCtx;

static MetalCtx *g_metal = NULL;

static MetalCtx *metal_setup(void) {
    MetalCtx *ctx = calloc(1, sizeof(MetalCtx));
    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "ERROR: No Metal device\n");
        free(ctx); return NULL;
    }
    printf("[metal] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "ERROR: No command queue\n");
        free(ctx); return NULL;
    }

    // Compile shaders from source
    NSError *error = nil;
    NSArray *paths = @[@"shaders.metal", @"metal_infer/shaders.metal"];
    NSString *src = nil;
    for (NSString *p in paths) {
        src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:&error];
        if (src) break;
    }
    if (!src) {
        fprintf(stderr, "ERROR: Cannot find shaders.metal\n");
        free(ctx); return NULL;
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_1;
    double t0 = now_ms();
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "FATAL: Shader compile failed: %s\n",
                [[error localizedDescription] UTF8String]);
        free(ctx); return NULL;
    }
    if (error) {
        // Non-fatal warnings from Metal compiler (e.g. variable initialization)
        fprintf(stderr, "[metal] Shader warning: %s\n",
                [[error localizedDescription] UTF8String]);
    }
    printf("[metal] Shader compile: %.0f ms\n", now_ms() - t0);

    // Create pipelines
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "ERROR: shader '%s' not found\n", [name UTF8String]); return (id<MTLComputePipelineState>)nil; }
        NSError *e2 = nil;
        id<MTLComputePipelineState> ps = [ctx->device newComputePipelineStateWithFunction:fn error:&e2];
        if (!ps) { fprintf(stderr, "ERROR: pipeline '%s': %s\n", [name UTF8String], [[e2 localizedDescription] UTF8String]); }
        return ps;
    };

    ctx->matvec_v3     = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_v5     = makePipe(@"dequant_matvec_4bit_v5");  // LUT variant (no uint→float conversions)
    ctx->matvec_fast   = makePipe(@"dequant_matvec_4bit_fast");
    ctx->matvec_1bit   = makePipe(@"dequant_matvec_1bit");
    ctx->matvec_2bit   = makePipe(@"dequant_matvec_2bit");
    ctx->matvec_3bit   = makePipe(@"dequant_matvec_3bit");
    ctx->matvec_qk     = makePipe(@"dequant_matvec_qk");      // Phase C: GGUF Q4_K/Q6_K
    ctx->fused_gate_up_swiglu_qk_pipe = makePipe(@"fused_gate_up_swiglu_qk");  // Phase C S2
    ctx->matvec_qk_prefill = makePipe(@"dequant_matvec_qk_prefill");            // Phase C S3
    ctx->matvec_8bit   = makePipe(@"dequant_matvec_8bit");
    ctx->fused_gate_up_swiglu_pipe      = makePipe(@"fused_gate_up_swiglu");
    ctx->fused_gate_up_swiglu_8bit_pipe = makePipe(@"fused_gate_up_swiglu_8bit");
    ctx->fused_gate_up_swiglu_2x_pipe = makePipe(@"fused_gate_up_swiglu_2x");
    ctx->gemv_bf16_pipe = makePipe(@"gemv_bf16");
    ctx->gemv_bf16_x2_pipe = makePipe(@"gemv_bf16_x2");
    ctx->rms_norm_sum  = makePipe(@"rms_norm_sum_sq");
    ctx->rms_norm_apply = makePipe(@"rms_norm_apply");
    ctx->rms_norm_apply_bf16 = makePipe(@"rms_norm_apply_bf16");
    ctx->residual_add  = makePipe(@"residual_add");
    ctx->residual_norm_fused = makePipe(@"residual_norm_fused");
    ctx->routing_batch_fused = makePipe(@"routing_batch_fused");
    ctx->swiglu        = makePipe(@"swiglu_fused");
    ctx->attn_scores_pipe  = makePipe(@"attn_scores_batched");
    ctx->attn_softmax_pipe = makePipe(@"attn_softmax_batched");
    ctx->attn_values_pipe  = makePipe(@"attn_values_batched");
    ctx->sigmoid_gate_pipe = makePipe(@"sigmoid_gate");
    ctx->attn_scores_prefill_pipe  = makePipe(@"attn_scores_batched_prefill");
    ctx->attn_softmax_prefill_pipe = makePipe(@"attn_softmax_batched_prefill");
    ctx->attn_values_prefill_pipe  = makePipe(@"attn_values_batched_prefill");
    ctx->sigmoid_gate_prefill_pipe = makePipe(@"sigmoid_gate_prefill");
    ctx->moe_combine_residual = makePipe(@"moe_combine_residual");
    ctx->delta_net_step    = makePipe(@"gated_delta_net_step");
    ctx->conv1d_step       = makePipe(@"conv1d_step");
    ctx->rms_norm_qk       = makePipe(@"rms_norm_qk");
    ctx->compute_decay_beta = makePipe(@"compute_decay_beta");
    ctx->gated_rms_norm    = makePipe(@"gated_rms_norm");
    ctx->fused_gdn_core    = makePipe(@"fused_gdn_core");
    ctx->fused_gdn_full    = makePipe(@"fused_gdn_full");
    ctx->fused_gdn_batched = makePipe(@"fused_gdn_batched");
    ctx->fused_gdn_batched_qk = makePipe(@"fused_gdn_batched_qk");
    ctx->fused_gate_up_swiglu_qk_pool_pipe = makePipe(@"fused_gate_up_swiglu_qk_pool_prefill");
    ctx->fused_gate_up_swiglu_qk_pool_gateonly_pipe = makePipe(@"fused_gate_up_swiglu_qk_pool_gateonly");
    ctx->fused_gate_up_swiglu_qk_pool_barrier_pipe = makePipe(@"fused_gate_up_swiglu_qk_pool_barrier");
    ctx->fused_gate_up_swiglu_qk_pool_xstage_pipe = makePipe(@"fused_gate_up_swiglu_qk_pool_xstage");
    ctx->fused_gate_up_swiglu_qk_pool_nox_pipe = makePipe(@"fused_gate_up_swiglu_qk_pool_nox");
    ctx->matvec_qk_pool_prefill_pipe       = makePipe(@"dequant_matvec_qk_pool_prefill");
    ctx->ka_nop_pipe                       = makePipe(@"ka_nop");
    ctx->swiglu_prefill_batch_pipe         = makePipe(@"swiglu_prefill_batch");
    ctx->rms_norm_sum_sq_prefill_batch_pipe = makePipe(@"rms_norm_sum_sq_prefill_batch");
    ctx->rms_norm_apply_bf16_prefill_batch_pipe = makePipe(@"rms_norm_apply_bf16_prefill_batch");
    ctx->moe_combine_residual_prefill_batch_pipe = makePipe(@"moe_combine_residual_prefill_batch");
    ctx->matvec_prefill_4bit  = makePipe(@"dequant_matvec_4bit_prefill");
    ctx->matvec_prefill_8bit  = makePipe(@"dequant_matvec_8bit_prefill");
    ctx->gemv_bf16_prefill    = makePipe(@"gemv_bf16_prefill");
    ctx->residual_norm_fused_prefill = makePipe(@"residual_norm_fused_prefill");
    ctx->rms_norm_sum_sq_prefill      = makePipe(@"rms_norm_sum_sq_prefill");
    ctx->rms_norm_apply_bf16_prefill  = makePipe(@"rms_norm_apply_bf16_prefill");
    ctx->moe_combine_residual_prefill = makePipe(@"moe_combine_residual_prefill");
    if (!ctx->moe_combine_residual) fprintf(stderr, "[metal] WARNING: moe_combine_residual pipeline failed\n");
    if (!ctx->delta_net_step) fprintf(stderr, "[metal] WARNING: gated_delta_net_step pipeline failed (CPU fallback)\n");
    if (!ctx->conv1d_step)    fprintf(stderr, "[metal] WARNING: conv1d_step pipeline failed (CPU fallback)\n");
    if (!ctx->rms_norm_qk)       fprintf(stderr, "[metal] WARNING: rms_norm_qk pipeline failed (CPU fallback)\n");
    if (!ctx->compute_decay_beta) fprintf(stderr, "[metal] WARNING: compute_decay_beta pipeline failed (CPU fallback)\n");
    if (!ctx->gated_rms_norm)     fprintf(stderr, "[metal] WARNING: gated_rms_norm pipeline failed (CPU fallback)\n");

    if (!ctx->matvec_v3 || !ctx->matvec_fast) {
        fprintf(stderr, "ERROR: Required Metal pipeline missing\n");
        free(ctx); return NULL;
    }

    // Allocate reusable buffers (large enough for biggest projection)
    // Q proj output is 16384 floats, lm_head output is 248320 floats
    // o_proj input is 8192, linear attn out_proj input is 8192
    size_t max_out = VOCAB_SIZE * sizeof(float);  // lm_head is largest
    size_t max_in = LINEAR_TOTAL_VALUE * sizeof(float);  // 4096 floats (linear_attn out_proj)
    if (max_in < (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float)) {
        max_in = (size_t)(NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float);  // o_proj input = 8192
    }
    ctx->buf_input  = [ctx->device newBufferWithLength:max_in  options:MTLResourceStorageModeShared];
    ctx->buf_output = [ctx->device newBufferWithLength:max_out options:MTLResourceStorageModeShared];

    // Batched matmul output slots — each large enough for the biggest projection
    // q_proj = 16384 floats, qkv_proj = 12288, z_proj = 8192, o_proj = 4096
    // lm_head (248320) uses buf_output directly, not batched.
    {
        size_t slot_size = (size_t)(NUM_ATTN_HEADS * HEAD_DIM * 2) * sizeof(float);  // 16384 floats
        if (slot_size < (size_t)LINEAR_CONV_DIM * sizeof(float))
            slot_size = (size_t)LINEAR_CONV_DIM * sizeof(float);  // 8192 floats
        for (int i = 0; i < MAX_BATCH_SLOTS; i++) {
            ctx->batch_out[i] = [ctx->device newBufferWithLength:slot_size
                                                         options:MTLResourceStorageModeShared];
        }
    }

    // Expert computation buffers (reused across all experts and layers)
    ctx->buf_expert_data  = [ctx->device newBufferWithLength:EXPERT_SIZE_MAX
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_gate  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_up    = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_act   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                     options:MTLResourceStorageModeShared];
    ctx->buf_expert_out   = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Multi-expert buffers: K independent slots (double-buffered data)
    // Expert data buffers use 2MB-aligned backing memory for DMA efficiency.
    // The pread DMA controller transfers 3.6x faster with 2MB alignment vs 16KB.
    ctx->buf_multi_expert_input = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    size_t expert_alloc_size = (EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);  // round up to 2MB
    for (int k = 0; k < MAX_K; k++) {
        // 2MB-aligned allocation for optimal DMA throughput
        void *aligned_data = NULL, *aligned_data_b = NULL;
        posix_memalign(&aligned_data,   2*1024*1024, expert_alloc_size);
        posix_memalign(&aligned_data_b, 2*1024*1024, expert_alloc_size);
        memset(aligned_data, 0, expert_alloc_size);
        memset(aligned_data_b, 0, expert_alloc_size);
        ctx->buf_multi_expert_data[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data
                                                                       length:expert_alloc_size
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
        ctx->buf_multi_expert_data_B[k] = [ctx->device newBufferWithBytesNoCopy:aligned_data_b
                                                                         length:expert_alloc_size
                                                                        options:MTLResourceStorageModeShared
                                                                    deallocator:nil];
        ctx->buf_multi_expert_gate[k] = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_up[k]   = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        ctx->buf_multi_expert_act[k]  = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        // Sized for prefill chunk slots ([PREFILL_CHUNK_MAX, HIDDEN_DIM]);
        // the per-token path only uses offset 0.
        ctx->buf_multi_expert_out[k]  = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * HIDDEN_DIM * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
    }

    // Shared expert buffers (for fused CMD2)
    ctx->buf_shared_gate = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_up   = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    ctx->buf_shared_act  = [ctx->device newBufferWithLength:SHARED_INTERMEDIATE * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    // Sized for prefill chunk slots ([PREFILL_CHUNK_MAX, HIDDEN_DIM]);
    // the per-token path only uses offset 0.
    ctx->buf_shared_out  = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * HIDDEN_DIM * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

    // Pool-mode expert pread buffers (--prefill-chunk N, N > 0). One pool
    // slot per expert (expert_alloc_size each) — position m uses slots
    // [8m..8m+7], so pool mode covers chunks of M <= slots/8 positions.
    // Memory-budget ladder: 64 (256MB) -> 32 -> 16 -> 0 (pool mode off,
    // per-position fallback). Requires ~512MB headroom beyond the pool.
    if (g_prefill_chunk > 0) {
        int p = 64;
        size_t avail = get_available_memory();
        while (p >= 16 && (avail < (size_t)p * expert_alloc_size + (size_t)512 * 1024 * 1024)) p /= 2;
        if (avail < (size_t)16 * expert_alloc_size + (size_t)512 * 1024 * 1024) p = 0;
        g_pf_pool_slots = p;
        if (p > 0) {
            void *pool_aligned = NULL;
            size_t pool_bytes = (size_t)p * expert_alloc_size;
            posix_memalign(&pool_aligned, 2*1024*1024, pool_bytes);
            memset(pool_aligned, 0, pool_bytes);
            ctx->buf_pool_expert_data = [ctx->device newBufferWithBytesNoCopy:pool_aligned
                                                                       length:pool_bytes
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
            size_t P2048 = (size_t)p * HIDDEN_DIM * sizeof(float);
            size_t P512  = (size_t)p * MOE_INTERMEDIATE * sizeof(float);
            ctx->buf_pf_expert_input = [ctx->device newBufferWithLength:P2048 options:MTLResourceStorageModeShared];
            for (int k = 0; k < MAX_K; k++) {
                ctx->buf_pf_expert_gate[k] = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
                ctx->buf_pf_expert_up[k]   = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
                ctx->buf_pf_expert_act[k]  = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            }
            ctx->buf_pf_shared_gate = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            ctx->buf_pf_shared_up   = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            ctx->buf_pf_shared_act  = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            fprintf(stderr, "[pf-pool] %d expert slots (%.0f MB), pool mode for chunks <= %d positions\n",
                    p, (double)pool_bytes / 1e6, p / MAX_K);
        } else {
            fprintf(stderr, "[pf-pool] disabled (insufficient memory) — per-position expert path\n");
        }
        // Prefill hot-set prefetch pools: 2 × hot_slots, alternating per
        // layer (hot(L) in pool[L%2], prefetched during layer L-1's compute).
        // Same memory-budget ladder; hot sets loaded from hot_sets.bin.
        // The prefetch re-reads the hot set once per chunk (redundant reads
        // served from the page cache) — it only pays when the OS has real
        // page-cache headroom, so require ~1 GB strictly-free memory beyond
        // the pools (reclaimable cache doesn't count: that IS the cache).
        // FINCHMOE_PF_PREFETCH=1 forces it on (benchmarking on healthy boxes).
        if (p >= 64) {
            int hp = 32;
            size_t free2 = get_free_memory();
            int force_pf = getenv("FINCHMOE_PF_PREFETCH") != NULL;
            while (!force_pf && hp >= 16 && free2 < (size_t)2 * hp * expert_alloc_size + (size_t)1024 * 1024 * 1024) hp /= 2;
            if (!force_pf && free2 < (size_t)2 * 16 * expert_alloc_size + (size_t)1024 * 1024 * 1024) hp = 0;
            g_pf_hot_slots = hp;
            for (int pi = 0; pi < 2; pi++) {
                if (hp == 0) break;
                void *hot_aligned = NULL;
                size_t hot_bytes = (size_t)hp * expert_alloc_size;
                posix_memalign(&hot_aligned, 2*1024*1024, hot_bytes);
                memset(hot_aligned, 0, hot_bytes);
                ctx->buf_prefetch_pool[pi] = [ctx->device newBufferWithBytesNoCopy:hot_aligned
                                                                             length:hot_bytes
                                                                            options:MTLResourceStorageModeShared
                                                                        deallocator:nil];
            }
            if (hp > 0) {
                fprintf(stderr, "[pf-prefetch] 2 x %d hot slots (%.0f MB), 1-layer-ahead hot-set prefetch\n",
                        hp, (double)2 * hp * expert_alloc_size / 1e6);
            } else {
                fprintf(stderr, "[pf-prefetch] disabled (insufficient page-cache headroom) — miss-only preads\n");
            }
        }
    }

    // Fused o_proj+norm+routing buffers
    ctx->buf_residual = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_h_mid    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    ctx->buf_sum_sq   = [ctx->device newBufferWithLength:sizeof(float)
                                                 options:MTLResourceStorageModeShared];

    // CMD3 GPU-side combine buffers
    ctx->buf_moe_hidden    = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
    ctx->buf_combine_params = [ctx->device newBufferWithLength:10 * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    ctx->buf_cmd3_sum_sq    = [ctx->device newBufferWithLength:sizeof(float)
                                                        options:MTLResourceStorageModeShared];

    // Prefill-batched scratch buffers (--prefill-chunk N). One slot per
    // position m in [0, PREFILL_CHUNK_MAX). ~48MB total at 256 positions.
    {
        size_t PF_2048 = (size_t)PREFILL_CHUNK_MAX * HIDDEN_DIM * sizeof(float);
        size_t PF_4096 = (size_t)PREFILL_CHUNK_MAX * 4096 * sizeof(float);
        size_t PF_8192 = (size_t)PREFILL_CHUNK_MAX * 8192 * sizeof(float);
        // Phase C S4: per-position delta-net scratch (chunked GGUF chain)
        size_t PF_KDIM = (size_t)PREFILL_CHUNK_MAX * LINEAR_TOTAL_KEY * sizeof(float);   // 2048/pos
        size_t PF_VDIM = (size_t)PREFILL_CHUNK_MAX * LINEAR_TOTAL_VALUE * sizeof(float); // 4096/pos
        size_t PF_H32  = (size_t)PREFILL_CHUNK_MAX * LINEAR_NUM_V_HEADS * sizeof(float);
        ctx->buf_pf_input    = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_residual = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_qkv      = [ctx->device newBufferWithLength:PF_8192 options:MTLResourceStorageModeShared];
        ctx->buf_pf_kv       = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * 1024 * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_z        = [ctx->device newBufferWithLength:PF_4096 options:MTLResourceStorageModeShared];
        ctx->buf_pf_ba       = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * 64 * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_q       = [ctx->device newBufferWithLength:PF_KDIM options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_k       = [ctx->device newBufferWithLength:PF_KDIM options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_v       = [ctx->device newBufferWithLength:PF_VDIM options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_g_decay = [ctx->device newBufferWithLength:PF_H32 options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_beta    = [ctx->device newBufferWithLength:PF_H32 options:MTLResourceStorageModeShared];
        ctx->buf_pf_delta_out     = [ctx->device newBufferWithLength:PF_VDIM options:MTLResourceStorageModeShared];
        ctx->buf_pf_oproj_in = [ctx->device newBufferWithLength:PF_4096 options:MTLResourceStorageModeShared];
        ctx->buf_pf_oproj_in2 = [ctx->device newBufferWithLength:PF_4096 options:MTLResourceStorageModeShared];
        ctx->buf_pf_oproj    = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_h_mid    = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_h_post   = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_gate_scores = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * 256 * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_shared   = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * 1024 * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_seg      = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_moe_hidden   = [ctx->device newBufferWithLength:PF_2048 options:MTLResourceStorageModeShared];
        ctx->buf_pf_combine_params = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * 10 * sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_pf_sum_sq   = [ctx->device newBufferWithLength:(size_t)PREFILL_CHUNK_MAX * sizeof(float) options:MTLResourceStorageModeShared];
    }

    // GPU attention buffers
    {
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;  // 512
        size_t kv_cache_size = g_gpu_kv_seq * kv_dim * sizeof(float);
        for (int i = 0; i < NUM_FULL_ATTN_LAYERS; i++) {
            ctx->buf_kv_k[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
            ctx->buf_kv_v[i] = [ctx->device newBufferWithLength:kv_cache_size
                                                        options:MTLResourceStorageModeShared];
        }
        ctx->buf_attn_q      = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_scores = [ctx->device newBufferWithLength:(size_t)NUM_ATTN_HEADS * g_gpu_kv_seq * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_out    = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        ctx->buf_attn_gate   = [ctx->device newBufferWithLength:NUM_ATTN_HEADS * HEAD_DIM * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        // Batched-attention prefill buffers (M queries per dispatch, M <= PF_ATTN_MAX)
        if (g_prefill_chunk > 0) {
            size_t q_dim = NUM_ATTN_HEADS * HEAD_DIM;  // 4096
            ctx->buf_pf_attn_q      = [ctx->device newBufferWithLength:(size_t)PF_ATTN_MAX * q_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
            ctx->buf_pf_attn_gate   = [ctx->device newBufferWithLength:(size_t)PF_ATTN_MAX * q_dim * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
            ctx->buf_pf_attn_scores = [ctx->device newBufferWithLength:(size_t)PF_ATTN_MAX * NUM_ATTN_HEADS * g_gpu_kv_seq * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];
        }
        printf("[metal] GPU attention buffers: %d KV caches (%.1f MB each), scores buf %.1f MB\n",
               NUM_FULL_ATTN_LAYERS, kv_cache_size / 1e6,
               (double)(NUM_ATTN_HEADS * g_gpu_kv_seq * sizeof(float)) / 1e6);
    }

    // Persistent GPU state buffers for delta-net (linear attention layers)
    if (ctx->delta_net_step) {
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            ctx->buf_delta_state[i] = [ctx->device newBufferWithLength:32*128*128*sizeof(float)
                                                               options:MTLResourceStorageModeShared];
            memset([ctx->buf_delta_state[i] contents], 0, 32*128*128*sizeof(float));
            ctx->buf_conv_state[i] = [ctx->device newBufferWithLength:3*LINEAR_CONV_DIM*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_state[i] contents], 0, 3*LINEAR_CONV_DIM*sizeof(float));
            ctx->buf_conv_qk[i] = [ctx->device newBufferWithLength:2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float)
                                                              options:MTLResourceStorageModeShared];
            memset([ctx->buf_conv_qk[i] contents], 0, 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float));
        }
        // Scratch buffers for delta-net inputs/outputs (allocated once, reused)
        ctx->buf_delta_q       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_k       = [ctx->device newBufferWithLength:2048*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_v       = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_delta_g_decay = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_beta    = [ctx->device newBufferWithLength:64*sizeof(float)    options:MTLResourceStorageModeShared];
        ctx->buf_delta_output  = [ctx->device newBufferWithLength:8192*sizeof(float)  options:MTLResourceStorageModeShared];
        ctx->buf_conv_input    = [ctx->device newBufferWithLength:LINEAR_CONV_DIM*sizeof(float) options:MTLResourceStorageModeShared];
        ctx->buf_conv_output   = [ctx->device newBufferWithLength:LINEAR_CONV_DIM*sizeof(float) options:MTLResourceStorageModeShared];
        printf("[metal] Delta-net GPU buffers: %d layers (%.1f MB state + %.1f MB scratch)\n",
               NUM_LINEAR_LAYERS,
               NUM_LINEAR_LAYERS * (32*128*128*4 + 3*8192*4) / 1e6,
               (2048+2048+8192+64+64+8192+12288+12288) * 4 / 1e6);
    }

    // Create shared event for CPU-GPU async pipeline
    ctx->pipeline_event = [ctx->device newSharedEvent];
    ctx->event_value = 0;
    // Create fence + shared event for inter-CB sync: fused expert CB -> combine CB
    ctx->expert_fence = [ctx->device newFence];
    ctx->expert_sync_event = [ctx->device newSharedEvent];
    ctx->expert_sync_value = 0;

    printf("[metal] Inference pipelines ready (multi-expert[%d] + shared buffers allocated)\n", MAX_K);
    return ctx;
}

// Reset delta-net and conv GPU state buffers (call at start of new generation)
static void reset_delta_net_state(void) {
    if (!g_metal || !g_metal->delta_net_step) return;
    for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
        if (g_metal->buf_delta_state[i])
            memset([g_metal->buf_delta_state[i] contents], 0, 32*128*128*sizeof(float));
        if (g_metal->buf_conv_state[i])
            memset([g_metal->buf_conv_state[i] contents], 0, 3*LINEAR_CONV_DIM*sizeof(float));
        if (g_metal->buf_conv_qk[i])
            memset([g_metal->buf_conv_qk[i] contents], 0, 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float));
    }
}

// Wrap the mmap'd weight file as a Metal buffer (zero-copy on unified memory)
// mmap returns page-aligned addresses, Metal requires the same.
// On Apple Silicon, page size is 16KB.
//
// MEMORY SAFETY: This function checks available system memory before wrapping.
// On machines without swap (or with limited swap), wrapping a multi-GB file
// can trigger SIGKILL/jetsam if the kernel can't find enough physical pages.
// When memory is tight, we refuse the wrap and let the caller fall back to
// per-tensor dispatch (slower but won't crash).
static void metal_set_weights(MetalCtx *ctx, void *data, size_t size) {
    // Round size up to page boundary (16KB)
    size_t page_size = 16384;
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    // ---- Memory safety check ----
    size_t avail_mem = get_available_memory();
    size_t needed = aligned_size + METAL_SAFETY_MARGIN_BYTES;

    printf("[metal] Available memory: %s, weight file: %s, safety margin: %s\n",
           format_mem_size(avail_mem), format_mem_size(aligned_size),
           format_mem_size(METAL_SAFETY_MARGIN_BYTES));

    if (avail_mem > 0 && avail_mem < needed) {
        // Tight memory — but on Apple Silicon unified memory, wrapping is
        // zero-copy (just GPU page-table mappings, no physical allocation).
        // Warn but attempt the wrap anyway. Only block if extremely tight
        // (less than 256 MB available — system is about to jetsam anyway).
        if (avail_mem < 256ULL * 1024 * 1024) {
            fprintf(stderr,
                    "\n"
                    "╔══════════════════════════════════════════════════════════════╗\n"
                    "║  MEMORY CRITICAL: < 256MB available — refusing Metal wrap  ║\n"
                    "╠══════════════════════════════════════════════════════════════╣\n"
                    "║  Available:  %7s                                      ║\n"
                    "║  Close apps or restart to restore GPU acceleration.         ║\n"
                    "╚══════════════════════════════════════════════════════════════╝\n"
                    "\n",
                    format_mem_size(avail_mem));
            return;
        }
        fprintf(stderr,
                "[metal] Memory tight (%s available) but attempting GPU wrap anyway\n"
                "        (Apple Silicon unified memory — zero-copy, no physical allocation)\n",
                format_mem_size(avail_mem));
    }

    ctx->wf_buf = [ctx->device newBufferWithBytesNoCopy:data
                                                 length:aligned_size
                                                options:MTLResourceStorageModeShared
                                            deallocator:nil];
    if (!ctx->wf_buf) {
        fprintf(stderr, "WARNING: Cannot wrap weight file as Metal buffer (size=%s)\n",
                format_mem_size(aligned_size));
        fprintf(stderr, "  data=%p, aligned_size=%zu — GPU matmul will fall back to per-tensor dispatch\n",
                data, aligned_size);
    } else {
        printf("[metal] Weight file wrapped as Metal buffer (%s, zero-copy)\n",
               format_mem_size(aligned_size));
    }
}

// GPU dequant matvec: out[out_dim] = W_4bit * x[in_dim]
// W_packed, scales, biases are pointers into mmap'd weight file
// x_f32 is CPU float array, result written back to out_f32
//
// We wrap the ENTIRE mmap'd weight file as a single Metal buffer and use
// byte offsets to point each shader argument at the right tensor.
// This avoids per-tensor buffer creation and the page-alignment constraint.
static void gpu_dequant_matvec(
    MetalCtx *ctx,
    const void *W_packed, const void *scales, const void *biases,
    const float *x_f32, float *out_f32,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size, int bits
) {
    // BF16 guard: GPU kernel is 4-bit only — crash early if BF16 passed
    if (!scales || !biases) {
        fprintf(stderr, "FATAL: gpu_dequant_matvec called with BF16 weights (scales=%p biases=%p)\n",
                (void*)scales, (void*)biases);
        fprintf(stderr, "  out_dim=%u in_dim=%u — BF16 must use cpu_dequant_matvec with bits=0\n",
                out_dim, in_dim);
        abort();
    }
    // Copy input to Metal buffer
    memcpy([ctx->buf_input contents], x_f32, in_dim * sizeof(float));

    size_t o_size = (size_t)out_dim * sizeof(float);

    // Compute offsets into the mmap'd weight buffer
    NSUInteger w_off = (NSUInteger)((const char *)W_packed - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales   - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases   - (const char *)[ctx->wf_buf contents]);

    // Ensure output buffer is large enough
    id<MTLBuffer> o_buf = ctx->buf_output;
    if (o_size > [o_buf length]) {
        o_buf = [ctx->device newBufferWithLength:o_size options:MTLResourceStorageModeShared];
    }

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];

    // v3 shader uses x_shared[4096], so can only handle in_dim <= 4096
    // For larger in_dim (e.g. o_proj with in_dim=8192), use matvec_fast.
    // 8-bit tensors use the tiled 8-bit kernel (same dispatch shape as v3).
    int use_v3 = (in_dim <= 4096);
    if (bits == 8 && ctx->matvec_8bit) {
        [enc setComputePipelineState:ctx->matvec_8bit];
    } else {
        [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    }
    [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
    [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
    [enc setBuffer:o_buf        offset:0     atIndex:4];
    [enc setBytes:&out_dim      length:4     atIndex:5];
    [enc setBytes:&in_dim       length:4     atIndex:6];
    [enc setBytes:&group_size   length:4     atIndex:7];

    if (use_v3 || (bits == 8 && ctx->matvec_8bit)) {
        // v3 / 8-bit: tiled threadgroups, 256 threads, 8 rows per TG
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        // fast: one threadgroup per output row, 64 threads per TG
        NSUInteger tg_size = 64;
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    }
    [enc endEncoding];
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back
    memcpy(out_f32, [o_buf contents], o_size);
}

// ============================================================================
// Phase C: GGUF Q4_K/Q6_K GPU matvec helpers.
// The whole GGUF mmap can't be wrapped as one Metal buffer (the driver
// rejects ~>8GB), so each tensor gets its own page-aligned zero-copy wrap
// covering [page_floor(off), page_ceil(off+size)). The kernel binds the
// buffer with the sub-page offset via setBuffer:offset:.
// ============================================================================
static id<MTLBuffer> gguf_tbuf_get(MetalCtx *ctx, const void *W, size_t size, uint32_t *delta) {
    size_t off = (size_t)((const char *)W - (const char *)g_gguf_data_base);
    size_t page = 16384;
    size_t lo = off & ~(page - 1);
    for (int i = 0; i < ctx->gguf_tbuf_count; i++) {
        if (ctx->gguf_tbufs[i].off == lo) {
            *delta = (uint32_t)(off - lo);
            return ctx->gguf_tbufs[i].buf;
        }
    }
    if (ctx->gguf_tbuf_count >= MAX_GGUF_TBUFS) return NULL;
    size_t hi = (off + size + page - 1) & ~(page - 1);
    id<MTLBuffer> buf = [ctx->device newBufferWithBytesNoCopy:(void *)((char *)g_gguf_data_base + lo)
                                                      length:hi - lo
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];
    if (!buf) return NULL;
    ctx->gguf_tbufs[ctx->gguf_tbuf_count].off = lo;
    ctx->gguf_tbufs[ctx->gguf_tbuf_count].buf = buf;
    ctx->gguf_tbuf_count++;
    *delta = (uint32_t)(off - lo);
    return buf;
}

// Phase C S4: lazy zero-copy wrap of the staged-BF16 conversion buffer.
// The stage is a static heap buffer finalized after open_gguf (which runs
// after metal_setup), so the wrap is created on first use. Lets the chunked
// GGUF path bind staged tensors (norms) at exact stage offsets.
static id<MTLBuffer> gguf_stage_mirror_get(MetalCtx *ctx) {
    if (!ctx->gguf_stage_gpu && g_gguf_stage && g_gguf_stage_len > 0) {
        ctx->gguf_stage_gpu = [ctx->device newBufferWithBytesNoCopy:g_gguf_stage
                                                             length:g_gguf_stage_len
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
    }
    return ctx->gguf_stage_gpu;
}

static void gguf_stage2_build(MetalCtx *ctx);   // defined after LayerWeightCache

// Phase C S4: GGUF expert pool slot size = max slab_total over all layers
// (gate+up+down per expert) rounded up to 2MB. Q4_K-only layers are ~1.7MB;
// layers with a Q6_K down_proj are ~4.1MB → 4MB slots.
static size_t gguf_exp_alloc_size(void) {
    size_t max_slab = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        size_t total = gguf_experts[l].gate_slab + gguf_experts[l].up_slab + gguf_experts[l].down_slab;
        if (total > max_slab) max_slab = total;
    }
    return (max_slab + 2*1024*1024 - 1) & ~(size_t)(2*1024*1024 - 1);
}

// Phase C S4: lazy GGUF expert-slab pool (copy-pool for chunked prefill).
// Same memory-budget ladder as the packed pool (64 → 32 → 16 → 0), computed
// against the 4MB slot size. 0 slots = per-position fallback path. Also
// guarantees the per-position PF expert scratch buffers exist — they're
// normally allocated by the packed-pool block, whose ladder can zero out
// independently of this one.
static id<MTLBuffer> gguf_pool_get(MetalCtx *ctx) {
    if (!g_gguf_stage) return NULL;
    if (!ctx->buf_pool_expert_data_gguf) {
        size_t exp_alloc = gguf_exp_alloc_size();
        g_gguf_exp_alloc = exp_alloc;
        int p = 64;
        size_t avail = get_available_memory();
        while (p >= 16 && (avail < (size_t)p * exp_alloc + (size_t)512 * 1024 * 1024)) p /= 2;
        if (avail < (size_t)16 * exp_alloc + (size_t)512 * 1024 * 1024) p = 0;
        g_pf_pool_slots_gguf = p;
        if (p > 0) {
            void *pool_aligned = NULL;
            size_t pool_bytes = (size_t)p * exp_alloc;
            posix_memalign(&pool_aligned, 2*1024*1024, pool_bytes);
            memset(pool_aligned, 0, pool_bytes);
            ctx->buf_pool_expert_data_gguf = [ctx->device newBufferWithBytesNoCopy:pool_aligned
                                                                            length:pool_bytes
                                                                           options:MTLResourceStorageModeShared
                                                                       deallocator:nil];
            fprintf(stderr, "[pf-pool-gguf] %d expert slots (%.0f MB), pool mode for chunks <= %d positions\n",
                    p, (double)pool_bytes / 1e6, p / MAX_K);
        } else {
            fprintf(stderr, "[pf-pool-gguf] disabled (insufficient memory) — per-position GGUF expert path\n");
        }
    }
    if (!ctx->buf_pf_expert_input) {
        size_t P2048 = (size_t)PREFILL_CHUNK_MAX * HIDDEN_DIM * sizeof(float);
        size_t P512  = (size_t)PREFILL_CHUNK_MAX * MOE_INTERMEDIATE * sizeof(float);
        ctx->buf_pf_expert_input = [ctx->device newBufferWithLength:P2048 options:MTLResourceStorageModeShared];
        for (int k = 0; k < MAX_K; k++) {
            ctx->buf_pf_expert_gate[k] = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            ctx->buf_pf_expert_up[k]   = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
            ctx->buf_pf_expert_act[k]  = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
        }
        ctx->buf_pf_shared_gate = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
        ctx->buf_pf_shared_up   = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
        ctx->buf_pf_shared_act  = [ctx->device newBufferWithLength:P512 options:MTLResourceStorageModeShared];
    }
    return ctx->buf_pool_expert_data_gguf;
}

// GGUF Q4_K/Q6_K matvec on GPU. Returns 1 on success (out filled), 0 if the
// caller should fall back to gguf_cpu_matvec. in_dim <= 4096 (x_shared limit).
static int gpu_gguf_dequant_matvec(MetalCtx *ctx, const void *W,
                                   const float *x, float *out,
                                   int out_dim, int in_dim, int ggml_type) {
    if (!ctx->matvec_qk || in_dim > 4096) return 0;
    size_t row_bytes = (size_t)(in_dim / 256) * (ggml_type == 12 ? 144 : 210);
    uint32_t delta = 0;
    id<MTLBuffer> tbuf = gguf_tbuf_get(ctx, W, row_bytes * (size_t)out_dim, &delta);
    if (!tbuf) return 0;
    memcpy([ctx->buf_input contents], x, (size_t)in_dim * sizeof(float));
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->matvec_qk];
    [enc setBuffer:tbuf offset:delta atIndex:0];
    [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
    [enc setBuffer:ctx->buf_output offset:0 atIndex:4];
    uint32_t od = (uint32_t)out_dim, id_ = (uint32_t)in_dim, gt = (uint32_t)ggml_type;
    [enc setBytes:&od length:4 atIndex:5];
    [enc setBytes:&id_ length:4 atIndex:6];
    [enc setBytes:&gt length:4 atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + 7) / 8, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    memcpy(out, [ctx->buf_output contents], (size_t)out_dim * sizeof(float));
    return 1;
}

// S7: slab pread task + pool dispatch, used by the GGUF expert path before
// their natural definitions in the I/O section below.
typedef struct InferPreadTask {
    int fd;
    void *dst;
    off_t offset;
    size_t size;
    ssize_t result;
    const void *mmap_base;  // if non-NULL, memcpy from mmap instead of pread
    // LZ4 compression fields (set by caller when reading compressed experts)
    void *lz4_comp_buf;     // if non-NULL: pread into this, then LZ4 decompress into dst
    uint32_t lz4_comp_size; // compressed size to read from disk
} InferPreadTask;
static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks);
static void async_pread_multi_start(const int *fds, const off_t *offsets,
                                    void *const *dsts, const size_t *sizes, int n);
typedef struct {
    InferPreadTask tasks[PREFILL_CHUNK_MAX * MAX_K];
    int num_tasks;
    int valid[PREFILL_CHUNK_MAX * MAX_K];
    dispatch_group_t group;
    int active;
} AsyncPreadState;
static AsyncPreadState g_async_pread;
static void async_pread_wait(void);

// Phase C S7 Lever 1: GGUF temporal-prediction preads into buffer set B.
// The guess for THIS token's routing is LAST token's (stored after the expert
// call). Fired at layer entry so the reads overlap the qkv/attn/cmd2 CPU work
// (~2.3ms window); the predicted slabs were read last token, so their pages
// are warm. Consumed (hit-checked) by gpu_gguf_experts_forward.
// Kill switch: FINCHMOE_GGUF_NOPRED=1.
static int g_gguf_pred_experts[NUM_LAYERS][MAX_K];
static int g_gguf_pred_count[NUM_LAYERS];
static int g_gguf_pred_valid = 0;
static int g_gguf_pred_fired = 0;
static long g_gguf_pred_hits = 0, g_gguf_pred_tries = 0;
static double g_gguf_pred_wait_ms = 0;
static void gguf_pred_fire(int layer_idx) {
    g_gguf_pred_fired = 0;
    if (!g_gguf_stage || !g_pred_generating || !g_gguf_pred_valid) return;
    if (g_gguf_pred_count[layer_idx] <= 0) return;
    static int nopred = -1;
    if (nopred < 0) nopred = getenv("FINCHMOE_GGUF_NOPRED") != NULL;
    if (nopred) return;
    GgufExpertInfo *ge = &gguf_experts[layer_idx];
    int n = g_gguf_pred_count[layer_idx];
    static int fds[PREFILL_CHUNK_MAX * MAX_K];
    static off_t offs[PREFILL_CHUNK_MAX * MAX_K];
    static void *dsts[PREFILL_CHUNK_MAX * MAX_K];
    static size_t sizes[PREFILL_CHUNK_MAX * MAX_K];
    int nr = 0;
    for (int p = 0; p < n; p++) {
        int e = g_gguf_pred_experts[layer_idx][p];
        uint8_t *dst = (uint8_t *)[g_metal->buf_multi_expert_data_B[p] contents];
        fds[nr] = g_gguf_fd;
        offs[nr] = (off_t)(ge->gate_off + (size_t)e * ge->gate_slab);
        dsts[nr] = dst; sizes[nr] = ge->gate_slab; nr++;
        fds[nr] = g_gguf_fd;
        offs[nr] = (off_t)(ge->up_off + (size_t)e * ge->up_slab);
        dsts[nr] = dst + ge->gate_slab; sizes[nr] = ge->up_slab; nr++;
        fds[nr] = g_gguf_fd;
        offs[nr] = (off_t)(ge->down_off + (size_t)e * ge->down_slab);
        dsts[nr] = dst + ge->gate_slab + ge->up_slab; sizes[nr] = ge->down_slab; nr++;
    }
    async_pread_multi_start(fds, offs, dsts, sizes, nr);
    g_gguf_pred_fired = 1;
}

// Phase C S2: GGUF routed experts on GPU. One command buffer per layer:
// per expert, a fused gate+up+SwiGLU (Q4_K slabs read from per-tensor wraps,
// no pread/copies) into buf_expert_act, then the down matvec (dequant_matvec_qk
// with x = buf_expert_act) into buf_multi_expert_out[k]. Returns 1 on success
// (moe_out accumulated with the same weighted-combine semantics as the CPU
// path), 0 if any wrap/pipeline is missing (caller falls back to CPU).
static int gpu_gguf_experts_forward(MetalCtx *ctx, WeightFile *wf, int layer_idx,
                                    const float *h_post,
                                    const int *expert_indices, const float *expert_weights,
                                    int K, float *moe_out) {
    if (!ctx->fused_gate_up_swiglu_qk_pipe || !ctx->matvec_qk) return 0;
    GgufExpertInfo *ge = &gguf_experts[layer_idx];
    if (ge->gate_type != 12 || ge->up_type != 12) return 0;  // fused kernel is Q4_K-only
    // Claim any pending prediction fire for THIS layer. Early-return paths
    // above discard it — a stale flag must never leak into the next layer's
    // call (its predictions are for a different layer's experts).
    int pred_fired_local = g_gguf_pred_fired;
    g_gguf_pred_fired = 0;

    // Copy the active experts' slabs into the stable preallocated expert
    // buffers (the packed path's design): the kernel reads from buffers that
    // never churn the GPU residency, and the mmap pages are touched once per
    // layer by a plain memcpy. Layout per slot: gate @ 0, up @ gate_slab,
    // down @ gate_slab + up_slab (all 32-byte aligned).
    size_t slab_total = ge->gate_slab + ge->up_slab + ge->down_slab;
    if (slab_total > EXPERT_SIZE_MAX) return 0;
    // Phase C S7: component-split probe (FINCHMOE_EXPTIME=2) — where the
    // per-layer expert time goes: slab copies vs encode vs sync wait vs combine.
    static int probe = -1;
    static double pmc = 0, penc = 0, pw = 0, pco = 0; static int pn = 0;
    if (probe < 0) { const char *e = getenv("FINCHMOE_EXPTIME"); probe = (e && atoi(e) >= 2); }
    double t_mc0 = 0, t_enc0 = 0, t_w0 = 0, t_co0 = 0;
    // Phase C S7 Lever 1: consume the prediction preads fired at layer entry.
    // A hit means buffer set B already holds this expert's slab (byte-identical
    // to what the copy would produce) — bind the GPU kernels to B and skip the
    // copy. Misses fall through to the copy paths below.
    id<MTLBuffer> expert_bufs[MAX_K];
    int pred_hit[MAX_K] = {0};
    for (int k = 0; k < MAX_K; k++) expert_bufs[k] = ctx->buf_multi_expert_data[k];
    if (pred_fired_local) {
        // Index pre-check: if no routed expert matches any predicted one, the
        // B data is useless — skip the wait entirely and go straight to the
        // copy paths.
        int want_wait = 0;
        for (int k = 0; k < K && !want_wait; k++) {
            for (int p = 0; p < g_gguf_pred_count[layer_idx]; p++) {
                if (expert_indices[k] == g_gguf_pred_experts[layer_idx][p]) { want_wait = 1; break; }
            }
        }
        int hit_count = 0;
        if (want_wait) {
            double t_pw0 = now_ms();
            async_pread_wait();
            g_gguf_pred_wait_ms += now_ms() - t_pw0;
            for (int k = 0; k < K; k++) {
                for (int p = 0; p < g_gguf_pred_count[layer_idx]; p++) {
                    if (expert_indices[k] == g_gguf_pred_experts[layer_idx][p] &&
                        g_async_pread.valid[p * 3] && g_async_pread.valid[p * 3 + 1] &&
                        g_async_pread.valid[p * 3 + 2]) {
                        expert_bufs[k] = ctx->buf_multi_expert_data_B[p];
                        pred_hit[k] = 1;
                        hit_count++;
                        break;
                    }
                }
            }
        }
        g_gguf_pred_hits += hit_count;
        g_gguf_pred_tries += K;
    }
    if (probe) t_mc0 = now_ms();
    // Phase C S7: parallel slab preads. The serial memcpy was latency-bound on
    // cold pages (one thread, one page at a time); io_pool preads (8 threads)
    // overlap the reads. Short reads fall back to inline memcpy.
    // Kill switch: FINCHMOE_GGUF_NOSLABCACHE=1 (original serial memcpy).
    // NOTE: no residency cache here — buf_multi_expert_data has only MAX_K
    // buffers shared by ALL layers, so a later layer always clobbers a
    // slot before the same layer reuses it next token. Cross-token reuse
    // needs a per-(layer,slot) pool (S7 follow-up).
    static int no_cache = -1;
    if (no_cache < 0) no_cache = getenv("FINCHMOE_GGUF_NOSLABCACHE") != NULL;
    if (no_cache) {
        for (int k = 0; k < K; k++) {
            if (pred_hit[k]) continue;
            int eidx = expert_indices[k];
            uint8_t *dst = (uint8_t *)[ctx->buf_multi_expert_data[k] contents];
            memcpy(dst,                          (const uint8_t *)wf->data + ge->gate_off + (size_t)eidx * ge->gate_slab, ge->gate_slab);
            memcpy(dst + ge->gate_slab,          (const uint8_t *)wf->data + ge->up_off   + (size_t)eidx * ge->up_slab,   ge->up_slab);
            memcpy(dst + ge->gate_slab + ge->up_slab,
                                                  (const uint8_t *)wf->data + ge->down_off + (size_t)eidx * ge->down_slab, ge->down_slab);
        }
    } else {
        InferPreadTask tasks[MAX_K * 3];
        int n_tasks = 0;
        int k_of_task[MAX_K * 3];
        for (int k = 0; k < K; k++) {
            if (pred_hit[k]) continue;
            int eidx = expert_indices[k];
            uint8_t *dst = (uint8_t *)[ctx->buf_multi_expert_data[k] contents];
            tasks[n_tasks].fd = g_gguf_fd;
            tasks[n_tasks].dst = dst;
            tasks[n_tasks].offset = (off_t)(ge->gate_off + (size_t)eidx * ge->gate_slab);
            tasks[n_tasks].size = ge->gate_slab;
            tasks[n_tasks].result = 0;
            tasks[n_tasks].mmap_base = NULL;
            tasks[n_tasks].lz4_comp_buf = NULL;
            tasks[n_tasks].lz4_comp_size = 0;
            k_of_task[n_tasks++] = k;
            tasks[n_tasks].fd = g_gguf_fd;
            tasks[n_tasks].dst = dst + ge->gate_slab;
            tasks[n_tasks].offset = (off_t)(ge->up_off + (size_t)eidx * ge->up_slab);
            tasks[n_tasks].size = ge->up_slab;
            tasks[n_tasks].result = 0;
            tasks[n_tasks].mmap_base = NULL;
            tasks[n_tasks].lz4_comp_buf = NULL;
            tasks[n_tasks].lz4_comp_size = 0;
            k_of_task[n_tasks++] = k;
            tasks[n_tasks].fd = g_gguf_fd;
            tasks[n_tasks].dst = dst + ge->gate_slab + ge->up_slab;
            tasks[n_tasks].offset = (off_t)(ge->down_off + (size_t)eidx * ge->down_slab);
            tasks[n_tasks].size = ge->down_slab;
            tasks[n_tasks].result = 0;
            tasks[n_tasks].mmap_base = NULL;
            tasks[n_tasks].lz4_comp_buf = NULL;
            tasks[n_tasks].lz4_comp_size = 0;
            k_of_task[n_tasks++] = k;
        }
        io_pool_dispatch(tasks, n_tasks);
        // Verify reads; any short read for expert k re-copies all 3 sections
        // inline from the mmap (original semantics).
        int bad[MAX_K] = {0};
        for (int t = 0; t < n_tasks; t++) {
            if (tasks[t].result != (ssize_t)tasks[t].size) bad[k_of_task[t]] = 1;
        }
        for (int k = 0; k < K; k++) {
            if (!bad[k]) continue;
            int eidx = expert_indices[k];
            uint8_t *dst = (uint8_t *)[ctx->buf_multi_expert_data[k] contents];
            memcpy(dst,                          (const uint8_t *)wf->data + ge->gate_off + (size_t)eidx * ge->gate_slab, ge->gate_slab);
            memcpy(dst + ge->gate_slab,          (const uint8_t *)wf->data + ge->up_off   + (size_t)eidx * ge->up_slab,   ge->up_slab);
            memcpy(dst + ge->gate_slab + ge->up_slab,
                                                  (const uint8_t *)wf->data + ge->down_off + (size_t)eidx * ge->down_slab, ge->down_slab);
        }
    }

    memcpy([ctx->buf_multi_expert_input contents], h_post, HIDDEN_DIM * sizeof(float));
    if (probe) t_enc0 = now_ms();
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    for (int k = 0; k < K; k++) {
        uint32_t od = MOE_INTERMEDIATE, id_ = HIDDEN_DIM;
        // fused gate+up+SwiGLU -> buf_expert_act (gate @ 0, up @ gate_slab)
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->fused_gate_up_swiglu_qk_pipe];
            [enc setBuffer:expert_bufs[k] offset:0 atIndex:0];
            [enc setBuffer:expert_bufs[k] offset:ge->gate_slab atIndex:1];
            [enc setBuffer:ctx->buf_multi_expert_input offset:0 atIndex:2];
            [enc setBuffer:ctx->buf_expert_act offset:0 atIndex:3];
            [enc setBytes:&od length:4 atIndex:4];
            [enc setBytes:&id_ length:4 atIndex:5];
            [enc dispatchThreadgroups:MTLSizeMake(MOE_INTERMEDIATE, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
        // down: buf_expert_act -> buf_multi_expert_out[k] offset 0
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            uint32_t dod = HIDDEN_DIM, did = MOE_INTERMEDIATE, dgt = (uint32_t)ge->down_type;
            [enc setComputePipelineState:ctx->matvec_qk];
            [enc setBuffer:expert_bufs[k] offset:ge->gate_slab + ge->up_slab atIndex:0];
            [enc setBuffer:ctx->buf_expert_act offset:0 atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
            [enc setBytes:&dod length:4 atIndex:5];
            [enc setBytes:&did length:4 atIndex:6];
            [enc setBytes:&dgt length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }
    [cb commit];
    if (probe) t_w0 = now_ms();
    [cb waitUntilCompleted];
    if (probe) t_co0 = now_ms();

    // CPU combine — mirrors the CPU fallback (weighted madd + finite guard
    // + total-weight renormalization).
    float total_weight = 0.0f;
    for (int k = 0; k < K; k++) {
        float *expert_out = (float *)[ctx->buf_multi_expert_out[k] contents];
        float er = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; j++) er += expert_out[j] * expert_out[j];
        if (isfinite(er) && er < 1e20f) {
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], HIDDEN_DIM);
            total_weight += expert_weights[k];
        }
    }
    if (total_weight > 0.0f && total_weight < 0.99f) {
        float inv_tw = 1.0f / total_weight;
        for (int i = 0; i < HIDDEN_DIM; i++) moe_out[i] *= inv_tw;
    }
    // Store this token's routing for the NEXT token's prediction preads.
    if (g_pred_generating) {
        for (int k = 0; k < K; k++) g_gguf_pred_experts[layer_idx][k] = expert_indices[k];
        g_gguf_pred_count[layer_idx] = K;
        if (layer_idx == NUM_LAYERS - 1) g_gguf_pred_valid = 1;
    }
    // Phase C S7 Lever 1: fire the NEXT layer's prediction preads now — the
    // window is this layer's CB wait+combine plus the next layer's
    // qkv/attn/cmd2 CPU work (~3.5-4ms total). Layer 0 of the next token is
    // fired by layer 39 here at the end of the previous token.
    if (layer_idx + 1 < NUM_LAYERS) gguf_pred_fire(layer_idx + 1);
    if (probe) {
        pmc += t_enc0 - t_mc0; penc += t_w0 - t_enc0; pw += t_co0 - t_w0; pco += now_ms() - t_co0;
        if (++pn % NUM_LAYERS == 0)
            fprintf(stderr, "[expsplit] copy %.3f encode %.3f wait %.3f combine %.3f ms/layer, predwait %.3f, pred hits %ld/%ld (n=%d)\n",
                    pmc/pn, penc/pn, pw/pn, pco/pn, g_gguf_pred_wait_ms / pn, g_gguf_pred_hits, g_gguf_pred_tries, pn);
    }
    return 1;
}

// Wrapper: use GPU if available and weight buffer is set, CPU otherwise
static void fast_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size, int bits
) {
    // BF16 path: no GPU kernel for raw BF16 matvec.
    // GGUF mode: Q4_K/Q6_K tensors (bits 10/11) also have no scales — try the
    // Phase C GPU block kernel first, else the CPU block dequant. Never the
    // raw-BF16 read.
    if (!scales || !biases) {
        if ((bits == 10 || bits == 11) && g_metal && g_metal->matvec_qk &&
            gpu_gguf_dequant_matvec(g_metal, W, x, out, out_dim, in_dim,
                                    bits == 10 ? 12 : 14)) {
            return;
        }
        cpu_dequant_matvec(W, NULL, NULL, x, out, out_dim, in_dim, group_size,
                           (bits == 10 || bits == 11) ? bits : 0);
        return;
    }
    if (g_metal && g_metal->wf_buf) {
        gpu_dequant_matvec(g_metal, W, scales, biases, x, out,
                           (uint32_t)out_dim, (uint32_t)in_dim, (uint32_t)group_size, bits);
    } else {
        cpu_dequant_matvec(W, scales, biases, x, out, out_dim, in_dim, group_size, bits);
    }
}

// ============================================================================
// Batched GPU matmul: encode N independent matmuls sharing the same input
// into ONE command buffer, reducing dispatch overhead by N-1 round-trips.
// ============================================================================

typedef struct {
    const void *W;           // packed weights (pointer into mmap'd file)
    const void *scales;      // scales (pointer into mmap'd file)
    const void *biases;      // biases (pointer into mmap'd file)
    float *out_cpu;          // CPU output pointer (result copied here after GPU finishes)
    uint32_t out_dim;
    uint32_t in_dim;
    uint32_t group_size;
    int batch_slot;          // which batch_out[slot] to use for GPU output
    int bits;                // packing width: 4 or 8 (ignored when scales==NULL → BF16 path)
} BatchMatvecSpec;

// Run N matmuls in a single command buffer. All share the same input vector.
// The input is copied once; all outputs go to preallocated batch_out slots.
static void gpu_batch_matvec(
    MetalCtx *ctx,
    const float *x_f32, uint32_t x_dim,  // shared input
    BatchMatvecSpec *specs, int num_specs
) {
    // Copy input once
    memcpy([ctx->buf_input contents], x_f32, x_dim * sizeof(float));

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];

        // Phase C: GGUF Q4_K/Q6_K specs (bits 10/11, scales==NULL) — encode
        // with the block-dequant kernel. Must be checked BEFORE the BF16
        // branch (which would read Q4_K bytes as BF16).
        if ((s->bits == 10 || s->bits == 11) && ctx->matvec_qk) {
            size_t row_bytes = (size_t)(s->in_dim / 256) * (s->bits == 10 ? 144 : 210);
            uint32_t delta = 0;
            id<MTLBuffer> tbuf = gguf_tbuf_get(ctx, s->W, row_bytes * (size_t)s->out_dim, &delta);
            if (tbuf) {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:ctx->matvec_qk];
                [enc setBuffer:tbuf offset:delta atIndex:0];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                [enc setBuffer:ctx->batch_out[s->batch_slot] offset:0 atIndex:4];
                uint32_t od = s->out_dim, id_ = s->in_dim, gt = (uint32_t)(s->bits == 10 ? 12 : 14);
                [enc setBytes:&od length:4 atIndex:5];
                [enc setBytes:&id_ length:4 atIndex:6];
                [enc setBytes:&gt length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((s->out_dim + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                continue;
            }
            cpu_dequant_matvec(s->W, NULL, NULL, x_f32, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size, s->bits);
            continue;
        }

        // BF16 path: use gemv_bf16 kernel (scales/biases are NULL for unquantized weights)
        if (!s->scales || !s->biases) {
            NSUInteger w_off = (NSUInteger)((const char *)s->W - (const char *)[ctx->wf_buf contents]);
            id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            // Use x2 kernel for out_dim >= 128: 2 rows/tg amortizes x-vector loads
            int use_x2 = (s->out_dim >= 128 && ctx->gemv_bf16_x2_pipe);
            [enc setComputePipelineState: use_x2 ? ctx->gemv_bf16_x2_pipe : ctx->gemv_bf16_pipe];
            [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
            [enc setBuffer:o_buf offset:0 atIndex:2];
            [enc setBytes:&s->out_dim length:4 atIndex:3];
            [enc setBytes:&s->in_dim  length:4 atIndex:4];
            uint32_t tgs = use_x2 ? (s->out_dim + 1) / 2 : s->out_dim;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            continue;
        }

        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        if (s->bits == 8 && ctx->matvec_8bit) {
            // 8-bit packed weights: same tiled layout as v3 (ROWS_PER_TG=8, 256 threads)
            [enc setComputePipelineState:ctx->matvec_8bit];
        } else {
            [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        }
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy results back to CPU
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// ============================================================================
// Encode-only variants: add dispatches to an EXISTING command buffer.
// These do NOT commit — the caller batches multiple encode calls into one
// command buffer and commits once, eliminating per-dispatch overhead.
// ============================================================================

// Encode N matmuls into cmdbuf. Input must already be in ctx->buf_input.
// BF16 specs (scales==NULL) are handled via CPU fallback automatically.
static void gpu_encode_batch_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    BatchMatvecSpec *specs, int num_specs
) {
    // Check for BF16 specs — use GPU BF16 kernel
    for (int i = 0; i < num_specs; i++) {
        // Phase C: GGUF Q4_K/Q6_K specs (bits 10/11, scales==NULL) — encode
        // with the block-dequant kernel reading a per-tensor wrapped buffer.
        // Must be checked BEFORE the BF16 branch below.
        if ((specs[i].bits == 10 || specs[i].bits == 11) && ctx->matvec_qk) {
            BatchMatvecSpec *s = &specs[i];
            size_t row_bytes = (size_t)(s->in_dim / 256) * (s->bits == 10 ? 144 : 210);
            uint32_t delta = 0;
            id<MTLBuffer> tbuf = gguf_tbuf_get(ctx, s->W, row_bytes * (size_t)s->out_dim, &delta);
            if (tbuf) {
                id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
                [enc setComputePipelineState:ctx->matvec_qk];
                [enc setBuffer:tbuf offset:delta atIndex:0];
                [enc setBuffer:ctx->buf_input offset:0 atIndex:3];
                [enc setBuffer:ctx->batch_out[s->batch_slot] offset:0 atIndex:4];
                uint32_t od = s->out_dim, id_ = s->in_dim, gt = (uint32_t)(s->bits == 10 ? 12 : 14);
                [enc setBytes:&od length:4 atIndex:5];
                [enc setBytes:&id_ length:4 atIndex:6];
                [enc setBytes:&gt length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((s->out_dim + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                continue;
            }
            // wrap failed (shouldn't happen for tensors <= 8GB) — compute on
            // CPU right here (the input is already in buf_input).
            cpu_dequant_matvec(s->W, NULL, NULL,
                               (const float *)[ctx->buf_input contents], s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size, s->bits);
            continue;
        }
        if (!specs[i].scales || !specs[i].biases) {
            BatchMatvecSpec *s = &specs[i];
            NSUInteger w_off = (NSUInteger)((const char *)s->W - (const char *)[ctx->wf_buf contents]);
            id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
            int use_x2 = (s->out_dim >= 128 && ctx->gemv_bf16_x2_pipe);
            [enc setComputePipelineState: use_x2 ? ctx->gemv_bf16_x2_pipe : ctx->gemv_bf16_pipe];
            [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
            [enc setBuffer:ctx->buf_input offset:0 atIndex:1];
            // output to batch_out slot so gpu_flush_batch_results can copy it
            [enc setBuffer:ctx->batch_out[s->batch_slot] offset:0 atIndex:2];
            [enc setBytes:&s->out_dim length:4 atIndex:3];
            [enc setBytes:&s->in_dim  length:4 atIndex:4];
            uint32_t tgs = use_x2 ? (s->out_dim + 1) / 2 : s->out_dim;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            continue;
        }
        // 4-bit/8-bit specs: encode with the matching dequant kernel
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        if (s->bits == 8 && ctx->matvec_8bit) {
            // 8-bit packed weights: same tiled layout as v3 (ROWS_PER_TG=8, 256 threads)
            [enc setComputePipelineState:ctx->matvec_8bit];
        } else {
            [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        }
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];
        if (use_v3 || (s->bits == 8 && ctx->matvec_8bit)) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
    return;  // Done — all specs encoded (BF16 on gemv_bf16, 8bit on matvec_8bit, 4bit on v3/fast)
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        NSUInteger w_off = (NSUInteger)((const char *)s->W      - (const char *)[ctx->wf_buf contents]);
        NSUInteger s_off = (NSUInteger)((const char *)s->scales  - (const char *)[ctx->wf_buf contents]);
        NSUInteger b_off = (NSUInteger)((const char *)s->biases  - (const char *)[ctx->wf_buf contents]);

        id<MTLBuffer> o_buf = ctx->batch_out[s->batch_slot];

        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        int use_v3 = (s->in_dim <= 4096);
        if (s->bits == 8 && ctx->matvec_8bit) {
            // 8-bit packed weights: same tiled layout as v3 (ROWS_PER_TG=8, 256 threads)
            [enc setComputePipelineState:ctx->matvec_8bit];
        } else {
            [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
        }
        [enc setBuffer:ctx->wf_buf  offset:w_off atIndex:0];
        [enc setBuffer:ctx->wf_buf  offset:s_off atIndex:1];
        [enc setBuffer:ctx->wf_buf  offset:b_off atIndex:2];
        [enc setBuffer:ctx->buf_input offset:0   atIndex:3];
        [enc setBuffer:o_buf        offset:0     atIndex:4];
        [enc setBytes:&s->out_dim   length:4     atIndex:5];
        [enc setBytes:&s->in_dim    length:4     atIndex:6];
        [enc setBytes:&s->group_size length:4    atIndex:7];

        if (use_v3) {
            uint32_t num_tgs = (s->out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        } else {
            [enc dispatchThreadgroups:MTLSizeMake(s->out_dim, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }
        [enc endEncoding];
    }
}

// ============================================================================
// Prefill-batched matvec: M positions x one weight matrix in a single
// dispatch. Math is bitwise-identical to the per-token kernels (same FMA
// form, same reduction order) — only the grid gains an M dimension.
// scales == NULL selects the BF16 gemv kernel (unquantized weights).
// ============================================================================

// Flush a buffer's GPU cache lines inside a command buffer so subsequent
// dispatches in the SAME CB see the latest device writes. On this GPU the
// scope-based memoryBarrierWithScope does NOT invalidate L2 (verified: the
// ~1e-2 prefill wobble persisted with barriers at every dependent encoder).
// synchronizeResource is the explicit flush primitive.
static void metal_sync_buffer(id<MTLCommandBuffer> cmdbuf, id<MTLBuffer> buf) {
    if (!cmdbuf || !buf) return;
    id<MTLBlitCommandEncoder> blit = [cmdbuf blitCommandEncoder];
    [blit synchronizeResource:buf];
    [blit endEncoding];
}
static void gpu_encode_prefill_matvec(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    uint32_t bits, uint32_t M, NSUInteger out_offset
) {
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    // Order this dispatch after ALL prior buffer writes in the command
    // buffer: out_proj reads buf_pf_oproj_in written by the GDN chains
    // (linear layers) or the batched-attention kernels (full layers) in the
    // same CB, and routing matmuls read buf_pf_h_post written by the
    // residual_norm dispatch — without the barrier the GPU may serve stale
    // cache lines (the run-to-run ~1e-2 logit wobble root cause).
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Phase C S3: GGUF Q4_K/Q6_K specs (bits 10/11, scales==NULL) — the
    // batched block-dequant kernel reading a per-tensor wrapped buffer.
    if ((bits == 10 || bits == 11) && ctx->matvec_qk_prefill) {
        size_t row_bytes = (size_t)(in_dim / 256) * (bits == 10 ? 144 : 210);
        uint32_t delta = 0;
        id<MTLBuffer> tbuf = NULL;
        // S6 stage2: tensors copied into the 2MB-aligned anonymous buffer
        // bind through the single stage2 wrap (avoids file-backed page walks).
        if (g_gguf_stage2_gpu && W >= (const void *)g_gguf_stage2 &&
            (const char *)W < g_gguf_stage2 + g_gguf_stage2_len) {
            tbuf = g_gguf_stage2_gpu;
            delta = (uint32_t)((const char *)W - g_gguf_stage2);
        } else {
            tbuf = gguf_tbuf_get(ctx, W, row_bytes * (size_t)out_dim, &delta);
        }
        if (tbuf) {
            [enc setComputePipelineState:ctx->matvec_qk_prefill];
            [enc setBuffer:tbuf     offset:delta      atIndex:0];
            [enc setBuffer:in_buf   offset:0          atIndex:3];
            [enc setBuffer:out_buf  offset:out_offset atIndex:4];
            uint32_t gt = (uint32_t)(bits == 10 ? 12 : 14);
            [enc setBytes:&out_dim length:4 atIndex:5];
            [enc setBytes:&in_dim  length:4 atIndex:6];
            [enc setBytes:&gt      length:4 atIndex:7];
            uint32_t num_row_tiles = (out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake((uint64_t)M * num_row_tiles, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
            return;
        }
        // wrap failed — leave the output stale is not acceptable; the caller
        // cannot handle this path (chunked prefill is gated off for GGUF
        // until S3 completes), so this is unreachable in practice.
    }

    // Phase C S4: bits-0 staged-BF16 tensors (GGUF mode) — bind the stage
    // mirror at the tensor's stage offset. The staged tensors live in a
    // heap buffer (not wf_buf), so the wf_buf pointer diff above is garbage.
    if (bits == 0 && g_gguf_stage && ctx->gguf_stage_gpu) {
        NSUInteger stage_off = (NSUInteger)((const char *)W - (const char *)g_gguf_stage);
        [enc setComputePipelineState:ctx->gemv_bf16_prefill];
        [enc setBuffer:ctx->gguf_stage_gpu offset:stage_off atIndex:0];
        [enc setBuffer:in_buf     offset:0     atIndex:1];
        [enc setBuffer:out_buf    offset:out_offset atIndex:2];
        [enc setBytes:&out_dim length:4 atIndex:3];
        [enc setBytes:&in_dim  length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)M * out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc endEncoding];
        return;
    }

    if (!scales || !biases) {
        // BF16 GEMV: grid linearized m * out_dim + row
        [enc setComputePipelineState:ctx->gemv_bf16_prefill];
        [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
        [enc setBuffer:in_buf     offset:0     atIndex:1];
        [enc setBuffer:out_buf    offset:out_offset atIndex:2];
        [enc setBytes:&out_dim length:4 atIndex:3];
        [enc setBytes:&in_dim  length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)M * out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // Writer-side barrier: make this dispatch's writes visible to
        // subsequent dispatches in the same CB (reader-side placement is a
        // no-op — the constraint is on THIS encoder's dispatches).
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc endEncoding];
        return;
    }

    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);

    if (bits == 8 && ctx->matvec_prefill_8bit) {
        [enc setComputePipelineState:ctx->matvec_prefill_8bit];
    } else {
        [enc setComputePipelineState:ctx->matvec_prefill_4bit];
    }
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf     offset:0     atIndex:3];
    [enc setBuffer:out_buf    offset:out_offset atIndex:4];
    [enc setBytes:&out_dim    length:4 atIndex:5];
    [enc setBytes:&in_dim     length:4 atIndex:6];
    [enc setBytes:&group_size length:4 atIndex:7];
    uint32_t num_row_tiles = (out_dim + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake((uint64_t)M * num_row_tiles, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    // Writer-side barrier (see BF16 branch comment).
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc endEncoding];
}

// Copy batch results from GPU buffers back to CPU pointers.
static void gpu_flush_batch_results(MetalCtx *ctx, BatchMatvecSpec *specs, int num_specs) {
    for (int i = 0; i < num_specs; i++) {
        BatchMatvecSpec *s = &specs[i];
        memcpy(s->out_cpu, [ctx->batch_out[s->batch_slot] contents],
               s->out_dim * sizeof(float));
    }
}

// Encode a single matvec reading from buf_expert_act into buf_expert_out,
// using weight pointers into the mmap'd weight file.
// Used for shared expert down_proj which reads from a different input than
// the attention projections.
static void gpu_encode_dequant_matvec_with_io_bufs(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    const void *W, const void *scales, const void *biases,
    id<MTLBuffer> in_buf, id<MTLBuffer> out_buf,
    uint32_t out_dim, uint32_t in_dim, uint32_t group_size,
    NSUInteger out_offset,
    NSUInteger in_offset
) {
    // BF16 guard: this function does 4-bit GPU dequant only
    if (!scales || !biases) {
        fprintf(stderr, "FATAL: gpu_encode_dequant_matvec_with_io_bufs called with BF16 weights\n");
        abort();
    }
    NSUInteger w_off = (NSUInteger)((const char *)W      - (const char *)[ctx->wf_buf contents]);
    NSUInteger s_off = (NSUInteger)((const char *)scales  - (const char *)[ctx->wf_buf contents]);
    NSUInteger b_off = (NSUInteger)((const char *)biases  - (const char *)[ctx->wf_buf contents]);

    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    // Barrier: the caller may have written the input buffer from a previous
    // encoder in this command buffer (e.g. shared expert SwiGLU -> down_proj).
    // Without this the GPU may run the two dispatches concurrently and this
    // kernel reads a half-written input. (Bit us: timing-dependent garbage.)
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    int use_v3 = (in_dim <= 4096);
    [enc setComputePipelineState: use_v3 ? ctx->matvec_v3 : ctx->matvec_fast];
    [enc setBuffer:ctx->wf_buf offset:w_off atIndex:0];
    [enc setBuffer:ctx->wf_buf offset:s_off atIndex:1];
    [enc setBuffer:ctx->wf_buf offset:b_off atIndex:2];
    [enc setBuffer:in_buf      offset:in_offset   atIndex:3];
    [enc setBuffer:out_buf     offset:out_offset atIndex:4];
    [enc setBytes:&out_dim     length:4     atIndex:5];
    [enc setBytes:&in_dim      length:4     atIndex:6];
    [enc setBytes:&group_size  length:4     atIndex:7];

    if (use_v3) {
        uint32_t num_tgs = (out_dim + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
    [enc endEncoding];
}

// Encode one expert forward using multi-expert slot k.
// Expert data must already be in buf_multi_expert_data[k].
// Input must already be in buf_multi_expert_input.
static void gpu_encode_expert_forward_slot(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k  // slot index
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_1bit) {
        gate_w_off = GATE_W_OFF_1; gate_s_off = GATE_S_OFF_1; gate_b_off = GATE_B_OFF_1;
        up_w_off   = UP_W_OFF_1;   up_s_off   = UP_S_OFF_1;   up_b_off   = UP_B_OFF_1;
        down_w_off = DOWN_W_OFF_1; down_s_off = DOWN_S_OFF_1; down_b_off = DOWN_B_OFF_1;
    } else if (g_use_2bit) {
        gate_w_off = GATE_W_OFF_2; gate_s_off = GATE_S_OFF_2; gate_b_off = GATE_B_OFF_2;
        up_w_off   = UP_W_OFF_2;   up_s_off   = UP_S_OFF_2;   up_b_off   = UP_B_OFF_2;
        down_w_off = DOWN_W_OFF_2; down_s_off = DOWN_S_OFF_2; down_b_off = DOWN_B_OFF_2;
    } else if (g_use_3bit) {
        gate_w_off = GATE_W_OFF_3; gate_s_off = GATE_S_OFF_3; gate_b_off = GATE_B_OFF_3;
        up_w_off   = UP_W_OFF_3;   up_s_off   = UP_S_OFF_3;   up_b_off   = UP_B_OFF_3;
        down_w_off = DOWN_W_OFF_3; down_s_off = DOWN_S_OFF_3; down_b_off = DOWN_B_OFF_3;
    } else if (g_use_int8) {
        gate_w_off = GATE_W_OFF_8; gate_s_off = GATE_S_OFF_8; gate_b_off = GATE_B_OFF_8;
        up_w_off   = UP_W_OFF_8;   up_s_off   = UP_S_OFF_8;   up_b_off   = UP_B_OFF_8;
        down_w_off = DOWN_W_OFF_8; down_s_off = DOWN_S_OFF_8; down_b_off = DOWN_B_OFF_8;
    } else {
        gate_w_off = GATE_W_OFF_4; gate_s_off = GATE_S_OFF_4; gate_b_off = GATE_B_OFF_4;
        up_w_off   = UP_W_OFF_4;   up_s_off   = UP_S_OFF_4;   up_b_off   = UP_B_OFF_4;
        down_w_off = DOWN_W_OFF_4; down_s_off = DOWN_S_OFF_4; down_b_off = DOWN_B_OFF_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_3bit ? ctx->matvec_3bit : (g_use_2bit ? ctx->matvec_2bit : (g_use_1bit ? ctx->matvec_1bit : (g_use_int8 ? ctx->matvec_8bit : ctx->matvec_v3)));

    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;

    // gate_proj: data[k] -> gate[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj: data[k] -> up[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k]  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU: gate[k], up[k] -> act[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj: act[k] -> out[k]
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_data[k] offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Encode one expert forward using explicit data buffer (for double buffering).
// Expert data must already be in data_buf.
// Input must already be in buf_multi_expert_input.
// Uses slot k's gate/up/act/out scratch buffers.
static void gpu_encode_expert_forward_slot_buf(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int k,                  // slot index (for gate/up/act/out scratch)
    id<MTLBuffer> data_buf  // expert weight data buffer (from either set A or B)
) {
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_1bit) {
        gate_w_off = GATE_W_OFF_1; gate_s_off = GATE_S_OFF_1; gate_b_off = GATE_B_OFF_1;
        up_w_off   = UP_W_OFF_1;   up_s_off   = UP_S_OFF_1;   up_b_off   = UP_B_OFF_1;
        down_w_off = DOWN_W_OFF_1; down_s_off = DOWN_S_OFF_1; down_b_off = DOWN_B_OFF_1;
    } else if (g_use_2bit) {
        gate_w_off = GATE_W_OFF_2; gate_s_off = GATE_S_OFF_2; gate_b_off = GATE_B_OFF_2;
        up_w_off   = UP_W_OFF_2;   up_s_off   = UP_S_OFF_2;   up_b_off   = UP_B_OFF_2;
        down_w_off = DOWN_W_OFF_2; down_s_off = DOWN_S_OFF_2; down_b_off = DOWN_B_OFF_2;
    } else if (g_use_3bit) {
        gate_w_off = GATE_W_OFF_3; gate_s_off = GATE_S_OFF_3; gate_b_off = GATE_B_OFF_3;
        up_w_off   = UP_W_OFF_3;   up_s_off   = UP_S_OFF_3;   up_b_off   = UP_B_OFF_3;
        down_w_off = DOWN_W_OFF_3; down_s_off = DOWN_S_OFF_3; down_b_off = DOWN_B_OFF_3;
    } else if (g_use_int8) {
        gate_w_off = GATE_W_OFF_8; gate_s_off = GATE_S_OFF_8; gate_b_off = GATE_B_OFF_8;
        up_w_off   = UP_W_OFF_8;   up_s_off   = UP_S_OFF_8;   up_b_off   = UP_B_OFF_8;
        down_w_off = DOWN_W_OFF_8; down_s_off = DOWN_S_OFF_8; down_b_off = DOWN_B_OFF_8;
    } else {
        gate_w_off = GATE_W_OFF_4; gate_s_off = GATE_S_OFF_4; gate_b_off = GATE_B_OFF_4;
        up_w_off   = UP_W_OFF_4;   up_s_off   = UP_S_OFF_4;   up_b_off   = UP_B_OFF_4;
        down_w_off = DOWN_W_OFF_4; down_s_off = DOWN_S_OFF_4; down_b_off = DOWN_B_OFF_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_3bit ? ctx->matvec_3bit : (g_use_2bit ? ctx->matvec_2bit : (g_use_1bit ? ctx->matvec_1bit : (g_use_int8 ? ctx->matvec_8bit : ctx->matvec_v3)));

    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:gate_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:gate_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_gate[k]   offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:up_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:up_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_input     offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_up[k]     offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_multi_expert_gate[k] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_multi_expert_up[k]   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_multi_expert_act[k]  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_buf                        offset:down_w_off  atIndex:0];
        [enc setBuffer:data_buf                        offset:down_s_off  atIndex:1];
        [enc setBuffer:data_buf                        offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_multi_expert_act[k]    offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]    offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched expert encoding: encode K experts using 2 encoders per expert
// (gate+up fused, SwiGLU+down fused) + 2 for shared = K*2 + 2 encoders total.
// With K=8: 18 encoders (vs. old 4*K + 2 = 34 with per-operation encoding).
// Each expert gets its own encoder pair for GPU parallelism across experts.
// Within each encoder, gate+up (or SwiGLU+down) are serialized but share
// encoder creation overhead. Net win: fewer encoders, same parallelism.
static void gpu_encode_experts_batched(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf,
    int K,                       // number of experts to encode
    const int *valid,            // which experts are valid [MAX_K]
    id<MTLBuffer> __strong *expert_bufs,  // per-expert weight data buffers [MAX_K]
    NSUInteger out_offset,       // byte offset for expert output buffers (prefill slot m)
    id<MTLBuffer> __unsafe_unretained *data_bufs,  // pool mode: per-expert weight buffers [MAX_K]
    const NSUInteger *data_offs, // pool mode: per-expert byte offsets (base of expert slot)
    uint32_t slot_m              // pool mode: position slot for input/gate/up/act bindings
) {
    // Select offsets and pipeline based on quantization mode
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_1bit) {
        gate_w_off = GATE_W_OFF_1; gate_s_off = GATE_S_OFF_1; gate_b_off = GATE_B_OFF_1;
        up_w_off   = UP_W_OFF_1;   up_s_off   = UP_S_OFF_1;   up_b_off   = UP_B_OFF_1;
        down_w_off = DOWN_W_OFF_1; down_s_off = DOWN_S_OFF_1; down_b_off = DOWN_B_OFF_1;
    } else if (g_use_2bit) {
        gate_w_off = GATE_W_OFF_2; gate_s_off = GATE_S_OFF_2; gate_b_off = GATE_B_OFF_2;
        up_w_off   = UP_W_OFF_2;   up_s_off   = UP_S_OFF_2;   up_b_off   = UP_B_OFF_2;
        down_w_off = DOWN_W_OFF_2; down_s_off = DOWN_S_OFF_2; down_b_off = DOWN_B_OFF_2;
    } else if (g_use_3bit) {
        gate_w_off = GATE_W_OFF_3; gate_s_off = GATE_S_OFF_3; gate_b_off = GATE_B_OFF_3;
        up_w_off   = UP_W_OFF_3;   up_s_off   = UP_S_OFF_3;   up_b_off   = UP_B_OFF_3;
        down_w_off = DOWN_W_OFF_3; down_s_off = DOWN_S_OFF_3; down_b_off = DOWN_B_OFF_3;
    } else if (g_use_int8) {
        gate_w_off = GATE_W_OFF_8; gate_s_off = GATE_S_OFF_8; gate_b_off = GATE_B_OFF_8;
        up_w_off   = UP_W_OFF_8;   up_s_off   = UP_S_OFF_8;   up_b_off   = UP_B_OFF_8;
        down_w_off = DOWN_W_OFF_8; down_s_off = DOWN_S_OFF_8; down_b_off = DOWN_B_OFF_8;
    } else {
        gate_w_off = GATE_W_OFF_4; gate_s_off = GATE_S_OFF_4; gate_b_off = GATE_B_OFF_4;
        up_w_off   = UP_W_OFF_4;   up_s_off   = UP_S_OFF_4;   up_b_off   = UP_B_OFF_4;
        down_w_off = DOWN_W_OFF_4; down_s_off = DOWN_S_OFF_4; down_b_off = DOWN_B_OFF_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_3bit ? ctx->matvec_3bit : (g_use_2bit ? ctx->matvec_2bit : (g_use_1bit ? ctx->matvec_1bit : (g_use_int8 ? ctx->matvec_8bit : ctx->matvec_v3)));
    // SiLU formula corrected in shaders: vg/(1+exp(-vg))*vu = SiLU(gate)*up.
    // Fused path verified bit-identical in isolation (finchTool).
    // For production: non-fused path for 4-bit (verified 6.2 tok/s coherent).
    // TODO: enable fused for 4-bit after inter-CB sync diagnosis.
    id<MTLComputePipelineState> fused_pipe = g_use_int8 ? ctx->fused_gate_up_swiglu_8bit_pipe : NULL;

    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;
    uint32_t gate_up_tgs = (gate_up_out + 7) / 8;
    uint32_t down_tgs    = (down_out + 7) / 8;

    // 2x path: both experts' gate+up+swiglu in ONE dispatch (K=2, both valid)
    if (0 && K == 2 && valid[0] && valid[1] && fused_pipe && ctx->fused_gate_up_swiglu_2x_pipe) {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->fused_gate_up_swiglu_2x_pipe];
        [enc setBuffer:expert_bufs[0]              offset:gate_w_off  atIndex:0];
        [enc setBuffer:expert_bufs[0]              offset:gate_s_off  atIndex:1];
        [enc setBuffer:expert_bufs[0]              offset:gate_b_off  atIndex:2];
        [enc setBuffer:expert_bufs[0]              offset:up_w_off    atIndex:3];
        [enc setBuffer:expert_bufs[0]              offset:up_s_off    atIndex:4];
        [enc setBuffer:expert_bufs[0]              offset:up_b_off    atIndex:5];
        [enc setBuffer:expert_bufs[1]              offset:gate_w_off  atIndex:6];
        [enc setBuffer:expert_bufs[1]              offset:gate_s_off  atIndex:7];
        [enc setBuffer:expert_bufs[1]              offset:gate_b_off  atIndex:8];
        [enc setBuffer:expert_bufs[1]              offset:up_w_off    atIndex:9];
        [enc setBuffer:expert_bufs[1]              offset:up_s_off    atIndex:10];
        [enc setBuffer:expert_bufs[1]              offset:up_b_off    atIndex:11];
        [enc setBuffer:ctx->buf_multi_expert_input offset:0           atIndex:12];
        [enc setBuffer:ctx->buf_multi_expert_act[0] offset:0          atIndex:13];
        [enc setBuffer:ctx->buf_multi_expert_act[1] offset:0          atIndex:14];
        [enc setBytes:&gate_up_out length:4 atIndex:15];
        [enc setBytes:&gate_up_in  length:4 atIndex:16];
        [enc setBytes:&gs          length:4 atIndex:17];
        // 2x kernel: 1 row per threadgroup (no ROWS_PER_TG), needs out_dim TGs.
        uint32_t gate_up_x2_tgs = gate_up_out;
        [enc dispatchThreadgroups:MTLSizeMake(gate_up_x2_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        // Barrier: 2x fused writes -> down_proj reads (pipeline state change
        // does NOT implicitly insert a barrier per Apple Metal docs).
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        for (int k = 0; k < 2; k++) {
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:expert_bufs[k]               offset:down_w_off  atIndex:0];
            [enc setBuffer:expert_bufs[k]               offset:down_s_off  atIndex:1];
            [enc setBuffer:expert_bufs[k]               offset:down_b_off  atIndex:2];
            [enc setBuffer:ctx->buf_multi_expert_act[k] offset:0           atIndex:3];
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:out_offset    atIndex:4];
            [enc setBytes:&down_out length:4 atIndex:5];
            [enc setBytes:&down_in  length:4 atIndex:6];
            [enc setBytes:&gs       length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
    } else {
        // FUSED PATH: dedicated synchronous CB per expert.
        // SiLU formula: vg/(1+exp(-vg))*vu = SiLU(gate)*up (verified correct).
        // MTLSharedEvent + MTLFence scaffolding for inter-CB sync.
        if (fused_pipe) {
            for (int k = 0; k < K; k++) {
                if (!valid[k]) continue;
                id<MTLCommandBuffer> fb = [ctx->queue commandBuffer];
                // Encoder 1: fused gate+up+swiglu
                {
                    id<MTLComputeCommandEncoder> fe = [fb computeCommandEncoder];
                    [fe setComputePipelineState:fused_pipe];
                    [fe setBuffer:expert_bufs[k]              offset:gate_w_off  atIndex:0];
                    [fe setBuffer:expert_bufs[k]              offset:gate_s_off  atIndex:1];
                    [fe setBuffer:expert_bufs[k]              offset:gate_b_off  atIndex:2];
                    [fe setBuffer:expert_bufs[k]              offset:up_w_off    atIndex:3];
                    [fe setBuffer:expert_bufs[k]              offset:up_s_off    atIndex:4];
                    [fe setBuffer:expert_bufs[k]              offset:up_b_off    atIndex:5];
                    [fe setBuffer:ctx->buf_multi_expert_input offset:0           atIndex:6];
                    [fe setBuffer:ctx->buf_multi_expert_act[k] offset:0          atIndex:7];
                    [fe setBytes:&gate_up_out length:4 atIndex:8];
                    [fe setBytes:&gate_up_in  length:4 atIndex:9];
                    [fe setBytes:&gs          length:4 atIndex:10];
                    uint32_t tgs = gate_up_out;
                    [fe dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [fe endEncoding];
                }
                // Encoder 2: down_proj (reads act[k], writes out[k])
                {
                    id<MTLComputeCommandEncoder> de = [fb computeCommandEncoder];
                    [de setComputePipelineState:expert_pipe];
                    [de setBuffer:expert_bufs[k]               offset:down_w_off  atIndex:0];
                    [de setBuffer:expert_bufs[k]               offset:down_s_off  atIndex:1];
                    [de setBuffer:expert_bufs[k]               offset:down_b_off  atIndex:2];
                    [de setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
                    [de setBuffer:ctx->buf_multi_expert_out[k]  offset:out_offset    atIndex:4];
                    [de setBytes:&down_out length:4 atIndex:5];
                    [de setBytes:&down_in  length:4 atIndex:6];
                    [de setBytes:&gs       length:4 atIndex:7];
                    [de dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    // Signal fence: guarantees out[k] writes are visible to
                    // subsequent command buffers that wait on this fence.
                    [de updateFence:ctx->expert_fence];
                    [de endEncoding];
                }
                // Signal event BEFORE commit: GPU signals after all dispatches finish
                ctx->expert_sync_value++;
                [fb encodeSignalEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
                [fb commit];
                // CPU-side guarantee that out[k] is valid
                [fb waitUntilCompleted];
            }
            return;  // done — out[k] buffers are valid, skip non-fused path entirely
        }

        for (int k = 0; k < K; k++) {
        if (!valid[k]) continue;
        // Pool mode (data_bufs != NULL): rebind weight data, input, and
        // gate/up/act scratch to position slot_m's regions. Pool mode only
        // runs on the non-fused path (g_use_int8 excluded by the caller).
        NSUInteger exp_alloc = (EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);
        id<MTLBuffer> data_src_buf = data_bufs ? data_bufs[k] : expert_bufs[k];
        NSUInteger data_src_off   = data_bufs ? data_offs[k] : 0;
        id<MTLBuffer> in_buf  = data_bufs ? ctx->buf_pf_expert_input : ctx->buf_multi_expert_input;
        NSUInteger in_off     = data_bufs ? (NSUInteger)slot_m * HIDDEN_DIM * sizeof(float) : 0;
        id<MTLBuffer> gate_buf = data_bufs ? ctx->buf_pf_expert_gate[k] : ctx->buf_multi_expert_gate[k];
        id<MTLBuffer> up_buf   = data_bufs ? ctx->buf_pf_expert_up[k]   : ctx->buf_multi_expert_up[k];
        id<MTLBuffer> act_buf  = data_bufs ? ctx->buf_pf_expert_act[k]  : ctx->buf_multi_expert_act[k];
        NSUInteger mid_off    = data_bufs ? (NSUInteger)slot_m * MOE_INTERMEDIATE * sizeof(float) : 0;
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        if (fused_pipe) {
            static int fused_call_count = 0;
            if (fused_call_count < 30) {
                fprintf(stderr, "[FUSED-CALL] k=%d layer expert_buf=%p\n", k, (__bridge void*)expert_bufs[k]);
                fused_call_count++;
            }
            // Fused gate+up+swiglu + down_proj on DEDICATED synchronous CB.
            // Runs fused kernel FIRST, waits, then runs down_proj on same CB.
            // This eliminates any cross-CB or encoder-sharing hazards.
            {
                id<MTLCommandBuffer> fb = [ctx->queue commandBuffer];
                id<MTLComputeCommandEncoder> fe = [fb computeCommandEncoder];
                [fe setComputePipelineState:fused_pipe];
                [fe setBuffer:expert_bufs[k]              offset:gate_w_off  atIndex:0];
                [fe setBuffer:expert_bufs[k]              offset:gate_s_off  atIndex:1];
                [fe setBuffer:expert_bufs[k]              offset:gate_b_off  atIndex:2];
                [fe setBuffer:expert_bufs[k]              offset:up_w_off    atIndex:3];
                [fe setBuffer:expert_bufs[k]              offset:up_s_off    atIndex:4];
                [fe setBuffer:expert_bufs[k]              offset:up_b_off    atIndex:5];
                [fe setBuffer:ctx->buf_multi_expert_input offset:0           atIndex:6];
                [fe setBuffer:ctx->buf_multi_expert_act[k] offset:0          atIndex:7];
                [fe setBytes:&gate_up_out length:4 atIndex:8];
                [fe setBytes:&gate_up_in  length:4 atIndex:9];
                [fe setBytes:&gs          length:4 atIndex:10];
                uint32_t gate_up_1x_tgs = gate_up_out;
                [fe dispatchThreadgroups:MTLSizeMake(gate_up_1x_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [fe endEncoding];
                // Now down_proj on same CB — guaranteed to see fused writes
                id<MTLComputeCommandEncoder> de = [fb computeCommandEncoder];
                [de setComputePipelineState:expert_pipe];
                [de setBuffer:expert_bufs[k]               offset:down_w_off  atIndex:0];
                [de setBuffer:expert_bufs[k]               offset:down_s_off  atIndex:1];
                [de setBuffer:expert_bufs[k]               offset:down_b_off  atIndex:2];
                [de setBuffer:ctx->buf_multi_expert_act[k]  offset:0           atIndex:3];
                [de setBuffer:ctx->buf_multi_expert_out[k]  offset:out_offset    atIndex:4];
                [de setBytes:&down_out length:4 atIndex:5];
                [de setBytes:&down_in  length:4 atIndex:6];
                [de setBytes:&gs       length:4 atIndex:7];
                [de dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [de endEncoding];
                [fb commit];
                [fb waitUntilCompleted];  // synchronous: out[k] is now valid
            }

            // DIAGNOSTIC: verify full expert output (gate→up→swiglu→down)
            // matches CPU computation. DISABLED for Private storage mode
            // (Private buffers can't be read via [buffer contents]).
#if 0
            static int diag_count = 0;
            if (diag_count < 30) {
                diag_count++;
                void *edata = malloc(active_expert_size());
                memcpy(edata, [expert_bufs[k] contents], active_expert_size());
                float *fused_act = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *fused_out = malloc(HIDDEN_DIM * sizeof(float));
                memcpy(fused_act, [ctx->buf_multi_expert_act[k] contents], MOE_INTERMEDIATE*sizeof(float));
                memcpy(fused_out, [ctx->buf_multi_expert_out[k] contents], HIDDEN_DIM*sizeof(float));
                float *inp = malloc(HIDDEN_DIM * sizeof(float));
                memcpy(inp, [ctx->buf_multi_expert_input contents], HIDDEN_DIM*sizeof(float));

                // CPU full expert: gate + up + swiglu + down
                float *cpu_gate = malloc(MOE_INTERMEDIATE*sizeof(float));
                float *cpu_up   = malloc(MOE_INTERMEDIATE*sizeof(float));
                float *cpu_act  = malloc(MOE_INTERMEDIATE*sizeof(float));
                float *cpu_out  = malloc(HIDDEN_DIM*sizeof(float));
                uint32_t *gw = (uint32_t*)((char*)edata + GATE_W_OFF_4);
                uint16_t *gs = (uint16_t*)((char*)edata + GATE_S_OFF_4);
                uint16_t *gb = (uint16_t*)((char*)edata + GATE_B_OFF_4);
                uint32_t *uw = (uint32_t*)((char*)edata + UP_W_OFF_4);
                uint16_t *us = (uint16_t*)((char*)edata + UP_S_OFF_4);
                uint16_t *ub = (uint16_t*)((char*)edata + UP_B_OFF_4);
                uint32_t *dw = (uint32_t*)((char*)edata + DOWN_W_OFF_4);
                uint16_t *ds = (uint16_t*)((char*)edata + DOWN_S_OFF_4);
                uint16_t *db = (uint16_t*)((char*)edata + DOWN_B_OFF_4);
                cpu_dequant_matvec(gw, gs, gb, inp, cpu_gate, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 4);
                cpu_dequant_matvec(uw, us, ub, inp, cpu_up,   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 4);
                for (int i = 0; i < MOE_INTERMEDIATE; i++) {
                    float g = cpu_gate[i];
                    cpu_act[i] = (g / (1.0f + expf(-g))) * cpu_up[i];  // SiLU(g) * up
                }
                cpu_dequant_matvec(dw, ds, db, cpu_act, cpu_out, HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, 4);

                float max_da = 0; int bad_a = -1;
                for (int i = 0; i < MOE_INTERMEDIATE; i++) {
                    float d = fabsf(cpu_act[i] - fused_act[i]);
                    if (d > max_da) { max_da = d; bad_a = i; }
                }
                float max_do = 0; int bad_o = -1;
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float d = fabsf(cpu_out[i] - fused_out[i]);
                    if (d > max_do) { max_do = d; bad_o = i; }
                }
                fprintf(stderr, "[DIAG-FULL] k=%d act:max_diff=%.2e out:max_diff=%.2e\n",
                    k, max_da, max_do);

                free(edata); free(fused_act); free(fused_out); free(inp);
                free(cpu_gate); free(cpu_up); free(cpu_act); free(cpu_out);
            }
#endif  // DIAG-FULL (disabled for Private storage mode)
            // Fused path already ran down_proj on dedicated CB — skip shared down_proj
            [enc endEncoding];
            continue;
        } else {
            // Fallback: separate gate + up dispatches
            [enc setComputePipelineState:expert_pipe];
            [enc setBuffer:data_src_buf                  offset:data_src_off + gate_w_off  atIndex:0];
            [enc setBuffer:data_src_buf                  offset:data_src_off + gate_s_off  atIndex:1];
            [enc setBuffer:data_src_buf                  offset:data_src_off + gate_b_off  atIndex:2];
            [enc setBuffer:in_buf                        offset:in_off       atIndex:3];
            [enc setBuffer:gate_buf                      offset:mid_off      atIndex:4];
            [enc setBytes:&gate_up_out length:4 atIndex:5];
            [enc setBytes:&gate_up_in  length:4 atIndex:6];
            [enc setBytes:&gs          length:4 atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // up_proj
            [enc setBuffer:data_src_buf                  offset:data_src_off + up_w_off  atIndex:0];
            [enc setBuffer:data_src_buf                  offset:data_src_off + up_s_off  atIndex:1];
            [enc setBuffer:data_src_buf                  offset:data_src_off + up_b_off  atIndex:2];
            [enc setBuffer:up_buf                        offset:mid_off      atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(gate_up_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // SwiGLU
            [enc setComputePipelineState:ctx->swiglu];
            [enc setBuffer:gate_buf offset:mid_off atIndex:0];
            [enc setBuffer:up_buf   offset:mid_off atIndex:1];
            [enc setBuffer:act_buf  offset:mid_off atIndex:2];
            [enc setBytes:&gate_up_out length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((gate_up_out + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Barrier: swiglu writes -> down_proj reads across pipeline change
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        // down_proj (always needed)
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:data_src_buf                  offset:data_src_off + down_w_off  atIndex:0];
        [enc setBuffer:data_src_buf                  offset:data_src_off + down_s_off  atIndex:1];
        [enc setBuffer:data_src_buf                  offset:data_src_off + down_b_off  atIndex:2];
        [enc setBuffer:act_buf                       offset:mid_off      atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k]  offset:out_offset   atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(down_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc endEncoding];
    }
    } // end else (non-2x path)
}

// Encode one expert forward (gate+up+swiglu+down) into cmdbuf.
// Expert data must already be in buf_expert_data.
// Input must already be in buf_expert_input.
__attribute__((unused))
static void gpu_encode_expert_forward(
    MetalCtx *ctx,
    id<MTLCommandBuffer> cmdbuf
) {
    NSUInteger gate_w_off = 0;
    NSUInteger gate_s_off = 524288;
    NSUInteger gate_b_off = 557056;
    NSUInteger up_w_off   = 589824;
    NSUInteger up_s_off   = 1114112;
    NSUInteger up_b_off   = 1146880;
    NSUInteger down_w_off = 1179648;
    NSUInteger down_s_off = 1703936;
    NSUInteger down_b_off = 1736704;

    uint32_t gate_up_out = MOE_INTERMEDIATE;
    uint32_t gate_up_in  = HIDDEN_DIM;
    uint32_t down_out    = HIDDEN_DIM;
    uint32_t down_in     = MOE_INTERMEDIATE;
    uint32_t gs          = GROUP_SIZE;

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // down_proj
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
}

// Batched wrapper: takes N matmul specs sharing the same input, dispatches
// via GPU batch if available, otherwise falls back to CPU.
static void fast_batch_matvec(
    const float *x, uint32_t x_dim,
    BatchMatvecSpec *specs, int num_specs
) {
    // GGUF mode: the GPU kernels don't know Q4_K/Q6_K (bits 10/11) yet —
    // stay on the CPU dequant path until the GPU block kernels land.
    if (g_metal && (g_metal->wf_buf || g_gguf_stage)) {
        gpu_batch_matvec(g_metal, x, x_dim, specs, num_specs);
    } else {
        for (int i = 0; i < num_specs; i++) {
            BatchMatvecSpec *s = &specs[i];
            cpu_dequant_matvec(s->W, s->scales, s->biases, x, s->out_cpu,
                               s->out_dim, s->in_dim, s->group_size, s->bits ? s->bits : 4);
        }
    }
}

// ============================================================================
// GPU expert forward: gate+up matvec -> SwiGLU -> down matvec
// All 3 matmuls + activation in a single command buffer submission.
// Expert data is copied into a reusable Metal buffer.
// ============================================================================

// expert_data_already_in_buffer: if true, expert data is already in buf_expert_data
//   (pread'd directly into it), skip the copy.
__attribute__((unused))
static void gpu_expert_forward(
    MetalCtx *ctx,
    const void *expert_data,     // EXPERT_SIZE_MAX bytes (may be buf_expert_data contents)
    const float *h_post,         // [HIDDEN_DIM] input
    float *expert_out,           // [HIDDEN_DIM] output
    int expert_data_already_in_buffer
) {
    // Expert layout offsets — select based on quantization mode
    NSUInteger gate_w_off, gate_s_off, gate_b_off;
    NSUInteger up_w_off, up_s_off, up_b_off;
    NSUInteger down_w_off, down_s_off, down_b_off;
    if (g_use_1bit) {
        gate_w_off = GATE_W_OFF_1; gate_s_off = GATE_S_OFF_1; gate_b_off = GATE_B_OFF_1;
        up_w_off   = UP_W_OFF_1;   up_s_off   = UP_S_OFF_1;   up_b_off   = UP_B_OFF_1;
        down_w_off = DOWN_W_OFF_1; down_s_off = DOWN_S_OFF_1; down_b_off = DOWN_B_OFF_1;
    } else if (g_use_2bit) {
        gate_w_off = GATE_W_OFF_2; gate_s_off = GATE_S_OFF_2; gate_b_off = GATE_B_OFF_2;
        up_w_off   = UP_W_OFF_2;   up_s_off   = UP_S_OFF_2;   up_b_off   = UP_B_OFF_2;
        down_w_off = DOWN_W_OFF_2; down_s_off = DOWN_S_OFF_2; down_b_off = DOWN_B_OFF_2;
    } else if (g_use_3bit) {
        gate_w_off = GATE_W_OFF_3; gate_s_off = GATE_S_OFF_3; gate_b_off = GATE_B_OFF_3;
        up_w_off   = UP_W_OFF_3;   up_s_off   = UP_S_OFF_3;   up_b_off   = UP_B_OFF_3;
        down_w_off = DOWN_W_OFF_3; down_s_off = DOWN_S_OFF_3; down_b_off = DOWN_B_OFF_3;
    } else if (g_use_int8) {
        gate_w_off = GATE_W_OFF_8; gate_s_off = GATE_S_OFF_8; gate_b_off = GATE_B_OFF_8;
        up_w_off   = UP_W_OFF_8;   up_s_off   = UP_S_OFF_8;   up_b_off   = UP_B_OFF_8;
        down_w_off = DOWN_W_OFF_8; down_s_off = DOWN_S_OFF_8; down_b_off = DOWN_B_OFF_8;
    } else {
        gate_w_off = GATE_W_OFF_4; gate_s_off = GATE_S_OFF_4; gate_b_off = GATE_B_OFF_4;
        up_w_off   = UP_W_OFF_4;   up_s_off   = UP_S_OFF_4;   up_b_off   = UP_B_OFF_4;
        down_w_off = DOWN_W_OFF_4; down_s_off = DOWN_S_OFF_4; down_b_off = DOWN_B_OFF_4;
    }
    id<MTLComputePipelineState> expert_pipe = g_use_3bit ? ctx->matvec_3bit : (g_use_2bit ? ctx->matvec_2bit : (g_use_1bit ? ctx->matvec_1bit : (g_use_int8 ? ctx->matvec_8bit : ctx->matvec_v3)));

    // Copy expert weights into Metal buffer only if not already there
    if (!expert_data_already_in_buffer) {
        memcpy([ctx->buf_expert_data contents], expert_data, active_expert_size());
    }
    memcpy([ctx->buf_expert_input contents], h_post, HIDDEN_DIM * sizeof(float));

    uint32_t gate_up_out = MOE_INTERMEDIATE;  // 512
    uint32_t gate_up_in  = HIDDEN_DIM;        // 2048
    uint32_t down_out    = HIDDEN_DIM;        // 2048
    uint32_t down_in     = MOE_INTERMEDIATE;  // 512
    uint32_t gs          = GROUP_SIZE;        // 64

    // Build one command buffer with all 4 dispatches:
    // 1. gate_proj matvec (h_post -> gate_out)
    // 2. up_proj matvec (h_post -> up_out)
    // 3. SwiGLU (gate_out, up_out -> act_out)
    // 4. down_proj matvec (act_out -> expert_out)

    id<MTLCommandBuffer> cmdbuf = [ctx->queue commandBuffer];

    // --- Dispatch 1: gate_proj [2048] -> [512] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:gate_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_gate  offset:0           atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 2: up_proj [2048] -> [512] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data  offset:up_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data  offset:up_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data  offset:up_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_input offset:0          atIndex:3];
        [enc setBuffer:ctx->buf_expert_up    offset:0          atIndex:4];
        [enc setBytes:&gate_up_out length:4 atIndex:5];
        [enc setBytes:&gate_up_in  length:4 atIndex:6];
        [enc setBytes:&gs          length:4 atIndex:7];
        uint32_t num_tgs = (gate_up_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 3: SwiGLU(gate, up) -> act ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_expert_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_expert_up   offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_expert_act  offset:0 atIndex:2];
        [enc setBytes:&gate_up_out length:4 atIndex:3];
        uint32_t swiglu_tgs = (gate_up_out + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // --- Dispatch 4: down_proj [512] -> [2048] ---
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:expert_pipe];
        [enc setBuffer:ctx->buf_expert_data offset:down_w_off  atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:down_s_off  atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:down_b_off  atIndex:2];
        [enc setBuffer:ctx->buf_expert_act  offset:0           atIndex:3];
        [enc setBuffer:ctx->buf_expert_out  offset:0           atIndex:4];
        [enc setBytes:&down_out length:4 atIndex:5];
        [enc setBytes:&down_in  length:4 atIndex:6];
        [enc setBytes:&gs       length:4 atIndex:7];
        uint32_t num_tgs = (down_out + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];

    // Copy result back to CPU
    memcpy(expert_out, [ctx->buf_expert_out contents], HIDDEN_DIM * sizeof(float));
}

// ============================================================================
// Rotary position embedding (for full attention layers)
// ============================================================================

static void apply_rotary_emb(float *q, float *k, int pos, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim) {
    // Apply RoPE to the first rotary_dim dimensions of each head
    // NON-TRADITIONAL (MLX default): pairs are (x[i], x[i + half_dim])
    // where half_dim = rotary_dim / 2
    int half = rotary_dim / 2;
    for (int h = 0; h < num_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float q0 = qh[i];
            float q1 = qh[i + half];
            qh[i]        = q0 * cos_a - q1 * sin_a;
            qh[i + half]  = q0 * sin_a + q1 * cos_a;
        }
    }
    for (int h = 0; h < num_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / rotary_dim);
            float angle = (float)pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);

            float k0 = kh[i];
            float k1 = kh[i + half];
            kh[i]        = k0 * cos_a - k1 * sin_a;
            kh[i + half]  = k0 * sin_a + k1 * cos_a;
        }
    }
}

// ============================================================================
// KV Cache for full attention layers
// ============================================================================

typedef struct {
    float *k_cache;  // [max_seq, num_kv_heads * head_dim] (FP32 mode)
    float *v_cache;  // [max_seq, num_kv_heads * head_dim] (FP32 mode)
    // Quantized storage (FP16 / TURBO modes)
    uint16_t *k16;       // FP16 mode: K halfs
    uint16_t *v16;       // FP16 mode: V halfs
    uint8_t  *k8;        // TURBO mode: K int8 + per-group float scales
    float    *k8_scales; // [max_seq * kv_dim/64]
    uint32_t *v4;        // TURBO mode: V 4-bit affine packed (8/uint32)
    float    *v4_scales; // [max_seq * kv_dim/64]
    float    *v4_biases; // [max_seq * kv_dim/64]
    int len;             // current number of cached entries
    int kv_type;         // 0 = FP32, 1 = FP16, 2 = TURBO (K int8 + V 4-bit)
} KVCache;

// KV mode (--kv-fp16 / --kv-turbo): FP32 is the default; the quant modes
// trade a little attention fidelity for 2x (FP16) or ~5x (TURBO) smaller
// KV caches. TURBO keeps K at 8-bit (scores are K-sensitive) and V at 4-bit.
#define KV_FP32 0
#define KV_FP16 1
#define KV_TURBO 2
static int g_kv_type = KV_FP32;

#define KV_DIM (NUM_KV_HEADS * HEAD_DIM)  // 512
#define KV_GROUPS (KV_DIM / 64)           // 8

static void kv_write(KVCache *kv, int pos, const float *k, const float *v) {
    if (kv->kv_type == KV_FP32) {
        memcpy(kv->k_cache + (size_t)pos * KV_DIM, k, KV_DIM * sizeof(float));
        memcpy(kv->v_cache + (size_t)pos * KV_DIM, v, KV_DIM * sizeof(float));
    } else if (kv->kv_type == KV_FP16) {
        uint16_t *k16 = kv->k16 + (size_t)pos * KV_DIM;
        uint16_t *v16 = kv->v16 + (size_t)pos * KV_DIM;
        for (int i = 0; i < KV_DIM; i++) {
            uint32_t kb; memcpy(&kb, &k[i], 4); k16[i] = (uint16_t)(kb >> 16);
            uint32_t vb; memcpy(&vb, &v[i], 4); v16[i] = (uint16_t)(vb >> 16);
        }
    } else {  // KV_TURBO: K int8 symmetric, V 4-bit affine (group 64)
        uint8_t *k8 = kv->k8 + (size_t)pos * KV_DIM;
        float *ks = kv->k8_scales + (size_t)pos * KV_GROUPS;
        for (int g = 0; g < KV_GROUPS; g++) {
            float mx = 0.0f;
            for (int i = 0; i < 64; i++) {
                float a = fabsf(k[g * 64 + i]);
                if (a > mx) mx = a;
            }
            float scale = mx > 0.0f ? mx / 127.0f : 1.0f;
            ks[g] = scale;
            for (int i = 0; i < 64; i++) {
                int q = (int)roundf(k[g * 64 + i] / scale);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                k8[g * 64 + i] = (uint8_t)(q & 0xFF);
            }
        }
        uint32_t *v4 = kv->v4 + (size_t)pos * (KV_DIM / 8);
        float *vs = kv->v4_scales + (size_t)pos * KV_GROUPS;
        float *vb = kv->v4_biases + (size_t)pos * KV_GROUPS;
        for (int g = 0; g < KV_GROUPS; g++) {
            float mn = v[g * 64], mx = v[g * 64];
            for (int i = 1; i < 64; i++) {
                if (v[g * 64 + i] < mn) mn = v[g * 64 + i];
                if (v[g * 64 + i] > mx) mx = v[g * 64 + i];
            }
            float scale = (mx - mn) / 15.0f;
            vs[g] = scale > 0.0f ? scale : 1.0f;
            vb[g] = mn;
            for (int i = 0; i < 64; i++) {
                int q = (int)roundf((v[g * 64 + i] - mn) / vs[g]);
                if (q > 15) q = 15;
                if (q < 0) q = 0;
                v4[g * 8 + i / 8] |= ((uint32_t)q) << (4 * (i % 8));
            }
        }
    }
}

static void kv_read_k(KVCache *kv, int pos, int kv_h, float *out) {
    if (kv->kv_type == KV_FP32) {
        memcpy(out, kv->k_cache + (size_t)pos * KV_DIM + (size_t)kv_h * HEAD_DIM,
               HEAD_DIM * sizeof(float));
    } else if (kv->kv_type == KV_FP16) {
        const uint16_t *k16 = kv->k16 + (size_t)pos * KV_DIM + (size_t)kv_h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM; i++) {
            uint32_t b = (uint32_t)k16[i] << 16;
            memcpy(&out[i], &b, 4);
        }
    } else {  // KV_TURBO
        const uint8_t *k8 = kv->k8 + (size_t)pos * KV_DIM + (size_t)kv_h * HEAD_DIM;
        const float *ks = kv->k8_scales + (size_t)pos * KV_GROUPS + (size_t)kv_h * (HEAD_DIM / 64);
        for (int g = 0; g < HEAD_DIM / 64; g++) {
            for (int i = 0; i < 64; i++)
                out[g * 64 + i] = (float)(int8_t)k8[g * 64 + i] * ks[g];
        }
    }
}

static void kv_read_v(KVCache *kv, int pos, int kv_h, float *out) {
    if (kv->kv_type == KV_FP32) {
        memcpy(out, kv->v_cache + (size_t)pos * KV_DIM + (size_t)kv_h * HEAD_DIM,
               HEAD_DIM * sizeof(float));
    } else if (kv->kv_type == KV_FP16) {
        const uint16_t *v16 = kv->v16 + (size_t)pos * KV_DIM + (size_t)kv_h * HEAD_DIM;
        for (int i = 0; i < HEAD_DIM; i++) {
            uint32_t b = (uint32_t)v16[i] << 16;
            memcpy(&out[i], &b, 4);
        }
    } else {  // KV_TURBO
        const uint32_t *v4 = kv->v4 + (size_t)pos * (KV_DIM / 8) + (size_t)kv_h * (HEAD_DIM / 8);
        const float *vs = kv->v4_scales + (size_t)pos * KV_GROUPS + (size_t)kv_h * (HEAD_DIM / 64);
        const float *vb = kv->v4_biases + (size_t)pos * KV_GROUPS + (size_t)kv_h * (HEAD_DIM / 64);
        for (int g = 0; g < HEAD_DIM / 64; g++) {
            for (int j = 0; j < 8; j++) {
                uint32_t w = v4[g * 8 + j];
                for (int i = 0; i < 8; i++)
                    out[g * 64 + j * 8 + i] = (float)((w >> (4 * i)) & 0xF) * vs[g] + vb[g];
            }
        }
    }
}

static KVCache *kv_cache_new(void) {
    KVCache *c = calloc(1, sizeof(KVCache));
    size_t n = (size_t)g_max_seq_len * KV_DIM;
    c->kv_type = g_kv_type;
    if (c->kv_type == KV_FP32) {
        c->k_cache = calloc(n, sizeof(float));
        c->v_cache = calloc(n, sizeof(float));
    } else if (c->kv_type == KV_FP16) {
        c->k16 = calloc(n, sizeof(uint16_t));
        c->v16 = calloc(n, sizeof(uint16_t));
    } else {
        c->k8 = calloc(n, sizeof(uint8_t));
        c->k8_scales = calloc((size_t)g_max_seq_len * KV_GROUPS, sizeof(float));
        c->v4 = calloc(n / 8, sizeof(uint32_t));
        c->v4_scales = calloc((size_t)g_max_seq_len * KV_GROUPS, sizeof(float));
        c->v4_biases = calloc((size_t)g_max_seq_len * KV_GROUPS, sizeof(float));
    }
    c->len = 0;
    return c;
}

static void kv_cache_free(KVCache *c) {
    if (c) {
        free(c->k_cache);
        free(c->v_cache);
        free(c->k16);
        free(c->v16);
        free(c->k8);
        free(c->k8_scales);
        free(c->v4);
        free(c->v4_scales);
        free(c->v4_biases);
        free(c);
    }
}

// ============================================================================
// Linear attention state (GatedDeltaNet recurrent state)
// ============================================================================

typedef struct {
    float *conv_state;  // [(kernel_size-1) * conv_dim] for conv1d
    float *ssm_state;   // [num_v_heads, head_v_dim, head_k_dim] recurrent state
} LinearAttnState;

static LinearAttnState *linear_attn_state_new(void) {
    LinearAttnState *s = calloc(1, sizeof(LinearAttnState));
    s->conv_state = calloc((CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM, sizeof(float));
    s->ssm_state = calloc(LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM, sizeof(float));
    return s;
}

static void linear_attn_state_free(LinearAttnState *s) {
    if (s) {
        free(s->conv_state);
        free(s->ssm_state);
        free(s);
    }
}

// ============================================================================
// Full attention layer forward (single token, incremental)
// ============================================================================

static int fa_debug_count = 0;

static float vec_rms(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += v[i] * v[i];
    return sqrtf(sum / n);
}

__attribute__((unused))
static void full_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,       // [HIDDEN_DIM] in/out
    KVCache *kv,
    int pos              // position in sequence
) {
    fa_debug_count++;
    int do_debug = 0;  // set to (fa_debug_count <= N) to enable debug

    char name[256];
    float *normed = malloc(HIDDEN_DIM * sizeof(float));
    float *residual = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(residual, hidden, HIDDEN_DIM);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] layer=%d pos=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, pos, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] normed_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(normed, HIDDEN_DIM), normed[0], normed[1], normed[2], normed[3], normed[4]);
    }

    // ---- QKV Projection ----
    // CRITICAL: Q projection outputs num_heads * head_dim * 2 = 16384
    // The second half is a sigmoid gate applied after attention
    int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;  // 32 * 256 * 2 = 16384
    int q_dim = NUM_ATTN_HEADS * HEAD_DIM;            // 32 * 256 = 8192
    int kv_dim = NUM_KV_HEADS * HEAD_DIM;             // 2 * 256 = 512

    float *q_proj_out = calloc(q_proj_dim, sizeof(float));
    float *k = calloc(kv_dim, sizeof(float));
    float *v = calloc(kv_dim, sizeof(float));

    // Batch Q/K/V projections into a single GPU command buffer
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", layer_idx);
    uint32_t *qw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", layer_idx);
    uint16_t *qs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", layer_idx);
    uint16_t *qb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", layer_idx);
    uint32_t *kw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", layer_idx);
    uint16_t *ks = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", layer_idx);
    uint16_t *kb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", layer_idx);
    uint32_t *vw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", layer_idx);
    uint16_t *vs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", layer_idx);
    uint16_t *vb = get_tensor_ptr(wf, name);

    // Batch Q/K/V into one command buffer (3 dispatches, 1 commit)
    if (qw && kw && vw /* BF16: scales may be NULL */) {
        char bn[256];
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.q_proj", layer_idx);
        int q_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.k_proj", layer_idx);
        int k_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.self_attn.v_proj", layer_idx);
        int v_bits = tensor_bits(wf, bn);
        BatchMatvecSpec qkv_specs[3] = {
            { qw, qs, qb, q_proj_out, (uint32_t)q_proj_dim, HIDDEN_DIM, GROUP_SIZE, 0, q_bits },
            { kw, ks, kb, k,          (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 1, k_bits },
            { vw, vs, vb, v,          (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 2, v_bits },
        };
        fast_batch_matvec(normed, HIDDEN_DIM, qkv_specs, 3);
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] q_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                q_proj_out[0], q_proj_out[1], q_proj_out[2], q_proj_out[3], q_proj_out[4]);
    }

    // Split q_proj_out into queries and gate
    float *q = calloc(q_dim, sizeof(float));
    float *q_gate = calloc(q_dim, sizeof(float));
    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        float *src = q_proj_out + h * (2 * HEAD_DIM);
        memcpy(q + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
        memcpy(q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    free(q_proj_out);

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] v_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(v, kv_dim), v[0], v[1], v[2], v[3], v[4]);
        fprintf(stderr, "[FA-DBG] q_gate_rms=%.6f gate_first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(q_gate, q_dim), q_gate[0], q_gate[1], q_gate[2], q_gate[3], q_gate[4]);
        float gate_sigmoid_sum = 0.0f;
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_sum += 1.0f / (1.0f + expf(-q_gate[i]));
        }
        fprintf(stderr, "[FA-DBG] gate_sigmoid_mean=%.6f\n", gate_sigmoid_sum / q_dim);
    }

    // ---- Q/K RMSNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", layer_idx);
    uint16_t *qnorm_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", layer_idx);
    uint16_t *knorm_w = get_tensor_ptr(wf, name);

    // Apply per-head Q norm
    if (qnorm_w) {
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            float *qh = q + h * HEAD_DIM;
            float sum_sq = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) sum_sq += qh[i] * qh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int i = 0; i < HEAD_DIM; i++) {
                qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
    }
    // Apply per-head K norm
    if (knorm_w) {
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            float *kh = k + h * HEAD_DIM;
            float sum_sq = 0.0f;
            for (int i = 0; i < HEAD_DIM; i++) sum_sq += kh[i] * kh[i];
            float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
            for (int i = 0; i < HEAD_DIM; i++) {
                kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }
    }


    // ---- RoPE ----
    apply_rotary_emb(q, k, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);

    // ---- Update KV cache ----
    int cache_pos = kv->len;
    kv_write(kv, cache_pos, k, v);
    kv->len++;

    // ---- Scaled dot-product attention ----
    // GQA: NUM_ATTN_HEADS=32 heads, NUM_KV_HEADS=2 kv heads
    // Each group of 16 query heads shares 1 kv head
    int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    float *attn_out = calloc(q_dim, sizeof(float));

    for (int h = 0; h < NUM_ATTN_HEADS; h++) {
        int kv_h = h / heads_per_kv;
        float *qh = q + h * HEAD_DIM;

        // Compute attention scores for all cached positions
        float *scores = malloc(kv->len * sizeof(float));
        static float kv_k_buf[HEAD_DIM], kv_v_buf[HEAD_DIM];
        for (int p = 0; p < kv->len; p++) {
            kv_read_k(kv, p, kv_h, kv_k_buf);
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * kv_k_buf[d];
            }
            scores[p] = dot * scale;
        }

        // Softmax
        cpu_softmax(scores, kv->len);

        // Weighted sum of values
        float *oh = attn_out + h * HEAD_DIM;
        for (int p = 0; p < kv->len; p++) {
            kv_read_v(kv, p, kv_h, kv_v_buf);
            for (int d = 0; d < HEAD_DIM; d++) {
                oh[d] += scores[p] * kv_v_buf[d];
            }
        }
        free(scores);
    }


    // ---- Apply sigmoid gate to attention output ----
    // MLX: return self.o_proj(output * mx.sigmoid(gate))
    // gate is reshaped to [B, L, num_heads*head_dim] = flat [q_dim]
    for (int i = 0; i < q_dim; i++) {
        float g = 1.0f / (1.0f + expf(-q_gate[i]));
        attn_out[i] *= g;
    }

    // ---- Output projection ----
    float *attn_projected = calloc(HIDDEN_DIM, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", layer_idx);
    uint32_t *ow = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", layer_idx);
    uint16_t *os_ptr = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", layer_idx);
    uint16_t *ob = get_tensor_ptr(wf, name);
    if (ow) {
        char obn[256];
        snprintf(obn, sizeof(obn), "model.layers.%d.self_attn.o_proj", layer_idx);
        fast_dequant_matvec(ow, os_ptr, ob, attn_out, attn_projected, HIDDEN_DIM,
                            q_dim, GROUP_SIZE, tensor_bits(wf, obn));
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] attn_out_rms=%.6f o_proj first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(attn_out, q_dim),
                attn_projected[0], attn_projected[1], attn_projected[2], attn_projected[3], attn_projected[4]);
    }

    // ---- Residual connection ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = residual[i] + attn_projected[i];
    }

    if (do_debug) {
        fprintf(stderr, "[FA-DBG] AFTER layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    free(normed);
    free(residual);
    free(q);
    free(q_gate);
    free(k);
    free(v);
    free(attn_out);
    free(attn_projected);
}

// ============================================================================
// Linear attention layer forward (GatedDeltaNet, single token, incremental)
// ============================================================================

// RMS norm without weights (just normalize)
static void cpu_rms_norm_bare(const float *x, float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv_rms;
}

// RMSNormGated: out = rms_norm(x) * silu(z)
static void cpu_rms_norm_gated(const float *x, const float *z, const uint16_t *w_bf16,
                                float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) {
        float w = bf16_to_f32(w_bf16[i]);
        float silu_z = z[i] / (1.0f + expf(-z[i]));
        out[i] = x[i] * inv_rms * w * silu_z;
    }
}

static int linear_attn_bypass = 0;  // set to 1 to skip linear attention (identity)
static int gpu_linear_attn_enabled = 1;  // fused GPU delta-net path (can disable via --cpu-linear)

__attribute__((unused))
static void linear_attention_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [HIDDEN_DIM] in/out
    LinearAttnState *state
) {
    // If bypass is enabled, just pass through (identity)
    if (linear_attn_bypass) {
        (void)wf; (void)layer_idx; (void)state;
        return;
    }

    static int la_debug_count = 0;
    la_debug_count++;
    int la_debug = 0;  // set to (la_debug_count <= N) to enable debug

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] layer=%d hidden_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(hidden, HIDDEN_DIM),
                hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    char name[256];
    float *normed = malloc(HIDDEN_DIM * sizeof(float));
    float *residual = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(residual, hidden, HIDDEN_DIM);

    // ---- Input LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);

    // ---- Batch QKV + Z + B + A projections (4 matmuls, 1 command buffer) ----
    int qkv_dim = LINEAR_CONV_DIM;  // 8192
    float *qkv = calloc(qkv_dim, sizeof(float));
    int z_dim = LINEAR_TOTAL_VALUE;  // 4096
    float *z = calloc(z_dim, sizeof(float));
    float *beta = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    float *alpha = calloc(LINEAR_NUM_V_HEADS, sizeof(float));

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", layer_idx);
    uint32_t *qkv_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", layer_idx);
    uint16_t *qkv_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", layer_idx);
    uint16_t *qkv_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", layer_idx);
    uint32_t *z_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", layer_idx);
    uint16_t *z_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", layer_idx);
    uint16_t *z_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", layer_idx);
    uint32_t *b_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", layer_idx);
    uint16_t *b_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", layer_idx);
    uint16_t *b_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", layer_idx);
    uint32_t *a_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", layer_idx);
    uint16_t *a_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", layer_idx);
    uint16_t *a_b = get_tensor_ptr(wf, name);

    if (qkv_w && z_w && b_w && a_w /* BF16: scales may be NULL */) {
        char bn[256];
        snprintf(bn, sizeof(bn), "model.layers.%d.linear_attn.in_proj_qkv", layer_idx);
        int qkv_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.linear_attn.in_proj_z", layer_idx);
        int z_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.linear_attn.in_proj_b", layer_idx);
        int b_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.linear_attn.in_proj_a", layer_idx);
        int a_bits = tensor_bits(wf, bn);
        BatchMatvecSpec la_specs[4] = {
            { qkv_w, qkv_s, qkv_b, qkv,   (uint32_t)qkv_dim,         HIDDEN_DIM, GROUP_SIZE, 0, qkv_bits },
            { z_w,   z_s,   z_b,   z,      (uint32_t)z_dim,           HIDDEN_DIM, GROUP_SIZE, 1, z_bits },
            { b_w,   b_s,   b_b,   beta,   (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 2, b_bits },
            { a_w,   a_s,   a_b,   alpha,  (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 3, a_bits },
        };
        fast_batch_matvec(normed, HIDDEN_DIM, la_specs, 4);
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] proj qkv_rms=%.6f z_rms=%.6f alpha_rms=%.6f beta_rms=%.6f\n",
                vec_rms(qkv, qkv_dim), vec_rms(z, z_dim),
                vec_rms(alpha, LINEAR_NUM_V_HEADS), vec_rms(beta, LINEAR_NUM_V_HEADS));
    }

    // ---- Conv1d step ----
    // conv_state holds last (kernel_size-1) inputs for each of the conv_dim channels
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", layer_idx);
    uint16_t *conv_w = get_tensor_ptr(wf, name);

    float *conv_out = calloc(qkv_dim, sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(state->conv_state, qkv, conv_w, conv_out,
                        qkv_dim, CONV_KERNEL_SIZE);
    }

    // Update conv state: shift left, append new input
    memmove(state->conv_state, state->conv_state + qkv_dim,
            (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
    memcpy(state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv,
           qkv_dim * sizeof(float));
    if (la_debug) {
        fprintf(stderr, "[LA-DBG] conv_out_rms=%.6f\n", vec_rms(conv_out, qkv_dim));
    }

    // ---- Split conv_out into q, k, v ----
    // q: [num_k_heads * head_k_dim] = [2048]
    // k: [num_k_heads * head_k_dim] = [2048]
    // v: [num_v_heads * head_v_dim] = [8192]
    float *lin_q = conv_out;  // first LINEAR_TOTAL_KEY elements
    float *lin_k = conv_out + LINEAR_TOTAL_KEY;  // next LINEAR_TOTAL_KEY
    float *lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;  // rest = LINEAR_TOTAL_VALUE

    // ---- RMS normalize q and k (bare, no weights) ----
    // q: scale = key_dim^(-0.5), normalize per head then scale by key_dim^(-1.0)
    // Actually from the code:
    //   inv_scale = k.shape[-1] ** -0.5 = head_k_dim^(-0.5) = 128^(-0.5)
    //   q = (inv_scale**2) * rms_norm(q) = (1/128) * rms_norm(q)
    //   k = inv_scale * rms_norm(k) = (1/sqrt(128)) * rms_norm(k)
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);

    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *qh = lin_q + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
        float q_scale = inv_scale * inv_scale;  // inv_scale^2 = 1/head_k_dim
        for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *kh = lin_k + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
        for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
    }

    // ---- Gated delta net recurrence ----
    // From gated_delta.py:
    //   g = exp(-exp(A_log) * softplus(a + dt_bias))   -- per-head decay
    //   beta_gate = sigmoid(b)                          -- per-head beta (NO dt_bias)
    //   For each v_head:
    //     state = state * g                             -- decay
    //     kv_mem = sum(state * k, axis=key_dim)         -- predict v from state
    //     delta = (v - kv_mem) * beta_gate              -- error signal
    //     state = state + outer(delta, k)               -- update state
    //     output = sum(state * q, axis=key_dim)         -- read from state

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", layer_idx);
    float *A_log = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", layer_idx);
    uint16_t *dt_bias_bf16 = get_tensor_ptr(wf, name);

    float *out_values = calloc(LINEAR_TOTAL_VALUE, sizeof(float));  // [num_v_heads * head_v_dim]

    int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;  // 32/16 = 2

    // Precompute per-head decay (g) and beta
    float g_decay[LINEAR_NUM_V_HEADS];
    float beta_gate[LINEAR_NUM_V_HEADS];
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        // g = exp(-exp(A_log) * softplus(a + dt_bias))
        float a_val = alpha[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        float A_val = A_log ? expf(A_log[vh]) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));  // softplus(a + dt_bias)
        g_decay[vh] = expf(-A_val * softplus_val);

        // beta = sigmoid(b)  (just b, NO dt_bias)
        beta_gate[vh] = cpu_sigmoid(beta[vh]);
    }

    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        // k-head mapping: torch .repeat() block convention (llama.cpp's
        // ggml_repeat) — v-heads 16..31 reuse k-heads 0..15, NOT vh/2.
        int kh = vh % LINEAR_NUM_K_HEADS;

        float g = g_decay[vh];
        float b_gate = beta_gate[vh];

        // state is [head_v_dim, head_k_dim]
        float *S = state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
        float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
        float *k_h = lin_k + kh * LINEAR_KEY_DIM;

        // Step 1: Decay state
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] *= g;
            }
        }

        // Step 2: Compute kv_mem[vi] = sum_ki(S[vi,ki] * k[ki])
        // Then delta[vi] = (v[vi] - kv_mem[vi]) * beta
        // Then state[vi,ki] += k[ki] * delta[vi]
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float kv_mem = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                kv_mem += S[vi * LINEAR_KEY_DIM + ki] * k_h[ki];
            }
            float delta = (v_h[vi] - kv_mem) * b_gate;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                S[vi * LINEAR_KEY_DIM + ki] += k_h[ki] * delta;
            }
        }

        // Step 3: Output: y[vi] = sum_ki(S[vi,ki] * q[ki])
        float *q_h = lin_q + kh * LINEAR_KEY_DIM;
        float *o_h = out_values + vh * LINEAR_VALUE_DIM;
        for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
            float sum = 0.0f;
            for (int ki = 0; ki < LINEAR_KEY_DIM; ki++) {
                sum += S[vi * LINEAR_KEY_DIM + ki] * q_h[ki];
            }
            o_h[vi] = sum;
        }
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] g_decay[0..2]=%.6f,%.6f,%.6f beta_gate[0..2]=%.6f,%.6f,%.6f out_values_rms=%.6f\n",
                g_decay[0], g_decay[1], g_decay[2],
                beta_gate[0], beta_gate[1], beta_gate[2],
                vec_rms(out_values, LINEAR_TOTAL_VALUE));
    }

    // ---- RMSNormGated: out = rms_norm(out_values_per_head) * silu(z_per_head) * weight ----
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", layer_idx);
    uint16_t *gated_norm_w = get_tensor_ptr(wf, name);

    float *gated_out = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float *oh = out_values + vh * LINEAR_VALUE_DIM;
        float *zh = z + vh * LINEAR_VALUE_DIM;
        float *gh = gated_out + vh * LINEAR_VALUE_DIM;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
        } else {
            memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
        }
    }

    // ---- Output projection: [value_dim=8192] -> [hidden_dim=2048] ----
    float *attn_out = calloc(HIDDEN_DIM, sizeof(float));
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", layer_idx);
    uint32_t *out_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", layer_idx);
    uint16_t *out_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", layer_idx);
    uint16_t *out_b = get_tensor_ptr(wf, name);
    if (out_w /* BF16: scales may be NULL */) {
        char obn[256];
        snprintf(obn, sizeof(obn), "model.layers.%d.linear_attn.out_proj", layer_idx);
        fast_dequant_matvec(out_w, out_s, out_b, gated_out, attn_out, HIDDEN_DIM,
                            LINEAR_TOTAL_VALUE, GROUP_SIZE, tensor_bits(wf, obn));
    }

    // ---- Residual ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = residual[i] + attn_out[i];
    }

    if (la_debug) {
        fprintf(stderr, "[LA-DBG] AFTER layer=%d out_proj_rms=%.6f gated_rms=%.6f hidden_rms=%.6f\n",
                layer_idx, vec_rms(attn_out, HIDDEN_DIM),
                vec_rms(gated_out, LINEAR_TOTAL_VALUE),
                vec_rms(hidden, HIDDEN_DIM));
    }

    free(normed);
    free(residual);
    free(qkv);
    free(z);
    free(beta);
    free(alpha);
    free(conv_out);
    free(out_values);
    free(gated_out);
    free(attn_out);
}

// ============================================================================
// MoE forward (routing + expert computation + shared expert)
// ============================================================================

static int moe_debug_count = 0;

__attribute__((unused))
static void moe_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,         // [HIDDEN_DIM] in/out
    const char *model_path __attribute__((unused)),
    int K,                 // number of active experts (e.g. 4)
    int packed_fd          // fd for this layer's packed expert file (-1 if not available)
) {
    moe_debug_count++;
    int moe_debug = 0;  // set to (moe_debug_count <= N) to enable debug
    int moe_dump = 0;

    char name[256];
    float *h_post = malloc(HIDDEN_DIM * sizeof(float));
    float *h_mid = malloc(HIDDEN_DIM * sizeof(float));
    cpu_vec_copy(h_mid, hidden, HIDDEN_DIM);

    // ---- Post-attention LayerNorm ----
    snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", layer_idx);
    uint16_t *norm_w = get_tensor_ptr(wf, name);
    cpu_rms_norm(hidden, norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);

    // ---- Batch routing gate + shared expert gate/up + shared_expert_gate (4 matmuls, 1 commit) ----
    float *gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    float *shared_gate = calloc(SHARED_INTERMEDIATE, sizeof(float));
    float *shared_up = calloc(SHARED_INTERMEDIATE, sizeof(float));
    float shared_gate_score = 0.0f;

    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", layer_idx);
    uint32_t *gate_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", layer_idx);
    uint16_t *gate_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", layer_idx);
    uint16_t *gate_b = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    uint32_t *sgw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    uint16_t *sgs = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    uint16_t *sgb = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    uint32_t *suw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    uint16_t *sus = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    uint16_t *sub = get_tensor_ptr(wf, name);

    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    uint32_t *seg_w = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    uint16_t *seg_s = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);
    uint16_t *seg_b = get_tensor_ptr(wf, name);

    // Routing gate + shared_expert_gate use 8-bit quantization (others use 4-bit)
    // Do 8-bit matvecs on CPU; batch 4-bit ones for GPU
    if (gate_w && sgw && suw && seg_w /* BF16: scales may be NULL */) {
        // 8-bit matvecs (CPU only — no 8-bit GPU kernel)
        cpu_dequant_matvec(gate_w, gate_s, gate_b, h_post, gate_scores,
                           NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 8);
        float seg_out;
        cpu_dequant_matvec(seg_w, seg_s, seg_b, h_post, &seg_out,
                           1, HIDDEN_DIM, GROUP_SIZE, 8);
        shared_gate_score = seg_out;  // sigmoid applied later

        // 4-bit matvecs (can use GPU)
        char bn[256];
        snprintf(bn, sizeof(bn), "model.layers.%d.mlp.shared_expert.gate_proj", layer_idx);
        int sg_bits = tensor_bits(wf, bn);
        snprintf(bn, sizeof(bn), "model.layers.%d.mlp.shared_expert.up_proj", layer_idx);
        int su_bits = tensor_bits(wf, bn);
        BatchMatvecSpec moe_specs[2] = {
            { sgw,    sgs,    sgb,    shared_gate, (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 0, sg_bits },
            { suw,    sus,    sub,    shared_up,   (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 1, su_bits },
        };
        fast_batch_matvec(h_post, HIDDEN_DIM, moe_specs, 2);
    }

    // Softmax routing scores
    cpu_softmax(gate_scores, NUM_EXPERTS);

    // Top-K expert selection
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] routing: K=%d experts=[", K);
        for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.4f)%s", expert_indices[k], expert_weights[k], k<K-1?",":"");
        fprintf(stderr, "]\n");
    }

    // ---- Routed expert computation ----
    float *moe_out = calloc(HIDDEN_DIM, sizeof(float));

    if (packed_fd >= 0) {
        float *expert_out = malloc(HIDDEN_DIM * sizeof(float));

        size_t esz = active_expert_size();
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;

            if (g_metal && g_metal->buf_expert_data) {
                // GPU path: pread directly into Metal buffer, run gate+up+swiglu+down on GPU
                void *expert_buf_ptr = [g_metal->buf_expert_data contents];
                ssize_t nread = pread(packed_fd, expert_buf_ptr, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    continue;
                }

                gpu_expert_forward(g_metal, expert_buf_ptr, h_post, expert_out, 1 /*already in buffer*/);
            } else {
                // CPU fallback
                void *expert_data = malloc(esz);
                ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
                if (nread != (ssize_t)esz) {
                    fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                            layer_idx, eidx, nread, esz);
                    free(expert_data);
                    continue;
                }

                uint32_t *gw = (uint32_t *)expert_data;
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + GATE_S_OFF);
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + GATE_B_OFF);
                uint32_t *uw = (uint32_t *)((char *)expert_data + UP_W_OFF);
                uint16_t *us_p = (uint16_t *)((char *)expert_data + UP_S_OFF);
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + UP_B_OFF);
                uint32_t *dw = (uint32_t *)((char *)expert_data + DOWN_W_OFF);
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + DOWN_S_OFF);
                uint16_t *db_p = (uint16_t *)((char *)expert_data + DOWN_B_OFF);

                float *gate_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *up_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *act_out = malloc(MOE_INTERMEDIATE * sizeof(float));

                int ebits = EXPERT_BITS;
                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                                   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, ebits);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                                   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, ebits);
                cpu_swiglu(gate_proj_out, up_proj_out, act_out, MOE_INTERMEDIATE);
                cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out,
                                   HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, ebits);

                free(gate_proj_out);
                free(up_proj_out);
                free(act_out);
                free(expert_data);
            }

            // Accumulate weighted
            if (moe_dump) {
                fprintf(stderr, "[MOE-DUMP] expert[%d] out_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                        eidx, vec_rms(expert_out, HIDDEN_DIM),
                        expert_out[0], expert_out[1], expert_out[2], expert_out[3], expert_out[4]);
            }
            cpu_vec_madd(moe_out, expert_out, expert_weights[k], HIDDEN_DIM);
        }

        free(expert_out);
    }

    // ---- Shared expert SwiGLU (gate_proj + up_proj already computed above) ----
    float *shared_out = calloc(HIDDEN_DIM, sizeof(float));
    float *shared_act = calloc(SHARED_INTERMEDIATE, sizeof(float));
    cpu_swiglu(shared_gate, shared_up, shared_act, SHARED_INTERMEDIATE);

    if (moe_dump) {
        fprintf(stderr, "[MOE-DUMP] layer=%d h_post_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                layer_idx, vec_rms(h_post, HIDDEN_DIM), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);
        fprintf(stderr, "[MOE-DUMP] gate_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_gate, SHARED_INTERMEDIATE),
                shared_gate[0], shared_gate[1], shared_gate[2], shared_gate[3], shared_gate[4]);
        fprintf(stderr, "[MOE-DUMP] up_proj_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_up, SHARED_INTERMEDIATE),
                shared_up[0], shared_up[1], shared_up[2], shared_up[3], shared_up[4]);
        fprintf(stderr, "[MOE-DUMP] swiglu_rms=%.6f first5=[%.6f,%.6f,%.6f,%.6f,%.6f]\n",
                vec_rms(shared_act, SHARED_INTERMEDIATE),
                shared_act[0], shared_act[1], shared_act[2], shared_act[3], shared_act[4]);
    }

    // shared_expert down_proj (separate dispatch — different input than h_post)
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    uint32_t *sdw = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    uint16_t *sds = get_tensor_ptr(wf, name);
    snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    uint16_t *sdb = get_tensor_ptr(wf, name);
    if (sdw) {
        char sdn[256];
        snprintf(sdn, sizeof(sdn), "model.layers.%d.mlp.shared_expert.down_proj", layer_idx);
        fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out, HIDDEN_DIM,
                            SHARED_INTERMEDIATE, GROUP_SIZE, tensor_bits(wf, sdn));
    }

    // ---- Shared expert gate (sigmoid) -- already computed above ----
    float shared_weight = cpu_sigmoid(shared_gate_score);

    // Scale shared expert output
    for (int i = 0; i < HIDDEN_DIM; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Combine: hidden = h_mid + moe_out + shared_out ----
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    if (moe_debug) {
        fprintf(stderr, "[MOE-DBG] layer=%d h_mid_rms=%.4f moe_rms=%.4f shared_rms=%.4f shared_gate=%.4f hidden_rms=%.4f\n",
                layer_idx, vec_rms(h_mid, HIDDEN_DIM), vec_rms(moe_out, HIDDEN_DIM),
                vec_rms(shared_out, HIDDEN_DIM), shared_weight,
                vec_rms(hidden, HIDDEN_DIM));
    }

    free(h_post);
    free(h_mid);
    free(gate_scores);
    free(moe_out);
    free(shared_out);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
}

// ============================================================================
// Embedding lookup (4-bit quantized)
// ============================================================================

static void embed_lookup(WeightFile *wf, int token_id, float *out) {
    TensorInfo *w_info = get_tensor_info(wf, "model.embed_tokens.weight");
    TensorInfo *s_info = get_tensor_info(wf, "model.embed_tokens.scales");
    TensorInfo *b_info = get_tensor_info(wf, "model.embed_tokens.biases");

    if (!w_info) {
        fprintf(stderr, "ERROR: embedding weight not found\n");
        memset(out, 0, HIDDEN_DIM * sizeof(float));
        return;
    }

    // GGUF mode: Q4_K / Q6_K block rows
    if (w_info->ggml_type == 12 || w_info->ggml_type == 14) {
        const uint8_t *W = (const uint8_t *)((char *)wf->data + w_info->offset);
        const size_t row_bytes = (size_t)(HIDDEN_DIM / 256) *
            (w_info->ggml_type == 12 ? 144 : 210);
        gguf_dequant_row(W + (size_t)token_id * row_bytes, out, HIDDEN_DIM,
                         w_info->ggml_type);
        return;
    }

    // BF16 path: no scales/biases, weight is raw BF16 [vocab_size, hidden_dim]
    if (!s_info || !b_info) {
        const uint16_t *W_bf16 = (const uint16_t *)((char *)wf->data + w_info->offset);
        int in_dim = w_info->shape[1];  // 2048 for BF16
        const uint16_t *row = W_bf16 + (size_t)token_id * in_dim;
        for (int i = 0; i < HIDDEN_DIM; i++) {
            out[i] = bf16_to_f32(row[i]);
        }
        return;
    }

    // 4-bit/8-bit path (bits from shapes; group_size = 64 convention)
    int packed_cols = w_info->shape[1];
    int num_groups = s_info->shape[1];
    int bits = (packed_cols == num_groups * 16) ? 8 : 4;
    int vals_per_u32 = 32 / bits;
    uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
    uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
    uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);
    const uint32_t *w_row = W + (size_t)token_id * packed_cols;
    const uint16_t *s_row = S + (size_t)token_id * num_groups;
    const uint16_t *b_row = B + (size_t)token_id * num_groups;
    int group_size = HIDDEN_DIM / num_groups;
    int packed_per_group = group_size / vals_per_u32;
    uint32_t mask = (1u << bits) - 1;
    for (int g = 0; g < num_groups; g++) {
        float scale = bf16_to_f32(s_row[g]);
        float bias = bf16_to_f32(b_row[g]);
        for (int p = 0; p < packed_per_group; p++) {
            uint32_t packed = w_row[g * packed_per_group + p];
            int base = g * group_size + p * vals_per_u32;
            for (int n = 0; n < vals_per_u32; n++) {
                uint32_t val = (packed >> (n * bits)) & mask;
                out[base + n] = (float)val * scale + bias;
            }
        }
    }
}

// ============================================================================
// LM head (logits projection)
// ============================================================================

static void lm_head_forward(WeightFile *wf, const float *hidden, float *logits) {
    TensorInfo *w_info = get_tensor_info(wf, "lm_head.weight");
    TensorInfo *s_info = get_tensor_info(wf, "lm_head.scales");
    TensorInfo *b_info = get_tensor_info(wf, "lm_head.biases");

    if (!w_info) {
        fprintf(stderr, "ERROR: lm_head weight not found\n");
        return;
    }

    // GGUF mode: Q4_K / Q6_K block matvec (Phase C GPU kernel, CPU fallback)
    if (w_info->ggml_type == 12 || w_info->ggml_type == 14) {
        const uint8_t *W = (const uint8_t *)((char *)wf->data + w_info->offset);
        if (g_metal && g_metal->matvec_qk &&
            gpu_gguf_dequant_matvec(g_metal, W, hidden, logits,
                                    VOCAB_SIZE, HIDDEN_DIM, w_info->ggml_type)) {
            return;
        }
        gguf_cpu_matvec(W, hidden, logits, VOCAB_SIZE, HIDDEN_DIM, w_info->ggml_type);
        return;
    }

    // 4-bit/8-bit path (GPU-accelerated). Use actual tensor shape for in_dim.
    if (s_info && b_info) {
        uint32_t *W = (uint32_t *)((char *)wf->data + w_info->offset);
        uint16_t *S = (uint16_t *)((char *)wf->data + s_info->offset);
        uint16_t *B = (uint16_t *)((char *)wf->data + b_info->offset);
        int bits = (w_info->shape[1] == s_info->shape[1] * 16) ? 8 : 4;
        int in_dim = w_info->shape[1] * (32 / bits);
        int group_size = in_dim / s_info->shape[1];
        fast_dequant_matvec(W, S, B, hidden, logits, VOCAB_SIZE, in_dim, group_size, bits);
        return;
    }

    int in_dim = w_info->shape[1];  // 2048

    // GPU path: use gemv_bf16_x2 kernel (reads BF16 directly from Metal buffer, zero-copy)
    if (g_metal && g_metal->wf_buf && (g_metal->gemv_bf16_x2_pipe || g_metal->gemv_bf16_pipe)) {
        int use_x2 = (g_metal->gemv_bf16_x2_pipe != NULL);
        NSUInteger w_off = w_info->offset;
        memcpy([g_metal->buf_input contents], hidden, in_dim * sizeof(float));
        id<MTLCommandBuffer> cmdbuf = [g_metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:use_x2 ? g_metal->gemv_bf16_x2_pipe : g_metal->gemv_bf16_pipe];
        [enc setBuffer:g_metal->wf_buf offset:w_off atIndex:0];
        [enc setBuffer:g_metal->buf_input offset:0 atIndex:1];
        [enc setBuffer:g_metal->buf_output offset:0 atIndex:2];
        uint32_t od = VOCAB_SIZE, id = (uint32_t)in_dim;
        [enc setBytes:&od length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&id length:sizeof(uint32_t) atIndex:4];
        uint32_t num_tgs = use_x2 ? ((VOCAB_SIZE + 1) / 2) : VOCAB_SIZE;
        uint32_t tg_size = use_x2 ? 256 : 256;
        [enc dispatchThreadgroups:MTLSizeMake(num_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
        [enc endEncoding];
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
        memcpy(logits, [g_metal->buf_output contents], VOCAB_SIZE * sizeof(float));
        return;
    }

    // CPU fallback: chunked BF16→F32 conversion + BLAS (avoids 2 GB allocation)
    {
        const uint16_t *W_bf16 = (const uint16_t *)((char *)wf->data + w_info->offset);
        #define LM_HEAD_CHUNK 4096  // process 4096 rows at a time (34 MB chunks)
        float *chunk_f32 = malloc((size_t)LM_HEAD_CHUNK * in_dim * sizeof(float));
        if (!chunk_f32) {
            fprintf(stderr, "ERROR: cannot allocate lm_head chunk buffer\n");
            return;
        }
        for (int chunk = 0; chunk < VOCAB_SIZE; chunk += LM_HEAD_CHUNK) {
            int rows = (chunk + LM_HEAD_CHUNK <= VOCAB_SIZE) ? LM_HEAD_CHUNK : (VOCAB_SIZE - chunk);
            // Convert BF16→F32 for this chunk
            size_t base = (size_t)chunk * in_dim;
            for (int i = 0; i < rows * in_dim; i++) {
                chunk_f32[i] = bf16_to_f32(W_bf16[base + i]);
            }
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        rows, in_dim,
                        1.0f, chunk_f32, in_dim,
                        hidden, 1,
                        0.0f, logits + chunk, 1);
        }
        free(chunk_f32);
        #undef LM_HEAD_CHUNK
    }
}

// ============================================================================
// Parallel I/O infrastructure for expert pread (from proven main.m pattern)
// ============================================================================

#define NUM_IO_THREADS 8  // 8 threads for K=8 experts (one per expert)

// (InferPreadTask + io_pool_dispatch prototype moved above — see S7 note
// before the GGUF expert path.)

typedef struct {
    InferPreadTask *tasks;
    int num_tasks;
    int thread_id;
} InferPreadThreadArg;

static void *infer_pread_thread_fn(void *arg) {
    InferPreadThreadArg *ta = (InferPreadThreadArg *)arg;
    for (int i = ta->thread_id; i < ta->num_tasks; i += NUM_IO_THREADS) {
        InferPreadTask *t = &ta->tasks[i];
        t->result = pread(t->fd, t->dst, t->size, t->offset);
    }
    return NULL;
}

// ============================================================================
// Persistent I/O Thread Pool — eliminates pthread_create/join per layer
// ============================================================================

typedef struct {
    pthread_t threads[NUM_IO_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t work_done;
    InferPreadTask *tasks;
    int num_tasks;
    int tasks_completed;
    int generation;          // incremented each dispatch — workers wait for new gen
    volatile int shutdown;
} IOThreadPool;

static IOThreadPool g_io_pool;
static int g_io_pool_initialized = 0;

static void *io_pool_worker(void *arg) {
    int tid = (int)(intptr_t)arg;
    int my_gen = 0;
    pthread_mutex_lock(&g_io_pool.mutex);
    while (1) {
        while (g_io_pool.generation == my_gen && !g_io_pool.shutdown)
            pthread_cond_wait(&g_io_pool.work_ready, &g_io_pool.mutex);
        if (g_io_pool.shutdown) break;
        my_gen = g_io_pool.generation;

        // Snapshot work for this generation
        int num_tasks = g_io_pool.num_tasks;
        InferPreadTask *tasks = g_io_pool.tasks;
        pthread_mutex_unlock(&g_io_pool.mutex);

        // Process assigned tasks (stride by thread count)
        for (int i = tid; i < num_tasks; i += NUM_IO_THREADS) {
            InferPreadTask *t = &tasks[i];
            if (t->lz4_comp_buf && t->lz4_comp_size > 0) {
                // LZ4 path: read compressed from SSD, decompress into dst
                ssize_t nr = pread(t->fd, t->lz4_comp_buf, t->lz4_comp_size, t->offset);
                if (nr == (ssize_t)t->lz4_comp_size) {
                    size_t dec = compression_decode_buffer(
                        t->dst, t->size, t->lz4_comp_buf, t->lz4_comp_size,
                        NULL, COMPRESSION_LZ4);
                    t->result = (ssize_t)dec;
                } else {
                    t->result = -1;
                }
            } else {
                t->result = pread(t->fd, t->dst, t->size, t->offset);
            }
        }

        pthread_mutex_lock(&g_io_pool.mutex);
        g_io_pool.tasks_completed++;
        if (g_io_pool.tasks_completed == NUM_IO_THREADS)
            pthread_cond_signal(&g_io_pool.work_done);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
    return NULL;
}

static void io_pool_init(void) {
    if (g_io_pool_initialized) return;
    pthread_mutex_init(&g_io_pool.mutex, NULL);
    pthread_cond_init(&g_io_pool.work_ready, NULL);
    pthread_cond_init(&g_io_pool.work_done, NULL);
    g_io_pool.shutdown = 0;
    g_io_pool.generation = 0;
    g_io_pool.tasks = NULL;
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_create(&g_io_pool.threads[i], NULL, io_pool_worker, (void*)(intptr_t)i);
    g_io_pool_initialized = 1;
}

static dispatch_queue_t g_io_gcd_queue = NULL;

static void io_pool_dispatch(InferPreadTask *tasks, int num_tasks) {
    if (num_tasks == 0) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.tasks = tasks;
    g_io_pool.num_tasks = num_tasks;
    g_io_pool.tasks_completed = 0;
    g_io_pool.generation++;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    while (g_io_pool.tasks_completed < NUM_IO_THREADS) {
        pthread_cond_wait(&g_io_pool.work_done, &g_io_pool.mutex);
    }
    pthread_mutex_unlock(&g_io_pool.mutex);
}

// ---- Async expert pread pipeline ----
// Starts pread on background GCD threads immediately after routing.
// The pread overlaps with shared expert prep + next layer's CMD1+attn+CMD2.
// Wait for completion right before CMD3 needs the expert data.
// (AsyncPreadState + g_async_pread + async_pread_wait are declared above —
// the GGUF prediction path uses them before their natural definitions.)
static AsyncPreadState g_async_pread = {0};

static void async_pread_start(int packed_fd, int *expert_indices, int K,
                               id<MTLBuffer> __strong *dst_bufs, const void *mmap_base) {
    size_t esz = active_expert_size();
    g_async_pread.num_tasks = K;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();

    for (int k = 0; k < K; k++) {
        g_async_pread.tasks[k].fd = packed_fd;
        g_async_pread.tasks[k].dst = [dst_bufs[k] contents];
        g_async_pread.tasks[k].offset = (off_t)expert_indices[k] * esz;
        g_async_pread.tasks[k].size = esz;
        g_async_pread.tasks[k].result = 0;
    }

    // Fire off parallel preads on GCD — returns immediately
    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int k = 0; k < K; k++) {
        InferPreadTask *t = &g_async_pread.tasks[k];
        dispatch_group_async(g_async_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

// Batched multi-position pread: n independent {fd, offset, dst, size} specs
// fired in ONE GCD group (pool mode reads all M positions' experts at once).
static void async_pread_multi_start(const int *fds, const off_t *offsets,
                                    void *const *dsts, const size_t *sizes, int n) {
    g_async_pread.num_tasks = n;
    g_async_pread.active = 1;
    if (!g_async_pread.group) g_async_pread.group = dispatch_group_create();

    for (int i = 0; i < n; i++) {
        g_async_pread.tasks[i].fd = fds[i];
        g_async_pread.tasks[i].dst = dsts[i];
        g_async_pread.tasks[i].offset = offsets[i];
        g_async_pread.tasks[i].size = sizes[i];
        g_async_pread.tasks[i].result = 0;
    }

    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int i = 0; i < n; i++) {
        InferPreadTask *t = &g_async_pread.tasks[i];
        dispatch_group_async(g_async_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

static void async_pread_wait(void) {
    if (!g_async_pread.active) return;
    dispatch_group_wait(g_async_pread.group, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < g_async_pread.num_tasks; k++) {
        // Compare against the task's OWN size — Phase C S4 reads GGUF
        // slabs (gate/up/down chunks), which differ from active_expert_size.
        // Packed callers always pass size == active_expert_size(), so their
        // behavior is unchanged.
        g_async_pread.valid[k] = (g_async_pread.tasks[k].result == (ssize_t)g_async_pread.tasks[k].size);
    }
    g_async_pread.active = 0;
}

// Second async-pread state for the prefill hot-set prefetcher (runs
// concurrently with the main expert preads, targeting the prefetch pools).
static AsyncPreadState g_prefetch_pread = {0};

static void async_pread_prefetch_start(const int *fds, const off_t *offsets,
                                       void *const *dsts, const size_t *sizes, int n) {
    g_prefetch_pread.num_tasks = n;
    g_prefetch_pread.active = 1;
    if (!g_prefetch_pread.group) g_prefetch_pread.group = dispatch_group_create();

    for (int i = 0; i < n; i++) {
        g_prefetch_pread.tasks[i].fd = fds[i];
        g_prefetch_pread.tasks[i].dst = dsts[i];
        g_prefetch_pread.tasks[i].offset = offsets[i];
        g_prefetch_pread.tasks[i].size = sizes[i];
        g_prefetch_pread.tasks[i].result = 0;
    }

    static dispatch_queue_t io_q = NULL;
    if (!io_q) io_q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    for (int i = 0; i < n; i++) {
        InferPreadTask *t = &g_prefetch_pread.tasks[i];
        dispatch_group_async(g_prefetch_pread.group, io_q, ^{
            t->result = pread(t->fd, t->dst, t->size, t->offset);
        });
    }
}

static void async_pread_prefetch_wait(void) {
    if (!g_prefetch_pread.active) return;
    dispatch_group_wait(g_prefetch_pread.group, DISPATCH_TIME_FOREVER);
    for (int k = 0; k < g_prefetch_pread.num_tasks; k++) {
        g_prefetch_pread.valid[k] = (g_prefetch_pread.tasks[k].result == (ssize_t)g_prefetch_pread.tasks[k].size);
    }
    g_prefetch_pread.active = 0;
}

static void io_pool_shutdown(void) {
    if (!g_io_pool_initialized) return;
    pthread_mutex_lock(&g_io_pool.mutex);
    g_io_pool.shutdown = 1;
    pthread_cond_broadcast(&g_io_pool.work_ready);
    pthread_mutex_unlock(&g_io_pool.mutex);
    for (int i = 0; i < NUM_IO_THREADS; i++)
        pthread_join(g_io_pool.threads[i], NULL);
    pthread_mutex_destroy(&g_io_pool.mutex);
    pthread_cond_destroy(&g_io_pool.work_ready);
    pthread_cond_destroy(&g_io_pool.work_done);
    g_io_pool_initialized = 0;
}

// Parallel pread of K experts into Metal buffers using pthreads.
// Returns number of successfully loaded experts, sets valid[] flags.
static int parallel_pread_experts(
    int packed_fd,
    int *expert_indices,
    int K,
    int *valid,  // [MAX_K] output: 1 if expert loaded successfully
    const void *mmap_base  // mmap'd layer file (NULL to use pread)
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
        tasks[k].mmap_base = mmap_base;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// ============================================================================
// Parallel pread into explicit buffer set (for double buffering).
// Same as parallel_pread_experts but reads into caller-specified MTLBuffers.
// ============================================================================
static int parallel_pread_experts_into(
    int packed_fd,
    int *expert_indices,
    int K,
    id<MTLBuffer> __strong *dst_bufs,  // target Metal buffers (set A or B)
    int *valid  // [MAX_K] output: 1 if expert loaded successfully
) {
    size_t esz = active_expert_size();
    InferPreadTask tasks[MAX_K];
    for (int k = 0; k < K; k++) {
        tasks[k].fd = packed_fd;
        tasks[k].dst = [dst_bufs[k] contents];
        tasks[k].offset = (off_t)expert_indices[k] * esz;
        tasks[k].size = esz;
        tasks[k].result = 0;
    }

    io_pool_dispatch(tasks, K);

    int loaded = 0;
    for (int k = 0; k < K; k++) {
        valid[k] = (tasks[k].result == (ssize_t)esz);
        if (valid[k]) loaded++;
        else {
            fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                    expert_indices[k], tasks[k].result, esz);
        }
    }
    return loaded;
}

// ============================================================================
// Expert LRU Cache: keeps recently-used expert Metal buffers in GPU memory.
//
// Key: (layer_idx, expert_idx) -> Metal buffer containing 7.08MB expert data.
// On cache HIT:  skip pread entirely, use the cached Metal buffer for GPU dispatch.
// On cache MISS: pread into a new/evicted Metal buffer, insert into cache.
// LRU eviction:  when cache is full, evict the least recently used entry.
//
// Memory budget: 2000 entries * 7.08MB = 14.2GB. With 5.5GB non-expert weights
// + 14.2GB cache = 19.7GB total. Fits in 48GB with room for OS.
//
// Unlike Python/MLX where LRU caching caused Metal heap pressure and slower
// mx.eval(), here Metal buffers ARE the cache -- no conversion overhead.
// ============================================================================

typedef struct {
    int layer_idx;
    int expert_idx;
    id<MTLBuffer> buffer;    // Metal buffer holding EXPERT_SIZE_MAX bytes
    uint64_t last_used;      // monotonic counter for LRU ordering
} ExpertCacheEntry;

typedef struct {
    ExpertCacheEntry *entries;
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[NUM_LAYERS][NUM_EXPERTS];
    uint64_t access_counter; // monotonic, incremented on every access
    id<MTLDevice> device;    // for allocating new Metal buffers
    // Stats
    uint64_t hits;
    uint64_t misses;
} ExpertLRUCache;

static ExpertLRUCache *g_expert_cache = NULL;

// Speculative early routing stats
static uint64_t g_spec_route_attempts = 0;   // total speculative routing attempts
static uint64_t g_spec_route_hits = 0;        // correctly predicted experts (found in cache at real routing time)
static uint64_t g_spec_route_preloads = 0;    // async preloads initiated (cache misses at speculation time)
static uint64_t g_spec_temporal_hits = 0;     // S8 probe: prev-position same-layer expert overlap

// ---- Temporal prediction pipeline ----
// Stores previous token's expert routing per layer. On the next token,
// predicted experts are preloaded into buf_multi_expert_data_B during CMD1_wait
// idle time. After routing, hits use buf_B, misses sync-pread into buf_A.
// Different from previous failed speculative attempts:
//   - Loads into scratch buffers (no cache pollution)
//   - Uses CMD1_wait idle time (no additional CPU cost)
//   - Only sync-preads misses (not all K experts)
static int g_pred_experts[60][MAX_K];              // previous token's expert indices per layer
static int g_pred_count[60];                       // how many experts stored per layer
static int g_pred_valid = 0;                       // 1 after first token completes (predictions available)
// g_pred_enabled, g_pred_hits, g_pred_misses, g_pred_layers declared near timing (line ~163)

static ExpertLRUCache *expert_cache_new(id<MTLDevice> device, int max_entries) {
    ExpertLRUCache *cache = calloc(1, sizeof(ExpertLRUCache));
    cache->entries = calloc(max_entries, sizeof(ExpertCacheEntry));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->device = device;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }
    // Pre-allocate ALL Metal buffers at startup (avoids allocation overhead at runtime)
    size_t esz = active_expert_size();
    double t_prealloc = now_ms();
    for (int i = 0; i < max_entries; i++) {
        cache->entries[i].buffer = [device newBufferWithLength:esz
                                                      options:MTLResourceStorageModeShared];
        cache->entries[i].layer_idx = -1;
        cache->entries[i].expert_idx = -1;
        cache->entries[i].last_used = 0;
        if (!cache->entries[i].buffer) {
            fprintf(stderr, "WARNING: expert_cache: pre-alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
    }
    cache->num_entries = max_entries; // All slots pre-allocated (but empty keys)
    printf("[expert_cache] Initialized: max_entries=%d (%.1f GB budget), pre-alloc %.0f ms\n",
           max_entries, (double)max_entries * esz / 1e9, now_ms() - t_prealloc);
    return cache;
}

static void expert_cache_free(ExpertLRUCache *cache) {
    if (!cache) return;
    printf("[expert_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    // Metal buffers released by ARC when entries are freed
    free(cache->entries);
    free(cache);
}

// Lookup: returns the cached Metal buffer if found, otherwise NULL.
// On hit, updates the LRU timestamp.
static id<MTLBuffer> expert_cache_lookup(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    int idx = cache->entry_idx[layer_idx][expert_idx];
    if (idx >= 0) {
        cache->entries[idx].last_used = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer_idx, expert_idx);
        return cache->entries[idx].buffer;
    }
    cache->misses++;
    cache_telemetry_miss(layer_idx, expert_idx);
    return nil;
}

// Insert: adds a new entry. If the cache is full, evicts the LRU entry.
// Returns the Metal buffer to pread into (either newly allocated or evicted+reused).
static id<MTLBuffer> expert_cache_insert(ExpertLRUCache *cache, int layer_idx, int expert_idx) {
    id<MTLBuffer> buf = nil;

    int existing = cache->entry_idx[layer_idx][expert_idx];
    if (existing >= 0) {
        cache->entries[existing].last_used = ++cache->access_counter;
        return cache->entries[existing].buffer;
    }

    // Find a slot: first try an unused slot (layer_idx == -1), then LRU evict
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }
    if (target >= 0) {
        // Unused pre-allocated slot
        buf = cache->entries[target].buffer;
        cache->entries[target].layer_idx = layer_idx;
        cache->entries[target].expert_idx = expert_idx;
        cache->entries[target].last_used = ++cache->access_counter;
        cache->entry_idx[layer_idx][expert_idx] = target;
        return buf;
    }

    // Cache full: find LRU entry (smallest last_used)
    int lru_idx = 0;
    uint64_t min_used = cache->entries[0].last_used;
    for (int i = 1; i < cache->num_entries; i++) {
        if (cache->entries[i].last_used < min_used) {
            min_used = cache->entries[i].last_used;
            lru_idx = i;
        }
    }

    // Reuse the evicted entry's Metal buffer (same size, no realloc needed)
    int old_layer = cache->entries[lru_idx].layer_idx;
    int old_expert = cache->entries[lru_idx].expert_idx;
    cache_telemetry_evict(old_layer, old_expert);
    if (old_layer >= 0 && old_expert >= 0) {
        cache->entry_idx[old_layer][old_expert] = -1;
    }
    buf = cache->entries[lru_idx].buffer;
    cache->entries[lru_idx].layer_idx = layer_idx;
    cache->entries[lru_idx].expert_idx = expert_idx;
    cache->entries[lru_idx].last_used = ++cache->access_counter;
    cache->entry_idx[layer_idx][expert_idx] = lru_idx;
    return buf;
}

// ============================================================================
// Malloc-based expert frequency cache.
// Stores expert data in regular malloc'd memory (not Metal buffers) to avoid
// GPU memory pressure. On hit, memcpy to Metal scratch buffer. Much larger
// capacity than Metal buffer LRU cache at the cost of one memcpy per hit.
// ============================================================================

typedef struct {
    void **data;           // [max_entries] page-aligned malloc'd EXPERT_SIZE_MAX buffers
    id<MTLBuffer> __strong *metal_bufs;  // [max_entries] zero-copy Metal buffer wrappers
    int *layer_idx;        // [max_entries] layer index for each entry
    int *expert_idx;       // [max_entries] expert index for each entry
    uint64_t *last_used;   // [max_entries] monotonic counter for LRU
    int max_entries;
    int num_entries;
    int used_entries;
    int entry_idx[NUM_LAYERS][NUM_EXPERTS];
    uint64_t access_counter;
    uint64_t hits;
    uint64_t misses;
} MallocExpertCache;

static MallocExpertCache *g_malloc_cache = NULL;

static MallocExpertCache *malloc_cache_init(int max_entries, id<MTLDevice> device) {
    MallocExpertCache *cache = calloc(1, sizeof(MallocExpertCache));
    cache->data = calloc(max_entries, sizeof(void *));
    cache->metal_bufs = (__strong id<MTLBuffer> *)calloc(max_entries, sizeof(id<MTLBuffer>));
    cache->layer_idx = calloc(max_entries, sizeof(int));
    cache->expert_idx = calloc(max_entries, sizeof(int));
    cache->last_used = calloc(max_entries, sizeof(uint64_t));
    cache->max_entries = max_entries;
    cache->num_entries = 0;
    cache->used_entries = 0;
    cache->access_counter = 0;
    cache->hits = 0;
    cache->misses = 0;
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cache->entry_idx[l][e] = -1;
        }
    }

    size_t esz = active_expert_size();
    printf("[malloc_cache] Initializing: %d entries (%.1f GB) with zero-copy Metal wrappers\n",
           max_entries, (double)max_entries * esz / 1e9);
    double t_start = now_ms();

    size_t page_size = (size_t)getpagesize();
    // Round expert size up to page boundary for newBufferWithBytesNoCopy
    size_t aligned_size = (esz + page_size - 1) & ~(page_size - 1);

    for (int i = 0; i < max_entries; i++) {
        // Page-aligned allocation for zero-copy Metal buffer
        void *buf = NULL;
        if (posix_memalign(&buf, page_size, aligned_size) != 0 || !buf) {
            fprintf(stderr, "WARNING: malloc_cache: alloc failed at entry %d\n", i);
            max_entries = i;
            cache->max_entries = i;
            break;
        }
        memset(buf, 0, aligned_size);
        cache->data[i] = buf;

        // Create zero-copy Metal buffer wrapping the malloc'd memory
        // nil deallocator = Metal doesn't free the memory
        cache->metal_bufs[i] = [device newBufferWithBytesNoCopy:buf
                                                         length:aligned_size
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        cache->layer_idx[i] = -1;
        cache->expert_idx[i] = -1;
        cache->last_used[i] = 0;
    }
    cache->num_entries = max_entries;

    printf("[malloc_cache] Pre-allocated %d entries in %.0f ms\n",
           max_entries, now_ms() - t_start);
    return cache;
}

// Lookup: returns Metal buffer wrapping cached data, or nil. Zero-copy dispatch.
static id<MTLBuffer> malloc_cache_lookup(MallocExpertCache *cache, int layer, int expert) {
    int idx = cache->entry_idx[layer][expert];
    if (idx >= 0) {
        cache->last_used[idx] = ++cache->access_counter;
        cache->hits++;
        cache_telemetry_touch(layer, expert);
        return cache->metal_bufs[idx];
    }
    cache->misses++;
    cache_telemetry_miss(layer, expert);
    return nil;
}

// Insert: evict LRU if needed, return entry index for pread target.
// Returns the Metal buffer for this entry (caller should pread into cache->data[idx]).
static id<MTLBuffer> malloc_cache_insert(MallocExpertCache *cache, int layer, int expert, int *out_idx) {
    int existing = cache->entry_idx[layer][expert];
    if (existing >= 0) {
        cache->last_used[existing] = ++cache->access_counter;
        if (out_idx) *out_idx = existing;
        return cache->metal_bufs[existing];
    }

    // Find a free slot (layer_idx == -1) or evict LRU
    int target = -1;
    if (cache->used_entries < cache->num_entries) {
        target = cache->used_entries++;
    }

    if (target < 0) {
        // Cache full: evict entry with smallest last_used
        target = 0;
        uint64_t min_used = cache->last_used[0];
        for (int i = 1; i < cache->num_entries; i++) {
            if (cache->last_used[i] < min_used) {
                min_used = cache->last_used[i];
                target = i;
            }
        }
        cache_telemetry_evict(cache->layer_idx[target], cache->expert_idx[target]);
        if (cache->layer_idx[target] >= 0 && cache->expert_idx[target] >= 0) {
            cache->entry_idx[cache->layer_idx[target]][cache->expert_idx[target]] = -1;
        }
    }

    cache->layer_idx[target] = layer;
    cache->expert_idx[target] = expert;
    cache->last_used[target] = ++cache->access_counter;
    cache->entry_idx[layer][expert] = target;
    if (out_idx) *out_idx = target;
    return cache->metal_bufs[target];
}

static void malloc_cache_free(MallocExpertCache *cache) {
    if (!cache) return;
    printf("[malloc_cache] Final stats: %llu hits, %llu misses (%.1f%% hit rate)\n",
           cache->hits, cache->misses,
           (cache->hits + cache->misses) > 0
               ? 100.0 * cache->hits / (cache->hits + cache->misses) : 0.0);
    for (int i = 0; i < cache->num_entries; i++) {
        cache->metal_bufs[i] = nil;  // release Metal buffer wrapper
        free(cache->data[i]);
    }
    free(cache->data);
    free(cache->metal_bufs);
    free(cache->layer_idx);
    free(cache->expert_idx);
    free(cache->last_used);
    free(cache);
}

// ============================================================================
// Background prefetch thread for double-buffered expert I/O (from main.m).
// Runs pread on a background thread while main thread does GPU compute.
// Uses pure C I/O plan to avoid ARC issues across threads.
// ============================================================================

typedef struct {
    void *dst[MAX_K];       // raw pointers from [buf contents] (no ARC)
    off_t offset[MAX_K];    // file offsets per expert
    int K;                  // number of experts
    int fd;                 // file descriptor for this layer
    int valid[MAX_K];       // output: 1 if pread succeeded
    int loaded;             // output: count of successfully loaded experts
} InferIOPlan;

typedef struct {
    InferIOPlan plan;       // pre-built I/O plan (pure C, no ARC)
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int start;              // signal: set to 1 to start prefetch
    int done;               // signal: set to 1 when prefetch complete
    int shutdown;           // signal: set to 1 to exit thread
} InferPrefetchCtx;

static void *infer_prefetch_thread_fn(void *arg) {
    InferPrefetchCtx *pf = (InferPrefetchCtx *)arg;

    while (1) {
        pthread_mutex_lock(&pf->mutex);
        while (!pf->start && !pf->shutdown) {
            pthread_cond_wait(&pf->cond, &pf->mutex);
        }
        if (pf->shutdown) {
            pthread_mutex_unlock(&pf->mutex);
            break;
        }
        pf->start = 0;
        pthread_mutex_unlock(&pf->mutex);

        // Execute parallel pread (pure C, no ARC objects)
        size_t esz = active_expert_size();
        InferIOPlan *plan = &pf->plan;
        InferPreadTask tasks[MAX_K];
        for (int k = 0; k < plan->K; k++) {
            tasks[k].fd = plan->fd;
            tasks[k].dst = plan->dst[k];
            tasks[k].offset = plan->offset[k];
            tasks[k].size = esz;
            tasks[k].result = 0;
        }

        io_pool_dispatch(tasks, plan->K);

        plan->loaded = 0;
        for (int k = 0; k < plan->K; k++) {
            plan->valid[k] = (tasks[k].result == (ssize_t)esz);
            if (plan->valid[k]) plan->loaded++;
        }

        // Signal completion
        pthread_mutex_lock(&pf->mutex);
        pf->done = 1;
        pthread_cond_signal(&pf->cond);
        pthread_mutex_unlock(&pf->mutex);
    }

    return NULL;
}

// Build I/O plan on main thread (ARC-safe: extracts void* from id<MTLBuffer>),
// then signal background prefetch thread.
static void infer_prefetch_start(InferPrefetchCtx *pf, int packed_fd,
                                  int *expert_indices, int K,
                                  id<MTLBuffer> __strong *dst_bufs) {
    pthread_mutex_lock(&pf->mutex);
    size_t esz = active_expert_size();
    InferIOPlan *plan = &pf->plan;
    plan->fd = packed_fd;
    plan->K = K;
    for (int k = 0; k < K; k++) {
        plan->dst[k] = [dst_bufs[k] contents];
        plan->offset[k] = (off_t)expert_indices[k] * esz;
        plan->valid[k] = 0;
    }
    plan->loaded = 0;
    pf->done = 0;
    pf->start = 1;
    pthread_cond_signal(&pf->cond);
    pthread_mutex_unlock(&pf->mutex);
}

// Wait for background prefetch to complete. Returns number of loaded experts.
// Copies valid[] flags into caller's array.
static int infer_prefetch_wait(InferPrefetchCtx *pf, int *valid_out, int K) {
    pthread_mutex_lock(&pf->mutex);
    while (!pf->done) {
        pthread_cond_wait(&pf->cond, &pf->mutex);
    }
    int loaded = pf->plan.loaded;
    for (int k = 0; k < K; k++) {
        valid_out[k] = pf->plan.valid[k];
    }
    pthread_mutex_unlock(&pf->mutex);
    return loaded;
}

static InferPrefetchCtx *g_prefetch = NULL;
static pthread_t g_prefetch_tid;

static void infer_prefetch_init(void) {
    if (g_prefetch) return;
    g_prefetch = calloc(1, sizeof(InferPrefetchCtx));
    pthread_mutex_init(&g_prefetch->mutex, NULL);
    pthread_cond_init(&g_prefetch->cond, NULL);
    g_prefetch->shutdown = 0;
    pthread_create(&g_prefetch_tid, NULL, infer_prefetch_thread_fn, g_prefetch);
}

static void infer_prefetch_shutdown(void) {
    if (!g_prefetch) return;
    pthread_mutex_lock(&g_prefetch->mutex);
    g_prefetch->shutdown = 1;
    pthread_cond_signal(&g_prefetch->cond);
    pthread_mutex_unlock(&g_prefetch->mutex);
    pthread_join(g_prefetch_tid, NULL);
    pthread_mutex_destroy(&g_prefetch->mutex);
    pthread_cond_destroy(&g_prefetch->cond);
    free(g_prefetch);
    g_prefetch = NULL;
}

// ============================================================================
// Per-layer weight pointer cache — built once, eliminates 40+ snprintf+lookup
// per layer per token. With 60 layers and 15 tokens = 36,000 lookups saved.
// ============================================================================

typedef struct {
    // Input/post-attention layer norms
    uint16_t *input_norm_w;
    uint16_t *post_attn_norm_w;

    // Full attention weights (non-NULL only for full attention layers)
    uint32_t *q_w; uint16_t *q_s, *q_b;
    uint32_t *k_w; uint16_t *k_s, *k_b;
    uint32_t *v_w; uint16_t *v_s, *v_b;
    uint32_t *o_w; uint16_t *o_s, *o_b;
    uint16_t *q_norm_w, *k_norm_w;

    // Linear attention weights (non-NULL only for linear attention layers)
    uint32_t *qkv_w; uint16_t *qkv_s, *qkv_b;
    uint32_t *z_w;   uint16_t *z_s, *z_b;
    uint32_t *b_w;   uint16_t *b_s, *b_b;
    uint32_t *a_w;   uint16_t *a_s, *a_b;
    uint16_t *conv1d_w;
    float *A_log;
    uint16_t *dt_bias;
    uint16_t *gated_norm_w;
    uint32_t *out_proj_w; uint16_t *out_proj_s, *out_proj_b;

    // MoE routing + shared expert weights
    uint32_t *gate_w; uint16_t *gate_s, *gate_b;
    uint32_t *sg_w;   uint16_t *sg_s, *sg_b;   // shared gate_proj
    uint32_t *su_w;   uint16_t *su_s, *su_b;   // shared up_proj
    uint32_t *sd_w;   uint16_t *sd_s, *sd_b;   // shared down_proj
    uint32_t *seg_w;  uint16_t *seg_s, *seg_b; // shared_expert_gate

    // Packing width per quantized tensor (4 or 8; 4 when BF16/unquantized)
    int q_bits, k_bits, v_bits, o_bits;
    int qkv_bits, z_bits, a_bits, b_bits, out_proj_bits;
    int gate_bits, sg_bits, su_bits, sd_bits, seg_bits;
} LayerWeightCache;

static LayerWeightCache layer_cache[NUM_LAYERS];
static int layer_cache_built = 0;

static void build_layer_cache(WeightFile *wf) {
    if (layer_cache_built) return;
    char name[256];

    for (int i = 0; i < NUM_LAYERS; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);

        // Norms
        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", i);
        lc->input_norm_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", i);
        lc->post_attn_norm_w = get_tensor_ptr(wf, name);

        if (is_full) {
            // Full attention
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", i);
            lc->q_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.scales", i);
            lc->q_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.biases", i);
            lc->q_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", i);
            lc->k_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.scales", i);
            lc->k_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.biases", i);
            lc->k_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", i);
            lc->v_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.scales", i);
            lc->v_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.biases", i);
            lc->v_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", i);
            lc->o_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.scales", i);
            lc->o_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.biases", i);
            lc->o_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", i);
            lc->q_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", i);
            lc->k_norm_w = get_tensor_ptr(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj", i);
            lc->q_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj", i);
            lc->k_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj", i);
            lc->v_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj", i);
            lc->o_bits = tensor_bits(wf, name);
        } else {
            // Linear attention
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            lc->qkv_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.scales", i);
            lc->qkv_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv.biases", i);
            lc->qkv_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.weight", i);
            lc->z_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.scales", i);
            lc->z_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z.biases", i);
            lc->z_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.weight", i);
            lc->b_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.scales", i);
            lc->b_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b.biases", i);
            lc->b_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.weight", i);
            lc->a_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.scales", i);
            lc->a_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a.biases", i);
            lc->a_b = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.conv1d.weight", i);
            lc->conv1d_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.A_log", i);
            lc->A_log = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.dt_bias", i);
            lc->dt_bias = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.norm.weight", i);
            lc->gated_norm_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.weight", i);
            lc->out_proj_w = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.scales", i);
            lc->out_proj_s = get_tensor_ptr(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj.biases", i);
            lc->out_proj_b = get_tensor_ptr(wf, name);

            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_qkv", i);
            lc->qkv_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_z", i);
            lc->z_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_b", i);
            lc->b_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.in_proj_a", i);
            lc->a_bits = tensor_bits(wf, name);
            snprintf(name, sizeof(name), "model.layers.%d.linear_attn.out_proj", i);
            lc->out_proj_bits = tensor_bits(wf, name);
        }

        // MoE weights (same for all layers)
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.weight", i);
        lc->gate_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.scales", i);
        lc->gate_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate.biases", i);
        lc->gate_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        lc->sg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", i);
        lc->sg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", i);
        lc->sg_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        lc->su_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.scales", i);
        lc->su_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj.biases", i);
        lc->su_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        lc->sd_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.scales", i);
        lc->sd_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj.biases", i);
        lc->sd_b = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.weight", i);
        lc->seg_w = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.scales", i);
        lc->seg_s = get_tensor_ptr(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate.biases", i);
        lc->seg_b = get_tensor_ptr(wf, name);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate", i);
        lc->gate_bits = tensor_bits(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.gate_proj", i);
        lc->sg_bits = tensor_bits(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.up_proj", i);
        lc->su_bits = tensor_bits(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert.down_proj", i);
        lc->sd_bits = tensor_bits(wf, name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.shared_expert_gate", i);
        lc->seg_bits = tensor_bits(wf, name);
    }

    layer_cache_built = 1;
    printf("[cache] Pre-computed weight pointers for %d layers\n", NUM_LAYERS);
}

// Phase C S6 probe: build the stage2 buffer (see g_gguf_stage2 above).
// Copies the GPU-read QK tensors at 2MB-aligned offsets and repoints the
// layer-cache pointers; the matvec encoder detects stage2 pointers and
// binds the single stage2 wrap at the tensor's offset.
//   scope 1 = cmdA's qkv/z (linear layers)
//   scope 2 = + cmdB tensors (out_proj/gate/seg/sg/su/sd, b/a) + full
//             layers' q/k/v/o
static void gguf_stage2_build(MetalCtx *ctx) {
    const char *se = getenv("FINCHMOE_GGUF_STAGE2");
    if (!se || !g_gguf_stage || g_gguf_stage2 || !ctx) return;
    int scope = atoi(se);
    if (scope < 1) scope = 1;
    // staging plan: {ptr-field offset, rows, in_dim, bits-field offset}
    typedef struct { size_t p_off, b_off; uint32_t rows, in_dim; } StPlan;
    StPlan plans[NUM_LAYERS][16];
    int npl[NUM_LAYERS] = {0};
    static size_t t_off[NUM_LAYERS][16];
    size_t total = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
        int np = 0;
#define ST2_PLAN(_w, _b, _rows, _in) \
        if ((lc->_w) && (lc->_b) >= 10) { \
            plans[i][np] = (StPlan){ offsetof(LayerWeightCache, _w), offsetof(LayerWeightCache, _b), \
                                     (uint32_t)(_rows), (uint32_t)(_in) }; np++; }
        if (!is_full) {
            ST2_PLAN(qkv_w, qkv_bits, LINEAR_CONV_DIM, HIDDEN_DIM);
            ST2_PLAN(z_w, z_bits, LINEAR_TOTAL_VALUE, HIDDEN_DIM);
            if (scope >= 2) {
                ST2_PLAN(out_proj_w, out_proj_bits, HIDDEN_DIM, LINEAR_TOTAL_VALUE);
                ST2_PLAN(gate_w, gate_bits, NUM_EXPERTS, HIDDEN_DIM);
                ST2_PLAN(seg_w, seg_bits, 1, HIDDEN_DIM);
                ST2_PLAN(sg_w, sg_bits, SHARED_INTERMEDIATE, HIDDEN_DIM);
                ST2_PLAN(su_w, su_bits, SHARED_INTERMEDIATE, HIDDEN_DIM);
                ST2_PLAN(sd_w, sd_bits, HIDDEN_DIM, SHARED_INTERMEDIATE);
                ST2_PLAN(b_w, b_bits, LINEAR_NUM_V_HEADS, HIDDEN_DIM);
                ST2_PLAN(a_w, a_bits, LINEAR_NUM_V_HEADS, HIDDEN_DIM);
            }
        } else if (scope >= 2) {
            ST2_PLAN(q_w, q_bits, NUM_ATTN_HEADS * HEAD_DIM * 2, HIDDEN_DIM);
            ST2_PLAN(k_w, k_bits, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            ST2_PLAN(v_w, v_bits, NUM_KV_HEADS * HEAD_DIM, HIDDEN_DIM);
            ST2_PLAN(o_w, o_bits, HIDDEN_DIM, NUM_ATTN_HEADS * HEAD_DIM);
            ST2_PLAN(gate_w, gate_bits, NUM_EXPERTS, HIDDEN_DIM);
            ST2_PLAN(seg_w, seg_bits, 1, HIDDEN_DIM);
            ST2_PLAN(sg_w, sg_bits, SHARED_INTERMEDIATE, HIDDEN_DIM);
            ST2_PLAN(su_w, su_bits, SHARED_INTERMEDIATE, HIDDEN_DIM);
            ST2_PLAN(sd_w, sd_bits, HIDDEN_DIM, SHARED_INTERMEDIATE);
        }
#undef ST2_PLAN
        npl[i] = np;
        for (int p = 0; p < np; p++) {
            StPlan *sp = &plans[i][p];
            int bits = *(int *)((char *)lc + sp->b_off);
            size_t rb = (size_t)(sp->in_dim / 256) * (bits == 10 ? 144 : 210);
            t_off[i][p] = (total + 2*1024*1024 - 1) & ~(size_t)(2*1024*1024 - 1);
            total = t_off[i][p] + (size_t)sp->rows * rb;
        }
    }
    if (!total) return;
    void *st2 = NULL;
    if (posix_memalign(&st2, 2*1024*1024, total) != 0) return;
    g_gguf_stage2 = (char *)st2;
    g_gguf_stage2_len = total;
    double t0 = now_ms();
    for (int i = 0; i < NUM_LAYERS; i++) {
        LayerWeightCache *lc = &layer_cache[i];
        for (int p = 0; p < npl[i]; p++) {
            StPlan *sp = &plans[i][p];
            int bits = *(int *)((char *)lc + sp->b_off);
            size_t rb = (size_t)(sp->in_dim / 256) * (bits == 10 ? 144 : 210);
            memcpy(g_gguf_stage2 + t_off[i][p], *(void **)((char *)lc + sp->p_off),
                   (size_t)sp->rows * rb);
            *(void **)((char *)lc + sp->p_off) = g_gguf_stage2 + t_off[i][p];
        }
    }
    g_gguf_stage2_gpu = [ctx->device newBufferWithBytesNoCopy:st2 length:total
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    fprintf(stderr, "[stage2] %zu MB 2MB-aligned copy of GPU-read QK tensors (scope %d) in %.0f ms\n",
            total >> 20, scope, now_ms() - t0);
}

// ============================================================================
// MTP (Multi-Token Prediction) — speculative decoding head
// ============================================================================

typedef struct {
    // Pre-fc norms (combine embedding and hidden before transformer layer)
    uint16_t *pre_fc_norm_embedding_w;  // [2048] BF16
    uint16_t *pre_fc_norm_hidden_w;     // [2048] BF16

    // Transformer layer (identical structure to main model layers)
    uint16_t *input_layernorm_w;        // [2048] BF16
    uint16_t *post_attn_norm_w;         // [2048] BF16

    // Attention (GQA, same as main model full-attention layers)
    uint32_t *q_w;  uint16_t *q_s, *q_b;  // [8192, 2048] 4-bit
    uint32_t *k_w;  uint16_t *k_s, *k_b;  // [512, 2048] 4-bit
    uint32_t *v_w;  uint16_t *v_s, *v_b;  // [512, 2048] 4-bit
    uint32_t *o_w;  uint16_t *o_s, *o_b;  // [2048, 4096] 4-bit
    uint16_t *k_norm_w, *q_norm_w;        // [256] BF16

    // Routing gate
    uint32_t *gate_w;  uint16_t *gate_s, *gate_b;  // [256, 2048] 8-bit

    // Shared expert
    uint32_t *shared_gate_w;         // [512, 2048] 4-bit
    uint16_t *shared_gate_s, *shared_gate_b;
    uint32_t *shared_up_w;           // [512, 2048] 4-bit
    uint16_t *shared_up_s, *shared_up_b;
    uint32_t *shared_down_w;         // [2048, 512] 4-bit
    uint16_t *shared_down_s, *shared_down_b;
    uint32_t *shared_gate_gate_w;         // [1, 2048] 8-bit
    uint16_t *shared_gate_gate_s, *shared_gate_gate_b;

    // Output projection
    uint32_t *fc_w;       uint16_t *fc_s, *fc_b;  // [2048, 512] 4-bit (packed [2048, 4096])
    uint16_t *final_norm_w;  // [2048] BF16

    // Expert file for routed experts
    int expert_fd;             // fd for layer_40.bin
    int expert_bits;           // 4 (matching main model)

    int loaded;
} MTPWeights;

static MTPWeights g_mtp = { .loaded = 0 };

static void mtp_init(WeightFile *wf, const char *model_path) {
    if (g_mtp.loaded) return;

    char name[256];
    #define MTP_T(fmt) (snprintf(name, sizeof(name), fmt, 0), (void*)get_tensor_ptr(wf, name))

    g_mtp.pre_fc_norm_embedding_w = (uint16_t *)MTP_T("mtp.pre_fc_norm_embedding.weight");
    g_mtp.pre_fc_norm_hidden_w    = (uint16_t *)MTP_T("mtp.pre_fc_norm_hidden.weight");

    g_mtp.input_layernorm_w       = (uint16_t *)MTP_T("mtp.layers.%d.input_layernorm.weight");
    g_mtp.post_attn_norm_w        = (uint16_t *)MTP_T("mtp.layers.%d.post_attention_layernorm.weight");

    g_mtp.q_w = (uint32_t *)MTP_T("mtp.layers.%d.self_attn.q_proj.weight");
    g_mtp.q_s = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.q_proj.scales");
    g_mtp.q_b = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.q_proj.biases");
    g_mtp.k_w = (uint32_t *)MTP_T("mtp.layers.%d.self_attn.k_proj.weight");
    g_mtp.k_s = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.k_proj.scales");
    g_mtp.k_b = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.k_proj.biases");
    g_mtp.v_w = (uint32_t *)MTP_T("mtp.layers.%d.self_attn.v_proj.weight");
    g_mtp.v_s = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.v_proj.scales");
    g_mtp.v_b = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.v_proj.biases");
    g_mtp.o_w = (uint32_t *)MTP_T("mtp.layers.%d.self_attn.o_proj.weight");
    g_mtp.o_s = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.o_proj.scales");
    g_mtp.o_b = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.o_proj.biases");
    g_mtp.k_norm_w = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.k_norm.weight");
    g_mtp.q_norm_w = (uint16_t *)MTP_T("mtp.layers.%d.self_attn.q_norm.weight");

    g_mtp.gate_w = (uint32_t *)MTP_T("mtp.layers.%d.mlp.gate.weight");
    g_mtp.gate_s = (uint16_t *)MTP_T("mtp.layers.%d.mlp.gate.scales");
    g_mtp.gate_b = (uint16_t *)MTP_T("mtp.layers.%d.mlp.gate.biases");

    g_mtp.shared_gate_w = (uint32_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.gate_proj.weight");
    g_mtp.shared_gate_s = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.gate_proj.scales");
    g_mtp.shared_gate_b = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.gate_proj.biases");
    g_mtp.shared_up_w   = (uint32_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.up_proj.weight");
    g_mtp.shared_up_s   = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.up_proj.scales");
    g_mtp.shared_up_b   = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.up_proj.biases");
    g_mtp.shared_down_w = (uint32_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.down_proj.weight");
    g_mtp.shared_down_s = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.down_proj.scales");
    g_mtp.shared_down_b = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert.down_proj.biases");
    g_mtp.shared_gate_gate_w = (uint32_t *)MTP_T("mtp.layers.%d.mlp.shared_expert_gate.weight");
    g_mtp.shared_gate_gate_s = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert_gate.scales");
    g_mtp.shared_gate_gate_b = (uint16_t *)MTP_T("mtp.layers.%d.mlp.shared_expert_gate.biases");

    g_mtp.fc_w = (uint32_t *)MTP_T("mtp.fc.weight");
    g_mtp.fc_s = (uint16_t *)MTP_T("mtp.fc.scales");
    g_mtp.fc_b = (uint16_t *)MTP_T("mtp.fc.biases");
    g_mtp.final_norm_w = (uint16_t *)MTP_T("mtp.norm.weight");

    #undef MTP_T

    // Check critical tensors
    int ok = (g_mtp.pre_fc_norm_embedding_w && g_mtp.pre_fc_norm_hidden_w &&
              g_mtp.fc_w && g_mtp.final_norm_w && g_mtp.input_layernorm_w);
    if (ok) {
        g_mtp.loaded = 1;
        // Try to open MTP expert file (in the model's packed_experts dir)
        char path[512];
        snprintf(path, sizeof(path), "%s/packed_experts/layer_40.bin", model_path);
        g_mtp.expert_fd = open(path, O_RDONLY);
        if (g_mtp.expert_fd >= 0) {
            fcntl(g_mtp.expert_fd, F_RDAHEAD, 0);
            g_mtp.expert_bits = 4;
            fprintf(stderr, "[mtp] MTP weights loaded, expert file: %s\n", path);
        } else {
            fprintf(stderr, "[mtp] MTP weights loaded, expert file NOT FOUND: %s\n", path);
        }
    } else {
        fprintf(stderr, "[mtp] MTP weights NOT FOUND — speculative decoding disabled\n");
    }
}

// MTP forward pass: predicts next token from hidden state + current token embedding.
// Returns 1 if a token was generated, 0 if MTP is not available.
// The predicted token is written to *next_token and hidden is updated in-place.
// MTP K/V cache: the MTP attention attends the FULL context — the cache is
// filled with the MTP's own K/V per token, both at prefill time (via
// mtp_cache_fill) and during generation (inside mtp_forward).
#define MTP_N_Q_HEADS 16
#define MTP_N_KV_HEADS 2
#define MTP_HEAD_DIM 256
#define MTP_Q_DIM (MTP_N_Q_HEADS * MTP_HEAD_DIM)    // 4096 (q only)
#define MTP_KV_DIM (MTP_N_KV_HEADS * MTP_HEAD_DIM)  // 512
#define MTP_O_IN_DIM (16 * MTP_HEAD_DIM)             // 4096
#define MTP_KV_CACHE_MAX 8192                        // 32 MB K + 32 MB V

static float *mtp_k_cache = NULL, *mtp_v_cache = NULL;
static int mtp_cache_len = 0;

// Compute the MTP K/V for one token (norm chain + projections + RoPE) and
// append to the cache. embed = token embedding, hidden = pre-final-norm
// hidden (post-MoE residual), pos = sequence position.
static void mtp_kv_append(WeightFile *wf, const float *embed, const float *hidden, int pos) {
    if (!g_mtp.loaded) return;
    if (!mtp_k_cache) {
        mtp_k_cache = calloc(MTP_KV_CACHE_MAX * MTP_KV_DIM, sizeof(float));
        mtp_v_cache = calloc(MTP_KV_CACHE_MAX * MTP_KV_DIM, sizeof(float));
        mtp_cache_len = 0;
    }
    if (mtp_cache_len >= MTP_KV_CACHE_MAX) {
        // Slide the window: drop the oldest half (the attention is local
        // enough that the full context beyond 8K rarely matters).
        memmove(mtp_k_cache, mtp_k_cache + (MTP_KV_CACHE_MAX / 2) * MTP_KV_DIM,
                (MTP_KV_CACHE_MAX / 2) * MTP_KV_DIM * sizeof(float));
        memmove(mtp_v_cache, mtp_v_cache + (MTP_KV_CACHE_MAX / 2) * MTP_KV_DIM,
                (MTP_KV_CACHE_MAX / 2) * MTP_KV_DIM * sizeof(float));
        mtp_cache_len = MTP_KV_CACHE_MAX / 2;
    }

    float emb_normed[HIDDEN_DIM], hidden_normed[HIDDEN_DIM], h[HIDDEN_DIM], normed[HIDDEN_DIM];
    cpu_rms_norm((float *)embed, g_mtp.pre_fc_norm_embedding_w, emb_normed, HIDDEN_DIM, RMS_NORM_EPS);
    cpu_rms_norm((float *)hidden, g_mtp.pre_fc_norm_hidden_w, hidden_normed, HIDDEN_DIM, RMS_NORM_EPS);
    for (int i = 0; i < HIDDEN_DIM; i++) h[i] = emb_normed[i] + hidden_normed[i];
    cpu_rms_norm(h, g_mtp.input_layernorm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);

    int kv_bits = (g_mtp.k_s && g_mtp.k_b) ? 4 : 0;
    float k_buf[MTP_KV_DIM], v_buf[MTP_KV_DIM];
    cpu_dequant_matvec(g_mtp.k_w, g_mtp.k_s, g_mtp.k_b, normed, k_buf, MTP_KV_DIM, HIDDEN_DIM, GROUP_SIZE, kv_bits);
    cpu_dequant_matvec(g_mtp.v_w, g_mtp.v_s, g_mtp.v_b, normed, v_buf, MTP_KV_DIM, HIDDEN_DIM, GROUP_SIZE, kv_bits);
    for (int hh = 0; hh < MTP_N_KV_HEADS; hh++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < MTP_HEAD_DIM; d++) sum_sq += k_buf[hh * MTP_HEAD_DIM + d] * k_buf[hh * MTP_HEAD_DIM + d];
        float inv_rms = 1.0f / sqrtf(sum_sq / MTP_HEAD_DIM + RMS_NORM_EPS);
        for (int d = 0; d < MTP_HEAD_DIM; d++)
            k_buf[hh * MTP_HEAD_DIM + d] *= inv_rms * bf16_to_f32(g_mtp.k_norm_w[d]);
    }
    // RoPE on K at its position (Q gets RoPE at the frontier in mtp_forward)
    apply_rotary_emb(k_buf, k_buf, pos, MTP_N_KV_HEADS, MTP_N_KV_HEADS, MTP_HEAD_DIM, ROTARY_DIM);

    memcpy(mtp_k_cache + mtp_cache_len * MTP_KV_DIM, k_buf, MTP_KV_DIM * sizeof(float));
    memcpy(mtp_v_cache + mtp_cache_len * MTP_KV_DIM, v_buf, MTP_KV_DIM * sizeof(float));
    mtp_cache_len++;
}

// Fill the MTP K/V cache for a batch of prefill positions (per-token or
// per-chunk hiddens with their embeddings).
static void mtp_cache_fill(WeightFile *wf, const int *tokens,
                           const float *embed_batch,
                           const float *hidden_batch, int count, int pos_base) {
    if (!g_use_mtp) return;
    for (int i = 0; i < count; i++) {
        // MTP reference dump (FINCHMOE_MTP_DUMP): prefill record
        if (getenv("FINCHMOE_MTP_DUMP")) {
            static FILE *md = NULL;
            if (!md) md = fopen("/tmp/mtp_ref_input.bin", "wb");
            if (md) {
                int32_t pos_i = pos_base + i, tok_i = tokens ? tokens[i] : -1;
                float zeros[VOCAB_SIZE];
                memset(zeros, 0, sizeof(zeros));
                fwrite(&pos_i, sizeof(int32_t), 1, md);
                fwrite(&tok_i, sizeof(int32_t), 1, md);
                fwrite(hidden_batch + (size_t)i * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, md);
                fwrite(zeros, sizeof(float), VOCAB_SIZE, md);
                fflush(md);
            }
        }
        mtp_kv_append(wf, embed_batch + (size_t)i * HIDDEN_DIM,
                      hidden_batch + (size_t)i * HIDDEN_DIM, pos_base + i);
    }
}

static int mtp_forward(WeightFile *wf, float *hidden, int current_token,
                       int *next_token, float *logits_buf) {
    if (!g_mtp.loaded || g_mtp.expert_fd < 0) return 0;

    // Step 1: Get embedding for current token
    float embed_buf[HIDDEN_DIM];
    embed_lookup(wf, current_token, embed_buf);

    // Step 2: Pre-norm embedding and hidden
    float emb_normed[HIDDEN_DIM], hidden_normed[HIDDEN_DIM];
    cpu_rms_norm(embed_buf, g_mtp.pre_fc_norm_embedding_w, emb_normed, HIDDEN_DIM, RMS_NORM_EPS);
    cpu_rms_norm(hidden,    g_mtp.pre_fc_norm_hidden_w,    hidden_normed, HIDDEN_DIM, RMS_NORM_EPS);

    // Step 3: Combine
    float h[HIDDEN_DIM];
    for (int i = 0; i < HIDDEN_DIM; i++) h[i] = emb_normed[i] + hidden_normed[i];

    // Step 4: Input norm
    float normed[HIDDEN_DIM];
    cpu_rms_norm(h, g_mtp.input_layernorm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);

    // Step 5: MTP attention (16 Q heads + gates, 2 KV heads, 256 dim).
    // The current token's K/V is appended to the full-context cache by
    // mtp_kv_append (which also computes the norm chain + projections).
    mtp_kv_append(wf, embed_buf, hidden, mtp_cache_len);

    // Q projection + gate (element-interleaved), 4-bit if scales present
    int q_bits = (g_mtp.q_s && g_mtp.q_b) ? 4 : 0;
    float q_raw[MTP_Q_DIM * 2];
    cpu_dequant_matvec(g_mtp.q_w, g_mtp.q_s, g_mtp.q_b, normed, q_raw, MTP_Q_DIM * 2, HIDDEN_DIM, GROUP_SIZE, q_bits);

    // De-interleave Q and gate — ELEMENT-interleaved per head (q,g,q,g...:
    // q at [h*512 + 2d], gate at [h*512 + 2d+1] — per the reference's
    // strided views of the [head × 2×dim] Q projection). Per-head QKNorm
    // (RMS, like the main model).
    float q_buf[MTP_Q_DIM], gate_buf[MTP_Q_DIM];
    for (int h = 0; h < MTP_N_Q_HEADS; h++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < MTP_HEAD_DIM; d++) {
            float qv = q_raw[h * 512 + 2 * d];
            q_buf[h * MTP_HEAD_DIM + d] = qv;
            gate_buf[h * MTP_HEAD_DIM + d] = q_raw[h * 512 + 2 * d + 1];
            sum_sq += qv * qv;
        }
        float inv_rms = 1.0f / sqrtf(sum_sq / MTP_HEAD_DIM + RMS_NORM_EPS);
        for (int d = 0; d < MTP_HEAD_DIM; d++)
            q_buf[h * MTP_HEAD_DIM + d] *= inv_rms * bf16_to_f32(g_mtp.q_norm_w[d]);
    }
    // RoPE on Q at the frontier position (K was rope'd by mtp_kv_append).
    static float mtp_k_scratch[MTP_KV_DIM];
    apply_rotary_emb(q_buf, mtp_k_scratch, mtp_cache_len - 1, MTP_N_Q_HEADS, MTP_N_KV_HEADS,
                     MTP_HEAD_DIM, ROTARY_DIM);

    // Multi-head attention with softmax over cached tokens
    int n_ctx = mtp_cache_len;
    float attn_out[MTP_O_IN_DIM];  // 4096: all 16 Q heads output
    memset(attn_out, 0, sizeof(attn_out));
    float inv_sqrt_dh = 1.0f / sqrtf((float)MTP_HEAD_DIM);

    // For each of the 16 Q heads (O projection input); GQA: 8 Q heads/KV head
    for (int qh = 0; qh < 16; qh++) {
        int kvh = qh / 8;
        float *q_head = q_buf + qh * MTP_HEAD_DIM;

        // Attention scores against all cached keys
        float scores[MTP_KV_CACHE_MAX];
        float max_s = -1e30f;
        for (int t = 0; t < n_ctx; t++) {
            float *k_head = mtp_k_cache + t * MTP_KV_DIM + kvh * MTP_HEAD_DIM;
            float s = 0;
            for (int d = 0; d < MTP_HEAD_DIM; d++) s += q_head[d] * k_head[d];
            s *= inv_sqrt_dh;
            scores[t] = s;
            if (s > max_s) max_s = s;
        }

        // Softmax
        float sum_exp = 0;
        for (int t = 0; t < n_ctx; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_exp += scores[t];
        }

        // Weighted sum of V, then the attention output gate (sigmoid)
        float *o_head = attn_out + qh * MTP_HEAD_DIM;
        for (int t = 0; t < n_ctx; t++) {
            float *v_head = mtp_v_cache + t * MTP_KV_DIM + kvh * MTP_HEAD_DIM;
            float w = scores[t] / sum_exp;
            for (int d = 0; d < MTP_HEAD_DIM; d++) o_head[d] += w * v_head[d];
        }
        for (int d = 0; d < MTP_HEAD_DIM; d++)
            o_head[d] *= 1.0f / (1.0f + expf(-gate_buf[qh * MTP_HEAD_DIM + d]));
    }

    // O projection: 4-bit if scales present, BF16 if NULL
    int o_bits = (g_mtp.o_s && g_mtp.o_b) ? 4 : 0;
    float attn_proj[HIDDEN_DIM];
    cpu_dequant_matvec(g_mtp.o_w, g_mtp.o_s, g_mtp.o_b, attn_out, attn_proj, HIDDEN_DIM, MTP_O_IN_DIM, GROUP_SIZE, o_bits);

    // Residual
    for (int i = 0; i < HIDDEN_DIM; i++) h[i] += attn_proj[i];

    #undef MTP_N_Q_HEADS
    #undef MTP_N_KV_HEADS
    #undef MTP_HEAD_DIM
    #undef MTP_Q_DIM
    #undef MTP_KV_DIM
    #undef MTP_O_IN_DIM
    #undef MTP_KV_CACHE_MAX

    // Post-attention norm
    float h_post[HIDDEN_DIM];
    cpu_rms_norm(h, g_mtp.post_attn_norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);

    // Step 6: MoE routing
    float gate_scores[256];
    int gate_bits = (g_mtp.gate_s && g_mtp.gate_b) ? 8 : 0;  // BF16 in the MTP manifest
    cpu_dequant_matvec(g_mtp.gate_w, g_mtp.gate_s, g_mtp.gate_b, h_post, gate_scores,
                       256, HIDDEN_DIM, GROUP_SIZE, gate_bits);
    cpu_softmax(gate_scores, 256);

    int K = 8;  // model trained with 8 experts/token (K=2 produces garbage)
    int expert_indices[8];
    float expert_weights[8];
    cpu_topk(gate_scores, 256, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    {
        static int rt_dbg = 0;
        if (rt_dbg < 3) {
            float ent = 0;
            for (int i = 0; i < 256; i++) if (gate_scores[i] > 0) ent -= gate_scores[i] * logf(gate_scores[i]);
            fprintf(stderr, "[mtp-route] top8: ");
            for (int k = 0; k < K; k++) fprintf(stderr, "%d(%.2f) ", expert_indices[k], expert_weights[k]);
            fprintf(stderr, "entropy=%.2f\n", ent);
            rt_dbg++;
        }
    }

    // Step 7: Shared expert gate
    float shared_gate_score;
    int sgg_bits = (g_mtp.shared_gate_gate_s && g_mtp.shared_gate_gate_b) ? 4 : 0;
    cpu_dequant_matvec(g_mtp.shared_gate_gate_w, g_mtp.shared_gate_gate_s,
                       g_mtp.shared_gate_gate_b, h_post, &shared_gate_score,
                       1, HIDDEN_DIM, GROUP_SIZE, sgg_bits);
    float shared_weight = 1.0f / (1.0f + expf(-shared_gate_score));  // sigmoid

    // Step 8: Shared expert gate/up
    float shared_gate[SHARED_INTERMEDIATE], shared_up[SHARED_INTERMEDIATE];
    int sg_bits = (g_mtp.shared_gate_s && g_mtp.shared_gate_b) ? 4 : 0;
    int su_bits = (g_mtp.shared_up_s && g_mtp.shared_up_b) ? 4 : 0;
    cpu_dequant_matvec(g_mtp.shared_gate_w, g_mtp.shared_gate_s, g_mtp.shared_gate_b,
                       h_post, shared_gate, 512, HIDDEN_DIM, GROUP_SIZE, sg_bits);
    cpu_dequant_matvec(g_mtp.shared_up_w, g_mtp.shared_up_s, g_mtp.shared_up_b,
                       h_post, shared_up, 512, HIDDEN_DIM, GROUP_SIZE, su_bits);

    // Shared expert SwiGLU
    float shared_act[SHARED_INTERMEDIATE];
    cpu_swiglu(shared_gate, shared_up, shared_act, 512);

    // Shared expert down
    float shared_out[HIDDEN_DIM];
    int sd_bits = (g_mtp.shared_down_s && g_mtp.shared_down_b) ? 4 : 0;
    cpu_dequant_matvec(g_mtp.shared_down_w, g_mtp.shared_down_s, g_mtp.shared_down_b,
                       shared_act, shared_out, HIDDEN_DIM, 512, GROUP_SIZE, sd_bits);

    // Step 9: Routed experts (persistent buffers, allocated once)
    // The MTP layer's expert file is packed in the 4-bit layout
    // (g_mtp.expert_bits = 4), independent of the main model's expert format.
    static float *moe_out = NULL;
    static void *expert_data = NULL;
    static size_t expert_data_sz = 0;
    size_t esz = EXPERT_SIZE_4BIT;
    if (!moe_out) moe_out = calloc(HIDDEN_DIM, sizeof(float));
    if (!expert_data || expert_data_sz != esz) {
        free(expert_data);
        expert_data = malloc(esz);
        expert_data_sz = esz;
    }
    memset(moe_out, 0, HIDDEN_DIM * sizeof(float));

    for (int k = 0; k < K; k++) {
        int eidx = expert_indices[k];
        float weight = expert_weights[k];

        ssize_t n = pread(g_mtp.expert_fd, expert_data, esz, (off_t)eidx * esz);
        if (n != (ssize_t)esz) { fprintf(stderr, "[mtp] expert %d pread fail: %zd/%zu\n", eidx, n, esz); continue; }


        uint32_t *ew = (uint32_t *)expert_data;  // gate_W at offset 0
        uint16_t *es = (uint16_t *)((char *)expert_data + GATE_S_OFF_4);  // gate scales
        uint16_t *eb = (uint16_t *)((char *)expert_data + GATE_B_OFF_4);  // gate biases
        uint32_t *uw = (uint32_t *)((char *)expert_data + UP_W_OFF_4);
        uint16_t *us = (uint16_t *)((char *)expert_data + UP_S_OFF_4);
        uint16_t *ub = (uint16_t *)((char *)expert_data + UP_B_OFF_4);
        uint32_t *dw = (uint32_t *)((char *)expert_data + DOWN_W_OFF_4);
        uint16_t *ds = (uint16_t *)((char *)expert_data + DOWN_S_OFF_4);
        uint16_t *db = (uint16_t *)((char *)expert_data + DOWN_B_OFF_4);

        float gate_out[MOE_INTERMEDIATE], up_out[MOE_INTERMEDIATE];
        cpu_dequant_matvec(ew, es, eb, h_post, gate_out, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 4);
        cpu_dequant_matvec(uw, us, ub, h_post, up_out,   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 4);

        float act_out[MOE_INTERMEDIATE];
        cpu_swiglu(gate_out, up_out, act_out, MOE_INTERMEDIATE);

        float expert_out[HIDDEN_DIM];
        cpu_dequant_matvec(dw, ds, db, act_out, expert_out, HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, 4);

        for (int i = 0; i < HIDDEN_DIM; i++) moe_out[i] += weight * expert_out[i];
    }
    // Buffers are persistent (static) — no free needed

    // Step 10: Combine
    for (int i = 0; i < HIDDEN_DIM; i++) {
        h[i] = h[i] + moe_out[i] + shared_weight * shared_out[i];
    }

    // Step 11: Final norm
    float final_hidden[HIDDEN_DIM];
    cpu_rms_norm(h, g_mtp.final_norm_w, final_hidden, HIDDEN_DIM, RMS_NORM_EPS);

    // (FC hidden-half norm variants were swept 2026-08-14: pre_fc_norm_hidden
    // → cos -0.59/-0.70, raw h → 0.58/0.78, input_layernorm → 0.58/0.79 —
    // mtp.norm is confirmed.)

    // Step 12: FC projection — maps [hidden; embedding] -> hidden
    // fc.weight shape: [2048, 4096] 4-bit packed = [2048, 512] uint32
    // Input: concatenate final_hidden (2048) + embed_normed (2048)
    float fc_in[4096];
    memcpy(fc_in, final_hidden, HIDDEN_DIM * sizeof(float));
    memcpy(fc_in + HIDDEN_DIM, emb_normed, HIDDEN_DIM * sizeof(float));
    float fc_out[HIDDEN_DIM];
    cpu_dequant_matvec(g_mtp.fc_w, g_mtp.fc_s, g_mtp.fc_b, fc_in, fc_out,
                       HIDDEN_DIM, 4096, GROUP_SIZE, 4);
    // Use lm_head to project fc_out to vocabulary logits
    lm_head_forward(wf, fc_out, logits_buf);

    // Debug: trace MTP forward pass (first call only)
    static int mtp_dbg = 0;
    if (mtp_dbg < 1) {
        fprintf(stderr, "[mtp-dbg] embed_rms=%.4f hidden_rms=%.4f moe_out_rms=%.4f logits_rms=%.4f\n",
                vec_rms(embed_buf, HIDDEN_DIM), vec_rms(hidden, HIDDEN_DIM),
                vec_rms(moe_out, HIDDEN_DIM), vec_rms(logits_buf, VOCAB_SIZE));
        mtp_dbg++;
    }

    // Step 13: Sample
    *next_token = cpu_sample_temp(logits_buf, VOCAB_SIZE, g_temperature, g_top_k);

    // Update hidden for potential next MTP step (chain)
    memcpy(hidden, h, HIDDEN_DIM * sizeof(float));

    return 1;
}

// ============================================================================
// Deferred expert state: holds state for async GPU expert compute.
// GPU experts are submitted async (commit without wait), and the wait+combine
// happens at the start of the NEXT layer. This overlaps ~1ms of GPU expert
// compute with the next layer's attention+routing CPU/GPU work.
// ============================================================================

typedef struct {
    int active;                         // 1 if there's a deferred GPU expert to wait for
    int gpu_combined;                   // 1 if CMD3 includes combine+residual+norm on GPU
                                        // (next layer can skip deferred_wait+finalize+input_norm
                                        //  and submit CMD1 immediately -- buf_input is ready)
    id<MTLCommandBuffer> cmd_experts;   // the async command buffer (committed but not waited)
    float expert_weights[MAX_K];        // routing weights for weighted accumulation
    int valid[MAX_K];                   // which experts loaded successfully
    int actual_K;                       // number of experts
    float h_mid[HIDDEN_DIM];            // saved h_mid for final combine
    float shared_gate_score;            // saved shared expert gate score
    float *hidden;                      // pointer to hidden state (for writing final result)
    int layer_idx;                      // which layer produced this deferred state
} DeferredExpertState;

static DeferredExpertState g_deferred = { .active = 0 };

// Wait for the deferred GPU expert command buffer to complete.
// Split from finalize so timing can be measured independently.
static void wait_deferred_experts_gpu(void) {
    if (!g_deferred.active) return;
    [g_deferred.cmd_experts waitUntilCompleted];
}

// CPU readback + accumulate + combine after GPU is done.
// Must be called after wait_deferred_experts_gpu().
// When gpu_combined=1, the GPU already computed the combine+residual+norm
// in CMD3, so we just need to read back the hidden state from buf_moe_hidden.
static void finalize_deferred_experts(void) {
    if (!g_deferred.active) return;

    if (g_deferred.gpu_combined) {
        // GPU-side combine: hidden state is already in buf_moe_hidden.
        // buf_input already has the normalized input for the next layer's CMD1.
        // Just read back hidden (needed for the residual connection in future layers).
        memcpy(g_deferred.hidden, [g_metal->buf_moe_hidden contents],
               HIDDEN_DIM * sizeof(float));
        if (getenv("FINCHMOE_DUMP_HIDDEN")) {
            static FILE *hf3 = NULL;
            if (!hf3) hf3 = fopen("/tmp/hidden_dump.bin", "wb");
            if (hf3) { fwrite(g_deferred.hidden, sizeof(float), HIDDEN_DIM, hf3); fflush(hf3); }
        }
    } else {
        // CPU-side combine (original path)
        // Debug: dump final-combine components (last layer, FINCHMOE_PF_DUMP)
        if (getenv("FINCHMOE_PF_DUMP") && g_deferred.layer_idx == NUM_LAYERS - 1) {
            static FILE *fcb = NULL;
            if (!fcb) fcb = fopen("/tmp/final_base.bin", "wb");
            if (fcb) {
                fwrite(g_deferred.h_mid, sizeof(float), HIDDEN_DIM, fcb);
                fwrite(&g_deferred.shared_gate_score, sizeof(float), 1, fcb);
                for (int k = 0; k < MAX_K; k++) {
                    fwrite(&g_deferred.expert_weights[k], sizeof(float), 1, fcb);
                    fwrite(&g_deferred.valid[k], sizeof(int), 1, fcb);
                    fwrite((const float *)[g_metal->buf_multi_expert_out[k] contents],
                           sizeof(float), HIDDEN_DIM, fcb);
                }
                fwrite((const float *)[g_metal->buf_shared_out contents],
                       sizeof(float), HIDDEN_DIM, fcb);
                fflush(fcb);
            }
        }
        // Read back and accumulate routed expert outputs
        float moe_out[HIDDEN_DIM];
        memset(moe_out, 0, sizeof(moe_out));
        for (int k = 0; k < g_deferred.actual_K; k++) {
            if (!g_deferred.valid[k]) continue;
            float *expert_result = (float *)[g_metal->buf_multi_expert_out[k] contents];
            cpu_vec_madd(moe_out, expert_result, g_deferred.expert_weights[k], HIDDEN_DIM);
        }

        // Read shared expert result
        float shared_out[HIDDEN_DIM];
        memcpy(shared_out, [g_metal->buf_shared_out contents], HIDDEN_DIM * sizeof(float));

        // Apply shared expert gate
        float shared_weight = cpu_sigmoid(g_deferred.shared_gate_score);
        for (int i = 0; i < HIDDEN_DIM; i++) {
            shared_out[i] *= shared_weight;
        }

        // Final combine: hidden = h_mid + moe_out + shared_out
        for (int i = 0; i < HIDDEN_DIM; i++) {
            g_deferred.hidden[i] = g_deferred.h_mid[i] + moe_out[i] + shared_out[i];
        }
        if (getenv("FINCHMOE_NANTRACE")) {
            fprintf(stderr, "[NANTRACE] L%d h_mid=%.6f moe=%.6f shared=%.6f shared_w=%.6f hidden=%.6f\n",
                    g_deferred.layer_idx,
                    vec_rms(g_deferred.h_mid, HIDDEN_DIM),
                    vec_rms(moe_out, HIDDEN_DIM),
                    vec_rms(shared_out, HIDDEN_DIM),
                    shared_weight,
                    vec_rms(g_deferred.hidden, HIDDEN_DIM));
        }
    }

    if (getenv("FINCHMOE_DUMP_HIDDEN")) {
        static FILE *hf2 = NULL;
        if (!hf2) hf2 = fopen("/tmp/hidden_dump.bin", "wb");
        if (hf2) { fwrite(g_deferred.hidden, sizeof(float), HIDDEN_DIM, hf2); fflush(hf2); }
    }

    g_deferred.active = 0;
    g_deferred.gpu_combined = 0;
    g_deferred.cmd_experts = nil;
}

// Complete the deferred GPU expert compute: wait for GPU, read back, accumulate, combine.
// Must be called before the next layer modifies static scratch buffers.
static void complete_deferred_experts(void) {
    wait_deferred_experts_gpu();
    finalize_deferred_experts();
}

// Discard the deferred GPU expert result: wait for GPU to finish (for buffer safety)
// but skip the CPU readback/combine. Used during prefill for intermediate tokens
// where the hidden state will be immediately overwritten by the next token's embedding.
// This saves ~0.1-0.2ms per prefill token (avoids unnecessary memcpy + combine work).
static void discard_deferred_experts(void) {
    wait_deferred_experts_gpu();
    // Clear deferred state without reading back results
    if (g_deferred.active) {
        g_deferred.active = 0;
        g_deferred.gpu_combined = 0;
        g_deferred.cmd_experts = nil;
    }
}

// ============================================================================
// Fused layer forward: GPU/CPU overlap + deferred expert pipeline
//
// Pipeline per layer (3 cmd buffers, GPU-side combine in CMD3):
//
//   FAST PATH (when previous CMD3 did GPU-side combine):
//     CMD1: submit immediately (buf_input already populated by CMD3(N-1))
//     WAIT: CMD1 complete (implies CMD3(N-1) also done, queue is serial)
//     CPU:  finalize deferred (read back hidden from buf_moe_hidden)
//
//   SLOW PATH (first layer, or last layer's CMD3 without GPU combine):
//     [DEFERRED] Wait for PREVIOUS layer's CMD3 (if any) + CPU combine
//     CPU:  input_norm(hidden) -> normed -> buf_input
//     CMD1: attention projections (commit)
//     WAIT: CMD1 complete
//
//   Then (both paths):
//     CPU:  attention compute (RoPE/softmax/delta-net)
//     CMD2: o_proj + residual + norm + routing + shared expert projs (8 encoders, 1 commit)
//     WAIT: CMD2 complete
//     CPU:  softmax + top-K routing
//     I/O:  parallel pread K experts (4 pthreads)
//     CMD3: K expert forwards + shared SwiGLU + shared down
//           + moe_combine_residual + rms_norm -> buf_input (ASYNC commit, NO wait)
//     RETURN: GPU experts + combine running async
//
// GPU-side combine eliminates the 0.83ms deferred_wait + CPU combine + input_norm
// at the start of each layer, allowing CMD1 to be submitted immediately.
//
// Key optimizations:
//   1. Parallel pread (4 threads) instead of sequential: ~4x I/O speedup
//   2. o_proj fused into CMD2 with routing (saves 1 commit+wait)
//   3. Deferred CMD3 (expert GPU compute overlapped with next layer)
//   4. GPU-side combine in CMD3 (eliminates CPU deferred_wait + combine + norm)
// ============================================================================

// Static scratch buffers — allocated once, reused across all 60 layers per token.
// Eliminates ~20 malloc/free per layer = ~1200 alloc/free per token.
static float *s_normed    = NULL;   // [HIDDEN_DIM]
static float *s_residual  = NULL;   // [HIDDEN_DIM]
static float *s_attn_proj = NULL;   // [HIDDEN_DIM]
static float *s_h_post    = NULL;   // [HIDDEN_DIM]
static float *s_h_mid     = NULL;   // [HIDDEN_DIM]
static float *s_gate_scores = NULL; // [NUM_EXPERTS]
static float *s_spec_gate_scores = NULL; // [NUM_EXPERTS] speculative routing scratch
static int s_spec_indices[MAX_K];         // speculative routing predicted expert indices
static int s_spec_count = 0;              // number of speculative predictions this layer
static float *s_shared_gate = NULL; // [SHARED_INTERMEDIATE]
static float *s_shared_up  = NULL;  // [SHARED_INTERMEDIATE]
static float *s_moe_out   = NULL;   // [HIDDEN_DIM]
static float *s_shared_out = NULL;  // [HIDDEN_DIM]
// Full attention scratch
static float *s_q_proj_out = NULL;  // [NUM_ATTN_HEADS * HEAD_DIM * 2]
static float *s_k_proj_out = NULL;  // [NUM_KV_HEADS * HEAD_DIM]
static float *s_v_proj_out = NULL;  // [NUM_KV_HEADS * HEAD_DIM]
static float *s_q         = NULL;   // [NUM_ATTN_HEADS * HEAD_DIM]
static float *s_q_gate    = NULL;   // [NUM_ATTN_HEADS * HEAD_DIM]
static float *s_attn_out  = NULL;   // [NUM_ATTN_HEADS * HEAD_DIM]
// Linear attention scratch
static float *s_qkv_proj_out = NULL;   // [LINEAR_CONV_DIM]
static float *s_z_proj_out   = NULL;   // [LINEAR_TOTAL_VALUE]
static float *s_beta_proj_out = NULL;  // [LINEAR_NUM_V_HEADS]
static float *s_alpha_proj_out = NULL; // [LINEAR_NUM_V_HEADS]
static float *s_conv_out  = NULL;   // [LINEAR_CONV_DIM]
static float *s_out_vals  = NULL;   // [LINEAR_TOTAL_VALUE]
static float *s_gated_out = NULL;   // [LINEAR_TOTAL_VALUE]

static void init_layer_scratch(void) {
    if (s_normed) return;  // already initialized
    s_normed     = calloc(HIDDEN_DIM, sizeof(float));
    s_residual   = calloc(HIDDEN_DIM, sizeof(float));
    s_attn_proj  = calloc(HIDDEN_DIM, sizeof(float));
    s_h_post     = calloc(HIDDEN_DIM, sizeof(float));
    s_h_mid      = calloc(HIDDEN_DIM, sizeof(float));
    s_gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    s_spec_gate_scores = calloc(NUM_EXPERTS, sizeof(float));
    s_shared_gate = calloc(SHARED_INTERMEDIATE, sizeof(float));
    s_shared_up  = calloc(SHARED_INTERMEDIATE, sizeof(float));
    s_moe_out    = calloc(HIDDEN_DIM, sizeof(float));
    s_shared_out = calloc(HIDDEN_DIM, sizeof(float));
    s_q_proj_out = calloc(NUM_ATTN_HEADS * HEAD_DIM * 2, sizeof(float));
    s_k_proj_out = calloc(NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    s_v_proj_out = calloc(NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    s_q          = calloc(NUM_ATTN_HEADS * HEAD_DIM, sizeof(float));
    s_q_gate     = calloc(NUM_ATTN_HEADS * HEAD_DIM, sizeof(float));
    s_attn_out   = calloc(NUM_ATTN_HEADS * HEAD_DIM, sizeof(float));
    s_qkv_proj_out = calloc(LINEAR_CONV_DIM, sizeof(float));
    s_z_proj_out   = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    s_beta_proj_out = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    s_alpha_proj_out = calloc(LINEAR_NUM_V_HEADS, sizeof(float));
    s_conv_out   = calloc(LINEAR_CONV_DIM, sizeof(float));
    s_out_vals   = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
    s_gated_out  = calloc(LINEAR_TOTAL_VALUE, sizeof(float));
}

// Encode the linear-attention GDN chain into cmdbuf. Uses the fully fused
// kernel (conv1d + qk-norm + decay/beta + delta-net + gated norm in ONE
// dispatch, no intermediate global-memory round trips) when available;
// otherwise falls back to the original encoder chain.
// Prefill variant: same fused_gdn_full kernel, inputs/outputs bound to
// position m's slot in the batched buffers. State buffers stay at offset 0
// (in-place update — Metal hazard tracking serializes the M sequential
// dispatches within one command buffer). Requires fused_gdn_full.
// Batched GDN encoder: ONE dispatch processes all M positions of the chunk
// sequentially inside the kernel (in-kernel m-loop). This eliminates the
// cross-dispatch recurrent-state L2 hazard that the M separate chain
// dispatches had on this GPU (the run-to-run wobble root cause).
static void gpu_encode_gdn_batched(MetalCtx *ctx, id<MTLCommandBuffer> cmdbuf,
                                   int linear_layer_idx, LayerWeightCache *lc,
                                   uint32_t M) {
    uint32_t conv_dim = LINEAR_CONV_DIM;
    NSUInteger conv_w_off   = (NSUInteger)((const char *)lc->conv1d_w   - (const char *)[ctx->wf_buf contents]);
    NSUInteger a_log_off    = (NSUInteger)((const char *)lc->A_log      - (const char *)[ctx->wf_buf contents]);
    NSUInteger dt_bias_off  = (NSUInteger)((const char *)lc->dt_bias    - (const char *)[ctx->wf_buf contents]);
    NSUInteger gnorm_w_off  = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[ctx->wf_buf contents]);
    uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
    uint32_t kdim = LINEAR_KEY_DIM, vdim = LINEAR_VALUE_DIM;
    float eps = RMS_NORM_EPS;
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->fused_gdn_batched];
    // Reader-side barrier: the S8 merged-CB path encodes the qkv/z/ba
    // matvecs in the SAME command buffer before this dispatch — the barrier
    // (paired with the caller's synchronizeResource) makes their writes
    // visible. Harmless in the 2-CB path.
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_pf_qkv       offset:0          atIndex:1];
    [enc setBuffer:ctx->wf_buf           offset:conv_w_off atIndex:2];
    [enc setBuffer:ctx->buf_pf_z         offset:0          atIndex:3];
    [enc setBuffer:ctx->buf_pf_ba        offset:0          atIndex:4];
    [enc setBuffer:ctx->wf_buf           offset:a_log_off   atIndex:5];
    [enc setBuffer:ctx->wf_buf           offset:dt_bias_off atIndex:6];
    [enc setBuffer:ctx->wf_buf           offset:gnorm_w_off atIndex:7];
    [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:8];
    [enc setBuffer:ctx->buf_pf_oproj_in offset:0          atIndex:9];
    [enc setBuffer:ctx->buf_conv_qk[linear_layer_idx] offset:0 atIndex:10]; // per-head q/k histories
    [enc setBytes:&conv_dim length:4 atIndex:11];
    [enc setBytes:&khpv     length:4 atIndex:12];
    [enc setBytes:&kdim     length:4 atIndex:13];
    [enc setBytes:&vdim     length:4 atIndex:14];
    [enc setBytes:&M        length:4 atIndex:15];
    [enc setBytes:&eps      length:4 atIndex:16];
    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
    // Writer-side barrier: state + oproj_in writes visible to subsequent
    // dispatches in this CB (the out_proj reads the CPU bridge copy anyway).
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc endEncoding];
}

// Phase C S5: fused_gdn_batched_qk encoder (GGUF chunked chain on GPU).
// Stage-mirror binds for the staged tensors (conv_w / A_log / dt_bias /
// gated_norm_w — same heap as the norm weights in the residual-norm block);
// b_w/a_w are Q4_K in the GGUF mmap → per-tensor wraps (gguf_tbuf_get).
// Returns 0 on success, -1 if a required wrap/stage bind failed (caller
// falls back to the CPU chain).
static int gpu_encode_gdn_batched_gguf(MetalCtx *ctx, id<MTLCommandBuffer> cmdbuf,
                                       int linear_layer_idx, LayerWeightCache *lc,
                                       uint32_t M) {
    if (!ctx->fused_gdn_batched_qk || !ctx->gguf_stage_gpu) return -1;
    // The fused kernel decodes Q4_K in_proj rows — only valid for GGUF files
    // where in_proj_a/b are Q4_K (bits 10). Files with BF16-staged in_proj
    // (bits 0) fall back to the CPU chain.
    if (lc->b_bits != 10 || lc->a_bits != 10) return -1;
    const char *stage = (const char *)g_gguf_stage;
    size_t stage_len = g_gguf_stage_len;
    // conv1d.weight / dt_bias / linear_attn.norm.weight are F32→BF16-staged
    // (gguf_needs_bf16_stage); A_log is F32 in the mmap (wrap separately).
    const char *ptrs[3] = {
        (const char *)lc->conv1d_w,
        (const char *)lc->dt_bias,
        (const char *)lc->gated_norm_w,
    };
    NSUInteger offs[3] = {0};
    for (int i = 0; i < 3; i++) {
        if (ptrs[i] < stage || ptrs[i] >= stage + stage_len) return -1;
        offs[i] = (NSUInteger)(ptrs[i] - stage);
    }
    // b_w / a_w: Q4_K rows of [32, HIDDEN_DIM] (144B/256).
    size_t row_bytes = (size_t)(HIDDEN_DIM / 256) * 144;
    uint32_t b_delta = 0, a_delta = 0, alog_delta = 0;
    id<MTLBuffer> b_buf = gguf_tbuf_get(ctx, lc->b_w, row_bytes * LINEAR_NUM_V_HEADS, &b_delta);
    id<MTLBuffer> a_buf = gguf_tbuf_get(ctx, lc->a_w, row_bytes * LINEAR_NUM_V_HEADS, &a_delta);
    id<MTLBuffer> alog_buf = gguf_tbuf_get(ctx, lc->A_log, LINEAR_NUM_V_HEADS * sizeof(float), &alog_delta);
    if (!b_buf || !a_buf || !alog_buf) return -1;

    uint32_t conv_dim = LINEAR_CONV_DIM;
    uint32_t in_dim = HIDDEN_DIM;
    uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
    uint32_t kdim = LINEAR_KEY_DIM, vdim = LINEAR_VALUE_DIM;
    float eps = RMS_NORM_EPS;
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc setComputePipelineState:ctx->fused_gdn_batched_qk];
    [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_pf_qkv       offset:0          atIndex:1];
    [enc setBuffer:ctx->gguf_stage_gpu   offset:offs[0]    atIndex:2];
    [enc setBuffer:ctx->buf_pf_z         offset:0          atIndex:3];
    [enc setBuffer:b_buf                 offset:b_delta    atIndex:4];
    [enc setBuffer:a_buf                 offset:a_delta    atIndex:5];
    [enc setBuffer:ctx->buf_pf_input     offset:0          atIndex:6];
    [enc setBuffer:alog_buf              offset:alog_delta atIndex:7];
    [enc setBuffer:ctx->gguf_stage_gpu   offset:offs[1]    atIndex:8];
    [enc setBuffer:ctx->gguf_stage_gpu   offset:offs[2]    atIndex:9];
    [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:10];
    [enc setBuffer:ctx->buf_pf_oproj_in offset:0          atIndex:11];
    [enc setBuffer:ctx->buf_conv_qk[linear_layer_idx] offset:0 atIndex:12];
    [enc setBytes:&conv_dim length:4 atIndex:13];
    [enc setBytes:&in_dim   length:4 atIndex:14];
    [enc setBytes:&khpv     length:4 atIndex:15];
    [enc setBytes:&kdim     length:4 atIndex:16];
    [enc setBytes:&vdim     length:4 atIndex:17];
    [enc setBytes:&M        length:4 atIndex:18];
    [enc setBytes:&eps      length:4 atIndex:19];
    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc endEncoding];
    return 0;
}

static void gpu_encode_gdn_chain_slot(MetalCtx *ctx, id<MTLCommandBuffer> cmdbuf,
                                      int linear_layer_idx, LayerWeightCache *lc,
                                      uint32_t m, uint32_t M) {
    uint32_t conv_dim = LINEAR_CONV_DIM;
    NSUInteger conv_w_off   = (NSUInteger)((const char *)lc->conv1d_w   - (const char *)[ctx->wf_buf contents]);
    NSUInteger a_log_off    = (NSUInteger)((const char *)lc->A_log      - (const char *)[ctx->wf_buf contents]);
    NSUInteger dt_bias_off  = (NSUInteger)((const char *)lc->dt_bias    - (const char *)[ctx->wf_buf contents]);
    NSUInteger gnorm_w_off  = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[ctx->wf_buf contents]);
    uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
    uint32_t kdim = LINEAR_KEY_DIM, vdim = LINEAR_VALUE_DIM;
    float eps = RMS_NORM_EPS;
    // buf_pf_ba layout: beta region [0, M*32), alpha region [M*32, 2*M*32)
    NSUInteger qkv_off   = (NSUInteger)m * LINEAR_CONV_DIM * sizeof(float);
    NSUInteger z_off     = (NSUInteger)m * (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) * sizeof(float);
    NSUInteger beta_off  = (NSUInteger)m * 32 * sizeof(float);
    NSUInteger alpha_off = (NSUInteger)(M + m) * 32 * sizeof(float);
    NSUInteger out_off   = (NSUInteger)m * (LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM) * sizeof(float);
    // Order this dispatch after ALL prior buffer writes in the command
    // buffer: the chain reads the recurrent state (buf_conv_state /
    // buf_delta_state) updated in place by the previous position's chain.
    // Scope barriers alone don't flush L2 on this GPU — explicit resource
    // syncs for the two in-place state buffers (see metal_sync_buffer).
    metal_sync_buffer(cmdbuf, ctx->buf_conv_state[linear_layer_idx]);
    metal_sync_buffer(cmdbuf, ctx->buf_delta_state[linear_layer_idx]);
    id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc setComputePipelineState:ctx->fused_gdn_full];
    [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_pf_qkv    offset:qkv_off       atIndex:1];  // qkv in
    [enc setBuffer:ctx->wf_buf        offset:conv_w_off    atIndex:2];  // conv weights
    [enc setBuffer:ctx->buf_pf_z      offset:z_off         atIndex:3];  // z
    [enc setBuffer:ctx->buf_pf_ba     offset:alpha_off     atIndex:4];  // alpha
    [enc setBuffer:ctx->buf_pf_ba     offset:beta_off      atIndex:5];  // beta
    [enc setBuffer:ctx->wf_buf        offset:a_log_off     atIndex:6];
    [enc setBuffer:ctx->wf_buf        offset:dt_bias_off   atIndex:7];
    [enc setBuffer:ctx->wf_buf        offset:gnorm_w_off   atIndex:8];
    [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:9];
    [enc setBuffer:ctx->buf_pf_oproj_in offset:out_off     atIndex:10]; // gated output
    [enc setBytes:&conv_dim length:4 atIndex:11];
    [enc setBytes:&khpv     length:4 atIndex:12];
    [enc setBytes:&kdim     length:4 atIndex:13];
    [enc setBytes:&vdim     length:4 atIndex:14];
    [enc setBytes:&eps      length:4 atIndex:15];
    [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
    // Writer-side barrier: make the recurrent-state + oproj_in writes of
    // this chain slot visible to the next slot / out_proj (reader-side
    // placement is a no-op).
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [enc endEncoding];
}

static void gpu_encode_gdn_chain(MetalCtx *ctx, id<MTLCommandBuffer> cmdbuf,
                                 int linear_layer_idx, LayerWeightCache *lc) {
    if (ctx->fused_gdn_full) {
        uint32_t conv_dim = LINEAR_CONV_DIM;
        NSUInteger conv_w_off   = (NSUInteger)((const char *)lc->conv1d_w   - (const char *)[ctx->wf_buf contents]);
        NSUInteger a_log_off    = (NSUInteger)((const char *)lc->A_log      - (const char *)[ctx->wf_buf contents]);
        NSUInteger dt_bias_off  = (NSUInteger)((const char *)lc->dt_bias    - (const char *)[ctx->wf_buf contents]);
        NSUInteger gnorm_w_off  = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[ctx->wf_buf contents]);
        uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
        uint32_t kdim = LINEAR_KEY_DIM, vdim = LINEAR_VALUE_DIM;
        float eps = RMS_NORM_EPS;
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->fused_gdn_full];
        [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
        [enc setBuffer:ctx->batch_out[0]  offset:0          atIndex:1];  // qkv in
        [enc setBuffer:ctx->wf_buf        offset:conv_w_off  atIndex:2];  // conv weights
        [enc setBuffer:ctx->batch_out[1]  offset:0          atIndex:3];  // z
        [enc setBuffer:ctx->batch_out[3]  offset:0          atIndex:4];  // alpha
        [enc setBuffer:ctx->batch_out[2]  offset:0          atIndex:5];  // beta
        [enc setBuffer:ctx->wf_buf        offset:a_log_off   atIndex:6];
        [enc setBuffer:ctx->wf_buf        offset:dt_bias_off atIndex:7];
        [enc setBuffer:ctx->wf_buf        offset:gnorm_w_off atIndex:8];
        [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:9];
        [enc setBuffer:ctx->batch_out[6]  offset:0          atIndex:10]; // gated output
        [enc setBuffer:ctx->buf_conv_qk[linear_layer_idx] offset:0 atIndex:11]; // per-head q/k histories
        [enc setBytes:&conv_dim length:4 atIndex:12];
        [enc setBytes:&khpv     length:4 atIndex:13];
        [enc setBytes:&kdim     length:4 atIndex:14];
        [enc setBytes:&vdim     length:4 atIndex:15];
        [enc setBytes:&eps      length:4 atIndex:16];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
        [enc endEncoding];
        return;
    }

    // ---- Fallback: original chain (conv1d + qk-norm + GDN + gated) ----
    uint32_t conv_dim = LINEAR_CONV_DIM;
    NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[ctx->wf_buf contents]);
    {
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->conv1d_step];
        [enc setBuffer:ctx->buf_conv_state[linear_layer_idx] offset:0 atIndex:0];
        [enc setBuffer:ctx->batch_out[0]    offset:0            atIndex:1];
        [enc setBuffer:ctx->wf_buf          offset:conv_w_off   atIndex:2];
        [enc setBuffer:ctx->buf_conv_output offset:0            atIndex:3];
        [enc setBytes:&conv_dim length:4 atIndex:4];
        uint32_t tgs = (conv_dim + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    {
        uint32_t key_dim = LINEAR_KEY_DIM;
        float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->rms_norm_qk];
        [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:1];
        [enc setBytes:&key_dim   length:4 atIndex:2];
        [enc setBytes:&inv_scale length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_K_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LINEAR_KEY_DIM, 1, 1)];
        [enc endEncoding];
    }
    if (ctx->fused_gdn_core) {
        NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[ctx->wf_buf contents]);
        NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[ctx->wf_buf contents]);
        NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[ctx->wf_buf contents]);
        uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
        uint32_t kdim = LINEAR_KEY_DIM, vdim = LINEAR_VALUE_DIM;
        float eps = RMS_NORM_EPS;
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->fused_gdn_core];
        [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:2];
        [enc setBuffer:ctx->buf_conv_output offset:2 * LINEAR_TOTAL_KEY * sizeof(float) atIndex:3];
        [enc setBuffer:ctx->batch_out[1]       offset:0          atIndex:4];
        [enc setBuffer:ctx->batch_out[3]       offset:0          atIndex:5];
        [enc setBuffer:ctx->batch_out[2]       offset:0          atIndex:6];
        [enc setBuffer:ctx->wf_buf             offset:a_log_off  atIndex:7];
        [enc setBuffer:ctx->wf_buf             offset:dt_bias_off atIndex:8];
        [enc setBuffer:ctx->wf_buf             offset:gnorm_w_off atIndex:9];
        [enc setBuffer:ctx->batch_out[6]       offset:0          atIndex:10];
        [enc setBytes:&khpv length:sizeof(khpv) atIndex:11];
        [enc setBytes:&kdim length:sizeof(kdim) atIndex:12];
        [enc setBytes:&vdim length:sizeof(vdim) atIndex:13];
        [enc setBytes:&eps  length:sizeof(eps)  atIndex:14];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
        [enc endEncoding];
    } else {
        NSUInteger a_log_off   = (NSUInteger)((const char *)lc->A_log   - (const char *)[ctx->wf_buf contents]);
        NSUInteger dt_bias_off = (NSUInteger)((const char *)lc->dt_bias  - (const char *)[ctx->wf_buf contents]);
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->compute_decay_beta];
        [enc setBuffer:ctx->batch_out[3]       offset:0          atIndex:0];
        [enc setBuffer:ctx->batch_out[2]       offset:0          atIndex:1];
        [enc setBuffer:ctx->wf_buf             offset:a_log_off  atIndex:2];
        [enc setBuffer:ctx->wf_buf             offset:dt_bias_off atIndex:3];
        [enc setBuffer:ctx->buf_delta_g_decay  offset:0          atIndex:4];
        [enc setBuffer:ctx->buf_delta_beta     offset:0          atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)];
        [enc endEncoding];

        uint32_t khpv = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;
        enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->delta_net_step];
        [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_conv_output offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_conv_output offset:LINEAR_TOTAL_KEY * sizeof(float) atIndex:2];
        [enc setBuffer:ctx->buf_conv_output offset:2 * LINEAR_TOTAL_KEY * sizeof(float) atIndex:3];
        [enc setBuffer:ctx->buf_delta_g_decay offset:0 atIndex:4];
        [enc setBuffer:ctx->buf_delta_beta    offset:0 atIndex:5];
        [enc setBuffer:ctx->buf_delta_output  offset:0 atIndex:6];
        [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];

        NSUInteger gnorm_w_off = (NSUInteger)((const char *)lc->gated_norm_w - (const char *)[ctx->wf_buf contents]);
        uint32_t value_dim = LINEAR_VALUE_DIM;
        float eps = RMS_NORM_EPS;
        enc = [cmdbuf computeCommandEncoder];
        [enc setComputePipelineState:ctx->gated_rms_norm];
        [enc setBuffer:ctx->buf_delta_output offset:0          atIndex:0];
        [enc setBuffer:ctx->batch_out[1]     offset:0          atIndex:1];
        [enc setBuffer:ctx->wf_buf           offset:gnorm_w_off atIndex:2];
        [enc setBuffer:ctx->batch_out[6]     offset:0          atIndex:3];
        [enc setBytes:&value_dim length:4 atIndex:4];
        [enc setBytes:&eps       length:4 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(LINEAR_VALUE_DIM, 1, 1)];
        [enc endEncoding];
    }
}

// CPU linear-attention chain for one position: conv1d + q/k RMSNorm +
// GDN recurrence + RMSNormGated. Inputs are the CMD1 projection outputs
// (static scratch). Updates la_state->conv_state (CPU) and, when the GPU
// recurrence pipeline exists, buf_delta_state[linear_layer_idx] in place.
// Returns s_gated_out (static scratch). Shared by the per-token path and
// the chunked GGUF driver's CPU fallback.
static float *linear_attn_chain_cpu(LayerWeightCache *lc, LinearAttnState *la_state,
                                    int layer_idx,
                                    const float *qkv_out, const float *z_out,
                                    const float *beta_out, const float *alpha_out) {
    int qkv_dim = LINEAR_CONV_DIM;

    // Conv1d step
    uint16_t *conv_w = lc->conv1d_w;
    float *conv_out = s_conv_out;
    memset(conv_out, 0, qkv_dim * sizeof(float));
    if (conv_w) {
        cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                        qkv_dim, CONV_KERNEL_SIZE);
    }
    // Update conv state
    memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
            (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
    memcpy(la_state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv_out,
           qkv_dim * sizeof(float));

    // TEMP DEBUG: per-token conv state dump for layer 2
    if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
        static FILE *cv2 = NULL;
        if (!cv2) cv2 = fopen("/tmp/conv2_ref.bin", "wb");
        if (cv2) {
            fwrite(la_state->conv_state, sizeof(float), 3 * LINEAR_CONV_DIM, cv2);
            fflush(cv2);
        }
    }

    // Split into q, k, v
    float *lin_q = conv_out;
    float *lin_k = conv_out + LINEAR_TOTAL_KEY;
    float *lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;

    // RMS normalize q and k
    float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *qh = lin_q + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
        float q_scale = inv_scale * inv_scale;
        for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
    }
    for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
        float *kh = lin_k + h * LINEAR_KEY_DIM;
        cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
        for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
    }

    // Gated delta net recurrence
    float *A_log = lc->A_log;
    uint16_t *dt_bias_bf16 = lc->dt_bias;

    float *out_values = s_out_vals;
    memset(out_values, 0, LINEAR_TOTAL_VALUE * sizeof(float));
    int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;

    float g_decay[LINEAR_NUM_V_HEADS];
    float beta_gate_arr[LINEAR_NUM_V_HEADS];
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float a_val = alpha_out[vh];
        float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
        // GGUF mode: ssm_a is stored already negated+exponentiated
        // (-exp(logA)); llama.cpp multiplies it directly
        // (qwen35moe.cpp: gate = ssm_a * softplus). The packed files
        // store the raw logA, which the engine exponentiates.
        float A_val = A_log ? (g_gguf_stage ? -A_log[vh] : expf(A_log[vh])) : 1.0f;
        float softplus_val = logf(1.0f + expf(a_val + dt_b));
        g_decay[vh] = expf(-A_val * softplus_val);
        beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
    }

    // Compute linear_layer_idx: count of non-full-attention layers before this one.
    // Full attention at (layer_idx+1) % 4 == 0, i.e. layers 3,7,11,...
    // linear_layer_idx = layer_idx - number_of_full_layers_at_or_before
    //                  = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL
    int linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;

    // GPU delta-net path (falls back to CPU if pipeline unavailable)
    if (g_metal && g_metal->delta_net_step &&
        linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS) {
        // Upload CPU-computed data to GPU scratch buffers
        memcpy([g_metal->buf_delta_q contents], lin_q, LINEAR_TOTAL_KEY * sizeof(float));
        memcpy([g_metal->buf_delta_k contents], lin_k, LINEAR_TOTAL_KEY * sizeof(float));
        memcpy([g_metal->buf_delta_v contents], lin_v, LINEAR_TOTAL_VALUE * sizeof(float));
        memcpy([g_metal->buf_delta_g_decay contents], g_decay, LINEAR_NUM_V_HEADS * sizeof(float));
        memcpy([g_metal->buf_delta_beta contents], beta_gate_arr, LINEAR_NUM_V_HEADS * sizeof(float));

        id<MTLCommandBuffer> cmd_dn = [g_metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
        [enc setComputePipelineState:g_metal->delta_net_step];
        [enc setBuffer:g_metal->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
        [enc setBuffer:g_metal->buf_delta_q       offset:0 atIndex:1];
        [enc setBuffer:g_metal->buf_delta_k       offset:0 atIndex:2];
        [enc setBuffer:g_metal->buf_delta_v       offset:0 atIndex:3];
        [enc setBuffer:g_metal->buf_delta_g_decay offset:0 atIndex:4];
        [enc setBuffer:g_metal->buf_delta_beta    offset:0 atIndex:5];
        [enc setBuffer:g_metal->buf_delta_output  offset:0 atIndex:6];
        uint32_t khpv = (uint32_t)k_heads_per_v;
        [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cmd_dn commit];
        [cmd_dn waitUntilCompleted];

        // Read back GPU result
        memcpy(out_values, [g_metal->buf_delta_output contents], LINEAR_TOTAL_VALUE * sizeof(float));

        // TEMP DEBUG: per-token state dump for layer 2 (cross-path state check)
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            static FILE *st2 = NULL;
            if (!st2) st2 = fopen("/tmp/state2_ref.bin", "wb");
            if (st2) {
                fwrite([g_metal->buf_delta_state[linear_layer_idx] contents], sizeof(float), 32*128*128, st2);
                fflush(st2);
            }
        }
    } else {
        // CPU delta-net with Accelerate BLAS
        for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
            int kh = vh % LINEAR_NUM_K_HEADS;  // torch .repeat() block mapping (llama.cpp)
            float g = g_decay[vh];
            float b_gate = beta_gate_arr[vh];
            float *S = la_state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
            float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
            float *k_h = lin_k + kh * LINEAR_KEY_DIM;

            // Step 1: Decay S *= g (BLAS sscal on entire state matrix)
            cblas_sscal(LINEAR_VALUE_DIM * LINEAR_KEY_DIM, g, S, 1);

            // Step 2: kv_mem = S @ k (each row dot k)
            // S is [VALUE_DIM x KEY_DIM] row-major, k is [KEY_DIM]
            // kv_mem[vi] = sum_ki(S[vi,ki] * k[ki]) = matrix-vector: S @ k
            float kv_mem_vec[LINEAR_VALUE_DIM];
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                        1.0f, S, LINEAR_KEY_DIM, k_h, 1,
                        0.0f, kv_mem_vec, 1);

            // Step 3: delta = (v - kv_mem) * beta, then rank-1 update S += k * delta^T
            // delta[vi] = (v[vi] - kv_mem[vi]) * beta
            float delta_vec[LINEAR_VALUE_DIM];
            for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
                delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
            }
            // S += delta @ k^T (rank-1 update: sger)
            // S[vi,ki] += delta[vi] * k[ki]
            cblas_sger(CblasRowMajor, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                       1.0f, delta_vec, 1, k_h, 1, S, LINEAR_KEY_DIM);

            // Step 4: output = S @ q (matrix-vector multiply)
            float *q_h = lin_q + kh * LINEAR_KEY_DIM;
            float *o_h = out_values + vh * LINEAR_VALUE_DIM;
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                        1.0f, S, LINEAR_KEY_DIM, q_h, 1,
                        0.0f, o_h, 1);
        }
    }

    // RMSNormGated
    uint16_t *gated_norm_w = lc->gated_norm_w;
    float *gated_out = s_gated_out;
    memset(gated_out, 0, LINEAR_TOTAL_VALUE * sizeof(float));
    for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
        float *oh = out_values + vh * LINEAR_VALUE_DIM;
        float *zh = z_out + vh * LINEAR_VALUE_DIM;
        float *gh = gated_out + vh * LINEAR_VALUE_DIM;
        if (gated_norm_w) {
            cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
        } else {
            memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
        }
    }

    if (g_debug_layers) {
        debug_print_hidden("delta-out", layer_idx, out_values, LINEAR_TOTAL_VALUE);
        debug_print_hidden("gated-out", layer_idx, gated_out, LINEAR_TOTAL_VALUE);
    }

    return gated_out;
}

static void fused_layer_forward(
    WeightFile *wf,
    int layer_idx,
    float *hidden,           // [HIDDEN_DIM] in/out
    KVCache *kv,             // non-NULL for full attention layers
    LinearAttnState *la_state, // non-NULL for linear attention layers
    int pos,                 // position for RoPE
    const void *mmap_base,   // mmap'd layer file (NULL if not available)
    int K,                   // number of active experts
    int packed_fd            // fd for packed expert file
) {
    double t_layer_start = 0, t0 = 0, t1 = 0;
    if (g_timing_enabled) { t_layer_start = now_ms(); }
    int pred_started = 0;  // set to 1 if we started prediction preads during CMD1_wait
    // CMD1+CMD2 fusion: for gpu_linear layers, CMD2's encoders (o_proj +
    // residual_add + norm + routing) are appended to the CMD1 command buffer,
    // giving ONE commit+wait round trip per layer instead of two.
    id<MTLCommandBuffer> cmd12 = nil;   // the fused buffer (== cmd1 when fused)
    int cmd12_fused = 0;

    init_layer_scratch();
    if (!layer_cache_built) build_layer_cache(wf);
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);

    debug_print_hidden("input", layer_idx, hidden, HIDDEN_DIM);
    if (g_debug_layers && layer_idx <= 2) {
        fprintf(stderr, "[LC-DBG] L%d qkv_w=%p z_w=%p b_w=%p a_w=%p (is_full=%d)\n",
                layer_idx, (void*)lc->qkv_w, (void*)lc->z_w, (void*)lc->b_w, (void*)lc->a_w, is_full);
    }

    // =====================================================================
    // PHASE 1: Deferred completion + CMD1 (attention projections)
    // =====================================================================

    // ---- Prepare attention projection specs (doesn't depend on hidden) ----
    int num_attn_specs = 0;
    BatchMatvecSpec attn_specs[5];
    float *q_proj_out = NULL, *k_out = NULL, *v_out = NULL;
    float *qkv_out = NULL, *z_out = NULL, *beta_out = NULL, *alpha_out = NULL;

    if (is_full) {
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;

        q_proj_out = s_q_proj_out;
        k_out = s_k_proj_out;
        v_out = s_v_proj_out;

        if (lc->q_w && lc->k_w && lc->v_w /* BF16: scales may be NULL */) {
            attn_specs[0] = (BatchMatvecSpec){ lc->q_w, lc->q_s, lc->q_b, q_proj_out, (uint32_t)q_proj_dim, HIDDEN_DIM, GROUP_SIZE, 0, lc->q_bits };
            attn_specs[1] = (BatchMatvecSpec){ lc->k_w, lc->k_s, lc->k_b, k_out,      (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 1, lc->k_bits };
            attn_specs[2] = (BatchMatvecSpec){ lc->v_w, lc->v_s, lc->v_b, v_out,      (uint32_t)kv_dim,     HIDDEN_DIM, GROUP_SIZE, 2, lc->v_bits };
            num_attn_specs = 3;
        }
    } else {
        int qkv_dim = LINEAR_CONV_DIM;
        int z_dim = LINEAR_TOTAL_VALUE;

        qkv_out = s_qkv_proj_out;
        z_out = s_z_proj_out;
        beta_out = s_beta_proj_out;
        alpha_out = s_alpha_proj_out;

        if (lc->qkv_w && lc->z_w && lc->b_w && lc->a_w /* BF16: scales may be NULL */) {
            attn_specs[0] = (BatchMatvecSpec){ lc->qkv_w, lc->qkv_s, lc->qkv_b, qkv_out,   (uint32_t)qkv_dim,            HIDDEN_DIM, GROUP_SIZE, 0, lc->qkv_bits };
            attn_specs[1] = (BatchMatvecSpec){ lc->z_w,   lc->z_s,   lc->z_b,   z_out,      (uint32_t)z_dim,              HIDDEN_DIM, GROUP_SIZE, 1, lc->z_bits };
            attn_specs[2] = (BatchMatvecSpec){ lc->b_w,   lc->b_s,   lc->b_b,   beta_out,   (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 2, lc->b_bits };
            attn_specs[3] = (BatchMatvecSpec){ lc->a_w,   lc->a_s,   lc->a_b,   alpha_out,  (uint32_t)LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, 3, lc->a_bits };
            num_attn_specs = 4;
        }
    }

    // ---- Deferred completion + CMD1 (sequential) ----
    float *normed = s_normed;
    float *residual = s_residual;
    id<MTLCommandBuffer> cmd1 = nil;
    int gpu_linear_attn = 0;  // set to 1 if GPU handles entire linear attention pipeline

    // Pre-compute linear_layer_idx for GPU linear attention encoding in CMD1
    int linear_layer_idx = -1;
    if (!is_full) {
        linear_layer_idx = layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;
    }
    // Can we run the full linear attention pipeline on GPU in CMD1?
    int can_gpu_linear = (gpu_linear_attn_enabled &&
                          !is_full && g_metal && g_metal->delta_net_step &&
                          g_metal->conv1d_step && g_metal->rms_norm_qk &&
                          g_metal->compute_decay_beta && g_metal->gated_rms_norm &&
                          g_metal->wf_buf &&
                          // Phase C: the GDN chain reads staged BF16 tensors
                          // (conv1d_w, A_log, dt_bias, gated_norm_w) whose
                          // wf_buf offsets would be garbage in GGUF mode —
                          // the chain stays CPU there.
                          !g_gguf_stage &&
                          linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS &&
                          lc->conv1d_w && lc->A_log && lc->dt_bias && lc->gated_norm_w &&
                          !linear_attn_bypass);

    // Check if previous layer's CMD3 already computed combine+residual+norm on GPU.
    // If so, buf_input already contains the normalized input for this layer's CMD1.
    // We can submit CMD1 immediately — the GPU queue serializes CMD3(N-1) then CMD1(N).
    int prev_gpu_combined = (g_deferred.active && g_deferred.gpu_combined);

    if (prev_gpu_combined && g_metal && g_metal->wf_buf && num_attn_specs > 0) {
        // ---- FAST PATH: GPU-combined previous CMD3 ----
        // buf_input already has the normalized hidden state from CMD3(N-1).
        // Submit CMD1 immediately — GPU runs CMD3(N-1) then CMD1(N) back-to-back.
        if (g_timing_enabled) { t0 = now_ms(); }

        cmd1 = [g_metal->queue commandBuffer];
        gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

        // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
        if (can_gpu_linear && num_attn_specs == 4) {
            // batch_out[0]=qkv(12288), [1]=z(8192), [2]=beta(64), [3]=alpha(64)
            uint32_t conv_dim = LINEAR_CONV_DIM;
            NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);

            gpu_encode_gdn_chain(g_metal, cmd1, linear_layer_idx, lc);
            gpu_linear_attn = 1;
        }

        if (gpu_linear_attn && cmd1) {
            // CMD1+CMD2 fusion: defer commit — CMD2's encoders append below.
            cmd12 = cmd1;
            cmd12_fused = 1;
        } else {
            [cmd1 commit];
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (implies CMD3(N-1) also done, since queue is serial)
        if (g_timing_enabled) { t0 = now_ms(); }
        if (!cmd12_fused) {
            [cmd1 waitUntilCompleted];
            if (getenv("FINCHMOE_PF_DUMP") && pos == 0 && layer_idx == 0) {
                static FILE *pf_after1 = NULL;
                if (!pf_after1) pf_after1 = fopen("/tmp/bufinput_after_cmd1.bin", "wb");
                if (pf_after1) {
                    fwrite([g_metal->buf_input contents], sizeof(float), HIDDEN_DIM, pf_after1);
                    fflush(pf_after1);
                }
            }
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }

        // Debug: check fast-path buffers for early layers
        if (!cmd12_fused && g_debug_layers && layer_idx <= 2) {
            float *fq = (float *)[g_metal->batch_out[0] contents];
            float *fg = (float *)[g_metal->batch_out[6] contents];
            float *fi = (float *)[g_metal->buf_input contents];
            fprintf(stderr, "[FAST-DBG] L%d batch_out[0] first4: [%.4f %.4f %.4f %.4f] batch_out[6] first4: [%.4f %.4f %.4f %.4f] buf_input first4: [%.4f %.4f %.4f %.4f]\n",
                    layer_idx, fq[0], fq[1], fq[2], fq[3], fg[0], fg[1], fg[2], fg[3], fi[0], fi[1], fi[2], fi[3]);
        }

        // Now CMD3(N-1) is done. Read back hidden state from GPU.
        if (g_timing_enabled) { t0 = now_ms(); }
        if (!cmd12_fused) {
            finalize_deferred_experts();  // reads buf_moe_hidden -> hidden

            // Start predicted expert preads AFTER CMD1_wait.
            // CMD3(N-1) is guaranteed done (serial queue), so buf_B is safe to overwrite.
            // Predictions overlap with CPU attn + CMD2 + routing (~0.6ms head start).
            // Predicted experts that hit page cache (same as previous token) complete in ~0.1ms.
            if (g_pred_enabled && g_pred_generating && g_pred_valid && packed_fd >= 0 &&
                g_metal->buf_multi_expert_data_B[0] && g_pred_count[layer_idx] > 0) {
                async_pread_start(packed_fd, g_pred_experts[layer_idx],
                                  g_pred_count[layer_idx],
                                  g_metal->buf_multi_expert_data_B, mmap_base);
                pred_started = 1;
            }
            // Set up residual for CMD2 (residual = hidden before this layer's attention)
            cpu_vec_copy(residual, hidden, HIDDEN_DIM);
        }
        // When fused (cmd12_fused): finalize + residual happen AFTER the single
        // commit+wait (in the CMD2 block); the GPU residual_add reads
        // buf_moe_hidden directly (== hidden before attention) instead of
        // buf_residual, avoiding the CPU round trip.
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // No input_norm needed — CMD3 already computed it into buf_input.
        // normed is only needed if speculative routing is enabled (currently disabled).
        // Skip the readback to avoid unnecessary overhead.
    } else {
        // ---- ORIGINAL PATH: CPU deferred completion + input norm ----
        // Complete deferred experts from previous layer
        if (g_timing_enabled) { t0 = now_ms(); }
        wait_deferred_experts_gpu();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_wait += t1 - t0; }

        if (g_timing_enabled) { t0 = now_ms(); }
        finalize_deferred_experts();
        if (g_timing_enabled) { t1 = now_ms(); g_timing.deferred_cpu += t1 - t0; }

        // Input norm
        if (g_timing_enabled) { t0 = now_ms(); }
        cpu_vec_copy(residual, hidden, HIDDEN_DIM);
        cpu_rms_norm(hidden, lc->input_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
        debug_print_hidden("post-norm", layer_idx, normed, HIDDEN_DIM);
        if (g_timing_enabled) { t1 = now_ms(); g_timing.input_norm += t1 - t0; }

        // Submit CMD1: attention projections
        if (g_timing_enabled) { t0 = now_ms(); }
        // Phase C: GGUF mode now uses the GPU batch matvec too (bits 10/11
        // encode via per-tensor wrapped buffers) — the GDN chain below stays
        // CPU (can_gpu_linear is gated on !g_gguf_stage).
        if (g_metal && (g_metal->wf_buf || g_gguf_stage) && num_attn_specs > 0) {
            if (getenv("FINCHMOE_PF_DUMP") && pos == 1 && layer_idx == 0) {
                static FILE *pf_st = NULL;
                if (!pf_st) pf_st = fopen("/tmp/state_per_token.bin", "wb");
                if (pf_st) {
                    fwrite([g_metal->buf_conv_state[0] contents], sizeof(float), 3*LINEAR_CONV_DIM, pf_st);
                    fwrite([g_metal->buf_delta_state[0] contents], sizeof(float), 32*128*128, pf_st);
                    fflush(pf_st);
                }
            }
            memcpy([g_metal->buf_input contents], normed, HIDDEN_DIM * sizeof(float));
            if (getenv("FINCHMOE_PF_DUMP") && pos <= 1 && layer_idx == 0) {
                static FILE *pf_early = NULL;
                if (!pf_early) pf_early = fopen("/tmp/bufinput_early.bin", "wb");
                if (pf_early) {
                    fwrite(normed, sizeof(float), HIDDEN_DIM, pf_early);
                    fflush(pf_early);
                }
            }
            cmd1 = [g_metal->queue commandBuffer];
            gpu_encode_batch_matvec(g_metal, cmd1, attn_specs, num_attn_specs);

            // GPU linear attention: encode conv1d + normalize + decay/beta + delta-net + gated_norm into CMD1
            if (can_gpu_linear && num_attn_specs == 4) {
                uint32_t conv_dim = LINEAR_CONV_DIM;
                NSUInteger conv_w_off = (NSUInteger)((const char *)lc->conv1d_w - (const char *)[g_metal->wf_buf contents]);

            gpu_encode_gdn_chain(g_metal, cmd1, linear_layer_idx, lc);
                gpu_linear_attn = 1;
            }

            if (gpu_linear_attn && cmd1) {
                // CMD1+CMD2 fusion: defer commit — CMD2's encoders append below.
                cmd12 = cmd1;
                cmd12_fused = 1;
            } else {
                [cmd1 commit];
            }
        } else {
            for (int i = 0; i < num_attn_specs; i++) {
                BatchMatvecSpec *s = &attn_specs[i];
                cpu_dequant_matvec(s->W, s->scales, s->biases, normed, s->out_cpu,
                                   s->out_dim, s->in_dim, s->group_size, s->bits);
            }
            if (g_debug_layers && !is_full && num_attn_specs == 4) {
                debug_print_hidden("qkv-proj", layer_idx, qkv_out, LINEAR_CONV_DIM);
                debug_print_hidden("z-proj", layer_idx, z_out, LINEAR_TOTAL_VALUE);
                debug_print_hidden("beta-proj", layer_idx, beta_out, LINEAR_NUM_V_HEADS);
                debug_print_hidden("alpha-proj", layer_idx, alpha_out, LINEAR_NUM_V_HEADS);
            }
            if (g_debug_layers && is_full && num_attn_specs >= 3) {
                debug_print_hidden("q-proj-fa", layer_idx, q_proj_out, NUM_ATTN_HEADS * HEAD_DIM * 2);
                debug_print_hidden("k-proj-fa", layer_idx, k_out, NUM_KV_HEADS * HEAD_DIM);
                debug_print_hidden("v-proj-fa", layer_idx, v_out, NUM_KV_HEADS * HEAD_DIM);
            }
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_submit += t1 - t0; }

        // Wait for CMD1 (skipped when fused — the single wait is in the CMD2 block)
        if (g_timing_enabled) { t0 = now_ms(); }
        if (cmd1 && !cmd12_fused) {
            [cmd1 waitUntilCompleted];
            if (getenv("FINCHMOE_PF_DUMP") && pos == 0 && layer_idx == 0) {
                static FILE *pf_after1 = NULL;
                if (!pf_after1) pf_after1 = fopen("/tmp/bufinput_after_cmd1.bin", "wb");
                if (pf_after1) {
                    fwrite([g_metal->buf_input contents], sizeof(float), HIDDEN_DIM, pf_after1);
                    fflush(pf_after1);
                }
            }
            if (!gpu_linear_attn) {
                gpu_flush_batch_results(g_metal, attn_specs, num_attn_specs);
            }
        }
        // Stage dump for layer-0 cross-validation (env-gated)
        // NOTE: when fused, the dump happens after the CMD2-block wait instead.
        if (!cmd12_fused && getenv("FINCHMOE_DUMP_STAGES") && layer_idx == 0 && pos == 0) {
            static FILE *sf1 = NULL;
            if (!sf1) sf1 = fopen("/tmp/stage_dump.bin", "wb");
            if (sf1) {
                fwrite([g_metal->buf_input contents], sizeof(float), HIDDEN_DIM, sf1);            // input 2048
                fwrite([g_metal->batch_out[0] contents], sizeof(float), LINEAR_CONV_DIM, sf1);    // qkv 8192
                fwrite([g_metal->batch_out[1] contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1); // z 4096
                fwrite([g_metal->batch_out[2] contents], sizeof(float), LINEAR_NUM_V_HEADS, sf1); // beta 32
                fwrite([g_metal->batch_out[3] contents], sizeof(float), LINEAR_NUM_V_HEADS, sf1); // alpha 32
                fwrite([g_metal->buf_conv_output contents], sizeof(float), LINEAR_CONV_DIM, sf1); // conv 8192
                fwrite([g_metal->buf_delta_output contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1); // delta out 4096
                fwrite([g_metal->batch_out[6] contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1); // gated 4096
                fflush(sf1);
            }
        }
        // Debug: check projection outputs after CMD1 completes
        if (g_debug_layers && !is_full && num_attn_specs == 4) {
            if (layer_idx <= 2) {
                BatchMatvecSpec *s0 = &attn_specs[0];
                NSUInteger w_off = (NSUInteger)((const char *)s0->W - (const char *)[g_metal->wf_buf contents]);
                NSUInteger s_off = (NSUInteger)((const char *)s0->scales - (const char *)[g_metal->wf_buf contents]);
                NSUInteger b_off = (NSUInteger)((const char *)s0->biases - (const char *)[g_metal->wf_buf contents]);
                const uint32_t *wptr = (const uint32_t *)[g_metal->wf_buf contents];
                const uint16_t *sptr = (const uint16_t *)[g_metal->wf_buf contents];
                fprintf(stderr, "[ATTN-DBG] L%d spec0: W=%p scales=%p biases=%p w_off=%lu s_off=%lu b_off=%lu\n",
                        layer_idx, (void*)s0->W, (void*)s0->scales, (void*)s0->biases,
                        (unsigned long)w_off, (unsigned long)s_off, (unsigned long)b_off);
                fprintf(stderr, "[ATTN-DBG] L%d spec0: W[0]=0x%08x S[0]=%.6f B[0]=%.6f\n",
                        layer_idx, wptr[w_off/4],
                        s0->scales ? bf16_to_f32(sptr[s_off/2]) : 0.0f,
                        s0->biases ? bf16_to_f32(sptr[b_off/2]) : 0.0f);
                float *dbg = (float *)[g_metal->batch_out[0] contents];
                fprintf(stderr, "[ATTN-DBG] L%d batch_out[0] first8: [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f]\n",
                        layer_idx, dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], dbg[6], dbg[7]);
                fprintf(stderr, "[ATTN-DBG] L%d gpu_linear_attn=%d qkv_out first8: [%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f] qkv_out=%p\n",
                        layer_idx, gpu_linear_attn, qkv_out[0], qkv_out[1], qkv_out[2], qkv_out[3],
                        qkv_out[4], qkv_out[5], qkv_out[6], qkv_out[7], (void*)qkv_out);
                fprintf(stderr, "[ATTN-DBG] L%d batch_out ptr=%p qkv_out ptr=%p\n",
                        layer_idx, (void*)[g_metal->batch_out[0] contents], (void*)qkv_out);
            }
        }
        // Debug: check projection outputs after CMD1 completes
        if (g_debug_layers && !is_full && num_attn_specs == 4) {
            debug_print_hidden("qkv-proj", layer_idx, qkv_out, LINEAR_CONV_DIM);
            debug_print_hidden("z-proj", layer_idx, z_out, LINEAR_TOTAL_VALUE);
            debug_print_hidden("beta-proj", layer_idx, beta_out, LINEAR_NUM_V_HEADS);
            debug_print_hidden("alpha-proj", layer_idx, alpha_out, LINEAR_NUM_V_HEADS);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd1_wait += t1 - t0; }
    }

    // =====================================================================
    // SPECULATIVE EARLY ROUTING — overlap expert I/O with CPU attention
    // =====================================================================
    // Compute approximate routing using the PRE-attention normed hidden state.
    // The real routing (in CMD2/PHASE 3) uses the POST-attention state, so this
    // is an approximation. Fire off async pread for predicted cache misses via
    // dispatch_group so the I/O runs concurrently with CPU attention compute.
    // After CPU attention, we wait for the group to finish. When the real routing
    // happens later, predicted experts are already in the LRU cache as hits.

    dispatch_group_t spec_group = NULL;
    int spec_preload_count = 0;
    int spec_routing_enabled = 0;  // DISABLED: cache pollution + overhead makes it slower

    if (g_timing_enabled) { t0 = now_ms(); }
    s_spec_count = 0;

    // Phase C S8: speculative-prefill accuracy probe (FINCHMOE_SPEC_PROBE=1).
    // Computes the spec top-K from the PRE-attention normed input and compares
    // it against the real post-gated-norm routing later in this call —
    // measures the ceiling for speculative expert preads in the chunked path.
    static int spec_probe = -1;
    if (spec_probe < 0) spec_probe = getenv("FINCHMOE_SPEC_PROBE") != NULL;
    if (spec_probe && packed_fd >= 0 && lc->gate_w && !g_gguf_stage) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, NUM_EXPERTS * sizeof(float));
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 8);
        cpu_softmax(spec_scores, NUM_EXPERTS);
        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, NUM_EXPERTS, spec_K, s_spec_indices, spec_weights);
        s_spec_count = spec_K;
        g_spec_route_attempts += spec_K;
    }

    if (spec_routing_enabled && (g_expert_cache || g_malloc_cache) && packed_fd >= 0 && lc->gate_w) {
        float *spec_scores = s_spec_gate_scores;
        memset(spec_scores, 0, NUM_EXPERTS * sizeof(float));

        // Gate projection matvec on pre-attention normed input (8-bit quantization)
        cpu_dequant_matvec(lc->gate_w, lc->gate_s, lc->gate_b,
                           normed, spec_scores,
                           NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 8);
        cpu_softmax(spec_scores, NUM_EXPERTS);

        int spec_K = (K > MAX_K) ? MAX_K : K;
        float spec_weights[MAX_K];
        cpu_topk(spec_scores, NUM_EXPERTS, spec_K, s_spec_indices, spec_weights);
        s_spec_count = spec_K;

        g_spec_route_attempts += spec_K;

        // Initialize GCD queue if needed
        if (!g_io_gcd_queue)
            g_io_gcd_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

        // Check cache for each predicted expert, start async I/O for misses
        size_t spec_esz = active_expert_size();
        if (g_malloc_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, eidx);
                if (!cached) {
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, eidx, &cidx);
                    if (buf && cidx >= 0) {
                        int fd_copy = packed_fd;
                        void *dst = g_malloc_cache->data[cidx];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        } else if (g_expert_cache) {
            spec_group = dispatch_group_create();
            for (int k = 0; k < spec_K; k++) {
                int eidx = s_spec_indices[k];
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, eidx);
                if (!cached) {
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, eidx);
                    if (buf) {
                        int fd_copy = packed_fd;
                        void *dst = [buf contents];
                        off_t offset = (off_t)eidx * spec_esz;
                        size_t sz = spec_esz;
                        dispatch_group_async(spec_group, g_io_gcd_queue, ^{
                            pread(fd_copy, dst, sz, offset);
                        });
                        spec_preload_count++;
                        g_spec_route_preloads++;
                    }
                }
            }
        }
    }
    (void)spec_preload_count;  // tracked via g_spec_route_preloads

    if (g_timing_enabled) { t1 = now_ms(); g_timing.spec_route += t1 - t0; }

    // =====================================================================
    // PHASE 2: CPU attention compute
    // =====================================================================

    if (g_timing_enabled) { t0 = now_ms(); }

    float *attn_projected = s_attn_proj;
    memset(attn_projected, 0, HIDDEN_DIM * sizeof(float));

    // Pre-lookup o_proj / out_proj weights (used after attention compute)
    // These are looked up NOW to avoid repeated snprintf later.
    uint32_t *oproj_w = NULL;
    uint16_t *oproj_s = NULL, *oproj_b = NULL;
    int oproj_in_dim = 0;
    int oproj_bits = 4;

    if (is_full) {
        oproj_w = lc->o_w; oproj_s = lc->o_s; oproj_b = lc->o_b;
        oproj_in_dim = NUM_ATTN_HEADS * HEAD_DIM;
        oproj_bits = lc->o_bits;
    } else if (!linear_attn_bypass) {
        oproj_w = lc->out_proj_w; oproj_s = lc->out_proj_s; oproj_b = lc->out_proj_b;
        oproj_in_dim = LINEAR_TOTAL_VALUE;
        oproj_bits = lc->out_proj_bits;
    }

    // All MoE weight pointers from cache (zero snprintf overhead)
    uint32_t *gate_w = lc->gate_w; uint16_t *gate_s = lc->gate_s, *gate_b = lc->gate_b;
    uint32_t *sgw = lc->sg_w;     uint16_t *sgs = lc->sg_s,       *sgb = lc->sg_b;
    uint32_t *suw = lc->su_w;     uint16_t *sus = lc->su_s,       *sub = lc->su_b;
    uint32_t *seg_w = lc->seg_w;  uint16_t *seg_s = lc->seg_s,   *seg_b = lc->seg_b;
    uint32_t *sdw = lc->sd_w;     uint16_t *sds = lc->sd_s,       *sdb = lc->sd_b;

    // ---- CPU attention compute (produces attn_out for o_proj) ----
    float *attn_out_for_oproj = NULL;

    if (is_full) {
        // ---- Full attention CPU compute ----
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
        int q_dim = NUM_ATTN_HEADS * HEAD_DIM;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;
        (void)q_proj_dim;

        float *q = s_q;
        float *q_gate = s_q_gate;
        for (int h = 0; h < NUM_ATTN_HEADS; h++) {
            float *src = q_proj_out + h * (2 * HEAD_DIM);
            memcpy(q + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
            memcpy(q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
        }

        // Q/K RMSNorm
        uint16_t *qnorm_w = lc->q_norm_w;
        uint16_t *knorm_w = lc->k_norm_w;
        if (qnorm_w) {
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                float *qh = q + h * HEAD_DIM;
                float sum_sq = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) sum_sq += qh[i] * qh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
                for (int i = 0; i < HEAD_DIM; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
            }
        }
        if (knorm_w) {
            for (int h = 0; h < NUM_KV_HEADS; h++) {
                float *kh = k_out + h * HEAD_DIM;
                float sum_sq = 0.0f;
                for (int i = 0; i < HEAD_DIM; i++) sum_sq += kh[i] * kh[i];
                float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
                for (int i = 0; i < HEAD_DIM; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
            }
        }

        // RoPE
        apply_rotary_emb(q, k_out, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);

        // Update KV cache (CPU + GPU mirror)
        int cache_pos = kv->len;
        kv_write(kv, cache_pos, k_out, v_out);

        int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
        if (g_metal && g_metal->attn_scores_pipe && fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
            cache_pos < g_gpu_kv_seq) {  // GPU mirror holds only [0, g_gpu_kv_seq)
            memcpy((float *)[g_metal->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                   k_out, kv_dim * sizeof(float));
            memcpy((float *)[g_metal->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                   v_out, kv_dim * sizeof(float));
        }
        kv->len++;

        // Scaled dot-product attention (GQA) — GPU or CPU
        int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float *attn_out = s_attn_out;
        memset(attn_out, 0, q_dim * sizeof(float));

        // GPU attention: defer dispatches to CMD2 (fused into single cmd buffer).
        // Only enabled when seq_len >= 32 (below that, CPU is faster).
        // Phase C S6 fix: GGUF mode defaults to CPU attention here — the
        // attention dispatches live inside the fully-fused CMD2, which is
        // gated off for GGUF (g_gguf_stage); with attn_out_for_oproj=NULL
        // and no dispatches, o_proj silently read STALE buf_attn_out from
        // token 32 onward (the 90-token per-token divergence, cos 0.818).
        // Phase C S7 L3: FINCHMOE_GGUF_GPU_ATTN=1 re-enables GPU attention for
        // GGUF — the dispatches are encoded in a dedicated CB in the branch
        // below (the native fused CMD2 stays gated off). CPU attention is
        // O(seq_len), so this also fixes long-context decode scaling.
        static int gguf_gpu_attn = -1;
        if (gguf_gpu_attn < 0) gguf_gpu_attn = getenv("FINCHMOE_GGUF_GPU_ATTN") != NULL;
        int gguf_oproj_fallback = 0;   // set if the GGUF CB couldn't encode o_proj
        int gpu_attn_ready = (g_metal && g_metal->attn_scores_pipe &&
                              fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                              kv->len >= 32 && kv->len < g_gpu_kv_seq &&
                              (!g_gguf_stage || gguf_gpu_attn));

        if (gpu_attn_ready) {
            // Copy Q and gate to GPU; attention dispatches will be in CMD2
            memcpy([g_metal->buf_attn_q contents], q, q_dim * sizeof(float));
            memcpy([g_metal->buf_attn_gate contents], q_gate, q_dim * sizeof(float));
            // attn_out_for_oproj will be set to NULL below — CMD2 reads buf_attn_out
            if (g_gguf_stage) {
                // ---- Phase C S7 L3: GGUF attention + o_proj in ONE CB ----
                // Same A1-A4 dispatches as the native fused CMD2 (with the
                // S4.1 sync pattern) plus the Q4_K/Q6_K o_proj matvec
                // (matvec_qk via the reused tensor-wrap buffer — the same
                // encode as gpu_gguf_dequant_matvec). Replaces the O(seq_len)
                // CPU attention loop and the one-off o_proj CB with a single
                // commit+wait.
                uint32_t ghd = HEAD_DIM, gkvd = (uint32_t)kv_dim;
                uint32_t gsl = (uint32_t)kv->len;
                uint32_t gseq = (uint32_t)g_gpu_kv_seq;
                uint32_t ghpkv = (uint32_t)heads_per_kv;
                id<MTLCommandBuffer> cmd_attn = [g_metal->queue commandBuffer];
                // Enc A1: attn_scores_batched
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->attn_scores_pipe];
                    [enc setBuffer:g_metal->buf_attn_q        offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_k[fa_idx]  offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:2];
                    [enc setBytes:&ghd       length:4 atIndex:3];
                    [enc setBytes:&gkvd      length:4 atIndex:4];
                    [enc setBytes:&gsl       length:4 atIndex:5];
                    [enc setBytes:&gseq      length:4 atIndex:6];
                    [enc setBytes:&scale     length:4 atIndex:7];
                    [enc setBytes:&ghpkv     length:4 atIndex:8];
                    [enc setBytes:&gsl       length:4 atIndex:9];
                    uint32_t total_tgs = gsl * NUM_ATTN_HEADS;
                    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                // Enc A2: attn_softmax_batched (writer sync + reader barrier)
                metal_sync_buffer(cmd_attn, g_metal->buf_attn_scores);
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                    [enc setBytes:&gsl       length:4 atIndex:1];
                    [enc setBytes:&gseq      length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                // Enc A3: attn_values_batched
                metal_sync_buffer(cmd_attn, g_metal->buf_attn_scores);
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:g_metal->attn_values_pipe];
                    [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                    [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                    [enc setBytes:&ghd      length:4 atIndex:3];
                    [enc setBytes:&gkvd     length:4 atIndex:4];
                    [enc setBytes:&gsl      length:4 atIndex:5];
                    [enc setBytes:&gseq     length:4 atIndex:6];
                    [enc setBytes:&ghpkv    length:4 atIndex:7];
                    uint32_t total_threads = HEAD_DIM * NUM_ATTN_HEADS;
                    uint32_t tgs = (total_threads + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                // Enc A4: sigmoid_gate
                metal_sync_buffer(cmd_attn, g_metal->buf_attn_out);
                {
                    uint32_t gqd = NUM_ATTN_HEADS * HEAD_DIM;
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:g_metal->sigmoid_gate_pipe];
                    [enc setBuffer:g_metal->buf_attn_out  offset:0 atIndex:0];
                    [enc setBuffer:g_metal->buf_attn_gate offset:0 atIndex:1];
                    [enc setBytes:&gqd length:4 atIndex:2];
                    uint32_t tgs = (gqd + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                // Enc 5: o_proj (matvec_qk from the mmap slab). Only for
                // in_dim <= 4096 — the verified one-off GPU path has the same
                // limit; full-attn layers (8192) keep the CPU o_proj via the
                // fallback flag below.
                metal_sync_buffer(cmd_attn, g_metal->buf_attn_out);
                id<MTLBuffer> o_tbuf = NULL;
                uint32_t o_delta = 0;
                if (oproj_in_dim <= 4096) {
                    size_t o_row_bytes = (size_t)(oproj_in_dim / 256) *
                                         ((oproj_bits == 10) ? 144 : 210);
                    o_tbuf = gguf_tbuf_get(g_metal, oproj_w,
                                           o_row_bytes * HIDDEN_DIM, &o_delta);
                }
                if (o_tbuf) {
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc setComputePipelineState:g_metal->matvec_qk];
                    [enc setBuffer:o_tbuf offset:o_delta atIndex:0];
                    [enc setBuffer:g_metal->buf_attn_out offset:0 atIndex:3];
                    [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
                    uint32_t ood = HIDDEN_DIM, oid_ = (uint32_t)oproj_in_dim;
                    uint32_t ogt = (uint32_t)(oproj_bits == 10 ? 12 : 14);
                    [enc setBytes:&ood length:4 atIndex:5];
                    [enc setBytes:&oid_ length:4 atIndex:6];
                    [enc setBytes:&ogt length:4 atIndex:7];
                    [enc dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 7) / 8, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                [cmd_attn commit];
                [cmd_attn waitUntilCompleted];
                memcpy(attn_out, [g_metal->buf_attn_out contents], q_dim * sizeof(float));
                if (o_tbuf) {
                    memcpy(attn_projected, [g_metal->buf_output contents],
                           HIDDEN_DIM * sizeof(float));
                } else {
                    gguf_oproj_fallback = 1;   // wrap alloc failed — CPU o_proj below
                }
            }
        } else {
            // CPU fallback
            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                int kv_h = h / heads_per_kv;
                float *qh = q + h * HEAD_DIM;
                float *scores = malloc(kv->len * sizeof(float));
                static float kv_k_buf2[HEAD_DIM], kv_v_buf2[HEAD_DIM];
                for (int p = 0; p < kv->len; p++) {
                    kv_read_k(kv, p, kv_h, kv_k_buf2);
                    float dot = 0.0f;
                    for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kv_k_buf2[d];
                    scores[p] = dot * scale;
                }
                cpu_softmax(scores, kv->len);
                float *oh = attn_out + h * HEAD_DIM;
                for (int p = 0; p < kv->len; p++) {
                    kv_read_v(kv, p, kv_h, kv_v_buf2);
                    for (int d = 0; d < HEAD_DIM; d++) oh[d] += scores[p] * kv_v_buf2[d];
                }
                free(scores);
            }
            for (int i = 0; i < q_dim; i++) {
                float g = 1.0f / (1.0f + expf(-q_gate[i]));
                attn_out[i] *= g;
            }
        }

        if (gpu_attn_ready && !gguf_oproj_fallback) {
            attn_out_for_oproj = NULL;  // signal CMD2 to use GPU buf_attn_out
        } else {
            attn_out_for_oproj = attn_out;
        }
        // q_proj_out, k_out, v_out, q, q_gate, attn_out are static scratch.
    } else if (gpu_linear_attn) {
        // ---- GPU linear attention: already computed in CMD1 ----
        // batch_out[6] already contains gated_rms_norm output (8192 floats)
        // Set a non-NULL sentinel so CMD2 enters fused path, but skip the memcpy
        static float gpu_linear_sentinel;
        attn_out_for_oproj = &gpu_linear_sentinel;
    } else {
        // ---- Linear attention CPU compute ----
        if (!linear_attn_bypass) {
            // TEMP DEBUG: per-token qkv/z CPU cross-check (layer 2)
            if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2 && normed) {
                static FILE *ptq = NULL;
                if (!ptq) ptq = fopen("/tmp/qkv2_ref.bin", "wb");
                if (ptq) {
                    static float cpu_qkv[LINEAR_CONV_DIM], cpu_z[LINEAR_TOTAL_VALUE];
                    gguf_cpu_matvec(lc->qkv_w, normed, cpu_qkv, LINEAR_CONV_DIM, HIDDEN_DIM,
                                    lc->qkv_bits == 10 ? 12 : 14);
                    gguf_cpu_matvec(lc->z_w, normed, cpu_z, LINEAR_TOTAL_VALUE, HIDDEN_DIM,
                                    lc->z_bits == 10 ? 12 : 14);
                    fwrite(cpu_qkv, sizeof(float), LINEAR_CONV_DIM, ptq);
                    fwrite(qkv_out, sizeof(float), LINEAR_CONV_DIM, ptq);
                    fwrite(cpu_z, sizeof(float), LINEAR_TOTAL_VALUE, ptq);
                    fwrite(z_out, sizeof(float), LINEAR_TOTAL_VALUE, ptq);
                    fwrite(normed, sizeof(float), HIDDEN_DIM, ptq);   // the chain input
                    fflush(ptq);
                }
            }
            attn_out_for_oproj = linear_attn_chain_cpu(lc, la_state, layer_idx,
                                                       qkv_out, z_out, beta_out, alpha_out);
            // conv_out, out_values are static — no free needed
            // gated_out is static — freed/released after CMD2 submission below
        }
        // else: linear_attn_bypass — attn_projected stays zero
        // qkv_out, z_out, beta_out, alpha_out are static scratch.
    }

    // =====================================================================
    // PHASE 3: FULLY FUSED CMD2 — o_proj + residual + norm + routing (1 cmd buffer)
    //   Eliminates 1 GPU round-trip vs old 2-buffer approach.
    //   GPU handles residual_add + rms_norm between o_proj and routing,
    //   so no CPU intervention is needed. 8 encoders, 1 commit+wait.
    //   Buffer flow: batch_out[6]->buf_output->buf_h_mid->buf_input->batch_out[0-3]
    // =====================================================================

    if (g_timing_enabled) { t1 = now_ms(); g_timing.cpu_attn += t1 - t0; }

    // Wait for speculative expert I/O to complete (overlapped with CPU attention)
    if (spec_group) {
        dispatch_group_wait(spec_group, DISPATCH_TIME_FOREVER);
        spec_group = NULL;  // ARC releases the group
    }

    if (g_timing_enabled) { t0 = now_ms(); }

    float *h_post = s_h_post;
    float *h_mid = s_h_mid;
    float *gate_scores = s_gate_scores;
    memset(gate_scores, 0, NUM_EXPERTS * sizeof(float));
    float *shared_gate = s_shared_gate;
    memset(shared_gate, 0, SHARED_INTERMEDIATE * sizeof(float));
    float *shared_up = s_shared_up;
    memset(shared_up, 0, SHARED_INTERMEDIATE * sizeof(float));
    float shared_gate_score = 0.0f;

    int have_moe_weights = (gate_w && sgw && suw && seg_w) /* BF16: scales may be NULL */;

    // gpu_attn_fuse: attention dispatches fused into CMD2 (full-attn layers only).
    // Only enabled when seq_len >= 32 — below that, CPU attention is faster
    // because GPU command encoder overhead dominates at short sequences.
    int gpu_attn_fuse = (is_full && !attn_out_for_oproj && g_metal && g_metal->attn_scores_pipe
                         && kv && kv->len >= 32 && kv->len < g_gpu_kv_seq);

    if ((attn_out_for_oproj || gpu_attn_fuse) && oproj_w /* BF16 OK: GPU gemv_bf16 */ &&
        g_metal && g_metal->wf_buf && !g_gguf_stage && have_moe_weights &&
        g_metal->residual_add && g_metal->rms_norm_sum &&
        g_metal->rms_norm_apply_bf16 && lc->post_attn_norm_w) {
        // ---- FULLY FUSED CMD2 ----
        // For GPU attention (full-attn layers): attention dispatches are prepended,
        //   o_proj reads from buf_attn_out instead of batch_out[6].
        // For CPU attention / linear attn: o_proj reads from batch_out[6] as before.
        //
        // GPU attn path (12 encoders):
        //   Enc 1-4: attn_scores + softmax + values + sigmoid -> buf_attn_out
        //   Enc 5:   o_proj (buf_attn_out -> buf_output)
        //   Enc 6-8: residual + norm -> buf_input
        //   Enc 9-12: routing + shared expert
        //
        // CPU attn path (8 encoders, unchanged):
        //   Enc 1:   o_proj (batch_out[6] -> buf_output)
        //   Enc 2-4: residual + norm -> buf_input
        //   Enc 5-8: routing + shared expert

        if (!gpu_attn_fuse && !gpu_linear_attn) {
            // CPU/linear attn: copy attention output to GPU input buffer
            memcpy([g_metal->batch_out[6] contents], attn_out_for_oproj,
                   oproj_in_dim * sizeof(float));
        }
        // gpu_linear_attn: batch_out[6] already has the result from CMD1 gated_rms_norm
        // Copy residual into GPU buffer for residual_add kernel
        memcpy([g_metal->buf_residual contents], residual, HIDDEN_DIM * sizeof(float));

        attn_out_for_oproj = NULL;

        id<MTLCommandBuffer> cmd_fused = cmd12 ? cmd12 : [g_metal->queue commandBuffer];

        // ---- GPU attention dispatches (only for full-attn layers with GPU path) ----
        if (gpu_attn_fuse) {
            int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
            int kv_dim = NUM_KV_HEADS * HEAD_DIM;
            int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            uint32_t hd = HEAD_DIM;
            uint32_t kvd = (uint32_t)kv_dim;
            uint32_t sl = (uint32_t)kv->len;
            uint32_t seq_stride = (uint32_t)g_gpu_kv_seq;
            uint32_t hpkv = (uint32_t)heads_per_kv;

            // Enc A1: attn_scores_batched
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc setComputePipelineState:g_metal->attn_scores_pipe];
                [enc setBuffer:g_metal->buf_attn_q          offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_scores     offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&scale     length:4 atIndex:7];
                [enc setBytes:&hpkv      length:4 atIndex:8];
                [enc setBytes:&sl        length:4 atIndex:9];
                uint32_t total_tgs = sl * NUM_ATTN_HEADS;
                [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            // Enc A2: attn_softmax_batched
            // Phase C S6 fix: the scores→softmax→values→sigmoid chain ran
            // with NO sync between dispatches — the same-CB stale-L2 hazard
            // (the run-to-run wobble root cause; the batched prefill CB
            // carries barriers + synchronizeResource and is bitwise vs CPU
            // at 90 tokens, this path was not). Writer-side syncs + reader
            // barriers per the S4.1 pattern.
            metal_sync_buffer(cmd_fused, g_metal->buf_attn_scores);
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:g_metal->attn_softmax_pipe];
                [enc setBuffer:g_metal->buf_attn_scores offset:0 atIndex:0];
                [enc setBytes:&sl         length:4 atIndex:1];
                [enc setBytes:&seq_stride  length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc endEncoding];
            }
            // Enc A3: attn_values_batched
            metal_sync_buffer(cmd_fused, g_metal->buf_attn_scores);
            {
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:g_metal->attn_values_pipe];
                [enc setBuffer:g_metal->buf_attn_scores   offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                [enc setBuffer:g_metal->buf_attn_out      offset:0 atIndex:2];
                [enc setBytes:&hd        length:4 atIndex:3];
                [enc setBytes:&kvd       length:4 atIndex:4];
                [enc setBytes:&sl        length:4 atIndex:5];
                [enc setBytes:&seq_stride length:4 atIndex:6];
                [enc setBytes:&hpkv      length:4 atIndex:7];
                uint32_t total_threads = HEAD_DIM * NUM_ATTN_HEADS;
                uint32_t tgs = (total_threads + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc endEncoding];
            }
            // Enc A4: sigmoid_gate
            metal_sync_buffer(cmd_fused, g_metal->buf_attn_out);
            {
                uint32_t qdim = NUM_ATTN_HEADS * HEAD_DIM;
                id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:g_metal->sigmoid_gate_pipe];
                [enc setBuffer:g_metal->buf_attn_out  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_attn_gate offset:0 atIndex:1];
                [enc setBytes:&qdim length:4 atIndex:2];
                uint32_t tgs = (qdim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc endEncoding];
            }
            // o_proj (Enc 5, below) reads buf_attn_out — sync it too.
            metal_sync_buffer(cmd_fused, g_metal->buf_attn_out);
        }

        // ---- o_proj matvec ----
        {
            NSUInteger w_off = (NSUInteger)((const char *)oproj_w - (const char *)[g_metal->wf_buf contents]);
            id<MTLBuffer> oproj_input = gpu_attn_fuse ? g_metal->buf_attn_out : g_metal->batch_out[6];
            uint32_t o_out_dim = HIDDEN_DIM;
            uint32_t o_in_dim = (uint32_t)oproj_in_dim;
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];

            if (!oproj_s || !oproj_b) {
                // BF16 path: use gemv_bf16_x2 for out_dim >= 128
                int use_x2 = (o_out_dim >= 128 && g_metal->gemv_bf16_x2_pipe);
                [enc setComputePipelineState: use_x2 ? g_metal->gemv_bf16_x2_pipe : g_metal->gemv_bf16_pipe];
                [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
                [enc setBuffer:oproj_input      offset:0    atIndex:1];
                [enc setBuffer:g_metal->buf_output offset:0 atIndex:2];
                [enc setBytes:&o_out_dim  length:4 atIndex:3];
                [enc setBytes:&o_in_dim   length:4 atIndex:4];
                uint32_t tgs = use_x2 ? (o_out_dim + 1) / 2 : o_out_dim;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else if (oproj_bits == 8 && g_metal->matvec_8bit) {
                // 8-bit path: dequant_matvec_8bit (tiled, same dispatch as v3)
                NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]);
                NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[g_metal->wf_buf contents]);
                uint32_t o_gs = GROUP_SIZE;
                [enc setComputePipelineState:g_metal->matvec_8bit];
                [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
                [enc setBuffer:g_metal->wf_buf  offset:s_off atIndex:1];
                [enc setBuffer:g_metal->wf_buf  offset:b_off atIndex:2];
                [enc setBuffer:oproj_input      offset:0    atIndex:3];
                [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
                [enc setBytes:&o_out_dim  length:4 atIndex:5];
                [enc setBytes:&o_in_dim   length:4 atIndex:6];
                [enc setBytes:&o_gs       length:4 atIndex:7];
                uint32_t tgs8 = (o_out_dim + 7) / 8;
                [enc dispatchThreadgroups:MTLSizeMake(tgs8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else if (o_in_dim <= 4096 && g_metal->matvec_v3) {
                // 4-bit path, in_dim <= 4096: tiled v3 (8 rows/TG x 256 threads)
                // — much faster than matvec_fast's 1 row/TG x 64 threads.
                NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]);
                NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[g_metal->wf_buf contents]);
                uint32_t o_gs = GROUP_SIZE;
                [enc setComputePipelineState:g_metal->matvec_v3];
                [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
                [enc setBuffer:g_metal->wf_buf  offset:s_off atIndex:1];
                [enc setBuffer:g_metal->wf_buf  offset:b_off atIndex:2];
                [enc setBuffer:oproj_input      offset:0    atIndex:3];
                [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
                [enc setBytes:&o_out_dim  length:4 atIndex:5];
                [enc setBytes:&o_in_dim   length:4 atIndex:6];
                [enc setBytes:&o_gs       length:4 atIndex:7];
                uint32_t tgsv = (o_out_dim + 7) / 8;
                [enc dispatchThreadgroups:MTLSizeMake(tgsv, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                // 4-bit path, in_dim > 4096: matvec_fast (no x_shared limit)
                NSUInteger s_off = (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]);
                NSUInteger b_off = (NSUInteger)((const char *)oproj_b - (const char *)[g_metal->wf_buf contents]);
                uint32_t o_gs = GROUP_SIZE;
                [enc setComputePipelineState:g_metal->matvec_fast];
                [enc setBuffer:g_metal->wf_buf  offset:w_off atIndex:0];
                [enc setBuffer:g_metal->wf_buf  offset:s_off atIndex:1];
                [enc setBuffer:g_metal->wf_buf  offset:b_off atIndex:2];
                [enc setBuffer:oproj_input      offset:0    atIndex:3];
                [enc setBuffer:g_metal->buf_output offset:0 atIndex:4];
                [enc setBytes:&o_out_dim  length:4 atIndex:5];
                [enc setBytes:&o_in_dim   length:4 atIndex:6];
                [enc setBytes:&o_gs       length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(o_out_dim, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            [enc endEncoding];
        }

        // ---- Enc 2: residual + rms_norm fused (one dispatch, writes
        //      buf_h_mid + buf_input) ----
        {
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            float eps = RMS_NORM_EPS;
            NSUInteger norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                               (const char *)[g_metal->wf_buf contents]);
            [enc setComputePipelineState:g_metal->residual_norm_fused];
            // Fused fast path: read buf_moe_hidden (== hidden before attention)
            // directly from the GPU — the CPU memcpy of buf_residual would
            // require the (now-skipped) CMD1 wait. GPU-side ordering is
            // guaranteed by the serial queue (CMD3(N-1) precedes this buffer).
            id<MTLBuffer> resid_src = (cmd12_fused && prev_gpu_combined)
                ? g_metal->buf_moe_hidden : g_metal->buf_residual;
            [enc setBuffer:resid_src              offset:0       atIndex:0];
            [enc setBuffer:g_metal->buf_output    offset:0       atIndex:1];
            [enc setBuffer:g_metal->wf_buf        offset:norm_off atIndex:2];
            [enc setBuffer:g_metal->buf_h_mid     offset:0       atIndex:3];
            [enc setBuffer:g_metal->buf_input     offset:0       atIndex:4];
            [enc setBytes:&dim length:4 atIndex:5];
            [enc setBytes:&eps length:4 atIndex:6];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // ---- Enc 3: routing + shared expert projections, ONE dispatch ----
        // Routing gate + shared_expert_gate are 8-bit packed; shared gate/up are 4-bit
        // (bits detected from manifest shapes at cache-build time).
        BatchMatvecSpec moe_specs[4] = {
            { gate_w, gate_s, gate_b, gate_scores,        (uint32_t)NUM_EXPERTS,        HIDDEN_DIM, GROUP_SIZE, 0, lc->gate_bits },
            { sgw,    sgs,    sgb,    shared_gate,         (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 1, lc->sg_bits },
            { suw,    sus,    sub,    shared_up,           (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 2, lc->su_bits },
            { seg_w,  seg_s,  seg_b,  &shared_gate_score,  1,                            HIDDEN_DIM, GROUP_SIZE, 3, lc->seg_bits },
        };
        if (g_metal->routing_batch_fused && gate_s && sgs && sus && seg_s) {
            // Single mixed-bits kernel: gate(8) + sg(4) + su(4) + seg(8)
            struct { uint32_t bits; uint32_t out_count; } secs[4] = {
                { (uint32_t)lc->gate_bits, NUM_EXPERTS },
                { (uint32_t)lc->sg_bits,   SHARED_INTERMEDIATE },
                { (uint32_t)lc->su_bits,   SHARED_INTERMEDIATE },
                { (uint32_t)lc->seg_bits,  1 },
            };
            uint32_t in_dim = HIDDEN_DIM, gs = GROUP_SIZE;
            id<MTLComputeCommandEncoder> enc = [cmd_fused computeCommandEncoder];
            [enc setComputePipelineState:g_metal->routing_batch_fused];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)gate_w - (const char *)[g_metal->wf_buf contents]) atIndex:0];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)gate_s - (const char *)[g_metal->wf_buf contents]) atIndex:1];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)gate_b - (const char *)[g_metal->wf_buf contents]) atIndex:2];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)sgw    - (const char *)[g_metal->wf_buf contents]) atIndex:3];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)sgs    - (const char *)[g_metal->wf_buf contents]) atIndex:4];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)sgb    - (const char *)[g_metal->wf_buf contents]) atIndex:5];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)suw    - (const char *)[g_metal->wf_buf contents]) atIndex:6];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)sus    - (const char *)[g_metal->wf_buf contents]) atIndex:7];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)sub    - (const char *)[g_metal->wf_buf contents]) atIndex:8];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)seg_w  - (const char *)[g_metal->wf_buf contents]) atIndex:9];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)seg_s  - (const char *)[g_metal->wf_buf contents]) atIndex:10];
            [enc setBuffer:g_metal->wf_buf offset:(NSUInteger)((const char *)seg_b  - (const char *)[g_metal->wf_buf contents]) atIndex:11];
            [enc setBuffer:g_metal->buf_input offset:0 atIndex:12];
            [enc setBuffer:g_metal->batch_out[0] offset:0 atIndex:13];
            [enc setBuffer:g_metal->batch_out[1] offset:0 atIndex:14];
            [enc setBuffer:g_metal->batch_out[2] offset:0 atIndex:15];
            [enc setBuffer:g_metal->batch_out[3] offset:0 atIndex:16];
            [enc setBytes:secs length:sizeof(secs) atIndex:17];
            [enc setBytes:&in_dim length:4 atIndex:18];
            [enc setBytes:&gs length:4 atIndex:19];
            uint32_t total_rows = NUM_EXPERTS + 2 * SHARED_INTERMEDIATE + 1;
            [enc dispatchThreadgroups:MTLSizeMake((total_rows + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        } else {
            gpu_encode_batch_matvec(g_metal, cmd_fused, moe_specs, 4);
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }

        // ---- Single commit+wait for all encoders ----
        if (g_timing_enabled) { t0 = now_ms(); }
        [cmd_fused commit];
        [cmd_fused waitUntilCompleted];

        // Fused fast path: the deferred finalize now runs after the single
        // wait (CMD3(N-1) is done — serial queue). residual stays CPU-side
        // for the next layer's non-fused paths / CPU combine.
        if (cmd12_fused && prev_gpu_combined) {
            finalize_deferred_experts();  // reads buf_moe_hidden -> hidden
            cpu_vec_copy(residual, hidden, HIDDEN_DIM);
        }
        // Debug: per-layer h_post dump for parity binary-search (pos 0)
        if (getenv("FINCHMOE_PF_DUMP") && pos == 0) {
            static FILE *pf_old = NULL;
            if (!pf_old) pf_old = fopen("/tmp/hpost_old.bin", "wb");
            if (pf_old) {
                fwrite([g_metal->buf_input contents], sizeof(float), HIDDEN_DIM, pf_old);
                fflush(pf_old);
            }
        }
        if (getenv("FINCHMOE_PF_DUMP") && cmd12_fused && layer_idx == 1 && pos == 0) {
            static FILE *pf_c3o = NULL;
            if (!pf_c3o) pf_c3o = fopen("/tmp/cmd3_components_old.bin", "wb");
            if (pf_c3o) {
                for (int k = 0; k < MAX_K; k++)
                    fwrite((const float *)[g_metal->buf_multi_expert_out[k] contents],
                           sizeof(float), HIDDEN_DIM, pf_c3o);
                fwrite((const float *)[g_metal->buf_shared_out contents],
                       sizeof(float), HIDDEN_DIM, pf_c3o);
                fwrite((const float *)[g_metal->buf_combine_params contents],
                       sizeof(float), 10, pf_c3o);
                fwrite((const float *)[g_metal->buf_h_mid contents],
                       sizeof(float), HIDDEN_DIM, pf_c3o);
                fflush(pf_c3o);
            }
        }
        if (getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && pos == 0 && cmd12_fused) {
            static FILE *pf_st3 = NULL;
            if (!pf_st3) pf_st3 = fopen("/tmp/state_after_tok0_chain.bin", "wb");
            if (pf_st3) {
                fwrite([g_metal->buf_conv_state[0] contents], sizeof(float), 3*LINEAR_CONV_DIM, pf_st3);
                fflush(pf_st3);
            }
        }
        // Fused layer-0 stage dump part 1 (CMD1 side was skipped)
        if (cmd12_fused && getenv("FINCHMOE_DUMP_STAGES") && layer_idx == 0 && pos <= 1) {
            static FILE *sf1f = NULL;
            if (!sf1f) sf1f = fopen("/tmp/stage_dump.bin", "wb");
            if (sf1f) {
                fwrite([g_metal->batch_out[0] contents], sizeof(float), LINEAR_CONV_DIM, sf1f);    // qkv 8192
                fwrite([g_metal->batch_out[1] contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1f); // z 4096
                fwrite([g_metal->batch_out[2] contents], sizeof(float), LINEAR_NUM_V_HEADS, sf1f); // beta 32
                fwrite([g_metal->batch_out[3] contents], sizeof(float), LINEAR_NUM_V_HEADS, sf1f); // alpha 32
                fwrite([g_metal->buf_conv_output contents], sizeof(float), LINEAR_CONV_DIM, sf1f); // conv 8192
                fwrite([g_metal->buf_delta_output contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1f); // delta out 4096
                fwrite([g_metal->batch_out[6] contents], sizeof(float), LINEAR_TOTAL_VALUE, sf1f); // gated 4096
                fflush(sf1f);
            }
        }

        // Read back results
        gpu_flush_batch_results(g_metal, moe_specs, 4);
        // Read h_mid from GPU buffer (needed for final combine)
        memcpy(h_mid, [g_metal->buf_h_mid contents], HIDDEN_DIM * sizeof(float));
        // Stage dump part 2: CMD2 outputs for layer-0 cross-validation
        if (getenv("FINCHMOE_DUMP_STAGES") && layer_idx == 0 && pos <= 1) {
            static FILE *sf2 = NULL;
            if (!sf2) sf2 = fopen("/tmp/stage_dump.bin", "ab");
            if (sf2) {
                fwrite([g_metal->buf_output contents], sizeof(float), HIDDEN_DIM, sf2); // o_proj 2048
                fwrite([g_metal->buf_h_mid contents], sizeof(float), HIDDEN_DIM, sf2);  // h_mid 2048
                fwrite([g_metal->buf_input contents], sizeof(float), HIDDEN_DIM, sf2);  // h_post 2048
                fflush(sf2);
            }
        }
        if (g_debug_layers && layer_idx <= 2) {
            float *ob = (float *)[g_metal->buf_output contents];
            float *hb = (float *)[g_metal->buf_h_mid contents];
            float *rb = (float *)[g_metal->buf_residual contents];
            NSUInteger ow_off = (NSUInteger)((const char *)oproj_w - (const char *)[g_metal->wf_buf contents]);
            NSUInteger os_off = oproj_s ? (NSUInteger)((const char *)oproj_s - (const char *)[g_metal->wf_buf contents]) : 0;
            const uint16_t *sp = (const uint16_t *)[g_metal->wf_buf contents];
            fprintf(stderr, "[OPROJ-DBG] L%d out[%.3f %.3f %.3f %.3f] h_mid[%.3f %.3f %.3f %.3f] resid[%.4f %.4f %.4f %.4f] w_off=%lu S[0]=%.6f\n",
                    layer_idx, ob[0], ob[1], ob[2], ob[3], hb[0], hb[1], hb[2], hb[3],
                    rb[0], rb[1], rb[2], rb[3],
                    (unsigned long)ow_off,
                    oproj_s ? bf16_to_f32(sp[os_off/2]) : 0.0f);
        }
        // Read h_post from buf_input (needed for expert input)
        memcpy(h_post, [g_metal->buf_input contents], HIDDEN_DIM * sizeof(float));
        if (getenv("FINCHMOE_DUMP_HPOST")) {
            static FILE *hf = NULL;
            if (!hf) hf = fopen("/tmp/hpost_dump.bin", "wb");
            if (hf) { fwrite(h_post, sizeof(float), HIDDEN_DIM, hf); fflush(hf); }
        }
        // Update hidden state to h_mid (= residual + o_proj)
        memcpy(hidden, h_mid, HIDDEN_DIM * sizeof(float));

        if (g_debug_layers) {
            debug_print_hidden("h_mid", layer_idx, h_mid, HIDDEN_DIM);
            debug_print_hidden("h_post", layer_idx, h_post, HIDDEN_DIM);
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_wait += t1 - t0; }

    } else {
        // ---- Non-fused fallback path ----
        // O projection
        if (attn_out_for_oproj && oproj_w /* BF16: scales may be NULL */) {
            fast_dequant_matvec(oproj_w, oproj_s, oproj_b, attn_out_for_oproj,
                                attn_projected, HIDDEN_DIM, oproj_in_dim, GROUP_SIZE, oproj_bits);
        }
        // attn_out_for_oproj is static — no free needed
        attn_out_for_oproj = NULL;

        // Residual connection
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = residual[i] + attn_projected[i];
        }
        // attn_projected, normed, residual are static — no free needed

        cpu_vec_copy(h_mid, hidden, HIDDEN_DIM);

        // Post-attention norm
        cpu_rms_norm(hidden, lc->post_attn_norm_w, h_post, HIDDEN_DIM, RMS_NORM_EPS);

        // Routing + shared expert batch
        if (have_moe_weights) {
            // 8-bit: routing gate + shared_expert_gate (CPU)
            cpu_dequant_matvec(gate_w, gate_s, gate_b, h_post, gate_scores,
                               NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, 8);
            float seg_buf;
            cpu_dequant_matvec(seg_w, seg_s, seg_b, h_post, &seg_buf,
                               1, HIDDEN_DIM, GROUP_SIZE, 8);
            shared_gate_score = seg_buf;
            // shared gate/up (bits from the manifest: 4 in the packed builds,
            // Q4_K/Q6_K = 10/11 in GGUF mode)
            BatchMatvecSpec moe_specs[2] = {
                { sgw,    sgs,    sgb,    shared_gate, (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 0, lc->sg_bits },
                { suw,    sus,    sub,    shared_up,   (uint32_t)SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, 1, lc->su_bits },
            };
            fast_batch_matvec(h_post, HIDDEN_DIM, moe_specs, 2);
        }
        if (g_timing_enabled) { t1 = now_ms(); g_timing.cmd2_encode += t1 - t0; }
    }

    // ---- Softmax + top-K (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    // Debug: per-(pos, layer) Phase-B input trace (FINCHMOE_DUMP_PHASEB)
    if (getenv("FINCHMOE_DUMP_PHASEB")) {
        static FILE *pb_ref = NULL;
        if (!pb_ref) pb_ref = fopen("/tmp/pb_ref.bin", "wb");
        if (pb_ref) {
            int32_t pi = (int32_t)pos, li = (int32_t)layer_idx, one = 1;
            fwrite(&pi, sizeof(int32_t), 1, pb_ref);
            fwrite(&li, sizeof(int32_t), 1, pb_ref);
            fwrite(&one, sizeof(int32_t), 1, pb_ref);
            fwrite(h_post, sizeof(float), HIDDEN_DIM, pb_ref);
            fwrite(gate_scores, sizeof(float), NUM_EXPERTS, pb_ref);
            fwrite(&shared_gate_score, sizeof(float), 1, pb_ref);
            // g_deferred.hidden = moe_hidden(L-1) equivalent (NULL before the
            // first dispatch — write zeros to keep the record format); h_mid
            // = resid+oproj. Both only compared for layers > 0.
            static float gdef_zero[HIDDEN_DIM];
            const float *gdef = g_deferred.hidden ? g_deferred.hidden : gdef_zero;
            fwrite(gdef, sizeof(float), HIDDEN_DIM, pb_ref);
            fwrite(h_mid, sizeof(float), HIDDEN_DIM, pb_ref);
            fflush(pb_ref);
        }
    }
    if (getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && pos == 0) {
        static FILE *pf_gsraw_old = NULL;
        if (!pf_gsraw_old) pf_gsraw_old = fopen("/tmp/gates_raw_old.bin", "wb");
        if (pf_gsraw_old) {
            fwrite(gate_scores, sizeof(float), NUM_EXPERTS, pf_gsraw_old);
            fwrite(gate_w, sizeof(uint16_t), 32, pf_gsraw_old);
            // TEMP DEBUG: per-token shared gate/up for the sg/su comparison
            fwrite(shared_gate, sizeof(float), SHARED_INTERMEDIATE, pf_gsraw_old);
            fwrite(shared_up, sizeof(float), SHARED_INTERMEDIATE, pf_gsraw_old);
            fflush(pf_gsraw_old);
        }
    }
    cpu_softmax(gate_scores, NUM_EXPERTS);
    int expert_indices[64];
    float expert_weights[64];
    cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    if (g_freq_tracking) {
        for (int k = 0; k < K; k++) {
            g_expert_freq[layer_idx][expert_indices[k]]++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }

    // Track speculative routing prediction accuracy
    if (s_spec_count > 0) {
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        for (int s = 0; s < s_spec_count; s++) {
            for (int r = 0; r < cmp_K; r++) {
                if (s_spec_indices[s] == expert_indices[r]) {
                    g_spec_route_hits++;
                    break;
                }
            }
        }
    }

    // Phase C S8 probe: temporal prediction accuracy for prefill — how many
    // of THIS position's experts were routed by the PREVIOUS position at the
    // same layer (the chunk g → g+1 prediction source).
    static int spec_temporal_last[60][MAX_K];
    static int spec_temporal_count[60];
    static int spec_temporal_init = 0;
    if (spec_probe) {
        if (!spec_temporal_init) {
            for (int l = 0; l < 60; l++) spec_temporal_count[l] = 0;
            spec_temporal_init = 1;
        }
        int cmp_K = (K > MAX_K) ? MAX_K : K;
        if (spec_temporal_count[layer_idx] > 0) {
            for (int r = 0; r < cmp_K; r++) {
                for (int t = 0; t < spec_temporal_count[layer_idx]; t++) {
                    if (spec_temporal_last[layer_idx][t] == expert_indices[r]) {
                        g_spec_temporal_hits++;
                        break;
                    }
                }
            }
        }
        for (int r = 0; r < cmp_K; r++) {
            spec_temporal_last[layer_idx][r] = expert_indices[r];
        }
        spec_temporal_count[layer_idx] = cmp_K;
    }

    if (g_timing_enabled) { t1 = now_ms(); g_timing.routing_cpu += t1 - t0; }

    // Log routing data for predictor training
    if (g_routing_log) {
        int32_t li = layer_idx;
        int32_t ki = (K > MAX_K) ? MAX_K : K;
        fwrite(&li, sizeof(int32_t), 1, g_routing_log);
        fwrite(&ki, sizeof(int32_t), 1, g_routing_log);
        fwrite(hidden, sizeof(float), HIDDEN_DIM, g_routing_log);
        fwrite(expert_indices, sizeof(int32_t), ki, g_routing_log);
        // Extended top-24 indices (gate scores pre-softmax — same ranking)
        // for predictor coverage analysis.
        int ext_indices[24];
        float ext_weights[24];
        cpu_topk(gate_scores, NUM_EXPERTS, 24, ext_indices, ext_weights);
        fwrite(ext_indices, sizeof(int32_t), 24, g_routing_log);
        g_routing_log_samples++;
    }

    // ---- Softmax + top-K (CPU) ----
    if (g_timing_enabled) { t0 = now_ms(); }
    if (getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && pos == 0) {
        static FILE *pf_gsraw_old = NULL;
        if (!pf_gsraw_old) pf_gsraw_old = fopen("/tmp/gates_raw_old.bin", "wb");
        if (pf_gsraw_old) {
            fwrite(gate_scores, sizeof(float), NUM_EXPERTS, pf_gsraw_old);
            fwrite(gate_w, sizeof(uint16_t), 32, pf_gsraw_old);
            // TEMP DEBUG: per-token shared gate/up for the sg/su comparison
            fwrite(shared_gate, sizeof(float), SHARED_INTERMEDIATE, pf_gsraw_old);
            fwrite(shared_up, sizeof(float), SHARED_INTERMEDIATE, pf_gsraw_old);
            fflush(pf_gsraw_old);
        }
    }
    cpu_softmax(gate_scores, NUM_EXPERTS);    // ---- Parallel pread + GPU experts ----
    if (g_timing_enabled) { t0 = now_ms(); }
    float *moe_out = s_moe_out;
    memset(moe_out, 0, HIDDEN_DIM * sizeof(float));
    float *shared_out = s_shared_out;
    memset(shared_out, 0, HIDDEN_DIM * sizeof(float));

    int actual_K = (K > MAX_K) ? MAX_K : K;

    // GPU expert path verified bit-identical to CPU path via --compare-experts.
    // Use --cpu-experts to force CPU path for debugging.
    // GGUF mode always takes the CPU path: the GPU expert kernels read the
    // packed 1/2/3/4/8-bit formats, not the Q4_K/Q6_K block slabs.
    int force_cpu_experts = (g_cpu_experts || g_gguf_stage) ? 1 : 0;
    if (force_cpu_experts) goto cpu_expert_fallback;

    if (packed_fd >= 0 && g_metal && g_metal->buf_multi_expert_data[0]) {
        // GPU multi-expert path with LRU cache + parallel I/O:
        // For each expert:
        //   - Cache HIT:  dispatch directly from cached Metal buffer (skip pread)
        //   - Cache MISS: pread into cache buffer, then dispatch from it
        // Falls back to original parallel_pread_experts when cache is disabled.

        int valid[MAX_K];
        id<MTLBuffer> expert_bufs[MAX_K];  // buffer to dispatch from per expert

        if (g_malloc_cache) {
            // ---- Malloc cache path (zero-copy Metal buffer wrappers) ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];
            int miss_cache_idx[MAX_K];  // cache entry index for each miss
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = malloc_cache_lookup(g_malloc_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: zero-copy dispatch directly from cache buffer
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert entry (get buffer to pread into)
                    int cidx = -1;
                    id<MTLBuffer> buf = malloc_cache_insert(g_malloc_cache, layer_idx, expert_indices[k], &cidx);
                    expert_bufs[k] = buf;
                    miss_indices[num_misses] = k;
                    miss_cache_idx[num_misses] = cidx;
                    num_misses++;
                    valid[k] = 0;
                }
            }

            // Phase 2: parallel pread misses directly into cache buffers (zero-copy)
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    int cidx = miss_cache_idx[m];
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = g_malloc_cache->data[cidx];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = NULL;  // always pread for cache population
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, esz);
                    }
                }
            }
        } else if (g_expert_cache) {
            // ---- Metal buffer LRU cache path ----
            // Phase 1: check cache for each expert, collect misses
            int miss_indices[MAX_K];       // indices into expert_indices[] for misses
            id<MTLBuffer> miss_bufs[MAX_K]; // cache buffers to pread into
            int num_misses = 0;

            for (int k = 0; k < actual_K; k++) {
                id<MTLBuffer> cached = expert_cache_lookup(g_expert_cache, layer_idx, expert_indices[k]);
                if (cached) {
                    // Cache hit: use this buffer directly for GPU dispatch
                    expert_bufs[k] = cached;
                    valid[k] = 1;
                } else {
                    // Cache miss: insert into cache (allocates or evicts), will pread below
                    id<MTLBuffer> buf = expert_cache_insert(g_expert_cache, layer_idx, expert_indices[k]);
                    if (buf) {
                        expert_bufs[k] = buf;
                        miss_indices[num_misses] = k;
                        miss_bufs[num_misses] = buf;
                        num_misses++;
                        valid[k] = 0;  // not yet loaded
                    } else {
                        expert_bufs[k] = nil;
                        valid[k] = 0;
                    }
                }
            }

            // Phase 2: parallel pread all cache misses
            if (num_misses > 0) {
                size_t esz = active_expert_size();
                InferPreadTask tasks[MAX_K];
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    tasks[m].fd = expert_pick_fd(layer_idx, expert_indices[k], packed_fd);
                    tasks[m].dst = [miss_bufs[m] contents];
                    tasks[m].offset = (off_t)expert_indices[k] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = mmap_base;
                }

                io_pool_dispatch(tasks, num_misses);

                // Mark successfully loaded misses as valid
                for (int m = 0; m < num_misses; m++) {
                    int k = miss_indices[m];
                    valid[k] = (tasks[m].result == (ssize_t)esz);
                    if (!valid[k]) {
                        fprintf(stderr, "WARNING: expert %d pread: %zd/%zu\n",
                                expert_indices[k], tasks[m].result, esz);
                    }
                }
            }
        } else if (pred_started) {
            // ---- Prediction path: predicted experts already loading into buf_B ----
            // Wait for predicted preads (they've had ~1.6ms: CMD1_wait + attn + CMD2 + routing)
            if (getenv("FINCHMOE_PRED_DEBUG")) fprintf(stderr, "[PRED] layer=%d wait start count=%d\n", layer_idx, g_pred_count[layer_idx]);
            async_pread_wait();
            g_pred_layers++;

            // Match predictions against actual routing
            int miss_ei[MAX_K];       // actual expert indices for misses
            int miss_k_slots[MAX_K];  // which k-slot each miss maps to
            int miss_count = 0;
            int hit_count = 0;

            for (int k = 0; k < actual_K; k++) {
                int found = 0;
                for (int p = 0; p < g_pred_count[layer_idx]; p++) {
                    if (expert_indices[k] == g_pred_experts[layer_idx][p] &&
                        g_async_pread.valid[p]) {
                        // Hit! This expert was pre-loaded into buf_B[p]
                        expert_bufs[k] = g_metal->buf_multi_expert_data_B[p];
                        valid[k] = 1;
                        found = 1;
                        hit_count++;
                        break;
                    }
                }
                if (!found) {
                    miss_ei[miss_count] = expert_indices[k];
                    miss_k_slots[miss_count] = k;
                    expert_bufs[k] = g_metal->buf_multi_expert_data[k];
                    miss_count++;
                }
            }
            if (getenv("FINCHMOE_PRED_DEBUG")) fprintf(stderr, "[PRED] layer=%d hits=%d misses=%d\n", layer_idx, hit_count, miss_count);
            g_pred_hits += hit_count;
            g_pred_misses += miss_count;

            // Parallel sync-pread misses into buf_A
            if (miss_count > 0) {
                InferPreadTask tasks[MAX_K];
                size_t esz = active_expert_size();
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    tasks[m].fd = packed_fd;
                    tasks[m].dst = [g_metal->buf_multi_expert_data[k] contents];
                    tasks[m].offset = (off_t)miss_ei[m] * esz;
                    tasks[m].size = esz;
                    tasks[m].result = 0;
                    tasks[m].mmap_base = NULL;
                    tasks[m].lz4_comp_buf = NULL;
                    tasks[m].lz4_comp_size = 0;
                }
                io_pool_dispatch(tasks, miss_count);
                for (int m = 0; m < miss_count; m++) {
                    int k = miss_k_slots[m];
                    valid[k] = (tasks[m].result == (ssize_t)active_expert_size());
                }
            }
        } else if (g_use_lz4 && g_lz4_index[layer_idx]) {
            // ---- LZ4 compressed path: read compressed + decompress via io_pool ----
            size_t esz = active_expert_size();
            InferPreadTask tasks[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                LZ4IndexEntry *ie = &g_lz4_index[layer_idx][expert_indices[k]];
                tasks[k].fd = packed_fd;
                tasks[k].dst = [g_metal->buf_multi_expert_data[k] contents];
                tasks[k].offset = ie->offset;
                tasks[k].size = esz;
                tasks[k].result = 0;
                tasks[k].mmap_base = NULL;
                tasks[k].lz4_comp_buf = g_lz4_comp_bufs[k];
                tasks[k].lz4_comp_size = ie->comp_size;
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
            io_pool_dispatch(tasks, actual_K);
            for (int k = 0; k < actual_K; k++) {
                valid[k] = (tasks[k].result == (ssize_t)esz);
            }
        } else {
            // ---- No cache, no prediction, no LZ4: ASYNC parallel pread ----
            async_pread_start(packed_fd, expert_indices, actual_K,
                              g_metal->buf_multi_expert_data, mmap_base);
            for (int k = 0; k < actual_K; k++) {
                expert_bufs[k] = g_metal->buf_multi_expert_data[k];
            }
        }

        // Shared expert prep (doesn't need expert data — can overlap with async pread)
        memcpy([g_metal->buf_multi_expert_input contents], h_post, HIDDEN_DIM * sizeof(float));
        memcpy([g_metal->buf_shared_gate contents], shared_gate,
               SHARED_INTERMEDIATE * sizeof(float));
        memcpy([g_metal->buf_shared_up contents], shared_up,
               SHARED_INTERMEDIATE * sizeof(float));

        // Wait for non-prediction async pread to complete
        if (!pred_started && g_async_pread.active) {
            async_pread_wait();
            for (int k = 0; k < actual_K; k++) {
                valid[k] = g_async_pread.valid[k];
            }
        }

        if (g_timing_enabled) { t1 = now_ms(); g_timing.expert_io += t1 - t0; }

        // Store this layer's routing for next token's temporal prediction.
        // MUST happen AFTER the prediction hit check above (which reads g_pred_experts).
        if (g_pred_enabled && g_pred_generating) {
            for (int k = 0; k < actual_K; k++) {
                g_pred_experts[layer_idx][k] = expert_indices[k];
            }
            g_pred_count[layer_idx] = actual_K;
            if (layer_idx == NUM_LAYERS - 1) {
                g_pred_valid = 1;
            }
        }

        if (g_timing_enabled) { t0 = now_ms(); }

        // Step 3: encode ALL experts + shared expert into ONE command buffer.
        // Batched encoding: 4 encoders for K experts + 2 for shared = 6 total
        // (vs. 4*K + 2 = 18 with old per-expert encoding).
        id<MTLCommandBuffer> cmd_experts = [g_metal->queue commandBuffer];
        // Wait for fused expert CB writes (no-op when non-fused path is active
        // since expert_sync_value starts at 0 and event is already at 0).
        [cmd_experts encodeWaitForEvent:g_metal->expert_sync_event
                                  value:g_metal->expert_sync_value];

        gpu_encode_experts_batched(g_metal, cmd_experts, actual_K, valid, expert_bufs, 0, NULL, NULL, 0);

        // Shared expert SwiGLU + down_proj (2 more encoders)
        // Note: shared_gate/up already copied to GPU buffers above (before async pread wait)

        // SwiGLU dispatch
        {
            id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
            [enc setComputePipelineState:g_metal->swiglu];
            [enc setBuffer:g_metal->buf_shared_gate offset:0 atIndex:0];
            [enc setBuffer:g_metal->buf_shared_up   offset:0 atIndex:1];
            [enc setBuffer:g_metal->buf_shared_act  offset:0 atIndex:2];
            uint32_t dim = SHARED_INTERMEDIATE;
            [enc setBytes:&dim length:4 atIndex:3];
            uint32_t swiglu_tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Shared down_proj dispatch
        if (sdw) { /* BF16: scales may be NULL */
            if (sds && sdb) {
                gpu_encode_dequant_matvec_with_io_bufs(
                    g_metal, cmd_experts, sdw, sds, sdb,
                    g_metal->buf_shared_act, g_metal->buf_shared_out,
                    HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, 0, 0);
            } else {
                // BF16: commit swiglu first so buf_shared_act is ready for CPU read
                [cmd_experts commit];
                [cmd_experts waitUntilCompleted];
                float *act = (float *)[g_metal->buf_shared_act contents];
                float *out = (float *)[g_metal->buf_shared_out contents];
                cpu_dequant_matvec(sdw, NULL, NULL, act, out,
                                   HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, lc->sd_bits);
                // Start a new command buffer for remaining dispatches
                cmd_experts = [g_metal->queue commandBuffer];
            }
        }

        // Step 4: GPU-side combine + residual + norm (if not last layer)
        // Appends dispatches to CMD3 so the next layer's CMD1 can submit immediately
        // without waiting for CMD3 to complete + CPU readback.
        //
        // For non-last layers with the combine pipeline available:
        //   Enc C1: moe_combine_residual (expert_outs + h_mid + shared_out -> buf_moe_hidden)
        //   Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
        //   Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_layer_norm_w -> buf_input)
        //
        // This makes CMD3 self-contained: it produces buf_input for the next layer's CMD1.
        // The next layer skips deferred_wait + finalize + input_norm entirely at layer start.

        int gpu_combine = (g_metal->moe_combine_residual &&
                           g_metal->rms_norm_sum &&
                           g_metal->rms_norm_apply_bf16 &&
                           g_metal->wf_buf &&
                           layer_idx < NUM_LAYERS - 1 &&
                           layer_cache[layer_idx + 1].input_norm_w != NULL);

        if (gpu_combine) {
            // Copy h_mid from buf_h_mid (populated by CMD2) — it's still valid on GPU.
            // h_mid is already in buf_h_mid from CMD2's residual_add dispatch.

            // Prepare combine params: expert_weights[0..K-1] + shared_gate_score
            {
                float *params = (float *)[g_metal->buf_combine_params contents];
                // Zero all 10 slots first (unused experts get weight=0)
                memset(params, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    params[k] = valid[k] ? expert_weights[k] : 0.0f;
                }
                params[8] = shared_gate_score;
            }

            // Enc C1: moe_combine_residual
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                // Wait for fused expert CB writes to out[k] to be visible
                [enc waitForFence:g_metal->expert_fence];
                // Barrier: expert out[k] + shared_out are written by earlier
                // encoders in THIS command buffer; ensure those dispatches
                // complete before the combine reads them.
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                [enc setComputePipelineState:g_metal->moe_combine_residual];
                [enc setBuffer:g_metal->buf_h_mid         offset:0 atIndex:0];   // h_mid
                [enc setBuffer:g_metal->buf_shared_out    offset:0 atIndex:1];   // shared_out
                [enc setBuffer:g_metal->buf_moe_hidden    offset:0 atIndex:2];   // output: hidden
                // Bind all 8 expert output buffers (unused ones have weight=0 in params)
                for (int k = 0; k < MAX_K; k++) {
                    [enc setBuffer:g_metal->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
                }
                [enc setBuffer:g_metal->buf_combine_params offset:0 atIndex:11]; // params
                uint32_t dim = HIDDEN_DIM;
                uint32_t k_val = (uint32_t)actual_K;
                [enc setBytes:&dim   length:4 atIndex:12];
                [enc setBytes:&k_val length:4 atIndex:13];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C2: rms_norm_sum_sq (buf_moe_hidden -> buf_cmd3_sum_sq)
            {
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = HIDDEN_DIM;
                [enc setComputePipelineState:g_metal->rms_norm_sum];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0 atIndex:0];
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0 atIndex:1];
                [enc setBytes:&dim length:4 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            // Enc C3: rms_norm_apply_bf16 (buf_moe_hidden + next_norm_w -> buf_input)
            {
                uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
                NSUInteger norm_off = (NSUInteger)((const char *)next_norm_w -
                                                   (const char *)[g_metal->wf_buf contents]);
                id<MTLComputeCommandEncoder> enc = [cmd_experts computeCommandEncoder];
                uint32_t dim = HIDDEN_DIM;
                float eps = RMS_NORM_EPS;
                [enc setComputePipelineState:g_metal->rms_norm_apply_bf16];
                [enc setBuffer:g_metal->buf_moe_hidden  offset:0       atIndex:0]; // x
                [enc setBuffer:g_metal->wf_buf          offset:norm_off atIndex:1]; // weight (bf16)
                [enc setBuffer:g_metal->buf_cmd3_sum_sq offset:0       atIndex:2]; // sum_sq
                [enc setBuffer:g_metal->buf_input       offset:0       atIndex:3]; // out = normed
                [enc setBytes:&dim length:4 atIndex:4];
                [enc setBytes:&eps length:4 atIndex:5];
                uint32_t tgs = (dim + 255) / 256;
                [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        // DEFERRED commit — submit async, don't wait.
        [cmd_experts commit];
        if (g_timing_enabled) {
            t1 = now_ms();
            g_timing.cmd3_encode += t1 - t0;
            g_timing.count++;
            g_timing.total += t1 - t_layer_start;
        }

        // Save state for deferred completion
        g_deferred.active = 1;
        g_deferred.gpu_combined = gpu_combine;
        g_deferred.cmd_experts = cmd_experts;
        g_deferred.actual_K = actual_K;
        g_deferred.shared_gate_score = shared_gate_score;
        g_deferred.hidden = hidden;
        g_deferred.layer_idx = layer_idx;
        if (!gpu_combine) {
            // Only need to save h_mid for CPU-side combine path
            memcpy(g_deferred.h_mid, h_mid, HIDDEN_DIM * sizeof(float));
        }
        for (int k = 0; k < actual_K; k++) {
            g_deferred.expert_weights[k] = expert_weights[k];
            g_deferred.valid[k] = valid[k];
        }

        // GPU-vs-CPU expert comparison diagnostic
        if (g_compare_experts == layer_idx) {
            [cmd_experts waitUntilCompleted];

            fprintf(stderr, "\n=== GPU vs CPU Expert Comparison (layer %d, K=%d) ===\n", layer_idx, actual_K);
            fprintf(stderr, "h_post rms=%.4f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                    vec_rms(h_post, HIDDEN_DIM), h_post[0], h_post[1], h_post[2], h_post[3], h_post[4]);

            size_t esz = active_expert_size();
            for (int k = 0; k < actual_K; k++) {
                if (!valid[k]) { fprintf(stderr, "  expert %d: INVALID (skipped)\n", k); continue; }

                // Copy expert data that GPU used from Metal buffer to CPU
                void *expert_data = malloc(esz);
                memcpy(expert_data, [expert_bufs[k] contents], esz);

                uint32_t *gw = (uint32_t *)((char *)expert_data + GATE_W_OFF);
                uint16_t *gs_p = (uint16_t *)((char *)expert_data + GATE_S_OFF);
                uint16_t *gb_p = (uint16_t *)((char *)expert_data + GATE_B_OFF);
                uint32_t *uw = (uint32_t *)((char *)expert_data + UP_W_OFF);
                uint16_t *us_p = (uint16_t *)((char *)expert_data + UP_S_OFF);
                uint16_t *ub_p = (uint16_t *)((char *)expert_data + UP_B_OFF);
                uint32_t *dw = (uint32_t *)((char *)expert_data + DOWN_W_OFF);
                uint16_t *ds_p = (uint16_t *)((char *)expert_data + DOWN_S_OFF);
                uint16_t *db_p = (uint16_t *)((char *)expert_data + DOWN_B_OFF);

                // --- CPU compute ---
                float *cpu_gate = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *cpu_up   = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *cpu_act  = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *cpu_out  = malloc(HIDDEN_DIM * sizeof(float));

                int cmp_bits = g_use_int8 ? 8 : (g_use_3bit ? 3 : 4);
                cpu_dequant_matvec(gw, gs_p, gb_p, h_post, cpu_gate, MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, cmp_bits);
                cpu_dequant_matvec(uw, us_p, ub_p, h_post, cpu_up,   MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, cmp_bits);
                cpu_swiglu(cpu_gate, cpu_up, cpu_act, MOE_INTERMEDIATE);
                cpu_dequant_matvec(dw, ds_p, db_p, cpu_act, cpu_out, HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, cmp_bits);

                // --- Read GPU results ---
                float *gpu_gate = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *gpu_up   = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *gpu_act  = malloc(MOE_INTERMEDIATE * sizeof(float));
                float *gpu_out  = malloc(HIDDEN_DIM * sizeof(float));

                memcpy(gpu_gate, [g_metal->buf_multi_expert_gate[k] contents], MOE_INTERMEDIATE * sizeof(float));
                memcpy(gpu_up,   [g_metal->buf_multi_expert_up[k] contents],   MOE_INTERMEDIATE * sizeof(float));
                memcpy(gpu_act,  [g_metal->buf_multi_expert_act[k] contents],  MOE_INTERMEDIATE * sizeof(float));
                memcpy(gpu_out,  [g_metal->buf_multi_expert_out[k] contents],  HIDDEN_DIM * sizeof(float));

                // --- Compare ---
                float max_dg = 0, max_du = 0, max_da = 0, max_do = 0;
                float sum_dg = 0, sum_du = 0, sum_da = 0, sum_do = 0;
                int first_bad_gate = -1, first_bad_up = -1, first_bad_act = -1, first_bad_out = -1;

                for (int i = 0; i < MOE_INTERMEDIATE; i++) {
                    float dg = fabsf(cpu_gate[i] - gpu_gate[i]);
                    float du = fabsf(cpu_up[i] - gpu_up[i]);
                    float da = fabsf(cpu_act[i] - gpu_act[i]);
                    sum_dg += dg; sum_du += du; sum_da += da;
                    if (dg > max_dg) { max_dg = dg; if (first_bad_gate < 0 && dg > 1e-3) first_bad_gate = i; }
                    if (du > max_du) { max_du = du; if (first_bad_up < 0 && du > 1e-3) first_bad_up = i; }
                    if (da > max_da) { max_da = da; if (first_bad_act < 0 && da > 1e-3) first_bad_act = i; }
                }
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float d = fabsf(cpu_out[i] - gpu_out[i]);
                    sum_do += d;
                    if (d > max_do) { max_do = d; if (first_bad_out < 0 && d > 1e-3) first_bad_out = i; }
                }

                fprintf(stderr, "\n  --- Expert %d (idx=%d, weight=%.4f) ---\n", k, expert_indices[k], expert_weights[k]);
                fprintf(stderr, "  %-12s  %8s  %8s  %8s  %12s  %12s\n", "stage", "cpu_rms", "gpu_rms", "max_diff", "avg_diff", "first_bad@");
                fprintf(stderr, "  %-12s  %8.4f  %8.4f  %8.2e  %8.2e  %s\n",
                    "gate_proj", vec_rms(cpu_gate, MOE_INTERMEDIATE), vec_rms(gpu_gate, MOE_INTERMEDIATE),
                    max_dg, sum_dg/MOE_INTERMEDIATE,
                    first_bad_gate >= 0 ? "idx=" : "OK");
                if (first_bad_gate >= 0) fprintf(stderr, "    gate[%d]: cpu=%.6f gpu=%.6f\n", first_bad_gate, cpu_gate[first_bad_gate], gpu_gate[first_bad_gate]);

                fprintf(stderr, "  %-12s  %8.4f  %8.4f  %8.2e  %8.2e  %s\n",
                    "up_proj", vec_rms(cpu_up, MOE_INTERMEDIATE), vec_rms(gpu_up, MOE_INTERMEDIATE),
                    max_du, sum_du/MOE_INTERMEDIATE,
                    first_bad_up >= 0 ? "idx=" : "OK");
                if (first_bad_up >= 0) fprintf(stderr, "    up[%d]: cpu=%.6f gpu=%.6f\n", first_bad_up, cpu_up[first_bad_up], gpu_up[first_bad_up]);

                fprintf(stderr, "  %-12s  %8.4f  %8.4f  %8.2e  %8.2e  %s\n",
                    "swiglu", vec_rms(cpu_act, MOE_INTERMEDIATE), vec_rms(gpu_act, MOE_INTERMEDIATE),
                    max_da, sum_da/MOE_INTERMEDIATE,
                    first_bad_act >= 0 ? "idx=" : "OK");
                if (first_bad_act >= 0) fprintf(stderr, "    act[%d]: cpu=%.6f gpu=%.6f\n", first_bad_act, cpu_act[first_bad_act], gpu_act[first_bad_act]);

                fprintf(stderr, "  %-12s  %8.4f  %8.4f  %8.2e  %8.2e  %s\n",
                    "down_proj", vec_rms(cpu_out, HIDDEN_DIM), vec_rms(gpu_out, HIDDEN_DIM),
                    max_do, sum_do/HIDDEN_DIM,
                    first_bad_out >= 0 ? "idx=" : "OK");
                if (first_bad_out >= 0) fprintf(stderr, "    out[%d]: cpu=%.6f gpu=%.6f\n", first_bad_out, cpu_out[first_bad_out], gpu_out[first_bad_out]);

                // Show first 5 values of each stage
                fprintf(stderr, "  gate first5 CPU: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", cpu_gate[0], cpu_gate[1], cpu_gate[2], cpu_gate[3], cpu_gate[4]);
                fprintf(stderr, "  gate first5 GPU: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", gpu_gate[0], gpu_gate[1], gpu_gate[2], gpu_gate[3], gpu_gate[4]);
                fprintf(stderr, "  out  first5 CPU: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], cpu_out[4]);
                fprintf(stderr, "  out  first5 GPU: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3], gpu_out[4]);

                // Check scale/bias data used by GPU vs CPU
                float gs0 = bf16_to_f32(gs_p[0]);
                float gb0 = bf16_to_f32(gb_p[0]);
                uint32_t gw0 = gw[0];
                fprintf(stderr, "  gate weights[0]: packed=0x%08X scale[0]=%.6f bias[0]=%.6f\n", gw0, gs0, gb0);
                fprintf(stderr, "  first nibble=%d effective_weight=%.6f\n", (gw0 & 0xF), (float)(gw0 & 0xF) * gs0 + gb0);

                // ---- Fused vs Non-Fused GPU Comparison (4-bit only) ----
                if (!g_use_int8 && !g_use_1bit && !g_use_2bit && g_metal->fused_gate_up_swiglu_pipe) {
                    id<MTLCommandBuffer> fbuf = [g_metal->queue commandBuffer];
                    id<MTLComputeCommandEncoder> fenc = [fbuf computeCommandEncoder];
                    [fenc setComputePipelineState:g_metal->fused_gate_up_swiglu_pipe];
                    NSUInteger gw_off = GATE_W_OFF_4, gs_off = GATE_S_OFF_4, gb_off = GATE_B_OFF_4;
                    NSUInteger uw_off = UP_W_OFF_4, us_off = UP_S_OFF_4, ub_off = UP_B_OFF_4;
                    memcpy([g_metal->buf_multi_expert_data[0] contents], expert_data, esz);
                    memcpy([g_metal->buf_multi_expert_input contents], h_post, HIDDEN_DIM*sizeof(float));
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gw_off atIndex:0];
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gs_off atIndex:1];
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gb_off atIndex:2];
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:uw_off atIndex:3];
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:us_off atIndex:4];
                    [fenc setBuffer:g_metal->buf_multi_expert_data[0] offset:ub_off atIndex:5];
                    [fenc setBuffer:g_metal->buf_multi_expert_input offset:0 atIndex:6];
                    // Fused writes to act[0]; also use gate[0]/up[0] buffers to capture
                    // non-fused intermediate for comparison below
                    [fenc setBuffer:g_metal->buf_multi_expert_act[0] offset:0 atIndex:7];
                    uint32_t go = MOE_INTERMEDIATE, gi = HIDDEN_DIM, gs = GROUP_SIZE;
                    [fenc setBytes:&go length:4 atIndex:8];
                    [fenc setBytes:&gi length:4 atIndex:9];
                    [fenc setBytes:&gs length:4 atIndex:10];
                    uint32_t fused_tgs = MOE_INTERMEDIATE;
                    [fenc dispatchThreadgroups:MTLSizeMake(fused_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [fenc endEncoding];
                    [fbuf commit];
                    [fbuf waitUntilCompleted];

                    float *fused_act = malloc(MOE_INTERMEDIATE * sizeof(float));
                    memcpy(fused_act, [g_metal->buf_multi_expert_act[0] contents], MOE_INTERMEDIATE*sizeof(float));

                    // Also run non-fused gate+up matvecs on the SAME data to get
                    // intermediate gate/up values for per-element comparison
                    id<MTLCommandBuffer> nfbuf = [g_metal->queue commandBuffer];
                    id<MTLComputeCommandEncoder> nfenc = [nfbuf computeCommandEncoder];
                    [nfenc setComputePipelineState:g_metal->matvec_v3];
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gw_off atIndex:0];
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gs_off atIndex:1];
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:gb_off atIndex:2];
                    [nfenc setBuffer:g_metal->buf_multi_expert_input offset:0 atIndex:3];
                    [nfenc setBuffer:g_metal->buf_multi_expert_gate[0] offset:0 atIndex:4];
                    [nfenc setBytes:&go length:4 atIndex:5];
                    [nfenc setBytes:&gi length:4 atIndex:6];
                    [nfenc setBytes:&gs length:4 atIndex:7];
                    uint32_t gate_tgs = (go + 7) / 8;
                    [nfenc dispatchThreadgroups:MTLSizeMake(gate_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    // up_proj on same encoder (serialized)
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:uw_off atIndex:0];
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:us_off atIndex:1];
                    [nfenc setBuffer:g_metal->buf_multi_expert_data[0] offset:ub_off atIndex:2];
                    [nfenc setBuffer:g_metal->buf_multi_expert_up[0] offset:0 atIndex:4];
                    [nfenc dispatchThreadgroups:MTLSizeMake(gate_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [nfenc endEncoding];
                    [nfbuf commit];
                    [nfbuf waitUntilCompleted];

                    float *nf_gate = malloc(MOE_INTERMEDIATE * sizeof(float));
                    float *nf_up   = malloc(MOE_INTERMEDIATE * sizeof(float));
                    memcpy(nf_gate, [g_metal->buf_multi_expert_gate[0] contents], MOE_INTERMEDIATE*sizeof(float));
                    memcpy(nf_up,   [g_metal->buf_multi_expert_up[0] contents],   MOE_INTERMEDIATE*sizeof(float));

                    // Compute non-fused SwiGLU: SiLU(gate) * up
                    float *nf_swiglu = malloc(MOE_INTERMEDIATE * sizeof(float));
                    for (int i = 0; i < MOE_INTERMEDIATE; i++) {
                        float g = nf_gate[i];
                        nf_swiglu[i] = (g / (1.0f + expf(-g))) * nf_up[i];
                    }

                    // Compare fused act vs non-fused swiglu (both should be SiLU(gate)*up)
                    float max_df = 0, sum_df = 0, sum_sq_nf = 0, sum_sq_f = 0, dot = 0;
                    int first_bad = -1;
                    for (int i = 0; i < MOE_INTERMEDIATE; i++) {
                        float d = fabsf(nf_swiglu[i] - fused_act[i]);
                        sum_df += d;
                        sum_sq_nf += nf_swiglu[i] * nf_swiglu[i];
                        sum_sq_f += fused_act[i] * fused_act[i];
                        dot += nf_swiglu[i] * fused_act[i];
                        if (d > max_df) { max_df = d; if (first_bad < 0 && d > 1e-3) first_bad = i; }
                    }
                    float cos_sim = (sum_sq_nf > 0 && sum_sq_f > 0) ?
                        dot / sqrtf(sum_sq_nf * sum_sq_f) : 0.0f;

                    // Also compare gate vs up RMS and first values
                    // The fused kernel's vg should equal nf_gate[i], vu = nf_up[i]
                    // We can't read vg/vu directly, but we can compare act outputs

                    fprintf(stderr, "\n  === Fused vs Non-Fused GPU Comparison (SwiGLU act) ===\n");
                    fprintf(stderr, "  non-fused gate rms=%.4f up rms=%.4f swiglu rms=%.4f\n",
                        vec_rms(nf_gate, MOE_INTERMEDIATE), vec_rms(nf_up, MOE_INTERMEDIATE),
                        vec_rms(nf_swiglu, MOE_INTERMEDIATE));
                    fprintf(stderr, "  fused swiglu rms=%.4f\n", vec_rms(fused_act, MOE_INTERMEDIATE));
                    fprintf(stderr, "  %-12s  %8s  %8s  %8s  %12s  %12s\n",
                        "stage", "nonfused_rms", "fused_rms", "max_diff", "avg_diff", "cos_sim");
                    fprintf(stderr, "  %-12s  %8.4f  %8.4f  %8.2e  %8.2e  %12.8f\n",
                        "swiglu_act",
                        vec_rms(nf_swiglu, MOE_INTERMEDIATE),
                        vec_rms(fused_act, MOE_INTERMEDIATE),
                        max_df, sum_df/MOE_INTERMEDIATE, cos_sim);
                    if (first_bad >= 0) {
                        fprintf(stderr, "  first_bad@%d: nf_swiglu=%.8f fused=%.8f diff=%.2e\n",
                            first_bad, nf_swiglu[first_bad], fused_act[first_bad],
                            fabsf(nf_swiglu[first_bad] - fused_act[first_bad]));
                        fprintf(stderr, "    nf_gate[%d]=%.6f nf_up[%d]=%.6f\n",
                            first_bad, nf_gate[first_bad], first_bad, nf_up[first_bad]);
                        // Print first 10 non-zero values for pattern analysis
                        fprintf(stderr, "  first10 nonfused swiglu: [");
                        for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f%s", nf_swiglu[i], i<9?",":"");
                        fprintf(stderr, "]\n  first10 fused swiglu:    [");
                        for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f%s", fused_act[i], i<9?",":"");
                        fprintf(stderr, "]\n  first10 nonfused gate:   [");
                        for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f%s", nf_gate[i], i<9?",":"");
                        fprintf(stderr, "]\n  first10 nonfused up:     [");
                        for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f%s", nf_up[i], i<9?",":"");
                        fprintf(stderr, "]\n");
                    }

                    free(nf_gate); free(nf_up); free(nf_swiglu);
                    free(fused_act);
                }

                free(cpu_gate); free(cpu_up); free(cpu_act); free(cpu_out);
                free(gpu_gate); free(gpu_up); free(gpu_act); free(gpu_out);
                free(expert_data);
            }
            fprintf(stderr, "=== End Comparison ===\n\n");
        }

        // Return immediately — GPU experts are running async.
        // The next call to fused_layer_forward() or complete_deferred_experts()
        // will wait for the GPU and apply the final combine.
        return;

    }
    cpu_expert_fallback:
    if (g_gguf_stage) {
        // GGUF mode: routed experts are stacked Q4_K/Q6_K slabs in the mmap —
        // zero-copy reads straight from the mapped file (no packed files).
        // Phase C S2: the GPU path dispatches the slabs from per-tensor wraps;
        // the CPU loop below is the fallback (--cpu-experts or wrap failure).
        GgufExpertInfo *ge = &gguf_experts[layer_idx];
        float *expert_out_cpu = malloc(HIDDEN_DIM * sizeof(float));
        float total_weight = 0.0f;
        // TEMP DEBUG: per-token routing dump (pos, layer, indices, weights) — G5 gate
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx <= 1) {
            static FILE *pte = NULL;
            if (!pte) pte = fopen("/tmp/pt_experts.bin", "wb");
            if (pte) {
                int32_t hd[2] = { pos, layer_idx };
                int32_t idx[8];
                for (int k = 0; k < 8; k++) idx[k] = k < K ? expert_indices[k] : -1;
                fwrite(hd, sizeof(int32_t), 2, pte);
                fwrite(idx, sizeof(int32_t), 8, pte);
                fwrite(expert_weights, sizeof(float), 8, pte);
                fwrite(gate_scores, sizeof(float), NUM_EXPERTS, pte);
                fflush(pte);
            }
        }
        // GPU expert dispatch via the stable copy-pool buffers (the packed
        // path's design): slabs copied once per layer into preallocated
        // buffers, no mmap page churn. Verified bit-correct (parity
        // cos 1.000000) and ~4x faster than the CPU fallback (5.1 tok/s).
        // --cpu-experts or a missing Metal device falls back to the CPU loop.
        int experts_on_gpu = 0;
        if (!g_cpu_experts && g_metal) {
            double t_exp = now_ms();
            experts_on_gpu = gpu_gguf_experts_forward(g_metal, wf, layer_idx, h_post,
                                                      expert_indices, expert_weights, K, moe_out);
            if (getenv("FINCHMOE_EXPTIME") && layer_idx < 40) {
                static double acc = 0; static int n = 0;
                acc += now_ms() - t_exp; n++;
                if (n % 40 == 0) fprintf(stderr, "[exptime] avg %.3f ms/layer (n=%d)\n", acc / n, n);
            }
        }
        if (experts_on_gpu) {
            // moe_out already accumulated + renormalized by the GPU path
        } else {
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            const uint8_t *gate_slab = (const uint8_t *)wf->data + ge->gate_off + (size_t)eidx * ge->gate_slab;
            const uint8_t *up_slab   = (const uint8_t *)wf->data + ge->up_off   + (size_t)eidx * ge->up_slab;
            const uint8_t *down_slab = (const uint8_t *)wf->data + ge->down_off + (size_t)eidx * ge->down_slab;
            if (getenv("FINCHMOE_LAYER_DUMP") && layer_idx == 0 && k == 0) {
                FILE *df = fopen("/tmp/expert_dump.bin", "wb");
                if (df) {
                    fwrite(h_post, sizeof(float), HIDDEN_DIM, df);
                    fwrite(gate_scores, sizeof(float), NUM_EXPERTS, df);
                    uint32_t ids[2] = {(uint32_t)eidx, (uint32_t)(ge->gate_off + (size_t)eidx * ge->gate_slab)};
                    fwrite(ids, 4, 2, df);
                    fclose(df);
                }
            }
            float *gate_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *up_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *act_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            gguf_cpu_matvec(gate_slab, h_post, gate_proj_out, MOE_INTERMEDIATE, HIDDEN_DIM, ge->gate_type);
            gguf_cpu_matvec(up_slab,   h_post, up_proj_out,   MOE_INTERMEDIATE, HIDDEN_DIM, ge->up_type);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, MOE_INTERMEDIATE);
            gguf_cpu_matvec(down_slab, act_out, expert_out_cpu, HIDDEN_DIM, MOE_INTERMEDIATE, ge->down_type);
            if (getenv("FINCHMOE_LAYER_DUMP") && layer_idx == 0 && k == 0) {
                FILE *df = fopen("/tmp/expert_dump.bin", "ab");
                if (df) {
                    fwrite(gate_proj_out, sizeof(float), MOE_INTERMEDIATE, df);
                    fwrite(expert_out_cpu, sizeof(float), HIDDEN_DIM, df);
                    fclose(df);
                }
            }
            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            // Guard against NaN/Inf from degenerate expert quantization
            float er = 0;
            for (int j = 0; j < HIDDEN_DIM; j++) er += expert_out_cpu[j] * expert_out_cpu[j];
            if (isfinite(er) && er < 1e20f) {
                cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], HIDDEN_DIM);
                total_weight += expert_weights[k];
            }
        }
        free(expert_out_cpu);
        if (total_weight > 0.0f && total_weight < 0.99f) {
            float inv_tw = 1.0f / total_weight;
            for (int i = 0; i < HIDDEN_DIM; i++) moe_out[i] *= inv_tw;
        }
        }  // else (CPU expert loop)
        // CPU shared expert (mirrors the packed-file paths below)
        float *shared_act = calloc(SHARED_INTERMEDIATE, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, SHARED_INTERMEDIATE);
        if (sdw) { /* scales may be NULL in GGUF mode */
            cpu_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                               HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, lc->sd_bits);
        }
        free(shared_act);
    } else if (packed_fd >= 0) {
        // CPU fallback for experts
        size_t esz = active_expert_size();
        float *expert_out_cpu = malloc(HIDDEN_DIM * sizeof(float));
        float total_weight = 0.0f;
        for (int k = 0; k < K; k++) {
            int eidx = expert_indices[k];
            off_t expert_offset = (off_t)eidx * esz;
            void *expert_data = malloc(esz);
            ssize_t nread = pread(packed_fd, expert_data, esz, expert_offset);
            if (nread != (ssize_t)esz) {
                fprintf(stderr, "WARNING: layer %d expert %d pread: %zd/%zu\n",
                        layer_idx, eidx, nread, esz);
                free(expert_data);
                continue;
            }

            // CPU fallback offsets — use correct layout based on quantization mode
            NSUInteger c_gate_w, c_gate_s, c_gate_b, c_up_w, c_up_s, c_up_b, c_down_w, c_down_s, c_down_b;
            int c_bits;
            if (g_use_1bit) {
                c_gate_w = GATE_W_OFF_1; c_gate_s = GATE_S_OFF_1; c_gate_b = GATE_B_OFF_1;
                c_up_w   = UP_W_OFF_1;   c_up_s   = UP_S_OFF_1;   c_up_b   = UP_B_OFF_1;
                c_down_w = DOWN_W_OFF_1; c_down_s = DOWN_S_OFF_1; c_down_b = DOWN_B_OFF_1;
                c_bits = 1;
            } else if (g_use_2bit) {
                c_gate_w = GATE_W_OFF_2; c_gate_s = GATE_S_OFF_2; c_gate_b = GATE_B_OFF_2;
                c_up_w   = UP_W_OFF_2;   c_up_s   = UP_S_OFF_2;   c_up_b   = UP_B_OFF_2;
                c_down_w = DOWN_W_OFF_2; c_down_s = DOWN_S_OFF_2; c_down_b = DOWN_B_OFF_2;
                c_bits = 2;
            } else if (g_use_3bit) {
                c_gate_w = GATE_W_OFF_3; c_gate_s = GATE_S_OFF_3; c_gate_b = GATE_B_OFF_3;
                c_up_w   = UP_W_OFF_3;   c_up_s   = UP_S_OFF_3;   c_up_b   = UP_B_OFF_3;
                c_down_w = DOWN_W_OFF_3; c_down_s = DOWN_S_OFF_3; c_down_b = DOWN_B_OFF_3;
                c_bits = 3;
            } else if (g_use_int8) {
                c_gate_w = GATE_W_OFF_8; c_gate_s = GATE_S_OFF_8; c_gate_b = GATE_B_OFF_8;
                c_up_w   = UP_W_OFF_8;   c_up_s   = UP_S_OFF_8;   c_up_b   = UP_B_OFF_8;
                c_down_w = DOWN_W_OFF_8; c_down_s = DOWN_S_OFF_8; c_down_b = DOWN_B_OFF_8;
                c_bits = 8;
            } else {
                c_gate_w = GATE_W_OFF_4; c_gate_s = GATE_S_OFF_4; c_gate_b = GATE_B_OFF_4;
                c_up_w   = UP_W_OFF_4;   c_up_s   = UP_S_OFF_4;   c_up_b   = UP_B_OFF_4;
                c_down_w = DOWN_W_OFF_4; c_down_s = DOWN_S_OFF_4; c_down_b = DOWN_B_OFF_4;
                c_bits = 4;
            }
            uint32_t *gw = (uint32_t *)((char *)expert_data + c_gate_w);
            uint16_t *gs_p = (uint16_t *)((char *)expert_data + c_gate_s);
            uint16_t *gb_p = (uint16_t *)((char *)expert_data + c_gate_b);
            uint32_t *uw = (uint32_t *)((char *)expert_data + c_up_w);
            uint16_t *us_p = (uint16_t *)((char *)expert_data + c_up_s);
            uint16_t *ub_p = (uint16_t *)((char *)expert_data + c_up_b);
            uint32_t *dw = (uint32_t *)((char *)expert_data + c_down_w);
            uint16_t *ds_p = (uint16_t *)((char *)expert_data + c_down_s);
            uint16_t *db_p = (uint16_t *)((char *)expert_data + c_down_b);

            float *gate_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *up_proj_out = malloc(MOE_INTERMEDIATE * sizeof(float));
            float *act_out = malloc(MOE_INTERMEDIATE * sizeof(float));

            cpu_dequant_matvec(gw, gs_p, gb_p, h_post, gate_proj_out,
                               MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, c_bits);
            cpu_dequant_matvec(uw, us_p, ub_p, h_post, up_proj_out,
                               MOE_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, c_bits);
            cpu_swiglu(gate_proj_out, up_proj_out, act_out, MOE_INTERMEDIATE);
            cpu_dequant_matvec(dw, ds_p, db_p, act_out, expert_out_cpu,
                               HIDDEN_DIM, MOE_INTERMEDIATE, GROUP_SIZE, c_bits);

            if (layer_idx == 7) {
                float gr=0, ur=0, ar=0, er=0;
                for(int j=0;j<MOE_INTERMEDIATE;j++){gr+=gate_proj_out[j]*gate_proj_out[j];ur+=up_proj_out[j]*up_proj_out[j];ar+=act_out[j]*act_out[j];}
                for(int j=0;j<HIDDEN_DIM;j++) er+=expert_out_cpu[j]*expert_out_cpu[j];
                fprintf(stderr,"[EXPERT-DBG] layer=%d k=%d expert=%d gate_rms=%.6f up_rms=%.6f act_rms=%.6f out_rms=%.6f weight=%.4f\n",
                  layer_idx, k, eidx, sqrtf(gr/MOE_INTERMEDIATE), sqrtf(ur/MOE_INTERMEDIATE), sqrtf(ar/MOE_INTERMEDIATE), sqrtf(er/HIDDEN_DIM), expert_weights[k]);
            }

            free(gate_proj_out);
            free(up_proj_out);
            free(act_out);
            free(expert_data);

            // Guard against NaN/Inf from degenerate expert quantization
            float er = 0;
            for (int j = 0; j < HIDDEN_DIM; j++) er += expert_out_cpu[j] * expert_out_cpu[j];
            if (isfinite(er) && er < 1e20f) {
                cpu_vec_madd(moe_out, expert_out_cpu, expert_weights[k], HIDDEN_DIM);
                total_weight += expert_weights[k];
            }
        }
        free(expert_out_cpu);

        // Renormalize MoE output when some experts were skipped
        if (total_weight > 0.0f && total_weight < 0.99f) {
            float inv_tw = 1.0f / total_weight;
            for (int i = 0; i < HIDDEN_DIM; i++) moe_out[i] *= inv_tw;
        }

        // CPU shared expert
        float *shared_act = calloc(SHARED_INTERMEDIATE, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, SHARED_INTERMEDIATE);
        if (sdw) { /* BF16: scales may be NULL */
            cpu_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                               HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, lc->sd_bits);
        }
        free(shared_act);
    } else {
        // No experts available -- still need shared expert
        float *shared_act = calloc(SHARED_INTERMEDIATE, sizeof(float));
        cpu_swiglu(shared_gate, shared_up, shared_act, SHARED_INTERMEDIATE);
        if (sdw) { /* BF16: scales may be NULL */
            fast_dequant_matvec(sdw, sds, sdb, shared_act, shared_out,
                                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, lc->sd_bits);
        }
        free(shared_act);
    }

    // ---- Shared expert gate ----
    float shared_weight = cpu_sigmoid(shared_gate_score);
    for (int i = 0; i < HIDDEN_DIM; i++) {
        shared_out[i] *= shared_weight;
    }

    // ---- Final combine: hidden = h_mid + moe_out + shared_out ----
    if (getenv("FINCHMOE_LAYER_DUMP")) {
        FILE *df = fopen("/tmp/layer_dump.bin", "ab");
        if (df) {
            uint32_t tagid = ('c'<<8)|'m';   // combine: h_mid | moe_out | shared_out
            uint32_t len = HIDDEN_DIM * 3;
            fwrite(&tagid, 4, 1, df);
            fwrite(&len, 4, 1, df);
            fwrite(h_mid, sizeof(float), HIDDEN_DIM, df);
            fwrite(moe_out, sizeof(float), HIDDEN_DIM, df);
            fwrite(shared_out, sizeof(float), HIDDEN_DIM, df);
            fclose(df);
        }
    }
    { float hmid_rms=0, moe_rms=0, shr_rms=0;
      for (int i=0;i<HIDDEN_DIM;i++){hmid_rms+=h_mid[i]*h_mid[i];moe_rms+=moe_out[i]*moe_out[i];shr_rms+=shared_out[i]*shared_out[i];}
      hmid_rms=sqrtf(hmid_rms/HIDDEN_DIM); moe_rms=sqrtf(moe_rms/HIDDEN_DIM); shr_rms=sqrtf(shr_rms/HIDDEN_DIM);
      if(g_debug_layers || !isfinite(hmid_rms)||!isfinite(moe_rms)||!isfinite(shr_rms)) {
        fprintf(stderr,"[CPU-COMBINE] layer=%d h_mid_rms=%.6f moe_rms=%.6f shared_rms=%.6f\n",
          layer_idx, hmid_rms, moe_rms, shr_rms);
        // Check moe_out first few values when debugging or NaN detected
        fprintf(stderr,"[CPU-COMBINE] moe_out[0..3]=[%.4f,%.4f,%.4f,%.4f]\n",
          moe_out[0], moe_out[1], moe_out[2], moe_out[3]);
      }
    }
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = h_mid[i] + moe_out[i] + shared_out[i];
    }

    // TEMP DEBUG: per-token post-combine hidden dump (pos, layer) — G5 gate
    if (getenv("FINCHMOE_GGUF_DBG")) {
        static FILE *pth = NULL;
        if (!pth) pth = fopen("/tmp/pt_hidden.bin", "wb");
        if (pth) {
            int32_t hd[2] = { pos, layer_idx };
            fwrite(hd, sizeof(int32_t), 2, pth);
            fwrite(hidden, sizeof(float), HIDDEN_DIM, pth);
            fflush(pth);
        }
    }

    debug_print_hidden("output", layer_idx, hidden, HIDDEN_DIM);

    if (g_timing_enabled) {
        t1 = now_ms();
        g_timing.cmd3_encode += t1 - t0;  // includes CPU expert compute for non-GPU paths
        g_timing.count++;
        g_timing.total += t1 - t_layer_start;
    }

    // h_post, h_mid, gate_scores, moe_out, shared_out, shared_gate, shared_up
    // are all static scratch buffers — no free needed.
}

// ============================================================================
// Chunked batched prefill (--prefill-chunk N)
//
// Phase A batches all non-expert matmuls over M prompt positions in ONE
// command buffer (bitwise-identical math to the per-token kernels: same FMA
// form, same reductions). Phase B runs routing + expert I/O + CMD3 per
// position (as today), with all GPU outputs written to position slot m.
// ============================================================================

// Phase B: per-position routing + expert compute + GPU combine into slot m.
// Mirrors the default-path expert tail of fused_layer_forward. Returns the
// committed (deferred) CMD3 command buffer.
// Encode CMD3 for position m: routed expert matmuls + shared SwiGLU/down +
// GPU combine + next-layer norm. pool_mode selects the per-position pooled
// buffers (expert data slots [8m..8m+7] of buf_pool_expert_data, pf slot m
// for input/gate/up/act/shared) instead of the single-slot buffers.
// buf_pf_combine_params slot m must already be filled by the caller.
// Phase C S4: chunked GGUF linear-attention chain. CPU per position (the
// exact arithmetic of linear_attn_chain_cpu: conv1d, q/k RMSNorm, decay/beta
// with the GGUF A_log ternary), then ONE command buffer of M sequential
// delta_net_step dispatches on buf_delta_state[linear_layer_idx] — bitwise
// vs the per-token baseline (same kernel, state buffer, inputs; the in-place
// state RMW is hazard-serialized by Metal). Without the pipeline, the CPU
// BLAS recurrence runs per position on la_state->ssm_state (same condition
// as the per-token chain). in_proj_a/b are computed on CPU (staged BF16,
// bits 0). Results land in buf_pf_oproj_in slots (RMSNormGated on CPU);
// beta/alpha mirror into buf_pf_ba slots so the parity dumps and Phase-B
// trace see the same layout as packed mode.
static void prefill_chunk_chain_gguf(MetalCtx *ctx, LayerWeightCache *lc,
                                     LinearAttnState *la_state, int layer_idx,
                                     int linear_layer_idx, uint32_t M) {
    int qkv_dim = LINEAR_CONV_DIM;
    uint16_t *conv_w = lc->conv1d_w;
    uint16_t *gated_norm_w = lc->gated_norm_w;
    const float *qkv_batch = (const float *)[ctx->buf_pf_qkv contents];
    const float *z_batch   = (const float *)[ctx->buf_pf_z contents];
    const float *input_batch = (const float *)[ctx->buf_pf_input contents];
    float *oproj_in = (float *)[ctx->buf_pf_oproj_in contents];
    float *ba_out   = (float *)[ctx->buf_pf_ba contents];

    int gpu_recur = (ctx->delta_net_step &&
                     linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS);

    if (linear_attn_bypass) {
        // Mirrors the per-token bypass: attn contribution is zero, so
        // o_proj reads zeros and h_mid = residual.
        memset(oproj_in, 0, (size_t)M * LINEAR_TOTAL_VALUE * sizeof(float));
        memset(ba_out, 0, (size_t)M * 64 * sizeof(float));
        return;
    }

    // Per-position CPU scratch (reused; chunked prefill never overlaps the
    // per-token path's use of these statics).
    float *conv_out = s_conv_out;
    float *out_values = s_out_vals;
    float *beta_out = s_beta_proj_out;
    float *alpha_out = s_alpha_proj_out;

    int k_heads_per_v = LINEAR_NUM_V_HEADS / LINEAR_NUM_K_HEADS;

    // Phase C S4 perf pass: split the chain into CPU loop / GPU delta wait /
    // CPU readback so the per-layer table can attribute the chain's time.
    double t_chain_cpu = 0, t_delta_wait = 0, t_chain_rb = 0;
    if (g_chunk_timing_enabled) t_chain_cpu = now_ms();

    for (uint32_t m = 0; m < M; m++) {
        const float *qkv_out = qkv_batch + (size_t)m * qkv_dim;
        const float *z_out   = z_batch   + (size_t)m * LINEAR_TOTAL_VALUE;
        const float *input_m = input_batch + (size_t)m * HIDDEN_DIM;

        // in_proj_a/b on CPU. In this GGUF they are Q4_K (bits 10); other
        // GGUF files stage them as BF16 (bits 0) — cpu_dequant_matvec
        // dispatches both (10/11 → gguf_cpu_matvec, 0 → raw BF16).
        cpu_dequant_matvec(lc->b_w, NULL, NULL, input_m, beta_out,
                           LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, lc->b_bits);
        cpu_dequant_matvec(lc->a_w, NULL, NULL, input_m, alpha_out,
                           LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, lc->a_bits);
        memcpy(ba_out + (size_t)m * 32, beta_out, 32 * sizeof(float));
        memcpy(ba_out + (size_t)(M + m) * 32, alpha_out, 32 * sizeof(float));

        // TEMP DEBUG: chunked conv state ENTRY dump for layer 2
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            static FILE *cv2e = NULL;
            if (!cv2e) cv2e = fopen("/tmp/conv2_entry.bin", "wb");
            if (cv2e) {
                fwrite(la_state->conv_state, sizeof(float), 3 * LINEAR_CONV_DIM, cv2e);
                fflush(cv2e);
            }
            fprintf(stderr, "[S4DBG] layer 2 chain: la_state=%p conv_state=%p\n",
                    (void *)la_state, (void *)la_state->conv_state);
        }

        // Conv1d step + state update (identical to linear_attn_chain_cpu)
        memset(conv_out, 0, qkv_dim * sizeof(float));
        if (conv_w) {
            cpu_conv1d_step(la_state->conv_state, qkv_out, conv_w, conv_out,
                            qkv_dim, CONV_KERNEL_SIZE);
        }
        memmove(la_state->conv_state, la_state->conv_state + qkv_dim,
                (CONV_KERNEL_SIZE - 2) * qkv_dim * sizeof(float));
        memcpy(la_state->conv_state + (CONV_KERNEL_SIZE - 2) * qkv_dim, qkv_out,
               qkv_dim * sizeof(float));

        // TEMP DEBUG: chunked conv state dump for layer 2 (entry + exit)
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            static FILE *cv2n = NULL;
            if (!cv2n) cv2n = fopen("/tmp/conv2_new.bin", "wb");
            if (cv2n) {
                fwrite(la_state->conv_state, sizeof(float), 3 * LINEAR_CONV_DIM, cv2n);
                fflush(cv2n);
            }
        }

        // Split into q, k, v + q/k RMSNorm (identical arithmetic)
        float *lin_q = conv_out;
        float *lin_k = conv_out + LINEAR_TOTAL_KEY;
        float *lin_v = conv_out + 2 * LINEAR_TOTAL_KEY;
        float inv_scale = 1.0f / sqrtf((float)LINEAR_KEY_DIM);
        for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
            float *qh = lin_q + h * LINEAR_KEY_DIM;
            cpu_rms_norm_bare(qh, qh, LINEAR_KEY_DIM, 1e-6f);
            float q_scale = inv_scale * inv_scale;
            for (int d = 0; d < LINEAR_KEY_DIM; d++) qh[d] *= q_scale;
        }
        for (int h = 0; h < LINEAR_NUM_K_HEADS; h++) {
            float *kh = lin_k + h * LINEAR_KEY_DIM;
            cpu_rms_norm_bare(kh, kh, LINEAR_KEY_DIM, 1e-6f);
            for (int d = 0; d < LINEAR_KEY_DIM; d++) kh[d] *= inv_scale;
        }

        // decay/beta (identical arithmetic, GGUF A_log ternary)
        float *A_log = lc->A_log;
        uint16_t *dt_bias_bf16 = lc->dt_bias;
        float g_decay[LINEAR_NUM_V_HEADS];
        float beta_gate_arr[LINEAR_NUM_V_HEADS];
        for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
            float a_val = alpha_out[vh];
            float dt_b = dt_bias_bf16 ? bf16_to_f32(dt_bias_bf16[vh]) : 0.0f;
            float A_val = A_log ? (g_gguf_stage ? -A_log[vh] : expf(A_log[vh])) : 1.0f;
            float softplus_val = logf(1.0f + expf(a_val + dt_b));
            g_decay[vh] = expf(-A_val * softplus_val);
            beta_gate_arr[vh] = cpu_sigmoid(beta_out[vh]);
        }

        if (gpu_recur) {
            // Stage into per-position GPU slots; the recurrence dispatches
            // run batched after the CPU loop.
            memcpy((float *)[ctx->buf_pf_delta_q contents] + (size_t)m * LINEAR_TOTAL_KEY,
                   lin_q, LINEAR_TOTAL_KEY * sizeof(float));
            memcpy((float *)[ctx->buf_pf_delta_k contents] + (size_t)m * LINEAR_TOTAL_KEY,
                   lin_k, LINEAR_TOTAL_KEY * sizeof(float));
            memcpy((float *)[ctx->buf_pf_delta_v contents] + (size_t)m * LINEAR_TOTAL_VALUE,
                   lin_v, LINEAR_TOTAL_VALUE * sizeof(float));
            memcpy((float *)[ctx->buf_pf_delta_g_decay contents] + (size_t)m * LINEAR_NUM_V_HEADS,
                   g_decay, LINEAR_NUM_V_HEADS * sizeof(float));
            memcpy((float *)[ctx->buf_pf_delta_beta contents] + (size_t)m * LINEAR_NUM_V_HEADS,
                   beta_gate_arr, LINEAR_NUM_V_HEADS * sizeof(float));
        } else {
            // CPU BLAS recurrence (mirrors linear_attn_chain_cpu's fallback)
            for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                int kh = vh % LINEAR_NUM_K_HEADS;  // torch .repeat() block mapping (llama.cpp)
                float g = g_decay[vh];
                float b_gate = beta_gate_arr[vh];
                float *S = la_state->ssm_state + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
                float *v_h = lin_v + vh * LINEAR_VALUE_DIM;
                float *k_h = lin_k + kh * LINEAR_KEY_DIM;
                cblas_sscal(LINEAR_VALUE_DIM * LINEAR_KEY_DIM, g, S, 1);
                float kv_mem_vec[LINEAR_VALUE_DIM];
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                            1.0f, S, LINEAR_KEY_DIM, k_h, 1,
                            0.0f, kv_mem_vec, 1);
                float delta_vec[LINEAR_VALUE_DIM];
                for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
                    delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                }
                cblas_sger(CblasRowMajor, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                           1.0f, delta_vec, 1, k_h, 1, S, LINEAR_KEY_DIM);
                float *q_h = lin_q + kh * LINEAR_KEY_DIM;
                float *o_h = out_values + vh * LINEAR_VALUE_DIM;
                cblas_sgemv(CblasRowMajor, CblasNoTrans,
                            LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                            1.0f, S, LINEAR_KEY_DIM, q_h, 1,
                            0.0f, o_h, 1);
            }
            // RMSNormGated + write oproj_in slot m
            float *slot_out = oproj_in + (size_t)m * LINEAR_TOTAL_VALUE;
            for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                float *oh = out_values + vh * LINEAR_VALUE_DIM;
                float *zh = z_out + vh * LINEAR_VALUE_DIM;
                float *gh = slot_out + vh * LINEAR_VALUE_DIM;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
                } else {
                    memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
                }
            }
        }
    }

    // TEMP DEBUG: CPU reference qkv/z for layer 2 (matvec cross-check)
    if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
        static FILE *q2dbg = NULL;
        if (!q2dbg) q2dbg = fopen("/tmp/qkv2_dbg.bin", "wb");
        if (q2dbg) {
            static float cpu_qkv[LINEAR_CONV_DIM], cpu_z[LINEAR_TOTAL_VALUE];
            gguf_cpu_matvec(lc->qkv_w, input_batch, cpu_qkv, LINEAR_CONV_DIM, HIDDEN_DIM,
                            lc->qkv_bits == 10 ? 12 : 14);
            gguf_cpu_matvec(lc->z_w, input_batch, cpu_z, LINEAR_TOTAL_VALUE, HIDDEN_DIM,
                            lc->z_bits == 10 ? 12 : 14);
            fwrite(cpu_qkv, sizeof(float), LINEAR_CONV_DIM, q2dbg);
            fwrite((const float *)[ctx->buf_pf_qkv contents], sizeof(float), LINEAR_CONV_DIM, q2dbg);
            fwrite(cpu_z, sizeof(float), LINEAR_TOTAL_VALUE, q2dbg);
            fwrite((const float *)[ctx->buf_pf_z contents], sizeof(float), LINEAR_TOTAL_VALUE, q2dbg);
            fflush(q2dbg);
        }
    }

    if (g_chunk_timing_enabled) {
        double d = now_ms() - t_chain_cpu;
        g_chunk_timing.chain_cpu += d;
        pf_per_layer_add(layer_idx, 1, d);
        t_delta_wait = now_ms();
    }

    if (gpu_recur) {
        // TEMP DEBUG: save pre-dispatch state for layer 2 (CPU ref cross-check)
        static float saved_state[32 * 128 * 128];
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            memcpy(saved_state, [ctx->buf_delta_state[linear_layer_idx] contents], sizeof(saved_state));
        }
        // M sequential delta_net_step dispatches in one CB — the in-place
        // state RMW is hazard-serialized by Metal (each dispatch rebinds the
        // per-position slots before dispatching).
        id<MTLCommandBuffer> cmd_dn = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd_dn computeCommandEncoder];
        [enc setComputePipelineState:ctx->delta_net_step];
        uint32_t khpv = (uint32_t)k_heads_per_v;
        for (uint32_t m = 0; m < M; m++) {
            [enc setBuffer:ctx->buf_delta_state[linear_layer_idx] offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_pf_delta_q       offset:(NSUInteger)m * LINEAR_TOTAL_KEY * sizeof(float) atIndex:1];
            [enc setBuffer:ctx->buf_pf_delta_k       offset:(NSUInteger)m * LINEAR_TOTAL_KEY * sizeof(float) atIndex:2];
            [enc setBuffer:ctx->buf_pf_delta_v       offset:(NSUInteger)m * LINEAR_TOTAL_VALUE * sizeof(float) atIndex:3];
            [enc setBuffer:ctx->buf_pf_delta_g_decay offset:(NSUInteger)m * LINEAR_NUM_V_HEADS * sizeof(float) atIndex:4];
            [enc setBuffer:ctx->buf_pf_delta_beta    offset:(NSUInteger)m * LINEAR_NUM_V_HEADS * sizeof(float) atIndex:5];
            [enc setBuffer:ctx->buf_pf_delta_out     offset:(NSUInteger)m * LINEAR_TOTAL_VALUE * sizeof(float) atIndex:6];
            [enc setBytes:&khpv length:sizeof(khpv) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake(LINEAR_NUM_V_HEADS, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [enc endEncoding];
        pf_note_gap(&g_chunk_timing.delta_gap);
        [cmd_dn commit];
        [cmd_dn waitUntilCompleted];
        pf_note_wait_done();
        if (g_chunk_timing_enabled) {
            double d = now_ms() - t_delta_wait;
            g_chunk_timing.delta_wait += d;
            pf_per_layer_add(layer_idx, 2, d);
            t_chain_rb = now_ms();
        }

        // Readback out slots + RMSNormGated per position
        const float *outs = (const float *)[ctx->buf_pf_delta_out contents];
        for (uint32_t m = 0; m < M; m++) {
            const float *out_m = outs + (size_t)m * LINEAR_TOTAL_VALUE;
            const float *z_out = z_batch + (size_t)m * LINEAR_TOTAL_VALUE;
            float *slot_out = oproj_in + (size_t)m * LINEAR_TOTAL_VALUE;
            for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                const float *oh = out_m + vh * LINEAR_VALUE_DIM;
                const float *zh = z_out + vh * LINEAR_VALUE_DIM;
                float *gh = slot_out + vh * LINEAR_VALUE_DIM;
                if (gated_norm_w) {
                    cpu_rms_norm_gated(oh, zh, gated_norm_w, gh, LINEAR_VALUE_DIM, RMS_NORM_EPS);
                } else {
                    memcpy(gh, oh, LINEAR_VALUE_DIM * sizeof(float));
                }
            }
        }
        if (g_chunk_timing_enabled) {
            double d = now_ms() - t_chain_rb;
            g_chunk_timing.chain_readback += d;
            pf_per_layer_add(layer_idx, 3, d);
        }

        // TEMP DEBUG: chunked post-dispatch state dump for layer 2
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            static FILE *st2n = NULL;
            if (!st2n) st2n = fopen("/tmp/state2_new.bin", "wb");
            if (st2n) {
                fwrite([ctx->buf_delta_state[linear_layer_idx] contents], sizeof(float), 32*128*128, st2n);
                fflush(st2n);
            }
        }

        // TEMP DEBUG: CPU-BLAS reference recurrence for layer 2 (from the
        // saved pre-dispatch state + the staged inputs), vs the GPU output.
        if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
            static FILE *c2dbg = NULL;
            if (!c2dbg) c2dbg = fopen("/tmp/chain2_dbg.bin", "wb");
            if (c2dbg) {
                static float S2[32 * 128 * 128];
                memcpy(S2, saved_state, sizeof(S2));
                const float *lq = (const float *)[ctx->buf_pf_delta_q contents];
                const float *lk = (const float *)[ctx->buf_pf_delta_k contents];
                const float *lv = (const float *)[ctx->buf_pf_delta_v contents];
                const float *gd = (const float *)[ctx->buf_pf_delta_g_decay contents];
                const float *bt = (const float *)[ctx->buf_pf_delta_beta contents];
                static float out_ref[LINEAR_TOTAL_VALUE];
                for (int vh = 0; vh < LINEAR_NUM_V_HEADS; vh++) {
                    int kh = vh % LINEAR_NUM_K_HEADS;
                    float g = gd[vh];
                    float b_gate = bt[vh];
                    float *S = S2 + vh * LINEAR_VALUE_DIM * LINEAR_KEY_DIM;
                    float *v_h = (float *)lv + vh * LINEAR_VALUE_DIM;
                    float *k_h = (float *)lk + kh * LINEAR_KEY_DIM;
                    cblas_sscal(LINEAR_VALUE_DIM * LINEAR_KEY_DIM, g, S, 1);
                    float kv_mem_vec[LINEAR_VALUE_DIM];
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                                1.0f, S, LINEAR_KEY_DIM, k_h, 1,
                                0.0f, kv_mem_vec, 1);
                    float delta_vec[LINEAR_VALUE_DIM];
                    for (int vi = 0; vi < LINEAR_VALUE_DIM; vi++) {
                        delta_vec[vi] = (v_h[vi] - kv_mem_vec[vi]) * b_gate;
                    }
                    cblas_sger(CblasRowMajor, LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                               1.0f, delta_vec, 1, k_h, 1, S, LINEAR_KEY_DIM);
                    float *q_h = (float *)lq + kh * LINEAR_KEY_DIM;
                    float *o_h = out_ref + vh * LINEAR_VALUE_DIM;
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                LINEAR_VALUE_DIM, LINEAR_KEY_DIM,
                                1.0f, S, LINEAR_KEY_DIM, q_h, 1,
                                0.0f, o_h, 1);
                }
                fwrite(out_ref, sizeof(float), LINEAR_TOTAL_VALUE, c2dbg);            // CPU ref
                fwrite((const float *)[ctx->buf_pf_delta_out contents], sizeof(float), LINEAR_TOTAL_VALUE, c2dbg);  // GPU
                fflush(c2dbg);
            }
        }
    }
}

// Phase C S4 perf: CMD3 component dumps for position m (FINCHMOE_PF_DUMP
// debug-only) — moved out of the per-position encode so the merged group CB
// can dump after its single wait.
static void pf_dump_cmd3_debug(id<MTLCommandBuffer> cmd, int layer_idx, uint32_t m,
                               id<MTLBuffer> sh_gate, id<MTLBuffer> sh_up,
                               id<MTLBuffer> sh_act, NSUInteger mid_off,
                               id<MTLBuffer> __unsafe_unretained *data_bufs,
                               const NSUInteger *data_offs) {
    MetalCtx *ctx = g_metal;
    LayerWeightCache *lc = &layer_cache[layer_idx];
    [cmd waitUntilCompleted];
    static FILE *pf_c3 = NULL;
    if (!pf_c3) pf_c3 = fopen("/tmp/cmd3_components.bin", "wb");
    if (pf_c3) {
        for (int k = 0; k < MAX_K; k++)
            fwrite((const float *)[ctx->buf_multi_expert_out[k] contents] + (size_t)m * HIDDEN_DIM,
                   sizeof(float), HIDDEN_DIM, pf_c3);
        fwrite((const float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM,
               sizeof(float), HIDDEN_DIM, pf_c3);
        fwrite((const float *)[ctx->buf_pf_combine_params contents] + (size_t)m * 10,
               sizeof(float), 10, pf_c3);
        fwrite((const float *)[ctx->buf_pf_h_mid contents] + (size_t)m * HIDDEN_DIM,
               sizeof(float), HIDDEN_DIM, pf_c3);
        fwrite((const float *)[sh_gate contents] + mid_off / sizeof(float), sizeof(float), SHARED_INTERMEDIATE, pf_c3);
        fwrite((const float *)[sh_up contents] + mid_off / sizeof(float), sizeof(float), SHARED_INTERMEDIATE, pf_c3);
        fwrite((const float *)[sh_act contents] + mid_off / sizeof(float), sizeof(float), SHARED_INTERMEDIATE, pf_c3);
        fflush(pf_c3);
    }
    // TEMP DEBUG: pool slot bytes AFTER the CMD3 completes (race check)
    if (getenv("FINCHMOE_GGUF_DBG") && g_gguf_stage && data_bufs) {
        static FILE *sl2 = NULL;
        if (!sl2) sl2 = fopen("/tmp/slot_after.bin", "wb");
        if (sl2) {
            GgufExpertInfo *gx = &gguf_experts[layer_idx];
            int32_t hdr = (int32_t)layer_idx;
            fwrite(&hdr, 4, 1, sl2);
            // CPU full expert chain from the slot bytes as they are NOW
            static float g_out[MOE_INTERMEDIATE], u_out[MOE_INTERMEDIATE], act[MOE_INTERMEDIATE], eo[HIDDEN_DIM];
            const float *x = (const float *)[ctx->buf_pf_expert_input contents];
            const char *slot0 = (const char *)[data_bufs[0] contents] + data_offs[0];
            gguf_cpu_matvec(slot0, x, g_out, MOE_INTERMEDIATE, HIDDEN_DIM, gx->gate_type);
            gguf_cpu_matvec(slot0 + gx->gate_slab, x, u_out, MOE_INTERMEDIATE, HIDDEN_DIM, gx->up_type);
            cpu_swiglu(g_out, u_out, act, MOE_INTERMEDIATE);
            gguf_cpu_matvec(slot0 + gx->gate_slab + gx->up_slab, act, eo, HIDDEN_DIM, MOE_INTERMEDIATE, gx->down_type);
            fwrite(eo, sizeof(float), HIDDEN_DIM, sl2);
            fwrite(x, sizeof(float), HIDDEN_DIM, sl2);   // the input as seen at CMD3 time
            fflush(sl2);
        }
    }

    // TEMP DEBUG: CPU reference for the shared down (same act)
    if (getenv("FINCHMOE_GGUF_DBG") && g_gguf_stage && lc->sd_w) {
        static FILE *sdbg = NULL;
        if (!sdbg) sdbg = fopen("/tmp/shared_cpu.bin", "wb");
        if (sdbg) {
            const float *act = (const float *)[sh_act contents] + mid_off / sizeof(float);
            static float cpu_so[HIDDEN_DIM];
            gguf_cpu_matvec(lc->sd_w, act, cpu_so, HIDDEN_DIM, SHARED_INTERMEDIATE,
                            lc->sd_bits == 10 ? 12 : 14);
            fwrite(cpu_so, sizeof(float), HIDDEN_DIM, sdbg);                     // CPU down
            fwrite((const float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM,
                   sizeof(float), HIDDEN_DIM, sdbg);                             // GPU down
            fwrite(act, sizeof(float), SHARED_INTERMEDIATE, sdbg);               // act
            fflush(sdbg);
        }
    }
}

// Phase C S4 perf: encodes position m's CMD3 work into *cmd_p (the caller
// owns the CB and commits it once per group — all kernels bind per-position
// offsets, so a whole group's positions append into ONE CB; the per-position
// CBs were ~7 extra scheduling boundaries per layer). The caller has already
// placed the expert_sync_event wait on the CB; the wrap-failure fallback
// paths below commit the partial CB, run the CPU fallback, and swap in a
// fresh CB (re-adding the event wait) — queue order keeps every recorded
// wait valid.
static void prefill_chunk_cmd3_encode(id<MTLCommandBuffer> *cmd_p,
                                       int layer_idx, uint32_t m,
                                       int actual_K, const int *valid,
                                       float shared_gate_score,
                                       int pool_mode,
                                       id<MTLBuffer> __unsafe_unretained *data_bufs,  // pool mode: per-expert weight buffers
                                       const NSUInteger *data_offs)  // pool mode: per-expert base offsets
{
    MetalCtx *ctx = g_metal;
    LayerWeightCache *lc = &layer_cache[layer_idx];

    NSUInteger out_off = (NSUInteger)m * HIDDEN_DIM * sizeof(float);
    NSUInteger mid_off = pool_mode ? (NSUInteger)m * MOE_INTERMEDIATE * sizeof(float) : 0;
    id<MTLBuffer> sh_gate = pool_mode ? ctx->buf_pf_shared_gate : ctx->buf_shared_gate;
    id<MTLBuffer> sh_up   = pool_mode ? ctx->buf_pf_shared_up   : ctx->buf_shared_up;
    id<MTLBuffer> sh_act  = pool_mode ? ctx->buf_pf_shared_act  : ctx->buf_shared_act;

    id<MTLCommandBuffer> cmd = *cmd_p;

    if (g_gguf_stage) {
        // Phase C S4: GGUF expert encode — the S2 kernels reading the
        // copy-pool slots (pool mode) or buf_multi_expert_data[k] (fallback,
        // slabs memcpy'd there by the caller). All bindings are per-position
        // offsets; no packed-quant layout constants.
        GgufExpertInfo *ge = &gguf_experts[layer_idx];
        NSUInteger x_off = pool_mode ? (NSUInteger)m * HIDDEN_DIM * sizeof(float) : 0;
        NSUInteger act_off = pool_mode ? (NSUInteger)m * MOE_INTERMEDIATE * sizeof(float) : 0;
        for (int k = 0; k < actual_K; k++) {
            if (!valid[k]) continue;
            id<MTLBuffer> dbuf = data_bufs ? data_bufs[k] : ctx->buf_multi_expert_data[k];
            NSUInteger doff = data_bufs ? data_offs[k] : 0;
            uint32_t od = MOE_INTERMEDIATE, id_ = HIDDEN_DIM;
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:ctx->fused_gate_up_swiglu_qk_pipe];
                [enc setBuffer:dbuf offset:doff atIndex:0];
                [enc setBuffer:dbuf offset:doff + ge->gate_slab atIndex:1];
                [enc setBuffer:ctx->buf_pf_expert_input offset:x_off atIndex:2];
                [enc setBuffer:ctx->buf_pf_expert_act[k] offset:act_off atIndex:3];
                [enc setBytes:&od length:4 atIndex:4];
                [enc setBytes:&id_ length:4 atIndex:5];
                [enc dispatchThreadgroups:MTLSizeMake(MOE_INTERMEDIATE, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
            {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                uint32_t dod = HIDDEN_DIM, did = MOE_INTERMEDIATE, dgt = (uint32_t)ge->down_type;
                [enc setComputePipelineState:ctx->matvec_qk];
                [enc setBuffer:dbuf offset:doff + ge->gate_slab + ge->up_slab atIndex:0];
                [enc setBuffer:ctx->buf_pf_expert_act[k] offset:act_off atIndex:3];
                [enc setBuffer:ctx->buf_multi_expert_out[k] offset:out_off atIndex:4];
                [enc setBytes:&dod length:4 atIndex:5];
                [enc setBytes:&did length:4 atIndex:6];
                [enc setBytes:&dgt length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }
    } else {
        gpu_encode_experts_batched(ctx, cmd, actual_K, valid, ctx->buf_multi_expert_data,
                                   out_off, data_bufs, data_offs, pool_mode ? m : 0);
    }

    // Shared expert SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:sh_gate offset:mid_off atIndex:0];
        [enc setBuffer:sh_up   offset:mid_off atIndex:1];
        [enc setBuffer:sh_act  offset:mid_off atIndex:2];
        uint32_t dim = SHARED_INTERMEDIATE;
        [enc setBytes:&dim length:4 atIndex:3];
        uint32_t swiglu_tgs = (dim + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(swiglu_tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // Shared down_proj (out at slot m)
    if (lc->sd_w) {
        if (g_gguf_stage && (lc->sd_bits == 10 || lc->sd_bits == 11)) {
            // Phase C S4: QK shared down via a per-tensor wrap (~40 wraps
            // total, one per layer — no gguf_tbufs pressure).
            size_t row_bytes = (size_t)(SHARED_INTERMEDIATE / 256) * (lc->sd_bits == 10 ? 144 : 210);
            uint32_t delta = 0;
            id<MTLBuffer> tbuf = gguf_tbuf_get(ctx, lc->sd_w, row_bytes * HIDDEN_DIM, &delta);
            if (tbuf) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                uint32_t od = HIDDEN_DIM, idm = SHARED_INTERMEDIATE;
                uint32_t gt = (uint32_t)(lc->sd_bits == 10 ? 12 : 14);
                [enc setComputePipelineState:ctx->matvec_qk];
                [enc setBuffer:tbuf offset:delta atIndex:0];
                [enc setBuffer:sh_act offset:mid_off atIndex:3];
                [enc setBuffer:ctx->buf_shared_out offset:out_off atIndex:4];
                [enc setBytes:&od length:4 atIndex:5];
                [enc setBytes:&idm length:4 atIndex:6];
                [enc setBytes:&gt length:4 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake((HIDDEN_DIM + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            } else {
                // wrap failed — commit first, CPU compute, restart the CB
                [cmd commit];
                [cmd waitUntilCompleted];
                float *act = (float *)[sh_act contents] + (pool_mode ? (size_t)m * MOE_INTERMEDIATE : 0);
                float *out = (float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM;
                gguf_cpu_matvec(lc->sd_w, act, out, HIDDEN_DIM, SHARED_INTERMEDIATE,
                                lc->sd_bits == 10 ? 12 : 14);
                cmd = [ctx->queue commandBuffer];
                [cmd encodeWaitForEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
            }
        } else if (lc->sd_s && lc->sd_b) {
            gpu_encode_dequant_matvec_with_io_bufs(
                ctx, cmd, lc->sd_w, lc->sd_s, lc->sd_b,
                sh_act, ctx->buf_shared_out,
                HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, out_off, mid_off);
        } else {
            // BF16 fallback: commit first, CPU compute, restart the CB
            [cmd commit];
            [cmd waitUntilCompleted];
            float *act = (float *)[sh_act contents] + (pool_mode ? (size_t)m * MOE_INTERMEDIATE : 0);
            float *out = (float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM;
            cpu_dequant_matvec(lc->sd_w, NULL, NULL, act, out,
                               HIDDEN_DIM, SHARED_INTERMEDIATE, GROUP_SIZE, 0);
            cmd = [ctx->queue commandBuffer];
            [cmd encodeWaitForEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
        }
    }


    // ---- GPU combine + residual + norm into slot m (if not last layer) ----
    int gpu_combine = (ctx->moe_combine_residual_prefill &&
                       ctx->rms_norm_sum_sq_prefill &&
                       ctx->rms_norm_apply_bf16_prefill &&
                       layer_idx < NUM_LAYERS - 1 &&
                       layer_cache[layer_idx + 1].input_norm_w != NULL);

    if (gpu_combine) {
        // Enc C1: moe_combine_residual_prefill (slot m)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc waitForFence:ctx->expert_fence];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->moe_combine_residual_prefill];
            [enc setBuffer:ctx->buf_pf_h_mid        offset:0 atIndex:0];   // h_mid
            [enc setBuffer:ctx->buf_shared_out      offset:0 atIndex:1];   // shared_out
            [enc setBuffer:ctx->buf_pf_moe_hidden   offset:0 atIndex:2];   // output
            for (int k = 0; k < MAX_K; k++) {
                [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
            }
            [enc setBuffer:ctx->buf_pf_combine_params offset:0 atIndex:11]; // params
            uint32_t dim = HIDDEN_DIM;
            uint32_t k_val = (uint32_t)actual_K;
            uint32_t m32 = m;
            [enc setBytes:&dim   length:4 atIndex:12];
            [enc setBytes:&k_val length:4 atIndex:13];
            [enc setBytes:&m32   length:4 atIndex:14];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Enc C2: rms_norm_sum_sq_prefill (slot m -> buf_pf_sum_sq slot m)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            uint32_t m32 = m;
            [enc setComputePipelineState:ctx->rms_norm_sum_sq_prefill];
            [enc setBuffer:ctx->buf_pf_moe_hidden offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_pf_sum_sq      offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc setBytes:&m32 length:4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        // Enc C3: rms_norm_apply_bf16_prefill (slot m -> buf_pf_input slot m)
        {
            uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
            // Phase C S4: GGUF norm weights live in the staged-BF16 heap —
            // bind the stage mirror (same ternary as the residual-norm site).
            NSUInteger norm_off;
            id<MTLBuffer> norm_buf;
            if (g_gguf_stage) {
                norm_buf = ctx->gguf_stage_gpu;
                norm_off = (NSUInteger)((const char *)next_norm_w -
                                        (const char *)g_gguf_stage);
            } else {
                norm_buf = ctx->wf_buf;
                norm_off = (NSUInteger)((const char *)next_norm_w -
                                        (const char *)[ctx->wf_buf contents]);
            }
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            uint32_t dim = HIDDEN_DIM;
            float eps = RMS_NORM_EPS;
            uint32_t m32 = m;
            [enc setComputePipelineState:ctx->rms_norm_apply_bf16_prefill];
            [enc setBuffer:ctx->buf_pf_moe_hidden offset:0       atIndex:0]; // x
            [enc setBuffer:norm_buf               offset:norm_off atIndex:1]; // weight
            [enc setBuffer:ctx->buf_pf_sum_sq     offset:0       atIndex:2]; // sum_sq
            [enc setBuffer:ctx->buf_pf_input      offset:0       atIndex:3]; // out
            [enc setBytes:&dim length:4 atIndex:4];
            [enc setBytes:&eps length:4 atIndex:5];
            [enc setBytes:&m32 length:4 atIndex:6];
            uint32_t tgs = (dim + 255) / 256;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }
    }

    // The caller owns the CB: commits it once per group (deferred, as
    // before) and times the encode.
    *cmd_p = cmd;
    (void)shared_gate_score;  // already encoded in buf_pf_combine_params by the caller
}

// Single-position wrapper (the packed fallback path's old contract):
// create + encode + commit, return the CB (deferred, as before).
static id<MTLCommandBuffer> prefill_chunk_cmd3(int layer_idx, uint32_t m,
                                                int actual_K, const int *valid,
                                                float shared_gate_score,
                                                int pool_mode,
                                                id<MTLBuffer> __unsafe_unretained *data_bufs,
                                                const NSUInteger *data_offs) {
    MetalCtx *ctx = g_metal;
    double t_c3 = 0;
    if (g_chunk_timing_enabled) t_c3 = now_ms();
    id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
    [cmd encodeWaitForEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
    prefill_chunk_cmd3_encode(&cmd, layer_idx, m, actual_K, valid,
                              shared_gate_score, pool_mode, data_bufs, data_offs);
    [cmd commit];
    if (g_chunk_timing_enabled) {
        double d = now_ms() - t_c3;
        g_chunk_timing.cmd3_encode += d;
        pf_per_layer_add(layer_idx, 8, d);
    }
    return cmd;
}

// Phase C S4.1: batched CMD3 for a whole GGUF group — folds the
// per-position dispatches (21/position) into 16 + 5 = 21 TOTAL dispatches
// (fused×K + down×K + shared swiglu + shared down + combine + sum_sq +
// apply), eliminating ~147 GPU launch overheads per layer (~14-18us each
// — the ~2ms CMD3 fixed cost). All arithmetic is verbatim from the
// per-position kernels (bitwise parity target). Requires ALL (m,k) valid,
// the QK shared-down wrap, and gpu_combine; returns -1 → the caller falls
// back to the per-position encodes. Deduped pool slots arrive as per-gm
// byte-offset arrays (setBytes); buffer slots stay absolute-position.
static int prefill_chunk_cmd3_batch_general(
    MetalCtx *ctx, id<MTLCommandBuffer> cmd, int layer_idx,
    uint32_t gbase, uint32_t gM, int actual_K, const int *valid_all,
    id<MTLBuffer> gate_buf, uint32_t gate_delta,
    id<MTLBuffer> up_buf,   uint32_t up_delta,
    id<MTLBuffer> down_buf, uint32_t down_delta,
    const uint32_t g_arr[MAX_K][PREFILL_CHUNK_MAX],
    const uint32_t u_arr[MAX_K][PREFILL_CHUNK_MAX],
    const uint32_t d_arr[MAX_K][PREFILL_CHUNK_MAX]) {
    if (!ctx->fused_gate_up_swiglu_qk_pool_pipe || !ctx->matvec_qk_pool_prefill_pipe ||
        !ctx->swiglu_prefill_batch_pipe || !ctx->rms_norm_sum_sq_prefill_batch_pipe ||
        !ctx->rms_norm_apply_bf16_prefill_batch_pipe ||
        !ctx->moe_combine_residual_prefill_batch_pipe)
        return -1;
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int gpu_combine = (ctx->moe_combine_residual_prefill &&
                       ctx->rms_norm_sum_sq_prefill &&
                       ctx->rms_norm_apply_bf16_prefill &&
                       layer_idx < NUM_LAYERS - 1 &&
                       layer_cache[layer_idx + 1].input_norm_w != NULL);
    if (!gpu_combine || !lc->sd_w || (lc->sd_bits != 10 && lc->sd_bits != 11))
        return -1;
    for (uint32_t gm = 0; gm < gM; gm++) {
        for (int k = 0; k < actual_K; k++) {
            if (!valid_all[((size_t)(gbase + gm)) * MAX_K + k]) return -1;
        }
    }
    // shared-down QK wrap (per-tensor, cached in gguf_tbufs)
    size_t row_bytes = (size_t)(SHARED_INTERMEDIATE / 256) * (lc->sd_bits == 10 ? 144 : 210);
    uint32_t sd_delta = 0;
    id<MTLBuffer> sd_buf = gguf_tbuf_get(ctx, lc->sd_w, row_bytes * HIDDEN_DIM, &sd_delta);
    if (!sd_buf) return -1;

    GgufExpertInfo *ge = &gguf_experts[layer_idx];
    uint32_t od = MOE_INTERMEDIATE, id_ = HIDDEN_DIM;
    uint32_t dod = HIDDEN_DIM, did = MOE_INTERMEDIATE, dgt = (uint32_t)ge->down_type;
    uint32_t mb = gbase;
    // Phase C S4.1 perf probe: FINCHMOE_PF_C3SKIP=1 skips the expert fused
    // dispatches, =2 skips fused+down (timing-only — outputs go stale).
    static int c3skip = -1;
    if (c3skip < 0) {
        const char *se = getenv("FINCHMOE_PF_C3SKIP");
        c3skip = se ? atoi(se) : 0;
        if (c3skip < 0) c3skip = 0;
    }

    // fused gate+up per expert (all group positions in ONE dispatch)
    if (c3skip >= 1) goto batch_shared;

    {
        static int fused_probe = -1;  // 0=default 1=gateonly 2=barrier 3=xstage 4=nox
        if (fused_probe < 0) {
            if (getenv("FINCHMOE_PF_GATEONLY")) fused_probe = 1;
            else if (getenv("FINCHMOE_PF_BARRIER")) fused_probe = 2;
            else if (getenv("FINCHMOE_PF_XSTAGE")) fused_probe = 3;
            else if (getenv("FINCHMOE_PF_NOX")) fused_probe = 4;
            else fused_probe = 0;
            if (fused_probe == 1) fprintf(stderr, "[probe] fused gate-only (up reads disabled, timing-only)\n");
            if (fused_probe == 2) fprintf(stderr, "[probe] fused barrier-lockstep (timing-only)\n");
            if (fused_probe == 3) fprintf(stderr, "[probe] fused x-staged in threadgroup (bitwise-safe candidate)\n");
            if (fused_probe == 4) fprintf(stderr, "[probe] fused no-x-reads (timing-only)\n");
        }
        for (int k = 0; k < actual_K; k++) {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:(fused_probe == 1 ? ctx->fused_gate_up_swiglu_qk_pool_gateonly_pipe
                                   : fused_probe == 2 ? ctx->fused_gate_up_swiglu_qk_pool_barrier_pipe
                                   : fused_probe == 3 ? ctx->fused_gate_up_swiglu_qk_pool_xstage_pipe
                                   : fused_probe == 4 ? ctx->fused_gate_up_swiglu_qk_pool_nox_pipe
                                                      : ctx->fused_gate_up_swiglu_qk_pool_pipe)];
        [enc setBuffer:gate_buf offset:gate_delta atIndex:0];
        [enc setBuffer:up_buf   offset:up_delta   atIndex:1];
        [enc setBuffer:ctx->buf_pf_expert_input offset:0 atIndex:2];
        [enc setBuffer:ctx->buf_pf_expert_act[k] offset:0 atIndex:3];
        [enc setBytes:&od length:4 atIndex:4];
        [enc setBytes:&id_ length:4 atIndex:5];
        [enc setBytes:g_arr[k] length:(NSUInteger)gM * 4 atIndex:6];
        [enc setBytes:u_arr[k] length:(NSUInteger)gM * 4 atIndex:7];
        [enc setBytes:&mb length:4 atIndex:8];
        uint32_t fused_row_tiles = (MOE_INTERMEDIATE + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)gM * fused_row_tiles, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        }
    }
    // expert downs (all group positions in ONE dispatch per expert)
    if (c3skip >= 2) goto batch_shared;
    for (int k = 0; k < actual_K; k++) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_qk_pool_prefill_pipe];
        [enc setBuffer:down_buf offset:down_delta atIndex:0];
        [enc setBuffer:ctx->buf_pf_expert_act[k] offset:0 atIndex:3];
        [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:4];
        [enc setBytes:&dod length:4 atIndex:5];
        [enc setBytes:&did length:4 atIndex:6];
        [enc setBytes:&dgt length:4 atIndex:7];
        [enc setBytes:d_arr[k] length:(NSUInteger)gM * 4 atIndex:8];
        [enc setBytes:&mb length:4 atIndex:9];
        uint32_t num_row_tiles = (HIDDEN_DIM + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)gM * num_row_tiles, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // shared SwiGLU (all group positions in ONE dispatch)
batch_shared:
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu_prefill_batch_pipe];
        [enc setBuffer:ctx->buf_pf_shared_gate offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_pf_shared_up offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_pf_shared_act offset:0 atIndex:2];
        uint32_t dim = SHARED_INTERMEDIATE;
        [enc setBytes:&dim length:4 atIndex:3];
        [enc setBytes:&mb length:4 atIndex:4];
        uint32_t tgs = (dim + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, gM, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // shared down via the S3 prefill kernel (uniform weights — buffer base
    // offsets map the kernel's m=0..gM-1 onto the absolute slots)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        uint32_t sod = HIDDEN_DIM, sid = SHARED_INTERMEDIATE;
        uint32_t sgt = (uint32_t)(lc->sd_bits == 10 ? 12 : 14);
        [enc setComputePipelineState:ctx->matvec_qk_prefill];
        [enc setBuffer:sd_buf offset:sd_delta atIndex:0];
        [enc setBuffer:ctx->buf_pf_shared_act offset:(NSUInteger)gbase * SHARED_INTERMEDIATE * sizeof(float) atIndex:3];
        [enc setBuffer:ctx->buf_shared_out offset:(NSUInteger)gbase * HIDDEN_DIM * sizeof(float) atIndex:4];
        [enc setBytes:&sod length:4 atIndex:5];
        [enc setBytes:&sid length:4 atIndex:6];
        [enc setBytes:&sgt length:4 atIndex:7];
        uint32_t num_row_tiles = (HIDDEN_DIM + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)gM * num_row_tiles, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    // combine + residual (all group positions in ONE dispatch)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc waitForFence:ctx->expert_fence];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc setComputePipelineState:ctx->moe_combine_residual_prefill_batch_pipe];
        [enc setBuffer:ctx->buf_pf_h_mid        offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_shared_out      offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_pf_moe_hidden   offset:0 atIndex:2];
        for (int k = 0; k < MAX_K; k++) {
            [enc setBuffer:ctx->buf_multi_expert_out[k] offset:0 atIndex:(3 + k)];
        }
        [enc setBuffer:ctx->buf_pf_combine_params offset:0 atIndex:11];
        uint32_t dim = HIDDEN_DIM;
        uint32_t k_val = (uint32_t)actual_K;
        uint32_t tgs = (dim + 255) / 256;
        [enc setBytes:&dim   length:4 atIndex:12];
        [enc setBytes:&k_val length:4 atIndex:13];
        [enc setBytes:&mb    length:4 atIndex:14];
        [enc setBytes:&tgs   length:4 atIndex:15];
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)tgs * gM, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // sum_sq (all group positions in ONE dispatch)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        uint32_t dim = HIDDEN_DIM;
        [enc setComputePipelineState:ctx->rms_norm_sum_sq_prefill_batch_pipe];
        [enc setBuffer:ctx->buf_pf_moe_hidden offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_pf_sum_sq      offset:0 atIndex:1];
        [enc setBytes:&dim length:4 atIndex:2];
        [enc setBytes:&mb length:4 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(gM, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // apply norm (all group positions in ONE dispatch)
    {
        uint16_t *next_norm_w = layer_cache[layer_idx + 1].input_norm_w;
        NSUInteger norm_off;
        id<MTLBuffer> norm_buf;
        if (g_gguf_stage) {
            norm_buf = ctx->gguf_stage_gpu;
            norm_off = (NSUInteger)((const char *)next_norm_w -
                                    (const char *)g_gguf_stage);
        } else {
            norm_buf = ctx->wf_buf;
            norm_off = (NSUInteger)((const char *)next_norm_w -
                                    (const char *)[ctx->wf_buf contents]);
        }
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        uint32_t dim = HIDDEN_DIM;
        float eps = RMS_NORM_EPS;
        [enc setComputePipelineState:ctx->rms_norm_apply_bf16_prefill_batch_pipe];
        [enc setBuffer:ctx->buf_pf_moe_hidden offset:0       atIndex:0];
        [enc setBuffer:norm_buf               offset:norm_off atIndex:1];
        [enc setBuffer:ctx->buf_pf_sum_sq     offset:0       atIndex:2];
        [enc setBuffer:ctx->buf_pf_input      offset:0       atIndex:3];
        uint32_t tgs = (dim + 255) / 256;
        [enc setBytes:&dim length:4 atIndex:4];
        [enc setBytes:&eps length:4 atIndex:5];
        [enc setBytes:&mb  length:4 atIndex:6];
        [enc setBytes:&tgs length:4 atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((uint64_t)tgs * gM, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    return 0;
}

// Pool-layout wrapper: the three slabs live inside ONE pool slot at fixed
// intra-slot offsets (gate at 0, up at +gate_slab, down at +gate_slab+up_slab),
// so all three offset arrays derive from the slot base and all three buffer
// binds are the pool.
static int prefill_chunk_cmd3_batch(MetalCtx *ctx, id<MTLCommandBuffer> cmd,
                                    int layer_idx, uint32_t gbase, uint32_t gM,
                                    int actual_K, const int *valid_all,
                                    id<MTLBuffer> __unsafe_unretained *data_bufs,
                                    const NSUInteger *data_offs) {
    (void)data_bufs;
    GgufExpertInfo *ge = &gguf_experts[layer_idx];
    static uint32_t g_arr[MAX_K][PREFILL_CHUNK_MAX];  // [k][gm] fused gate offsets
    static uint32_t u_arr[MAX_K][PREFILL_CHUNK_MAX];  // [k][gm] fused up offsets
    static uint32_t d_arr[MAX_K][PREFILL_CHUNK_MAX];  // [k][gm] down offsets
    for (uint32_t gm = 0; gm < gM; gm++) {
        for (int k = 0; k < actual_K; k++) {
            uint32_t base = (uint32_t)data_offs[((size_t)(gbase + gm)) * MAX_K + k];
            g_arr[k][gm] = base;
            u_arr[k][gm] = base + (uint32_t)ge->gate_slab;
            d_arr[k][gm] = base + (uint32_t)ge->gate_slab + (uint32_t)ge->up_slab;
        }
    }
    return prefill_chunk_cmd3_batch_general(ctx, cmd, layer_idx, gbase, gM,
                                            actual_K, valid_all,
                                            ctx->buf_pool_expert_data_gguf, 0,
                                            ctx->buf_pool_expert_data_gguf, 0,
                                            ctx->buf_pool_expert_data_gguf, 0,
                                            g_arr, u_arr, d_arr);
}

// Phase C S6 probe (FINCHMOE_PF_NOPREAD): batched CMD3 reading the expert
// slabs DIRECTLY from the GGUF mmap via per-tensor wraps — no pool, no
// preads. Eliminates the routing+pread CPU gap before the CMD3 commit (the
// ~2.4ms gap that pays a ~3.6ms GPU wake tax — FINCHMOE_CBLAT curve) plus
// the ~63MB/layer CPU copy. The preads were verbatim slab copies, so the
// kernels read bitwise-identical bytes → outputs match the pool path.
// Returns -1 (wrap alloc fail / missing pipe) → caller falls back to the
// pool path for this group.
static int prefill_chunk_cmd3_batch_nopread(
    MetalCtx *ctx, id<MTLCommandBuffer> cmd, int layer_idx,
    uint32_t gbase, uint32_t gM, int actual_K, const int *valid_all,
    const int pf_idx[PREFILL_CHUNK_MAX][MAX_K]) {
    GgufExpertInfo *ge = &gguf_experts[layer_idx];
    uint32_t gate_delta = 0, up_delta = 0, down_delta = 0;
    id<MTLBuffer> gate_buf = gguf_tbuf_get(ctx, (const char *)g_gguf_data_base + ge->gate_off,
                                           256 * ge->gate_slab, &gate_delta);
    id<MTLBuffer> up_buf   = gguf_tbuf_get(ctx, (const char *)g_gguf_data_base + ge->up_off,
                                           256 * ge->up_slab, &up_delta);
    id<MTLBuffer> down_buf = gguf_tbuf_get(ctx, (const char *)g_gguf_data_base + ge->down_off,
                                           256 * ge->down_slab, &down_delta);
    if (!gate_buf || !up_buf || !down_buf) return -1;
    if (getenv("FINCHMOE_PF_NOPREAD_DBG")) {
        fprintf(stderr, "[nopread] L%d gate: off=%zu delta=%u wrap_len=%zu need=%zu | up: %zu/%u len=%zu need=%zu | down: %zu/%u len=%zu need=%zu\n",
                layer_idx, ge->gate_off, gate_delta, (size_t)[gate_buf length], 256 * ge->gate_slab,
                ge->up_off, up_delta, (size_t)[up_buf length], 256 * ge->up_slab,
                ge->down_off, down_delta, (size_t)[down_buf length], 256 * ge->down_slab);
    }
    static uint32_t g_arr[MAX_K][PREFILL_CHUNK_MAX];
    static uint32_t u_arr[MAX_K][PREFILL_CHUNK_MAX];
    static uint32_t d_arr[MAX_K][PREFILL_CHUNK_MAX];
    for (uint32_t gm = 0; gm < gM; gm++) {
        for (int k = 0; k < actual_K; k++) {
            uint32_t e = (uint32_t)pf_idx[(size_t)(gbase + gm)][k];
            g_arr[k][gm] = e * (uint32_t)ge->gate_slab;
            u_arr[k][gm] = e * (uint32_t)ge->up_slab;
            d_arr[k][gm] = e * (uint32_t)ge->down_slab;
        }
    }
    return prefill_chunk_cmd3_batch_general(ctx, cmd, layer_idx, gbase, gM,
                                            actual_K, valid_all,
                                            gate_buf, gate_delta,
                                            up_buf, up_delta,
                                            down_buf, down_delta,
                                            g_arr, u_arr, d_arr);
}

// Per-position expert path (fallback and M > pool/8): routing → pread →
// staging → CMD3. Verbatim flow from before the pool-mode refactor.
static id<MTLCommandBuffer> prefill_chunk_experts(
    int layer_idx, uint32_t m, uint32_t M,
    const float *gate_scores_batch,  // [M * NUM_EXPERTS] CPU
    const float *seg_batch,          // [M] CPU
    const float *shared_batch,       // [M * 1024] CPU (gate then up)
    const float *h_post_batch,       // [M * HIDDEN_DIM] CPU
    float *expert_weights_out,       // [MAX_K]
    int *valid_out,                  // [MAX_K]
    const void *mmap_base, int K, int packed_fd)
{
    MetalCtx *ctx = g_metal;

    // ---- Routing (CPU): softmax + top-K for this position ----
    double t_route = 0, t_pread = 0;
    if (g_chunk_timing_enabled) t_route = now_ms();
    float gate_scores[NUM_EXPERTS];
    memcpy(gate_scores, gate_scores_batch + (size_t)m * NUM_EXPERTS,
           NUM_EXPERTS * sizeof(float));
    float shared_gate_score = seg_batch[m];
    cpu_softmax(gate_scores, NUM_EXPERTS);
    int expert_indices[MAX_K];
    float expert_weights[MAX_K];
    cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, K);
    int actual_K = (K > MAX_K) ? MAX_K : K;
    if (g_chunk_timing_enabled) g_chunk_timing.routing_cpu += now_ms() - t_route;

    if (g_freq_tracking) {
        for (int k = 0; k < K; k++) {
            g_expert_freq[layer_idx][expert_indices[k]]++;
        }
        if (layer_idx == 0) g_freq_total_tokens++;
    }

    // ---- Parallel pread (default async path — chunk mode requires it) ----
    async_pread_start(packed_fd, expert_indices, actual_K,
                      ctx->buf_multi_expert_data, mmap_base);

    // Shared expert prep (overlaps async pread)
    memcpy([ctx->buf_multi_expert_input contents],
           h_post_batch + (size_t)m * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
    // Batch layout: gate region [0, M*512), up region [M*512, 2*M*512).
    memcpy([ctx->buf_shared_gate contents],
           shared_batch + (size_t)m * SHARED_INTERMEDIATE,
           SHARED_INTERMEDIATE * sizeof(float));
    memcpy([ctx->buf_shared_up contents],
           shared_batch + (size_t)M * SHARED_INTERMEDIATE + (size_t)m * SHARED_INTERMEDIATE,
           SHARED_INTERMEDIATE * sizeof(float));

    if (g_chunk_timing_enabled) t_pread = now_ms();
    async_pread_wait();
    if (g_chunk_timing_enabled) g_chunk_timing.pread_wait += now_ms() - t_pread;
    int valid[MAX_K];
    for (int k = 0; k < actual_K; k++) valid[k] = g_async_pread.valid[k];

    // Prepare combine params: expert_weights[0..K-1] + shared_gate_score.
    // Written for every position — the final-token CPU completion reads them
    // even for the last layer (which has no GPU combine).
    {
        float *params = (float *)[ctx->buf_pf_combine_params contents] + (size_t)m * 10;
        memset(params, 0, 10 * sizeof(float));
        for (int k = 0; k < actual_K; k++) {
            params[k] = valid[k] ? expert_weights[k] : 0.0f;
        }
        params[8] = shared_gate_score;
    }

    for (int k = 0; k < actual_K; k++) {
        expert_weights_out[k] = expert_weights[k];
        valid_out[k] = valid[k];
    }
    return prefill_chunk_cmd3(layer_idx, m, actual_K, valid, shared_gate_score, 0, NULL, NULL);
}

// Deferred CMD3 slots for the chunked path (file-scope: strong refs keep the
// command buffers alive until the next position's backpressure wait).
static id<MTLCommandBuffer> pf_cmd3_slots[PREFILL_CHUNK_MAX];

// One layer of chunked prefill for M positions (chunk_base .. chunk_base+M-1).
// Fills pf_cmd3_slots[m] with each position's deferred CMD3 and returns the
// final position's CMD3 via *last_cmd3_out.
static int gguf_gdn_gpu_enabled(void);  // defined with gguf_chunk_enabled below

static void prefill_chunk_layer(WeightFile *wf, int layer_idx,
                                const float *embed_batch, int chunk_base,
                                uint32_t M, int pos_base,
                                KVCache *kv, LinearAttnState *la_state,
                                const void *mmap_base, int K, int packed_fd,
                                int *layer_fds,   // all layers' packed expert fds (hot-set prefetch)
                                id<MTLCommandBuffer> __strong *last_cmd3_out)
{
    MetalCtx *ctx = g_metal;
    LayerWeightCache *lc = &layer_cache[layer_idx];
    int is_full = (kv != NULL);
    int linear_layer_idx = is_full ? -1 : layer_idx - (layer_idx + 1) / FULL_ATTN_INTERVAL;
    double t_layer = 0, t_ph = 0;
    if (g_chunk_timing_enabled) t_layer = now_ms();

    // ===================== Phase A ====================
    double t_cbc = 0, t_pregap = 0;
    if (g_chunk_timing_enabled) t_cbc = now_ms();
    id<MTLCommandBuffer> cmdA = [ctx->queue commandBuffer];
    if (g_chunk_timing_enabled) { t_pregap = now_ms(); g_pf_cbcreate_ms += t_pregap - t_cbc; g_pf_cbcreate_n++; }
    if (g_chunk_timing_enabled) t_ph = now_ms();

    // Layer-0 input: CPU RMS norm from embed batch + residual = embed batch.
    // Layers >= 1: buf_pf_input / buf_pf_residual already hold previous
    // layer's CMD3 output (queue order guarantees CMD3(L-1) completed).
    // Debug: embed_batch integrity check (chunk 0, layer 0)
    if (getenv("FINCHMOE_PF_DUMP") && chunk_base == 0 && layer_idx == 0) {
        static FILE *pf_emb = NULL;
        if (!pf_emb) pf_emb = fopen("/tmp/embed_integrity.bin", "wb");
        if (pf_emb) {
            fwrite(embed_batch, sizeof(float), 6, pf_emb);   // entry
        }
    }

    if (getenv("FINCHMOE_PF_DUMP") && chunk_base >= 1 && layer_idx == 0) {
        static FILE *pf_st2 = NULL;
        if (!pf_st2) pf_st2 = fopen("/tmp/state_chunked.bin", "wb");
        if (pf_st2) {
            fwrite((const float *)[ctx->buf_conv_state[0] contents], sizeof(float), 3*LINEAR_CONV_DIM, pf_st2);
            fwrite((const float *)[ctx->buf_delta_state[0] contents], sizeof(float), 32*128*128, pf_st2);
            fflush(pf_st2);
        }
    }
    // Debug: hidden-output dump (chunk 0, ALL positions) — mirrors the
    // per-token FINCHMOE_DUMP_HIDDEN stage: hidden(L-1, m) = moe_hidden
    // slot m written by CMD3(L-1, m). Wait for the CBs, then dump slots.
    if (getenv("FINCHMOE_PF_DUMP") && layer_idx > 0) {
        static FILE *pf_hid = NULL;
        if (!pf_hid) pf_hid = fopen("/tmp/hidden_new.bin", "wb");
        if (pf_hid) {
            for (uint32_t m = 0; m < M; m++) {
                if (pf_cmd3_slots[m]) [pf_cmd3_slots[m] waitUntilCompleted];
                fwrite((const float *)[ctx->buf_pf_moe_hidden contents] + (size_t)m * HIDDEN_DIM,
                       sizeof(float), HIDDEN_DIM, pf_hid);
            }
            fflush(pf_hid);
        }
    }

    if (layer_idx == 0) {
        float *inp = (float *)[ctx->buf_pf_input contents];
        for (uint32_t m = 0; m < M; m++) {
            cpu_rms_norm(embed_batch + (size_t)(chunk_base + m) * HIDDEN_DIM,
                         lc->input_norm_w, inp + (size_t)m * HIDDEN_DIM,
                         HIDDEN_DIM, RMS_NORM_EPS);
        }
        memcpy([ctx->buf_pf_residual contents],
               embed_batch + (size_t)chunk_base * HIDDEN_DIM,
               (size_t)M * HIDDEN_DIM * sizeof(float));
        if (getenv("FINCHMOE_PF_DUMP") && chunk_base == 0) {
            static FILE *pf_new0 = NULL;
            if (!pf_new0) pf_new0 = fopen("/tmp/bufinput_chunked.bin", "wb");
            if (pf_new0) {
                fwrite((const float *)[ctx->buf_pf_input contents], sizeof(float), HIDDEN_DIM, pf_new0);
                fwrite((const float *)[ctx->buf_pf_input contents] + HIDDEN_DIM, sizeof(float), HIDDEN_DIM, pf_new0);
                fflush(pf_new0);
            }
        }
    }

    if (!is_full) {
        if (g_gguf_stage) {
            // ---- Phase C S4: GGUF linear layer ----
            // Batched QK projections (S3 kernel); in_proj_a/b run on CPU
            // inside the chain (staged BF16, bits 0 — the nil-wf_buf BF16
            // branch would rely on the raw-address accident, so the chain
            // computes them explicitly).
            // Phase C S4 perf pass: FINCHMOE_PF_KLOOP=N repeats the QKV+Z
            // encodes N times inside cmdA (outputs overwritten each pass —
            // timing-only mode). cmdA_wait/N isolates per-iteration kernel
            // time from CB commit/wait latency.
            {
                static int kloop = 0, kparsed = 0;
                if (!kparsed) {
                    const char *ke = getenv("FINCHMOE_PF_KLOOP");
                    kloop = ke ? atoi(ke) : 1;
                    if (kloop < 1) kloop = 1;
                    kparsed = 1;
                    if (kloop > 1) fprintf(stderr, "[kloop] cmdA iteration x%d (timing-only)\n", kloop);
                }
                for (int ki = 0; ki < kloop; ki++) {
                    gpu_encode_prefill_matvec(ctx, cmdA, lc->qkv_w, lc->qkv_s, lc->qkv_b,
                        ctx->buf_pf_input, ctx->buf_pf_qkv,
                        LINEAR_CONV_DIM, HIDDEN_DIM, GROUP_SIZE, lc->qkv_bits, M, 0);
                    gpu_encode_prefill_matvec(ctx, cmdA, lc->z_w, lc->z_s, lc->z_b,
                        ctx->buf_pf_input, ctx->buf_pf_z,
                        LINEAR_TOTAL_VALUE, HIDDEN_DIM, GROUP_SIZE, lc->z_bits, M, 0);
                }
            }
            // Phase C S6a (FINCHMOE_GGUF_GDN_GPU>=2): qkv/z matvecs + fused
            // chain in ONE CB — the fused chain reads the matvec outputs
            // after syncResource (the S4.1-verified same-CB chain pattern;
            // the wobble-era "CB boundary only" rule predates the
            // per-thread-history fix for the conv-state head-pair race, the
            // real root cause). Saves one kernel-CB dispatch (~0.26ms) and
            // the delta-gap wake tax (~0.5ms) per linear layer. The encoder
            // resolves every wrap BEFORE encoding, so a failure leaves the
            // CB holding only the matvecs — safe to fall back.
            int gdn_merged = 0;
            {
                static int gdn_mode = -1;
                if (gdn_mode < 0) {
                    const char *gme = getenv("FINCHMOE_GGUF_GDN_GPU");
                    gdn_mode = gme ? atoi(gme) : 0;
                    if (gdn_mode < 0) gdn_mode = 0;
                    if (gdn_mode >= 2)
                        fprintf(stderr, "[S6a] FINCHMOE_GGUF_GDN_GPU=%d: qkv/z matvecs + fused chain in ONE CB\n", gdn_mode);
                }
                if (gdn_mode >= 2) {
                    metal_sync_buffer(cmdA, ctx->buf_pf_qkv);
                    metal_sync_buffer(cmdA, ctx->buf_pf_z);
                    if (ctx->fused_gdn_batched_qk &&
                        gpu_encode_gdn_batched_gguf(ctx, cmdA, linear_layer_idx, lc, M) == 0) {
                        if (g_chunk_timing_enabled) {
                            g_chunk_timing.cmdA_encode += now_ms() - t_ph;
                            t_ph = now_ms();
                        }
                        pf_note_gap(&g_chunk_timing.cmdA_gap);
                        [cmdA commit];
                        [cmdA waitUntilCompleted];
                        pf_note_wait_done();
                        if (g_chunk_timing_enabled) {
                            double d = now_ms() - t_ph;
                            g_chunk_timing.cmdA_wait += d;
                            pf_per_layer_add(layer_idx, 0, d);
                        }
                        gdn_merged = 1;
                    }
                }
            }
            if (!gdn_merged) {
                if (g_chunk_timing_enabled) {
                    g_chunk_timing.cmdA_encode += now_ms() - t_ph;
                    t_ph = now_ms();
                }
                pf_note_gap(&g_chunk_timing.cmdA_gap);
                [cmdA commit];
                [cmdA waitUntilCompleted];
                pf_note_wait_done();
                if (g_chunk_timing_enabled) {
                    double d = now_ms() - t_ph;
                    g_chunk_timing.cmdA_wait += d;
                    pf_per_layer_add(layer_idx, 0, d);
                }
            }
            // CPU chain per position + batched GPU recurrence; writes
            // buf_pf_oproj_in slots (and buf_pf_ba for the dumps).
            // Phase C S5: FINCHMOE_GGUF_GDN_GPU selects the fused GPU chain
            // (conv + in_proj + decay + delta + gated norm in one CB — the
            // packed 2-CB structure; qkv/z are cross-CB reads, coherent).
            // Falls back to the CPU chain when the pipe is missing or a
            // wrap/stage bind fails. The CPU conv state is bridged by the
            // driver after the chunk loop (prefill_chunked_run's GDN bridge).
            if (!gdn_merged) {
                int fused_ok = 0;
                if (gguf_gdn_gpu_enabled()) {
                    id<MTLCommandBuffer> cmdG = [ctx->queue commandBuffer];
                    // FINCHMOE_PF_GKLOOP=N repeats the fused dispatch N times
                    // (timing-only — the state RMW repeats, corrupting it).
                    static int gkloop = 0, gkparsed = 0;
                    if (!gkparsed) {
                        const char *ke = getenv("FINCHMOE_PF_GKLOOP");
                        gkloop = ke ? atoi(ke) : 1;
                        if (gkloop < 1) gkloop = 1;
                        gkparsed = 1;
                        if (gkloop > 1) fprintf(stderr, "[gkloop] fused GDN iteration x%d (timing-only)\n", gkloop);
                    }
                    int enc_ok = 0;
                    for (int gi = 0; gi < gkloop; gi++) {
                        enc_ok = gpu_encode_gdn_batched_gguf(ctx, cmdG, linear_layer_idx, lc, M);
                        if (enc_ok != 0) break;
                    }
                    if (ctx->fused_gdn_batched_qk && enc_ok == 0) {
                        double t_gdn = 0;
                        if (g_chunk_timing_enabled) t_gdn = now_ms();
                        pf_note_gap(&g_chunk_timing.delta_gap);
                        [cmdG commit];
                        [cmdG waitUntilCompleted];
                        pf_note_wait_done();
                        if (g_chunk_timing_enabled) {
                            double d = now_ms() - t_gdn;
                            g_chunk_timing.delta_wait += d;
                            pf_per_layer_add(layer_idx, 2, d);
                        }
                        fused_ok = 1;
                    }
                }
                if (!fused_ok) {
                    prefill_chunk_chain_gguf(ctx, lc, la_state, layer_idx,
                                             linear_layer_idx, M);
                }
            }
        } else {
            // ---- Linear-attention layer: batched projections ----
            if (g_chunk_timing_enabled) g_pf_pregap_ms += now_ms() - t_pregap;
            if (g_chunk_timing_enabled) { double te = now_ms();
            gpu_encode_prefill_matvec(ctx, cmdA, lc->qkv_w, lc->qkv_s, lc->qkv_b,
                ctx->buf_pf_input, ctx->buf_pf_qkv,
                LINEAR_CONV_DIM, HIDDEN_DIM, GROUP_SIZE, lc->qkv_bits, M, 0);
            gpu_encode_prefill_matvec(ctx, cmdA, lc->z_w, lc->z_s, lc->z_b,
                ctx->buf_pf_input, ctx->buf_pf_z,
                LINEAR_TOTAL_VALUE, HIDDEN_DIM, GROUP_SIZE, lc->z_bits, M, 0);
            // beta region [0, M*32), alpha region [M*32, 2*M*32)
            gpu_encode_prefill_matvec(ctx, cmdA, lc->b_w, lc->b_s, lc->b_b,
                ctx->buf_pf_input, ctx->buf_pf_ba,
                LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, lc->b_bits, M, 0);
            gpu_encode_prefill_matvec(ctx, cmdA, lc->a_w, lc->a_s, lc->a_b,
                ctx->buf_pf_input, ctx->buf_pf_ba,
                LINEAR_NUM_V_HEADS, HIDDEN_DIM, GROUP_SIZE, lc->a_bits, M,
                (NSUInteger)M * 32 * sizeof(float));
            g_pf_enc_micro += now_ms() - te; g_pf_enc_n++; }
            // The GDN must NOT read the matvec outputs (qkv/z/ba) in the same
            // command buffer — the GPU serves stale L2 lines for same-CB
            // device-to-device reads (the run-to-run wobble root cause,
            // verified: the kernel's qkv read differed between runs while the
            // CPU readback of the same buffer was bitwise). A CB boundary is
            // the only reliable flush on this GPU.
            if (g_chunk_timing_enabled) {
                g_chunk_timing.cmdA_encode += now_ms() - t_ph;
                t_ph = now_ms();
            }
            // Phase C S8: matvecs + fused GDN in ONE CB (the S4.1 sync
            // pattern), DEFAULT ON. The old 2-CB split enforced the retired
            // "CB boundary = only reliable flush" rule; S6a verified same-CB
            // chaining (syncResource + reader barrier) bitwise on the GGUF
            // mode-2 path, and this merge is bitwise vs the 2-CB path
            // (cmp-clean -I A/B). Saves one commit+wait + wake tax per
            // linear layer-pass (-7.1% 90-token prefill). Kill switch:
            // FINCHMOE_PF_GDNMERGE=0.
            static int gdnmerge = -1;
            if (gdnmerge < 0) {
                const char *ge = getenv("FINCHMOE_PF_GDNMERGE");
                gdnmerge = (ge && atoi(ge) == 0) ? 0 : 1;
            }
            if (gdnmerge) {
                metal_sync_buffer(cmdA, ctx->buf_pf_qkv);
                metal_sync_buffer(cmdA, ctx->buf_pf_z);
                metal_sync_buffer(cmdA, ctx->buf_pf_ba);
                gpu_encode_gdn_batched(ctx, cmdA, linear_layer_idx, lc, M);
                [cmdA commit];
                [cmdA waitUntilCompleted];
                if (g_chunk_timing_enabled) g_chunk_timing.cmdA_wait += now_ms() - t_ph;
            } else {
            [cmdA commit];
            [cmdA waitUntilCompleted];
            if (g_chunk_timing_enabled) {
                g_chunk_timing.cmdA_wait += now_ms() - t_ph;
                t_ph = now_ms();   // S8 accounting fix — the wait was leaking
                                   // into the next cmdA_encode bucket
            }
            cmdA = [ctx->queue commandBuffer];
            static double g_pf_gdnenc_ms = 0; static int g_pf_gdnenc_n = 0;
            double t_gn0 = 0; if (g_chunk_timing_enabled) t_gn0 = now_ms();
            gpu_encode_gdn_batched(ctx, cmdA, linear_layer_idx, lc, M);
            if (g_chunk_timing_enabled) { g_pf_gdnenc_ms += now_ms() - t_gn0; g_pf_gdnenc_n++; }
            if (g_pf_gdnenc_n == 360)
                fprintf(stderr, "[gdn-enc] avg %.3f ms after 360 calls\n", g_pf_gdnenc_ms / g_pf_gdnenc_n);
            // CB boundary + CPU-copy bridge: the out_proj (encoded below) must
            // NOT read the chains' oproj_in device-to-device — barriers and
            // synchronizeResource both fail to flush L2 on this GPU (the
            // run-to-run wobble root cause). A CPU readback after the wait is
            // always coherent, so route the data through buf_pf_oproj_in2.
            if (g_chunk_timing_enabled) {
                g_chunk_timing.cmdA_encode += now_ms() - t_ph;
                t_ph = now_ms();
            }
            [cmdA commit];
            [cmdA waitUntilCompleted];
            if (g_chunk_timing_enabled) g_chunk_timing.cmdA_wait += now_ms() - t_ph;
            }   // end gdnmerge else
        }
        if (g_chunk_timing_enabled) t_ph = now_ms();
        memcpy([ctx->buf_pf_oproj_in2 contents], [ctx->buf_pf_oproj_in contents],
               (size_t)M * LINEAR_TOTAL_VALUE * sizeof(float));
        if (g_chunk_timing_enabled) {
            double d = now_ms() - t_ph;
            g_chunk_timing.bridge += d;
            pf_per_layer_add(layer_idx, 4, d);
        }
    } else {
        // ---- Full-attention layer: batched q/k/v ----
        if (g_chunk_timing_enabled) g_pf_pregap_ms += now_ms() - t_pregap;
        if (g_chunk_timing_enabled) { double te = now_ms();
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;
        gpu_encode_prefill_matvec(ctx, cmdA, lc->q_w, lc->q_s, lc->q_b,
            ctx->buf_pf_input, ctx->buf_pf_qkv,
            q_proj_dim, HIDDEN_DIM, GROUP_SIZE, lc->q_bits, M, 0);
        gpu_encode_prefill_matvec(ctx, cmdA, lc->k_w, lc->k_s, lc->k_b,
            ctx->buf_pf_input, ctx->buf_pf_kv,
            kv_dim, HIDDEN_DIM, GROUP_SIZE, lc->k_bits, M, 0);
        // v region after k region: [M*kv_dim, 2*M*kv_dim)
        gpu_encode_prefill_matvec(ctx, cmdA, lc->v_w, lc->v_s, lc->v_b,
            ctx->buf_pf_input, ctx->buf_pf_kv,
            kv_dim, HIDDEN_DIM, GROUP_SIZE, lc->v_bits, M,
            (NSUInteger)M * kv_dim * sizeof(float));
        g_pf_enc_micro += now_ms() - te; g_pf_enc_n++; }
    }
    if (is_full) {
        if (g_chunk_timing_enabled) {
            g_chunk_timing.cmdA_encode += now_ms() - t_ph;
            t_ph = now_ms();
        }
        pf_note_gap(&g_chunk_timing.cmdA_gap);
        [cmdA commit];
        [cmdA waitUntilCompleted];
        pf_note_wait_done();
        if (g_chunk_timing_enabled) g_chunk_timing.cmdA_wait += now_ms() - t_ph;
    }
    // (linear layers: cmdA stays open — residual_norm + routing encoders are
    // appended to the SAME command buffer below, one commit+wait per layer.
    // The qkv_after_cmdA / stage_pf debug dumps moved below the merged wait
    // since those buffers are now un-waited at this point for linear layers.)

    // ---- CPU attention compute (per position, sequential) ----
    if (g_chunk_timing_enabled && is_full) t_ph = now_ms();
    // Batched GPU attention: active when the pipelines exist and the chunk
    // fits the staged-query buffers (M <= PF_ATTN_MAX). Masked positions
    // (sl < 32 or sl >= g_gpu_kv_seq) keep CPU attention via causal_len = 0.
    static uint32_t pf_causal_len[PF_ATTN_MAX];
    uint32_t pf_sl_max = 0;
    int batch_attn_ok = (is_full && M <= PF_ATTN_MAX &&
                         ctx->attn_scores_prefill_pipe && ctx->attn_softmax_prefill_pipe &&
                         ctx->attn_values_prefill_pipe && ctx->sigmoid_gate_prefill_pipe &&
                         ctx->buf_pf_attn_q && ctx->buf_pf_attn_gate && ctx->buf_pf_attn_scores);
    if (batch_attn_ok) memset(pf_causal_len, 0, M * sizeof(uint32_t));
    if (is_full) {
        int q_proj_dim = NUM_ATTN_HEADS * HEAD_DIM * 2;
        int q_dim = NUM_ATTN_HEADS * HEAD_DIM;
        int kv_dim = NUM_KV_HEADS * HEAD_DIM;
        const float *q_batch = (const float *)[ctx->buf_pf_qkv contents];
        const float *k_batch = (const float *)[ctx->buf_pf_kv contents];
        const float *v_batch = k_batch + (size_t)M * kv_dim;
        float *oproj_in = (float *)[ctx->buf_pf_oproj_in contents];

        for (uint32_t m = 0; m < M; m++) {
            int pos = pos_base + (int)m;
            // Per-position scratch (static, mirrors s_q / s_k_proj_out usage)
            static float pf_q[NUM_ATTN_HEADS * HEAD_DIM];
            static float pf_q_gate[NUM_ATTN_HEADS * HEAD_DIM];
            static float pf_k[NUM_KV_HEADS * HEAD_DIM];
            static float pf_v[NUM_KV_HEADS * HEAD_DIM];
            static float pf_attn_out[NUM_ATTN_HEADS * HEAD_DIM];

            const float *q_proj_m = q_batch + (size_t)m * q_proj_dim;
            const float *k_m = k_batch + (size_t)m * kv_dim;
            const float *v_m = v_batch + (size_t)m * kv_dim;

            for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                const float *src = q_proj_m + h * (2 * HEAD_DIM);
                memcpy(pf_q + h * HEAD_DIM, src, HEAD_DIM * sizeof(float));
                memcpy(pf_q_gate + h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
            }

            // Q/K RMSNorm
            uint16_t *qnorm_w = lc->q_norm_w;
            uint16_t *knorm_w = lc->k_norm_w;
            if (qnorm_w) {
                for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                    float *qh = pf_q + h * HEAD_DIM;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < HEAD_DIM; i++) sum_sq += qh[i] * qh[i];
                    float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
                    for (int i = 0; i < HEAD_DIM; i++) qh[i] = qh[i] * inv_rms * bf16_to_f32(qnorm_w[i]);
                }
            }
            memcpy(pf_k, k_m, kv_dim * sizeof(float));
            memcpy(pf_v, v_m, kv_dim * sizeof(float));
            if (knorm_w) {
                for (int h = 0; h < NUM_KV_HEADS; h++) {
                    float *kh = pf_k + h * HEAD_DIM;
                    float sum_sq = 0.0f;
                    for (int i = 0; i < HEAD_DIM; i++) sum_sq += kh[i] * kh[i];
                    float inv_rms = 1.0f / sqrtf(sum_sq / HEAD_DIM + RMS_NORM_EPS);
                    for (int i = 0; i < HEAD_DIM; i++) kh[i] = kh[i] * inv_rms * bf16_to_f32(knorm_w[i]);
                }
            }

            // RoPE
            apply_rotary_emb(pf_q, pf_k, pos, NUM_ATTN_HEADS, NUM_KV_HEADS, HEAD_DIM, ROTARY_DIM);

            // KV cache update (CPU + GPU mirror)
            int cache_pos = kv->len;
            kv_write(kv, cache_pos, pf_k, pf_v);

            int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
            if (ctx->attn_scores_pipe && fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                cache_pos < g_gpu_kv_seq) {  // mirror holds only [0, g_gpu_kv_seq)
                memcpy((float *)[ctx->buf_kv_k[fa_idx] contents] + cache_pos * kv_dim,
                       pf_k, kv_dim * sizeof(float));
                memcpy((float *)[ctx->buf_kv_v[fa_idx] contents] + cache_pos * kv_dim,
                       pf_v, kv_dim * sizeof(float));
            }
            kv->len++;

            // GQA attention — replicate the per-token branch selection exactly
            int heads_per_kv = NUM_ATTN_HEADS / NUM_KV_HEADS;
            float scale = 1.0f / sqrtf((float)HEAD_DIM);
            float *attn_out = pf_attn_out;
            memset(attn_out, 0, q_dim * sizeof(float));

            int gpu_attn_ready = (ctx->attn_scores_pipe &&
                                  fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS &&
                                  kv->len >= 32 && kv->len < g_gpu_kv_seq);

            if (batch_attn_ok && gpu_attn_ready) {
                // Batched GPU attention: stage q/gate, mark causal length.
                memcpy((float *)[ctx->buf_pf_attn_q contents] + (size_t)m * q_dim,
                       pf_q, q_dim * sizeof(float));
                memcpy((float *)[ctx->buf_pf_attn_gate contents] + (size_t)m * q_dim,
                       pf_q_gate, q_dim * sizeof(float));
                pf_causal_len[m] = (uint32_t)kv->len;  // = pos_base + m + 1
                if (pf_causal_len[m] > pf_sl_max) pf_sl_max = pf_causal_len[m];
            } else if (gpu_attn_ready) {
                // Per-position GPU attention in its own CB (same 4 kernels)
                memcpy([ctx->buf_attn_q contents], pf_q, q_dim * sizeof(float));
                memcpy([ctx->buf_attn_gate contents], pf_q_gate, q_dim * sizeof(float));
                id<MTLCommandBuffer> cmd_attn = [ctx->queue commandBuffer];
                uint32_t hd = HEAD_DIM;
                uint32_t kvd = (uint32_t)kv_dim;
                uint32_t sl = (uint32_t)kv->len;
                uint32_t seq_stride = (uint32_t)g_gpu_kv_seq;
                uint32_t hpkv = (uint32_t)heads_per_kv;
                {
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc setComputePipelineState:ctx->attn_scores_pipe];
                    [enc setBuffer:ctx->buf_attn_q          offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_kv_k[fa_idx]    offset:0 atIndex:1];
                    [enc setBuffer:ctx->buf_attn_scores     offset:0 atIndex:2];
                    [enc setBytes:&hd        length:4 atIndex:3];
                    [enc setBytes:&kvd       length:4 atIndex:4];
                    [enc setBytes:&sl        length:4 atIndex:5];
                    [enc setBytes:&seq_stride length:4 atIndex:6];
                    [enc setBytes:&scale     length:4 atIndex:7];
                    [enc setBytes:&hpkv      length:4 atIndex:8];
                    [enc setBytes:&sl        length:4 atIndex:9];
                    uint32_t total_tgs = sl * NUM_ATTN_HEADS;
                    [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc endEncoding];
                }
                {
                    // S6 fix: intra-CB syncs (same stale-L2 hazard class as
                    // the per-token CMD2 chain — see the comment there).
                    metal_sync_buffer(cmd_attn, ctx->buf_attn_scores);
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:ctx->attn_softmax_pipe];
                    [enc setBuffer:ctx->buf_attn_scores offset:0 atIndex:0];
                    [enc setBytes:&sl         length:4 atIndex:1];
                    [enc setBytes:&seq_stride length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(NUM_ATTN_HEADS, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                {
                    metal_sync_buffer(cmd_attn, ctx->buf_attn_scores);
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:ctx->attn_values_pipe];
                    [enc setBuffer:ctx->buf_attn_scores   offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_kv_v[fa_idx]  offset:0 atIndex:1];
                    [enc setBuffer:ctx->buf_attn_out      offset:0 atIndex:2];
                    [enc setBytes:&hd        length:4 atIndex:3];
                    [enc setBytes:&kvd       length:4 atIndex:4];
                    [enc setBytes:&sl        length:4 atIndex:5];
                    [enc setBytes:&seq_stride length:4 atIndex:6];
                    [enc setBytes:&hpkv      length:4 atIndex:7];
                    uint32_t total_threads = HEAD_DIM * NUM_ATTN_HEADS;
                    uint32_t tgs = (total_threads + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                {
                    metal_sync_buffer(cmd_attn, ctx->buf_attn_out);
                    uint32_t qdim = NUM_ATTN_HEADS * HEAD_DIM;
                    id<MTLComputeCommandEncoder> enc = [cmd_attn computeCommandEncoder];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc setComputePipelineState:ctx->sigmoid_gate_pipe];
                    [enc setBuffer:ctx->buf_attn_out  offset:0 atIndex:0];
                    [enc setBuffer:ctx->buf_attn_gate offset:0 atIndex:1];
                    [enc setBytes:&qdim length:4 atIndex:2];
                    uint32_t tgs = (qdim + 255) / 256;
                    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc endEncoding];
                }
                [cmd_attn commit];
                [cmd_attn waitUntilCompleted];
                memcpy(oproj_in + (size_t)m * q_dim, [ctx->buf_attn_out contents],
                       q_dim * sizeof(float));
            } else {
                // CPU attention
                if (batch_attn_ok) pf_causal_len[m] = 0;  // batched kernels skip this position
                for (int h = 0; h < NUM_ATTN_HEADS; h++) {
                    int kv_h = h / heads_per_kv;
                    float *qh = pf_q + h * HEAD_DIM;
                    float *scores = malloc(kv->len * sizeof(float));
                    for (int p = 0; p < kv->len; p++) {
                        static float kv_k_buf3[HEAD_DIM];
                        kv_read_k(kv, p, kv_h, kv_k_buf3);
                        float dot = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kv_k_buf3[d];
                        scores[p] = dot * scale;
                    }
                    cpu_softmax(scores, kv->len);
                    float *oh = attn_out + h * HEAD_DIM;
                    for (int p = 0; p < kv->len; p++) {
                        static float kv_v_buf3[HEAD_DIM];
                        kv_read_v(kv, p, kv_h, kv_v_buf3);
                        for (int d = 0; d < HEAD_DIM; d++) oh[d] += scores[p] * kv_v_buf3[d];
                    }
                    free(scores);
                }
                for (int i = 0; i < q_dim; i++) {
                    float g = 1.0f / (1.0f + expf(-pf_q_gate[i]));
                    attn_out[i] *= g;
                }
                memcpy(oproj_in + (size_t)m * q_dim, attn_out, q_dim * sizeof(float));
            }
        }
    }
    // (linear layers: buf_pf_oproj_in already filled by the GDN chain on GPU)
    if (g_chunk_timing_enabled && is_full) g_chunk_timing.attn_cpu += now_ms() - t_ph;

    // ===================== Phase A2: batched o_proj + residual + norm + routing =====================
    // cmdB is always a fresh command buffer now (linear layers commit cmdA
    // after the GDN chains for the CPU-copy bridge; full layers commit cmdA
    // above). The qkv_before_cmdB debug dump moved below the merged wait.
    id<MTLCommandBuffer> cmdB = [ctx->queue commandBuffer];
    if (g_chunk_timing_enabled) t_ph = now_ms();

    if (batch_attn_ok && pf_sl_max > 0) {
        // Batched GPU attention: all M queries in 4 dispatches (one per
        // kernel) in their OWN command buffer — the o_proj below must not
        // read the attention output device-to-device (same L2 hazard as
        // the linear chain→out_proj path), so this CB is waited and the
        // output is bridged through a CPU copy before o_proj encodes.
        // CPU-handled positions are skipped via causal_len = 0.
        id<MTLCommandBuffer> cmd_attn_b = [ctx->queue commandBuffer];
        int fa_idx = (layer_idx + 1) / FULL_ATTN_INTERVAL - 1;
        uint32_t hd = HEAD_DIM;
        uint32_t kvd = NUM_KV_HEADS * HEAD_DIM;
        uint32_t seq_stride = (uint32_t)g_gpu_kv_seq;
        float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
        uint32_t hpkv = NUM_ATTN_HEADS / NUM_KV_HEADS;
        uint32_t nh = NUM_ATTN_HEADS;
        uint32_t qdim = NUM_ATTN_HEADS * HEAD_DIM;
        uint32_t sl_max = pf_sl_max;
        uint32_t m32 = (uint32_t)M;
        {
            id<MTLComputeCommandEncoder> enc = [cmd_attn_b computeCommandEncoder];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->attn_scores_prefill_pipe];
            [enc setBuffer:ctx->buf_pf_attn_q       offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_kv_k[fa_idx]    offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_pf_attn_scores  offset:0 atIndex:2];
            [enc setBytes:&hd          length:4 atIndex:3];
            [enc setBytes:&kvd         length:4 atIndex:4];
            [enc setBytes:&seq_stride  length:4 atIndex:5];
            [enc setBytes:&attn_scale  length:4 atIndex:6];
            [enc setBytes:&hpkv        length:4 atIndex:7];
            [enc setBytes:&nh          length:4 atIndex:8];
            [enc setBytes:&sl_max      length:4 atIndex:9];
            [enc setBytes:pf_causal_len length:(NSUInteger)M * 4 atIndex:10];
            uint32_t total_tgs = m32 * sl_max * nh;
            [enc dispatchThreadgroups:MTLSizeMake(total_tgs, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Writer-side barrier: scores writes visible to softmax.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }
        metal_sync_buffer(cmdB, ctx->buf_pf_attn_scores);
        {
            id<MTLComputeCommandEncoder> enc = [cmd_attn_b computeCommandEncoder];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->attn_softmax_prefill_pipe];
            [enc setBuffer:ctx->buf_pf_attn_scores offset:0 atIndex:0];
            [enc setBytes:&seq_stride length:4 atIndex:1];
            [enc setBytes:&nh         length:4 atIndex:2];
            [enc setBytes:pf_causal_len length:(NSUInteger)M * 4 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(m32 * nh, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Writer-side barrier: softmax writes visible to values.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }
        metal_sync_buffer(cmdB, ctx->buf_pf_attn_scores);
        {
            id<MTLComputeCommandEncoder> enc = [cmd_attn_b computeCommandEncoder];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->attn_values_prefill_pipe];
            [enc setBuffer:ctx->buf_pf_attn_scores  offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_kv_v[fa_idx]    offset:0 atIndex:1];
            [enc setBuffer:ctx->buf_pf_oproj_in     offset:0 atIndex:2];
            [enc setBytes:&hd         length:4 atIndex:3];
            [enc setBytes:&kvd        length:4 atIndex:4];
            [enc setBytes:&seq_stride length:4 atIndex:5];
            [enc setBytes:&hpkv       length:4 atIndex:6];
            [enc setBytes:&nh         length:4 atIndex:7];
            [enc setBytes:pf_causal_len length:(NSUInteger)M * 4 atIndex:8];
            uint32_t total_threads = m32 * nh * hd;
            [enc dispatchThreadgroups:MTLSizeMake((total_threads + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Writer-side barrier: values writes visible to sigmoid.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }
        metal_sync_buffer(cmdB, ctx->buf_pf_oproj_in);
        {
            id<MTLComputeCommandEncoder> enc = [cmd_attn_b computeCommandEncoder];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:ctx->sigmoid_gate_prefill_pipe];
            [enc setBuffer:ctx->buf_pf_oproj_in   offset:0 atIndex:0];
            [enc setBuffer:ctx->buf_pf_attn_gate  offset:0 atIndex:1];
            [enc setBytes:&qdim       length:4 atIndex:2];
            [enc setBytes:pf_causal_len length:(NSUInteger)M * 4 atIndex:3];
            uint32_t total_threads = m32 * qdim;
            [enc dispatchThreadgroups:MTLSizeMake((total_threads + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            // Writer-side barrier: sigmoid writes visible to o_proj.
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc endEncoding];
        }
        pf_note_gap(&g_chunk_timing.cmdB_gap);
        [cmd_attn_b commit];
        [cmd_attn_b waitUntilCompleted];
        pf_note_wait_done();
    }
    if (is_full) {
        // Bridge copy for BOTH attention paths (CPU-written or batched):
        // o_proj reads buf_pf_oproj_in2 only — no dev-to-dev L2 dependency.
        memcpy([ctx->buf_pf_oproj_in2 contents], [ctx->buf_pf_oproj_in contents],
               (size_t)M * (NUM_ATTN_HEADS * HEAD_DIM) * sizeof(float));
    }

    if (is_full) {
        gpu_encode_prefill_matvec(ctx, cmdB, lc->o_w, lc->o_s, lc->o_b,
            ctx->buf_pf_oproj_in2, ctx->buf_pf_oproj,
            HIDDEN_DIM, NUM_ATTN_HEADS * HEAD_DIM, GROUP_SIZE, lc->o_bits, M, 0);
    } else {
        // Linear out_proj (reads the bridge copy of the chains' output)
        gpu_encode_prefill_matvec(ctx, cmdB, lc->out_proj_w, lc->out_proj_s, lc->out_proj_b,
            ctx->buf_pf_oproj_in2, ctx->buf_pf_oproj,
            HIDDEN_DIM, LINEAR_TOTAL_VALUE, GROUP_SIZE, lc->out_proj_bits, M, 0);
    }


    // Residual + post-attn norm (batched, one TG per position)
    {
        // Phase C S4: GGUF norm weights are staged BF16 in the heap stage
        // buffer — bind the stage mirror at the stage offset (the wf_buf
        // pointer diff would be garbage: wf_buf is nil in GGUF mode).
        NSUInteger norm_off;
        id<MTLBuffer> norm_buf;
        if (g_gguf_stage) {
            norm_buf = ctx->gguf_stage_gpu;
            norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                    (const char *)g_gguf_stage);
        } else {
            norm_buf = ctx->wf_buf;
            norm_off = (NSUInteger)((const char *)lc->post_attn_norm_w -
                                    (const char *)[ctx->wf_buf contents]);
        }
        // Order after the o_proj matvec (and GDN chains) in the same CB —
        // same stale-cache-line hazard as gpu_encode_prefill_matvec.
        metal_sync_buffer(cmdB, ctx->buf_pf_oproj);
        id<MTLComputeCommandEncoder> enc = [cmdB computeCommandEncoder];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        uint32_t dim = HIDDEN_DIM;
        float eps = RMS_NORM_EPS;
        id<MTLBuffer> resid_buf = (layer_idx == 0) ? ctx->buf_pf_residual : ctx->buf_pf_moe_hidden;
        [enc setComputePipelineState:ctx->residual_norm_fused_prefill];
        [enc setBuffer:resid_buf            offset:0       atIndex:0];
        [enc setBuffer:ctx->buf_pf_oproj    offset:0       atIndex:1];
        [enc setBuffer:norm_buf             offset:norm_off atIndex:2];
        [enc setBuffer:ctx->buf_pf_h_mid    offset:0       atIndex:3];
        [enc setBuffer:ctx->buf_pf_h_post   offset:0       atIndex:4];
        [enc setBytes:&dim length:4 atIndex:5];
        [enc setBytes:&eps length:4 atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        // Writer-side barrier: h_mid/h_post writes visible to the routing
        // matmuls that follow in the same CB.
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [enc endEncoding];
    }

    // Routing matmuls (batched): gate/seg BF16, sg/su 4-bit
    metal_sync_buffer(cmdB, ctx->buf_pf_h_post);
    gpu_encode_prefill_matvec(ctx, cmdB, lc->gate_w, lc->gate_s, lc->gate_b,
        ctx->buf_pf_h_post, ctx->buf_pf_gate_scores,
        NUM_EXPERTS, HIDDEN_DIM, GROUP_SIZE, lc->gate_bits, M, 0);
    gpu_encode_prefill_matvec(ctx, cmdB, lc->seg_w, lc->seg_s, lc->seg_b,
        ctx->buf_pf_h_post, ctx->buf_pf_seg,
        1, HIDDEN_DIM, GROUP_SIZE, lc->seg_bits, M, 0);
    gpu_encode_prefill_matvec(ctx, cmdB, lc->sg_w, lc->sg_s, lc->sg_b,
        ctx->buf_pf_h_post, ctx->buf_pf_shared,
        SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, lc->sg_bits, M, 0);
    // up region after gate region: [M*512, 2*M*512)
    gpu_encode_prefill_matvec(ctx, cmdB, lc->su_w, lc->su_s, lc->su_b,
        ctx->buf_pf_h_post, ctx->buf_pf_shared,
        SHARED_INTERMEDIATE, HIDDEN_DIM, GROUP_SIZE, lc->su_bits, M,
        (NSUInteger)M * SHARED_INTERMEDIATE * sizeof(float));
    if (g_chunk_timing_enabled) {
        g_chunk_timing.cmdB_encode += now_ms() - t_ph;
        t_ph = now_ms();
    }
    pf_note_gap(&g_chunk_timing.cmdB_gap);
    [cmdB commit];
    [cmdB waitUntilCompleted];
    pf_note_wait_done();
    if (g_chunk_timing_enabled) {
        double d = now_ms() - t_ph;
        g_chunk_timing.cmdB_wait += d;
        pf_per_layer_add(layer_idx, 5, d);
    }

    // Debug dumps that read cmdA/cmdB-written buffers (moved below the merged
    // wait — for linear layers cmdA is no longer waited mid-layer).
    if (getenv("FINCHMOE_PF_DUMP") && chunk_base == 0 && layer_idx == 0) {
        static FILE *pf_emb2 = NULL;
        if (!pf_emb2) pf_emb2 = fopen("/tmp/embed_integrity.bin", "ab");
        if (pf_emb2) {
            static FILE *pf_qa = NULL;
            if (!pf_qa) pf_qa = fopen("/tmp/qkv_after_cmdA.bin", "wb");
            if (pf_qa) {
                fwrite((const float *)[ctx->buf_pf_qkv contents] + LINEAR_CONV_DIM,
                       sizeof(float), LINEAR_CONV_DIM, pf_qa);
                fflush(pf_qa);
            }
            fwrite(embed_batch, sizeof(float), 6, pf_emb2);   // after merged wait
        }
    }

    // Debug: parity dump for layer 0 position 0, fused path (mirrors the
    // cmd12_fused FINCHMOE_DUMP_STAGES layout: qkv, z, beta, alpha, conv,
    // delta, gated — conv/delta are stale buffers in the fused path).
    if (!is_full && layer_idx == 0 && chunk_base == 0 && getenv("FINCHMOE_DUMP_STAGES")) {
        static FILE *pf_sf = NULL;
        if (!pf_sf) pf_sf = fopen("/tmp/stage_pf.bin", "wb");
        if (pf_sf) {
            fwrite((const float *)[ctx->buf_pf_qkv contents], sizeof(float), LINEAR_CONV_DIM, pf_sf);
            fwrite((const float *)[ctx->buf_pf_z contents], sizeof(float), LINEAR_TOTAL_VALUE, pf_sf);
            fwrite((const float *)[ctx->buf_pf_ba contents], sizeof(float), LINEAR_NUM_V_HEADS, pf_sf);
            fwrite((const float *)[ctx->buf_pf_ba contents] + (size_t)M * LINEAR_NUM_V_HEADS, sizeof(float), LINEAR_NUM_V_HEADS, pf_sf);
            static float pf_zeros[8192] = {0};
            fwrite(pf_zeros, sizeof(float), LINEAR_CONV_DIM, pf_sf);
            fwrite(pf_zeros, sizeof(float), LINEAR_TOTAL_VALUE, pf_sf);
            fwrite((const float *)[ctx->buf_pf_oproj_in contents], sizeof(float), LINEAR_TOTAL_VALUE, pf_sf);
            fwrite((const float *)[ctx->buf_pf_oproj contents], sizeof(float), HIDDEN_DIM, pf_sf);   // o_proj
            fwrite((const float *)[ctx->buf_pf_h_mid contents], sizeof(float), HIDDEN_DIM, pf_sf);    // h_mid
            fwrite((const float *)[ctx->buf_pf_h_post contents], sizeof(float), HIDDEN_DIM, pf_sf);   // h_post
            fflush(pf_sf);
        }
    }

    if (getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && chunk_base == 0) {
        static FILE *pf_qb = NULL;
        if (!pf_qb) pf_qb = fopen("/tmp/qkv_before_cmdB.bin", "wb");
        if (pf_qb) {
            fwrite((const float *)[ctx->buf_pf_qkv contents] + LINEAR_CONV_DIM,
                   sizeof(float), LINEAR_CONV_DIM, pf_qb);
            fflush(pf_qb);
        }
    }

    if (getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && chunk_base <= 1) {
        static FILE *pf_qa2 = NULL;
        if (!pf_qa2) pf_qa2 = fopen("/tmp/qkv_after_cmdB.bin", "wb");
        if (pf_qa2) {
            fwrite((const float *)[ctx->buf_pf_qkv contents] + LINEAR_CONV_DIM,
                   sizeof(float), LINEAR_CONV_DIM, pf_qa2);
            fflush(pf_qa2);
        }
        static FILE *pf_hp2 = NULL;
        if (!pf_hp2) pf_hp2 = fopen("/tmp/stage_pf2.bin", "wb");
        if (pf_hp2) {
            fprintf(stderr, "[PF-PTR] qkv=%p oproj_in=%p input=%p resid=%p h_mid=%p h_post=%p shared=%p gate_scores=%p\n",
                    (void *)[ctx->buf_pf_qkv contents], (void *)[ctx->buf_pf_oproj_in contents],
                    (void *)[ctx->buf_pf_input contents], (void *)[ctx->buf_pf_residual contents],
                    (void *)[ctx->buf_pf_h_mid contents], (void *)[ctx->buf_pf_h_post contents],
                    (void *)[ctx->buf_pf_shared contents], (void *)[ctx->buf_pf_gate_scores contents]);
            for (uint32_t dm = 0; dm < 2; dm++) {
                fwrite((const float *)[ctx->buf_pf_z contents] + (size_t)dm * LINEAR_TOTAL_VALUE,
                       sizeof(float), LINEAR_TOTAL_VALUE, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_ba contents] + (size_t)dm * LINEAR_NUM_V_HEADS,
                       sizeof(float), LINEAR_NUM_V_HEADS, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_ba contents] + (size_t)(M + dm) * LINEAR_NUM_V_HEADS,
                       sizeof(float), LINEAR_NUM_V_HEADS, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_qkv contents] + (size_t)dm * LINEAR_CONV_DIM,
                       sizeof(float), LINEAR_CONV_DIM, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_oproj_in contents] + (size_t)dm * LINEAR_TOTAL_VALUE,
                       sizeof(float), LINEAR_TOTAL_VALUE, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_oproj contents] + (size_t)dm * HIDDEN_DIM,
                       sizeof(float), HIDDEN_DIM, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_h_mid contents] + (size_t)dm * HIDDEN_DIM,
                       sizeof(float), HIDDEN_DIM, pf_hp2);
                fwrite((const float *)[ctx->buf_pf_h_post contents] + (size_t)dm * HIDDEN_DIM,
                       sizeof(float), HIDDEN_DIM, pf_hp2);
            }
            fflush(pf_hp2);
        }
    }

    if (0 && getenv("FINCHMOE_PF_DUMP") && layer_idx == 0 && chunk_base == 0) {
        static FILE *pf_hp = NULL;
        if (!pf_hp) pf_hp = fopen("/tmp/hpost_slot0.bin", "wb");
        if (pf_hp) {
            fwrite((const float *)[ctx->buf_pf_oproj_in contents], sizeof(float), LINEAR_TOTAL_VALUE, pf_hp);
            fwrite((const float *)[ctx->buf_pf_h_post contents], sizeof(float), HIDDEN_DIM, pf_hp);
            fwrite((const float *)[ctx->buf_pf_shared contents], sizeof(float), SHARED_INTERMEDIATE, pf_hp);
            fwrite((const float *)[ctx->buf_pf_shared contents] + (size_t)M * SHARED_INTERMEDIATE,
                   sizeof(float), SHARED_INTERMEDIATE, pf_hp);
            fflush(pf_hp);
        }
    }

    // Debug: parity dump part 2 (o_proj/h_mid/h_post — mirrors sf2 in the
    // per-token path), appended for layer 0 position 0.
    if (!is_full && layer_idx == 0 && chunk_base == 0 && getenv("FINCHMOE_DUMP_STAGES")) {
        static FILE *pf_sf2 = NULL;
        if (!pf_sf2) pf_sf2 = fopen("/tmp/stage_pf.bin", "ab");
        if (pf_sf2) {
            fwrite(embed_batch + (size_t)chunk_base * HIDDEN_DIM, sizeof(float), HIDDEN_DIM, pf_sf2);
            fwrite((const float *)[ctx->buf_pf_residual contents], sizeof(float), HIDDEN_DIM, pf_sf2);
            fwrite((const float *)[ctx->buf_pf_oproj contents], sizeof(float), HIDDEN_DIM, pf_sf2);
            fwrite((const float *)[ctx->buf_pf_h_mid contents], sizeof(float), HIDDEN_DIM, pf_sf2);
            fwrite((const float *)[ctx->buf_pf_h_post contents], sizeof(float), HIDDEN_DIM, pf_sf2);
            fflush(pf_sf2);
        }
    }

    // Debug: gate-scores parity dump (chunk 0, layer 0)
    if (getenv("FINCHMOE_PF_DUMP") && chunk_base == 0 && layer_idx == 0) {
        static FILE *pf_gs_new = NULL;
        if (!pf_gs_new) pf_gs_new = fopen("/tmp/gates_new.bin", "wb");
        if (pf_gs_new) {
            const float *gs = (const float *)[ctx->buf_pf_gate_scores contents];
            const float *sh = (const float *)[ctx->buf_pf_shared contents];
            const float *sg = (const float *)[ctx->buf_pf_seg contents];
            fwrite(gs, sizeof(float), NUM_EXPERTS, pf_gs_new);
            fwrite(sh, sizeof(float), SHARED_INTERMEDIATE, pf_gs_new);
            fwrite(sh + (size_t)M * SHARED_INTERMEDIATE, sizeof(float), SHARED_INTERMEDIATE, pf_gs_new);
            fwrite(sg, sizeof(float), 1, pf_gs_new);
            fflush(pf_gs_new);
        }
    }

    // CPU readbacks for Phase B
    const float *gate_scores_batch = (const float *)[ctx->buf_pf_gate_scores contents];
    const float *seg_batch = (const float *)[ctx->buf_pf_seg contents];
    const float *shared_batch = (const float *)[ctx->buf_pf_shared contents];
    const float *h_post_batch = (const float *)[ctx->buf_pf_h_post contents];

    // Debug: per-(chunk, layer) Phase-B input trace (FINCHMOE_DUMP_PHASEB).
    // Accumulated in memory and written once at the end of the prefill —
    // per-layer fwrite/fflush perturbed the CPU/GPU interleave enough to
    // mask the race this trace exists to catch.
    if (getenv("FINCHMOE_DUMP_PHASEB")) {
        static size_t pb_cap = 0;
        size_t need = 3 + (size_t)M * (HIDDEN_DIM + NUM_EXPERTS + 1 + 4 * HIDDEN_DIM)
                    + (size_t)M * (LINEAR_CONV_DIM + LINEAR_TOTAL_VALUE + 64)
                    + (size_t)M * LINEAR_TOTAL_VALUE + 2;
        if (g_pb_len + need > pb_cap) {
            size_t ncap = pb_cap ? pb_cap * 2 : 1 << 20;
            while (ncap < g_pb_len + need) ncap *= 2;
            g_pb_acc = realloc(g_pb_acc, ncap * sizeof(float));
            pb_cap = ncap;
        }
        int32_t hdr[3] = { (int32_t)chunk_base, (int32_t)layer_idx, (int32_t)M };
        memcpy(g_pb_acc + g_pb_len, hdr, 3 * sizeof(int32_t));
        g_pb_len += 3;
        memcpy(g_pb_acc + g_pb_len, h_post_batch, (size_t)M * HIDDEN_DIM * sizeof(float));
        g_pb_len += (size_t)M * HIDDEN_DIM;
        memcpy(g_pb_acc + g_pb_len, gate_scores_batch, (size_t)M * NUM_EXPERTS * sizeof(float));
        g_pb_len += (size_t)M * NUM_EXPERTS;
        memcpy(g_pb_acc + g_pb_len, seg_batch, M * sizeof(float));
        g_pb_len += M;
        // Post-cmdB boundary values (all GPU-written, completed by the
        // cmdA/B wait above) for the CMD3-vs-cmdA divergence split:
        // moe_hidden = CMD3(L-1) combine out, oproj = layer L o_proj out,
        // h_mid = resid+oproj, input = normed hidden fed to layer L's cmdA.
        const float *moe_b = (const float *)[ctx->buf_pf_moe_hidden contents];
        const float *opr_b = (const float *)[ctx->buf_pf_oproj contents];
        const float *mid_b = (const float *)[ctx->buf_pf_h_mid contents];
        const float *inp_b = (const float *)[ctx->buf_pf_input contents];
        memcpy(g_pb_acc + g_pb_len, moe_b, (size_t)M * HIDDEN_DIM * sizeof(float));
        g_pb_len += (size_t)M * HIDDEN_DIM;
        memcpy(g_pb_acc + g_pb_len, opr_b, (size_t)M * HIDDEN_DIM * sizeof(float));
        g_pb_len += (size_t)M * HIDDEN_DIM;
        memcpy(g_pb_acc + g_pb_len, mid_b, (size_t)M * HIDDEN_DIM * sizeof(float));
        g_pb_len += (size_t)M * HIDDEN_DIM;
        memcpy(g_pb_acc + g_pb_len, inp_b, (size_t)M * HIDDEN_DIM * sizeof(float));
        g_pb_len += (size_t)M * HIDDEN_DIM;
        // Chain-stage inputs (matvec outputs per slot) + recurrent-state
        // fingerprints: qkv/z/ba per slot show whether the chain's INPUTS
        // were corrupted; the 16-float state fingerprints (8 delta + 8 conv)
        // show whether THIS layer's chains corrupted the recurrent state.
        memcpy(g_pb_acc + g_pb_len, (const float *)[ctx->buf_pf_qkv contents],
               (size_t)M * LINEAR_CONV_DIM * sizeof(float));
        g_pb_len += (size_t)M * LINEAR_CONV_DIM;
        memcpy(g_pb_acc + g_pb_len, (const float *)[ctx->buf_pf_z contents],
               (size_t)M * LINEAR_TOTAL_VALUE * sizeof(float));
        g_pb_len += (size_t)M * LINEAR_TOTAL_VALUE;
        memcpy(g_pb_acc + g_pb_len, (const float *)[ctx->buf_pf_ba contents],
               (size_t)M * 64 * sizeof(float));
        g_pb_len += (size_t)M * 64;
        // Chain OUTPUT (oproj_in per slot) + full recurrent-state FNV hashes
        // (8 floats is too weak a fingerprint — a stale state read can be
        // anywhere in the 2MB delta matrix).
        memcpy(g_pb_acc + g_pb_len, (const float *)[ctx->buf_pf_oproj_in contents],
               (size_t)M * LINEAR_TOTAL_VALUE * sizeof(float));
        g_pb_len += (size_t)M * LINEAR_TOTAL_VALUE;
        if (!is_full && linear_layer_idx >= 0 && linear_layer_idx < NUM_LINEAR_LAYERS) {
            static uint32_t h1 = 0, h2 = 0;
            const float *ds = (const float *)[ctx->buf_delta_state[linear_layer_idx] contents];
            const float *cs = (const float *)[ctx->buf_conv_state[linear_layer_idx] contents];
            uint32_t ha = 2166136261u, hb = 2166136261u;
            size_t dn = 32 * 128 * 128, cn = 3 * LINEAR_CONV_DIM;
            for (size_t i = 0; i < dn; i++) {
                uint32_t b; memcpy(&b, ds + i, 4);
                ha = (ha ^ b) * 16777619u;
            }
            for (size_t i = 0; i < cn; i++) {
                uint32_t b; memcpy(&b, cs + i, 4);
                hb = (hb ^ b) * 16777619u;
            }
            h1 = ha; h2 = hb;
            memcpy(g_pb_acc + g_pb_len, &h1, 4);
            g_pb_len += 1;
            memcpy(g_pb_acc + g_pb_len, &h2, 4);
            g_pb_len += 1;
        } else {
            memset(g_pb_acc + g_pb_len, 0, 8);
            g_pb_len += 2;
        }
    }

    // ===================== Phase B: per-position experts =====================
    static float pf_expert_weights[MAX_K];
    static int pf_valid[MAX_K];

    // Pool mode: routing pass for all M + ONE batched pread + M back-to-back
    // CMD3s (no backpressure — every buffer is per-position-disjoint).
    // With hot sets: layer L+1's hot experts are prefetched NOW (during this
    // layer's routing/preads/CMD3s) into pool[(L+1)%2]; this layer's hot
    // experts were prefetched during layer L-1 into pool[L%2].
    int pool_ok = (g_pf_pool_slots >= (int)(8 * M)) && !g_use_int8 && ctx->buf_pool_expert_data;
    if (g_gguf_stage) {
        // ---- Phase C S4: GGUF expert dispatch ----
        // Uniform group-pool path: positions process in groups of
        // G = pool_slots/8 (the 4MB copy pool covers G positions' experts
        // at once). Per group: routing → 3 slab preads per expert into the
        // slots → per-position CMD3s (S2 kernels reading the stable pool
        // buffers) → wait for the group's last CMD3 before the next group's
        // preads overwrite the slots. No per-position waits inside a group
        // (slots are per-position-disjoint); the final group keeps its
        // deferred overlap with the next layer's cmdA.
        static int pf_idx[PREFILL_CHUNK_MAX][MAX_K];
        static float pf_w[PREFILL_CHUNK_MAX][MAX_K];
        static id<MTLBuffer> __unsafe_unretained pf_data_bufs[PREFILL_CHUNK_MAX * MAX_K];
        static NSUInteger pf_data_offs[PREFILL_CHUNK_MAX * MAX_K];
        static int pf_valid_all[PREFILL_CHUNK_MAX * MAX_K];
        static int fds[PREFILL_CHUNK_MAX * MAX_K * 3];
        static off_t offs[PREFILL_CHUNK_MAX * MAX_K * 3];
        static void *dsts[PREFILL_CHUNK_MAX * MAX_K * 3];
        static size_t sizes[PREFILL_CHUNK_MAX * MAX_K * 3];
        int actual_K = (K > MAX_K) ? MAX_K : K;
        GgufExpertInfo *ge = &gguf_experts[layer_idx];
        NSUInteger exp_alloc = g_gguf_exp_alloc;
        int G = g_pf_pool_slots_gguf / MAX_K;
        if (G < 1) G = 1;
        if (G > (int)M) G = (int)M;
        if (getenv("FINCHMOE_PF_NOPOOL")) G = 1;   // debug: force per-position groups
        // Phase C S6: ping-pong pool groups. Split the pool into two
        // halves; consecutive groups pread into alternate halves, so group
        // g+1's routing+preads overlap CMD3(g) on the GPU instead of idling
        // it (a ~2.4ms gap before a CMD3 commit pays a GPU wake tax —
        // FINCHMOE_CBLAT curve). Measured NEUTRAL on M4 (the wake saving is
        // offset by the doubled per-CB dispatch cost of 2 smaller CMD3s), so
        // it stays opt-in (FINCHMOE_PF_PINGPONG=1). Bitwise-verified.
        static int pingpong = -1;
        if (pingpong < 0) pingpong = getenv("FINCHMOE_PF_PINGPONG") ? 1 : 0;
        int Gpp = G, half_slots = 0;
        if (pingpong && G >= 2) {
            Gpp = G / 2;
            half_slots = Gpp * MAX_K;   // slots per half
        }

        for (uint32_t gbase = 0; gbase < M; gbase += (uint32_t)Gpp) {
            int gM = ((gbase + (uint32_t)Gpp) <= M) ? Gpp : (int)(M - gbase);
            NSUInteger half_off = ((NSUInteger)(gbase / (uint32_t)Gpp) & 1) *
                                  (NSUInteger)half_slots * exp_alloc;
            char *pool_base = (char *)[ctx->buf_pool_expert_data_gguf contents];

            // Pass 1: routing + staging for this group's positions
            for (int gm = 0; gm < gM; gm++) {
                uint32_t m = gbase + (uint32_t)gm;
                double t_route = 0;
                if (g_chunk_timing_enabled) t_route = now_ms();
                float gate_scores[NUM_EXPERTS];
                memcpy(gate_scores, gate_scores_batch + (size_t)m * NUM_EXPERTS,
                       NUM_EXPERTS * sizeof(float));
                cpu_softmax(gate_scores, NUM_EXPERTS);
                int expert_indices[MAX_K];
                float expert_weights[MAX_K];
                cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
                cpu_normalize_weights(expert_weights, K);
                if (g_chunk_timing_enabled) {
                    double d = now_ms() - t_route;
                    g_chunk_timing.routing_cpu += d;
                    pf_per_layer_add(layer_idx, 6, d);
                }
                if (g_freq_tracking) {
                    for (int k = 0; k < K; k++) {
                        g_expert_freq[layer_idx][expert_indices[k]]++;
                    }
                    if (layer_idx == 0) g_freq_total_tokens++;
                }
                for (int k = 0; k < MAX_K; k++) {
                    pf_idx[m][k] = expert_indices[k];
                    pf_w[m][k] = expert_weights[k];
                }
                memcpy((float *)[ctx->buf_pf_expert_input contents] + (size_t)m * HIDDEN_DIM,
                       h_post_batch + (size_t)m * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
                memcpy((float *)[ctx->buf_pf_shared_gate contents] + (size_t)m * SHARED_INTERMEDIATE,
                       shared_batch + (size_t)m * SHARED_INTERMEDIATE,
                       SHARED_INTERMEDIATE * sizeof(float));
                memcpy((float *)[ctx->buf_pf_shared_up contents] + (size_t)m * SHARED_INTERMEDIATE,
                       shared_batch + (size_t)M * SHARED_INTERMEDIATE + (size_t)m * SHARED_INTERMEDIATE,
                       SHARED_INTERMEDIATE * sizeof(float));
            }

            // Phase C S6 probe (FINCHMOE_PF_NOPREAD): skip the pool/preads
            // entirely — CMD3 binds the expert slabs via mmap wraps.
            static int nopread = -1;
            if (nopread < 0) {
                nopread = getenv("FINCHMOE_PF_NOPREAD") ? 1 : 0;
                if (nopread) fprintf(stderr, "[nopread] CMD3 reads expert slabs directly from the GGUF mmap (no pool)\n");
            }
            if (nopread) {
                for (int gm = 0; gm < gM; gm++)
                    for (int k = 0; k < actual_K; k++)
                        pf_valid_all[((size_t)(gbase + gm)) * MAX_K + k] = 1;
                // madvise(WILLNEED) on every unique routed expert's slabs —
                // async kernel readahead replaces the blocking pool preads.
                // Resident pages are DART-mapped, so the CMD3 kernels read
                // them without the catastrophic GPU fault path.
                double t_wl = 0;
                if (g_chunk_timing_enabled) t_wl = now_ms();
                for (int gm = 0; gm < gM; gm++) {
                    for (int k = 0; k < actual_K; k++) {
                        uint32_t e = (uint32_t)pf_idx[(size_t)(gbase + gm)][k];
                        int seen = 0;
                        for (int gm2 = 0; gm2 < gm && !seen; gm2++)
                            for (int k2 = 0; k2 < actual_K && !seen; k2++)
                                if ((uint32_t)pf_idx[(size_t)(gbase + gm2)][k2] == e) seen = 1;
                        for (int k2 = 0; k2 < k && !seen; k2++)
                            if ((uint32_t)pf_idx[(size_t)(gbase + gm)][k2] == e) seen = 1;
                        if (seen) continue;
                        madvise((char *)g_gguf_data_base + ge->gate_off + (size_t)e * ge->gate_slab,
                                ge->gate_slab, MADV_WILLNEED);
                        madvise((char *)g_gguf_data_base + ge->up_off + (size_t)e * ge->up_slab,
                                ge->up_slab, MADV_WILLNEED);
                        madvise((char *)g_gguf_data_base + ge->down_off + (size_t)e * ge->down_slab,
                                ge->down_slab, MADV_WILLNEED);
                    }
                }
                if (g_chunk_timing_enabled) {
                    double d = now_ms() - t_wl;
                    g_chunk_timing.pread_wait += d;   // same bucket: replaces the preads
                    pf_per_layer_add(layer_idx, 7, d);
                }
                double t_c3e = 0;
                if (g_chunk_timing_enabled) t_c3e = now_ms();
                id<MTLCommandBuffer> group_cb = [ctx->queue commandBuffer];
                [group_cb encodeWaitForEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
                if (prefill_chunk_cmd3_batch_nopread(ctx, group_cb, layer_idx, gbase, (uint32_t)gM,
                                                     actual_K, pf_valid_all, pf_idx) == 0) {
                    for (int gm = 0; gm < gM; gm++) pf_cmd3_slots[gbase + (uint32_t)gm] = group_cb;
                    pf_note_gap(&g_chunk_timing.cmd3_gap);
                    [group_cb commit];
                    if (g_chunk_timing_enabled) {
                        double d = now_ms() - t_c3e;
                        g_chunk_timing.cmd3_encode += d;
                        pf_per_layer_add(layer_idx, 8, d);
                    }
                    continue;   // no pool writes → no backpressure wait
                }
                // wrap alloc / missing pipe → fall through to the pool path
            }

            // 3 slab preads per expert into a pool slot — deduped across the
            // group (Phase C S4 perf pass): positions sharing an expert read
            // ONE copy into ONE slot, and their CMD3s share the slot (the S2
            // kernels only READ the pool, so sharing is safe). Pool bytes are
            // bitwise identical to the per-position layout, so parity with
            // the per-token baseline is preserved. Both the pread copy
            // traffic and the CMD3 dequant reads scale with unique experts.
            // Escape hatch: FINCHMOE_PF_NODEDUP forces the old layout.
            {
                static int pf_seen_e[PREFILL_CHUNK_MAX * MAX_K];
                static int pf_uniq_owner[PREFILL_CHUNK_MAX * MAX_K];
                static int pf_uniq_valid[PREFILL_CHUNK_MAX * MAX_K];
                int dedup = !getenv("FINCHMOE_PF_NODEDUP");
                int n_seen = 0, n_read = 0;
                for (int gm = 0; gm < gM; gm++) {
                    uint32_t m = gbase + (uint32_t)gm;
                    for (int k = 0; k < actual_K; k++) {
                        int e = pf_idx[m][k];
                        int owner = -1;
                        if (dedup) {
                            for (int s = 0; s < n_seen; s++) {
                                if (pf_seen_e[s] == e) { owner = s; break; }
                            }
                        }
                        if (owner < 0) {
                            owner = n_seen++;
                            pf_seen_e[owner] = e;
                            size_t slot = (size_t)half_off + (size_t)owner * exp_alloc;
                            // PERF PROBE (FINCHMOE_PF_DOWNFIRST): write the
                            // down slab FIRST so the SLC tail holds gate+up
                            // instead (tests whether the down kernel's
                            // SLC hits are a write-order artifact).
                            static int downfirst = -1;
                            if (downfirst < 0) downfirst = getenv("FINCHMOE_PF_DOWNFIRST") ? 1 : 0;
                            const off_t slab_gate = (off_t)(ge->gate_off + (size_t)e * ge->gate_slab);
                            const off_t slab_up   = (off_t)(ge->up_off + (size_t)e * ge->up_slab);
                            const off_t slab_down = (off_t)(ge->down_off + (size_t)e * ge->down_slab);
                            if (!downfirst) {
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_gate;
                                dsts[n_read] = pool_base + slot;
                                sizes[n_read] = ge->gate_slab;
                                n_read++;
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_up;
                                dsts[n_read] = pool_base + slot + ge->gate_slab;
                                sizes[n_read] = ge->up_slab;
                                n_read++;
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_down;
                                dsts[n_read] = pool_base + slot + ge->gate_slab + ge->up_slab;
                                sizes[n_read] = ge->down_slab;
                                n_read++;
                            } else {
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_down;
                                dsts[n_read] = pool_base + slot + ge->gate_slab + ge->up_slab;
                                sizes[n_read] = ge->down_slab;
                                n_read++;
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_up;
                                dsts[n_read] = pool_base + slot + ge->gate_slab;
                                sizes[n_read] = ge->up_slab;
                                n_read++;
                                fds[n_read] = g_gguf_fd;
                                offs[n_read] = slab_gate;
                                dsts[n_read] = pool_base + slot;
                                sizes[n_read] = ge->gate_slab;
                                n_read++;
                            }
                        }
                        pf_data_bufs[(size_t)m * MAX_K + k] = ctx->buf_pool_expert_data_gguf;
                        pf_data_offs[(size_t)m * MAX_K + k] = half_off + (NSUInteger)owner * exp_alloc;
                        pf_uniq_owner[(size_t)m * MAX_K + k] = owner;
                    }
                }
                double t_pread = 0;
                if (g_chunk_timing_enabled) t_pread = now_ms();
                // Ping-pong: before preads overwrite THIS group's half, the
                // same half's previous CMD3 (group gi-2) must have finished
                // reading the pool. (Group gi-1 lives on the other half —
                // no wait for it; its CMD3 overlaps these preads.)
                if (pingpong && gbase >= 2 * (uint32_t)Gpp &&
                    pf_cmd3_slots[gbase - (uint32_t)Gpp - 1]) {
                    [pf_cmd3_slots[gbase - (uint32_t)Gpp - 1] waitUntilCompleted];
                    pf_note_wait_done();
                }
                async_pread_multi_start(fds, offs, dsts, sizes, n_read);
                async_pread_wait();
                // validity: all 3 reads of an expert must have completed —
                // per unique owner, then fanned out to the (m, k) entries.
                for (int s = 0; s < n_seen; s++) {
                    int v = 1;
                    for (int r = 0; r < 3; r++) {
                        if (!g_async_pread.valid[s * 3 + r]) { v = 0; break; }
                    }
                    pf_uniq_valid[s] = v;
                }
                for (int gm = 0; gm < gM; gm++) {
                    uint32_t m = gbase + (uint32_t)gm;
                    for (int k = 0; k < actual_K; k++) {
                        pf_valid_all[(size_t)m * MAX_K + k] =
                            pf_uniq_valid[pf_uniq_owner[(size_t)m * MAX_K + k]];
                    }
                }
                if (g_chunk_timing_enabled) {
                    g_gguf_dedup_unique += n_seen;
                    g_gguf_dedup_slots += (int)gM * actual_K;
                }
                // TEMP DEBUG: pool-slot-vs-mmap byte check for layer 1 k=0
                if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 1) {
                    static FILE *pb2 = NULL;
                    if (!pb2) pb2 = fopen("/tmp/poolslot.bin", "wb");
                    if (pb2) {
                        int e = pf_idx[0][0];
                        char *slot = pool_base + 0 * exp_alloc;
                        const char *mm = (const char *)wf->data + ge->gate_off + (size_t)e * ge->gate_slab;
                        fwrite(slot, 1, ge->gate_slab, pb2);
                        fwrite(mm, 1, ge->gate_slab, pb2);
                        const char *slot_up = slot + ge->gate_slab;
                        const char *mm_up = (const char *)wf->data + ge->up_off + (size_t)e * ge->up_slab;
                        fwrite(slot_up, 1, ge->up_slab, pb2);
                        fwrite(mm_up, 1, ge->up_slab, pb2);
                        const char *slot_dn = slot_up + ge->up_slab;
                        const char *mm_dn = (const char *)wf->data + ge->down_off + (size_t)e * ge->down_slab;
                        fwrite(slot_dn, 1, ge->down_slab, pb2);
                        fwrite(mm_dn, 1, ge->down_slab, pb2);
                        fflush(pb2);
                    }
                }
                if (g_chunk_timing_enabled) {
                    double d = now_ms() - t_pread;
                    g_chunk_timing.pread_wait += d;
                    pf_per_layer_add(layer_idx, 7, d);
                }
            }

            // Combine params + validity for the group
            {
                float *params = (float *)[ctx->buf_pf_combine_params contents];
                for (int gm = 0; gm < gM; gm++) {
                    uint32_t m = gbase + (uint32_t)gm;
                    memset(params + (size_t)m * 10, 0, 10 * sizeof(float));
                    for (int k = 0; k < actual_K; k++) {
                        int v = pf_valid_all[(size_t)m * MAX_K + k];
                        params[(size_t)m * 10 + k] = v ? pf_w[m][k] : 0.0f;
                    }
                    params[(size_t)m * 10 + 8] = seg_batch[m];
                }
            }

            // TEMP DEBUG: CPU reference for C3's input norm (layer 1's C3 output)
            if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 2) {
                static FILE *indbg = NULL;
                if (!indbg) indbg = fopen("/tmp/inputnorm_cpu.bin", "wb");
                if (indbg) {
                    const float *moe_b = (const float *)[ctx->buf_pf_moe_hidden contents];
                    static float normed[HIDDEN_DIM];
                    cpu_rms_norm(moe_b, layer_cache[layer_idx].input_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                    fwrite(normed, sizeof(float), HIDDEN_DIM, indbg);                            // CPU ref
                    fwrite((const float *)[ctx->buf_pf_input contents], sizeof(float), HIDDEN_DIM, indbg);  // GPU C3
                    // TEMP: residual-norm cross-check: CPU norm of h_mid vs GPU h_post
                    static float hp_cpu[HIDDEN_DIM];
                    cpu_rms_norm((const float *)[ctx->buf_pf_h_mid contents],
                                 lc->post_attn_norm_w, hp_cpu, HIDDEN_DIM, RMS_NORM_EPS);
                    fwrite(hp_cpu, sizeof(float), HIDDEN_DIM, indbg);
                    fwrite((const float *)[ctx->buf_pf_h_post contents], sizeof(float), HIDDEN_DIM, indbg);  // GPU residual_norm
                    fwrite((const float *)[ctx->buf_pf_h_mid contents], sizeof(float), HIDDEN_DIM, indbg);
                    fflush(indbg);
                }
            }

            // TEMP DEBUG: CPU reference expert outputs for layer 0 + 1 + 39 (mmap slabs)
            if (getenv("FINCHMOE_GGUF_DBG") && (layer_idx == 0 || layer_idx == 1 || layer_idx == NUM_LAYERS - 1)) {
                static FILE *edbg = NULL;
                if (!edbg) edbg = fopen("/tmp/expert_cpu0.bin", "wb");
                if (edbg) {
                    int32_t hdr[2] = { (int32_t)layer_idx, (int32_t)gM };
                    fwrite(hdr, sizeof(int32_t), 2, edbg);
                    const float *x = h_post_batch;   // position 0 of the group
                    for (int k = 0; k < actual_K; k++) {
                        int e = pf_idx[0][k];
                        static float g_out[MOE_INTERMEDIATE], u_out[MOE_INTERMEDIATE], act[MOE_INTERMEDIATE], eo[HIDDEN_DIM];
                        gguf_cpu_matvec((const uint8_t *)wf->data + ge->gate_off + (size_t)e * ge->gate_slab,
                                        x, g_out, MOE_INTERMEDIATE, HIDDEN_DIM, ge->gate_type);
                        gguf_cpu_matvec((const uint8_t *)wf->data + ge->up_off + (size_t)e * ge->up_slab,
                                        x, u_out, MOE_INTERMEDIATE, HIDDEN_DIM, ge->up_type);
                        cpu_swiglu(g_out, u_out, act, MOE_INTERMEDIATE);
                        gguf_cpu_matvec((const uint8_t *)wf->data + ge->down_off + (size_t)e * ge->down_slab,
                                        act, eo, HIDDEN_DIM, MOE_INTERMEDIATE, ge->down_type);
                        fwrite(&e, 4, 1, edbg);
                        fwrite(eo, sizeof(float), HIDDEN_DIM, edbg);
                    }
                    fflush(edbg);
                }
                // TEMP DEBUG: the Phase-B input for the CPU ref
                static FILE *ib = NULL;
                if (!ib) ib = fopen("/tmp/expert_input_phaseb.bin", "wb");
                if (ib) {
                    int32_t hdr = (int32_t)layer_idx;
                    fwrite(&hdr, 4, 1, ib);
                    fwrite(h_post_batch, sizeof(float), HIDDEN_DIM, ib);
                    fflush(ib);
                }
            }

            // TEMP DEBUG: at chunk 10 layer 1, CPU refs from ALL layers' slabs
            if (getenv("FINCHMOE_GGUF_DBG") && layer_idx == 1) {
                static int l1calls = 0;
                if (l1calls == 10) {
                    static FILE *xdbg = NULL;
                    if (!xdbg) xdbg = fopen("/tmp/expert_xlayer.bin", "wb");
                    if (xdbg) {
                        const float *x = h_post_batch;
                        for (int XL = 0; XL < NUM_LAYERS; XL++) {
                            GgufExpertInfo *gx = &gguf_experts[XL];
                            for (int k = 0; k < actual_K; k++) {
                                int e = pf_idx[0][k];
                                static float g_out[MOE_INTERMEDIATE], u_out[MOE_INTERMEDIATE], act[MOE_INTERMEDIATE], eo[HIDDEN_DIM];
                                gguf_cpu_matvec((const uint8_t *)wf->data + gx->gate_off + (size_t)e * gx->gate_slab,
                                                x, g_out, MOE_INTERMEDIATE, HIDDEN_DIM, gx->gate_type);
                                gguf_cpu_matvec((const uint8_t *)wf->data + gx->up_off + (size_t)e * gx->up_slab,
                                                x, u_out, MOE_INTERMEDIATE, HIDDEN_DIM, gx->up_type);
                                cpu_swiglu(g_out, u_out, act, MOE_INTERMEDIATE);
                                gguf_cpu_matvec((const uint8_t *)wf->data + gx->down_off + (size_t)e * gx->down_slab,
                                                act, eo, HIDDEN_DIM, MOE_INTERMEDIATE, gx->down_type);
                                fwrite(eo, sizeof(float), HIDDEN_DIM, xdbg);
                            }
                        }
                        fflush(xdbg);
                    }
                }
                l1calls++;
            }

            // Pass 2: ONE command buffer for the whole group (Phase C S4
            // perf): the per-position CBs cost ~7 extra scheduling
            // boundaries per layer — the ~2ms CMD3 fixed cost (dedup's
            // marginal pool bandwidth ~140GB/s proved the kernels are not
            // the bottleneck). All kernels bind per-position offsets on
            // disjoint slots, so the group's positions append into one CB;
            // the wrap-failure fallbacks inside the encode commit the
            // partial CB and swap in a fresh one (pf_cmd3_slots records
            // whichever CB holds each position's tail — queue order keeps
            // every recorded wait valid). Deferred commit, as before.
            {
                double t_c3e = 0;
                if (g_chunk_timing_enabled) t_c3e = now_ms();
                id<MTLCommandBuffer> group_cb = [ctx->queue commandBuffer];
                [group_cb encodeWaitForEvent:ctx->expert_sync_event value:ctx->expert_sync_value];
                // Phase C S4.1: batched CMD3 (21 dispatches for the group)
                // with a per-position fallback (invalid experts / missing
                // wraps — rare). FINCHMOE_PF_C3LOOP=N repeats the batch N
                // times (timing-only — outputs overwritten each pass).
                {
                    static int c3loop = 0, c3parsed = 0;
                    if (!c3parsed) {
                        const char *ke = getenv("FINCHMOE_PF_C3LOOP");
                        c3loop = ke ? atoi(ke) : 1;
                        if (c3loop < 1) c3loop = 1;
                        c3parsed = 1;
                        if (c3loop > 1) fprintf(stderr, "[c3loop] batched CMD3 iteration x%d (timing-only)\n", c3loop);
                    }
                    int b_ok = 1;
                    for (int ci = 0; ci < c3loop; ci++) {
                        if (prefill_chunk_cmd3_batch(ctx, group_cb, layer_idx, gbase, (uint32_t)gM,
                                                     actual_K, pf_valid_all,
                                                     pf_data_bufs, pf_data_offs) != 0) {
                            b_ok = 0;
                            break;
                        }
                    }
                    if (b_ok) {
                        for (int gm = 0; gm < gM; gm++) {
                            pf_cmd3_slots[gbase + (uint32_t)gm] = group_cb;
                        }
                    } else {
                        for (int gm = 0; gm < gM; gm++) {
                            uint32_t m = gbase + (uint32_t)gm;
                            int valid[MAX_K];
                            for (int k = 0; k < actual_K; k++) {
                                valid[k] = pf_valid_all[(size_t)m * MAX_K + k];
                            }
                            prefill_chunk_cmd3_encode(&group_cb, layer_idx, m, actual_K, valid,
                                                      seg_batch[m], 1,
                                                      &pf_data_bufs[(size_t)m * MAX_K],
                                                      &pf_data_offs[(size_t)m * MAX_K]);
                            pf_cmd3_slots[m] = group_cb;
                        }
                    }
                }
                pf_note_gap(&g_chunk_timing.cmd3_gap);
                [group_cb commit];
                if (g_chunk_timing_enabled) {
                    double d = now_ms() - t_c3e;
                    g_chunk_timing.cmd3_encode += d;
                    pf_per_layer_add(layer_idx, 8, d);
                }
                // Debug dumps read the CMD3 outputs — wait the group CB
                // (debug-only; the deferral is not needed for parity runs).
                if (getenv("FINCHMOE_PF_DUMP") && (layer_idx == 0 || layer_idx == 1)) {
                    pf_dump_cmd3_debug(group_cb, layer_idx, 0,
                                       ctx->buf_pf_shared_gate, ctx->buf_pf_shared_up,
                                       ctx->buf_pf_shared_act, 0,
                                       &pf_data_bufs[0], &pf_data_offs[0]);
                }
            }
            // Group backpressure: wait for the group's last CMD3 before the
            // next group's preads overwrite the pool slots. The final group
            // keeps its deferred overlap with the next layer's cmdA.
            // (Ping-pong mode: consecutive groups use different halves, so
            // no wait here — the half-reuse wait above covers gi >= 2.)
            if (!pingpong && gbase + (uint32_t)Gpp < M && pf_cmd3_slots[gbase + (uint32_t)gM - 1]) {
                if (g_chunk_timing_enabled) t_ph = now_ms();
                [pf_cmd3_slots[gbase + (uint32_t)gM - 1] waitUntilCompleted];
                pf_note_wait_done();
                if (g_chunk_timing_enabled) g_chunk_timing.backpressure += now_ms() - t_ph;
            }
        }
        // Phase C S4 perf pass: optional explicit wait for the final group's
        // last CMD3 (FINCHMOE_PF_CMD3WAIT) — attributes the deferred-overlap
        // time that normally hides inside the next layer's cmdA_wait.
        if (getenv("FINCHMOE_PF_CMD3WAIT") && pf_cmd3_slots[M - 1]) {
            if (g_chunk_timing_enabled) t_ph = now_ms();
            [pf_cmd3_slots[M - 1] waitUntilCompleted];
            pf_note_wait_done();
            if (g_chunk_timing_enabled) {
                double d = now_ms() - t_ph;
                g_chunk_timing.cmd3_wait += d;
                pf_per_layer_add(layer_idx, 9, d);
            }
        }
    } else if (pool_ok) {
        static int pf_idx[PREFILL_CHUNK_MAX][MAX_K];
        static float pf_w[PREFILL_CHUNK_MAX][MAX_K];
        static id<MTLBuffer> __unsafe_unretained pf_data_bufs[PREFILL_CHUNK_MAX * MAX_K];
        static NSUInteger pf_data_offs[PREFILL_CHUNK_MAX * MAX_K];
        static int pf_valid_all[PREFILL_CHUNK_MAX * MAX_K];
        static int pf_miss_map[PREFILL_CHUNK_MAX * MAX_K];
        int actual_K = (K > MAX_K) ? MAX_K : K;
        NSUInteger exp_alloc = (EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);
        size_t esz = active_expert_size();

        // Wait for this layer's hot-set prefetch FIRST (fired during layer
        // L-1's Phase B — or at the chunk start for layer 0; ~0 residual:
        // it had a full layer of GPU work to complete in). Waiting before
        // firing the next prefetch keeps the two on separate group cycles.
        // Snapshot validity before the next fire overwrites the task array.
        static int pf_prefetch_valid[PF_HOT_MAX];
        async_pread_prefetch_wait();
        for (int s = 0; s < g_pf_hot_slots; s++) pf_prefetch_valid[s] = g_prefetch_pread.valid[s];

        // ---- Prefetch layer L+1's hot set (async; overlaps this layer's
        // routing + miss preads + CMD3s + next layer's cmdA/cmdB) ----
        if (g_hot_loaded && layer_idx < NUM_LAYERS - 1 && layer_fds) {
            int pi = (layer_idx + 1) % 2;
            if (ctx->buf_prefetch_pool[pi]) {
                static int pf_fds[PF_HOT_MAX];
                static off_t pf_offs[PF_HOT_MAX];
                static void *pf_dsts[PF_HOT_MAX];
                static size_t pf_sizes[PF_HOT_MAX];
                char *pp = (char *)[ctx->buf_prefetch_pool[pi] contents];
                for (int s = 0; s < g_pf_hot_slots; s++) {
                    int e = g_hot_expert[layer_idx + 1][s];
                    pf_fds[s] = layer_fds[layer_idx + 1];   // layer L+1's own expert file
                    pf_offs[s] = (off_t)e * (off_t)esz;
                    pf_dsts[s] = pp + (size_t)s * exp_alloc;
                    pf_sizes[s] = esz;
                }
                async_pread_prefetch_start(pf_fds, pf_offs, pf_dsts, pf_sizes, g_pf_hot_slots);
            }
        }

        // Pass 1: routing + staging for all M positions
        for (uint32_t m = 0; m < M; m++) {
            double t_route = 0;
            if (g_chunk_timing_enabled) t_route = now_ms();
            float gate_scores[NUM_EXPERTS];
            memcpy(gate_scores, gate_scores_batch + (size_t)m * NUM_EXPERTS,
                   NUM_EXPERTS * sizeof(float));
            cpu_softmax(gate_scores, NUM_EXPERTS);
            int expert_indices[MAX_K];
            float expert_weights[MAX_K];
            cpu_topk(gate_scores, NUM_EXPERTS, K, expert_indices, expert_weights);
            cpu_normalize_weights(expert_weights, K);
            if (g_chunk_timing_enabled) g_chunk_timing.routing_cpu += now_ms() - t_route;
            if (g_freq_tracking) {
                for (int k = 0; k < K; k++) {
                    g_expert_freq[layer_idx][expert_indices[k]]++;
                }
                if (layer_idx == 0) g_freq_total_tokens++;
            }
            for (int k = 0; k < MAX_K; k++) {
                pf_idx[m][k] = expert_indices[k];
                pf_w[m][k] = expert_weights[k];
            }
            // Stage per-position expert input + shared expert gate/up
            memcpy((float *)[ctx->buf_pf_expert_input contents] + (size_t)m * HIDDEN_DIM,
                   h_post_batch + (size_t)m * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
            memcpy((float *)[ctx->buf_pf_shared_gate contents] + (size_t)m * SHARED_INTERMEDIATE,
                   shared_batch + (size_t)m * SHARED_INTERMEDIATE,
                   SHARED_INTERMEDIATE * sizeof(float));
            memcpy((float *)[ctx->buf_pf_shared_up contents] + (size_t)m * SHARED_INTERMEDIATE,
                   shared_batch + (size_t)M * SHARED_INTERMEDIATE + (size_t)m * SHARED_INTERMEDIATE,
                   SHARED_INTERMEDIATE * sizeof(float));
        }

        // Hit/miss split: hot experts bind directly from pool[L%2]; misses
        // are pread into the main pool slots [8m..8m+7] (one batched read).
        {
            static int fds[PREFILL_CHUNK_MAX * MAX_K];
            static off_t offs[PREFILL_CHUNK_MAX * MAX_K];
            static void *dsts[PREFILL_CHUNK_MAX * MAX_K];
            static size_t sizes[PREFILL_CHUNK_MAX * MAX_K];
            char *pool_base = (char *)[ctx->buf_pool_expert_data contents];
            int n_miss = 0;
            for (uint32_t m = 0; m < M; m++) {
                for (int k = 0; k < MAX_K; k++) {
                    int e = pf_idx[m][k];
                    int s = (k < actual_K) ? g_hot_slot[layer_idx][e] : -1;
                    // A hot slot only counts if its prefetch pread completed
                    // with a full read (short reads under IO pressure would
                    // otherwise silently corrupt the expert computation).
                    if (s >= 0 && !pf_prefetch_valid[s]) s = -1;
                    if (s >= 0) {
                        // Hot hit — bind the prefetch pool slot directly.
                        pf_data_bufs[(size_t)m * MAX_K + k] = ctx->buf_prefetch_pool[layer_idx % 2];
                        pf_data_offs[(size_t)m * MAX_K + k] = (NSUInteger)s * exp_alloc;
                        pf_valid_all[(size_t)m * MAX_K + k] = 1;
                    } else {
                        // Cold miss — pread into the main pool slot.
                        int n = n_miss++;
                        fds[n] = packed_fd;
                        offs[n] = (off_t)e * (off_t)esz;
                        dsts[n] = pool_base + ((size_t)(8 * m) + (size_t)k) * exp_alloc;
                        sizes[n] = esz;
                        pf_miss_map[n] = (int)((size_t)m * MAX_K + k);
                        pf_data_bufs[(size_t)m * MAX_K + k] = ctx->buf_pool_expert_data;
                        pf_data_offs[(size_t)m * MAX_K + k] = ((size_t)(8 * m) + (size_t)k) * exp_alloc;
                        pf_valid_all[(size_t)m * MAX_K + k] = 0;
                    }
                }
            }
            double t_pread = 0;
            if (g_chunk_timing_enabled) t_pread = now_ms();
            if (n_miss > 0) {
                async_pread_multi_start(fds, offs, dsts, sizes, n_miss);
                async_pread_wait();
            }
            for (int n = 0; n < n_miss; n++) {
                pf_valid_all[pf_miss_map[n]] = g_async_pread.valid[n];
            }
            if (g_chunk_timing_enabled) g_chunk_timing.pread_wait += now_ms() - t_pread;
        }

        // Combine params + validity
        {
            float *params = (float *)[ctx->buf_pf_combine_params contents];
            for (uint32_t m = 0; m < M; m++) {
                memset(params + (size_t)m * 10, 0, 10 * sizeof(float));
                for (int k = 0; k < actual_K; k++) {
                    int v = pf_valid_all[(size_t)m * MAX_K + k];
                    params[(size_t)m * 10 + k] = v ? pf_w[m][k] : 0.0f;
                }
                params[(size_t)m * 10 + 8] = seg_batch[m];
            }
        }

        // Pass 2: encode + commit all M CMD3s back-to-back, no waits.
        for (uint32_t m = 0; m < M; m++) {
            int valid[MAX_K];
            for (int k = 0; k < actual_K; k++) {
                valid[k] = pf_valid_all[(size_t)m * MAX_K + k];
            }
            pf_cmd3_slots[m] = prefill_chunk_cmd3(layer_idx, m, actual_K, valid,
                                                  seg_batch[m], 1,
                                                  &pf_data_bufs[(size_t)m * MAX_K],
                                                  &pf_data_offs[(size_t)m * MAX_K]);
        }
    } else {
        for (uint32_t m = 0; m < M; m++) {
            // Buffer-safety backpressure: CMD3(m-1) may still read
            // buf_multi_expert_data[k] / buf_multi_expert_out[k] when the next
            // position's preads overwrite them.
            if (m > 0 && pf_cmd3_slots[m - 1]) {
                if (g_chunk_timing_enabled) t_ph = now_ms();
                [pf_cmd3_slots[m - 1] waitUntilCompleted];
                if (g_chunk_timing_enabled) g_chunk_timing.backpressure += now_ms() - t_ph;
            }
            pf_cmd3_slots[m] = prefill_chunk_experts(layer_idx, m, M,
                gate_scores_batch, seg_batch, shared_batch, h_post_batch,
                pf_expert_weights, pf_valid, mmap_base, K, packed_fd);
        }
    }
    if (last_cmd3_out) *last_cmd3_out = pf_cmd3_slots[M - 1];
    if (g_chunk_timing_enabled) {
        g_chunk_timing.total += now_ms() - t_layer;
        g_chunk_timing.layers++;
        // Phase C S4 perf pass: per-layer phase attribution (col 10 = total;
        // the other columns are recorded per-delta at their accumulation
        // sites via pf_per_layer_add).
        pf_per_layer_add(layer_idx, 10, now_ms() - t_layer);
        g_pf_per_layer_count[layer_idx]++;
    }
    (void)la_state;
}

// Chunk mode prerequisites: fused GPU delta-net + default async-pread expert
// path (no malloc cache / Metal LRU / LZ4 / CPU experts).
// GGUF mode: FINCHMOE_GGUF_CHUNK env gate (the S0 hazard guard originally
// excluded GGUF entirely — the prefill encoders didn't know Q4_K/Q6_K bits
// 10/11; S3 landed dequant_matvec_qk_prefill, S4 lands the driver branch).
static int gguf_chunk_enabled(void) {
    static int parsed = 0, val = 0;
    if (!parsed) {
        const char *e = getenv("FINCHMOE_GGUF_CHUNK");
        // C3: default-on (unset = 1) after the G5 parity gate went green.
        // G5 result: all components parity-clean (bitwise where kernels are
        // shared, <= 1e-5 where they differ); the sole divergence is an exact
        // gate-score tie (0 ULP) at one (pos, layer) flipping topk — final
        // logits cos 0.99943 vs per-token, argmax identical. Escape hatch:
        // FINCHMOE_GGUF_CHUNK=0.
        val = !(e && strcmp(e, "0") == 0);
        parsed = 1;
    }
    return val;
}

// Phase C S5: fused GPU GDN chain for the chunked GGUF driver (the
// fused_gdn_batched_qk kernel). Opt-in via FINCHMOE_GGUF_GDN_GPU=1.
// Measured on the M4 (2026-08-17): bitwise parity (cos 1.000000) and a
// working decode bridge, but perf-neutral vs the CPU chain (fused ~1.37ms
// vs ~1.1ms per layer — the delta state RMW dominates either way, and the
// chain's CPU work was already off the GPU's critical path). Default OFF;
// revisit on CPU-weak targets (iPhone tier) where the CPU chain's serial
// work is relatively more expensive.
static int gguf_gdn_gpu_enabled(void) {
    static int parsed = 0, val = 0;
    if (!parsed) {
        const char *e = getenv("FINCHMOE_GGUF_GDN_GPU");
        val = (e != NULL && strcmp(e, "0") != 0);
        parsed = 1;
    }
    return val;
}

static int prefill_chunk_available(void) {
    if (!(g_prefill_chunk > 0 && g_metal &&
          !g_malloc_cache && !g_expert_cache && !g_use_lz4 && !g_cpu_experts))
        return 0;
    if (g_gguf_stage) {
        // Phase C S4: GGUF chunked driver. Env-gated (FINCHMOE_GGUF_CHUNK,
        // default off until the G5 parity gate goes green). Requires the
        // stage mirror + expert pool (both lazy-created on first call).
        // NOTE: gpu_linear_attn_enabled is deliberately NOT required — the
        // bench harness passes -L, and the chunked GGUF chain runs on CPU
        // regardless (the per-token baseline does too).
        return gguf_chunk_enabled() && !linear_attn_bypass &&
               g_metal->matvec_qk_prefill && g_metal->fused_gate_up_swiglu_qk_pipe &&
               g_metal->matvec_qk && g_metal->gemv_bf16_prefill &&
               g_metal->residual_norm_fused_prefill &&
               g_metal->moe_combine_residual_prefill &&
               g_metal->rms_norm_sum_sq_prefill &&
               g_metal->rms_norm_apply_bf16_prefill &&
               gguf_stage_mirror_get(g_metal) && gguf_pool_get(g_metal);
    }
    return gpu_linear_attn_enabled && !linear_attn_bypass &&
           g_metal->fused_gdn_full && g_metal->fused_gdn_batched &&
           g_metal->matvec_prefill_4bit && g_metal->gemv_bf16_prefill;
}

// Chunked prefill driver: replaces the per-token prefill loops.
static void prefill_chunked_run(WeightFile *wf, float *hidden,
                                const float *embed_batch, int count, int *pos,
                                const int *mtp_tokens,
                                KVCache **kv_caches, void **layer_states,
                                void **layer_mmaps, int K, int *layer_fds,
                                int chunk_size)
{
    if (chunk_size <= 0 || chunk_size > PREFILL_CHUNK_MAX) chunk_size = PREFILL_CHUNK_MAX;

    // Chunked-path phase timing (env-gated, reset per prefill call).
    g_chunk_timing_enabled = getenv("FINCHMOE_PF_TIMING") != NULL;
    if (g_chunk_timing_enabled) memset(&g_chunk_timing, 0, sizeof(g_chunk_timing));

    // The layer weight cache is normally built lazily inside
    // fused_layer_forward's first call — the chunked path runs first.
    if (!layer_cache_built) build_layer_cache(wf);
    gguf_stage2_build(g_metal);   // S6 probe: FINCHMOE_GGUF_STAGE2
    // Same for the static scratch (the chunked GGUF chain uses
    // s_conv_out/s_out_vals/s_beta_proj_out/s_alpha_proj_out).
    init_layer_scratch();

    // Phase C S5: seed the GPU conv histories from the CPU la_state (fresh
    // sessions are zero; serve continuations carry the prior turn's state).
    // buf_conv_qk holds the per-head q/k histories, buf_conv_state the
    // v-channel history — both are what fused_gdn_batched_qk maintains in
    // place. The ssm state (buf_delta_state) needs no seed: the serve sync
    // uploads it, and the CPU chain's GPU delta shares the same buffer.
    if (g_gguf_stage && gguf_gdn_gpu_enabled() && g_metal->fused_gdn_batched_qk) {
        int li = 0;
        for (int i = 0; i < NUM_LAYERS; i++) {
            if ((i + 1) % FULL_ATTN_INTERVAL == 0) continue;
            LinearAttnState *la = (LinearAttnState *)layer_states[i];
            if (li >= NUM_LINEAR_LAYERS || !la) break;
            float *qh = (float *)[g_metal->buf_conv_qk[li] contents];
            float *cs = (float *)[g_metal->buf_conv_state[li] contents];
            for (int s = 0; s < CONV_KERNEL_SIZE - 1; s++) {
                for (int h = 0; h < LINEAR_NUM_V_HEADS; h++) {
                    // q history: qk_hist[h*3*kd + s*kd]
                    memcpy(qh + (size_t)h * (3 * LINEAR_KEY_DIM) + (size_t)s * LINEAR_KEY_DIM,
                           la->conv_state + (size_t)s * LINEAR_CONV_DIM + (size_t)h * LINEAR_KEY_DIM,
                           LINEAR_KEY_DIM * sizeof(float));
                    // k history: qk_hist[32*3*kd + h*3*kd + s*kd]
                    memcpy(qh + (size_t)(LINEAR_NUM_V_HEADS * 3 * LINEAR_KEY_DIM
                                         + h * 3 * LINEAR_KEY_DIM) + (size_t)s * LINEAR_KEY_DIM,
                           la->conv_state + (size_t)s * LINEAR_CONV_DIM + LINEAR_TOTAL_KEY
                                         + (size_t)h * LINEAR_KEY_DIM,
                           LINEAR_KEY_DIM * sizeof(float));
                }
                // v history: buf_conv_state[s*conv_dim + 2*total_key]
                memcpy(cs + (size_t)s * LINEAR_CONV_DIM + 2 * LINEAR_TOTAL_KEY,
                       la->conv_state + (size_t)s * LINEAR_CONV_DIM + 2 * LINEAR_TOTAL_KEY,
                       2 * LINEAR_TOTAL_KEY * sizeof(float));
            }
            li++;
        }
    }

    id<MTLCommandBuffer> last_cmd3 = nil;   // final chunk's last-layer CMD3 (final position)
    int last_M = 0;

    for (int cbase = 0; cbase < count; cbase += chunk_size) {
        int M = (cbase + chunk_size <= count) ? chunk_size : (count - cbase);
        // Kick off layer 0's hot-set prefetch for THIS chunk before the
        // chunk's layer-0 cmdA so it overlaps the first layer's GPU work
        // (layers >= 1 prefetch during the previous layer's Phase B).
        // pool[0] is free here: the previous chunk's last pool[0] reader was
        // CMD3(38), completed before that chunk's cmdA(39).
        if (g_hot_loaded && g_pf_hot_slots > 0 && g_metal->buf_prefetch_pool[0] && layer_fds) {
            static int pf_fds[PF_HOT_MAX];
            static off_t pf_offs[PF_HOT_MAX];
            static void *pf_dsts[PF_HOT_MAX];
            static size_t pf_sizes[PF_HOT_MAX];
            size_t esz = active_expert_size();
            NSUInteger exp_alloc = (EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);
            char *pp = (char *)[g_metal->buf_prefetch_pool[0] contents];
            for (int s = 0; s < g_pf_hot_slots; s++) {
                int e = g_hot_expert[0][s];
                pf_fds[s] = layer_fds[0];
                pf_offs[s] = (off_t)e * (off_t)esz;
                pf_dsts[s] = pp + (size_t)s * exp_alloc;
                pf_sizes[s] = esz;
            }
            async_pread_prefetch_start(pf_fds, pf_offs, pf_dsts, pf_sizes, g_pf_hot_slots);
        }
        id<MTLCommandBuffer> chunk_last = nil;
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            id<MTLCommandBuffer> layer_last = nil;
            prefill_chunk_layer(wf, layer, embed_batch, cbase, (uint32_t)M, *pos,
                                is_full ? kv_caches[layer] : NULL,
                                is_full ? NULL : (LinearAttnState *)layer_states[layer],
                                layer_mmaps[layer], K, layer_fds[layer], layer_fds,
                                &layer_last);
            if (layer == NUM_LAYERS - 1) chunk_last = layer_last;
            if (cbase + M >= count && layer == NUM_LAYERS - 1) {
                last_cmd3 = layer_last;
                last_M = M;
            }
        }
        // Fill the MTP K/V cache for this chunk's prompt tokens. The MTP's
        // hidden input = the post-MoE hidden per position — layer 39's
        // CMD3s don't combine into a buffer (only the driver's finalize
        // does, for the last position), so wait for the chunk's last CB and
        // CPU-combine per position, mirroring the finalize arithmetic.
        if (g_use_mtp && g_metal && g_mtp.loaded && chunk_last) {
            [chunk_last waitUntilCompleted];
            const float *h_mid = (const float *)[g_metal->buf_pf_h_mid contents];
            const float *params = (const float *)[g_metal->buf_pf_combine_params contents];
            static float mtp_hidden_batch[PREFILL_CHUNK_MAX * HIDDEN_DIM];
            for (uint32_t m = 0; m < (uint32_t)M; m++) {
                float *h_out = mtp_hidden_batch + (size_t)m * HIDDEN_DIM;
                float shared_gate = 1.0f / (1.0f + expf(-params[m * 10 + 8]));
                memcpy(h_out, h_mid + (size_t)m * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
                for (int k = 0; k < MAX_K; k++) {
                    float w = params[m * 10 + k];
                    if (w == 0.0f) continue;
                    const float *eo = (const float *)[g_metal->buf_multi_expert_out[k] contents]
                                      + (size_t)m * HIDDEN_DIM;
                    cpu_vec_madd(h_out, eo, w, HIDDEN_DIM);
                }
                const float *so = (const float *)[g_metal->buf_shared_out contents] + (size_t)m * HIDDEN_DIM;
                for (int i = 0; i < HIDDEN_DIM; i++) h_out[i] += shared_gate * so[i];
            }
            mtp_cache_fill(wf, mtp_tokens ? mtp_tokens + cbase : NULL,
                           embed_batch + (size_t)cbase * HIDDEN_DIM,
                           mtp_hidden_batch, M, cbase);
        }
        *pos += M;
    }

    // Final position: wait for the last layer's CMD3 and CPU-combine into
    // hidden so lm_head sees the complete state (mirrors
    // finalize_deferred_experts with slot (last_M-1) reads).
    if (last_cmd3 && last_M > 0) {
        int m = last_M - 1;
        [last_cmd3 waitUntilCompleted];
        MetalCtx *ctx = g_metal;
        const float *h_mid = (const float *)[ctx->buf_pf_h_mid contents] + (size_t)m * HIDDEN_DIM;
        const float *seg = (const float *)[ctx->buf_pf_seg contents] + m;
        // Debug: dump final-combine components (FINCHMOE_PF_DUMP)
        if (getenv("FINCHMOE_PF_DUMP")) {
            static FILE *fch = NULL;
            if (!fch) fch = fopen("/tmp/final_chunked.bin", "wb");
            if (fch) {
                fwrite(h_mid, sizeof(float), HIDDEN_DIM, fch);
                fwrite(seg, sizeof(float), 1, fch);
                const float *params = (const float *)[ctx->buf_pf_combine_params contents] + (size_t)m * 10;
                fwrite(params, sizeof(float), 10, fch);
                for (int k = 0; k < MAX_K; k++) {
                    fwrite((const float *)[ctx->buf_multi_expert_out[k] contents] + (size_t)m * HIDDEN_DIM,
                           sizeof(float), HIDDEN_DIM, fch);
                }
                fwrite((const float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM,
                       sizeof(float), HIDDEN_DIM, fch);
                fflush(fch);
            }
        }
        float shared_gate = 1.0f / (1.0f + expf(-seg[0]));
        // Mirror finalize_deferred_experts' CPU combine arithmetic order
        // exactly (separate cpu_vec_madd accumulation + premultiplied shared
        // gate) so the final hidden is bitwise-identical to the per-token
        // path: hidden[i] = h_mid[i] + moe_out[i] + shared_gate * shared_out[i].
        float moe_out[HIDDEN_DIM];
        memset(moe_out, 0, sizeof(moe_out));
        const float *params = (const float *)[ctx->buf_pf_combine_params contents] + (size_t)m * 10;
        // Phase C S4: the per-token GGUF combine skips non-finite expert
        // outputs and renormalizes the surviving weights (gpu_gguf_experts_
        // forward's CPU combine). Mirror it here so the final hidden matches
        // the GGUF baseline's semantics; packed mode keeps the exact
        // finalize_deferred_experts arithmetic (the guard never fires on
        // healthy outputs, so the extra branches cost nothing).
        float gguf_total_weight = 0.0f;
        for (int k = 0; k < MAX_K; k++) {
            float w = params[k];
            if (w == 0.0f) continue;  // invalid expert (mirrors finalize's valid-skip)
            const float *eo = (const float *)[ctx->buf_multi_expert_out[k] contents]
                              + (size_t)m * HIDDEN_DIM;
            if (g_gguf_stage) {
                float er = 0.0f;
                for (int j = 0; j < HIDDEN_DIM; j++) er += eo[j] * eo[j];
                if (!(isfinite(er) && er < 1e20f)) continue;
                gguf_total_weight += w;
            }
            cpu_vec_madd(moe_out, eo, w, HIDDEN_DIM);
        }
        if (g_gguf_stage && gguf_total_weight > 0.0f && gguf_total_weight < 0.99f) {
            float inv_tw = 1.0f / gguf_total_weight;
            for (int i = 0; i < HIDDEN_DIM; i++) moe_out[i] *= inv_tw;
        }
        // Premultiply shared output by the gate first (separate loop, mirrors
        // finalize's in-place shared_out *= sigmoid(score)), then plain adds.
        const float *so = (const float *)[ctx->buf_shared_out contents] + (size_t)m * HIDDEN_DIM;
        static float shared_scaled[HIDDEN_DIM];
        for (int i = 0; i < HIDDEN_DIM; i++) shared_scaled[i] = shared_gate * so[i];
        for (int i = 0; i < HIDDEN_DIM; i++) {
            hidden[i] = h_mid[i] + moe_out[i] + shared_scaled[i];
        }
    }

    // Phase C S5: bridge the GPU conv histories back to the CPU la_state
    // layout so the decode chain (per-token CPU conv) continues from the
    // post-prefill state. Inverse of the seed above.
    if (g_gguf_stage && gguf_gdn_gpu_enabled() && g_metal->fused_gdn_batched_qk) {
        int li = 0;
        for (int i = 0; i < NUM_LAYERS; i++) {
            if ((i + 1) % FULL_ATTN_INTERVAL == 0) continue;
            LinearAttnState *la = (LinearAttnState *)layer_states[i];
            if (li >= NUM_LINEAR_LAYERS || !la) break;
            const float *qh = (const float *)[g_metal->buf_conv_qk[li] contents];
            const float *cs = (const float *)[g_metal->buf_conv_state[li] contents];
            for (int s = 0; s < CONV_KERNEL_SIZE - 1; s++) {
                for (int h = 0; h < LINEAR_NUM_V_HEADS; h++) {
                    memcpy(la->conv_state + (size_t)s * LINEAR_CONV_DIM + (size_t)h * LINEAR_KEY_DIM,
                           qh + (size_t)h * (3 * LINEAR_KEY_DIM) + (size_t)s * LINEAR_KEY_DIM,
                           LINEAR_KEY_DIM * sizeof(float));
                    memcpy(la->conv_state + (size_t)s * LINEAR_CONV_DIM + LINEAR_TOTAL_KEY
                                         + (size_t)h * LINEAR_KEY_DIM,
                           qh + (size_t)(LINEAR_NUM_V_HEADS * 3 * LINEAR_KEY_DIM
                                         + h * 3 * LINEAR_KEY_DIM) + (size_t)s * LINEAR_KEY_DIM,
                           LINEAR_KEY_DIM * sizeof(float));
                }
                memcpy(la->conv_state + (size_t)s * LINEAR_CONV_DIM + 2 * LINEAR_TOTAL_KEY,
                       cs + (size_t)s * LINEAR_CONV_DIM + 2 * LINEAR_TOTAL_KEY,
                       2 * LINEAR_TOTAL_KEY * sizeof(float));
            }
            li++;
        }
    }

    // Flush the accumulated Phase-B trace (one write at the end — per-layer
    // I/O perturbed the timing landscape and masked the race).
    if (g_pb_acc && g_pb_len > 0) {
        FILE *f = fopen("/tmp/pb_new.bin", "wb");
        if (f) {
            fwrite(g_pb_acc, sizeof(float), g_pb_len, f);
            fclose(f);
        }
        g_pb_len = 0;
    }

    if (g_chunk_timing_enabled) chunk_timing_print();
}

// ============================================================================
// Main inference loop
// ============================================================================

// ============================================================================
// Expert frequency analysis (--freq)
// ============================================================================

static int freq_cmp_desc(const void *a, const void *b) {
    return *(const int *)b - *(const int *)a;
}

static void freq_print_analysis(int K) {
    if (!g_freq_tracking || g_freq_total_tokens == 0) return;

    int total_activations_per_layer = g_freq_total_tokens * K;

    fprintf(stderr, "\n=== Expert Frequency Analysis ===\n");
    fprintf(stderr, "Tokens tracked: %d, K=%d, activations/layer=%d\n\n",
            g_freq_total_tokens, K, total_activations_per_layer);

    // Per-layer analysis
    int experts_for_80_total = 0;  // sum across layers for overall estimate

    for (int l = 0; l < NUM_LAYERS; l++) {
        // Count unique experts and sort frequencies descending
        int sorted[NUM_EXPERTS];
        memcpy(sorted, g_expert_freq[l], NUM_EXPERTS * sizeof(int));
        qsort(sorted, NUM_EXPERTS, sizeof(int), freq_cmp_desc);

        int unique = 0;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            if (sorted[e] > 0) unique++;
        }

        // Compute cumulative coverage thresholds
        int cum = 0;
        int top10_cov = 0, top30_cov = 0, top60_cov = 0;
        int n_for_50 = 0, n_for_80 = 0, n_for_90 = 0;
        for (int e = 0; e < NUM_EXPERTS; e++) {
            cum += sorted[e];
            if (e == 9)  top10_cov = cum;
            if (e == 29) top30_cov = cum;
            if (e == 59) top60_cov = cum;
            if (n_for_50 == 0 && cum * 100 >= total_activations_per_layer * 50)
                n_for_50 = e + 1;
            if (n_for_80 == 0 && cum * 100 >= total_activations_per_layer * 80)
                n_for_80 = e + 1;
            if (n_for_90 == 0 && cum * 100 >= total_activations_per_layer * 90)
                n_for_90 = e + 1;
        }

        double pct10 = 100.0 * top10_cov / total_activations_per_layer;
        double pct30 = 100.0 * top30_cov / total_activations_per_layer;
        double pct60 = 100.0 * top60_cov / total_activations_per_layer;

        fprintf(stderr, "Layer %2d: %3d unique experts, "
                "top-10 cover %.0f%%, top-30 cover %.0f%%, top-60 cover %.0f%% "
                "(50%%@%d, 80%%@%d, 90%%@%d)\n",
                l, unique, pct10, pct30, pct60, n_for_50, n_for_80, n_for_90);

        experts_for_80_total += n_for_80;
    }

    // Overall summary: average experts needed for 80% across all layers
    double avg_experts_80 = (double)experts_for_80_total / NUM_LAYERS;
    // Expert size in GB: each expert is active_expert_size() bytes
    double expert_gb = (double)active_expert_size() / (1024.0 * 1024.0 * 1024.0);
    double total_pin_gb = avg_experts_80 * NUM_LAYERS * expert_gb;

    fprintf(stderr, "\n--- Overall Summary ---\n");
    fprintf(stderr, "To achieve 80%% hit rate across all layers, need %d experts pinned "
            "(avg %.0f/layer, %.2f GB)\n",
            experts_for_80_total, avg_experts_80, total_pin_gb);
    fprintf(stderr, "Expert size: %zu bytes (%.3f MB), %d layers x %d experts = %d total\n",
            active_expert_size(), (double)active_expert_size() / (1024.0 * 1024.0),
            NUM_LAYERS, NUM_EXPERTS, NUM_LAYERS * NUM_EXPERTS);
}

#ifndef CHAT_MODE

// ============================================================================
// HTTP Serve Mode — OpenAI-compatible /v1/chat/completions (SSE streaming)
// ============================================================================

// Read exactly n bytes from fd, returns 0 on success, -1 on error/EOF
static int read_exact(int fd, char *buf, int n) {
    int got = 0;
    while (got < n) {
        ssize_t r = read(fd, buf + got, n - got);
        if (r <= 0) return -1;
        got += (int)r;
    }
    return 0;
}

// Read HTTP request into buf (up to bufsz-1). Returns total bytes read, or -1.
// Reads headers, then Content-Length body if present.
static int read_http_request(int fd, char *buf, int bufsz) {
    int total = 0;
    // Read until we find \r\n\r\n (end of headers)
    while (total < bufsz - 1) {
        ssize_t r = read(fd, buf + total, 1);
        if (r <= 0) return -1;
        total++;
        if (total >= 4 &&
            buf[total-4] == '\r' && buf[total-3] == '\n' &&
            buf[total-2] == '\r' && buf[total-1] == '\n') {
            break;
        }
    }
    buf[total] = '\0';

    // Find Content-Length
    const char *cl = strcasestr(buf, "Content-Length:");
    if (cl) {
        int content_len = atoi(cl + 15);
        if (content_len > 0 && total + content_len < bufsz - 1) {
            if (read_exact(fd, buf + total, content_len) < 0) return -1;
            total += content_len;
            buf[total] = '\0';
        }
    }
    return total;
}

// Extract the last "content" value from an OpenAI messages array.
// Minimal JSON parsing: find last "content":" and extract the string value.
// Returns pointer into buf (null-terminated in place), or NULL.
static char *extract_last_content(char *buf) {
    char *last = NULL;
    char *p = buf;
    for (;;) {
        p = strstr(p, "\"content\"");
        if (!p) break;
        p += 9; // skip "content"
        // Skip whitespace and colon
        while (*p == ' ' || *p == '\t' || *p == ':') p++;
        if (*p == '"') {
            p++; // skip opening quote
            last = p;
            // Find closing quote (handle escapes)
            while (*p && !(*p == '"' && *(p-1) != '\\')) p++;
        }
    }
    if (last) {
        // Null-terminate the content string (overwrite closing quote)
        char *end = last;
        while (*end && !(*end == '"' && (end == last || *(end-1) != '\\'))) end++;
        *end = '\0';
        // Unescape \\n -> \n, \\" -> ", \\\\ -> backslash inline
        char *r = last, *w = last;
        while (*r) {
            if (*r == '\\' && *(r+1)) {
                r++;
                switch (*r) {
                    case 'n':  *w++ = '\n'; r++; break;
                    case 't':  *w++ = '\t'; r++; break;
                    case '"':  *w++ = '"';  r++; break;
                    case '\\': *w++ = '\\'; r++; break;
                    default:   *w++ = '\\'; *w++ = *r++; break;
                }
            } else {
                *w++ = *r++;
            }
        }
        *w = '\0';
    }
    return last;
}

// Extract "prompt" from JSON body (/v1/completions).
// Returns malloc'd string (caller frees), or NULL if not found.
static char *extract_prompt(const char *buf) {
    const char *p = strstr(buf, "\"prompt\"");
    if (!p) return NULL;
    p += 8; // skip "prompt"
    // skip whitespace and colon
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return NULL;
    p++; // skip opening quote
    char *result = malloc(65536);
    if (!result) return NULL;
    int i = 0;
    while (*p && *p != '"' && i < 65535) {
        // Proper JSON unescape — the old code dropped the backslash and
        // copied the next char verbatim, so "\n" became the literal letter
        // 'n' in the prompt. The model then imitated the mangled input and
        // generated 'n' where newlines belong (HumanEval harness bug).
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case 'n':  result[i++] = '\n'; p++; break;
                case 't':  result[i++] = '\t'; p++; break;
                case 'r':  result[i++] = '\r'; p++; break;
                case 'b':  result[i++] = '\b'; p++; break;
                case 'f':  result[i++] = '\f'; p++; break;
                case '"':  result[i++] = '"';  p++; break;
                case '/':  result[i++] = '/';  p++; break;
                case '\\': result[i++] = '\\'; p++; break;
                default:   result[i++] = '\\'; result[i++] = *p++; break;
            }
        } else {
            result[i++] = *p++;
        }
    }
    result[i] = '\0';
    return result;
}

// Extract "max_tokens" or "max_completion_tokens" from JSON body. Returns value or default.
static int extract_max_tokens(const char *buf, int default_val) {
    const char *p = strstr(buf, "\"max_completion_tokens\"");
    if (!p) p = strstr(buf, "\"max_tokens\"");
    if (!p) return default_val;
    p = strchr(p, ':');
    if (!p) return default_val;
    return atoi(p + 1);
}

// Save a conversation turn to ~/.flash-moe/sessions/<session_id>.jsonl
// Shared data store with the chat client.
static void server_save_turn(const char *session_id, const char *role, const char *content) {
    if (!session_id || !session_id[0] || !content) return;
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    char dir[1024], path[1024];
    snprintf(dir, sizeof(dir), "%s/.flash-moe/sessions", home);
    mkdir(dir, 0755);
    char parent[1024];
    snprintf(parent, sizeof(parent), "%s/.flash-moe", home);
    mkdir(parent, 0755);
    mkdir(dir, 0755);
    snprintf(path, sizeof(path), "%s/%s.jsonl", dir, session_id);
    FILE *f = fopen(path, "a");
    if (!f) return;
    // JSON-escape content
    size_t clen = strlen(content);
    char *escaped = malloc(clen * 2 + 1);
    int j = 0;
    for (size_t i = 0; i < clen; i++) {
        switch (content[i]) {
            case '"': escaped[j++]='\\'; escaped[j++]='"'; break;
            case '\\': escaped[j++]='\\'; escaped[j++]='\\'; break;
            case '\n': escaped[j++]='\\'; escaped[j++]='n'; break;
            case '\r': escaped[j++]='\\'; escaped[j++]='r'; break;
            case '\t': escaped[j++]='\\'; escaped[j++]='t'; break;
            default: escaped[j++]=content[i]; break;
        }
    }
    escaped[j] = 0;
    fprintf(f, "{\"role\":\"%s\",\"content\":\"%s\"}\n", role, escaped);
    free(escaped);
    fclose(f);
}

// Extract "session_id" string from JSON body. Copies into out_buf (max out_size).
// Returns 1 if found, 0 if missing.
static int extract_session_id(const char *buf, char *out_buf, int out_size) {
    const char *p = strstr(buf, "\"session_id\"");
    if (!p) return 0;
    p += 12; // skip "session_id"
    while (*p == ' ' || *p == '\t' || *p == ':') p++;
    if (*p != '"') return 0;
    p++; // skip opening quote
    int i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        out_buf[i++] = *p++;
    }
    out_buf[i] = '\0';
    return i > 0 ? 1 : 0;
}

// Write a full HTTP response string to fd
static void http_write(int fd, const char *data, int len) {
    int sent = 0;
    while (sent < len) {
        ssize_t w = write(fd, data + sent, len - sent);
        if (w <= 0) break;
        sent += (int)w;
    }
}

static void http_write_str(int fd, const char *s) {
    http_write(fd, s, (int)strlen(s));
}

// Send an SSE chunk with a token delta
// Returns 0 on success, -1 if client disconnected
static int sse_send_delta(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    // Escape the token text for JSON
    char escaped[2048];
    char *w = escaped;
    for (const char *r = token_text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"':  *w++ = '\\'; *w++ = '"';  break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n';  break;
            case '\r': *w++ = '\\'; *w++ = 'r';  break;
            case '\t': *w++ = '\\'; *w++ = 't';  break;
            default:
                // Raw byte-fallback bytes (>= 0x80) are invalid UTF-8 in the
                // JSON stream — clients (and the eval harness) drop the whole
                // delta when json parsing fails. Escape as \u00XX.
                if ((unsigned char)*r >= 0x80) {
                    static const char hexd[] = "0123456789abcdef";
                    unsigned char b = (unsigned char)*r;
                    *w++ = '\\'; *w++ = 'u'; *w++ = '0'; *w++ = '0';
                    *w++ = hexd[b >> 4]; *w++ = hexd[b & 0xF];
                } else {
                    *w++ = *r;
                }
                break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
        request_id, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done(int fd, const char *request_id,
                          int prompt_tokens, int completion_tokens,
                          double prefill_ms, double gen_ms) {
    char chunk[2048];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
        "\"total_tokens\":%d,\"prefill_ms\":%.0f,\"generation_ms\":%.0f,"
        "\"tokens_per_second\":%.1f}}\n\n"
        "data: [DONE]\n\n",
        request_id,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        prefill_ms, gen_ms,
        completion_tokens > 0 ? completion_tokens * 1000.0 / gen_ms : 0.0);
    http_write(fd, chunk, n);
}

// Send SSE delta for /v1/completions format (text_completion instead of chat.completion.chunk)
static int sse_send_delta_completion(int fd, const char *request_id, const char *token_text) {
    char chunk[4096];
    char escaped[2048];
    char *w = escaped;
    for (const char *r = token_text; *r && w < escaped + sizeof(escaped) - 8; r++) {
        switch (*r) {
            case '"':  *w++ = '\\'; *w++ = '"';  break;
            case '\\': *w++ = '\\'; *w++ = '\\'; break;
            case '\n': *w++ = '\\'; *w++ = 'n';  break;
            case '\r': *w++ = '\\'; *w++ = 'r';  break;
            case '\t': *w++ = '\\'; *w++ = 't';  break;
            default:
                // Raw byte-fallback bytes (>= 0x80) are invalid UTF-8 in the
                // JSON stream — json parsers drop the whole delta (the eval
                // harness silently skips them). Escape as \u00XX.
                if ((unsigned char)*r >= 0x80) {
                    static const char hexd[] = "0123456789abcdef";
                    unsigned char b = (unsigned char)*r;
                    *w++ = '\\'; *w++ = 'u'; *w++ = '0'; *w++ = '0';
                    *w++ = hexd[b >> 4]; *w++ = hexd[b & 0xF];
                } else {
                    *w++ = *r;
                }
                break;
        }
    }
    *w = '\0';
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"text_completion\","
        "\"choices\":[{\"index\":0,\"text\":\"%s\",\"finish_reason\":null}]}\n\n",
        request_id, escaped);
    ssize_t wr = write(fd, chunk, n);
    return (wr <= 0) ? -1 : 0;
}

static void sse_send_done_completion(int fd, const char *request_id,
                                     int prompt_tokens, int completion_tokens,
                                     double prefill_ms, double gen_ms) {
    char chunk[2048];
    int n = snprintf(chunk, sizeof(chunk),
        "data: {\"id\":\"%s\",\"object\":\"text_completion\","
        "\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"stop\"}],"
        "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
        "\"total_tokens\":%d,\"prefill_ms\":%.0f,\"generation_ms\":%.0f,"
        "\"tokens_per_second\":%.1f}}\n\n"
        "data: [DONE]\n\n",
        request_id,
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
        prefill_ms, gen_ms,
        completion_tokens > 0 ? completion_tokens * 1000.0 / gen_ms : 0.0);
    http_write(fd, chunk, n);
}

static const char *SSE_HEADERS =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: text/event-stream\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "\r\n";

static const char *CORS_RESPONSE =
    "HTTP/1.1 204 No Content\r\n"
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
    "Access-Control-Max-Age: 86400\r\n"
    "\r\n";

// Build the think suffix based on g_no_think.
// Thinking ON:  <think>\n           (model generates reasoning inside <think>...</think>)
// Thinking OFF: <think>\n\n</think>\n\n (empty think block — skip reasoning)
static void build_think_suffix(char *buf, size_t bufsz) {
    if (g_no_think) {
        // Empty think block: tells model thinking is done, answer directly
        snprintf(buf, bufsz, "<think>\n\n</think>\n\n");
    } else {
        // Let model output <think> itself — prepending <think>\n confuses
        // the quantized model into closing the think block immediately.
        buf[0] = '\0';
    }
}

// Tokenize a user turn (system prompt already cached in KV).
// Encodes: <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n
static PromptTokens *tokenize_user_turn(const char *user_content) {
    const char *prefix = "<|im_start|>user\n";
    char think[32]; build_think_suffix(think, sizeof(think));
    char suffix[256];
    snprintf(suffix, sizeof(suffix), "<|im_end|>\n<|im_start|>assistant\n%s", think);

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Tokenize a continuation turn for session caching.
// Prefixes with \n to follow the previous assistant's <|im_end|>, then the new user turn.
static PromptTokens *tokenize_continuation_turn(const char *user_content) {
    const char *prefix = "\n<|im_start|>user\n";
    char think[32]; build_think_suffix(think, sizeof(think));
    char suffix[256];
    snprintf(suffix, sizeof(suffix), "<|im_end|>\n<|im_start|>assistant\n%s", think);

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;
    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Load custom system prompt from ~/.flash-moe/system.md, or use default
static char *load_system_prompt(void) {
    const char *home = getenv("HOME");
    if (home) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/.flash-moe/system.md", home);
        FILE *f = fopen(path, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char *buf = malloc(sz + 1);
            size_t n = fread(buf, 1, sz, f);
            buf[n] = 0;
            fclose(f);
            fprintf(stderr, "[serve] Loaded custom system prompt from %s (%ld bytes)\n", path, sz);
            return buf;
        }
    }
    return strdup("You are a helpful assistant.");
}

// Tokenize ONLY the system prompt (for prefill caching).
// Encodes: <|im_start|>system\n{system_prompt}<|im_end|>\n
// The user turn + assistant prompt are added per-request by tokenize_user_turn.
static PromptTokens *tokenize_system_prompt(void) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();

    size_t sys_len = strlen(sys_prompt_text);
    size_t total = 64 + sys_len;
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n", sys_prompt_text);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Tokenize a full chat message (system prompt + user turn) for first-time use.
static PromptTokens *tokenize_chat_message(const char *user_content) {
    static char *sys_prompt_text = NULL;
    if (!sys_prompt_text) sys_prompt_text = load_system_prompt();

    // Build: <|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n
    char think[32]; build_think_suffix(think, sizeof(think));
    size_t sys_len = strlen(sys_prompt_text);
    size_t user_len = strlen(user_content);
    size_t total = 80 + sys_len + user_len + strlen(think);
    char *prompt = malloc(total);
    if (!prompt) return NULL;
    snprintf(prompt, total, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s",
             sys_prompt_text, user_content, think);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// Keep old signature for backward compat (unused but prevents compiler warning)
__attribute__((unused))
static PromptTokens *tokenize_chat_message_old(const char *user_content) {
    const char *prefix =
        "<|im_start|>system\nYou are a helpful assistant. /think<|im_end|>\n"
        "<|im_start|>user\n";
    const char *suffix = "<|im_end|>\n<|im_start|>assistant\n";

    size_t prompt_len = strlen(prefix) + strlen(user_content) + strlen(suffix) + 1;
    char *prompt = malloc(prompt_len);
    if (!prompt) return NULL;

    snprintf(prompt, prompt_len, "%s%s%s", prefix, user_content, suffix);
    PromptTokens *pt = encode_prompt_text_to_tokens(prompt);
    free(prompt);
    return pt;
}

// The main serve loop. Model state must already be initialized.
// Sync CPU linear attention state → GPU buffers
static void sync_cpu_to_gpu_delta_state_serve(void **layer_states) {
    if (!g_metal || !g_metal->delta_net_step || !layer_states) return;
    int li = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if ((i + 1) % FULL_ATTN_INTERVAL == 0) continue;
        if (!layer_states[i]) { li++; continue; }
        LinearAttnState *la = (LinearAttnState *)layer_states[i];
        if (li < NUM_LINEAR_LAYERS) {
            if (g_metal->buf_delta_state[li] && la->ssm_state)
                memcpy([g_metal->buf_delta_state[li] contents], la->ssm_state,
                       LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float));
            if (g_metal->buf_conv_state[li] && la->conv_state)
                memcpy([g_metal->buf_conv_state[li] contents], la->conv_state,
                       (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float));
        }
        li++;
    }
}

// ============================================================================
// Request queue — enables concurrent clients without blocking the accept loop.
// A single worker thread drains the queue sequentially (GPU is not concurrent).
// ============================================================================

#define SERVE_QUEUE_MAX 16

typedef struct {
    int client_fd;
    char *content;          // malloc'd, freed after processing
    int max_gen;
    char session_id[64];
    int has_session;
    char request_id[64];
    int is_completion;      // 1 = /v1/completions (raw prompt), 0 = chat
} ServeQueueEntry;

typedef struct {
    ServeQueueEntry entries[SERVE_QUEUE_MAX];
    int head, tail, count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int shutdown;
} ServeQueue;

static ServeQueue g_serve_queue = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond  = PTHREAD_COND_INITIALIZER,
};

// All shared state needed by the worker thread to process a request
typedef struct {
    WeightFile *wf;
    Vocabulary *vocab;
    void **layer_states;
    KVCache **kv_caches;
    void **layer_mmaps;
    int *layer_fds;
    uint16_t *final_norm_w;
    int K;
    // System prompt cache (read-only after init)
    int sys_prompt_len;
    // Snapshot storage
    float *kv_k_snapshots[NUM_LAYERS];
    float *kv_v_snapshots[NUM_LAYERS];
    int   kv_snapshot_len[NUM_LAYERS];
    float *la_conv_snapshots[NUM_LAYERS];
    float *la_ssm_snapshots[NUM_LAYERS];
    void  *gpu_delta_snapshots[NUM_LINEAR_LAYERS];
    void  *gpu_conv_snapshots[NUM_LINEAR_LAYERS];
    void  *gpu_conv_qk_snapshots[NUM_LINEAR_LAYERS];
    // Pre-turn snapshot (rollback target for truncated turns)
    int   pre_turn_pos;
    float *pre_kv_k[NUM_LAYERS];
    float *pre_kv_v[NUM_LAYERS];
    int   pre_kv_len[NUM_LAYERS];
    float *pre_la_conv[NUM_LAYERS];
    float *pre_la_ssm[NUM_LAYERS];
    void  *pre_gpu_delta[NUM_LINEAR_LAYERS];
    void  *pre_gpu_conv[NUM_LINEAR_LAYERS];
    void  *pre_gpu_qk[NUM_LINEAR_LAYERS];
    // Session tracking (protected by session_mutex)
    char active_session_id[64];
    int  session_pos;
    pthread_mutex_t session_mutex;
} ServeState;

// Forward declaration
static void process_chat_request(ServeState *s, int client_fd,
                                 const char *content, int max_gen,
                                 const char *session_id, int has_session,
                                 const char *request_id, int is_completion);

// ============================================================================
// Worker thread: dequeues requests and processes them sequentially
// ============================================================================

static void *serve_worker(void *arg) {
    ServeState *s = (ServeState *)arg;
    ServeQueue *q = &g_serve_queue;

    for (;;) {
        pthread_mutex_lock(&q->mutex);
        while (q->count == 0 && !q->shutdown)
            pthread_cond_wait(&q->cond, &q->mutex);
        if (q->shutdown && q->count == 0) {
            pthread_mutex_unlock(&q->mutex);
            break;
        }
        ServeQueueEntry e = q->entries[q->head];
        q->head = (q->head + 1) % SERVE_QUEUE_MAX;
        q->count--;
        pthread_mutex_unlock(&q->mutex);

        process_chat_request(s, e.client_fd,
                             e.content, e.max_gen,
                             e.has_session ? e.session_id : NULL, e.has_session,
                             e.request_id, e.is_completion);

        free(e.content);
    }
    return NULL;
}

// ============================================================================
// process_chat_request — full generation pipeline for a single chat request
// (extracted from serve_loop so the worker thread can call it)
// ============================================================================

static void process_chat_request(ServeState *s, int client_fd,
                                 const char *content, int max_gen,
                                 const char *session_id, int has_session,
                                 const char *request_id, int is_completion) {

    // Per-request sampler isolation: the n-gram blocker + rep-penalty ring
    // are global scratch — without a reset here, greedy sampling is stateful
    // across requests (identical prompts gave different completions).
    sampler_state_reset();

    float *hidden = calloc(HIDDEN_DIM, sizeof(float));
    float *logits = calloc(VOCAB_SIZE, sizeof(float));

    // Session continuation: enabled — the same session_id continues the
    // conversation from the previous turn's KV/GDN state. (Previously
    // disabled after the "template tokens leak" bug; that leak was caused
    // by truncated turns poisoning the next continuation, now handled by
    // the pre-turn snapshot rollback below. FINCHMOE_SERVE_NOCONTINUE=1
    // restores stateless per-turn mode for debugging.)
    int is_continuation = 0;
    if (!getenv("FINCHMOE_SERVE_NOCONTINUE")) {
        is_continuation = (has_session &&
                           s->active_session_id[0] != '\0' &&
                           strcmp(session_id, s->active_session_id) == 0);
    }

    fprintf(stderr, "[serve] %s content=%zu chars, max_tokens=%d, session=%s%s%s\n",
            request_id, strlen(content), max_gen,
            has_session ? session_id : "(none)",
            is_continuation ? " [CONTINUE]" : " [NEW]",
            is_completion ? " [COMPLETION]" : "");

    // ---- Tokenize ----
    PromptTokens *pt;
    if (is_completion) {
        // Raw prompt — no chat template wrapping
        pt = encode_prompt_text_to_tokens(content);
    } else if (is_continuation) {
        pt = tokenize_continuation_turn(content);
    } else {
        pt = tokenize_user_turn(content);
    }
    if (!pt) {
        http_write_str(client_fd,
            "HTTP/1.1 500 Internal Server Error\r\nConnection: close\r\n\r\n"
            "{\"error\":\"tokenization failed\"}\n");
        free(hidden); free(logits); close(client_fd);
        return;
    }

    fprintf(stderr, "[serve] %s prompt=%d tokens%s\n", request_id, pt->count,
            is_continuation ? " (continuation — skipping snapshot restore)" : "");

    size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
    size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);

    int pos;
    if (is_continuation) {
        pos = s->session_pos;
    } else {
        // Restore state from system prompt snapshot
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (s->kv_caches[i] && s->kv_k_snapshots[i]) {
                size_t sz = s->sys_prompt_len * kv_dim * sizeof(float);
                for (int p = 0; p < s->sys_prompt_len; p++)
                    kv_write(s->kv_caches[i], p,
                             s->kv_k_snapshots[i] + (size_t)p * kv_dim,
                             s->kv_v_snapshots[i] + (size_t)p * kv_dim);
                s->kv_caches[i]->len = s->kv_snapshot_len[i];
                if (g_metal) {
                    int fa_idx = (i + 1) / FULL_ATTN_INTERVAL - 1;
                    if (fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS) {
                        memcpy([g_metal->buf_kv_k[fa_idx] contents],
                               s->kv_k_snapshots[i], sz);
                        memcpy([g_metal->buf_kv_v[fa_idx] contents],
                               s->kv_v_snapshots[i], sz);
                    }
                }
            } else if (s->kv_caches[i]) {
                s->kv_caches[i]->len = 0;
            }
            if (s->layer_states[i] && s->la_conv_snapshots[i]) {
                LinearAttnState *ls = (LinearAttnState *)s->layer_states[i];
                memcpy(ls->conv_state, s->la_conv_snapshots[i], conv_state_size);
                memcpy(ls->ssm_state, s->la_ssm_snapshots[i], ssm_state_size);
            } else if (s->layer_states[i]) {
                LinearAttnState *ls = (LinearAttnState *)s->layer_states[i];
                memset(ls->conv_state, 0, conv_state_size);
                memset(ls->ssm_state, 0, ssm_state_size);
            }
        }
        // Restore GPU delta-net state
        if (g_metal && g_metal->delta_net_step) {
            for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
                if (s->gpu_delta_snapshots[i] && g_metal->buf_delta_state[i])
                    memcpy([g_metal->buf_delta_state[i] contents],
                           s->gpu_delta_snapshots[i], 32*128*128*sizeof(float));
                if (s->gpu_conv_snapshots[i] && g_metal->buf_conv_state[i])
                    memcpy([g_metal->buf_conv_state[i] contents],
                           s->gpu_conv_snapshots[i], 3*LINEAR_CONV_DIM*sizeof(float));
                if (s->gpu_conv_qk_snapshots[i] && g_metal->buf_conv_qk[i])
                    memcpy([g_metal->buf_conv_qk[i] contents],
                           s->gpu_conv_qk_snapshots[i], 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float));
            }
        } else {
            reset_delta_net_state();
        }
        pos = s->sys_prompt_len;
        // Update active session
        pthread_mutex_lock(&s->session_mutex);
        if (has_session) {
            strncpy(s->active_session_id, session_id, sizeof(s->active_session_id) - 1);
            s->active_session_id[sizeof(s->active_session_id) - 1] = '\0';
        } else {
            s->active_session_id[0] = '\0';
        }
        pthread_mutex_unlock(&s->session_mutex);
    }
    if (g_cache_telemetry_enabled) cache_telemetry_reset();

    // ---- Send SSE headers ----
    // Send SSE headers (same for both chat and completions)
    http_write_str(client_fd, SSE_HEADERS);

    // ---- Pre-turn snapshot (session requests only) ----
    // A truncated assistant turn poisons the next continuation (the model
    // imitates the abrupt end and stops immediately) — snapshot the state
    // so a truncated turn can be rolled back, keeping the session clean.
    if (has_session) {
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
        size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
        size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (s->kv_caches[i]) {
                size_t sz = (size_t)s->kv_caches[i]->len * kv_dim * sizeof(float);
                s->pre_kv_k[i] = realloc(s->pre_kv_k[i], sz);
                s->pre_kv_v[i] = realloc(s->pre_kv_v[i], sz);
                for (int p = 0; p < s->kv_caches[i]->len; p++)
                    for (int hh = 0; hh < NUM_KV_HEADS; hh++) {
                        kv_read_k(s->kv_caches[i], p, hh,
                                  s->pre_kv_k[i] + (size_t)p * kv_dim + (size_t)hh * HEAD_DIM);
                        kv_read_v(s->kv_caches[i], p, hh,
                                  s->pre_kv_v[i] + (size_t)p * kv_dim + (size_t)hh * HEAD_DIM);
                    }
                s->pre_kv_len[i] = s->kv_caches[i]->len;
            }
            if (s->layer_states[i]) {
                LinearAttnState *ls = (LinearAttnState *)s->layer_states[i];
                s->pre_la_conv[i] = realloc(s->pre_la_conv[i], conv_state_size);
                s->pre_la_ssm[i] = realloc(s->pre_la_ssm[i], ssm_state_size);
                memcpy(s->pre_la_conv[i], ls->conv_state, conv_state_size);
                memcpy(s->pre_la_ssm[i], ls->ssm_state, ssm_state_size);
            }
        }
        if (g_metal && g_metal->delta_net_step) {
            for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
                if (g_metal->buf_delta_state[i]) {
                    size_t sz = 32*128*128*sizeof(float);
                    s->pre_gpu_delta[i] = realloc(s->pre_gpu_delta[i], sz);
                    memcpy(s->pre_gpu_delta[i], [g_metal->buf_delta_state[i] contents], sz);
                }
                if (g_metal->buf_conv_state[i]) {
                    size_t sz = 3*LINEAR_CONV_DIM*sizeof(float);
                    s->pre_gpu_conv[i] = realloc(s->pre_gpu_conv[i], sz);
                    memcpy(s->pre_gpu_conv[i], [g_metal->buf_conv_state[i] contents], sz);
                }
                if (g_metal->buf_conv_qk[i]) {
                    size_t sz = 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float);
                    s->pre_gpu_qk[i] = realloc(s->pre_gpu_qk[i], sz);
                    memcpy(s->pre_gpu_qk[i], [g_metal->buf_conv_qk[i] contents], sz);
                }
            }
        }
        s->pre_turn_pos = pos;
    }

    // ---- Batch prefill ----
    double t_prefill = now_ms();
    float *serve_embed_batch = NULL;
    if (pt->count > 1) {
        serve_embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
        for (int i = 0; i < pt->count; i++) {
            embed_lookup(s->wf, pt->ids[i], serve_embed_batch + (size_t)i * HIDDEN_DIM);
        }
    }
    if (prefill_chunk_available() && pt->count > 1 && serve_embed_batch) {
        // Chunked batched prefill — completes the final hidden and advances
        // pos (same contract as the interactive path).
        prefill_chunked_run(s->wf, hidden, serve_embed_batch, pt->count, &pos,
                            pt->ids,
                            s->kv_caches, s->layer_states, s->layer_mmaps,
                            s->K, s->layer_fds, g_prefill_chunk);
    } else {
        for (int i = 0; i < pt->count - 1; i++) {
            cache_telemetry_note_token();
            if (serve_embed_batch) {
                memcpy(hidden, serve_embed_batch + (size_t)i * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(s->wf, pt->ids[i], hidden);
            }
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(s->wf, layer, hidden,
                                    is_full ? s->kv_caches[layer] : NULL,
                                    is_full ? NULL : s->layer_states[layer],
                                    pos,
                                    s->layer_mmaps[layer] != MAP_FAILED ? s->layer_mmaps[layer] : NULL,
                                    s->K, s->layer_fds[layer]);
            }
            discard_deferred_experts();
            pos++;
            if (g_use_mtp && serve_embed_batch)
                mtp_cache_fill(s->wf, pt->ids + i, serve_embed_batch + (size_t)i * HIDDEN_DIM,
                               hidden, 1, pos - 1);
        }
        // Last prefill token
        {
            cache_telemetry_note_token();
            if (serve_embed_batch) {
                memcpy(hidden, serve_embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(s->wf, pt->ids[0], hidden);
            }
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(s->wf, layer, hidden,
                                    is_full ? s->kv_caches[layer] : NULL,
                                    is_full ? NULL : s->layer_states[layer],
                                    pos,
                                    s->layer_mmaps[layer] != MAP_FAILED ? s->layer_mmaps[layer] : NULL,
                                    s->K, s->layer_fds[layer]);
            }
            complete_deferred_experts();
            pos++;
            if (g_use_mtp && serve_embed_batch)
                mtp_cache_fill(s->wf, pt->ids + (pt->count - 1),
                               serve_embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                               hidden, 1, pos - 1);
        }
    }
    if (serve_embed_batch) { free(serve_embed_batch); serve_embed_batch = NULL; }
    double prefill_ms = now_ms() - t_prefill;
    fprintf(stderr, "[serve] %s prefill=%d tokens in %.0fms\n",
            request_id, pt->count, prefill_ms);

    // ---- Final norm + LM head for first token ----
    if (s->final_norm_w) {
        float *normed = malloc(HIDDEN_DIM * sizeof(float));
        cpu_rms_norm(hidden, s->final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
        memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
        free(normed);
    }
    lm_head_forward(s->wf, hidden, logits);
    if (getenv("FINCHMOE_SERVE_DEBUG")) {
        fprintf(stderr, "[serve-dbg] %s prefill hidden_rms=%.4f logits_rms=%.4f top3=%d(%.2f),%d(%.2f),%d(%.2f)\n",
                request_id, vec_rms(hidden, HIDDEN_DIM), vec_rms(logits, VOCAB_SIZE),
                (int)cpu_argmax(logits, VOCAB_SIZE), 0.0f, 0, 0.0f, 0, 0.0f);
    }
    int next_token = cpu_sample_temp(logits, VOCAB_SIZE, g_temperature, g_top_k);
    logit_diag_dump(logits, VOCAB_SIZE, next_token, 0);

    // ---- Auto-regressive generation with SSE streaming ----
    if (g_pred_enabled || g_gguf_stage) {   // GGUF: S7 Lever 1 temporal prediction
        g_pred_generating = 1;
        g_pred_valid = 0;
    }
    double t_gen = now_ms();
    int gen_count = 0;
    int in_think = 0;
    int think_ended = 0;   // once the think block closes, think tokens are banned
    int think_tokens = 0;
    char *gen_response = calloc(1, 256 * 1024);
    int gen_resp_len = 0;

    for (int gen = 0; gen < max_gen; gen++) {
        if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
            // EOS mid-think leaves an unclosed <think> in the context, which
            // the next continuation imitates (ends immediately). Close the
            // think in the context before embedding the EOS.
            if (in_think) {
                cache_telemetry_note_token();
                embed_lookup(s->wf, THINK_END_TOKEN, hidden);
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(s->wf, layer, hidden,
                                        is_full ? s->kv_caches[layer] : NULL,
                                        is_full ? NULL : s->layer_states[layer],
                                        pos,
                                        s->layer_mmaps[layer] != MAP_FAILED ? s->layer_mmaps[layer] : NULL,
                                        s->K, s->layer_fds[layer]);
                }
                discard_deferred_experts();
                pos++;
                in_think = 0;
            }
            cache_telemetry_note_token();
            embed_lookup(s->wf, next_token, hidden);
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(s->wf, layer, hidden,
                                    is_full ? s->kv_caches[layer] : NULL,
                                    is_full ? NULL : s->layer_states[layer],
                                    pos,
                                    s->layer_mmaps[layer] != MAP_FAILED ? s->layer_mmaps[layer] : NULL,
                                    s->K, s->layer_fds[layer]);
            }
            discard_deferred_experts();
            pos++;
            break;
        }

        if (next_token == THINK_START_TOKEN) in_think = 1;
        if (next_token == THINK_END_TOKEN) { in_think = 0; think_ended = 1; }
        if (in_think) {
            think_tokens++;
            if (g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = THINK_END_TOKEN;
                in_think = 0;
                think_ended = 1;
            }
        }

        const char *tok_str = decode_token(s->vocab, next_token);
        if (getenv("FINCHMOE_SERVE_DEBUG")) {
            fprintf(stderr, "[serve-tok] id=%d text=%s\n", next_token,
                    tok_str ? tok_str : "(null)");
        }
        if (!in_think && tok_str && gen_resp_len + (int)strlen(tok_str) < 256*1024 - 1) {
            int tlen = (int)strlen(tok_str);
            memcpy(gen_response + gen_resp_len, tok_str, tlen);
            gen_resp_len += tlen;
            gen_response[gen_resp_len] = 0;
        }
        int disconnected;
        if (is_completion)
            disconnected = (sse_send_delta_completion(client_fd, request_id, tok_str) < 0);
        else
            disconnected = (sse_send_delta(client_fd, request_id, tok_str) < 0);
        if (disconnected) {
            fprintf(stderr, "[serve] %s client disconnected, stopping generation\n", request_id);
            break;
        }
        gen_count++;

        cache_telemetry_note_token();
        embed_lookup(s->wf, next_token, hidden);
        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
            fused_layer_forward(s->wf, layer, hidden,
                                is_full ? s->kv_caches[layer] : NULL,
                                is_full ? NULL : s->layer_states[layer],
                                pos,
                                s->layer_mmaps[layer] != MAP_FAILED ? s->layer_mmaps[layer] : NULL,
                                s->K, s->layer_fds[layer]);
        }
        complete_deferred_experts();
        pos++;

        if (s->final_norm_w) {
            float *normed = malloc(HIDDEN_DIM * sizeof(float));
            cpu_rms_norm(hidden, s->final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
            memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
            free(normed);
        }
        lm_head_forward(s->wf, hidden, logits);
        // Once the think block has closed, ban re-entering it — the model
        // otherwise loops "<think>…</think>" fragments mid-answer (the
        // serve long-generation repetition driver).
        if (think_ended) {
            logits[THINK_START_TOKEN] = -INFINITY;
            logits[THINK_END_TOKEN] = -INFINITY;
        }
        next_token = cpu_sample_temp(logits, VOCAB_SIZE, g_temperature, g_top_k);
        logit_diag_dump(logits, VOCAB_SIZE, next_token, gen + 1);
    }

    // Roll back a truncated turn (no EOS — max_tokens exhausted or client
    // disconnected). An abruptly-ended assistant turn poisons the next
    // continuation: the model imitates the turn's ending shape — an
    // unclosed think or a cut-off answer makes the next turn end
    // immediately (the multi-turn "empty turn-2" bug). Restoring the
    // pre-turn snapshot keeps the session context well-formed; the
    // truncated response was already delivered to the client, it just
    // doesn't enter the history. (Natural-EOS turns accumulate normally.)
    if (next_token != EOS_TOKEN_1 && next_token != EOS_TOKEN_2 && has_session) {
        size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
        size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
        size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (s->kv_caches[i] && s->pre_kv_k[i]) {
                size_t sz = (size_t)s->pre_kv_len[i] * kv_dim * sizeof(float);
                for (int p = 0; p < s->pre_kv_len[i]; p++)
                    kv_write(s->kv_caches[i], p,
                             s->pre_kv_k[i] + (size_t)p * kv_dim,
                             s->pre_kv_v[i] + (size_t)p * kv_dim);
                s->kv_caches[i]->len = s->pre_kv_len[i];
                if (g_metal) {
                    int fa_idx = (i + 1) / FULL_ATTN_INTERVAL - 1;
                    if (fa_idx >= 0 && fa_idx < NUM_FULL_ATTN_LAYERS) {
                        memcpy([g_metal->buf_kv_k[fa_idx] contents], s->pre_kv_k[i], sz);
                        memcpy([g_metal->buf_kv_v[fa_idx] contents], s->pre_kv_v[i], sz);
                    }
                }
            }
            if (s->layer_states[i] && s->pre_la_conv[i]) {
                LinearAttnState *ls = (LinearAttnState *)s->layer_states[i];
                memcpy(ls->conv_state, s->pre_la_conv[i], conv_state_size);
                memcpy(ls->ssm_state, s->pre_la_ssm[i], ssm_state_size);
            }
        }
        if (g_metal && g_metal->delta_net_step) {
            for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
                if (s->pre_gpu_delta[i] && g_metal->buf_delta_state[i])
                    memcpy([g_metal->buf_delta_state[i] contents],
                           s->pre_gpu_delta[i], 32*128*128*sizeof(float));
                if (s->pre_gpu_conv[i] && g_metal->buf_conv_state[i])
                    memcpy([g_metal->buf_conv_state[i] contents],
                           s->pre_gpu_conv[i], 3*LINEAR_CONV_DIM*sizeof(float));
                if (s->pre_gpu_qk[i] && g_metal->buf_conv_qk[i])
                    memcpy([g_metal->buf_conv_qk[i] contents],
                           s->pre_gpu_qk[i], 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float));
            }
        }
        pos = s->pre_turn_pos;
        fprintf(stderr, "[serve] %s truncated turn rolled back (session kept clean)\n", request_id);
    }

    double gen_ms = now_ms() - t_gen;
    if (is_completion)
        sse_send_done_completion(client_fd, request_id,
                                  pt->count, gen_count, prefill_ms, gen_ms);
    else
        sse_send_done(client_fd, request_id,
                      pt->count, gen_count, prefill_ms, gen_ms);

    free(gen_response);
    // Save session position for potential continuation
    pthread_mutex_lock(&s->session_mutex);
    s->session_pos = pos;
    pthread_mutex_unlock(&s->session_mutex);
    fprintf(stderr, "[serve] %s session_pos=%d (session=%s)\n",
            request_id, pos,
            s->active_session_id[0] ? s->active_session_id : "(none)");

    fprintf(stderr, "[serve] %s generated=%d tokens in %.0fms (%.2f tok/s)\n",
            request_id, gen_count, gen_ms,
            gen_count > 0 ? gen_count * 1000.0 / gen_ms : 0.0);
    if (g_expert_cache) {
        cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
    } else if (g_malloc_cache) {
        cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
    }

    free(pt->ids);
    free(pt);
    free(hidden);
    free(logits);
    close(client_fd);
}

static void serve_loop(
    int port,
    WeightFile *wf, Vocabulary *vocab,
    void **layer_states, KVCache **kv_caches,
    void **layer_mmaps, int *layer_fds,
    float *hidden, float *logits,
    uint16_t *final_norm_w, int K)
{
    // Ignore SIGPIPE (client disconnect mid-write)
    signal(SIGPIPE, SIG_IGN);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind"); close(server_fd); return;
    }
    if (listen(server_fd, 8) < 0) {
        perror("listen"); close(server_fd); return;
    }

    printf("[serve] Listening on http://0.0.0.0:%d\n", port);
    printf("[serve] Endpoints: POST /v1/chat/completions, POST /v1/completions, GET /v1/models, GET /health\n");
    printf("[serve] Queue: max %d pending requests\n", SERVE_QUEUE_MAX);
    fflush(stdout);

    static uint64_t req_counter = 0;

    // ---- Initialize ServeState ----
    ServeState s_state;
    memset(&s_state, 0, sizeof(s_state));
    s_state.wf = wf;
    s_state.vocab = vocab;
    s_state.layer_states = layer_states;
    s_state.kv_caches = kv_caches;
    s_state.layer_mmaps = layer_mmaps;
    s_state.layer_fds = layer_fds;
    s_state.final_norm_w = final_norm_w;
    s_state.K = K;
    s_state.session_mutex = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;

    // ---- System prompt cache: prefill system prompt once at startup ----
    fprintf(stderr, "[serve] Pre-caching system prompt...\n");
    PromptTokens *sys_pt = tokenize_system_prompt();
    int sys_pos = 0;
    if (sys_pt && sys_pt->count > 0) {
        float *sys_embed_batch = NULL;
        if (sys_pt->count > 1) {
            sys_embed_batch = malloc((size_t)sys_pt->count * HIDDEN_DIM * sizeof(float));
            for (int i = 0; i < sys_pt->count; i++) {
                embed_lookup(wf, sys_pt->ids[i], sys_embed_batch + (size_t)i * HIDDEN_DIM);
            }
        }
        if (prefill_chunk_available() && sys_pt->count > 1 && sys_embed_batch) {
            // Chunked batched prefill for the system prompt.
            prefill_chunked_run(wf, hidden, sys_embed_batch, sys_pt->count, &sys_pos,
                                sys_pt->ids,
                                kv_caches, layer_states, layer_mmaps, K, layer_fds,
                                g_prefill_chunk);
        } else {
            for (int i = 0; i < sys_pt->count - 1; i++) {
                cache_telemetry_note_token();
                if (sys_embed_batch) {
                    memcpy(hidden, sys_embed_batch + (size_t)i * HIDDEN_DIM,
                           HIDDEN_DIM * sizeof(float));
                } else {
                    embed_lookup(wf, sys_pt->ids[i], hidden);
                }
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        sys_pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                discard_deferred_experts();
                sys_pos++;
            }
            {
                cache_telemetry_note_token();
                if (sys_embed_batch) {
                    memcpy(hidden, sys_embed_batch + (size_t)(sys_pt->count - 1) * HIDDEN_DIM,
                           HIDDEN_DIM * sizeof(float));
                } else {
                    embed_lookup(wf, sys_pt->ids[0], hidden);
                }
                for (int layer = 0; layer < NUM_LAYERS; layer++) {
                    int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                    fused_layer_forward(wf, layer, hidden,
                                        is_full ? kv_caches[layer] : NULL,
                                        is_full ? NULL : layer_states[layer],
                                        sys_pos,
                                        layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                        K, layer_fds[layer]);
                }
                complete_deferred_experts();
                sys_pos++;
            }
        }
        if (sys_embed_batch) { free(sys_embed_batch); sys_embed_batch = NULL; }
        sync_cpu_to_gpu_delta_state_serve(layer_states);
        fprintf(stderr, "[serve] System prompt cached: %d tokens prefilled\n", sys_pos);
    }
    free(sys_pt);

    s_state.sys_prompt_len = sys_pos;

    // ---- Save snapshots into ServeState ----
    size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;
    size_t conv_state_size = (CONV_KERNEL_SIZE - 1) * LINEAR_CONV_DIM * sizeof(float);
    size_t ssm_state_size = LINEAR_NUM_V_HEADS * LINEAR_VALUE_DIM * LINEAR_KEY_DIM * sizeof(float);

    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_caches[i]) {
            size_t sz = sys_pos * kv_dim * sizeof(float);
            s_state.kv_k_snapshots[i] = malloc(sz);
            s_state.kv_v_snapshots[i] = malloc(sz);
            for (int p = 0; p < sys_pos; p++)
                for (int hh = 0; hh < NUM_KV_HEADS; hh++) {
                    kv_read_k(kv_caches[i], p, hh,
                              s_state.kv_k_snapshots[i] + (size_t)p * kv_dim + (size_t)hh * HEAD_DIM);
                    kv_read_v(kv_caches[i], p, hh,
                              s_state.kv_v_snapshots[i] + (size_t)p * kv_dim + (size_t)hh * HEAD_DIM);
                }
            s_state.kv_snapshot_len[i] = kv_caches[i]->len;
        }
        if (layer_states[i]) {
            LinearAttnState *ls = (LinearAttnState *)layer_states[i];
            s_state.la_conv_snapshots[i] = malloc(conv_state_size);
            s_state.la_ssm_snapshots[i] = malloc(ssm_state_size);
            memcpy(s_state.la_conv_snapshots[i], ls->conv_state, conv_state_size);
            memcpy(s_state.la_ssm_snapshots[i], ls->ssm_state, ssm_state_size);
        }
    }
    if (g_metal && g_metal->delta_net_step) {
        for (int i = 0; i < NUM_LINEAR_LAYERS; i++) {
            if (g_metal->buf_delta_state[i]) {
                size_t sz = 32*128*128*sizeof(float);
                s_state.gpu_delta_snapshots[i] = malloc(sz);
                memcpy(s_state.gpu_delta_snapshots[i], [g_metal->buf_delta_state[i] contents], sz);
            }
            if (g_metal->buf_conv_state[i]) {
                size_t sz = 3*LINEAR_CONV_DIM*sizeof(float);
                s_state.gpu_conv_snapshots[i] = malloc(sz);
                memcpy(s_state.gpu_conv_snapshots[i], [g_metal->buf_conv_state[i] contents], sz);
                size_t qksz = 2*LINEAR_NUM_V_HEADS*3*LINEAR_KEY_DIM*sizeof(float);
                s_state.gpu_conv_qk_snapshots[i] = malloc(qksz);
                memcpy(s_state.gpu_conv_qk_snapshots[i], [g_metal->buf_conv_qk[i] contents], qksz);
            }
        }
    }

    // ---- Start worker thread ----
    pthread_t worker;
    if (pthread_create(&worker, NULL, serve_worker, &s_state) != 0) {
        fprintf(stderr, "[serve] ERROR: failed to create worker thread\n");
        close(server_fd);
        return;
    }
    pthread_detach(worker);
    fprintf(stderr, "[serve] Worker thread started\n");

    // ---- Accept loop: enqueue POST requests, answer GET immediately ----
    for (;;) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept"); continue; }

        char *reqbuf = malloc(1024 * 1024);
        int reqlen = read_http_request(client_fd, reqbuf, 1024 * 1024);
        if (reqlen <= 0) { free(reqbuf); close(client_fd); continue; }

        char method[16] = {0}, path[256] = {0};
        sscanf(reqbuf, "%15s %255s", method, path);

        // CORS preflight
        if (strcmp(method, "OPTIONS") == 0) {
            http_write_str(client_fd, CORS_RESPONSE);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /health — always respond immediately
        if (strcmp(method, "GET") == 0 && strcmp(path, "/health") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"status\":\"ok\",\"model\":\"qwen3.6-35b-a3b\"}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // GET /v1/models — always respond immediately
        if (strcmp(method, "GET") == 0 && strcmp(path, "/v1/models") == 0) {
            const char *resp =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n"
                "\r\n"
                "{\"object\":\"list\",\"data\":[{\"id\":\"qwen3.6-35b-a3b\","
                "\"object\":\"model\",\"owned_by\":\"local\"}]}\n";
            http_write_str(client_fd, resp);
            free(reqbuf); close(client_fd);
            continue;
        }

        // POST /v1/chat/completions — enqueue for worker thread
        if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/chat/completions") == 0) {
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            int max_gen = extract_max_tokens(body, 8192);
            if (max_gen > 32768) max_gen = 32768;

            char req_session_id[64] = {0};
            int has_session = extract_session_id(body, req_session_id, sizeof(req_session_id));

            char *content = extract_last_content(body);
            if (!content || strlen(content) == 0) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no content in messages\"}\n");
                free(reqbuf); close(client_fd); continue;
            }

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "chatcmpl-%llu", ++req_counter);

            // Try to enqueue
            pthread_mutex_lock(&g_serve_queue.mutex);
            if (g_serve_queue.count >= SERVE_QUEUE_MAX) {
                pthread_mutex_unlock(&g_serve_queue.mutex);
                // Queue full — return 503
                char busy_resp[512];
                int nr = snprintf(busy_resp, sizeof(busy_resp),
                    "HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Type: application/json\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Retry-After: 3\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                    "{\"error\":\"server busy\",\"queue_depth\":%d,\"retry_after_s\":3}\n",
                    SERVE_QUEUE_MAX);
                http_write(client_fd, busy_resp, nr);
                fprintf(stderr, "[serve] %s 503 queue full (depth=%d)\n",
                        request_id, SERVE_QUEUE_MAX);
                free(reqbuf); close(client_fd); continue;
            }

            int slot = g_serve_queue.tail;
            g_serve_queue.entries[slot].client_fd = client_fd;
            g_serve_queue.entries[slot].content = strdup(content);
            g_serve_queue.entries[slot].max_gen = max_gen;
            if (has_session) {
                strncpy(g_serve_queue.entries[slot].session_id, req_session_id, 63);
                g_serve_queue.entries[slot].session_id[63] = '\0';
            } else {
                g_serve_queue.entries[slot].session_id[0] = '\0';
            }
            g_serve_queue.entries[slot].has_session = has_session;
            strncpy(g_serve_queue.entries[slot].request_id, request_id, 63);
            g_serve_queue.entries[slot].request_id[63] = '\0';

            g_serve_queue.tail = (g_serve_queue.tail + 1) % SERVE_QUEUE_MAX;
            g_serve_queue.count++;
            pthread_cond_signal(&g_serve_queue.cond);
            int qdepth = g_serve_queue.count;
            pthread_mutex_unlock(&g_serve_queue.mutex);

            fprintf(stderr, "[serve] %s enqueued (depth=%d/%d)\n",
                    request_id, qdepth, SERVE_QUEUE_MAX);
            free(reqbuf);
            // Note: client_fd is NOT closed — worker thread will close it after generation
            continue;
        }

        // POST /v1/completions — raw text completion (no chat template)
        if (strcmp(method, "POST") == 0 && strcmp(path, "/v1/completions") == 0) {
            char *body = strstr(reqbuf, "\r\n\r\n");
            if (!body) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no body\"}\n");
                free(reqbuf); close(client_fd); continue;
            }
            body += 4;

            int max_gen = extract_max_tokens(body, 16);  // completions default: 16 tokens
            if (max_gen > 32768) max_gen = 32768;

            char *prompt = extract_prompt(body);
            if (!prompt || strlen(prompt) == 0) {
                http_write_str(client_fd,
                    "HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"no prompt field\"}\n");
                free(prompt); free(reqbuf); close(client_fd); continue;
            }

            char request_id[64];
            snprintf(request_id, sizeof(request_id), "cmpl-%llu", ++req_counter);

            // Enqueue
            pthread_mutex_lock(&g_serve_queue.mutex);
            if (g_serve_queue.count >= SERVE_QUEUE_MAX) {
                pthread_mutex_unlock(&g_serve_queue.mutex);
                char busy_resp[512];
                int nr = snprintf(busy_resp, sizeof(busy_resp),
                    "HTTP/1.1 503 Service Unavailable\r\n"
                    "Content-Type: application/json\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Retry-After: 3\r\nConnection: close\r\n\r\n"
                    "{\"error\":\"server busy\",\"queue_depth\":%d,\"retry_after_s\":3}\n",
                    SERVE_QUEUE_MAX);
                http_write(client_fd, busy_resp, nr);
                free(prompt); free(reqbuf); close(client_fd); continue;
            }

            int slot = g_serve_queue.tail;
            g_serve_queue.entries[slot].client_fd = client_fd;
            g_serve_queue.entries[slot].content = prompt;  // raw prompt, no chat template
            g_serve_queue.entries[slot].max_gen = max_gen;
            g_serve_queue.entries[slot].has_session = 0;
            g_serve_queue.entries[slot].session_id[0] = '\0';
            g_serve_queue.entries[slot].is_completion = 1;
            strncpy(g_serve_queue.entries[slot].request_id, request_id, 63);
            g_serve_queue.entries[slot].request_id[63] = '\0';

            g_serve_queue.tail = (g_serve_queue.tail + 1) % SERVE_QUEUE_MAX;
            g_serve_queue.count++;
            pthread_cond_signal(&g_serve_queue.cond);
            int qdepth = g_serve_queue.count;
            pthread_mutex_unlock(&g_serve_queue.mutex);

            fprintf(stderr, "[serve] %s completions enqueued prompt=%zu chars max_tokens=%d (depth=%d/%d)\n",
                    request_id, strlen(prompt), max_gen, qdepth, SERVE_QUEUE_MAX);
            free(reqbuf);
            continue;
        }

        // Unknown endpoint
        const char *resp404 =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n"
            "{\"error\":\"not found\"}\n";
        http_write_str(client_fd, resp404);
        free(reqbuf);
        close(client_fd);
    }
}

// ============================================================================

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH         Model path\n");
    printf("  --weights PATH       model_weights.bin path\n");
    printf("  --manifest PATH      model_weights.json path\n");
    printf("  --vocab PATH         vocab.bin path\n");
    printf("  --prompt-tokens PATH prompt_tokens.bin path\n");
    printf("  --prompt TEXT         Prompt text (requires encode_prompt.py)\n");
    printf("  --tokens N           Max tokens to generate (default: 20)\n");
    printf("  --k N                Active experts per layer (default: 8)\n");
    printf("  --cache-entries N    Expert LRU cache size (default: 2500, 0 = disabled)\n");
    printf("  --malloc-cache N     Malloc expert cache entries (e.g., 2581 = 17GB for 80%% hit)\n");
    printf("  --cpu-linear         Disable fused GPU delta-net and use the older CPU/hybrid linear path\n");
    printf("  --timing             Enable per-layer timing breakdown\n");
    printf("  --freq               Enable expert frequency tracking + analysis\n");
    printf("  --cache-telemetry    Report cold vs eviction misses and reuse distance\n");
    printf("  --2bit               Use 2-bit quantized experts (packed_experts_2bit/)\n");
    printf("  --3bit               Use 3-bit quantized experts (packed_experts_3bit/) [DEFAULT]\n");
    printf("  --4bit               Use 4-bit quantized experts (packed_experts/)\n");
    printf("  --int8-experts       Use 8-bit quantized experts (packed_experts_8bit/)\n");
    printf("  --gpu-linear         Alias for the fused GPU delta-net path (default)\n");
    printf("  --predict            Enable temporal expert prediction (prefetch during CMD1_wait)\n");
    printf("  --collect-routing F  Log routing data to binary file F (for predictor training)\n");
    printf("  --think-budget N     Max thinking tokens before force </think> (default: 2048, 0=unlimited)\n");
    printf("  --debug-layers       Print per-layer hidden state statistics\n");
    printf("  --gpu-experts        Use GPU expert path (now default, ~15 tok/s)\n");
    printf("  --cpu-experts        Use CPU expert path for debugging (~2 tok/s)\n");
    printf("  --compare-experts N  Compare GPU vs CPU expert outputs for layer N\n");
    printf("  --temperature F      Sampling temperature (default: 0.3, 0=greedy)\n");
    printf("  --top-k N            Top-k sampling (default: 40, 1=greedy)\n");
    printf("  --rep-penalty F      Repetition penalty (default: 1.15, 1.0=disabled)\n");
    printf("  --no-think           Disable thinking mode (empty <think/> block)\n");
    printf("  --serve PORT         Run HTTP server (OpenAI-compatible API)\n");
    printf("  --max-seq-len N      Max context length for KV cache (default: 262144 = 256K, model limit)\n");
    printf("  --gpu-kv-seq N       GPU KV buffer pre-allocation in tokens (default: 8192)\n");
    printf("  --low-memory         Skip Metal weight buffer wrap (slower, safe for 16GB)\n");
    printf("  --logit-diag N       Dump logit top-20 + entropy every N tokens (debug drift)\n");
    printf("  --help               This message\n");
}

// Phase C S6: GPU keep-alive ticker (FINCHMOE_CBLAT=KA / FINCHMOE_PF_KEEPALIVE).
// Commits a 1-thread no-op CB every ~300us while g_ka_running — continuous
// tiny submissions keep the GPU clocked through CPU gaps (chain/pread),
// killing the wake tax on the next real submission (CBLAT: 0.013ms
// back-to-back vs 4.4ms after a 3ms gap).
static int g_ka_running = 0;
static void *ka_ticker_main(void *arg) {
    (void)arg;
    MetalCtx *ctx = g_metal;
    if (!ctx || !ctx->ka_nop_pipe) return NULL;
    id<MTLBuffer> dummy = [ctx->device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    *((float *)[dummy contents]) = 0.0f;
    // Probe knobs: FINCHMOE_KA_US interval, FINCHMOE_KA_THREADS per no-op.
    int ka_us = 300, ka_threads = 1;
    {
        const char *e1 = getenv("FINCHMOE_KA_US");
        if (e1 && atoi(e1) > 0) ka_us = atoi(e1);
        const char *e2 = getenv("FINCHMOE_KA_THREADS");
        if (e2 && atoi(e2) > 0) ka_threads = atoi(e2);
    }
    int ticks = 0;
    while (g_ka_running) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->ka_nop_pipe];
            [enc setBuffer:dummy offset:0 atIndex:0];
            uint32_t nt = (uint32_t)ka_threads;
            [enc dispatchThreadgroups:MTLSizeMake((nt + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
        }
        ticks++;
        usleep(ka_us);
    }
    fprintf(stderr, "[keepalive] ticker stopped after %d no-op submissions\n", ticks);
    return NULL;
}
static pthread_t g_ka_thread;
static void ka_ticker_start(void) {
    if (g_ka_running || !g_metal || !g_metal->ka_nop_pipe) return;
    g_ka_running = 1;
    pthread_create(&g_ka_thread, NULL, ka_ticker_main, NULL);
    fprintf(stderr, "[keepalive] ticker started (300us no-op CBs)\n");
}
static void ka_ticker_stop(void) {
    if (!g_ka_running) return;
    g_ka_running = 0;
    pthread_join(g_ka_thread, NULL);
}

int main(int argc, char **argv) {
    // Unbuffered output — critical for diagnosing early-exit issues
    setlinebuf(stdout);
    setlinebuf(stderr);

    srand48(time(NULL));
    @autoreleasepool {
        // ---- Ultra-early memory safety check ----
        // Before ANY allocation, verify the system has enough free memory.
        // On machines without swap, Metal buffer allocation + large mmap
        // can trigger jetsam (SIGKILL) if the system can't find physical pages.
        // This check runs before Metal init, before mmap, before everything.
        {
            size_t avail = get_available_memory();
            printf("[bootstrap] Available memory: %s (free + purgeable + speculative)\n",
                   format_mem_size(avail));

            // Hard limits:
            //   < 3GB: REFUSE to run — mmap alone needs ~3GB of address space
            //                    backing; SIGKILL is virtually certain.
            //   3-7GB: WARN but allow — may work with --low-memory + small KV,
            //          but the weight file mmap still risks jetsam.
            //   >= 7GB: Safe for full Metal zero-copy.
            size_t hard_min_memory  = 3ULL * 1024 * 1024 * 1024;  // 3GB — absolute floor
            size_t soft_min_memory  = 7ULL * 1024 * 1024 * 1024;  // 7GB — safe for full speed

            if (avail > 0 && avail < hard_min_memory) {
                fprintf(stderr,
                        "\n"
                        "╔══════════════════════════════════════════════════════════════════╗\n"
                        "║  FATAL: Not enough physical memory to run safely               ║\n"
                        "╠══════════════════════════════════════════════════════════════════╣\n"
                        "║  Available: %7s  (need at least %s)                    ║\n"
                        "║  Metal GPU buffers require wired (non-swappable) physical pages ║\n"
                        "║  on Apple Silicon unified memory. The 5GB weight file + GPU     ║\n"
                        "║  buffers need ~3GB of immediately available physical pages.     ║\n"
                        "╠══════════════════════════════════════════════════════════════════╣\n"
                        "║  Fixes:                                                         ║\n"
                        "║  1. Close memory-heavy apps (Claude Code = ~2GB)                ║\n"
                        "║  2. Wait 5-10 min for system to settle after restart            ║\n"
                        "║     (Spotlight indexing, caches rebuilding etc.)                ║\n"
                        "║  3. Use --low-memory --gpu-kv-seq 512 for minimal GPU footprint ║\n"
                        "╚══════════════════════════════════════════════════════════════════╝\n"
                        "\n",
                        format_mem_size(avail), format_mem_size(hard_min_memory));
                return 1;
            }

            if (avail > 0 && avail < soft_min_memory) {
                fprintf(stderr,
                        "\n"
                        "╔══════════════════════════════════════════════════════════════════╗\n"
                        "║  WARNING: Tight memory — may trigger SIGKILL/jetsam            ║\n"
                        "╠══════════════════════════════════════════════════════════════════╣\n"
                        "║  Available: %7s  (recommended minimum: %s)             ║\n"
                        "║  Consider: --low-memory --gpu-kv-seq 512 for safest path        ║\n"
                        "╚══════════════════════════════════════════════════════════════════╝\n"
                        "\n",
                        format_mem_size(avail), format_mem_size(soft_min_memory));
                // Continue with warning — per-phase checks provide finer guards
            }
        }

        const char *model_path = MODEL_PATH_DEFAULT;
        int model_path_from_user = 0;  // set when --model flag used
        const char *weights_path = NULL;
        const char *gguf_path = NULL;
        const char *manifest_path = NULL;
        const char *vocab_path = NULL;
        const char *prompt_tokens_path = NULL;
        const char *prompt_text = NULL;
        int max_tokens = 20;
        int K = 8;  // model trained with 8 experts/token (K=2 produces garbage)
        int cache_entries = 0;  // default 0: trust OS page cache (38% faster than Metal LRU)
        int malloc_cache_entries = 0;  // 0 = disabled (override with --malloc-cache)
        int serve_port = 0;  // 0 = disabled, >0 = HTTP serve mode

        static struct option long_options[] = {
            {"model",         required_argument, 0, 'm'},
            {"weights",       required_argument, 0, 'w'},
            {"manifest",      required_argument, 0, 'j'},
            {"vocab",         required_argument, 0, 'v'},
            {"prompt-tokens", required_argument, 0, 'p'},
            {"prompt",        required_argument, 0, 'P'},
            {"tokens",        required_argument, 0, 't'},
            {"k",             required_argument, 0, 'k'},
            {"cache-entries",  required_argument, 0, 'C'},
            {"malloc-cache",   required_argument, 0, 'M'},
            {"cpu-linear",    no_argument,       0, 'L'},
            {"skip-linear",   no_argument,       0, 'S'},
            {"timing",        no_argument,       0, 'T'},
            {"freq",          no_argument,       0, 'F'},
            {"cache-telemetry", no_argument,     0, 'E'},
            {"2bit",          no_argument,       0, '2'},
            {"3bit",          no_argument,       0, '3'},
            {"4bit",          no_argument,       0, '4'},
            {"int8-experts",  no_argument,       0, '8'},
            {"gpu-linear",    no_argument,       0, 'G'},
            {"think-budget",  required_argument, 0, 'B'},
            {"temperature",   required_argument, 0, 'e'},
            {"top-k",         required_argument, 0, 'o'},
            {"no-think",      no_argument,       0, 'H'},
            {"serve",         required_argument, 0, 'R'},
            {"predict",       no_argument,       0, 'D'},
            {"mtp",           no_argument,       0, 'J'},
            {"rep-penalty",   required_argument, 0, 'r'},
            {"min-p",         required_argument, 0, 702},
            {"gguf",          required_argument, 0, 703},
            {"prefill-chunk", required_argument, 0, 'b'},
            {"debug-layers",  no_argument,       0, 'X'},
            {"gpu-experts",   no_argument,       0, 'U'},
            {"cpu-experts",   no_argument,       0, 'V'},
            {"compare-experts", required_argument, 0, 'Y'},
            {"collect-routing", required_argument, 0, 'Z'},
            {"max-seq-len",   required_argument, 0, 'N'},
            {"gpu-kv-seq",    required_argument, 0, 'Q'},
            {"low-memory",    no_argument,       0, 'l'},
            {"kv-fp16",       no_argument,       0, 700},
            {"kv-turbo",      no_argument,       0, 701},
            {"dump-logits",   required_argument, 0, 'I'},
            {"logit-diag",    required_argument, 0, 'A'},
            {"help",          no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int c;
        while ((c = getopt_long(argc, argv, "m:w:j:v:p:P:t:k:C:M:R:B:N:Q:e:o:I:r:b:lHLSTFE234GhXUY:VJ", long_options, NULL)) != -1) {
            switch (c) {
                case 'm': model_path = optarg; model_path_from_user = 1; break;
                case 'w': weights_path = optarg; break;
                case 'j': manifest_path = optarg; break;
                case 'v': vocab_path = optarg; break;
                case 'p': prompt_tokens_path = optarg; break;
                case 'P': prompt_text = optarg; break;
                case 't': max_tokens = atoi(optarg); break;
                case 'k': K = atoi(optarg); break;
                case 'C': cache_entries = atoi(optarg); break;
                case 'M': malloc_cache_entries = atoi(optarg); break;
                case 'L': gpu_linear_attn_enabled = 0; break;
                case 'S': linear_attn_bypass = 1; break;
                case 'T': g_timing_enabled = 1; break;
                case 'F': g_freq_tracking = 1; break;
                case 'E': g_cache_telemetry_enabled = 1; break;
                case '2': g_use_2bit = 1; g_use_3bit = 0; g_use_int8 = 0; g_use_1bit = 0; break;
                case '3': g_use_3bit = 1; g_use_2bit = 0; g_use_int8 = 0; g_use_1bit = 0; break;
                case '4': g_use_3bit = 0; g_use_2bit = 0; g_use_int8 = 0; g_use_1bit = 0; break;
                case '8': g_use_int8 = 1; g_use_3bit = 0; g_use_2bit = 0; g_use_1bit = 0; break;
                case 'G': gpu_linear_attn_enabled = 1; break;
                case 'D': g_pred_enabled = 1; break;
                case 'J': g_use_mtp = 1; break;
                case 'r': g_rep_penalty = atof(optarg); break;
                case 'b': g_prefill_chunk = atoi(optarg);
                           if (g_prefill_chunk > PREFILL_CHUNK_MAX) g_prefill_chunk = PREFILL_CHUNK_MAX;
                           break;
                case 'X': g_debug_layers = 1; break;
                case 'U': g_gpu_experts = 1; break;
                case 'V': g_cpu_experts = 1; break;
                case 'Y': g_compare_experts = atoi(optarg); break;
                case 'Z':
                    g_routing_log = fopen(optarg, "wb");
                    if (!g_routing_log) {
                        fprintf(stderr, "ERROR: cannot open routing log: %s\n", optarg);
                        return 1;
                    }
                    break;
                case 'B': g_think_budget = atoi(optarg); break;
                case 'e': g_temperature = atof(optarg); break;
                case 'o': g_top_k = atoi(optarg); break;
                case 'H': g_no_think = 1; break;
                case 'l': g_low_memory = 1; break;
            case 700: g_kv_type = KV_FP16; break;
            case 702: g_min_p = atof(optarg); break;
            case 703: gguf_path = optarg; break;
            case 701: g_kv_type = KV_TURBO; break;
                case 'I': g_dump_logits_path = optarg; break;
                case 'A': g_logit_diag_interval = atoi(optarg); break;
                case 'N': g_max_seq_len = atoi(optarg); break;
                case 'Q': g_gpu_kv_seq = atoi(optarg); break;
                case 'R': serve_port = atoi(optarg); break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        // Build default paths
        char default_weights[1024], default_manifest[1024], default_vocab[1024];

        // Try to find files relative to the executable
        if (!weights_path) {
            snprintf(default_weights, sizeof(default_weights),
                     "metal_infer/model_weights.bin");
            if (access(default_weights, R_OK) != 0) {
                snprintf(default_weights, sizeof(default_weights),
                         "model_weights.bin");
            }
            weights_path = default_weights;
        }
        if (!manifest_path) {
            snprintf(default_manifest, sizeof(default_manifest),
                     "metal_infer/model_weights.json");
            if (access(default_manifest, R_OK) != 0) {
                snprintf(default_manifest, sizeof(default_manifest),
                         "model_weights.json");
            }
            manifest_path = default_manifest;
        }
        if (!vocab_path) {
            snprintf(default_vocab, sizeof(default_vocab),
                     "metal_infer/vocab.bin");
            if (access(default_vocab, R_OK) != 0) {
                snprintf(default_vocab, sizeof(default_vocab),
                         "vocab.bin");
            }
            vocab_path = default_vocab;
        }

        // ---- Initialize Metal ----
        printf("[phase] Initializing Metal (device + shaders + GPU buffers)...\n");
        g_metal = metal_setup();
        if (!g_metal) {
            fprintf(stderr, "WARNING: Metal init failed, falling back to CPU\n");
        } else {
            printf("[phase] Metal initialization complete\n");
            // PERF PROBE (FINCHMOE_CBLAT=N): empty command-buffer commit+wait
            // round-trip latency — back-to-back AND after idle gaps (the GPU
            // wake-latency theory: each wait in the chunked-prefill pipeline
            // follows a CPU gap (chain/pread), so the gap variants measure
            // the real per-wait cost).
            {
                const char *ce = getenv("FINCHMOE_CBLAT");
                if (ce) {
                    int n = atoi(ce);
                    if (n < 1) n = 200;
                    const int gaps[4] = {0, 100, 1000, 3000};  // us
                    for (int gi = 0; gi < 4; gi++) {
                        double t0 = now_ms();
                        for (int i = 0; i < n; i++) {
                            id<MTLCommandBuffer> cb = [g_metal->queue commandBuffer];
                            [cb commit];
                            [cb waitUntilCompleted];
                            if (gaps[gi] > 0) usleep(gaps[gi]);
                        }
                        double dt = now_ms() - t0;
                        fprintf(stderr, "[cblat] gap=%5dus: %d CBs in %.3f ms = %.4f ms avg\n",
                                gaps[gi], n, dt, dt / n);
                    }
                    // Kernel variant (FINCHMOE_CBLAT=K): each CB carries one
                    // 1-thread ka_nop dispatch — isolates whether the ~2ms
                    // per-CB overhead scales with CB content vs the empty-CB
                    // floor (13.5us).
                    if (strcmp(ce, "K") == 0) {
                        const int gaps3[4] = {0, 100, 1000, 3000};
                        for (int gi = 0; gi < 4; gi++) {
                            double t0 = now_ms();
                            for (int i = 0; i < n; i++) {
                                id<MTLCommandBuffer> cb = [g_metal->queue commandBuffer];
                                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                                [enc setComputePipelineState:g_metal->ka_nop_pipe];
                                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                                    threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                [enc endEncoding];
                                [cb commit];
                                [cb waitUntilCompleted];
                                if (gaps3[gi] > 0) usleep(gaps3[gi]);
                            }
                            double dt = now_ms() - t0;
                            fprintf(stderr, "[cblat-K] gap=%5dus: %d kernel CBs in %.3f ms = %.4f ms avg\n",
                                    gaps3[gi], n, dt, dt / n);
                        }
                    }
                    // Keep-alive variant (FINCHMOE_CBLAT=KA): a ticker thread
                    // commits a 1-thread no-op CB every 300us during the
                    // gap — tests whether continuous tiny submissions keep
                    // the GPU clocked (kills the wake tax on the next real
                    // submission).
                    if (strcmp(ce, "KA") == 0) {
                        ka_ticker_start();
                        usleep(100000);   // let the ticker ramp the GPU
                        const int gaps2[1] = {3000};
                        for (int gi = 0; gi < 1; gi++) {
                            double t0 = now_ms();
                            for (int i = 0; i < n; i++) {
                                id<MTLCommandBuffer> cb = [g_metal->queue commandBuffer];
                                [cb commit];
                                [cb waitUntilCompleted];
                                if (gaps2[gi] > 0) usleep(gaps2[gi]);
                            }
                            double dt = now_ms() - t0;
                            fprintf(stderr, "[cblat-KA] gap=%5dus + ticker: %d CBs in %.3f ms = %.4f ms avg\n",
                                    gaps2[gi], n, dt, dt / n);
                        }
                        ka_ticker_stop();
                    }
                }
            }
        }

        // ---- Initialize persistent I/O thread pool ----
        io_pool_init();

        // ---- Initialize malloc expert cache (if requested) ----
        // (Dead weight in GGUF mode — the expert slabs live in the mmap and
        // are read directly; the cache would pre-allocate GBs never used.)
        if (malloc_cache_entries > 0 && !gguf_path) {
            g_malloc_cache = malloc_cache_init(malloc_cache_entries, g_metal ? g_metal->device : MTLCreateSystemDefaultDevice());
            cache_entries = 0;  // disable Metal LRU cache when malloc cache is active
        }

        // ---- Initialize expert LRU cache ----
        if (cache_entries > 0 && g_metal && !gguf_path) {
            g_expert_cache = expert_cache_new(g_metal->device, cache_entries);
        }

        printf("=== Qwen3.6-35B-A3B Metal Inference Engine ===\n");
        printf("Model:    %s\n", model_path);
        printf("Weights:  %s\n", weights_path);
        printf("Manifest: %s\n", manifest_path);
        printf("Vocab:    %s\n", vocab_path);
        printf("K:        %d experts/layer\n", K);
        printf("Sample:   temp=%.2f top_k=%d%s\n", g_temperature, g_top_k,
               g_temperature <= 0.0f ? " (greedy)" : "");
        printf("Tokens:   %d\n", max_tokens);
        if (g_malloc_cache) {
            printf("Cache:    malloc %d entries (%.1f GB)\n",
                   malloc_cache_entries, (double)malloc_cache_entries * active_expert_size() / 1e9);
        } else {
            printf("Cache:    %d entries%s\n", cache_entries,
                   cache_entries > 0 ? "" : " (disabled)");
        }
        printf("Context:  max_seq=%d (%.0fK tokens), GPU_KV=%d tokens\n",
               g_max_seq_len, g_max_seq_len / 1000.0, g_gpu_kv_seq);

        // ---- Memory budget report ----
        // Compute expected GPU memory usage before loading weights.
        // This helps users tune flags to avoid OOM on memory-constrained machines.
        {
            // Stat weight file to get its size
            struct stat wf_st;
            size_t wf_size = 0;
            if (stat(weights_path, &wf_st) == 0) wf_size = wf_st.st_size;

            size_t kv_dim = NUM_KV_HEADS * HEAD_DIM;  // 512
            size_t kv_cache_mem = NUM_FULL_ATTN_LAYERS * 2 * kv_dim * g_gpu_kv_seq * sizeof(float);
            size_t delta_state_mem = NUM_LINEAR_LAYERS * (32*128*128 + 3*LINEAR_CONV_DIM) * sizeof(float);
            size_t delta_scratch_mem = (2048+2048+8192+64+64+8192+12288+12288) * sizeof(float);
            size_t attn_mem = (NUM_ATTN_HEADS * HEAD_DIM * 3 +
                               (size_t)NUM_ATTN_HEADS * g_gpu_kv_seq +
                               NUM_ATTN_HEADS * HEAD_DIM * 2) * sizeof(float);
            size_t expert_mem_per_slot = ((EXPERT_SIZE_MAX + 2*1024*1024 - 1) & ~(2*1024*1024 - 1)) +
                                         4 * MOE_INTERMEDIATE * sizeof(float) + HIDDEN_DIM * sizeof(float);
            size_t expert_mem = MAX_K * 2 * expert_mem_per_slot;
            size_t shared_expert_mem = (3 * SHARED_INTERMEDIATE + HIDDEN_DIM) * sizeof(float);
            size_t misc_mem = (HIDDEN_DIM * 4 + sizeof(float) * 3 + 10 * sizeof(float) +
                               LINEAR_TOTAL_VALUE * sizeof(float) + VOCAB_SIZE * sizeof(float) +
                               MAX_BATCH_SLOTS * (NUM_ATTN_HEADS * HEAD_DIM * 2) * sizeof(float));
            size_t total_gpu_mem = kv_cache_mem + delta_state_mem + delta_scratch_mem +
                                   attn_mem + expert_mem + shared_expert_mem + misc_mem;

            printf("\n");
            printf("  ┌─ GPU Memory Budget ─────────────────────────────────────┐\n");
            printf("  │ Weight file (mmap):          %7s                     │\n", format_mem_size(wf_size));
            printf("  │ Weight Metal wrap (if safe): %7s  (zero-copy)         │\n", format_mem_size(wf_size));
            printf("  │ KV caches (%d layers):       %7s  (%.1f MB each)     │\n",
                   NUM_FULL_ATTN_LAYERS, format_mem_size(kv_cache_mem),
                   (double)(2 * kv_dim * g_gpu_kv_seq * sizeof(float)) / 1e6);
            printf("  │ Delta-net state (%d layers): %7s                     │\n",
                   NUM_LINEAR_LAYERS, format_mem_size(delta_state_mem));
            printf("  │ Delta-net scratch:           %7s                     │\n",
                   format_mem_size(delta_scratch_mem));
            printf("  │ Expert multi-buf (%d slots): %7s  (2MB-aligned)      │\n",
                   MAX_K * 2, format_mem_size(expert_mem));
            printf("  │ Other GPU bufs:              %7s                     │\n",
                   format_mem_size(attn_mem + shared_expert_mem + misc_mem));
            printf("  ├────────────────────────────────────────────────────────┤\n");
            printf("  │ GPU bufs (excl. weights):    %7s                     │\n",
                   format_mem_size(total_gpu_mem));
            printf("  │ GPU bufs + weight wrap:      %7s                     │\n",
                   format_mem_size(total_gpu_mem + wf_size));
            printf("  └────────────────────────────────────────────────────────┘\n");

            size_t avail = get_available_memory();
            if (avail > 0 && wf_size > 0) {
                size_t peak = total_gpu_mem + wf_size;
                if (avail < peak) {
                    printf("  ⚠️  Available memory (%s) < peak GPU usage (%s)\n",
                           format_mem_size(avail), format_mem_size(peak));
                    printf("     Suggestions: --gpu-kv-seq 512 (saves %s), close other apps\n",
                           format_mem_size(kv_cache_mem - (NUM_FULL_ATTN_LAYERS * 2 * kv_dim * 512 * sizeof(float))));
                } else {
                    printf("  ✅ Available memory (%s) >= peak GPU usage (%s)\n",
                           format_mem_size(avail), format_mem_size(peak));
                }
            }
            printf("\n");
        }

        double t0 = now_ms();

        // ---- Load weights ----
        printf("[phase] Loading weights (mmap + manifest)...\n");
        WeightFile *wf = gguf_path ? open_gguf(gguf_path)
                                 : open_weights(weights_path, manifest_path);
        if (!wf) {
            fprintf(stderr, "ERROR: Failed to load weights\n");
            return 1;
        }

        // If user didn't specify --model, use the path from the manifest.
        // The manifest's "model" field points to the directory containing
        // packed_experts_2bit/ etc., which is needed for auto-detection.
        if (!model_path_from_user && wf->manifest->model_path) {
            model_path = wf->manifest->model_path;
        }

        // Wrap weight file for Metal GPU access (zero-copy, requires ~5GB free)
        // When --low-memory is set, skip wrapping: all GPU matmuls fall back
        // to CPU (reading directly from mmap). This is slower (1-3 tok/s vs
        // 10-15 tok/s) but won't trigger OOM/jetsam on 16GB machines.
        if (g_metal && !g_low_memory) {
            metal_set_weights(g_metal, wf->data, wf->size);
        } else if (g_metal && g_low_memory) {
            printf("[metal] --low-memory: skipping weight buffer wrap (CPU fallback)\n");
        }

        // Print GPU matmul mode
        if (g_metal && g_metal->wf_buf) {
            printf("[mode]  GPU matmuls: zero-copy (fast, %.2f GB Metal buffer)\n",
                   (double)wf->size / 1e9);
        } else if (g_metal) {
            printf("[mode]  GPU matmuls: fallback (safe, CPU reads from mmap, slower)\n");
        } else {
            printf("[mode]  GPU matmuls: CPU only (no Metal device)\n");
        }

        // Initialize MTP (Multi-Token Prediction) speculative decoding head
        // MTP weights are only needed with --mtp (the head is not shippable —
        // see the MTP notes); the 4.96 GB model_weights_mtp.bin + the 453 MB
        // layer_40 pack can be deleted when MTP is not used.
        if (g_use_mtp) mtp_init(wf, model_path);
        else fprintf(stderr, "[mtp] skipped (--mtp not set — MTP weight files optional)\n");

        // ---- Load vocabulary ----
        fflush(stdout); fflush(stderr);
        printf("[phase] Loading vocabulary...\n");
        fflush(stdout);
        Vocabulary *vocab = load_vocab(vocab_path);
        if (!vocab) {
            fprintf(stderr, "ERROR: Failed to load vocabulary\n");
            return 1;
        }

        // ---- Get prompt tokens (skip in serve mode) ----
        PromptTokens *pt = NULL;
        if (serve_port == 0) {
            if (prompt_text) {
                // Use ChatML template for instruct model
                pt = tokenize_chat_message(prompt_text);
                if (!pt) {
                    fprintf(stderr, "ERROR: Failed to encode prompt. Make sure encode_prompt.py exists.\n");
                    return 1;
                }
            } else if (!prompt_tokens_path) {
                pt = encode_prompt_text_to_tokens("Hello, what is");
                if (!pt) {
                    fprintf(stderr, "ERROR: No prompt tokens and encode_prompt.py not found\n");
                    return 1;
                }
            } else {
                pt = load_prompt_tokens(prompt_tokens_path);
            }

            if (!pt) {
                fprintf(stderr, "ERROR: Failed to load prompt tokens from %s\n", prompt_tokens_path);
                return 1;
            }
            printf("[prompt] %d tokens:", pt->count);
            for (int i = 0; i < pt->count && i < 20; i++) {
                printf(" %d", pt->ids[i]);
            }
            printf("\n");
        }

        // ---- Auto-detect expert format (only when no explicit flag; 3-bit is the default) ----
        if (!g_use_1bit && !g_use_2bit && !g_use_3bit && !g_use_int8) {
            char probe[1024];
            // Check 1-bit first (smallest, fastest)
            snprintf(probe, sizeof(probe), "%s/packed_experts_1bit/layer_00.bin", model_path);
            int pfd1 = open(probe, O_RDONLY);
            if (pfd1 >= 0) { close(pfd1); g_use_1bit = 1; printf("[auto] Using 1-bit experts\n"); }
        }
        if (!g_use_1bit && !g_use_2bit && !g_use_3bit && !g_use_int8) {
            char probe[1024];
            // Check 8-bit
            snprintf(probe, sizeof(probe), "%s/packed_experts_8bit/layer_00.bin", model_path);
            int pfd8 = open(probe, O_RDONLY);
            if (pfd8 >= 0) { close(pfd8); g_use_int8 = 1; }
            if (!g_use_int8) {
                // Check 2-bit
                snprintf(probe, sizeof(probe), "%s/packed_experts_2bit/layer_00.bin", model_path);
                int pfd2 = open(probe, O_RDONLY);
                if (pfd2 >= 0) {
                    close(pfd2);
                    // Only use 2-bit if 4-bit not found
                    snprintf(probe, sizeof(probe), "%s/packed_experts/layer_00.bin", model_path);
                    int pfd4 = open(probe, O_RDONLY);
                    if (pfd4 < 0) {
                        g_use_2bit = 1;
                        printf("[auto] Using 2-bit experts (4-bit not found)\n");
                    } else {
                        close(pfd4);
                    }
                }
            }
        }
        if (g_use_int8) printf("[auto] Using 8-bit experts\n");
        if (g_use_2bit) printf("[auto] Using 2-bit experts\n");

        // 3-bit is the default; if its files are missing, fall back to 4-bit.
        if (g_use_3bit) {
            char probe[1024];
            snprintf(probe, sizeof(probe), "%s/packed_experts_3bit/layer_00.bin", model_path);
            int pfd3 = open(probe, O_RDONLY);
            if (pfd3 < 0) {
                g_use_3bit = 0;
                printf("[auto] packed_experts_3bit missing — falling back to 4-bit experts\n");
            } else {
                close(pfd3);
            }
        }

        // Print quant and linear info now that auto-detect has settled
        printf("Quant:    %s experts (%zu bytes each)\n",
               g_use_1bit ? "1-bit" : (g_use_2bit ? "2-bit" : (g_use_3bit ? "3-bit" : (g_use_int8 ? "8-bit" : "4-bit"))),
               active_expert_size());
        printf("Linear:   %s\n", gpu_linear_attn_enabled ? "fused GPU delta-net" : "CPU/hybrid fallback");

        // ---- Load static per-layer hot sets (prefill expert prefetch) ----
        // hot_sets.bin: [NUM_LAYERS][PF_HOT_MAX] int32 expert ids, most
        // frequently selected first (built by build_hot_sets.py from
        // --collect-routing logs). The first g_pf_hot_slots entries per
        // layer are prefetched one layer ahead during chunked prefill.
        memset(g_hot_slot, -1, sizeof(g_hot_slot));
        memset(g_hot_expert, 0, sizeof(g_hot_expert));
        if (g_prefill_chunk > 0 && g_pf_hot_slots > 0) {
            FILE *hf = fopen("hot_sets.bin", "rb");
            if (hf) {
                int32_t hot[PF_HOT_MAX];
                for (int L = 0; L < NUM_LAYERS; L++) {
                    if (fread(hot, sizeof(int32_t), PF_HOT_MAX, hf) != PF_HOT_MAX) break;
                    for (int s = 0; s < g_pf_hot_slots; s++) {
                        if (hot[s] >= 0 && hot[s] < NUM_EXPERTS) {
                            g_hot_slot[L][hot[s]] = s;
                            g_hot_expert[L][s] = hot[s];
                        }
                    }
                }
                fclose(hf);
                g_hot_loaded = 1;
                printf("[pf-prefetch] hot_sets.bin loaded (%d hot slots/layer)\n", g_pf_hot_slots);
            } else {
                fprintf(stderr, "[pf-prefetch] hot_sets.bin not found — prefetch disabled "
                                "(build with build_hot_sets.py)\n");
            }
        }

        // ---- Open + mmap packed expert files ----
        // Tiered I/O: two fds per layer file.
        //   layer_fds[i]      = warm fd (page cached) — for experts seen before
        //   layer_fds_cold[i] = cold fd (F_NOCACHE)   — for first-time expert reads
        // Seen-expert bitset tracks which (layer, expert) pairs have been read before.
        // First read goes through cold fd (no page cache pollution).
        // Subsequent reads go through warm fd (page cache hit = 32 GB/s vs 5.5 GB/s).
        int layer_fds[NUM_LAYERS];
        int layer_fds_cold[NUM_LAYERS];
        void *layer_mmaps[NUM_LAYERS];
        size_t layer_mmap_sizes[NUM_LAYERS];
        int expert_layers_available = 0;

        // Reset the global seen-expert bitset
        memset(g_expert_seen, 0, sizeof(g_expert_seen));

        for (int i = 0; i < NUM_LAYERS; i++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s/layer_%02d.bin", model_path,
                     g_use_1bit ? "packed_experts_1bit" :
                     g_use_2bit ? "packed_experts_2bit" :
                     g_use_3bit ? "packed_experts_3bit" :
                     g_use_int8 ? "packed_experts_8bit" : "packed_experts", i);
            layer_fds[i] = open(path, O_RDONLY);
            layer_fds_cold[i] = -1;  // no longer used (trust OS page cache)
            layer_mmaps[i] = MAP_FAILED;
            layer_mmap_sizes[i] = 0;
            if (layer_fds[i] >= 0) {
                expert_layers_available++;
                // Disable readahead: expert reads are random (different offsets per token).
                // Read-ahead prefetches adjacent data we won't use, wasting SSD bandwidth.
                fcntl(layer_fds[i], F_RDAHEAD, 0);
                struct stat st;
                if (fstat(layer_fds[i], &st) == 0 && st.st_size > 0) {
                    layer_mmaps[i] = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, layer_fds[i], 0);
                    if (layer_mmaps[i] != MAP_FAILED) {
                        layer_mmap_sizes[i] = st.st_size;
                        // No madvise: kernel default is best.
                        // MADV_RANDOM disables readahead (tested: hurts).
                        // MADV_SEQUENTIAL doesn't reduce I/O fragmentation (tested: no effect).
                        // The kernel fragments 3.9MB preads into ~5.7 disk ops regardless
                        // of hints — this is inherent to the page cache's physical page layout.
                    }
                }
            }
        }
        printf("[experts] %d/%d packed layer files available (mmap'd)\n", expert_layers_available, NUM_LAYERS);

        // ---- LZ4 compressed experts: auto-detect and load ----
        {
            char lz4_probe[1024];
            snprintf(lz4_probe, sizeof(lz4_probe), "%s/packed_experts_lz4/layer_00.bin", model_path);
            if (!g_use_2bit && access(lz4_probe, R_OK) == 0) {
                int lz4_layers = 0;
                for (int i = 0; i < NUM_LAYERS; i++) {
                    char lz4_path[1024];
                    snprintf(lz4_path, sizeof(lz4_path), "%s/packed_experts_lz4/layer_%02d.bin", model_path, i);
                    int lz4_fd = open(lz4_path, O_RDONLY);
                    if (lz4_fd >= 0) {
                        // Load index header (512 entries × 16 bytes = 8KB)
                        g_lz4_index[i] = malloc(NUM_EXPERTS * sizeof(LZ4IndexEntry));
                        ssize_t nr = pread(lz4_fd, g_lz4_index[i],
                                           NUM_EXPERTS * sizeof(LZ4IndexEntry), 0);
                        if (nr == NUM_EXPERTS * (ssize_t)sizeof(LZ4IndexEntry)) {
                            // Replace the raw fd with the LZ4 fd
                            close(layer_fds[i]);
                            layer_fds[i] = lz4_fd;
                            fcntl(lz4_fd, F_RDAHEAD, 1);
                            lz4_layers++;
                        } else {
                            free(g_lz4_index[i]);
                            g_lz4_index[i] = NULL;
                            close(lz4_fd);
                        }
                    }
                }
                if (lz4_layers > 0) {
                    g_use_lz4 = 1;
                    // Allocate compressed read buffers (one per expert slot)
                    for (int k = 0; k < MAX_K; k++) {
                        g_lz4_comp_bufs[k] = malloc(EXPERT_SIZE_MAX + 4096);
                    }
                    printf("[lz4] %d/%d layers using LZ4 compressed experts\n",
                           lz4_layers, NUM_LAYERS);
                }
            }
        }

        // Wire up tiered I/O globals
        g_layer_fds_cold = layer_fds_cold;
        if (!g_use_lz4)
            printf("[tiered-io] Cold fds (F_NOCACHE) + warm fds (page cached) active\n");

        // Warm page cache hint
        if (expert_layers_available > 0) {
            double t_warm = now_ms();
            for (int i = 0; i < NUM_LAYERS; i++) {
                if (layer_fds[i] >= 0) {
                    char dummy[4096];
                    pread(layer_fds[i], dummy, sizeof(dummy), 0);
                }
            }
            printf("[warmup] Page cache hint: %.1f ms\n", now_ms() - t_warm);
        }

        // ---- Allocate per-layer state ----
        void **layer_states = calloc(NUM_LAYERS, sizeof(void *));
        KVCache **kv_caches = calloc(NUM_LAYERS, sizeof(KVCache *));

        for (int i = 0; i < NUM_LAYERS; i++) {
            int is_full = ((i + 1) % FULL_ATTN_INTERVAL == 0);
            if (is_full) {
                kv_caches[i] = kv_cache_new();
            } else {
                layer_states[i] = linear_attn_state_new();
            }
        }

        double t_init = now_ms();
        printf("[init] Setup: %.1f ms\n\n", t_init - t0);
        if (g_missing_tensor_count > 0) {
            printf("[init] %d missing scale/bias tensors (expected — using BF16 fallback)\n\n",
                   g_missing_tensor_count);
        }

        // ---- Allocate working buffers ----
        float *hidden = calloc(HIDDEN_DIM, sizeof(float));
        float *logits = calloc(VOCAB_SIZE, sizeof(float));
        uint16_t *final_norm_w = get_tensor_ptr(wf, "model.norm.weight");

        // ---- Serve mode: enter HTTP server loop (never returns) ----
        if (serve_port > 0) {
            double gpu_kv_gb = (double)g_gpu_kv_seq * NUM_KV_HEADS * HEAD_DIM * sizeof(float)
                               * NUM_FULL_ATTN_LAYERS * 2 / 1e9;
            printf("[config] Context: max_seq=%d (%.0fK), gpu_kv=%d tokens (%.1f GB GPU KV buffers)\n",
                   g_max_seq_len, g_max_seq_len / 1000.0,
                   g_gpu_kv_seq, gpu_kv_gb);
            reset_delta_net_state();
            serve_loop(serve_port, wf, vocab,
                       layer_states, kv_caches,
                       (void **)layer_mmaps, layer_fds,
                       hidden, logits, final_norm_w, K);
            // serve_loop never returns, but cleanup just in case
            free(hidden); free(logits);
            return 0;
        }

        // ---- Generate tokens ----
        reset_delta_net_state();  // zero GPU delta-net state before generation
        if (g_cache_telemetry_enabled) cache_telemetry_reset();
        printf("--- Generating %d tokens ---\n", max_tokens);
        int pos = 0;  // position counter for RoPE

        // ---- Batch prefill: pre-embed all prompt tokens ----
        // Embedding all tokens upfront into a batch buffer avoids interleaving
        // embed_lookup with GPU work, and enables the optimized prefill loop below.
        float *embed_batch = NULL;
        if (pt->count > 1) {
            embed_batch = malloc((size_t)pt->count * HIDDEN_DIM * sizeof(float));
            double t_embed = now_ms();
            for (int i = 0; i < pt->count; i++) {
                embed_lookup(wf, pt->ids[i], embed_batch + (size_t)i * HIDDEN_DIM);
            }
            double embed_ms = now_ms() - t_embed;
            printf("  [prefill] batch embed %d tokens: %.1f ms\n", pt->count, embed_ms);
        }

        // ---- Batch prefill loop ----
        // Process all prompt tokens through the model. For intermediate tokens
        // (not the last), we use discard_deferred_experts() which waits for the GPU
        // but skips the CPU readback/combine of the last layer's expert outputs.
        // This is safe because the hidden state from intermediate prefill tokens
        // is immediately overwritten by the next token's embedding — the recurrent
        // state (KV cache, delta-net state) is already updated inside fused_layer_forward.
        if (pt->count > 1) {
            double t_prefill_batch = now_ms();
            double first_tok_ms = 0;

            if (prefill_chunk_available()) {
                // ---- Chunked batched prefill (all prompt tokens, then the
                // driver CPU-combines the last token for lm_head) ----
                prefill_chunked_run(wf, hidden, embed_batch, pt->count, &pos,
                                    pt->ids,
                                    kv_caches, (void **)layer_states,
                                    layer_mmaps, K, layer_fds, g_prefill_chunk);
                first_tok_ms = 0;  // not measured per token in chunk mode
            } else {
                for (int token_idx = 0; token_idx < pt->count - 1; token_idx++) {
                    double t_tok = now_ms();

                    // Load pre-embedded token from batch buffer
                    cache_telemetry_note_token();
                    memcpy(hidden, embed_batch + (size_t)token_idx * HIDDEN_DIM,
                           HIDDEN_DIM * sizeof(float));

                    // Run through all 40 transformer layers
                    for (int layer = 0; layer < NUM_LAYERS; layer++) {
                        int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                        fused_layer_forward(wf, layer, hidden,
                                            is_full ? kv_caches[layer] : NULL,
                                            is_full ? NULL : layer_states[layer],
                                            pos,
                                            layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                            K, layer_fds[layer]);
                    }

                    // Discard last layer's expert output — hidden will be overwritten
                    // by the next token's embedding. Only wait for GPU (buffer safety).
                    discard_deferred_experts();
                    pos++;

                    if (g_use_mtp && embed_batch)
                        mtp_cache_fill(wf, pt->ids + token_idx,
                                       embed_batch + (size_t)token_idx * HIDDEN_DIM,
                                       hidden, 1, pos - 1);

                    if (token_idx == 0) {
                        first_tok_ms = now_ms() - t_tok;
                    }
                }
            }

            double prefill_batch_ms = now_ms() - t_prefill_batch;
            double avg_ms = (pt->count > 2) ?
                (prefill_batch_ms - first_tok_ms) / (pt->count - 2) : first_tok_ms;
            printf("  [prefill] %d/%d tokens: %.0f ms (first: %.0f ms, rest avg: %.0f ms)%s\n",
                   pt->count - 1, pt->count, prefill_batch_ms, first_tok_ms, avg_ms,
                   (prefill_chunk_available() && pt->count > 1) ? " [chunked]" : "");
        }

        // ---- Last prefill token (or single-token prompt) ----
        // This one needs full completion since we need hidden state for logits.
        // Skipped entirely in chunked mode — prefill_chunked_run completes the
        // final position's hidden state itself and advances pos. Running this
        // block in chunked mode would clobber hidden with the raw last-token
        // embedding (leaving the lm_head to sample from garbage) and
        // double-increment pos (corrupting RoPE/KV positions for generation).
        if (!(prefill_chunk_available() && pt->count > 1)) {
            cache_telemetry_note_token();
            if (embed_batch) {
                memcpy(hidden, embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                       HIDDEN_DIM * sizeof(float));
            } else {
                embed_lookup(wf, pt->ids[0], hidden);
            }

            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                double t_lay = now_ms();
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
                if (getenv("FINCHMOE_LAYERTIME")) {
                    static double acc_l = 0; static int n_l = 0;
                    acc_l += now_ms() - t_lay; n_l++;
                    if (n_l % 40 == 0) fprintf(stderr, "[layertime] avg %.3f ms/layer (n=%d)\n", acc_l / n_l, n_l);
                }
                { float hr=0; for(int j=0;j<HIDDEN_DIM;j++) hr+=hidden[j]*hidden[j];
                  hr=sqrtf(hr/HIDDEN_DIM);
                  if(!isfinite(hr)) fprintf(stderr,"[LOOP] layer %d: hidden rms=nan!\n",layer); }
            }
            // Full completion — need hidden state for final norm + lm_head
            complete_deferred_experts();
            pos++;
            if (g_use_mtp && embed_batch)
                mtp_cache_fill(wf, pt->ids + (pt->count - 1),
                               embed_batch + (size_t)(pt->count - 1) * HIDDEN_DIM,
                               hidden, 1, pos - 1);
        }

        if (embed_batch) { free(embed_batch); embed_batch = NULL; }

        // ---- Final norm ----
        { float hr=0; for(int j=0;j<HIDDEN_DIM;j++) hr+=hidden[j]*hidden[j];
          fprintf(stderr,"[PRE-NORM] hidden rms=%.6f isfinite=%d\n", sqrtf(hr/HIDDEN_DIM), isfinite(sqrtf(hr/HIDDEN_DIM))); }
        if (final_norm_w) {
            float *normed = malloc(HIDDEN_DIM * sizeof(float));
            cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
            memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
            free(normed);
        }

        // ---- LM head (GPU gemv_bf16_x2, zero-copy) ----
        double t_lm = now_ms();
        lm_head_forward(wf, hidden, logits);
        double lm_ms = now_ms() - t_lm;

        // ---- Dump logits for cross-validation (first token truncates, later steps append) ----
        if (g_dump_logits_path) {
            FILE *df = fopen(g_dump_logits_path, "wb");
            if (df) {
                fwrite(logits, sizeof(float), VOCAB_SIZE, df);
                fclose(df);
                fprintf(stderr, "[dump-logits] step 0: %d floats (%.1f MB) -> %s\n",
                        VOCAB_SIZE, (double)(VOCAB_SIZE * sizeof(float)) / 1e6, g_dump_logits_path);
            }
        }

        // Sampler isolation for the (single) CLI generation — mirrors the
        // serve-path reset; makes T=0 deterministic regardless of any prior
        // sampler use in this process.
        sampler_state_reset();

        // ---- Sample first token ----
        int next_token = cpu_sample_temp(logits, VOCAB_SIZE, g_temperature, g_top_k);
        logit_diag_dump(logits, VOCAB_SIZE, next_token, 0);
        double ttft_ms = now_ms() - t0;

        // Debug: show top-5 logits for first token
        {
            // Find top 5 manually
            int top5[5] = {0,0,0,0,0};
            float topv[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
            for (int i = 0; i < VOCAB_SIZE; i++) {
                int min_k = 0;
                for (int k = 1; k < 5; k++) if (topv[k] < topv[min_k]) min_k = k;
                if (logits[i] > topv[min_k]) { topv[min_k] = logits[i]; top5[min_k] = i; }
            }
            fprintf(stderr, "[debug] Top 5 logits (next_token=%d):\n", next_token);
            for (int i = 0; i < 5; i++) {
                fprintf(stderr, "  token %d (\"%s\") logit=%.4f\n",
                        top5[i], decode_token(vocab, top5[i]), topv[i]);
            }
            fprintf(stderr, "[debug] hidden rms after final_norm=%.4f, logits rms=%.4f\n",
                    vec_rms(hidden, HIDDEN_DIM), vec_rms(logits, VOCAB_SIZE));
        }
        printf("[ttft] %.0f ms (prefill %d tokens + lm_head %.0f ms)\n",
               ttft_ms, pt->count, lm_ms);

        printf("\n--- Output ---\n");
        printf("%s", decode_token(vocab, next_token));
        fflush(stdout);

        int total_generated = 1;
        int in_think = (next_token == THINK_START_TOKEN) ? 1 : 0;
        int think_ended = 0;   // once the think block closes, think tokens are banned
        int think_tokens = 0;

        // ---- Auto-regressive generation ----
        if (g_timing_enabled) timing_reset();
        if (g_pred_enabled || g_gguf_stage) {   // GGUF: S7 Lever 1 temporal prediction
            g_pred_generating = 1;  // enable prediction storage/use during generation
            g_pred_valid = 0;       // reset — first gen token builds predictions
        }
        for (int gen = 1; gen < max_tokens; gen++) {
            double t_gen_start = now_ms();

            // Check EOS
            if (next_token == EOS_TOKEN_1 || next_token == EOS_TOKEN_2) {
                fprintf(stderr, "\n[eos] Token %d at position %d\n", next_token, gen);
                break;
            }

            // Think budget enforcement + one-shot re-entry ban
            if (next_token == THINK_START_TOKEN) in_think = 1;
            if (next_token == THINK_END_TOKEN) { in_think = 0; think_ended = 1; }
            if (in_think) think_tokens++;

            // Embed the just-generated token (next iteration)
            cache_telemetry_note_token();
            embed_lookup(wf, next_token, hidden);

            // Run 60 layers (fused: 1+K cmd buffers per layer)
            for (int layer = 0; layer < NUM_LAYERS; layer++) {
                int is_full = ((layer + 1) % FULL_ATTN_INTERVAL == 0);
                fused_layer_forward(wf, layer, hidden,
                                    is_full ? kv_caches[layer] : NULL,
                                    is_full ? NULL : layer_states[layer],
                                    pos,
                                    layer_mmaps[layer] != MAP_FAILED ? layer_mmaps[layer] : NULL,
                                    K, layer_fds[layer]);
            }
            // Complete last layer's deferred GPU experts before final norm
            complete_deferred_experts();
            pos++;

            // Final norm — save the pre-norm hidden for the MTP head
            // (the reference MTP consumes the post-MoE residual, not the
            // final-normed hidden).
            static float mtp_hidden_in[HIDDEN_DIM];
            memcpy(mtp_hidden_in, hidden, HIDDEN_DIM * sizeof(float));
            if (final_norm_w) {
                float *normed = malloc(HIDDEN_DIM * sizeof(float));
                cpu_rms_norm(hidden, final_norm_w, normed, HIDDEN_DIM, RMS_NORM_EPS);
                memcpy(hidden, normed, HIDDEN_DIM * sizeof(float));
                free(normed);
            }

            // LM head
            lm_head_forward(wf, hidden, logits);

            // Once the think block has closed, ban re-entering it (the model
            // otherwise loops "<think>…</think>" fragments mid-answer).
            if (think_ended) {
                logits[THINK_START_TOKEN] = -INFINITY;
                logits[THINK_END_TOKEN] = -INFINITY;
            }

            // Per-step logit dump for drift cross-validation
            if (g_dump_logits_path) {
                FILE *df = fopen(g_dump_logits_path, "ab");
                if (df) {
                    fwrite(logits, sizeof(float), VOCAB_SIZE, df);
                    fclose(df);
                }
            }

            // Temperature sample
            next_token = cpu_sample_temp(logits, VOCAB_SIZE, g_temperature, g_top_k);
            logit_diag_dump(logits, VOCAB_SIZE, next_token, total_generated);

            // Think budget: force end thinking if over budget
            if (in_think && g_think_budget > 0 && think_tokens >= g_think_budget) {
                next_token = THINK_END_TOKEN;
                in_think = 0;
            }
            total_generated++;

            // Print decoded token
            printf("%s", decode_token(vocab, next_token));
            fflush(stdout);

            double t_gen_end = now_ms();
            double tok_time = t_gen_end - t_gen_start;

            // Print progress to stderr
            fprintf(stderr, "  [gen %d/%d] token_id=%d (%.0f ms, %.2f tok/s)",
                    gen, max_tokens, next_token, tok_time, 1000.0 / tok_time);

            // MTP speculative draft (persistent buffer, allocated once)
            static int mtp_attempts = 0, mtp_accepted = 0;
            static float *mtp_logits = NULL;
            if (!mtp_logits) {
                mtp_logits = malloc(VOCAB_SIZE * sizeof(float));
                fprintf(stderr, "[MTP] logits buffer alloc: %p\n", (void*)mtp_logits);
            }
            // MTP reference dump (FINCHMOE_MTP_DUMP): record (token, pre-norm
            // hidden, main logits) per generation step for the Python
            // reference comparison.
            if (g_use_mtp && getenv("FINCHMOE_MTP_DUMP")) {
                static FILE *md = NULL;
                if (!md) md = fopen("/tmp/mtp_ref_input.bin", "ab");
                if (md) {
                    int32_t pos_i = pos, tok = next_token;
                    fwrite(&pos_i, sizeof(int32_t), 1, md);
                    fwrite(&tok, sizeof(int32_t), 1, md);
                    fwrite(mtp_hidden_in, sizeof(float), HIDDEN_DIM, md);
                    fwrite(logits, sizeof(float), VOCAB_SIZE, md);
                    fflush(md);
                }
            }
            if (g_use_mtp && g_mtp.loaded && mtp_logits && total_generated < max_tokens - 1) {
                int mtp_token;
                float saved_hidden[HIDDEN_DIM];
                memcpy(saved_hidden, hidden, HIDDEN_DIM * sizeof(float));

                if (mtp_forward(wf, mtp_hidden_in, next_token, &mtp_token, mtp_logits)) {
                    mtp_attempts++;
                    int draft_match = (mtp_token == next_token);
                    if (draft_match) mtp_accepted++;
                    if (mtp_attempts % 10 == 0 || mtp_attempts < 3) {
                        double md = 0, mn1 = 0, mn2 = 0;
                        for (int i = 0; i < VOCAB_SIZE; i++) {
                            md += (double)logits[i] * mtp_logits[i];
                            mn1 += (double)logits[i] * logits[i];
                            mn2 += (double)mtp_logits[i] * mtp_logits[i];
                        }
                        fprintf(stderr, "  [mtp-cos] main_vs_draft_logit_cos=%.4f (main_rms=%.2f draft_rms=%.2f)\n",
                                md / sqrt(mn1 * mn2),
                                vec_rms(logits, VOCAB_SIZE), vec_rms(mtp_logits, VOCAB_SIZE));
                    }
                    if (mtp_attempts % 10 == 0 || draft_match || mtp_attempts < 5) {
                        fprintf(stderr, "  mtp=%d draft=%d %s (rate=%d/%d=%.0f%%)\n",
                                next_token, mtp_token,
                                draft_match ? "ACCEPT" : "reject",
                                mtp_accepted, mtp_attempts,
                                100.0 * mtp_accepted / mtp_attempts);
                    }
                    memcpy(hidden, saved_hidden, HIDDEN_DIM * sizeof(float));
                }
            }
            fprintf(stderr, "\n");
        }

        if (g_timing_enabled) timing_print();
        printf("\n\n--- Statistics ---\n");
        double total_time = now_ms() - t0;
        printf("Total time:     %.1f s\n", total_time / 1000.0);
        printf("TTFT:           %.0f ms\n", ttft_ms);
        printf("Tokens:         %d generated\n", total_generated);
        if (total_generated > 1) {
            double gen_time = total_time - ttft_ms;
            printf("Generation:     %.1f s (%.2f tok/s)\n",
                   gen_time / 1000.0, (total_generated - 1) * 1000.0 / gen_time);
        }
        printf("Config:         K=%d experts, %d layers\n", K, NUM_LAYERS);
        if (g_expert_cache) {
            uint64_t total = g_expert_cache->hits + g_expert_cache->misses;
            printf("Expert cache:   %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_expert_cache->hits, g_expert_cache->misses,
                   total > 0 ? 100.0 * g_expert_cache->hits / total : 0.0,
                   g_expert_cache->num_entries, g_expert_cache->max_entries);
            cache_telemetry_print(g_expert_cache->hits, g_expert_cache->misses);
        } else if (g_malloc_cache) {
            uint64_t total = g_malloc_cache->hits + g_malloc_cache->misses;
            printf("Expert cache:   malloc %llu hits, %llu misses (%.1f%% hit rate), %d/%d entries used\n",
                   g_malloc_cache->hits, g_malloc_cache->misses,
                   total > 0 ? 100.0 * g_malloc_cache->hits / total : 0.0,
                   g_malloc_cache->num_entries, g_malloc_cache->max_entries);
            cache_telemetry_print(g_malloc_cache->hits, g_malloc_cache->misses);
        }

        if (g_spec_route_attempts > 0) {
            printf("Spec routing:   %llu attempts, %llu preloads, %llu hits (%.1f%% prediction accuracy)\n",
                   g_spec_route_attempts, g_spec_route_preloads, g_spec_route_hits,
                   g_spec_route_attempts > 0
                       ? 100.0 * g_spec_route_hits / g_spec_route_attempts : 0.0);
            if (getenv("FINCHMOE_SPEC_PROBE")) {
                printf("Spec temporal:  %llu hits / %llu attempts (%.1f%% prev-position overlap)\n",
                       g_spec_temporal_hits, g_spec_route_attempts,
                       g_spec_route_attempts > 0
                           ? 100.0 * g_spec_temporal_hits / g_spec_route_attempts : 0.0);
            }
        }

        if (g_freq_tracking) freq_print_analysis(K);
        if (g_routing_log) {
            fclose(g_routing_log);
            fprintf(stderr, "[routing] Logged %d samples to routing data file\n",
                    g_routing_log_samples);
            g_routing_log = NULL;
        }

        // ---- Cleanup ----
        io_pool_shutdown();
        if (g_malloc_cache) {
            malloc_cache_free(g_malloc_cache);
            g_malloc_cache = NULL;
        }
        if (g_expert_cache) {
            expert_cache_free(g_expert_cache);
            g_expert_cache = NULL;
        }
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (kv_caches[i]) kv_cache_free(kv_caches[i]);
            if (layer_states[i]) linear_attn_state_free(layer_states[i]);
            if (layer_mmaps[i] != MAP_FAILED) munmap(layer_mmaps[i], layer_mmap_sizes[i]);
            if (layer_fds[i] >= 0) close(layer_fds[i]);
            if (layer_fds_cold[i] >= 0) close(layer_fds_cold[i]);
        }
        free(layer_states);
        free(kv_caches);
        free(hidden);
        free(logits);

        return 0;
    }
}
#endif // CHAT_MODE

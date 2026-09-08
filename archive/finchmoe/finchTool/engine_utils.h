/*
 * engine_utils.h — Shared Metal engine primitives for finchTool diagnostics.
 *
 * Extracted from infer.m to provide a minimal, self-contained Metal setup
 * for kernel isolation tests, pipeline auditing, and layer diagnostics.
 */

#ifndef ENGINE_UTILS_H
#define ENGINE_UTILS_H

#include <Metal/Metal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ---- Model dimensions (match infer.m) ----
#define HIDDEN_DIM          2048
#define NUM_LAYERS          40
#define NUM_EXPERTS         256
#define NUM_EXPERTS_PER_TOK 8
#define MOE_INTERMEDIATE    512
#define GROUP_SIZE          64
#define SHARED_INTERMEDIATE 2048  // shared expert intermediate size

// ---- Expert size macros (match infer.m lines 160-219) ----
#define EXPERT_SIZE_4BIT   1769472
#define EXPERT_SIZE_8BIT   3342336
#define EXPERT_SIZE_1BIT    589824
#define EXPERT_SIZE_2BIT    983040
#define EXPERT_SIZE_MAX    3932160

// 4-bit expert buffer offsets (match infer.m lines 160-169)
#define GATE_W_OFF_4  0
#define GATE_S_OFF_4  524288
#define GATE_B_OFF_4  557056
#define UP_W_OFF_4    589824
#define UP_S_OFF_4    1114112
#define UP_B_OFF_4    1146880
#define DOWN_W_OFF_4  1179648
#define DOWN_S_OFF_4  1703936
#define DOWN_B_OFF_4  1736704

// 8-bit expert buffer offsets (match infer.m lines 172-183)
#define GATE_W_OFF_8  0
#define GATE_S_OFF_8  1048576
#define GATE_B_OFF_8  1114112
#define UP_W_OFF_8    1179648
#define UP_S_OFF_8    2228224
#define UP_B_OFF_8    2293760
#define DOWN_W_OFF_8  2359296
#define DOWN_S_OFF_8  3276800
#define DOWN_B_OFF_8  3342336

// 1-bit expert buffer offsets (match infer.m lines 184-196)
#define GATE_W_OFF_1  0
#define GATE_S_OFF_1  65536
#define GATE_B_OFF_1  69632
#define UP_W_OFF_1    73728
#define UP_S_OFF_1    139264
#define UP_B_OFF_1    143360
#define DOWN_W_OFF_1  147456
#define DOWN_S_OFF_1  491520
#define DOWN_B_OFF_1  532480

// 2-bit expert buffer offsets (match infer.m lines 197-208)
#define GATE_W_OFF_2  0
#define GATE_S_OFF_2  131072
#define GATE_B_OFF_2  139264
#define UP_W_OFF_2    147456
#define UP_S_OFF_2    278528
#define UP_B_OFF_2    286720
#define DOWN_W_OFF_2  294912
#define DOWN_S_OFF_2  819200
#define DOWN_B_OFF_2  860160

// ---- Minimal Metal context for diagnostics ----
typedef struct {
    id<MTLDevice>        device;
    id<MTLCommandQueue>  queue;
    id<MTLLibrary>       library;

    // Core compute pipelines
    id<MTLComputePipelineState> matvec_v3;          // 4-bit dequant matvec
    id<MTLComputePipelineState> matvec_8bit;        // 8-bit dequant matvec
    id<MTLComputePipelineState> matvec_2bit;        // 2-bit dequant matvec
    id<MTLComputePipelineState> matvec_1bit;        // 1-bit dequant matvec
    id<MTLComputePipelineState> fused_gate_up_swiglu;      // 4-bit fused gate+up+SiLU
    id<MTLComputePipelineState> fused_gate_up_swiglu_8bit; // 8-bit fused variant
    id<MTLComputePipelineState> fused_gate_up_swiglu_2x;   // 2x expert fused
    id<MTLComputePipelineState> swiglu;             // SwiGLU activation
    id<MTLComputePipelineState> moe_combine_residual; // MoE combine + residual
    id<MTLComputePipelineState> gemv_bf16;           // BF16 GEMV
    id<MTLComputePipelineState> gemv_bf16_x2;        // BF16 GEMV x2

    // Diagnostic buffers (allocated per-test for isolation)
    id<MTLBuffer> buf_input;        // [HIDDEN_DIM] float input
    id<MTLBuffer> buf_expert_data;  // [EXPERT_SIZE_MAX] expert weights
    id<MTLBuffer> buf_gate;         // [MOE_INTERMEDIATE] float
    id<MTLBuffer> buf_up;           // [MOE_INTERMEDIATE] float
    id<MTLBuffer> buf_act;          // [MOE_INTERMEDIATE] float
    id<MTLBuffer> buf_out;          // [HIDDEN_DIM] float

    // Pipeline sync
    id<MTLFence>        sync_fence;
    id<MTLSharedEvent>  sync_event;
    uint64_t            sync_value;
} DiagMetalCtx;

// ---- CPU reference functions (extracted from infer.m) ----

// BF16 to float32 conversion (infer.m line 509)
static inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

// ---- Function declarations ----

// Initialize Metal device, queue, shader library, and pipelines.
// Returns NULL on failure (prints error to stderr).
DiagMetalCtx *diag_metal_init(void);

// Free all Metal resources.
void diag_metal_free(DiagMetalCtx *ctx);

// Allocate fresh diagnostic buffers (input, expert_data, gate, up, act, out).
// Previous buffers are freed first. All use MTLResourceStorageModeShared.
void diag_metal_alloc_buffers(DiagMetalCtx *ctx);

// Load expert weight data for a specific expert from a layer file.
// Returns malloc'd buffer of active_expert_size() bytes, or NULL on failure.
// Caller must free().
void *diag_load_expert_data(const char *layer_file_path, int expert_idx, int bits);

// CPU 4-bit dequant matvec: out = W * x  (infer.m line 967)
// bits: 0=BF16 (scales==NULL), 1=1-bit, 2=2-bit, 4=4-bit, 8=8-bit
void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size, int bits);

// CPU SwiGLU: out[i] = SiLU(gate[i]) * up[i]  (infer.m line 1032)
void cpu_swiglu(const float *gate, const float *up, float *out, int dim);

// RMS of a float vector (infer.m line 2875)
float vec_rms(const float *v, int n);

// Print first N values of a float array (for debugging)
void print_first_n(const char *label, const float *v, int n);

// Get the active expert size for a given bit width
static inline size_t active_expert_size(int bits) {
    switch (bits) {
        case 1:  return EXPERT_SIZE_1BIT;
        case 2:  return EXPERT_SIZE_2BIT;
        case 8:  return EXPERT_SIZE_8BIT;
        case 4:
        default: return EXPERT_SIZE_4BIT;
    }
}

#endif // ENGINE_UTILS_H

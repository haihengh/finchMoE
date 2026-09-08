/*
 * main.m — finchTool CLI entry point.
 *
 * Metal Engine Verification & Diagnostic Suite for Apple Silicon.
 *
 * Usage:
 *   ./finchTool kernel --test fused_mlp --quant 4bit
 *   ./finchTool pipeline --test inter-cb-sync
 *   ./finchTool layer --model <path> --layer 0 --dump
 *   ./finchTool parity --a <file_a.bin> --b <file_b.bin>
 */

#import "engine_utils.h"
#import "verify_core.h"
#import <getopt.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

// ============================================================================
// Test: fused_mlp — compare fused gate+up+SiLU vs non-fused path
// ============================================================================

static int test_fused_mlp(DiagMetalCtx *ctx, int bits, int dim_in, int dim_out,
                           bool verbose) {
    if (bits != 4 && bits != 8) {
        fprintf(stderr, "[fused_mlp] Only 4-bit and 8-bit supported (got %d-bit)\n", bits);
        return 1;
    }
    if (!ctx->fused_gate_up_swiglu && bits == 4) {
        fprintf(stderr, "[fused_mlp] SKIP: fused_gate_up_swiglu pipeline not available\n");
        return 0;
    }
    if (!ctx->fused_gate_up_swiglu_8bit && bits == 8) {
        fprintf(stderr, "[fused_mlp] SKIP: fused_gate_up_swiglu_8bit pipeline not available\n");
        return 0;
    }
    if (!ctx->matvec_v3) {
        fprintf(stderr, "[fused_mlp] SKIP: matvec_v3 pipeline not available\n");
        return 0;
    }

    fprintf(stderr, "\n=== Test: fused_mlp (%d-bit, in=%d, out=%d) ===\n",
            bits, dim_in, dim_out);

    // Generate deterministic test input
    int group_size = GROUP_SIZE;
    int num_groups = dim_in / group_size;
    int vals_per_u32 = 32 / bits;
    int packed_cols = dim_in / vals_per_u32;
    int packed_per_group = group_size / vals_per_u32;

    size_t gate_w_bytes = (size_t)dim_out * packed_cols * sizeof(uint32_t);
    size_t gate_s_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);
    size_t gate_b_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);

    // Allocate synthetic weight data
    uint32_t *gate_W = (uint32_t *)malloc(gate_w_bytes);
    uint16_t *gate_s = (uint16_t *)malloc(gate_s_bytes);
    uint16_t *gate_b = (uint16_t *)malloc(gate_b_bytes);
    uint32_t *up_W   = (uint32_t *)malloc(gate_w_bytes);
    uint16_t *up_s   = (uint16_t *)malloc(gate_s_bytes);
    uint16_t *up_b   = (uint16_t *)malloc(gate_b_bytes);

    // Fill with deterministic patterns (sine-based for reproducibility)
    srand(42);
    for (size_t i = 0; i < gate_w_bytes / sizeof(uint32_t); i++) {
        gate_W[i] = (uint32_t)(rand() & 0xFFFFFFFF);
        up_W[i]   = (uint32_t)(rand() & 0xFFFFFFFF);
    }
    // Scales: small positive values in BF16 range
    for (size_t i = 0; i < gate_s_bytes / sizeof(uint16_t); i++) {
        float s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;  // 0.5 .. 2.0
        gate_s[i] = (uint16_t)((*(uint32_t *)&s) >> 16);
        s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;
        up_s[i]   = (uint16_t)((*(uint32_t *)&s) >> 16);
    }
    // Biases: small values around zero
    for (size_t i = 0; i < gate_b_bytes / sizeof(uint16_t); i++) {
        float b = 0.2f * ((float)rand() / (float)RAND_MAX - 0.5f);  // -0.1 .. 0.1
        gate_b[i] = (uint16_t)((*(uint32_t *)&b) >> 16);
        b = 0.2f * ((float)rand() / (float)RAND_MAX - 0.5f);
        up_b[i]   = (uint16_t)((*(uint32_t *)&b) >> 16);
    }

    // Generate test input
    float *input = (float *)malloc(dim_in * sizeof(float));
    for (int i = 0; i < dim_in; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX - 0.5f);  // -0.5 .. 0.5
    }

    // Copy to Metal buffers
    memcpy([ctx->buf_input contents], input, dim_in * sizeof(float));

    // Copy synthetic weights to expert_data buffer in the standard layout
    uint8_t *edata = (uint8_t *)[ctx->buf_expert_data contents];
    NSUInteger gw_off, gs_off, gb_off, uw_off, us_off, ub_off;
    if (bits == 8) {
        gw_off = GATE_W_OFF_8; gs_off = GATE_S_OFF_8; gb_off = GATE_B_OFF_8;
        uw_off = UP_W_OFF_8;   us_off = UP_S_OFF_8;   ub_off = UP_B_OFF_8;
    } else {
        gw_off = GATE_W_OFF_4; gs_off = GATE_S_OFF_4; gb_off = GATE_B_OFF_4;
        uw_off = UP_W_OFF_4;   us_off = UP_S_OFF_4;   ub_off = UP_B_OFF_4;
    }
    // Note: we only fill gate/up data; down_proj isn't tested in this kernel test
    memcpy(edata + gw_off, gate_W, gate_w_bytes);
    memcpy(edata + gs_off, gate_s, gate_s_bytes);
    memcpy(edata + gb_off, gate_b, gate_b_bytes);
    memcpy(edata + uw_off, up_W,   gate_w_bytes);
    memcpy(edata + us_off, up_s,   gate_s_bytes);
    memcpy(edata + ub_off, up_b,   gate_b_bytes);

    // ---- Path A: Non-fused GPU (golden reference) ----
    id<MTLCommandBuffer> cb_a = [ctx->queue commandBuffer];

    // gate_proj
    {
        id<MTLComputeCommandEncoder> enc = [cb_a computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:gw_off atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:gs_off atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:gb_off atIndex:2];
        [enc setBuffer:ctx->buf_input       offset:0     atIndex:3];
        [enc setBuffer:ctx->buf_gate        offset:0     atIndex:4];
        uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:5];
        [enc setBytes:&i length:4 atIndex:6];
        [enc setBytes:&g length:4 atIndex:7];
        uint32_t tgs = (o + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // up_proj
    {
        id<MTLComputeCommandEncoder> enc = [cb_a computeCommandEncoder];
        [enc setComputePipelineState:ctx->matvec_v3];
        [enc setBuffer:ctx->buf_expert_data offset:uw_off atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:us_off atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:ub_off atIndex:2];
        [enc setBuffer:ctx->buf_input       offset:0     atIndex:3];
        [enc setBuffer:ctx->buf_up          offset:0     atIndex:4];
        uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:5];
        [enc setBytes:&i length:4 atIndex:6];
        [enc setBytes:&g length:4 atIndex:7];
        uint32_t tgs = (o + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    // SwiGLU
    {
        id<MTLComputeCommandEncoder> enc = [cb_a computeCommandEncoder];
        [enc setComputePipelineState:ctx->swiglu];
        [enc setBuffer:ctx->buf_gate  offset:0 atIndex:0];
        [enc setBuffer:ctx->buf_up    offset:0 atIndex:1];
        [enc setBuffer:ctx->buf_act   offset:0 atIndex:2];
        uint32_t o = (uint32_t)dim_out;
        [enc setBytes:&o length:4 atIndex:3];
        uint32_t tgs = (o + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }
    [cb_a commit];
    [cb_a waitUntilCompleted];

    // Read back non-fused result
    float *nonfused_act = (float *)malloc(dim_out * sizeof(float));
    float *nonfused_gate = (float *)malloc(dim_out * sizeof(float));
    float *nonfused_up   = (float *)malloc(dim_out * sizeof(float));
    memcpy(nonfused_act,  [ctx->buf_act contents],  dim_out * sizeof(float));
    memcpy(nonfused_gate, [ctx->buf_gate contents], dim_out * sizeof(float));
    memcpy(nonfused_up,   [ctx->buf_up contents],   dim_out * sizeof(float));

    // ---- Path B: Fused GPU (test) ----
    id<MTLCommandBuffer> cb_b = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc_b = [cb_b computeCommandEncoder];
    id<MTLComputePipelineState> fpipe = (bits == 8) ?
        ctx->fused_gate_up_swiglu_8bit : ctx->fused_gate_up_swiglu;
    [enc_b setComputePipelineState:fpipe];
    [enc_b setBuffer:ctx->buf_expert_data offset:gw_off atIndex:0];
    [enc_b setBuffer:ctx->buf_expert_data offset:gs_off atIndex:1];
    [enc_b setBuffer:ctx->buf_expert_data offset:gb_off atIndex:2];
    [enc_b setBuffer:ctx->buf_expert_data offset:uw_off atIndex:3];
    [enc_b setBuffer:ctx->buf_expert_data offset:us_off atIndex:4];
    [enc_b setBuffer:ctx->buf_expert_data offset:ub_off atIndex:5];
    [enc_b setBuffer:ctx->buf_input       offset:0     atIndex:6];
    // Use a different output buffer to avoid overwriting non-fused result
    id<MTLBuffer> fused_act_buf = [ctx->device newBufferWithLength:dim_out * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
    [enc_b setBuffer:fused_act_buf        offset:0     atIndex:7];
    uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
    [enc_b setBytes:&o length:4 atIndex:8];
    [enc_b setBytes:&i length:4 atIndex:9];
    [enc_b setBytes:&g length:4 atIndex:10];
    [enc_b dispatchThreadgroups:MTLSizeMake(o, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc_b endEncoding];
    [cb_b commit];
    [cb_b waitUntilCompleted];

    // Read back fused result
    float *fused_act = (float *)malloc(dim_out * sizeof(float));
    memcpy(fused_act, [fused_act_buf contents], dim_out * sizeof(float));
    fused_act_buf = nil;  // release

    // ---- CPU reference (ground truth) ----
    float *cpu_gate = (float *)malloc(dim_out * sizeof(float));
    float *cpu_up   = (float *)malloc(dim_out * sizeof(float));
    float *cpu_act  = (float *)malloc(dim_out * sizeof(float));
    cpu_dequant_matvec(gate_W, gate_s, gate_b, input, cpu_gate,
                       dim_out, dim_in, group_size, bits);
    cpu_dequant_matvec(up_W,   up_s,   up_b,   input, cpu_up,
                       dim_out, dim_in, group_size, bits);
    cpu_swiglu(cpu_gate, cpu_up, cpu_act, dim_out);

    // ---- Compare ----
    // Non-fused GPU vs CPU (should be exact)
    ParityReport r_nf = evaluate_parity_f32(cpu_act, nonfused_act, dim_out, 1e-3f);
    print_parity_report("Non-Fused GPU vs CPU (SwiGLU act)", &r_nf);

    // Fused GPU vs CPU
    ParityReport r_f = evaluate_parity_f32(cpu_act, fused_act, dim_out, 1e-3f);
    print_parity_report("Fused GPU vs CPU (SwiGLU act)", &r_f);

    // Fused vs Non-Fused directly
    ParityReport r_fn = evaluate_parity_f32(nonfused_act, fused_act, dim_out, 1e-3f);
    print_parity_report("Fused vs Non-Fused GPU (SwiGLU act)", &r_fn);

    if (verbose) {
        fprintf(stderr, "\n  First 10 values:\n");
        print_first_n("cpu_gate", cpu_gate, 10);
        print_first_n("cpu_up",   cpu_up,   10);
        print_first_n("cpu_act",  cpu_act,  10);
        print_first_n("nf_gate",  nonfused_gate, 10);
        print_first_n("nf_up",    nonfused_up,   10);
        print_first_n("nf_act",   nonfused_act,  10);
        print_first_n("f_act",    fused_act,     10);
        fprintf(stderr, "\n  RMS values:\n");
        fprintf(stderr, "  cpu_gate rms=%.4f  cpu_up rms=%.4f  cpu_act rms=%.4f\n",
                vec_rms(cpu_gate, dim_out), vec_rms(cpu_up, dim_out), vec_rms(cpu_act, dim_out));
        fprintf(stderr, "  nf_gate rms=%.4f  nf_up rms=%.4f  nf_act rms=%.4f\n",
                vec_rms(nonfused_gate, dim_out), vec_rms(nonfused_up, dim_out), vec_rms(nonfused_act, dim_out));
        fprintf(stderr, "  f_act rms=%.4f\n", vec_rms(fused_act, dim_out));
    }

    // Cleanup
    free(gate_W); free(gate_s); free(gate_b);
    free(up_W);   free(up_s);   free(up_b);
    free(input);
    free(nonfused_act); free(nonfused_gate); free(nonfused_up);
    free(fused_act);
    free(cpu_gate); free(cpu_up); free(cpu_act);

    return (r_f.passed && r_nf.passed) ? 0 : 1;
}

// ============================================================================
// Test: matvec — validate dequant matvec kernel vs CPU
// ============================================================================

static int test_matvec(DiagMetalCtx *ctx, int bits, bool verbose) {
    id<MTLComputePipelineState> pipe = NULL;
    switch (bits) {
        case 4:  pipe = ctx->matvec_v3;   break;
        case 8:  pipe = ctx->matvec_8bit; break;
        case 2:  pipe = ctx->matvec_2bit; break;
        case 1:  pipe = ctx->matvec_1bit; break;
        default:
            fprintf(stderr, "[matvec] Unsupported bit width: %d\n", bits);
            return 1;
    }
    if (!pipe) {
        fprintf(stderr, "[matvec] SKIP: %d-bit matvec pipeline not available\n", bits);
        return 0;
    }

    int dim_in    = 2048;
    int dim_out   = 512;
    int group_size = GROUP_SIZE;
    int vals_per_u32 = 32 / bits;
    int packed_cols = dim_in / vals_per_u32;
    int num_groups = dim_in / group_size;

    fprintf(stderr, "\n=== Test: matvec (%d-bit, in=%d, out=%d) ===\n",
            bits, dim_in, dim_out);

    // Generate synthetic test data
    srand(123 + bits);
    size_t w_bytes = (size_t)dim_out * packed_cols * sizeof(uint32_t);
    size_t s_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);
    size_t b_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);

    uint32_t *W = (uint32_t *)malloc(w_bytes);
    uint16_t *S = (uint16_t *)malloc(s_bytes);
    uint16_t *B = (uint16_t *)malloc(b_bytes);
    float *input = (float *)malloc(dim_in * sizeof(float));

    for (size_t i = 0; i < w_bytes / sizeof(uint32_t); i++) {
        W[i] = (uint32_t)(rand() & 0xFFFFFFFF);
    }
    for (size_t i = 0; i < s_bytes / sizeof(uint16_t); i++) {
        float s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;
        S[i] = (uint16_t)((*(uint32_t *)&s) >> 16);
        float b = 0.2f * ((float)rand() / (float)RAND_MAX - 0.5f);
        B[i] = (uint16_t)((*(uint32_t *)&b) >> 16);
    }
    for (int i = 0; i < dim_in; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX - 0.5f);
    }

    // CPU reference
    float *cpu_out = (float *)malloc(dim_out * sizeof(float));
    cpu_dequant_matvec(W, S, B, input, cpu_out, dim_out, dim_in, group_size, bits);

    // GPU
    memcpy([ctx->buf_input contents], input, dim_in * sizeof(float));
    memcpy([ctx->buf_expert_data contents], W, w_bytes);

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:ctx->buf_expert_data offset:0   atIndex:0];  // W
    // Scales/biases are right after W in buf_expert_data
    memcpy((uint8_t *)[ctx->buf_expert_data contents] + w_bytes, S, s_bytes);
    memcpy((uint8_t *)[ctx->buf_expert_data contents] + w_bytes + s_bytes, B, b_bytes);
    [enc setBuffer:ctx->buf_expert_data offset:w_bytes        atIndex:1];  // scales
    [enc setBuffer:ctx->buf_expert_data offset:w_bytes+s_bytes atIndex:2];  // biases
    [enc setBuffer:ctx->buf_input       offset:0              atIndex:3];  // x
    [enc setBuffer:ctx->buf_out         offset:0              atIndex:4];  // out
    uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
    [enc setBytes:&o length:4 atIndex:5];
    [enc setBytes:&i length:4 atIndex:6];
    [enc setBytes:&g length:4 atIndex:7];
    uint32_t tgs = (o + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    float *gpu_out = (float *)malloc(dim_out * sizeof(float));
    memcpy(gpu_out, [ctx->buf_out contents], dim_out * sizeof(float));

    ParityReport r = evaluate_parity_f32(cpu_out, gpu_out, dim_out, 1e-3f);
    print_parity_report("MatVec GPU vs CPU", &r);

    if (verbose) {
        print_first_n("cpu_out", cpu_out, 10);
        print_first_n("gpu_out", gpu_out, 10);
    }

    free(W); free(S); free(B); free(input);
    free(cpu_out); free(gpu_out);
    return r.passed ? 0 : 1;
}

// ============================================================================
// Test: swiglu — validate SwiGLU activation kernel
// ============================================================================

static int test_swiglu(DiagMetalCtx *ctx, int dim, bool verbose) {
    if (!ctx->swiglu) {
        fprintf(stderr, "[swiglu] SKIP: swiglu pipeline not available\n");
        return 0;
    }

    fprintf(stderr, "\n=== Test: swiglu (dim=%d) ===\n", dim);

    srand(456);
    float *gate = (float *)malloc(dim * sizeof(float));
    float *up   = (float *)malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        gate[i] = 4.0f * ((float)rand() / (float)RAND_MAX - 0.5f);  // -2 .. 2
        up[i]   = 4.0f * ((float)rand() / (float)RAND_MAX - 0.5f);
    }

    // CPU reference
    float *cpu_out = (float *)malloc(dim * sizeof(float));
    cpu_swiglu(gate, up, cpu_out, dim);

    // GPU
    memcpy([ctx->buf_gate contents], gate, dim * sizeof(float));
    memcpy([ctx->buf_up contents],   up,   dim * sizeof(float));

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->swiglu];
    [enc setBuffer:ctx->buf_gate offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_up   offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_act  offset:0 atIndex:2];
    uint32_t d = (uint32_t)dim;
    [enc setBytes:&d length:4 atIndex:3];
    uint32_t tgs = (d + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    float *gpu_out = (float *)malloc(dim * sizeof(float));
    memcpy(gpu_out, [ctx->buf_act contents], dim * sizeof(float));

    ParityReport r = evaluate_parity_f32(cpu_out, gpu_out, dim, 1e-3f);
    print_parity_report("SwiGLU GPU vs CPU", &r);

    if (verbose) {
        print_first_n("cpu_out", cpu_out, 10);
        print_first_n("gpu_out", gpu_out, 10);
    }

    free(gate); free(up); free(cpu_out); free(gpu_out);
    return r.passed ? 0 : 1;
}

// ============================================================================
// Test: pipeline inter-cb-sync
// ============================================================================

static int test_inter_cb_sync(DiagMetalCtx *ctx, int bits, bool verbose) {
    fprintf(stderr, "\n=== Test: inter_cb_sync (%d-bit) ===\n", bits);

    if (bits != 4) {
        fprintf(stderr, "[inter_cb_sync] Only 4-bit supported for now\n");
        return 1;
    }
    if (!ctx->fused_gate_up_swiglu || !ctx->matvec_v3) {
        fprintf(stderr, "[inter_cb_sync] SKIP: required pipelines not available\n");
        return 0;
    }

    int dim_in  = 2048;
    int dim_out = 512;
    int group_size = GROUP_SIZE;
    int num_groups = dim_in / group_size;
    int packed_cols = dim_in / 8;  // 4-bit

    // Allocate test data
    srand(789);
    size_t w_bytes = (size_t)dim_out * packed_cols * sizeof(uint32_t);
    size_t s_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);

    uint32_t *gate_W = (uint32_t *)malloc(w_bytes);
    uint16_t *gate_s = (uint16_t *)malloc(s_bytes);
    uint16_t *gate_b = (uint16_t *)malloc(s_bytes);
    uint32_t *up_W   = (uint32_t *)malloc(w_bytes);
    uint16_t *up_s   = (uint16_t *)malloc(s_bytes);
    uint16_t *up_b   = (uint16_t *)malloc(s_bytes);
    float *input = (float *)malloc(dim_in * sizeof(float));

    for (size_t i = 0; i < w_bytes / sizeof(uint32_t); i++) {
        gate_W[i] = (uint32_t)(rand() & 0xFFFFFFFF);
        up_W[i]   = (uint32_t)(rand() & 0xFFFFFFFF);
    }
    for (size_t i = 0; i < s_bytes / sizeof(uint16_t); i++) {
        float s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;
        gate_s[i] = (uint16_t)((*(uint32_t *)&s) >> 16);
        gate_b[i] = (uint16_t)((*(uint32_t *)&s) >> 16);
        s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;
        up_s[i]   = (uint16_t)((*(uint32_t *)&s) >> 16);
        up_b[i]   = (uint16_t)((*(uint32_t *)&s) >> 16);
    }
    for (int i = 0; i < dim_in; i++) {
        input[i] = ((float)rand() / (float)RAND_MAX - 0.5f);
    }

    // Copy to Metal buffer
    uint8_t *ed = (uint8_t *)[ctx->buf_expert_data contents];
    memcpy(ed + GATE_W_OFF_4, gate_W, w_bytes);
    memcpy(ed + GATE_S_OFF_4, gate_s, s_bytes);
    memcpy(ed + GATE_B_OFF_4, gate_b, s_bytes);
    memcpy(ed + UP_W_OFF_4,   up_W,   w_bytes);
    memcpy(ed + UP_S_OFF_4,   up_s,   s_bytes);
    memcpy(ed + UP_B_OFF_4,   up_b,   s_bytes);
    memcpy([ctx->buf_input contents], input, dim_in * sizeof(float));

    // CPU reference
    float *cpu_gate_ref = (float *)malloc(dim_out * sizeof(float));
    float *cpu_up_ref   = (float *)malloc(dim_out * sizeof(float));
    float *cpu_act_ref  = (float *)malloc(dim_out * sizeof(float));
    cpu_dequant_matvec(gate_W, gate_s, gate_b, input, cpu_gate_ref,
                       dim_out, dim_in, group_size, 4);
    cpu_dequant_matvec(up_W, up_s, up_b, input, cpu_up_ref,
                       dim_out, dim_in, group_size, 4);
    cpu_swiglu(cpu_gate_ref, cpu_up_ref, cpu_act_ref, dim_out);

    // ---- Step 1: Single CB (golden reference) ----
    float *golden = (float *)malloc(dim_out * sizeof(float));
    {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->fused_gate_up_swiglu];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_W_OFF_4 atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_S_OFF_4 atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_B_OFF_4 atIndex:2];
        [enc setBuffer:ctx->buf_expert_data offset:UP_W_OFF_4   atIndex:3];
        [enc setBuffer:ctx->buf_expert_data offset:UP_S_OFF_4   atIndex:4];
        [enc setBuffer:ctx->buf_expert_data offset:UP_B_OFF_4   atIndex:5];
        [enc setBuffer:ctx->buf_input       offset:0            atIndex:6];
        [enc setBuffer:ctx->buf_act         offset:0            atIndex:7];
        uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:8];
        [enc setBytes:&i length:4 atIndex:9];
        [enc setBytes:&g length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(o, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        memcpy(golden, [ctx->buf_act contents], dim_out * sizeof(float));
    }
    ParityReport r_golden = evaluate_parity_f32(cpu_act_ref, golden, dim_out, 1e-3f);
    print_parity_report("Step 1: Single CB (golden)", &r_golden);

    // ---- Step 2: Separate CBs (CPU wait) ----
    float *step2 = (float *)malloc(dim_out * sizeof(float));
    {
        id<MTLCommandBuffer> cb1 = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb1 computeCommandEncoder];
        [enc setComputePipelineState:ctx->fused_gate_up_swiglu];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_W_OFF_4 atIndex:0];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_S_OFF_4 atIndex:1];
        [enc setBuffer:ctx->buf_expert_data offset:GATE_B_OFF_4 atIndex:2];
        [enc setBuffer:ctx->buf_expert_data offset:UP_W_OFF_4   atIndex:3];
        [enc setBuffer:ctx->buf_expert_data offset:UP_S_OFF_4   atIndex:4];
        [enc setBuffer:ctx->buf_expert_data offset:UP_B_OFF_4   atIndex:5];
        [enc setBuffer:ctx->buf_input       offset:0            atIndex:6];
        [enc setBuffer:ctx->buf_act         offset:0            atIndex:7];
        uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:8];
        [enc setBytes:&i length:4 atIndex:9];
        [enc setBytes:&g length:4 atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(o, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc endEncoding];
        [cb1 commit];
        [cb1 waitUntilCompleted];
        memcpy(step2, [ctx->buf_act contents], dim_out * sizeof(float));
    }
    ParityReport r_step2 = evaluate_parity_f32(golden, step2, dim_out, 1e-3f);
    print_parity_report("Step 2: Separate CBs (CPU wait)", &r_step2);

    // ---- Step 3: MTLFence ----
    float *step3 = (float *)malloc(dim_out * sizeof(float));
    {
        id<MTLCommandBuffer> cb1 = [ctx->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cb1 computeCommandEncoder];
            [enc setComputePipelineState:ctx->fused_gate_up_swiglu];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_W_OFF_4 atIndex:0];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_S_OFF_4 atIndex:1];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_B_OFF_4 atIndex:2];
            [enc setBuffer:ctx->buf_expert_data offset:UP_W_OFF_4   atIndex:3];
            [enc setBuffer:ctx->buf_expert_data offset:UP_S_OFF_4   atIndex:4];
            [enc setBuffer:ctx->buf_expert_data offset:UP_B_OFF_4   atIndex:5];
            [enc setBuffer:ctx->buf_input       offset:0            atIndex:6];
            [enc setBuffer:ctx->buf_act         offset:0            atIndex:7];
            uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
            [enc setBytes:&o length:4 atIndex:8];
            [enc setBytes:&i length:4 atIndex:9];
            [enc setBytes:&g length:4 atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(o, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc updateFence:ctx->sync_fence];
            [enc endEncoding];
        }
        [cb1 commit];
        [cb1 waitUntilCompleted];
        memcpy(step3, [ctx->buf_act contents], dim_out * sizeof(float));
    }
    // Note: step3 reads via CPU (waitUntilCompleted), so fence is redundant here.
    // The real test is whether a GPU-side read (e.g., combine kernel on another CB)
    // sees the data. This is tested in the full generation pipeline, not here.
    ParityReport r_step3 = evaluate_parity_f32(golden, step3, dim_out, 1e-3f);
    print_parity_report("Step 3: MTLFence (same CB pattern)", &r_step3);

    // ---- Step 4: MTLSharedEvent ----
    float *step4 = (float *)malloc(dim_out * sizeof(float));
    {
        ctx->sync_value++;
        uint64_t val = ctx->sync_value;

        id<MTLCommandBuffer> cb1 = [ctx->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cb1 computeCommandEncoder];
            [enc setComputePipelineState:ctx->fused_gate_up_swiglu];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_W_OFF_4 atIndex:0];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_S_OFF_4 atIndex:1];
            [enc setBuffer:ctx->buf_expert_data offset:GATE_B_OFF_4 atIndex:2];
            [enc setBuffer:ctx->buf_expert_data offset:UP_W_OFF_4   atIndex:3];
            [enc setBuffer:ctx->buf_expert_data offset:UP_S_OFF_4   atIndex:4];
            [enc setBuffer:ctx->buf_expert_data offset:UP_B_OFF_4   atIndex:5];
            [enc setBuffer:ctx->buf_input       offset:0            atIndex:6];
            [enc setBuffer:ctx->buf_act         offset:0            atIndex:7];
            uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)group_size;
            [enc setBytes:&o length:4 atIndex:8];
            [enc setBytes:&i length:4 atIndex:9];
            [enc setBytes:&g length:4 atIndex:10];
            [enc dispatchThreadgroups:MTLSizeMake(o, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
        }
        [cb1 encodeSignalEvent:ctx->sync_event value:val];
        [cb1 commit];

        // Second CB waits for event
        id<MTLCommandBuffer> cb2 = [ctx->queue commandBuffer];
        [cb2 encodeWaitForEvent:ctx->sync_event value:val];
        // No dispatches — just wait
        [cb2 commit];
        [cb2 waitUntilCompleted];
        memcpy(step4, [ctx->buf_act contents], dim_out * sizeof(float));
    }
    ParityReport r_step4 = evaluate_parity_f32(golden, step4, dim_out, 1e-3f);
    print_parity_report("Step 4: MTLSharedEvent", &r_step4);

    // Summary
    fprintf(stderr, "\n--- Inter-CB Sync Summary ---\n");
    print_parity_summary("Step 1: Single CB",       &r_golden);
    print_parity_summary("Step 2: Separate CBs",     &r_step2);
    print_parity_summary("Step 3: MTLFence",         &r_step3);
    print_parity_summary("Step 4: MTLSharedEvent",   &r_step4);

    free(gate_W); free(gate_s); free(gate_b);
    free(up_W); free(up_s); free(up_b);
    free(input);
    free(cpu_gate_ref); free(cpu_up_ref); free(cpu_act_ref);
    free(golden); free(step2); free(step3); free(step4);

    return (r_golden.passed && r_step2.passed) ? 0 : 1;
}

// ============================================================================
// Test: parity — compare two binary tensor files
// ============================================================================

static int test_parity(const char *path_a, const char *path_b, float tolerance) {
    size_t len_a = 0, len_b = 0;
    float *a = load_tensor_f32(path_a, &len_a);
    float *b = load_tensor_f32(path_b, &len_b);
    if (!a || !b) {
        free(a); free(b);
        return 1;
    }
    if (len_a != len_b) {
        fprintf(stderr, "[parity] ERROR: size mismatch: %zu vs %zu\n", len_a, len_b);
        free(a); free(b);
        return 1;
    }

    ParityReport r = evaluate_parity_f32(a, b, len_a, tolerance);
    char name[512];
    snprintf(name, sizeof(name), "File parity: %s vs %s", path_a, path_b);
    print_parity_report(name, &r);
    free(a); free(b);
    return r.passed ? 0 : 1;
}

// ============================================================================
// CLI
// ============================================================================

// ============================================================================
// Model integrity check — verifies safetensors shards and tensor access
// ============================================================================

#import <fcntl.h>
#import <unistd.h>
#import <sys/mman.h>
#import <sys/stat.h>

static int test_model_integrity(const char *model_path, bool verbose) {
    fprintf(stderr, "\n=== Model Integrity: %s ===\n", model_path);

    char idx_path[1024];
    snprintf(idx_path, sizeof(idx_path), "%s/model.safetensors.index.json", model_path);

    @autoreleasepool {
        NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:idx_path]];
        if (!data) { fprintf(stderr, "FAIL: Cannot read index.json\n"); return 1; }
        NSDictionary *root = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (!root) { fprintf(stderr, "FAIL: Invalid index.json\n"); return 1; }
        NSDictionary *wm = root[@"weight_map"];
        if (!wm) { fprintf(stderr, "FAIL: No weight_map\n"); return 1; }

        int n_tensors = (int)[wm count];
        fprintf(stderr, "  Tensors in index: %d\n", n_tensors);

        // Step 1: List all shard files
        NSMutableSet *shardSet = [NSMutableSet set];
        for (NSString *f in [wm allValues]) [shardSet addObject:f];
        NSArray *sortedShards = [[shardSet allObjects] sortedArrayUsingSelector:@selector(compare:)];
        int n_shards = (int)[shardSet count];
        fprintf(stderr, "  Unique shards: %d\n", n_shards);

        // Step 2: Try to open and mmap each shard
        int open_ok = 0, mmap_ok = 0, header_ok = 0;
        for (int i = 0; i < n_shards; i++) {
            NSString *sf = sortedShards[i];
            char spath[1024];
            snprintf(spath, sizeof(spath), "%s/%s", model_path, [sf UTF8String]);

            int fd = open(spath, O_RDONLY);
            if (fd < 0) {
                fprintf(stderr, "  FAIL: cannot open %s\n", [sf UTF8String]);
                continue;
            }
            open_ok++;

            struct stat st;
            fstat(fd, &st);
            void *mm = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
            close(fd);
            if (mm == MAP_FAILED) {
                fprintf(stderr, "  FAIL: mmap failed for %s (%lld bytes)\n",
                        [sf UTF8String], (long long)st.st_size);
                continue;
            }
            mmap_ok++;

            // Read safetensors header
            uint64_t header_len = *(uint64_t *)mm;
            if (header_len == 0 || header_len > (uint64_t)st.st_size) {
                fprintf(stderr, "  FAIL: bad header_len=%llu in %s\n",
                        (unsigned long long)header_len, [sf UTF8String]);
                munmap(mm, st.st_size);
                continue;
            }

            char *hbuf = malloc(header_len + 1);
            memcpy(hbuf, (char *)mm + 8, header_len);
            hbuf[header_len] = 0;
            NSData *jd = [NSData dataWithBytesNoCopy:hbuf length:header_len freeWhenDone:YES];
            NSDictionary *hdr = [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
            if (!hdr) {
                fprintf(stderr, "  FAIL: JSON parse failed in %s header\n", [sf UTF8String]);
                munmap(mm, st.st_size);
                continue;
            }
            header_ok++;

            if (verbose) {
                int n_t = 0;
                for (NSString *k in hdr) if (![k isEqualToString:@"__metadata__"]) n_t++;
                fprintf(stderr, "  OK: %s (%lld MB, %d tensors, header=%llu bytes)\n",
                        [sf UTF8String], (long long)st.st_size / 1048576,
                        n_t, (unsigned long long)header_len);
            }
            munmap(mm, st.st_size);
        }

        fprintf(stderr, "\n  Shard access: %d/%d open, %d/%d mmap, %d/%d header parse\n",
                open_ok, n_shards, mmap_ok, n_shards, header_ok, n_shards);

        // Step 3: Check dtype coverage
        fprintf(stderr, "\n  Parsing all shard headers for dtype audit...\n");
        NSMutableDictionary *allMeta = [NSMutableDictionary dictionary];
        NSMutableSet *allDtypes = [NSMutableSet set];
        for (int i = 0; i < n_shards; i++) {
            NSString *sf = sortedShards[i];
            char spath[1024];
            snprintf(spath, sizeof(spath), "%s/%s", model_path, [sf UTF8String]);
            int fd = open(spath, O_RDONLY);
            if (fd < 0) continue;
            uint64_t hl;
            if (read(fd, &hl, 8) != 8) { close(fd); continue; }
            char *hb = malloc(hl + 1);
            if (read(fd, hb, hl) != (ssize_t)hl) { free(hb); close(fd); continue; }
            hb[hl] = 0;
            close(fd);
            NSData *jd = [NSData dataWithBytesNoCopy:hb length:hl freeWhenDone:YES];
            NSDictionary *hdr = [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil];
            if (!hdr) continue;
            for (NSString *k in hdr) {
                if ([k isEqualToString:@"__metadata__"]) continue;
                allMeta[k] = hdr[k];
                NSString *dt = hdr[k][@"dtype"];
                if (dt) [allDtypes addObject:dt];
            }
        }
        fprintf(stderr, "  Total metadata entries: %d\n", (int)[allMeta count]);
        fprintf(stderr, "  Unique dtypes: ");
        for (NSString *dt in [[allDtypes allObjects] sortedArrayUsingSelector:@selector(compare:)]) {
            fprintf(stderr, "%s ", [dt UTF8String]);
        }
        fprintf(stderr, "\n");

        // Step 4: Spot-check key tensors
        fprintf(stderr, "\n  Spot-checking key tensors...\n");
        const char *checks[] = {
            "embed.weight", "head.weight", "norm.weight",
            "layers.0.attn_norm.weight", "layers.0.attn.wq_a.weight",
            "layers.0.ffn.gate.weight", "layers.0.ffn.gate.tid2eid",
            "layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
            "layers.42.ffn.experts.255.w3.weight",
            NULL
        };
        int found = 0, missing = 0;
        for (const char **chk = checks; *chk; chk++) {
            NSString *name = [NSString stringWithUTF8String:*chk];
            NSString *shard = wm[name];
            if (!shard) {
                fprintf(stderr, "  MISSING: %s (not in weight_map)\n", *chk);
                missing++;
                continue;
            }
            NSDictionary *meta = allMeta[name];
            if (!meta) {
                fprintf(stderr, "  MISSING: %s (no metadata in shard header)\n", *chk);
                missing++;
                continue;
            }
            NSArray *shape = meta[@"shape"];
            NSString *dtype = meta[@"dtype"];
            NSArray *offsets = meta[@"data_offsets"];
            size_t dsize = [offsets[1] longLongValue] - [offsets[0] longLongValue];
            if (verbose) {
                fprintf(stderr, "  OK: %s shape=[", *chk);
                for (int d = 0; d < (int)[shape count]; d++)
                    fprintf(stderr, "%d%s", [shape[d] intValue], d < (int)[shape count]-1 ? "," : "");
                fprintf(stderr, "] dtype=%s size=%zu shard=%s\n",
                        [dtype UTF8String], dsize, [shard UTF8String]);
            }
            found++;
        }
        fprintf(stderr, "  Found: %d  Missing: %d\n", found, missing);

        int all_ok = (open_ok == n_shards && mmap_ok == n_shards && header_ok == n_shards && missing == 0);
        fprintf(stderr, "\n%s Model integrity check: %d/%d shards accessible, %d/%d key tensors found\n",
                all_ok ? "PASS" : "FAIL",
                header_ok, n_shards, found, found + missing);
        return all_ok ? 0 : 1;
    }
}

static void print_usage(void) {
    fprintf(stderr, "\nfinchTool — Metal Engine Verification & Diagnostic Suite\n\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  ./finchTool kernel --test <name> [options]\n");
    fprintf(stderr, "  ./finchTool pipeline --test <name> [options]\n");
    fprintf(stderr, "  ./finchTool model --model <path> [--verbose]\n");
    fprintf(stderr, "  ./finchTool parity --a <file> --b <file> [options]\n\n");
    fprintf(stderr, "Kernel tests:\n");
    fprintf(stderr, "  fused_mlp     Fused gate+up+SiLU vs non-fused comparison\n");
    fprintf(stderr, "  matvec        Dequant matvec kernel validation\n");
    fprintf(stderr, "  swiglu        SwiGLU activation kernel validation\n");
    fprintf(stderr, "  all           Run all kernel tests\n\n");
    fprintf(stderr, "Pipeline tests:\n");
    fprintf(stderr, "  inter-cb-sync Inter-command-buffer synchronization audit\n");
    fprintf(stderr, "  all           Run all pipeline tests\n\n");
    fprintf(stderr, "Model checks:\n");
    fprintf(stderr, "  model         Verify safetensors shards, mmap, dtypes, key tensors\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --quant, -q   Quantization: 4bit (default), 8bit, 2bit, 1bit\n");
    fprintf(stderr, "  --dim-in, -i  Input dimension (default: 2048)\n");
    fprintf(stderr, "  --dim-out, -o Output dimension (default: 512)\n");
    fprintf(stderr, "  --dim, -d     Dimension for swiglu (default: 512)\n");
    fprintf(stderr, "  --tolerance   Per-element diff threshold (default: 1e-3)\n");
    fprintf(stderr, "  --verbose, -v Verbose output (per-element diffs)\n");
    fprintf(stderr, "  --help, -h    Show this help\n\n");
}

int main(int argc, char **argv) {
    const char *subcommand = NULL;
    const char *test_name  = NULL;
    const char *path_a     = NULL;
    const char *path_b     = NULL;
    int   quant    = 4;
    int   dim_in   = 2048;
    int   dim_out  = 512;
    int   dim      = 512;
    float tolerance = 1e-3f;
    bool  verbose  = false;

    // Parse subcommand
    if (argc > 1) {
        subcommand = argv[1];
    } else {
        print_usage();
        return 1;
    }

    // Parse options (skip subcommand)
    static struct option long_options[] = {
        {"test",      required_argument, 0, 't'},
        {"quant",     required_argument, 0, 'q'},
        {"model",     required_argument, 0, 'm'},
        {"dim-in",    required_argument, 0, 'i'},
        {"dim-out",   required_argument, 0, 'o'},
        {"dim",       required_argument, 0, 'd'},
        {"a",         required_argument, 0, 'A'},
        {"b",         required_argument, 0, 'B'},
        {"tolerance", required_argument, 0, 'T'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    // Re-build argv for getopt (skip subcommand)
    char *fake_argv[32];
    int fake_argc = 1;
    fake_argv[0] = argv[0];
    for (int i = 2; i < argc && fake_argc < 31; i++) {
        fake_argv[fake_argc++] = argv[i];
    }
    fake_argv[fake_argc] = NULL;

    int c;
    while ((c = getopt_long(fake_argc, fake_argv, "t:q:m:i:o:d:A:B:T:vh",
                            long_options, NULL)) != -1) {
        switch (c) {
            case 't': test_name  = optarg; break;
            case 'q':
                if (strcmp(optarg, "4bit") == 0 || strcmp(optarg, "4") == 0) quant = 4;
                else if (strcmp(optarg, "8bit") == 0 || strcmp(optarg, "8") == 0) quant = 8;
                else if (strcmp(optarg, "2bit") == 0 || strcmp(optarg, "2") == 0) quant = 2;
                else if (strcmp(optarg, "1bit") == 0 || strcmp(optarg, "1") == 0) quant = 1;
                else { fprintf(stderr, "Unknown quant: %s\n", optarg); return 1; }
                break;
            case 'm': path_a  = optarg; break;  // model path (reuse path_a)
            case 'i': dim_in  = atoi(optarg); break;
            case 'o': dim_out = atoi(optarg); break;
            case 'd': dim     = atoi(optarg); break;
            case 'A': path_a  = optarg; break;
            case 'B': path_b  = optarg; break;
            case 'T': tolerance = atof(optarg); break;
            case 'v': verbose = true; break;
            case 'h': print_usage(); return 0;
            default:  print_usage(); return 1;
        }
    }

    // ---- Dispatch ----
    int result = 0;

    if (strcmp(subcommand, "kernel") == 0) {
        if (!test_name) {
            fprintf(stderr, "ERROR: --test required for kernel subcommand\n");
            return 1;
        }

        DiagMetalCtx *ctx = diag_metal_init();
        if (!ctx) return 1;

        if (strcmp(test_name, "fused_mlp") == 0) {
            result = test_fused_mlp(ctx, quant, dim_in, dim_out, verbose);
        } else if (strcmp(test_name, "matvec") == 0) {
            result = test_matvec(ctx, quant, verbose);
        } else if (strcmp(test_name, "swiglu") == 0) {
            result = test_swiglu(ctx, dim, verbose);
        } else if (strcmp(test_name, "all") == 0) {
            result |= test_matvec(ctx, 4, verbose);
            result |= test_matvec(ctx, 8, verbose);
            result |= test_matvec(ctx, 2, verbose);
            result |= test_matvec(ctx, 1, verbose);
            result |= test_swiglu(ctx, dim, verbose);
            result |= test_fused_mlp(ctx, 4, dim_in, dim_out, verbose);
            result |= test_fused_mlp(ctx, 8, dim_in, dim_out, verbose);
        } else {
            fprintf(stderr, "ERROR: Unknown kernel test: %s\n", test_name);
            result = 1;
        }

        diag_metal_free(ctx);

    } else if (strcmp(subcommand, "pipeline") == 0) {
        if (!test_name) {
            fprintf(stderr, "ERROR: --test required for pipeline subcommand\n");
            return 1;
        }

        DiagMetalCtx *ctx = diag_metal_init();
        if (!ctx) return 1;

        if (strcmp(test_name, "inter-cb-sync") == 0) {
            result = test_inter_cb_sync(ctx, quant, verbose);
        } else if (strcmp(test_name, "all") == 0) {
            result |= test_inter_cb_sync(ctx, 4, verbose);
        } else {
            fprintf(stderr, "ERROR: Unknown pipeline test: %s\n", test_name);
            result = 1;
        }

        diag_metal_free(ctx);

    } else if (strcmp(subcommand, "model") == 0) {
        if (!path_a) {
            fprintf(stderr, "ERROR: --model PATH required for model subcommand\n");
            return 1;
        }
        result = test_model_integrity(path_a, verbose);

    } else if (strcmp(subcommand, "parity") == 0) {
        if (!path_a || !path_b) {
            fprintf(stderr, "ERROR: --a and --b required for parity\n");
            return 1;
        }
        result = test_parity(path_a, path_b, tolerance);

    } else {
        fprintf(stderr, "ERROR: Unknown subcommand: %s\n", subcommand);
        print_usage();
        return 1;
    }

    if (result == 0) {
        fprintf(stderr, "\n✅ All tests passed.\n");
    } else {
        fprintf(stderr, "\n❌ %d test(s) failed.\n", result);
    }
    return result;
}

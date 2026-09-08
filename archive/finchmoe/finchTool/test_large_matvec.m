// test_large_matvec.m — Test 4-bit dequant matvec at non-expert scale
// Tests [2048, 2048] (attention O-proj) and larger
// Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation test_large_matvec.m -o test_large_matvec
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Mirrors the production kernel constants
#define GROUP_SIZE 64

// BF16 conversion helper
static inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// CPU reference: dequant + matvec
static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out, int out_dim, int in_dim, int group_size, int bits)
{
    int vals_per_u32 = 32 / bits;
    int groups = in_dim / group_size;
    uint32_t mask = (1u << bits) - 1;
    for (int row = 0; row < out_dim; row++) {
        double acc = 0.0;
        for (int g = 0; g < groups; g++) {
            float s = bf16_to_f32(scales[row * groups + g]);
            float b = bf16_to_f32(biases[row * groups + g]);
            for (int u = 0; u < group_size / vals_per_u32; u++) {
                uint32_t packed = W[row * (groups * group_size / vals_per_u32) + g * (group_size / vals_per_u32) + u];
                for (int v = 0; v < vals_per_u32; v++) {
                    int idx = g * group_size + u * vals_per_u32 + v;
                    float w = (float)((packed >> (v * bits)) & mask) * s + b;
                    acc += (double)w * (double)x[idx];
                }
            }
        }
        out[row] = (float)acc;
    }
}

static float vec_rms(const float *x, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * x[i];
    return (float)sqrt(sum / n);
}

static void print_first_n(const char *label, const float *x, int n) {
    fprintf(stderr, "  %s (first %d): ", label, n);
    for (int i = 0; i < n && i < 10; i++) fprintf(stderr, "%.4f ", x[i]);
    fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    int dim_in = 2048;
    int dim_out = 2048;  // default: attention O-proj
    int bits = 4;
    if (argc > 1) dim_out = atoi(argv[1]);
    if (argc > 2) bits = atoi(argv[2]);

    fprintf(stderr, "=== Test: large matvec (%d-bit, in=%d, out=%d) ===\n", bits, dim_in, dim_out);

    int vals_per_u32 = 32 / bits;
    int packed_cols = dim_in / vals_per_u32;
    int num_groups = dim_in / GROUP_SIZE;
    size_t w_bytes = (size_t)dim_out * packed_cols * sizeof(uint32_t);
    size_t s_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);
    size_t b_bytes = (size_t)dim_out * num_groups * sizeof(uint16_t);
    size_t x_bytes = (size_t)dim_in * sizeof(float);
    size_t out_bytes = (size_t)dim_out * sizeof(float);

    fprintf(stderr, "  W: %zu bytes (%.2f MB), S: %zu bytes, B: %zu bytes\n",
            w_bytes, (double)w_bytes / 1e6, s_bytes, b_bytes);

    // Generate synthetic test data
    srand(42);
    uint32_t *W = (uint32_t *)malloc(w_bytes);
    uint16_t *S = (uint16_t *)malloc(s_bytes);
    uint16_t *B = (uint16_t *)malloc(b_bytes);
    float *input = (float *)malloc(x_bytes);
    float *cpu_out = (float *)malloc(out_bytes);
    float *gpu_out = (float *)malloc(out_bytes);

    for (size_t i = 0; i < w_bytes / sizeof(uint32_t); i++)
        W[i] = (uint32_t)(rand() & 0xFFFFFFFF);
    for (size_t i = 0; i < s_bytes / sizeof(uint16_t); i++) {
        float s = 0.5f + 1.5f * (float)rand() / (float)RAND_MAX;
        S[i] = (uint16_t)((*(uint32_t *)&s) >> 16);
        float b = 0.2f * ((float)rand() / (float)RAND_MAX - 0.5f);
        B[i] = (uint16_t)((*(uint32_t *)&b) >> 16);
    }
    for (int i = 0; i < dim_in; i++)
        input[i] = ((float)rand() / (float)RAND_MAX - 0.5f);

    // CPU reference
    cpu_dequant_matvec(W, S, B, input, cpu_out, dim_out, dim_in, GROUP_SIZE, bits);
    fprintf(stderr, "  CPU: rms=%.6f\n", vec_rms(cpu_out, dim_out));

    // Metal setup
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { fprintf(stderr, "No Metal device\n"); return 1; }

    // Compile shader
    NSString *shader_path = @"../shaders.metal";
    NSString *src = [NSString stringWithContentsOfFile:shader_path encoding:NSUTF8StringEncoding error:nil];
    if (!src) { fprintf(stderr, "Cannot read %s\n", [shader_path UTF8String]); return 1; }
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:nil];
    if (!lib) { fprintf(stderr, "Shader compile failed\n"); return 1; }

    id<MTLFunction> fn = [lib newFunctionWithName:
        bits == 4 ? @"dequant_matvec_4bit_v3" :
        bits == 8 ? @"dequant_matvec_8bit" :
        bits == 2 ? @"dequant_matvec_2bit" :
        @"dequant_matvec_1bit"];
    if (!fn) { fprintf(stderr, "Function not found\n"); return 1; }

    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:nil];
    if (!pipe) { fprintf(stderr, "Pipeline creation failed\n"); return 1; }

    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Allocate GPU buffers — need enough for the largest case
    size_t gpu_data_size = (w_bytes + s_bytes + b_bytes + 256) & ~255;
    id<MTLBuffer> buf_data = [device newBufferWithLength:gpu_data_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_input = [device newBufferWithLength:(x_bytes > out_bytes ? x_bytes : out_bytes) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [device newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];

    if (!buf_data || !buf_input || !buf_out) {
        fprintf(stderr, "Buffer allocation failed (requested %.1f MB)\n", (double)gpu_data_size / 1e6);
        return 1;
    }

    // Copy data to GPU
    memcpy([buf_data contents], W, w_bytes);
    memcpy((uint8_t *)[buf_data contents] + w_bytes, S, s_bytes);
    memcpy((uint8_t *)[buf_data contents] + w_bytes + s_bytes, B, b_bytes);
    memcpy([buf_input contents], input, x_bytes);

    // Dispatch
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:buf_data  offset:0              atIndex:0];  // W
    [enc setBuffer:buf_data  offset:w_bytes        atIndex:1];  // scales
    [enc setBuffer:buf_data  offset:w_bytes+s_bytes atIndex:2]; // biases
    [enc setBuffer:buf_input offset:0              atIndex:3];  // x
    [enc setBuffer:buf_out   offset:0              atIndex:4];  // out
    uint32_t o = (uint32_t)dim_out, i = (uint32_t)dim_in, g = (uint32_t)GROUP_SIZE;
    [enc setBytes:&o length:4 atIndex:5];
    [enc setBytes:&i length:4 atIndex:6];
    [enc setBytes:&g length:4 atIndex:7];
    uint32_t tgs = (o + 7) / 8;
    [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if ([cb error]) {
        fprintf(stderr, "GPU error: %s\n", [[cb error].description UTF8String]);
        return 1;
    }

    memcpy(gpu_out, [buf_out contents], out_bytes);

    // Compare
    double max_diff = 0, sum_diff = 0, cpu_sum_sq = 0;
    for (int i = 0; i < dim_out; i++) {
        double d = fabs((double)cpu_out[i] - (double)gpu_out[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
        cpu_sum_sq += (double)cpu_out[i] * cpu_out[i];
    }
    double avg_diff = sum_diff / dim_out;
    double dot = 0, cpu_norm = 0, gpu_norm = 0;
    for (int i = 0; i < dim_out; i++) {
        dot += (double)cpu_out[i] * (double)gpu_out[i];
        cpu_norm += (double)cpu_out[i] * cpu_out[i];
        gpu_norm += (double)gpu_out[i] * gpu_out[i];
    }
    double cos_sim = dot / (sqrt(cpu_norm) * sqrt(gpu_norm));
    int passed = (cos_sim > 0.999);

    fprintf(stderr, "\n  CosSim=%.8f  MaxDiff=%.4e  AvgDiff=%.4e\n", cos_sim, max_diff, avg_diff);
    fprintf(stderr, "  GPU rms=%.6f (CPU rms=%.6f)\n", vec_rms(gpu_out, dim_out), vec_rms(cpu_out, dim_out));
    fprintf(stderr, "  %s\n", passed ? "PASS" : "FAIL");

    if (!passed) {
        print_first_n("cpu_out", cpu_out, 10);
        print_first_n("gpu_out", gpu_out, 10);
    }

    free(W); free(S); free(B); free(input); free(cpu_out); free(gpu_out);
    return passed ? 0 : 1;
}

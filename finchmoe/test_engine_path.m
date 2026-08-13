// Test EXACT engine code path: single wf_buf with offsets
// Build: clang -O2 -fobjc-arc -framework Metal -framework Foundation test_engine_path.m -o test_engine_path
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>

static inline float bf16_to_f32_cpu(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f; memcpy(&f, &u, sizeof(f)); return f;
}

// bits: 4 or 8. W layout matches engine: [out_dim, row_u32]
static void cpu_dequant_matvec_bits(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out, int out_dim, int in_dim, int group_size, int bits)
{
    int vals_per_u32 = 32 / bits;
    int groups = in_dim / group_size;
    uint32_t mask = (1u << bits) - 1;
    for (int row = 0; row < out_dim; row++) {
        double acc = 0.0;
        const uint32_t *w_row = W + row * (groups * group_size / vals_per_u32);
        const uint16_t *s_row = scales + row * groups;
        const uint16_t *b_row = biases + row * groups;
        for (int g = 0; g < groups; g++) {
            float s = bf16_to_f32_cpu(s_row[g]);
            float b = bf16_to_f32_cpu(b_row[g]);
            for (int u = 0; u < group_size / vals_per_u32; u++) {
                uint32_t packed = w_row[g * (group_size / vals_per_u32) + u];
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

// Struct describing one test
typedef struct {
    const char *name;
    int bits;          // correct packing (4 or 8)
    int force_group;   // group size (64 for all current tensors)
} TestSpec;

// Compare GPU result vs CPU reference
static void compare_and_report(const char *label, const char *tname,
                               float *cpu_out, float *gpu_out, int out_dim)
{
    double maxd = 0, dot = 0, n1 = 0, n2 = 0;
    int nan_count = 0;
    for (int j = 0; j < out_dim; j++) {
        double d = fabs((double)cpu_out[j] - gpu_out[j]);
        if (d > maxd) maxd = d;
        dot += (double)cpu_out[j] * gpu_out[j];
        n1 += (double)cpu_out[j] * cpu_out[j];
        n2 += (double)gpu_out[j] * gpu_out[j];
        if (!isfinite(gpu_out[j])) nan_count++;
    }
    double cos_sim = (n1 > 0 && n2 > 0) ? dot / sqrt(n1 * n2) : -99.0;
    printf("  [%s] %s: CosSim=%.8f MaxDiff=%.2e nan=%d  %s\n",
           label, tname, cos_sim, maxd, nan_count,
           (cos_sim > 0.999 && nan_count == 0) ? "PASS" : "FAIL");
    if (nan_count > 0 || cos_sim < 0.999) {
        printf("    GPU first5: [%.4f %.4f %.4f %.4f %.4f]\n",
               gpu_out[0], gpu_out[1], gpu_out[2], gpu_out[3], gpu_out[4]);
        printf("    CPU first5: [%.4f %.4f %.4f %.4f %.4f]\n",
               cpu_out[0], cpu_out[1], cpu_out[2], cpu_out[3], cpu_out[4]);
    }
}

int main() {
    const char *json_path = "quant_test/model_weights.json";
    const char *bin_path = "quant_test/model_weights.bin";

    NSString *json_str = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:json_path] encoding:NSUTF8StringEncoding error:nil];
    NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:[json_str dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    NSDictionary *tensors = manifest[@"tensors"];

    int fd = open(bin_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *mmap_data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("Mmap'd %.2f MB at %p\n", st.st_size / 1e6, mmap_data);

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal" encoding:NSUTF8StringEncoding error:nil];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:nil];
    id<MTLFunction> fn_v3 = [lib newFunctionWithName:@"dequant_matvec_4bit_v3"];
    id<MTLFunction> fn_fast = [lib newFunctionWithName:@"dequant_matvec_4bit_fast"];
    id<MTLFunction> fn_8bit = [lib newFunctionWithName:@"dequant_matvec_8bit"];
    id<MTLComputePipelineState> pipe_v3 = [device newComputePipelineStateWithFunction:fn_v3 error:nil];
    id<MTLComputePipelineState> pipe_fast = [device newComputePipelineStateWithFunction:fn_fast error:nil];
    id<MTLComputePipelineState> pipe_8bit = [device newComputePipelineStateWithFunction:fn_8bit error:nil];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    id<MTLBuffer> wf_buf = [device newBufferWithBytesNoCopy:mmap_data length:st.st_size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];

    // ---- Test list ----
    // kernel: 0=v3, 1=fast, 2=8bit, 3=v3-on-8bit-data (mimics engine bug)
    struct { const char *name; int kernel; } tests[] = {
        // CMD3 shared expert down_proj [2048, 512] via v3
        {"model.layers.0.mlp.shared_expert.down_proj", 0},
        // CMD2 shared gate/up [512, 2048] via v3
        {"model.layers.0.mlp.shared_expert.gate_proj", 0},
        {"model.layers.0.mlp.shared_expert.up_proj",   0},
        // CMD2 4-bit o_proj path uses matvec_fast
        {"model.layers.0.linear_attn.out_proj", 1},
        {"model.layers.3.self_attn.o_proj",     1},
        // Routing gate: 8-bit data. kernel 3 = what the engine does (v3/4-bit).
        {"model.layers.0.mlp.gate",              3},
        {"model.layers.0.mlp.gate",              2},
        // shared_expert_gate: 8-bit data [1, 2048]
        {"model.layers.0.mlp.shared_expert_gate", 3},
        {"model.layers.0.mlp.shared_expert_gate", 2},
    };
    int n_tests = sizeof(tests) / sizeof(tests[0]);

    for (int ti = 0; ti < n_tests; ti++) {
        char wname[256], sname[256], bname[256];
        snprintf(wname, sizeof(wname), "%s.weight", tests[ti].name);
        snprintf(sname, sizeof(sname), "%s.scales", tests[ti].name);
        snprintf(bname, sizeof(bname), "%s.biases", tests[ti].name);

        NSDictionary *wi = tensors[[NSString stringWithUTF8String:wname]];
        NSDictionary *si = tensors[[NSString stringWithUTF8String:sname]];
        NSDictionary *bi = tensors[[NSString stringWithUTF8String:bname]];

        if (!wi || !si || !bi) {
            printf("SKIP %s (missing)\n", tests[ti].name);
            continue;
        }

        int out_dim = [wi[@"shape"][0] intValue];
        int groups = [si[@"shape"][1] intValue];
        int row_u32 = [wi[@"shape"][1] intValue];
        int group_size = 64;
        // Detect packing: 4-bit => row_u32 == groups*8; 8-bit => row_u32 == groups*16
        int bits = (row_u32 == groups * 16) ? 8 : 4;
        int in_dim = groups * group_size;

        size_t w_off = [wi[@"offset"] unsignedLongValue];
        size_t s_off = [si[@"offset"] unsignedLongValue];
        size_t b_off = [bi[@"offset"] unsignedLongValue];

        printf("\n--- %s (kernel=%d) out=%d in=%d bits=%d groups=%d ---\n",
               tests[ti].name, tests[ti].kernel, out_dim, in_dim, bits, groups);

        // Input
        srand(42 + ti);
        float *input = malloc(in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) input[i] = ((float)rand() / RAND_MAX - 0.5f);

        id<MTLBuffer> buf_x = [device newBufferWithBytes:input length:in_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_o = [device newBufferWithLength:out_dim*sizeof(float) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        int kernel = tests[ti].kernel;
        // For kernel 3 (v3 on 8-bit data): dispatch v3 with 4-bit geometry, same as engine
        if (kernel == 3) kernel = 0;

        if (kernel == 0) [enc setComputePipelineState:pipe_v3];
        else if (kernel == 1) [enc setComputePipelineState:pipe_fast];
        else [enc setComputePipelineState:pipe_8bit];

        [enc setBuffer:wf_buf offset:w_off atIndex:0];
        [enc setBuffer:wf_buf offset:s_off atIndex:1];
        [enc setBuffer:wf_buf offset:b_off atIndex:2];
        [enc setBuffer:buf_x  offset:0     atIndex:3];
        [enc setBuffer:buf_o  offset:0     atIndex:4];
        uint32_t o = (uint32_t)out_dim, i = (uint32_t)in_dim, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:5];
        [enc setBytes:&i length:4 atIndex:6];
        [enc setBytes:&g length:4 atIndex:7];
        if (kernel == 1) {
            [enc dispatchThreadgroups:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else {
            uint32_t tgs = (out_dim + 7) / 8;
            [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *gpu_out = malloc(out_dim * sizeof(float));
        memcpy(gpu_out, [buf_o contents], out_dim * sizeof(float));

        // CPU reference with the tensor's TRUE packing
        uint32_t *W = (uint32_t*)((char*)mmap_data + w_off);
        uint16_t *S = (uint16_t*)((char*)mmap_data + s_off);
        uint16_t *B = (uint16_t*)((char*)mmap_data + b_off);
        float *cpu_out = malloc(out_dim * sizeof(float));
        cpu_dequant_matvec_bits(W, S, B, input, cpu_out, out_dim, in_dim, group_size, bits);

        const char *label = tests[ti].kernel == 3 ? "v3-on-8bit (ENGINE BUG REPRO)"
                          : tests[ti].kernel == 2 ? "8bit-kernel"
                          : tests[ti].kernel == 1 ? "fast"
                          : "v3";
        compare_and_report(label, tests[ti].name, cpu_out, gpu_out, out_dim);

        free(input); free(cpu_out); free(gpu_out);
    }

    munmap(mmap_data, st.st_size);
    close(fd);
    return 0;
}

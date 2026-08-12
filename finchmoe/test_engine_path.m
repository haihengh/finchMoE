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

static void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out, int out_dim, int in_dim, int group_size)
{
    int vals_per_u32 = 8;
    int groups = in_dim / group_size;
    int packed_per_group = group_size / vals_per_u32;
    for (int row = 0; row < out_dim; row++) {
        double acc = 0.0;
        const uint32_t *w_row = W + row * packed_per_group * groups;
        const uint16_t *s_row = scales + row * groups;
        const uint16_t *b_row = biases + row * groups;
        for (int g = 0; g < groups; g++) {
            float s = bf16_to_f32_cpu(s_row[g]);
            float b = bf16_to_f32_cpu(b_row[g]);
            for (int p = 0; p < packed_per_group; p++) {
                uint32_t packed = w_row[g * packed_per_group + p];
                int base = g * group_size + p * vals_per_u32;
                for (int n = 0; n < vals_per_u32; n++) {
                    float w_val = (float)((packed >> (n * 4)) & 0xF) * s + b;
                    acc += (double)w_val * (double)x[base + n];
                }
            }
        }
        out[row] = (float)acc;
    }
}

int main() {
    // Load JSON manifest
    const char *json_path = "quant_test/model_weights.json";
    const char *bin_path = "quant_test/model_weights.bin";

    NSString *json_str = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:json_path] encoding:NSUTF8StringEncoding error:nil];
    NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:[json_str dataUsingEncoding:NSUTF8StringEncoding] options:0 error:nil];
    NSDictionary *tensors = manifest[@"tensors"];

    // mmap the EXACT same file the engine uses
    int fd = open(bin_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    void *mmap_data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    printf("Mmap'd %.2f MB at %p\n", st.st_size / 1e6, mmap_data);

    // Metal setup
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSString *src = [NSString stringWithContentsOfFile:@"shaders.metal" encoding:NSUTF8StringEncoding error:nil];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:nil];
    id<MTLFunction> fn = [lib newFunctionWithName:@"dequant_matvec_4bit_v3"];
    id<MTLComputePipelineState> pipe = [device newComputePipelineStateWithFunction:fn error:nil];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // Wrap as Metal buffer — EXACTLY like engine
    id<MTLBuffer> wf_buf = [device newBufferWithBytesNoCopy:mmap_data length:st.st_size
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    printf("Metal buffer: %p length=%lu\n", (void*)[wf_buf contents], (unsigned long)[wf_buf length]);

    // Test key tensors
    struct { const char *name; int out_dim; int in_dim; } tests[] = {
        {"model.layers.0.linear_attn.in_proj_qkv", 8192, 2048},
        {"model.layers.3.self_attn.q_proj", 8192, 2048},
        {"model.layers.0.linear_attn.out_proj", 2048, 4096},
        {"model.layers.3.self_attn.o_proj", 2048, 4096},
    };

    for (int ti = 0; ti < 4; ti++) {
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
        int in_dim = out_dim == [wi[@"shape"][0] intValue] ? 2048 : 4096;
        int group_size = 64;

        // Get byte offsets — EXACTLY like engine
        size_t w_off = [wi[@"offset"] unsignedLongValue];
        size_t s_off = [si[@"offset"] unsignedLongValue];
        size_t b_off = [bi[@"offset"] unsignedLongValue];

        printf("\n--- %s out=%d in=%d ---\n", tests[ti].name, out_dim, in_dim);
        printf("  w_off=%lu s_off=%lu b_off=%lu\n", w_off, s_off, b_off);
        printf("  W data first 4 bytes: 0x%08x\n", *(uint32_t*)((char*)mmap_data + w_off));
        printf("  S data first BF16: %.6f\n", bf16_to_f32_cpu(*(uint16_t*)((char*)mmap_data + s_off)));
        printf("  B data first BF16: %.6f\n", bf16_to_f32_cpu(*(uint16_t*)((char*)mmap_data + b_off)));

        // Input
        srand(42 + ti);
        float *input = malloc(in_dim * sizeof(float));
        for (int i = 0; i < in_dim; i++) input[i] = ((float)rand() / RAND_MAX - 0.5f);

        // GPU input buffer
        id<MTLBuffer> buf_x = [device newBufferWithBytes:input length:in_dim*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_o = [device newBufferWithLength:out_dim*sizeof(float) options:MTLResourceStorageModeShared];

        // Dispatch — EXACTLY like engine's gpu_encode_batch_matvec
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:wf_buf offset:w_off atIndex:0];
        [enc setBuffer:wf_buf offset:s_off atIndex:1];
        [enc setBuffer:wf_buf offset:b_off atIndex:2];
        [enc setBuffer:buf_x  offset:0     atIndex:3];
        [enc setBuffer:buf_o  offset:0     atIndex:4];
        uint32_t o = (uint32_t)out_dim, i = (uint32_t)in_dim, g = (uint32_t)group_size;
        [enc setBytes:&o length:4 atIndex:5];
        [enc setBytes:&i length:4 atIndex:6];
        [enc setBytes:&g length:4 atIndex:7];
        uint32_t tgs = (o + 7) / 8;
        [enc dispatchThreadgroups:MTLSizeMake(tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *gpu_out = malloc(out_dim * sizeof(float));
        memcpy(gpu_out, [buf_o contents], out_dim * sizeof(float));

        // CPU reference
        uint32_t *W = (uint32_t*)((char*)mmap_data + w_off);
        uint16_t *S = (uint16_t*)((char*)mmap_data + s_off);
        uint16_t *B = (uint16_t*)((char*)mmap_data + b_off);
        float *cpu_out = malloc(out_dim * sizeof(float));
        cpu_dequant_matvec(W, S, B, input, cpu_out, out_dim, in_dim, group_size);

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
        double cos_sim = dot / sqrt(n1 * n2);
        printf("  CosSim=%.8f MaxDiff=%.2e nan_count=%d  %s\n",
               cos_sim, maxd, nan_count,
               (cos_sim > 0.999 && nan_count == 0) ? "PASS" : "FAIL");

        if (nan_count > 0) {
            printf("  First 5 GPU outputs: ");
            for (int j = 0; j < 5; j++) printf("%.4f ", gpu_out[j]);
            printf("\n  First 5 CPU outputs: ");
            for (int j = 0; j < 5; j++) printf("%.4f ", cpu_out[j]);
            printf("\n");
        }

        free(input); free(cpu_out); free(gpu_out);
        // buf_x and buf_o auto-released
    }

    munmap(mmap_data, st.st_size);
    close(fd);
    return 0;
}

// Test matvec_fast kernel (used by o_proj 4-bit path)
int main2() {
    // ... reuse Metal setup from main
    return 0;
}

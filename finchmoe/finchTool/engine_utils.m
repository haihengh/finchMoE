/*
 * engine_utils.m — Shared Metal engine primitives for finchTool diagnostics.
 *
 * Extracted and adapted from infer.m. Provides minimal Metal setup,
 * CPU reference functions, and expert data loading utilities.
 */

#import "engine_utils.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <fcntl.h>
#import <unistd.h>
#import <sys/stat.h>
#import <sys/mman.h>

// ============================================================================
// CPU reference functions (extracted from infer.m)
// ============================================================================

void cpu_dequant_matvec(
    const uint32_t *W, const uint16_t *scales, const uint16_t *biases,
    const float *x, float *out,
    int out_dim, int in_dim, int group_size, int bits)
{
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
    int vals_per_u32 = 32 / bits;
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

void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    for (int i = 0; i < dim; i++) {
        float g = gate[i];
        // SiLU(g) * up = g * sigmoid(g) * up
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

float vec_rms(const float *v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)v[i] * (double)v[i];
    }
    return (float)sqrt(sum / n);
}

void print_first_n(const char *label, const float *v, int n) {
    fprintf(stderr, "  %s first%d: [", label, n);
    for (int i = 0; i < n; i++) {
        fprintf(stderr, "%.4f%s", v[i], i < n - 1 ? ", " : "");
    }
    fprintf(stderr, "]\n");
}

// ============================================================================
// Metal setup
// ============================================================================

DiagMetalCtx *diag_metal_init(void) {
    DiagMetalCtx *ctx = calloc(1, sizeof(DiagMetalCtx));
    if (!ctx) return NULL;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) {
        fprintf(stderr, "[diag] ERROR: Metal is not supported on this device\n");
        free(ctx);
        return NULL;
    }
    fprintf(stderr, "[diag] Device: %s\n", [[ctx->device name] UTF8String]);

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) {
        fprintf(stderr, "[diag] ERROR: Failed to create command queue\n");
        free(ctx);
        return NULL;
    }

    // Load and compile shaders from source (runtime compilation, same as infer.m)
    NSArray *paths = @[@"shaders.metal", @"../shaders.metal", @"metal_infer/shaders.metal"];
    NSString *src = nil;
    for (NSString *p in paths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
            src = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding error:nil];
            if (src) {
                fprintf(stderr, "[diag] Loaded shaders from %s\n", [p UTF8String]);
                break;
            }
        }
    }
    if (!src) {
        fprintf(stderr, "[diag] ERROR: Cannot find shaders.metal\n");
        diag_metal_free(ctx);
        return NULL;
    }

    MTLCompileOptions *opts = [MTLCompileOptions new];
    opts.languageVersion = MTLLanguageVersion3_1;
    NSError *error = nil;
    ctx->library = [ctx->device newLibraryWithSource:src options:opts error:&error];
    if (!ctx->library) {
        fprintf(stderr, "[diag] ERROR: Shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        diag_metal_free(ctx);
        return NULL;
    }
    fprintf(stderr, "[diag] Shader library compiled successfully\n");

    // Helper to create a compute pipeline
    id<MTLComputePipelineState> (^makePipe)(NSString *) = ^(NSString *name) {
        id<MTLFunction> fn = [ctx->library newFunctionWithName:name];
        if (!fn) {
            fprintf(stderr, "[diag] WARNING: Function '%s' not found in shader library\n",
                    [name UTF8String]);
            return (id<MTLComputePipelineState>)nil;
        }
        NSError *err = nil;
        id<MTLComputePipelineState> pipe =
            [ctx->device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            fprintf(stderr, "[diag] WARNING: Pipeline '%s' failed: %s\n",
                    [name UTF8String], [[err localizedDescription] UTF8String]);
        }
        return pipe;
    };

    ctx->matvec_v3                = makePipe(@"dequant_matvec_4bit_v3");
    ctx->matvec_8bit              = makePipe(@"dequant_matvec_8bit");
    ctx->matvec_2bit              = makePipe(@"dequant_matvec_2bit");
    ctx->matvec_1bit              = makePipe(@"dequant_matvec_1bit");
    ctx->fused_gate_up_swiglu     = makePipe(@"fused_gate_up_swiglu");
    ctx->fused_gate_up_swiglu_8bit = makePipe(@"fused_gate_up_swiglu_8bit");
    ctx->fused_gate_up_swiglu_2x  = makePipe(@"fused_gate_up_swiglu_2x");
    ctx->swiglu                   = makePipe(@"swiglu_fused");
    ctx->moe_combine_residual     = makePipe(@"moe_combine_residual");
    ctx->gemv_bf16                = makePipe(@"gemv_bf16");
    ctx->gemv_bf16_x2             = makePipe(@"gemv_bf16_x2");

    // Critical check: matvec_v3 is required for 4-bit diagnostics
    if (!ctx->matvec_v3) {
        fprintf(stderr, "[diag] ERROR: matvec_v3 pipeline required for 4-bit tests\n");
        diag_metal_free(ctx);
        return NULL;
    }

    // Create sync primitives
    ctx->sync_fence = [ctx->device newFence];
    ctx->sync_event = [ctx->device newSharedEvent];
    ctx->sync_value = 0;

    // Allocate initial buffers
    diag_metal_alloc_buffers(ctx);

    fprintf(stderr, "[diag] Metal initialization complete\n");
    return ctx;
}

void diag_metal_free(DiagMetalCtx *ctx) {
    if (!ctx) return;
    ctx->buf_input     = nil;
    ctx->buf_expert_data = nil;
    ctx->buf_gate      = nil;
    ctx->buf_up        = nil;
    ctx->buf_act       = nil;
    ctx->buf_out       = nil;
    ctx->sync_fence    = nil;
    ctx->sync_event    = nil;
    ctx->library       = nil;
    ctx->queue         = nil;
    ctx->device        = nil;
    free(ctx);
}

void diag_metal_alloc_buffers(DiagMetalCtx *ctx) {
    if (!ctx || !ctx->device) return;

    // Free previous allocations
    ctx->buf_input       = nil;
    ctx->buf_expert_data = nil;
    ctx->buf_gate        = nil;
    ctx->buf_up          = nil;
    ctx->buf_act         = nil;
    ctx->buf_out         = nil;

    NSUInteger opts = MTLResourceStorageModeShared;
    ctx->buf_input       = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float) options:opts];
    ctx->buf_expert_data = [ctx->device newBufferWithLength:EXPERT_SIZE_MAX options:opts];
    ctx->buf_gate        = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float) options:opts];
    ctx->buf_up          = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float) options:opts];
    ctx->buf_act         = [ctx->device newBufferWithLength:MOE_INTERMEDIATE * sizeof(float) options:opts];
    ctx->buf_out         = [ctx->device newBufferWithLength:HIDDEN_DIM * sizeof(float) options:opts];
}

// ============================================================================
// Expert data loading
// ============================================================================

void *diag_load_expert_data(const char *layer_file_path, int expert_idx, int bits) {
    if (!layer_file_path) return NULL;

    int fd = open(layer_file_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[diag] ERROR: Cannot open expert file: %s\n", layer_file_path);
        return NULL;
    }

    // Disable read-ahead for random access
    fcntl(fd, F_RDAHEAD, 0);

    size_t esz = active_expert_size(bits);
    off_t offset = (off_t)expert_idx * esz;

    void *data = malloc(esz);
    if (!data) {
        fprintf(stderr, "[diag] ERROR: malloc(%zu) failed\n", esz);
        close(fd);
        return NULL;
    }

    ssize_t n = pread(fd, data, esz, offset);
    close(fd);

    if (n != (ssize_t)esz) {
        fprintf(stderr, "[diag] ERROR: pread expert %d: %zd/%zu bytes\n",
                expert_idx, n, esz);
        free(data);
        return NULL;
    }

    return data;
}

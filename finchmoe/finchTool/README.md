# finchTool — Metal Engine Verification & Diagnostic Suite

A standalone diagnostic tool for validating custom Metal-based LLM/MoE inference engines on Apple Silicon. Runs kernel isolation tests, pipeline synchronization audits, and tensor parity checks independently of the production generation loop, while sharing the exact same `shaders.metal` kernels.

## Why finchTool Exists

During development of the fused expert kernel, we spent dozens of iterations manually:
1. Adding inline `fprintf` diagnostic code to `infer.m`
2. Rebuilding, running, checking output
3. Disabling diagnostic code with `#if 0`, leaving dead code
4. Repeating

This process was slow, error-prone, and left ~150 lines of disabled diagnostic scaffolding in the production engine. More critically, a **SiLU formula regression** (`vg*vg` instead of `vg`) went undetected for hours because the diagnostic code computing the CPU reference had the **same bug** — both sides of the comparison were equally wrong, producing a false "bit-identical" verdict.

`finchTool` replaces this manual loop with structured, reproducible, independently-verified tests.

## Quick Start

```bash
cd finchmoe/finchTool
make                          # Build
./finchTool kernel --test all  # Run all kernel tests
```

**Prerequisites**: macOS with Metal support (Apple Silicon), `shaders.metal` in the parent directory.

## CLI Reference

### Kernel Isolation Tests

Test individual Metal shaders with fresh buffers and synthetic test vectors. Each test compares GPU output against an independent CPU reference implementation.

```bash
# Test the fused gate+up+SiLU kernel against the non-fused path
./finchTool kernel --test fused_mlp --quant 4bit

# Test with verbose per-element output
./finchTool kernel --test fused_mlp --quant 4bit --verbose

# Test dequant matvec kernels at various bit widths
./finchTool kernel --test matvec --quant 4bit
./finchTool kernel --test matvec --quant 8bit
./finchTool kernel --test matvec --quant 2bit
./finchTool kernel --test matvec --quant 1bit

# Test SwiGLU activation
./finchTool kernel --test swiglu

# Run all kernel tests
./finchTool kernel --test all
```

**Available tests**:

| Test | What it validates | Kernels used |
|------|-------------------|-------------|
| `fused_mlp` | Fused gate+up+SiLU vs non-fused (gate→up→swiglu) | `fused_gate_up_swiglu`, `dequant_matvec_4bit_v3`, `swiglu_fused` |
| `matvec` | Dequant matvec at specified bit width | `dequant_matvec_4bit_v3`, `_8bit`, `_2bit`, `_1bit` |
| `swiglu` | SiLU activation gate×sigmoid(gate)×up | `swiglu_fused` |

### Pipeline Synchronization Audit

Tests inter-command-buffer memory visibility to detect GPU cache coherency issues (L2/SLC staleness on Apple Silicon).

```bash
./finchTool pipeline --test inter-cb-sync --quant 4bit
```

The test runs the fused kernel and a downstream consumer across 4 synchronization configurations:

| Step | Configuration | What it tests |
|------|---------------|---------------|
| 1 | Single CB (golden) | Baseline: both dispatches on one command buffer |
| 2 | Separate CBs + `waitUntilCompleted` | CPU-side sync only, no GPU barrier |
| 3 | Separate CBs + `MTLFence` | Encoder-level GPU fence |
| 4 | Separate CBs + `MTLSharedEvent` | Command-buffer-level GPU event |

If Step 1 passes but Steps 2-4 fail, an **L2/SLC cache incoherency** is confirmed — the exact bug discovered during fused kernel integration. Resolution requires `MTLBlitCommandEncoder::synchronizeResource:` or investigation with the Xcode GPU frame debugger.

### Tensor Parity Check

Compare two binary float32 tensor files (e.g., GPU output vs PyTorch/MLX reference).

```bash
./finchTool parity --a output_gpu.bin --b output_cpu.bin
./finchTool parity --a layer7_act.bin --b golden/layer7_act.bin --tolerance 1e-4
```

### Common Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--quant, -q 4bit` | `4bit` | Quantization: `4bit`, `8bit`, `2bit`, `1bit` |
| `--dim-in, -i 2048` | `2048` | Input dimension for matvec/fused_mlp |
| `--dim-out, -o 512` | `512` | Output dimension for matvec/fused_mlp |
| `--dim, -d 512` | `512` | Dimension for swiglu test |
| `--tolerance 1e-3` | `1e-3` | Per-element diff threshold for first-bad-index |
| `--verbose, -v` | off | Print first-10 values of each tensor |
| `--help, -h` | — | Show usage |

## Interpreting Results

### Parity Report Format

```
========================================================
 DIAGNOSTIC REPORT: Fused GPU vs CPU (SwiGLU act)
========================================================
 Status             : PASS (exact)
 Cosine Similarity  : 1.00000000
 Max Abs Difference : 4.7684e-07
 Avg Abs Difference : 2.0614e-08
========================================================
```

### Parity Status Levels

| Status | CosSim | MaxDiff | Meaning |
|--------|--------|---------|---------|
| `PASS (exact)` | ≥ 0.999999 | < 1e-5 | Bit-identical or trivial rounding |
| `PASS (acceptable)` | ≥ 0.999 | < 1e-2 | FP accumulation order reordering |
| `WARN (degraded)` | ≥ 0.98 | < 1e-1 | Quantization or precision loss |
| `FAIL` | < 0.98 | any | Math bug, memory hazard, or corrupt data |

### What "acceptable" difference means

FP addition is commutative but NOT associative. Different accumulation orders (tile sizes, reduction tree topologies, thread mappings) produce slightly different rounding. A `MaxDiff` of 1e-4 with `CosSim` of 1.0 is normal and harmless — it will not cause cascading drift across 40 layers.

## Architecture

```
finchmoe/
├── finchTool/
│   ├── main.m                   # CLI entry + all test implementations
│   ├── engine_utils.h/m         # Shared Metal setup, CPU reference, expert I/O
│   ├── verify_core.h/m          # ParityReport, metrics, tensor serialization
│   ├── Makefile                 # Build: make, make test, make clean
│   └── README.md                # This file
└── shaders.metal                # Shared shader library (runtime-compiled)
```

### Design Principles

1. **Separate binary, shared shaders**: `finchTool` compiles independently but loads the identical `shaders.metal` at runtime. No dependency on `infer.m`'s full pipeline. A kernel that passes in `finchTool` uses the exact same GPU code as production.

2. **Synchronous by default**: All tests run synchronously (`commit` + `waitUntilCompleted`). This eliminates timing-dependent bugs and makes output deterministic. Pipeline tests explicitly introduce async patterns to test synchronization.

3. **Fresh buffers per test**: Each test allocates its own Metal buffers. No buffer recycling eliminates stale-data hazards from the diagnostic tool itself.

4. **CPU reference is the source of truth**: Every GPU kernel test computes an independent CPU reference using the same mathematical formula. No golden files needed — the CPU is the oracle. This prevents the "both sides equally wrong" class of bugs.

5. **Binary export for cross-validation**: `dump_tensor_f32()` writes raw float32 arrays for comparison with Python, PyTorch, or MLX reference implementations.

### Module Design

#### `engine_utils` — Shared Engine Helpers

Extracted from `infer.m` (lines 509, 967, 1032, 2875) into a standalone module.

| Component | Source | Purpose |
|-----------|--------|---------|
| `DiagMetalCtx` | Adapted from `MetalCtx` | Minimal Metal setup: device, queue, shader library, key pipelines, diagnostic buffers |
| `bf16_to_f32()` | `infer.m:509` | BF16 → float32 conversion |
| `cpu_dequant_matvec()` | `infer.m:967` | CPU reference for 1/2/4/8-bit + BF16 matvec |
| `cpu_swiglu()` | `infer.m:1032` | CPU reference for SiLU activation |
| `vec_rms()` | `infer.m:2875` | RMS sanity metric |
| `diag_load_expert_data()` | New | `pread` a single expert from a layer file for model-dependent tests |

Shader loading mirrors `infer.m` exactly: runtime source compilation via `newLibraryWithSource:` with `MTLMathModeFast` and `MTLLanguageVersion3_1`.

#### `verify_core` — Validation Engine Core

Standardized numerical comparison with automatic parity classification.

```c
typedef struct {
    float max_diff;         // L-infinity: max |a[i] - b[i]|
    float avg_diff;         // Mean absolute difference
    float cos_sim;          // Cosine similarity
    float rmse;             // Relative mean squared error
    size_t first_bad_index; // First element exceeding tolerance
    float val_a_at_bad;     // Reference value at first-bad
    float val_b_at_bad;     // Test value at first-bad
    bool passed;            // PARITY_EXACT or PARITY_ACCEPTABLE
    ParityStatus status;    // Enum: EXACT, ACCEPTABLE, DEGRADED, FAILED
} ParityReport;
```

#### `main.m` — Test Implementations

Each test follows the same pattern:
1. Generate deterministic synthetic inputs (`srand(seed)`)
2. Copy inputs to Metal buffers
3. Run GPU kernel(s) on their own command buffer, commit, wait
4. Read back GPU results
5. Compute CPU reference independently
6. Evaluate parity and print report

## Real-World Usage: Detecting the SiLU Regression

The most impactful use of `finchTool` to date was catching a regression in the fused kernel's SwiGLU activation:

**Original code** (correct):
```metal
out[tgid] = (vg/(1.0f+exp(-vg))) * vu;  // SiLU(gate) * up
```

**"Fix" that was a regression** (wrong):
```metal
out[tgid] = (vg*vg/(1.0f+exp(-vg))) * vu;  // gate * SiLU(gate) * up — EXTRA vg!
```

The diagnostic code in `infer.m` had the **same bug** (`g*g` instead of `g`), so it reported "bit-identical" — both GPU and CPU reference were equally wrong. `finchTool`'s independent `cpu_swiglu()` implementation caught the 100× magnitude discrepancy immediately:

```
Fused GPU vs CPU: CosSim=0.95, MaxDiff=7.6e+06, Status=FAIL
  cpu_act rms=5347  f_act rms=711495  (133× too large!)
```

## Adding New Tests

Tests follow a standard pattern. To add a new kernel test:

```c
static int test_my_kernel(DiagMetalCtx *ctx, bool verbose) {
    // 1. Check pipeline availability
    if (!ctx->my_pipeline) { /* SKIP */ return 0; }

    // 2. Generate deterministic test data
    srand(FIXED_SEED);
    float *input = malloc(dim * sizeof(float));
    // ... fill with random data ...

    // 3. CPU reference
    float *cpu_out = malloc(dim * sizeof(float));
    my_cpu_reference(input, cpu_out, dim);

    // 4. GPU execution
    memcpy([ctx->buf_input contents], input, dim * sizeof(float));
    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->my_pipeline];
    // ... bind buffers, set bytes, dispatch ...
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    // 5. Read back and compare
    float *gpu_out = malloc(dim * sizeof(float));
    memcpy(gpu_out, [ctx->buf_out contents], dim * sizeof(float));
    ParityReport r = evaluate_parity_f32(cpu_out, gpu_out, dim, 1e-3f);
    print_parity_report("MyKernel GPU vs CPU", &r);

    // 6. Cleanup
    free(input); free(cpu_out); free(gpu_out);
    return r.passed ? 0 : 1;
}
```

Then register it in `main()` under the `kernel` subcommand.

## Integration with infer.m

`finchTool` extracts and standardizes patterns that were ad-hoc in `infer.m`:

| `infer.m` pattern | `finchTool` equivalent |
|-------------------|------------------------|
| `--compare-experts N` flag | `./finchTool kernel --test fused_mlp` (runs on synthetic data, no model needed) |
| `DIAG-FULL` inline code (`#if 0`) | `ParityReport` struct + `print_parity_report()` |
| `fprintf(stderr, "max_diff=...")` scattered prints | Structured `ParityReport` with automatic classification |
| `--dump-logits F` flag | `dump_tensor_f32()` / `load_tensor_f32()` |
| Manual `vec_rms()` checks | `ParityReport.cos_sim` + automatic threshold comparison |

The vision is that `infer.m` eventually delegates all diagnostic checks to `finchTool`'s libraries, keeping the production engine clean of debugging scaffolding.

## Future Extensions

- **Layer boundary sanity runner**: Run a single-token forward pass through all 40 layers, dump activations at checkpoints, compare against golden reference
- **SIMD reduction auditor**: Test `simd_sum` with varying active lane counts (1..32) to detect divergence UB
- **RoPE rotation test**: Compare half-split vs interleaved implementations
- **Gated DeltaNet test**: Validate the GDN recurrence kernel against CPU reference
- **CI integration**: `make test` as a pre-commit hook to catch regressions before they reach production

## tools/ — Python & Shell Diagnostics

Standalone analysis and debug scripts (consolidated from the engine's root
directory 2026-08-18). Run from anywhere; they read `/tmp` dumps produced by
the engine's env-gated debug flags and model files by absolute path.

**Logits / trace comparison**
- `compare_gguf_logits.py ref.bin new.bin` — cos/argmax/max-diff of `-I`
  logit dumps (used by `../bench_gguf.sh`)
- `compare_pb_traces.py pb_ref.bin pb_new.bin` — first-diverging
  (token, layer) on the Phase-B traces (`FINCHMOE_DUMP_PHASEB`)
- `analyze_s4_dumps.py` — Phase C S4 kernel-parity dumps

**Long-generation health**
- `analyze_longgen.py logfile` — n-gram repetition, EOS, think-tag
  balance, topical drift (Bug 15 toolkit)

**Quantization quality**
- `quant_audit.py` — per-tensor CosSim audit vs pristine BF16 (259
  non-expert tensors + 3/4/8-bit expert packs, per-role aggregates).
  Ships crash guardrails from the 2026-08-20 kernel panic: shared
  heavy-job lock with `repack_experts.py`, memory guards, streamed
  expert refs (peak ~1.7 GB).

**Session debug one-offs** (kept for provenance; each reads its own
`FINCHMOE_*_DBG` dumps): `debug_2token_gdn.py`, `debug_bf16_vs_4bit.py`,
`debug_compare.py`, `debug_e2e_logits.py`, `debug_full_attn_moe.py`,
`debug_full_forward.py`, `debug_gdn_compare.py`, `debug_gdn_reference.py`,
`debug_layer_compare.py`, `debug_layer_diff.py`, `debug_mlx_inference.py`

**MTP / rebuild / wobble**
- `extract_mtp_experts.py`, `mtp_reference.py` — MTP head extraction +
  numpy reference
- `verify_clean_rebuild.py` — single-stage rebuild verification
- `wobble_hunt.sh`, `wobble_trace_hunt.sh` — run-to-run wobble hunting

The build/quant pipeline scripts (`extract_weights.py`,
`generate_expert_index.py`, `repack_experts.py`, `compress_experts.py`,
`export_tokenizer.py`, `quantize_model.py`, `quantize_non_experts.py`,
`build_hot_sets.py`) and the bench runners (`bench_*.sh`,
`run_logit_dump_safe.sh`) stay in the engine root — the Makefile targets
and the bench scripts' relative paths depend on that location.

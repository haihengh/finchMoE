/*
 * verify_core.h — Validation Engine Core (VEC) for finchTool.
 *
 * Standardized numerical comparison between tensor outputs.
 * All diagnostic modules use this for consistent, structured reporting.
 */

#ifndef VERIFY_CORE_H
#define VERIFY_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    PARITY_EXACT       = 0,  // CosSim >= 0.999999, MaxDiff < 1e-5
    PARITY_ACCEPTABLE  = 1,  // CosSim >= 0.999, MaxDiff < 1e-2  (FP reordering)
    PARITY_DEGRADED    = 2,  // CosSim >= 0.98, MaxDiff < 1e-1   (quant loss)
    PARITY_FAILED      = 3,  // CosSim < 0.98                     (math bug/hazard)
} ParityStatus;

typedef struct {
    // Core metrics
    float max_diff;         // L-infinity: max absolute difference
    float avg_diff;         // mean absolute difference
    float cos_sim;          // cosine similarity
    float rmse;             // relative mean squared error

    // First-bad-index tracking
    size_t first_bad_index;
    float  val_a_at_bad;    // value from reference tensor A
    float  val_b_at_bad;    // value from test tensor B

    // Status flags
    bool         passed;
    ParityStatus status;
    const char  *status_str;  // human-readable status string
} ParityReport;

// ---- Core evaluation functions ----

// Compare two float32 tensors element-by-element.
// `a` is the reference (e.g., CPU golden), `b` is the test (e.g., GPU output).
// `tolerance` is the per-element diff threshold for first_bad_index tracking.
ParityReport evaluate_parity_f32(const float *a, const float *b,
                                  size_t length, float tolerance);

// Compare two BF16 tensors (converts to float32 internally).
ParityReport evaluate_parity_bf16(const uint16_t *a, const uint16_t *b,
                                   size_t length, float tolerance);

// ---- Reporting ----

// Print a formatted parity report to stderr.
void print_parity_report(const char *test_name, const ParityReport *report);

// Print a compact one-line summary for batch test runs.
void print_parity_summary(const char *test_name, const ParityReport *report);

// ---- Serialization ----

// Dump a float32 tensor to a binary file for cross-validation.
// Format: raw little-endian float32 array, `length` elements.
// Returns 0 on success, -1 on failure.
int dump_tensor_f32(const char *path, const float *data, size_t length);

// Load a binary float32 tensor file. Returns malloc'd buffer (caller frees).
float *load_tensor_f32(const char *path, size_t *length_out);

#endif // VERIFY_CORE_H

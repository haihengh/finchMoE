/*
 * verify_core.m — Validation Engine Core (VEC) implementation.
 *
 * Provides standardized numerical comparison between tensor outputs
 * with automatic parity classification and formatted reporting.
 */

#import "verify_core.h"
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

// ============================================================================
// Core evaluation
// ============================================================================

static ParityStatus classify_parity(float cos_sim, float max_diff) {
    if (cos_sim >= 0.999999f && max_diff < 1e-5f)  return PARITY_EXACT;
    if (cos_sim >= 0.999f    && max_diff < 1e-2f)  return PARITY_ACCEPTABLE;
    if (cos_sim >= 0.98f     && max_diff < 1e-1f)  return PARITY_DEGRADED;
    return PARITY_FAILED;
}

static const char *status_string(ParityStatus s) {
    switch (s) {
        case PARITY_EXACT:      return "PASS (exact)";
        case PARITY_ACCEPTABLE: return "PASS (acceptable)";
        case PARITY_DEGRADED:   return "WARN (degraded)";
        case PARITY_FAILED:     return "FAIL";
        default:                return "UNKNOWN";
    }
}

ParityReport evaluate_parity_f32(const float *a, const float *b,
                                  size_t length, float tolerance) {
    ParityReport r = {0};
    r.max_diff = 0.0f;
    r.first_bad_index = (size_t)-1;

    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    double sum_diff = 0.0;

    for (size_t i = 0; i < length; i++) {
        float va = a[i];
        float vb = b[i];
        float diff = fabsf(va - vb);

        sum_diff += (double)diff;
        if (diff > r.max_diff) {
            r.max_diff = diff;
        }

        if (diff > tolerance && r.first_bad_index == (size_t)-1) {
            r.first_bad_index = i;
            r.val_a_at_bad = va;
            r.val_b_at_bad = vb;
        }

        dot_product += (double)va * (double)vb;
        norm_a += (double)va * (double)va;
        norm_b += (double)vb * (double)vb;
    }

    r.avg_diff = (float)(sum_diff / (double)length);
    r.rmse = r.avg_diff;  // simplified: mean absolute diff

    double na = sqrt(norm_a);
    double nb = sqrt(norm_b);
    if (na > 0.0 && nb > 0.0) {
        r.cos_sim = (float)(dot_product / (na * nb));
    } else {
        r.cos_sim = (na == 0.0 && nb == 0.0) ? 1.0f : 0.0f;
    }

    r.status = classify_parity(r.cos_sim, r.max_diff);
    r.status_str = status_string(r.status);
    r.passed = (r.status <= PARITY_ACCEPTABLE);

    return r;
}

ParityReport evaluate_parity_bf16(const uint16_t *a, const uint16_t *b,
                                   size_t length, float tolerance) {
    // Convert BF16 to float32 for comparison
    float *fa = (float *)malloc(length * sizeof(float));
    float *fb = (float *)malloc(length * sizeof(float));
    if (!fa || !fb) {
        ParityReport r = {0};
        r.status = PARITY_FAILED;
        r.status_str = "FAIL (malloc)";
        free(fa); free(fb);
        return r;
    }

    for (size_t i = 0; i < length; i++) {
        uint32_t bits_a = (uint32_t)a[i] << 16;
        uint32_t bits_b = (uint32_t)b[i] << 16;
        memcpy(&fa[i], &bits_a, sizeof(float));
        memcpy(&fb[i], &bits_b, sizeof(float));
    }

    ParityReport r = evaluate_parity_f32(fa, fb, length, tolerance);
    free(fa); free(fb);
    return r;
}

// ============================================================================
// Reporting
// ============================================================================

void print_parity_report(const char *test_name, const ParityReport *report) {
    fprintf(stderr, "\n");
    fprintf(stderr, "========================================================\n");
    fprintf(stderr, " DIAGNOSTIC REPORT: %s\n", test_name);
    fprintf(stderr, "========================================================\n");
    fprintf(stderr, " Status             : %s\n", report->status_str);
    fprintf(stderr, " Cosine Similarity  : %.8f\n", report->cos_sim);
    fprintf(stderr, " Max Abs Difference : %.4e\n", report->max_diff);
    fprintf(stderr, " Avg Abs Difference : %.4e\n", report->avg_diff);
    if (!report->passed && report->first_bad_index != (size_t)-1) {
        fprintf(stderr, " First Mismatch @%zu  ref=%.8f  test=%.8f  diff=%.4e\n",
                report->first_bad_index,
                report->val_a_at_bad, report->val_b_at_bad,
                fabsf(report->val_a_at_bad - report->val_b_at_bad));
    }
    fprintf(stderr, "========================================================\n");
}

void print_parity_summary(const char *test_name, const ParityReport *report) {
    const char *icon = report->passed ? "OK" :
                       (report->status == PARITY_DEGRADED ? "WARN" : "FAIL");
    fprintf(stderr, "  [%s] %-40s  cos=%.7f  max_diff=%.2e\n",
            icon, test_name, report->cos_sim, report->max_diff);
}

// ============================================================================
// Serialization
// ============================================================================

int dump_tensor_f32(const char *path, const float *data, size_t length) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[verify] ERROR: Cannot open %s for writing\n", path);
        return -1;
    }
    size_t nw = fwrite(data, sizeof(float), length, f);
    fclose(f);
    if (nw != length) {
        fprintf(stderr, "[verify] ERROR: Short write to %s: %zu/%zu\n",
                path, nw, length);
        return -1;
    }
    fprintf(stderr, "[verify] Dumped %zu floats (%.1f KB) -> %s\n",
            length, (double)(length * sizeof(float)) / 1024.0, path);
    return 0;
}

float *load_tensor_f32(const char *path, size_t *length_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[verify] ERROR: Cannot open %s for reading\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % sizeof(float) != 0) {
        fprintf(stderr, "[verify] ERROR: Invalid file size %ld for %s\n", sz, path);
        fclose(f);
        return NULL;
    }
    size_t length = sz / sizeof(float);
    float *data = (float *)malloc(sz);
    if (!data) {
        fclose(f);
        return NULL;
    }
    size_t nr = fread(data, sizeof(float), length, f);
    fclose(f);
    if (nr != length) {
        fprintf(stderr, "[verify] ERROR: Short read from %s: %zu/%zu\n",
                path, nr, length);
        free(data);
        return NULL;
    }
    if (length_out) *length_out = length;
    return data;
}

// test/engine_test.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Smoke test: open a .finchmoe file, check dense tensors, acquire one expert.
// ─────────────────────────────────────────────────────────────────────────────

#include "engine_core.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: finchmoe_test <model.finchmoe>\n");
        return 1;
    }

    finchmoe::EngineConfig cfg;
    cfg.verbose          = true;
    cfg.memory_budget_bytes = finchmoe::BUDGET_4GB;
    cfg.soft_budget_bytes   = finchmoe::BUDGET_3_5GB;
    cfg.io_threads          = 4;

    finchmoe::EngineCore engine;
    auto s = engine.init(argv[1], cfg);
    if (!finchmoe::ok(s)) {
        fprintf(stderr, "init failed: %s\n", finchmoe::status_str(s));
        return 1;
    }

    const auto& c = engine.config();
    printf("Model: %u layers  %u experts/layer  top-%u active\n",
           c.num_hidden_layers, c.num_experts, c.num_experts_per_tok);
    printf("Active bytes/token: %.2f GB\n",
           (double)c.active_bytes_per_token() / 1e9);

    // Verify embed_tokens is loaded
    auto tv = engine.dense("model.embed_tokens.weight");
    if (!tv.data) {
        fprintf(stderr, "FAIL: embed_tokens not found\n");
        return 1;
    }
    printf("embed_tokens: [%d, %d]  dtype=%s\n",
           tv.shape[0], tv.shape[1], finchmoe::dtype_name(tv.dtype));

    // Acquire expert (layer=0, expert=0)
    finchmoe::Status es;
    auto ev = engine.acquire_expert(0, 0, &es);
    if (!finchmoe::ok(es)) {
        fprintf(stderr, "FAIL: acquire_expert(0,0): %s\n", finchmoe::status_str(es));
        return 1;
    }
    printf("Expert (0,0) gate_proj: [%d, %d]  data=%p\n",
           ev.gate_proj.shape[0], ev.gate_proj.shape[1], ev.gate_proj.data);
    engine.release_expert(0, 0);

    engine.print_stats();
    printf("PASS\n");
    return 0;
}

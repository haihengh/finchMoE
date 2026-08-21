# HumanEval on RTX 3090: Qwen3.8-27B vs Qwen3.6-35B-A3B

Benchmark of two Qwen models served locally with llama.cpp on a single NVIDIA RTX 3090 (24 GB).
Results generated 2026-08-20.

## Hardware and software

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, 24 GB VRAM, driver 610.47 |
| llama.cpp | built from source at commit `a30273376` (CUDA backend, sm_86, MSVC/VS2026) |
| Server | `llama-server` from that build |
| Harness | [EvalPlus](https://github.com/evalplus/evalplus) 0.3.1 (HumanEval + HumanEval+) |
| Host | Windows 11, Python 3.12 |

## Models under test

| Model | File | Quant | Architecture |
|---|---|---|---|
| Qwen3.8-27B | `Qwen3.8-27B-Q4_K_M.gguf` (15.66 GB) | Q4_K_M | `qwen35` hybrid: 64 layers = 48 gated delta-net + 16 full attention + 1 MTP head; native ctx 262144 |
| Qwen3.6-35B-A3B | `Qwen3.6-35B-A3B-Q4_K_M.gguf` (19.71 GB) | Q4_K_M | `qwen35moe` MoE: 256 experts / 8 active, 40 layers = 30 delta-net + 10 full attention; native ctx 262144 |

Both from `lmstudio-community` on the local LM Studio models dir.

## Protocol

- Dataset: HumanEval, all 164 problems
- Decoding: greedy (temperature 0), 1 sample per problem, max 512 new tokens
- Thinking disabled on both models (`--reasoning off`) for a fair classic-completion comparison
- Server: full GPU offload (`-ngl 99`), flash attention on, KV cache q8_0
  - 27B: `-c 131072` + mmproj loaded
  - 35B: `-c 32768`, no mmproj (context length irrelevant for this benchmark)
- Generation: `evalplus.codegen` (backend `openai`) against the local server's OpenAI-compatible endpoint
- Evaluation: `evalplus.evaluate` with `--dataset humaneval` (base tests + HumanEval+ extra tests)
- Metric: pass@1, as reported by the harness

## Results

| Model | HumanEval base pass@1 | HumanEval+ pass@1 | Passed (base / plus) |
|---|---|---|---|
| **Qwen3.8-27B** (dense) | **93.3%** | **90.9%** | 153 / 149 of 164 |
| **Qwen3.6-35B-A3B** (MoE) | **91.5%** | **89.0%** | 150 / 147 of 164 |

Measured generation speed during the run (average over all 164 requests):

| Model | Prompt processing | Decode |
|---|---|---|
| Qwen3.8-27B | 478 t/s | 38 t/s |
| Qwen3.6-35B-A3B | 946 t/s | 136 t/s |

## Takeaways

- The 27B dense model scores ~2 points higher on both suites: all 27B parameters run on
  every token, while the MoE activates only ~3B per token. Dense capacity per token wins
  on hard algorithmic problems.
- The 35B-A3B is 2x faster on prompt processing and 3.6x faster on decode (136 t/s feels
  instant), and its hybrid design makes the KV cache cheap enough for the full native
  256K context (with q4_0 KV) on 24 GB.
- Trade: quality-critical coding/agents -> 27B dense; throughput/long-context serving -> 35B-A3B.

## VRAM sizing notes (measured on this card)

KV cache memory per token (only full-attention layers allocate KV; delta-net layers use a
small fixed recurrent state):

| Model | full-attn layers | KV dims | f16 / q8_0 / q4_0 per token |
|---|---|---|---|
| Qwen3.8-27B | 16 | 4 heads x 256 = 1024 | 64 / 34 / 18 KB |
| Qwen3.6-35B-A3B | 10 | 2 heads x 256 = 512 | 20 / 10.6 / 5.6 KB |

Fit at 23.5 GB ceiling (weights + mmproj + ~0.8 GB overhead):

- 27B: 128K ctx q8_0 comfortable (128K is the practical ceiling; 256K needs q4_0 KV)
- 35B: 128K q8_0 comfortable, 192K q8_0 tight, full 256K with q4_0 KV (~22.8 GB)

## Files

```
humaneval_3090/
  README.md                                  this document
  humaneval_gen.py                           generation driver (Windows-safe evalplus codegen)
  results/
    qwen3.8-27b_openai_temp_0.0.jsonl        samples submitted for evaluation
    qwen3.8-27b_openai_temp_0.0.raw.jsonl    raw model outputs (before sanitization)
    qwen3.8-27b_openai_temp_0.0_eval_results.json   per-problem verdicts (base/plus)
    qwen3.6-35b-a3b_openai_temp_0.0.jsonl
    qwen3.6-35b-a3b_openai_temp_0.0.raw.jsonl
    qwen3.6-35b-a3b_openai_temp_0.0_eval_results.json
```

`eval_results.json` schema: `eval[task_id]` is a list of samples, each with fields
`task_id`, `solution`, `base_status` (`pass`/`fail`/`timeout`), `plus_status`,
`base_fail_tests`, `plus_fail_tests`.

## Reproduce

1. Build llama.cpp with CUDA (see `docs/build.md`):
   ```
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release --target llama-server -j
   ```
2. Start the server, e.g.:
   ```
   llama-server.exe -m <model.gguf> -c 32768 -ctk q8_0 -ctv q8_0 -fa on -ngl 99 --reasoning off --host 127.0.0.1 --port 8080
   ```
3. `pip install evalplus`
4. Apply the two Windows compatibility patches to `evalplus/eval/utils.py`
   (required; `time_limit()` uses Unix interval timers, `reliability_guard()` imports
   the Unix-only `resource` module):
   - `time_limit()`: no-op when `signal.setitimer`/`SIGALRM` are missing (the parent
     process enforces per-test timeouts via `Process.join()` + kill).
   - `reliability_guard()`: wrap the `import resource` / `setrlimit` block in
     `try/except ImportError`.
   - Generation also needs the SIGALRM-free request helper; `humaneval_gen.py` patches
     `evalplus.gen.util.openai_request.make_auto_request` at runtime, so no file edit
     is needed there.
5. Generate:
   ```
   python humaneval_gen.py <model-id>   # e.g. qwen3.8-27b
   ```
6. Evaluate:
   ```
   python -m evalplus.evaluate --dataset humaneval --samples evalplus_results/humaneval/<model-id>_openai_temp_0.0.jsonl --i-just-wanna-run
   ```

## Caveats

- Single greedy sample per problem (n=1). Standard leaderboard numbers use many samples
  with temperature 0.2-0.8, so these are slightly conservative.
- Thinking disabled; hybrid-thinking behavior may improve scores further.
- Contexts are tiny here, so KV cache quantization (q8_0) does not affect these results;
  this measures the Q4_K_M weights and the harness, not the KV tuning.
- EvalPlus 0.3.1; entry point is `evalplus.codegen` (not `evalplus.generate`).

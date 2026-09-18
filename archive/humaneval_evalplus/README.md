# HumanEval (EvalPlus) on FinchMoE: 3-bit / 4-bit / GGUF vs the 3090 llama.cpp numbers

> Root-cause investigation and experiment plan: `../finchmoe/HUMANEVAL_GAP_INVESTIGATION.md`.
> Headline: template and tokenizer both verified identical to the 3090's; the
> remaining suspects are decode accumulation, serve-mode state, and sampling.

Counterpart to `humaneval_3090/` (branch `3090`): the same EvalPlus 0.3.1 harness and
protocol, run against our engine on the M4 Mac mini. The 3090 run published
**91.5% base / 89.0% plus** for llama.cpp + Qwen3.6-35B-A3B Q4_K_M; our old homegrown
harness (`humaneval_m1/`) published 22.0% for what we believed was the same GGUF. This
run exists to measure the same thing the 3090 measured, with byte-identical tooling,
so the remaining gap is a real result instead of a harness artifact.

## The matrix

| # | Engine | Weights | Where | base / plus |
|---|---|---|---|---|
| A | llama.cpp (a30273376) | lmstudio-community Q4_K_M | RTX 3090 | 91.5% / 89.0% |
| B | FinchMoE | 3-bit native | this Mac | 13.4% / 12.8% |
| C | FinchMoE | 4-bit native | this Mac | 12.8% / 12.8% |
| D | FinchMoE | **our** Q4_K_M GGUF | this Mac | 12.8% / 12.8% |
| D′ | FinchMoE | **the 3090's exact GGUF** | this Mac | 11.6% / 11.6% |
| E1 | llama.cpp (CPU) | **the 3090's exact GGUF** | this Mac | **90.9% / 88.4%** |

## Protocol (identical to humaneval_3090)

- EvalPlus 0.3.1, `evalplus.codegen` with backend `openai`, greedy (T=0), 1 sample/problem
- EvalPlus's own instruction prompt (instruction prefix + task fenced in ```python),
  chat role, system message `You are a helpful assistant good at coding.`
  (the system message is injected by `humaneval_gen.py`'s monkeypatch, exactly as on the
  3090 — upstream EvalPlus sends none)
- `top_p=0.95` (request) — ignored by our engine, which samples by CLI flag only:
  `-e 0 --top-k 1` = greedy. llama.cpp at T=0 is also effectively greedy, so this is
  equivalent.
- Decode cap: **768 tokens**, not 512. `humaneval_3090/README.md` documents 512, but
  `make_request`'s `max_tokens=512` default is dead code — EvalPlus's decoder always
  passes `max_tokens=self.max_new_tokens` explicitly, and that default is 768
  (`evalplus/provider/base.py`). Confirmed on the wire.
- Evaluation: `evalplus.evaluate --dataset humaneval` → base pass@1 and HumanEval+ pass@1

## Running the cell: two things that cost a cycle

- **Score with the server stopped.** `run_server_cell.sh` scores in the same
  process tree that just served 164 problems, so the server is still resident
  while evalplus starts its own workers, and memguard killed the scoring pass at
  7.2 GB with the cell's own ceiling. The generations are written before scoring,
  so the fix is to re-run the scoring command alone
  (`python -m evalplus.evaluate --dataset humaneval --samples <stem>.jsonl
  --i-just-wanna-run`) — which is only possible because the samples are on disk,
  so do not let a driver delete them between the two phases.
- **Budget ~2 minutes per problem, not seconds.** 164 problems at the 768-token
  cap took 5 h 24 min on the 125B install. The timestamps on a previous arm's
  artifacts are last-write times, not durations; reading them as duration
  underestimates the run by hours.

## Platform adaptations

The 3090 documented two Windows patches. macOS needs neither, but it needs its own:

- **`reliability_guard` setrlimit**: `resource.setrlimit(RLIMIT_AS, ...)` raises
  `ValueError: current limit exceeds maximum limit` on macOS and killed every test
  subprocess. evalplus already special-cases Darwin for RLIMIT_STACK but not
  RLIMIT_AS/RLIMIT_DATA. `patch_evalplus.py` wraps all three in try/except
  (idempotent; the guard is explicitly not a sandbox, so losing the memory cap is
  acceptable for scoring our own generations).

## How the pieces fit

```
evalplus --openai SDK--> shim.py :8080 --SSE--> finchmoe-infer :9000
```

`finchmoe-infer`'s serve mode is not OpenAI-compatible enough for the SDK: it always
answers SSE (never non-streaming), ignores the request's system message, and ignores
temperature/top_p/n in the body. `shim.py` bridges all three without touching the
validated engine. Two details mattered:

- **The byte round-trip.** The engine escapes every byte ≥ 0x80 as `\u00XX` — raw
  bytes, so multi-byte characters are split across chunks. The shim decodes with
  `latin-1`, not UTF-8 (UTF-8 there produces mojibake). Unit-tested, including one
  byte per chunk as the worst case.
- **`ensure_ascii=False`** when forwarding — the engine's minimal unescaper would
  otherwise feed literal `\uXXXX` text into the prompt.

`--no-think` is our stand-in for the 3090's `--reasoning off`, but it is weaker: it
only prefixes an empty think block; nothing prevents the model from sampling think
tokens. The shim strips think blocks from the output, but any tokens spent thinking
still count against the 768-token budget. Measured leak in cell B: 2/164 (1%).

## Results

### Cell B — 3-bit native: 13.4% base / 12.8% plus (22/164)

LOWER than the old raw-completion harness's 29.9% on the same tier. The protocols are
not comparable: this one asks for a complete self-contained script (signature +
docstring + body), which is a harder task for a model whose greedy decoding falls into
docstring reproduction loops. Diagnostics on the 164 raw outputs:

- 100% correctly fenced markdown code blocks (protocol is working)
- 23/164 (14%) exhausted the 768-token cap without finishing (the engine always
  reports `finish_reason: stop`; the cap is the only truncation signal)
- 16/164 (10%) sanitized to imports-only — no extractable function
- 12/164 (7%) contain a 5x+ repeated line (greedy attractor loops, e.g. the model
  re-emitting docstring example lines forever)

The loops dominate the failures: they consume the budget, and whatever code survives
is usually syntactically broken. This is the same greedy-attractor signature as
Bug 15 (long-generation loops) surfacing inside docstrings.

### Cells C, D, D′ — the decisive cluster

All three score 12.8% / 12.8% / 11.6% — statistically indistinguishable from cell B
and from each other, across four different weight files:

- 3-bit native experts
- 4-bit native experts
- our locally-quantized Q4_K_M GGUF
- **the exact lmstudio-community Q4_K_M GGUF that scores 91.5% on the 3090
  through llama.cpp**

The fourth is the control that settles the experiment. The same file, the same
harness, the same protocol: **91.5% under llama.cpp, 11.6% under FinchMoE.**
Identical failure signatures across all four cells (docstring reproduction loops
under greedy decoding, 13–21% repetition loops, 10–18% sanitizing to imports-only).

**Conclusion: the deficit is in our engine, tokenizer, or chat-template path —
not in the quantization.** Every weight format we feed the engine converges to
~13%; the 3090's own file is no better than our worst quant. This refutes the
`QUANT_QUALITY_PLAN.md` conclusion that "32.3% is the ceiling" — that ceiling
was an artifact of our harness and engine, not the weights.

Prime suspects for the next step (in order of suspicion):

1. **Chat-template construction.** The engine's GGUF raw-completion path was
   cross-validated bitwise against llama.cpp (90-token logits, cos 0.998, argmax
   match), but the chat path never was. llama.cpp applies the real Qwen3
   chat template with `--reasoning off`; our engine hand-builds
   `<|im_start|>system\n...<|im_end|>...` itself (`tokenize_chat_message`,
   infer.m:13739) and prepends an empty think block. A small template deviation
   (extra newline, misplaced `<think>`, different role punctuation) would
   degrade every cell uniformly — exactly what we see.
2. **Tokenizer** — the template is tokenized by our vocab; a systematic
   token-level mismatch would survive the logit-level cross-validation of raw
   continuations less well, but cannot be ruled out.
3. **Engine numerical state** — least likely given the bitwise parity work, but
   the uniformity of the ~13% ceiling is what a shared compute bug looks like.

Next experiment (cheap, decisive): run the same HumanEval protocol through
llama.cpp *on this Mac* with the same GGUF. If llama.cpp here reproduces ~90%,
the harness and file are vindicated on this hardware and the hunt narrows to a
differential debug of our chat template vs llama.cpp's.

## Caveats

1. **Cell D vs D′ use different GGUF files.** D is our local quantization
   (`general.name = Qwen3.6 35B A3B Bf16`, 21,713,462,624 bytes, no imatrix).
   D′ is the 3090's exact lmstudio-community file (`general.name =
   Qwen_Qwen3.6 35B A3B`, 733 tensors, 21,166,757,728 bytes, imatrix-quantized),
   copied to `finchmoe/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf`
   (2026-08-21). With D′ the provenance gap is closed; both score ~12%.
2. Greedy decoding (T=0) is the harshest setting for this model — the known
   low-temperature attractor. The 3090 numbers are greedy too, so the comparison is
   fair, but non-greedy sampling would likely raise every cell.
3. Single sample per problem, no thinking: a conservative protocol by design.
4. Our engine ignores top_p in requests; llama.cpp applies top_p=0.95. At T=0
   (greedy) this is a no-op on both sides.

## Files

```
humaneval_evalplus/
  README.md              this document
  setup.sh               python3.12 venv + evalplus==0.3.1 + openai; dataset pre-warm
  patch_evalplus.py      macOS setrlimit patch for the installed evalplus (idempotent)
  shim.py                OpenAI-compatible proxy -> finchmoe-infer SSE
  test_shim.py           unit tests for the shim's SSE reassembly (no engine needed)
  humaneval_gen.py       codegen driver (3090's copy, only root= changed)
  smoke_check.py         protocol diagnostics on a partial or full sample set
  run_cell.sh            one tier end-to-end with tier-load assertion + teardown
  run_all.sh             sequential sweep (lockfile + port guards; one engine at a time)
                         (run_cell.sh also takes a `gguf3090` tier for the 3090's file)
  run_server_cell.sh     one cell against a FinchMoEServer install — the Swift engine,
                         no shim; the 3.6/3.8 `.finch` cells, which are NOT in the
                         matrix above (that matrix is the archive C engine). Publishes
                         to quality/humaneval/. See its header for usage.
  reference/             the 3090's published qwen3.6-35b-a3b samples, for cross-checking
  results/               evalplus output per tier
```

The `.finch` cells use this same EvalPlus install, the same `humaneval_gen.py`,
and the same scoring — only the engine on port 8080 differs, which is what lets
a 3.8 cell be read against a 3.6 one. Scores, and the two harness traps they hit
(the 4 GB compressor ceiling is *binding* for a 3.8 server run; scoring itself is
the memory hog, so score standalone with `--parallel 2`), are recorded in
`docs/QWEN38_PORT.md` §8.

## Reproduce

```sh
./setup.sh                      # venv + evalplus + macOS patch
./run_cell.sh 3bit              # one cell (engine + shim + generate + evaluate + teardown)
./run_cell.sh gguf3090          # cell D': the 3090's exact GGUF through FinchMoE
./run_all.sh                    # or: the whole sweep, strictly sequential
```

The engine's startup log is asserted per cell (`Quant: 3-bit experts (1376256 bytes
each)` etc.) because `--4bit` was once silently overridden by expert auto-detect
(commit 6c2e264). Engine runs are strictly sequential and port-guarded: concurrent
model runs on this machine have caused kernel panics three times.

## Cross-check of the evaluation half

`evalplus.evaluate` on the 3090's own published samples with our (patched) install
reproduces their numbers exactly: **0.915 base / 0.890 plus**. Scoring is therefore
provably identical across machines; any score differences come from generation.

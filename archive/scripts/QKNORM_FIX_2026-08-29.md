# LOCALIZED ROOT CAUSE: GDN q/k normalization convention (2026-08-29)

## Finding (Phase-1 differential, reproducible)

With identical inputs, numpy's norm + projection match finchmoe to 1e-7
(diff_h1: attn_norm maxd 1e-6, qkv maxd 2e-6). But the GDN delta-out (`de`)
shows a real per-op difference:

| q/k norm style (numpy simulation) | de maxd at L0 p0 | de rms |
|---|---|---|
| numpy-style (eps on SUM, scale at read) | 2.103e-03 | 4.726e-05 |
| **finchmoe-style (eps on MEAN, pre-scale)** | **1.381e-04** | **3.628e-06** |

So finchmoe's current convention reproduces its own `de` to 1.4e-4 when
simulated in numpy — but numpy/llama use the OTHER convention (2.1e-3 gap).
The ~2e-3-per-step difference integrates through the recurrent GDN state into
the observed directional L38 drift (p0 0.003 → p135 0.18) that flips greedy
argmaxes.

## The three conventions

**finchmoe (current, all 3 sites):**
- `q / sqrt(mean(q²) + 1e-6) * (1/128)`, then delta-out `s @ q` directly
- `k / sqrt(mean(k²) + 1e-6) * (1/√128)`
- Sites: infer.m `cpu_rms_norm_bare` usage at 5344/5350
  (`linear_attention_forward`), 7986/7992 (`linear_attn_chain_cpu`, the GGUF
  eval path), and shaders.metal 1757/1774 (the GPU `delta_net_step` norm
  kernel, also eval path). Metal kernel delta-out uses q directly (line 1666).

**llama.cpp (qwen35moe.cpp:454-457):**
- `q = ggml_l2_norm(q, eps_norm)` → `scale = 1/max(sqrt(sum(q²)), eps)`
  (eps on the SUM, fmaxf clamp — ops.cpp:4190)
- the 1/√128 scale is applied inside `build_recurrent_attn` (read time)

**numpy (numpy_forward.py:453-464, the 100% reference):**
- `q = q / sqrt(sum(q²) + 1e-6)`, `k = k / sqrt(sum(k²) + 1e-6)`
- delta-out: `s @ (q * q_scale)` where `q_scale = 1/√128`

numpy == llama convention (eps on sum, scale at read). finchmoe deviates:
eps on the mean (128× different eps term) and scale folded into q pre-read.

## The fix (Option C, first step)

Change finchmoe's GDN q/k normalization to the llama/numpy convention:
`q = q / sqrt(sum(q²) + eps)` and apply `1/√128` at the delta-out read
(matching numpy's `s @ (q * q_scale)`), in:

1. `linear_attn_chain_cpu` (infer.m ~7983-7994) — GGUF eval CPU path
2. `linear_attention_forward` (infer.m ~5340-5352) — legacy CPU path
3. shaders.metal `gated_delta_net_step` norm (~1757, ~1774) — GPU eval path:
   `q_inv_rms = rsqrt(q_sum_sq + 1e-6f)` (no /key_dim), `q = qval * q_inv_rms`
   (no inv_scale²), and apply `1/√128` in the delta-out read (line 1666:
   `out_val += state * (q * inv_scale)`)

## Verification

1. Rebuild finchmoe, regenerate /tmp/famprobe_baseline.dump.bin
2. numpy forward de-vs-de at L0 p0 should drop from 2.1e-3 → ~1e-4
3. L38 h-vs-llama / h-vs-numpy direction cosine should move toward numpy
4. 20-task slice toward the ~90% band (rep105 stays shipped until beat)

## STATUS (2026-08-29 09:25)

**Fix implemented in engine (all q/k norm + delta-out read sites):**

shaders.metal:
- `rms_norm_qk` (1732): `rsqrt(sum+1e-6)`, no /key_dim, no inv_scale pre-scale
- `gated_delta_net_step` (eval-path GPU kernel): delta-out read × 1/√128
- `fused_gdn_core` (2008): delta-out read × 1/√128
- `fused_gdn_full` (2167): same
- 2 pooled kernels (2907, 3125): norm + read fixed

infer.m:
- `linear_attn_chain_cpu` (~7983): L2 norm (eps on sum), no pre-scale;
  cblas delta-out read × 1/√128
- prefill path (~10496): L2 norm; cblas read × 1/√128
- `linear_attention_forward` (~5340): L2 norm; serial read × 1/√128

**Per-op verification (FIXED engine dump vs numpy-default):**

| (pos, layer) | de maxd BEFORE | de maxd AFTER |
|---|---|---|
| p0 L0 | 2.103e-03 | **1.363e-04** |
| p0 L1 | 7.897e-04 | **2.173e-04** |
| p1 L0 | 1.903e-03 | 1.424e-03 |
| p1 L1 | 1.957e-03 | 7.153e-04 |

L0 improves 15× as predicted. p1 residual (1.4e-3 at p1 L0) is the second
candidate (delta-rule accumulation order s@k / s@q / outer, serial-vs-BLAS)
— evaluate after the full L38/slice check.

**Next:** L38 h-vs-llama direction cosine + 20-task slice.

## STATUS (2026-08-29 09:55) — L38 verification + slice running

**Fixed engine vs numpy (h-vs-numpy rmsd at L38):**

| position | BEFORE fix | AFTER fix |
|---|---|---|
| p0 | 0.00345 | 0.00332 |
| p50 | 0.00392 | 0.00390 |
| p100 | 0.02802 | 0.02609 |
| p135 | **0.18129** | **0.04430** |

The deep-position divergence (p135, the directional-drift carrier) dropped
4× (0.18 → 0.044). L20 p135: 0.0088 → 0.0056. The engine is collapsing
toward numpy (the 100% reference). Remaining p135 gap (0.044) is the second
candidate: delta-rule accumulation order (s@k / s@q / outer, serial-vs-BLAS).

**20-task slice with the fixed engine (`qknorm-fix`) is running** — the
decisive generation-level test. rep105 baseline = 50%; target ~90% band.
Results: humaneval_evalplus/results/humaneval/qknorm-fix_openai_temp_0.0.raw.jsonl

## NEGATIVE RESULT + REVERT (2026-08-29 10:12)

The 20-task slice with the q/k norm fix ran to 12 tasks: **5/12 (41.7%)**,
BELOW the baseline (10/20 = 50%). The L38 deep-position h-direction improved
4× (p135 0.18 → 0.044) but the generation-level newline/indent suppression
PERSISTED and regressed 3 baseline-passing tasks:

| task | baseline | qknorm-fix |
|---|---|---|
| 0 | FAIL | PASS (gained) |
| 3 | PASS | NO_ENTRY_POINT (lost) |
| 9 | PASS | PASS |
| 10 | PASS | EXEC_ERR(SyntaxError) (lost) |
| 11 | PASS | NO_ENTRY_POINT (lost) |

**Conclusion: the q/k norm convention is NOT the cause of the slice failure.**
Aligning it with llama/numpy was mathematically correct (the engine deviated)
but does not fix the newline/indent suppression. The failure mechanism lives
elsewhere — likely the FINAL-LAYER output path (output_norm + lm_head +
residual at L39), where the fixed engine still showed elevated divergence
(L39 p135 0.057).

**Action taken: the q/k norm change was REVERTED** (infer.m + shaders.metal
restored to HEAD; engine rebuilt 10:12). rep105 remains the shipped config.

**Next lead:** the newline/indent token suppression at the logits level —
compare final-layer (L39) output_norm + lm_head logits between finchmoe and
numpy on the exact same input, focusing on newline/indent token ids. This is
where the 50% band originates, and it is NOT the GDN.

## The 0.036-logit knife-edge (2026-08-29 10:40) — what the newline suppression actually is

At the exact context where finchmoe emitted the wrong indent (`sorted_numbers =
sorted(numbers)` at column 1 instead of 4), numpy's (100%-reference) logits:

| token | id | numpy logit |
|---|---|---|
| 2-space | 256 | **15.350** (model's clear choice) |
| 4-space | 257 | 14.541 (correct indent) |
| 1-space | 220 | 14.505 (the wrong choice finchmoe made) |

**gap(4sp − 1sp) = +0.036** — a razor-thin margin. The model makes dozens of
near-tied whitespace decisions per task (gap ~0.03–0.1). numpy's drift lands
on the right side of most; finchmoe's larger final-layer drift (L39 p135
~0.05–0.25) flips a fraction of these 0.03-margin decisions → ~half the tasks
fail.

**Implication for Option C:** it is NOT about any single op (q/k norm, delta
order, etc. all verified near-identical). It is about reducing the TOTAL
accumulated hidden-state drift at the FINAL layer below the ~0.03 whitespace-
margin threshold, in the RIGHT direction. The q/k fix reduced L39 p135 drift
(0.249 → 0.057) yet the slice got WORSE (50% → 41.7%) — the residual drift
direction at the borderline contexts matters more than its magnitude.

The target is precise: final-layer h-vs-numpy rmsd must be ≲0.03 at the
whitespace-decision positions (not just L38/L39 mean), with the drift
direction matching numpy's. This requires either (a) full arithmetic
alignment so the drift is bounded below the threshold, or (b) a per-token
logit correction on the whitespace class (the earlier FINCHMOE_NL_BIAS probe
did 45% — it compensated one class but not all borderline classes).

## MARGIN PROBE + the REAL mechanism: per-occurrence rep-penalty crushing (2026-08-30)

The 0.036 knife-edge narrative was INCOMPLETE. A 6-task margin probe (tasks 0-5,
`--logit-diag 1`, 1145 decision steps, raw top-20 + whitespace logits dumped
per step) shows the knife-edge decisions are NOT where the failures come from:

**Raw whitespace margins are LARGE (1-7 logits), not 0.03.** NL (198) is raw
top-1 at 104/1145 steps with margins of 1.1-6.9 (median ~7-10) over the #2
token. In 17 of those the engine STILL emitted a code token. Mechanism
verified (102/104 predictions): `rep_penalty_apply` divides the logit by
1.05 **per occurrence in the 64-token ring** — newline appears in the ring
constantly (every line ends in NL), so k≈5-8 and the NL logit is crushed by
1.05^5..8 ≈ 1.28-1.48×, dropping below the runner-up code token → line
collapse / indent mixing. Numpy (no penalty) emits NL at those steps — hence
numpy 20/20 vs engine 10/20.

Concrete: task-5 (`intersperse`) whole-body collapse — at the docstring close
step, raw NL = 21.02 vs `if` = 18.36 (margin 2.66); after k=5 NLs in ring:
21.02/1.05^5 = 16.47 < 18.36 → `if` chosen → `"""if not numbers:return []…`
Task-1 (`separate_paren_groups`) mixed 3sp/4sp indents: same mechanism on
220/256/257.

**Fix (option-b calibration, sampler-level):** exempt structural whitespace
tokens {198 NL, 220 1sp, 256 2sp, 257 4sp, 9 tab} from the rep penalty AND the
n-gram blocker (their repetition is code structure, not an attractor loop).
Code tokens keep rep105 exactly as shipped (docstring-loop fix preserved).
A/B gate: FINCHMOE_PENALIZE_WS=1 restores the old path. This is strictly
closer to numpy (which has neither penalty nor blocker): whitespace decisions
follow the raw model ranking in every case, not just ≤0.05 margins.

## RESULT 1: whitespace exemption — 6/6 slice (2026-08-30)

The sampler-level calibration was implemented and tested on the 6-task margin
slice:

| config | 6-task slice | notes |
|---|---|---|
| baseline rep105 | 4/6 | tasks 1,5 fail |
| exemption {198,220,256,257,9} only | 4/6 | task 5 fixed (collapse gone), task 0 REGRESSED (` for` at 1sp: merged 3sp token 262 n-gram-blocked; token 9 wrongly exempted — it is `*` not tab) |
| **vocab-derived exemption (422 ws-only tokens)** | **6/6 = 100%** | tasks 0,1,5 all PASS |

Key corrections along the way:
- id 9 is `*` (star) in this vocab, NOT tab — exempting it would disable the
  "* * * *" attractor guard. The hardcoded list was wrong; the exempt set is
  now derived from the vocab at init: `decode(tok)` non-empty and every char
  in {space,\n,\r,\t} → 422 tokens (covers merged tokens like 262="   ",
  271="\n\n", 6987="\n    \n").
- The n-gram blocker ALSO had to exempt whitespace: at task-0 step 156 the
  raw top-1 was 262 ("   ", 3sp) but the 2-gram blocker fired on its
  repetition and the engine fell to 364 (" for", 1sp) — the single bad line
  that regressed task 0.
- Post-fix diag verification: ws_top1=145 steps, ws_flips=0, nl_top1=99,
  nl_suppressed=0 (before: 153/27/104/17).

Mechanism recap (margin probe, 1145 steps): raw NL margins are 1-7 logits
(not 0.03); rep_penalty_apply divides per-occurrence in the 64-token ring so
1.05^k with k=5-8 crushes NL below code tokens → line collapse. The exemption
restores raw-model (numpy) ranking for ALL whitespace decisions.

PENDING: full 20-task slice (baseline 10/20 = 50%) — result in next section.

## RESULT 2: full 20-task slice — 18/20 = 90% (2026-08-30)

| config | 20-task slice |
|---|---|
| baseline rep105 (shipped) | 10/20 = 50% |
| **+ whitespace exemption (422 ws-only tokens)** | **18/20 = 90%** |

Fixed: tasks 1, 5, 6, 8, 12, 17, 18 (all PASS now). Remaining failures:
- HumanEval/11 (`string_xor`): `for i in range(max len):` — missing underscore
  (`max_len` → `max len`), a CONTENT token-choice error, not whitespace.
- HumanEval/19 (`sort_numbers`): dict literal with dropped quotes
  (`two': 2`), a CONTENT quote-emission error, not whitespace.

The calibration hit the ~90% band target on the slice. rep105 value unchanged
(1.05); only its application to whitespace-only tokens changed (exempt), with
an A/B gate (FINCHMOE_PENALIZE_WS=1 restores old path). Final diag
verification: ws_top1=146 steps, flips=0, nl_suppressed=0.

NEXT: the 2 residual failures are non-whitespace; extending the band further
would need content-class calibration (out of option-b scope) or the full
164-task official sweep to confirm the headline number (earlier official:
18.3% with rep105).

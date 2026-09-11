# Runtime controls

The Mac app exposes generation and runtime controls in its fixed right
settings pane. FP16 is the fixed KV format. Generation settings apply to the
next request; load-time settings require a reload.

## Generation controls

The Mac app and CLI expose these generation controls:

| Control | Mac values | CLI flag | Default | Effect |
| --- | --- | --- | --- | --- |
| Maximum response | Automatic | `--max-new` | App: remaining context; CLI: 1,024 tokens | The app can use the context space left after formatting the prompt. The CLI uses its explicit or default `--max-new` limit. |
| Maximum context | 4K, 8K, 16K, 32K, 64K | `--max-context` | 4K | Sets prompt plus response capacity. The app shows an FP16 KV-memory delta — see the caveat below. |
| Temperature | 0...2 in 0.05 steps | `--temperature` | 0.2 | `0` is greedy; positive values sample. |
| Top-K | Off or 1...256 | `--top-k` | 64 | Keeps at most K candidates. CLI `0` turns it off. |
| Top-P | Off or 0.01...1 | `--top-p` | 0.95 | Applies nucleus truncation before Top-K and is effective only while Top-K is enabled. |

With positive temperature, a CLI Top-P below `1` requires Top-K between `1`
and `256`. To disable both truncation controls, pass `--top-k 0 --top-p 1`.
Generation controls apply to the next request and do not require a model
reload. They are interactive product settings, not the fixed community
benchmark protocol.

**KV-memory caveat.** "FP16" is accurate — the Swift engine has no other KV
format (the `--kv-fp16` / `--kv-turbo` flags are archive-era and absent here).
But the context menu's `+85 MB` / `+250 MB` / `+590 MB` / `+1.26 GB` labels are
literals computed for Gemma 4 26B-A4B (`AppContextLengthOption.menuLabel`), and
the instance property behind them pins `.gemma4_26B_A4B`. So for a Qwen 3.6 or
3.8 install the menu under-reports the real delta —
`AppContextLengthOption.fp16KVBytes(tokens:architecture:)` is architecture-correct
(counting full-attention layers only, since Qwen's linear layers hold no KV) but
is currently reached only by tests. Qwen 3.8 has 12 full-attention layers of 48.

## Runtime settings

| Control | Values | Production default | Effect |
| --- | --- | --- | --- |
| Expert-cache slots | 8, 16, 24, 32 | 16 | More slots can retain more routed experts and reduce later reads, but values above 16 use more RAM. The CLI takes the same list as `--expert-cache-slots`; a count below the model's top-k is rejected rather than left to trap. |
| Prompt prefill | On, off | On | On processes known prompt tokens through the chunked prefill path. Off disables that path. |
| RDADVISE | Off, Default, Bounded, Adaptive | Off | Applies experimental read advice. Its effect depends on the workload; it may help a short decode and slow a long one. |
| Model verification | Automatic, Full SHA-256, Trust verified install | Automatic | Automatic uses `verified-install.json` when it is present and valid and hashes everything otherwise. Full SHA-256 always hashes. Trust verified install requires the receipt and fails without one. The CLI and the server take the same three modes as `--verify auto\|full-sha256\|trusted-install`. |

Changing context length, expert-cache slots, RDADVISE, or model verification requires a reload.
Some sampling changes also require a reload because greedy and sampled
generation use different output-head paths. Prompt-prefill settings apply to
each request and do not require a reload.

## Run an experiment

1. Start from 4K context, 16 expert-cache slots, prefill on, and RDADVISE off.
2. Keep the prompt and generation controls fixed.
3. Record a baseline after a warmup.
4. Change one runtime control and reload the model.
5. Compare prompt prefill, request TTFT, decode rate, peak memory, and I/O per
   token over repeated runs.
6. Restore the production defaults when the experiment ends.

A production result uses the default runtime controls; the
[community benchmark protocol](COMMUNITY_BENCHMARKS.md) is the standard way to
record one. A run with changed runtime controls is experimental and must name
the changed setting.

## Read the results

- **Decode rate** measures generated tokens per second after prompt prefill.
- **Request TTFT** includes prompt prefill and the wait for the first generated
  token.
- **Peak memory** in Last run is the highest decode-service memory observed
  during the request. The HUD shows the service's current memory instead of the
  much smaller foreground UI process.
- **I/O / token** reports routed-expert read time per generated token.
- **Advanced** shows decode duration and per-token cb1, cb2, and output-head
  time. When RDADVISE runs, it also shows time, calls, data, and skipped advice.
  The CLI has the same breakdown behind `--counters`, which prints one extra
  stderr line after the timing footer: the `cb1` sub-buckets (with an
  `identity=` field that reads `exact` when they tile `cb1`), `io`, the head,
  expert-cache hits and misses, command buffers, and — when GPU timestamps are
  readable — summed GPU time. It accumulates unconditionally and is not
  suppressed by `--quiet`, which covers the timing footer only. Read the clock
  kinds: the `cb1` buckets are CPU encode-and-commit clocks that exclude the
  pipeline wait, while `io` and the head are wall clocks that include theirs, so
  the figures overlap and are not a serial timeline. See
  [System design](SYSTEM_DESIGN.md) for the bucket table.

During chunked prefill, the phase label reports exact progress, for example
`Prefill (128/514)`. Errors and unsupported configurations appear only when
they occur. RDADVISE remains experimental and is off by default. A measured
result is a data point, not a performance ceiling.

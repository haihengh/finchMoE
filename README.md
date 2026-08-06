

Recommended design: LieMoE-flash (Local inference engine) — SSD-streamed Metal runtime for Apple Silicon

For a Qwen 3.6 35B-A3B–class MoE and derived fine-tunes, I would build a model-family-aware engine, not a general-purpose MLX wrapper. The closest base architecture is TurboFieldfare, combined with Flash-MoE’s low-level Metal and I/O lessons, while borrowing oMLX only at the serving/KV-cache layer.

Your hard requirement—under 4 GB active unified memory—is feasible only if that means engine-owned active memory excluding macOS file-cache pages. It is not feasible to hold all model weights resident in 4 GB. The engine must keep the dense/shared path compact and SSD-stream routed experts per layer.

The exact resident-memory budget depends on the actual checkpoint config: layers, hidden size, expert count, top-k, shared expert structure, attention type, and KV dimensions. Do not hard-code it around the marketing label “35B-A3B”; inspect each checkpoint’s configuration and tensor shapes at conversion time.

<svg width="100%" viewBox="0 0 680 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="AeroMoE Apple Silicon streaming architecture">
  <defs>
    <marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--color-text-secondary)"/></marker>
  </defs>
  <style>
    text{font-family:var(--font-sans);fill:var(--color-text-primary)} .th{font-size:14px;font-weight:500}.ts{font-size:12px;fill:var(--color-text-secondary)}.arr{stroke:var(--color-text-secondary);stroke-width:1.2;fill:none;marker-end:url(#a)}.box{stroke-width:.6}
  </style>
  <g class="c-gray">
    <rect class="box" x="18" y="35" width="128" height="305" rx="12"/>
    <text class="th" x="82" y="62" text-anchor="middle">NVMe SSD</text>
    <text class="ts" x="82" y="81" text-anchor="middle">packed expert shards</text>
    <rect x="35" y="108" width="94" height="38" rx="5" fill="var(--color-background-tertiary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="ts" x="82" y="127" text-anchor="middle" dominant-baseline="central">L00 experts</text>
    <rect x="35" y="160" width="94" height="38" rx="5" fill="var(--color-background-tertiary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="ts" x="82" y="179" text-anchor="middle" dominant-baseline="central">L01 experts</text>
    <rect x="35" y="212" width="94" height="38" rx="5" fill="var(--color-background-tertiary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="ts" x="82" y="231" text-anchor="middle" dominant-baseline="central">… layer shards</text>
  </g>
  <path class="arr" d="M146 185 L198 185"/>
  <g class="c-amber">
    <rect class="box" x="199" y="122" width="145" height="126" rx="12"/>
    <text class="th" x="271" y="150" text-anchor="middle">I/O + cache planner</text>
    <text class="ts" x="271" y="171" text-anchor="middle">bounded parallel pread</text>
    <text class="ts" x="271" y="190" text-anchor="middle">8–16 expert slots</text>
    <text class="ts" x="271" y="209" text-anchor="middle">LFU / recency admission</text>
  </g>
  <path class="arr" d="M344 185 L393 185"/>
  <g class="c-purple">
    <rect class="box" x="394" y="35" width="267" height="305" rx="16"/>
    <text class="th" x="527" y="62" text-anchor="middle">Apple unified memory + Metal GPU</text>
    <rect x="414" y="88" width="227" height="52" rx="7" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="th" x="527" y="107" text-anchor="middle" dominant-baseline="central">Resident backbone</text>
    <text class="ts" x="527" y="126" text-anchor="middle" dominant-baseline="central">embedding · attention · router · shared path</text>
    <rect x="414" y="159" width="227" height="52" rx="7" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="th" x="527" y="178" text-anchor="middle" dominant-baseline="central">Metal expert slots</text>
    <text class="ts" x="527" y="197" text-anchor="middle" dominant-baseline="central">direct Q4 matvec + fused SwiGLU</text>
    <rect x="414" y="230" width="227" height="52" rx="7" fill="var(--color-background-secondary)" stroke="var(--color-border-secondary)" stroke-width=".5"/>
    <text class="th" x="527" y="249" text-anchor="middle" dominant-baseline="central">Bounded KV cache</text>
    <text class="ts" x="527" y="268" text-anchor="middle" dominant-baseline="central">paged · quantized · sliding window first</text>
    <text class="ts" x="527" y="316" text-anchor="middle">one token: backbone → route → fetch → MoE → next layer</text>
  </g>
</svg>

What to reuse from each project
Source	Reuse	Avoid / change
Flash-MoE	Hand-written Metal, direct quantized matvec, fused operations, pread()-based SSD reads, treating Apple unified-memory bandwidth as the primary constraint	Its architecture is tailored to a very different model family; do not use its tensor layout or routing assumptions blindly. Its “trust macOS page cache” approach is less safe on an 8 GB machine.
TurboFieldfare	The most relevant blueprint: strict resident-memory accounting, dedicated model format, per-layer expert streaming, bounded cache, async overlap of shared-expert GPU work with SSD reads, chunked prefill	Its Gemma-specific graph, top-8 behavior, tensor names, and .gturbo format must be replaced with a Qwen-family converter and runtime plan.
oMLX	Server lifecycle, OpenAI-compatible interface, paged KV-cache ideas, request scheduling, model management patterns	Do not put Python/MLX in the hot decode loop for this target. It is a robust serving layer, not a sub-4 GB expert-streaming execution core.
Memory contract

Target a 3.6–3.9 GB engine allocation ceiling, leaving a safety margin for macOS, UI, Metal driver allocation, and file cache.

Region	Target allocation	Notes
Quantized resident backbone	1.6–2.1 GB	Embedding, output head, attention projections, norms, routers, shared experts, plus any non-MoE FFNs. Must be Q4 or mixed Q4/Q5.
KV cache	0.6–1.0 GB	Context-dependent. Use paged allocation and quantization; do not promise full long context on 8 GB machines.
Metal activation/scratch arena	192–320 MB	Fixed-size, reusable; no allocation within decoding.
Routed-expert slots	512–896 MB	A small cache of complete expert tensors, sized using actual converted expert sizes.
CPU scheduler, tokenizer, metadata	100–180 MB	Keep router output small; no large CPU tensor copies.
Total	≤ 3.9 GB	Enforced by a runtime allocator and admission control.

The main feasibility gate is the non-expert core. If the Qwen-derived model’s required resident backbone cannot be packed below about 2.1 GB without unacceptable accuracy degradation, then a genuine 4 GB runtime budget is not achievable. Do not silently spill backbone tensors to SSD: that would cause an SSD fetch in every layer regardless of routing and collapse decode speed.

Model storage format: .aeromoe

Do not stream generic safetensors or ordinary MLX shards during token decode. Build a conversion pipeline that emits an inference-only format:

model.aeromoe/
  manifest.json            model shape, context/KV layout, quant schemes
  tokenizer.*              copied tokenizer assets
  core.bin                 packed resident tensors, 4 KB-aligned
  experts/
    layer_000.bin          all routed experts of layer 0
    layer_001.bin
    ...
  index.bin                offset, byte length, quant metadata per expert

Design rules
One independently readable, contiguous slice per expert.
Each slice includes all matrices needed by that expert: gate/up projection and down projection.
Alignment should favor direct I/O and Metal buffer binding; use a converter-defined, fixed alignment such as 64 KB.
Separate files per transformer layer are preferable to a huge interleaved file. The decoder knows which layer it is executing, and the I/O planner can read a selected expert as one contiguous range.
Pack directly for Metal. The SSD representation must be the GPU-consumable layout—not an intermediate format that requires unpacking or dequantizing into a larger buffer.
Mixed quantization:
Router: 8-bit.
Norms, RoPE constants, and numerically sensitive small tensors: FP16 or FP32 where necessary.
Dense backbone: Q4 with Q5/Q6 selectively for quality-sensitive tensors.
Routed experts: Q4 first; Q5 is an optional quality profile, not the low-RAM default.
The converter must validate architecture and tensor coverage before writing output. “Opus distilled” or uncensored variants may alter tokenizer/configuration, but should only be accepted if their actual graph matches a supported Qwen adapter.
Decode execution plan

Per generated token, per transformer layer:

Backbone Metal command buffer A

RMSNorm.
Q/K/V and output projections.
RoPE and decode attention against the paged KV cache.
Router logits.
Router top-k selection on GPU if practical; otherwise copy only the small expert-ID/weight result to CPU.

CPU planner

Looks up the selected expert slices in index.bin.
Checks the bounded Metal-visible expert-slot cache.
Submits cache misses as bounded, parallel pread() operations.
Performs no weight transformations and no full-tensor copies.

Overlap only useful work

While SSD reads run, execute the shared expert or any independent layer operations in Metal command buffer B.
Do not aggressively prefetch predicted experts. Flash-MoE’s published experiments found that speculative routing/prefetch hurt performance due to poor hit rate and unified-memory contention.

Expert Metal command buffer C

Bind resident cache hits and freshly filled slots.
Quantized GEMV for gate/up.
Fuse SiLU and element-wise gating.
Quantized down projection.
Apply routing weights, combine output, residual add, and next RMSNorm in as few passes as possible.

Eviction

Evict only after a slot is needed.
Use a small adaptive policy: protected “used this token/layer” set + frequency score + recency tiebreaker.
On an 8 GB Mac, issue F_NOCACHE or equivalent only for data unlikely to be reused; otherwise macOS may fill available memory with expert pages and create memory pressure.

The key difference from the simplistic “load then unload every token” design is that loading is still demand-driven, but recently used experts remain in a strictly capped cache. This is essential because routing repeats enough that blind unloads waste SSD bandwidth.

Apple-specific choices
GPU: Metal is the primary execution device

Use Metal, not ANE, for the decode path:

You need custom Q4/Q5 matrix-vector kernels, dynamic expert binding, command-buffer control, and predictable buffer ownership.
The ANE is not a good fit for frequent dynamic expert-swapping or low-level MoE routing.
Apple Silicon has unified memory, so there is no PCIe transfer between CPU and GPU. The bottleneck is shared DRAM bandwidth and SSD DMA contention, not GPU-to-CPU copying.
I/O
Use pread() on a dedicated I/O queue with a small number of in-flight reads.
Start with 2–4 parallel reads, then autotune by machine class. More concurrency can increase latency through memory-controller contention.
Allocate fixed, aligned, Metal-visible buffers for expert slots. Reads should fill those buffers directly if the API path allows it; otherwise copy only once.
Benchmark cold SSD, warm OS cache, and memory-pressure cases separately. Warm-cache throughput is useful but should not be advertised as sustained cold-cache throughput.
CPU
Use Swift for packaging/UI/service orchestration if desired, but put the runtime core in C++/Objective-C++ plus Metal or Swift with a thin C/Obj-C Metal layer.
Use GCD / Swift concurrency for I/O coordination, but avoid task allocation in inner decode loops.
Pin no threads manually until profiling proves a benefit; Apple’s scheduler generally does better with QoS and bounded work queues.
Prefill and KV strategy

Decode is the memory-critical mode. Prefill needs different behavior:

Process prefill in small chunks—start with 32 tokens, autotune up to 128 only if memory permits.
Deduplicate selected experts across all rows in a chunk: a fetched expert serves every routed row that needs it.
Use a paged KV layout with a hard token budget.
For a 4 GB active-memory profile:
Default to a conservative context cap.
Prefer sliding-window attention where the model supports it.
Quantize older KV blocks before considering SSD KV offload.
SSD KV is a latency recovery mechanism for resumed prefixes, not part of the critical decode loop.

oMLX’s tiered KV-cache approach is useful for resumed conversation prefixes, but expert SSD streaming and SSD KV restoration must never compete unrestrictedly. Give expert I/O priority during generation.

Build order

Architecture probe + converter

Parse the actual model config and weights.
Emit manifest, core.bin, layer-expert shards, and index.
Add an offline reference test comparing individual layer outputs to the source implementation.

Resident-only Metal decoder

Load a small compatible Qwen-family MoE entirely into memory.
Validate tokenizer, RoPE, attention, routing, and exact expert combination before adding SSD behavior.

One-layer expert-streaming prototype

Replace one MoE layer with SSD pread() and 2–4 fixed expert slots.
Verify numerics and profile real cold-read latency.

Full streaming and capped memory allocator

Add strict allocation accounting.
Fail fast or reduce KV capacity when the 3.9 GB ceiling would be exceeded—never let macOS swap.

Prefill, server, and fine-tune adapters

Add chunked prefill.
Then expose a local OpenAI-compatible endpoint.
Support additional distills/fine-tunes through explicit architecture adapters and conversion validation.
What performance to expect

I would not promise a token/sec figure without the target Mac model and a converted checkpoint. It will be driven by:

Apple GPU and unified-memory bandwidth,
internal SSD speed and free space,
actual expert size and top-k,
resident-core size,
context/KV-cache size,
cache hit rate,
quantization quality profile,
whether the model’s dense/shared path fits the resident budget.

The appropriate success criterion for version one is:

correct output matching a reference runtime within an agreed tolerance, no system swap, active engine allocation below 4 GB, and stable cold-cache decoding.

TurboFieldfare demonstrates the closest relevant proof point: it reports a 26B-A4B model with a roughly 2 GB weight-and-KV budget by keeping a 1.35 GB shared core resident and streaming selected experts from SSD. TurboFieldfare’s design is therefore the best structural starting point; Flash-MoE supplies the stronger low-level kernel and SSD-streaming techniques for large MoEs.

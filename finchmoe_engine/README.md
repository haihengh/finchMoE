# FinchMoE Engine — Complete Reference

> **Production-quality C++/Metal inference engine for Qwen3.6-35B-A3B on Apple Silicon**
> Target: macOS 14+ · M1–M4 · Metal-only (no ANE) · < 4 GB active RAM

---

## Quick start

```bash
# 1. Convert Hugging Face weights → .finchmoe format
python3 finchmoe_convert.py  \
    --model   /path/to/Qwen3.6-35B-A3B-Instruct \
    --output  qwen3.finchmoe

# 2. Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.logicalcpu)

# 3a. One-shot generation
./build/finchmoe_generate  qwen3.finchmoe  "Explain MoE routing in one paragraph"

# 3b. Interactive chat (with tool-calling)
./build/finchmoe_chat  qwen3.finchmoe  qwen3.tiktoken \
    --system "You are a helpful assistant." \
    --ctx 4096  --temp 0.7  --top-p 0.9
```

---

## Repository layout

```
finchmoe_engine/
│
├── CMakeLists.txt               Build system (CMake 3.22+, C++20, ObjC++20)
│
│  ── Session 1: wire format ───────────────────────────────────────────────────
├── finchmoe_types.h              DType, ModelConfig, TensorView, Status
├── finchmoe_format.h             .finchmoe file parser: header + dense/expert indexes
├── memory_ledger.h              Lock-free atomic memory budget tracker
│
│  ── Session 2: engine core ───────────────────────────────────────────────────
├── expert_cache.h / .mm         LFU+recency expert slab cache (Metal buffers)
├── io_planner.h / .cpp          Parallel pread() I/O worker thread pool
├── engine_core.h / .mm          Metal device + command queue + file handle
│
│  ── Session 3: Metal kernels ─────────────────────────────────────────────────
├── kernels/
│   ├── rope_kernels.metal       Rotary position embedding (RoPE)
│   ├── norm_kernels.metal       RMSNorm
│   ├── gemv_kernels.metal       bf16 matrix×vector (GEMV)
│   ├── attention_kernels.metal  GQA flash-attention (no materialized QK^T)
│   ├── moe_kernels.metal        SiGLU MoE gating + expert accumulation
│   ├── kernel_dispatch.h        Pure-C++ dispatch interface
│   └── kernel_dispatch.mm       Pipeline state cache + MTLCommandBuffer management
│
│  ── Session 4: inference loop ────────────────────────────────────────────────
├── kv_cache.h / .mm             GQA key-value cache (ring buffer, bf16)
├── sampler.h / .cpp             Temperature / top-p / top-k sampling
├── model_runner.h / .mm         94-layer Qwen3.6 forward pass
├── inference_engine.h / .mm     Public C++ API: InferenceEngine
│
│  ── Session 5: tool-calling layer ────────────────────────────────────────────
├── tokenizer.h / .cpp           BPE tiktoken tokenizer (loads .tiktoken vocab)
├── chat_template.h / .cpp       ChatML formatter + tool-schema JSON injection
├── tool_schema.h                ToolParam / ToolSchema / ToolCall / ToolResult types
├── tool_parser.h / .cpp         <tool_call> XML block parser + recursive JSON parser
├── tool_engine.h / .cpp         Multi-turn agentic loop + streaming callback
│
│  ── Executables ──────────────────────────────────────────────────────────────
├── tools/
│   ├── generate_main.cpp        CLI: one-shot text generation
│   └── chat_main.cpp            CLI: interactive multi-turn chat (REPL + demo tools)
│
│  ── Tests ────────────────────────────────────────────────────────────────────
└── test/
    └── engine_test.cpp          Engine smoke test (load → generate → verify)
```

Converter (lives outside the engine tree):

```
finchmoe_convert.py               safetensors → .finchmoe weight converter
```

---

## Architecture overview

```
User prompt
    │
    ▼
┌─────────────┐    .tiktoken vocab     ┌──────────────┐
│  Tokenizer  │◄────────────────────── │  vocab file  │
│  (BPE)      │                        └──────────────┘
└──────┬──────┘
       │ token IDs
       ▼
┌──────────────────┐
│  ChatTemplate    │  Formats ChatML prompt, injects tool schemas as JSON
└──────┬───────────┘
       │ formatted string → re-tokenize
       ▼
┌──────────────────────────────────────────────────────────┐
│                  InferenceEngine  (C++ facade)            │
│                                                          │
│  ┌────────────┐  acquire_expert()  ┌─────────────────┐   │
│  │ EngineCore │◄──────────────────►│  ExpertCache    │   │
│  │ Metal dev  │                    │  LFU+recency    │   │
│  │ cmd queue  │   pread() batch    │  evict → disk   │   │
│  └─────┬──────┘◄──────────────────┤  IOPlanner      │   │
│        │                          └─────────────────┘   │
│        │ MTLCommandBuffer                                 │
│        ▼                                                  │
│  ┌─────────────────────────────────────────────────┐      │
│  │            ModelRunner  (94 layers)             │      │
│  │                                                 │      │
│  │  for each layer:                                │      │
│  │    RMSNorm → QKV proj → RoPE                   │      │
│  │    GQA Flash-Attention (4 KV heads, 64 Q heads) │      │
│  │    RMSNorm → MoE gate → top-8 routing           │      │
│  │    8 expert GEMV (gate · up → SiGLU → down)    │      │
│  │    KV-cache update                              │      │
│  └─────────────────────────────────────────────────┘      │
│        │ logits [vocab_size=151936]                        │
│        ▼                                                   │
│  ┌────────────┐                                            │
│  │  Sampler   │  temperature · top-p · top-k              │
│  └────────────┘                                            │
└──────────────────────────────────────────────────────────┘
       │ generated token(s)
       ▼
┌──────────────────────────────────────────────────────────┐
│               ToolEngine  (multi-turn agentic loop)       │
│                                                          │
│  while not done:                                         │
│    text = InferenceEngine::generate(prompt)              │
│    if <tool_call> block detected:                        │
│      call = ToolParser::parse(text)                      │
│      result = registered_handler[call.name](call.args)   │
│      prompt += ChatTemplate::append_tool_result(result)  │
│    else:                                                 │
│      emit final answer                                   │
└──────────────────────────────────────────────────────────┘
       │ final text
       ▼
  User / streaming callback
```

---

## Memory budget (Qwen3.6-35B-A3B, bf16, 4096-token context)

| Component            | Size      | Notes |
|----------------------|-----------|-------|
| Dense backbone       | ~600 MB   | Persistent; loaded once at startup |
| KV cache             | ~400 MB   | 4 KV heads × 64 dim × 4096 ctx × 94 layers × 2 (K+V) × 2 B |
| Activation scratch   | ~128 MB   | Two ping-pong hidden-state buffers |
| Expert cache pool    | ~2.9 GB   | LFU+recency managed; hard eviction before 4 GB |
| **Total cap**        | **4 GB**  | Enforced by `MemoryLedger` |

Each expert slab = gate(768×4096) + up(768×4096) + down(4096×768) = **18 MB bf16**
→ ~160 expert slabs can be hot simultaneously (128 experts/layer, 8 active per token)

---

## Metal kernel summary

| Kernel | Dispatch shape | Notes |
|--------|---------------|-------|
| `rope`        | `[seq_len, n_heads]` | Complex rotation in fp32 accumulation |
| `rmsnorm`     | `[seq_len]` threads, 256 simdgroup | Welford variance, eps=1e-6 |
| `gemv_bf16`   | `[out_dim]` rows, 4 simdgroups/row | 4× unrolled inner loop |
| `attention`   | `[n_heads, seq_len]` | Flash-attention tiling; GQA KV broadcast |
| `moe_gate`    | `[seq_len]` | Softmax + top-k argmax in one threadgroup |
| `moe_expert`  | `[out_dim]` per active expert | Fused gate×up SiGLU + down projection |

All kernels operate on `MTLStorageModeShared` buffers — zero-copy between CPU and GPU on Apple Silicon.

---

## Qwen3.6-35B-A3B model config

| Parameter | Value |
|-----------|-------|
| `hidden_size` | 4096 |
| `num_hidden_layers` | 94 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 4 |
| `head_dim` | 64 |
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 768 |
| `intermediate_size` | 2048 |
| `vocab_size` | 151936 |
| `rope_theta` | 1000000.0 |
| Special: `<\|im_start\|>` | 151644 |
| Special: `<\|im_end\|>` | 151645 (stop token) |

---

## Tool-calling format (Qwen3.6 official)

The model emits tool calls as XML blocks:

```xml
<tool_call>
{"name": "calculator", "arguments": {"expression": "sqrt(2) * 3.14159"}}
</tool_call>
```

`ToolParser` detects these, extracts the JSON, and dispatches to the registered C++ handler. The result is fed back as a `<tool_response>` in the next turn.

### Registering a tool

```cpp
ToolEngine engine(model_path, vocab_path);

engine.register_tool(ToolSchema{
    .name        = "get_weather",
    .description = "Get current weather for a city",
    .parameters  = {
        ToolParam{"city",  "string",  "City name", true},
        ToolParam{"units", "string",  "'celsius' or 'fahrenheit'", false},
    },
}, [](const ToolCall& call) -> ToolResult {
    auto city  = call.arguments.at("city");
    // ... fetch weather ...
    return ToolResult{call.id, call.name, R"({"temp":22,"condition":"sunny"})"};
});

// Run one turn (blocking) or streaming
auto reply = engine.chat("What is the weather in Tokyo?");
```

---

## Build targets

| Target | Command | Description |
|--------|---------|-------------|
| `finchmoe_engine` | *(static lib)* | All engine sources; linked by the executables |
| `finchmoe_test` | `cmake --build build -t finchmoe_test` | Smoke test: load + generate 20 tokens |
| `finchmoe_generate` | `cmake --build build -t finchmoe_generate` | One-shot generate CLI |
| `finchmoe_chat` | `cmake --build build -t finchmoe_chat` | Interactive REPL with tool-calling |

### finchmoe_chat flags

| Flag | Default | Description |
|------|---------|-------------|
| `--system TEXT` | *(none)* | System prompt injected at turn 0 |
| `--ctx N` | 4096 | KV context window in tokens |
| `--temp F` | 0.7 | Sampling temperature |
| `--top-p F` | 0.9 | Nucleus sampling p |
| `--top-k N` | 50 | Top-k sampling |

### REPL commands

| Command | Effect |
|---------|--------|
| `/quit` | Exit |
| `/clear` | Reset conversation history |
| `/stats` | Print runtime stats (tokens/s, memory usage) |
| `/tools` | List registered tool schemas |
| `/system TEXT` | Replace system prompt for next turn |

---

## Requirements

- macOS 14.0+ (Sonoma or later)
- Apple Silicon: M1, M2, M3, or M4 chip
- CMake ≥ 3.22
- Xcode 15+ (for Metal shader compilation and ObjC++ 20)
- Python 3.10+ with `safetensors`, `torch` (for converter only)

---

## Sessions roadmap

| # | Component | Status |
|---|-----------|--------|
| 1 | `safetensors → .finchmoe` converter + wire format types | ✅ Complete |
| 2 | Engine core: expert cache, I/O planner, Metal device init | ✅ Complete |
| 3 | Metal kernels: RoPE, RMSNorm, GEMV, GQA attention, MoE | ✅ Complete |
| 4 | Inference loop: KV cache, sampler, 94-layer forward pass | ✅ Complete |
| 5 | Tool-calling: tokenizer, chat template, tool parser, agentic loop | ✅ Complete |

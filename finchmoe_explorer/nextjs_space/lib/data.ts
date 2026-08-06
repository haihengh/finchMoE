// All static data for FinchMoE Explorer

export const MODEL_CONFIG = {
  name: 'Qwen3.6-35B-A3B',
  hidden_size: 4096,
  num_hidden_layers: 94,
  num_attention_heads: 64,
  num_key_value_heads: 4,
  gqa_ratio: 16,
  head_dim: 64,
  num_experts: 128,
  num_experts_per_tok: 8,
  moe_intermediate_size: 768,
  vocab_size: 151936,
  expert_slab_mb: 18,
  dense_backbone_mb: 600,
  cache_budget_gb: 4,
  cache_capacity_slabs: 160,
  stop_token: 151645,
  activation_scratch_mb: 128,
} as const;

export function kvCacheMB(seqLen: number): number {
  const bytes = MODEL_CONFIG.num_hidden_layers * MODEL_CONFIG.num_key_value_heads * MODEL_CONFIG.head_dim * seqLen * 2 * 2;
  return bytes / (1024 * 1024);
}

export interface SourceFile {
  name: string;
  path: string;
  session: number;
  lines: number;
  type: 'header' | 'source' | 'metal' | 'cmake' | 'markdown' | 'test';
  description: string;
  symbols: string[];
}

export const SESSION_LABELS: Record<number, { title: string; color: string; description: string }> = {
  1: { title: 'Wire Format', color: '#3b82f6', description: 'Binary file format, type definitions, and memory budget tracking' },
  2: { title: 'Engine Core', color: '#06b6d4', description: 'Expert caching, I/O planning, and core engine orchestration' },
  3: { title: 'Metal Kernels', color: '#8b5cf6', description: 'GPU compute kernels for RoPE, norms, GEMV, attention, and MoE dispatch' },
  4: { title: 'Inference Loop', color: '#10b981', description: 'KV cache, sampling, model runner, and the main inference engine' },
  5: { title: 'Tool-Calling', color: '#f59e0b', description: 'Tokenizer, chat templates, tool parsing, and agentic execution' },
  0: { title: 'Build', color: '#ef4444', description: 'Build system, documentation, and tests' },
};

export const SOURCE_FILES: SourceFile[] = [
  // Session 1
  { name: 'finchmoe_types.h', path: 'src/', session: 1, lines: 184, type: 'header', description: 'Core type definitions: LayerMeta, ExpertSlab, ModelHeader, bf16 helpers', symbols: ['LayerMeta', 'ExpertSlab', 'ModelHeader', 'bf16_to_float', 'float_to_bf16'] },
  { name: 'finchmoe_format.h', path: 'src/', session: 1, lines: 315, type: 'header', description: '.finchmoe binary format reader/writer with 64KB alignment and mmap support', symbols: ['FinchMoeFormat', 'FormatReader', 'FormatWriter', 'ALIGNMENT_64KB', 'MAGIC_NUMBER'] },
  { name: 'memory_ledger.h', path: 'src/', session: 1, lines: 148, type: 'header', description: 'Real-time memory budget tracker with soft/hard caps and eviction triggers', symbols: ['MemoryLedger', 'BudgetEntry', 'EvictionPolicy', 'SoftCap', 'HardCap'] },
  // Session 2
  { name: 'expert_cache.h', path: 'src/', session: 2, lines: 152, type: 'header', description: 'LFU expert cache with Metal buffer management and async prefetch', symbols: ['ExpertCache', 'CacheEntry', 'EvictionResult', 'PrefetchHint'] },
  { name: 'expert_cache.mm', path: 'src/', session: 2, lines: 295, type: 'source', description: 'ExpertCache implementation: LFU eviction, buffer pool, async loading', symbols: ['ExpertCache::load', 'ExpertCache::evict_lfu', 'ExpertCache::prefetch'] },
  { name: 'io_planner.h', path: 'src/', session: 2, lines: 119, type: 'header', description: 'Async I/O scheduler for expert slab loading with priority queue', symbols: ['IOPlanner', 'IORequest', 'IOPriority', 'CompletionCallback'] },
  { name: 'io_planner.cpp', path: 'src/', session: 2, lines: 202, type: 'source', description: 'IOPlanner implementation: double-buffered reads, deadline scheduling', symbols: ['IOPlanner::submit', 'IOPlanner::poll', 'IOPlanner::cancel'] },
  { name: 'engine_core.h', path: 'src/', session: 2, lines: 198, type: 'header', description: 'Top-level engine coordinator: ties cache, I/O, and ledger together', symbols: ['EngineCore', 'EngineConfig', 'LayerPlan', 'ExecutionContext'] },
  { name: 'engine_core.mm', path: 'src/', session: 2, lines: 315, type: 'source', description: 'EngineCore implementation: layer-by-layer execution with expert staging', symbols: ['EngineCore::step_layer', 'EngineCore::stage_experts', 'EngineCore::run_attention'] },
  // Session 3
  { name: 'rope_kernels.metal', path: 'src/kernels/', session: 3, lines: 189, type: 'metal', description: 'Rotary Position Embedding kernel for query/key rotation', symbols: ['rope_forward', 'rope_inverse', 'compute_freqs'] },
  { name: 'norm_kernels.metal', path: 'src/kernels/', session: 3, lines: 198, type: 'metal', description: 'RMSNorm kernel with fused residual add', symbols: ['rmsnorm_forward', 'rmsnorm_residual'] },
  { name: 'gemv_kernels.metal', path: 'src/kernels/', session: 3, lines: 205, type: 'metal', description: 'General matrix-vector multiply kernels for dense projections', symbols: ['gemv_bf16', 'gemv_f32', 'gemv_fused_gelu'] },
  { name: 'attention_kernels.metal', path: 'src/kernels/', session: 3, lines: 311, type: 'metal', description: 'Multi-head attention with GQA broadcast (4 KV → 64 Q heads)', symbols: ['attention_qkv', 'attention_softmax', 'attention_output', 'gqa_broadcast'] },
  { name: 'moe_kernels.metal', path: 'src/kernels/', session: 3, lines: 319, type: 'metal', description: 'MoE dispatch: gate scoring, top-8 selection, weighted expert combine', symbols: ['moe_gate_score', 'moe_topk_select', 'moe_expert_ffn', 'moe_weighted_sum'] },
  { name: 'kernel_dispatch.h', path: 'src/kernels/', session: 3, lines: 283, type: 'header', description: 'Metal pipeline state management and kernel launch configuration', symbols: ['KernelDispatch', 'PipelineCache', 'DispatchConfig', 'ThreadgroupSize'] },
  { name: 'kernel_dispatch.mm', path: 'src/kernels/', session: 3, lines: 399, type: 'source', description: 'KernelDispatch implementation: pipeline creation, encoder setup', symbols: ['KernelDispatch::compile', 'KernelDispatch::dispatch', 'KernelDispatch::encode'] },
  // Session 4
  { name: 'kv_cache.h', path: 'src/', session: 4, lines: 167, type: 'header', description: 'KV cache with per-layer Metal buffers and sliding window support', symbols: ['KVCache', 'KVCacheConfig', 'CacheSlice', 'EvictOldest'] },
  { name: 'kv_cache.mm', path: 'src/', session: 4, lines: 162, type: 'source', description: 'KVCache implementation: buffer allocation, append, slice operations', symbols: ['KVCache::append', 'KVCache::slice', 'KVCache::resize'] },
  { name: 'sampler.h', path: 'src/', session: 4, lines: 108, type: 'header', description: 'Token sampler with temperature, top-k, top-p, and repetition penalty', symbols: ['Sampler', 'SamplerConfig', 'LogitProcessor'] },
  { name: 'sampler.cpp', path: 'src/', session: 4, lines: 191, type: 'source', description: 'Sampler implementation: softmax, multinomial, argmax paths', symbols: ['Sampler::sample', 'Sampler::apply_temperature', 'Sampler::top_p_filter'] },
  { name: 'model_runner.h', path: 'src/', session: 4, lines: 283, type: 'header', description: 'Complete model forward pass: embed → 94 layers → logits', symbols: ['ModelRunner', 'ForwardConfig', 'LayerOutput', 'LogitsResult'] },
  { name: 'model_runner.mm', path: 'src/', session: 4, lines: 734, type: 'source', description: 'ModelRunner implementation: the main inference loop over 94 layers', symbols: ['ModelRunner::forward', 'ModelRunner::run_layer', 'ModelRunner::compute_logits'] },
  { name: 'inference_engine.h', path: 'src/', session: 4, lines: 199, type: 'header', description: 'High-level inference API: generate(), stream(), and batch support', symbols: ['InferenceEngine', 'GenerateConfig', 'StreamCallback', 'StopCriteria'] },
  { name: 'inference_engine.mm', path: 'src/', session: 4, lines: 325, type: 'source', description: 'InferenceEngine implementation: autoregressive generation loop', symbols: ['InferenceEngine::generate', 'InferenceEngine::stream', 'InferenceEngine::check_stop'] },
  { name: 'generate_main.cpp', path: 'tools/', session: 4, lines: 174, type: 'source', description: 'CLI tool for raw text generation (no chat template)', symbols: ['main', 'parse_args', 'run_generation'] },
  // Session 5
  { name: 'tokenizer.h', path: 'src/', session: 5, lines: 135, type: 'header', description: 'BPE tokenizer with special token handling for Qwen vocabulary', symbols: ['Tokenizer', 'TokenizerConfig', 'SpecialTokens'] },
  { name: 'tokenizer.cpp', path: 'src/', session: 5, lines: 433, type: 'source', description: 'Tokenizer implementation: BPE merge, encode/decode, special tokens', symbols: ['Tokenizer::encode', 'Tokenizer::decode', 'Tokenizer::add_special'] },
  { name: 'chat_template.h', path: 'src/', session: 5, lines: 129, type: 'header', description: 'ChatML template engine for multi-turn conversations', symbols: ['ChatTemplate', 'Message', 'Role', 'FormatOptions'] },
  { name: 'chat_template.cpp', path: 'src/', session: 5, lines: 257, type: 'source', description: 'ChatTemplate implementation: ChatML formatting with tool support', symbols: ['ChatTemplate::format', 'ChatTemplate::append_tool_result'] },
  { name: 'tool_schema.h', path: 'src/', session: 5, lines: 98, type: 'header', description: 'JSON Schema definitions for tool function declarations', symbols: ['ToolSchema', 'ParameterDef', 'FunctionDef'] },
  { name: 'tool_parser.h', path: 'src/', session: 5, lines: 70, type: 'header', description: 'XML parser for <tool_call> extraction from model output', symbols: ['ToolParser', 'ParsedCall'] },
  { name: 'tool_parser.cpp', path: 'src/', session: 5, lines: 337, type: 'source', description: 'ToolParser implementation: streaming XML extraction with soft-stop sequences', symbols: ['ToolParser::parse', 'ToolParser::extract_xml', 'ToolParser::check_soft_stop'] },
  { name: 'tool_engine.h', path: 'src/', session: 5, lines: 166, type: 'header', description: 'Agentic tool execution engine with handler registry', symbols: ['ToolEngine', 'ToolHandler', 'ToolResult', 'HandlerRegistry'] },
  { name: 'tool_engine.cpp', path: 'src/', session: 5, lines: 272, type: 'source', description: 'ToolEngine implementation: dispatch, execute, result formatting', symbols: ['ToolEngine::register_handler', 'ToolEngine::dispatch', 'ToolEngine::execute'] },
  { name: 'chat_main.cpp', path: 'tools/', session: 5, lines: 431, type: 'source', description: 'Interactive chat CLI with tool-calling and multi-turn support', symbols: ['main', 'chat_loop', 'handle_tool_call', 'display_response'] },
  // Build
  { name: 'CMakeLists.txt', path: '', session: 0, lines: 83, type: 'cmake', description: 'CMake build configuration for macOS with Metal framework linking', symbols: ['project(FinchMoE)', 'find_library(Metal)', 'add_executable'] },
  { name: 'README.md', path: '', session: 0, lines: 94, type: 'markdown', description: 'Project documentation: build instructions, usage, architecture overview', symbols: ['Build', 'Usage', 'Architecture', 'License'] },
  { name: 'engine_test.cpp', path: 'test/', session: 0, lines: 58, type: 'test', description: 'Basic engine integration test: load model, generate one token', symbols: ['test_load_model', 'test_generate_token', 'test_expert_cache'] },
];

export const TOTAL_FILES = SOURCE_FILES?.length ?? 0;
export const TOTAL_LINES = SOURCE_FILES?.reduce?.((sum: number, f: SourceFile) => sum + (f?.lines ?? 0), 0) ?? 0;
export const TOTAL_SESSIONS = 5;

export interface ArchitectureLayer {
  id: string;
  label: string;
  components: string[];
  session: number;
  color: string;
  description: string;
  files: string[];
}

export const ARCHITECTURE_LAYERS: ArchitectureLayer[] = [
  {
    id: 'cli',
    label: 'CLI Tools',
    components: ['finchmoe_chat', 'finchmoe_generate'],
    session: 5,
    color: '#f59e0b',
    description: 'Command-line interfaces for interactive chat and raw text generation',
    files: ['chat_main.cpp', 'generate_main.cpp'],
  },
  {
    id: 'tooling',
    label: 'Tool-Calling Layer',
    components: ['ToolEngine', 'ChatTemplate', 'Tokenizer'],
    session: 5,
    color: '#f59e0b',
    description: 'Agentic tool execution: XML parsing, handler registry, ChatML formatting, BPE tokenization',
    files: ['tool_engine.h', 'tool_engine.cpp', 'chat_template.h', 'chat_template.cpp', 'tokenizer.h', 'tokenizer.cpp', 'tool_parser.h', 'tool_parser.cpp', 'tool_schema.h'],
  },
  {
    id: 'inference',
    label: 'Inference Engine',
    components: ['InferenceEngine', 'ModelRunner', 'Sampler'],
    session: 4,
    color: '#10b981',
    description: 'Autoregressive generation loop: forward pass over 94 layers, logits computation, token sampling',
    files: ['inference_engine.h', 'inference_engine.mm', 'model_runner.h', 'model_runner.mm', 'sampler.h', 'sampler.cpp', 'kv_cache.h', 'kv_cache.mm'],
  },
  {
    id: 'kernels',
    label: 'Metal Kernels',
    components: ['KernelDispatch', 'RoPE', 'Norms', 'GEMV', 'Attention', 'MoE'],
    session: 3,
    color: '#8b5cf6',
    description: 'GPU compute kernels: rotary embeddings, RMSNorm, matrix-vector multiply, GQA attention, MoE dispatch',
    files: ['kernel_dispatch.h', 'kernel_dispatch.mm', 'rope_kernels.metal', 'norm_kernels.metal', 'gemv_kernels.metal', 'attention_kernels.metal', 'moe_kernels.metal'],
  },
  {
    id: 'core',
    label: 'Engine Core',
    components: ['EngineCore', 'ExpertCache', 'IOPlanner'],
    session: 2,
    color: '#06b6d4',
    description: 'Expert caching (LFU), async I/O scheduling, and layer-by-layer execution coordination',
    files: ['engine_core.h', 'engine_core.mm', 'expert_cache.h', 'expert_cache.mm', 'io_planner.h', 'io_planner.cpp'],
  },
  {
    id: 'format',
    label: 'Wire Format',
    components: ['finchmoe_types', 'finchmoe_format', 'MemoryLedger'],
    session: 1,
    color: '#3b82f6',
    description: '.finchmoe binary format with 64KB alignment, type definitions, and real-time memory budget tracking',
    files: ['finchmoe_types.h', 'finchmoe_format.h', 'memory_ledger.h'],
  },
  {
    id: 'foundation',
    label: 'Foundation',
    components: ['.finchmoe file', 'Apple Silicon GPU'],
    session: 0,
    color: '#64748b',
    description: 'The model weights file and the Metal-capable Apple Silicon hardware',
    files: [],
  },
];

export const PIPELINE_STEPS = [
  {
    id: 'input',
    title: 'Input',
    description: 'User types a prompt. Each character appears in the input field.',
    detail: 'The raw text string is passed to the tokenizer.',
  },
  {
    id: 'tokenize',
    title: 'Tokenize',
    description: 'BPE tokenizer splits text into subword tokens.',
    detail: 'Qwen uses a 151,936 token vocabulary with byte-level BPE. Special tokens like <|im_start|> have reserved IDs.',
  },
  {
    id: 'chatml',
    title: 'ChatML Format',
    description: 'Wraps tokens with ChatML markers for multi-turn context.',
    detail: 'Format: <|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n',
  },
  {
    id: 'embedding',
    title: 'Embedding',
    description: 'Token IDs map to 4096-dimensional dense vectors.',
    detail: 'The embedding table is part of the dense backbone (~600 MB). Each token becomes a float16 vector of size 4096.',
  },
  {
    id: 'layers',
    title: 'Layer Loop',
    description: 'Process through all 94 transformer layers sequentially.',
    detail: 'Each layer: RMSNorm → Attention (GQA) → RMSNorm → MoE FFN. The hidden state is refined layer by layer.',
  },
  {
    id: 'attention',
    title: 'Attention (GQA)',
    description: 'Grouped Query Attention: 4 KV heads serve 64 query heads.',
    detail: 'Q = Wq·x (64 heads), K = Wk·x (4 heads), V = Wv·x (4 heads). Each KV head broadcasts to 16 Q heads. Scores = softmax(Q·K^T / √64) · V.',
  },
  {
    id: 'moe',
    title: 'MoE Routing',
    description: 'Gate network selects top-8 of 128 experts per token.',
    detail: 'Gate: g = softmax(Wg·x) → top-8 indices + weights. Each expert is a 768-wide FFN (gate+up+down, 18 MB). Experts may need to be loaded from disk into the GPU cache.',
  },
  {
    id: 'sampling',
    title: 'Sampling',
    description: 'Convert logits to probabilities and sample next token.',
    detail: 'Apply temperature scaling, top-k filtering, top-p (nucleus) sampling, then multinomial sample from the filtered distribution.',
  },
  {
    id: 'output',
    title: 'Output',
    description: 'Decoded token is appended to the response.',
    detail: 'The sampled token ID is decoded back to text via the tokenizer. Generation continues until stop token (151645) or max length.',
  },
];

export const TOOL_CALL_DEMO_STEPS = [
  {
    id: 'user-prompt',
    label: 'User Prompt',
    node: 'user',
    content: 'What is sqrt(144) × the current hour?',
    type: 'user' as const,
  },
  {
    id: 'format-chatml',
    label: 'Format ChatML',
    node: 'format',
    content: '<|im_start|>user\nWhat is sqrt(144) × the current hour?<|im_end|>\n<|im_start|>assistant',
    type: 'system' as const,
  },
  {
    id: 'generate-1',
    label: 'Generate (call 1)',
    node: 'generate',
    content: '<tool_call>\n<name>calculator</name>\n<arguments>{"expression": "sqrt(144)"}</arguments>\n</tool_call>',
    type: 'model' as const,
  },
  {
    id: 'parse-1',
    label: 'Parse Tool Call',
    node: 'parse',
    content: 'Extracted: calculator({"expression": "sqrt(144)"})',
    type: 'system' as const,
  },
  {
    id: 'execute-1',
    label: 'Execute: calculator',
    node: 'dispatch',
    content: '<tool_response>\n{"result": 12}\n</tool_response>',
    type: 'tool' as const,
  },
  {
    id: 'generate-2',
    label: 'Generate (call 2)',
    node: 'generate',
    content: '<tool_call>\n<name>get_time</name>\n<arguments>{}</arguments>\n</tool_call>',
    type: 'model' as const,
  },
  {
    id: 'parse-2',
    label: 'Parse Tool Call',
    node: 'parse',
    content: 'Extracted: get_time({})',
    type: 'system' as const,
  },
  {
    id: 'execute-2',
    label: 'Execute: get_time',
    node: 'dispatch',
    content: '<tool_response>\n{"hour": 12, "time": "12:00 PM"}\n</tool_response>',
    type: 'tool' as const,
  },
  {
    id: 'generate-3',
    label: 'Generate (final)',
    node: 'generate',
    content: 'sqrt(144) = 12, and the current hour is 12.\n12 × 12 = **144**.',
    type: 'model' as const,
  },
  {
    id: 'final',
    label: 'Final Answer',
    node: 'output',
    content: 'The answer is 144.',
    type: 'final' as const,
  },
];

cmake_minimum_required(VERSION 3.22)
project(finchmoe_engine LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD 20)

# ── macOS / Apple Silicon only ────────────────────────────────────────────────
if(NOT APPLE)
    message(FATAL_ERROR "FinchMoE engine requires macOS (Apple Silicon)")
endif()

# ── compiler flags ────────────────────────────────────────────────────────────
add_compile_options(
    -O3
    -march=armv8.5-a+fp16+bf16   # enable NEON bf16 / fp16 native
    -ffast-math
    -Wall -Wextra -Wno-unused-parameter
)

# ── Metal framework ───────────────────────────────────────────────────────────
find_library(METAL_LIB Metal REQUIRED)
find_library(FOUNDATION_LIB Foundation REQUIRED)

# ── engine library ────────────────────────────────────────────────────────────
add_library(finchmoe_engine STATIC
    # Session 2 — engine core
    expert_cache.mm
    io_planner.cpp
    engine_core.mm

    # Session 3 — Metal kernel dispatch
    kernels/kernel_dispatch.mm

    # Session 4 — inference loop
    kv_cache.mm
    sampler.cpp
    model_runner.mm
    inference_engine.mm

    # Session 5 — tool-calling layer (pure C++; no Metal ObjC calls)
    tokenizer.cpp
    chat_template.cpp
    tool_parser.cpp
    tool_engine.cpp
)

target_include_directories(finchmoe_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(finchmoe_engine PUBLIC ${METAL_LIB} ${FOUNDATION_LIB})

# Suppress ObjC++ mixed selector warnings that come from Metal headers
target_compile_options(finchmoe_engine PRIVATE -Wno-deprecated-declarations)

# ── optional test binary ──────────────────────────────────────────────────────
add_executable(finchmoe_test test/engine_test.cpp)
target_link_libraries(finchmoe_test PRIVATE finchmoe_engine)

# ── simple generate smoke-test ────────────────────────────────────────────────
# Build with:  cmake -B build && cmake --build build --target finchmoe_generate
# Run with:    ./build/finchmoe_generate /path/to/model.finchmoe "Hello"
add_executable(finchmoe_generate
    tools/generate_main.cpp
)
target_link_libraries(finchmoe_generate PRIVATE finchmoe_engine)

# ── interactive chat CLI (Session 5) ─────────────────────────────────────────
# Build with:  cmake --build build --target finchmoe_chat
# Run with:    ./build/finchmoe_chat /path/to/model.finchmoe /path/to/model.tiktoken
#
# Flags:
#   --system  "You are a helpful assistant."   (optional system prompt)
#   --ctx     4096                             (KV context window, default 4096)
#   --temp    0.7                              (sampling temperature)
#   --top-p   0.9                              (nucleus sampling p)
#   --top-k   50                               (top-k sampling)
#
# Built-in demo tools exposed to the model:
#   calculator  –  evaluate arithmetic expressions
#   get_time    –  return current UTC time
#   echo        –  echo a message (round-trip sanity check)
add_executable(finchmoe_chat
    tools/chat_main.cpp
)
target_link_libraries(finchmoe_chat PRIVATE finchmoe_engine)

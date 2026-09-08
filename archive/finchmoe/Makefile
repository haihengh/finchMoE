# Makefile for finchmoe — C/Metal Qwen3.6-35B-A3B inference engine
#
# Targets:
#   make              — build finchmoe infer binary
#   make chat         — build interactive chat TUI
#   make extract      — extract non-expert weights from safetensors
#   make index        — generate expert_index.json
#   make repack       — repack expert weights (all layers)
#   make repack L=0-4 — repack specific layers
#   make clean        — remove build artifacts
#
# Based on flash-moe's metal_infer/Makefile, adapted for Qwen3.6-35B-A3B.

CC = clang
# -mmacosx-version-min=13.0 keeps the binary runnable on older Apple
# Silicon Macs (M1 mini etc.) — the default SDK targets the host OS only.
CFLAGS = -O2 -g -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK -mmacosx-version-min=13.0
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

TARGET = finchmoe-infer
SRC = infer.m

CHAT_TARGET = chat
CHAT_SRC = chat.m

# Optional: pre-compiled Metal library
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal

MODEL_DIR = ../models/Qwen3.6-35B-A3B-4bit-custom

.PHONY: all clean extract index repack chat

all: $(TARGET)

$(TARGET): $(SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(SRC) -o $(TARGET)

$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(CHAT_SRC) linenoise.c -o $(CHAT_TARGET)

chat: $(CHAT_TARGET)

clean:
	rm -f $(TARGET) $(CHAT_TARGET)

# Model preparation targets
extract:
	python3 extract_weights.py --model $(MODEL_DIR) --output .

index:
	python3 generate_expert_index.py --model $(MODEL_DIR)

repack:
	python3 repack_experts.py --index expert_index.json $(if $(L),--layers $(L),)

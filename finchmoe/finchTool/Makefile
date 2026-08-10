# Makefile for finchTool — Metal Engine Verification & Diagnostic Suite
#
# Builds a standalone diagnostic binary that shares shaders.metal with finchmoe-infer
# but has its own minimal Metal setup and independent test harness.
#
# Targets:
#   make              — build finchTool
#   make clean        — remove finchTool
#   make test         — build and run quick sanity check (kernel all)
#   make test-verbose — build and run with verbose output

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc
FRAMEWORKS = -framework Metal -framework Foundation
LDFLAGS =

TARGET = finchTool
SRCS = main.m engine_utils.m verify_core.m
OBJS = $(SRCS:.m=.o)
HEADERS = engine_utils.h verify_core.h

# Shader search paths (relative to finchTool binary location)
# The diagnostic tool looks for shaders.metal in: . , .. , metal_infer/
# Running from finchmoe/finchTool/: ../shaders.metal resolves to finchmoe/shaders.metal

.PHONY: all clean test test-verbose

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(SRCS) -o $(TARGET)

test: $(TARGET)
	./$(TARGET) kernel --test all

test-verbose: $(TARGET)
	./$(TARGET) kernel --test fused_mlp --verbose

clean:
	rm -f $(TARGET) *.o

#!/usr/bin/env python3
"""
Debug script: Run MLX inference on the quantized model to verify correctness.
If this produces coherent output, the quantization is correct and the bug is in the C engine.
If this produces garbage, the quantization itself is broken.

Usage: python debug_mlx_inference.py [--custom | --community] [--prompt "Hello"]
"""

import argparse
import sys
import os
import json
import numpy as np

import mlx.core as mx
from mlx_lm import load, generate
from transformers import AutoTokenizer


# Resolve paths relative to project root (parent of finchmoe/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_CUSTOM = os.path.join(_PROJECT_ROOT, "models/Qwen3.6-35B-A3B-4bit-custom")
MODEL_COMMUNITY = os.path.join(_PROJECT_ROOT, "models/Qwen3.6-35B-A3B-4bit")
MODEL_BF16 = os.path.join(_PROJECT_ROOT, "models/Qwen3.6-35B-A3B-bf16")


def test_inference(model_path, prompt, max_tokens=50):
    """Run inference and print results."""
    print(f"\n{'='*80}")
    print(f"Model: {model_path}")
    print(f"Prompt: {prompt!r}")
    print(f"Max tokens: {max_tokens}")
    print(f"{'='*80}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
    print(f"Model loaded. Vocab size: {tokenizer.vocab_size}")

    # Tokenize
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"Chat template applied: {text[-200:]!r}")
    else:
        text = prompt

    tokens = tokenizer.encode(text)
    print(f"Input tokens ({len(tokens)}): {tokens[:20]}...")

    # Generate
    print(f"\n--- Generating {max_tokens} tokens ---")
    response = generate(
        model,
        tokenizer,
        prompt=text,
        max_tokens=max_tokens,
        verbose=True,
    )
    print(f"\n--- Output ---")
    print(response)
    return response


def main():
    parser = argparse.ArgumentParser(description="MLX inference debug")
    parser.add_argument("--custom", action="store_true", help="Use self-quantized model")
    parser.add_argument("--community", action="store_true", help="Use mlx-community model")
    parser.add_argument("--bf16", action="store_true", help="Use original BF16 model")
    parser.add_argument("--prompt", default="Hello, what is 2+2?", help="Input prompt")
    parser.add_argument("--tokens", type=int, default=30, help="Max tokens to generate")
    args = parser.parse_args()

    # Default to custom unless specified
    if args.bf16:
        model_path = MODEL_BF16
    elif args.community:
        model_path = MODEL_COMMUNITY
    else:
        model_path = MODEL_CUSTOM

    # Check model exists
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    test_inference(model_path, args.prompt, args.tokens)


if __name__ == "__main__":
    main()

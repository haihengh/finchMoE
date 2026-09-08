#!/usr/bin/env python3
"""test_shim.py — unit tests for shim.py's SSE reassembly.

The byte round-trip is the one place a silent bug would corrupt every result,
so it is tested without needing the engine running.
    python3 test_shim.py
"""
import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location(
    "shim", os.path.join(os.path.dirname(os.path.abspath(__file__)), "shim.py"))
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)

DONE = (b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":'
        b'[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":'
        b'{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}'
        b'\n\ndata: [DONE]\n\n')


def ev(content):
    """One SSE event, escaped the way infer.m's sse_send_delta does it."""
    payload = json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": content},
                     "finish_reason": None}]})
    return b"data: " + payload.encode()


def test_byte_roundtrip():
    # "café ✓" -- the engine escapes each raw UTF-8 byte as \u00XX,
    # so multi-byte characters arrive split across chunks.
    raw = "café ✓\ndef f():\n\treturn 1".encode("utf-8")
    parts = [chr(b) for b in raw]          # one chunk per byte: worst case
    body = b"\n\n".join(ev(p) for p in parts) + b"\n\n" + DONE

    text, usage, rid = shim.decode_sse(body)
    assert text == "café ✓\ndef f():\n\treturn 1", repr(text)
    assert usage["completion_tokens"] == 8, usage
    assert rid == "chatcmpl-1", rid
    print("PASS byte round-trip (per-byte chunks):", repr(text))


def test_malformed_chunk_skipped():
    body = ev("ok ") + b"\n\ndata: {broken\n\n" + ev("still here") + b"\n\n" + DONE
    text, _, _ = shim.decode_sse(body)
    assert text == "ok still here", repr(text)
    print("PASS malformed chunk skipped:", repr(text))


def test_no_usage_chunk():
    # engine killed mid-stream: no finish chunk, no [DONE]
    text, usage, _ = shim.decode_sse(ev("partial") + b"\n\n")
    assert text == "partial" and usage is None
    print("PASS truncated stream tolerated")


def test_think_stripping():
    assert shim.strip_think("<think>reason</think>code()") == "code()"
    assert shim.strip_think("code()<think>dangling") == "code()"
    assert shim.strip_think("plain") == "plain"
    assert shim.strip_think("<think>a</think>x<think>b</think>y") == "xy"
    print("PASS think stripping (closed, unclosed, multiple)")


def test_clean_for_engine():
    got = shim.clean_for_engine("a\r\nb\tc\x00d\x07e")
    assert got == "a\nb\tcde", repr(got)
    assert shim.clean_for_engine("x\rz") == "x\nz"
    print("PASS clean_for_engine:", repr(got))


def test_extract_user_content():
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"}]
    assert shim.extract_user_content(msgs) == "hello"
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"},
                                          {"type": "text", "text": "b"}]}]
    assert shim.extract_user_content(parts) == "ab"
    assert shim.extract_user_content([]) is None
    print("PASS extract_user_content (string, parts, empty)")


if __name__ == "__main__":
    for fn in [test_byte_roundtrip, test_malformed_chunk_skipped,
               test_no_usage_chunk, test_think_stripping,
               test_clean_for_engine, test_extract_user_content]:
        fn()
    print("\nALL SHIM UNIT TESTS PASSED")

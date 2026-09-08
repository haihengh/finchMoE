#!/usr/bin/env python3
"""shim.py — OpenAI-compatible front end for finchmoe-infer's serve mode.

EvalPlus drives models through the official `openai` SDK, which issues a
NON-streaming POST /v1/chat/completions and expects a single JSON body.
finchmoe-infer only ever answers with an SSE stream, ignores the request's
system message, and ignores temperature/top_p/n in the body (sampling is set by
CLI flags). This shim bridges that gap so the exact EvalPlus harness that
produced the 3090 numbers can run against our engine unmodified.

    evalplus --openai SDK--> shim.py :8080 --SSE--> finchmoe-infer :9000

Engine wire contract (finchmoe/infer.m):
  - reply is close-delimited: Connection: close, no Content-Length, no chunked
    framing (SSE_HEADERS, infer.m:13636) -> read to EOF
  - events are LF-only `data: {...}\\n\\n`; no role-only preamble
  - terminator is a finish_reason:"stop" chunk carrying usage, immediately
    followed by `data: [DONE]`, both in ONE write (sse_send_done, infer.m:13564)
  - finish_reason is ALWAYS "stop", even on truncation (infer.m:14337) --
    completion_tokens hitting max_tokens is the only truncation signal
  - every byte >= 0x80 is escaped as \\u00XX -- RAW BYTES, not codepoints
    (sse_send_delta, infer.m:13528)

Usage: shim.py [--port 8080] [--upstream-port 9000] [--keep-think]
"""
import argparse
import http.client
import json
import re
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 9000
CONNECT_TIMEOUT = 10      # engine accept() should be instant once up
READ_TIMEOUT = 600        # engine writes continuously while generating;
                          # 120 s of silence means it is wedged
MAX_503_RETRIES = 8
STRIP_THINK = True

# The engine serialises generation anyway AND keeps a single global session,
# so more than one in-flight upstream request would clobber state.
_upstream_lock = threading.Lock()

_THINK_BLOCK = re.compile(r"(?s)<think>.*?</think>\s*")


def log(msg):
    print(f"[shim] {msg}", file=sys.stderr, flush=True)


def clean_for_engine(text):
    """Make text safe for the engine's minimal unescaper.

    extract_last_content (infer.m:13368) only unescapes \\n \\t \\" \\\\ --
    everything else keeps its backslash and lands in the prompt as literal
    characters. We forward with ensure_ascii=False so non-ASCII stays raw, but
    C0 control characters would still be emitted as \\uXXXX by json.dumps, so
    strip them here. \\n and \\t are safe and preserved.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(c for c in text if c >= " " or c in "\n\t")


def extract_user_content(messages):
    """Last user message, flattened. The engine reads only ONE content string."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # OpenAI content-parts form; the engine cannot parse arrays at all.
            return "".join(p.get("text", "") for p in content
                           if isinstance(p, dict) and p.get("type") == "text")
    return None


def decode_sse(body):
    """Reassemble the engine's SSE stream into (text, usage, request_id).

    The byte round-trip is the subtle part. The engine escapes each raw byte
    >= 0x80 as \\u00XX, so a multi-byte UTF-8 character arrives as several
    escapes and may be split across chunks. json.loads turns \\u00XX into the
    codepoint U+00XX, whose value IS the original byte, so latin-1 recovers it
    exactly. Encoding as UTF-8 here would re-encode U+00E9 as 0xC3 0xA9 and
    produce mojibake -- that is the classic bug this function exists to avoid.
    """
    buf = bytearray()
    usage = None
    request_id = None
    bad_chunks = 0

    for raw_event in body.split(b"\n\n"):
        raw_event = raw_event.strip()
        if not raw_event.startswith(b"data:"):
            continue
        payload = raw_event[len(b"data:"):].strip()
        if payload == b"[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # The engine's per-token escape buffer is 2 KB; a pathological
            # token could truncate mid-escape. Skip that one event.
            bad_chunks += 1
            continue

        if request_id is None:
            request_id = chunk.get("id")
        if chunk.get("usage"):
            usage = chunk["usage"]

        choices = chunk.get("choices") or [{}]
        content = (choices[0].get("delta") or {}).get("content")
        if not content:
            continue
        try:
            buf.extend(content.encode("latin-1"))
        except UnicodeEncodeError:
            # Should not happen given the engine's escaping, but never drop text.
            buf.extend(content.encode("utf-8"))

    if bad_chunks:
        log(f"WARNING: skipped {bad_chunks} unparseable SSE chunk(s)")

    text = bytes(buf).decode("utf-8", errors="replace")
    return text, usage, request_id


def strip_think(text):
    """Remove <think>...</think>, including an unclosed trailing block.

    --no-think does NOT suppress reasoning in this engine: it only prepends an
    empty think block as a hint, and the per-token SSE send is unconditional
    (infer.m:14242). An EOS mid-think can also leave the block unclosed.
    """
    text = _THINK_BLOCK.sub("", text)
    i = text.find("<think>")
    if i >= 0:
        text = text[:i]
    return text


def call_engine(content, max_tokens):
    """POST to the engine and return (text, usage, request_id). Retries 503."""
    body = json.dumps(
        {"messages": [{"role": "user", "content": content}],
         "max_tokens": max_tokens},
        ensure_ascii=False,          # the engine cannot unescape \uXXXX
    ).encode("utf-8")

    for attempt in range(MAX_503_RETRIES):
        conn = None
        try:
            conn = http.client.HTTPConnection(
                UPSTREAM_HOST, UPSTREAM_PORT, timeout=CONNECT_TIMEOUT)
            conn.connect()
            conn.sock.settimeout(READ_TIMEOUT)
            conn.request("POST", "/v1/chat/completions", body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()

            if resp.status == 503:
                resp.read()
                wait = int(resp.getheader("Retry-After") or 3)
                log(f"engine busy (503), retry in {wait}s "
                    f"[{attempt + 1}/{MAX_503_RETRIES}]")
                time.sleep(wait)
                continue
            if resp.status != 200:
                detail = resp.read()[:200].decode("utf-8", "replace")
                raise RuntimeError(f"engine HTTP {resp.status}: {detail}")

            # Close-delimited body: read() loops until EOF, and each underlying
            # recv carries READ_TIMEOUT, so a stalled engine raises rather than
            # hanging forever.
            return decode_sse(resp.read())
        except (socket.timeout, TimeoutError):
            raise RuntimeError(f"engine stalled >{READ_TIMEOUT}s with no output")
        except (ConnectionError, http.client.HTTPException) as e:
            log(f"engine connection error ({e}), retry in 3s "
                f"[{attempt + 1}/{MAX_503_RETRIES}]")
            time.sleep(3)
        finally:
            if conn is not None:
                conn.close()

    raise RuntimeError(f"engine unavailable after {MAX_503_RETRIES} attempts")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # requires an accurate Content-Length

    def log_message(self, fmt, *args):
        pass  # we do our own logging

    def _send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [{"id": self.server.model_id, "object": "model",
                          "created": 0, "owned_by": "finchmoe"}],
            })
        elif self.path.rstrip("/") == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions",
                                         "/chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return

        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError) as e:
            self._send_json(400, {"error": {"message": f"bad request: {e}"}})
            return

        content = extract_user_content(req.get("messages"))
        if not content:
            self._send_json(400, {"error": {"message": "no user content"}})
            return

        # The engine ignores temperature/top_p/n/stream/model and the system
        # message; sampling comes from its CLI flags. Dropping them here is
        # honest -- forwarding would imply they take effect.
        max_tokens = req.get("max_completion_tokens") or req.get("max_tokens") or 768
        model = req.get("model") or self.server.model_id

        t0 = time.time()
        try:
            with _upstream_lock:
                text, usage, rid = call_engine(clean_for_engine(content),
                                               max_tokens)
        except Exception as e:
            log(f"ERROR: {e}")
            self._send_json(502, {"error": {"message": str(e)}})
            return

        if self.server.strip_think:
            text = strip_think(text)

        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens, completion_tokens = 0, 0

        elapsed = time.time() - t0
        trunc = " TRUNCATED" if completion_tokens >= max_tokens else ""
        log(f"{rid or '-'} {elapsed:.1f}s "
            f"{completion_tokens}/{max_tokens} tok {len(text)} chars{trunc}")

        self._send_json(200, {
            "id": rid or "chatcmpl-shim",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",   # all the engine ever reports
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        })


def main():
    global UPSTREAM_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--upstream-port", type=int, default=9000)
    ap.add_argument("--model-id", default="qwen3.6-35b-a3b")
    ap.add_argument("--keep-think", action="store_true",
                    help="do not strip <think> blocks (diagnostics only)")
    args = ap.parse_args()

    UPSTREAM_PORT = args.upstream_port

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.model_id = args.model_id
    server.strip_think = not args.keep_think
    log(f"listening on http://127.0.0.1:{args.port}/v1 "
        f"-> engine 127.0.0.1:{UPSTREAM_PORT} "
        f"(strip_think={server.strip_think})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()

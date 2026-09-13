#!/bin/bash
#
# run_server_cell.sh — an EvalPlus HumanEval cell against a FinchMoEServer install.
#
# Why this exists next to run_cell.sh rather than inside it: run_cell.sh drives
# the *archive-era C engine* through shim.py, and that engine physically cannot
# load a 125B install — `archive/finchmoe/infer.m:145,151,152` are compile-time
# (`#define NUM_LAYERS 40`, `NUM_EXPERTS 256`, `NUM_EXPERTS_PER_TOK 8`) and the
# string `qwen3_8` appears nowhere in that file. The Swift engine needs no shim:
# FinchMoEServer is already OpenAI-compatible, so evalplus talks to it directly
# on 8080.
#
# That is the topology docs/QWEN36_PORT.md §6 used for the Qwen 3.6 reference
# run (base 0.909, plus 0.878) — but it was driven by hand and the command was
# never written down, so neither that run nor its protocol could be re-checked
# afterwards. This script is that missing record. Any published number should be
# reproducible by re-running this file.
#
# The PROTOCOL is deliberately not defined here — it lives in humaneval_gen.py
# (system "You are a helpful assistant good at coding.", greedy T=0, 1 sample,
# top_p 0.95, 768-token cap via evalplus's OpenAIChatDecoder). Do not add
# sampling flags here: sharing that file is what makes cells comparable across
# engines, and it is why a 3.8 cell can be read against the 3.6 one.
#
# Usage:
#   run_server_cell.sh <model-id> <model-dir> [start end]
#
#   # smoke slice first — proves the protocol before committing hours
#   run_server_cell.sh finchmoe-qwen38 "models/Qwen3.8-Flash-Next-125B.finch" 0 20
#   # full 164-problem sweep
#   run_server_cell.sh finchmoe-qwen38 "models/Qwen3.8-Flash-Next-125B.finch"
#
# <model-id> is sent as the OpenAI `model` field, and OpenAIModels.swift rejects
# any request whose model != the server's --model-id. Both come from $1 here on
# purpose: a mismatch 400s every request, and evalplus's retry loop would spin on
# it for the whole run rather than fail fast.
#
# Env: PORT_ENGINE (default 8080), MAX_CONTEXT (default 4096), VERIFY_MODE
#      (default trusted-install).
#
set -uo pipefail

MODEL_ID=${1:?usage: run_server_cell.sh <model-id> <model-dir> [start end]}
MODEL_DIR=${2:?usage: run_server_cell.sh <model-id> <model-dir> [start end]}
START=${3:-}
END=${4:-}

BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$BASE/../.." && pwd)
PORT=${PORT_ENGINE:-8080}
CTX=${MAX_CONTEXT:-4096}
# trusted-install skips the SHA-256 over ~167 GB of layer+PLE files that
# otherwise lands in the first prefill of every request. It still verifies the
# receipt, including a physical-path binding: the receipt records the lowercase
# checkout path (".../code/finchmoe/models/...") while --model is given through
# ".../finchMoE/models/...". Both survive that compare because BOTH sides
# resolve the models/ symlink, and the resulting ".." collapses the component
# that differs. Checked against Foundation before the first run rather than
# assumed — a mismatch here throws trustedReceiptInvalid and the server never
# loads, so it is worth not guessing at.
VERIFY_MODE=${VERIFY_MODE:-trusted-install}
PY="$BASE/.venv/bin/python"
SERVER_BIN="$REPO/.build/release/FinchMoEServer"
SERVER_LOG="$BASE/results/${MODEL_ID}_server.log"

case "$MODEL_DIR" in /*) MODEL_PATH="$MODEL_DIR" ;; *) MODEL_PATH="$REPO/$MODEL_DIR" ;; esac

mkdir -p "$BASE/results"
[ -x "$SERVER_BIN" ] || { echo "ABORT: no server binary at $SERVER_BIN — build it first"; exit 1; }
[ -x "$PY" ]         || { echo "ABORT: no venv python at $PY — run setup.sh"; exit 1; }
[ -f "$MODEL_PATH/manifest.json" ] || { echo "ABORT: no manifest.json under $MODEL_PATH"; exit 1; }

# One engine at a time on this box: three of its panics were concurrent model
# runs. Refuse rather than queue.
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "ABORT: port $PORT is already serving — refusing to start a second engine"
  exit 1
fi

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cell] stopping server $SERVER_PID"
    kill -TERM "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

echo "[cell] model-id=$MODEL_ID ctx=$CTX port=$PORT verify=$VERIFY_MODE"
echo "[cell] model=$MODEL_PATH"
echo "[cell] starting server -> $SERVER_LOG"
"$SERVER_BIN" --model "$MODEL_PATH" --port "$PORT" --model-id "$MODEL_ID" \
  --max-context "$CTX" --verify "$VERIFY_MODE" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[cell] server pid $SERVER_PID"

# Loading a 125B install is not fast; 600s is generous, and the health gate is
# what stops the generator from hammering a socket that is not listening yet.
for i in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ABORT: server exited during load. Last lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  if [ "$i" = 600 ]; then
    echo "ABORT: server not healthy after 600s. Last lines:"; tail -30 "$SERVER_LOG"; exit 1
  fi
  sleep 1
done
echo "[cell] server healthy after ${i}s"

if [ -n "$START" ] && [ -n "$END" ]; then
  echo "[cell] generating range $START..$END"
  OPENAI_API_KEY=none "$PY" "$BASE/humaneval_gen.py" "$MODEL_ID" "$START" "$END" || exit 1
else
  echo "[cell] generating full set"
  OPENAI_API_KEY=none "$PY" "$BASE/humaneval_gen.py" "$MODEL_ID" || exit 1
fi

STEM="$BASE/results/humaneval/${MODEL_ID}_openai_temp_0.0"

if [ -n "$START" ] && [ -n "$END" ]; then
  # evalplus refuses to score a partial file ("Missing problems in samples"),
  # which is fine: the smoke exists to prove the protocol, not to score.
  echo
  echo "[cell] partial run — running protocol diagnostics instead of scoring"
  "$PY" "$BASE/smoke_check.py" "$MODEL_ID" || true
  echo
  echo "[cell] GATE: fenced >= 0.8 and think == 0 before committing to the full sweep."
  exit 0
fi

echo "[cell] scoring with evalplus (base + HumanEval+)"
"$PY" -m evalplus.evaluate --dataset humaneval --samples "$STEM.jsonl" \
  --i-just-wanna-run 2>&1 | tee "$BASE/results/${MODEL_ID}_eval.txt"

# Mirror the reference run's artifact location so the 3.8 cell sits beside the
# 3.6 one instead of only in the harness's own tree.
echo "[cell] publishing artifacts to quality/humaneval/"
mkdir -p "$REPO/quality/humaneval"
for f in "$STEM.jsonl" "$STEM.raw.jsonl" "${STEM}_eval_results.json"; do
  [ -f "$f" ] && cp "$f" "$REPO/quality/humaneval/"
done
ls -l "$REPO/quality/humaneval/" | grep -- "$MODEL_ID" || true
echo "[cell] done: $MODEL_ID"

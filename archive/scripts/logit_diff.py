#!/usr/bin/env python3
"""logit_diff.py — §15.4 of protocol_improvement.md.

Find the first generation step where finchMoE's greedy argmax diverges from
llama.cpp's greedy argmax on the identical GGUF file and prompt.

Inputs:
  --finchmoelog  engine log containing [logit-diag] blocks (step/top1/entropy)
  --llamacpp     llama-server response JSON with per-token logprobs (n_probs=20)
  --tokenizer    HF tokenizer dir (default: models/Qwen3.6-35B-A3B-bf16)

Outputs a report: matched prefix length, first divergence step with both
engines' top-5 decoded, top-20 membership checks, and the decoded text
context around the divergence.
"""
import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------- finchMoE

def parse_finchmoe_log(path):
    """Parse [logit-diag] blocks.

    Block layout (infer.m logit_diag_dump):
      [logit-diag] step=N token=ID ("text") entropy=X max_logit=Y
      [logit-diag] top-20: V(T), V(T), ...

    The token TEXT in the dump is unreliable (decode_token gets a NULL vocab
    and returns "<unk>"), and tokens may contain newlines/commas/parens, so
    we segment on the step marker and only trust the header's numeric fields.
    """
    with open(path, "r", errors="replace") as f:
        content = f.read()

    steps = []
    header_re = re.compile(r"\[logit-diag\] step=(\d+) token=(\d+)")
    matches = list(header_re.finditer(content))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        seg = content[m.start():end]
        step = int(m.group(1))
        top1_id = int(m.group(2))
        ent = maxl = None
        e = re.search(r"entropy=([\d.]+)", seg)
        if e:
            ent = float(e.group(1))
        x = re.search(r"max_logit=([\d.]+)", seg)
        if x:
            maxl = float(x.group(1))
        top20_vals = []
        t = seg.find("top-20: ")
        if t >= 0:
            vals = re.findall(r"(-?[\d.]+)\(", seg[t:])
            top20_vals = [float(v) for v in vals[:20]]
        steps.append({
            "step": step,
            "top1_id": top1_id,
            "entropy": ent,
            "max_logit": maxl,
            "top20_logit_vals": top20_vals,
            # NOTE: top-20 token IDs are not in the current dump (only values).
            "top20_ids": None,
        })
    return steps

# ------------------------------------------------------------- llama.cpp

def parse_llamacpp(path):
    with open(path) as f:
        r = json.load(f)
    if "error" in r:
        raise SystemExit(f"llama.cpp response is an error: {r['error']}")
    ch = r["choices"][0]
    lp = ch.get("logprobs")
    steps = []
    if lp and lp.get("content") is not None:
        # Modern OpenAI-style shape: content = [{token, logprob, top_logprobs}]
        for i, entry in enumerate(lp["content"]):
            top1_id = entry["token"]
            top20 = []
            for c in entry.get("top_logprobs") or []:
                top20.append({"id": c["token"], "logprob": c.get("logprob"),
                              "text": c.get("token_text") if "token_text" in c else None})
            steps.append({
                "step": i, "top1_id": top1_id,
                "top1_logprob": entry.get("logprob"),
                "top20": top20,
            })
    elif lp is not None:
        # llama.cpp legacy shape
        toks = lp.get("tokens") or []
        tlogprobs = lp.get("token_logprobs") or []
        top_lp = lp.get("top_logprobs") or []
        for i, tok in enumerate(toks):
            top20 = []
            for c in (top_lp[i] if i < len(top_lp) else []):
                top20.append({"id": c.get("token_id", c.get("id")),
                              "p": c.get("prob"),
                              "text": c.get("text")})
            steps.append({
                "step": i, "top1_id": tok,
                "top1_logprob": tlogprobs[i] if i < len(tlogprobs) else None,
                "top20": top20,
            })
    else:
        raise SystemExit(f"no logprobs in response; keys={list(ch.keys())}")
    return steps

# ------------------------------------------------------------- tokenizer

class Tok:
    """Decode token ids to text. Prefers HF transformers; falls back to a
    minimal pure-Python GGUF token-list reader."""

    def __init__(self, hf_dir=None, gguf_path=None):
        self._hf = None
        self._gguf = None
        if hf_dir and os.path.exists(hf_dir):
            try:
                from transformers import AutoTokenizer
                self._hf = AutoTokenizer.from_pretrained(hf_dir, trust_remote_code=True)
                return
            except Exception as e:
                print(f"[warn] HF tokenizer load failed: {e}", file=sys.stderr)
        if gguf_path and os.path.exists(gguf_path):
            self._gguf = self._load_gguf_tokens(gguf_path)
            return
        raise SystemExit("no tokenizer source available")

    @staticmethod
    def _load_gguf_tokens(path):
        import struct
        with open(path, "rb") as f:
            magic = f.read(4)
            assert magic == b"GGUF", "not a GGUF file"
            ver = struct.unpack("<I", f.read(4))[0]
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            def read_val(ty):
                if ty == 0:
                    return f.read(1)
                if ty == 1:
                    return struct.unpack("<b", f.read(1))[0]
                if ty == 2:
                    return struct.unpack("<B", f.read(1))[0]
                if ty == 3:
                    return struct.unpack("<h", f.read(2))[0]
                if ty == 4:
                    return struct.unpack("<H", f.read(2))[0]
                if ty == 5:
                    return struct.unpack("<i", f.read(4))[0]
                if ty == 6:
                    return struct.unpack("<I", f.read(4))[0]
                if ty == 7:
                    return struct.unpack("<q", f.read(8))[0]
                if ty == 8:
                    return struct.unpack("<Q", f.read(8))[0]
                if ty == 9:
                    return struct.unpack("<f", f.read(4))[0]
                if ty == 10:
                    return struct.unpack("<d", f.read(8))[0]
                if ty == 11:
                    return struct.unpack("<?", f.read(1))[0]
                if ty == 12:
                    n = struct.unpack("<Q", f.read(8))[0]
                    et = struct.unpack("<B", f.read(1))[0]
                    return [read_val(et) for _ in range(n)]
                if ty == 13:
                    ln = struct.unpack("<Q", f.read(8))[0]
                    return f.read(ln).decode("utf-8", "replace")
                raise ValueError(f"bad gguf type {ty}")
            for _ in range(n_tensors):
                _ = f.read(512)  # name
                _ = struct.unpack("<Q", f.read(8))[0]  # n_dims
                _ = struct.unpack("<I", f.read(4))[0]  # type
                _ = struct.unpack("<Q", f.read(8))[0]  # offset
            for _ in range(n_kv):
                name = read_val(13)
                ty = read_val(10) if False else struct.unpack("<B", f.read(1))[0]
                if name == "llama.tokenlist":
                    toks = read_val(12)
                    return [t.decode("utf-8", "replace") if isinstance(t, bytes) else t
                            for t in toks]
                read_val(ty)
        raise ValueError("llama.tokenlist not found")

    def decode(self, tid):
        if self._hf is not None:
            try:
                s = self._hf.decode([int(tid)])
                return s
            except Exception:
                return f"<id:{tid}>"
        if self._gguf is not None and 0 <= int(tid) < len(self._gguf):
            return self._gguf[int(tid)]
        return f"<id:{tid}>"

# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finchmoelog", required=True)
    ap.add_argument("--llamacpp", required=True)
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer dir")
    ap.add_argument("--gguf", default=None, help="GGUF fallback for token text")
    ap.add_argument("--context", type=int, default=8,
                    help="steps of context around the divergence")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    finch = parse_finchmoe_log(args.finchmoelog)
    llama = parse_llamacpp(args.llamacpp)
    print(f"parsed: finchMoE {len(finch)} steps, llama.cpp {len(llama)} steps")

    tok = None
    try:
        tok = Tok(args.tokenizer, args.gguf)
    except SystemExit as e:
        print(f"[warn] {e} — ids will be shown raw")

    def dec(tid):
        return tok.decode(tid) if tok else ""

    n = min(len(finch), len(llama))
    first_div = None
    for i in range(n):
        if finch[i]["top1_id"] != llama[i]["top1_id"]:
            first_div = i
            break

    rep = {
        "finchmoe_steps": len(finch),
        "llamacpp_steps": len(llama),
        "first_divergence_step": first_div,
    }

    print("\n=== matched prefix ===")
    show = min(6, n)
    for i in range(show):
        f, l = finch[i], llama[i]
        mark = "OK " if f["top1_id"] == l["top1_id"] else "!! "
        print(f"{mark}step {i}: finch={f['top1_id']} ({dec(f['top1_id'])!r}) "
              f"llama={l['top1_id']} ({dec(l['top1_id'])!r})")
    if n > 6:
        print("...")

    if first_div is None:
        print("\n*** NO DIVERGENCE within the compared prefix — greedy argmax "
              "matched on every compared step.")
        rep["classification"] = "no divergence in compared range"
    else:
        d = first_div
        f, l = finch[d], llama[d]
        print(f"\n=== FIRST DIVERGENCE at step {d} ===")
        print(f"finchMoE: top1={f['top1_id']} text={dec(f['top1_id'])!r} "
              f"entropy={f['entropy']} max_logit={f['max_logit']}")
        print(f"llama.cpp: top1={l['top1_id']} text={dec(l['top1_id'])!r} "
              f"top1_logprob={l.get('top1_logprob')}")

        # Membership checks (as far as the data allows)
        llama_top1 = l["top1_id"]
        finch_top1 = f["top1_id"]
        llama_top20_ids = [c["id"] for c in l.get("top20") or []]
        llama_top1_in_finch_top20 = None
        if f.get("top20_ids"):
            llama_top1_in_finch_top20 = llama_top1 in f["top20_ids"]
        finch_top1_in_llama_top20 = finch_top1 in llama_top20_ids

        print(f"\nllama top-5: " + ", ".join(
            f"{c['id']}({dec(c['id'])!r},p={c.get('p', c.get('logprob')):.4f}"
            for c in (l.get("top20") or [])[:5] if c.get("id") is not None))
        print(f"finch top-20 logit values: {f['top20_logit_vals'][:5]} ... "
              f"(top-20 ids not in current dump)")
        print(f"\nfinch top1 in llama top-20 ids: {finch_top1_in_llama_top20}")
        print(f"llama top1 in finch top-20 ids: {llama_top1_in_finch_top20} "
              f"({'unavailable: dump has no top-20 ids' if llama_top1_in_finch_top20 is None else ''})")

        print(f"\n=== context around divergence (steps {max(0, d - 2)}..{min(n - 1, d + args.context)} ===")
        for i in range(max(0, d - 2), min(n, d + args.context + 1)):
            ff, ll = finch[i], llama[i]
            mark = "  " if ff["top1_id"] == ll["top1_id"] else "**"
            print(f"{mark} {i}: finch={dec(ff['top1_id'])!r:<24} "
                  f"llama={dec(ll['top1_id'])!r:<24} "
                  f"{'MATCH' if ff['top1_id'] == ll['top1_id'] else 'DIVERGE'}")

        rep["divergence"] = {
            "step": d,
            "finchmoe_top1": {"id": finch_top1, "text": dec(finch_top1),
                              "entropy": f["entropy"], "max_logit": f["max_logit"],
                              "top20_logit_vals": f["top20_logit_vals"]},
            "llamacpp_top1": {"id": llama_top1, "text": dec(llama_top1),
                              "top1_logprob": l.get("top1_logprob")},
            "llamacpp_top5": [{"id": c["id"], "text": dec(c["id"]),
                               "p": c.get("p", c.get("logprob"))}
                              for c in (l.get("top20") or [])[:5]],
            "finch_top1_in_llama_top20": finch_top1_in_llama_top20,
            "llama_top1_in_finch_top20": llama_top1_in_finch_top20,
        }
        # Heuristic classification per §15.4 table
        if d <= 5:
            rep["classification"] = ("early (step<=5): prompt encoding / RoPE base / "
                                     "first layers / template difference")
        elif llama_top1_in_finch_top20:
            rep["classification"] = "precision: llama top-1 in finch top-20"
        elif finch_top1_in_llama_top20:
            rep["classification"] = "precision: finch top-1 in llama top-20 (small drift)"
        else:
            rep["classification"] = ("late + not in each other's top-20: qualitative "
                                     "divergence or attractor amplifying early drift")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\n[written] {args.json_out}")

if __name__ == "__main__":
    main()

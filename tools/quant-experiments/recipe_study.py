#!/usr/bin/env python3
"""Recipe study: which writer-side quantizer math to use for low-bit routed
experts, measured on real Qwen3.6 expert tensors.

Context (2026-10-04): the shipped `.finch` format is group-64 affine with a
BF16 scale+bias per group, and the writer picks the scale from min/max and
rounds to nearest. The 3-bit experiment on that recipe collapsed HumanEval
(149/164 -> 28/164). An offline comparison against llama.cpp's IQ2_XS format
(`scratch/iq2_vs_ours_results.md` in the experiment checkout) showed the
collapse is not a raw-error story; still, several cheap writer-side changes
have never been measured on our distribution. This script measures them.

Every strategy here changes ONLY how the writer chooses (scale, bias); the
decoded value stays `q * scale + bias`, so no runtime or kernel change is
involved.

Strategies:
  rtn           current shipping recipe: min/max scale+bias, round-to-nearest
  mse           search candidate scales around min/max, keep min squared error
  clip(p%)      clip group extremes at the p-th percentile, then rtn
  clip+mse      clipped range, then scale search
  g32           group 32 instead of 64 (doubles the side-info cost)

Usage:
  python3 tools/quant-experiments/recipe_study.py            # full sweep
  python3 tools/quant-experiments/recipe_study.py --layer 20 # another layer
"""
import argparse
import json
import mmap
import struct

import numpy as np

SNAPSHOT = "models/Qwen3.6-35B-A3B-bf16"


def load_experts(name, experts, rows_per_expert):
    idx = json.load(open(f"{SNAPSHOT}/model.safetensors.index.json"))["weight_map"]
    path = f"{SNAPSHOT}/{idx[name]}"
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    t = hdr[name]
    E, R, C = t["shape"]
    off = 8 + n + t["data_offsets"][0]
    row_ids = np.linspace(0, R - 1, rows_per_expert).astype(int)
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        parts = []
        for e in experts:
            for r in row_ids:
                start = off + (e * R + r) * C * 2
                parts.append(np.frombuffer(mm, dtype=np.uint16, count=C, offset=start))
        arr = np.concatenate(parts)
    return (arr.astype(np.uint32) << 16).view(np.float32).reshape(-1, C).copy()


def bf16_round(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) & 0xFFFF0000).view(np.float32)


def quantize(w, bits, group=64, clip_pct=None, mse=False, weight=None,
             symmetric=False):
    """Return (reconstruction, effective bits per weight).

    mse=True searches candidate scales (x0.55..1.3) and keeps the per-group
    minimum of the (optionally weighted) squared error. weight='abs' or
    'sq' reweights the error by |w| or w^2 — a heuristic stand-in for an
    activation-importance matrix (no calibration data exists yet).
    symmetric=True drops the bias: levels are ±(2k+1)·s for the 2^(bits-1)
    magnitudes, i.e. a 2-bit symmetric code holds {±s, ±3s} in four codes.
    Half the side info (no bias bytes) and no min/max skew on zero-mean
    distributions — the structure the IQ2 family approximates with a
    lattice codebook.
    """
    n, C = w.shape
    g = w.reshape(n, C // group, group).astype(np.float32)
    qmax = (1 << bits) - 1

    if symmetric:
        # 2^(bits-1) symmetric magnitudes; scale from the group's max |w|,
        # then LS-optimal if mse — no bias term at all.
        top = (1 << bits) - 1  # largest magnitude multiplier (odd)
        mx = np.abs(g).max(-1, keepdims=True)
        base = bf16_round(np.where(mx > 0, mx / top, 1.0))
        mags = np.arange(1, top + 1, 2, dtype=np.float32)  # 1, 3, ...

        def rec_sym(scale):
            eff = np.where(scale == 0, 1.0, scale)
            idx = np.clip(np.round((np.abs(g) / eff - 1.0) / 2.0),
                          0, len(mags) - 1)
            mag = mags[idx.astype(np.int64)]
            return np.sign(g) * mag * eff

        best_err = None
        best_rec = None
        mults = np.linspace(0.6, 1.4, 33) if mse else np.array([1.0])
        for mult in mults:
            rec = rec_sym(bf16_round(base * mult))
            err = ((g - rec) ** 2).sum(-1, keepdims=True)
            if best_err is None:
                best_err, best_rec = err, rec
            else:
                m = err < best_err
                best_err = np.where(m, err, best_err)
                best_rec = np.where(m, rec, best_rec)
        bits_eff = bits + 16.0 / group  # one BF16 scale per group
        return best_rec.reshape(n, C), bits_eff

    if clip_pct:
        lo = np.percentile(g, clip_pct, axis=-1, keepdims=True)
        hi = np.percentile(g, 100 - clip_pct, axis=-1, keepdims=True)
        g_c = np.clip(g, lo, hi)
    else:
        g_c = g
    mn = g_c.min(-1, keepdims=True)
    mx = g_c.max(-1, keepdims=True)
    base = bf16_round(np.where(mx > mn, (mx - mn) / qmax, 1.0))
    bias = bf16_round(mn)

    if weight == "abs":
        wgt = np.abs(g)
    elif weight == "sq":
        wgt = g * g
    else:
        wgt = np.ones_like(g)

    best_err = None
    best_rec = None
    mults = np.linspace(0.55, 1.3, 31) if mse else np.array([1.0])
    for mult in mults:
        scale = bf16_round(base * mult)
        eff = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.round((g - bias) / eff), 0, qmax)
        rec = q * eff + bias
        err = (wgt * (g - rec) ** 2).sum(-1, keepdims=True)
        if best_err is None:
            best_err, best_rec = err, rec
        else:
            m = err < best_err
            best_err = np.where(m, err, best_err)
            best_rec = np.where(m, rec, best_rec)
    bits_eff = bits + 32.0 / group  # two BF16 aux values per group
    return best_rec.reshape(n, C), bits_eff


def report(tag, w, rec, bits_eff):
    err = np.sqrt(((w - rec) ** 2).sum() / (w ** 2).sum()) * 100
    # error on the largest weights: |w| in the top 1% of the tensor
    thr = np.percentile(np.abs(w), 99)
    big = np.abs(w) >= thr
    big_err = np.sqrt(((w[big] - rec[big]) ** 2).sum() / (w[big] ** 2).sum()) * 100
    print(f"  {tag:<16} bpw {bits_eff:4.2f}  RMS {err:6.2f}%  top1% {big_err:6.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--rows", type=int, default=12)
    args = ap.parse_args()

    tensors = [
        (f"model.language_model.layers.{args.layer}.mlp.experts.gate_up_proj", "gate_up"),
        (f"model.language_model.layers.{args.layer}.mlp.experts.down_proj", "down_proj"),
    ]
    for name, short in tensors:
        w = load_experts(name, range(args.experts), args.rows)
        print(f"{short}: {w.shape[0]} rows x {w.shape[1]} cols, "
              f"mean|w| {np.abs(w).mean():.4f} max|w| {np.abs(w).max():.4f}")
        for bits in (2, 3, 4):
            for tag, kw in [
                ("rtn", dict()),
                ("mse", dict(mse=True)),
                ("mse|w|", dict(mse=True, weight="abs")),
                ("sym", dict(symmetric=True)),
                ("sym+mse|w|", dict(symmetric=True, mse=True, weight="abs")),
                ("sym g32", dict(symmetric=True, mse=True, weight="abs",
                                 group=32)),
            ]:
                rec, eff = quantize(w, bits, **kw)
                report(f"{bits}bit {tag}", w, rec, eff)
        print()


if __name__ == "__main__":
    main()

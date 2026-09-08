#!/usr/bin/env python3
"""GDN branch-fidelity drill: engine stage dump vs qwen3_5_moe torch oracle.

Engine side: /tmp/fq_rows.bin produced by QwenLayer0DebugTests.dumpRowsForTorchProbe
(FQ_TORCH_DUMP=1) — per-layer hidden/dense rows, pre-residual xa rows and the
per-stage pfGdn.normed/conv/o/h/z/g/beta snapshots for the 30 linear layers.

Torch side: reference oracle from transformers 5.14 qwen3_5_moe
(Qwen3_5MoeGatedDeltaNet.forward slow path — no fla / causal-conv1d installed,
so causal_conv1d_fn/chunk_gated_delta_rule are None and the module falls back
to F.silu(conv1d) + torch_chunk_gated_delta_rule, which this drill replicates
step-for-step in fp32 with the checkpoint's bf16 weights upcast).

Two comparison modes, both feeding the ENGINE's own rows so the stage that
first diverges localizes the fault:
  Mode A (normed):   torch input_layernorm(engine hidden input row) vs engine
                     pfGdn.normed  (isolates the input norm).
  Mode B (branch):   torch linear_attn math on the ENGINE's normed row
                     (fp32) vs engine conv / z / g / beta / o / h / xa
                     (isolates the GDN branch from the norm).

Runs layers 0 and 1 (the layers the engine leaves the bf16 reference at per
the chain bisect), all 5 probe tokens, and prints per-stage per-token
maxAbs + RMS diffs plus the L1-vs-L0 t0 ratios (the documented asymmetry).

Usage:
  /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
      tools/qwen-probe/stage_drill.py [/tmp/fq_rows.bin]

Reference: docs/QWEN36_PORT.md "GDN decode math (locked)" + qwen3_5_moe
modeling_qwen3_5_moe.py lines ~438-560 (forward, no-cache path).
"""

import json
import struct
import sys

import torch
import torch.nn.functional as F

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
)

torch.set_num_threads(4)

REPO = "/Volumes/samsung 2t/code/finchmoe"
CKPT = f"{REPO}/models/Qwen3.6-35B-A3B-bf16"
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fq_rows.bin"
LINEAR_LAYERS = (0, 1)  # drill targets per docs/QWEN36_PORT.md remaining work


# ---------------------------------------------------------------- dump parse
def parse_dump(path):
    d = open(path, "rb").read()
    o = [0]

    def rd(fmt):
        v = struct.unpack_from(fmt, d, o[0])
        o[0] += struct.calcsize(fmt)
        return v[0] if len(v) == 1 else v

    magic, T, D, L = rd("4i")
    assert magic == 0x00A10003, hex(magic)
    ids = list(rd(f"{T}i"))
    embed = [list(rd(f"{D}f")) for _ in range(T)]
    hidden, dense = {}, {}
    for l in range(L):
        hidden[l] = [list(rd(f"{D}f")) for _ in range(T)]
    for l in range(L):
        dense[l] = [list(rd(f"{D}f")) for _ in range(T)]
    cnt = rd("i")
    o[0] += cnt * 4 + cnt * 4  # route ids + weights
    m2, nL, T2, D2 = rd("4i")
    assert m2 == 0x00A20003, hex(m2)
    xa = {}
    for l in range(nL):
        xa[l] = [list(rd(f"{D2}f")) for _ in range(T2)]
    m3, linCount, T3, D3, qkvDim, valueDim, numV = rd("7i")
    assert m3 == 0x00A20005, hex(m3)
    assert (T3, D3) == (T, D)
    stages = {}
    for _ in range(linCount):
        l = rd("i")
        sizes = {"normed": T * D, "conv": T * qkvDim, "o": T * valueDim,
                 "h": T * valueDim, "z": T * valueDim, "g": T * numV,
                 "beta": T * numV}
        stages[l] = {}
        for name, n in sizes.items():
            vals = list(rd(f"{n}f"))
            stages[l][name] = vals
            if n != len(vals):
                raise SystemExit(f"stage {l}/{name}: got {len(vals)} want {n}")
    return dict(T=T, D=D, L=L, ids=ids, embed=embed, hidden=hidden,
                dense=dense, xa=xa, stages=stages, qkvDim=qkvDim,
                valueDim=valueDim, numV=numV, linCount=linCount)


def rows(arr, T, D, t):
    return torch.tensor(arr[t * D:(t + 1) * D], dtype=torch.float32)


# ------------------------------------------------------------------ weights
def load_text_config():
    cfg = json.load(open(f"{CKPT}/config.json"))
    return Qwen3_5MoeTextConfig(**cfg["text_config"])


def needed_keys():
    keys = set()
    for l in LINEAR_LAYERS:
        keys |= {f"model.language_model.layers.{l}.input_layernorm.weight"}
        for p in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                  "conv1d", "out_proj"):
            keys.add(f"model.language_model.layers.{l}.linear_attn.{p}.weight")
        for p in ("A_log", "dt_bias"):
            keys.add(f"model.language_model.layers.{l}.linear_attn.{p}")
        keys.add(f"model.language_model.layers.{l}.linear_attn.norm.weight")
    return keys


def load_layer_weights(l, keys, wmap):
    import safetensors
    want = {k for k in keys if f".layers.{l}." in k
            or k.endswith(f".layers.{l}.input_layernorm.weight")}
    # group by shard
    from collections import defaultdict
    by_shard = defaultdict(list)
    for k in want:
        by_shard[wmap[k]].append(k)
    tensors = {}
    for shard, ks in by_shard.items():
        with safetensors.safe_open(f"{CKPT}/{shard}", framework="pt") as sf:
            for k in ks:
                t = sf.get_tensor(k)
                tensors[k] = t.to(torch.float32) if t.is_floating_point() else t
    return tensors


def build_module_pair(l, cfg, tensors):
    pre = f"model.language_model.layers.{l}"
    inorm = Qwen3_5MoeRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    inorm.weight.data.copy_(
        tensors[f"{pre}.input_layernorm.weight"].to(torch.float32))
    attn = Qwen3_5MoeGatedDeltaNet(cfg, l)
    for name, p in attn.named_parameters():
        key = f"{pre}.linear_attn.{name}"
        if name == "A_log" or name == "dt_bias":
            p.data.copy_(tensors[key].to(torch.float32))
        else:  # nn.Linear / conv1d weights
            p.data.copy_(tensors[key].to(torch.float32))
    return inorm, attn


# ------------------------------------------------------------- oracle math
def gdn_branch(attn, x):
    """Replicate Qwen3_5MoeGatedDeltaNet.forward no-cache slow path in fp32.

    x: [1, T, D] fp32 (the ENGINE's normed rows). Returns a dict of the
    same named stages the engine snapshots.
    """
    B, T, D = x.shape
    with torch.no_grad():
        mixed = attn.in_proj_qkv(x).transpose(1, 2)          # [B, C, T]
        zr = attn.in_proj_z(x)                               # [B, T, value]
        b = attn.in_proj_b(x)
        a = attn.in_proj_a(x)
        conv_out = F.silu(attn.conv1d(mixed)[:, :, :T])      # slow path
        conv_out = conv_out.transpose(1, 2)                  # [B, T, C]
        q, k, v = torch.split(conv_out, [attn.key_dim,
                                         attn.key_dim,
                                         attn.value_dim], dim=-1)
        q = q.reshape(B, T, -1, attn.head_k_dim)
        k = k.reshape(B, T, -1, attn.head_k_dim)
        v = v.reshape(B, T, -1, attn.head_v_dim)
        beta = b.sigmoid()
        g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
        q = q.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
        k = k.repeat_interleave(attn.num_v_heads // attn.num_k_heads, dim=2)
        core, _ = attn.chunk_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=None,
            output_final_state=False, use_qk_l2norm_in_kernel=True)
        core = core.reshape(B, T, attn.num_v_heads, attn.head_v_dim)
        z = zr.reshape(B, T, attn.num_v_heads, attn.head_v_dim)
        # Qwen3_5MoeRMSNormGated: norm before gate, fp32 math
        o2 = core.reshape(-1, attn.head_v_dim)
        z2 = z.reshape(-1, attn.head_v_dim)
        var = o2.pow(2).mean(-1, keepdim=True)
        h = o2 * torch.rsqrt(var + attn.layer_norm_epsilon)
        h = h * attn.norm.weight * F.silu(z2)
        h = h.reshape(B, T, attn.value_dim)
        xa = attn.out_proj(h)
    return {"conv": conv_out.reshape(B * T, -1),
            "z": zr.reshape(B * T, -1),
            "g": g.reshape(B * T, -1),
            "beta": beta.reshape(B * T, -1),
            "o": core.reshape(B * T, -1),
            "h": h.reshape(B * T, -1),
            "xa": xa.reshape(B * T, -1)}


def metric(engine_row, torch_row):
    d = torch_row - engine_row
    return (d.abs().max().item(),
            float(d.pow(2).mean().sqrt()),
            float(torch_row.pow(2).mean().sqrt()))


def main():
    cfg = load_text_config()
    dump = parse_dump(DUMP)
    T, D = dump["T"], dump["D"]
    print(f"dump: ids={dump['ids']} T={T} D={D} layers={dump['L']} "
          f"linCount={dump['linCount']} qkvDim={dump['qkvDim']} "
          f"valueDim={dump['valueDim']} numV={dump['numV']}")
    wmap = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    keys = needed_keys()
    missing = [k for k in keys if k not in wmap]
    if missing:
        raise SystemExit(f"weights missing from index: {missing[:5]}")

    for l in LINEAR_LAYERS:
        inorm, attn = build_module_pair(l, cfg, load_layer_weights(l, keys,
                                                                   wmap))
        st = dump["stages"][l]
        print(f"\n=== layer {l} ({'linear' if l else 'linear'} GDN) ===")
        eng_in = dump["embed"] if l == 0 else dump["hidden"][l - 1]
        x_chain = torch.tensor(eng_in, dtype=torch.float32).unsqueeze(0)
        normed_t = inorm(x_chain)                         # [1, T, D] Mode A
        branch = gdn_branch(attn, normed_t)               # Mode B
        # Mode A: the input norm
        print(f"{'stage':8} {'tok':>3} {'maxAbs':>12} {'rmsDiff':>10} "
              f"{'rmsRef':>10}")
        for t in range(T):
            e = rows(st["normed"], T, D, t)
            r = normed_t[0, t]
            mx, rms, ref = metric(e, r)
            print(f"{'normed':8} {t:>3} {mx:12.4g} {rms:10.4g} {ref:10.4g}")
        # Mode B: branch stages (xa lives in the separate xa block, compared
        # below against the engine's pre-residual rows)
        for name in ("conv", "z", "g", "beta", "o", "h"):
            n = {"conv": dump["qkvDim"], "z": dump["valueDim"],
                 "o": dump["valueDim"], "h": dump["valueDim"],
                 "g": dump["numV"], "beta": dump["numV"]}[name]
            r = branch[name]
            for t in range(T):
                e = rows(st[name], T, n, t)
                mx, rms, ref = metric(e, r[t])
                print(f"{name:8} {t:>3} {mx:12.4g} {rms:10.4g} {ref:10.4g}")
        # xa: engine pre-residual rows (xa block) vs torch out_proj
        for t in range(T):
            e = torch.tensor(dump["xa"][l][t], dtype=torch.float32)
            mx, rms, ref = metric(e, branch["xa"][t])
            print(f"{'xa':8} {t:>3} {mx:12.4g} {rms:10.4g} {ref:10.4g}")
        # per-value-head diagnostics: engine o/h vs torch, head rms ratios.
        # A uniform ratio ~sqrt(head_dim)=11.31 on o says the q readout scale
        # cancels inside the engine's l2norm; h ratio ~1 says the gated norm
        # removes it per head; any scatter says the two are misaligned.
        nv, hd = dump["numV"], dump["valueDim"] // dump["numV"]
        for name in ("o", "h"):
            ref_r = {"o": branch["o"], "h": branch["h"]}[name]
            for t in (0, T - 1):
                E = torch.tensor(st[name][t * dump["valueDim"]:
                                         (t + 1) * dump["valueDim"]],
                                 dtype=torch.float32).reshape(nv, hd)
                R = ref_r[t].reshape(nv, hd)
                er = E.pow(2).mean(1).sqrt()
                rr = R.pow(2).mean(1).sqrt()
                rat = er / rr
                print(f"{name:4} t{t} head-rms eng/torch ratio: "
                      f"min {float(rat.min()):.4g}  "
                      f"max {float(rat.max()):.4g}  "
                      f"mean {float(rat.mean()):.4g}  "
                      f"(uniform ~11.31 => readout scale canceled; ~1 => "
                      f"norm removes it)")
    print("\nnote: g/beta in the dump are fp16-rounded; engine o is fp16 "
          "math. diffs are engine(quantized+fp16) - oracle(fp32, bf16 "
          "weights upcast)")


if __name__ == "__main__":
    main()

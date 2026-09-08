#!/usr/bin/env python3
"""Phase 0: Per-tensor quantization audit (non-experts + packed experts).

Compares every quantized tensor against the BF16 safetensors reference and
reports per-role CosSim aggregates (+ RMS ratio, which catches scale
calibration errors that CosSim hides), so Phase 1 knows where precision pays.

Covers:
  1. Non-expert tensors in model_weights_quant.bin (manifest-indexed)
  2. Packed expert slabs (packed_experts_3bit/ and packed_experts_4bit/)
     - all 40 layers, all 256 experts, gate/up/down, dequantized with the
       same layout constants as repack_experts.py

Usage:
    python3 quant_audit.py [--top N]
    python3 quant_audit.py --experts-only
    python3 quant_audit.py --no-experts
    python3 quant_audit.py --pack3 PATH --pack4 PATH
    python3 quant_audit.py --expert-sample 4     # audit every 4th expert (fast)
"""
import argparse
import fcntl
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np

# __file__ = .../finchmoe/finchTool/tools/quant_audit.py

# ── Crash guardrails (2026-08-20 kernel panic lesson) ──────────────────────
# Two concurrent audit runs held 21.1 + 15.3 GB RSS on the 16 GB machine:
# compressor hit 100% of segments, swap ran low, watchdogd starved for 92 s,
# kernel panic. Same signature as the 2026-08-15 logit_dump panic.
# Rule: one heavy job at a time, and abort long before RAM runs out.

HEAVY_JOB_LOCK = '/tmp/finchmoe_heavy_job.lock'


def acquire_heavy_job_lock():
    """flock a shared lockfile; exit if another heavy finchmoe job runs."""
    fd = os.open(HEAVY_JOB_LOCK, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = 'unknown'
        try:
            holder = os.read(fd, 64).decode().strip()
        except Exception:
            pass
        os.close(fd)
        print(f'ABORT: another finchmoe heavy job is running '
              f'(lock held by: {holder}). One job at a time — '
              f'crash history 2026-08-15 / 2026-08-20.', file=sys.stderr)
        sys.exit(1)
    os.ftruncate(fd, 0)
    os.write(fd, f'pid {os.getpid()} {sys.argv[0]}'.encode())
    return fd  # keep open: released when the process exits


def _free_mb():
    """Estimate available RAM (MB): free + inactive + speculative + purgeable."""
    vm = subprocess.check_output(['vm_stat']).decode()
    page = int(re.search(r'page size of (\d+) bytes', vm).group(1))
    total = 0
    for key in ('Pages free', 'Pages inactive',
                'Pages speculative', 'Pages purgeable'):
        m = re.search(key + r':\s+(\d+)\.', vm)
        total += int(m.group(1))
    return total * page // (1024 * 1024)


def _check_memory(need_mb, ctx):
    """Abort if available RAM drops below need_mb — never let the machine
    thrash into a watchdog panic again."""
    free = _free_mb()
    if free < need_mb:
        raise MemoryError(
            f'ABORT ({ctx}): only {free} MB available, need {need_mb} MB. '
            f'Refusing to push the machine into swap — crash history '
            f'2026-08-15 / 2026-08-20.')

_FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_DIR = os.path.dirname(_FT)
MANIFEST = os.path.join(_MANIFEST_DIR, 'model_weights.json')
BF16_DIR = os.path.join(os.path.dirname(_MANIFEST_DIR), 'models', 'Qwen3.6-35B-A3B-bf16')

# Packed expert layouts — keep in sync with repack_experts.py.
# w_off = byte offset of the component's weight within one expert block;
# scales follow at w_off + rows*groups*(3 bytes*8vals) for 3-bit else *4,
# biases follow scales at + rows*groups*2.
EXPERT_SIZE_3BIT = 1376256
EXPERT_SIZE_4BIT = 1769472
EXPERT_SIZE_8BIT = 3342336
EXPERT_COMPONENTS = {
    3: {
        'gate': {'rows': 512, 'groups': 32, 'w_off': 0},
        'up':   {'rows': 512, 'groups': 32, 'w_off': 458752},
        'down': {'rows': 2048, 'groups': 8, 'w_off': 917504},
    },
    4: {
        'gate': {'rows': 512, 'groups': 32, 'w_off': 0},
        'up':   {'rows': 512, 'groups': 32, 'w_off': 589824},
        'down': {'rows': 2048, 'groups': 8, 'w_off': 1179648},
    },
    8: {
        'gate': {'rows': 512, 'groups': 32, 'w_off': 0},
        'up':   {'rows': 512, 'groups': 32, 'w_off': 1114112},
        'down': {'rows': 2048, 'groups': 8, 'w_off': 2228224},
    },
}
# BF16 reference slices (pristine Qwen3.6-35B-A3B-bf16):
#   gate_up_proj fused [256 experts, 1024 rows, 2048 cols]; rows 0-511 = gate, 512-1023 = up
#   down_proj [256 experts, 512 rows, 2048 cols]
GROUP_SIZE = 64


# ── BF16 safetensors access (header-only; never read whole shards) ─────────

_sf_cache = {}  # fname -> {name: (shape, dtype_str, data_off, sz)}

def _load_sf_meta(fname):
    """Read ONLY the header of a safetensors file (never the data section)."""
    path = os.path.join(BF16_DIR, fname)
    if path in _sf_cache:
        return _sf_cache[path]
    with open(path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    aligned_start = (8 + hlen + 7) & ~7
    tensors = {}
    for tname, tmeta in hdr.items():
        if isinstance(tmeta, dict) and 'shape' in tmeta:
            off = aligned_start + tmeta['data_offsets'][0]
            sz = tmeta['data_offsets'][1] - tmeta['data_offsets'][0]  # end-offset -> byte size
            tensors[tname] = (tuple(tmeta['shape']), tmeta['dtype'], off, sz)
    _sf_cache[path] = tensors
    return tensors


def _bf16_to_f32(u16):
    """BF16 u16 array -> float32, converted in chunks so the astype+shift
    temporaries stay ~256 MB even on multi-GB tensors (embed/lm_head)."""
    n = u16.size
    out = np.empty(n, dtype=np.float32)
    CH = 1 << 26  # 64M elements per chunk
    for s in range(0, n, CH):
        e = min(n, s + CH)
        out[s:e] = (u16[s:e].astype(np.uint32) << 16).view(np.float32)
    return out


def load_bf16_tensor(name, flat_slice=None):
    """Load BF16 tensor (or a contiguous flat element slice) as float32.

    flat_slice=(start, end) reads elements [start:end] of the flattened
    tensor — used for fused expert tensors (one contiguous seek).
    """
    idx = json.load(open(os.path.join(BF16_DIR, 'model.safetensors.index.json')))
    wm = idx['weight_map']
    if name not in wm:
        # tolerate prefix differences: match on the last two name parts
        # (e.g. 'embed_tokens.weight') — never just the final part, which
        # collides with every '*.weight' tensor
        key = '.'.join(name.rsplit('.', 2)[-2:]) if '.' in name else name
        hits = [k for k in wm if k.endswith(key)]
        if not hits:
            return None
        name = hits[0]

    shape, dtype_str, data_off, sz = _load_sf_meta(wm[name])[name]
    n = sz // 2

    if flat_slice is not None:
        s, e = flat_slice
        data_off += s * 2
        n = e - s

    with open(os.path.join(BF16_DIR, wm[name]), 'rb') as f:
        f.seek(data_off)
        data = f.read(n * 2)

    if dtype_str == 'BF16':
        arr = _bf16_to_f32(np.frombuffer(data, dtype=np.uint16))
    elif dtype_str in ('F16', 'float16'):
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
    elif dtype_str in ('F32',):
        arr = np.frombuffer(data, dtype=np.float32)
    else:
        return None
    return arr.reshape(shape) if flat_slice is None else arr


def cos_sim(a, b):
    dot = float(np.dot(a.ravel(), b.ravel()))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def rms_norm(x):
    # linalg.norm avoids the x**2 temporary (multi-GB on embed/lm_head)
    return float(np.linalg.norm(x)) / math.sqrt(x.size)


# ── Dequantization ─────────────────────────────────────────────────────────

def _bf16_u16(u):
    return (np.frombuffer(np.asarray(u, dtype=np.uint16), dtype=np.uint16)
            .astype(np.uint32) << 16).view(np.float32)


def dequant_block(bits, weight_raw, scales_raw, biases_raw, rows, groups):
    """Dequant one packed expert block. weight_raw is bytes, scales/biases
    are BF16 bytes (rows*groups*2). Returns float32 (rows, groups*64)."""
    if bits == 3:
        w = np.frombuffer(weight_raw, dtype=np.uint8).reshape(rows, groups, 8, 3).astype(np.uint32)
        v24 = w[:, :, :, 0] | (w[:, :, :, 1] << np.uint32(8)) | (w[:, :, :, 2] << np.uint32(16))
        shifts = np.array([0, 3, 6, 9, 12, 15, 18, 21], dtype=np.uint32)
        nib = ((v24[:, :, :, np.newaxis] >> shifts[np.newaxis, np.newaxis, np.newaxis, :]) & 7)
        vals = nib.astype(np.float32).reshape(rows, -1)
    elif bits == 4:
        w = np.frombuffer(weight_raw, dtype=np.uint32).reshape(rows, groups, 8)
        shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
        nib = ((w[:, :, :, np.newaxis] >> shifts[np.newaxis, np.newaxis, np.newaxis, :]) & 0xF)
        vals = nib.astype(np.float32).reshape(rows, -1)
    elif bits == 8:
        w = np.frombuffer(weight_raw, dtype=np.uint32).reshape(rows, groups, 16)
        shifts = np.array([0, 8, 16, 24], dtype=np.uint32)
        nib = ((w[:, :, :, np.newaxis] >> shifts[np.newaxis, np.newaxis, np.newaxis, :]) & 0xFF)
        vals = nib.astype(np.float32).reshape(rows, -1)
    else:
        raise ValueError(f'unsupported bits: {bits}')

    S = _bf16_u16(np.frombuffer(scales_raw, dtype=np.uint16)).reshape(rows, groups)
    B = _bf16_u16(np.frombuffer(biases_raw, dtype=np.uint16)).reshape(rows, groups)
    S_exp = np.repeat(S, GROUP_SIZE, axis=1)
    B_exp = np.repeat(B, GROUP_SIZE, axis=1)
    return vals * S_exp + B_exp


# ── Name mapping ───────────────────────────────────────────────────────────

_TOP_LEVEL = {'lm_head.weight', 'mtp.fc.weight'}


# Linear-attn head-pair interleave: the engine's kernel pairs v-heads
# (h, h+16) and the extractor stores them adjacent, so the stored head
# order is [0, 16, 1, 17, ..., 15, 31] = g^{-1}(j) = 16*j mod 31.
# To compare against the raw HF layout we un-permute with PINV.
_HEAD_DIM = 128
_PINV = [16 * j % 31 if 1 <= j <= 30 else j for j in range(32)]


def unpermute_linear_attn(arr, kind):
    """Undo the engine's head-pair interleave on a dequantized tensor.

    kind: 'qkv' -> v-block rows only; 'z' -> all rows; 'out' -> columns.
    """
    out = arr.copy()
    if kind == 'qkv':
        for j in range(32):
            src = 4096 + _HEAD_DIM * _PINV[j]
            out[4096 + _HEAD_DIM * j:4096 + _HEAD_DIM * (j + 1)] = arr[src:src + _HEAD_DIM]
    elif kind == 'z':
        for j in range(32):
            src = _HEAD_DIM * _PINV[j]
            out[_HEAD_DIM * j:_HEAD_DIM * (j + 1)] = arr[src:src + _HEAD_DIM]
    elif kind == 'out':
        for j in range(32):
            src = _HEAD_DIM * _PINV[j]
            out[:, _HEAD_DIM * j:_HEAD_DIM * (j + 1)] = arr[:, src:src + _HEAD_DIM]
    return out


def bf16_name(qname):
    """Map a quantized manifest tensor name to its BF16 safetensors name."""
    if qname in _TOP_LEVEL:
        return qname
    if qname == 'model.embed_tokens.weight':
        return 'model.language_model.embed_tokens.weight'
    if qname == 'model.norm.weight':
        return 'model.language_model.norm.weight'
    if qname.startswith('model.layers.'):
        return 'model.language_model.' + qname[len('model.'):]
    if qname.startswith('mtp.'):
        # MTP lives at top level in the safetensors (like lm_head)
        return qname
    return None


def classify_role(name):
    """Classify a tensor name into a role group."""
    parts = name.split('.')

    if 'exps' in name:
        return f'expert.{parts[-1].replace("_exps", "")}'
    if 'shared_expert' in name and '.weight' in name:
        return f'shared_expert.{parts[-1].replace("_proj", "")}'
    if name.startswith('mtp.'):
        return f'mtp.{parts[-1].replace(".weight", "")}'
    if 'lm_head' in name:
        return 'lm_head'
    if 'embed_tokens' in name:
        return 'embed'
    if 'linear_attn' in name:
        if 'in_proj_qkv' in name:
            return 'attn.qkv'
        if 'in_proj_z' in name:
            return 'attn.z'
        if 'out_proj' in name:
            return 'attn.out'
        if 'conv1d' in name:
            return 'attn.conv1d'
        if 'A_log' in name:
            return 'attn.A_log'
        if 'dt_bias' in name:
            return 'attn.dt_bias'
        if 'norm.weight' in name:
            return 'attn.norm'
        return f'attn.{parts[-1].replace(".weight", "")}'
    if 'self_attn' in name:
        if 'q_proj' in name:
            return 'full_attn.q'
        if 'k_proj' in name:
            return 'full_attn.k'
        if 'v_proj' in name:
            return 'full_attn.v'
        if 'o_proj' in name:
            return 'full_attn.o'
        if 'q_norm' in name:
            return 'full_attn.q_norm'
        if 'k_norm' in name:
            return 'full_attn.k_norm'
        return f'full_attn.{parts[-1].replace(".weight", "")}'
    if 'input_layernorm' in name:
        return 'norm.input_layernorm'
    if 'post_attention_layernorm' in name:
        return 'norm.post_attention_layernorm'
    if 'mlp.gate.weight' in name:
        return 'mlp.gate'
    if 'mlp.shared_expert_gate.weight' in name:
        return 'mlp.shared_gate'
    if 'model.norm.weight' in name:
        return 'norm.final'
    if 'gated_norm' in name:
        return 'gdn'
    return f'other.{parts[-1].replace(".weight", "")}'


# ── Non-expert audit ───────────────────────────────────────────────────────

def load_quant_tensor(name, manifest, weights_path):
    """Load + dequantize a non-expert tensor from the manifest bin. float32."""
    info = manifest[name]
    offset, size = info['offset'], info['size']
    with open(weights_path, 'rb') as f:
        f.seek(offset)
        data = f.read(size)
    dtype = info['dtype']
    shape = tuple(info['shape'])

    if dtype == 'U16':  # BF16
        return _bf16_to_f32(np.frombuffer(data, dtype=np.uint16)).reshape(shape)
    if dtype == 'F32':
        return np.frombuffer(data, dtype=np.float32).reshape(shape)
    if dtype == 'U32':
        bits = info.get('bits', 4)
        s_name = name.replace('.weight', '.scales')
        b_name = name.replace('.weight', '.biases')
        si, bi = manifest[s_name], manifest[b_name]
        with open(weights_path, 'rb') as f:
            f.seek(si['offset']); sd = f.read(si['size'])
            f.seek(bi['offset']); bd = f.read(bi['size'])
        rows, cols = shape
        # Manifest shape IS the packed shape: cols = u32 per row.
        # vals/row = cols * (32/bits); groups = vals/row / group_size.
        vals_per_row = cols * (32 // bits)
        groups = vals_per_row // GROUP_SIZE
        return dequant_block(bits, data, sd, bd, rows, groups)
    return None


def audit_non_experts(weights_path):
    """Audit every quantized non-expert tensor vs BF16. Returns list of
    (qname, cos, bf_rms, q_rms, shape, bits, role)."""
    manifest = json.load(open(MANIFEST))['tensors']
    weight_names = sorted(n for n in manifest if n.endswith('.weight'))

    results = []
    errors = []
    for i, qname in enumerate(weight_names):
        info = manifest[qname]
        bits = str(info.get('bits', 'BF16'))
        if bits in ('BF16', 'none'):
            continue
        ref_name = bf16_name(qname)
        if ref_name is None:
            errors.append((qname, 'no BF16 mapping'))
            continue
        _check_memory(3072, f'before {qname}')
        bf16_arr = load_bf16_tensor(ref_name)
        if bf16_arr is None:
            errors.append((qname, f'BF16 tensor not found: {ref_name}'))
            continue
        q_arr = load_quant_tensor(qname, manifest, weights_path)
        if q_arr is None or q_arr.shape != bf16_arr.shape:
            errors.append((qname, f'shape mismatch {getattr(q_arr, "shape", None)} vs {bf16_arr.shape}'))
            continue
        if 'linear_attn.in_proj_qkv' in qname:
            q_arr = unpermute_linear_attn(q_arr, 'qkv')
        elif 'linear_attn.in_proj_z' in qname:
            q_arr = unpermute_linear_attn(q_arr, 'z')
        elif 'linear_attn.out_proj' in qname:
            q_arr = unpermute_linear_attn(q_arr, 'out')
        results.append((qname, cos_sim(bf16_arr, q_arr), rms_norm(bf16_arr),
                        rms_norm(q_arr), tuple(info['shape']), bits, classify_role(qname)))
        if (i + 1) % 200 == 0:
            print(f'  ...non-experts {i+1}/{len(weight_names)}')

    return results, errors


# ── Expert audit ───────────────────────────────────────────────────────────

def load_layer_experts_bf16(layer):
    """Read one layer's fused expert tensors as u16 BF16 (~1.5 GB retained).

    gate_up: [256, 1024, 2048] flat, expert e rows e*1024..e*1024+1023
    down:    [256, 512, 2048] flat, expert e rows e*512..e*512+511
    Per-expert float32 slices come from ref_f32() one expert at a time —
    materializing all 256 at once was ~3 GB and, run twice concurrently,
    helped drive the 2026-08-20 panic.
    """
    idx = json.load(open(os.path.join(BF16_DIR, 'model.safetensors.index.json')))
    wm = idx['weight_map']
    gate_up_name = f'model.language_model.layers.{layer}.mlp.experts.gate_up_proj'
    down_name = f'model.language_model.layers.{layer}.mlp.experts.down_proj'
    if gate_up_name not in wm or down_name not in wm:
        return None, None

    gate_up_u16, down_u16 = [], []
    for name, n in ((gate_up_name, 256 * 1024 * 2048), (down_name, 256 * 512 * 2048)):
        shape, dtype_str, data_off, sz = _load_sf_meta(wm[name])[name]
        with open(os.path.join(BF16_DIR, wm[name]), 'rb') as f:
            f.seek(data_off)
            raw = f.read(n * 2)
        u16 = np.frombuffer(raw, dtype=np.uint16)
        (gate_up_u16 if name == gate_up_name else down_u16).append(u16)
    return gate_up_u16[0], down_u16[0]


def ref_f32(gate_up_u16, down_u16, e):
    """One expert's gate/up/down as float32 (~12 MB — free immediately)."""
    gu_base = e * 1024 * 2048
    dn_base = e * 512 * 2048
    return {
        'gate': _bf16_to_f32(gate_up_u16[gu_base: gu_base + 512 * 2048]).reshape(512, 2048),
        'up': _bf16_to_f32(gate_up_u16[gu_base + 512 * 2048: gu_base + 1024 * 2048]).reshape(512, 2048),
        'down': _bf16_to_f32(down_u16[dn_base: dn_base + 512 * 2048]).reshape(512, 2048),
    }


def audit_experts(bits, pack_dir, sample=1):
    """Audit a packed-expert directory vs pristine BF16.

    Returns records: (layer, expert, comp, cos, bf_rms, q_rms).
    """
    if not os.path.isdir(pack_dir):
        print(f'  SKIP: pack dir not found: {pack_dir}')
        return None

    expert_size = (EXPERT_SIZE_3BIT if bits == 3
                   else EXPERT_SIZE_8BIT if bits == 8 else EXPERT_SIZE_4BIT)
    comps = EXPERT_COMPONENTS[bits]
    records = []

    for layer in range(40):
        layer_file = os.path.join(pack_dir, f'layer_{layer:02d}.bin')
        if not os.path.exists(layer_file):
            print(f'  SKIP: {layer_file} missing')
            continue
        gu, dn = load_layer_experts_bf16(layer)
        if gu is None:
            print(f'  SKIP: BF16 refs missing for layer {layer}')
            continue
        _check_memory(4096, f'before {bits}bit layer {layer}')
        with open(layer_file, 'rb') as f:
            for e in range(0, 256, sample):
                base = e * expert_size
                refs = ref_f32(gu, dn, e)  # ~12 MB — transient
                for comp, c in comps.items():
                    rows, groups = c['rows'], c['groups']
                    w_off = base + c['w_off']
                    weight_bytes = rows * groups * (24 if bits == 3 else 64 if bits == 8 else 32)
                    s_off = w_off + weight_bytes
                    b_off = s_off + rows * groups * 2
                    f.seek(w_off)
                    w_raw = f.read(weight_bytes)
                    f.seek(s_off)
                    s_raw = f.read(rows * groups * 2)
                    f.seek(b_off)
                    b_raw = f.read(rows * groups * 2)
                    deq = dequant_block(bits, w_raw, s_raw, b_raw, rows, groups)
                    ref = refs[comp]
                    records.append((layer, e, comp, cos_sim(ref, deq),
                                    rms_norm(ref), rms_norm(deq)))
                del refs
        del gu, dn  # release the layer's BF16 u16 arrays (~1.5GB)
        print(f'  ...{bits}bit layer {layer} done')

    return records


# ── Reporting ──────────────────────────────────────────────────────────────

def report(results, expert_records, args):
    """Print the ranked role map + worst lists."""
    role_results = {}
    for qname, cos, bf_rms, q_rms, shape, bits, role in results:
        role_results.setdefault(role, []).append((cos, bf_rms, q_rms, shape, bits))

    print()
    print('=' * 110)
    print('PER-ROLE AGGREGATES (non-expert tensors, vs BF16)')
    print('=' * 110)
    print(f"{'Role':<30} {'N':>5} {'CosMin':>8} {'CosMean':>9} {'CosMed':>8} {'BF16_RMS':>11} {'Q_RMS':>11} {'Bits':>12}")
    print('-' * 110)

    role_stats = []
    for role in sorted(role_results):
        entries = role_results[role]
        cos_vals = [e[0] for e in entries]
        bits_set = sorted(set(e[4] for e in entries))
        role_stats.append((role, min(cos_vals), float(np.mean(cos_vals)),
                           float(np.median(cos_vals)),
                           float(np.mean([e[1] for e in entries])),
                           float(np.mean([e[2] for e in entries])),
                           bits_set, len(entries)))
        print(f"{role:<30} {len(entries):>5} {min(cos_vals):>8.5f} {np.mean(cos_vals):>9.5f} "
              f"{np.median(cos_vals):>8.5f} {np.mean([e[1] for e in entries]):>11.4f} "
              f"{np.mean([e[2] for e in entries]):>11.4f} {','.join(str(b) for b in bits_set):>12}")

    # expert summary
    for bits, records in expert_records.items():
        if not records:
            continue
        print()
        print('=' * 110)
        print(f'EXPERT PACK ({bits}bit, {len(records)} tensors) — per component')
        print('=' * 110)
        print(f"{'Comp':<8} {'N':>6} {'CosMin':>8} {'CosMean':>9} {'CosMed':>8} {'RMSratio(bf/q)':>14} {'WorstLayer':>10}")
        for comp in ['gate', 'up', 'down']:
            rr = [r for r in records if r[2] == comp]
            if not rr:
                continue
            cos_vals = [r[3] for r in rr]
            ratios = [r[4] / r[5] for r in rr]
            # worst layer = layer with the lowest min-cos
            worst_layer, worst_cos = -1, 2.0
            for L in sorted(set(r[0] for r in rr)):
                Lmin = min((r[3] for r in rr if r[0] == L), default=2.0)
                if Lmin < worst_cos:
                    worst_layer, worst_cos = L, Lmin
            print(f"{comp:<8} {len(rr):>6} {min(cos_vals):>8.5f} {np.mean(cos_vals):>9.5f} "
                  f"{np.median(cos_vals):>8.5f} {np.mean(ratios):>14.4f} {worst_layer:>10}")

        # worst 10 per component
        print()
        for comp in ['gate', 'up', 'down']:
            rr = sorted((r for r in records if r[2] == comp), key=lambda r: r[3])[:10]
            if rr:
                print(f'  worst-10 {comp}: ' + ', '.join(f'L{r[0]}e{r[1]}={r[3]:.4f}' for r in rr))

    # worst-10 overall
    if expert_records:
        all_exp = [r for recs in expert_records.values() if recs for r in recs]
        if all_exp:
            print()
            print('=' * 110)
            print('WORST-10 EXPERT TENSORS')
            print('=' * 110)
            for r in sorted(all_exp, key=lambda r: r[3])[:10]:
                print(f'  L{r[0]:02d} e{r[1]:>3} {r[2]:<5} cos={r[3]:.5f}')

    if results:
        print()
        print('=' * 110)
        print('WORST-10 NON-EXPERT TENSORS')
        print('=' * 110)
        for r in sorted(results, key=lambda r: r[1])[:10]:
            print(f'  {r[6]:<28} {r[5]:>5} {r[4]} cos={r[1]:.6f}  {r[0]}')

    # summary — one unified ranked list of role min-CosSim, experts included
    print()
    print('=' * 110)
    print('SUMMARY: roles below CosSim thresholds (min-cos per role)')
    print('=' * 110)
    role_cos = {r[0]: r[1] for r in role_stats}  # role -> min cos
    for bits, records in expert_records.items():
        if not records:
            continue
        for comp in ['gate', 'up', 'down']:
            rr = [r[3] for r in records if r[2] == comp]
            if rr:
                role_cos[f'expert.{comp}({bits}bit)'] = min(rr)
    for threshold in [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]:
        below = sorted((k, v) for k, v in role_cos.items() if v < threshold)
        if below:
            print(f'  cos<{threshold:>8.3f}: ' + ', '.join(f'{k}={v:.4f}' for k, v in below[:10]))


def main():
    p = argparse.ArgumentParser(description='Phase 0: quantization audit')
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--no-experts', action='store_true')
    p.add_argument('--experts-only', action='store_true')
    p.add_argument('--pack3', default=os.path.join(_MANIFEST_DIR, 'packed_experts_3bit'))
    p.add_argument('--pack4', default=os.path.join(os.path.dirname(_MANIFEST_DIR), 'models',
                                                   'Qwen3.6-35B-A3B-4bit-dense', 'packed_experts'))
    p.add_argument('--pack8', default=os.path.join(os.path.dirname(_MANIFEST_DIR), 'models',
                                                   'Qwen3.6-35B-A3B-bf16', 'packed_experts_8bit'))
    p.add_argument('--expert-sample', type=int, default=1)
    args = p.parse_args()

    # One heavy job at a time, and leave headroom: the 2026-08-20 kernel
    # panic was two concurrent audit runs exhausting the 16 GB machine.
    acquire_heavy_job_lock()
    _check_memory(6144, 'startup')

    # pick the live weight file (largest candidate)
    candidates = [os.path.join(_MANIFEST_DIR, 'model_weights.bin'),
                  os.path.join(_MANIFEST_DIR, 'quant_clean_pi', 'model_weights_quant.bin'),
                  os.path.join(_MANIFEST_DIR, 'quant_clean', 'model_weights_quant.bin')]
    weights_path = max((c for c in candidates if os.path.exists(c)), key=os.path.getsize, default=None)
    if weights_path is None:
        print('ERROR: no weight file found')
        sys.exit(1)

    print(f'Manifest: {MANIFEST} -> {os.path.realpath(MANIFEST)}')
    print(f'Weights:  {weights_path}')
    print(f'BF16 dir: {BF16_DIR}')

    results, errors = [], []
    if not args.experts_only:
        print()
        print('=== auditing non-expert tensors ===')
        results, errors = audit_non_experts(weights_path)
        print(f'  non-expert tensors audited: {len(results)}, errors: {len(errors)}')

    expert_records = {}
    if not args.no_experts:
        print()
        print('=== auditing 3-bit expert pack ===')
        r3 = audit_experts(3, args.pack3, sample=args.expert_sample)
        if r3:
            expert_records[3] = r3
        print()
        print('=== auditing 4-bit expert pack ===')
        r4 = audit_experts(4, args.pack4, sample=args.expert_sample)
        if r4:
            expert_records[4] = r4
        print()
        print('=== auditing 8-bit expert pack ===')
        r8 = audit_experts(8, args.pack8, sample=args.expert_sample)
        if r8:
            expert_records[8] = r8

    report(results, expert_records, args)

    if errors:
        print()
        print(f'ERRORS ({len(errors)}):')
        for qname, reason in errors[:20]:
            print(f'  {qname}: {reason}')


if __name__ == '__main__':
    main()

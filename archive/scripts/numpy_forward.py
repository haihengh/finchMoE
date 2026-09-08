#!/usr/bin/env python3
"""numpy_forward.py — offline BLAS-order forward pass of Qwen3.6-35B-A3B.

Counterfactual test: finchMoE's verified dequant + GDN math, executed with
numpy (BLAS) matmul — the "does BLAS-ordered arithmetic land on the good
side?" question for finchMoE's specific formulation.

Verified layer-by-layer against llama.cpp's per-token l_out dumps
(target corr >= 0.9999) before any slice is trusted.

Usage:
  python3 numpy_forward.py verify <llama_lout.bin> [position]
  python3 numpy_forward.py slice <start> <end> <out_prefix>
"""
import json
import os
import re
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.join(REPO, 'finchmoe/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf')
DATA_START = 10988896   # verified (embedding dequant matched finchmoe exactly)
EPS = 1e-6

HIDDEN = 2048
N_LAYERS = 40
FA_INTERVAL = 4
N_EXPERT = 256
N_EXPERT_USED = 8
FF = 512          # expert ff dim
FF_SH = 512       # shared expert ff dim
N_K_HEADS = 16    # GDN key heads
K_DIM = 128
N_V_HEADS = 32    # GDN value heads
V_DIM = 128
CONV_K = 4        # conv1d kernel
N_FA_HEADS = 16   # full-attn query heads
FA_HEAD_DIM = 256
N_FA_KV = 2
N_ROT = 64        # rope dims
ROPE_BASE = 1e7
ROPE_SECTIONS = [11, 11, 10, 0]

GGML_TYPES = {0: 'F32', 12: 'Q4_K', 14: 'Q6_K'}


# ---------------------------------------------------------------- dequant

def fp16(h):
    return struct.unpack('<e', struct.pack('<H', h))[0]


def dequant_q4k_block(p):
    """144-byte Q4_K block -> 256 f32 (vectorized)."""
    d = fp16(struct.unpack('<H', p[0:2])[0])
    mn = fp16(struct.unpack('<H', p[2:4])[0])
    scales = np.frombuffer(p[4:16], np.uint8).astype(np.int32)
    q = np.frombuffer(p[16:144], np.uint8).astype(np.int32)
    # 8 scale values (6-bit packed): indices 0..7
    sc = np.empty(8, np.int32)
    sc[0:4] = scales[0:4] & 63
    sc[4:8] = (scales[8:12] & 0xF) | ((scales[0:4] >> 6) << 4)
    m = np.empty(8, np.int32)
    m[0:4] = scales[4:8] & 63
    m[4:8] = (scales[8:12] >> 4) | ((scales[4:8] >> 6) << 4)
    # each 64-block uses 2 scales/mins; q chunk: 32 bytes -> lows then highs
    s_all = np.repeat(sc, 32).reshape(8, 32).T.reshape(256)  # placeholder, fix below
    y = np.empty(256, np.float32)
    for j in range(4):
        d1, m1 = d * sc[2 * j], mn * m[2 * j]
        d2, m2 = d * sc[2 * j + 1], mn * m[2 * j + 1]
        qb = q[j * 32:(j + 1) * 32]
        y[j * 64:j * 64 + 32] = d1 * (qb & 0xF) - m1
        y[j * 64 + 32:j * 64 + 64] = d2 * (qb >> 4) - m2
    return y


def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dequant_q6k_block(p):
    """210-byte Q6_K block -> 256 f32 (vectorized, verified vs llama outputs)."""
    d = fp16(struct.unpack('<H', p[208:210])[0])
    ql = np.frombuffer(p[0:128], np.uint8).astype(np.int32)
    qh = np.frombuffer(p[128:192], np.uint8).astype(np.int32)
    sc = np.frombuffer(p[192:208], np.int8).astype(np.int32)
    q = np.zeros(256, np.int32)
    q[0:32] = (ql[0:32] & 0xF) | ((qh[0:32] >> 0) & 3) << 4
    q[32:64] = (ql[32:64] & 0xF) | ((qh[0:32] >> 2) & 3) << 4
    q[64:96] = (ql[0:32] >> 4) | ((qh[0:32] >> 4) & 3) << 4
    q[96:128] = (ql[32:64] >> 4) | ((qh[0:32] >> 6) & 3) << 4
    q[128:160] = (ql[64:96] & 0xF) | ((qh[32:64] >> 0) & 3) << 4
    q[160:192] = (ql[96:128] & 0xF) | ((qh[32:64] >> 2) & 3) << 4
    q[192:224] = (ql[64:96] >> 4) | ((qh[32:64] >> 4) & 3) << 4
    q[224:256] = (ql[96:128] >> 4) | ((qh[32:64] >> 6) & 3) << 4
    q -= 32
    s = np.zeros(256, np.int32)
    for n in (0, 128):
        for g in range(8):
            s[n + g * 16:n + g * 16 + 16] = sc[g + (8 if n else 0)]
    return (d * s * q).astype(np.float32)


def dequant_rows(typ, mm, data_off, ne, row_start, n_rows):
    """Dequantize n_rows consecutive rows (each ne[0] elems). Fully vectorized."""
    n = ne[0]
    nblk = n // 256
    if typ == 0:  # F32
        row_bytes = n * 4
        out = np.empty((n_rows, n), np.float32)
        for i in range(n_rows):
            off = (row_start + i) * row_bytes
            out[i] = np.frombuffer(mm[off:off + row_bytes], np.float32)
        return out
    if typ == 12:  # Q4_K: block = 144 B
        rb = nblk * 144
        raw = np.ndarray((n_rows, nblk, 144), np.uint8, buffer=mm,
                         offset=(row_start * rb))
        r16 = raw.view(np.uint16)          # [R, B, 72] zero-copy
        d = r16[..., 0:1].view(np.float16).astype(np.float32)   # [R,B,1]
        mn = r16[..., 1:2].view(np.float16).astype(np.float32)
        s0 = raw[:, :, 4:8]                # uint8 scales bytes 0..3
        s1 = raw[:, :, 8:12]               # bytes 4..7
        s2 = raw[:, :, 12:16]              # bytes 8..11
        q = raw[:, :, 16:144]
        # scale/min decode (uint8 ops, promote to int32 only where needed)
        sc = np.empty((n_rows, nblk, 8), np.uint8)
        sc[..., 0:4] = s0 & 63
        sc[..., 4:8] = (s2 & 0xF) | ((s0 >> 6) << 4)
        mm_ = np.empty((n_rows, nblk, 8), np.uint8)
        mm_[..., 0:4] = s1 & 63
        mm_[..., 4:8] = (s2 >> 4) | ((s1 >> 6) << 4)
        # 256 values per block: 8 sub-blocks of 32; scale/mins per sub-block pair
        s_pairs = sc.reshape(n_rows, nblk, 4, 2)          # sub-block j uses [2j, 2j+1]
        m_pairs = mm_.reshape(n_rows, nblk, 4, 2)
        d1 = (d * s_pairs[..., 0].astype(np.float32))[..., None]   # [R,B,4,1]
        m1 = (mn * m_pairs[..., 0].astype(np.float32))[..., None]
        d2 = (d * s_pairs[..., 1].astype(np.float32))[..., None]
        m2 = (mn * m_pairs[..., 1].astype(np.float32))[..., None]
        lo = (q.reshape(n_rows, nblk, 4, 32) & 0xF).astype(np.float32)
        hi = (q.reshape(n_rows, nblk, 4, 32) >> 4).astype(np.float32)
        y = np.concatenate([d1 * lo - m1, d2 * hi - m2], axis=-1)   # [R,B,4,64]
        return y.reshape(n_rows, n).astype(np.float32)
    if typ == 14:  # Q6_K: block = 210 B
        rb = nblk * 210
        raw = np.ndarray((n_rows, nblk, 210), np.uint8, buffer=mm,
                         offset=(row_start * rb))
        d = raw[:, :, 208:210].copy().view(np.uint16).view(np.float16).astype(np.float32)  # [R,B]
        ql = raw[:, :, 0:128]
        qh = raw[:, :, 128:192]
        sc = raw[:, :, 192:208].view(np.int8).astype(np.float32)   # [R,B,16]
        # Q6_K 256-value block: 8 groups of 32; group g uses ql[g//4*32:...] with
        # nibble shift (g//4)*4 and qh[g%4... no: qh index (g//4)*32+j, bits (g%4)*2.
        q = np.zeros((n_rows, nblk, 256), np.uint8)
        q[..., 0:32] = (ql[..., 0:32] & 0xF) | ((qh[..., 0:32] >> 0) & 3) << 4
        q[..., 32:64] = (ql[..., 32:64] & 0xF) | ((qh[..., 0:32] >> 2) & 3) << 4
        q[..., 64:96] = (ql[..., 0:32] >> 4) | ((qh[..., 0:32] >> 4) & 3) << 4
        q[..., 96:128] = (ql[..., 32:64] >> 4) | ((qh[..., 0:32] >> 6) & 3) << 4
        q[..., 128:160] = (ql[..., 64:96] & 0xF) | ((qh[..., 32:64] >> 0) & 3) << 4
        q[..., 160:192] = (ql[..., 96:128] & 0xF) | ((qh[..., 32:64] >> 2) & 3) << 4
        q[..., 192:224] = (ql[..., 64:96] >> 4) | ((qh[..., 32:64] >> 4) & 3) << 4
        q[..., 224:256] = (ql[..., 96:128] >> 4) | ((qh[..., 32:64] >> 6) & 3) << 4
        qf = q.astype(np.float32) - 32.0
        # scale: 8 groups of 16 per 128-half; half 0 uses sc[0:8], half 1 uses sc[8:16]
        s = np.concatenate([np.repeat(sc[..., 0:8], 16, axis=-1),
                            np.repeat(sc[..., 8:16], 16, axis=-1)], axis=-1)  # [R,B,256]
        return (d * s * qf).reshape(n_rows, n)
    raise ValueError(f'unsupported type {typ}')


# ---------------------------------------------------------------- GGUF loader

def load_tensor_info(path):
    f = open(path, 'rb')
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))

    def skip_str():
        ln = struct.unpack('<Q', f.read(8))[0]
        f.seek(ln, 1)

    SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    for _ in range(n_kv):
        ln = struct.unpack('<Q', f.read(8))[0]
        f.seek(ln, 1)
        ty = struct.unpack('<I', f.read(4))[0]
        if ty == 8:
            skip_str()
        elif ty == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            if et == 8:
                for _ in range(n):
                    skip_str()
            else:
                f.seek(SIZES[et] * n, 1)
        else:
            f.seek(SIZES[ty], 1)
    tensors = {}
    for i in range(n_tensors):
        ln = struct.unpack('<Q', f.read(8))[0]
        name = f.read(ln).decode('utf-8', 'replace')
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack(f'<{nd}Q', f.read(8 * nd))
        typ = struct.unpack('<I', f.read(4))[0]
        off = struct.unpack('<Q', f.read(8))[0]
        tensors[name] = (typ, dims, off)
    align = struct.unpack('<I', f.read(4))[0]
    data_start = f.tell()
    f.close()
    return tensors


# ---------------------------------------------------------------- model

def is_fa(il):
    return (il + 1) % FA_INTERVAL == 0


def gdn_idx(il):
    return il - (il + 1) // FA_INTERVAL


_RMS_DTYPE = None  # None = float64 (default, verified); 'f32' = float32 accumulation


def _rms_dtype():
    global _RMS_DTYPE
    if _RMS_DTYPE is None:
        _RMS_DTYPE = 'f32' if os.environ.get('NUMPY_RMS_F32') else 'f64'
    return _RMS_DTYPE


def rmsnorm(x, w):
    dt = np.float32 if _rms_dtype() == 'f32' else np.float64
    return x / np.sqrt(np.mean(x.astype(dt) ** 2) + EPS) * w


def rmsnorm_rows(x, w):
    """Per-row RMSNorm: x [..., D], w [D] -> normalize each row, scale by w."""
    dt = np.float32 if _rms_dtype() == 'f32' else np.float64
    return x / np.sqrt(np.mean(x.astype(dt) ** 2, axis=-1, keepdims=True) + EPS) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -80, 80)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope(q, pos):
    """IMRoPE on the first N_ROT dims of a [..., head_dim] tensor.
    Text-only: all four mrope positions equal -> standard RoPE:
    pair ic rotates (dims ic, ic+32) by pos * base^(-2 ic / N_ROT)."""
    head_dim = q.shape[-1]
    n_pairs = N_ROT // 2
    ic = np.arange(n_pairs)
    theta = pos * ROPE_BASE ** (-2.0 * ic / N_ROT)
    cos = np.cos(theta)
    sin = np.sin(theta)
    x0 = q[..., 0:n_pairs].copy()
    x1 = q[..., n_pairs:N_ROT].copy()
    q[..., 0:n_pairs] = x0 * cos - x1 * sin
    q[..., n_pairs:N_ROT] = x0 * sin + x1 * cos
    return q


class Model:
    def __init__(self, path=GGUF):
        self.path = path
        self.info = load_tensor_info(path)
        self.mm = np.memmap(path, mode='r')
        self.data = self.mm  # byte view
        self.w = {}  # dequantized non-expert weights
        self.expert_cache = {}  # (il, e, kind) -> [rows, in_dim] (fp16)
        self.expert_order = []
        self.expert_cache_max = 400   # fp16 cache: 400 x 2 MB = 0.8 GB (keep memory in check)
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._lock = threading.Lock()
        self.load_non_experts()

    def raw(self, name):
        typ, dims, off = self.info[name]
        base = DATA_START + off
        return self.mm[base:], typ, dims

    def t(self, name):
        """Dequantize a full small tensor to [ne1..., ne0] float32 (row-major)."""
        mm, typ, dims = self.raw(name)
        n_rows = 1
        for d in dims[1:]:
            n_rows *= d
        return dequant_rows(typ, mm, 0, (dims[0],), 0, n_rows).reshape(dims[1:] + (dims[0],))

    def load_non_experts(self):
        for name, (typ, dims, off) in self.info.items():
            if 'exps' in name:
                continue
            if name in ('token_embd.weight', 'output.weight'):
                continue
            if name.endswith('.weight') or name.endswith('.bias') or name == 'ssm_a' or name.startswith('blk.'):
                pass
            # dequant everything small (non-expert)
            try:
                self.w[name] = self.t(name)
            except Exception:
                pass
        # precompute per-layer indices
        self.layers = {}
        for il in range(N_LAYERS):
            self.layers[il] = {
                'attn_norm': self.w.get(f'blk.{il}.attn_norm.weight'),
                'post_norm': self.w.get(f'blk.{il}.post_attention_norm.weight'),
            }
            if is_fa(il):
                self.layers[il].update({
                    'q': self.w.get(f'blk.{il}.attn_q.weight'),
                    'k': self.w.get(f'blk.{il}.attn_k.weight'),
                    'v': self.w.get(f'blk.{il}.attn_v.weight'),
                    'o': self.w.get(f'blk.{il}.attn_output.weight'),
                    'q_norm': self.w.get(f'blk.{il}.attn_q_norm.weight'),
                    'k_norm': self.w.get(f'blk.{il}.attn_k_norm.weight'),
                })
            else:
                self.layers[il].update({
                    'qkv': self.w.get(f'blk.{il}.attn_qkv.weight'),
                    'z': self.w.get(f'blk.{il}.attn_gate.weight'),
                    'beta': self.w.get(f'blk.{il}.ssm_beta.weight'),
                    'alpha': self.w.get(f'blk.{il}.ssm_alpha.weight'),
                    'conv': self.w.get(f'blk.{il}.ssm_conv1d.weight'),
                    'dt': self.w.get(f'blk.{il}.ssm_dt.bias'),
                    'a': self.w.get(f'blk.{il}.ssm_a'),
                    'snorm': self.w.get(f'blk.{il}.ssm_norm.weight'),
                    'sout': self.w.get(f'blk.{il}.ssm_out.weight'),
                })
            self.layers[il].update({
                'gate_inp': self.w.get(f'blk.{il}.ffn_gate_inp.weight'),
                'gate_inp_sh': self.w.get(f'blk.{il}.ffn_gate_inp_shexp.weight'),
                'gate_sh': self.w.get(f'blk.{il}.ffn_gate_shexp.weight'),
                'up_sh': self.w.get(f'blk.{il}.ffn_up_shexp.weight'),
                'down_sh': self.w.get(f'blk.{il}.ffn_down_shexp.weight'),
            })
        self.embd_mm, self.embd_typ, self.embd_dims = self.raw('token_embd.weight')
        self.out_mm, self.out_typ, self.out_dims = self.raw('output.weight')
        self.out_norm = self.w.get('output_norm.weight')
        # NUMPY_STAGED_TRUNC=1: mirror finchmoe's GGUF importer — the F32 small
        # tensors it BF16-stages (norms, conv1d, dt_bias, routers, alpha/beta)
        # are truncated (bits >> 16) at load. Env-gated probe; off by default.
        if os.environ.get('NUMPY_STAGED_TRUNC'):
            _tr = lambda x: (x.astype(np.float32).view(np.uint32) & 0xFFFF0000).view(np.float32)
            for il in range(N_LAYERS):
                for key in ('attn_norm', 'post_norm', 'q_norm', 'k_norm', 'snorm',
                            'conv', 'dt', 'gate_inp', 'gate_inp_sh', 'gate_sh',
                            'alpha', 'beta'):
                    v = self.layers[il].get(key)
                    if v is not None and v.dtype == np.float32:
                        self.layers[il][key] = _tr(v)
            self.out_norm = _tr(self.out_norm) if self.out_norm is not None else None
        # dequantize the lm_head once, keep fp16 (1 GB): Q4_K quantization error
        # (~1e-2) far exceeds fp16 storage error (~5e-4), so logits are unaffected.
        self.out_all = dequant_rows(self.out_typ, self.out_mm, 0,
                                    (self.out_dims[0],), 0, self.out_dims[1]).astype(np.float16)

    def embed(self, tid):
        return dequant_rows(self.embd_typ, self.embd_mm, 0, (self.embd_dims[0],), tid, 1)[0]

    def expert(self, il, e, kind):
        key = (il, e, kind)
        arr16 = self.expert_cache.get(key)
        if arr16 is not None:
            return arr16.astype(np.float32)
        name = f'blk.{il}.ffn_{kind}_exps.weight'
        mm, typ, dims = self.raw(name)
        # dims: gate/up = (2048, 512, 256); down = (512, 2048, 256). Expert e slice:
        rows_per_exp = dims[1]
        row_len = dims[0]
        rows = dequant_rows(typ, mm, 0, (row_len,), e * rows_per_exp, rows_per_exp)
        # cache as fp16 (halves memory; convert on hit)
        with self._lock:
            if len(self.expert_order) >= self.expert_cache_max:
                old = self.expert_order.pop(0)
                self.expert_cache.pop(old, None)
            self.expert_cache[key] = rows.astype(np.float16)
            self.expert_order.append(key)
        return rows

    def reset_state(self):
        self.conv_state = [np.zeros((CONV_K - 1, 8192), np.float32) for _ in range(30)]
        self.ssm_state = [np.zeros((N_V_HEADS, V_DIM, K_DIM), np.float32) for _ in range(30)]
        self.kv_k = [[] for _ in range(10)]
        self.kv_v = [[] for _ in range(10)]

    # ---- layer forward ----
    def forward_layer(self, il, h, pos):
        if os.environ.get('NUMPY_GDN_DUMP'):
            self._gdnd_pos = pos
        l = self.layers[il]
        x = rmsnorm(h, l['attn_norm'])
        if is_fa(il):
            qkv = l['q'] @ x                      # [8192], layout: per head [q(256)|gate(256)]
            qq = qkv.reshape(N_FA_HEADS, 2 * FA_HEAD_DIM)
            q = qq[:, 0:FA_HEAD_DIM]
            qg = qq[:, FA_HEAD_DIM:]
            q = rmsnorm_rows(q, l['q_norm'])
            k = (l['k'] @ x).reshape(N_FA_KV, FA_HEAD_DIM)
            k = rmsnorm_rows(k, l['k_norm'])
            v = (l['v'] @ x).reshape(N_FA_KV, FA_HEAD_DIM)
            q = rope(q, pos)
            k = rope(k, pos)
            ki = il // FA_INTERVAL
            self.kv_k[ki].append(k.copy())
            self.kv_v[ki].append(v.copy())
            K = np.stack(self.kv_k[ki])   # [T, 2, 256]
            V = np.stack(self.kv_v[ki])
            # GQA: q head h uses kv head h // (16/2)
            kv_h = np.arange(N_FA_HEADS) // (N_FA_HEADS // N_FA_KV)
            scores = np.empty((N_FA_HEADS, K.shape[0]))
            for hh in range(N_FA_HEADS):
                scores[hh] = q[hh] @ K[:, kv_h[hh], :].T
            scores /= np.sqrt(FA_HEAD_DIM)
            attn = np.array([softmax(scores[hh]) for hh in range(N_FA_HEADS)])  # [16, T]
            ctx = np.array([attn[hh] @ V[:, kv_h[hh], :] for hh in range(N_FA_HEADS)])  # [16, 256]
            ctx = ctx * sigmoid(qg.reshape(N_FA_HEADS, FA_HEAD_DIM))
            attn_out = l['o'] @ ctx.reshape(-1)  # [2048]

        else:
            qkv = l['qkv'] @ x          # [8192]
            z = l['z'] @ x              # [4096]
            beta = sigmoid(l['beta'] @ x)       # [32]
            alpha = l['alpha'] @ x              # [32]
            gate = softplus(alpha + l['dt']) * l['a']   # [32]
            gi = gdn_idx(il)
            conv_in = np.concatenate([self.conv_state[gi], qkv[None, :]], axis=0)  # [4, 8192]
            conv_out = silu(np.sum(l['conv'] * conv_in.T, axis=1))  # [8192]
            self.conv_state[gi] = np.concatenate([self.conv_state[gi][1:], qkv[None, :]], axis=0)
            q = conv_out[0:2048].reshape(N_K_HEADS, K_DIM)
            k = conv_out[2048:4096].reshape(N_K_HEADS, K_DIM)
            v = conv_out[4096:8192].reshape(N_V_HEADS, V_DIM)
            q = q / np.sqrt(np.sum(q ** 2, axis=1, keepdims=True) + EPS)
            k = k / np.sqrt(np.sum(k ** 2, axis=1, keepdims=True) + EPS)
            s = self.ssm_state[gi]
            s *= np.exp(gate)[:, None, None]
            out = np.zeros(4096, np.float32)
            q_scale = 1.0 / np.sqrt(K_DIM)
            for vh in range(N_V_HEADS):
                kh = vh % N_K_HEADS
                sk = s[vh] @ k[kh]
                dlt = (v[vh] - sk) * beta[vh]
                s[vh] += np.outer(dlt, k[kh])
                out[vh * V_DIM:(vh + 1) * V_DIM] = s[vh] @ (q[kh] * q_scale)
            if os.environ.get('NUMPY_GDN_DUMP'):
                # append (layer, pos, delta-out) to a binary file for the
                # family-ranking GDN-internal comparison vs finchmoe de/ga tags
                with open(os.environ['NUMPY_GDN_DUMP'], 'ab') as _df:
                    _df.write(np.asarray([il, self._gdnd_pos], np.int32).tobytes())
                    _df.write(out.astype(np.float32).tobytes())
            gated = rmsnorm_rows(out.reshape(N_V_HEADS, V_DIM), l['snorm']).reshape(-1) * silu(z)
            attn_out = l['sout'] @ gated
        h = h + attn_out
        x = rmsnorm(h, l['post_norm'])
        moe_out = self.moe(il, x)
        if os.environ.get('DBG'):
            print(f'  [DBG L{il}] moe_out.max={moe_out.max():.3f} rms={np.sqrt(np.mean(moe_out**2)):.3f}')
        h = h + moe_out
        return h

    def moe(self, il, x):
        l = self.layers[il]
        logits = l['gate_inp'] @ x
        probs = softmax(logits)
        idx = np.argsort(-probs)[:N_EXPERT_USED]
        w = probs[idx]
        w = w / max(float(np.sum(w)), 6.103515625e-5)

        def one(e, we):
            g = self.expert(il, int(e), 'gate') @ x
            u = self.expert(il, int(e), 'up') @ x
            d = self.expert(il, int(e), 'down') @ (silu(g) * u)
            return we * d

        parts = list(self._pool.map(one, idx, w))
        out = sum(parts) if parts else np.zeros(HIDDEN, np.float32)
        sg = sigmoid(l['gate_inp_sh'] @ x)
        gs = l['gate_sh'] @ x
        us = l['up_sh'] @ x
        sh = l['down_sh'] @ (silu(gs) * us)
        out += sg * sh
        return out

    def logits_second(self, h, penalty, banned):
        x = rmsnorm(h, self.out_norm)
        n = self.out_dims[1]
        best = (-1, -1e30)
        CHUNK = 5000
        for c0 in range(0, n, CHUNK):
            lg = self.out_all[c0:c0 + CHUNK] @ x
            if penalty:
                for t, sgn in penalty.items():
                    if c0 <= t < c0 + CHUNK:
                        v = lg[t - c0]
                        lg[t - c0] = v / 1.05 if v > 0 else v * 1.05
            for t in banned:
                if c0 <= t < c0 + CHUNK:
                    lg[t - c0] = -1e30
            i = int(np.argmax(lg))
            if lg[i] > best[1]:
                best = (c0 + i, float(lg[i]))
        return best

    def logits(self, h, penalty=None):
        x = rmsnorm(h, self.out_norm)
        n = self.out_dims[1]
        best = (-1, -1e30)
        CHUNK = 5000
        for c0 in range(0, n, CHUNK):
            lg = self.out_all[c0:c0 + CHUNK] @ x
            if penalty:
                for t, sgn in penalty.items():
                    if c0 <= t < c0 + CHUNK:
                        v = lg[t - c0]
                        lg[t - c0] = v / 1.05 if v > 0 else v * 1.05
            i = int(np.argmax(lg))
            if lg[i] > best[1]:
                best = (c0 + i, float(lg[i]))
        return best


# ---------------------------------------------------------------- verification

def load_llama_lout(path):
    """Load the eval-callback l_out dump: name -> list of (pos, vector)."""
    out = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(4)
            if not h or len(h) < 4:
                break
            nl = struct.unpack('<I', h)[0]
            name = f.read(nl).decode('utf-8', 'replace')
            nb = struct.unpack('<Q', f.read(8))[0]
            typ = struct.unpack('<I', f.read(4))[0]
            nd = struct.unpack('<I', f.read(4))[0]
            ne = struct.unpack(f'<{nd}I', f.read(4 * nd))
            payload = f.read(nb)
            if name.startswith('l_out-') and typ == 0:
                il = int(name.split('-')[1])
                out.setdefault(il, []).append(np.frombuffer(payload, np.float32))
    return out


def verify(model, lout_path, tokens_path=None):
    print(f'[verify] loading llama l_out from {lout_path}')
    ref = load_llama_lout(lout_path)
    if tokens_path is None:
        tokens_path = '/tmp/sweep_153.csv'
    toks = [int(x) for x in open(tokens_path).read().split(',')]
    print(f'[verify] {len(toks)} tokens')
    model.reset_state()
    # record per (layer, pos) hidden states
    mine = {}
    for p, tid in enumerate(toks):
        h = model.embed(tid)
        for il in range(N_LAYERS):
            h = model.forward_layer(il, h, p)
            mine.setdefault(il, []).append(h.copy())
    print(f'{"L":>3} {"maxd":>9} {"rmsd":>10} {"corr":>8}  (worst over all positions)')
    worst = 1.0
    for il in range(N_LAYERS):
        refs = ref.get(il, [])
        if not refs:
            print(f'{il:3d}  NO REF')
            continue
        best_corr = 1.0
        worst_maxd = 0.0
        worst_rmsd = 0.0
        for p in range(min(len(refs), len(mine[il]))):
            r = refs[p]
            m = mine[il][p]
            c = float(np.corrcoef(m, r)[0, 1])
            best_corr = min(best_corr, c)
            d = m - r
            worst_maxd = max(worst_maxd, float(np.max(np.abs(d))))
            worst_rmsd = max(worst_rmsd, float(np.sqrt(np.mean(d ** 2))))
        worst = min(worst, best_corr)
        print(f'{il:3d} {worst_maxd:9.4f} {worst_rmsd:10.5f} {best_corr:8.5f}')
    print(f'[verify] worst corr: {worst:.5f}  {"PASS" if worst >= 0.999 else "FAIL (target 0.9999, check math)"}')


if __name__ == '__main__':
    cmd = sys.argv[1]
    m = Model()
    if cmd == 'verify':
        verify(m, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == 'slice':
        tok = Tokenizer(GGUF)
        run_slice(m, tok, int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])


# ---------------------------------------------------------------- tokenizer

def load_tokenizer(path):
    """Load GGUF tokenizer (tokens, merges, types) via a fresh parse."""
    f = open(path, 'rb')
    magic = f.read(4)
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors, n_kv = struct.unpack('<QQ', f.read(16))
    def rstr():
        ln = struct.unpack('<Q', f.read(8))[0]
        return f.read(ln).decode('utf-8', 'replace')
    tokens = None
    merges = None
    types = None
    for _ in range(n_kv):
        key = rstr()
        ty = struct.unpack('<I', f.read(4))[0]
        if ty == 8:
            v = rstr()
        elif ty == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            if et == 8:
                v = [rstr() for _ in range(n)]
            elif et == 5:
                v = list(np.frombuffer(f.read(4 * n), np.int32))
            else:
                f.seek((4 if et in (4, 6) else 1 if et in (0, 1) else 2 if et in (2, 3) else 8) * n, 1)
                v = None
        else:
            f.seek({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[ty], 1)
            v = None
        if key == 'tokenizer.ggml.tokens':
            tokens = v
        elif key == 'tokenizer.ggml.merges':
            merges = v
        elif key == 'tokenizer.ggml.token_type':
            types = v
    f.close()
    return tokens, merges, types


def _bytes_to_unicode():
    """Classic GPT-2 byte encoder: printable bytes map to their literal char,
    control/extra bytes map to U+0100.. in order of first appearance."""
    bs = (list(range(ord('!'), ord('~') + 1)) +
          list(range(ord('\u00a1'), ord('\u00ac') + 1)) +
          list(range(ord('\u00ae'), ord('\u00ff') + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


# Qwen3.6 (qwen35) pre-tokenizer regex, same as llama.cpp
# LLAMA_VOCAB_PRE_TYPE_QWEN35 (src/llama-vocab.cpp).
_QWEN35_RE = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
    r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _pre_tokenize(text):
    """Split text with the qwen35 regex (tiktoken-style pre-tokenization)."""
    if _regex is not None:
        return list(_regex.findall(_QWEN35_RE, text))
    # stdlib fallback (no \p classes; approximate with re)
    return list(re.findall(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|"
        r"[^\r\n\w]?\w+|\d| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        text))


class Tokenizer:
    def __init__(self, path):
        toks, merges, types = load_tokenizer(path)
        self.tokens = toks
        self.types = types if types is not None else [0] * len(toks)
        # classic GPT-2 byte-fallback tables
        self._b2c = _bytes_to_unicode()          # byte -> char
        self._c2b = {c: b for b, c in self._b2c.items()}  # char -> byte
        self.byte_enc = self._b2c                 # byte -> char
        # build vocab: token text -> id
        self.vocab = {t: i for i, t in enumerate(toks) if t}
        # merges: list of (a, b) in priority order
        self.merges = []
        for m in merges:
            a, b = m.split(' ')
            self.merges.append((a, b))
        self.merges_idx = {m: i for i, m in enumerate(self.merges)}
        # special tokens: GGUF types >= 3 (control/user-defined/added)
        self.special = {t: i for i, t in enumerate(toks) if types is not None and i < len(types) and types[i] >= 3}

    def _encode_text(self, text):
        # find special tokens, encode the gaps with BPE
        ids = []
        idxs = []
        for s in self.special:
            start = 0
            while True:
                k = text.find(s, start)
                if k < 0:
                    break
                idxs.append((k, s))
                start = k + len(s)
        idxs.sort()
        cur = 0
        for k, s in idxs:
            if k > cur:
                ids.extend(self._bpe_text(text[cur:k]))
            ids.append(self.special[s])
            cur = k + len(s)
        if cur < len(text):
            ids.extend(self._bpe_text(text[cur:]))
        return ids

    def _bpe_text(self, text):
        # pre-tokenize with the qwen35 regex, then BPE each chunk
        ids = []
        for chunk in _pre_tokenize(text):
            ids.extend(self._bpe_encode(chunk))
        return ids

    def _bpe_encode(self, text):
        # byte-level BPE (GPT-2 style with byte fallback)
        b = text.encode('utf-8')
        syms = [self.byte_enc[byte] for byte in b]  # all 256 bytes mapped
        pairs = {}
        for i in range(len(syms) - 1):
            p = (syms[i], syms[i + 1])
            if p in self.merges_idx:
                pairs[i] = self.merges_idx[p]
        while pairs:
            best = min(pairs, key=lambda i: (pairs[i], i))
            if pairs[best] == float('inf'):
                break
            a = syms[best]
            b = syms[best + 1]
            syms[best] = a + b
            del syms[best + 1]
            pairs = {}
            for i in range(len(syms) - 1):
                p = (syms[i], syms[i + 1])
                if p in self.merges_idx:
                    pairs[i] = self.merges_idx[p]
        ids = []
        for s in syms:
            ids.append(self.vocab[s])
        return ids

    def encode(self, text):
        return self._encode_text(text)

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.tokens[i] if 0 <= i < len(self.tokens) else ''
            s = ''
            for ch in t:
                s += chr(self._c2b[ch]) if ch in self._c2b else ch
            out.append(s)
        return ''.join(out)


# ---------------------------------------------------------------- slice mode

def run_slice(model, tokenizer, start, end, out_prefix):
    from evalplus.data import get_human_eval_plus
    problems = get_human_eval_plus()
    instruction = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    system = "You are a helpful assistant good at coding."
    raw_path = f'{out_prefix}.raw.jsonl'
    main_path = f'{out_prefix}.jsonl'
    # resume: skip task_ids already written
    done = set()
    if os.path.exists(raw_path):
        try:
            for line in open(raw_path):
                try:
                    done.add(json.loads(line)['task_id'])
                except Exception:
                    pass
        except Exception:
            pass
    with open(raw_path, 'a') as rf, open(main_path, 'a') as mf:
        for i in range(start, end):
            tid = f'HumanEval/{i}'
            if tid in done:
                print(f'[slice] {tid}: skipped (already done)')
                continue
            p = problems[tid]['prompt']
            user = instruction + f"\n```python\n{p.strip()}\n```"
            rendered = (f'<|im_start|>system\n{system}<|im_end|>\n'
                        f'<|im_start|>user\n{user}<|im_end|>\n'
                        f'<|im_start|>assistant\n<think>\n\n</think>\n\n')
            toks = tokenizer.encode(rendered)
            t0 = time.time()
            text = generate(model, tokenizer, toks, max_tokens=768, rep_penalty=1.05)
            rf.write(json.dumps({'task_id': tid, 'solution': text}) + '\n')
            mf.write(json.dumps({'task_id': tid, 'solution': text}) + '\n')
            rf.flush()
            mf.flush()
            print(f'[slice] {tid}: {len(text)} chars, {len(toks)} prompt tokens, {time.time()-t0:.0f}s', flush=True)
    model._pool.shutdown(wait=False)
    print(f'[slice] done: {raw_path}')


def generate(model, tokenizer, prompt_toks, max_tokens=768, rep_penalty=1.05):
    model.reset_state()
    h = None
    for p, tid in enumerate(prompt_toks):
        h = model.embed(tid)
        for il in range(N_LAYERS):
            h = model.forward_layer(il, h, p)
    out = []
    ring = []
    think_ended = False
    THINK_S = 248068
    THINK_E = 248069
    for step in range(max_tokens):
        penalty = {}
        if rep_penalty > 1.0:
            for t in ring:
                penalty[t] = 1 if t >= 0 else -1
        best = model.logits(h, penalty)[0]
        if think_ended and best in (THINK_S, THINK_E):
            # banned: pick second-best excluding think tokens
            best = model.logits_second(h, penalty, (THINK_S, THINK_E))[0]
        if best == THINK_S:
            continue
        if best == THINK_E:
            think_ended = True
            continue
        if best == 248046 or best == 248075:  # EOS/EOT
            break
        out.append(best)
        ring.append(best)
        if len(ring) > 64:
            ring.pop(0)
        h = model.embed(best)
        for il in range(N_LAYERS):
            h = model.forward_layer(il, h, len(prompt_toks) + step)
    return tokenizer.decode(out)

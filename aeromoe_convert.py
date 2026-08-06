#!/usr/bin/env python3
"""
aeromoe_convert.py
==================
safetensors → .aeromoe converter
Target: Qwen3.6-35B-A3B and compatible Qwen3 MoE variants
         (Qwen3-30B-A3B, Qwen3.6-35B-A3B-Instruct, abliterated forks, etc.)

.aeromoe format layout (all sections 64 KB-aligned)
────────────────────────────────────────────────────
  [0x0000_0000]  FILE HEADER      512 B   magic + version + model config
  [0x0000_0200]  DENSE INDEX      ≤64 KB  offset/shape table for backbone tensors
  [aligned]      EXPERT INDEX     var     offset table for every (layer, expert) slab
  [aligned]      DENSE WEIGHTS    var     all non-expert tensors, tightly packed
  [aligned]      EXPERT SLABS     var     one aligned slab per (layer, expert)

Expert slab internal layout
───────────────────────────
  gate_proj.weight  |  up_proj.weight  |  down_proj.weight
  (all bf16/fp16, row-major, no padding between projections)
  Slab is zero-padded to the next 64 KB boundary.

Dense index entry (40 B each)
──────────────────────────────
  name_hash  : uint64   FNV-1a of tensor name
  offset     : uint64   byte offset from file start
  nbytes     : uint64   raw byte count
  ndim       : uint32   number of dimensions (≤ 4)
  shape      : uint32×4 dimension sizes (unused dims = 0)
  dtype_code : uint32   0=bf16 1=fp16 2=fp32 3=fp8_e4m3

Expert index entry (24 B each)
───────────────────────────────
  layer_idx  : uint32
  expert_idx : uint32
  offset     : uint64   byte offset from file start
  nbytes     : uint64   slab byte count (includes padding)

Usage
─────
  python aeromoe_convert.py \\
      --model-dir /path/to/Qwen3.6-35B-A3B-Instruct \\
      --output    output.aeromoe \\
      [--dtype    bfloat16]    # keep source dtype if omitted
      [--verify]               # checksum re-read after write
      [--workers  4]           # parallel shard-loading threads
      [--dry-run]              # print plan, do not write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ─── optional tqdm ──────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable
            self._desc = kwargs.get("desc", "")
            self._total = kwargs.get("total", None)
            self._n = 0
        def __iter__(self):
            for item in self._it:
                self._n += 1
                pct = f"{100*self._n/self._total:.0f}%" if self._total else str(self._n)
                print(f"\r{self._desc}: {pct}  ", end="", flush=True)
                yield item
            print()
        def __enter__(self): return self
        def __exit__(self, *a): print()
        def update(self, n=1):
            self._n += n
            pct = f"{100*self._n/self._total:.0f}%" if self._total else str(self._n)
            print(f"\r{self._desc}: {pct}  ", end="", flush=True)

# ─── constants ───────────────────────────────────────────────────────────────

MAGIC           = b"AEROMOE\x00"   # 8 bytes
FORMAT_VERSION  = 0x0001_0000      # major=1, minor=0
ALIGN           = 64 * 1024        # 64 KB alignment

HEADER_SIZE     = 512              # bytes, fixed
DENSE_IDX_ENTRY = 40               # bytes per dense index entry
EXPERT_IDX_ENTRY= 24               # bytes per expert index entry

# dtype codes used in the file format
DTYPE_BF16  = 0
DTYPE_FP16  = 1
DTYPE_FP32  = 2
DTYPE_FP8   = 3   # e4m3

# map numpy dtype → format dtype code
NP_TO_DTYPE_CODE: Dict[str, int] = {
    "bfloat16": DTYPE_BF16,
    "float16":  DTYPE_FP16,
    "float32":  DTYPE_FP32,
}

# Tensor name patterns that are MoE expert weights
# Qwen3 naming: model.layers.{i}.mlp.experts.{j}.{proj}.weight
EXPERT_PROJ_KEYS = ("gate_proj", "up_proj", "down_proj")
EXPERT_PROJ_ORDER = list(EXPERT_PROJ_KEYS)   # gate | up | down inside slab

# Shared expert (always-active) keys — treated as dense backbone
SHARED_EXPERT_KEYS = (
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",          # variant naming
)

# ─── tiny FNV-1a hash ────────────────────────────────────────────────────────

def fnv1a_64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFF_FFFF_FFFF_FFFF
    return h

# ─── data classes ────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Distilled from config.json — only fields AeroMoE engine needs."""
    model_type:              str   = "qwen3_moe"
    hidden_size:             int   = 4096
    intermediate_size:       int   = 2048      # dense (shared expert)
    moe_intermediate_size:   int   = 768       # per-expert dim
    num_hidden_layers:       int   = 94
    num_attention_heads:     int   = 64
    num_key_value_heads:     int   = 4
    vocab_size:              int   = 151936
    max_position_embeddings: int   = 131072
    num_experts:             int   = 128       # routed experts per layer
    num_experts_per_tok:     int   = 8         # top-k active
    num_shared_experts:      int   = 1
    norm_topk_prob:          bool  = True
    rms_norm_eps:            float = 1e-6
    rope_theta:              float = 1_000_000.0
    rope_scaling_type:       str   = "yarn"
    tie_word_embeddings:     bool  = False
    dtype:                   str   = "bfloat16"  # target storage dtype

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        with open(path) as f:
            cfg = json.load(f)
        obj = cls()
        mapping = {
            "hidden_size":             "hidden_size",
            "intermediate_size":       "intermediate_size",
            "moe_intermediate_size":   "moe_intermediate_size",
            "num_hidden_layers":       "num_hidden_layers",
            "num_attention_heads":     "num_attention_heads",
            "num_key_value_heads":     "num_key_value_heads",
            "vocab_size":              "vocab_size",
            "max_position_embeddings": "max_position_embeddings",
            "num_experts":             "num_experts",
            "num_experts_per_tok":     "num_experts_per_tok",
            "num_shared_experts":      "num_shared_experts",
            "norm_topk_prob":          "norm_topk_prob",
            "rms_norm_eps":            "rms_norm_eps",
            "rope_theta":              "rope_theta",
        }
        for json_key, field_name in mapping.items():
            if json_key in cfg:
                setattr(obj, field_name, cfg[json_key])
        # rope_scaling sub-dict
        if "rope_scaling" in cfg and isinstance(cfg["rope_scaling"], dict):
            obj.rope_scaling_type = cfg["rope_scaling"].get("type", "yarn")
        obj.model_type = cfg.get("model_type", "qwen3_moe")
        obj.tie_word_embeddings = cfg.get("tie_word_embeddings", False)
        return obj

    def active_param_bytes(self, dtype_bytes: int = 2) -> int:
        """Rough active-parameter byte count for sanity-check reporting."""
        attn = self.hidden_size * (
            self.num_attention_heads * (self.hidden_size // self.num_attention_heads) +
            self.num_key_value_heads * (self.hidden_size // self.num_attention_heads) * 2 +
            self.hidden_size
        )
        expert_active = self.num_experts_per_tok * 3 * self.hidden_size * self.moe_intermediate_size
        shared = self.num_shared_experts * 3 * self.hidden_size * self.intermediate_size
        per_layer = attn + expert_active + shared
        return per_layer * self.num_hidden_layers * dtype_bytes


@dataclass
class TensorMeta:
    name:   str
    dtype:  str          # numpy dtype string
    shape:  Tuple[int, ...]
    shard:  Path         # source safetensors file
    offset: int          # data_offset inside the safetensors DATA section
    nbytes: int          # raw byte count in source

    @property
    def is_expert(self) -> bool:
        return ".mlp.experts." in self.name

    @property
    def expert_layer(self) -> int:
        # model.layers.{i}.mlp.experts.{j}.xxx
        parts = self.name.split(".")
        return int(parts[2])

    @property
    def expert_idx(self) -> int:
        parts = self.name.split(".")
        return int(parts[5])

    @property
    def expert_proj(self) -> str:
        # gate_proj / up_proj / down_proj
        parts = self.name.split(".")
        return parts[6]  # e.g. "gate_proj"

# ─── safetensors reader ──────────────────────────────────────────────────────

SAFETENSOR_DTYPE_TO_NP = {
    "BF16": "bfloat16",
    "F16":  "float16",
    "F32":  "float32",
    "F8_E4M3": "float8_e4m3fn",   # numpy may not have this — handled below
    "I32":  "int32",
    "I64":  "int64",
}

def _read_safetensors_header(path: Path) -> Tuple[dict, int]:
    """Return (header_dict, data_start_offset)."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header_raw = f.read(header_len)
    header = json.loads(header_raw)
    data_start = 8 + header_len
    return header, data_start


def iter_shard_tensors(shard: Path) -> Iterator[TensorMeta]:
    """Yield TensorMeta for every tensor in one safetensors shard."""
    header, data_start = _read_safetensors_header(shard)
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype_str = SAFETENSOR_DTYPE_TO_NP.get(meta["dtype"], meta["dtype"].lower())
        shape = tuple(meta["data_offsets"][1] - meta["data_offsets"][0]
                      for _ in [1])  # placeholder — use actual shape below
        shape = tuple(meta.get("shape", []))
        raw_start, raw_end = meta["data_offsets"]
        yield TensorMeta(
            name   = name,
            dtype  = dtype_str,
            shape  = shape,
            shard  = shard,
            offset = data_start + raw_start,
            nbytes = raw_end - raw_start,
        )


def discover_shards(model_dir: Path) -> List[Path]:
    """Find all .safetensors shards, ordered by name."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
    return shards


def load_all_tensor_meta(model_dir: Path, workers: int = 4) -> Dict[str, TensorMeta]:
    """Scan all shards in parallel, return name → TensorMeta."""
    shards = discover_shards(model_dir)
    print(f"  Found {len(shards)} safetensors shard(s)")
    all_meta: Dict[str, TensorMeta] = {}
    lock = threading.Lock()

    def _scan(shard: Path):
        local = {m.name: m for m in iter_shard_tensors(shard)}
        with lock:
            all_meta.update(local)

    with ThreadPoolExecutor(max_workers=min(workers, len(shards))) as ex:
        futs = [ex.submit(_scan, s) for s in shards]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="  Scanning shards"):
            fut.result()   # re-raise exceptions

    return all_meta

# ─── dtype conversion helpers ────────────────────────────────────────────────

def _dtype_itemsize(dtype_str: str) -> int:
    sizes = {"bfloat16": 2, "float16": 2, "float32": 4,
             "float8_e4m3fn": 1, "int32": 4, "int64": 8}
    return sizes.get(dtype_str, 2)

def _load_raw_bytes(meta: TensorMeta) -> bytes:
    """Read raw bytes from source shard (zero-copy via pread-style seek)."""
    with open(meta.shard, "rb") as f:
        f.seek(meta.offset)
        return f.read(meta.nbytes)

def _convert_to_target(raw: bytes, src_dtype: str, tgt_dtype: str) -> bytes:
    """
    Convert raw bytes from src_dtype to tgt_dtype.
    Supported paths: fp32→bf16, fp32→fp16, fp16→bf16, bf16→fp16, identity.
    For bf16 we treat numpy uint16 with reinterpretation.
    """
    if src_dtype == tgt_dtype:
        return raw

    src_map = {"float32": np.float32, "float16": np.float16,
               "bfloat16": None,        # special-cased below
               "int32": np.int32, "int64": np.int64}

    if src_dtype == "bfloat16":
        # reinterpret as uint16 then bit-shift to float32
        u16 = np.frombuffer(raw, dtype=np.uint16)
        f32 = np.zeros(len(u16), dtype=np.float32)
        np.copyto(f32.view(np.uint32), u16.astype(np.uint32) << 16)
        arr = f32
    else:
        arr = np.frombuffer(raw, dtype=src_map[src_dtype])

    if tgt_dtype == "bfloat16":
        # truncate float32 mantissa → bf16 stored as uint16
        f32 = arr.astype(np.float32)
        u32 = f32.view(np.uint32)
        # round-to-nearest-even
        rounding = (u32 >> 16) & 1
        u32r = u32 + 0x7FFF + rounding
        u16 = (u32r >> 16).astype(np.uint16)
        return u16.tobytes()
    elif tgt_dtype == "float16":
        return arr.astype(np.float16).tobytes()
    elif tgt_dtype == "float32":
        return arr.astype(np.float32).tobytes()
    else:
        raise ValueError(f"Unsupported target dtype: {tgt_dtype}")

# ─── alignment helpers ───────────────────────────────────────────────────────

def align_up(n: int, alignment: int = ALIGN) -> int:
    return (n + alignment - 1) & ~(alignment - 1)

def pad_to_align(f, alignment: int = ALIGN):
    """Write zero bytes until file position is aligned."""
    pos = f.tell()
    rem = pos % alignment
    if rem:
        f.write(b"\x00" * (alignment - rem))

# ─── file header ─────────────────────────────────────────────────────────────

def build_file_header(cfg: ModelConfig, dtype_code: int,
                      dense_idx_offset: int,  dense_idx_count: int,
                      expert_idx_offset: int, expert_idx_count: int,
                      dense_data_offset: int,
                      expert_data_offset: int) -> bytes:
    """
    Pack the 512-byte file header.

    Layout (all little-endian):
      0x000   magic           8 B
      0x008   version         4 B  uint32
      0x00C   dtype_code      4 B  uint32
      0x010   hidden_size     4 B  uint32
      0x014   inter_size      4 B  uint32  (dense / shared expert)
      0x018   moe_inter_size  4 B  uint32  (per-expert)
      0x01C   num_layers      4 B  uint32
      0x020   n_heads         4 B  uint32
      0x024   n_kv_heads      4 B  uint32
      0x028   vocab_size      4 B  uint32
      0x02C   max_seq_len     4 B  uint32
      0x030   num_experts     4 B  uint32  (routed, per layer)
      0x034   num_experts_tok 4 B  uint32  (top-k)
      0x038   num_shared_exp  4 B  uint32
      0x03C   norm_topk_prob  1 B  uint8
      0x03D   tie_embeddings  1 B  uint8
      0x03E   _pad            2 B
      0x040   rms_norm_eps    4 B  float32
      0x044   rope_theta      4 B  float32
      0x048   rope_type       8 B  char[8]  e.g. "yarn\0..."
      0x050   dense_idx_off   8 B  uint64
      0x058   dense_idx_count 4 B  uint32
      0x05C   _pad            4 B
      0x060   expert_idx_off  8 B  uint64
      0x068   expert_idx_cnt  4 B  uint32
      0x06C   _pad            4 B
      0x070   dense_data_off  8 B  uint64
      0x078   expert_data_off 8 B  uint64
      0x080   reserved        0x180 B  (zeros, future use)
      0x200   [end of header - 512 bytes total]
    """
    buf = bytearray(512)
    # magic + version
    struct.pack_into("<8sI", buf, 0x000, MAGIC, FORMAT_VERSION)
    struct.pack_into("<I", buf, 0x00C, dtype_code)
    # model config
    struct.pack_into("<IIIIIII", buf, 0x010,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.moe_intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
    )
    struct.pack_into("<I", buf, 0x02C, cfg.max_position_embeddings)
    struct.pack_into("<III", buf, 0x030,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.num_shared_experts,
    )
    struct.pack_into("<BB", buf, 0x03C,
        int(cfg.norm_topk_prob),
        int(cfg.tie_word_embeddings),
    )
    struct.pack_into("<ff", buf, 0x040, cfg.rms_norm_eps, cfg.rope_theta)
    # rope type string, null-padded to 8 bytes
    rope_b = cfg.rope_scaling_type.encode()[:8].ljust(8, b"\x00")
    struct.pack_into("<8s", buf, 0x048, rope_b)
    # index / data offsets
    struct.pack_into("<QI", buf, 0x050, dense_idx_offset, dense_idx_count)
    struct.pack_into("<QI", buf, 0x060, expert_idx_offset, expert_idx_count)
    struct.pack_into("<QQ", buf, 0x070, dense_data_offset, expert_data_offset)
    return bytes(buf)

# ─── index builders ──────────────────────────────────────────────────────────

def build_dense_index(dense_tensors: List[TensorMeta],
                      offsets: List[int]) -> bytes:
    """
    Encode the dense index as a packed byte string.
    Each entry is DENSE_IDX_ENTRY (40) bytes.
    """
    entries = bytearray()
    for meta, off in zip(dense_tensors, offsets):
        h = fnv1a_64(meta.name)
        code = NP_TO_DTYPE_CODE.get(meta.dtype, DTYPE_BF16)
        # shape: up to 4 dims, zero-padded
        shape4 = list(meta.shape) + [0] * (4 - len(meta.shape))
        shape4 = shape4[:4]
        nbytes_stored = meta.nbytes if meta.dtype == "bfloat16" else \
                        meta.nbytes   # conversion sizes handled at write time
        entry = struct.pack("<QQQIIiii",
            h, off, meta.nbytes,
            len(meta.shape), *shape4)
        # fix: struct above is 8+8+8+4+16 = 44, need 40 — adjust
        # Correct layout: hash(8) offset(8) nbytes(8) ndim(4) shape(4×4=16) = 44 B
        # Widen entry size to 48 B (round up nicely), update constant
        entries += entry
    return bytes(entries)

# NOTE: recalculate — hash(8)+offset(8)+nbytes(8)+ndim(4)+shape[4](16) = 44 bytes
# We'll use 48 bytes per entry (add 4 B pad) for 16-byte alignment.
DENSE_IDX_ENTRY = 48   # override module constant

def _pack_dense_entry(name: str, offset: int, nbytes: int,
                      shape: tuple, dtype_code: int) -> bytes:
    h = fnv1a_64(name)
    ndim = len(shape)
    s = list(shape) + [0] * (4 - min(ndim, 4))
    s = s[:4]
    # 8+8+8+4+4+4×4+4 = 48 bytes
    return struct.pack("<QQQIiiiII",
        h, offset, nbytes, ndim,
        s[0], s[1], s[2], s[3], dtype_code)
# that's 8+8+8+4+4+4+4+4+4 = 48 ✓

def _pack_expert_entry(layer: int, expert: int,
                       offset: int, nbytes: int) -> bytes:
    # uint32 + uint32 + uint64 + uint64 = 24 bytes ✓
    return struct.pack("<IIQQ", layer, expert, offset, nbytes)

# ─── main converter class ────────────────────────────────────────────────────

class AeroMoEConverter:
    """
    Orchestrates the full safetensors → .aeromoe conversion.

    Pass 1 — scan metadata: identify dense vs expert tensors, compute
             byte sizes after dtype conversion, lay out file sections.
    Pass 2 — write: emit header placeholder, indexes, dense weights,
             then expert slabs.
    Pass 3 (optional) — verify: re-read random sample and checksum.
    """

    def __init__(
        self,
        model_dir: Path,
        output_path: Path,
        target_dtype: Optional[str] = None,
        workers: int = 4,
        dry_run: bool = False,
        verify: bool = False,
    ):
        self.model_dir    = model_dir
        self.output_path  = output_path
        self.target_dtype = target_dtype      # None = keep source dtype
        self.workers      = workers
        self.dry_run      = dry_run
        self.verify       = verify

        self.cfg: Optional[ModelConfig] = None
        self.all_meta: Dict[str, TensorMeta] = {}

        # Split after scanning
        self.dense_tensors:  List[TensorMeta] = []
        self.expert_tensors: Dict[Tuple[int,int], Dict[str, TensorMeta]] = {}
        # (layer, expert) → {proj_name: meta}

    # ── Step 1: scan ─────────────────────────────────────────────────────────

    def scan(self):
        print("\n[1/4] Scanning model directory …")
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_dir}")
        self.cfg = ModelConfig.from_json(config_path)
        self._print_config()

        self.all_meta = load_all_tensor_meta(self.model_dir, self.workers)
        print(f"  Total tensors found: {len(self.all_meta)}")

        self._split_tensors()
        self._print_split_summary()

    def _print_config(self):
        c = self.cfg
        print(f"  Model type          : {c.model_type}")
        print(f"  Hidden size         : {c.hidden_size}")
        print(f"  MoE inter size      : {c.moe_intermediate_size}")
        print(f"  Layers              : {c.num_hidden_layers}")
        print(f"  Experts/layer       : {c.num_experts}  (top-{c.num_experts_per_tok} active)")
        print(f"  Shared experts      : {c.num_shared_experts}")
        print(f"  Attention heads     : {c.num_attention_heads} / KV {c.num_kv_heads}")
        print(f"  Vocab               : {c.vocab_size}")
        active_gb = c.active_param_bytes() / 1e9
        print(f"  Est. active params  : ~{active_gb:.2f} GB (bf16)")

    def _split_tensors(self):
        """Partition all_meta into dense and expert groups."""
        for name, meta in self.all_meta.items():
            if meta.is_expert:
                key = (meta.expert_layer, meta.expert_idx)
                if key not in self.expert_tensors:
                    self.expert_tensors[key] = {}
                self.expert_tensors[key][meta.expert_proj] = meta
            else:
                self.dense_tensors.append(meta)

        # Sort dense by name for determinism
        self.dense_tensors.sort(key=lambda m: m.name)

    def _print_split_summary(self):
        n_exp_layers = len({k[0] for k in self.expert_tensors})
        n_exp_slots  = len(self.expert_tensors)
        dense_gb = sum(m.nbytes for m in self.dense_tensors) / 1e9
        expert_gb = sum(
            sum(m.nbytes for m in projs.values())
            for projs in self.expert_tensors.values()
        ) / 1e9
        print(f"  Dense tensors       : {len(self.dense_tensors)}  ({dense_gb:.2f} GB)")
        print(f"  Expert (layer,slot) : {n_exp_slots} across {n_exp_layers} MoE layers")
        print(f"  Expert weights      : {expert_gb:.2f} GB")
        print(f"  Total on-disk       : {(dense_gb+expert_gb):.2f} GB")

    # ── Step 2: layout planning ───────────────────────────────────────────────

    def _resolve_dtype(self, meta: TensorMeta) -> str:
        return self.target_dtype if self.target_dtype else meta.dtype

    def _converted_nbytes(self, meta: TensorMeta) -> int:
        tgt = self._resolve_dtype(meta)
        n_elements = math.prod(meta.shape) if meta.shape else meta.nbytes // _dtype_itemsize(meta.dtype)
        return n_elements * _dtype_itemsize(tgt)

    def plan_layout(self) -> dict:
        """
        Compute all file offsets without writing anything.
        Returns a layout dict used by both dry-run and actual write.
        """
        dtype_code = NP_TO_DTYPE_CODE.get(
            self.target_dtype or self.dense_tensors[0].dtype,
            DTYPE_BF16
        )

        # ── Section offsets ───────────────────────────────────────────────
        header_end   = HEADER_SIZE                              # 512
        dense_idx_off = align_up(header_end)                   # 64 KB

        n_dense = len(self.dense_tensors)
        dense_idx_bytes = align_up(n_dense * DENSE_IDX_ENTRY)

        n_expert = len(self.expert_tensors)
        expert_idx_off = dense_idx_off + dense_idx_bytes
        expert_idx_bytes = align_up(n_expert * EXPERT_IDX_ENTRY)

        dense_data_off = expert_idx_off + expert_idx_bytes

        # Dense data: tensors packed tightly, section 64 KB-aligned total
        dense_tensor_offsets = []
        cursor = dense_data_off
        for meta in self.dense_tensors:
            dense_tensor_offsets.append(cursor)
            cursor += self._converted_nbytes(meta)
        dense_data_end = align_up(cursor)

        expert_data_off = dense_data_end

        # Expert slabs: sorted by (layer, expert_idx)
        sorted_keys = sorted(self.expert_tensors.keys())
        expert_slab_offsets = {}
        expert_slab_sizes   = {}
        cursor = expert_data_off
        for key in sorted_keys:
            projs = self.expert_tensors[key]
            slab_bytes = sum(
                self._converted_nbytes(projs[p])
                for p in EXPERT_PROJ_ORDER if p in projs
            )
            expert_slab_offsets[key] = cursor
            expert_slab_sizes[key]   = align_up(slab_bytes)
            cursor += align_up(slab_bytes)

        total_size = cursor

        return dict(
            dtype_code          = dtype_code,
            dense_idx_off       = dense_idx_off,
            dense_idx_count     = n_dense,
            expert_idx_off      = expert_idx_off,
            expert_idx_count    = n_expert,
            dense_data_off      = dense_data_off,
            dense_tensor_offsets= dense_tensor_offsets,
            expert_data_off     = expert_data_off,
            expert_slab_offsets = expert_slab_offsets,
            expert_slab_sizes   = expert_slab_sizes,
            sorted_expert_keys  = sorted_keys,
            total_size          = total_size,
        )

    # ── Step 3: write ─────────────────────────────────────────────────────────

    def write(self, layout: dict):
        print(f"\n[3/4] Writing {self.output_path} …")
        print(f"  Estimated output size: {layout['total_size']/1e9:.2f} GB")

        if self.dry_run:
            print("  [DRY RUN] — no file written.")
            self._print_layout(layout)
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_path, "wb") as f:
            # ── Header (placeholder — will patch offsets) ──────────────────
            hdr = build_file_header(
                self.cfg,
                layout["dtype_code"],
                layout["dense_idx_off"],  layout["dense_idx_count"],
                layout["expert_idx_off"], layout["expert_idx_count"],
                layout["dense_data_off"],
                layout["expert_data_off"],
            )
            f.write(hdr)

            # ── Dense index ────────────────────────────────────────────────
            pad_to_align(f)
            assert f.tell() == layout["dense_idx_off"], \
                f"Dense index offset mismatch: {f.tell()} != {layout['dense_idx_off']}"

            for meta, off in zip(self.dense_tensors, layout["dense_tensor_offsets"]):
                nbytes_out = self._converted_nbytes(meta)
                dtype_code = NP_TO_DTYPE_CODE.get(self._resolve_dtype(meta), DTYPE_BF16)
                entry = _pack_dense_entry(meta.name, off, nbytes_out,
                                          meta.shape, dtype_code)
                f.write(entry)
            pad_to_align(f)

            # ── Expert index ───────────────────────────────────────────────
            assert f.tell() == layout["expert_idx_off"], \
                f"Expert index offset mismatch: {f.tell()} != {layout['expert_idx_off']}"

            for key in layout["sorted_expert_keys"]:
                layer, expert = key
                entry = _pack_expert_entry(
                    layer, expert,
                    layout["expert_slab_offsets"][key],
                    layout["expert_slab_sizes"][key],
                )
                f.write(entry)
            pad_to_align(f)

            # ── Dense weights ──────────────────────────────────────────────
            assert f.tell() == layout["dense_data_off"], \
                f"Dense data offset mismatch: {f.tell()} != {layout['dense_data_off']}"

            with tqdm(total=len(self.dense_tensors), desc="  Dense weights") as bar:
                for meta in self.dense_tensors:
                    raw   = _load_raw_bytes(meta)
                    tgt   = self._resolve_dtype(meta)
                    data  = _convert_to_target(raw, meta.dtype, tgt)
                    f.write(data)
                    bar.update(1)
            pad_to_align(f)

            # ── Expert slabs ───────────────────────────────────────────────
            assert f.tell() == layout["expert_data_off"], \
                f"Expert data offset mismatch: {f.tell()} != {layout['expert_data_off']}"

            total_slabs = len(layout["sorted_expert_keys"])
            with tqdm(total=total_slabs, desc="  Expert slabs") as bar:
                for key in layout["sorted_expert_keys"]:
                    projs = self.expert_tensors[key]
                    slab_start = f.tell()
                    for proj in EXPERT_PROJ_ORDER:
                        if proj not in projs:
                            # Warn but don't crash — some models omit bias
                            print(f"\n  WARNING: missing {proj} for expert {key}")
                            continue
                        meta = projs[proj]
                        raw  = _load_raw_bytes(meta)
                        tgt  = self._resolve_dtype(meta)
                        data = _convert_to_target(raw, meta.dtype, tgt)
                        f.write(data)
                    # pad slab to alignment
                    slab_written = f.tell() - slab_start
                    expected_size = layout["expert_slab_sizes"][key]
                    pad = expected_size - slab_written
                    if pad > 0:
                        f.write(b"\x00" * pad)
                    bar.update(1)

        actual_size = self.output_path.stat().st_size
        print(f"\n  Written: {actual_size/1e9:.3f} GB → {self.output_path}")

    def _print_layout(self, layout: dict):
        print(f"\n  Layout plan:")
        print(f"    Header         : 0x{0:08X} – 0x{HEADER_SIZE:08X}  ({HEADER_SIZE} B)")
        print(f"    Dense index    : 0x{layout['dense_idx_off']:08X}  ({layout['dense_idx_count']} entries × {DENSE_IDX_ENTRY} B)")
        print(f"    Expert index   : 0x{layout['expert_idx_off']:08X}  ({layout['expert_idx_count']} entries × {EXPERT_IDX_ENTRY} B)")
        print(f"    Dense weights  : 0x{layout['dense_data_off']:08X}")
        print(f"    Expert slabs   : 0x{layout['expert_data_off']:08X}")
        print(f"    Total          : {layout['total_size']/1e9:.2f} GB")

    # ── Step 4: verify ────────────────────────────────────────────────────────

    def verify_output(self, layout: dict, n_samples: int = 32):
        """
        Re-read n_samples random tensors from the output file and compare
        SHA-256 digest against a freshly-loaded-and-converted source.
        """
        print("\n[4/4] Verifying output …")
        import random, hashlib
        random.seed(42)

        # Build a combined list: (name, src_meta, file_offset, file_nbytes)
        candidates = []
        for meta, off in zip(self.dense_tensors, layout["dense_tensor_offsets"]):
            candidates.append((meta.name, meta, off, self._converted_nbytes(meta)))

        sample = random.sample(candidates, min(n_samples, len(candidates)))
        errors = 0

        with open(self.output_path, "rb") as f:
            for name, meta, off, nbytes in tqdm(sample, desc="  Verifying"):
                # expected
                raw = _load_raw_bytes(meta)
                tgt = self._resolve_dtype(meta)
                expected = _convert_to_target(raw, meta.dtype, tgt)
                expected_hash = hashlib.sha256(expected).hexdigest()

                # actual
                f.seek(off)
                actual = f.read(nbytes)
                actual_hash = hashlib.sha256(actual).hexdigest()

                if expected_hash != actual_hash:
                    print(f"\n  MISMATCH: {name}")
                    errors += 1

        if errors == 0:
            print(f"  ✓ All {len(sample)} samples verified OK")
        else:
            print(f"  ✗ {errors} mismatch(es) detected — output may be corrupt")
            sys.exit(1)

    # ── Orchestrate ───────────────────────────────────────────────────────────

    def run(self):
        t0 = time.time()
        self.scan()
        print("\n[2/4] Planning file layout …")
        layout = self.plan_layout()
        self._print_layout(layout)
        self.write(layout)
        if self.verify and not self.dry_run:
            self.verify_output(layout)
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s")

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Qwen3 MoE safetensors → .aeromoe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Directory containing safetensors shards + config.json")
    p.add_argument("--output",    required=True, type=Path,
                   help="Output .aeromoe file path")
    p.add_argument("--dtype",
                   choices=["bfloat16", "float16", "float32"],
                   default=None,
                   help="Target storage dtype (default: keep source dtype, usually bfloat16)")
    p.add_argument("--workers",  type=int, default=4,
                   help="Parallel threads for shard scanning (default: 4)")
    p.add_argument("--verify",   action="store_true",
                   help="Re-read random samples after write and checksum them")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print layout plan only, do not write any file")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.model_dir.is_dir():
        print(f"ERROR: model-dir does not exist: {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.dry_run and args.output.exists():
        print(f"WARNING: output file already exists: {args.output}")
        resp = input("  Overwrite? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            sys.exit(0)

    converter = AeroMoEConverter(
        model_dir   = args.model_dir,
        output_path = args.output,
        target_dtype= args.dtype,
        workers     = args.workers,
        dry_run     = args.dry_run,
        verify      = args.verify,
    )
    converter.run()


if __name__ == "__main__":
    main()

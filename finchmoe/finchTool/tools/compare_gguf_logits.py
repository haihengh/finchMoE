#!/usr/bin/env python3
"""Compare finchmoe --gguf first-token logits against llama.cpp logit_dump."""
import numpy as np, sys

def load(p):
    a = np.fromfile(p, dtype=np.float32)
    print(f"{p}: {len(a)} floats, rms={np.sqrt(np.mean(a**2)):.4f}, "
          f"max={a.max():.4f}, min={a.min():.4f}, finite={np.isfinite(a).all()}")
    return a

ref = load(sys.argv[1])
fm  = load(sys.argv[2])
n = min(len(ref), len(fm))
ref, fm = ref[:n], fm[:n]

# cosine similarity
cos = np.dot(ref, fm) / (np.linalg.norm(ref) * np.linalg.norm(fm))
# centered cosine (logit-mean removed — robust to constant offset)
rc, fc = ref - ref.mean(), fm - fm.mean()
ccos = np.dot(rc, fc) / (np.linalg.norm(rc) * np.linalg.norm(fc))
# correlation
corr = np.corrcoef(ref, fm)[0, 1]

argmax_r, argmax_f = int(ref.argmax()), int(fm.argmax())
top10_r = set(np.argsort(ref)[::-1][:10])
top10_f = set(np.argsort(fm)[::-1][:10])
top100_r = set(np.argsort(ref)[::-1][:100])
top100_f = set(np.argsort(fm)[::-1][:100])

print(f"\ncos_sim        = {cos:.6f}")
print(f"centered_cos   = {ccos:.6f}")
print(f"pearson_corr   = {corr:.6f}")
print(f"argmax         = ref {argmax_r} vs finchmoe {argmax_f} {'MATCH' if argmax_r==argmax_f else 'DIFF'}")
print(f"top10 overlap  = {len(top10_r & top10_f)}/10")
print(f"top100 overlap = {len(top100_r & top100_f)}/100")
print(f"mean|diff|     = {np.abs(ref-fm).mean():.4f}  max|diff| = {np.abs(ref-fm).max():.4f}")

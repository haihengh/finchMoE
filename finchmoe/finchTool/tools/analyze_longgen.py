#!/usr/bin/env python3
"""Analyze a finchmoe-infer generation log for long-generation health:
repetition, EOS handling, think-tag behavior, coherence heuristics."""
import re, sys
from collections import Counter

def parse(log):
    """Extract generated text from [gen N/M] token_id=... lines and the output section."""
    text_parts = []
    for line in open(log, errors='replace'):
        m = re.match(r'^(.*?)  \[gen (\d+)/', line)
        if m:
            text_parts.append(m.group(1).strip())
        m2 = re.match(r'^(.*?)  \[eos\]', line)
        if m2:
            text_parts.append(m2.group(1).strip() + ' <EOS>')
    return ' '.join(t for t in text_parts if t)

def ngram_reps(text, n=16):
    """Find word n-grams that repeat >= 2 times."""
    words = text.split()
    if len(words) < n: return []
    c = Counter(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    return [(k, v) for k, v in c.items() if v >= 2]

def main():
    log = sys.argv[1] if len(sys.argv) > 1 else '/tmp/longgen_full.log'
    text = parse(log)
    words = text.split()
    print(f"generated: {len(words)} words, {len(text)} chars")
    if not text:
        print("NO TEXT PARSED")
        return
    print(f"\n--- first 200 chars ---\n{text[:200]}")
    print(f"\n--- last 200 chars ---\n{text[-200:]}")
    # repetition analysis
    for n in (8, 16, 32):
        reps = ngram_reps(text, n)
        print(f"{n}-gram repeats: {len(reps)}")
        for k, v in reps[:5]:
            print(f"   x{v}: {' '.join(k)[:120]}")
    # EOS
    print(f"\nEOS reached: {'<EOS>' in text}")
    # think tags
    print(f"think tags: open={text.count('<think>')} close={text.count('</think>')}")
    # unique-word ratio (last 500 words vs first 500)
    if len(words) > 1000:
        a = Counter(words[:500]); b = Counter(words[-500:])
        shared = sum(min(a[w], b[w]) for w in a if w in b)
        print(f"first500/last500 shared word mass: {shared/500:.2%} (low = topical drift)")

if __name__ == '__main__':
    main()

# Diagnostic: compare each tensor's expected byte size (per gguf-py's type
# table) against the actual gap between consecutive tensor offsets in the file.
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\code\llama.cpp\gguf-py")

from gguf.gguf_reader import GGUFReader, ReaderField
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType


class DiagReader(GGUFReader):
    def _build_tensors(self, start_offs: int, fields: list[ReaderField]) -> None:
        infos = []
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            name = str(bytes(name_data), encoding="utf-8")
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = 1
            for dim in dims.tolist():
                n_elems *= int(dim)
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            infos.append((name, ggml_type, tuple(dims.tolist()), n_elems, n_bytes, int(start_offs + offset_tensor[0])))
        infos.sort(key=lambda t: t[5])
        by_type = defaultdict(lambda: [0, 0, 0])
        print(f"{'tensor':<28} {'type':<10} {'dims':<24} {'exp_bytes':>12} {'actual':>12} {'delta':>10}")
        for i, (name, t, d, n_elems, n_bytes, off) in enumerate(infos):
            if i + 1 < len(infos):
                actual = infos[i + 1][5] - off
            else:
                actual = None  # last tensor: unknown file length contribution
            if actual is None:
                continue
            by_type[t][0] += 1
            by_type[t][1] += n_bytes
            by_type[t][2] += actual
            if n_bytes != actual:
                print(f"{name:<28} {t.name:<10} {str(d):<24} {n_bytes:>12} {actual:>12} {actual - n_bytes:>+10}")
        print("\nper-type summary (expected vs actual total bytes):")
        for t, (cnt, exp, act) in sorted(by_type.items(), key=lambda kv: -abs(kv[1][2] - kv[1][1])):
            print(f"  {t.name:<12} tensors={cnt:<4} exp={exp:>12} actual={act:>12} delta={act-exp:>+12}")
        self.tensors = []


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\LM studio\prism-ml\Ternary-Bonsai-27B-gguf\Ternary-Bonsai-27B-Q2_0.gguf"
    r = DiagReader(path, "r")
    print("kv keys:")
    for k in list(r.fields.keys())[:60]:
        print("  ", k)


if __name__ == "__main__":
    main()

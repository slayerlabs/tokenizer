"""Pomiar tokenizerów na held-oucie.

Użycie: python evaluate.py [plik_heldoutu.jsonl] [plik_wyniku.json]

Definicje (te same dla każdego tokenizera, liczone na tym samym tekście):
- słowo: ciąg znaków bez białych znaków, czyli wynik text.split();
  "kot," to jedno słowo razem z przecinkiem,
- fertility: tokeny / słowa (mniej = lepiej),
- znaki/token i bajty/token: ile tekstu niesie średnio jeden token (więcej = lepiej),
- round-trip: czy decode(encode(tekst)) == tekst dla każdego dokumentu.

Nasz tokenizer mierzymy w trzech rozmiarach (8k, 16k, 32k). Mniejsze to
prefiksy reguł dużego, więc nie wymagają osobnego treningu.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

from bpe import Tokenizer

HERE = Path(__file__).resolve().parent


def load_docs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def tokenizers():
    ours = Tokenizer.load(HERE / "artifacts" / "pl_bpe_32k.json")
    out = {}
    for size in (8000, 16000):
        t = ours.truncated(size - 1)  # minus jeden na token specjalny
        out[f"nasz BPE {size // 1000}k"] = (t.encode_ordinary, t.decode)
    out["nasz BPE 32k"] = (ours.encode_ordinary, ours.decode)
    for name in ("cl100k_base", "o200k_base"):
        e = tiktoken.get_encoding(name)
        out[name] = (e.encode_ordinary, e.decode)
    return out


def measure(docs, encode, decode):
    """Zwraca sumy dla całości i per źródło."""
    total = defaultdict(int)
    per_source = defaultdict(lambda: defaultdict(int))
    for d in docs:
        text = d["text"]
        ids = encode(text)
        row = {
            "tokens": len(ids),
            "words": len(text.split()),
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "roundtrip_fail": int(decode(ids) != text),
            "docs": 1,
        }
        for k, v in row.items():
            total[k] += v
            per_source[d["source"]][k] += v
    return total, per_source


def ratios(s):
    return {
        "fertility": s["tokens"] / s["words"],
        "chars_per_token": s["chars"] / s["tokens"],
        "bytes_per_token": s["bytes"] / s["tokens"],
        "tokens": s["tokens"],
        "words": s["words"],
        "roundtrip_fail": s["roundtrip_fail"],
        "docs": s["docs"],
    }


def main():
    heldout = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "heldout.jsonl"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "results" / "heldout_random.json"
    out.parent.mkdir(exist_ok=True)
    docs = load_docs(heldout)
    results = {}
    print(f"{'tokenizer':16s} {'fertility':>9s} {'znaki/tok':>9s} {'bajty/tok':>9s} "
          f"{'tokeny':>10s} {'round-trip':>10s}")
    for name, (enc, dec) in tokenizers().items():
        total, per_source = measure(docs, enc, dec)
        r = ratios(total)
        results[name] = {"total": r, "per_source": {k: ratios(v) for k, v in per_source.items()}}
        rt = "ok" if r["roundtrip_fail"] == 0 else f"{r['roundtrip_fail']} błędów"
        print(f"{name:16s} {r['fertility']:9.3f} {r['chars_per_token']:9.2f} "
              f"{r['bytes_per_token']:9.2f} {r['tokens']:10,} {rt:>10s}", flush=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"zapisano {out}")


if __name__ == "__main__":
    main()

"""Ewaluacja tokenizerów na held-oucie z hplt-v3-pl-cleaned.

Metryki (i po co):
  chars/token      kompresja — ile znaków mieści się w jednym tokenie, ↑ lepiej
  fertility        tokeny/słowo — standard w ewaluacji multilingual, ↓ lepiej
  bytes/token      to samo co wyżej, ale odporne na wieloznakowe kodowanie
  Rényi efficiency równomierność rozkładu tokenów (α=2.5, Zouhar i in. 2023), ↑ lepiej
  round-trip       decode(encode(x)) == x — twardy warunek poprawności

Uwaga metodyczna: fertility i Rényi są DYSTRYBUCYJNE. Mierzą kompresję i rozkład,
nie „rozumienie” morfologii — dobra fertility nie gwarantuje sensownych granic
morfemów (patrz sekcja „Fleksja pod mikroskopem” w README).

    uv run --with regex --with tiktoken python evaluate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from collections import Counter

import regex as re

from tokenizer import BPETokenizer

WORD_RE = re.compile(r"\p{L}+")


def disp(b: bytes) -> str:
    """Token do pokazania człowiekowi: bajty -> tekst, spacja -> widoczne ␣.

    Token, który jest urwanym kawałkiem sekwencji UTF-8 (np. pierwszy bajt „ę”),
    da U+FFFD — i dobrze, bo tak właśnie wygląda rozjechanie się na bajtach.
    """
    return b.decode("utf-8", errors="replace").replace(" ", "\u2423")

# zdanie ze slajdu 16 — bezpośrednie porównanie z warsztatem
DEMO_PL = "Niezaprzeczalnie skomplikowana morfosyntaktyczna struktura języka polskiego."
DEMO_EN = "The undeniably complicated morphosyntactic structure of the English language."


def load_heldout(path: str, limit_mb: float) -> list[str]:
    budget = int(limit_mb * 1024 * 1024)
    docs, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d["text"]
            docs.append(t)
            total += len(t.encode("utf-8"))
            if total >= budget:
                break
    return docs


def renyi_efficiency(freqs: Counter, vocab_size: int, alpha: float = 2.5) -> float:
    """H_alpha(p) / ln(V). Normalizujemy przez PEŁNY rozmiar słownika —
    tokeny nigdy nieużyte są karane (bo płacimy za nie w macierzy embeddingów)."""
    n = sum(freqs.values())
    if n == 0 or vocab_size <= 1:
        return 0.0
    s = sum((c / n) ** alpha for c in freqs.values())
    h = math.log(s) / (1.0 - alpha)
    return h / math.log(vocab_size)


def eval_encoder(name: str, encode, decode, vocab_size: int,
                 docs: list[str]) -> dict:
    n_tok = n_chars = n_bytes = n_words = 0
    freqs: Counter = Counter()
    rt_ok = True
    t0 = time.time()
    for d in docs:
        ids = encode(d)
        freqs.update(ids)
        n_tok += len(ids)
        n_chars += len(d)
        n_bytes += len(d.encode("utf-8"))
        n_words += len(WORD_RE.findall(d))
        if rt_ok and decode(ids) != d:
            rt_ok = False
    el = time.time() - t0
    return {
        "name": name,
        "vocab_size": vocab_size,
        "n_docs": len(docs),
        "n_tokens": n_tok,
        "chars_per_token": round(n_chars / n_tok, 3),
        "bytes_per_token": round(n_bytes / n_tok, 3),
        "fertility": round(n_tok / n_words, 4),
        "renyi_efficiency": round(renyi_efficiency(freqs, vocab_size), 4),
        "vocab_used": len(freqs),
        "vocab_used_pct": round(100 * len(freqs) / vocab_size, 1),
        "round_trip_lossless": rt_ok,
        "encode_mb_per_s": round(n_bytes / 1e6 / el, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default="../data/hplt-pl/corpus_heldout.jsonl")
    ap.add_argument("--limit-mb", type=float, default=10.0)
    ap.add_argument("--tokenizers", default="tokenizers/*.json")
    ap.add_argument("--out", default="results/metrics.json")
    ap.add_argument("--viz-out", default="results/viz_data.json")
    args = ap.parse_args()

    docs = load_heldout(args.heldout, args.limit_mb)
    mb = sum(len(d.encode("utf-8")) for d in docs) / 1e6
    print(f"held-out: {len(docs)} dok., {mb:.1f} MB "
          f"(row-group nieużywany w treningu)\n")

    results = []
    ours: dict[str, BPETokenizer] = {}
    for path in sorted(glob.glob(args.tokenizers)):
        tok = BPETokenizer.load(path)
        name = tok.meta.get("name") or os.path.basename(path)[:-5]
        ours[name] = tok
        print(f"-> {name} ...", flush=True)
        results.append(eval_encoder(name, tok.encode_ordinary, tok.decode,
                                    tok.vocab_size, docs))

    # --- odniesienia zewnętrzne
    ext = {}
    try:
        import tiktoken
        for label, enc_name in [("cl100k_base (GPT-4)", "cl100k_base"),
                                ("o200k_base (GPT-4o)", "o200k_base"),
                                ("gpt2", "gpt2")]:
            try:
                enc = tiktoken.get_encoding(enc_name)
            except Exception as e:                       # brak sieci / cache
                print(f"   (pomijam {enc_name}: {e})")
                continue
            print(f"-> {label} ...", flush=True)
            ext[label] = enc
            results.append(eval_encoder(label, enc.encode_ordinary, enc.decode,
                                        enc.n_vocab, docs))
    except ImportError:
        print("   (tiktoken niedostępny — pomijam odniesienia zewnętrzne)")

    results.sort(key=lambda r: r["fertility"])

    # --- tabela
    hdr = (f"\n{'tokenizer':<24}{'vocab':>8}{'zn/tok':>9}{'fert.':>8}"
           f"{'Rényi':>8}{'użyty':>8}{'RT':>5}")
    print(hdr)
    print("-" * len(hdr.strip()))
    for r in results:
        print(f"{r['name']:<24}{r['vocab_size']:>8}{r['chars_per_token']:>9.2f}"
              f"{r['fertility']:>8.3f}{r['renyi_efficiency']:>8.3f}"
              f"{r['vocab_used_pct']:>7.1f}%{'  ok' if r['round_trip_lossless'] else ' BAD':>5}")

    # --- zdanie ze slajdu 16
    print(f"\nzdanie ze slajdu 16:\n  PL: {DEMO_PL}\n  EN: {DEMO_EN}\n")
    demo = {}
    all_encoders = {n: (t.encode_ordinary,
                        lambda i, t=t: [disp(t.vocab[x]) for x in i])
                    for n, t in ours.items()}
    for label, enc in ext.items():
        all_encoders[label] = (
            enc.encode_ordinary,
            lambda i, e=enc: [disp(e.decode_single_token_bytes(x)) for x in i],
        )
    for label, (enc_fn, tostr) in all_encoders.items():
        pl, en = enc_fn(DEMO_PL), enc_fn(DEMO_EN)
        demo[label] = {
            "pl_n": len(pl), "en_n": len(en),
            "ratio": round(len(pl) / len(en), 2),
            "pl_tokens": tostr(pl),
        }
        print(f"  {label:<24} PL {len(pl):>3}  EN {len(en):>3}  "
              f"PL/EN {len(pl)/len(en):.2f}x")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"heldout_mb": round(mb, 1), "heldout_docs": len(docs),
                   "results": results, "demo_sentence": demo,
                   "demo_pl": DEMO_PL, "demo_en": DEMO_EN},
                  f, ensure_ascii=False, indent=1)
    print(f"\nzapisano {args.out}")

    # --- dane dla animacji (scena 5)
    main_name = "hplt-pl-32k-pl"
    if main_name in demo:
        by_name = {r["name"]: r for r in results}
        viz = {
            "demo_pl": DEMO_PL,
            "ours": {"name": main_name,
                     "tokens": demo[main_name]["pl_tokens"],
                     "n": demo[main_name]["pl_n"],
                     **{k: by_name[main_name][k] for k in
                        ("chars_per_token", "fertility", "vocab_size")}},
        }
        ref = next((k for k in demo if k.startswith("cl100k")), None)
        if ref:
            viz["ref"] = {"name": "cl100k_base (GPT-4)",
                          "tokens": demo[ref]["pl_tokens"],
                          "n": demo[ref]["pl_n"],
                          **{k: by_name[ref][k] for k in
                             ("chars_per_token", "fertility", "vocab_size")}}
        with open(args.viz_out, "w", encoding="utf-8") as f:
            json.dump(viz, f, ensure_ascii=False, indent=1)
        print(f"zapisano {args.viz_out}")


if __name__ == "__main__":
    main()

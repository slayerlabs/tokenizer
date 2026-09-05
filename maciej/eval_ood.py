"""Ewaluacja poza domeną treningową (out-of-domain).

Tokenizer trenowaliśmy na web-crawlu (HPLT) i held-out też jest z web-crawla —
to ta sama dystrybucja. Przewaga nad cl100k mogłaby być artefaktem dopasowania
do domeny. Ten skrypt sprawdza to na dwóch korpusach, których tokenizer nie
widział i które wyglądają zupełnie inaczej:

  literatura  — Wolne Lektury, XIX-wieczna proza (Prus, Sienkiewicz, Żeromski)
  encyklopedia — polska Wikipedia (współczesny styl encyklopedyczny)

    uv run --with regex --with tiktoken python eval_ood.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

from evaluate import eval_encoder
from tokenizer import BPETokenizer

UA = "slayerlabs-course-tokenizer-eval/1.0 (educational; BPE fertility study)"

LEKTURY = [
    "lalka-tom-pierwszy",
    "ogniem-i-mieczem-tom-pierwszy",
    "quo-vadis",
    "chlopi-czesc-pierwsza-jesien",
]
WIKI = [
    "Język polski", "Polska", "Warszawa", "Kraków", "Mikołaj Kopernik",
    "Tatry", "Chemia", "Fizyka", "Biologia", "Matematyka",
    "II wojna światowa", "Fotosynteza", "Prawo rzymskie", "Konstytucja",
    "Muzyka poważna", "Gospodarka Polski", "Uniwersytet Jagielloński",
    "Historia Polski", "Literatura polska", "Adam Mickiewicz",
    "Morze Bałtyckie", "Unia Europejska", "Komputer", "Medycyna",
    "Ewolucja", "Układ Słoneczny", "Sztuczna inteligencja", "Bitwa pod Grunwaldem",
]


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_lektury(cache: str) -> list[str]:
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    docs = []
    for slug in LEKTURY:
        try:
            txt = _get(f"https://wolnelektury.pl/media/book/txt/{slug}.txt")
        except Exception as e:          # slug zmieniony / książka zdjęta
            print(f"  {slug}: POMINIĘTE ({e})")
            continue
        print(f"  {slug}: {len(txt)/1e3:.0f}k znaków")
        docs.append(txt)
    json.dump(docs, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return docs


def fetch_wiki(cache: str) -> list[str]:
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    docs = []
    # po jednym tytule na zapytanie — exlimit potrafi uciąć resztę batcha
    for title in WIKI:
        url = ("https://pl.wikipedia.org/w/api.php?action=query&format=json"
               "&prop=extracts&explaintext=1&redirects=1&titles="
               + urllib.parse.quote(title))
        try:
            d = json.loads(_get(url))
        except Exception as e:
            print(f"  {title}: POMINIĘTE ({e})")
            continue
        for p in d["query"]["pages"].values():
            ex = p.get("extract", "")
            if len(ex) > 500:
                docs.append(ex)
                print(f"  {p['title']}: {len(ex)/1e3:.0f}k znaków")
    json.dump(docs, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="../data/ood")
    ap.add_argument("--out", default="results/metrics_ood.json")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    print("pobieram korpusy out-of-domain...")
    domains = {
        "literatura (Wolne Lektury)":
            fetch_lektury(os.path.join(args.cache_dir, "lektury.json")),
        "encyklopedia (Wikipedia PL)":
            fetch_wiki(os.path.join(args.cache_dir, "wiki.json")),
    }

    encoders = {}
    for name in ["hplt-pl-32k-pl", "hplt-pl-64k-pl"]:
        p = f"tokenizers/{name}.json"
        if os.path.exists(p):
            t = BPETokenizer.load(p)
            encoders[name] = (t.encode_ordinary, t.decode, t.vocab_size)
    import tiktoken
    for label, enc_name in [("cl100k_base (GPT-4)", "cl100k_base"),
                            ("o200k_base (GPT-4o)", "o200k_base")]:
        e = tiktoken.get_encoding(enc_name)
        encoders[label] = (e.encode_ordinary, e.decode, e.n_vocab)

    out = {}
    for dom, docs in domains.items():
        mb = sum(len(d.encode("utf-8")) for d in docs) / 1e6
        print(f"\n=== {dom}: {len(docs)} dok., {mb:.1f} MB ===")
        print(f"{'tokenizer':<24}{'zn/tok':>9}{'fert.':>8}{'RT':>5}")
        rows = []
        for name, (enc, dec, vs) in encoders.items():
            r = eval_encoder(name, enc, dec, vs, docs)
            rows.append(r)
            print(f"{name:<24}{r['chars_per_token']:>9.2f}{r['fertility']:>8.3f}"
                  f"{'  ok' if r['round_trip_lossless'] else ' BAD':>5}")
        out[dom] = {"mb": round(mb, 1), "n_docs": len(docs), "results": rows}

    json.dump(out, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nzapisano {args.out}")


if __name__ == "__main__":
    main()

"""Wypluj tabele markdown z results/metrics.json (żeby README nie rozjechało się
z artefaktami). Wynik wklejany do README.md."""

import json
import sys

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/metrics.json",
                   encoding="utf-8"))
res = {r["name"]: r for r in d["results"]}
demo = d["demo_sentence"]

OURS = ["hplt-pl-8k-pl", "hplt-pl-16k-pl", "hplt-pl-32k-pl",
        "hplt-pl-64k-pl", "hplt-pl-32k-gpt2"]
REF = [k for k in res if k not in OURS]


def row(name, label=None):
    r = res[name]
    dm = demo.get(name, {})
    return (f"| {label or name} | {r['vocab_size']:,} | {r['chars_per_token']:.2f} | "
            f"{r['fertility']:.3f} | {r['renyi_efficiency']:.3f} | "
            f"{r['vocab_used_pct']:.1f}% | {dm.get('pl_n','-')} | "
            f"{'tak' if r['round_trip_lossless'] else 'NIE'} |").replace(",", " ")


print("### Porównanie (held-out %s MB, %s dok.)\n" % (d["heldout_mb"], d["heldout_docs"]))
print("| tokenizer | słownik | zn/tok | fertility | Rényi | użyty słownik | zdanie ze slajdu 16 | round-trip |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for n in ["hplt-pl-32k-pl"]:
    print(row(n, "**hplt-pl-32k-pl** (nasz)"))
for n in sorted(REF, key=lambda k: res[k]["fertility"]):
    print(row(n))

print("\n### Krzywa vocab -> fertility (pretokenizer `pl`)\n")
print("| słownik | zn/tok | fertility | Rényi | użyty słownik | rozmiar pliku |")
print("|---:|---:|---:|---:|---:|---:|")
import os
for n in ["hplt-pl-8k-pl", "hplt-pl-16k-pl", "hplt-pl-32k-pl", "hplt-pl-64k-pl"]:
    if n not in res:
        continue
    r = res[n]
    p = f"tokenizers/{n}.json"
    sz = f"{os.path.getsize(p)/1e6:.1f} MB" if os.path.exists(p) else "-"
    print(f"| {r['vocab_size']:,} | {r['chars_per_token']:.2f} | {r['fertility']:.3f} | "
          f"{r['renyi_efficiency']:.3f} | {r['vocab_used_pct']:.1f}% | {sz} |"
          .replace(",", " "))

print("\n### Pretokenizer: `pl` vs `gpt2` (oba 32k, ten sam korpus)\n")
print("| pretokenizer | zn/tok | fertility | Rényi | użyty słownik |")
print("|---|---:|---:|---:|---:|")
for n, lab in [("hplt-pl-32k-pl", "`pl` (bez angielskich skrótów, cyfry ≤3)"),
               ("hplt-pl-32k-gpt2", "`gpt2` (klasyczny)")]:
    if n not in res:
        continue
    r = res[n]
    print(f"| {lab} | {r['chars_per_token']:.2f} | {r['fertility']:.3f} | "
          f"{r['renyi_efficiency']:.3f} | {r['vocab_used_pct']:.1f}% |")

print("\n### Zdanie ze slajdu 16 (PL vs EN)\n")
print("| tokenizer | PL | EN | PL/EN |")
print("|---|---:|---:|---:|")
for n in ["hplt-pl-32k-pl"] + sorted(REF, key=lambda k: demo.get(k, {}).get("pl_n", 999)):
    if n in demo:
        m = demo[n]
        print(f"| {n} | {m['pl_n']} | {m['en_n']} | {m['ratio']}× |")

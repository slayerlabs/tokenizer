#!/usr/bin/env python3
"""Research question 3.1: vocabulary stability between two disjoint random halves of the corpus.

If a token is a property of the language, a tokenizer trained on the other half should
learn it too. If it is a sampling accident, it will not. We measure:
  * Jaccard of the merged-token sets (as byte strings) at each vocab size,
  * coverage per rank window: share of A's tokens in ranks [r, r+w) that exist anywhere
    in B (same size), plotted against the pair frequency at merge time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bpe.evaluation import merge_counts, size_dirs  # noqa: E402
from bpe.tokenizer import Tokenizer  # noqa: E402

A, B = Path("artifacts/half_a"), Path("artifacts/half_b")
RES, FIG = Path("results"), Path("report/figures")
WINDOW = 1024


def merged_tokens(tok: Tokenizer) -> list[bytes]:
    return [tok.vocab[i] for i in range(256, tok.vocab_size)]


def main() -> None:
    RES.mkdir(exist_ok=True)
    jaccard = {}
    for v, pa in size_dirs(A):
        ta, tb = set(merged_tokens(Tokenizer.load(pa))), set(merged_tokens(Tokenizer.load(B / pa.name)))
        jaccard[v] = len(ta & tb) / len(ta | tb)
        print(f"v={v:>6} jaccard={jaccard[v]:.3f}")

    # Tolerant metric (added after seeing that strict Jaccard is dominated by tokens that
    # sit a few ranks across the cut-off): share of A@V tokens that exist in B@2V.
    tolerant = {}
    for v, pa in size_dirs(A):
        if (B / f"v{2 * v}").exists():
            ta = merged_tokens(Tokenizer.load(pa))
            tb2 = set(merged_tokens(Tokenizer.load(B / f"v{2 * v}")))
            tolerant[v] = sum(t in tb2 for t in ta) / len(ta)
            print(f"v={v:>6} share of A@V in B@2V={tolerant[v]:.3f}")

    vmax, pmax = size_dirs(A)[-1]
    a_tokens = merged_tokens(Tokenizer.load(pmax))
    b_set = set(merged_tokens(Tokenizer.load(B / pmax.name)))
    counts = merge_counts(pmax)
    windows = []
    for start in range(0, len(a_tokens), WINDOW):
        chunk = a_tokens[start:start + WINDOW]
        cov = sum(t in b_set for t in chunk) / len(chunk)
        cnt = sorted(counts[start:start + WINDOW])[len(chunk) // 2]
        windows.append({"rank_start": 256 + start, "coverage": cov, "median_merge_count": cnt})

    # Main tokenizer agreement with each half: an upper bound on how stable it can look.
    main_set = set(merged_tokens(Tokenizer.load(Path("artifacts/main") / pmax.name)))
    agree_main = len(main_set & set(a_tokens)) / len(main_set | set(a_tokens))

    out = {"jaccard": jaccard, "tolerant_a_in_b2v": tolerant, "windows": windows, "jaccard_main_vs_half_a_at_max": agree_main}
    (RES / "stability.json").write_text(json.dumps(out, indent=1))
    print(f"main vs half-a jaccard at {vmax}: {agree_main:.3f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    sizes = sorted(jaccard)
    a1.plot(sizes, [jaccard[v] for v in sizes], "o-", lw=2, label="Jaccard(A@V, B@V) — metryka z prerejestracji")
    ts = sorted(tolerant)
    a1.plot(ts, [tolerant[v] for v in ts], "s--", lw=2, label="% A@V obecnych w B@2V (tolerancyjna)")
    a1.legend(fontsize=7, loc="center left")
    a1.axhline(0.95, ls=":", color="green")
    a1.text(sizes[0], 0.953, "hipoteza 1: ≥0,95 do 4k", color="green", fontsize=8)
    a1.axhline(0.70, ls=":", color="red")
    a1.text(sizes[-1], 0.705, "hipoteza 2: <0,70 przy 32k", color="red", fontsize=8, ha="right")
    a1.set_ylim(0.68, 1.01)
    a1.set_xscale("log", base=2)
    a1.set_xticks(sizes, [f"{v // 1024}k" for v in sizes])
    a1.set_ylabel("zgodność słowników A i B")
    a1.set_title("Pokrycie słowników dwóch połówek korpusu")
    a1.grid(alpha=0.3)

    xs = [w["rank_start"] for w in windows]
    a2.plot(xs, [100 * w["coverage"] for w in windows], "-", color="#4C72B0")
    a2.set_xlabel("ranga merge'a w tokenizerze A")
    a2.set_ylabel("% tokenów z okna obecnych w B", color="#4C72B0")
    a2.set_title(f"Stabilność per okno {WINDOW} rang (vs B@64k)")
    a2.axvspan(vmax // 2, vmax, color="gray", alpha=0.15)
    a2.text(vmax // 2 + 500, 52, "efekt granicy:\nB nie ma\nzapasu rang", fontsize=7, color="gray")
    a2.grid(alpha=0.3)
    a3 = a2.twinx()
    a3.plot(xs, [w["median_merge_count"] for w in windows], "--", color="#DD8452")
    a3.set_yscale("log")
    a3.set_ylabel("mediana częstości pary przy merge'u", color="#DD8452")
    fig.tight_layout()
    fig.savefig(FIG / "stability.png", dpi=130)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Research question 3.4: how numbers are split under three digit pre-tokenization variants.

All three tokenizers have 32k vocab and identical pre-tokenization except for digits.
For 1000..9999 we record the split pattern (token lengths in digits, e.g. (2, 2)).
Digits are pre-tokenized separately from the preceding space in all variants, so
encoding the bare number equals encoding it in running text.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import regex  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from bpe.evaluation import encode_corpus, load_jsonl  # noqa: E402
from bpe.text import count_words  # noqa: E402
from bpe.tokenizer import Tokenizer  # noqa: E402

RES, FIG = Path("results"), Path("report/figures")
VARIANTS = {"group3": Path("artifacts/main/v32768"), "single": Path("artifacts/digits_single/v32768"),
            "free": Path("artifacts/digits_free/v32768")}


def pattern(tok: Tokenizer, s: str) -> tuple[int, ...]:
    return tuple(len(tok.vocab[i]) for i in tok.encode(s))


def main() -> None:
    RES.mkdir(exist_ok=True)
    docs = load_jsonl()
    texts = [d["text"] for d in docs]
    words = sum(count_words(t) for t in texts)
    numpat = regex.compile(r"\p{N}+")
    held_numbers = [m for t in texts for m in numpat.findall(t)]

    out, grids = {}, {}
    for name, path in VARIANTS.items():
        tok = Tokenizer.load(path)
        pats = {n: pattern(tok, str(n)) for n in range(1000, 10000)}
        dist = Counter(pats.values())
        top, top_n = dist.most_common(1)[0]
        entropy = -sum(c / 9000 * math.log2(c / 9000) for c in dist.values())
        adjacent_change = sum(pats[n] != pats[n + 1] for n in range(1000, 9999)) / 8999
        years = [pats[y] == (4,) for y in range(1950, 2026)]
        held_tokens = sum(len(tok.encode(n)) for n in held_numbers)
        total = sum(encode_corpus(path, texts)[0])
        out[name] = {
            "top_pattern": "+".join(map(str, top)), "irregular_pct": 100 * (1 - top_n / 9000),
            "pattern_entropy_bits": entropy, "n_patterns": len(dist),
            "pattern_dist": {"+".join(map(str, k)): v for k, v in dist.most_common(8)},
            "adjacent_pattern_change_pct": 100 * adjacent_change,
            "years_1950_2025_single_token_pct": 100 * sum(years) / len(years),
            "single_token_4digit_pct": 100 * dist.get((4,), 0) / 9000,
            "heldout_tokens_per_number": held_tokens / len(held_numbers),
            "heldout_fertility": total / words,
            "digit_tokens_in_vocab": sum(1 for i in range(256, tok.vocab_size) if tok.vocab[i].isdigit()),
        }
        grids[name] = (pats, dist)
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)

    base = out["group3"]["heldout_fertility"]
    for name in out:
        out[name]["fertility_change_vs_group3_pct"] = 100 * (out[name]["heldout_fertility"] - base) / base
    (RES / "numbers.json").write_text(json.dumps(out, indent=1))

    # Pattern map for the 'free' variant: rows = first two digits, cols = last two.
    pats, dist = grids["free"]
    labels = [p for p, _ in dist.most_common(7)]
    idx = {p: i for i, p in enumerate(labels)}
    grid = np.full((90, 100), len(labels))
    for n, p in pats.items():
        grid[n // 100 - 10, n % 100] = idx.get(p, len(labels))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#cccccc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    # Fixed vmin/vmax so a missing category cannot shift the colour mapping.
    a1.imshow(grid, cmap=ListedColormap(colors[:len(labels)] + [colors[-1]]), aspect="auto",
              vmin=-0.5, vmax=len(labels) + 0.5, extent=(-0.5, 99.5, 99.5, 9.5), interpolation="nearest")
    a1.set_xlabel("dwie ostatnie cyfry")
    a1.set_ylabel("dwie pierwsze cyfry")
    a1.set_title("Wariant free: wzorzec podziału liczb 1000–9999")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(labels))]
    a1.legend(handles, ["+".join(map(str, p)) for p in labels], fontsize=7, loc="lower right", title="cyfry/token")

    names = list(VARIANTS)
    x = np.arange(len(names))
    a2.bar(x - 0.2, [out[n]["irregular_pct"] for n in names], 0.4, label="% nieregularnych (≠ dominujący wzorzec)")
    a2.bar(x + 0.2, [out[n]["adjacent_pattern_change_pct"] for n in names], 0.4,
           label="% sąsiednich liczb (n, n+1) o innym wzorcu")
    a2.axhline(40, ls=":", color="red")
    a2.set_xticks(x, names)
    a2.set_ylabel("%")
    a2.set_title("Spójność podziału per wariant")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "numbers.png", dpi=130)


if __name__ == "__main__":
    main()

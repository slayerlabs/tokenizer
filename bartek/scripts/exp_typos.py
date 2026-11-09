#!/usr/bin/env python3
"""Research question 3.2: how many extra tokens does a single typo cost, per vocab size?

Words (>= 4 letters) are sampled from held-out text in proportion to their frequency
and encoded the way they appear in running text (with a leading space). Each word
gets exactly one random typo: adjacent swap, deletion, insertion, or substitution
with a neighbouring QWERTY key. Delta = tokens(typo) - tokens(original).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402
import regex  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bpe.evaluation import load_jsonl, size_dirs  # noqa: E402
from bpe.tokenizer import Tokenizer  # noqa: E402

N_WORDS = 20_000
RES, FIG = Path("results"), Path("report/figures")
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
LETTERS = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż"


def neighbours(c: str) -> str:
    lc = c.lower()
    for r, row in enumerate(ROWS):
        if lc in row:
            i = row.index(lc)
            near = row[max(i - 1, 0):i] + row[i + 1:i + 2]
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(ROWS):
                    near += ROWS[rr][max(i - 1, 0):i + 1]
            return near
    return "aeiou"


def make_typo(word: str, rng: random.Random) -> tuple[str, str]:
    op = rng.choice(["swap", "delete", "insert", "substitute"])
    i = rng.randrange(len(word))
    if op == "swap":
        i = min(i, len(word) - 2)
        return op, word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return op, word[:i] + word[i + 1:]
    if op == "insert":
        return op, word[:i] + rng.choice(LETTERS) + word[i:]
    return op, word[:i] + rng.choice(neighbours(word[i])) + word[i + 1:]


def main() -> None:
    RES.mkdir(exist_ok=True)
    rng = random.Random(0)
    pat = regex.compile(r"\b\p{L}{4,}\b")
    occurrences = [w for d in load_jsonl() for w in pat.findall(d["text"])]
    words = rng.sample(occurrences, N_WORDS)
    pairs = []
    for w in words:
        op, t = make_typo(w, rng)
        if t != w:
            pairs.append((w, t, op))

    results = {}
    for v, path in size_dirs("artifacts/main"):
        tok = Tokenizer.load(path)
        rows = []
        for w, t, op in pairs:
            a, b = tok.encode(" " + w), tok.encode(" " + t)
            rows.append((len(a), len(b), op))
        deltas = [b - a for a, b, _ in rows]
        single = [b - a for a, b, _ in rows if a == 1]
        results[v] = {
            "mean_delta": mean(deltas),
            "mean_rel": mean(b / a for a, b, _ in rows),
            "mean_orig_tokens": mean(a for a, _, _ in rows),
            "share_single_token_words": len(single) / len(rows),
            "mean_delta_single_token_words": mean(single) if single else None,
            "share_delta_ge2": sum(d >= 2 for d in deltas) / len(deltas),
            "by_op": {op: mean(b - a for a, b, o in rows if o == op) for op in ("swap", "delete", "insert", "substitute")},
        }
        r = results[v]
        print(f"v={v:>6} Δ={r['mean_delta']:.3f} rel={r['mean_rel']:.2f} orig={r['mean_orig_tokens']:.2f} "
              f"single={r['share_single_token_words']:.2f}", flush=True)
    (RES / "typos.json").write_text(json.dumps({"n": len(pairs), "results": results}, indent=1))

    sizes = sorted(results)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(sizes, [results[v]["mean_delta"] for v in sizes], "o-", lw=2, label="Δ tokenów (średnio)")
    a1.plot(sizes, [results[v]["mean_orig_tokens"] for v in sizes], "s--", label="tokeny oryginału")
    a1.axhline(1.5, ls=":", color="red")
    a1.text(sizes[0], 1.52, "hipoteza 1: >1,5 przy 32k", color="red", fontsize=8)
    a1.axhline(1.0, ls=":", color="green")
    a1.text(sizes[0], 1.02, "hipoteza 2: <1,0 przy 4k", color="green", fontsize=8)
    a1.set_xscale("log", base=2)
    a1.set_xticks(sizes, [f"{v // 1024}k" for v in sizes])
    a1.set_title("Koszt jednej literówki")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.plot(sizes, [results[v]["mean_rel"] for v in sizes], "o-", color="#C44E52")
    a2.set_xscale("log", base=2)
    a2.set_xticks(sizes, [f"{v // 1024}k" for v in sizes])
    a2.set_ylabel("tokeny(literówka) / tokeny(oryginał)")
    a2.set_title("Względny wzrost liczby tokenów")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "typos.png", dpi=130)


if __name__ == "__main__":
    main()

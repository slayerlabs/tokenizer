#!/usr/bin/env python3
"""Part 2: vocab_size experiment, measured on the held-out split only.

For each trained size: fertility (tokens/word), compression (bytes/token), per-source
breakdown, vocab usage on held-out, token-length distribution, and the share of
"rare" tokens (pair frequency at merge time below a threshold). tiktoken cl100k_base
and o200k_base are evaluated on the same text as an external reference.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bpe.evaluation import SOURCES, encode_corpus, load_jsonl, merge_counts, size_dirs  # noqa: E402
from bpe.text import count_words  # noqa: E402
from bpe.tokenizer import Tokenizer  # noqa: E402

ROOT = Path("artifacts/main")
RES = Path("results")
FIG = Path("report/figures")
RARE_THRESHOLD = 200  # a token seen < 200 times in 172 MB of training text is "rare"


def aggregate(docs, counts):
    by = defaultdict(lambda: Counter())
    for d, n in zip(docs, counts):
        for key in ("all", d["source"]):
            by[key]["tokens"] += n
            by[key]["words"] += count_words(d["text"])
            by[key]["bytes"] += len(d["text"].encode())
            by[key]["chars"] += len(d["text"])
    out = {}
    for key, c in by.items():
        out[key] = {"fertility": c["tokens"] / c["words"], "bytes_per_token": c["bytes"] / c["tokens"],
                    "chars_per_token": c["chars"] / c["tokens"], "tokens": c["tokens"], "words": c["words"]}
    out["macro"] = {"fertility": sum(out[s]["fertility"] for s in SOURCES) / len(SOURCES)}
    return out


def tiktoken_reference(docs):
    import tiktoken
    res = {}
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        counts = [len(enc.encode(d["text"], disallowed_special=())) for d in docs]
        res[name] = aggregate(docs, counts)
    return res


def main() -> None:
    RES.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    docs = load_jsonl()
    texts = [d["text"] for d in docs]
    results = {}
    for v, path in size_dirs(ROOT):
        counts, usage = encode_corpus(path, texts)
        tok = Tokenizer.load(path)
        mc = merge_counts(path)
        lengths = [len(tok.vocab[i]) for i in range(256, tok.vocab_size)]
        invalid = 0
        for i in range(256, tok.vocab_size):
            try:
                tok.vocab[i].decode("utf-8")
            except UnicodeDecodeError:
                invalid += 1
        used = sum(1 for i in range(tok.vocab_size) if usage[i] > 0)
        results[v] = {
            "metrics": aggregate(docs, counts),
            "used_on_heldout": used,
            "unused_merged_tokens": sum(1 for i in range(256, tok.vocab_size) if usage[i] == 0),
            "rare_merges": sum(1 for c in mc if c < RARE_THRESHOLD),
            "min_merge_count": min(mc),
            "invalid_utf8_tokens": invalid,
            "token_len_bytes": Counter(lengths),
        }
        m = results[v]["metrics"]["all"]
        print(f"v={v:>6} fertility={m['fertility']:.3f} bytes/tok={m['bytes_per_token']:.2f} "
              f"unused={results[v]['unused_merged_tokens']} rare={results[v]['rare_merges']} "
              f"min_count={min(mc)}", flush=True)

    ref = tiktoken_reference(docs)
    (RES / "vocab_metrics.json").write_text(json.dumps({"ours": results, "tiktoken": ref}, indent=1))
    plot(results, ref)


def plot(results, ref):
    sizes = sorted(results)
    # 1. fertility vs vocab size (overall + per source) with tiktoken references
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, [results[v]["metrics"]["all"]["fertility"] for v in sizes], "o-", lw=2.5, color="black",
            label="całość (held-out)")
    for s in SOURCES:
        ax.plot(sizes, [results[v]["metrics"][s]["fertility"] for v in sizes], ".--", alpha=0.8, label=s)
    for name, style in (("cl100k_base", ":"), ("o200k_base", "-.")):
        ax.axhline(ref[name]["all"]["fertility"], ls=style, color="gray")
        ax.text(sizes[-1], ref[name]["all"]["fertility"] + 0.02, f"{name} (całość)", color="gray", fontsize=8,
                ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{v // 1024}k" for v in sizes])
    ax.set_xlabel("vocab_size")
    ax.set_ylabel("fertility (tokeny / słowo)")
    ax.set_title("Fertility vs vocab_size — held-out 10,4 MB")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fertility_vs_vocab.png", dpi=130)

    # 2. marginal gain per doubling + rare / unused tokens
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    fert = [results[v]["metrics"]["all"]["fertility"] for v in sizes]
    gains = [100 * (fert[i - 1] - fert[i]) / fert[i - 1] for i in range(1, len(sizes))]
    a1.bar([f"{sizes[i - 1] // 1024}k→{sizes[i] // 1024}k" for i in range(1, len(sizes))], gains, color="#4C72B0")
    a1.set_ylabel("spadek fertility [%]")
    a1.set_title("Zysk z każdego podwojenia słownika")
    a1.tick_params(axis="x", rotation=30)
    for i, g in enumerate(gains):
        a1.text(i, g + 0.1, f"{g:.1f}%", ha="center", fontsize=8)
    a2.plot(sizes, [100 * results[v]["rare_merges"] / v for v in sizes], "o-",
            label=f"rzadkie (< {RARE_THRESHOLD} wystąpień w treningu)")
    a2.plot(sizes, [100 * results[v]["unused_merged_tokens"] / v for v in sizes], "s-",
            label="nieużyte ani razu na held-out")
    a2.set_xscale("log", base=2)
    a2.set_xticks(sizes, [f"{v // 1024}k" for v in sizes])
    a2.set_ylabel("% słownika")
    a2.set_title("Ile słownika jest „słabo wyuczone”")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "gain_and_rare.png", dpi=130)

    # 3. token length histograms
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in [s for s in sizes if s in (4096, 16384, 65536)]:
        h = results[v]["token_len_bytes"]
        total = sum(h.values())
        xs = list(range(1, 21))
        ax.plot(xs, [100 * h.get(x, 0) / total for x in xs], "o-", label=f"{v // 1024}k")
    ax.set_xlabel("długość tokenu [bajty UTF-8]")
    ax.set_ylabel("% tokenów (bez 256 bajtów bazowych)")
    ax.set_title("Rozkład długości tokenów w słowniku")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "token_lengths.png", dpi=130)


if __name__ == "__main__":
    main()

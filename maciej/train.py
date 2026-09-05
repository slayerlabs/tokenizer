"""Trening byte-level BPE na SlayerLab/hplt-v3-pl-cleaned.

Ten sam algorytm co `bpe.py`, ale liczony tak, żeby dowieźć 32k merge'y na
setkach MB tekstu zamiast na akapicie. Trzy różnice — wyłącznie
inżynieryjne, nie algorytmiczne:

  1. PRETOKENIZACJA (slajd 18). Korpus -> słownik {pretoken: częstość}. Zamiast
     300 mln symboli mamy ~2 mln unikalnych typów słów z wagami. Merge'e nie
     przechodzą przez granice słów — to świadomy koszt, tak robi GPT-2.
  2. INKREMENTALNE LICZNIKI. Naiwna wersja przelicza WSZYSTKIE pary po każdym
     merge'u (slajd 4.3). Tutaj po merge'u dotykamy tylko słów, które tę parę
     zawierały, i korygujemy liczniki różnicowo.
  3. KOPIEC LENIWY. `max()` po ~2 mln par x 32k merge'y to 6.4e10 operacji.
     Kopiec z leniwym usuwaniem: wrzucamy nową wartość, a nieaktualne wpisy
     odrzucamy przy zdejmowaniu.

Parytet z `bpe.py` jest testowany 1:1 (test_bpe.py), z tie_break="min_pair".

    uv run --with regex --with pyarrow python train.py --vocab-size 32000
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import time
from collections import Counter
from multiprocessing import Pool

import regex as re

from tokenizer import PATTERNS, BPETokenizer

_PAT = None


def _init(pattern: str) -> None:
    global _PAT
    _PAT = re.compile(pattern)


def _count_chunk(chunk: str) -> Counter:
    return Counter(_PAT.findall(chunk))


def _read_chunks(path: str, chunk_bytes: int):
    """Czytaj plik kawałkami wyrównanymi do końca linii."""
    with open(path, encoding="utf-8") as f:
        buf = []
        size = 0
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= chunk_bytes:
                yield "".join(buf)
                buf, size = [], 0
        if buf:
            yield "".join(buf)


def count_pretokens(path: str, pattern: str, workers: int,
                    chunk_mb: float = 8.0) -> Counter:
    t0 = time.time()
    total = Counter()
    chunks = _read_chunks(path, int(chunk_mb * 1024 * 1024))
    if workers > 1:
        with Pool(workers, initializer=_init, initargs=(pattern,)) as pool:
            for i, c in enumerate(pool.imap_unordered(_count_chunk, chunks, 1)):
                total.update(c)
                if (i + 1) % 10 == 0:
                    print(f"  ~{(i+1)*chunk_mb:.0f} MB, {len(total):,} typów "
                          f"({time.time()-t0:.0f}s)", flush=True)
    else:
        _init(pattern)
        for c in map(_count_chunk, chunks):
            total.update(c)
    print(f"pretokenizacja: {len(total):,} unikalnych typów, "
          f"{sum(total.values()):,} wystąpień ({time.time()-t0:.1f}s)")
    return total


def _pairs_of(sym: list[int]) -> dict[tuple[int, int], int]:
    d: dict[tuple[int, int], int] = {}
    for p in zip(sym, sym[1:]):
        d[p] = d.get(p, 0) + 1
    return d


def _apply(sym: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(sym)
    while i < n:
        if i < n - 1 and sym[i] == pair[0] and sym[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return out


def train_from_counts(word_counts: dict[str, int], n_merges: int,
                      min_freq: int = 2, verbose_every: int = 2000):
    """BPE na typach słów z wagami. Zwraca merges: (a, b) -> new_id."""
    t0 = time.time()
    words: list[list[int]] = []
    freqs: list[int] = []
    for w, c in word_counts.items():
        words.append(list(w.encode("utf-8")))
        freqs.append(c)

    pair_counts: dict[tuple[int, int], int] = {}
    pair_words: dict[tuple[int, int], set[int]] = {}
    for wi, sym in enumerate(words):
        f = freqs[wi]
        for p, c in _pairs_of(sym).items():
            pair_counts[p] = pair_counts.get(p, 0) + c * f
            pair_words.setdefault(p, set()).add(wi)

    # kopiec: (-count, a, b) -> zdejmuje najpierw największą częstość,
    # a przy remisie parę o najmniejszych ID (tie_break="min_pair")
    heap = [(-c, p[0], p[1]) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    print(f"start: {len(words):,} typów, {len(pair_counts):,} par "
          f"({time.time()-t0:.1f}s)", flush=True)

    merges: dict[tuple[int, int], int] = {}
    next_id = 256
    t1 = time.time()

    for step in range(n_merges):
        # zdejmij szczyt, pomijając nieaktualne wpisy
        best = None
        while heap:
            negc, a, b = heapq.heappop(heap)
            p = (a, b)
            cur = pair_counts.get(p, 0)
            if cur and -negc == cur:
                best = p
                best_count = cur
                break
        if best is None or best_count < min_freq:
            print(f"stop na {step} merge'ach (brak par o częstości >= {min_freq})")
            break

        new_id = next_id
        for wi in list(pair_words.get(best, ())):
            sym = words[wi]
            f = freqs[wi]
            old = _pairs_of(sym)
            if best not in old:
                continue
            new_sym = _apply(sym, best, new_id)
            new = _pairs_of(new_sym)
            words[wi] = new_sym
            touched = set(old) | set(new)
            for p in touched:
                delta = (new.get(p, 0) - old.get(p, 0)) * f
                if delta:
                    c = pair_counts.get(p, 0) + delta
                    if c <= 0:
                        pair_counts.pop(p, None)
                    else:
                        pair_counts[p] = c
                        heapq.heappush(heap, (-c, p[0], p[1]))
                if new.get(p, 0) == 0:
                    s = pair_words.get(p)
                    if s is not None:
                        s.discard(wi)
                elif old.get(p, 0) == 0:
                    pair_words.setdefault(p, set()).add(wi)
        pair_counts.pop(best, None)
        pair_words.pop(best, None)

        merges[best] = new_id
        next_id += 1
        if verbose_every and (step + 1) % verbose_every == 0:
            el = time.time() - t1
            print(f"  merge {step+1}/{n_merges} id={new_id} freq={best_count:,} "
                  f"| {el:.0f}s, {(step+1)/el:.0f} merge/s", flush=True)

    print(f"trening BPE: {len(merges)} merge'y w {time.time()-t1:.1f}s")
    return merges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/hplt-pl/corpus_train.txt")
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--pattern", default="pl", choices=list(PATTERNS))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--counts-cache", default=None,
                    help="cache JSON ze zliczonymi pretokenami (oszczędza minutę)")
    args = ap.parse_args()

    name = args.name or f"hplt-pl-{args.vocab_size//1000}k-{args.pattern}"
    out = args.out or f"tokenizers/{name}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    specials = {"<|endoftext|>": args.vocab_size - 1}
    n_merges = args.vocab_size - 256 - len(specials)

    print(f"=== {name}: vocab {args.vocab_size} "
          f"({n_merges} merge'y + 256 bajtów + {len(specials)} specjalne) ===")

    cache = args.counts_cache or f"../data/hplt-pl/pretokens-{args.pattern}.json"
    if os.path.exists(cache):
        print(f"wczytuję zliczone pretokeny z {cache}")
        with open(cache, encoding="utf-8") as f:
            counts = json.load(f)
    else:
        counts = dict(count_pretokens(args.corpus, PATTERNS[args.pattern],
                                      args.workers))
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(counts, f, ensure_ascii=False)
        print(f"zapisano cache -> {cache}")

    corpus_bytes = os.path.getsize(args.corpus)
    t0 = time.time()
    merges = train_from_counts(counts, n_merges)
    train_s = time.time() - t0

    tok = BPETokenizer(
        merges=merges,
        pattern_name=args.pattern,
        special_tokens=specials,
        meta={
            "name": name,
            "dataset": "SlayerLab/hplt-v3-pl-cleaned",
            "shard": "data/european_hplt_v3_pl_bin8_6_p0.parquet",
            "corpus_bytes": corpus_bytes,
            "corpus_mb": round(corpus_bytes / 1e6, 1),
            "n_word_types": len(counts),
            "n_word_occurrences": sum(counts.values()),
            "tie_break": "min_pair",
            "train_seconds": round(train_s, 1),
            "trained_at": time.strftime("%Y-%m-%d"),
        },
    )
    tok.save(out)
    print(f"zapisano {out} ({os.path.getsize(out)/1e6:.1f} MB), "
          f"vocab_size={tok.vocab_size}")

    demo = "Niezaprzeczalnie skomplikowana morfosyntaktyczna struktura języka polskiego."
    ids = tok.encode(demo)
    print(f"demo: {len(ids)} tokenów")
    print("  ", [tok.token_str(i) for i in ids])
    assert tok.decode(ids) == demo, "round-trip się nie zgadza!"


if __name__ == "__main__":
    main()

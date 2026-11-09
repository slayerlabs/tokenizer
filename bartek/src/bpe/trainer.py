"""Byte-level BPE training.

The algorithm is the same as in the workshop (count pairs -> merge the most frequent
-> repeat); only the bookkeeping changes so it can run on ~100 MB in pure Python:

* the corpus is a {pre-token: frequency} dict instead of one byte stream, so repeated
  words are processed once;
* pair counts are updated incrementally: after a merge only the words that contained
  the merged pair are recounted (pair -> words index);
* the most frequent pair comes from a heap with lazy deletion of stale entries.

Ties are broken deterministically: highest count first, then the smallest pair (tuple
comparison of IDs). Since the algorithm does not depend on the target size, the merges
for vocab_size=V are exactly a prefix of the merges for any larger V, so one training
run up to 64k yields the 1k...64k tokenizers.
"""

from __future__ import annotations

import heapq
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

Pair = tuple[int, int]


@dataclass(frozen=True)
class Merge:
    pair: Pair
    new_id: int
    count: int  # pair occurrences in the corpus at merge time


def _merge_word(word: list[int], pair: Pair, new_id: int) -> list[int]:
    a, b = pair
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train(word_counts: Counter[bytes] | dict[bytes, int], vocab_size: int,
          min_count: int = 2, verbose: bool = True) -> list[Merge]:
    """Train BPE on a pre-token frequency dict. Returns merges in rank order."""
    words: list[list[int]] = []
    freqs: list[int] = []
    for w, f in word_counts.items():
        if len(w) >= 2:  # single-byte pre-tokens have no pairs
            words.append(list(w))
            freqs.append(f)

    pair_counts: dict[Pair, int] = defaultdict(int)
    where: dict[Pair, set[int]] = defaultdict(set)
    for idx, (w, f) in enumerate(zip(words, freqs)):
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            where[p].add(idx)

    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[Merge] = []
    next_id = 256
    t0 = time.time()
    while next_id < vocab_size and heap:
        neg, pair = heapq.heappop(heap)
        cur = pair_counts.get(pair, 0)
        if cur != -neg:  # stale entry
            if cur > 0:
                heapq.heappush(heap, (-cur, pair))
            continue
        if cur < min_count:
            break  # no pair is worth merging anymore

        new_id = next_id
        next_id += 1
        merges.append(Merge(pair, new_id, cur))

        changed: set[Pair] = set()
        for idx in where.pop(pair):
            w = words[idx]
            f = freqs[idx]
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                changed.add(p)
            w = _merge_word(w, pair, new_id)
            words[idx] = w
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                where[p].add(idx)
                changed.add(p)
        pair_counts.pop(pair, None)
        changed.discard(pair)
        for p in changed:
            c = pair_counts[p]
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                del pair_counts[p]
                where.pop(p, None)

        if verbose and (len(merges) % 2000 == 0):
            print(f"  merge {len(merges):>6}  count={cur:>8}  {time.time() - t0:7.1f}s", flush=True)
    return merges

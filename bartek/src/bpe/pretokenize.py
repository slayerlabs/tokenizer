"""Pre-tokenization: split text into chunks inside which BPE operates.

BPE still works on UTF-8 bytes; the regex only defines boundaries that a merge may
not cross (e.g. no token like "a." or "ę w" can appear). The pattern is based on
cl100k_base (GPT-4) without the English contractions ('s, 't, ...), which are noise
for Polish. The digit handling variant is a parameter: it is the axis of research question 3.4 (numbers).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from multiprocessing import Pool

import regex

# {digits} is replaced by the selected number-handling variant
_TEMPLATE = (
    r"[^\r\n\p{{L}}\p{{N}}]?+\p{{L}}+"  # word with an optional leading char (usually a space)
    r"|{digits}"  # numbers
    r"| ?[^\s\p{{L}}\p{{N}}]++[\r\n]*"  # punctuation / symbols
    r"|\s*[\r\n]"  # newlines
    r"|\s+(?!\S)"  # whitespace (except the last space before a word)
    r"|\s+"
)

DIGIT_MODES = {
    "group3": r"\p{N}{1,3}",  # GPT-4: groups of up to 3 digits, left-aligned
    "single": r"\p{N}",  # Llama/Mistral-like: every digit separately
    "free": r"\p{N}+",  # no constraint: BPE decides on its own
}


def compile_pattern(digits: str = "group3") -> regex.Pattern:
    return regex.compile(_TEMPLATE.format(digits=DIGIT_MODES[digits]))


def pretokenize(text: str, pattern: regex.Pattern) -> list[str]:
    return pattern.findall(text)


_worker_pattern: regex.Pattern | None = None


def _init(digits: str) -> None:
    global _worker_pattern
    _worker_pattern = compile_pattern(digits)


def _count_chunk(texts: list[str]) -> Counter[str]:
    c: Counter[str] = Counter()
    for t in texts:
        c.update(_worker_pattern.findall(t))
    return c


def count_pretokens(texts: Iterable[str], digits: str = "group3", workers: int = 8,
                    chunk: int = 200) -> Counter[bytes]:
    """Count pre-tokens in the corpus (in parallel) and return them as UTF-8 bytes."""
    batches: list[list[str]] = []
    cur: list[str] = []
    for t in texts:
        cur.append(t)
        if len(cur) >= chunk:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    total: Counter[str] = Counter()
    with Pool(workers, initializer=_init, initargs=(digits,)) as pool:
        for part in pool.imap_unordered(_count_chunk, batches):
            total.update(part)
    return Counter({k.encode("utf-8"): v for k, v in total.items()})

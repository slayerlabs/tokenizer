"""Shared helpers for held-out evaluation (parallel encoding, data loading)."""

from __future__ import annotations

import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from .tokenizer import Tokenizer

HELDOUT = Path("data/clean/heldout.jsonl")
SOURCES = ["web", "wiki", "literature", "parliament", "law"]


def load_jsonl(path: Path = HELDOUT) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


_tok: Tokenizer | None = None


def _init(tok_dir: str) -> None:
    global _tok
    _tok = Tokenizer.load(tok_dir)


def _encode_batch(texts: list[str]) -> tuple[list[int], Counter]:
    counts, usage = [], Counter()
    for t in texts:
        ids = _tok.encode(t)
        counts.append(len(ids))
        usage.update(ids)
    return counts, usage


def encode_corpus(tok_dir: str | Path, texts: list[str], workers: int = 14,
                  batch: int = 50) -> tuple[list[int], Counter]:
    """Encode texts in parallel. Returns per-text token counts and global token-id usage."""
    batches = [texts[i:i + batch] for i in range(0, len(texts), batch)]
    counts: list[int] = []
    usage: Counter = Counter()
    with Pool(workers, initializer=_init, initargs=(str(tok_dir),)) as pool:
        for c, u in pool.imap(_encode_batch, batches):  # imap keeps the order
            counts.extend(c)
            usage.update(u)
    return counts, usage


def size_dirs(root: str | Path) -> list[tuple[int, Path]]:
    dirs = [(int(p.name[1:]), p) for p in Path(root).glob("v*") if (p / "merges.txt").exists()]
    return sorted(dirs)


def merge_counts(tok_dir: str | Path) -> list[int]:
    """Pair frequency at merge time for each merge (column 4 of merges.txt)."""
    out = []
    for line in open(Path(tok_dir) / "merges.txt", encoding="utf-8"):
        if not line.startswith("#"):
            out.append(int(line.split(" | ", 1)[0].split()[3]))
    return out

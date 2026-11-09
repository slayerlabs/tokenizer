"""Tokenizer: encode/decode plus saving/loading artifacts (merges.txt, vocab.json)."""

from __future__ import annotations

import json
from pathlib import Path

from .pretokenize import compile_pattern
from .trainer import Merge, Pair


class Tokenizer:
    def __init__(self, merges: list[Pair], digits: str = "group3"):
        self.merges = list(merges)
        self.digits = digits
        self.pattern = compile_pattern(digits)
        self.ranks: dict[Pair, int] = {p: 256 + i for i, p in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.ranks.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def truncated(self, vocab_size: int) -> Tokenizer:
        """A smaller-vocab tokenizer is a prefix of the merge list (see trainer.py)."""
        assert 256 <= vocab_size <= self.vocab_size
        return Tokenizer(self.merges[: vocab_size - 256], self.digits)

    # --- encoding --------------------------------------------------------------
    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk)
        ranks = self.ranks
        while len(ids) >= 2:
            # pair with the lowest ID = learned earliest (same rule as in the workshop)
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            pair = (ids[best_i], ids[best_i + 1])
            # merge all occurrences of this pair in the chunk
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    out.append(best_rank)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        result = tuple(ids)
        if len(self._cache) < 2_000_000:
            self._cache[chunk] = result
        return result

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for piece in self.pattern.findall(text):
            out.extend(self._encode_chunk(piece.encode("utf-8")))
        return out

    def count_tokens(self, text: str) -> int:
        return sum(len(self._encode_chunk(p.encode("utf-8"))) for p in self.pattern.findall(text))

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def piece(self, token_id: int) -> str:
        raw = self.vocab[token_id]
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "<" + raw.hex(" ") + ">"

    # --- artifacts ----------------------------------------------------------------
    def save(self, directory: str | Path, merge_counts: list[int] | None = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "merges.txt", "w", encoding="utf-8") as f:
            f.write(f"#bpe-bytes v1 digits={self.digits} vocab_size={self.vocab_size}\n")
            f.write("#new_id left_id right_id count | left + right -> merged (preview)\n")
            for i, (a, b) in enumerate(self.merges):
                nid = 256 + i
                cnt = merge_counts[i] if merge_counts else -1
                f.write(f"{nid} {a} {b} {cnt} | {self.piece(a)!r} + {self.piece(b)!r} -> {self.piece(nid)!r}\n")
        vocab = {str(i): {"hex": self.vocab[i].hex(), "text": self.piece(i)} for i in range(self.vocab_size)}
        with open(d / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, directory: str | Path) -> Tokenizer:
        merges, digits = [], "group3"
        with open(Path(directory) / "merges.txt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    for tok in line.split():
                        if tok.startswith("digits="):
                            digits = tok.split("=", 1)[1]
                    continue
                parts = line.split(" | ", 1)[0].split()
                merges.append((int(parts[1]), int(parts[2])))
        return cls(merges, digits)


def from_training(merges: list[Merge], digits: str) -> Tokenizer:
    return Tokenizer([m.pair for m in merges], digits)

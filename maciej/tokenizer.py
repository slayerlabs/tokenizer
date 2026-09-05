"""BPETokenizer — artefakt: pretokenizacja + merge'e + tokeny specjalne + I/O.

Rdzeń algorytmu jest w `bpe.py` (wersja ze slajdów). Tutaj jest to, co odróżnia
"działający kod z warsztatu" od "tokenizera, który da się wersjonować i wgrać do
treningu modelu":

  * pretokenizacja regexem (slajd 18: produkcyjne BPE nie trenuje na surowym
    strumieniu bajtów — merge'e nie przechodzą przez granice słów),
  * tokeny specjalne z zarezerwowanymi ID, ATOMOWE (nigdy nie przechodzą przez BPE),
  * zapis/odczyt JSON z metadanymi treningu — tokenizer to osobny artefakt,
    wersjonowany niezależnie od wag modelu.

Zależność: `regex` (potrzebne \\p{L}/\\p{N} — moduł `re` nie zna kategorii Unicode,
a bez nich polskie "ą" nie jest literą). Sam BPE dalej jest czystym Pythonem.
"""

from __future__ import annotations

import json
from functools import lru_cache

import regex as re

# --- wzorce pretokenizacji --------------------------------------------------
# GPT-2: klasyka ze slajdu 18. Angielskie skróty ('s, 't, 're...) są dla polskiego
# martwym balastem — zjadają merge'e na sekwencje, których w korpusie prawie nie ma.
PATTERNS = {
    "gpt2": r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    # "pl": bez angielskich skrótów, cyfry w runach <=3 (jak cl100k — inaczej BPE
    # marnuje słownik na daty i numery telefonów z web-crawla).
    "pl": r""" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
}

DEFAULT_PATTERN = "pl"


@lru_cache(maxsize=None)
def bytes_to_unicode() -> dict[int, str]:
    """Bajt -> drukowalny znak (mapowanie z GPT-2).

    Po co: w JSON-ie chcemy ZOBACZYĆ tokeny, a surowy bajt 0x0A czy 0xC4 nie jest
    drukowalny. Mapowanie jest bijekcją, więc artefakt zostaje bezstratny.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _unicode_to_bytes() -> dict[str, int]:
    return {v: k for k, v in bytes_to_unicode().items()}


class BPETokenizer:
    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        pattern_name: str = DEFAULT_PATTERN,
        special_tokens: dict[str, int] | None = None,
        meta: dict | None = None,
    ):
        self.merges = dict(merges or {})          # (a, b) -> new_id, w kolejności uczenia
        self.pattern_name = pattern_name
        self.pattern = PATTERNS[pattern_name]
        self._re = re.compile(self.pattern)
        self.special_tokens = dict(special_tokens or {})
        self.meta = dict(meta or {})
        self._rebuild()

    # --- stan pochodny ---
    def _rebuild(self) -> None:
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._inv_special = {v: k for k, v in self.special_tokens.items()}
        for tok, tid in self.special_tokens.items():
            self.vocab[tid] = tok.encode("utf-8")
        self._special_re = (
            re.compile("(" + "|".join(re.escape(k) for k in self.special_tokens) + ")")
            if self.special_tokens else None
        )
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    # --- kodowanie ---
    def _encode_chunk(self, piece: bytes) -> tuple[int, ...]:
        """BPE na pojedynczym pretokenie. Merge'e w kolejności uczenia (rank)."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece)
        while len(ids) >= 2:
            # najniższy rank = reguła wyuczona najwcześniej (slajd 4.4)
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id = self.merges[best]
            merged, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        out = tuple(ids)
        if len(self._cache) < 1_000_000:
            self._cache[piece] = out
        return out

    def encode_ordinary(self, text: str) -> list[int]:
        """Bez obsługi tokenów specjalnych — tekst użytkownika."""
        out: list[int] = []
        for piece in self._re.findall(text):
            out.extend(self._encode_chunk(piece.encode("utf-8")))
        return out

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Tokeny specjalne są ATOMOWE: wycinamy je przed BPE (slajd 18)."""
        if not allowed_special or not self._special_re:
            return self.encode_ordinary(text)
        out: list[int] = []
        for part in self._special_re.split(text):
            if not part:
                continue
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_str(self, tid: int) -> str:
        """Czytelna forma tokenu (mapowanie bajt->znak z GPT-2)."""
        if tid in self._inv_special:
            return self._inv_special[tid]
        b2u = bytes_to_unicode()
        return "".join(b2u[b] for b in self.vocab[tid])

    # --- I/O ---
    def save(self, path: str) -> None:
        b2u = bytes_to_unicode()
        payload = {
            "format": "slayerlabs-bpe",
            "format_version": 1,
            "pattern_name": self.pattern_name,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "n_merges": len(self.merges),
            "special_tokens": self.special_tokens,
            "meta": self.meta,
            # merge i-ty ma ID 256+i; zapisujemy pary ID -> kolejność jest jawna
            "merges": [[a, b] for (a, b) in self.merges],
            # czytelny podgląd: ID -> token w mapowaniu bajt->znak (bezstratne)
            "vocab": {str(i): "".join(b2u[b] for b in self.vocab[i])
                      for i in range(256 + len(self.merges))},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        merges = {(a, b): 256 + i for i, (a, b) in enumerate(d["merges"])}
        return cls(
            merges=merges,
            pattern_name=d.get("pattern_name", DEFAULT_PATTERN),
            special_tokens=d.get("special_tokens") or {},
            meta=d.get("meta") or {},
        )

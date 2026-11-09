"""Small text utilities shared by training and evaluation."""

from __future__ import annotations

_PL = "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
_ASCII = "acelnoszzACELNOSZZ"
_STRIP = str.maketrans(_PL, _ASCII)


def strip_diacritics(text: str) -> str:
    """Remove Polish diacritics the way people type without them (ą->a, ł->l, ż/ź->z)."""
    return text.translate(_STRIP)


def count_words(text: str) -> int:
    """Word = whitespace-delimited segment (the usual definition behind 'fertility')."""
    return len(text.split())

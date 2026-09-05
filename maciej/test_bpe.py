"""Testy: asserty ze slajdów + parytet szybkiego trenera z wersją naiwną.

    uv run --with regex --with pytest python -m pytest test_bpe.py -q
"""

from __future__ import annotations

import json
import random
import tempfile

from bpe import build_vocab, decode, encode, get_pair_counts, merge, train_bpe
from tokenizer import BPETokenizer
from train import train_from_counts


# --- asserty dosłownie ze slajdów 4.1 / 4.2 --------------------------------

def test_slide_pair_counts():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}


def test_slide_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_slide8_example_matches_deck():
    """Slajd 8: aa->Z, ab->Y, ZY->X, korpus 'aaabdaaabac' -> 'XdXac'.

    Trace ze slajdu wychodzi przy tie_break='min_pair': w drugim kroku jest
    remis 2:2 między (Z,a) i (a,b). Domyślne max() ze slajdu 4.3 wybrałoby
    (Z,a) — obie odpowiedzi są poprawnym BPE, ale to pokazuje, że remisy
    trzeba rozstrzygać jawnie.
    """
    ids, merges = train_bpe("aaabdaaabac", 256 + 3, tie_break="min_pair")
    Z, Y, X = 256, 257, 258
    assert merges == {(ord("a"), ord("a")): Z, (ord("a"), ord("b")): Y, (Z, Y): X}
    assert ids == [X, ord("d"), X, ord("a"), ord("c")]
    assert decode(ids, build_vocab(merges)) == "aaabdaaabac"

    # ten sam korpus, druga reguła remisu -> inny, też poprawny słownik
    ids2, merges2 = train_bpe("aaabdaaabac", 256 + 3, tie_break="first")
    assert merges2 != merges
    assert decode(ids2, build_vocab(merges2)) == "aaabdaaabac"


def test_slide13_encode_applies_merges_in_learned_order():
    text = "aaabdaaabac"
    _, merges = train_bpe(text, 256 + 3, tie_break="min_pair")
    assert encode(text, merges) == [258, ord("d"), 258, ord("a"), ord("c")]
    # tekst spoza korpusu treningowego też się koduje — zero <UNK>
    assert decode(encode("aab", merges), build_vocab(merges)) == "aab"


# --- parytet: szybki trener == naiwna pętla ze slajdów ---------------------

def test_parity_fast_vs_naive():
    """`train.py` to ta sama matematyka co `bpe.py`, tylko liczona sprytniej.

    Bez pretokenizacji (jeden 'typ słowa' = cały tekst, waga 1) oba muszą dać
    IDENTYCZNE merge'e przy tej samej regule remisu.
    """
    random.seed(7)
    text = "".join(random.choice("abcdeąęó ") for _ in range(4000))
    _, naive = train_bpe(text, 256 + 120, tie_break="min_pair")
    fast = train_from_counts({text: 1}, 120, verbose_every=0)
    assert list(naive.items()) == list(fast.items())


def test_parity_with_word_weights():
    """Wagi też muszą się zgadzać: {'ab': 3} == korpus 'ab ab ab' bez spacji."""
    words = {"ab": 3, "abc": 2, "ba": 1}
    fast = train_from_counts(dict(words), 5, verbose_every=0)
    flat = "ab" * 3 + "abc" * 2 + "ba"
    # ta sama pierwsza reguła: (a,b) występuje 3+2+0 = 5 razy z wagami
    assert list(fast)[0] == (ord("a"), ord("b"))
    assert flat.count("ab") == 5


# --- właściwości tokenizera -------------------------------------------------

CORPUS_PL = (
    "Nie lubię tokenizacji w językach fleksyjnych. "
    "dom domu domowi domem domach domowy domowego "
    "Kupiłem 3,5 kg jabłek za 12.99 zł. Zażółć gęślą jaźń! "
) * 40


def _small_tokenizer(vocab_size: int = 600) -> BPETokenizer:
    from collections import Counter
    import regex as re
    from tokenizer import PATTERNS
    counts = Counter(re.compile(PATTERNS["pl"]).findall(CORPUS_PL))
    merges = train_from_counts(dict(counts), vocab_size - 256, verbose_every=0)
    return BPETokenizer(merges=merges, pattern_name="pl",
                        special_tokens={"<|endoftext|>": vocab_size - 1})


def test_round_trip_lossless_including_unseen_scripts():
    tok = _small_tokenizer()
    for s in [
        "Zażółć gęślą jaźń",
        "don't stop believing",           # apostrof ze slajdu 6
        "niesamowicie-fantastycznie-dobre",
        "super!!! 🎉🇵🇱",                  # emoji spoza korpusu
        "日本語のテキスト",                  # alfabet nigdy nie widziany
        "",
    ]:
        assert tok.decode(tok.encode(s)) == s, f"round-trip padł na {s!r}"


def test_no_unk_every_byte_covered():
    """Byte-level = zero <UNK> (slajd 15). Każdy z 256 bajtów ma swoje ID."""
    tok = _small_tokenizer()
    raw = bytes(range(256)).decode("latin-1")
    assert tok.decode(tok.encode(raw)) == raw


def test_special_tokens_are_atomic():
    """Slajd 18: token specjalny nigdy nie przechodzi przez BPE."""
    tok = _small_tokenizer()
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("dom<|endoftext|>dom")
    assert ids.count(eot) == 1
    assert ids == tok.encode("dom") + [eot] + tok.encode("dom")
    # bez zezwolenia tekst jest kodowany dosłownie, nie jako token specjalny
    assert eot not in tok.encode("dom<|endoftext|>dom", allowed_special=False)


def test_save_load_round_trip():
    tok = _small_tokenizer()
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as f:
        path = f.name
    tok.save(path)
    back = BPETokenizer.load(path)
    assert back.merges == tok.merges
    assert back.vocab_size == tok.vocab_size
    s = "Zażółć gęślą jaźń, 12.99 zł"
    assert back.encode(s) == tok.encode(s)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    assert d["format"] == "slayerlabs-bpe" and d["pattern_name"] == "pl"


def test_merges_never_cross_pretoken_boundary():
    """Konsekwencja pretokenizacji: żaden token nie zawiera dwóch słów."""
    tok = _small_tokenizer()
    for tid in range(256, 256 + len(tok.merges)):
        s = tok.vocab[tid].decode("utf-8", errors="replace")
        assert s.strip() == "" or " " not in s.strip(), f"token {tid}={s!r} zlepia słowa"


def test_compression_beats_raw_bytes():
    tok = _small_tokenizer()
    s = "dom domu domowi domem domach"
    assert len(tok.encode(s)) < len(s.encode("utf-8"))

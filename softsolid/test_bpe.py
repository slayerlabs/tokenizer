"""Testy bpe.py. Uruchomienie: ../.venv/bin/python -I test_bpe.py"""

import json
from collections import Counter
from pathlib import Path

import regex

from bpe import SPLIT_PATTERN, Tokenizer, get_pair_counts, merge, train, train_naive

HERE = Path(__file__).resolve().parent


def test_primitives():
    assert get_pair_counts([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([5, 5, 5], (5, 5), 9) == [9, 5]


def test_readme_example():
    # aaabdaaabac: aa -> Z, ab -> Y, ZY -> X, wynik XdXac (README zadania)
    merges, ids = train_naive("aaabdaaabac", 3)
    a, b = ord("a"), ord("b")
    assert merges == {(a, a): 256, (a, b): 257, (256, 257): 258}
    assert ids == [258, ord("d"), 258, ord("a"), ord("c")]


def naive_train_on_chunks(texts, vocab_size):
    """Wolny wzorzec: te same kawałki co train, ale pary liczone od nowa co krok."""
    pat = regex.compile(SPLIT_PATTERN)
    cc = Counter()
    for t in texts:
        cc.update(pat.findall(t))
    words = [list(c.encode("utf-8")) for c in cc]
    freqs = list(cc.values())
    merges = {}
    for new_id in range(256, vocab_size):
        counts = {}
        for w, f in zip(words, freqs):
            get_pair_counts(w, counts, f)
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        merges[pair] = new_id
        words = [merge(w, pair, new_id) for w in words]
    return merges


def sample_texts(n_chars):
    texts, total = [], 0
    with open(HERE / "data" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"][:3000]
            texts.append(t)
            total += len(t)
            if total >= n_chars:
                break
    return texts


def test_fast_equals_naive():
    texts = sample_texts(60_000)
    fast = train(texts, 256 + 400, log_every=0, log=lambda *a: None)
    slow = naive_train_on_chunks(texts, 256 + 400)
    assert fast.merges == slow, "szybki trening różni się od wzorca"


def test_roundtrip_and_special():
    texts = sample_texts(60_000)
    tok = train(texts, 256 + 300, log_every=0, log=lambda *a: None)
    samples = texts[:50] + ["Zażółć gęślą jaźń 🙂 ∑ 日本語", "", "  \n\t x"]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s
    eot = tok.special_tokens["<|endoftext|>"]
    ids = tok.encode("ala<|endoftext|>kot", allowed_special=True)
    assert ids.count(eot) == 1 and tok.decode(ids) == "ala<|endoftext|>kot"
    # bez zgody napis specjalny to zwykły tekst
    assert eot not in tok.encode("ala<|endoftext|>kot")
    # prefiks słownika jest prefiksem reguł
    small = tok.truncated(256 + 100)
    assert len(small.merges) == 100
    # zapis i odczyt nie zmieniają kodowania
    p = HERE / "data" / "_test_tok.json"
    tok.save(p)
    tok2 = Tokenizer.load(p)
    p.unlink()
    assert tok2.encode(texts[0]) == tok.encode(texts[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

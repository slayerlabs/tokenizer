"""Byte-level BPE od zera, w czystym Pythonie (bez tokenizers i sentencepiece).

Trzy elementy, w kolejności czytania:

1. get_pair_counts i merge: dwie prymitywne operacje, na których stoi cały BPE.
2. train: uczenie się listy reguł scalania (merges) z korpusu.
3. encode i decode: stosowanie reguł do nowego tekstu i droga powrotna.

Słownik bazowy to 256 bajtów, więc dowolny tekst UTF-8 da się zakodować bez
tokenu <UNK>. Każda reguła scalania dokłada jeden token: id 256, 257, ...
"""

import heapq
import json
import time
from collections import Counter, defaultdict

import regex

# Wzorzec pretokenizacji z cl100k_base (GPT-4). Tnie tekst na kawałki: słowa
# razem z poprzedzającą spacją, liczby po 1-3 cyfry, interpunkcję, białe znaki.
# BPE działa potem WEWNĄTRZ kawałka, więc nigdy nie skleja końca jednego słowa
# z początkiem następnego. \p{L} obejmuje polskie litery, więc "żółć" to jedno słowo.
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

ENDOFTEXT = "<|endoftext|>"


# ---------------------------------------------------------------------------
# 1. Dwie prymitywne operacje
# ---------------------------------------------------------------------------

def get_pair_counts(ids, counts=None, weight=1):
    """Liczy sąsiadujące pary w ciągu identyfikatorów.

    [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}

    weight pozwala policzyć parę od razu tyle razy, ile razy ciąg występuje
    w korpusie (patrz train), counts pozwala dopisywać do istniejącego licznika.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge(ids, pair, new_id):
    """Zastępuje każde wystąpienie pary nowym identyfikatorem, od lewej.

    merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]

    Przy parze z powtórzeniem idziemy od lewej i nie nakładamy dopasowań:
    merge([5, 5, 5], (5, 5), 9) -> [9, 5]
    """
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# 2. Trening
# ---------------------------------------------------------------------------

def train_naive(text, num_merges):
    """Podręcznikowa wersja: po każdym scaleniu liczymy wszystkie pary od nowa.

    Poprawna, ale na dużym korpusie nieużywalna (każdy krok to przejście po
    całym tekście). Służy do sprawdzenia przykładu z README zadania
    i jako wzorzec, z którym porównujemy szybką wersję w testach.
    """
    ids = list(text.encode("utf-8"))
    merges = {}
    for k in range(num_merges):
        counts = get_pair_counts(ids)
        if not counts:
            break
        # najczęstsza para; remis rozstrzyga mniejsza para (deterministycznie)
        pair = min(counts, key=lambda p: (-counts[p], p))
        new_id = 256 + k
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
    return merges, ids


def train(texts, vocab_size, log_every=1000, log=print):
    """Uczy się reguł scalania tak, by słownik miał vocab_size tokenów.

    Dwa triki odróżniają tę wersję od train_naive, a wynik jest ten sam:

    a) Liczymy unikalne kawałki z pretokenizacji razem z częstością. Słowo
       " jest" występuje w korpusie dziesiątki tysięcy razy, ale scalamy je
       raz i mnożymy przez częstość. Tak robi też biblioteka tokenizers.

    b) Po scaleniu nie liczymy wszystkich par od nowa. Poprawiamy liczniki
       tylko w słowach, które zawierały scalaną parę (indeks `where`),
       a najczęstszą parę trzymamy w kopcu (heap), z leniwym usuwaniem
       nieaktualnych wpisów.
    """
    assert vocab_size >= 256
    pat = regex.compile(SPLIT_PATTERN)

    # a) unikalne kawałki i ich częstości
    t0 = time.time()
    chunk_counts = Counter()
    for text in texts:
        chunk_counts.update(pat.findall(text))
    words = [list(ch.encode("utf-8")) for ch in chunk_counts]
    freqs = list(chunk_counts.values())
    log(f"pretokenizacja: {sum(freqs):,} kawałków, {len(words):,} unikalnych, "
        f"{time.time() - t0:.1f} s")

    # liczniki par ważone częstością i indeks: para -> słowa, w których jest
    pair_counts = {}
    where = defaultdict(set)
    for i, w in enumerate(words):
        get_pair_counts(w, pair_counts, freqs[i])
        for pair in zip(w, w[1:]):
            where[pair].add(i)

    # kopiec z wpisami (-licznik, para); najmniejszy wpis = najczęstsza para
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)

    merges = {}
    t0 = time.time()
    for new_id in range(256, vocab_size):
        # zdejmujemy wpisy, aż trafimy na aktualny (licznik się zgadza)
        while heap:
            neg, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) == -neg and -neg > 0:
                break
        else:
            log("brak par do scalenia, koniec wcześniej")
            break

        merges[pair] = new_id
        touched = set()
        for i in where.pop(pair):
            w = words[i]
            if len(w) < 2:
                continue
            new_w = merge(w, pair, new_id)
            if len(new_w) == len(w):
                continue  # nieaktualny wpis indeksu, para już tu nie występuje
            f = freqs[i]
            # odejmujemy stare pary słowa, dodajemy nowe
            for p in zip(w, w[1:]):
                pair_counts[p] -= f
                touched.add(p)
            for p in zip(new_w, new_w[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                where[p].add(i)
                touched.add(p)
            words[i] = new_w
        pair_counts.pop(pair, None)
        touched.discard(pair)
        for p in touched:
            c = pair_counts.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))
            else:
                pair_counts.pop(p, None)

        k = new_id - 255
        if log_every and k % log_every == 0:
            log(f"scalenie {k:>6,} / {vocab_size - 256:,}  "
                f"({time.time() - t0:.0f} s)  ostatni token: {_bytes_of(merges, new_id)!r}")

    return Tokenizer(merges)


def _bytes_of(merges, token_id):
    """Pomocniczo: bajty tokenu, tylko do logów."""
    return Tokenizer(merges).vocab[token_id]


# ---------------------------------------------------------------------------
# 3. Tokenizer: encode, decode, zapis
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, merges, special_tokens=None):
        # merges: {(a, b): new_id}, kolejność id = kolejność nauki
        self.merges = dict(merges)
        self.pat = regex.compile(SPLIT_PATTERN)
        # słownik id -> bajty, budowany w kolejności nauki
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        # tokeny specjalne nie pochodzą z BPE: dostają zarezerwowane id
        # za ostatnim tokenem z BPE i są atomowe (nigdy nie dzielone)
        if special_tokens is None:
            special_tokens = {ENDOFTEXT: 256 + len(self.merges)}
        self.special_tokens = dict(special_tokens)
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def truncated(self, vocab_size):
        """Tokenizer z pierwszych reguł. Reguły tworzą ciąg, więc mniejszy
        słownik to po prostu prefiks większego, bez ponownego treningu."""
        n = vocab_size - 256
        return Tokenizer({p: i for p, i in self.merges.items() if i < 256 + n})

    def _encode_chunk(self, chunk_bytes):
        """Koduje jeden kawałek z pretokenizacji.

        Kluczowa zasada: scalamy w kolejności NAUKI reguł, a nie od pary
        najczęstszej w tym tekście. W każdym kroku spośród par obecnych
        w kawałku wybieramy tę, której reguła powstała najwcześniej.
        """
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            pairs = get_pair_counts(ids)
            pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # żadna para nie ma reguły, koniec
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text):
        """Kodowanie bez tokenów specjalnych."""
        out = []
        for chunk in self.pat.findall(text):
            ids = self._cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk.encode("utf-8"))
                self._cache[chunk] = ids
            out.extend(ids)
        return out

    def encode(self, text, allowed_special=False):
        """Kodowanie. Przy allowed_special=True napis <|endoftext|> w tekście
        staje się jednym tokenem specjalnym, w przeciwnym razie zwykłym tekstem."""
        if not allowed_special or not self.special_tokens:
            return self.encode_ordinary(text)
        special_pat = "(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")"
        out = []
        for part in regex.split(special_pat, text):
            if part in self.special_tokens:
                out.append(self.special_tokens[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.inverse_special:
                parts.append(self.inverse_special[i].encode("utf-8"))
            else:
                raise ValueError(f"nieznany identyfikator tokenu: {i}")
        # pojedynczy token może być połową znaku UTF-8, stąd errors="replace"
        return b"".join(parts).decode("utf-8", errors="replace")

    # --- zapis i odczyt ---------------------------------------------------

    def save(self, path):
        """JSON z regułami w kolejności nauki. Słownik da się z nich odtworzyć."""
        data = {
            "type": "byte-level BPE",
            "pattern": SPLIT_PATTERN,
            "merges": [[a, b] for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])],
            "special_tokens": self.special_tokens,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merges = {(a, b): 256 + k for k, (a, b) in enumerate(data["merges"])}
        return cls(merges, data["special_tokens"])

    def save_vocab_readable(self, path):
        """Słownik do oglądania: id, tekst tokenu, z jakiej pary powstał."""
        parent = {idx: pair for pair, idx in self.merges.items()}
        with open(path, "w", encoding="utf-8") as f:
            for i in range(len(self.vocab)):
                s = self.vocab[i].decode("utf-8", errors="replace")
                # znaki niedrukowalne (\n, \x00 itd.) jako kody, żeby plik był czystym tekstem
                s = "".join(c if c.isprintable() else repr(c)[1:-1] for c in s)
                if i in parent:
                    a, b = parent[i]
                    f.write(f"{i}\t[{s}]\t<- {a} + {b}\n")
                else:
                    f.write(f"{i}\t[{s}]\n")

"""
Byte-level BPE od zera — rdzeń dydaktyczny.

Ten plik jest celowo *dosłownym* rozwinięciem slajdów 4.1–4.4 z warsztatu
(materials/tokenizer.pdf): cztery funkcje, zero zależności, zero optymalizacji.
Jest wolny i taki ma być — to wersja, którą się czyta, a nie ta, którą się
trenuje na gigabajtach (do tego jest `train.py`, sprawdzony na parytet
1:1 z tym plikiem — patrz `test_bpe.py::test_parity_fast_vs_naive`).

    get_pair_counts  ->  policz sąsiadujące pary
    merge            ->  zlej jedną parę w nowy token
    train_bpe        ->  powtórz N razy, zapisz reguły
    encode           ->  zastosuj reguły W KOLEJNOŚCI POWSTANIA
    decode           ->  posklejaj bajty z powrotem
"""

from __future__ import annotations


# --- 4.1 zliczanie par ------------------------------------------------------

def get_pair_counts(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Zlicz wystąpienia każdej sąsiadującej pary tokenów."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


# --- 4.2 merge --------------------------------------------------------------

def merge(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Zamień wszystkie (niezachodzące) wystąpienia pary na nowy token."""
    new_ids: list[int] = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(token_ids[i])
            i += 1
    return new_ids


# --- 4.3 pętla treningowa ---------------------------------------------------

def train_bpe(text: str, vocab_size: int, verbose: bool = False,
              tie_break: str = "first"):
    """Wytrenuj merge'e na surowym strumieniu bajtów UTF-8.

    Zwraca (token_ids, merges). `merges` to dict (a, b) -> new_id, a kolejność
    wstawiania do dicta = kolejność uczenia (Python 3.7+ gwarantuje insertion
    order) — z tego korzysta `encode`.

    `tie_break` — co zrobić, gdy dwie pary mają identyczną częstość. BPE tego NIE
    definiuje, a to realna pułapka: dwie poprawne implementacje rozjadą się na
    remisie i wyprodukują różne (obie prawidłowe) słowniki.
      "first"    — para napotkana najwcześniej; to zachowanie `max(counts,
                   key=counts.get)` ze slajdu (dict trzyma kolejność wstawiania).
      "min_pair" — para o najmniejszych ID. Nie zależy od kolejności skanowania,
                   więc da się ją odtworzyć w trenerze inkrementalnym — `train.py`
                   używa właśnie tej reguły (patrz test parytetu).
    """
    if vocab_size < 256:
        raise ValueError("vocab_size musi być >= 256 (sam alfabet bajtowy to 256)")

    token_ids = list(text.encode("utf-8"))   # start: surowe bajty, nie znaki
    merges: dict[tuple[int, int], int] = {}  # (a, b) -> new_id
    next_id = 256                            # 0-255 zajęte przez bajty

    for _ in range(vocab_size - 256):
        counts = get_pair_counts(token_ids)
        if not counts:
            break
        if tie_break == "first":
            # max() po wartości; przy remisie wygrywa para napotkana najwcześniej
            best_pair = max(counts, key=counts.get)
        elif tie_break == "min_pair":
            best_pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        else:
            raise ValueError(f"nieznany tie_break: {tie_break}")
        if counts[best_pair] < 2:
            break  # nie ma już nic, co powtarza się częściej niż raz
        token_ids = merge(token_ids, best_pair, next_id)
        merges[best_pair] = next_id
        if verbose:
            print(f"merge {next_id}: {best_pair} x{counts[best_pair]} "
                  f"-> len {len(token_ids)}")
        next_id += 1

    return token_ids, merges


# --- 4.4 encoder ------------------------------------------------------------

def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """Zastosuj wyuczone merge'e do nowego tekstu.

    Tu ludzie psują: merge'y NIE są aplikowane "najczęstszy najpierw" ani w
    dowolnej kolejności — muszą iść w tej samej kolejności, w jakiej powstały
    podczas treningu. Stąd `min(...)` po ID merge'a: najniższe ID = najwcześniej
    wyuczona reguła.
    """
    token_ids = list(text.encode("utf-8"))
    while len(token_ids) >= 2:
        counts = get_pair_counts(token_ids)
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nic więcej się nie da zlać
        token_ids = merge(token_ids, pair, merges[pair])
    return token_ids


# --- decoder ----------------------------------------------------------------

def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """ID -> bajty. Buduje się rekurencyjnie, bo merge'y są uporządkowane."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), new_id in merges.items():
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in token_ids).decode("utf-8", errors="replace")


# --- demo ze slajdu 8 -------------------------------------------------------

if __name__ == "__main__":
    ids, merges = train_bpe("aaabdaaabac", vocab_size=256 + 3, verbose=True)
    vocab = build_vocab(merges)
    print("merges:", merges)
    print("ids   :", ids)
    print("decode:", decode(ids, vocab))

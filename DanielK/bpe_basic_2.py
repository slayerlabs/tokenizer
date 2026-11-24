#!/usr/bin/env python3
"""Własny Byte-level BPE od zera (bez `tokenizers`, bez `sentencepiece`).

Rozwinięcie Twojego `bpe_basic_2.py`:
pre_tokenize -> count_frequency -> find_most_frequent_pair -> merge -> encode / decode,
z obsługą trybów (`standard`, `polish`, `polish,cnc`), wielokrotnych ścieżek `--input`
oraz zapisem `merges.txt`, `vocab.json` i `tokenizer.json` do osobnych katalogów.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

Pair = tuple[int, int]

CNC_DOMAIN_PATTERN = (
    r" ?(?:"
    r"\d+(?:[.,]\d+)? ?(?:mm|cm|m|µm|um|nm|HRC|HV|HB)"
    r"|R[az] ?\d+(?:[.,]\d+)?"
    r"|M\d+(?:[x×]\d+(?:[.,]\d+)?)?"
    r"|± ?\d+(?:[.,]\d+)?"
    r"|\b(?:js|JS|[A-Za-z])\d{1,2}\b"
    r")"
)

POLISH_ABBREV_PATTERN = (
    r" ?(?:m\.in\.|itd\.|itp\.|tzn\.|tj\.|np\.|ok\.|inż\.|prof\.|mgr\.|dr\."
    r"|ul\.|nr\.|art\.|ust\.|pkt\.|rys\.|tab\.|poz\.|godz\.|min\.|sek\.|tys\.|mln\.|mld\.)"
)

POLISH_PATTERN = (
    rf"{POLISH_ABBREV_PATTERN}"
    r"| ?(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)"
    r"| ?[^\W\d_]+(?:-[^\W\d_]+)*"
    r"| ?\d+(?:[.,]\d+)*"
    r"| ?[^\s\w]+"
    r"|\s*[\r\n]+"
    r"|\s+"
)

STANDARD_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?[^\W\d_]+"
    r"| ?\d+"
    r"| ?[^\s\w]+"
    r"|\s+"
)

VALID_MODES = {"standard", "polish", "cnc"}

DEMO_CORPUS = (
    "Tolerancja otworu wynosi H7, a chropowatość powierzchni Ra 0.8.\n"
    "Ustawiamy tolerancję do jednej setki, czyli 0.01 mm albo 0.001 mm (mikron).\n"
    "Gwint metryczny drobnozwojny M12x1,5 oraz M8 wymagają dokładnego sprawdzenia.\n"
    "Narzędzie wkładamy do maszyny i zapisujemy wartość offsetu w bibliotece z offsetami.\n"
    "Kalibracja sondy i calibration offset dla osi Z wymaga tolerancji ±0.02 mm.\n"
)


def bytes_to_unicode() -> dict[int, str]:
    """Standardowe mapowanie 256 bajtów na czytelne znaki Unicode (styl GPT-2 / HF ByteLevel)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


def build_pre_tokenize_regex(mode: str) -> re.Pattern[str]:
    """Buduje regex pre-tokenizera dla wybranego trybu (`standard`, `polish`, `polish,cnc`)."""
    modes = [m.strip().lower() for m in mode.split(",") if m.strip()]
    unknown = [m for m in modes if m not in VALID_MODES]
    if not modes or unknown:
        raise ValueError(f"Nieprawidłowy tryb --mode: {mode!r}. Dozwolone: {sorted(VALID_MODES)}.")

    unique_modes = list(dict.fromkeys(modes))
    if "standard" in unique_modes and len(unique_modes) > 1:
        raise ValueError("Tryb 'standard' nie może być łączony z 'polish' ani 'cnc'.")
    if unique_modes == ["standard"]:
        return re.compile(STANDARD_PATTERN)

    parts: list[str] = []
    if "cnc" in unique_modes:
        parts.append(CNC_DOMAIN_PATTERN)
    if "polish" in unique_modes:
        parts.append(POLISH_PATTERN)
    else:
        parts.append(STANDARD_PATTERN)
    return re.compile("|".join(parts))


def init_vocab() -> dict[int, str]:
    """Bazowy słownik 256 bajtów (0..255)."""
    return {b: BYTE_TO_CHAR[b] for b in range(256)}


def pre_tokenize(texts: Iterator[str], pattern: re.Pattern[str]) -> Counter[tuple[int, ...]]:
    """Dzieli teksty regexem i zlicza częstości unikalnych pre-tokenów (w bajtach UTF-8)."""
    word_counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        for piece in pattern.findall(text):
            byte_ids = tuple(piece.encode("utf-8"))
            if byte_ids:
                word_counts[byte_ids] += 1
    return word_counts


def count_frequency(word_counts: Counter[tuple[int, ...]]) -> Counter[Pair]:
    """Liczy częstość wszystkich sąsiadujących par tokenów."""
    pair_frequency: Counter[Pair] = Counter()
    for token_ids, freq in word_counts.items():
        if len(token_ids) < 2:
            continue
        for pair in zip(token_ids, token_ids[1:]):
            pair_frequency[pair] += freq
    return pair_frequency


def find_most_frequent_pair(pair_frequency: Counter[Pair]) -> tuple[Pair | None, int]:
    """Zwraca najczęstszą parę oraz jej częstość."""
    if not pair_frequency:
        return None, 0
    return pair_frequency.most_common(1)[0]


def update_merges_and_vocab(
    pair: Pair, merges: dict[Pair, int], vocab: dict[int, str]
) -> int:
    """Dodaje nową regułę merge do `merges` i nowy token do `vocab`."""
    new_id = 256 + len(merges)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return new_id


def merge_sequence(token_ids: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    """Scala wszystkie wystąpienia `pair` w jednej sekwencji (poprawne także dla `aaa`)."""
    merged: list[int] = []
    i = 0
    n = len(token_ids)
    p0, p1 = pair
    while i < n:
        if i < n - 1 and token_ids[i] == p0 and token_ids[i + 1] == p1:
            merged.append(new_id)
            i += 2
        else:
            merged.append(token_ids[i])
            i += 1
    return tuple(merged)


def merge(
    word_counts: Counter[tuple[int, ...]], pair: Pair, new_id: int
) -> Counter[tuple[int, ...]]:
    """Aplikuje scalenie `pair -> new_id` w całym korpusie pre-tokenów."""
    updated: Counter[tuple[int, ...]] = Counter()
    p0, p1 = pair
    for token_ids, freq in word_counts.items():
        if len(token_ids) >= 2 and p0 in token_ids and p1 in token_ids:
            updated[merge_sequence(token_ids, pair, new_id)] += freq
        else:
            updated[token_ids] += freq
    return updated


def train_bpe(
    texts: Iterator[str],
    vocab_size: int = 4096,
    min_frequency: int = 2,
    mode: str = "polish,cnc",
) -> tuple[dict[Pair, int], dict[int, str], re.Pattern[str]]:
    """Pętla treningowa Byte-level BPE z przyrostowym indeksem par (ta sama logika, ~100x szybciej)."""
    if vocab_size < 256:
        raise ValueError("vocab_size musi być >= 256.")

    pattern = build_pre_tokenize_regex(mode)
    vocab = init_vocab()
    merges: dict[Pair, int] = {}
    word_counts = pre_tokenize(texts, pattern)
    num_merges = vocab_size - 256

    words: list[tuple[int, ...]] = []
    freqs: list[int] = []
    pair_frequency: Counter[Pair] = Counter()
    pair_index: dict[Pair, set[int]] = defaultdict(set)

    for idx, (token_ids, freq) in enumerate(word_counts.items()):
        words.append(token_ids)
        freqs.append(freq)
        if len(token_ids) >= 2:
            for pair in zip(token_ids, token_ids[1:]):
                pair_frequency[pair] += freq
                pair_index[pair].add(idx)

    for step in range(num_merges):
        best_pair, best_freq = find_most_frequent_pair(pair_frequency)
        if best_pair is None or best_freq < min_frequency:
            print(f"Stop na kroku {step}: max frequency = {best_freq} < {min_frequency}")
            break

        new_id = update_merges_and_vocab(best_pair, merges, vocab)
        affected = list(pair_index.pop(best_pair, ()))
        pair_frequency.pop(best_pair, None)

        for idx in affected:
            old_ids = words[idx]
            freq = freqs[idx]
            new_ids = merge_sequence(old_ids, best_pair, new_id)
            if new_ids == old_ids:
                continue

            for p in zip(old_ids, old_ids[1:]):
                if p != best_pair:
                    new_cnt = pair_frequency[p] - freq
                    if new_cnt <= 0:
                        pair_frequency.pop(p, None)
                    else:
                        pair_frequency[p] = new_cnt
                    pair_index[p].discard(idx)

            words[idx] = new_ids
            if len(new_ids) >= 2:
                for p in zip(new_ids, new_ids[1:]):
                    pair_frequency[p] += freq
                    pair_index[p].add(idx)

        if step < 10 or (step + 1) % 250 == 0:
            print(
                f"krok {step + 1:04d} | {vocab[best_pair[0]]!r} + {vocab[best_pair[1]]!r} "
                f"-> {vocab[new_id]!r} (id={new_id}, freq={best_freq})"
            )

    return merges, vocab, pattern


def encode(text: str, merges: dict[Pair, int], pattern: re.Pattern[str]) -> list[int]:
    """Koduje tekst UTF-8 do listy ID tokenów zgodnie z kolejnością `merges`."""
    output_ids: list[int] = []
    for piece in pattern.findall(text):
        ids = tuple(piece.encode("utf-8"))
        while len(ids) >= 2:
            best = min(
                (
                    (merges[(ids[i], ids[i + 1])], (ids[i], ids[i + 1]))
                    for i in range(len(ids) - 1)
                    if (ids[i], ids[i + 1]) in merges
                ),
                default=None,
            )
            if best is None:
                break
            new_id, pair = best
            ids = merge_sequence(ids, pair, new_id)
        output_ids.extend(ids)
    return output_ids


def decode(token_ids: list[int], vocab: dict[int, str]) -> str:
    """Dekoduje listę ID tokenów z powrotem na tekst UTF-8."""
    byte_level_str = "".join(vocab[tid] for tid in token_ids)
    raw_bytes = bytes(CHAR_TO_BYTE[ch] for ch in byte_level_str)
    return raw_bytes.decode("utf-8", errors="replace")


def save_hf_files(
    output_path: Path, merges: dict[Pair, int], vocab: dict[int, str], pattern: re.Pattern[str]
) -> None:
    """Zapisuje `merges.txt`, `vocab.json` i `tokenizer.json` w katalogu `output_path.parent`."""
    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    str_to_id = {tok_str: tid for tid, tok_str in sorted(vocab.items())}
    (out_dir / "vocab.json").write_text(
        json.dumps(str_to_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    merge_lines = ["#version: 0.2"]
    merge_pairs_str: list[str] = []
    for (left_id, right_id), _new_id in sorted(merges.items(), key=lambda x: x[1]):
        rule = f"{vocab[left_id]} {vocab[right_id]}"
        merge_lines.append(rule)
        merge_pairs_str.append(rule)
    (out_dir / "merges.txt").write_text("\n".join(merge_lines) + "\n", encoding="utf-8")

    # Zapisujemy dokładny regex naszego pre-tokenizera w `tokenizer.json`,
    # dzięki czemu `evaluate_tokenizer.py` odtworzy go identycznie.
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": pattern.pattern},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": False,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": False,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": str_to_id,
            "merges": merge_pairs_str,
        },
    }
    output_path.write_text(
        json.dumps(tokenizer_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def iter_texts_from_paths(
    paths: list[Path], text_field: str = "text", max_mb: float = 0.0
) -> Iterator[str]:
    """Strumieniuje teksty z podanych plików/katalogów (.txt/.jsonl) z opcjonalnym limitem MB."""
    max_bytes = int(max_mb * 1024 * 1024) if max_mb > 0 else 0
    total_bytes = 0

    for path in paths:
        files = [path] if path.is_file() else sorted(path.rglob("*"))
        for file_path in files:
            if not file_path.is_file():
                continue
            if file_path.suffix == ".jsonl":
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        text = str(json.loads(line).get(text_field, "")).strip()
                        if not text:
                            continue
                        b = len(text.encode("utf-8"))
                        if max_bytes and total_bytes + b > max_bytes:
                            return
                        yield text
                        total_bytes += b
            elif file_path.suffix in {".txt", ".md", ".csv"}:
                with file_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        text = line.strip()
                        if not text:
                            continue
                        b = len(text.encode("utf-8"))
                        if max_bytes and total_bytes + b > max_bytes:
                            return
                        yield text
                        total_bytes += b


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Własny Byte-level BPE od zera.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        help="Jedna lub więcej ścieżek do plików (.txt/.jsonl) albo katalogów.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/bpe2/tokenizer.json"),
        help="Ścieżka do `tokenizer.json` (w tym samym folderze zapisze `merges.txt` i `vocab.json`).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="polish,cnc",
        help="Tryb pre-tokenizera po przecinku: 'standard', 'polish', 'cnc' (domyślnie 'polish,cnc').",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Docelowy rozmiar słownika (domyślnie 4096).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimalna częstość pary do scalenia (domyślnie 2).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=0.0,
        help="Opcjonalny limit MB wczytywanego korpusu (0 = bez limitu).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Nazwa pola z tekstem dla plików .jsonl.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Uruchom na wbudowanym mini-korpusie.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        texts = iter([DEMO_CORPUS] * 20)
        vocab_size = min(args.vocab_size, 520)
    elif args.input is not None:
        for path in args.input:
            if not path.exists():
                raise FileNotFoundError(f"Nie znaleziono ścieżki wejściowej: {path}")
        texts = iter_texts_from_paths(args.input, text_field=args.text_field, max_mb=args.max_mb)
        vocab_size = args.vocab_size
    else:
        raise SystemExit("Podaj --input <ścieżka> albo użyj --demo.")

    merges, vocab, pattern = train_bpe(
        texts=texts,
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        mode=args.mode,
    )
    save_hf_files(args.output, merges, vocab, pattern)

    sample = "tolerancja do jednej setki albo do 0.01 mm, otwór H7, Ra 0.8, gwint M12x1,5"
    ids = encode(sample, merges, pattern)
    decoded = decode(ids, vocab)
    assert decoded == sample, f"Round-trip failed: {decoded!r} != {sample!r}"

    print(f"\nZapisano do {args.output.parent}/ (merges.txt, vocab.json, tokenizer.json)")
    print(f"Rozmiar słownika: {len(vocab)}")
    print(f"Sample ({len(ids)} tokenów): {[vocab[i] for i in ids]}")


if __name__ == "__main__":
    main()

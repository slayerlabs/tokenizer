#!/usr/bin/env python3
"""Ewaluacja wytrenowanych tokenizerów pod kątem metryk z `hipotezy.md`.

Mierzone metryki:
- Fertility: średnia liczba tokenów na jedno słowo.
- Compression Ratio: Chars/Token oraz Bytes/Token.
- Rényi Efficiency (alpha=2.0) oraz odsetek aktywnych / martwych tokenów słownika.
- Boundary Preservation (%): jaki odsetek terminów domenowych przetrwał jako 1 token.
- Continued Words Rate (%): jaki odsetek słów został rozbity na więcej niż 1 token.

Przykład użycia:
    python evaluate_tokenizer.py --tokenizers artifacts/t1_standard.json artifacts/t2_cnc.json
    python evaluate_tokenizer.py --tokenizers artifacts/*.json --test-file dane/test_cnc.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

DEFAULT_TEST_TEXTS = [
    "tolerancja do jednej setki albo do 0.01 mm i 0.001 mm",
    "otwór wykonany w pasowaniu H7 z chropowatością powierzchni Ra 0.8",
    "gwint metryczny M12x1,5 oraz tolerancja ±0.02 mm na długości",
    "narzędzie wkładamy do maszyny i zapisujemy wartość offsetu w bibliotece z offsetami",
    "kalibracja sondy oraz calibration offset dla mikrometr i micrometer",
]

DEFAULT_DOMAIN_TERMS = [
    "0.001 mm",
    "0.01 mm",
    "0.1 mm",
    "H7",
    "H6",
    "Ra 0.8",
    "M12x1,5",
    "±0.02",
]

DEFAULT_CLASSIC_TEXTS = [
    "Azaliż nie rzekł był pan starosta, iżby tedy waćpan nie zwlekał jeno szedł ku grodowi?",
    "Białogłowa ona wielce zacna była, a dwór jej słynął z obfitości i gościnności.",
    "Kędyż podziały się one czasy, gdy rycerstwo w zbrojach stawało na wezwanie króla?",
]

# Pary PL-EN z hipotezy H4: wspólny prefiks ortograficzny vs kontrola negatywna (mikrometr/micrometer).
DEFAULT_PL_EN_POSITIVE_PAIRS = [
    ("tolerancja", "tolerance"),
    ("kalibracja", "calibration"),
    ("offsetu", "offset"),
]
DEFAULT_PL_EN_NEGATIVE_PAIR = ("mikrometr", "micrometer")


class TokenizerAdapter(Protocol):
    name: str
    vocab_size: int

    def encode_ids(self, text: str) -> list[int]: ...
    def encode_tokens(self, text: str) -> list[str]: ...
    def decode_ids(self, ids: list[int]) -> str: ...


class HFTokenizerAdapter:
    """Adapter dla lokalnych plików `tokenizer.json` i modeli z HuggingFace Hub."""

    def __init__(self, name: str, tokenizer: Tokenizer) -> None:
        self.name = name
        self._tok = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()

    def encode_ids(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def encode_tokens(self, text: str) -> list[str]:
        return self._tok.encode(text).tokens

    def decode_ids(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class TiktokenAdapter:
    """Adapter dla zewnętrznych baseline'ów `tiktoken` (np. `cl100k_base`, `o200k_base`)."""

    def __init__(self, encoding_name: str) -> None:
        import tiktoken

        self.name = encoding_name
        self._enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self._enc.n_vocab

    def encode_ids(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def encode_tokens(self, text: str) -> list[str]:
        return [
            self._enc.decode_single_token_bytes(tid).decode("utf-8", errors="replace")
            for tid in self._enc.encode(text)
        ]

    def decode_ids(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


def load_adapters(tokenizer_paths: list[Path], baselines_spec: str) -> list[TokenizerAdapter]:
    """Ładuje lokalne tokenizery oraz zewnętrzne baseline'y (np. `cl100k_base`, `hf:gpt2`)."""
    adapters: list[TokenizerAdapter] = []
    for path in tokenizer_paths:
        name = path.parent.name if path.stem == "tokenizer" else path.stem
        adapters.append(HFTokenizerAdapter(name, Tokenizer.from_file(str(path))))

    baseline_items = [b.strip() for b in baselines_spec.split(",") if b.strip()]
    for item in baseline_items:
        if item.lower() == "none":
            continue
        if item.startswith("hf:"):
            repo_id = item[3:]
            short_name = repo_id.split("/")[-1]
            adapters.append(HFTokenizerAdapter(short_name, Tokenizer.from_pretrained(repo_id)))
        else:
            adapters.append(TiktokenAdapter(item))
    return adapters


@dataclass(frozen=True)
class EvalMetrics:
    name: str
    vocab_size: int
    fertility: float
    chars_per_token: float
    bytes_per_token: float
    classic_fertility: float
    classic_bytes_per_token: float
    renyi_efficiency: float
    active_tokens_pct: float
    continued_words_pct: float
    boundary_preservation_pct: float
    shared_prefix_pct: float
    round_trip_pct: float


def compute_renyi_efficiency(token_counts: Counter[int], vocab_size: int, alpha: float = 2.0) -> float:
    """Oblicza znormalizowaną entropię Rényiego rzędu `alpha` w zakresie [0, 1]."""
    total = sum(token_counts.values())
    if total == 0 or vocab_size <= 1:
        return 0.0
    if alpha == 1.0:
        entropy = -sum((c / total) * math.log2(c / total) for c in token_counts.values())
    else:
        sum_p_alpha = sum((c / total) ** alpha for c in token_counts.values())
        entropy = math.log2(sum_p_alpha) / (1.0 - alpha)
    return entropy / math.log2(vocab_size)


def compute_corpus_stats(
    adapter: TokenizerAdapter, texts: list[str]
) -> tuple[float, float, float, float, float, Counter[int]]:
    """Zwraca (fertility, chars_per_token, bytes_per_token, continued_words_pct, round_trip_pct, token_counts)."""
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_words = 0
    continued_words = 0
    round_trip_ok = 0
    token_counts: Counter[int] = Counter()

    for text in texts:
        ids = adapter.encode_ids(text)
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(ids)
        token_counts.update(ids)
        if adapter.decode_ids(ids) == text:
            round_trip_ok += 1

        words = re.findall(r"\S+", text)
        total_words += len(words)
        for word in words:
            if len(adapter.encode_ids(word)) > 1:
                continued_words += 1

    fertility = total_tokens / total_words if total_words else 0.0
    chars_per_token = total_chars / total_tokens if total_tokens else 0.0
    bytes_per_token = total_bytes / total_tokens if total_tokens else 0.0
    continued_pct = (continued_words / total_words * 100.0) if total_words else 0.0
    round_trip_pct = (round_trip_ok / len(texts) * 100.0) if texts else 0.0
    return fertility, chars_per_token, bytes_per_token, continued_pct, round_trip_pct, token_counts


def evaluate_single(
    adapter: TokenizerAdapter,
    texts: list[str],
    classic_texts: list[str],
    domain_terms: list[str],
    alpha: float = 2.0,
) -> EvalMetrics:
    vocab_size = adapter.vocab_size

    fertility, chars_per_token, bytes_per_token, continued_pct, round_trip_pct, token_counts = (
        compute_corpus_stats(adapter, texts)
    )
    classic_fertility, _, classic_bytes_per_token, _, _, _ = compute_corpus_stats(
        adapter, classic_texts
    )

    preserved = sum(1 for term in domain_terms if len(adapter.encode_ids(term)) == 1)

    shared_prefix_hits = 0
    for pl_word, en_word in DEFAULT_PL_EN_POSITIVE_PAIRS:
        pl_ids = adapter.encode_ids(pl_word)
        en_ids = adapter.encode_ids(en_word)
        # Wspólny pierwszy token (prefiks/rdzeń) lub pełne scalenie.
        if pl_ids and en_ids and pl_ids[0] == en_ids[0]:
            shared_prefix_hits += 1

    renyi_eff = compute_renyi_efficiency(token_counts, vocab_size, alpha=alpha)
    active_pct = (len(token_counts) / vocab_size * 100.0) if vocab_size else 0.0
    boundary_pct = (preserved / len(domain_terms) * 100.0) if domain_terms else 0.0
    shared_prefix_pct = (
        shared_prefix_hits / len(DEFAULT_PL_EN_POSITIVE_PAIRS) * 100.0
        if DEFAULT_PL_EN_POSITIVE_PAIRS
        else 0.0
    )

    return EvalMetrics(
        name=adapter.name,
        vocab_size=vocab_size,
        fertility=fertility,
        chars_per_token=chars_per_token,
        bytes_per_token=bytes_per_token,
        classic_fertility=classic_fertility,
        classic_bytes_per_token=classic_bytes_per_token,
        renyi_efficiency=renyi_eff,
        active_tokens_pct=active_pct,
        continued_words_pct=continued_pct,
        boundary_preservation_pct=boundary_pct,
        shared_prefix_pct=shared_prefix_pct,
        round_trip_pct=round_trip_pct,
    )


def print_table(results: list[EvalMetrics]) -> None:
    header = (
        f"{'Model':<18} | {'Vocab':>6} | {'Fert':>6} | {'ClsFert':>7} | "
        f"{'Byt/Tok':>7} | {'ClsB/T':>6} | {'Rényi':>6} | {'Act%':>6} | {'Cont%':>6} | {'Bound%':>6} | {'Stem%':>5} | {'RT%':>4}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<18} | {r.vocab_size:>6} | {r.fertility:>6.3f} | {r.classic_fertility:>7.3f} | "
            f"{r.bytes_per_token:>7.2f} | {r.classic_bytes_per_token:>6.2f} | "
            f"{r.renyi_efficiency:>6.3f} | {r.active_tokens_pct:>5.1f}% | "
            f"{r.continued_words_pct:>5.1f}% | {r.boundary_preservation_pct:>5.1f}% | "
            f"{r.shared_prefix_pct:>4.0f}% | {r.round_trip_pct:>3.0f}%"
        )


def print_h4_stem_analysis(adapters: list[TokenizerAdapter]) -> None:
    """Wypisuje podział par PL-EN z hipotezy H4 (wspólne prefiksy vs kontrola negatywna)."""
    print("\n--- H4: Analiza wspólnych rdzeni PL-EN ---")
    all_pairs = [*DEFAULT_PL_EN_POSITIVE_PAIRS, DEFAULT_PL_EN_NEGATIVE_PAIR]
    for adapter in adapters:
        print(f"\n[{adapter.name}]")
        for pl_word, en_word in all_pairs:
            pl_ids = adapter.encode_ids(pl_word)
            en_ids = adapter.encode_ids(en_word)
            pl_toks = adapter.encode_tokens(pl_word)
            en_toks = adapter.encode_tokens(en_word)
            shared = bool(pl_ids and en_ids and pl_ids[0] == en_ids[0])
            tag = "NEG" if (pl_word, en_word) == DEFAULT_PL_EN_NEGATIVE_PAIR else "POS"
            print(
                f"  ({tag}) {pl_word} -> {pl_toks} | "
                f"{en_word} -> {en_toks} | wspólny prefiks={shared}"
            )


def _save_bar_comparison(
    plt,
    names: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Rysuje pogrupowany wykres słupkowy z wartościami nad słupkami."""
    x = list(range(len(names)))
    n_series = len(series)
    width = 0.72 / max(1, n_series)

    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2.2), 4.8))
    for idx, (label, values, color) in enumerate(series):
        offset = (idx - (n_series - 1) / 2.0) * width
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, values, width=width * 0.9, label=label, color=color, alpha=0.9)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if n_series > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_plots(
    results: list[EvalMetrics],
    adapters: list[TokenizerAdapter],
    domain_terms: list[str],
    plots_dir: Path,
) -> None:
    """Generuje zestaw wykresów matplotlib przypisanych do hipotez H1–H4."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    names = [r.name for r in results]

    # H1: Odarchaizowanie korpusu — Fertility i Bytes/Token na współczesnym vs klasycznym PL
    _save_bar_comparison(
        plt,
        names,
        [
            ("Modern Fertility ↓", [r.fertility for r in results], "#E07A5F"),
            ("Classic Fertility", [r.classic_fertility for r in results], "#F2CC8F"),
            ("Modern Bytes/Tok ↑", [r.bytes_per_token for r in results], "#3D405B"),
            ("Classic Bytes/Tok", [r.classic_bytes_per_token for r in results], "#81B29A"),
        ],
        title="H1: Współczesny PL vs klasyka (niższe Fertility, wyższy Bytes/Token = lepiej)",
        ylabel="Wartość",
        out_path=plots_dir / "h1_fertility_compression.png",
    )

    # H2: Domenowy pre-tokenizer — Boundary Preservation (%) i Continued Words Rate (%)
    _save_bar_comparison(
        plt,
        names,
        [
            ("Boundary Preservation (%) ↑", [r.boundary_preservation_pct for r in results], "#2A9D8F"),
            ("Continued Words Rate (%) ↓", [r.continued_words_pct for r in results], "#E76F51"),
        ],
        title="H2: Zachowanie granic jednostek CNC i fragmentacja słów",
        ylabel="Procent (%)",
        out_path=plots_dir / "h2_boundary_fragmentation.png",
    )

    # H2 (szczegóły): Heatmapa liczby tokenów na pojedynczy termin domenowy CNC
    matrix = [[len(adapter.encode_ids(term)) for term in domain_terms] for adapter in adapters]

    fig, ax = plt.subplots(figsize=(max(8, len(domain_terms) * 1.1), max(3.5, len(names) * 0.9)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=1)
    ax.set_xticks(range(len(domain_terms)))
    ax.set_xticklabels(domain_terms, rotation=25, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("H2: Liczba tokenów dla terminów CNC (1 = zachowany w całości)", fontweight="bold")
    for i in range(len(names)):
        for j in range(len(domain_terms)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="Liczba tokenów")
    fig.tight_layout()
    fig.savefig(plots_dir / "h2_cnc_terms_heatmap.png", dpi=150)
    plt.close(fig)

    # H3: Równomierność słownika — Rényi Efficiency oraz odsetek aktywnych tokenów
    _save_bar_comparison(
        plt,
        names,
        [
            ("Rényi Efficiency (×100) ↑", [r.renyi_efficiency * 100.0 for r in results], "#457B9D"),
            ("Aktywne tokeny (%) ↑", [r.active_tokens_pct for r in results], "#A8DADC"),
        ],
        title="H3: Efektywność Rényiego i wykorzystanie słownika",
        ylabel="Skala (%)",
        out_path=plots_dir / "h3_renyi_vocab_usage.png",
    )

    # H4: Gęstość słownika — Bytes/Token, Continued Words Rate i wspólne prefiksy PL-EN
    _save_bar_comparison(
        plt,
        names,
        [
            ("Bytes / Token ↑", [r.bytes_per_token for r in results], "#264653"),
            ("Continued Words / 10 ↓", [r.continued_words_pct / 10.0 for r in results], "#F4A261"),
            ("Shared Prefix PL-EN / 10 ↑", [r.shared_prefix_pct / 10.0 for r in results], "#2A9D8F"),
        ],
        title="H4: Gęstość słownika i współdzielenie prefiksów PL-EN",
        ylabel="Wartość",
        out_path=plots_dir / "h4_dictionary_density.png",
    )

    print(f"\nZapisano wykresy hipotez H1–H4 w katalogu: {plots_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Porównanie metryk tokenizerów.")
    parser.add_argument(
        "--tokenizers",
        type=Path,
        nargs="+",
        required=True,
        help="Lista plików tokenizer.json do porównania.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Opcjonalny plik .txt ze współczesnym korpusem testowym (domyślnie próbka z hipotezy.md).",
    )
    parser.add_argument(
        "--classic-file",
        type=Path,
        help="Opcjonalny plik .txt z klasycznym korpusem PL do sprawdzenia spadku na klasyce (H1).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Rząd entropii Rényiego (domyślnie 2.0).",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="cl100k_base",
        help="Zewnętrzne baseline'y po przecinku: np. 'cl100k_base,o200k_base,hf:gpt2' lub 'none'.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        help="Opcjonalny katalog zapisu wykresów matplotlib dla hipotez H1–H4.",
    )
    return parser.parse_args()


def _load_lines(path: Path | None, fallback: list[str]) -> list[str]:
    """Wczytuje linie tekstu z pliku (.txt/.jsonl) lub katalogu takich plików."""
    if path is None:
        return fallback
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    lines: list[str] = []
    for file_path in files:
        if not file_path.is_file():
            continue
        if file_path.suffix == ".jsonl":
            with file_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    text = str(json.loads(raw).get("text", "")).strip()
                    if text:
                        lines.append(text)
        elif file_path.suffix in {".txt", ".md"}:
            lines.extend(
                ln.strip()
                for ln in file_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()
            )
    return lines


def main() -> None:
    args = parse_args()
    texts = _load_lines(args.test_file, DEFAULT_TEST_TEXTS)
    classic_texts = _load_lines(args.classic_file, DEFAULT_CLASSIC_TEXTS)
    adapters = load_adapters(args.tokenizers, args.baselines)

    results = [
        evaluate_single(adapter, texts, classic_texts, DEFAULT_DOMAIN_TERMS, alpha=args.alpha)
        for adapter in adapters
    ]
    print_table(results)
    print_h4_stem_analysis(adapters)
    if args.plots_dir is not None:
        generate_plots(results, adapters, DEFAULT_DOMAIN_TERMS, args.plots_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Uwspółcześnianie klasycznych tekstów (Wolne Lektury) przez Gemini API pod eksperyment H1.

Skrypt czyta plik `.jsonl` (np. `dane_klasyka/wolne_lektury_pl.jsonl`), dzieli długie utwory
na spójne fragmenty (~2000 znaków) i zapisuje równolegle:
1. surowy fragment (`--raw-output`) — do treningu T0,
2. uwspółcześniony odpowiednik (`--output`) — do treningu T1.

Dzięki temu T0 i T1 dostają dokładnie tę samą treść i objętość, różniąc się wyłącznie
epoką języka (czysta ablacja pod H1). Obsługuje wznowienie po przerwaniu oraz współbieżność.

Wymaga zmiennej środowiskowej `GEMINI_API_KEY`:
    export GEMINI_API_KEY="AIza..."
    python modernize_corpus.py --input dane_klasyka/wolne_lektury_pl.jsonl --max-mb 15
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

SYSTEM_PROMPT = (
    "Jesteś redaktorem języka polskiego. Przepisz poniższy fragment tekstu literackiego "
    "na naturalną, współczesną polszczyznę (XXI wiek). Zastąp archaizmy, dawną składnię "
    "i przestarzałą ortografię współczesnymi odpowiednikami. Zachowaj dokładnie sens, "
    "fakty, imiona własne oraz ton wypowiedzi. Zwróć wyłącznie przepisany tekst, bez komentarzy."
)

GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
)


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Dzieli długi tekst na fragmenty po granicach akapitów/zdań do `max_chars` znaków."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if current:
                chunks.append("\n".join(current))
                current, current_len = [], 0
            for start in range(0, len(para), max_chars):
                piece = para[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        added = len(para) + (1 if current else 0)
        if current_len + added > max_chars:
            chunks.append("\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += added

    if current:
        chunks.append("\n".join(current))
    return chunks


def iter_chunks(input_path: Path, max_mb: float, max_chunks: int) -> Iterator[str]:
    """Strumieniuje fragmenty z pliku JSONL aż do osiągnięcia limitu MB lub liczby fragmentów."""
    max_bytes = int(max_mb * 1024 * 1024) if max_mb > 0 else 0
    total_bytes = 0
    count = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record.get("text", "")).strip()
            if not text:
                continue
            for chunk in chunk_text(text):
                chunk_bytes = len(chunk.encode("utf-8"))
                if max_bytes and total_bytes + chunk_bytes > max_bytes:
                    return
                yield chunk
                total_bytes += chunk_bytes
                count += 1
                if max_chunks and count >= max_chunks:
                    return


def call_gemini(text: str, api_key: str, model: str, max_retries: int = 5) -> str:
    """Wywołuje Gemini REST API z wykładniczym backoffem dla błędów 429/5xx."""
    url = GEMINI_URL_TEMPLATE.format(model=model, api_key=api_key)
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.2},
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
            return ""
        except urllib.error.HTTPError as err:
            if err.code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            detail = err.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {err.code}: {detail}") from err
        except urllib.error.URLError:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            raise
    return ""


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uwspółcześnianie klasyki PL przez Gemini API (ablacja H1)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dane_klasyka/wolne_lektury_pl.jsonl"),
        help="Wejściowy plik JSONL z klasyką PL.",
    )
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=Path("dane_h1/raw_klasyka.jsonl"),
        help="Wyjściowy plik JSONL z dokładnie tymi samymi surowymi fragmentami (dla T0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dane_h1/modern_pl.jsonl"),
        help="Wyjściowy plik JSONL ze uwspółcześnionymi fragmentami (dla T1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-3.5-flash-lite",
        help="Model Gemini (domyślnie gemini-3.5-flash-lite — najtańszy i najszybszy).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=15.0,
        help="Limit surowego tekstu wejściowego w MB (domyślnie 15 MB; 0 = bez limitu).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Opcjonalny limit liczby fragmentów (0 = bez limitu).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Liczba równoległych zapytań do API (domyślnie 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Brak pliku wejściowego: {args.input}")

    already_done = count_lines(args.output)
    all_chunks = list(iter_chunks(args.input, max_mb=args.max_mb, max_chunks=args.max_chunks))
    pending = all_chunks[already_done:]

    print(
        f"Fragmenty łącznie: {len(all_chunks)} | Już przetworzone: {already_done} | Do zrobienia: {len(pending)}"
    )
    if not pending:
        return

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Ustaw zmienną środowiskową GEMINI_API_KEY przed uruchomieniem.")

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Przetwarzamy w paczkach po `workers`, żeby zachować kolejność 1:1 między raw i modern.
    batch_size = max(1, args.workers * 2)
    processed = already_done

    with (
        args.raw_output.open("a", encoding="utf-8") as f_raw,
        args.output.open("a", encoding="utf-8") as f_mod,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            futures = {
                pool.submit(call_gemini, chunk, api_key, args.model): idx
                for idx, chunk in enumerate(batch)
            }
            results = [""] * len(batch)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

            for raw_chunk, mod_chunk in zip(batch, results):
                if mod_chunk:
                    f_raw.write(json.dumps({"text": raw_chunk}, ensure_ascii=False) + "\n")
                    f_mod.write(json.dumps({"text": mod_chunk}, ensure_ascii=False) + "\n")
            f_raw.flush()
            f_mod.flush()
            processed += len(batch)
            print(f"Postęp: {processed}/{len(all_chunks)}")


if __name__ == "__main__":
    main()

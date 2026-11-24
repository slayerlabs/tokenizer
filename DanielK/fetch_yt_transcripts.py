#!/usr/bin/env python3
"""Pobieranie i czyszczenie polskich transkrypcji (napisów) z kanału YouTube przez `yt-dlp`.

Nie pobiera plików wideo (`skip_download=True`) — ściąga wyłącznie napisy `.vtt` (ręczne
lub automatyczne `pl`), czyści znaczniki czasu, tagi `<c>` oraz zduplikowane linie
charakterystyczne dla napisów YouTube, a następnie zapisuje czysty tekst do `.jsonl`.

Obsługuje podział train/test (domyślnie 80% do treningu, 20% jako zbiór testowy).

Przykłady użycia:
    # Test na 3 najnowszych filmach
    python fetch_yt_transcripts.py --channel "https://www.youtube.com/@<kanal_cnc>/videos" --max-videos 3 --train-output dane_test_pl/yt_cnc.jsonl --test-output dane_test_eval/yt_cnc_test.jsonl

    # Pełny zaciąg z kanału (80% train / 20% test)
    python fetch_yt_transcripts.py --channel "https://www.youtube.com/@<kanal_cnc>/videos" --train-output dane_pl/yt_cnc.jsonl --test-output dane_eval/yt_cnc_test.jsonl
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import yt_dlp

DEFAULT_CHANNEL_URL = ""

TIMESTAMP_RE = re.compile(
    r"^(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}\s*-->\s*(?:\d{2}:)?\d{2}:\d{2}[.,]\d{3}.*$"
)
TAG_RE = re.compile(r"<[^>]+>")


def clean_vtt(vtt_path: Path) -> str:
    """Czyści plik .vtt z nagłówków, timestampów, tagów i powtarzających się linii."""
    lines: list[str] = []
    prev_line = ""

    for raw_line in vtt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE")):
            continue
        if line.isdigit() or TIMESTAMP_RE.match(line):
            continue

        # Usuwamy tagi typu <00:00:01.200><c>tekst</c> i encje HTML.
        cleaned = TAG_RE.sub("", line)
        cleaned = html.unescape(cleaned).strip()
        cleaned = " ".join(cleaned.split())
        if not cleaned or cleaned == prev_line:
            continue

        lines.append(cleaned)
        prev_line = cleaned

    return " ".join(lines)


def list_channel_video_urls(channel_url: str, max_videos: int) -> list[tuple[str, str]]:
    """Zwraca listę (url, title) filmów z kanału bez pobierania samych filmów."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
        "ignoreerrors": True,
    }
    if max_videos > 0:
        ydl_opts["playlistend"] = max_videos

    videos: list[tuple[str, str]] = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
        if not info:
            return videos
        for entry in info.get("entries") or []:
            if not entry:
                continue
            video_id = entry.get("id")
            url = entry.get("url") or (f"https://www.youtube.com/watch?v={video_id}" if video_id else None)
            if url:
                videos.append((url, entry.get("title") or video_id or "video"))
    return videos


def download_subtitles(videos: list[tuple[str, str]], subs_dir: Path, lang: str = "pl") -> list[Path]:
    """Pobiera wyłącznie napisy dla podanych filmów i zwraca posortowaną listę plików .vtt."""
    subs_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt",
        "outtmpl": str(subs_dir / "%(id)s.%(ext)s"),
        "ignoreerrors": True,
    }
    urls = [url for url, _ in videos]
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)

    return sorted(subs_dir.glob(f"*.{lang}.vtt"))


def write_split_jsonl(
    vtt_files: list[Path],
    train_output: Path,
    test_output: Path | None,
    train_ratio: float,
) -> tuple[int, int]:
    """Zapisuje oczyszczone transkrypcje do plików JSONL z deterministycznym podziałem train/test."""
    train_output.parent.mkdir(parents=True, exist_ok=True)
    if test_output is not None:
        test_output.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    # Deterministyczny podział co N-ty film do testu (np. dla 0.8 co 5. film idzie do testu).
    test_every = max(2, round(1.0 / max(1e-6, 1.0 - train_ratio))) if test_output and train_ratio < 1.0 else 0

    f_train = train_output.open("w", encoding="utf-8")
    f_test = test_output.open("w", encoding="utf-8") if test_output is not None else None

    try:
        for idx, vtt_path in enumerate(vtt_files):
            text = clean_vtt(vtt_path)
            if len(text) < 50:
                continue
            record = json.dumps({"text": text, "source": vtt_path.stem}, ensure_ascii=False) + "\n"
            if f_test is not None and test_every and (idx % test_every == test_every - 1):
                f_test.write(record)
                test_count += 1
            else:
                f_train.write(record)
                train_count += 1
    finally:
        f_train.close()
        if f_test is not None:
            f_test.close()

    return train_count, test_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pobieranie polskich transkrypcji z kanału YouTube przez yt-dlp."
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL_URL,
        help="URL kanału lub playlisty YouTube.",
    )
    parser.add_argument(
        "--subs-dir",
        type=Path,
        default=Path("dane_yt_vtt"),
        help="Katalog tymczasowy na pobrane pliki .vtt (domyślnie dane_yt_vtt).",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("dane_pl/yt_cnc.jsonl"),
        help="Wyjściowy plik JSONL z częścią treningową (domyślnie 80%%).",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("dane_eval/yt_cnc_test.jsonl"),
        help="Wyjściowy plik JSONL z częścią testową (domyślnie 20%%).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proporcja treningowa (domyślnie 0.8 = 80%% train / 20%% test).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maksymalna liczba filmów z kanału (0 = wszystkie dostępne).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="pl",
        help="Kod języka napisów (domyślnie 'pl').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.channel:
        raise SystemExit("Podaj URL kanału lub playlisty YouTube przez --channel <URL>.")
    print(f"[yt] Pobieranie listy filmów z {args.channel}...")
    videos = list_channel_video_urls(args.channel, max_videos=args.max_videos)
    print(f"[yt] Znaleziono {len(videos)} filmów. Pobieranie napisów ({args.lang})...")
    if not videos:
        raise SystemExit("Nie znaleziono żadnych filmów pod podanym adresem.")

    vtt_files = download_subtitles(videos, subs_dir=args.subs_dir, lang=args.lang)
    train_n, test_n = write_split_jsonl(
        vtt_files,
        train_output=args.train_output,
        test_output=args.test_output,
        train_ratio=args.train_ratio,
    )
    print(
        f"[yt] Zapisano transkrypcje -> train: {train_n} ({args.train_output}) | "
        f"test: {test_n} ({args.test_output})"
    )


if __name__ == "__main__":
    main()

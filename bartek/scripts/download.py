#!/usr/bin/env python3
"""Download raw text from five legally reusable sources into data/raw/<source>.jsonl.

Each line: {"source": ..., "id": ..., "text": ...}. The script is idempotent: a source
whose output file already exists is skipped. Size limits (MB) leave headroom because
cleaning and deduplication will drop part of the text.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import requests

RAW = Path("data/raw")
CACHE = Path("data/cache")
HF = "https://huggingface.co/datasets"
UA = {"User-Agent": "tokenizer-course-research/1.0 (BPE homework; low rate)"}

# Target amount of raw text (MB) per source, before cleaning.
TARGET_MB = {
    "web": 75,  # FineWeb-2 pol_Latn
    "wiki": 35,  # Wikipedia PL
    "literature": 30,  # Wolne Lektury
    "parliament": 30,  # Polish Sejm speeches
    "law": 25,  # legal acts from Dziennik Ustaw (ELI API)
}


def fetch_file(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with requests.get(url, headers=UA, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def write_docs(name: str, docs) -> None:
    out = RAW / f"{name}.jsonl"
    limit = TARGET_MB[name] * 1_000_000
    total = n = 0
    tmp = out.with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for doc_id, text in docs:
            if not text or not text.strip():
                continue
            f.write(json.dumps({"source": name, "id": str(doc_id), "text": text}, ensure_ascii=False) + "\n")
            total += len(text.encode("utf-8"))
            n += 1
            if total >= limit:
                break
    tmp.rename(out)
    print(f"[{name}] {n} documents, {total / 1e6:.1f} MB")


def parquet_docs(path: Path, text_col: str, id_col: str | None, seed: int):
    table = pq.read_table(path, columns=[c for c in (text_col, id_col) if c])
    idx = list(range(table.num_rows))
    random.Random(seed).shuffle(idx)  # random sample, not the first N rows
    texts = table.column(text_col)
    ids = table.column(id_col) if id_col else None
    for i in idx:
        yield (ids[i].as_py() if ids is not None else i), texts[i].as_py()


def get_web() -> None:
    p = fetch_file(f"{HF}/HuggingFaceFW/fineweb-2/resolve/main/data/pol_Latn/test/000_00000.parquet",
                   CACHE / "fineweb2_pol_test.parquet")
    write_docs("web", parquet_docs(p, "text", "id", seed=1))


def get_wiki() -> None:
    p = fetch_file(f"{HF}/wikimedia/wikipedia/resolve/main/20231101.pl/train-00005-of-00006.parquet",
                   CACHE / "wiki_pl_00005.parquet")
    write_docs("wiki", parquet_docs(p, "text", "id", seed=2))


def get_parliament() -> None:
    p = fetch_file(f"{HF}/PiotrSty/sejm-speeches-corpus/resolve/main/data/train.parquet",
                   CACHE / "sejm_speeches_train.parquet")
    cols = pq.read_schema(p).names
    id_col = "id" if "id" in cols else None
    write_docs("parliament", parquet_docs(p, "text", id_col, seed=3))


def get_literature() -> None:
    books = requests.get("https://wolnelektury.pl/api/books/", headers=UA, timeout=60).json()
    random.Random(4).shuffle(books)

    def one(book):
        slug = book["slug"]
        dest = CACHE / "wolnelektury" / f"{slug}.txt"
        try:
            fetch_file(f"https://wolnelektury.pl/media/book/txt/{slug}.txt", dest)
            time.sleep(0.2)
            return slug, dest.read_text(encoding="utf-8", errors="replace")
        except requests.RequestException:
            return slug, ""

    def gen():
        with ThreadPoolExecutor(4) as ex:
            yield from ex.map(one, books)

    write_docs("literature", gen())


class _TextExtractor(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    ex = _TextExtractor()
    ex.feed(html)
    text = "".join(ex.parts)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def get_law() -> None:
    items = []
    for year in (2020, 2021, 2022, 2023, 2024, 2025):
        d = requests.get(f"https://api.sejm.gov.pl/eli/acts/DU/{year}", headers=UA, timeout=60).json()
        items += [it for it in d["items"] if it.get("textHTML")]
    random.Random(5).shuffle(items)

    def one(it):
        eli = it["ELI"]
        dest = CACHE / "eli" / (eli.replace("/", "_") + ".html")
        try:
            fetch_file(f"https://api.sejm.gov.pl/eli/acts/{eli}/text.html", dest)
            time.sleep(0.3)
            return eli, html_to_text(dest.read_text(encoding="utf-8", errors="replace"))
        except requests.RequestException:
            return eli, ""

    def gen():
        with ThreadPoolExecutor(3) as ex:
            yield from ex.map(one, items)

    write_docs("law", gen())


SOURCES = {"web": get_web, "wiki": get_wiki, "literature": get_literature,
           "parliament": get_parliament, "law": get_law}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="*", default=list(SOURCES))
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    for name in args.sources:
        if (RAW / f"{name}.jsonl").exists():
            print(f"[{name}] already present, skipping")
            continue
        SOURCES[name]()


if __name__ == "__main__":
    main()

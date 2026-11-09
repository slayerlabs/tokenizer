#!/usr/bin/env python3
"""Clean, deduplicate and split the raw corpus.

Pipeline (each step logs what it removed, per source, into data/clean/stats.json):
  1. normalize: Unicode NFC, drop control chars, collapse spaces/blank lines
  2. source-specific cleanup (license footers, metadata trailers, hard line wraps, UI strings)
  3. web only: drop boilerplate lines repeated across many documents
  4. quality filters: minimum length, letter ratio, Polish stop-word ratio
  5. exact dedup (normalized text hash)
  6. containment dedup: drop docs whose long paragraphs were mostly seen already
     (catches book parts vs whole books, copied articles, consolidated legal texts)
  7. near-dup dedup: MinHash LSH on word 5-shingles (Jaccard >= 0.8)
  8. document-level split: ~5% held-out per source, by a stable hash of the doc id

The held-out split is decided per document *before* training, so no paragraph of a
held-out document can be in the training data (step 6 runs on the union first).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from datasketch import MinHash, MinHashLSH

RAW = Path("data/raw")
OUT = Path("data/clean")
SOURCES = ["web", "wiki", "literature", "parliament", "law"]
HELDOUT_PCT = 5

# A short list of very frequent Polish function words used by the language check.
STOPWORDS = set("""
i w na z do że się nie to jest o a jak po co ale od za przez dla tak już czy tylko jego
jej ich ten ta te tego tej tym są być był była było może przy oraz także jako który która
które którego mnie mi go je pan pani bardzo też jeszcze gdy kiedy więc bo ze we ku nad pod
""".split())

CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‎‏﻿]")
SEJM_STAGE = re.compile(r"\((?:Oklaski|Dzwonek|Gwar na sali|Poruszenie na sali|Wesołość na sali)[^)]{0,40}\)")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = CTRL.sub("", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines (PDF-like layout of legal acts): a line that does not end a
    sentence, followed by a line starting with a lowercase letter, is a single sentence."""
    return re.sub(r"(?<=[^.:;!?\n])\n(?=[a-ząćęłńóśźż(])", " ", text)


def source_specific(source: str, text: str) -> str:
    if source == "literature":
        k = text.rfind("\n-----\n")  # Wolne Lektury license/editorial footer
        if k > 0:
            text = text[:k]
    elif source == "parliament":
        k = text.find(" Przebieg posiedzenia ")  # metadata trailer appended to each speech
        if k > 0:
            text = text[:k]
        text = SEJM_STAGE.sub("", text)
    elif source == "law":
        text = text.replace("Pokaż całość", "")
        text = unwrap_lines(text)
    return normalize(text)


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def quality_reason(text: str) -> str | None:
    if len(text) < 300:
        return "too_short"
    non_space = sum(1 for c in text if not c.isspace())
    letters = sum(1 for c in text if c.isalpha())
    if letters / max(non_space, 1) < 0.65:
        return "low_letter_ratio"
    # Mojibake: literal JSON escapes or UTF-8 Polish read as GBK (e.g. "wi臋kszy").
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    if "\\u00" in text or "\\u01" in text or cjk / max(letters, 1) > 0.005:
        return "mojibake"
    # Language check: Polish text has either function words or Polish diacritics.
    # Deliberately loose: telegraphic Polish (bios, listings) must survive.
    w = words(text)
    stop_ratio = sum(1 for x in w if x in STOPWORDS) / max(len(w), 1)
    diacritic_ratio = sum(1 for c in text if c in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ") / max(letters, 1)
    if stop_ratio < 0.04 and diacritic_ratio < 0.01:
        return "not_polish"
    return None


def norm_hash(text: str) -> str:
    return hashlib.sha1(" ".join(words(text)).encode()).hexdigest()


def is_heldout(source: str, doc_id: str) -> bool:
    h = int(hashlib.blake2b(f"{source}:{doc_id}".encode(), digest_size=8).hexdigest(), 16)
    return h % 100 < HELDOUT_PCT


def minhash(text: str, num_perm: int = 64) -> MinHash:
    w = words(text[:50_000])
    shingles = {" ".join(w[i:i + 5]).encode() for i in range(max(len(w) - 4, 1))}
    m = MinHash(num_perm=num_perm)
    m.update_batch(list(shingles))
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Counter] = defaultdict(Counter)

    def drop(src: str, reason: str, text: str) -> None:
        stats[src][f"drop_{reason}_docs"] += 1
        stats[src][f"drop_{reason}_bytes"] += len(text.encode())

    docs = []
    for src in SOURCES:
        for line in open(RAW / f"{src}.jsonl", encoding="utf-8"):
            d = json.loads(line)
            stats[src]["raw_docs"] += 1
            stats[src]["raw_bytes"] += len(d["text"].encode())
            d["text"] = source_specific(src, normalize(d["text"]))
            docs.append(d)

    # 3. web boilerplate: short-ish lines that repeat in >= 5 different web documents
    line_df = Counter()
    for d in docs:
        if d["source"] == "web":
            line_df.update({ln for ln in d["text"].split("\n") if len(ln) < 200})
    boiler = {ln for ln, c in line_df.items() if c >= 5}
    for d in docs:
        if d["source"] == "web":
            kept = [ln for ln in d["text"].split("\n") if ln not in boiler]
            removed = len(d["text"]) - len("\n".join(kept))
            stats["web"]["boilerplate_lines_chars"] += removed
            d["text"] = "\n".join(kept)

    # 4. quality filters
    good = []
    for d in docs:
        reason = quality_reason(d["text"])
        if reason:
            drop(d["source"], reason, d["text"])
        else:
            good.append(d)

    # 5. exact dedup
    seen, uniq = set(), []
    for d in good:
        h = norm_hash(d["text"])
        if h in seen:
            drop(d["source"], "exact_dup", d["text"])
        else:
            seen.add(h)
            uniq.append(d)

    # 6. containment dedup, longest documents first so a whole book wins over its parts
    uniq.sort(key=lambda d: -len(d["text"]))
    para_seen, kept = set(), []
    for d in uniq:
        paras = [p for p in d["text"].split("\n") if len(p) >= 120]
        hs = [hashlib.sha1(" ".join(words(p)).encode()).digest() for p in paras]
        if hs and sum(h in para_seen for h in hs) / len(hs) > 0.5:
            drop(d["source"], "contained_dup", d["text"])
            continue
        para_seen.update(hs)
        kept.append(d)

    # 7. near-dup via MinHash LSH
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    final = []
    for i, d in enumerate(kept):
        m = minhash(d["text"])
        if lsh.query(m):
            drop(d["source"], "near_dup", d["text"])
            continue
        lsh.insert(str(i), m)
        final.append(d)

    # 8. split
    final.sort(key=lambda d: (d["source"], d["id"]))
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as ftr, \
            open(OUT / "heldout.jsonl", "w", encoding="utf-8") as fho:
        for d in final:
            split = "heldout" if is_heldout(d["source"], d["id"]) else "train"
            rec = {"source": d["source"], "id": d["id"], "text": d["text"]}
            (fho if split == "heldout" else ftr).write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[d["source"]][f"{split}_docs"] += 1
            stats[d["source"]][f"{split}_bytes"] += len(d["text"].encode())

    report = {s: dict(c) for s, c in stats.items()}
    (OUT / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    tot = Counter()
    for s, c in report.items():
        tot.update(c)
        print(f"{s:<11} raw {c.get('raw_bytes', 0) / 1e6:6.1f} MB -> train {c.get('train_bytes', 0) / 1e6:6.1f} MB"
              f" + heldout {c.get('heldout_bytes', 0) / 1e6:5.1f} MB")
    print(f"{'TOTAL':<11} raw {tot['raw_bytes'] / 1e6:6.1f} MB -> train {tot['train_bytes'] / 1e6:6.1f} MB"
          f" + heldout {tot['heldout_bytes'] / 1e6:5.1f} MB")


if __name__ == "__main__":
    main()

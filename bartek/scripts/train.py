#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer on data/clean/train.jsonl.

Variants (used by the experiments):
  --subset all      the full training split (main tokenizer)
  --subset half-a   random half of the training docs (research question 3.1, stability)
  --subset half-b   the complementary half (research question 3.1, stability)
  --strip-frac 0.1  strip Polish diacritics from this fraction of docs (research question 3.3, diacritics)
  --digits MODE     number pre-tokenization: group3 | single | free (research question 3.4, numbers)

One run trains up to the largest size; smaller tokenizers are prefixes of the merge
list and are written to <out>/v<size>/ (merges.txt + vocab.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpe.pretokenize import count_pretokens  # noqa: E402
from bpe.text import strip_diacritics  # noqa: E402
from bpe.tokenizer import from_training  # noqa: E402
from bpe.trainer import train  # noqa: E402


def doc_bucket(doc: dict, salt: str) -> int:
    key = f"{salt}:{doc['source']}:{doc['id']}".encode()
    return int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16) % 1000


def load_texts(path: Path, subset: str, strip_frac: float) -> list[str]:
    texts = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if subset in ("half-a", "half-b"):
            in_a = doc_bucket(d, "half") < 500
            if in_a != (subset == "half-a"):
                continue
        t = d["text"]
        if strip_frac and doc_bucket(d, "strip") < strip_frac * 1000:
            t = strip_diacritics(t)
        texts.append(t)
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--subset", default="all", choices=["all", "half-a", "half-b"])
    ap.add_argument("--strip-frac", type=float, default=0.0)
    ap.add_argument("--digits", default="group3", choices=["group3", "single", "free"])
    ap.add_argument("--data", default="data/clean/train.jsonl")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    sizes = sorted(int(s) for s in args.sizes.split(","))
    out = Path(args.out)
    if (out / f"v{sizes[-1]}" / "merges.txt").exists():
        print(f"{out} already trained, skipping")
        return

    t0 = time.time()
    texts = load_texts(Path(args.data), args.subset, args.strip_frac)
    n_bytes = sum(len(t.encode()) for t in texts)
    print(f"{len(texts)} docs, {n_bytes / 1e6:.1f} MB; counting pre-tokens...", flush=True)
    wc = count_pretokens(texts, args.digits, workers=args.workers)
    del texts
    print(f"{len(wc)} unique pre-tokens ({time.time() - t0:.0f}s); training to {sizes[-1]}...", flush=True)

    merges = train(wc, sizes[-1])
    tok = from_training(merges, args.digits)
    counts = [m.count for m in merges]
    for v in sizes:
        v_eff = min(v, tok.vocab_size)
        tok.truncated(v_eff).save(out / f"v{v}", counts[: v_eff - 256])
    meta = {"subset": args.subset, "strip_frac": args.strip_frac, "digits": args.digits,
            "train_docs_bytes": n_bytes, "unique_pretokens": len(wc), "merges": len(merges),
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

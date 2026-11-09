#!/usr/bin/env python3
"""Export trained tokenizers to the HuggingFace `tokenizer.json` format.

Writes <run>/v<size>/tokenizer.json next to merges.txt / vocab.json. The file loads
with `tokenizers.Tokenizer.from_file` or `transformers.PreTrainedTokenizerFast`.

Mapping to the HF pipeline:
- pre-tokenizer: Split with our regex (Isolated) followed by ByteLevel without its own
  regex, so the chunk boundaries are exactly ours;
- model: BPE over GPT-2 byte-to-unicode strings, merges in training (rank) order;
- decoder: ByteLevel.
No normalizer: our encoder does not normalize either (NFC is applied during cleaning).

After writing, each file is verified: HF ids must equal our ids on held-out documents.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bpe.evaluation import load_jsonl, size_dirs  # noqa: E402
from bpe.pretokenize import DIGIT_MODES, _TEMPLATE  # noqa: E402
from bpe.tokenizer import Tokenizer  # noqa: E402

VERIFY_DOCS = 300


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable unicode char mapping used by HF ByteLevel."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def to_hf(tok: Tokenizer) -> dict:
    b2u = bytes_to_unicode()

    def s(i: int) -> str:
        return "".join(b2u[b] for b in tok.vocab[i])

    vocab = {s(i): i for i in range(tok.vocab_size)}
    if len(vocab) != tok.vocab_size:
        raise ValueError("duplicate byte sequences in vocab; cannot express in HF format")
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": _TEMPLATE.format(digits=DIGIT_MODES[tok.digits])},
                 "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
            ],
        },
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
            "vocab": vocab,
            "merges": [[s(a), s(b)] for a, b in tok.merges],
        },
    }


def verify(path: Path, tok: Tokenizer, texts: list[str]) -> None:
    from tokenizers import Tokenizer as HFTokenizer

    hf = HFTokenizer.from_file(str(path))
    for t in texts:
        ours = tok.encode(t)
        theirs = hf.encode(t).ids
        if ours != theirs:
            raise AssertionError(f"{path}: HF ids differ from ours on: {t[:80]!r}")
        if hf.decode(theirs) != t:
            raise AssertionError(f"{path}: HF decode is not a round-trip on: {t[:80]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["artifacts/main"])
    args = ap.parse_args()
    texts = [d["text"] for d in load_jsonl()[:VERIFY_DOCS]]
    for run in args.runs:
        for v, d in size_dirs(Path(run)):
            tok = Tokenizer.load(d)
            out = d / "tokenizer.json"
            out.write_text(json.dumps(to_hf(tok), ensure_ascii=False), encoding="utf-8")
            verify(out, tok, texts)
            print(f"{out}: ok ({len(texts)} held-out docs identical)", flush=True)


if __name__ == "__main__":
    main()

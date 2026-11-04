"""Trening tokenizera na data/train.jsonl.

Użycie: python train_tokenizer.py [vocab_size] [plik_wyjściowy]
Domyślnie 32000 tokenów (tyle co Mistral i APT4 z Bielika v3), wynik
w artifacts/pl_bpe_32k.json oraz czytelny słownik obok.
"""

import json
import sys
import time
from pathlib import Path

from bpe import train

HERE = Path(__file__).resolve().parent


def texts():
    with open(HERE / "data" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def main():
    vocab_size = int(sys.argv[1]) if len(sys.argv) > 1 else 32000
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "artifacts" / "pl_bpe_32k.json"
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()
    # vocab_size liczy też token specjalny, więc BPE uczy się o jeden mniej
    tok = train(texts(), vocab_size - 1, log=lambda m: print(m, flush=True))
    tok.save(out)
    tok.save_vocab_readable(out.with_suffix(".vocab.txt"))
    print(f"gotowe: {out}, {tok.vocab_size} tokenów, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

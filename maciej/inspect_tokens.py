"""Jakościowy podgląd: co tokenizer robi z fleksją i z przypadkami ze slajdu 6.

Fertility mówi, ILE tokenów. To mówi, JAKICH — i czy granice mają sens.

    uv run --with regex --with tiktoken python inspect_tokens.py
"""

from __future__ import annotations

import argparse

from evaluate import disp
from tokenizer import BPETokenizer

PARADYGMAT = ["dom", "domu", "domowi", "domem", "domach", "domowy", "domowego",
              "domownikami"]
SLAJD6 = ["don't stop believing", "Kupiłem 3,5 kg jabłek za 12.99 zł",
          "super!!! 🎉", "niesamowicie-fantastycznie-dobre"]
FLEKSJA = ["książka", "książki", "książce", "książkę", "książką", "książkach",
           "pisać", "napisać", "przepisać", "zapisywać", "niedopisany"]


def show(tok, label, items, ref=None, with_space: bool = True) -> None:
    """with_space: pokaż też formę z wiodącą spacją.

    To nie kosmetyka. Pretokenizator dokleja spację do słowa (` ?\\p{L}+`), więc
    "␣dom" i "dom" to DWA RÓŻNE tokeny. W prawdziwym tekście słowo prawie zawsze
    ma przed sobą spację — i tylko ta forma mówi coś o realnej wydajności.
    """
    print(f"\n--- {label} ---")
    for s in items:
        for variant in ([s, " " + s] if with_space else [s]):
            ids = tok.encode_ordinary(variant)
            parts = " | ".join(disp(tok.vocab[i]) for i in ids)
            line = f"  {variant!r:<16} {len(ids):>2}  {parts:<30}"
            if ref is not None:
                rids = ref.encode_ordinary(variant)
                line += f"{len(rids):>3}  " + " | ".join(
                    disp(ref.decode_single_token_bytes(i)) for i in rids)
            print(line)
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizers/hplt-pl-32k-pl.json")
    ap.add_argument("--compare", action="store_true",
                    help="pokaż obok siebie z cl100k_base")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    ref = None
    if args.compare:
        import tiktoken
        ref = tiktoken.get_encoding("cl100k_base")

    print(f"tokenizer: {tok.meta.get('name')}  (vocab {tok.vocab_size}, "
          f"pretokenizer '{tok.pattern_name}')")
    print(f"{'':<19}{'nasz':<35}{'cl100k':>6}" if ref else "")
    show(tok, "paradygmat: dom (slajd 4)", PARADYGMAT, ref)
    show(tok, "fleksja i derywacja", FLEKSJA, ref)
    show(tok, "przypadki brzegowe ze slajdu 6", SLAJD6, ref, with_space=False)

    print("\n--- round-trip na tym wszystkim ---")
    cases = PARADYGMAT + FLEKSJA + SLAJD6
    cases += [" " + c for c in cases]
    bad = [s for s in cases if tok.decode(tok.encode_ordinary(s)) != s]
    print("  OK, bezstratnie" if not bad else f"  BŁĄD: {bad}")


if __name__ == "__main__":
    main()

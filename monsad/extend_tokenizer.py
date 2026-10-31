#!/usr/bin/env python3
"""
Rozszerza tokenizer Bielika o śląskie litery i częste śląskie cząstki słów.

Nie dokleja tokenów "z boku" (add_tokens), tylko douczа sam model BPE:
  1. dodaje do słownika litery, których tokenizer nie zna (ō, ŏ, ô, õ, ã ...),
  2. uczy się nowych połączeń (merges) na śląskim korpusie i dopisuje je NA KOŃCU
     listy, więc dotychczasowe reguły zachowują pierwszeństwo.

Wymaga tylko: pip install tokenizers
(opcjonalnie: pip install tiktoken - wtedy raport porównuje też z cl100k_base i o200k_base)

Przykład:
    python extend_tokenizer.py --tokenizer-dir bielik-4.5b-mlx \
        --train szl-data-v2/train.jsonl --valid szl-data-v2/valid.jsonl \
        --merges 2000 --out tokenizer-szl

Wynik (katalog --out):
    tokenizer.json, tokenizer_config.json, chat_template.jinja  - gotowy tokenizer
    new_tokens.json  - dla każdego nowego tokenu: z jakich starych tokenów się składa
                       (potrzebne do inicjalizacji embeddingów przed treningiem)
    report.json      - tokeny na słowo przed i po, wynik testów poprawności
"""
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


def read_texts(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh if line.strip()]


def is_byte_token(tok):
    return tok.startswith("<0x") and tok.endswith(">")


def mergeable(a, b):
    """Łączymy tylko zwykłe litery; lewa część może zaczynać się od ▁ (początek słowa)."""
    if is_byte_token(a) or is_byte_token(b) or b.startswith("▁"):
        return False
    core = a[1:] if a.startswith("▁") else a
    return (core == "" or core.isalpha()) and b.isalpha()


def learn_merges(word_tokens, word_freq, n_merges, min_count):
    """Klasyczny BPE startujący od segmentacji istniejącego tokenizera."""
    words = [list(t) for t in word_tokens]
    pair_count = Counter()
    pair_words = defaultdict(set)
    for wi, toks in enumerate(words):
        f = word_freq[wi]
        for a, b in zip(toks, toks[1:]):
            if mergeable(a, b):
                pair_count[(a, b)] += f
                pair_words[(a, b)].add(wi)

    merges = []
    while len(merges) < n_merges and pair_count:
        (a, b), cnt = max(pair_count.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt < min_count:
            break
        new = a + b
        merges.append((a, b, new, cnt))
        for wi in list(pair_words[(a, b)]):
            toks, f = words[wi], word_freq[wi]
            for x, y in zip(toks, toks[1:]):      # zdejmij stare pary tego słowa
                if mergeable(x, y):
                    pair_count[(x, y)] -= f
                    pair_words[(x, y)].discard(wi)
            out, i = [], 0
            while i < len(toks):                  # scal od lewej do prawej
                if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                    out.append(new)
                    i += 2
                else:
                    out.append(toks[i])
                    i += 1
            words[wi] = out
            for x, y in zip(out, out[1:]):        # dodaj nowe pary
                if mergeable(x, y):
                    pair_count[(x, y)] += f
                    pair_words[(x, y)].add(wi)
        pair_count.pop((a, b), None)
        pair_words.pop((a, b), None)
        pair_count = +pair_count                  # usuń zera
    return merges, words


def tokens_per_word(tok, texts):
    n_tok = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    return n_tok / max(1, sum(len(t.split()) for t in texts))


def chars_per_token(tok, texts):
    n_tok = sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False))
    return sum(len(t) for t in texts) / max(1, n_tok)


def openai_baselines(texts):
    """Ten sam tekst w tokenizerach OpenAI - punkt odniesienia. Pomija, jeśli brak tiktoken."""
    try:
        import tiktoken
    except ImportError:
        return {}
    n_words = max(1, sum(len(t.split()) for t in texts))
    n_chars = sum(len(t) for t in texts)
    out = {}
    for name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(name)
        n_tok = sum(len(ids) for ids in enc.encode_ordinary_batch(texts))
        out[name] = {"tokens_per_word": round(n_tok / n_words, 3),
                     "chars_per_token": round(n_chars / max(1, n_tok), 3)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer-dir", required=True, help="katalog z tokenizer.json Bielika")
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--merges", type=int, default=2000, help="ile nowych cząstek słów dodać")
    ap.add_argument("--min-count", type=int, default=20, help="minimalna liczba wystąpień pary")
    ap.add_argument("--letters", default="ãŏōôõÃŎŌÔÕ",
                    help="litery, które mają dostać własny token (domyślnie alfabet ślabikŏrzowy)")
    ap.add_argument("--out", default="tokenizer-szl")
    args = ap.parse_args()

    src = Path(args.tokenizer_dir)
    spec = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    if spec["model"]["type"] != "BPE":
        sys.exit("Ten skrypt obsługuje tylko tokenizery BPE.")
    vocab = spec["model"]["vocab"]
    base_size = len(vocab)
    base = Tokenizer.from_file(str(src / "tokenizer.json"))

    train, valid = read_texts(args.train), read_texts(args.valid)
    word_count = Counter(w for t in train for w in t.split())

    # --- 1. nowe litery ---
    char_count = Counter()
    for w, c in word_count.items():
        for ch in w:
            if ch in args.letters and ch not in vocab:
                char_count[ch] += c
    new_chars = [ch for ch in args.letters if ch not in vocab]   # także te rzadkie, np. wielkie Õ
    new_tokens = []   # (token, [stare id-ki do inicjalizacji], liczba wystąpień)
    decomp = {}
    for ch in new_chars:
        ids = [vocab[f"<0x{b:02X}>"] for b in ch.encode("utf-8")]
        vocab[ch] = len(vocab)
        decomp[ch] = ids
        new_tokens.append((ch, ids, char_count[ch]))

    step = Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))   # tokenizer z samymi literami

    # --- 2. nowe połączenia ---
    uniq = list(word_count)
    freq = [word_count[w] for w in uniq]
    seg = [e.tokens for e in step.encode_batch(uniq, add_special_tokens=False)]
    merges, final_seg = learn_merges(seg, freq, args.merges, args.min_count)

    def ids_of(tok):
        return decomp[tok] if tok in decomp else [vocab[tok]]

    for a, b, new, cnt in merges:
        if new in vocab:      # token już istniał, brakowało tylko reguły łączenia
            spec["model"]["merges"].append([a, b])
            continue
        decomp[new] = ids_of(a) + ids_of(b)
        vocab[new] = len(vocab)
        spec["model"]["merges"].append([a, b])
        new_tokens.append((new, decomp[new], cnt))

    # --- 3. zapis ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    for extra in ("tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)
    (out / "new_tokens.json").write_text(json.dumps({
        "base_vocab_size": base_size,
        "new_vocab_size": len(vocab),
        "tokens": [{"id": vocab[t], "token": t, "init_ids": ids, "count": c} for t, ids, c in new_tokens],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- 4. testy poprawności ---
    new = Tokenizer.from_file(str(out / "tokenizer.json"))
    roundtrip_bad = sum(
        new.decode(e.ids) != base.decode(b.ids)
        for e, b in zip(new.encode_batch(valid, add_special_tokens=False),
                        base.encode_batch(valid, add_special_tokens=False))
    )
    sample = uniq[:20000]
    sim_bad = sum(e.tokens != final_seg[i]
                  for i, e in enumerate(new.encode_batch(sample, add_special_tokens=False)))
    used = Counter(i for e in new.encode_batch(valid, add_special_tokens=False) for i in e.ids if i >= base_size)

    report = {
        "base_vocab_size": base_size,
        "new_vocab_size": len(vocab),
        "new_letters": new_chars,
        "new_merges": len(merges),
        "tokens_per_word_valid": {
            "before": round(tokens_per_word(base, valid), 3),
            "letters_only": round(tokens_per_word(step, valid), 3),
            "after": round(tokens_per_word(new, valid), 3),
        },
        "chars_per_token_valid": {
            "before": round(chars_per_token(base, valid), 3),
            "after": round(chars_per_token(new, valid), 3),
        },
        "openai_baselines_valid": openai_baselines(valid),
        "checks": {
            "valid_paragraphs": len(valid),
            "decode_mismatches": roundtrip_bad,
            "segmentation_mismatches_in_20k_words": sim_bad,
            "new_tokens_used_in_valid": len(used),
        },
        "examples": {w: new.encode(w, add_special_tokens=False).tokens
                     for w in ["ôd", "sōm", "Ślōnska", "gŏdka", "nojlepszy", "miyszkańcōw", "Katowice"]},
        "first_new_tokens": [t for t, _, _ in new_tokens[:40]],
    }
    report["tokens_per_word_valid"]["reduction_percent"] = round(
        100 * (1 - report["tokens_per_word_valid"]["after"] / report["tokens_per_word_valid"]["before"]), 1)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nZapisano do: {out}/")


if __name__ == "__main__":
    main()

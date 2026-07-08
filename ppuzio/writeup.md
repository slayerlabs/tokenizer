# Podsumowanie tokenizatora Byte-Level BPE

## Cel

Byte-level BPE dla polskiego. **Wersje są kumulatywne** — każdy trening trafia do `tokenizers/<name>/`, stare wersje zostają do porównań.

Źródło książki v1: [Wolne Lektury — Ogniem i mieczem](https://wolnelektury.pl/media/book/txt/ogniem-i-mieczem-tom-pierwszy.txt).

Korpus DynaWord: 12 źródeł parquet w `polish-dynaword/data/` (Wikipedia, Wikisource, Wolne Lektury, powieści, wiki*, prawo).

---

## Struktura

```text
ppuzio/
  bpe_skeleton.py          # v1: edukacyjny trening (czysty Python, wolny)
  train_dynaword.py        # produkcyjny trening → tokenizers/<name>/
  compare_fertility.py     # tabela fertility ze wszystkich wersji
  inspect_pretokenization.py
  export_v1.py
  tokenizers/
    registry.json
    fertility_comparison.json
    v1-ogniem-10k/
    dynaword-{10,14,32,64,128}k-*/
```

---

## Trening

```bash
pip install pyarrow tokenizers

# szybki eksperyment (~6M znaków)
python train_dynaword.py --vocab-size 10000 --cap-chars 500000

# standard: 220 MB general + 50 MB legal (~1.25 GB)
python train_dynaword.py --vocab-size 128000 --cap-mb 220 --legal-cap-mb 50

# więcej Wikipedii / Wikisource (500 MB general)
python train_dynaword.py --vocab-size 128000 --cap-mb 500 --legal-cap-mb 50

# cały sensowny korpus general (~4.5 GB) + 50 MB legal per źródło prawne
python train_dynaword.py --vocab-size 128000 --general-full --legal-cap-mb 50

python compare_fertility.py
```

**Źródła w `--general-full`:** cała Wikipedia (1.8 GB), Wikisource (2.0 GB), Wolne Lektury, 1000 noveli CLARIN-PL, Wikinews, Wikibooks, Wikiquote, Wikivoyage, ELTEC. **Legal** (eurlex, parliamentary, dziennik_ustaw) ograniczone do 50 MB/źródło.

Trening: biblioteka **`tokenizers`** (Rust), nie tiktoken. Parquet ułatwia streaming; główny zysk prędkości vs `bpe_skeleton.py` to implementacja BPE (~16 min na jednej książce → ~1–2 min na GB danych).

---

## Wyniki fertility (held-out)

**Split:** ostatnie 10% każdego źródła parquet (max 50 MB/źródło) **nie trafia do treningu**. Eval na zarezerwowanym sufiksie ze wszystkich 12 źródeł (~48k docs, ~47M słów, ~330 MB).

Fertility = tokeny/słowo (niżej lepiej). Metryki z `compare_fertility.py` (domyślnie reserved suffix; stary split: `--legacy-held-out`).

### Krzywa vocab (220 MB / źródło)

| Vocab | Held-out | Ogniem |
|------:|---------:|-------:|
| 10k | 2,07 | 1,95 |
| 14k | 1,90 | 1,83 |
| 32k | 1,70 | 1,63 |
| 64k | 1,57 | 1,51 |
| 128k | 1,51 | 1,41 |

### 128k — wpływ rozmiaru danych (50 MB legal)

| Dane treningowe | Train (≈) | Held-out | Ogniem |
|-----------------|----------:|---------:|-------:|
| 220 MB general / źródło | 1,25 GB | 1,499 | 1,393 |
| 500 MB general | 1,87 GB | 1,500 | 1,401 |
| **full general** | **4,3 GB** | **1,509** | 1,413 |

*Cap „220 MB / 500 MB” = limit **na każde źródło osobno**; trening pomija też sufiks eval (~330 MB). Szczegóły w tabeli na końcu.*

Przy **128k** różnica 220 MB → full to ~0,01 fertility — dane szybko się nasycają.

### Porównanie z baseline'ami

| Tokenizer | Vocab | Held-out |
|-----------|------:|---------:|
| **ppuzio 128k (full general)** | 128k | **1,509** |
| ppuzio 64k (220 MB) | 64k | 1,575 |
| Kuba 14k | 14k | 2,150 |

### Artefakt do oddania

**`tokenizers/dynaword-128k-gfull-l50mb/`** — najlepsza fertility w serii (train + eval z reserved suffix).

Format: Hugging Face `tokenizer.json` (ByteLevel + BPE), ładowanie:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizers/dynaword-128k-gfull-l50mb/tokenizer.json")
```

---

## v1 — lekcja overfitu

`bpe_skeleton.py` na jednej książce, bez pretokenizacji. BPE scala całe frazy (`! — rzekł pan Skrzetuski`). In-domain fertility na Ogniem ~1,42 — klasyczny overfit względem DynaWord. Implementacja O(n×merges) w Pythonie.

## v2+ — DynaWord

- Zróżnicowany korpus zamiast jednej powieści
- `tokenizers` (Rust) zamiast czystego Pythona
- ByteLevel pretokenizacja (jak w eksporcie Kuby — regex Split z writeupu Kuby nie widać w jego JSON)
- `--legal-cap-mb 50` ogranicza dominację prawa (z ~39% do ~3% przy full general)

---

## Wnioski

1. **Overfit na jednej książce** daje dobrą kompresję in-domain, słabą generalizację poza nią.
2. **Vocab ↑** to największa dźwignia (10k→128k: ~2,07→1,51).
3. **Więcej danych** pomaga, ale przy 128k szybko nasyca (220 MB ≈ full).
4. **Mniej prawa w treningu** — sensowne przy skalowaniu danych.
5. **128k + DynaWord** bije Kuba 14k na reserved held-out (1,51 vs 2,15).

---

## Korpus treningowy — co znaczy cap MB?

Dane z `polish-dynaword/data/` (12 plików parquet). Trening czyta każde źródło od początku i przestaje po capie; **sufiks eval (10%, max 50 MB/źródło) jest pomijany**.

**Zasady capów:**
- `--cap-mb N` — max **N MB na źródło general** (Wikipedia, Wikisource, …)
- `--legal-cap-mb 50` — max **50 MB na źródło legal** (eurlex, parliamentary, dziennik_ustaw)
- `--general-full` — całe źródła general bez limitu; legal nadal @ 50 MB
- `*` w tabeli = źródło większe niż cap, ucięte do limitu

| Źródło | Typ | Dostępne | @220 MB | @500 MB | full |
|--------|-----|--------:|--------:|--------:|-----:|
| wikipedia | general | 1,84 GB | 231 MB* | 524 MB* | 1,84 GB |
| wikisource | general | 1,97 GB | 231 MB* | 524 MB* | 1,97 GB |
| wolne_lektury | general | 262 MB | 231 MB* | 262 MB | 262 MB |
| 1000_novels | general | 152 MB | 152 MB | 152 MB | 152 MB |
| wikiquote | general | 82 MB | 82 MB | 82 MB | 82 MB |
| eltec_pol | general | 54 MB | 54 MB | 54 MB | 54 MB |
| wikivoyage | general | 45 MB | 45 MB | 45 MB | 45 MB |
| wikibooks | general | 39 MB | 39 MB | 39 MB | 39 MB |
| wikinews | general | 33 MB | 33 MB | 33 MB | 33 MB |
| eurlex | legal | 5,98 GB | 52 MB* | 52 MB* | 52 MB* |
| parliamentary | legal | 4,49 GB | 52 MB* | 52 MB* | 52 MB* |
| dziennik_ustaw | legal | 1,23 GB | 52 MB* | 52 MB* | 52 MB* |
| **Σ general** | | **4,49 GB** | **1,10 GB** | **1,72 GB** | **4,49 GB** |
| **Σ legal** | | | **157 MB** | **157 MB** | **157 MB** |
| **Σ łącznie** | | | **1,25 GB** | **1,87 GB** | **4,64 GB** |

Małe źródła (wikinews, wikibooks, …) wchodzą w całości przy każdym capie — stąd 220 MB/źródło **nie** daje 12 × 220 MB. Przy full general dominują Wikipedia i Wikisource (~80% korpusu general).

## Otwarte

- Ewaluacja na wspólnej próbce SpeakLeash-clean (jak wykres Slayer)
- Eksperyment z regex pretokenizacją (`inspect_pretokenization.py`)
- `max_token_len` / zakaz merge'y ze `\n` (jak u Patryka)

# Runbook — uruchamianie eksperymentów

Instrukcja operacyjna krok po kroku do odtworzenia treningu i ewaluacji tokenizerów z poziomu `DanielK/`.

---

## 1. Setup środowiska (`venv` + biblioteki)

```bash
cd DanielK
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Szybki test offline (`bpe_basic_2.py --demo` + `evaluate_tokenizer.py` + `matplotlib`):
```bash
make test-demo PYTHON=.venv/bin/python
```

---

## 2. Klucz API Gemini (do modernizacji klasyki w H1)

Wymagany wyłącznie przez `modernize_corpus.py`, jeśli `dane_h1/modern_pl.jsonl` nie jest jeszcze wygenerowany:

```bash
export GEMINI_API_KEY="AIza..."
```

*(Jeśli `dane_h1/modern_pl.jsonl` jest już gotowy, skrypt automatycznie pomija wywołania API.)*

---

## 3. Przygotowanie i weryfikacja danych

- `dane_klasyka/wolne_lektury_pl.jsonl` — klasyka PL (`30 MB`)
- `dane_klasyka/klasyka_test.jsonl` — klasyczny zbiór kontrolny *held-out* (`1.93 MB`, 8 niewidzianych podczas treningu utworów z Wolnych Lektur)
- `dane_h1/raw_klasyka.jsonl` oraz `dane_h1/modern_pl.jsonl` — para 1:1 pod H1 (`po 15 MB`)
- `dane_pl/` — `nkjp_pl.jsonl` (`7.4 MB`), `dynaword_general_pl.jsonl` (`48 MB`), `dynaword_tech_pl.jsonl` (`17 MB`), `wiki_pl_tech.jsonl` (`18 MB`), `yt_cnc.jsonl` (`4.2 MB`)
- `dane_en/` — `wiki_en_tech.jsonl` (`15 MB`)
- `dane_eval/` — zbiór *held-out* (`yt_cnc_test.jsonl` `1.0 MB` + `dynaword_tech_test.jsonl` `1.0 MB`)

Uwspółcześnienie 15 MB klasyki (jeśli nie zostało jeszcze uruchomione):
```bash
make h1-modernize PYTHON=.venv/bin/python
```

Pobranie transkrypcji YouTube (80% train / 20% test):
```bash
.venv/bin/python fetch_yt_transcripts.py \
  --channel "https://www.youtube.com/@<kanal_cnc>/videos" \
  --subs-dir dane_yt_vtt \
  --train-output dane_pl/yt_cnc.jsonl \
  --test-output dane_eval/yt_cnc_test.jsonl
```

---

## 4. Trening własnych tokenizerów (`bpe_basic_2.py`)

Każdy wariant zapisuje `merges.txt`, `vocab.json` i `tokenizer.json` do osobnego katalogu z sufiksem `$(VOCAB_TAG)`:

```bash
# 4k
make train PYTHON=.venv/bin/python VOCAB_SIZE=4096 VOCAB_TAG=4k

# 16k
make train PYTHON=.venv/bin/python VOCAB_SIZE=16384 VOCAB_TAG=16k

# 32k
make train PYTHON=.venv/bin/python VOCAB_SIZE=32768 VOCAB_TAG=32k
```

Ręczne uruchomienie pojedynczego wariantu:
```bash
.venv/bin/python bpe_basic_2.py \
  --input dane_pl dane_en \
  --mode polish,cnc \
  --vocab-size 16384 \
  --output artifacts/moj_eksperyment/tokenizer.json
```

---

## 5. Benchmarki na *held-oucie* i wykresy (`matplotlib`)

```bash
.venv/bin/python evaluate_tokenizer.py \
  --tokenizers \
    artifacts/bpe2_t0_klasyka_polish_16k/tokenizer.json \
    artifacts/bpe2_t1_h1_modern_16k/tokenizer.json \
    artifacts/bpe2_t1_pl_polish_16k/tokenizer.json \
    artifacts/bpe2_t2_pl_cnc_16k/tokenizer.json \
    artifacts/bpe2_t3_pl_en_cnc_16k/tokenizer.json \
  --test-file dane_eval \
  --classic-file dane_klasyka/klasyka_test.jsonl \
  --plots-dir artifacts/plots_16k_heldout
```

Skrypt porównuje modele z `cl100k_base` (`tiktoken`) i zapisuje 5 wykresów (`h1_fertility_compression.png`, `h2_boundary_fragmentation.png`, `h2_cnc_terms_heatmap.png`, `h3_renyi_vocab_usage.png`, `h4_dictionary_density.png`).

---

## 6. Sprzątanie plików testowych

```bash
make clean-test
```

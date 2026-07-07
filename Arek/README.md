# Arek — byte-level BPE tokenizery (polski)

Byte-level BPE trenowany **od zera w czystym Pythonie** (wariant `naive`/`fast` bez bibliotek),
na kilku korpusach i dwóch schematach pre-tokenizacji — do **kontrolowanego, empirycznego**
porównania. Wszystkie: alfabet bazowy 256 bajtów, **zero `<UNK>`, round-trip lossless**.

## Cztery rodziny wariantów

| katalog | korpus | pre-tok | vocab | rola |
|---|---|---|---|---|
| `naive/` | Wolne Lektury (2,71 M zn.) | brak | 512–2048 | dydaktyczny, `O(n·m)` |
| `fast/` | Wolne Lektury (2,71 M zn.) | GPT-2 regex | 512–15000 | produkcyjny (heap) |
| `fast-speakleash/` | **SpeakLeash ~3–4 GB** | GPT-2 regex | 32k, 64k | produkcja / skala |
| `fast-speakleash-gpt4/` | SpeakLeash ~3 GB | **GPT-4 (cl100k) regex** | 32k | ablacja pre-tok |

`naive` vs `fast`: ta sama jakość, `fast` skaluje (regex pre-tok + bucketing + liczniki
inkrementalne + best-pair przez lazy-heap; parytet 1:1 z rdzeniem referencyjnym).

## Metryki (i po co)

- **zn/tok** (chars/token) — kompresja, ↑ lepiej.
- **fertility** (tokeny/słowo) — ↓ lepiej. Standard w ewaluacji multilingual.
- **Rényi efficiency** (α=2,5) — równomierność rozkładu tokenów, 0–1, ↑ lepiej. Informacyjno-teoretyczny
  proxy jakości z „noiseless channel" (Zouhar i in. 2023, ACL; ~0,78 korelacji z BLEU w MT).
- **round-trip lossless** — `decode(encode(x)) == x`.

> **Uwaga metodyczna (kluczowa):** fertility i Rényi są **dystrybucyjne** — mierzą kompresję i rozkład,
> **nie** gramatyczność/morfologię. BPE optymalizuje kompresję, nie spójność lingwistyczną — dlatego
> powstały osobne metryki morfologiczne (MorphScore, MorphAcc). Do jakości morfologicznej trzeba
> **osobnej osi** (trafienie w granice morfemów ≈ MorphScore), nie fertility.

## Tabela 1 — mały korpus (held-out: Mickiewicz „Pan Tadeusz", 445 637 zn.)

| wariant | vocab | zn/tok | fertility | Rényi |
|---|---|---|---|---|
| naive-512 | 512 | 1,76 | 3,718 | 0,768 |
| naive-1024 | 1024 | 2,11 | 3,094 | 0,767 |
| naive-2048 | 2048 | 2,43 | 2,685 | 0,743 |
| fast-512 | 512 | 1,78 | 3,663 | 0,730 |
| fast-1024 | 1024 | 2,13 | 3,074 | 0,662 |
| fast-2048 | 2048 | 2,41 | 2,712 | 0,585 |
| fast-4096 | 4096 | 2,69 | 2,426 | 0,518 |
| fast-8192 | 8192 | 3,00 | 2,177 | 0,460 |
| fast-15000 | 15000 | 3,25 | 2,010 | 0,418 |
| GPT-2 (ref.) | 50257 | 1,73 | 3,782 | 0,391 |

## Tabela 2 — produkcja SpeakLeash (held-out: shard 0005, ROZŁĄCZNY z treningiem)

| wariant | vocab | pre-tok | zn/tok | fertility | Rényi | round-trip |
|---|---|---|---|---|---|---|
| fast-speakleash | 32000 | GPT-2 | 4,03 | 1,765 | 0,451 | ✅ |
| fast-speakleash | 64000 | GPT-2 | 4,37 | **1,630** | 0,412 | ✅ |
| fast-speakleash-gpt4 | 32000 | GPT-4 | 4,04 | 1,763 | **0,473** | ✅ |

> **Nie porównuj Tabeli 1 z Tabelą 2** — inne held-outy → liczby nieporównywalne wprost. To samo
> dotyczy porównań między RÓŻNYMI tokenizerami mierzonymi na różnych held-outach: absolutne liczby
> są nieważne bez **wspólnego** held-outu i wspólnej definicji słowa.

## Co zmierzyliśmy i dlaczego

1. **Vocab ↑ → fertility ↓ (monotonicznie, z saturacją).** 512→15k: zn/tok 1,78→3,25 (Tab. 1).
   Standardowa zależność; przy dużym vocab zyski maleją.
2. **Skala korpusu jest ogromną dźwignią — do saturacji.** Nasz stary `fast` 15k (2,71 M literatury)
   na held-oucie SpeakLeash = **2,388**; `fast-speakleash` 32k = **1,765** (−0,62 tok/słowo). Ale dalej
   **saturuje**: 3 GB → 4,14 GB @64k → 1,632 → 1,630 (szum). Zgodne z „Diminishing Returns of
   Tokenization Training Data" (arXiv 2502.20273). **Korpus > algorytm**, ale ma sufit.
3. **Pre-tok GPT-4 vs GPT-2 @32k:** fertility bez zmian (1,765 → 1,763), Rényi ↑ (0,451 → **0,473**).
   Lepsze traktowanie liczb/granic kategorii → **zdrowszy rozkład**, nie krótszy zapis.
4. **fertility/Rényi są ślepe na morfologię** (nasz pomiar + literatura): formy fleksyjne jednego lematu
   wchodzą jako **osobne atomy**, wspólny morfem (`-ość`) nie ma wspólnego tokenu — a metryki tego nie
   widzą. Do oceny „czy przeżył paradygmat fleksyjny" trzeba osi morfologicznej (MorphScore-like).
5. **BPE vs Unigram (SentencePiece, matched vocab):** u nas BPE dał **nieco niższą fertility** niż Unigram
   — zgodne z „BPE jest zachłannie kompresyjny" (Bostrom & Durrett 2020). Przewaga Unigrama leży w
   downstream/morfologii, nie w surowej kompresji; literatura o samej fertility jest **mieszana**.
6. **Kierunek morfem-aware** (wstrzyknięcie granic morfemów do BPE): podnosi trafienie w morfemy, ale
   **zysk maleje przy dużym vocab**, a dopasowanie morfologiczne **słabo koreluje z wydajnością modelu**
   (arXiv 2507.06378). Morfem-**pre**-segmentacja wspiera BPE downstream (MorphBPE 2502.00894) — kierunek otwarty.

## Uczciwe zastrzeżenia

- **Held-out i definicja słowa są nasze** (`[^\W\d_]+`, ta sama dystrybucja). Liczby pokazują
  **magnitudy i tendencje**; czyste 1:1 z innym tokenizerem wymaga **wspólnego** held-outu/protokołu.
- **α w Rényim = konwencja** (nie strojona pod wynik — to byłby Goodhart).
- Metryki tu są **dystrybucyjne**; jakość morfologiczna i downstream to osobne osie (nie mierzone tu w pełni).

## Format i wczytywanie

Każdy plik to JSON w formacie zbliżonym do HuggingFace: blok `model` z `vocab`
(`bytes_to_unicode` → id) i `merges` (lista `"lewy prawy"` w kolejności uczenia).

```python
import json
m = json.load(open("fast/2048.json", encoding="utf-8"))["model"]
vocab = m["vocab"]                       # repr -> id
merges = {}
for line in m["merges"]:                 # kolejność = rank
    l, r = line.split(" ")
    merges[(vocab[l], vocab[r])] = vocab[l + r]
```

## Literatura

- Zouhar i in. 2023 — *Tokenization and the Noiseless Channel*, ACL (Rényi efficiency).
- Bostrom & Durrett 2020 — *Byte Pair Encoding is Suboptimal for Language Model Pretraining* (BPE vs Unigram).
- *Evaluating Morphological Alignment of Tokenizers in 70 Languages*, arXiv 2507.06378 (MorphScore; alignment ≠ downstream).
- MorphBPE, arXiv 2502.00894 · *Diminishing Returns of Tokenization Training Data*, arXiv 2502.20273.
- Petrov i in. 2023 — *Language Model Tokenizers Introduce Unfairness Between Languages*, NeurIPS.

## Autor
Arek — Kurs Tokenizer, Slayer Labs.

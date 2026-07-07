# Arek — byte-level BPE tokenizery (polski)

Byte-level BPE trenowany **od zera w czystym Pythonie** (bez bibliotek). Wszystkie: alfabet bazowy 256 bajtów,
**zero `<UNK>`, round-trip lossless**.

**Organizacja repo = katalog per rozmiar słownika (`vocab`)** — bo *vocab to główna dźwignia fertility*
(zmierzone, patrz „Krzywa vocab"). W każdym katalogu warianty w nazwie: **`<korpus>-<pretok>.json`**.

> **Najlepszy fertility:** [`128000/speakleash-gpt4.json`](128000/speakleash-gpt4.json) — **1,524 tok/słowo**
> (Rényi 0,398, GPT-4 pre-tok + pełne cyfry). To Pareto-wybór — tradeoff niżej.

## Struktura (po vocabie)

| katalog | warianty (pliki) | korpus · pre-tok |
|---|---|---|
| `512/` `1024/` `2048/` | `lektury-naive`, `lektury-fast` | Wolne Lektury (2,71 M zn.) · brak / GPT-2 |
| `4096/` `8192/` `15000/` | `lektury-fast` | Wolne Lektury · GPT-2 |
| `32000/` `64000/` | `speakleash-gpt2`, `speakleash-gpt4` | SpeakLeash ~4,14 GB · GPT-2 / GPT-4+cyfry |
| `128000/` | `speakleash-gpt4` | SpeakLeash 4,14 GB · GPT-4 + pełne cyfry |

Warianty: **`naive`** = dydaktyczny `O(n·m)`, bez pre-tok · **`fast`** / **`speakleash-gpt2`** = pre-tok GPT-2
(bucketing + lazy-heap, parytet 1:1 z rdzeniem) · **`speakleash-gpt4`** = pre-tok GPT-4 (cl100k regex)
**+ pełne runy cyfr `\p{N}+`** — spójnie dla 32k/64k/128k.

## Metryki (i po co)

- **zn/tok** (chars/token) — kompresja, ↑ lepiej.
- **fertility** (tokeny/słowo) — ↓ lepiej. Standard w ewaluacji multilingual.
- **Rényi efficiency** (α=2,5) — równomierność rozkładu tokenów, 0–1, ↑ lepiej (Zouhar i in. 2023, ACL;
  ~0,78 korelacji z BLEU w MT).
- **round-trip lossless** — `decode(encode(x)) == x`.

> **Uwaga metodyczna:** fertility i Rényi są **dystrybucyjne** — mierzą kompresję/rozkład, **nie**
> gramatyczność. Patrz „Fleksja pod mikroskopem".

## ⭐ Krzywa vocab → fertility (główny wynik; held-out SpeakLeash shard 0005, def. słowa `[^\W\d_]+`)

| vocab | wariant | zn/tok | fertility | Rényi | RT |
|---|---|---|---|---|---|
| 32000 | speakleash-gpt2 | 4,03 | 1,765 | 0,451 | ✅ |
| 32000 | speakleash-gpt4 | 4,05 | 1,756 | **0,472** | ✅ |
| 64000 | speakleash-gpt2 | 4,37 | 1,630 | 0,412 | ✅ |
| 64000 | speakleash-gpt4 | 4,39 | 1,619 | 0,431 | ✅ |
| **128000** | **speakleash-gpt4** | **4,67** | **1,524** | 0,398 | ✅ |

**Vocab dominuje.** 32k→64k→128k (gpt4): fertility **1,756 → 1,619 → 1,524**. Dla kontrastu: zmiana pre-toka
(GPT-2→GPT-4) daje ~**−0,01**, a **6× więcej danych 0,00** (saturacja — patrz „Skala danych"). Ogon 128k
jest **zdrowy** (`best_c≈500` na ostatnich merge'ach → 4,14 GB wystarczyło, bez glitch-tokenów).

**Tradeoff (nie darmowe):** większy vocab → **Rényi ↓** (0,451→0,398) i **tabela embeddingów rośnie** (128k =
vocab Llama-3). Wybór vocab = kompromis fertility ↔ Rényi ↔ rozmiar modelu.

## Tabela — mały korpus (held-out: Mickiewicz „Pan Tadeusz", 445 637 zn.)

| plik | vocab | zn/tok | fertility | Rényi |
|---|---|---|---|---|
| `512/lektury-naive` | 512 | 1,76 | 3,718 | 0,768 |
| `2048/lektury-naive` | 2048 | 2,43 | 2,685 | 0,743 |
| `2048/lektury-fast` | 2048 | 2,41 | 2,712 | 0,585 |
| `8192/lektury-fast` | 8192 | 3,00 | 2,177 | 0,460 |
| `15000/lektury-fast` | 15000 | 3,25 | 2,010 | 0,418 |
| GPT-2 (ref.) | 50257 | 1,73 | 3,782 | 0,391 |

> **Nie porównuj tej tabeli z „Krzywą vocab"** — inne held-outy (Pan Tadeusz vs SpeakLeash) → liczby
> nieporównywalne wprost. Absolutne liczby są nieważne bez **wspólnego** held-outu i definicji słowa.

## Co zmierzyliśmy i dlaczego

1. **Vocab ↑ → fertility ↓ — NAJWIĘKSZA dźwignia.** 32k→128k (gpt4): 1,756→1,524 (−0,23). Największy z efektów.
2. **Pre-tok GPT-4 + pełne cyfry — mały, ale realny zysk.** @32k neutralny na fert (Rényi↑); **@64k wygrywa
   obie osie** (1,630→1,619, Rényi 0,412→0,431). Najlepszy pre-tok, ~10× słabszy efekt niż vocab.
3. **Skala danych: nasycona JUŻ przy ~3 GB (przy stałym vocab).** dynaword 2,84→17 GB (6×): fertility płaski.
   Więcej danych **nie** obniża fertility — ogranicza je **rozmiar słownika**. *(„Diminishing Returns…" arXiv
   2502.20273 mierzy jakość przy ZMIENNYM vocab, EN ~150/RU ~200 GB; nasz **stały** vocab nasyca się dużo
   wcześniej.)* Dane stają się lewarem **dopiero razem z większym vocab** (napełnić ogon). Patrz „Skala danych".
4. **fertility/Rényi są ślepe na morfologię** — patrz „Fleksja pod mikroskopem".
5. **BPE vs Unigram (matched vocab):** BPE dał nieco niższą fertility (zachłannie kompresyjny — Bostrom &
   Durrett 2020); przewaga Unigrama w downstream/morfologii, nie w kompresji.

## Skala danych i dystrybucja — eksperyment dynaword (zmierzony)

Retrain 64k (pipeline GPT-4 + pełne cyfry) na [`SlayerLab/polish-dynaword`](https://huggingface.co/datasets/SlayerLab/polish-dynaword)
(human-text, 12 źródeł, CC-BY-SA). Zmienne rozłożone; pomiar na OBU held-outach.

| trening (64k) | fert @SpeakLeash-0005 | fert @dyna-held | gap |
|---|---|---|---|
| SpeakLeash 4,14 GB | **1,619** | 1,829 | 0,210 |
| dynaword 2,84 GB | 1,758 | 1,726 | — |
| dynaword 17,0 GB | 1,757 | 1,733 | 0,024 |

- **Rozmiar: nasycony do ~3 GB** (dynaword 2,84→17 GB = fertility płaski).
- **Dystrybucja dominuje:** SpeakLeash-trained (4,14 GB) bije dynaword-trained (17 GB) na SpeakLeash — mimo 4× mniej danych.
- **Robustność:** SpeakLeash-trained = home-turf (gap 0,21); dynaword-trained = robust (gap 0,024). Publikujemy
  wariant SpeakLeash jako najlepszy **na benchmarku**; dynaword bywa lepszy jako **ogólny**.

## Fleksja pod mikroskopem — tokeny vs metryka (dowód punktu 4)

Realny podział na tokeny `64000/speakleash-gpt2` (zweryfikowany 1:1 z encoderem; słowa w izolacji):

| słowo | tokeny | `-ość` / rdzeń |
|---|---|---|
| odpowiedzialność | `odpowie · dzialność` | rdzeń bez wspólnego tokenu |
| odpowiedzialności | `odpowie · dzia · lności` | inny podział niż mianownik |
| wolność | `wo · lność` | `-ość` → `lność` |
| radość | `ra · dość` | `-ość` → `dość` |

Ten sam morfem `-ość` **nie ma jednego tokenu**. Zmierzone (UniMorph PL, N=86 k; recall-only,
within-tokenizer): tokenizer trafia w granicę `stem|końcówka` tylko ~**19%** przypadków. **Metryka (fertility
~1,5–1,6) mówi „dobrze skompresowane", a paradygmat fleksyjny jest rozbity** — fertility/Rényi tego nie widzą.

## Uczciwe zastrzeżenia

- **Held-out i definicja słowa są nasze** (`[^\W\d_]+`). Liczby pokazują **magnitudy i tendencje**; czyste 1:1
  z innym tokenizerem wymaga **wspólnego** held-outu/protokołu.
- **α w Rényim = konwencja** (nie strojona pod wynik).
- „Najlepszy" znaczy **na benchmarku SpeakLeash** — na innej dystrybucji ranking się zmienia (patrz 2×2).
  Jakość **modelu** (perplexity/downstream) to osobna oś, **tu niemierzona** (fertility ≠ downstream).

## Format i wczytywanie

Każdy plik to JSON w formacie zbliżonym do HuggingFace: blok `model` z `vocab` (`bytes_to_unicode` → id) i
`merges` (lista `"lewy prawy"` w kolejności uczenia). **Pre-tok (zdeterminowany nazwą pliku):** `lektury-fast`
i `speakleash-gpt2` → regex GPT-2; **każdy `speakleash-gpt4`** (32k/64k/128k) → regex GPT-4 (cl100k) **+ pełne
runy cyfr `\p{N}+`**; `lektury-naive` → bez pre-tok.

```python
import json
m = json.load(open("128000/speakleash-gpt4.json", encoding="utf-8"))["model"]
vocab = m["vocab"]                       # repr -> id
merges = {}
for line in m["merges"]:                 # kolejność = rank
    l, r = line.split(" ")
    merges[(vocab[l], vocab[r])] = vocab[l + r]
```

## Literatura

- Zouhar i in. 2023 — *Tokenization and the Noiseless Channel*, ACL (Rényi efficiency).
- Bostrom & Durrett 2020 — *BPE is Suboptimal for Language Model Pretraining* (BPE vs Unigram).
- *Evaluating Morphological Alignment of Tokenizers in 70 Languages*, arXiv 2507.06378 (MorphScore; alignment ≠ downstream).
- *Diminishing Returns of Tokenization Training Data*, arXiv 2502.20273 · MorphBPE, arXiv 2502.00894.
- Petrov i in. 2023 — *Language Model Tokenizers Introduce Unfairness Between Languages*, NeurIPS.
- Korpus skali: [`SlayerLab/polish-dynaword`](https://huggingface.co/datasets/SlayerLab/polish-dynaword) (CC-BY-SA-4.0).

## Autor
Arek — Kurs Tokenizer, Slayer Labs.

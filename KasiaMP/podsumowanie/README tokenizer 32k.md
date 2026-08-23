# Kasia — byte-level BPE 32k (polski)

Byte-level BPE trenowany na **czyszczonym korpusie SpeakLeash**, biblioteka HuggingFace `tokenizers`
(`BpeTrainer`). Alfabet bazowy 256 bajtów na ID 0–255, **zero `<UNK>`, round-trip lossless**.

Artefakt jest jeden i produkcyjny: `tokenizer.json` — **32 768 slotów** (32 568 tokenów BPE + 200 tokenów
specjalnych). Rozmiar to wielokrotność 128 (wyrównanie pod kernele GPU). Docelowe zastosowanie: pretrening
polskiego modelu ~125 M na pojedynczym GPU.

## Parametry

| pole | wartość |
|---|---|
| typ | byte-level BPE (HF `tokenizers`, format `tokenizer.json` v1.0) |
| vocab (łącznie) | **32 768** |
| tokeny BPE | 32 568 |
| merge'y | 32 312 |
| alfabet bazowy | 256 bajtów na ID **0–255** |
| tokeny specjalne | 200, na ID **32 568–32 767** |
| normalizer | `null` (NFC wykonany **w pipelinie korpusu**, nie w tokenizerze) |
| pre-tokenizer | `Split` (regex cl100k) → `ByteLevel(add_prefix_space=false)` |
| decoder | `ByteLevel` |
| `byte_fallback` / `unk_token` | `false` / `None` — niepotrzebne przy bazie bajtowej |
| `dropout` | `None` |

**Alfabet bajtowy na ID 0–255 jest wymuszony celowo.** Podanie `special_tokens` do `BpeTrainer` przesuwa bajty
poza ten zakres; tokeny specjalne są więc wstrzykiwane **po treningu**, a `sanity_check` zawiera asercję
pilnującą tego niezmiennika.

## Pre-tokenizacja

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
```

Regex w stylu **cl100k (GPT-4)**, z jedną świadomą decyzją: **`\p{N}{1,3}`**, czyli maksymalnie 3 cyfry na
pre-token. Konsekwencje zmierzone na artefakcie:

- **0** tokenów z ≥4 cyframi; 412 czystych ciągów cyfr, wszystkie długości 1–3 → `1234567` → `123|45|6|7`.
  Bez tego limitu BPE uczy się arbitralnych tokenów typu `2020`, a arytmetyka staje się loterią.
- Skróty/apostrofy **case-insensitive** (`(?i:...)`) — brak niespójności przy tekście pisanym kapitalikami.
- Spacja wiodąca doklejona do słowa, **nigdy** jako samodzielny token między słowami.
- Whitespace jest merge'owalny — w vocabie są tokeny wcięć o długości **1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64**
  spacji, więc kod w korpusie nie eksploduje na sekwencje pojedynczych spacji.
- Merge nie przekracza newline'a w drugą stronę: `\n\n` jest jednym tokenem.

## Tokeny specjalne (200, ID 32 568–32 767)

14 nazwanych + 186 slotów rezerwowych (`<|reserved_0|>` … `<|reserved_185|>`):

| grupa | tokeny |
|---|---|
| dokument | `<\|endoftext\|>`, `<\|begin_of_text\|>` |
| czat | `<\|im_start\|>`, `<\|im_end\|>`, `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` |
| FIM | `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` |
| narzędzia | `<\|tool_call\|>`, `<\|tool_result\|>` |
| techniczne | `<\|pad\|>`, `<\|unk\|>` |

Blok jest zarezerwowany **z góry**, żeby nie robić resize'u macierzy embeddingów w połowie życia modelu.
Wszystkie mają `normalized: false` i `special: true` — encoder nie wyprodukuje ich z surowego tekstu
użytkownika.

## Korpus

- **Źródło:** SpeakLeash, 23 zbiory, **~2,40 GB** tekstu po czyszczeniu.
- **Pipeline:** `pl_clean.py` (`pl-clean-v3`) — deterministyczny i wersjonowany, przeznaczony do odpalenia
  **identycznie** na danych treningowych modelu. Kroki: normalizacja NFC → usuwanie znaków kontrolnych →
  filtr językowy per dokument → line-level boilerplate filtering (z progiem minimalnej długości linii, żeby
  nie wycinać krótkich nagłówków typu „Wstęp") → detektor uszkodzeń kodowania → deduplikacja krzyżowa
  względem zbioru held-out.
- **Held-out:** ~9,5 M znaków, 2 125 dokumentów — kilkadziesiąt utworów literackich + Konstytucja RP.
  Pięć zbiorów SpeakLeash zostało **wykluczonych z treningu**, żeby wyeliminować kontaminację.

## Wyniki

- **Round-trip lossless** na wszystkich 2 125 dokumentach held-out.
- **Wykorzystanie vocabu: 98,4%** na held-oucie (cienki ogon martwych tokenów → mniejsze ryzyko glitch tokens).
- Rozkład długości tokenów (bajty): moda w przedziale 4–6 B, ogon do 20 B dla tekstu; powyżej tego wyłącznie
  tokeny wcięć (24/32/64 B).

Porównanie kontrolne wobec klasowego byte-level BPE 6 756 (bez pre-tokenizacji, korpus ~9,4 M znaków:
2 książki + fragment Konstytucji), na wspólnym held-oucie 725 słów / 4 978 znaków w 6 rejestrach
(proza, potoczny, publicystyka, prawniczy, naukowy, liczby+kod):

| tokenizer | vocab | tok/słowo | tok/1000 zn. | zn./tok |
|---|---:|---:|---:|---:|
| **ten (32k)** | 32 768 | **1,665** | **242,5** | **4,12** |
| klasowy BPE 6,7k | 6 756 | 2,338 | 340,5 | 2,94 |

Różnica: **−28,8% tokenów** na tym samym tekście. Rozstrzał per rejestr: od −13% (proza) do −40%
(tekst naukowy) — mały tokenizer wygląda przyzwoicie tylko na domenie zbliżonej do swojego korpusu.

## Uczciwe zastrzeżenia

- Tabela powyżej to **mały held-out ad hoc** (725 słów) i definicja słowa `\S+`. Pokazuje magnitudę
  i tendencję, nie jest pomiarem publikowalnym — porównanie 1:1 z innym tokenizerem wymaga wspólnego
  held-outu i wspólnej definicji słowa.
- Fertility to metryka **dystrybucyjna**: mierzy kompresję, nie gramatyczność ani zgodność z granicami
  morfemów. Dla języka fleksyjnego dobrymi tokenami są często całe częste formy fleksyjne — morfologicznie
  „brzydkie", statystycznie opłacalne.
- Jakość **modelu** (perplexity, downstream) to osobna oś i nie jest tu mierzona.
- `normalizer: null` w pliku **nie znaczy „bez normalizacji"** — NFC jest w pipelinie korpusu. Jeżeli dane
  treningowe modelu pójdą inną ścieżką preprocessingu, rozjazd wyprodukuje glitch tokens.

## Format i wczytywanie

```python
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("Zażółć gęślą jaźń.").ids
assert tok.decode(ids) == "Zażółć gęślą jaźń."
```

Plik jest w natywnym formacie HF, więc działa też przez `PreTrainedTokenizerFast`:

```python
from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

## Autor

Kasia — Kurs Tokenizer, Slayer Labs.

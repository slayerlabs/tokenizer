# Tokenizer BPE - budowa

Celem niniejszego zadania było zbudowanie własnego tokenizera typu BPE (**Byte-pair encoding**) jako podstawy do zrozumienia działania i możliwości budowy własnych modeli językowych od podstaw.

## Opis

W skład czynności wchodziło:
- Wytrenowanie słowników metodą BPE różnej wielkości na różnych korpusach oraz porównanie liczby tokenów i **fertility** (tokeny / słowo).
- Wyniki w postaci słowników (dołączone w formacie HuggingFace `tokenizer.json`).
- Porównanie trzech wariantów tokenizacji (BPE, BPE + pretokenizacja, naiwny) — opisane niżej.
- Test kodowania na tekstach spoza korpusu (różne pisma/języki).

Na tym etapie nie przeprowadziłem żadnych badań — wyniki służyły głównie poznaniu podstaw i możliwości dostępnego hardware, żeby ocenić wykonalność przyszłych testów z użyciem moich zasobów.

## Warianty tokenizera

**1. Naiwny tokenizer** — Najbardziej prymitywna metoda, bez uczenia - słownik nie występuje. Podział na tokeny przez prosty regex:

```
\w+|[^\w\s]
```

**2. BPE (byte-level)** — wariant bazowy. Trening i kodowanie na surowym strumieniu bajtów
UTF-8 (alfabet bazowy 256 bajtów).

**3. BPE + pretokenizacja (styl GPT-2)** — jak wyżej, ale tekst jest najpierw dzielony
regexem na pretokeny, a merge'y działają **tylko wewnątrz pretokenu** (nigdy przez granicę
słowa/spacji). Regex (klasy `\p{L}` / `\p{N}` jak w GPT-2):

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

dzieli na: końcówki typu `'s`, słowo z wiodącą spacją, liczbę, ciąg interpunkcji, białe znaki.
Pretokenizacja również znacząco przyspiesza budowanie słownika.

## Bezstratność jako kryterium poprawności wykonanego zadania

BPE jest metodą bezstratną. Kryterium akceptacji definiującym prawidłowe działanie tokenizera
jest zgodność danych wejściowych z wyjściowymi (round-trip `decode(encode(x)) == x`). Warunek
spełniony dla wszystkich wytrenowanych słowników (encode/decode na całych korpusach).

## Dane treningowe

### Korpusy

| Korpus | Źródło | Znaki | Bajty (UTF-8) | Słowa | Rozmiar |
| --- | --- | ---: | ---: | ---: | ---: |
| Stefan Żeromski - Syzyfowe prace | https://wolnelektury.pl/media/book/txt/syzyfowe-prace.txt | 423 491 | 457 396 | 63 437 | 3 022 linie |
| Slayer Dynaword / Wikinews | `SlayerLab/polish-dynaword/data/wikinews` | 32 549 032 | 34 200 000 | 4 245 681 | 24 386 artykułów |
| Slayer Dynaword / 100 novels | `SlayerLab/polish-dynaword/1000_novels` | 151 347 756 | 164 289 857 | 23 007 564 | 164 MB |

- Rozmiary słowników: **5k** i **16k** dla korpusów Syzyfowe prace i Wikinews; **1 500** dla 100 novels.
- Metryka: **fertility = tokeny / słowa** (słowa = podział po białych znakach).
- Dynaword 100 novels użyty wyłącznie w wariancie z pretokenizacją.

### Teksty do testów liczby tokenów

Krótkie teksty spoza korpusów treningowych — różne pisma/języki do testu tokenizacji na materiale nieznanym słownikom. 

| Tekst | Język / pismo | Znaki | Źródło |
| --- | --- | ---: | --- |
| Lorem ipsum | łacina (placeholder) | 1 015 | <https://www.lipsum.com/> |
| Polski (Chopin) | polski / łaciński | 1 008 | <https://pl.wikipedia.org/w/index.php?title=Fryderyk_Chopin&oldid=80062101> |
| Japoński (日本語) | japoński / kanji+kana | 316 | <https://ja.wikipedia.org/w/index.php?title=%E6%97%A5%E6%9C%AC%E8%AA%9E&oldid=109750467> |
| Rosyjski (cyrylica) | rosyjski / cyrylica | 3 038 | <https://ru.wikipedia.org/w/index.php?title=%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9_%D1%8F%D0%B7%D1%8B%D0%BA&oldid=153722526> |

## Wyniki

### Słowniki

Byte-level BPE (osobno per korpus):

| Korpus | Vocab size | Tokeny | Fertility |
| --- | --- | ---: | ---: |
| Syzyfowe prace | 5k | 117 709 | 1,8555 |
| Syzyfowe prace | 16k | 85 668 | 1,3504 |
| Wikinews | 5k | 9 516 504 | 2,2415 |
| Wikinews | 16k | 7 249 669 | 1,7075 |

Byte-level BPE  z pretokenizacją (styl GPT-2):

| Korpus | Vocab size | Tokeny | Fertility |
| --- | --- | ---: | ---: |
| Syzyfowe prace | 5k | 120 118 | 1,8935 |
| Wikinews | 16k | 7 969 087 | 1,8770 |
| 100 novels | 1 500 | 58 178 100 | 2,5287 |

### Testy na tekstach spoza korpusu

Liczba tokenów:

| Tekst | Znaki | naiwny | syz 5k | syz 16k | wiki 5k | wiki 16k | syz-pre 5k | wiki-pre 16k | 100-nov 1 500 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lorem ipsum | 1 015 | 171 | 549 | 505 | 500 | 443 | 560 | 446 | 628 |
| Polski (Chopin) | 1 008 | 170 | 414 | 364 | 328 | 259 | 413 | 255 | 470 |
| Japoński (日本語) | 316 | 44 | 928 | 928 | 924 | 787 | 928 | 788 | 928 |
| Rosyjski (cyrylica) | 3 038 | 466 | 5 488 | 5 467 | 4 533 | 3 461 | 5 534 | 3 489 | 5 539 |

Skróty: **syz** = Syzyfowe prace, **wiki** = Wikinews, **100-nov** = 100 novels; sufiks
**-pre** = wariant z pretokenizacją (100-nov trenowany wyłącznie w tym wariancie).

#### Nieznane słowa dla naiwnego tokenizera

Naiwny tokenizer sam nie ma słownika. Na potrzeby tego testu nadajemy mu **słownik
odniesienia** = wszystkie słowa, jakie wyprodukował na danym korpusie, i liczymy, ile słów
tekstu testowego w nim nie występuje.

Liczba nieznanych słów (w nawiasie rozmiar słownika odniesienia — unikalne słowa):

| Tekst | Syzyfowe prace (19 269) | Wikinews (266 782) | 100 novels (647 138) |
| --- | ---: | ---: | ---: |
| Lorem ipsum | 136 | 119 | 40 |
| Polski (Chopin) | 81 | 9 | 10 |
| Japoński (日本語) | 43 | 27 | 43 |
| Rosyjski (cyrylica) | 374 | 290 | 362 |

Przewaga BPE:
- Brak nieznanych tokenów: — alfabet bazowy 256 bajtów pokrywa cały UTF-8, więc każdy znak jest reprezentowalny (w razie potrzeby jako pojedyncze bajty).
- Rozmiar słownika jest hiperparametrem - to twórca ustala jego wielkość.
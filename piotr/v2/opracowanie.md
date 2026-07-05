# Fabryka AI vol 2 // W1 - tokenizer (v2)

## Wstęp

Podsumowanie zadania "TYDZ. 01 - Tokenizer" - wersja v2 rozszerza v1 o dwa nowe
korpusy (chiński `wikipedia-zh` i anglojęzyczny `random-text-files`), trzeci
rozmiar słownika (10 000) oraz dotrenowywanie przez checkpointy. Łącznie
**6 korpusów x 3 rozmiary słownika = 18 przebiegów** `TokenizerUTF8`.

Teza: kompresja zależy od dopasowania dziedziny - tokenizer stosuje tylko te
scalenia, które były częste w jego korpusie. Widać to symetrycznie w obu kierunkach:
korpusy polskie/łacińskie w ogóle nie kompresują chińskiego (`cn_moon` = 1.0),
a korpus chiński (`wikipedia-zh`) kompresuje go do **2.99**, będąc jednocześnie
najgorszym dla polskiego i angielskiego. Kompresja nie jest więc własnością tekstu -
jest miarą zgodności tekstu z korpusem treningowym.

## Dane

Sześć korpusów przy trzech rozmiarach słownika (1 000, 5 000, 10 000):

| korpus | dokumenty | znaki | bajty UTF-8 | unikalne znaki | pochodzenie |
| --- | ---: | ---: | ---: | ---: | --- |
| lorem-ipsum | 2 | 1 811 605 | 1 811 605 | 41 | Generator "lorem ipsum" (czysty ASCII) |
| nad-niemnem | 3 | 1 081 002 | 1 175 927 | 96 | E. Orzeszkowa, "Nad Niemnem" |
| random-text-files | 486 | 5 348 450 | 5 366 695 | 252 | 486 anglojęzycznych artykułów publicystycznych (.txt) |
| wolne-lektury | 6 141 | 259 776 717 | 281 206 491 | 420 | HF `polish-dynaword` / `wolne_lektury` |
| wikipedia-pl | 1 171 897 | 1 844 140 560 | 1 944 925 241 | 8 070 | HF `polish-dynaword` / `wikipedia` |
| wikipedia-zh | 1 384 748 | 1 030 636 635 | 2 602 952 850 | 29 554 | HF `wikipedia` / `20231101.zh` |

Widać już z samego zestawienia:

- **lorem-ipsum i random-text-files: bajty ≈ znaki** - tekst niemal czysto ASCII
  (1 bajt/znak), więc kompresja bajtowa i znakowa się pokrywają.
- **Korpusy polskie: bajty > znaki** - znaki diakrytyczne to 2 bajty UTF-8, stąd
  `nad-niemnem` ma 1 175 927 bajtów przy 1 081 002 znakach.
- **`wikipedia-zh`: bajty >> znaki** - znak CJK to 3 bajty UTF-8, więc 1,03 mld
  znaków rozdmuchuje się do 2,6 mld bajtów (najcięższy korpus mimo najmniejszej
  liczby znaków po `wikipedia-pl`).
- **Różnorodność rośnie z rozmiarem** (41 -> 96 -> 252 -> 420 -> 8 070 -> 29 554
  unikalnych znaków). 8 070 w `wikipedia-pl` to *nie* polskie litery - polska
  Wikipedia cytuje obce nazwy i terminy, więc siedzą w niej całe alfabety (długi
  ogon, z którego nie powstają częste pary). W `wikipedia-zh` te 29 554 znaki to
  już rdzeń korpusu, nie ogon.

## Tokenizery

Eksperymenty na tokenizerze bajtowym `TokenizerUTF8` (BPE na poziomie bajtów):
bazą jest 256 surowych bajtów UTF-8 (id `0x00..0xff`), a każde wyuczone scalenie
dostaje kolejny identyfikator powyżej `0xff` i trafia do `merge_map` (id -> para).
Trening to klasyczne BPE: w pętli znajdź najczęstszą sąsiadującą parę, scal ją w
nowy token, powtarzaj aż do zadanego rozmiaru słownika lub braku par do scalenia
(min. 2 wystąpienia).

Dwie konsekwencje warstwy bajtowej, które widać w wynikach:

- **Odtworzenie wejścia jest zawsze bezstratne.** Każdy tekst UTF-8 sprowadza się
  do ciągu bajtów bazowych, więc `decode(encode(x)) == x` zachodzi nawet dla tekstu
  całkiem spoza dziedziny.
- **Tekst spoza dziedziny = brak kompresji, nie błąd.** Nieznany tekst po prostu
  zwraca surowe bajty (kompresja 1.0), bo nie stosuje się do niego żadne scalenie.

Nowość w v2: **trening warm-start przez checkpointy**. BPE jest zachłanne
i deterministyczne, więc pierwszych N scaleń nie zależy od docelowego rozmiaru
słownika - model 5k ma dokładnie te same pierwsze 1 000 par co model 1k, a model
10k te same pierwsze 5 000 co model 5k (zweryfikowane bajt w bajt na tablicach
scaleń). Dzięki temu przebieg 5k wczytuje checkpoint 1k i dolicza tylko brakujące
4 000 scaleń, a 10k rusza z checkpointu 5k. Zob. sekcję "Dotrenowywanie".

Sprawdziłem po drodze też tokenizer na 1/2 bajtu - pary
(`0xfe => (0xf >> 4, 0xe)`, 16 znaków bazowych `[0-f]`) - ale poza wartością
rozrywkowo-edukacyjną nic to nie wniosło.

## Metodyka

Każdy z 18 tokenizerów koduje i dekoduje ten sam zestaw 5 tekstów testowych
(`test_data.strings_to_encode`), po jednym na każdą dziedzinę:

```python
strings_to_encode = {
    "pl_frenzy": "ą, ć, ę, ł, ń, ó, ś, ź, ż, Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż",
    "cn_moon": "月球条目讨论汉漢不转换阅读编辑查看历史外观 隐藏文本小标准大宽度标准宽颜色 （测试版）自动浅色深色维基百科，自由的百科全书提示：此条目的主题不是月或明月。关于与「月球」標題相近或相同的条目",
    "table": "Stół z powyłamywanymi nogami",
    "fox": "The quick brown fox jumps over the lazy dog",
    "lorem_ipsum": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Etiam bibendum, lacus nec consequat iaculis, augue urna posuere leo, id pretium orci ante ac nulla.",
}
```

Artefakty lądują w `data/output/`: słownik per przebieg (`*_tokens.csv`, pominięty
w eksportowanym zestawie danych z racji rozmiaru), statystyki korpusu/treningu 
(`*_stats.csv`) i wspólny plik odtworzeń `in_out.csv`.

## Wyniki i statystyki

### Statystyki treningu (per przebieg)

Kompresja liczona jako bajty bazowe UTF-8 na token wyjściowy (wyżej = lepiej),
mierzona **na korpusie treningowym**. Czas to łączny czas dojścia do danego
słownika (wliczony od zera przez cały łańcuch checkpointów).

| korpus | słownik (cel) | scalenia | słownik (końcowy) | cel? | tokeny (korpus) | kompresja B/tok | znaki/tok | czas (łącznie) |
| --- | ---: | ---: | ---: | :-: | ---: | ---: | ---: | ---: |
| lorem-ipsum | 1 000 | 744 | 1 000 | tak | 286 810 | 6.316 | 6.316 | 0.17 s |
| lorem-ipsum | 5 000 | 886 | **1 142** | **NIE** | 267 984 | 6.76 | 6.76 | 0.17 s |
| lorem-ipsum | 10 000 | 886 | **1 142** | **NIE** | 267 984 | 6.76 | 6.76 | 0.17 s |
| nad-niemnem | 1 000 | 744 | 1 000 | tak | 433 985 | 2.71 | 2.491 | 3.65 s |
| nad-niemnem | 5 000 | 4 744 | 5 000 | tak | 288 551 | 4.075 | 3.746 | 4.49 s |
| nad-niemnem | 10 000 | 9 744 | 10 000 | tak | 248 360 | 4.735 | 4.353 | 4.96 s |
| random-text-files | 1 000 | 744 | 1 000 | tak | 2 114 261 | 2.538 | 2.53 | 5.46 s |
| random-text-files | 5 000 | 4 744 | 5 000 | tak | 1 356 498 | 3.956 | 3.943 | 7.21 s |
| random-text-files | 10 000 | 9 744 | 10 000 | tak | 1 165 836 | 4.603 | 4.588 | 8.19 s |
| wolne-lektury | 1 000 | 744 | 1 000 | tak | 109 246 722 | 2.574 | 2.378 | 275 s (~4.6 min) |
| wolne-lektury | 5 000 | 4 744 | 5 000 | tak | 76 999 486 | 3.652 | 3.374 | 313 s (~5.2 min) |
| wolne-lektury | 10 000 | 9 744 | 10 000 | tak | 67 930 443 | 4.14 | 3.824 | 329 s (~5.5 min) |
| wikipedia-pl | 1 000 | 744 | 1 000 | tak | 826 361 071 | 2.354 | 2.232 | 1 188 s (~19.8 min) |
| wikipedia-pl | 5 000 | 4 744 | 5 000 | tak | 559 412 112 | 3.477 | 3.297 | 1 396 s (~23.3 min) |
| wikipedia-pl | 10 000 | 9 744 | 10 000 | tak | 481 575 372 | 4.039 | 3.829 | 1 485 s (~24.8 min) |
| wikipedia-zh | 1 000 | 744 | 1 000 | tak | 1 238 625 822 | 2.156 | 0.832 | 585 s (~9.8 min) |
| wikipedia-zh | 5 000 | 4 744 | 5 000 | tak | 904 114 273 | 2.953 | 1.14 | 927 s (~15.4 min) |
| wikipedia-zh | 10 000 | 9 744 | 10 000 | tak | 871 933 024 | 3.062 | 1.182 | 1 023 s (~17.0 min) |

Uwagi:

- **`lorem-ipsum` powyżej 1 142 nie osiąga celu**: częste pary kończą się już przy
  słowniku 1 142. Tylko 41 unikalnych znaków i mały, powtarzalny zestaw słów, więc
  każdy docelowy słownik >~1 142 (5k, 10k) daje ten sam tokenizer.
- **Znaki/token < bajty/token dla wielobajtowych alfabetów.** Dla ASCII (`lorem`,
  `random`) obie liczby są bliskie; dla polskiego niższe (diakryt = 2 bajty),
  a dla `wikipedia-zh` bardzo niskie - przy 1k to zaledwie **0.832 znaku/token**
  (znak CJK = 3 bajty, słownik ledwie obejmuje pojedyncze znaki).

### Dotrenowywanie (warm-start i efekt kuli śnieżnej)

Każdy przebieg 5k wystartował z checkpointu 1k (dokładając 4 000 scaleń), a każdy
10k z checkpointu 5k (dokładając 5 000). Ponieważ pierwsze N scaleń jest niezależne
od celu, dolane słowniki są identyczne z tymi z osobnego treningu - nic się nie
traci. Kluczowa obserwacja: **dokładanie kolejnych scaleń jest coraz tańsze**, bo
po wczesnych scaleniach korpus jest reprezentowany w coraz mniejszej liczbie tokenów,
więc każde kolejne przejście liczące pary skanuje krótszy strumień:

| krok | scalenia dołożone | nad-niemnem | random-text-files | wolne-lektury | wikipedia-pl | wikipedia-zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 -> 1k | 744 | 3.65 s | 5.46 s | 275 s | 1 188 s | 585 s |
| 1k -> 5k | +4 000 | +0.84 s | +1.75 s | +38.7 s | +208 s | +342 s |
| 5k -> 10k | +5 000 | +0.47 s | +0.98 s | +15.6 s | +89 s | +96 s |

Choć krok `5k -> 10k` dokłada więcej scaleń (5 000) niż pierwszy krok od zera (744),
kosztuje ułamek jego czasu. W praktyce: aby zbudować tokenizer 10k nie trenuje się
od zera - rozszerza się mniejszy, a tokenizery 1k i 5k dostaje się "przy okazji"
jako produkty pośrednie.

### Odtworzenie tekstów testowych (`in_out.csv`)

Kolumny `encoded_output` i `decoded_output` pominięte - odtworzenie było **bezstratne
we wszystkich 90 przypadkach** (`identical=1`). Liczy się więc `tokens_count`
(mniej = lepsza kompresja przy stałej liczbie bajtów wejścia). Bajty wejścia (`base`,
stałe dla tekstu): `pl_frenzy` = 70, `cn_moon` = 278, `table` = 31, `fox` = 43,
`lorem_ipsum` = 156.

**Liczba tokenów** (mniej = lepiej):

| tokenizer | `pl_frenzy` | `cn_moon` | `table` | `fox` | `lorem_ipsum` |
| --- | ---: | ---: | ---: | ---: | ---: |
| lorem-ipsum 1k | 70 | 278 | 26 | 31 | 26 |
| lorem-ipsum 5k | 70 | 278 | 26 | 31 | 24 |
| lorem-ipsum 10k | 70 | 278 | 26 | 31 | 24 |
| nad-niemnem 1k | 50 | 278 | 14 | 31 | 100 |
| nad-niemnem 5k | 45 | 278 | 12 | 27 | 84 |
| nad-niemnem 10k | 45 | 278 | 10 | 25 | 78 |
| random-text-files 1k | 70 | 278 | 23 | 24 | 87 |
| random-text-files 5k | 70 | 278 | 23 | 16 | 68 |
| random-text-files 10k | 70 | 275 | 19 | 15 | 62 |
| wolne-lektury 1k | 49 | 278 | 14 | 30 | 97 |
| wolne-lektury 5k | 44 | 278 | 12 | 24 | 77 |
| wolne-lektury 10k | 41 | 278 | 10 | 22 | 73 |
| wikipedia-pl 1k | 51 | 278 | 14 | 29 | 93 |
| wikipedia-pl 5k | 44 | 278 | 13 | 23 | 76 |
| wikipedia-pl 10k | 40 | 278 | 11 | 22 | 68 |
| wikipedia-zh 1k | 70 | 134 | 30 | 38 | 141 |
| wikipedia-zh 5k | 69 | 93 | 21 | 22 | 86 |
| wikipedia-zh 10k | 53 | 93 | 16 | 18 | 73 |

**Kompresja** (bajty/token = `base` / `tokens_count`, wyżej = lepiej):

| tokenizer | `pl_frenzy` | `cn_moon` | `table` | `fox` | `lorem_ipsum` |
| --- | ---: | ---: | ---: | ---: | ---: |
| lorem-ipsum 1k | 1.00 | 1.00 | 1.192 | 1.387 | 6.00 |
| lorem-ipsum 5k | 1.00 | 1.00 | 1.192 | 1.387 | 6.50 |
| lorem-ipsum 10k | 1.00 | 1.00 | 1.192 | 1.387 | 6.50 |
| nad-niemnem 1k | 1.40 | 1.00 | 2.214 | 1.387 | 1.56 |
| nad-niemnem 5k | 1.556 | 1.00 | 2.583 | 1.593 | 1.857 |
| nad-niemnem 10k | 1.556 | 1.00 | 3.10 | 1.72 | 2.00 |
| random-text-files 1k | 1.00 | 1.00 | 1.348 | 1.792 | 1.793 |
| random-text-files 5k | 1.00 | 1.00 | 1.348 | **2.688** | 2.294 |
| random-text-files 10k | 1.00 | 1.011 | 1.632 | **2.867** | 2.516 |
| wolne-lektury 1k | 1.429 | 1.00 | 2.214 | 1.433 | 1.608 |
| wolne-lektury 5k | 1.591 | 1.00 | 2.583 | 1.792 | 2.026 |
| wolne-lektury 10k | 1.707 | 1.00 | 3.10 | 1.955 | 2.137 |
| wikipedia-pl 1k | 1.373 | 1.00 | 2.214 | 1.483 | 1.677 |
| wikipedia-pl 5k | 1.591 | 1.00 | 2.385 | 1.87 | 2.053 |
| wikipedia-pl 10k | 1.75 | 1.00 | 2.818 | 1.955 | 2.294 |
| wikipedia-zh 1k | 1.00 | **2.075** | 1.033 | 1.132 | 1.106 |
| wikipedia-zh 5k | 1.014 | **2.989** | 1.476 | 1.955 | 1.814 |
| wikipedia-zh 10k | 1.321 | **2.989** | 1.938 | 2.389 | 2.137 |

### Wnioski

- **Dziedzina treningu decyduje o kompresji.** Każdy tekst najlepiej kompresuje
  tokenizer z jego dziedziny: `lorem_ipsum` -> lorem (6.0-6.5), polskie `table` ->
  korpusy polskie (do 3.1), angielski `fox` -> `random-text-files` (do 2.867),
  chiński `cn_moon` -> `wikipedia-zh` (do 2.99). Poza dziedziną kompresja spada
  ku 1.0.
- **`wikipedia-zh` domyka tezę symetrycznie.** Chiński kompresuje się tylko na
  korpusie chińskim (do 2.99) - i jednocześnie ten sam tokenizer jest najgorszy
  dla polskiego i angielskiego (przy 1k prawie wszystko ~1.0-1.13). Dopiero od
  5k/10k łapie łacińskie pary z tekstu wtrąconego w chińską Wikipedię, ale wciąż
  przegrywa z korpusami polskimi przy tym samym słowniku.
- **Liczy się częstość pary, nie sama obecność w korpusie.** `wikipedia-pl` ma
  w długim ogonie tysiące znaków CJK (8 070 unikalnych), a `cn_moon` i tak daje na
  niej 1.0 przy 1k/5k/10k - te pary są zbyt rzadkie, by wejść do słownika. Odwrotnie
  na `wikipedia-zh`, gdzie chiński jest rdzeniem, a nie ogonem.
- **Większy słownik pomaga z malejącym zyskiem - i tylko w dziedzinie.** `nad-niemnem`
  2.71 -> 4.075 -> 4.735, `wolne-lektury` 2.574 -> 3.652 -> 4.14 (1k -> 5k -> 10k),
  ale `cn_moon` na tych korpusach zostaje na 1.0 niezależnie od słownika. Im bardziej
  różnorodny korpus, tym niższa kompresja własnego tekstu przy tym samym słowniku
  (`nad-niemnem` 2.71 > `wikipedia-pl` 2.354 przy 1k) - tekst powtarzalny domyka się
  tymi samymi scaleniami.
- **Warstwa bajtowa gwarantuje bezstratność.** Odtworzenie było bezstratne we
  wszystkich 90 przypadkach - cokolwiek nie zostanie scalone, wraca do surowych
  bajtów jako najmniejszej jednostki.

## Otwarte pytania

1. W jaki sposób wydajnie robić trening na korpusach większych niż dostępna pamięć?
Za każdym razem trzeba re-merge'ować/re-trenować od zera, bo zmieniają się najpopularniejsze
pary, więc koniec końców do trenowania musi trafić cały korpus. Jakie są istniejące
implementacje umożliwiające strumieniowanie dowolnych ilości tekstów daleko wykraczających
poza dostępną ilość pamięci?

2. ~~Skoro mam tokenizer wytrenowany na 1k vocab size, to teoretycznie mogę dla tego
samego tekstu dotrenować do 10k zaczynając od modelu vocab = 1k?~~
  - Tak - v2 to potwierdza: checkpoint 1k jest dokładnym prefiksem 5k, a 5k prefiksem
    10k, więc rozszerzanie jest bezstratne i tańsze niż trening od zera (zob.
    "Dotrenowywanie").

3. Jak zbudować tokenizer wielojęzyczny, który kompresuje i polski, i chiński naraz?
Każdy korpus specjalizuje się w swojej dziedzinie kosztem pozostałych. Czy korpus
mieszany (pl + zh) da przyzwoitą kompresję obu przy jednym słowniku, i jak duży
słownik jest do tego potrzebny, skoro same znaki CJK to dziesiątki tysięcy jednostek?

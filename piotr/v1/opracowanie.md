# Fabryka AI vol 2 // W1 - tokenizer

## Wstęp

Podsumowanie zadania "TYDZ. 01 - Tokenizer". Przeprowadziłem kilka eksperymentów
na różnych korpusach treningowych, żeby porównać poziom kompresji zależnie od
użytego zbioru. Założenie: im więcej znanych (wytrenowanych) tokenów, tym lepsza
kompresja - a tokenizer wytrenowany na polskich znakach nie skompresuje tekstu
chińskiego. Wyniki potwierdzają to wprost: tekst chiński (`cn_moon`) daje kompresję
**1.0 dla każdego** z ośmiu tokenizerów, bo żaden korpus nie zawierał dość znaków
chińskich, więc każdy znak zostaje przy swoich surowych bajtach UTF-8.

## Dane

Cztery korpusy przy dwóch rozmiarach słownika (1 000 i 5 000), wszystko na
`TokenizerUTF8` - łącznie 8 przebiegów treningowych:

| korpus | dokumenty | znaki | bajty UTF-8 | unikalne znaki | pochodzenie |
| --- | ---: | ---: | ---: | ---: | --- |
| lorem-ipsum | 2 | 1 811 605 | 1 811 605 | 41 | Generator "lorem ipsum" (czysty ASCII) |
| nad-niemnem | 3 | 1 081 002 | 1 175 927 | 96 | E. Orzeszkowa, "Nad Niemnem" |
| wolne-lektury | 6 141 | 259 776 717 | 281 206 491 | 420 | HF `polish-dynaword` / `wolne_lektury` |
| wikipedia | 1 171 897 | 1 844 140 560 | 1 944 925 241 | 8 070 | HF `polish-dynaword` / `wikipedia` |

Widać już z samego zestawienia:

- **lorem-ipsum: bajty = znaki** - czysty ASCII (1 bajt/znak), więc kompresja
  bajtowa i znakowa się pokrywają.
- **Korpusy polskie: bajty > znaki** - znaki diakrytyczne to 2 bajty UTF-8, stąd
  `nad-niemnem` ma 1 175 927 bajtów przy 1 081 002 znakach.
- **Różnorodność rośnie z rozmiarem** (41 -> 96 -> 420 -> 8 070 unikalnych znaków).
  Te 8 070 to *nie* polskie litery - polska Wikipedia cytuje obce nazwy i terminy,
  więc siedzą w niej całe alfabety (np. chiński). To długi ogon, z którego w trakcie
  treningu nie powstają częste pary.

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
  całkiem spoza dziedziny (chiński).
- **Tekst spoza dziedziny = brak kompresji, nie błąd.** Nieznany tekst po prostu
  zwraca surowe bajty (kompresja 1.0), bo nie stosuje się do niego żadne scalenie.

Sprawdziłem po drodze też tokenizer na 1/2 bajtu - pary
(`0xfe => (0xf >> 4, 0xe)`, 16 znaków bazowych `[0-f]`) - ale poza wartością
rozrywkowo-edukacyjną nic to nie wniosło.

## Metodyka

Każdy z 8 tokenizerów koduje i dekoduje ten sam zestaw 5 tekstów testowych
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

Artefakty lądują w `data/output/`: słownik per przebieg (`*_tokens.csv`, pominięty),
statystyki korpusu/treningu (`*_stats.csv`) i wspólny plik odtworzeń `in_out.csv`.

## Wyniki i statystyki

### Statystyki treningu (per przebieg)

Kompresja liczona jako bajty bazowe UTF-8 na token wyjściowy (wyżej = lepiej),
mierzona **na korpusie treningowym**.

| korpus | słownik (cel) | scalenia | słownik (końcowy) | cel? | tokeny (korpus) | kompresja B/tok | znaki/tok | czas trenowania |
| --- | ---: | ---: | ---: | :-: | ---: | ---: | ---: | ---: |
| lorem-ipsum | 1 000 | 744 | 1 000 | tak | 286 810 | 6.316 | 6.316 | 0.11 s |
| lorem-ipsum | 5 000 | 886 | **1 142** | **NIE** | 267 984 | 6.76 | 6.76 | 0.11 s |
| nad-niemnem | 1 000 | 744 | 1 000 | tak | 433 985 | 2.71 | 2.491 | 3.56 s |
| nad-niemnem | 5 000 | 4 744 | 5 000 | tak | 288 551 | 4.075 | 3.746 | 4.66 s |
| wolne-lektury | 1 000 | 744 | 1 000 | tak | 109 246 722 | 2.574 | 2.378 | 279 s (~4.6 min) |
| wolne-lektury | 5 000 | 4 744 | 5 000 | tak | 76 999 486 | 3.652 | 3.374 | 388 s (~6.5 min) |
| wikipedia | 1 000 | 744 | 1 000 | tak | 826 358 183 | 2.354 | 2.232 | 12 608 s (~3.5 h) |
| wikipedia | 5 000 | 4 744 | 5 000 | tak | 559 410 829 | 3.477 | 3.297 | 13 111 s (~3.6 h) |

Uwagi:

- **`lorem-ipsum` przy 5 000 nie osiągnął celu**: częste pary skończyły się już
  przy słowniku 1 142. Tylko 41 unikalnych znaków i mały, powtarzalny zestaw słów,
  więc każdy docelowy słownik powyżej ~1 142 daje ten sam tokenizer.
- **Znaki/token < bajty/token dla polskiego, równe dla łaciny.** Dla `lorem-ipsum`
  obie liczby są identyczne (ASCII); dla korpusów polskich znaki/token są niższe,
  bo znak diakrytyczny = 2 bajty (`nad-niemnem-5000`: 4.075 B/tok, ale 3.746 znaku/tok).

### Odtworzenie tekstów testowych (`in_out.csv`)

Kolumny `encoded_output` i `decoded_output` pominięte - odtworzenie było **bezstratne
we wszystkich 40 przypadkach** (`identical=1`). Liczy się więc `tokens_count`
(mniej = lepsza kompresja przy stałej liczbie bajtów wejścia).

**Liczba tokenów** (`base` = bajty wejścia, stałe dla danego tekstu):

| tekst | base | lorem 1k | lorem 5k | nad 1k | nad 5k | wolne 1k | wolne 5k | wiki 1k | wiki 5k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pl_frenzy` | 70 | 70 | 70 | 50 | 45 | 49 | 44 | 51 | 44 |
| `cn_moon` | 278 | 278 | 278 | 278 | 278 | 278 | 278 | 278 | 278 |
| `table` | 31 | 26 | 26 | 14 | 12 | 14 | 12 | 14 | 13 |
| `fox` | 43 | 31 | 31 | 31 | 27 | 30 | 24 | 29 | 23 |
| `lorem_ipsum` | 156 | 26 | 24 | 100 | 84 | 97 | 77 | 93 | 76 |

**Kompresja** (bajty/token = `base` / `tokens_count`, wyżej = lepiej):

| tekst | lorem 1k | lorem 5k | nad 1k | nad 5k | wolne 1k | wolne 5k | wiki 1k | wiki 5k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pl_frenzy` | 1.00 | 1.00 | 1.40 | 1.556 | 1.429 | 1.591 | 1.373 | 1.591 |
| `cn_moon` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `table` | 1.192 | 1.192 | 2.214 | 2.583 | 2.214 | 2.583 | 2.214 | 2.385 |
| `fox` | 1.387 | 1.387 | 1.387 | 1.593 | 1.433 | 1.792 | 1.483 | 1.87 |
| `lorem_ipsum` | 6.00 | 6.50 | 1.56 | 1.857 | 1.608 | 2.026 | 1.677 | 2.053 |

### Wnioski

- **Dziedzina treningu decyduje o kompresji.** "lorem ipsum" najlepiej kompresują
  tokenizery lorem (6.0-6.5), polskie słabo (1.56-2.05); polskie zdanie `table`
  odwrotnie - lorem najgorzej (1.19), korpusy polskie ~2.2-2.6. Chiński stoi na 1.0
  wszędzie przez to, ze we wszystkich treningach powstało za mało chińskich par
- **Liczy się częstość pary, nie sama obecność w korpusie.** Wikipedia zawiera 917+
  znaków chińskich, a `cn_moon` i tak daje na niej 1.0 - żadne z 4 744 scaleń
  `wikipedia-5000` nie zawiera znaku chińskiego. Chiński to mala część korpusu, więc jego
  pary przegrywają walkę o miejsce w słowniku. Kompresja pojawiłaby się dopiero przy
  większym słowniku albo korpusie, w którym te pary są częste.
- **Większy słownik pomaga z malejącym zyskiem - i tylko w dziedzinie.** `nad-niemnem`
  2.71 -> 4.075, `wolne-lektury` 2.574 -> 3.652 (1k -> 5k), ale `cn_moon` zostaje na 1.0.
  Im bardziej różnorodny korpus, tym niższa kompresja własnego tekstu przy tym samym
  słowniku (`nad-niemnem` 2.71 > `wikipedia` 2.354 przy 1k) - tekst powtarzalny domyka
  się tymi samymi scaleniami.
- **Warstwa bajtowa gwarantuje bezstratność.** Odtworzenie było bezstratne we wszystkich
  40 przypadkach, także dla chińskiego - cokolwiek nie zostanie scalone, wraca do
  surowych bajtów jako najmniejszej jednostki.

## Otwarte pytania

1. W jaki sposób wydajnie robić trening na korpusach większych niż dostępna pamięć?
Za każdym razem trzeba re-merge'owac/re-trenować od zera bo zmieniają się najpopularniejsze
pary więc koniec końców, do trenowania musi trafić cały korpus. Jakie są istniejące
implementacje umożliwiające strumieniowanie dowolnych ilości tekstów daleko wykraczających
poza dostępną ilość pamięci?

2. Skoro mam tokenizer wytrenowany na 1k vocab size to teoretycznie, dla tego samego
tekstu, dotrenować do 10k zaczynając od modelu vocab = 1k?

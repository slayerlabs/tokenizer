# Byte-level BPE na zrównoważonej próbce Polish Dynaword

Tokenizer napisany od zera w czystym Pythonie, bez bibliotek `tokenizers` i `sentencepiece`, wytrenowany na próbce polskiego korpusu pretreningowego Polish Dynaword i zmierzony na held-oucie obok `cl100k_base` oraz `o200k_base`. Słownik ma 32 000 tokenów, czyli tyle samo co tokenizer Mistrala i APT4 z Bielika v3, a pełny trening trwa niecałą minutę na laptopie bez karty graficznej.

## Na czym trenowałem

Polish Dynaword składa się z 19 plików źródłowych o bardzo różnym charakterze: prawo, protokoły parlamentarne, Wikipedia i jej projekty siostrzane, literatura, prasa, strony z sieci, korpus NKJP. W pełnym korpusie dominuje objętościowo prawo i teksty urzędowe, dlatego tokenizer uczony na losowym wycinku nauczyłby się głównie języka ustaw. Próbka bierze więc z każdego źródła mniej więcej tyle samo tekstu, około dwóch milionów znaków, czerpiąc dokumenty z sześciu równo rozłożonych miejsc pliku, a nie tylko z jego początku. Długie dokumenty, czyli powieści i ustawy, są przycinane do 50 000 znaków, żeby jedno źródło nie składało się z trzech dokumentów.

Podział na trening i held-out idzie po skrócie SHA-1 identyfikatora dokumentu: co dziesiąty dokument trafia do held-outu. Podział jest deterministyczny i dokumentowy, więc żaden dokument nie leży w obu częściach jednocześnie. Część treningowa liczy 34 507 829 znaków w 21 211 dokumentach, held-out 3 845 918 znaków w 2 383 dokumentach.

## Jak to działa

Przed BPE tekst jest dzielony na kawałki tym samym wyrażeniem regularnym co w `cl100k_base`: słowo razem z poprzedzającą spacją, liczby po trzy cyfry, interpunkcja, białe znaki. BPE działa wyłącznie wewnątrz kawałka, więc nigdy nie skleja końcówki jednego słowa z początkiem następnego. Alfabetem bazowym jest 256 bajtów, dzięki czemu każdy tekst, łącznie z emoji i chińskimi znakami, da się zakodować bez tokenu nieznanego.

Plik `bpe.py` zawiera dwie prymitywne operacje (`get_pair_counts` i `merge`), podręcznikową wersję treningu `train_naive` oraz wersję szybką `train`. Szybka wersja daje ten sam wynik, a różni się dwoma zabiegami. Pierwszy polega na tym, że uczy się na unikalnych kawałkach z ich częstością, zamiast na strumieniu tekstu: 6,3 mln kawałków próbki to tylko 417 tys. unikalnych. Drugi polega na tym, że po scaleniu nie liczy par od nowa, tylko poprawia liczniki w słowach, które scaloną parę zawierały, a najczęstszą parę trzyma w kopcu. Test `test_fast_equals_naive` sprawdza, że obie wersje uczą się identycznych 400 reguł na tym samym fragmencie korpusu, a `test_readme_example` odtwarza przykład `aaabdaaabac` z README warsztatu.

Kodowanie stosuje reguły w kolejności, w jakiej powstały: w każdym kroku spośród par obecnych w kawałku scalana jest ta, której reguła została wyuczona najwcześniej. Token `<|endoftext|>` dostaje zarezerwowany identyfikator 31 999 i jest atomowy, a w tekście rozpoznawany wyłącznie na wyraźne życzenie (`allowed_special=True`). Ponieważ reguły tworzą ciąg, tokenizer o słowniku 8k albo 16k jest po prostu prefiksem reguł dużego, więc trzy rozmiary słownika wychodzą z jednego treningu.

## Jak mierzyłem

Wszystkie tokenizery kodują ten sam held-out, dokument po dokumencie, bez tokenów specjalnych. Słowo to ciąg znaków bez białych znaków, czyli wynik `text.split()`, więc wyraz z przecinkiem liczy się jako jedno słowo. Fertility to liczba tokenów podzielona przez liczbę słów (mniej znaczy lepiej), a znaki na token i bajty na token mówią, ile tekstu niesie średnio jeden token (więcej znaczy lepiej). Round-trip sprawdza dla każdego dokumentu, czy po zakodowaniu i odkodowaniu wraca dokładnie ten sam tekst. Held-out ma 543 073 słowa.

## Ile wyszło

| tokenizer | słownik | fertility | znaki/token | bajty/token | tokeny | round-trip |
|---|---:|---:|---:|---:|---:|---|
| nasz BPE | 8 000 | 2,091 | 3,39 | 3,60 | 1 135 307 | 2383/2383 |
| nasz BPE | 16 000 | 1,868 | 3,79 | 4,03 | 1 014 668 | 2383/2383 |
| **nasz BPE** | **32 000** | **1,693** | **4,18** | **4,44** | **919 508** | 2383/2383 |
| `cl100k_base` | 100 277 | 2,729 | 2,60 | 2,76 | 1 482 026 | 2383/2383 |
| `o200k_base` | 200 019 | 2,370 | 2,99 | 3,17 | 1 286 962 | 2383/2383 |

Na tym samym polskim tekście tokenizer 32k potrzebuje o 38% mniej tokenów niż `cl100k_base` (1,61 raza mniej) i o 29% mniej niż `o200k_base` (1,40 raza mniej), mając przy tym słownik odpowiednio trzy i sześć razy mniejszy.

Zysk względem `cl100k_base` w rozbiciu na źródła held-outu:

| źródło | fertility nasz 32k | fertility `cl100k_base` | ile razy mniej tokenów |
|---|---:|---:|---:|
| parliamentary | 1,47 | 2,75 | 1,87 |
| govpl | 1,61 | 2,85 | 1,77 |
| european_hplt_v3_pl | 1,51 | 2,59 | 1,71 |
| wikinews | 1,69 | 2,87 | 1,69 |
| 1000_novels | 1,55 | 2,62 | 1,69 |
| samorzad_gov_pl | 1,71 | 2,87 | 1,67 |
| nkjp1m | 1,60 | 2,66 | 1,66 |
| wolne_lektury | 1,61 | 2,62 | 1,63 |
| dziennik_ustaw | 1,68 | 2,73 | 1,63 |
| wikisource | 1,61 | 2,62 | 1,62 |
| eltec_pol | 1,57 | 2,53 | 1,61 |
| wiktionary_examples | 1,70 | 2,72 | 1,60 |
| wikiquote | 1,74 | 2,78 | 1,60 |
| eurlex | 1,88 | 2,98 | 1,58 |
| wikivoyage | 1,85 | 2,91 | 1,57 |
| wikipedia | 1,76 | 2,75 | 1,56 |
| biblioteka_nauki | 1,77 | 2,72 | 1,54 |
| global_voices | 1,82 | 2,64 | 1,45 |
| wikibooks | 2,05 | 2,88 | 1,41 |

Część źródeł ma w held-oucie tylko kilka dokumentów (1000_novels cztery, wolne_lektury i eltec_pol po dziewięć), więc ich pozycja w tabeli jest obarczona dużym rozrzutem.

## Co mnie zaskoczyło

Najbardziej zaskoczyło mnie to, że polski tokenizer o słowniku zaledwie 8 000 tokenów koduje polski tekst krócej niż `o200k_base` ze słownikiem dwudziestopięciokrotnie większym (1,14 mln tokenów wobec 1,29 mln). To mocne potwierdzenie tezy z warsztatu, że liczy się fertility w danym języku, a nie sam rozmiar słownika: większość z 200 tysięcy tokenów generycznego tokenizera po prostu nie dotyczy polszczyzny.

Kolejność pierwszych reguł czyta się jak lekcję polskiej ortografii. Już w pierwszych dwudziestu scaleniach pojawiają się `ie`, `ni`, `nie`, `cz`, `rz`, `ch`, `st`, a także litery `ł`, `ę`, `ą`, `ó`, `ż`, które w UTF-8 zajmują po dwa bajty i dopiero scalenie skleja je w jeden token. Litera `ł` jest drugą regułą w ogóle, przed jakimkolwiek słowem ze spacją.

W polskim tokenizerze znalazło się 55 tokenów z chińskimi albo japońskimi znakami. Ich źródłem są artykuły Global Voices, które cytują wpisy z Chin i Japonii w oryginale; w jedenastu dokumentach treningowych tego źródła jest od około 300 do ponad 2 000 takich znaków. Przy zrównoważonej próbce kilka dokumentów jednego źródła wystarcza, żeby zająć słownik obcym pismem, a to samo źródło ma też jeden z najmniejszych zysków w tabeli. Najsłabiej wypada wikibooks. Nie badałem tego dokładnie, ale w próbce treningowej trafiła się tam między innymi tabela znaków Unicode z kodami szesnastkowymi, więc część tego źródła to raczej dane niż proza.

Najwięcej zyskują protokoły parlamentarne, a nie teksty prawne. Spodziewałem się, że powtarzalny język ustaw wypadnie najlepiej, tymczasem eurlex znalazł się w dolnej części tabeli, a dziennik_ustaw w środku, Moja hipoteza, jeszcze niesprawdzona, jest taka, że oba źródła są pełne numerów, odsyłaczy i skrótów, które tną się na krótkie tokeny w każdym tokenizerze.

## Co dalej

Ten held-out jest zwykły: podział jest losowy po dokumentach i nikt nie sprawdzał, czy w treningu nie leżą bliźniacze wersje dokumentów z held-outu. Polish Dynaword nie ma near-dedupu, a w Dzienniku Ustaw prawie identyczne teksty jednolite ustaw stanowią około jednej piątej tokenów. Następny krok to ten sam pomiar na held-oucie oczyszczonym z bliźniaków metodą MinHash i LSH, żeby sprawdzić, o ile zwykły held-out zawyża wynik tokenizera.

## Jak odtworzyć

Wymagane pakiety: `regex`, `pyarrow` (do próbki), `tiktoken` (do porównania), oraz lokalna kopia plików parquet zbioru [SlayerLab/polish-dynaword](https://huggingface.co/datasets/SlayerLab/polish-dynaword), po jednym pliku na źródło.

```
python sample_corpus.py <katalog_z_parquetami>   # data/train.jsonl, data/heldout.jsonl, około 20 s
python test_bpe.py               # testy, kilka sekund
python train_tokenizer.py 32000  # artifacts/pl_bpe_32k.json, około minuty
python evaluate.py               # tabela i results/heldout_random.json
```

Artefakt `artifacts/pl_bpe_32k.json` zawiera reguły scalania w kolejności nauki, wzorzec pretokenizacji i tokeny specjalne, a `artifacts/pl_bpe_32k.vocab.txt` pozwala obejrzeć słownik: identyfikator, tekst tokenu i parę, z której powstał.

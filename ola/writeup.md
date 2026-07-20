# Byte-level BPE od zera - Quo Vadis

Autor: Aleksandra (Ola) Jachymiak

Tokenizer **Byte-Pair Encoding na poziomie bajtów**, napisany od zera w czystym
Pythonie (bez bibliotek ML). Projekt zaczyna się od prostego baseline'u na *Quo Vadis*
(vocab 512, bez pretokenizacji) i iteracyjnie dochodzi do finalnej wersji: diverse korpus
~8,9M znaków, pretokenizacja regexem GPT-2, vocab 8000. Celem było przejść cały
mechanizm ze zrozumieniem - styl Karpathy/minbpe, każda wersja jako kontrolowany
eksperyment z jedną zmienną.

## Ścieżka eksperymentów

Zadanie rozwiązywałam iteracyjnie - każda wersja to **kontrolowany eksperyment** z jedną
zmienną, aż do złożenia wszystkiego razem w Wersji 4.

| Wersja | Korpus | Vocab | Pretokenizacja | Co badałam |
|---|---|---|---|---|
| **v1** | Quo Vadis (~1M znaków) | 512 | ✗ | baseline: czy BPE w ogóle działa na polskim? |
| **v2** | Wikipedia PL (~750k znaków) | 512 | ✗ | wpływ korpusu przy stałym vocab |
| **v3** | Quo Vadis | 512 → 4096 | ✗ | wpływ rozmiaru słownika przy stałym korpusie |
| **v4** | literatura + wiki (~8,9M znaków) | 8000 | ✅ GPT-2 regex | wszystko razem: produkcyjny tokenizer |

**Logika progresji:**

**v1 → v2** izoluje zmienną *korpus* (vocab identyczny = 512). Wynik: tokenizer z jednej
książki marnuje sloty na imiona własne (`Petroniusz`, `Winicjusz`); tokenizer z Wikipedii
uczy się końcówek fleksyjnych (`nych `, `ały `, `ść `), które działają na dowolnym polskim
tekście. Każdy wygrywa na swojej domenie, ale diverse generalnie lepiej generalizuje.

**v1 → v3** izoluje zmienną *rozmiar słownika* (korpus identyczny = Quo Vadis). Wynik:
fertility spada monotoniczne 3.31 → 1.85 (−44%) wraz ze wzrostem vocab 512 → 4096.
Przy 4096 ten tokenizer „bije" GPT-4 na tym held-out - ale to złudne: ewaluacja
in-domain + brak pretokenizacji pozwolił tokenom pochłonąć całe frazy z `\n`
(`'.\n\nLecz Winicjusz'`, `'rzekł:\n\n- Nie'`). Dobra kompresja tej jednej książki,
bezużyteczna poza nią.

**v4** składa wszystkie lekcje: diverse korpus (uogólnienie z v2) + duży vocab (siła z v3)
+ pretokenizacja regexem GPT-2 (merge nie przekracza granicy słowa/znaku nowej linii) +
inkrementalne liczniki par (83 s zamiast 20 min). Najdłuższe nauczone tokeny to prawdziwe
polskie słowa (`Rzeczypospolitej`, `sprawiedliwości`), nie śmieciowe frazy.

**Przewodnia lekcja:** niską fertility łatwo „kupić" - przeuczając na jednej książce albo
pozostawiając merge'om wolną rękę nad whitespace. *Dobry* tokenizer optymalizuje
generalizację, nie kompresję jednego tekstu.

---

## Cel

Zbudować bajtowy tokenizer BPE, który:
- koduje **dowolny** tekst Unicode bez tokenów `<UNK>` (pełne pokrycie bajtowe 0-255),
- uczy się częstych fragmentów polskich słów z danych,
- jest w pełni odwracalny: `decode(encode(text)) == text`.

## Podejście

Tokenizer startuje od **surowych bajtów UTF-8** (256 tokenów bazowych, 0-255), więc
każdy znak - polski, obcy, emoji - rozkłada się na bajty i nigdy nie powoduje `<UNK>`.

Algorytm (6 kroków, każdy jako osobna funkcja):

1. `text.encode("utf-8")` - tekst na listę bajtów.
2. `get_pair_counts` - policz wszystkie sąsiadujące pary (okno przesuwane).
3. `merge` - sklej wybraną parę w nowy token (id 256, 257, ...).
4. `train_bpe` - pętla: policz pary → wybierz najczęstszą → sklej → zapisz regułę → powtórz.
5. `encode` - nowy tekst na tokeny, aplikując reguły **w kolejności uczenia**
   (para o najniższym id merge'a najpierw).
6. `build_vocab` + `decode` - tokeny z powrotem na tekst.

**Świadome uproszczenie (Wersja 1 - baseline):** w tej wersji trenujemy na **surowym
strumieniu bajtów, bez pretokenizacji regexem** (w przeciwieństwie do GPT-2, który
najpierw dzieli tekst na słowa/liczby/interpunkcję). To sprawia, że tokeny mogą obejmować
interpunkcję razem ze spacją lub znakiem nowej linii (np. `", że "`). Pretokenizacja
regexem GPT-2 zostaje dodana w Wersji 4.

## Dane treningowe

- **Korpus:** *Quo Vadis*, Henryk Sienkiewicz - Wolne Lektury (domena publiczna).
- **Czyszczenie:** odcięto nagłówek (autor/tytuł/ISBN) oraz stopkę licencyjną Wolnych Lektur.
- **Czysty tekst:** 1 086 738 znaków / 1 179 102 bajty UTF-8.
- **Podział train/held-out:** trening na pierwszych 1 046 738 znakach; **ostatnie 40 000
  znaków** odłożono jako zbiór testowy, którego tokenizer **nie widział** podczas treningu
  (uczciwa ewaluacja).

## Parametry

- **Rozmiar słownika:** 512 (256 bajtów bazowych + 256 nauczonych reguł merge).
- **Czas treningu:** ~75 s (czysty Python, jednowątkowo).

## Artefakt: `quo_vadis_tokenizer.json`

Format własny, zbliżony do stylu HuggingFace, ale bez mapowania `bytes_to_unicode` (dla
prostoty tokeny opisujemy wprost bajtami + czytelnym tekstem):

- `merges` - lista reguł w kolejności uczenia, każda jako `[lewy_id, prawy_id, nowy_id]`.
  Kolejność jest istotna: wcześniejsze reguły mają niższy id i są stosowane najpierw.
- `vocab` - słownik `id -> {bytes, text}`, gdzie `bytes` to surowe bajty tokenu, a `text`
  to jego czytelna reprezentacja (`decode` z `errors="replace"`).
- `eval_heldout` - metryki policzone na zbiorze testowym.

## Wyniki

**Poprawność:** `decode(encode(held-out)) == held-out` → **True** (idealna rekonstrukcja,
0 tokenów `<UNK>`).

**Metryki na held-out (40 000 znaków, 6 261 słów):**

| metryka | wartość |
|---|---|
| tokeny | 20 741 |
| fertility | **3.313 tok/słowo** |
| znaki/token | 1.929 |
| kompresja (bajty/tokeny) | 2.082× |

**Najdłuższe nauczone tokeny** (algorytm sam je odkrył, bez wiedzy o polskim):

`Petroniusz`, `Winicjusz`, `etroniusz`, `roniusz`, `nicjusz`, `, że `, `, ale `,
`, któr`, `ł się`

Widać dwie rzeczy: (1) tokenizer nauczył się **imion głównych bohaterów** (bo występują
setki razy) oraz (2) **częstych polskich konstrukcji łączących** (`, że `, `, ale `).

## Porównanie z GPT-4 (tiktoken `cl100k_base`) na tym samym held-out

| tokenizer | słownik | tokeny | fertility |
|---|---|---|---|
| mój (od zera) | 512 | 20 741 | 3.313 |
| GPT-4 `cl100k` | ~100 000 | 15 833 | 2.529 |

GPT-4 jest lepszy (niższa fertility) - i tak ma być: ma ~200× większy słownik. Mój wynik
przy zaledwie 512 tokenach jest sensowny. Pokazuje to główną zależność BPE: **większy
słownik → grubsze tokeny → mniej tokenów na słowo** (kosztem większej macierzy embeddingów).

## Wersja 2: diverse korpus (Wikipedia PL) - czy lepszy dataset pomaga?

Żeby sprawdzić wpływ **jakości korpusu**, wytrenowałam drugi tokenizer na próbce
polskiej Wikipedii (10 dużych artykułów z różnych dziedzin: historia, geografia, nauka -
~749 000 znaków, licencja CC BY-SA). **Kluczowe: ten sam `vocab_size = 512`** co przy
Quo Vadis - dzięki temu jedyną zmienną jest korpus, nie rozmiar słownika (kontrolowany
eksperyment).

**Eksperyment 2×2 - oba tokenizery na obu tekstach** (fertility, niżej = lepiej):

| tokenizer \ tekst | QV (literacki) | Wiki (encyklopedyczny) |
|---|---|---|
| QV-tok (512) | **3.313** | 4.592 |
| Wiki-tok (512) | 3.549 | **4.268** |

Każdy tokenizer wygrywa na **swojej** domenie. Przy tak małym słowniku (512) diverse
korpus nie zmiażdżył wersji literackiej - ale prawdziwa różnica jest widoczna w tym,
**czego** się nauczyły:

| QV-tok (jedna książka) | Wiki-tok (diverse) |
|---|---|
| `Petroniusz`, `Winicjusz`, `roniusz` | `przez `, `oraz `, `któr`, `się ` |
| (imiona bohaterów - bezużyteczne indziej) | `nych `, `ały `, `ść ` (końcówki fleksyjne!) |

**Wniosek:** tokenizer z jednej książki marnuje sloty na imiona własne; tokenizer z
diverse korpusu uczy się **reużywalnej gramatyki polskiej** (końcówki, łączniki), więc
lepiej generalizuje na nieznany tekst. Do prawdziwego LLM liczy się właśnie generalizacja
- czyli **różnorodny korpus wygrywa**, mimo że na jednej konkretnej książce jej własny
tokenizer wypada lepiej. To praktyczna ilustracja problemu fleksji z teorii.

Odniesienie: GPT-4 `cl100k` na tych samych tekstach - fertility 2.53 (QV) / 2.80 (Wiki).
Wciąż lepszy, bo ~100 000 tokenów vs moje 512.

Artefakt: `diverse_tokenizer.json`, skrypt: `homework_diverse.py`.

## Wersja 3: większy słownik (sweep 512 → 4096) - najprostsza dźwignia

Sekcja *Ograniczenia* wskazywała mały słownik (512) jako największą łatwą dźwignię.
Tu ją sprawdzam. Kluczowa obserwacja: merge'e BPE są **prefiksowe** - słownik 4096
zawiera pierwsze 512/1024/2048 merge'y jako swój początek. Dlatego **jednym treningiem**
do 4096 robię snapshoty metryk na held-out w checkpointach i dostaję całą krzywą
(kontrolowany eksperyment: jedyna zmienna to rozmiar słownika; korpus i split te same
co w Wersji 1).

**Krzywa fertility vs rozmiar słownika (ten sam held-out, 40 000 znaków, 6 261 słów):**

| vocab | tokeny | fertility | znaki/token | roundtrip |
|---|---|---|---|---|
| 512 | 20 741 | 3.313 | 1.929 | ✔ |
| 1024 | 16 378 | 2.616 | 2.442 | ✔ |
| 2048 | 13 758 | 2.197 | 2.907 | ✔ |
| 4096 | 11 600 | **1.853** | 3.448 | ✔ |

Fertility spada monotonicznie 3.31 → **1.85** (–44%) - dokładnie zależność BPE z Wersji 1:
większy słownik → grubsze tokeny → mniej tokenów na słowo. Wartość przy 512 (3.313)
odtworzyła się co do cyfry z baseline'u (trening jest deterministyczny). Trening do 4096
zajął ~20 min (czysty Python, naiwne `train_bpe` przeliczające wszystkie pary co rundę -
patrz *Ograniczenia*, „naiwna złożoność").

**Uwaga - dlaczego przy 4096 bijemy GPT-4 i dlaczego to złudne.** Na tym held-out ten
tokenizer (fertility 1.853) wypada lepiej niż GPT-4 `cl100k` (2.529, ~100 000 tokenów).
To **nie** znaczy, że jest lepszy - wynika z dwóch rzeczy:

1. **Ewaluacja in-domain.** Held-out to końcówka *tej samej książki*, na której trenował.
   Słownik jest przeuczony na styl i słownictwo Sienkiewicza.
2. **Brak pretokenizacji.** Bez regexu GPT-2 tokeny mogą obejmować całe wielowyrazowe
   frazy z nowymi liniami. Najdłuższe nauczone tokeny przy 4096 mówią wszystko:
   `'dawało mu się, że '`, `'- odpowiedział '`, `'.\n\nLecz Winicjusz'`, `'! - zawołał '`,
   `'rzekł:\n\n- Nie'`. To świetna kompresja *tego konkretnego* tekstu i bezużyteczne
   gdzie indziej.

Innymi słowy niska fertility jest tu kupiona przeuczeniem i brakiem pretokenizacji, nie
lepszym tokenizerem. Na tekście spoza domeny (jak w eksperymencie 2×2 z Wersji 2) GPT-4
zmiażdżyłby ten słownik. To ta sama lekcja co w Wersji 2 - **liczy się generalizacja, nie
kompresja jednej książki** - tylko pokazana od strony rozmiaru słownika.

Artefakt: `quo_vadis_tokenizer_4k.json`, skrypt: `homework_vocab_sweep.py`.

## Wersja 4: „ten dobry" tokenizer - diverse korpus + pretokenizacja (8000)

Wersje 1–3 były kontrolowanymi eksperymentami (izolacja jednej zmiennej). Tu składam
wszystko w jeden **produkcyjny** tokenizer, stosując materiał kursowy „Jak zrobić dobry
tokenizer dla LLM" (Slayer AI Lab).

**Zastosowane lekcje z materiału** (numery sekcji):
- **Pretokenizacja regexem GPT-2/cl100k** [2.1] - merge nigdy nie przekracza granicy
  pretokenu; litery/cyfry(1–3)/interpunkcja osobno; spacja wiodąca doklejona do słowa;
  **newline oddzielony od spacji** (żeby nie sklejać whitespace przez nowe linie).
- **NFC + czyszczenie** [2.2/4.1] - soft-hyphen, znaki kontrolne, CRLF→\n, BOM.
- **Mapa pretokenów + inkrementalne liczniki par** [3.3] - trening na `{pretoken: count}`,
  aktualizacja tylko par sąsiadujących z merge'em.
- **Higiena merge'ów** [5.2] - `min_freq=2`, `max_token_bytes=32`.
- **Blok tokenów specjalnych** zarezerwowany z góry [6].

**Korpus (diverse, ~8,9M znaków):** 10 książek Wolne Lektury (Prus, Żeromski, Sienkiewicz,
Wyspiański, Fredro, Orzeszkowa - proza + dramat) + ~90 artykułów Wikipedii PL z ~12 dziedzin.
Podział ~63% literatura / 37% encyklopedia. **Pan Tadeusz świadomie POZA treningiem** - zbiór ewaluacyjny.

**Trening:** 7739 merge'y (vocab 8000) w **83 s**. Dla porównania naiwna pętla przy 4096
(Wersja 3) zajęła ~20 min. To praktyczny dowód tezy z materiału [3.3]: mapa pretokenów +
inkrementalne liczniki zamieniają godziny w minuty.

**Wyniki na całym Panu Tadeuszu (445 646 znaków, 68 905 słów), roundtrip = True:**

| metryka | wartość |
|---|---|
| vocab | 8000 |
| fertility | **2.205 tok/słowo** |
| roundtrip | ✔ |

Dla porównania GPT-4 `cl100k` na tym samym tekście: fertility ~2.53. Wynik bliski
produkcyjnemu pomimo ~12× mniejszego słownika - efekt doboru korpusu (literatura polska)
zbliżonego do domeny ewaluacyjnej. Pełniejsze porównanie wymagałoby zrównania rozmiaru
vocab i ewaluacji wielodomenowej [7]. Najdłuższe nauczone tokeny to prawdziwe polskie słowa
(`Rzeczypospolitej`, `sprawiedliwości`, `społeczeństwa`), a nie frazy z \n jak w Wersji 3 -
pretokenizacja zadziałała.

**Napotkany i naprawiony defekt (warto zapamiętać):** pierwsza wersja pretokenizera łączyła
spacje z nowymi liniami w jeden pretoken (`\s+`), więc BPE nauczył się wielo-liniowych
„potworów" whitespace i marnował na nie sloty - dokładnie to, przed czym ostrzega materiał
(„merge nie przekracza newline" [2.1]). Fix: rozdzielenie `\n+` od `[^\S\n]+` + higiena
`max_token_bytes`. **Fertility po fixie nie drgnęła** (2.205 → 2.205), co dowodzi, że potwory
nie podbijały wyniku - po prostu zaśmiecały słownik. Rezydualnie zostały jeszcze pojedyncze
runy myślników/spacji (boilerplate) - dobiłoby je czyszczenie na poziomie linii albo
min-document-frequency [4.2/5.2].

**Vocab utilization:** 5723 / 8000 (71,5%) tokenów „strzela" na Panu Tadeuszu; reszta to
tokeny wiki/innych domen + zarezerwowane specjalne - spodziewane przy ewaluacji na jednej
domenie.

Artefakt: `polish_bpe_8k.json`.

## Ograniczenia i następne kroki

- **Mały słownik (512).** Zaadresowane w *Wersji 3*: sweep do 4096 obniżył fertility
  3.31 → 1.85 na held-out. To była najprostsza dźwignia. Dalej ograniczeniem robi się już
  brak pretokenizacji (poniżej) i koszt naiwnego treningu (na dole).
- **Brak pretokenizacji.** Zaadresowane w *Wersji 4*: regex GPT-2 przed BPE (merge nie
  przekracza granicy pretokenu) usunął frazowe tokeny z \n i podbił jakość podziału.
- **Jedna książka (baseline).** Słownik Quo Vadis jest dopasowany do stylu Sienkiewicza
  (imiona rzymskie). Zaadresowane w *Wersji 2* (diverse Wikipedia) i domknięte w *Wersji 4*
  (literatura + wiki, ~8,9M znaków).
- **Naiwna złożoność.** `train_bpe` przelicza wszystkie pary od nowa co rundę (O(rund × n)).
  Zaadresowane w *Wersji 4*: mapa pretokenów + inkrementalne liczniki par → 8000 vocab w 83 s.
- **Pozostaje (v4):** resztki boilerplate (runy myślników/spacji) do wycięcia przez
  min-document-frequency / czyszczenie na poziomie linii [4.2/5.2]; chunkowanie cyfr od
  prawej (Llama 3); ewaluacja wielodomenowa; entropia Rényi jako lepszy predyktor niż sama
  kompresja [7]; porównanie z produkcyjnym tokenizerem Bielika.

## Artefakty

Repozytorium zawiera słowniki wszystkich wersji w formacie JSON:

- `quo_vadis_tokenizer.json` - Wersja 1 (vocab 512, Quo Vadis)
- `diverse_tokenizer.json` - Wersja 2 (vocab 512, Wikipedia PL)
- `quo_vadis_tokenizer_4k.json` - Wersja 3 (vocab 4096, sweep Quo Vadis)
- `polish_bpe_8k.json` - Wersja 4 (vocab 8000, diverse korpus + pretokenizacja)

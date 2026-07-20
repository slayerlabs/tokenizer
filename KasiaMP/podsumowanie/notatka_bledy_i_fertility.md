# Notatka projektowa: cztery błędy i wyniki fertility

Projekt: byte-level BPE trenowany od zera na korpusie polskiej literatury (9 363 020 znaków po czyszczeniu), vocab 6756 (256 bazy + 6500 merge), porównany z tokenizerami HuggingFace trenowanymi na identycznym korpusie i z identycznym rozmiarem słownika.

---

## Przypadek 1: Monster tokeny z boilerplate'u PDF

**Objaw.** Tokenizer uczył się bardzo długich, wielowyrazowych tokenów odpowiadających stopkom i nagłówkom z plików PDF (np. fragmenty formuły "Materiał ćwiczeniowy dla uczniów niepełnosprawnych…", powtórzonej ok. 211 razy w korpusie). Kompresja in-domain wyglądała świetnie, ale tokeny były bezużyteczne poza korpusem.

**Przyczyna.** Powtarzający się boilerplate dominował statystykę par. BPE wybiera zawsze najczęstszą parę, więc frazy powtórzone setki razy wygrywały rundy merge, mimo że nie reprezentują języka, tylko artefakt źródła danych.

**Naprawa.** Filtr częstości linii przed treningiem: usunięcie każdej linii występującej w korpusie 30 lub więcej razy (`PROG = 30`), wraz z liniami pustymi.

**Lekcja.** Jakość danych poprzedza trening. Dobra metryka kompresji na danych treningowych może być fałszywym sygnałem — wynikiem przeuczenia się na śmieciach, nie jakości tokenizera.

---

## Przypadek 2: Zepsuta baza bajtowa w eksporcie JSON (błąd główny)

**Objaw.** Uwaga prowadzącego: "nie nałożyłaś podstawowego 256 vocabu na standardowe znaki; do 255 powinnaś mieć tylko jedną literkę". W wyeksportowanym JSON-ie wpisy o ID 0–127 były poprawne (pojedyncze znaki ASCII), ale wszystkie 128 wpisów o ID 128–255 zawierało ten sam znak — spację.

**Przyczyna.** Linia eksportu:

```python
"vocab": {str(i): vocab[i].decode("utf-8", errors="replace") for i in sorted(vocab)}
```

Pojedynczy bajt z zakresu 128–255 nie jest samodzielnym znakiem UTF-8 — to fragment sekwencji wielobajtowej (np. bajt 197 to dopiero pierwsza połowa litery `ł` = bajty 197+130). Dekodowanie takiego bajtu w pojedynkę musi się nie powieść, a `errors="replace"` zamiast zgłosić błąd podstawiał znak zastępczy — ten sam dla wszystkich 128 bajtów. Mapa bajt→znak przestała być funkcją różnowartościową (iniekcją), więc odtworzenie bajtów ze stringów stało się matematycznie niemożliwe: dekoder nie istnieje.

**Istotne rozróżnienie.** Algorytm treningu był poprawny przez cały czas — słownik w pamięci (`vocab = {i: bytes([i]) …}`) zawierał komplet 256 różnych bajtów, a reguły merge zgadzały się z tokenizerem referencyjnym (pierwszy nietrywialny merge `(197, 130) → ID 257 = ł` identyczny jak w pliku HF kolegi). Zepsuł się wyłącznie artefakt wyjściowy — jedna linijka serializacji.

**Naprawa.** Odwracalna mapa bajt→znak w stylu GPT-2 ByteLevel: bajty drukowalne (33–126, 161–172, 174–255) pozostają sobą; bajty niedrukowalne (0–32, 127–160, 173 — soft hyphen) otrzymują kolejne unikalne znaki zastępcze od kodepunktu 256 wzwyż (stąd `Ġ` = spacja = chr(288)). Linia eksportu zastąpiona przepuszczeniem każdego bajtu osobno przez tę mapę. Naprawa wdrożona i zweryfikowana: baza zapisanego JSON-a zawiera 256 unikalnych wpisów, round-trip przechodzi dla wszystkich 256 bajtów i dla wszystkich 6756 tokenów słownika, a zapisy ID 65 = `A`, ID 197 = `Å`, ID 130 = `Ĥ`, ID 257 = `ÅĤ` są zgodne z konwencją HuggingFace ByteLevel. Dodano również brakujący dekoder — round-trip `decode(encode(x)) == x` przechodzi na tekstach polskich, emoji i piśmie chińskim.

**Lekcja.** Poprawny algorytm + zepsuta serializacja = zepsuty artefakt. Eksport słownika musi być iniekcją, inaczej dekodowanie jest niemożliwe. `errors="replace"` w eksporcie danych to cicha utrata informacji.

---

## Przypadek 3: Konwencja rozmiaru słownika (6500 vs 6756)

**Objaw.** Uwaga prowadzącego: "vocab_size + base_vocab = realny vocab". Ryzyko raportowania rozmiaru słownika jako 6500 (same merge), podczas gdy realny słownik to 6756.

**Przyczyna.** Nieutrwalona konwencja: rozmiar słownika obejmuje wszystkie tokeny posiadające ID — bazę bajtową (256), tokeny z merge (6500) i ewentualne tokeny specjalne. Tak liczy cała branża (GPT-2: 50257 = 256 + 50000 + 1) i tak liczy parametr `vocab_size` w bibliotece HuggingFace. Liczba wierszy przyszłej macierzy embeddingów modelu równa się temu całkowitemu rozmiarowi.

**Naprawa.** Jednolita formuła w całej dokumentacji: "Vocab size 6756: 256 tokenów bazy bajtowej + 6500 tokenów z reguł merge". Przy treningu porównawczym w HF ustawiono `vocab_size=6756`, aby całkowite rozmiary słowników były równe.

**Lekcja.** Powiązany drobiazg: baza bajtowa liczy 256 wartości (ID 0–255 włącznie), nie 255 — klasyczny błąd off-by-one, który pojawiał się w notatkach kilkukrotnie.

---

## Przypadek 4: Rozjazd korpusu przy rekonstrukcji eksperymentu

**Objaw.** Plik korpusu przygotowany dla HF miał 9 485 253 znaki zamiast oczekiwanych 9 363 020 (+122 233 znaki, +1,3%). Trening HF na takim pliku byłby nieuczciwy — porównanie wymagało identycznego tekstu.

**Przyczyna.** Do pliku trafił korpus surowy, sprzed czyszczenia. Diagnoza licznikiem linii potwierdziła obecność boilerplate'u (linie powtórzone 210–211 razy). Pierwsza próba rekonstrukcji filtra nie trafiła w cel o 2349 znaków, ponieważ rekonstrukcja zakładała próg `PROG = 100` i pomijała etap normalizacji — w oryginalnym kodzie z Colaba próg wynosił `PROG = 30`, a po filtrze następowała normalizacja (ujednolicenie cudzysłowów i myślników, redukcja wielokrotnych spacji i pustych linii).

**Naprawa.** Odtworzenie pełnego potoku z oryginalnego skryptu: filtr `PROG = 30` → normalizacja. Wynik: 9 363 020 znaków, różnica od celu równa 0 — korpus odtworzony co do znaku.

**Lekcja.** Liczba znaków korpusu działa jak suma kontrolna eksperymentu: pozwala wykryć każdą rozbieżność w danych zanim zniekształci wyniki. Uczciwe porównanie wymaga identycznych danych wejściowych, zweryfikowanych liczbowo, nie "na oko".

---

## Fałszywy alarm (nie-błąd): inne ID bazy w pliku HF

W wyeksportowanym pliku HF znaki bazowe mają inne numery niż w pliku referencyjnym (`'A'` = 32 zamiast 65, `'Ġ'` = 220 zamiast 32). To nie jest błąd: HF posortował alfabet bazowy po kodepunktach znaków zastępczych, a nie po wartościach bajtów. Weryfikacja programowa: 256 wpisów, wszystkie różne (iniekcja), każdy o długości jednego znaku, zbiór identyczny z alfabetem GPT-2 ByteLevel. Konkretne numery ID nie niosą znaczenia — liczy się kompletność mapy, różnowartościowość i to, że encoder i dekoder używają tej samej mapy.

---

## Wyniki fertility na held-out

Warunki pomiaru: wspólny held-out (2696 znaków, 371 słów, tekst spoza korpusu treningowego), identyczny korpus treningowy (9 363 020 znaków, zgodność co do znaku), identyczny całkowity rozmiar słownika (6756), identyczny kod metryk (mianownik: `len(held_out.split())`).

| tokenizer | tokeny | tokens/word | chars/token |
|---|---|---|---|
| HF z regex + normalizator (korpus przefiltrowany) | **833** | **2,245** | **3,236** |
| HF z regex + normalizator (korpus surowy) | 833 | 2,245 | 3,236 |
| Mój BPE (Colab, bez pre-tokenizacji) | 836 | 2,253 | 3,225 |
| HF ByteLevel, z regex (pre-tokenizacja) | 840 | 2,264 | 3,210 |
| HF ByteLevel, bez regex | 844 | 2,275 | 3,194 |

Interpretacja:

**Wszystkie warianty mieszczą się w przedziale różnic 1,3%** (833–844 tokeny). Przy held-oucie o rozmiarze 371 słów różnice tego rzędu są w granicach szumu pomiarowego i nie uprawniają do mocnych wniosków o przewadze któregokolwiek wariantu.

**Efekt normalizacji, po izolacji zmiennej.** Wariant z normalizatorem wytrenowano powtórnie na korpusie już przefiltrowanym, aby jedyną różnicą wobec wariantu „HF z regex" pozostał normalizator. Porównanie 840 → 833 daje poprawę 0,8%. Zastrzeżenie: korpus przeszedł wcześniej ręczną normalizację (cudzysłowy, myślniki, wielokrotne spacje), więc mierzony jest efekt resztkowy normalizatora — NFC, usunięcie soft hyphenów i znaków kontrolnych.

**Filtr boilerplate'u nie wpłynął na wynik na held-oucie.** Oba warianty z normalizatorem — trenowany na korpusie surowym i na przefiltrowanym — dały identyczną liczbę 833 tokenów, mimo że ich słowniki różnią się 155 tokenami (2,3% słownika). Różniące się tokeny to fragmenty boilerplate'u obecne wyłącznie w wariancie surowym; ponieważ held-out nie zawiera stopek PDF, tokeny te nie miały czego kompresować, a wspólne 97,7% słownika wykonało całą pracę. Wniosek: przy włączonej pre-tokenizacji ochrona przed monster tokenami działa podwójnie — filtr danych i regex realizują częściowo to samo zadanie.

**Własna implementacja ≈ implementacja produkcyjna.** Różnica między moim BPE a HF w tej samej konfiguracji (bez pre-tokenizacji) wynosi 0,9% (836 vs 844). To empiryczny dowód poprawności własnej implementacji algorytmu — dwie niezależne realizacje tego samego algorytmu na tych samych danych dają praktycznie ten sam wynik. Drobna pozostała różnica wynika z odmiennej obsługi granic: HF trenuje per linia (merge nie przekracza `\n`), moja implementacja pracuje na ciągłym strumieniu bajtów.

**Pre-tokenizacja nie poprawiła istotnie fertility** (844 → 840, ok. 0,5%), wbrew przewidywaniu o wyraźnej poprawie generalizacji. Jej realna wartość leży gdzie indziej: tokenizer bez pre-tokenizacji wydaje część budżetu merge na tokeny wielowyrazowe przekraczające granice słów (np. `", bo "`, `" się "` — widoczne zarówno w moim słowniku, jak i w wariancie HF bez regex), które kompresują wyłącznie frazy znane z treningu. Pre-tokenizacja wymusza tokeny nieprzekraczające granic słów (interpunkcja osobno, spacja wiodąca doklejona do słowa: `Ġdo`, `Ġbo`), czyli jednostki wielokrotnego użytku o lepszych granicach morfologicznych. Na held-oucie zbliżonym stylistycznie do korpusu efekt jest mały; różnica powinna rosnąć wraz z odległością domenową tekstu testowego.

**Czas treningu:** HF (Rust, trening wielordzeniowy, na zliczonych pre-tokenach) wytrenował oba warianty łącznie w 8,1 s czasu zegarowego (52,3 s CPU). Różnica wydajności względem implementacji własnej w czystym Pythonie wynika z języka i optymalizacji algorytmicznej, nie z różnicy w samym algorytmie BPE.

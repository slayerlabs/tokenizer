# Polski Byte-Level BPE — implementacja własna i porównanie z HuggingFace

## Przegląd

Artefaktem projektu jest byte-level BPE zaimplementowany od zera w Pythonie,
wytrenowany na korpusie polskiej literatury, oraz zestaw tokenizerów
HuggingFace wytrenowanych na tym samym korpusie jako punkt odniesienia.

```text

wynik FINALNY.json             - tokenizer na HF
wyniki.json                    — tokenizer własny (format autorski)
wyniki/hf\\\_bez\\\_regex.json       — HF ByteLevel, bez pre-tokenizacji
wyniki/hf\\\_z\\\_regex.json         — HF ByteLevel, z pre-tokenizacją
wyniki/hf\\\_normalizowany.json   — HF ByteLevel, z pre-tokenizacją i normalizatorem
```

Najlepszy wynik na zbiorze held-out:

```text
tokens/word = 2.245
chars/token = 3.236
vocab       = 6756
```

## Pytanie badawcze

Celem nie było uzyskanie najlepszego możliwego tokenizera, lecz odpowiedź na
pytanie, którego nie da się postawić, korzystając wyłącznie z gotowej
biblioteki:

* czy implementacja BPE napisana od zera daje wyniki porównywalne z biblioteką
produkcyjną przy identycznych warunkach;
* jak duży jest realny wpływ pre-tokenizacji na kompresję tekstu spoza domeny;
* jakie błędy popełnia się, budując tokenizer ręcznie, i czego one uczą o
samym algorytmie.

Punktem wyjścia była implementacja własna, a nie biblioteka — dzięki temu
porównanie mierzy konkretną decyzję projektową (obecność pre-tokenizacji),
a nie różnicę między dwoma nieznanymi czarnymi skrzynkami.

## Projekt tokenizera

Tokenizer używa bazowego słownika bajtowego, więc dowolny ciąg wejściowy da się
zakodować bez tokenów `<UNK>`.

Konfiguracja implementacji własnej:

* baza: 256 bajtów UTF-8, ID 0–255;
* merge: ID od 256 wzwyż, 6500 reguł;
* całkowity rozmiar słownika: 6756 (256 bazy + 6500 merge);
* limit długości tokenu: 20 bajtów;
* zliczanie par przez `Counter`, jedna reguła merge na rundę;
* `\\\\n` zachowany w korpusie, bez lowercasingu, bez usuwania interpunkcji;
* polskie znaki diakrytyczne zachowane (na poziomie bajtów);
* brak pre-tokenizacji — trening na ciągłym strumieniu bajtów.

Warianty HuggingFace wytrenowane dla porównania:

|wariant|pre-tokenizacja|normalizator|
|-|-|-|
|bez regex|nie|nie|
|z regex|tak|nie|
|z normalizatorem|tak|tak|

Normalizator w trzecim wariancie obejmuje: NFC, ujednolicenie końców linii do
`\\\\n`, usunięcie soft hyphena, znaków zero-width, BOM i znaków kontrolnych,
redukcję wielokrotnych spacji i pustych linii.

## Dane treningowe

Korpus złożony z trzech tekstów: *Tajemniczy ogród* (F.H. Burnett), *200 mil*
oraz fragment Konstytucji RP.

Czyszczenie przed treningiem:

* usunięcie linii powtarzających się w korpusie 30 lub więcej razy (stopki i
nagłówki z plików PDF) wraz z liniami pustymi;
* ujednolicenie cudzysłowów i myślników;
* redukcja wielokrotnych spacji i potrójnych końców linii.

Rozmiar korpusu po czyszczeniu: **9 363 020 znaków**.

Zbiór held-out: **2696 znaków, 371 słów**, tekst spoza korpusu treningowego.

## Protokół ewaluacji

Metryki:

* fertility (tokens/word) — niższe znaczy lepiej;
* chars/token — wyższe znaczy lepiej;
* liczba tokenów na całym held-oucie.

Warunki uczciwości porównania:

* identyczny korpus treningowy dla wszystkich wariantów, zgodność
zweryfikowana liczbą znaków (różnica 0);
* identyczny całkowity rozmiar słownika (6756) we wszystkich wariantach;
* identyczny kod metryk, ten sam podział na słowa (`str.split()`);
* held-out sprawdzony pod kątem braku nakładania się z korpusem.

## Wyniki na held-out

|tokenizer|korpus|tokeny|tokens/word|chars/token|
|-|-|-:|-:|-:|
|HF z regex + normalizator (wynik FINALNY.json)|przefiltrowany|**833**|**2.245**|**3.236**|
|HF z regex + normalizator|surowy|833|2.245|3.236|
|Implementacja własna|przefiltrowany|836|2.253|3.225|
|HF z regex|przefiltrowany|840|2.264|3.210|
|HF bez regex|przefiltrowany|844|2.275|3.194|

Obserwacje:

1. **Implementacja własna dorównuje bibliotece produkcyjnej.** Różnica między
implementacją własną a wariantem HF w tej samej konfiguracji (bez
pre-tokenizacji) wynosi 0,9% (836 vs 844). Dwie niezależne realizacje tego
samego algorytmu na tych samych danych dają praktycznie ten sam wynik, co
stanowi empiryczne potwierdzenie poprawności implementacji.
2. **Pre-tokenizacja poprawiła fertility nieznacznie** (844 → 840, ok. 0,5%),
wyraźnie mniej, niż zakładało przewidywanie postawione przed
eksperymentem.
3. **Rozstęp całej stawki wynosi 1,3%.** Przy held-oucie o rozmiarze 371 słów
różnice tego rzędu mieszczą się w granicach szumu i nie uprawniają do
mocnych wniosków o przewadze konkretnego wariantu.
4. **Normalizacja dała poprawę 0,8% po izolacji zmiennej.** Wariant z
normalizatorem wytrenowano powtórnie na korpusie przefiltrowanym, aby
jedyną różnicą wobec wariantu „HF z regex" pozostał normalizator: 840 →
5. Mierzony jest efekt resztkowy, ponieważ korpus przeszedł wcześniej
ręczną normalizację cudzysłowów, myślników i wielokrotnych spacji —
normalizatorowi HF pozostało NFC, usunięcie soft hyphenów i znaków
kontrolnych.
6. **Filtr boilerplate'u nie wpłynął na wynik na held-oucie.** Warianty z
normalizatorem trenowane na korpusie surowym i przefiltrowanym dały
identyczną liczbę 833 tokenów, mimo że ich słowniki różnią się 155
tokenami (2,3%). Różniące się tokeny to fragmenty boilerplate'u obecne
wyłącznie w wariancie surowym; held-out nie zawiera stopek PDF, więc nie
miały czego kompresować. Przy włączonej pre-tokenizacji ochrona przed
monster tokenami działa podwójnie — filtr danych i regex realizują
częściowo to samo zadanie.

## Jakość tokenów

Metryki zbiorcze nie pokazują, czego BPE faktycznie się nauczył, dlatego
porównano cięcie tego samego zdania przez warianty z pre-tokenizacją i bez.

Zdanie testowe: *Zadzwoniłam do serwisu, bo laptop przestał się ładować.*

|wariant|podział|
|-|-|
|bez regex|`Za \| dzwo \| ni \| łam␣ \| do␣ \| ser \| wi \| su \| ,␣bo␣ \| la \| p \| to \| p \| ␣prze \| stał \| ␣się␣ \| ład \| ować \| .`|
|z regex|`Za \| dzwo \| ni \| łam \| ␣do \| ␣ser \| wi \| su \| , \| ␣bo \| ␣la \| p \| to \| p \| ␣prze \| stał \| ␣się \| ␣ła \| dować \| .`|

Wnioski:

* wariant bez pre-tokenizacji uczy się tokenów **wielowyrazowych**
przekraczających granice słów i sklejających interpunkcję: `,␣bo␣`, `␣się␣`,
`łam␣`, `do␣` — spacja doklejona z tyłu;
* wariant z pre-tokenizacją utrzymuje interpunkcję osobno, a spację dokleja
**z przodu** słowa (`␣do`, `␣bo`), zgodnie z konwencją GPT-2;
* identyczny wzorzec tokenów wielowyrazowych występuje w implementacji
własnej (`", bo "`, `" się "`, `" dopiero "`), co potwierdza, że jest to
skutek braku pre-tokenizacji, a nie błędu implementacji;
* tokeny wielowyrazowe kompresują wyłącznie frazy znane z korpusu
treningowego, więc ich wartość spada wraz z odległością domenową tekstu
testowego.

Zbieżność z tokenizerem referencyjnym: pierwszy nietrywialny merge w
implementacji własnej to para bajtów (197, 130) → ID 257, czyli litera `ł`.
Identyczny merge pod identycznym ID występuje w tokenizerze HuggingFace
wytrenowanym niezależnie przez inną osobę na polskim korpusie. Statystyka
języka przebija się przez różne implementacje.

## Wydajność

|implementacja|czas treningu|
|-|-|
|HF (Rust, wielordzeniowo, dwa warianty łącznie)|8,1 s zegarowe / 52,3 s CPU|
|Implementacja własna (czysty Python, jednordzeniowo)|ok. 15 500 s (ok. 4,3 h)|

Przyspieszenie rzędu 4000× na jeden trening. Stosunek czasu CPU do czasu
zegarowego w HF (52,3 / 8,1 ≈ 6,5) pokazuje, że biblioteka zrównolegliła
pracę. Różnica wydajności wynika z języka implementacji, zrównoleglenia i
optymalizacji algorytmicznej (trening na zliczonych pre-tokenach zamiast na
pełnym strumieniu), nie z różnicy w samym algorytmie BPE.

Krzywa częstości zwycięskich par w implementacji własnej: runda 0 — 228 141
wystąpień, runda 500 — 1689, runda 1000 — 778, runda 3000 — 202, runda 6000 —
87, ostatnia runda — 79. Spadek jest gwałtowny na początku i bardzo powolny
w dalszej części treningu.

## Poprawność artefaktu

Po naprawie serializacji artefakt implementacji własnej przechodzi komplet
testów poprawności:

|test|wynik|
|-|-|
|mapa bajt→znak: liczba wpisów|256|
|mapa bajt→znak: iniekcja (unikalne wartości)|tak|
|round-trip bajt → znak → bajt, 256 asercji|przechodzi|
|odwracalność zapisu znakowego dla całego słownika (6756 tokenów)|przechodzi|
|liczba unikalnych wartości w bazie zapisanego JSON-a|256|

Round-trip `decode(encode(x)) == x` na tekstach:

|tekst|wynik|
|-|-|
|zdanie demonstracyjne (polski)|OK|
|pełny held-out|OK|
|`Zażółć gęślą jaźń`|OK|
|`emoji: 🙂 i chiński: 中文`|OK|
|cyfry i znaki specjalne|OK|

Ostatni przypadek jest istotny: emoji i pismo chińskie nie występują w
korpusie treningowym ani razu, a mimo to kodują się i dekodują bezstratnie.
Jest to praktyczne potwierdzenie gwarancji braku tokenów `<UNK>` wynikającej
z kompletnej bazy bajtowej.

Zgodność z konwencją HuggingFace ByteLevel: w zapisanym słowniku ID 65 to
`A`, ID 197 to `Å`, ID 130 to `Ĥ`, a ID 257 to `ÅĤ` — czyli dokładnie te same
zapisy, jakie występują w tokenizerach eksportowanych przez bibliotekę.

## Wnioski

1. **Byte-level gwarantuje brak `<UNK>`.** Kompletna baza 256 bajtów pokrywa
każdy możliwy tekst, kosztem 68 wpisów na bajty niedrukowalne, które
praktycznie nie występują.
2. **Poprawny algorytm nie wystarczy — artefakt musi być odwracalny.**
Serializacja słownika wymaga mapy bajt→znak będącej iniekcją; bez tego
dekoder nie istnieje, mimo w pełni poprawnego treningu.
3. **Pre-tokenizacja poprawia jakość granic, nie kompresję.** Jej efekt na
fertility był marginalny, ale zmienia charakter uczonych tokenów z
frazowych na wielokrotnego użytku.
4. **Kompresja in-domain bywa myląca.** Boilerplate w korpusie generuje długie
tokeny, które poprawiają metryki na danych podobnych do treningowych i nie
dają nic poza nimi.
5. **Konwencja rozmiaru słownika obejmuje bazę.** 6756 = 256 + 6500;
analogicznie GPT-2: 50257 = 256 + 50000 + 1.

## Ograniczenia

* Implementacja własna nie ma pre-tokenizacji ani tokenów specjalnych.
* Nie zmierzono utylizacji słownika ani bytes/token.
* Nie przeprowadzono testów odporności (liczby, kod, markdown, emoji, tekst
bez diakrytyków, CAPS LOCK).
* Nie przeprowadzono ablacji po rozmiarze słownika — testowano wyłącznie 6756.
* Held-out liczy 371 słów, co jest zbyt małą próbą, aby różnice rzędu 1%
uznać za istotne.
* Nie porównano wyników z tokenizerami innych osób na wspólnym zbiorze
testowym.
* Efekt normalizatora zmierzono wyłącznie jako resztkowy, na korpusie
poddanym wcześniej ręcznej normalizacji.

## Kolejne kroki

* Ewaluacja na held-oucie spoza domeny literackiej, gdzie przewaga
pre-tokenizacji powinna być wyraźniejsza.
* Testy odporności na liczbach, kodzie i tekście bez diakrytyków.

## Artefakt końcowy

Artefaktem zgłoszonym w projekcie jest implementacja własna byte-level BPE
(vocab 6756) wraz z zestawem trzech tokenizerów HuggingFace stanowiących
kontrolę porównawczą. Wartością zgłoszenia jest metodologia porównania:
zrównane warunki, izolowana zmienna i udokumentowany proces diagnozy błędów.


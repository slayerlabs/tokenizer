# Podsumowanie tokenizatora BPE dla języka polskiego

## Cel

Celem jest bajtowy tokenizator BPE dla tekstu polskiego. Tokenizator ma kodować dowolny tekst Unicode bez tokenów nieznanych, dobrze kompresować polskie słowa i jednocześnie zachować pełną odwracalność: `decode(encode(text)) == text`.

## Podejście

Tokenizator używa podejścia podobnego do GPT-2: najpierw dzielimy tekst wyrażeniem regularnym na fragmenty należące do różnych kategorii znaków, a dopiero potem uczymy BPE wewnątrz tych fragmentów.

Użyty wzorzec pretokenizacji:

```python
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

W praktyce oznacza to, że tokenizator osobno traktuje słowa, liczby, interpunkcję i białe znaki. Wyjątkiem jest opcjonalna spacja przed słowem lub liczbą, np. `" kot"`, co pozwala BPE nauczyć się częstych fragmentów ze spacją na początku. To istotnie poprawia kompresję tekstu bez agresywnego łączenia słów z interpunkcją.

Istotna konsekwencja tego kroku jest taka, że BPE nie widzi całego tekstu jako jednej długiej sekwencji bajtów. Najpierw regex dzieli tekst na niezależne fragmenty, a scalania BPE są uczone tylko wewnątrz tych fragmentów. Dzięki temu tokenizator nie może stworzyć pojedynczego tokenu obejmującego kilka słów, np. `hello world`, bo granica między `hello` i ` world` została ustalona już na etapie pretokenizacji. Może natomiast nauczyć się tokenu ` world`, czyli słowa razem ze spacją poprzedzającą. To zachowuje informację o odstępach i poprawia kompresję, ale nie pozwala słownikowi marnować miejsc na całe frazy wielowyrazowe, które są mniej uniwersalne niż pojedyncze słowa lub części słów.

To jest zgodne z obserwacją z pracy GPT-2, `Language Models are Unsupervised Multitask Learners`:

> We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . This results in a sub-optimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

W naszym przypadku zapobiega to marnowaniu słownika na wiele wariantów tego samego słowa z różnymi znakami interpunkcyjnymi, np. `słowo`, `słowo.`, `słowo,`, `słowo?`. Dzięki temu ograniczona liczba tokenów jest przeznaczana głównie na produktywne fragmenty słów i częste konstrukcje języka polskiego.

## Dane treningowe

Tokenizator został wytrenowany na korpusie z polskiej Wikipedii, przygotowanym z dumpów Wikimedia.

```text
plwiki-latest-pages-articles1.xml-p1p187037.bz2
```

Statystyki korpusu treningowego:

- Liczba artykułów: `3 570`
- Liczba słów: `5 081 838`
- Liczba znaków: `37 076 542`
- Liczba bajtów UTF-8: `39 190 360`

Podczas przygotowania danych usunęliśmy oczywiste artefakty dumpów Wiki, m.in. fragmenty LaTeX, znaczniki `<ref>`, `<math>`, szablony, tabele i surowy markup typu `[[...]]` oraz `{{...}}`.

## Artefakt

Wygenerowany artefakt tokenizatora znajduje się w pliku:

```text
tokenizer.json
```

Format jest zgodny ze stylem Hugging Face `tokenizers`, co pozwala na użycie wytrenowanych "wag" z biblioteką `Toknizers` od HF:

- `model.type = "BPE"` oznacza model Byte Pair Encoding.
- `model.vocab` mapuje reprezentację tokenu na jego identyfikator liczbowy.
- `model.merges` zawiera uporządkowaną listę reguł scalania BPE.
- `pre_tokenizer.type = "ByteLevel"` oznacza bajtowy tokenizer zgodny z mapowaniem GPT-2.
- `decoder.type = "ByteLevel"` pozwala odwrócić tokeny z powrotem do tekstu.

Słownik składa się z dwóch części:

- `256` tokenów bazowych odpowiadających pojedynczym bajtom.
- `5 976` tokenów nauczonych przez BPE jako kolejne reguły scalania.

Reguła merge mówi, że dwa istniejące tokeny mogą zostać zastąpione nowym tokenem. Na przykład uproszczona reguła:

```text
Ġ p
```

oznacza, że token reprezentujący spację (`Ġ`) oraz token `p` mogą zostać scalone w jeden nowy token. Kolejność reguł jest ważna, bo wcześniejsze reguły mają niższy rank i są stosowane jako pierwsze.

## Główne statystyki tokenizatora

- Typ: `Byte-level BPE`
- Rozmiar słownika: `6 232`
- Liczba reguł merge: `5 976`
- Liczba tokenów bazowych bajtowych: `256`
- Liczba tokenów wieloznakowych nauczonych przez BPE: `5 976`
- Średnia długość wpisu słownikowego: `5.10` znaków w reprezentacji byte-level
- Najdłuższy wpis słownikowy: `17` znaków w reprezentacji byte-level

Kilka najdłuższych tokenów nauczonych przez BPE:
` Rzeczypospolitej`,
` przeciwieństwie`,
` najważniejszych`,
` przedstawicieli`,
` niepodległości`,
` rzeczywistości`,
` poszczególnych`,
` prawdopodobnie`,
` współczesnych`

## Metryki na korpusie treningowym

- Liczba tokenów: `11 344 286`
- Fertility: `2.232 tok/słowo`
- Współczynnik kompresji znakowej: `69.40%`
- Średnia liczba znaków na token: `3.27`

## Metryki na tekście ewaluacyjnym

Pomiar wykonany na całości Pana Tadeusza:

- Liczba znaków: `447 334`
- Liczba słów: `69 095`
- Liczba tokenów: `177 519`
- Współczynnik kompresji znakowej: `60.32%`
- Tekst w tokenach jest około `2.52` razy krótszy niż tekst liczony znakowo.
- Fertility: `2.569 tok/słowo`

Interpretacja: wynik `2.569 tok/słowo` oznacza, że jedno polskie słowo wymaga średnio około 2.57 tokenu. Dla języka fleksyjnego, takiego jak polski, jest to rozsądny wynik przy słowniku rzędu kilku tysięcy tokenów.

## Brak lowercasingu

Nie zamieniamy tekstu na małe litery przed treningiem tokenizatora. To świadoma decyzja: lowercasing zwykle poprawia kompresję i zmniejsza liczbę wariantów słów, ale jednocześnie usuwa informację semantyczną.

## Ograniczenia

Tokenizer był trenowany głównie na Wikipedii, więc jego słownik jest dostosowany do stylu encyklopedycznego. Może być mniej optymalny dla poezji, czatów, tekstów prawnych, forów internetowych, kodu źródłowego lub tekstów silnie potocznych.

Drugie ograniczenie to rozmiar słownika. `6 232` tokeny dają dobrą prostotę i pełne pokrycie bajtowe, ale większy słownik, np. `16k` lub `32k`, prawdopodobnie zmniejszyłby fertility i skrócił sekwencje wejściowe kosztem większej tablicy embeddingów w modelu.

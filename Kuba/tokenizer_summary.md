# Podsumowanie tokenizatora BPE dla języka polskiego

## Cel

Celem jest bajtowy tokenizator BPE dla tekstu polskiego. Tokenizator ma kodować dowolny tekst Unicode bez tokenów nieznanych, dobrze kompresować polskie słowa i jednocześnie zachować pełną odwracalność: `decode(encode(text)) == text`.

## Podejście

Tokenizator używa podejścia podobnego do GPT: najpierw dzielimy tekst wyrażeniem regularnym na fragmenty należące do różnych kategorii znaków, a dopiero potem uczymy BPE wewnątrz tych fragmentów.

Użyty wzorzec pretokenizacji:

```python
pat = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]+|\s++$|\s[\r\n]|\s+(?!\S)|\s""")
```

W praktyce oznacza to, że tokenizator osobno traktuje słowa, liczby, interpunkcję i białe znaki. Wyjątkiem jest opcjonalna spacja przed słowem lub liczbą, np. `" kot"`, co pozwala BPE nauczyć się częstych fragmentów ze spacją na początku. To istotnie poprawia kompresję tekstu bez agresywnego łączenia słów z interpunkcją.

Istotna konsekwencja tego kroku jest taka, że BPE nie widzi całego tekstu jako jednej długiej sekwencji bajtów. Najpierw regex dzieli tekst na niezależne fragmenty, a scalania BPE są uczone tylko wewnątrz tych fragmentów. Dzięki temu tokenizator nie może stworzyć pojedynczego tokenu obejmującego kilka słów, np. `hello world`, bo granica między `hello` i ` world` została ustalona już na etapie pretokenizacji. Może natomiast nauczyć się tokenu ` world`, czyli słowa razem ze spacją poprzedzającą. To zachowuje informację o odstępach i poprawia kompresję, ale nie pozwala słownikowi marnować miejsc na całe frazy wielowyrazowe, które są mniej uniwersalne niż pojedyncze słowa lub części słów.

To jest zgodne z obserwacją z pracy GPT-2, `Language Models are Unsupervised Multitask Learners`:

> We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . This results in a sub-optimal allocation of limited vocabulary slots and model capacity. To avoid this, we prevent BPE from merging across character categories for any byte sequence. We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

W naszym przypadku zapobiega to marnowaniu słownika na wiele wariantów tego samego słowa z różnymi znakami interpunkcyjnymi, np. `słowo`, `słowo.`, `słowo,`, `słowo?`. Dzięki temu ograniczona liczba tokenów jest przeznaczana głównie na produktywne fragmenty słów i częste konstrukcje języka polskiego.

## Dane treningowe

Korpus treningowy został przygotowany z polskiej Wikipedii na podstawie dumpów Wikimedia. Został ograniczony do `30M` słów, żeby uzyskać stabilne statystyki merge dla tokenizatora BPE bez niepotrzebnego powiększania pliku treningowego.

```text
plwiki-latest-pages-articles1.xml-p1p187037.bz2
```

Statystyki korpusu treningowego:

- Liczba artykułów: `40 709`
- Liczba słów: `30 000 000`
- Liczba znaków: `219 651 686`
- Liczba bajtów UTF-8: `231 522 600`

Podczas przygotowania danych usunięte zostały oczywiste artefakty dumpów Wiki, m.in. fragmenty LaTeX, znaczniki `<ref>`, `<math>`, szablony, tabele i surowy markup typu `[[...]]` oraz `{{...}}`.

## Artefakt

Wygenerowany artefakt tokenizatora znajduje się w pliku:

```text
tokenizer.json
```

Format jest zgodny ze stylem Hugging Face `tokenizers`, co pozwala używać wytrenowanych "wag" tokenizatora bezpośrednio z biblioteką `tokenizers` od HF. W przypadku BPE tymi wagami są głównie słownik `model.vocab` oraz lista reguł scalania `model.merges`:

- `model.type = "BPE"` oznacza model Byte Pair Encoding.
- `model.vocab` mapuje reprezentację tokenu na jego identyfikator liczbowy.
- `model.merges` zawiera uporządkowaną listę reguł scalania BPE.
- `pre_tokenizer.type = "ByteLevel"` oznacza bajtowy tokenizer zgodny z mapowaniem GPT-2.
- `decoder.type = "ByteLevel"` pozwala odwrócić tokeny z powrotem do tekstu.

Słownik składa się z dwóch części:

- `256` tokenów bazowych odpowiadających pojedynczym bajtom.
- `13 744` tokenów nauczonych przez BPE jako kolejne reguły scalania.

Reguła merge mówi, że dwa istniejące tokeny mogą zostać zastąpione nowym tokenem. Na przykład uproszczona reguła:

```text
Ġ p
```

oznacza, że token reprezentujący spację (`Ġ`) oraz token `p` mogą zostać scalone w jeden nowy token. Kolejność reguł jest ważna, bo wcześniejsze reguły mają niższy rank i są stosowane jako pierwsze.

## Główne statystyki tokenizatora

- Typ: `Byte-level BPE`
- Rozmiar słownika: `14 000`
- Liczba reguł merge: `13 744`
- Liczba tokenów bazowych bajtowych: `256`
- Liczba tokenów wieloznakowych nauczonych przez BPE: `13 744`
- Średnia długość wpisu słownikowego: `5.91` znaków w reprezentacji byte-level
- Najdłuższy wpis słownikowy: `23` znaki w reprezentacji byte-level

Przykładowe długie tokeny nauczone przez BPE obejmują częste polskie formy i wyrażenia z korpusu, np. odpowiedniki fragmentów takich jak `południowoafrykański`, `południowokoreański`, `niepodległościowy`, `najprawdopodobniej`, `charakterystyczne`, `przedsiębiorstwo`.

## Metryki na tekście ewaluacyjnym

Pomiar wykonany na całości Pana Tadeusza:

- Liczba znaków: `447 334`
- Liczba słów: `69 095`
- Liczba tokenów: `163 043`
- Współczynnik kompresji znakowej: `63.55%`
- Tekst w tokenach jest około `2.74` razy krótszy niż tekst liczony znakowo.
- Fertility: `2.360 tok/słowo`

Interpretacja: wynik `2.360 tok/słowo` oznacza, że jedno polskie słowo wymaga średnio około 2.36 tokenu. Dla języka fleksyjnego, takiego jak polski, jest to rozsądny wynik przy słowniku rzędu kilkunastu tysięcy tokenów.

## Brak lowercasingu

Nie zamieniamy tekstu na małe litery przed treningiem tokenizatora. To świadoma decyzja: lowercasing zwykle poprawia kompresję i zmniejsza liczbę wariantów słów, ale jednocześnie usuwa informację semantyczną.

## Ograniczenia

Tokenizer był trenowany głównie na Wikipedii, więc jego słownik jest dostosowany do stylu encyklopedycznego. Może być mniej optymalny dla poezji, czatów, tekstów prawnych, forów internetowych, kodu źródłowego lub tekstów silnie potocznych.

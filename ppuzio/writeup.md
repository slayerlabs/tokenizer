# Podsumowanie tokenizatora Byte-Level BPE

## Cel

Celem jest prosty **byte-level BPE** tokenizer dla polskiego tekstu literackiego, zaimplementowany od zera w czystym Pythonie (`bpe_skeleton.py`). Tokenizer ma:

- kodować dowolny tekst UTF-8 bez tokenów `<UNK>`,
- zachować pełną odwracalność: `decode(encode(text)) == text`,
- wygenerować artefakt `tokenizer.json` z regułami scalania.

## Podejście

Tekst jest kodowany na bajty UTF-8 (256 tokenów bazowych: `0–255`). Algorytm BPE w pętli:

1. liczy częstość sąsiadujących par tokenów w całym korpusie,
2. scala najczęstszą parę w jeden nowy token,
3. powtarza, aż słownik osiągnie docelowy rozmiar.

W przeciwieństwie do tokenizerów w stylu GPT-2 (np. u Kuby) **nie stosujemy pretokenizacji regexem** ani mapowania `bytes_to_unicode` (`Ġ` itd.). BPE widzi cały tekst jako jeden ciąg bajtów. To upraszcza implementację, ale pozwala na scalanie ponad granicami słów i akapitów — widać to w najdłuższych nauczonych tokenach.

Implementacja encode stosuje reguły merge w kolejności treningu (najniższe ID = najwyższy priorytet), zgodnie z typową inferencją BPE.

## Dane treningowe

Korpus: **„Ogniem i mieczem”, tom pierwszy** — Henryk Sienkiewicz, Wolne Lektury.

- Źródło: [ogniem-i-mieczem-tom-pierwszy.txt](https://wolnelektury.pl/media/book/txt/ogniem-i-mieczem-tom-pierwszy.txt)
- Lokalny plik: `ogniem.txt`

Statystyki korpusu:

| Metryka | Wartość |
|---------|---------|
| Liczba znaków | 791 246 |
| Liczba bajtów UTF-8 | 859 818 |
| Liczba słów (`split()`) | 125 371 |
| Liczba linii | 8 018 |

Korpus jest znacznie mniejszy niż u pozostałych (Wikipedia, DynaWord) — słownik jest więc mocno dopasowany do stylu jednej powieści historycznej.

## Artefakt

Plik `tokenizer.json` zawiera:

- `type`: `"byte_bpe"`
- `vocab_size`: `10 000` (256 bajtów + 9 744 merge'y)
- `merges`: lista reguł `[left_id, right_id, new_id]` — bezpośrednio z `merges.txt`

Format jest bliższy schematowi numerycznemu (jak u Patryka) niż pełnemu JSON Hugging Face (jak u Kuby), ale jest w pełni deterministyczny i łatwy do odtworzenia z istniejącego kodu.

Dodatkowe pliki po treningu:

- `merges.txt` — reguły scalania, jedna na linię
- `token_ids.txt` — tokenizacja całego korpusu po treningu

## Główne statystyki tokenizatora

| Parametr | Wartość |
|----------|---------|
| Typ | Byte-level BPE (bez pretokenizacji) |
| Docelowy rozmiar słownika | 10 000 |
| Tokeny bazowe (bajty) | 256 |
| Reguły merge | 9 744 |
| Średnia długość tokenu | 6,23 bajta |
| Najdłuższy token | 27 bajtów |

Przykłady najdłuższych tokenów (wielowyrazowe fragmenty dialogów — efekt braku ograniczeń na merge):

- `! — rzekł pan Skrzetuski`
- `południowych województwa`
- `wasza książęca mość`

## Metryki na korpusie treningowym

Pomiar na całym `ogniem.txt` po treningu (`token_ids.txt`):

| Metryka | Wartość |
|---------|---------|
| Liczba tokenów | 177 752 |
| Fertility | **1,418 tok/słowo** |
| Średnia liczba znaków na token | 4,45 |
| Współczynnik kompresji znakowej | 77,54% |
| Tekst w tokenach jest ~4,45× krótszy niż liczony znakami | |
| Rekonstrukcja `decode(encode(text))` | **100%** — idealna |
| Tokeny `<UNK>` | **0** |

Niska fertility (w porównaniu do ~2,2–2,6 u tokenizerów z pretokenizacją) wynika częściowo z tego, że BPE łączy całe frazy i dialogi w pojedyncze tokeny. To poprawia „kompresję” na tym korpusie, ale niekoniecznie daje dobrze uogólnialne jednostki pod model językowy.

## Próbka podziału tekstu

| Tekst wejściowy | Tokeny |
|-----------------|--------|
| `Jan Skrzetuski` | `['Jan ', 'Skrzetuski']` |
| `Pan Zagłoba` | `['Pan ', 'Zagłoba']` |
| `Ogniem i mieczem` | `['O', 'gnie', 'm i ', 'mie', 'czem']` |
| `czyli ostatni zajazd na Litwie` | `['czyli ', 'ostatni ', 'za', 'jazd ', 'na ', 'Lit', 'wie']` |
| `Rzeczpospolita` | `['Rzeczpospoli', 'ta']` |
| `Wojska Koronnego` | `['W', 'ojska ', 'Ko', 'ron', 'nego']` |

## Ograniczenia i możliwe usprawnienia

1. **Brak pretokenizacji** — merge'y przechodzą przez spacje, interpunkcję i znaki nowej linii; powstają tokeny wielowyrazowe.
2. **Jeden mały korpus** — słownik jest overfitowany do „Ogniem i mieczem”, nie do ogólnego polskiego.
3. **Wolny trening** — każda iteracja skanuje cały korpus (`O(n × merges)`); na większych danych potrzebna byłaby optymalizacja.
4. **Brak eksportu HF** — `tokenizer.json` nie ładuje się wprost przez `tokenizers.Tokenizer.from_file()`; wymagałby konwersji do formatu ByteLevel + `bytes_to_unicode`.

Kolejne kroki (gdyby rozwijać ćwiczenie): pretokenizacja regex (jak GPT-2), limit długości tokenu, zakaz merge'y ze znakiem nowej linii, większy / zróżnicowany korpus.

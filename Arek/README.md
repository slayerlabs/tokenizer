# Arek — byte-level BPE (naive vs fast)

Dwa warianty tokenizera BPE trenowane od zera w czystym Pythonie (bez bibliotek),
na tym samym polskim korpusie i tych samych rozmiarach słownika — do porównania 1:1.
Alfabet bazowy: 256 bajtów, zero `<UNK>`, round-trip lossless.

## Warianty

- **`naive/`** — BPE bez pre-tokenizacji. Liczy najczęstszą parę sąsiednich bajtów
  w całym ciągu, scala, powtarza. Złożoność `O(n·m)` (przeliczenie par po całym
  korpusie co scalenie) — nie skaluje się na duże korpusy. Wariant dydaktyczny.
- **`fast/`** — BPE produkcyjny (styl GPT-2 / minbpe): regex pre-tokenizacja
  (scalanie nie przekracza granic słów), bucketing po unikalnych fragmentach,
  liczniki inkrementalne (po scaleniu aktualizowane są tylko słowa zawierające
  scaloną parę). Znacznie szybszy trening przy tej samej jakości.

Rozmiary słownika: `naive` — 512/1024/2048; `fast` — 512/1024/2048/4096/8192/15000.

## Korpus

Polska literatura z domeny publicznej (Wolne Lektury): Prus „Lalka" (t. I–II),
Sienkiewicz „Ogniem i mieczem", Wyspiański „Wesele", Shakespeare „Romeo i Julia"
(tłum.). Trening ~2,71 M znaków, 22,5% diakrytyków. Held-out do ewaluacji:
Mickiewicz „Pan Tadeusz" (445 637 znaków, ten sam rejestr, brak przecieku).

## Metryki (held-out „Pan Tadeusz")

- **zn/tok** — znaki na token (wyżej = gęstszy zapis).
- **fertility** — tokeny na słowo (niżej = krótszy zapis).
- **Rényi efficiency** — równomierność rozkładu tokenów, zakres 0–1 (α=2,5).
  Niżej = kilka tokenów dominuje, więcej rzadkich/niedotrenowanych.

| wariant      | vocab | tokeny  | zn/tok | fertility | Rényi |
|--------------|-------|---------|--------|-----------|-------|
| naive-512    | 512   | 253 549 | 1,76   | 3,718     | 0,768 |
| naive-1024   | 1024  | 210 983 | 2,11   | 3,094     | 0,767 |
| naive-2048   | 2048  | 183 067 | 2,43   | 2,685     | 0,743 |
| fast-512     | 512   | 249 784 | 1,78   | 3,663     | 0,730 |
| fast-1024    | 1024  | 209 638 | 2,13   | 3,074     | 0,662 |
| fast-2048    | 2048  | 184 953 | 2,41   | 2,712     | 0,585 |
| fast-4096    | 4096  | 165 438 | 2,69   | 2,426     | 0,518 |
| fast-8192    | 8192  | 148 451 | 3,00   | 2,177     | 0,460 |
| fast-15000   | 15000 | 137 083 | 3,25   | 2,010     | 0,418 |
| GPT-2 (ref.) | 50257 | 257 901 | 1,73   | 3,782     | 0,391 |

## Obserwacje

1. Wraz ze wzrostem vocab kompresja rośnie (zn/tok 1,78 → 3,25), a Rényi spada
   (0,730 → 0,418). Zależność jest liniowa, bez ostrego progu: każde podwojenie
   słownika daje ok. +0,28 zn/tok kosztem ok. −0,06 Rényi.
2. Wszystkie warianty przewyższają GPT-2 na tekście polskim — gęstszy zapis
   (fast-2048: 2,41 vs 1,73, czyli 1,39×) i wyższe Rényi (0,585 vs 0,391).
   GPT-2 ma 50 257 tokenów, ale był uczony na angielskim, więc na polskim
   fragmentuje słowa na krótkie tokeny.
3. fast-15000 osiąga Rényi 0,418 — poziom GPT-2 (0,391). Dla korpusu 2,71 M to
   za duży słownik (za dużo rzadkich tokenów). Rozsądny zakres dla tego korpusu:
   2048–4096.
4. `naive` ma wyższe Rényi niż `fast` przy tym samym vocab (2048: 0,743 vs 0,585),
   ale scala przez granice słów — ok. 12,4% tokenów to fragmenty cross-boundary
   (spacja+słowo, interpunkcja, newline). Wyższe Rényi nie oznacza tu lepszej
   jakości: równomierność wynika ze sklejek formatujących, nie z czystych
   jednostek językowych. Rozstrzygnąć może dopiero ewaluacja downstream
   [do zweryfikowania].

## Format i wczytywanie

Każdy plik to JSON zbliżony do formatu HuggingFace: blok `model` z `vocab`
(reprezentacja `bytes_to_unicode` → id) i `merges` (lista `"lewy prawy"` w kolejności
uczenia). Odtworzenie reguł scalania do postaci `{(id_lewego, id_prawego): id_scalonego}`:

```python
import json
m = json.load(open("fast/2048.json", encoding="utf-8"))["model"]
vocab = m["vocab"]                       # repr -> id
merges = {}
for line in m["merges"]:                 # kolejność = rank
    l, r = line.split(" ")
    merges[(vocab[l], vocab[r])] = vocab[l + r]
```

## Autor

Arek — Kurs Tokenizer, Slayer Labs.

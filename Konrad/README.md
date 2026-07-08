# Tokenizer BPE — Konrad

Bajtowy Byte-Pair Encoding (jak GPT-2) uczony na słowniku **SJP.pl** (~178 mln znaków).

## Zawartość

- **`tokenizer.json`** — gotowy tokenizer w formacie Hugging Face (byte-level BPE, 200 merge'y, vocab 456). Ładuje się przez:
  ```python
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file("tokenizer.json")
  ```
- **`tokens_per_word.html`** — długość słowa vs liczba tokenów; modele uczone na rosnących fragmentach SJP (10 tys. → całość), 50 merge'y.
- **`tokens_per_word_labels.html`** — to samo dla zestawu polskich słów (2–20 znaków), słowa jako etykiety na punktach.
- **`tokens_per_merges.html`** — wpływ liczby merge'y (**50 / 100 / 200**) przy uczeniu na **całym** SJP. Więcej merge'y → mniej tokenów na słowo (539 → 453 → 395 tok. łącznie).

## Jak powstało

1. Implementacja BPE (`bpe.py`): bajty UTF-8 → liczenie par → scalanie najczęstszej pary N razy.
2. Trening na SJP i generowanie raportów HTML (Bootstrap + ECharts) z `report.py`.
3. Model 200-merge (jeden trening, modele 50/100 skrócone z niego) wyeksportowany do `tokenizer.json` — zgodność z własnym enkoderem zweryfikowana biblioteką `tokenizers`.

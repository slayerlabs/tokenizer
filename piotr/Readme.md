# W1 - tokenizer (Piotr) - lookup

Eksperymenty z bajtowym tokenizerem BPE (`TokenizerUTF8`): kompresja zależnie od korpusu treningowego i rozmiaru słownika.

## `v1/`

Wersja pierwsza. `opracowanie.md`: podsumowanie treningu BPE na 4 korpusach (lorem-ipsum, nad-niemnem, wolne-lektury, wikipedia-pl) przy słownikach 1k i 5k - 8 przebiegów, z tabelami kompresji i wnioskami (tekst chiński `cn_moon` = 1.0 wszędzie). `data/`: statystyki per przebieg (`*_stats.csv`) i odtworzenia tekstów testowych (`in_out.csv`).

## `v2/`

Rozszerza v1 o korpus chiński (wikipedia-zh) i angielski (random-text-files), trzeci słownik 10k oraz dotrenowywanie z checkpointów (warm-start) - łącznie 6 korpusów x 3 słowniki = 18 przebiegów. `opracowanie.md`: pełne wyniki, dowód bezstratności i wnioski. `data/`: `*_stats.csv` per przebieg oraz wspólny `in_out.csv` z odtworzeniami.

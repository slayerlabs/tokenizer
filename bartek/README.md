# Polski tokenizer BPE od zera

Byte-level BPE w czystym Pythonie, trenowany na zróżnicowanym korpusie polskim
(172 MB treningu i 10 MB held-out). Repo zawiera eksperyment z rozmiarem słownika
i cztery pytania badawcze. Raport: [`report/REPORT.md`](report/REPORT.md).

## Uruchomienie (jedna komenda)

```bash
make -j6 all
```

Komenda wykonuje kolejno:

1. tworzy `.venv` i instaluje zależności z `requirements.txt`;
2. pobiera dane (`scripts/download.py`) do `data/raw/`;
3. czyści dane, deduplikuje i dzieli na train/held-out (`scripts/clean.py`) do `data/clean/`;
4. trenuje 6 tokenizerów (równolegle przy `-j6`) do `artifacts/`;
5. liczy metryki i rysuje wykresy do `results/` i `report/figures/`;
6. eksportuje główny tokenizer do formatu HuggingFace (`tokenizer.json`) i sprawdza,
   że daje identyczne id jak nasza implementacja.

Wymagania: Python ≥ 3.11, ok. 1 GB miejsca na dane, ok. 8 GB RAM. Pełny przebieg
na 16 rdzeniach trwa ok. 15 minut, a większość tego czasu zajmuje pobieranie danych.
Poszczególne etapy można uruchamiać osobno: `make data`, `make train`, `make eval`.

## Struktura

| Ścieżka | Zawartość |
|---|---|
| `src/bpe/trainer.py` | trening BPE: warsztatowy algorytm z przyrostowymi licznikami par i kopcem |
| `src/bpe/tokenizer.py` | encode/decode, zapis i odczyt `merges.txt` / `vocab.json` |
| `src/bpe/pretokenize.py` | regex w stylu GPT-4 z trzema wariantami obsługi cyfr |
| `scripts/` | pobieranie, czyszczenie, trening, ewaluacja, eksperymenty do pytań badawczych 3.1–3.4 |
| `artifacts/<run>/v<size>/` | **artefakty**: `merges.txt`, `vocab.json` i `tokenizer.json` (HF, tylko `main`) dla każdego rozmiaru |
| `report/PREREGISTRATION.md` | hipotezy zapisane przed treningiem |
| `report/REPORT.md` | raport |

## Artefakty

`artifacts/main/v{1024,2048,4096,8192,16384,32768,65536}/` to główny tokenizer
w 7 rozmiarach i jako jedyny jest w repo. Tokenizery eksperymentów `make train`
odtwarza lokalnie w `artifacts/`:

| Katalog | Pytanie badawcze | Opis |
|---|---|---|
| `half_a`, `half_b` | 3.1 stabilność słownika | dwie rozłączne połówki korpusu |
| `strip10` | 3.3 bez ogonków | 10% dokumentów bez ogonków |
| `digits_single`, `digits_free` | 3.4 liczby | inne warianty pretokenizacji cyfr |

Format `merges.txt`:

```
<new_id> <left_id> <right_id> <częstość pary przy merge'u> | 'lewy' + 'prawy' -> 'wynik'
```

Kolumna z częstością pozwala odtworzyć, jak rzadkie były późne merge'e. Plik
`vocab.json` mapuje `id` na `{hex, text}`. Tokeny, które nie są poprawnym UTF-8
(fragmenty znaków wielobajtowych), mają w polu `text` zapis `<hex>`.

Plik `tokenizer.json` (tylko `main`) to ten sam tokenizer w formacie HuggingFace
`tokenizers`: regex pretokenizacji jako `Split`, bajty w mapowaniu GPT-2 (`ByteLevel`),
merge'e w kolejności treningu. `scripts/export_hf.py` po eksporcie sprawdza, że id
są identyczne z naszymi na 300 dokumentach held-out.

Użycie:

```python
from bpe.tokenizer import Tokenizer   # PYTHONPATH=src
tok = Tokenizer.load("artifacts/main/v32768")
ids = tok.encode("Zażółć gęślą jaźń")
assert tok.decode(ids) == "Zażółć gęślą jaźń"

# albo przez HuggingFace (te same id):
from tokenizers import Tokenizer as HFTokenizer
hf = HFTokenizer.from_file("artifacts/main/v32768/tokenizer.json")
assert hf.encode("Zażółć gęślą jaźń").ids == ids
```

## Źródła danych i licencje

| Źródło | Licencja / podstawa |
|---|---|
| FineWeb-2 `pol_Latn` (HuggingFaceFW) | ODC-By 1.0 (+ warunki Common Crawl) |
| Wikipedia PL (`wikimedia/wikipedia`, 20231101.pl) | CC BY-SA 3.0 / GFDL |
| Wolne Lektury (wolnelektury.pl) | domena publiczna / CC BY-SA 3.0 |
| Wystąpienia w Sejmie (`PiotrSty/sejm-speeches-corpus`, dane z api.sejm.gov.pl) | art. 4 pkt 2 pr. aut. (dokumenty urzędowe) + ustawa o otwartych danych |
| Akty prawne Dz.U. (api.sejm.gov.pl/eli) | art. 4 pkt 1 pr. aut. (akty normatywne nie są przedmiotem prawa autorskiego) |

Dane pobiera i odtwarza deterministycznie (stałe seedy) `make data`.

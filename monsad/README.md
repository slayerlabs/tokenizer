# Bielik + śląski: rozszerzenie tokenizera zamiast nowego

Wariant zadania „pod prąd": zamiast trenować BPE od zera, **douczam istniejący
tokenizer Bielika v3 (APT4, 32 000 tokenów)** pod język śląski (szl). Powód jest
ten sam, który pada w głównym README — tokenizera nie da się podmienić bez
przetrenowania modelu. Jeśli chcę dostroić gotowego Bielika do śląskiego,
mogę tylko **dopisać** tokeny, nie zmieniając żadnego istniejącego ID.

## Na czym trenowałam

- Korpus śląski w ortografii Ślabikŏrza: 15 997 unikalnych akapitów
  (~926 tys. słów) po odfiltrowaniu szablonowych artykułów (geo/bio) i
  akapitów w innej ortografii.
- Podział: 15 518 akapitów train, **479 akapitów held-out** (żaden nie trafił
  do treningu merge'y).

## Co robi `extend_tokenizer.py`

1. Dodaje litery, których APT4 nie zna w ogóle (`ã ŏ ō ô õ` + wielkie) — bez tego
   każda z nich rozpada się na 2 tokeny bajtowe `<0x..>`.
2. Uczy się 2000 nowych merge'y klasycznym BPE, ale **startując od segmentacji
   Bielika**, nie od bajtów. Nowe reguły idą **na koniec** listy merge'y, więc
   stare zachowują pierwszeństwo.
3. Zapisuje `new_tokens.json`: z jakich starych tokenów składa się każdy nowy —
   embedding nowego tokenu = średnia embeddingów jego składników (start treningu
   nie od szumu).

Merge'y łączą tylko litery (bez bajtów, bez przechodzenia przez `▁`), żeby nie
uczyć się śmieci z interpunkcji.

## Jak mierzyłam

- **Słowo** = fragment po `str.split()` (białe znaki), **znaki** = `len()` w
  code pointach.
- Fertility i znaki/token liczone na held-oucie, ten sam tekst dla wszystkich
  tokenizerów; OpenAI przez `tiktoken` (`encode_ordinary`).
- Testy poprawności: round-trip `decode` vs stary tokenizer na każdym akapicie
  held-outu; zgodność segmentacji z symulacją BPE na 20 tys. słów.
- Regresja na polskim: `pl_check.py`, 20 artykułów polskiej Wikipedii
  (2229 akapitów, ~127,5 tys. słów).

## Ile wyszło

Śląski, held-out (479 akapitów):

| tokenizer | tokeny/słowo | znaki/token |
|---|---:|---:|
| cl100k_base | 2,864 | 2,23 |
| o200k_base | 2,596 | 2,46 |
| Bielik APT4 (bazowy) | 2,788 | 2,29 |
| APT4 + same litery | 2,566 | — |
| **APT4 + litery + 2000 merge'y** | **1,930** | **3,31** |

−30,8% tokenów na słowo, słownik 32 000 → 34 010 (+6%).
Round-trip: 0 różnic / 479 akapitów. Segmentacja: 0 różnic / 20 000 słów.
Użytych nowych tokenów na held-oucie: 1803 z 2010.

Polski (Wikipedia), czy nie zepsułam:

| tokenizer | tokeny/słowo |
|---|---:|
| cl100k_base | 2,826 |
| o200k_base | 2,442 |
| Bielik APT4 | 1,810 |
| APT4 + szl | 1,809 |

Fertility bez zmian, ale **120 z 2229 akapitów (5%) tnie się inaczej** — nowe
merge'y łapią też część polskich słów. Dla modelu to inne ID, więc
na polskim też trzeba chwilę douczyć.

Przykłady po rozszerzeniu: `ôd` → `▁ôd`, `sōm` → `▁sōm`, `Ślōnska` → `▁Ślōnska`,
`miyszkańcōw` → `▁miyszkańcōw`, `nojlepszy` → `▁nojlep · szy`.

## Co mnie zaskoczyło

- **Polski tokenizer na śląskim jest prawie tak zły jak angielski.** APT4 robi
  z polskiego 1,81 tokenu/słowo, ze śląskiego 2,79 — gorzej niż `o200k_base`
  (2,60) i niewiele lepiej niż `cl100k_base`. Język bliski nie wystarczy, jeśli
  ortografia ma znaki, których tokenizer nigdy nie widział.
- **Same litery dają mało** (2,79 → 2,57). Prawie cały zysk to merge'y na
  częste śląskie cząstki (`ōm`, `▁niy`, `▁ôd`, `▁przi`, `▁tyż`).
- **„Dopisz na koniec" nie znaczy „nie ruszaj starego".** Fertility na polskim
  się nie zmienia, ale 5% akapitów dostaje inną segmentację.

## Odtworzenie

```bash
pip install tokenizers tiktoken
python extend_tokenizer.py --tokenizer-dir <katalog z tokenizer.json Bielika> \
    --train train.jsonl --valid valid.jsonl --merges 2000 --out tokenizer-szl
python pl_check.py <bielik>/tokenizer.json tokenizer-szl/tokenizer.json
```

`train.jsonl`/`valid.jsonl` to `{"text": "..."}` po jednym akapicie. Wynik
pomiarów: `report.json`.

# Arek — fast BPE (korpus SpeakLeash ~3 GB)

Byte-level BPE (ten sam wariant `fast` co `../fast/`, ten sam pre-tokenizer),
ale trenowany na **~3 GB czystego SpeakLeash** (`speakleash-tokenizer-5gb-sample`,
shardy 0001–0003, streamowane) zamiast literatury (2,71 M zn.). Cel: pomiar
**dźwigni korpusu** na held-oucie SpeakLeash (nitka `Optymalizacja-Tokenizer`).

- Alfabet bazowy: 256 bajtów, zero `<UNK>`, round-trip lossless.
- Pre-tok: `" ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+"` (GPT-2-style; cyfry jako pełne runy).
- Trening: liczniki inkrementalne + **selekcja best-pair przez kopiec (lazy-heap)**,
  z parytetem 1:1 vs rdzeń referencyjny — skala do 3 GB (31 744 merges w ~236 s).
- `min_freq=3` (prune rzadkich typów; parametr).

## Pliki
| plik | vocab | uwaga |
|---|---|---|
| `32000.json` | 32 000 | format klasy (`model.vocab` + `model.merges`) |
| `64000.json` | 64 000 | jw. |

## Metryki (held-out = shard 0005, ROZŁĄCZNY z treningiem; fertility = tok/słowo, ↓ lepiej)
| vocab | fertility | zn/tok | Rényi (α=2,5) | round-trip | ref. Slayer |
|---|---|---|---|---|---|
| 32 000 | 1,765 | 4,03 | 0,451 | ✅ | 1,771 |
| 64 000 | _(w trakcie)_ | | | | 1,607 |

**Uczciwie (apple-to-apple):** to NASZ held-out z tej samej dystrybucji i NASZA
definicja słowa (`[^\W\d_]+`) — nie dokładny held-out/def. Slayera. Liczby pokazują
**magnitudę dźwigni korpusu** (stary fast 15k na literaturze = 2,388 → tu 1,765);
czysty 1:1 ze Slayerem wymaga jego held-outu/protokołu (α w Rényim ustalamy wg
jego metryki, ale **nie** stroimy α pod wynik — to byłby Goodhart).

## Wczytanie
Jak w `../README.md`: `model.vocab` (repr `bytes_to_unicode` → id) + `model.merges`
(lista `"lewy prawy"` w kolejności uczenia).

# tokenizer

*🇵🇱 polski · [🇬🇧 English](README.en.md)*

Tokenizery z warsztatu **[Slayer](https://slayer.fabryka.ai/)** — otwartego
laboratorium [Fabryka AI](https://fabryka.ai/).

> **Budujemy polskie modele. Otwieramy cały proces.**

Slayer łączy ludzi, compute i metodę: trenuje modele, przygotowuje zbiory i
publikuje pomiary, które da się sprawdzić. Praca idzie trzema torami — **modele**,
**dane**, **pomiary**; artefakty lądują na [HF: SlayerLab](https://huggingface.co/SlayerLab).

To repo jest z warsztatu „Jak zbudować tokenizer dla LLM" — byte-level BPE od
zera, w czystym Pythonie. Tokenizer to pierwsza decyzja w pipelinie pretreningu
i jedna z niewielu, których nie da się cofnąć bez przetrenowania modelu od nowa.

---

## Dlaczego w ogóle tokenizacja

Model nie czyta tekstu znak po znaku z trzech powodów:

1. **Długość sekwencji** — okno kontekstu jest skończone. Znak jako jednostka to
   4–5× dłuższe sekwencje, czyli 4–5× wyższy koszt.
2. **Słowa też nie działają** — słownik eksploduje, zwłaszcza we fleksji.
   `dom · domu · domowi · domem · domach · domowy · domowego` to jedno pojęcie
   i kilkanaście ID. Nowa forma = `<UNK>` = utrata informacji.
3. **Subword to kompromis** — jednostki między znakiem a słowem.

> Tokenizer to **kompresja stratna na poziomie reprezentacji, nie treści**. Uczy
> się, które sekwencje bajtów występują razem na tyle często, że opłaca się dać
> im jeden identyfikator.

## BPE w czterech krokach

1. Zacznij od najmniejszych jednostek (bajty).
2. Policz częstość wszystkich **sąsiadujących par**.
3. Najczęstszą parę zlej w nowy token, zapisz regułę (merge).
4. Powtórz N razy, gdzie `N = docelowy słownik − alfabet bazowy`.

Na korpusie `aaabdaaabac`:

| para | merge | korpus |
|---|---|---|
| `aa` → `Z` | 1 | `ZabdZabac` |
| `ab` → `Y` | 2 | `ZYdZYac` |
| `ZY` → `X` | 3 | `XdXac` |

Trzy merge'e, słownik `{a, b, c, d, Z, Y, X}`.

**Trening BPE = uczenie się słownika i reguł merge. Tokenizacja = aplikowanie
tych reguł w tej samej kolejności, w jakiej powstały.** Na tym drugim ludzie
najczęściej się wykładają — merge'y nie idą „od najczęstszego", tylko od
najwcześniej wyuczonego.

## Czemu bajty, a nie znaki

- Alfabet znakowy Unicode to ~150 tys. code pointów do pokrycia (albo `<UNK>`
  na wszystko, czego nie ma — emoji, rzadkie alfabety).
- Bajtów UTF-8 jest **zawsze 256**. Każdy tekst, dowolny język i alfabet,
  rozkłada się na bajty **bez wyjątków i bez `<UNK>`**.
- Koszt: znak spoza ASCII to 2–4 bajty zamiast jednego. Ale BPE szybko zleje
  częste sekwencje z powrotem — **jeśli** są częste w korpusie treningowym.

To ostatnie „jeśli" jest sednem: tokenizer trenowany głównie na angielskim nie da
polskim znakom własnych tokenów, więc rozpadną się na surowe bajty.

## Dlaczego to boli po polsku

To samo zdanie po polsku kosztuje zwykle **1,5–3× więcej tokenów** niż po
angielsku. Konsekwencje są policzalne: **koszt** (płacisz per token),
**kontekst** (to samo okno mieści mniej) i **jakość** (z drobniejszych kawałków
trudniej złożyć znaczenie).

Ale rozmiar słownika to myląca metryka. Bielik v3 ma tokenizer APT4 o **tych
samych 32 000 tokenów co Mistral** — tyle że trenowany na polskim:

| tokenizer | znaki/token | tokeny (Konstytucja) |
|---|---:|---:|
| Mistral (generyczny) | 2,40 | 747 |
| APT4 (polski) | **4,78** | **375** |

Dwa razy mniej tokenów na tym samym tekście, **bez większego słownika**. Liczy
się **fertility** (tokeny/słowo dla Twojego języka), nie sam rozmiar słownika.

## Pułapki produkcyjne

- **Tokeny specjalne** (`<|endoftext|>`, role) nie wynikają z BPE — dodaje się je
  ręcznie z zarezerwowanymi ID i muszą być atomowe.
- **Rozmiar słownika to hiperparametr.** Większy = krótsze sekwencje, ale większa
  macierz embeddingów i rzadziej widziane tokeny.
- **Tokenizer to osobny artefakt**, wersjonowany niezależnie od wag. Podmiana =
  model produkuje bzdury **bez błędu** (embeddingi indeksowane po ID).
- **Pretokenizacja przed BPE** — produkcyjne tokenizery dzielą tekst regexem po
  granicach słów, rzadko trenują na surowym strumieniu bajtów.

---

## Zadanie

Byte-level BPE **od zera** (bez `tokenizers`, bez `sentencepiece`), wytrenowany
na polskim korpusie i zmierzony na held-oucie: `get_pair_counts` → `merge` →
pętla treningowa → `encode` / `decode`, plus porównanie z tokenizerem
generycznym (`cl100k_base`, `o200k_base`).

## Organizacja i jak dorzucić swój

**Katalog na osobę** (`Arek/`, `ola/`, `ppuzio/`, …), w środku artefakt
tokenizera, kod i writeup. Nie ma narzuconej struktury — byle dało się odtworzyć.
Zrób forka, dodaj swój katalog, otwórz PR do `main`.

Dobry writeup odpowiada na cztery pytania: **na czym trenowałeś** (korpus i jego
rozmiar), **jak mierzyłeś** (held-out, definicja słowa), **ile wyszło**
(fertility ↓, znaki/token ↑, round-trip) i **co cię zaskoczyło**. Dwie uwagi:

- **Porównuj z czymś.** Sama liczba nic nie mówi — dopiero obok `cl100k_base` na
  tym samym tekście widać, ile daje trening na polskim.
- **Wynik negatywny to też wynik.** Jeśli hipoteza się nie potwierdziła, napisz
  to wprost.

## Materiały

- [Slayer](https://slayer.fabryka.ai/) · [Fabryka AI](https://fabryka.ai/) · [HF: SlayerLab](https://huggingface.co/SlayerLab)
- Karpathy, „Let's build the GPT Tokenizer" · [minbpe](https://github.com/karpathy/minbpe)

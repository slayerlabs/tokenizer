
Co się zazwyczaj mierzy?
• Fertility: Ile średnio tokenów przypada na jedno słowo w tekście.
• Compression Ratio (Chars/Token lub Bytes/Token): Ile znaków/bajtów mieści się w jednym tokenie.
• Rényi Efficiency: Jak równomiernie wykorzystywany jest cały słownik (czy nie ma pustych/martwych tokenów).
• Token Fragmentation / Boundary Preservation: Jaki procent kluczowych terminów domenowych (np. 0.001 mm, H7, Ra 0.8) przetrwał w całości jako pojedynczy token, zamiast zostać pociętym na kawałki.

# Zestaw 4 hipotez badawczych

## Hipoteza 1 Odarchaizowanie korpusu ogólnego:
Zastąpienie w korpusie treningowym tokenizera BPE tekstów klasycznych ich uwspółcześnionymi odpowiednikami obniży współczynnik fertility oraz poprawi kompresję (Bytes/Token) na współczesnych domenach językowych, przy kontrolowanym spadku wydajności tokenizacji tekstów historycznych.

## Hipoteza 2 Domenowy pre-tokenizer i ochrona jednostek:
Zastosowanie domenowego pre-tokenizera chroniącego zapisy wymiarowe, tolerancje oraz oznaczenia technologiczne (np. `0.001 mm`, `0.01 mm`, `H7`, `Ra 0.8`, `M12x1,5`) obniży fertility w tekstach technicznych przy jednoczesnym wzroście zachowania granic jednostek (Boundary Preservation).

### Hipoteza 2.1 Ewaluacja na modelu downstream: 
Mały model językowy (SLM) wytrenowany z wykorzystaniem domenowego tokenizera osiągnie wyższą dokładność (Test Accuracy) na zadaniach wymagających operowania na tolerancjach i żargonie warsztatowym (np. przeliczanie `„setek”` i `„mikronów”` na `milimetry`) w porównaniu do modelu z bazowym tokenizerem.

## Hipoteza 3: Równomierność słownika i code-switching PL-EN:
Dorzucenie małego, celowanego korpusu języka angielskiego w domenie technicznej (gdzie język polski naturalnie miesza się z angielskim) poprawi równomierność wykorzystania słownika — zwiększając efektywność Rényiego (Rényi efficiency) oraz zmniejszając odsetek nieużywanych tokenów na dwujęzycznych tekstach technicznych (np. odmiana zapożyczeń typu `offset` / `offsetu` / `offsetem`).

## Hipoteza 4: Gęstość słownika i współdzielenie prefiksów (rdzeni ortograficznych) PL-EN:
Obecność celowanego korpusu technicznego PL-EN pozwoli algorytmowi BPE wyodrębnić wspólne `subwordy` dla `zapożyczeń` o wspólnym prefiksie ortograficznym (np. `toleranc-` dla `tolerancja` / `tolerance`, `kalibr-` dla `kalibracja` / `calibration`, `offset-` dla `offset` / `offsetu`), co zwiększy gęstość informacyjną słownika (wyższy Compression Ratio oraz wyższy odsetek współdzielonych tokenów między językami) przy stałym rozmiarze słownika.

### Uwaga:
Oczekuję, że nie zmniejszy to fragmentacji spowodowanej różnicami pisowni (jak w parze `mikrometr` vs `micrometer`).

### Założenie negatywne hipotezy:
Zjawisko to dotyczy wyłącznie ciągów znaków identycznych od lewej strony.
Oczekuję, że dorzucenie korpusu EN nie zmniejszy fragmentacji dla par pojęć o rozbieżnej pisowni
na wczesnym etapie słowa (np. `mikro`+`metr` vs `micro`+`meter`), dla których algorytm BPE
wygeneruje osobne tokeny rdzeniowe (`mikro` oraz `micro`).

# Podsumowanie: Jaka metryka weryfikuje którą hipotezę?

| Hipoteza | Główny czynnik (zmienna) | Kluczowe metryki weryfikacji |
| :--- | :--- | :--- |
| **H1** | Zamiana klasyki na współczesny PL | **Fertility**, **Compression Ratio** (współczesny PL vs. klasyczny PL) |
| **H2 (+ H2.1)** | Reguły regex w pre-tokenizerze | **Boundary Preservation (%)**, **Fertility**, a na modelu: **Test Accuracy** |
| **H3** | Domieszka technicznego EN (code-switching) | **Rényi Efficiency**, odsetek aktywnych/martwych tokenów |
| **H4** | Wspólne rdzenie techniczne PL-EN | **Compression Ratio**, **Continued Words Rate**, analiza wspólnych tokenów rdzeniowych |

# Projekt eksperymentu:
Podejście ablacyjne — zmieniamy jeden czynnik naraz przy stałym `vocab_size` (`4k`, `16k`, `32k`), szczegóły w [experiment_design.md](experiment_design.md):

- **Baseline (`T0`)**: BPE wytrenowany na surowym korpusie klasyki PL (`15 MB` Wolne Lektury) + polski pre-tokenizer (`--mode polish`).
- **Eksperyment 1 (`T1 H1-modern` / `T1 PL`)**: zamiana klasyki na uwspółcześniony odpowiednik 1:1 (`15 MB` po Gemini) oraz pełny współczesny korpus PL (`94.6 MB`) z polskim pre-tokenizerem (`--mode polish`) — weryfikacja **H1**.
- **Eksperyment 2 (`T2 PL+CNC`)**: `T1 PL` + domenowy pre-tokenizer regexowy dla CNC/tolerancji (`--mode polish,cnc`) — weryfikacja **H2**.
- **Eksperyment 3 (`T3 PL+EN+CNC`)**: `T2` + domieszka korpusu technicznego EN (`15 MB`, `13.7%` całości) — weryfikacja **H3 i H4**.

### Użyte datasety (`109.6 MB` korpus główny + `2.0 MB` held-out):
- **50.5% (`55.4 MB`)** — współczesny język ogólny (`NKJP` `7.4 MB` + `Polish Dynaword` ogólny `48 MB`), żeby tokenizer nie stał się kaleką poza warsztatem.
- **35.8% (`39.2 MB`)** — polska domena techniczna (`Polish Dynaword` filtr tech `17 MB` + `Wikipedia PL` filtr tech `18 MB` + `80%` transkrypcji YT o tematyce CNC/obróbki metalu `4.2 MB`).
- **13.7% (`15.0 MB`)** — angielska domena techniczna (`Wikipedia EN` filtr tech `15 MB`).
- **Klasyka (`15 MB` vs `15 MB` modern)** — Wolne Lektury dla `T0` i czystej ablacji `H1` oraz zbiór kontrolny klasyki.
- **Held-out (`2.0 MB`)** — `20%` odłożonych transkrypcji YT (`1.0 MB`) + `Polish Dynaword` tech po pominięciu pierwszych `20 MB` (`1.0 MB`).


## Przykłady do sprawdzenia intuicji:

### 1 Analizowany Tekst:
`tolerancja do jednej setki albo do 0.01 mm`

### APT4 (Bielik): Tokenizer Polski 
13 tokenów
toleran cja ␣do ␣jednej ␣setki ␣albo ␣do ␣0 . 0 1 ␣mm

### Bielik 11B (Mistral) Tokenizer wielojęzyczny
17 tokenów
`toler anc ja ␣do ␣jed nej ␣set ki ␣al bo ␣do ␣ 0 . 0 1 ␣mm`

### GPT-2 Tokenizer uniwersalny
18 tokenów 
`t oler anc ja ␣do ␣j ed ne j ␣set ki ␣al bo ␣do ␣0 . 01 ␣mm`

### 2 Analizowany tekst:
`narzędzie wkładamy do maszyny i zapisujemy wartość offsetu w bibliotece z offsetami`

### Bielik (APT4) Polski tokenizer
12 tokenów 
`zapisu jemy ␣wartość ␣off se tu ␣w ␣bibliotece ␣z ␣off se tami`

### Bielik 11B (Mistral) Tokenizer wielojęzyczny
16 tokenów 
`zap is uj emy ␣w arto ść ␣offset u ␣w ␣bibli ote ce ␣z ␣offset ami`

### GPT-2 Tokenizer uniwersalny
20 tokenów 
`z ap is uj emy ␣wart o Å Ľ ć ␣offset u ␣w ␣b ibli ote ce ␣z ␣offset ami`



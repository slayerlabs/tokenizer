# Projekt eksperymentu (Experiment Design)

## 1. Cel i własna implementacja tokenizera
Celem eksperymentu jest sprawdzenie, jak dobór korpusu (współczesny vs klasyczny polski), reguł pre-tokenizacji (ochrona zapisów technicznych CNC) oraz kontrolowana domieszka języka angielskiego wpływają na jakość tokenizacji w małym modelu językowym (SLM).

Wszystkie badane tokenizery zostały wytrenowane **od zera** własną implementacją Byte-level BPE ([`bpe_basic_2.py`](bpe_basic_2.py)), bez użycia zewnętrznych bibliotek treningowych (`tokenizers`, `sentencepiece`), w trzech rozmiarach słownika:
- **4k** (`4 096` tokenów),
- **16k** (`16 384` tokenów),
- **32k** (`32 768` tokenów).

Każdy model zapisuje `merges.txt`, `vocab.json` oraz `tokenizer.json` w osobnym katalogu.

---

## 2. Korpusy treningowe, przygotowanie danych i uzasadnienie

| Kategoria | Zbiór / Źródło | Rozmiar | Sposób przygotowania i powód użycia |
| :--- | :--- | :---: | :--- |
| **Klasyka PL (T0 / H1)** | Wolne Lektury (próbka surowa) | **15.0 MB** | Surowe fragmenty literatury dawnej (wycięte 1:1 z większej próbki 30 MB). Służy jako punkt odniesienia dla archaicznej polszczyzny. |
| **Współczesny PL 1:1 (H1)** | Wolne Lektury (odarchaizowane) | **15.0 MB** | Dokładnie te same fragmenty przepisane przez LLM (`gemini-3.5-flash-lite`) na współczesną polszczyznę. Eliminuje różnicę tematyki — jedyną zmienną jest epoka języka. |
| **Współczesny PL ogólny (~50.5%)** | NKJP (Podkorpus Milionowy) + Polish Dynaword (ogólny) | **55.4 MB** *(7.4 + 48 MB)* | Zapewnia bazową znajomość współczesnej polszczyzny ogólnej. |
| **Polska domena techniczna (~35.8%)** | Polish Dynaword (filtr tech) + Wikipedia PL (filtr tech) + transkrypcje YT (80% filmów o tematyce cnc, obrobka metalu etc.) | **39.2 MB** *(17 + 18 + 4.2 MB)* | Odfiltrowane po rdzeniach obróbki skrawaniem i metrologii (`obróbk`, `skrawan`, `tokar`, `frez`, `tolerancj`, `gwint`, `cnc` itd.) + mówiony żargon warsztatowy. |
| **Angielska domena techniczna (~13.7%)** | Wikipedia EN (filtr tech) | **15.0 MB** | Odfiltrowana po słowach kluczowych (`machining`, `cnc`, `milling`, `lathe`, `tolerance`, `metrology` itd.) pod code-switching PL-EN. |
| **Zbiór testowy (*held-out*)** | Transkrypcje YT (20% odłożonych filmów) + Polish Dynaword (filtr tech, po pominięciu pierwszych 20 MB) | **2.0 MB** *(1.0 + 1.0 MB)* | Dane niewidoczne podczas treningu, łączące język mówiony warsztatowy i pisany techniczny. |
| **Zbiór kontrolny klasyki (`--classic-file`)** | Wolne Lektury (8 niewidzianych utworów, m.in. Słowacki, Sienkiewicz, Kraszewski, Wyspiański) | **1.93 MB** | Niewidziany podczas treningu zbiór klasyki do pomiaru kontrolowanego spadku w H1 (`ClsFert`, `ClsB/T`). |

**Uzasadnienie proporcji głównego korpusu (`109.6 MB`):**
- **50.5%** współczesny PL ogólny,
- **35.8%** polska domena techniczna,
- **13.7%** angielska domena techniczna.

---

## 3. Warianty ablacyjne (T0 → T3)
Eksperyment ma charakter ablacyjny — w każdym kolejnym kroku zmieniany jest jeden czynnik przy stałym `vocab_size`:

1. **`t0_klasyka_polish`** — surowa klasyka PL (`15 MB`) + polski pre-tokenizer (`--mode polish`). Ten sam pre-tokenizer co w `T1`, dzięki czemu przejście `T0 → T1 (H1-modern)` izoluje wyłącznie zmianę epoki języka.
2. **`t1_h1_modern`** — uwspółcześniona klasyka PL (`15 MB`) + polski pre-tokenizer (`--mode polish`).
3. **`t1_pl_polish`** — pełny współczesny korpus PL (`94.6 MB`, bez EN) + polski pre-tokenizer (`--mode polish`, chroniący skróty typu `m.in.`, złożenia z dywizem i liczby dziesiętne).
4. **`t2_pl_cnc`** — ten sam korpus PL (`94.6 MB`, bez EN) + domenowy pre-tokenizer CNC (`--mode polish,cnc`, izolujący `0.001 mm`, `H7`, `Ra 0.8`, `M12x1,5`, `±0.02`).
5. **`t3_pl_en_cnc`** — korpus PL + 13.7% EN tech (`109.6 MB`) + domenowy pre-tokenizer CNC (`--mode polish,cnc`).

---

## 4. Tokenizer porównawczy (Baseline)
Wszystkie warianty są porównywane na tym samym zbiorze *held-out* z zewnętrznym tokenizerem **`cl100k_base`** (OpenAI GPT-4, `tiktoken`, rozmiar słownika: **100 277** tokenów). Pozwala to ocenić, ile realnie daje trening na polskim korpusie domenowym mimo 3–25× mniejszego słownika.

---

## 5. Badane hipotezy

### H1 — Odarchaizowanie korpusu ogólnego
- **Hipoteza:** Zastąpienie w korpusie treningowym tokenizera BPE tekstów klasycznych ich uwspółcześnionymi odpowiednikami obniży współczynnik *Fertility* oraz poprawi kompresję (*Bytes/Token*) na współczesnych domenach językowych, przy kontrolowanym spadku wydajności tokenizacji tekstów historycznych.
- **Co testujemy:** porównujemy model uczony na surowej klasyce (`T0`) z modelem uczonym na dokładnie tych samych fragmentach po uwspółcześnieniu (`T1 H1-modern`) oraz z modelem na pełnym współczesnym korpusie (`T1 PL`).
- **Oczekiwany rezultat:** spadek *Fertility* i wzrost *Bytes/Token* na współczesnym technicznym *held-oucie* (`dane_eval/`), przy jednoczesnym niewielkim pogorszeniu tych samych metryk na tekstach klasycznych (`ClsFert` ↑, `ClsB/T` ↓).

### H2 — Domenowy pre-tokenizer i ochrona jednostek CNC
- **Hipoteza:** Zastosowanie domenowego pre-tokenizera chroniącego zapisy wymiarowe, tolerancje oraz oznaczenia technologiczne (np. `0.001 mm`, `0.01 mm`, `H7`, `Ra 0.8`, `M12x1,5`, `±0.02`) obniży *Fertility* w tekstach technicznych przy jednoczesnym wzroście zachowania granic jednostek (*Boundary Preservation*).
- **Co testujemy:** porównujemy `T1 (PL)` (polski pre-tokenizer bez reguł CNC) z `T2 (PL+CNC)` (ten sam korpus PL, ale z regułami regex izolującymi zapisy CNC przed BPE).
- **Oczekiwany rezultat:** wzrost *Boundary Preservation (%)* oraz spadek *Fertility* i *Continued Words Rate (%)* na tekstach technicznych.

### H3 — Równomierność słownika i code-switching PL-EN
- **Hipoteza:** Dorzucenie małego, celowanego korpusu języka angielskiego w domenie technicznej (gdzie język polski naturalnie miesza się z angielskim) poprawi równomierność wykorzystania słownika — zwiększając efektywność Rényiego (*Rényi Efficiency*) oraz zmniejszając odsetek nieużywanych tokenów na dwujęzycznych tekstach technicznych (np. odmiana zapożyczeń typu `offset` / `offsetu` / `offsetem`).
- **Co testujemy:** porównujemy `T2 (PL+CNC)` (wyłącznie polski korpus) z `T3 (PL+EN+CNC)` (ten sam korpus PL + 13.7% technicznego EN).
- **Oczekiwany rezultat:** wyższe *Rényi Efficiency* oraz wyższy odsetek aktywnych tokenów słownika (`Active%`) przy stałym `vocab_size`.

### H4 — Gęstość słownika i współdzielenie prefiksów (rdzeni ortograficznych) PL-EN
- **Hipoteza:** Obecność celowanego korpusu technicznego PL-EN pozwoli algorytmowi BPE wyodrębnić wspólne *subwordy* dla zapożyczeń o wspólnym prefiksie ortograficznym (np. `toleranc-` dla `tolerancja` / `tolerance`, `kalibr-` dla `kalibracja` / `calibration`, `offset-` dla `offset` / `offsetu`), co zwiększy gęstość informacyjną słownika (wyższy *Compression Ratio* i niższy *Continued Words Rate*) przy stałym rozmiarze słownika.
- **Założenie negatywne:** zjawisko to dotyczy wyłącznie ciągów znaków identycznych od lewej strony. Dorzucenie korpusu EN **nie** zmniejszy fragmentacji dla par o rozbieżnej pisowni na wczesnym etapie słowa (np. `mikrometr` vs `micrometer`), dla których BPE wygeneruje osobne tokeny rdzeniowe (`mikro` vs `micro`).
- **Co testujemy:** porównujemy `T2` vs `T3` pod kątem kompresji oraz tokenizacji 3 par pozytywnych i 1 pary negatywnej.

| Hipoteza | Porównywane modele | Kluczowe metryki |
| :--- | :--- | :--- |
| **H1 (Odarchaizowanie)** | `T0` vs `T1 (H1-modern)` oraz `T1 (PL)` | **Fertility ↓**, **Bytes/Token ↑** (współczesny PL) vs `ClsFert`, `ClsB/T` (klasyka) |
| **H2 (Pre-tokenizer CNC)** | `T1 (PL)` vs `T2 (PL+CNC)` | **Boundary Preservation (%) ↑**, **Fertility ↓**, **Continued Words Rate (%) ↓** |
| **H3 (Code-switching PL-EN)** | `T2 (PL+CNC)` vs `T3 (PL+EN+CNC)` | **Rényi Efficiency ↑** ($\alpha = 2.0$), **Active Tokens (%) ↑** |
| **H4 (Wspólne rdzenie PL-EN)** | `T2` vs `T3` | **Bytes/Token ↑**, **Continued Words Rate ↓**, **Stem%**, kontrola negatywna `mikrometr`/`micrometer` |

---

## 6. Jak mierzymy — definicje metryk i interpretacja zmian

Wszystkie pomiary wykonywane są w [`evaluate_tokenizer.py`](evaluate_tokenizer.py) na tym samym zbiorze *held-out* (`dane_eval/`, 2.0 MB) oraz na kontrolnym zbiorze klasyki (`--classic-file dane_klasyka/klasyka_test.jsonl`, 1.93 MB):

1. **Fertility (`Fert`, tokeny / słowo) i `ClsFert` (na klasyce)**
   - **Jak mierzymy:** łączna liczba wygenerowanych tokenów podzielona przez liczbę słów wydzielonych białymi znakami (`\S+`).
   - **Spadek (↓):** lepiej — model potrzebuje mniej tokenów na jedno słowo (krótsze sekwencje dla transformera, szybsze uczenie i dłuższy efektywny kontekst).
   - **Wzrost (↑):** gorzej — słowa są silniej cięte na drobne kawałki.

2. **Compression Ratio (`Bytes/Token`, `Byt/Tok` oraz `ClsB/T` na klasyce)**
   - **Jak mierzymy:** łączna liczba bajtów UTF-8 tekstu wejściowego podzielona przez łączną liczbę tokenów.
   - **Wzrost (↑):** lepiej — pojedynczy token przenosi średnio więcej informacji (bajtów).
   - **Spadek (↓):** gorzej — tokenizer produkuje krótsze, bardziej rozdrobnione tokeny.

3. **Rényi Efficiency (`Rényi`, $\alpha = 2.0$)**
   - **Jak mierzymy:** znormalizowana entropia Rényiego rzędu $\alpha=2.0$ rozkładu częstości tokenów na zbiorze testowym, w skali $[0, 1]$: $H_2(p) / \log_2(|V|)$.
   - **Wzrost (↑):** lepiej — słownik jest wykorzystywany bardziej równomiernie (mniej „martwych” lub skrajnie dominujących tokenów).
   - **Spadek (↓):** gorzej — rozkład użycia tokenów jest bardziej nierównomierny (np. duży ogon rzadko używanych tokenów przy większym `vocab_size`).

4. **Active Tokens (`Act%`)**
   - **Jak mierzymy:** odsetek unikalnych ID ze słownika, które pojawiły się co najmniej raz podczas tokenizacji zbioru testowego.
   - **Wzrost (↑):** lepiej — większa część budżetu słownika realnie pracuje na danej domenie (mniej „martwych” wag w macierzy embeddingów).
   - **Spadek (↓):** gorzej — duża część słownika pozostaje nieużywana na docelowych tekstach.

5. **Continued Words Rate (`Cont%`)**
   - **Jak mierzymy:** odsetek słów (`\S+`), które po zakodowaniu zostały rozbite na więcej niż 1 token (`len(encode(word)) > 1`).
   - **Spadek (↓):** lepiej — więcej pełnych słów mieści się w jednym tokenie.
   - **Wzrost (↑):** gorzej — więcej słów ulega fragmentacji na podsłowa.

6. **Boundary Preservation (`Bound%`)**
   - **Jak mierzymy:** odsetek kluczowych terminów domenowych (`0.001 mm`, `0.01 mm`, `0.1 mm`, `H7`, `H6`, `Ra 0.8`, `M12x1,5`, `±0.02`), które przetrwały jako dokładnie **1 token**.
   - **Wzrost (↑):** lepiej — zapisy techniczne i tolerancje nie są cięte na pojedyncze cyfry/znaki.
   - **Spadek (↓):** gorzej — jednostki i tolerancje są rozbijane na fragmenty.

7. **Shared Prefix PL-EN (`Stem%`)**
   - **Jak mierzymy:** odsetek par pozytywnych PL-EN (`tolerancja`/`tolerance`, `kalibracja`/`calibration`, `offsetu`/`offset`), które dzielą identyczny pierwszy token prefiksowy (oraz osobne sprawdzenie kontroli negatywnej `mikrometr`/`micrometer`).
   - **Wzrost (↑):** większe współdzielenie rdzeni ortograficznych między polskim a angielskim.
   - **Spadek (↓):** rozdzielenie reprezentacji na osobne tokeny dla obu języków.

8. **Round-trip (`RT%`)**
   - **Jak mierzymy:** odsetek tekstów testowych spełniających `decode(encode(text)) == text`.
   - **100%:** pełna bezstratność kodowania UTF-8; każdy wynik `< 100%` oznacza utratę lub zniekształcenie znaków.

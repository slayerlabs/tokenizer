# Własny Byte-level BPE dla języka polskiego i domeny CNC

Badanie wpływu doboru korpusu, reguł pre-tokenizacji oraz dwujęzycznej domieszki PL-EN na jakość tokenizacji dla małego modelu językowego (SLM) w domenie obróbki skrawaniem i metrologii (CNC).

Wszystkie modele zostały wytrenowane **od zera** własną implementacją Byte-level BPE ([bpe_basic_2.py](bpe_basic_2.py)), bez użycia zewnętrznych bibliotek treningowych (`tokenizers`, `sentencepiece`), w trzech rozmiarach słownika (**4k**, **16k**, **32k**) i porównane na **2.0 MB** zbiorze *held-out* z zewnętrznym baseline'em **`cl100k_base`** (OpenAI GPT-4, `100 277` tokenów).

---

## Podsumowanie wyników

- **Lepsza kompresja i krótsze sekwencje niż `cl100k_base`:** już przy **4k** tokenów własny tokenizer osiąga niższe *Fertility* (`2.096` vs `2.428`), a przy **32k** schodzi do **`1.485`** tokena/słowo i **`4.69`** bajta/token (wobec `2.87` w `cl100k_base`).
- **H1 (Odarchaizowanie korpusu) — potwierdzona:** uwspółcześnienie tej samej 15 MB próbki klasyki obniża *Fertility* i poprawia *Bytes/Token* we wszystkich rozmiarach słownika, a pełny współczesny korpus PL daje najlepszą kompresję na tekstach technicznych przy kontrolowanym spadku na klasyce.
- **H2 (Pre-tokenizer CNC) — częściowo obalona:** sama izolacja regexowa (`0.001 mm`, `H7`, `Ra 0.8`, `M12x1,5`) nie wystarczyła, by rzadkie zapisy CNC weszły do top-32k scaleń BPE (`Bound% = 0.0%`).
- **H3 (Code-switching PL-EN) — potwierdzona dla *Rényi Efficiency*:** domieszka 13.7% technicznego EN podnosi równomierność wykorzystania słownika we wszystkich rozmiarach (`4k / 16k / 32k`).
- **H4 (Wspólne rdzenie PL-EN) — częściowo potwierdzona:** potwierdzono założenie negatywne (rozbieżne pary typu `mikrometr` / `micrometer` rozbijają się na osobne rdzenie).

---

## Dokumentacja projektu

- **[hipotezy.md](hipotezy.md)** — sformułowanie hipotez badawczych (H1–H4, w tym planowane H2.1 na modelu SLM) oraz intuicje początkowe.
- **[experiment_design.md](experiment_design.md)** — projekt eksperymentu: korpusy i ich przygotowanie (`109.6 MB` w proporcjach `50.5% / 35.8% / 13.7%`), warianty ablacyjne `T0 → T3`, baseline `cl100k_base` oraz dokładne definicje metryk.
- **[experiment_results.md](experiment_results.md)** — surowe tabele wyników (`4k / 16k / 32k` na *held-oucie*), wykresy `matplotlib` oraz szczegółowa analiza każdej hipotezy i metryki.
- **[runbook.md](runbook.md)** — instrukcja krok po kroku uruchamiania środowiska, treningu i benchmarków.

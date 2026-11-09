# Prerejestracja: przewidywania i hipotezy (spisane PRZED treningiem)

Przewidywania i hipotezy spisane przed uruchomieniem jakiegokolwiek treningu. Raport
odwołuje się do tych liczb bez ich zmieniania.

## Korpus → tokenizer (przewidywanie do części 1)

Korpus to 172 MB treningowego tekstu: web 41%, Wikipedia 18%, literatura 15%, prawo 13%,
Sejm 12%. Przewiduję, że:

1. **Web zdominuje środek słownika.** Słownictwo usługowo-handlowe i newsowe („oferta”,
   „produktu”, „klientów”) dostanie własne tokeny wcześniej niż słownictwo literackie.
2. **Prawo „kupi” sobie tokeny ponad swój udział.** Jest ekstremalnie powtarzalne („ust.”,
   „pkt”, „Dz. U.”, „rozporządzenia”, „poz.”), więc na held-out prawo będzie miało
   **najniższą** fertility ze wszystkich źródeł, mimo że to tylko 13% danych.
3. **Literatura będzie miała najwyższą fertility.** Archaizmy, rzadka leksyka, dialogi
   i dawna ortografia z Wolnych Lektur.
4. **Sejm dostanie tokeny formułkowe** typu „ Marszałku”, „ Izbo”, „ posłanki”.
5. **Fertility przy 32k na held-out wyjdzie ok. 1,55–1,70 tokenu/słowo.** Benchmark
   z materiałów kursu podaje 1,616 przy 32k na innym korpusie.
6. **Słabości:** angielskie wtrącenia, kod, liczby i nazwy obce (z Wikipedii) będą
   rozbijane mocno; tekst bez ogonków będzie rozbijany mocno.

## Vocab size (część 2)

- Przewiduję malejące zyski: każde podwojenie vocabu da mniejszy spadek fertility niż
  poprzednie. Kolano krzywej spodziewam się w okolicy **16k–32k**.
- Przy 64k ponad 2% vocabu nie zostanie użyte ani razu na held-out (~10 MB).

## Pytania badawcze

Metryki: *fertility* = liczba tokenów / liczba słów (słowo = segment rozdzielony białym
znakiem), *compression* = bajty UTF-8 / liczba tokenów. Wszystko liczone na held-out.

### 3.1 Stabilność słownika: od której rangi merge'e są szumem próbkowania?
- **Metoda:** dwa tokenizery na rozłącznych, losowych połowach korpusu treningowego
  (podział po dokumentach), każdy do 65 536. Mierzę Jaccard zbiorów tokenów (jako bajty)
  przy rozmiarach 1k…64k oraz pokrycie w oknach rang: jaki odsetek tokenów z rang
  [r, r+w) tokenizera A istnieje gdziekolwiek w tokenizerze B tego samego rozmiaru.
- **Hipoteza 1:** Jaccard ≥ 0,95 do 4k włącznie.
- **Hipoteza 2:** Jaccard < 0,70 przy 32k.
- **Falsyfikacja:** hipoteza 1 pada, jeśli Jaccard przy 4k < 0,95. Hipoteza 2 pada, jeśli Jaccard przy 32k ≥ 0,70.

### 3.2 Odporność na literówki a rozmiar słownika
- **Metoda:** 20 000 losowych słów (≥ 4 litery) z held-out, kodowanych w kontekście
  (z wiodącą spacją). Dla każdego jedna losowa literówka: zamiana sąsiednich liter,
  usunięcie, wstawienie lub podmiana na sąsiedni klawisz QWERTY. Mierzę Δ = tokeny(literówka)
  − tokeny(oryginał) dla każdego rozmiaru vocabu.
- **Hipoteza 1:** średnie Δ przy 32k > 1,5 tokenu.
- **Hipoteza 2:** średnie Δ przy 4k < 1,0 tokenu.
- **Hipoteza 3 (kierunek):** Δ rośnie monotonicznie z vocab_size.
- **Falsyfikacja:** hipotezy 1, 2 i 3 obalane niezależnie, według progów powyżej.

### 3.3 Polski „bez ogonków”
- **Metoda (obserwacja):** held-out po usunięciu diakrytyków (ą→a, ł→l, ż/ź→z itd.).
  Porównuję liczbę bajtów i liczbę tokenów z oryginałem dla każdego vocabu.
- **Metoda (interwencja):** tokenizer 32k trenowany na korpusie, w którym 10% dokumentów
  pozbawiono diakrytyków. Porównanie z głównym 32k na obu wersjach held-out.
- **Hipoteza 1:** bajty spadają o ok. 3%, ale tokeny rosną o > 15% (przy 32k).
- **Hipoteza 2:** domieszka 10% odzyskuje ponad połowę straty na tekście bez ogonków,
  kosztem < 2% wzrostu fertility na normalnym tekście.
- **Falsyfikacja:** hipoteza 1 pada, jeśli wzrost tokenów ≤ 15%. Hipoteza 2 pada, jeśli odzyskane < 50%
  luki albo koszt ≥ 2%.

### 3.4 Liczby i daty
- **Metoda:** trzy tokenizery 32k różniące się wyłącznie pretokenizacją cyfr: `group3`
  (GPT-4: grupy ≤ 3 cyfr od lewej), `single` (każda cyfra osobno), `free` (`\p{N}+`,
  BPE decyduje). Dla liczb 1000–9999 (kodowanych jak w tekście, po spacji) mierzę wzorzec
  podziału (np. 2+2, 3+1, 4). „Nieregularny” podział = inny niż najczęstszy wzorzec.
  Dodatkowo: odsetek lat 1950–2025 jako jeden token, tokeny na liczbę w held-out
  i ogólna fertility.
- **Hipoteza 1:** w wariancie `free` ponad 40% liczb 4-cyfrowych ma nieregularny podział.
- **Hipoteza 2:** w wariancie `free` lata 1950–2025 to w ≥ 90% pojedynczy token.
- **Hipoteza 3:** wybór wariantu cyfr zmienia ogólną fertility na held-out o < 1%.
- **Falsyfikacja:** progi jak wyżej.

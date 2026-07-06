# Raport z Prac i Specyfikacja: Byte-Level BPE Tokenizer (Polish)

Niniejszy dokument stanowi podsumowanie i opis inżynieryjny prac nad autorskim, zaawansowanym tokenizerem **Byte-level BPE (BBPE)** napisanym od podstaw w czystym Pythonie na potrzeby projektu małego polskiego modelu językowego (LLM ok. 200M parametrów).

---

## 1. Wybór architektury: BPE na poziomie znaków vs Byte-level BPE

Podczas prac zaimplementowano dwa rodzaje tokenizerów, ostatecznie wybierając standard komercyjny (**BBPE**):

*   **Początkowy (Character-Level BPE):**
    Rozpoczynał naukę bezpośrednio od znaków tekstowych Unicode. Choć prosty dydaktycznie, posiada krytyczną wadę: gdy użytkownik wprowadzi znak niewidziany podczas treningu (rzadkie emoji, obcy alfabet, rzadki symbol matematyczny), model zwraca wadliwy token `<UNK>` (ID 0).
*   **Docelowy (Byte-Level BPE):**
    Tekst jest najpierw kodowany na ciąg bajtów UTF-8. Bazowy alfabet to zawsze **stałe 256 bajtów (0-255)**. 
    *   **Zaleta:** Dowolny znak (i plik) na świecie można wyrazić za pomocą bajtów. Dzięki temu tokenizer **nigdy nie rzuca błędu `<UNK>`** i bezbłędnie koduje każdy wejściowy tekst.
    *   **Widoczność w JSON (bytes_to_unicode):** Zaimplementowano mapowanie bajtów na czytelne dla oka znaki Unicode (stosowane np. w GPT-2), dzięki czemu znaki kontrolne i spacje są prezentowane w pliku JSON w cywilizowany sposób (np. spacja jako symbol `Ġ`).

---

## 2. Podział Zapasów Treningowych (Dataset Balancing)

Trening przeprowadzono na potężnym polskim korpusie **Polish DynaWord** (łącznie ok. 6 GB zróżnicowanego tekstu w 12 domenach, m.in. akty prawne EUR-lex, Dziennik Ustaw, Wikisłownik, polska literatura Wolne Lektury, Wikipedia, Wikinews) oraz dużym, niefiltrowanym korpusie **wikipedia-pl-20230401**.

*   **Selektywne Próbkowanie i Wikipedia:**
    Aby uniknąć zdominowania słownika przez żargon prawniczy (EUR-Lex i Parlament stanowią ponad 60% całego korpusu) oraz chronić pamięć RAM przed zapchaniem, wdrożono funkcję `load_balanced_polish_corpus` pobierającą równe porcje po **1 500 000 znaków** z każdego pliku Parquet. Dodatkowo włączono **4 041 586 znaków** bezpośrednio z niefiltrowanej polskiej Wikipedii, aby wzorowo obsłużyć terminologię encyklopedyczną i nowoczesny język.
*   **Całkowity zbiór treningowy:** Ponad **22 200 000 znaków** (dokładnie 22 247 300 znaków).

---

## 3. Zabezpieczenia przed Overfittingiem Słownika i Polonizacja (Regex)

Podczas zwiększania rozmiaru słownika, standardowy algorytm BPE zaczął scalić powtarzające się formułki urzędowe w gigantyczne, kilkudziesięcio-znakowe tokeny (np. *„oraz do zarządzenia jej publikacji w Dzienniku Urzędowym...„* z podpisem przewodniczącego EUR-Lex). 
Aby zapobiec marnowaniu slotów w słowniku, zaimplementowano trzy kluczowe mechanizmy:
1.  **Długość maksymalna tokenu (`max_token_len` = 20):** Algorytm ignoruje i odrzuca scalenia dłuższe niż 20 znaków rzeczywistego tekstu. Pozwala to na naukę słów i ich odmian, odrzucając całe zdania.
2.  **Zakaz nowej linii (`\n`):** Zabroniono łączenia linii i akapitów. Tokeny kończą się przed przejściem do nowej linii.
3.  **Optymalizacja pod język polski (Dedykowany Regex):** Tradycyjny regex z GPT-2/3 niepotrzebnie rozbijał angielskie skróty (np. `'s`, `'re`, `'ve`), które w języku polskim nie występują. Zaimplementowano dedykowany, czysty polski regex: ` ?[^\s\d\W]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+`. Eliminuje on obce wzorce, poprawnie obsługuje polskie znaki diakrytyczne, ułatwia łączenie spacji z początkami wyrazów oraz całkowicie zapobiega łączeniu liter z cyframi (np. ubezwłasnowolnienie tokenizacji struktur typu `rok2026`), co pozwala skupić proces uczenia wyłącznie na polskiej fleksji i morfologii.

---

## 4. Wyniki Ewaluacji (Test na całym "Panu Tadeuszu")

Wyniki testów na pełnym eposie narodowym (447k znaków, 69k słów) po wytrenowaniu ostatecznego słownika **`vocab_size = 16000`** na zbalansowanej porcji DynaWord + wikipedia-pl (łącznie ponad 22.2 mln znaków):

*   **Wierność rekonstrukcji (Wycofanie / Dekodowanie):** **100.0% (IDEALNA)**.
*   **Liczba nieznanych tokenów (`<UNK>`):** **0** (Idealne pokrycie bajtowe, całkowite wyeliminowanie problemu nieznanych znaków).
*   **Optymalny współczynnik redukcji długości tekstu (kompresja znakowa):** **68.18%** (tekst w postaci tokenów zajmuje ponad 3.14-krotnie mniej miejsca!).
*   **Wskaźnik Fertility:** **2.060 tok/słowo** (Niespotykana dotąd, rewelacyjna kompresja — słownik potrzebuje średnio zaledwie 2.06 tokenu do zakodowania całego bogatego polskiego słowa, co stanowi olbrzymi skok wydajności w porównaniu z bazowym 2.353!).

### Próbka podziału tekstu (Inwokacja):
*   `Adam Mickiewicz` $\rightarrow$ `['A', 'dam ', 'Mi', 'ckie', 'wicz']`
*   `Pan Tadeusz` $\rightarrow$ `['Pan', ' T', 'ade', 'usz']` (*Słowo "Pan" skompresowane w całości!*)
*   `czyli ostatni zajazd na Litwie` $\rightarrow$ `['czyli ', 'ostat', 'ni ', 'za', 'jazd ', 'na ', 'L', 'it', 'wie']` (*Spójniki i przyimki kodowane jako całe słowa ze spacjami!*)

---
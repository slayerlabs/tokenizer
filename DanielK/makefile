PYTHON ?= python3
VOCAB_SIZE ?= 16384
VOCAB_TAG ?= 16k
NKJP_ARCHIVE ?= dane_raw/NKJP-PodkorpusMilionowy-1.2.tar.gz
YT_CHANNEL ?=
TEST_FILE ?= dane_eval
CLASSIC_FILE ?= dane_klasyka/klasyka_test.jsonl

.PHONY: help install test-demo test-fetch test-h1-modernize test-train fetch fetch-yt h1-modernize train eval eval-heldout clean-test

help:
	@echo "Dostępne komendy (uruchamiaj z katalogu DanielK/ lub przez 'make -C DanielK <cel>'):"
	@echo "  make install           - instaluje zależności z requirements.txt"
	@echo "  make test-demo         - szybki test offline na mini-korpusie + wykresy"
	@echo "  make test-fetch        - testowy zaciąg próbek (wymaga download_datasets.py + YT_CHANNEL)"
	@echo "  make test-h1-modernize - test uwspółcześniania przez Gemini API (5 fragmentów, pomija jeśli gotowe)"
	@echo "  make test-train        - pełny test T0->T3 + Gemini H1 (osobne foldery) + benchmark + wykresy"
	@echo "  make fetch             - zaciąg korpusów + uwspółcześnienie 15 MB klasyki (wymaga download_datasets.py + YT_CHANNEL)"
	@echo "  make fetch-yt          - pobranie transkrypcji YT (wymaga YT_CHANNEL=<URL>)"
	@echo "  make h1-modernize      - samo uwspółcześnienie 15 MB Wolnych Lektur (pomija jeśli już zrobione)"
	@echo "  make train             - trening T0, T1(H1-modern), T1(PL), T2(PL+CNC), T3(PL+EN+CNC)"
	@echo "  make eval              - benchmark wszystkich modeli (--classic-file $(CLASSIC_FILE)) + wykresy w artifacts/plots_<VOCAB_TAG>/"
	@echo "  make eval-heldout      - benchmark na $(TEST_FILE) i --classic-file $(CLASSIC_FILE) + wykresy w artifacts/plots_<VOCAB_TAG>_heldout/"
	@echo "  make clean-test        - usuwa katalogi dane_test_*, dane_h1_test i testowe artefakty"

install:
	$(PYTHON) -m pip install -r requirements.txt

# --- TESTY ---

test-demo:
	$(PYTHON) bpe_basic_2.py --demo --mode standard --output artifacts/demo_bpe2_std/tokenizer.json
	$(PYTHON) bpe_basic_2.py --demo --mode polish --output artifacts/demo_bpe2_polish/tokenizer.json
	$(PYTHON) bpe_basic_2.py --demo --mode polish,cnc --output artifacts/demo_bpe2_polish_cnc/tokenizer.json
	$(PYTHON) evaluate_tokenizer.py --tokenizers artifacts/demo_bpe2_std/tokenizer.json artifacts/demo_bpe2_polish/tokenizer.json artifacts/demo_bpe2_polish_cnc/tokenizer.json --plots-dir artifacts/demo_plots

test-fetch:
	@test -f download_datasets.py || (echo "Brak skryptu download_datasets.py w katalogu DanielK/." && exit 1)
	$(PYTHON) download_datasets.py --sources wolne-lektury --output-dir dane_test_klasyka --max-mb 2 --max-rows 200
	$(PYTHON) download_datasets.py --sources nkjp,wiki-pl-tech,dynaword --nkjp-archive $(NKJP_ARCHIVE) --output-dir dane_test_pl --max-mb 2 --max-rows 200
	$(PYTHON) fetch_yt_transcripts.py --channel "$(YT_CHANNEL)" --max-videos 3 --subs-dir dane_test_yt_vtt --train-output dane_test_pl/yt_cnc.jsonl --test-output dane_test_eval/yt_cnc_test.jsonl
	$(PYTHON) download_datasets.py --sources wiki-en-tech,en-mfg --output-dir dane_test_en --max-mb 2 --max-rows 200

test-h1-modernize: test-fetch
	$(PYTHON) modernize_corpus.py --input dane_test_klasyka/wolne_lektury_pl.jsonl --raw-output dane_h1_test/raw_klasyka.jsonl --output dane_h1_test/modern_pl.jsonl --max-chunks 5 --workers 2

test-train: test-fetch test-h1-modernize
	$(PYTHON) bpe_basic_2.py --input dane_h1_test/raw_klasyka.jsonl --mode polish --vocab-size 1024 --output artifacts/test_bpe2_t0_klasyka/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_h1_test/modern_pl.jsonl --mode polish --vocab-size 1024 --output artifacts/test_bpe2_t1_h1_modern/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_test_pl --mode polish --vocab-size 1024 --output artifacts/test_bpe2_t1_pl/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_test_pl --mode polish,cnc --vocab-size 1024 --output artifacts/test_bpe2_t2_pl_cnc/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_test_pl dane_test_en --mode polish,cnc --vocab-size 1024 --output artifacts/test_bpe2_t3_pl_en_cnc/tokenizer.json
	$(PYTHON) evaluate_tokenizer.py --tokenizers artifacts/test_bpe2_t0_klasyka/tokenizer.json artifacts/test_bpe2_t1_h1_modern/tokenizer.json artifacts/test_bpe2_t1_pl/tokenizer.json artifacts/test_bpe2_t2_pl_cnc/tokenizer.json artifacts/test_bpe2_t3_pl_en_cnc/tokenizer.json --plots-dir artifacts/test_plots

# --- WŁAŚCIWY ZACIĄG I TRENING (ABLACJA T0 -> T1 -> T2 -> T3) ---

h1-modernize:
	$(PYTHON) modernize_corpus.py --input dane_klasyka/wolne_lektury_pl.jsonl --raw-output dane_h1/raw_klasyka.jsonl --output dane_h1/modern_pl.jsonl --max-mb 15 --workers 8

fetch-yt:
	$(PYTHON) fetch_yt_transcripts.py --channel "$(YT_CHANNEL)" --subs-dir dane_yt_vtt --train-output dane_pl/yt_cnc.jsonl --test-output dane_eval/yt_cnc_test.jsonl

fetch:
	@test -f download_datasets.py || (echo "Brak skryptu download_datasets.py w katalogu DanielK/." && exit 1)
	@echo "=== T0/H1 (klasyka PL): Wolne Lektury -> dane_klasyka/ ==="
	$(PYTHON) download_datasets.py --sources wolne-lektury --output-dir dane_klasyka --max-mb 30
	@echo "=== H1 (uwspółcześnienie 15 MB klasyki przez Gemini API — pomija jeśli gotowe) ==="
	$(MAKE) h1-modernize
	@echo "=== T1/T2 (~95 MB PL: 55 MB ogólny PL [NKJP+dynaword-general] + 40 MB PL tech [dynaword+wiki-pl-tech+YT]) -> dane_pl/ ==="
	$(PYTHON) download_datasets.py --sources nkjp --nkjp-archive $(NKJP_ARCHIVE) --output-dir dane_pl --max-mb 10
	$(PYTHON) download_datasets.py --sources dynaword-general --output-dir dane_pl --max-mb 48
	$(PYTHON) download_datasets.py --sources dynaword --output-dir dane_pl --max-mb 18
	$(PYTHON) download_datasets.py --sources wiki-pl-tech --output-dir dane_pl --max-mb 18
	$(MAKE) fetch-yt
	@echo "=== T3 (~15 MB / 13.6% EN technicznego): wiki-en-tech -> dane_en/ ==="
	$(PYTHON) download_datasets.py --sources wiki-en-tech --output-dir dane_en --max-mb 15

train:
	$(PYTHON) bpe_basic_2.py --input dane_h1/raw_klasyka.jsonl --mode polish --vocab-size $(VOCAB_SIZE) --output artifacts/bpe2_t0_klasyka_polish_$(VOCAB_TAG)/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_h1/modern_pl.jsonl --mode polish --vocab-size $(VOCAB_SIZE) --output artifacts/bpe2_t1_h1_modern_$(VOCAB_TAG)/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_pl --mode polish --vocab-size $(VOCAB_SIZE) --output artifacts/bpe2_t1_pl_polish_$(VOCAB_TAG)/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_pl --mode polish,cnc --vocab-size $(VOCAB_SIZE) --output artifacts/bpe2_t2_pl_cnc_$(VOCAB_TAG)/tokenizer.json
	$(PYTHON) bpe_basic_2.py --input dane_pl dane_en --mode polish,cnc --vocab-size $(VOCAB_SIZE) --output artifacts/bpe2_t3_pl_en_cnc_$(VOCAB_TAG)/tokenizer.json

eval:
	$(PYTHON) evaluate_tokenizer.py --tokenizers artifacts/bpe2_t0_klasyka_polish_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t1_h1_modern_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t1_pl_polish_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t2_pl_cnc_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t3_pl_en_cnc_$(VOCAB_TAG)/tokenizer.json --classic-file $(CLASSIC_FILE) --plots-dir artifacts/plots_$(VOCAB_TAG)

eval-heldout:
	$(PYTHON) evaluate_tokenizer.py --tokenizers artifacts/bpe2_t0_klasyka_polish_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t1_h1_modern_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t1_pl_polish_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t2_pl_cnc_$(VOCAB_TAG)/tokenizer.json artifacts/bpe2_t3_pl_en_cnc_$(VOCAB_TAG)/tokenizer.json --test-file $(TEST_FILE) --classic-file $(CLASSIC_FILE) --plots-dir artifacts/plots_$(VOCAB_TAG)_heldout

clean-test:
	rm -rf dane_test_klasyka dane_test_pl dane_test_en dane_test_eval dane_test_yt_vtt dane_h1_test artifacts/demo_* artifacts/test_*

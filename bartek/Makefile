# One command: `make all` (setup -> data -> training -> experiments).
PY := .venv/bin/python
W  := --workers 4

.PHONY: all setup data train eval clean-results

all: setup data train eval

setup: .venv/.ok
.venv/.ok: requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements.txt
	touch $@

data: data/clean/train.jsonl
data/clean/train.jsonl: .venv/.ok
	$(PY) scripts/download.py
	$(PY) scripts/clean.py

# Training runs are independent; `make -j6 all` runs them in parallel.
TRAIN_TARGETS := artifacts/main/meta.json artifacts/half_a/meta.json artifacts/half_b/meta.json \
                 artifacts/strip10/meta.json artifacts/digits_single/meta.json artifacts/digits_free/meta.json
train: $(TRAIN_TARGETS)
$(TRAIN_TARGETS): data/clean/train.jsonl

artifacts/main/meta.json:
	$(PY) scripts/train.py --out artifacts/main --workers 12
artifacts/half_a/meta.json:
	$(PY) scripts/train.py --out artifacts/half_a --subset half-a $(W)
artifacts/half_b/meta.json:
	$(PY) scripts/train.py --out artifacts/half_b --subset half-b $(W)
artifacts/strip10/meta.json:
	$(PY) scripts/train.py --out artifacts/strip10 --strip-frac 0.1 --sizes 32768 $(W)
artifacts/digits_single/meta.json:
	$(PY) scripts/train.py --out artifacts/digits_single --digits single --sizes 32768 $(W)
artifacts/digits_free/meta.json:
	$(PY) scripts/train.py --out artifacts/digits_free --digits free --sizes 32768 $(W)

eval: train
	$(PY) scripts/eval_vocab.py
	$(PY) scripts/exp_stability.py
	$(PY) scripts/exp_typos.py
	$(PY) scripts/exp_diacritics.py
	$(PY) scripts/exp_numbers.py
	$(PY) scripts/export_hf.py

clean-results:
	rm -rf results report/figures/*.png

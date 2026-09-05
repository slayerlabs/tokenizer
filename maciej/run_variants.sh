#!/bin/bash
# Warianty do krzywej vocab->fertility oraz test wpływu pretokenizera.
set -e
for v in 8000 16000 64000; do
  uv run --quiet --with regex --with pyarrow python train.py --vocab-size $v --pattern pl 2>&1 | tee results/train-${v}-pl.log
done
uv run --quiet --with regex --with pyarrow python train.py --vocab-size 32000 --pattern gpt2 2>&1 | tee results/train-32k-gpt2.log
echo "ALL VARIANTS DONE"

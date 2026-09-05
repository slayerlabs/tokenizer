#!/bin/bash
# Render wszystkich scen + sklejenie w jeden film.
#   ./render.sh          -> podgląd (480p15, szybko)
#   ./render.sh -qh      -> produkcja (1080p60)
set -e
Q="${1:--ql}"
SCENES="S1_Dlaczego S2_Algorytm S3_Bajty S4_Encode S5_Fertility"
for s in $SCENES; do
  echo "=== $s ==="
  uv run --quiet --with manim manim $Q --disable_caching scenes.py "$s"
done
DIR=$(find media/videos/scenes -maxdepth 1 -type d -name "*p*" | head -1)
: > /tmp/bpe_concat.txt
for s in $SCENES; do echo "file '$PWD/$DIR/$s.mp4'" >> /tmp/bpe_concat.txt; done
ffmpeg -y -loglevel error -f concat -safe 0 -i /tmp/bpe_concat.txt -c copy "$DIR/bpe-full.mp4"
echo "-> $DIR/bpe-full.mp4"

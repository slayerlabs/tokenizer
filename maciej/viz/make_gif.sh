#!/bin/bash
# GIF do README (paleta + skalowanie, żeby zmieścić się w kilku MB).
set -e
cd "$(dirname "$0")"
uv run --quiet --with manim manim -qm --disable_caching scenes.py S2_Gif
SRC=$(find media/videos/scenes -name "S2_Gif.mp4" | head -1)
ffmpeg -y -loglevel error -i "$SRC" -vf "fps=12,scale=720:-1:flags=lanczos,palettegen=stats_mode=diff" /tmp/pal.png
ffmpeg -y -loglevel error -i "$SRC" -i /tmp/pal.png \
  -lavfi "fps=12,scale=720:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" bpe-algorithm.gif
ls -lh bpe-algorithm.gif

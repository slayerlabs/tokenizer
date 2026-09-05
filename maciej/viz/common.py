"""Wspólna warstwa wizualna dla scen BPE (paleta, pudełka tokenów, tytuły)."""

from __future__ import annotations

import json
import os

from manim import *

# --- paleta -----------------------------------------------------------------
BG = "#111520"
FG = "#E9ECF4"
MUTED = "#8B93A7"
ACCENT = "#FFC857"      # żółty — to, na co patrzymy teraz
BYTE = "#59A5D8"        # niebieski — surowe bajty
TOKEN = "#7BD389"       # zielony — token po merge'u
WARN = "#EF6461"        # czerwony — problem / <UNK>
VIOLET = "#B085F5"

SANS = "Helvetica Neue"
MONO = "Menlo"

config.background_color = BG

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "viz_data.json")


def viz_data() -> dict:
    """Zmierzone liczby z evaluate.py. Świadomie BEZ fallbacku —
    lepiej, żeby render padł, niż żeby film pokazał zmyślone wyniki."""
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


# --- podstawowe elementy ----------------------------------------------------

def title(text: str, sub: str | None = None) -> VGroup:
    t = Text(text, font=SANS, font_size=42, color=FG, weight=BOLD)
    g = VGroup(t)
    if sub:
        s = Text(sub, font=SANS, font_size=24, color=MUTED)
        s.next_to(t, DOWN, buff=0.18)
        g.add(s)
    g.to_edge(UP, buff=0.45)
    return g


def caption(text: str, color: str = MUTED, size: int = 26) -> Text:
    return Text(text, font=SANS, font_size=size, color=color).to_edge(DOWN, buff=0.6)


def tok_box(label: str, color: str = BYTE, font_size: int = 30,
            height: float = 0.68, min_width: float = 0.62) -> VGroup:
    """Jedno pudełko = jeden token. Szerokość rośnie z długością napisu."""
    txt = Text(label, font=MONO, font_size=font_size, color=FG)
    w = max(txt.width + 0.3, min_width)
    box = RoundedRectangle(
        corner_radius=0.09, width=w, height=height,
        stroke_color=color, stroke_width=2.5,
        fill_color=color, fill_opacity=0.16,
    )
    txt.move_to(box)
    g = VGroup(box, txt)
    g.label = label
    g.color_key = color
    return g


def tok_row(labels, colors=None, buff: float = 0.09, font_size: int = 30) -> VGroup:
    if colors is None:
        colors = [BYTE] * len(labels)
    if isinstance(colors, str):
        colors = [colors] * len(labels)
    boxes = [tok_box(l, c, font_size=font_size) for l, c in zip(labels, colors)]
    return VGroup(*boxes).arrange(RIGHT, buff=buff)


def fit_width(mob: Mobject, max_w: float = 12.6) -> Mobject:
    if mob.width > max_w:
        mob.scale(max_w / mob.width)
    return mob


def counter_label(n: int, what: str = "tokenów", color: str = ACCENT) -> VGroup:
    num = Text(str(n), font=SANS, font_size=46, color=color, weight=BOLD)
    lab = Text(what, font=SANS, font_size=24, color=MUTED)
    return VGroup(num, lab).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def rule_row(pair: str, arrow_to: str, color: str = TOKEN,
             font_size: int = 26) -> VGroup:
    """Wiersz tabeli merge'y:  ('a','a')  ->  Z"""
    left = Text(pair, font=MONO, font_size=font_size, color=FG)
    arr = Text("→", font=SANS, font_size=font_size, color=MUTED)
    right = Text(arrow_to, font=MONO, font_size=font_size, color=color)
    return VGroup(left, arr, right).arrange(RIGHT, buff=0.22)

"""Wizualizacja BPE — sceny 1-5. Render: ./render.sh

Sceny odpowiadają blokom warsztatu (materials/tokenizer.pdf):
  S1 slajdy 3-4    dlaczego nie znaki i nie słowa
  S2 slajdy 7-8    algorytm BPE na 'aaabdaaabac'
  S3 slajd 15      czemu bajty, a nie znaki
  S4 slajd 13      encoder — kolejność merge'y
  S5 slajdy 16-20  fertility: nasz tokenizer vs cl100k (liczby zmierzone)
"""

from __future__ import annotations

from manim import *

from common import (ACCENT, BYTE, FG, MONO, MUTED, SANS, TOKEN, VIOLET, WARN,
                    caption, counter_label, fit_width, rule_row, title,
                    tok_box, tok_row, viz_data)


def merge_in_row(scene, boxes, center, positions, new_label, new_color,
                 buff=0.09, font_size=30, run_time=1.1):
    """Zlej pary boxes[i],boxes[i+1] dla i w `positions` w jedno pudełko.

    Zwraca nową listę pudełek. Pudełka muszą być dodane do sceny pojedynczo
    (nie jako VGroup), żeby ReplacementTransform mógł je podmienić.
    """
    new_boxes, anims = [], []
    idx = 0
    pos = set(positions)
    while idx < len(boxes):
        if idx in pos:
            b1, b2 = boxes[idx], boxes[idx + 1]
            nb = tok_box(new_label, new_color, font_size=font_size)
            nb.move_to((b1.get_center() + b2.get_center()) / 2)
            anims.append(ReplacementTransform(VGroup(b1, b2), nb))
            new_boxes.append(nb)
            idx += 2
        else:
            new_boxes.append(boxes[idx])
            idx += 1
    scene.play(*anims, run_time=run_time)
    grp = VGroup(*new_boxes)
    scene.play(grp.animate.arrange(RIGHT, buff=buff).move_to(center), run_time=0.6)
    return new_boxes


def counts_panel(rows, highlight=None, highlight_color=ACCENT, font_size=25):
    """Tabelka 'para -> licznik'. rows: [(etykieta, licznik)]"""
    items = []
    for lab, cnt in rows:
        c = highlight_color if (highlight and lab in highlight) else FG
        left = Text(lab, font=MONO, font_size=font_size, color=c)
        right = Text(str(cnt), font=MONO, font_size=font_size,
                     color=c if c != FG else MUTED)
        row = VGroup(left, right).arrange(RIGHT, buff=0.45)
        items.append(row)
    g = VGroup(*items).arrange(DOWN, buff=0.16, aligned_edge=LEFT)
    for row in items:                       # wyrównaj liczniki do prawej
        row[1].align_to(g, RIGHT)
    return g


# ---------------------------------------------------------------- S1
class S1_Dlaczego(Scene):
    def construct(self):
        t = title("Dlaczego nie znak po znaku?",
                  "i dlaczego nie całymi słowami")
        self.play(FadeIn(t, shift=DOWN * 0.3))
        self.wait(0.4)

        # --- 1. znaki: sekwencja puchnie
        sent = "Kupiłem dom w Warszawie."
        chars = tok_row(list(sent.replace(" ", "␣")), BYTE,
                        buff=0.05, font_size=22)
        fit_width(chars, 12.4).move_to(UP * 0.75)
        lab1 = Text("1. znak jako jednostka", font=SANS, font_size=28,
                    color=ACCENT).next_to(chars, UP, buff=0.6)
        self.play(FadeIn(lab1), LaggedStart(*[FadeIn(b) for b in chars],
                                            lag_ratio=0.03, run_time=1.6))
        c1 = counter_label(len(sent), "tokenów na jedno zdanie")
        c1.next_to(chars, DOWN, buff=0.85)
        self.play(FadeIn(c1, shift=UP * 0.2))
        cap = Text("okno kontekstu jest skończone — 4-5× dłuższe sekwencje "
                   "to 4-5× wyższy koszt", font=SANS, font_size=26, color=MUTED)
        cap.next_to(c1, DOWN, buff=0.9)
        self.play(FadeIn(cap))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (chars, c1, cap, lab1)])

        # --- 2. słowa: eksplozja słownika we fleksji (slajd 4)
        lab2 = Text("2. słowo jako jednostka", font=SANS, font_size=28,
                    color=ACCENT).move_to(UP * 2.1)
        forms = ["dom", "domu", "domowi", "domem", "domach", "domowy", "domowego"]
        frow = tok_row(forms, VIOLET, buff=0.14, font_size=28)
        fit_width(frow, 12.4).move_to(UP * 1.15)
        ids = VGroup(*[Text(f"id {1000+i}", font=MONO, font_size=19, color=MUTED)
                       .next_to(b, DOWN, buff=0.16) for i, b in enumerate(frow)])
        self.play(FadeIn(lab2))
        self.play(LaggedStart(*[FadeIn(b) for b in frow], lag_ratio=0.12,
                              run_time=1.6))
        self.play(FadeIn(ids, shift=UP * 0.1))
        one = Text("jedno pojęcie  =  7 osobnych ID", font=SANS,
                   font_size=30, color=FG).next_to(ids, DOWN, buff=0.6)
        self.play(FadeIn(one))
        self.wait(1.2)

        new = tok_box("domownikami", WARN, font_size=28).next_to(one, DOWN, buff=0.5)
        arrow = Text("→", font=SANS, font_size=34, color=MUTED)
        unk = tok_box("<UNK>", WARN, font_size=28)
        grp = VGroup(new, arrow, unk).arrange(RIGHT, buff=0.3)
        grp.next_to(one, DOWN, buff=0.5)
        self.play(FadeIn(new, shift=UP * 0.2))
        self.play(FadeIn(arrow), FadeIn(unk, shift=RIGHT * 0.2))
        self.play(Indicate(unk, color=WARN, scale_factor=1.12))
        cap2 = Text("nowa forma fleksyjna = utrata informacji", font=SANS,
                    font_size=26, color=WARN).next_to(grp, DOWN, buff=0.75)
        self.play(FadeIn(cap2))
        self.wait(1.6)
        self.play(*[FadeOut(m) for m in (frow, ids, one, grp, cap2, lab2)])

        # --- 3. subword
        lab3 = Text("3. subword — kompromis", font=SANS, font_size=28,
                    color=ACCENT).move_to(UP * 2.1)
        sub = tok_row(["dom", "own", "ikami"], TOKEN, buff=0.12, font_size=34)
        sub.move_to(UP * 0.95)
        self.play(FadeIn(lab3), LaggedStart(*[FadeIn(b) for b in sub],
                                            lag_ratio=0.18, run_time=1.2))
        note = Text("jednostki między znakiem a słowem —\n"
                    "„dom” użyte ponownie, nic nie ginie",
                    font=SANS, font_size=27, color=MUTED,
                    line_spacing=0.8).next_to(sub, DOWN, buff=0.75)
        self.play(FadeIn(note))
        self.wait(1.4)

        key = Text("Tokenizer to kompresja stratna na poziomie\n"
                   "reprezentacji, nie treści.",
                   font=SANS, font_size=34, color=FG, line_spacing=0.9)
        key.move_to(DOWN * 1.35)
        self.play(FadeOut(note), FadeIn(key, shift=UP * 0.2))
        self.wait(2.2)
        self.play(*[FadeOut(m) for m in (t, lab3, sub, key)])


# ---------------------------------------------------------------- S2
class S2_Algorytm(Scene):
    """Algorytm BPE, krok po kroku, na korpusie ze slajdu 8."""

    def construct(self):
        t = title("Byte-Pair Encoding", "policz pary · zlej najczęstszą · powtórz")
        self.play(FadeIn(t, shift=DOWN * 0.3))

        steps = VGroup(
            Text("1. zacznij od najmniejszych jednostek", font=SANS, font_size=27, color=FG),
            Text("2. policz wszystkie sąsiadujące pary", font=SANS, font_size=27, color=FG),
            Text("3. najczęstszą zlej w nowy token, zapisz regułę", font=SANS, font_size=27, color=FG),
            Text("4. powtórz N razy", font=SANS, font_size=27, color=FG),
        ).arrange(DOWN, buff=0.32, aligned_edge=LEFT).move_to(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT * 0.25) for s in steps],
                              lag_ratio=0.35, run_time=2.4))
        self.wait(1.4)
        self.play(FadeOut(steps))

        CENTER = UP * 1.35
        corpus = "aaabdaaabac"
        boxes = list(tok_row(list(corpus), BYTE, buff=0.1, font_size=32)
                     .move_to(CENTER))
        self.add(*boxes)
        clab = Text("korpus", font=SANS, font_size=23, color=MUTED)
        clab.next_to(VGroup(*boxes), UP, buff=0.3)
        self.play(FadeIn(clab), LaggedStart(*[FadeIn(b) for b in boxes],
                                            lag_ratio=0.05, run_time=1.2))

        rules_title = Text("reguły (merge)", font=SANS, font_size=23, color=MUTED)
        rules_title.move_to(RIGHT * 4.2 + DOWN * 0.55)
        self.play(FadeIn(rules_title))
        rules = VGroup()

        # ---- merge 1: (a,a) = 4
        rows = [("('a','a')", 4), ("('a','b')", 2), ("('b','d')", 1),
                ("('d','a')", 1), ("('b','a')", 1), ("('a','c')", 1)]
        panel = counts_panel(rows, highlight={"('a','a')"})
        panel.move_to(LEFT * 4.3 + DOWN * 1.05)
        ptitle = Text("licznik par", font=SANS, font_size=23, color=MUTED)
        ptitle.next_to(panel, UP, buff=0.3)
        self.play(FadeIn(ptitle), LaggedStart(*[FadeIn(r) for r in panel],
                                              lag_ratio=0.12, run_time=1.4))
        self.play(Indicate(panel[0], color=ACCENT, scale_factor=1.1))

        # subtelność: 4 wystąpienia, ale merge jest niezachodzący -> 2 zlania
        note = Text("policzona 4×, ale merge jest niezachodzący:\n"
                    "w „aaa” zlewa się tylko raz",
                    font=SANS, font_size=24, color=ACCENT, line_spacing=0.85)
        note.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(note))
        self.play(*[Indicate(boxes[i], color=ACCENT, scale_factor=1.15)
                    for i in (0, 1, 5, 6)], run_time=1.0)
        self.wait(1.5)
        self.play(FadeOut(note))

        boxes = merge_in_row(self, boxes, CENTER, [0, 5], "Z", TOKEN,
                             buff=0.1, font_size=32)
        r1 = rule_row("('a','a')", "Z")
        r1.next_to(rules_title, DOWN, buff=0.3)
        rules.add(r1)
        # licznik przeliczamy od razu — stary pokazywałby pary, których już nie ma
        rows2 = [("('Z','a')", 2), ("('a','b')", 2), ("('b','d')", 1),
                 ("('d','Z')", 1), ("('b','a')", 1), ("('a','c')", 1)]
        panel2 = counts_panel(rows2, highlight={"('Z','a')", "('a','b')"})
        panel2.move_to(panel.get_center()).align_to(panel, UP + LEFT)
        self.play(FadeIn(r1, shift=UP * 0.15), ReplacementTransform(panel, panel2))
        self.wait(0.8)

        # ---- merge 2: REMIS (Z,a)=2 vs (a,b)=2
        tie = Text("remis 2 : 2 — BPE nie definiuje, co wtedy",
                   font=SANS, font_size=25, color=ACCENT).to_edge(DOWN, buff=0.75)
        tie2 = Text("trzeba wybrać regułę i trzymać się jej "
                    "(tu: para o mniejszych ID)",
                    font=SANS, font_size=22, color=MUTED).next_to(tie, DOWN, buff=0.18)
        self.play(FadeIn(tie))
        self.wait(1.3)
        self.play(FadeIn(tie2))
        self.wait(1.5)
        self.play(FadeOut(tie), FadeOut(tie2))

        boxes = merge_in_row(self, boxes, CENTER, [1, 4], "Y", TOKEN,
                             buff=0.1, font_size=32)
        r2 = rule_row("('a','b')", "Y")
        r2.next_to(r1, DOWN, buff=0.26).align_to(r1, LEFT)
        rules.add(r2)
        self.play(FadeIn(r2, shift=UP * 0.15))
        self.wait(0.7)

        # ---- merge 3: (Z,Y) = 2
        rows3 = [("('Z','Y')", 2), ("('Y','d')", 1), ("('d','Z')", 1),
                 ("('Y','a')", 1), ("('a','c')", 1)]
        panel3 = counts_panel(rows3, highlight={"('Z','Y')"})
        panel3.move_to(panel2.get_center()).align_to(panel2, UP + LEFT)
        self.play(ReplacementTransform(panel2, panel3))
        self.wait(0.6)
        boxes = merge_in_row(self, boxes, CENTER, [0, 3], "X", TOKEN,
                             buff=0.1, font_size=32)
        r3 = rule_row("('Z','Y')", "X")
        r3.next_to(r2, DOWN, buff=0.26).align_to(r2, LEFT)
        rules.add(r3)
        self.play(FadeIn(r3, shift=UP * 0.15))
        self.wait(0.8)

        # ---- podsumowanie
        self.play(FadeOut(panel3), FadeOut(ptitle))
        before = Text("11 tokenów", font=SANS, font_size=30, color=MUTED)
        arrow = Text("→", font=SANS, font_size=30, color=MUTED)
        after = Text("5 tokenów", font=SANS, font_size=30, color=TOKEN)
        summ = VGroup(before, arrow, after).arrange(RIGHT, buff=0.3)
        summ.move_to(LEFT * 3.6 + DOWN * 1.0)
        self.play(FadeIn(summ))
        key = Text("Trening BPE = uczenie się słownika i reguł merge.\n"
                   "Tokenizacja = aplikowanie tych reguł w tej samej kolejności.",
                   font=SANS, font_size=27, color=FG, line_spacing=0.9)
        key.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(key, shift=UP * 0.2))
        self.wait(2.6)
        self.play(*[FadeOut(m) for m in
                    (t, clab, summ, key, rules, rules_title, *boxes)])


# ---------------------------------------------------------------- S3
class S3_Bajty(Scene):
    """Slajd 15: alfabet bazowy to 256 bajtów, nie ~150k znaków Unicode."""

    def construct(self):
        t = title("Czemu bajty, a nie znaki?", "alfabet bazowy tokenizera")
        self.play(FadeIn(t, shift=DOWN * 0.3))

        # --- dwa alfabety obok siebie
        big = RoundedRectangle(corner_radius=0.12, width=5.4, height=2.5,
                               stroke_color=WARN, stroke_width=3,
                               fill_color=WARN, fill_opacity=0.12)
        big_t = VGroup(
            Text("znaki Unicode", font=SANS, font_size=27, color=FG),
            Text("~150 000", font=SANS, font_size=46, color=WARN, weight=BOLD),
            Text("code pointów do pokrycia", font=SANS, font_size=21, color=MUTED),
        ).arrange(DOWN, buff=0.16).move_to(big)
        left = VGroup(big, big_t).move_to(LEFT * 3.3 + UP * 0.55)

        small = RoundedRectangle(corner_radius=0.12, width=5.4, height=2.5,
                                 stroke_color=BYTE, stroke_width=3,
                                 fill_color=BYTE, fill_opacity=0.12)
        small_t = VGroup(
            Text("bajty UTF-8", font=SANS, font_size=27, color=FG),
            Text("zawsze 256", font=SANS, font_size=46, color=BYTE, weight=BOLD),
            Text("bez wyjątków, bez <UNK>", font=SANS, font_size=21, color=MUTED),
        ).arrange(DOWN, buff=0.16).move_to(small)
        right = VGroup(small, small_t).move_to(RIGHT * 3.3 + UP * 0.55)

        self.play(FadeIn(left, shift=RIGHT * 0.2))
        self.wait(0.7)
        self.play(FadeIn(right, shift=LEFT * 0.2))
        self.wait(1.3)

        cap = caption("każdy tekst — dowolny język, alfabet, emoji — "
                      "rozkłada się na bajty")
        self.play(FadeIn(cap))
        self.wait(1.6)
        self.play(FadeOut(left), FadeOut(right), FadeOut(cap))

        # --- koszt: polski znak to 2 bajty
        ch = tok_box("ę", VIOLET, font_size=44).move_to(UP * 1.25 + LEFT * 3.2)
        chl = Text("1 znak", font=SANS, font_size=22, color=MUTED).next_to(ch, DOWN, buff=0.2)
        arr = Text("→", font=SANS, font_size=40, color=MUTED).move_to(UP * 1.25 + LEFT * 1.5)
        b1 = tok_box("0xC4", BYTE, font_size=28)
        b2 = tok_box("0x99", BYTE, font_size=28)
        bs = VGroup(b1, b2).arrange(RIGHT, buff=0.14).move_to(UP * 1.25 + RIGHT * 0.9)
        bsl = Text("2 bajty", font=SANS, font_size=22, color=MUTED).next_to(bs, DOWN, buff=0.2)
        self.play(FadeIn(ch), FadeIn(chl))
        self.play(FadeIn(arr), FadeIn(bs, shift=RIGHT * 0.2), FadeIn(bsl))
        self.wait(1.0)

        cost = Text("znak spoza ASCII kosztuje 2-4 bajty zamiast jednego",
                    font=SANS, font_size=26, color=WARN).move_to(DOWN * 0.3)
        self.play(FadeIn(cost))
        self.wait(1.4)

        # --- ale BPE to odkręca, JEŚLI sekwencja jest częsta
        fix = Text("...ale BPE zlewa częste sekwencje z powrotem",
                   font=SANS, font_size=26, color=TOKEN).move_to(DOWN * 0.3)
        self.play(FadeOut(cost), FadeIn(fix))
        merged = tok_box("ę", TOKEN, font_size=34).move_to(bs.get_center())
        self.play(ReplacementTransform(bs, merged),
                  FadeOut(bsl), run_time=1.1)
        self.play(Indicate(merged, color=TOKEN, scale_factor=1.15))
        self.wait(1.2)

        # --- warunek: jeśli częste W TYM korpusie
        cond = Text("„jeśli częste” — w korpusie, na którym trenowano tokenizer",
                    font=SANS, font_size=27, color=ACCENT).move_to(DOWN * 1.45)
        cond2 = Text("tokenizer trenowany głównie na angielskim nie da polskim\n"
                     "znakom własnych tokenów — rozpadną się na surowe bajty",
                     font=SANS, font_size=25, color=MUTED, line_spacing=0.85)
        cond2.next_to(cond, DOWN, buff=0.35)
        self.play(FadeOut(fix), FadeIn(cond))
        self.wait(1.2)
        self.play(FadeIn(cond2))
        self.wait(2.4)
        self.play(*[FadeOut(m) for m in (t, ch, chl, arr, merged, cond, cond2)])


# ---------------------------------------------------------------- S4
class S4_Encode(Scene):
    """Slajd 13: merge'e aplikuje się w kolejności, w jakiej powstały."""

    def construct(self):
        t = title("Encoder — tu ludzie psują", "kolejność merge'y nie jest dowolna")
        self.play(FadeIn(t, shift=DOWN * 0.3))

        # wyuczone reguły z rangami
        rt = Text("wyuczone reguły", font=SANS, font_size=23, color=MUTED)
        rows = VGroup(
            VGroup(Text("rank 0", font=MONO, font_size=22, color=ACCENT),
                   rule_row("('a','a')", "Z")).arrange(RIGHT, buff=0.45),
            VGroup(Text("rank 1", font=MONO, font_size=22, color=MUTED),
                   rule_row("('a','b')", "Y")).arrange(RIGHT, buff=0.45),
            VGroup(Text("rank 2", font=MONO, font_size=22, color=MUTED),
                   rule_row("('Z','Y')", "X")).arrange(RIGHT, buff=0.45),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        panel = VGroup(rt, rows).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        panel.to_edge(UP, buff=1.7).to_edge(LEFT, buff=0.9)
        self.play(FadeIn(panel))
        self.wait(0.8)

        new = Text('nowy tekst:  "aab"', font=SANS, font_size=30, color=FG)
        new.to_edge(UP, buff=1.9).to_edge(RIGHT, buff=1.4)
        self.play(FadeIn(new))
        self.wait(0.6)

        # --- dwie ścieżki
        okl = Text("dobrze: najniższy rank najpierw", font=SANS, font_size=25, color=TOKEN)
        badl = Text("źle: dowolna kolejność", font=SANS, font_size=25, color=WARN)
        okl.move_to(LEFT * 3.4 + DOWN * 0.15)
        badl.move_to(RIGHT * 3.4 + DOWN * 0.15)
        self.play(FadeIn(okl), FadeIn(badl))

        okr = list(tok_row(list("aab"), BYTE, buff=0.1, font_size=30)
                   .move_to(LEFT * 3.4 + DOWN * 0.9))
        badr = list(tok_row(list("aab"), BYTE, buff=0.1, font_size=30)
                    .move_to(RIGHT * 3.4 + DOWN * 0.9))
        self.add(*okr, *badr)
        self.play(*[FadeIn(b) for b in okr + badr])
        self.wait(0.6)

        okr = merge_in_row(self, okr, LEFT * 3.4 + DOWN * 0.9, [0], "Z", TOKEN,
                           font_size=30, run_time=0.9)
        badr = merge_in_row(self, badr, RIGHT * 3.4 + DOWN * 0.9, [1], "Y", WARN,
                            font_size=30, run_time=0.9)
        self.wait(0.7)

        okid = Text("[Z, b]", font=MONO, font_size=30, color=TOKEN)
        okid.next_to(VGroup(*okr), DOWN, buff=0.45)
        badid = Text("[a, Y]", font=MONO, font_size=30, color=WARN)
        badid.next_to(VGroup(*badr), DOWN, buff=0.45)
        self.play(FadeIn(okid), FadeIn(badid))
        self.wait(1.2)

        # --- puenta: obie wersje DEKODUJĄ się poprawnie
        both = Text('obie dekodują się do "aab" — nic nie wybucha',
                    font=SANS, font_size=27, color=MUTED).to_edge(DOWN, buff=0.95)
        self.play(FadeIn(both))
        self.wait(1.6)
        punch = Text("ale model ma embeddingi indeksowane po ID:\n"
                     "to dla niego dwa różne wejścia. Błąd jest CICHY.",
                     font=SANS, font_size=28, color=ACCENT, line_spacing=0.9)
        punch.to_edge(DOWN, buff=0.6)
        self.play(FadeOut(both), FadeIn(punch, shift=UP * 0.15))
        self.wait(2.6)
        self.play(*[FadeOut(m) for m in
                    (t, panel, new, okl, badl, okid, badid, punch, *okr, *badr)])


# ---------------------------------------------------------------- S5
class S5_Fertility(Scene):
    """Slajdy 16-20 na zmierzonych danych: nasz tokenizer vs cl100k."""

    def construct(self):
        d = viz_data()
        ours, ref = d["ours"], d["ref"]

        t = title("Ile to kosztuje po polsku?",
                  "to samo zdanie, dwa tokenizery")
        self.play(FadeIn(t, shift=DOWN * 0.3))

        sent = Text(d["demo_pl"], font=SANS, font_size=26, color=FG)
        fit_width(sent, 12.0).move_to(UP * 1.95)
        self.play(Write(sent, run_time=1.6))
        self.wait(0.5)

        def block(info, color, y, sublabel):
            lab = Text(info["name"], font=SANS, font_size=24, color=color)
            sub = Text(sublabel, font=SANS, font_size=20, color=MUTED)
            head = VGroup(lab, sub).arrange(RIGHT, buff=0.35)
            row = tok_row(info["tokens"], color, buff=0.05, font_size=19)
            fit_width(row, 10.2)
            cnt = counter_label(info["n"], "tok.", color)
            cnt.scale(0.72)
            # licznik w JEDNYM rzędzie z paskiem tokenów — inaczej ucieka poza kadr
            body = VGroup(row, cnt).arrange(RIGHT, buff=0.45)
            g = VGroup(head, body).arrange(DOWN, buff=0.28)
            g.move_to(UP * y)
            return g, cnt, row

        g1, c1, row1 = block(ref, WARN, 0.55, f"słownik {ref['vocab_size']:,}".replace(",", " "))
        self.play(FadeIn(g1[0]))
        self.play(LaggedStart(*[FadeIn(b) for b in row1], lag_ratio=0.03, run_time=1.6))
        self.play(FadeIn(c1, shift=LEFT * 0.2))
        self.wait(1.3)

        g2, c2, row2 = block(ours, TOKEN, -1.25, f"słownik {ours['vocab_size']:,}".replace(",", " "))
        self.play(FadeIn(g2[0]))
        self.play(LaggedStart(*[FadeIn(b) for b in row2], lag_ratio=0.05, run_time=1.6))
        self.play(FadeIn(c2, shift=LEFT * 0.2))
        self.wait(1.6)

        ratio = ref["n"] / ours["n"]
        punch = Text(f"{ratio:.0f}× mniej tokenów — przy 3× MNIEJSZYM słowniku",
                     font=SANS, font_size=30, color=ACCENT)
        punch.to_edge(DOWN, buff=0.75)
        self.play(FadeIn(punch, shift=UP * 0.2))
        self.wait(2.4)

        self.play(*[FadeOut(m) for m in (sent, g1, g2, c1, c2, punch)])

        # --- liczby z held-outu
        h = Text("held-out, 10 MB polskiego tekstu z HPLT v3",
                 font=SANS, font_size=24, color=MUTED).move_to(UP * 1.9)
        self.play(FadeIn(h))

        def stat(v_ours, v_ref, label, better_ours, y):
            lab = Text(label, font=SANS, font_size=25, color=MUTED)
            a = Text(v_ref, font=SANS, font_size=34, color=WARN, weight=BOLD)
            b = Text(v_ours, font=SANS, font_size=34,
                     color=TOKEN if better_ours else WARN, weight=BOLD)
            g = VGroup(a, lab, b).arrange(RIGHT, buff=0.9)
            lab.move_to(ORIGIN + UP * y)
            a.next_to(lab, LEFT, buff=1.1)
            b.next_to(lab, RIGHT, buff=1.1)
            return VGroup(a, lab, b).move_to(UP * y)

        hdr = VGroup(
            Text("cl100k (GPT-4)", font=SANS, font_size=22, color=WARN).move_to(LEFT * 3.6 + UP * 1.0),
            Text("nasz 32k", font=SANS, font_size=22, color=TOKEN).move_to(RIGHT * 3.6 + UP * 1.0),
        )
        self.play(FadeIn(hdr))
        s1 = stat(f"{ours['chars_per_token']:.2f}", f"{ref['chars_per_token']:.2f}",
                  "znaków / token", True, 0.25)
        s2 = stat(f"{ours['fertility']:.3f}", f"{ref['fertility']:.3f}",
                  "tokenów / słowo", True, -0.6)
        for s in (s1, s2):
            self.play(FadeIn(s, shift=UP * 0.15))
            self.wait(0.8)

        cons = VGroup(
            Text("koszt: płacisz per token", font=SANS, font_size=24, color=FG),
            Text("kontekst: to samo okno mieści więcej", font=SANS, font_size=24, color=FG),
            Text("jakość: mniej sklejania znaczenia z okruchów", font=SANS, font_size=24, color=FG),
        ).arrange(DOWN, buff=0.2).move_to(DOWN * 2.15)
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in cons],
                              lag_ratio=0.35, run_time=1.8))
        self.wait(2.0)

        self.play(*[FadeOut(m) for m in (h, hdr, s1, s2, cons)])
        end = Text("Liczy się fertility dla TWOJEGO języka,\nnie sam rozmiar słownika.",
                   font=SANS, font_size=34, color=FG, line_spacing=0.9)
        self.play(FadeIn(end, shift=UP * 0.2))
        self.wait(2.6)
        self.play(FadeOut(end), FadeOut(t))


# ---------------------------------------------------------------- GIF
class S2_Gif(Scene):
    """Skrócona wersja S2 pod GIF do README (bez narracji, ~15 s)."""

    def construct(self):
        CENTER = UP * 0.9
        head = Text("BPE: policz pary · zlej najczęstszą · powtórz",
                    font=SANS, font_size=30, color=FG).move_to(UP * 2.4)
        self.add(head)

        boxes = list(tok_row(list("aaabdaaabac"), BYTE, buff=0.1, font_size=34)
                     .move_to(CENTER))
        self.add(*boxes)
        self.play(LaggedStart(*[FadeIn(b) for b in boxes], lag_ratio=0.05,
                              run_time=1.0))

        rules = VGroup()
        for pair, new, positions, cnt in [
            ("('a','a')", "Z", [0, 5], 4),
            ("('a','b')", "Y", [1, 4], 2),
            ("('Z','Y')", "X", [0, 3], 2),
        ]:
            tag = Text(f"najczęstsza para: {pair} × {cnt}", font=SANS,
                       font_size=26, color=ACCENT).move_to(DOWN * 1.9)
            self.play(FadeIn(tag), run_time=0.4)
            boxes = merge_in_row(self, boxes, CENTER, positions, new, TOKEN,
                                 buff=0.1, font_size=34, run_time=0.8)
            r = rule_row(pair, new, font_size=25)
            if len(rules) == 0:
                r.move_to(DOWN * 0.6)
            else:
                r.next_to(rules[-1], DOWN, buff=0.22).align_to(rules[0], LEFT)
            rules.add(r)
            self.play(FadeIn(r, shift=UP * 0.1), FadeOut(tag), run_time=0.5)

        done = Text("11 → 5 tokenów", font=SANS, font_size=30, color=TOKEN)
        done.move_to(DOWN * 2.5)
        self.play(FadeIn(done), run_time=0.6)
        self.wait(2.0)

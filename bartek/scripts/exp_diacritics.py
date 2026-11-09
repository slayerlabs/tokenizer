#!/usr/bin/env python3
"""Research question 3.3: Polish typed without diacritics ("bez ogonków").

Observation: encode held-out as-is and with diacritics stripped, for every main size.
Intervention: a 32k tokenizer trained with 10% of documents stripped, compared with
the main 32k on both versions of held-out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bpe.evaluation import encode_corpus, load_jsonl, size_dirs  # noqa: E402
from bpe.text import count_words, strip_diacritics  # noqa: E402

RES, FIG = Path("results"), Path("report/figures")
MIX = Path("artifacts/strip10/v32768")
MAIN32 = Path("artifacts/main/v32768")


def main() -> None:
    RES.mkdir(exist_ok=True)
    docs = load_jsonl()
    orig = [d["text"] for d in docs]
    strip = [strip_diacritics(t) for t in orig]
    words = sum(count_words(t) for t in orig)
    b_orig = sum(len(t.encode()) for t in orig)
    b_strip = sum(len(t.encode()) for t in strip)

    obs = {}
    for v, path in size_dirs("artifacts/main"):
        t_o = sum(encode_corpus(path, orig)[0])
        t_s = sum(encode_corpus(path, strip)[0])
        obs[v] = {"tokens_orig": t_o, "tokens_strip": t_s, "token_increase_pct": 100 * (t_s - t_o) / t_o,
                  "fertility_orig": t_o / words, "fertility_strip": t_s / words}
        print(f"v={v:>6} tokens +{obs[v]['token_increase_pct']:.1f}%", flush=True)

    m_o, m_s = obs[32768]["tokens_orig"], obs[32768]["tokens_strip"]
    x_o = sum(encode_corpus(MIX, orig)[0])
    x_s = sum(encode_corpus(MIX, strip)[0])
    interv = {
        "main_orig": m_o, "main_strip": m_s, "mix_orig": x_o, "mix_strip": x_s,
        "gap_recovered_pct": 100 * (m_s - x_s) / (m_s - m_o),
        "cost_on_normal_pct": 100 * (x_o - m_o) / m_o,
        "mix_strip_vs_main_orig_pct": 100 * (x_s - m_o) / m_o,
    }
    out = {"bytes_orig": b_orig, "bytes_strip": b_strip, "byte_change_pct": 100 * (b_strip - b_orig) / b_orig,
           "observation": obs, "intervention": interv}
    (RES / "diacritics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(interv, indent=1), f"bytes {out['byte_change_pct']:.2f}%")

    sizes = sorted(obs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar([f"{v // 1024}k" for v in sizes], [obs[v]["token_increase_pct"] for v in sizes], color="#4C72B0",
           label="tokeny")
    a1.axhline(out["byte_change_pct"], color="black", ls="--", label=f"bajty ({out['byte_change_pct']:.1f}%)")
    a1.axhline(15, color="red", ls=":", label="próg hipotezy 1 (+15%)")
    a1.set_ylabel("zmiana vs tekst z ogonkami [%]")
    a1.set_title("Ten sam tekst bez ogonków")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3, axis="y")
    labels = ["główny 32k\nz ogonkami", "główny 32k\nbez ogonków", "mix 10% 32k\nz ogonkami", "mix 10% 32k\nbez ogonków"]
    vals = [m_o / words, m_s / words, x_o / words, x_s / words]
    a2.bar(labels, vals, color=["#55A868", "#C44E52", "#8DD3A0", "#E8A0A2"])
    for i, val in enumerate(vals):
        a2.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    a2.set_ylabel("fertility")
    a2.set_title("Interwencja: 10% treningu bez ogonków")
    a2.set_ylim(min(vals) * 0.9, max(vals) * 1.05)
    fig.tight_layout()
    fig.savefig(FIG / "diacritics.png", dpi=130)


if __name__ == "__main__":
    main()

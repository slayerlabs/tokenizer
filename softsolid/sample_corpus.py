"""Wybiera próbkę Polish Dynaword do treningu i oceny tokenizera.

Z każdego z 19 źródeł bierzemy podobną ilość tekstu (domyślnie ok. 2 MB), żeby
tokenizer nie zobaczył samego prawa, które w pełnym korpusie dominuje objętością.
Dokumenty czytamy z kilku równo rozłożonych grup wierszy (row groups) pliku,
a nie tylko z początku, bo początek pliku bywa nietypowy (np. posortowany po dacie).

Podział trening / held-out robimy po skrócie identyfikatora dokumentu, więc jest
deterministyczny: ten sam dokument zawsze trafia w to samo miejsce. To jest
zwykły, losowy podział, bez sprawdzania, czy w obu częściach nie ma bliźniaków.
Tym zajmie się krok drugi pracy (czysty held-out).

Wynik: data/train.jsonl i data/heldout.jsonl, jeden dokument na linię:
{"id": ..., "source": ..., "text": ...}
"""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

# katalog z plikami *.parquet zbioru SlayerLab/polish-dynaword (pierwszy argument)
DYNAWORD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dynaword")
OUT = Path(__file__).resolve().parent / "data"

BUDGET_PER_SOURCE = 2_000_000   # znaków tekstu na źródło
MAX_DOC_CHARS = 50_000          # długie dokumenty (powieści, ustawy) przycinamy
ROW_GROUPS_TO_SAMPLE = 6        # z ilu miejsc pliku czerpiemy
HELDOUT_EVERY = 10              # co dziesiąty dokument (po skrócie id) idzie do held-outu


def is_heldout(doc_id):
    """Deterministyczny podział po skrócie identyfikatora."""
    h = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16)
    return h % HELDOUT_EVERY == 0


def sample_source(path):
    """Zwraca listę dokumentów z jednego pliku, aż do wyczerpania budżetu."""
    f = pq.ParquetFile(path)
    n = f.metadata.num_row_groups
    k = min(ROW_GROUPS_TO_SAMPLE, n)
    # równo rozłożone numery grup wierszy: 0, n/k, 2n/k, ...
    groups = sorted({int(i * n / k) for i in range(k)})
    per_group = BUDGET_PER_SOURCE // len(groups)

    docs = []
    for g in groups:
        taken = 0
        # małe paczki, bo niektóre grupy wierszy mają kilka GB po rozpakowaniu
        for batch in f.iter_batches(batch_size=500, row_groups=[g],
                                    columns=["id", "text", "source"]):
            for row in batch.to_pylist():
                text = (row["text"] or "")[:MAX_DOC_CHARS]
                if not text.strip():
                    continue
                docs.append({"id": row["id"], "source": row["source"], "text": text})
                taken += len(text)
                if taken >= per_group:
                    break
            if taken >= per_group:
                break
    return docs


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(DYNAWORD.glob("*.parquet"))
    stats = {}
    with open(OUT / "train.jsonl", "w", encoding="utf-8") as ftr, \
         open(OUT / "heldout.jsonl", "w", encoding="utf-8") as fho:
        for path in files:
            docs = sample_source(path)
            s = {"train_docs": 0, "train_chars": 0, "heldout_docs": 0, "heldout_chars": 0}
            for d in docs:
                part = "heldout" if is_heldout(d["id"]) else "train"
                (fho if part == "heldout" else ftr).write(json.dumps(d, ensure_ascii=False) + "\n")
                s[part + "_docs"] += 1
                s[part + "_chars"] += len(d["text"])
            stats[path.stem] = s
            print(f"{path.stem:24s} train {s['train_chars']:>9,} zn. ({s['train_docs']} dok.)"
                  f"  held-out {s['heldout_chars']:>8,} zn. ({s['heldout_docs']} dok.)",
                  file=sys.stderr, flush=True)
    with open(OUT / "sample_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

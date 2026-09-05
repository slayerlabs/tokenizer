"""Wytnij korpus treningowy i held-out z SlayerLab/hplt-v3-pl-cleaned.

Held-out pochodzi z INNYCH row-groupów parquet niż trening (brak wycieku).
Pliki lądują w data/ (gitignore) — nie trzymamy korpusu w repo.

    uv run --with pyarrow --with huggingface_hub python prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

DATASET = "SlayerLab/hplt-v3-pl-cleaned"
SHARD = "data/european_hplt_v3_pl_bin8_6_p0.parquet"


def ensure_shard(local_dir: str) -> str:
    path = os.path.join(local_dir, SHARD)
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download
    print(f"pobieram {DATASET}/{SHARD} ...")
    return hf_hub_download(
        repo_id=DATASET, filename=SHARD, repo_type="dataset", local_dir=local_dir
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/hplt-pl")
    ap.add_argument("--out-dir", default="../data/hplt-pl")
    ap.add_argument("--train-mb", type=float, default=300.0)
    ap.add_argument("--heldout-mb", type=float, default=20.0)
    ap.add_argument("--heldout-row-group", type=int, default=24,
                    help="row-group trzymany wyłącznie na eval (ostatni)")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    path = ensure_shard(args.data_dir)
    pf = pq.ParquetFile(path)
    n_rg = pf.num_row_groups
    print(f"{path}: {pf.metadata.num_rows} dok., {n_rg} row-groupów")

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "corpus_train.txt")
    held_path = os.path.join(args.out_dir, "corpus_heldout.jsonl")

    train_budget = int(args.train_mb * 1024 * 1024)
    held_budget = int(args.heldout_mb * 1024 * 1024)

    t0 = time.time()
    written = docs = 0
    with open(train_path, "w", encoding="utf-8") as f:
        for rg in range(n_rg):
            if rg == args.heldout_row_group:
                continue
            tbl = pf.read_row_group(rg, columns=["text"])
            for text in tbl.column("text").to_pylist():
                if not text:
                    continue
                f.write(text)
                f.write("\n")
                written += len(text.encode("utf-8")) + 1
                docs += 1
                if written >= train_budget:
                    break
            print(f"  rg{rg}: {written/1e6:.1f} MB / {docs} dok.", flush=True)
            if written >= train_budget:
                break
    print(f"train -> {train_path}: {written/1e6:.1f} MB, {docs} dok. "
          f"({time.time()-t0:.1f}s)")

    hw = hdocs = 0
    tbl = pf.read_row_group(args.heldout_row_group, columns=["id", "text"])
    with open(held_path, "w", encoding="utf-8") as f:
        for i, text in zip(tbl.column("id").to_pylist(),
                           tbl.column("text").to_pylist()):
            if not text:
                continue
            f.write(json.dumps({"id": i, "text": text}, ensure_ascii=False) + "\n")
            hw += len(text.encode("utf-8"))
            hdocs += 1
            if hw >= held_budget:
                break
    print(f"heldout -> {held_path}: {hw/1e6:.1f} MB, {hdocs} dok. "
          f"(row-group {args.heldout_row_group}, nieużywany w treningu)")


if __name__ == "__main__":
    main()

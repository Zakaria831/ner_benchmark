# make_splits_winer.py

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List


def _year_from_relpath(rel_txt: Path) -> str:
    for part in rel_txt.parts:
        if part.isdigit() and len(part) == 4:
            return part
    return "unknown"


def _stable_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="splits")
    ap.add_argument("--dev_ratio", type=float, default=0.10, help="Dev ratio within TRAIN years")
    ap.add_argument("--train_years", type=str, default="2016,2017")
    ap.add_argument("--test_years", type=str, default="2018")
    ap.add_argument("--seed", type=int, default=13, help="Salt deterministic split")
    args = ap.parse_args()

    winer_root = Path(args.winer_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_years = {y.strip() for y in args.train_years.split(",") if y.strip()}
    test_years = {y.strip() for y in args.test_years.split(",") if y.strip()}

    # Collect records
    records: List[Dict] = []
    for txt_path in winer_root.rglob("*.txt"):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        rel_txt = txt_path.relative_to(winer_root)
        rel_noext = rel_txt.with_suffix("") 
        doc_id = str(rel_noext).replace("\\", "/")  
        year = _year_from_relpath(rel_txt)

        records.append(
            {
                "doc_id": doc_id,
                "txt_path": str(txt_path),
                "ann_path": str(ann_path),
                "year": year,
                "rel_path": str(rel_txt).replace("\\", "/"),
            }
        )

    if not records:
        raise SystemExit("No .txt/.ann pairs found. Check --winer_root.")

    train_pool = [r for r in records if r["year"] in train_years]
    test_pool = [r for r in records if r["year"] in test_years]

    if not train_pool:
        raise SystemExit(f"No training records found for years={sorted(train_years)}")
    if not test_pool:
        raise SystemExit(f"No test records found for years={sorted(test_years)}")

    # Deterministic dev split from train_pool
    dev: List[Dict] = []
    train: List[Dict] = []
    for r in train_pool:
        key = f"{r['doc_id']}|{r['year']}|{args.seed}"
        p = (_stable_hash(key) % 10_000) / 10_000.0
        if p < args.dev_ratio:
            dev.append(r)
        else:
            train.append(r)

    if len(dev) < 10:
        print(f"[WARN] dev is very small ({len(dev)}). Consider increasing --dev_ratio.")

    split_obj = {
        "meta": {
            "winer_root": str(winer_root),
            "train_years": sorted(train_years),
            "test_years": sorted(test_years),
            "dev_ratio": args.dev_ratio,
            "seed": args.seed,
            "doc_id_format": "relative path without extension, e.g. 2016/01/article123",
        },
        "train": train,
        "dev": dev,
        "test": test_pool,
    }

    out_json = out_dir / "winer_splits.json"
    out_json.write_text(json.dumps(split_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "train_ids.txt").write_text("\n".join([r["doc_id"] for r in train]) + "\n", encoding="utf-8")
    (out_dir / "dev_ids.txt").write_text("\n".join([r["doc_id"] for r in dev]) + "\n", encoding="utf-8")
    (out_dir / "test_ids.txt").write_text("\n".join([r["doc_id"] for r in test_pool]) + "\n", encoding="utf-8")

    print(f"Saved splits to: {out_json}")
    print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test_pool)}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import torch

from load_winer import load_winer

Entity = Tuple[int, int, str]
WINER_TARGET_LABELS = ["Person", "Location", "Organization", "Date", "Event", "Product", "Hour"]


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    _ensure_outdir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_ids_file(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"ids_file not found: {p}")
    ids: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        ids.append(s)
    return ids


def _filter_docs_by_ids(docs: List[dict], ids: List[str]) -> List[dict]:
    want = set(ids)
    out: List[dict] = []
    for d in docs:
        doc_id = str(d.get("doc_id", ""))
        if doc_id in want or Path(doc_id).name in want:
            out.append(d)
    return out


def _unique_sorted(entities: List[Entity]) -> List[Entity]:
    return sorted(set(entities), key=lambda x: (x[0], x[1], x[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="urchade/gliner_multi-v2.1")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--ids_file", type=str, default="")
    args = ap.parse_args()

    docs = load_winer(args.winer_root)
    if args.ids_file:
        ids = _read_ids_file(args.ids_file)
        docs = _filter_docs_by_ids(docs, ids)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("No documents found (after filtering). Check --winer_root/--ids_file.")

    device = "cpu"
    if (not args.force_cpu) and torch.cuda.is_available():
        device = "cuda"

    # GLiNER import (package name depends on install)
    try:
        from gliner import GLiNER
    except Exception as e:
        raise SystemExit(
            "GLiNER not installed. Try: pip install gliner\n"
            f"Import error: {e}"
        )

    model = GLiNER.from_pretrained(args.model)
    model.to(device)

    out_path = Path(args.out) if args.out else Path(f"results/gliner_{args.model.replace('/', '_')}_winer.jsonl")

    rows: List[dict] = []
    t_all0 = time.perf_counter()

    for d in docs:
        t0 = time.perf_counter()

        preds = model.predict_entities(
            d["text"],
            labels=WINER_TARGET_LABELS,
            threshold=args.threshold,
        )
        ents: List[Entity] = []
        for p in preds:
            s = int(p.get("start", -1))
            e = int(p.get("end", -1))
            lab = str(p.get("label", "")).strip()
            if s >= 0 and e > s and lab:
                ents.append((s, e, lab))

        ents = _unique_sorted(ents)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {"doc_id": d["doc_id"], "model": args.model, "entities": ents, "elapsed_ms": round(elapsed_ms, 3)}
        )

    total_ms = (time.perf_counter() - t_all0) * 1000.0
    _write_jsonl(out_path, rows)

    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Docs:   {len(rows)}")
    print(f"Out:    {out_path}")
    print(f"Total:  {total_ms:.2f} ms | Avg: {total_ms / max(1,len(rows)):.2f} ms/doc")


if __name__ == "__main__":
    main()
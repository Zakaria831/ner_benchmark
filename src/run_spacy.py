# run_spacy.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import spacy

from load_winer import load_winer

Entity = Tuple[int, int, str]


@dataclass
class DocPred:
    doc_id: str
    entities: List[Entity]
    elapsed_ms: float


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _iter_docs(docs: List[Dict], limit: Optional[int] = None) -> Iterable[Dict]:
    if limit is None or limit <= 0:
        return docs
    return docs[:limit]


def _predict_spacy(
    nlp,
    docs: List[Dict],
    batch_size: int,
    n_process: int,
) -> List[DocPred]:
    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    preds: List[DocPred] = []
    t0 = time.perf_counter()

    for doc_id, doc in zip(doc_ids, nlp.pipe(texts, batch_size=batch_size, n_process=n_process)):
        t_doc0 = time.perf_counter()
        ents: List[Entity] = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        elapsed_ms = (time.perf_counter() - t_doc0) * 1000.0
        preds.append(DocPred(doc_id=doc_id, entities=ents, elapsed_ms=elapsed_ms))

    _ = t0  # keeps structure simple; overall time computed outside if needed
    return preds


def _write_jsonl(out_path: Path, rows: Iterable[dict]) -> None:
    _ensure_outdir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True, help="Path to WiNER-fr root folder")
    ap.add_argument("--model", type=str, default="fr_core_news_md", help="SpaCy model name")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path (default: results/spacy_<model>_winer.jsonl)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_process", type=int, default=1, help=">1 uses multiprocessing (CPU only)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all)")
    ap.add_argument("--use_gpu", action="store_true", help="Try to use GPU (mainly for transformer pipelines)")
    ap.add_argument("--disable", type=str, default="", help="Comma-separated spaCy components to disable (optional)")
    args = ap.parse_args()

    winer_root = args.winer_root
    model_name = args.model

    disable = [c.strip() for c in args.disable.split(",") if c.strip()]
    if args.use_gpu:
        try:
            spacy.require_gpu()
        except Exception:
            pass

    nlp = spacy.load(model_name, disable=disable)

    docs = load_winer(winer_root)
    docs = list(_iter_docs(docs, None if args.limit <= 0 else args.limit))

    if not docs:
        raise SystemExit("No documents found. Check --winer_root path.")

    out_path = Path(
        args.out if args.out else f"results/spacy_{model_name.replace('/', '_')}_winer.jsonl"
    )

    t0 = time.perf_counter()
    preds = _predict_spacy(nlp, docs, batch_size=args.batch_size, n_process=args.n_process)
    total_ms = (time.perf_counter() - t0) * 1000.0

    rows = (
        {
            "doc_id": p.doc_id,
            "model": model_name,
            "entities": p.entities,
            "elapsed_ms": round(p.elapsed_ms, 3),
        }
        for p in preds
    )
    _write_jsonl(out_path, rows)

    avg_ms = total_ms / max(1, len(preds))
    print(f"Model: {model_name}")
    print(f"Docs:  {len(preds)}")
    print(f"Out:   {out_path}")
    print(f"Total: {total_ms:.2f} ms | Avg: {avg_ms:.2f} ms/doc")


if __name__ == "__main__":
    main()
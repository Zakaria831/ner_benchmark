from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import spacy

from load_winer import load_winer

Entity = Tuple[int, int, str]


def _ensure_outdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(out_path: Path, rows: Iterable[dict]) -> None:
    _ensure_outdir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _iter_docs(docs: List[Dict], limit: Optional[int] = None) -> List[Dict]:
    if limit is None or limit <= 0:
        return docs
    return docs[:limit]


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


def _filter_docs_by_ids(docs: List[Dict], ids: List[str]) -> List[Dict]:
    want = set(ids)
    out: List[Dict] = []
    for d in docs:
        doc_id = str(d.get("doc_id", ""))
        if doc_id in want or Path(doc_id).name in want:
            out.append(d)
    return out


def _to_winer_label(raw: str) -> Optional[str]:
    up = str(raw).strip().upper()
    if not up:
        return None

    # spaCy FR often uses PER/LOC/ORG
    if up in ("PER", "PERSON"):
        return "Person"
    if up in ("LOC", "GPE", "LOCATION"):
        return "Location"
    if up in ("ORG", "ORGANIZATION"):
        return "Organization"
    if up == "DATE":
        return "Date"
    if up in ("TIME",):
        return "Hour"

    # Ignore other types (MISC, etc.)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="fr_core_news_md")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_process", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--disable", type=str, default="")
    ap.add_argument("--ids_file", type=str, default="")
    args = ap.parse_args()

    disable = [c.strip() for c in args.disable.split(",") if c.strip()]

    if args.use_gpu:
        try:
            spacy.require_gpu()
        except Exception:
            pass

    nlp = spacy.load(args.model, disable=disable)

    docs = load_winer(args.winer_root)
    if args.ids_file:
        ids = _read_ids_file(args.ids_file)
        docs = _filter_docs_by_ids(docs, ids)

    docs = _iter_docs(docs, None if args.limit <= 0 else args.limit)
    if not docs:
        raise SystemExit("No documents found (after filtering). Check --winer_root/--ids_file.")

    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    out_path = Path(args.out) if args.out else Path(f"results/predictions/spacy_{args.model.replace('/', '_')}_winer.jsonl")

    t0 = time.perf_counter()
    docs_iter = list(nlp.pipe(texts, batch_size=args.batch_size, n_process=args.n_process))
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_ms / max(1, len(docs_iter))

    rows: List[dict] = []
    for doc_id, doc in zip(doc_ids, docs_iter):
        ents: List[Entity] = []
        for ent in doc.ents:
            lab = _to_winer_label(ent.label_)
            if lab is None:
                continue
            if ent.end_char > ent.start_char:
                ents.append((ent.start_char, ent.end_char, lab))

        rows.append({"doc_id": doc_id, "model": f"spacy_{args.model}", "entities": ents, "elapsed_ms": round(avg_ms, 3)})

    _write_jsonl(out_path, rows)

    print(f"Model: {args.model}")
    print(f"Docs:  {len(rows)}")
    print(f"Out:   {out_path}")
    print(f"Total: {total_ms:.2f} ms | Avg: {avg_ms:.2f} ms/doc")


if __name__ == "__main__":
    main()
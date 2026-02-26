from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import flair
from flair.models import SequenceTagger
from flair.splitter import SegtokSentenceSplitter

from load_winer import load_winer

Entity = Tuple[int, int, str]


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


def _to_winer_label(raw: str) -> Optional[str]:
    up = str(raw).strip().upper()
    if not up:
        return None
    if up in ("PER", "PERSON"):
        return "Person"
    if up in ("LOC", "LOCATION", "GPE"):
        return "Location"
    if up in ("ORG", "ORGANIZATION"):
        return "Organization"
    if up == "DATE":
        return "Date"
    if up in ("TIME", "HOUR"):
        return "Hour"
    if up == "EVENT":
        return "Event"
    if up == "PRODUCT":
        return "Product"
    if up == "MISC":
        return None
    return None


def predict_flair_doc(
    text: str,
    tagger: SequenceTagger,
    splitter: SegtokSentenceSplitter,
    batch_size: int,
) -> List[Entity]:
    sentences = splitter.split(text)
    if not sentences:
        return []

    tagger.predict(sentences, mini_batch_size=batch_size, verbose=False)

    ents: List[Entity] = []
    for sent in sentences:
        base = getattr(sent, "start_position", 0) or 0
        for span in sent.get_spans("ner"):
            raw = span.get_label("ner").value
            lab = _to_winer_label(raw)
            if lab is None:
                continue

            start = base + span.start_position
            end = base + span.end_position
            if end > start:
                ents.append((start, end, lab))

    return _unique_sorted(ents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="flair/ner-french")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force_cpu", action="store_true")
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

    device = torch.device("cpu")
    if (not args.force_cpu) and torch.cuda.is_available():
        device = torch.device("cuda:0")
    flair.device = device

    tagger: SequenceTagger = SequenceTagger.load(args.model)
    splitter = SegtokSentenceSplitter()

    out_path = Path(args.out) if args.out else Path(f"results/predictions/flair_{args.model.replace('/', '_')}_winer.jsonl")

    rows: List[dict] = []
    t_all0 = time.perf_counter()

    for d in docs:
        t0 = time.perf_counter()
        ents = predict_flair_doc(d["text"], tagger, splitter, args.batch_size)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"doc_id": d["doc_id"], "model": f"flair_{args.model}", "entities": ents, "elapsed_ms": round(elapsed_ms, 3)})

    total_ms = (time.perf_counter() - t_all0) * 1000.0
    _write_jsonl(out_path, rows)

    avg_ms = total_ms / max(1, len(rows))
    print(f"Device: {device}")
    print(f"Model:  {args.model}")
    print(f"Docs:   {len(rows)}")
    print(f"Out:    {out_path}")
    print(f"Total:  {total_ms:.2f} ms | Avg: {avg_ms:.2f} ms/doc")


if __name__ == "__main__":
    main()
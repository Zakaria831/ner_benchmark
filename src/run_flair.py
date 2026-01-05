# src/run_flair.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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


def _unique_sorted(entities: List[Entity]) -> List[Entity]:
    return sorted(set(entities), key=lambda x: (x[0], x[1], x[2]))


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
            label = span.get_label("ner").value  # e.g., PER, LOC, ORG, MISC
            start = base + span.start_position
            end = base + span.end_position
            if end > start:
                ents.append((start, end, label))

    return _unique_sorted(ents)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True, help="Path to WiNER-fr root folder")
    ap.add_argument("--model", type=str, default="flair/ner-french", help="Flair model name")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    ap.add_argument("--batch_size", type=int, default=16, help="Mini-batch size for Flair tagging")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of docs (0 = all)")
    ap.add_argument("--force_cpu", action="store_true", help="Force CPU even if GPU is available")
    args = ap.parse_args()

    docs = load_winer(args.winer_root)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("No documents found. Check --winer_root path.")

    device = torch.device("cpu")
    if (not args.force_cpu) and torch.cuda.is_available():
        device = torch.device("cuda:0")
    flair.device = device

    tagger: SequenceTagger = SequenceTagger.load(args.model)
    splitter = SegtokSentenceSplitter()

    out_path = Path(args.out if args.out else f"results/flair_{args.model.replace('/', '_')}_winer.jsonl")

    rows: List[dict] = []
    t_all0 = time.perf_counter()

    for d in docs:
        t0 = time.perf_counter()
        ents = predict_flair_doc(
            text=d["text"],
            tagger=tagger,
            splitter=splitter,
            batch_size=args.batch_size,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "doc_id": d["doc_id"],
                "model": args.model,
                "entities": ents,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

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
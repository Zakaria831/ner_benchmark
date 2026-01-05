# src/run_hf.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from load_winer import load_winer

Entity = Tuple[int, int, str]


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _unique_sorted(entities: List[Entity]) -> List[Entity]:
    return sorted(set(entities), key=lambda x: (x[0], x[1], x[2]))


def _decode_iob_entities(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    pred_label_ids: List[int],
    id2label: Dict[int, str],
) -> List[Entity]:
    entities: List[Entity] = []
    cur_label: Optional[str] = None
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    def _close():
        nonlocal cur_label, cur_start, cur_end
        if cur_label is not None and cur_start is not None and cur_end is not None and cur_end > cur_start:
            entities.append((cur_start, cur_end, cur_label))
        cur_label, cur_start, cur_end = None, None, None

    for (start, end), lid in zip(offset_mapping, pred_label_ids):
        if start == end:  # special tokens
            continue

        tag = id2label[int(lid)]
        if tag == "O":
            _close()
            continue

        if "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = "B", tag

        if prefix == "B":
            _close()
            cur_label, cur_start, cur_end = label, start, end
        elif prefix == "I":
            if cur_label == label and cur_end is not None and start >= cur_end:
                cur_end = end
            else:
                _close()
                cur_label, cur_start, cur_end = label, start, end
        else:
            _close()

    _close()
    return entities


@torch.no_grad()
def predict_doc_entities(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    stride: int,
) -> List[Entity]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    id2label = model.config.id2label
    all_entities: List[Entity] = []

    input_ids_list = enc["input_ids"]
    attn_list = enc["attention_mask"]
    offsets_list = enc["offset_mapping"]

    for input_ids, attn, offsets in zip(input_ids_list, attn_list, offsets_list):
        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attn], device=device)

        logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()

        ents = _decode_iob_entities(text, offsets, pred_ids, id2label)
        all_entities.extend(ents)

    return _unique_sorted(all_entities)


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    _ensure_outdir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True, help="Path to WiNER-fr root folder")
    ap.add_argument("--model", type=str, default="Jean-Baptiste/camembert-ner")
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    model.eval()

    out_path = Path(
        args.out if args.out else f"results/hf_{args.model.replace('/', '_')}_winer.jsonl"
    )

    rows: List[dict] = []
    t_all0 = time.perf_counter()

    for d in docs:
        t0 = time.perf_counter()
        ents = predict_doc_entities(
            text=d["text"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
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
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import torch
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer

Entity = Tuple[int, int, str]


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _unique_sorted(entities: List[Entity]) -> List[Entity]:
    return sorted(set(entities), key=lambda x: (x[0], x[1], x[2]))


def _merge_overlapping_same_label(ents: List[Entity]) -> List[Entity]:
    if not ents:
        return []
    ents = sorted(ents, key=lambda x: (x[2], x[0], x[1]))
    merged: List[Entity] = []
    cur_s, cur_e, cur_lab = ents[0]
    for s, e, lab in ents[1:]:
        if lab == cur_lab and s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_lab))
            cur_s, cur_e, cur_lab = s, e, lab
    merged.append((cur_s, cur_e, cur_lab))
    return _unique_sorted(merged)


def _decode_iob_entities(
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
        if start == end:
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
            if cur_label == label and cur_start is not None and cur_end is not None:
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
    merge_windows: bool = True,
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

    for input_ids, attn, offsets in zip(enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]):
        input_ids_t = torch.tensor([input_ids], device=device)
        attn_t = torch.tensor([attn], device=device)

        logits = model(input_ids=input_ids_t, attention_mask=attn_t).logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        all_entities.extend(_decode_iob_entities(offsets, pred_ids, id2label))

    if merge_windows:
        all_entities = _merge_overlapping_same_label(all_entities)

    return _unique_sorted(all_entities)


def _load_brat_docs_rel_docid(winer_root: Path) -> List[dict]:
    docs: List[dict] = []
    for txt_path in winer_root.rglob("*.txt"):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8")
        rel_noext = txt_path.relative_to(winer_root).with_suffix("")
        doc_id = str(rel_noext).replace("\\", "/")
        docs.append({"doc_id": doc_id, "text": text})
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, required=True, help=".../dataset")
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--model_dir", type=str, required=True, help=".../best_model")
    ap.add_argument("--out_jsonl", type=str, required=True)

    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--no_merge_windows", action="store_true")
    ap.add_argument("--model_name_in_jsonl", type=str, default="", help="value stored in JSONL field 'model'")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    winer_root = Path(args.winer_root)
    model_dir = Path(args.model_dir)
    out_jsonl = Path(args.out_jsonl)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")
    if not winer_root.exists():
        raise SystemExit(f"WiNER root not found: {winer_root}")
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    ds = load_from_disk(str(dataset_dir))
    if "test" not in ds:
        raise SystemExit("Dataset has no 'test' split.")
    test_doc_ids: Set[str] = set(ds["test"]["doc_id"])

    docs = _load_brat_docs_rel_docid(winer_root)
    docs = [d for d in docs if d["doc_id"] in test_doc_ids]
    if not docs:
        raise SystemExit("No BRAT docs matched TEST doc_ids (doc_id format mismatch).")

    device = torch.device("cpu")
    if (not args.force_cpu) and torch.cuda.is_available():
        device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    rows = []
    t_all0 = time.perf_counter()

    model_name_field = args.model_name_in_jsonl.strip() or model_dir.parent.name

    for d in docs:
        t0 = time.perf_counter()
        ents = predict_doc_entities(
            text=d["text"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
            merge_windows=(not args.no_merge_windows),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        rows.append(
            {
                "doc_id": d["doc_id"],
                "model": model_name_field,
                "entities": ents,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    total_ms = (time.perf_counter() - t_all0) * 1000.0
    _ensure_outdir(out_jsonl)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Device: {device}")
    print(f"Test docs: {len(rows)}")
    print(f"Out: {out_jsonl}")
    print(f"Total: {total_ms:.2f} ms | Avg: {total_ms / max(1,len(rows)):.2f} ms/doc")


if __name__ == "__main__":
    main()
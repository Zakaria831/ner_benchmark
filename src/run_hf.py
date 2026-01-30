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
    """
    Map common NER labels to WiNER labels.
    Returns None if not supported (ex: MISC).
    """
    if raw is None:
        return None
    up = str(raw).strip().upper()
    if not up:
        return None

    # Common variants
    if up in ("PER", "PERSON"):
        return "Person"
    if up in ("LOC", "LOCATION", "GPE"):
        return "Location"
    if up in ("ORG", "ORGANIZATION"):
        return "Organization"
    if up in ("DATE",):
        return "Date"
    if up in ("TIME", "HOUR"):
        return "Hour"

    # WiNER-only labels are not predicted by most HF NER models,
    # but keep them if they appear
    if up == "EVENT":
        return "Event"
    if up == "PRODUCT":
        return "Product"

   
    if up == "MISC":
        return None


    return None


def _decode_iob_entities(
    offset_mapping: List[Tuple[int, int]],
    pred_label_ids: List[int],
    id2label: Dict[int, str],
) -> List[Entity]:
    """
    Convert BIO/IOB token tags to char-span entities using offset_mapping.
    Output labels are normalized to WiNER label names.
    """
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
            continue  # special tokens / padding

        tag = str(id2label[int(lid)])
        if tag == "O":
            _close()
            continue

        # Handle BIO or plain labels
        if "-" in tag:
            prefix, raw_label = tag.split("-", 1)
        else:
            prefix, raw_label = "B", tag

        winer_label = _to_winer_label(raw_label)
        if winer_label is None:
            # treat as O for our eval purposes
            _close()
            continue

        if prefix == "B":
            _close()
            cur_label, cur_start, cur_end = winer_label, start, end
        elif prefix == "I":
            # extend if same entity is open
            if cur_label == winer_label and cur_start is not None and cur_end is not None:
                cur_end = end
            else:
                _close()
                cur_label, cur_start, cur_end = winer_label, start, end
        else:
            _close()

    _close()
    return entities


def _merge_overlapping_same_label(ents: List[Entity]) -> List[Entity]:
    """Merge overlapping/touching spans with same label (helps with sliding windows)."""
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

        ents = _decode_iob_entities(offsets, pred_ids, id2label)
        all_entities.extend(ents)

    if merge_windows:
        all_entities = _merge_overlapping_same_label(all_entities)

    return _unique_sorted(all_entities)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--model", type=str, default="Jean-Baptiste/camembert-ner")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--no_merge_windows", action="store_true")
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    model.eval()

    out_path = Path(args.out) if args.out else Path(f"results/predictions/hf_{args.model.replace('/', '_')}_winer.jsonl")

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
            merge_windows=not args.no_merge_windows,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "doc_id": d["doc_id"],
                "model": f"hf_{args.model}",
                "entities": ents,  # always [s,e,"Person|Location|..."]
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
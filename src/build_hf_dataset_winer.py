from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

Entity = Tuple[int, int, str]  # (start, end, label)

WINER_LABELS = ["Person", "Organization", "Location", "Date", "Hour", "Event", "Product"]

# Priority order (highest first) used to flatten nested/overlapping entities
PRIORITY = {
    "Person": 7,
    "Organization": 6,
    "Location": 5,
    "Date": 4,
    "Hour": 3,
    "Event": 2,
    "Product": 1,
}


def load_brat(txt_path: Path, ann_path: Path) -> Tuple[str, List[Entity]]:
    text = txt_path.read_text(encoding="utf-8")
    ents: List[Entity] = []

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue

        meta = parts[1].strip()  # "Label s e" or "Label s e;s e"
        if not meta:
            continue

        label = meta.split()[0].strip()
        if label not in PRIORITY:
            continue

        rest = meta[len(label) :].strip()
        chunks = [c.strip() for c in rest.split(";") if c.strip()]
        for ch in chunks:
            nums = ch.split()
            if len(nums) != 2:
                continue
            try:
                s = int(nums[0])
                e = int(nums[1])
            except ValueError:
                continue
            if e > s:
                ents.append((s, e, label))

    return text, ents


def choose_entity_for_token(token_span: Tuple[int, int], gold_ents: List[Entity]) -> Optional[Entity]:
    ts, te = token_span
    if ts == te:
        return None

    best: Optional[Entity] = None
    best_score = -1

    for (s, e, lab) in gold_ents:
        if max(ts, s) < min(te, e):  # overlap
            score = PRIORITY.get(lab, 0)
            if score > best_score:
                best_score = score
                best = (s, e, lab)
            elif score == best_score and best is not None:
                bs, be, _ = best
                if (e - s) > (be - bs):  # tie-breaker: longer
                    best = (s, e, lab)

    return best


def build_bio_labels(
    offsets: List[Tuple[int, int]],
    gold_ents: List[Entity],
    ignore_special: bool = True,
) -> List[str]:
    labels: List[str] = []
    prev_ent: Optional[Entity] = None

    for (s, e) in offsets:
        if ignore_special and s == e:
            labels.append("IGN")  # -> -100
            prev_ent = None
            continue

        chosen = choose_entity_for_token((s, e), gold_ents)
        if chosen is None:
            labels.append("O")
            prev_ent = None
            continue

        ent_s, _, ent_lab = chosen
        if s == ent_s or prev_ent != chosen:
            labels.append(f"B-{ent_lab}")
        else:
            labels.append(f"I-{ent_lab}")

        prev_ent = chosen

    return labels


def make_label_maps() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = ["O"]
    for lab in WINER_LABELS:
        labels.append(f"B-{lab}")
        labels.append(f"I-{lab}")
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label


def encode_one(
    tokenizer,
    text: str,
    ents: List[Entity],
    label2id: Dict[str, int],
    max_length: int,
    stride: int,
) -> List[Dict]:
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        padding=False,
    )

    examples: List[Dict] = []
    for input_ids, attn, offsets in zip(enc["input_ids"], enc["attention_mask"], enc["offset_mapping"]):
        bio = build_bio_labels(offsets, ents)

        lab_ids: List[int] = []
        for t in bio:
            if t == "IGN":
                lab_ids.append(-100)
            else:
                lab_ids.append(label2id.get(t, label2id["O"]))

        examples.append({"input_ids": input_ids, "attention_mask": attn, "labels": lab_ids})

    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits_json).read_text(encoding="utf-8"))
    labels, label2id, id2label = make_label_maps()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "label2id.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "id2label.json").write_text(json.dumps(id2label, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def build_split(split_name: str) -> Dataset:
        rows = []
        recs = splits[split_name]
        print(f"[{split_name}] docs={len(recs)}")

        for idx, rec in enumerate(recs, start=1):
            if idx % 200 == 0:
                print(f"[{split_name}] processed {idx}/{len(recs)} | rows={len(rows)}")

            txt_path = Path(rec["txt_path"])
            ann_path = Path(rec["ann_path"])
            doc_id = rec["doc_id"]

            text, ents = load_brat(txt_path, ann_path)
            examples = encode_one(
                tokenizer=tokenizer,
                text=text,
                ents=ents,
                label2id=label2id,
                max_length=args.max_length,
                stride=args.stride,
            )
            for i, ex in enumerate(examples):
                rows.append({"doc_id": doc_id, "window_id": i, **ex})

        print(f"[{split_name}] done | rows={len(rows)}")
        return Dataset.from_list(rows)

    ds = DatasetDict(
        {
            "train": build_split("train"),
            "validation": build_split("dev"),
            "test": build_split("test"),
        }
    )

    ds.save_to_disk(str(out_dir / "dataset"))
    print(f"Saved HF dataset to: {out_dir / 'dataset'}")
    print(ds)
    print("Priority order:", PRIORITY)


if __name__ == "__main__":
    main()
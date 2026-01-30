from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

Entity = Tuple[int, int, str]


def _parse_brat_spans(span_str: str) -> List[Tuple[int, int]]:
    """
    BRAT can represent discontinuous spans:
      "Label 10 20;30 40"
    WiNER is usually contiguous, but we support multi-span safely.
    We return a list of (start, end).
    """
    spans: List[Tuple[int, int]] = []
    # span_str looks like: "Organization 10 13" or "Organization 10 13;15 20"
    parts = span_str.split()
    if len(parts) < 3:
        return spans

    # Remove the label, keep everything else as span tokens
    # Example: ["Organization", "10", "13;15", "20"] -> tricky because ';' can stick
    # Better: rebuild from original after label
    rest = span_str[len(parts[0]):].strip()  # after label
    # rest example: "10 13" or "10 13;15 20"
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
            spans.append((s, e))
    return spans


def load_brat_file(txt_path: Path, ann_path: Path) -> Dict:
    text = txt_path.read_text(encoding="utf-8")
    entities: List[Entity] = []

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        # parts[1] contains label + span(s)
        # example: "Organization 10 13" or "Organization 10 13;15 20"
        meta = parts[1].strip()
        if not meta:
            continue

        label = meta.split()[0].strip()
        spans = _parse_brat_spans(meta)

        # If multi-span, we add each segment as an entity.
        # (Alternative: merge into one span; but BRAT discontinuous spans are not contiguous)
        for (start, end) in spans:
            entities.append((start, end, label))

    return {
        "text": text,
        "entities": entities,
        "doc_id": txt_path.stem,
    }


def load_winer(root_dir: str) -> List[Dict]:
    root = Path(root_dir).resolve()
    documents: List[Dict] = []

    for txt_path in sorted(root.rglob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        rel_id = txt_path.relative_to(root).with_suffix("").as_posix()  # unique id
        doc = load_brat_file(txt_path, ann_path)
        doc["doc_id"] = rel_id
        documents.append(doc)

    return documents



if __name__ == "__main__":
    winer_path = "data/WiNER-fr"
    docs = load_winer(winer_path)

    print(f"Loaded {len(docs)} documents")
    if docs:
        print(docs[0]["doc_id"])
        print(docs[0]["entities"][:5])

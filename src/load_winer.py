from pathlib import Path
from typing import List, Tuple, Dict


Entity = Tuple[int, int, str]


def load_brat_file(txt_path: Path, ann_path: Path) -> Dict:
    text = txt_path.read_text(encoding="utf-8")
    entities: List[Entity] = []

    for line in ann_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue

        parts = line.split("\t")
        if len(parts) < 3:
            continue

        meta = parts[1].split()
        label = meta[0]

        try:
            start = int(meta[1])
            end = int(meta[2])
        except ValueError:
            continue

        entities.append((start, end, label))

    return {
        "text": text,
        "entities": entities,
        "doc_id": txt_path.stem,
    }


def load_winer(root_dir: str) -> List[Dict]:
    root = Path(root_dir)
    documents: List[Dict] = []

    for txt_path in root.rglob("*.txt"):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        doc = load_brat_file(txt_path, ann_path)
        documents.append(doc)

    return documents


if __name__ == "__main__":
    winer_path = "data/WiNER-fr"
    docs = load_winer(winer_path)

    print(f"Loaded {len(docs)} documents")
    print(docs[0]["doc_id"])
    print(docs[0]["entities"][:5])
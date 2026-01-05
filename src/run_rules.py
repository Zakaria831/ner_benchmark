# src/run_rules.py
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

from load_winer import load_winer

Entity = Tuple[int, int, str]


MONTHS_FR = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)

DATE_PATTERNS = [
    # 18/02/2017 or 18-02-2017 or 18.02.2017
    re.compile(r"\b([0-3]?\d)[/\-.]([01]?\d)[/\-.](\d{4})\b"),
    # 2017-02-18
    re.compile(r"\b(\d{4})-([01]\d)-([0-3]\d)\b"),
    # 18 février 2017 / 1er janvier 2020
    re.compile(rf"\b([0-3]?\d|1er)\s+({MONTHS_FR})\s+(\d{{4}})\b", re.IGNORECASE),
    # février 2017
    re.compile(rf"\b({MONTHS_FR})\s+(\d{{4}})\b", re.IGNORECASE),
    # relative dates
    re.compile(r"\b(aujourd’hui|aujourd'hui|hier|demain|avant-hier|après-demain|apres-demain)\b", re.IGNORECASE),
    # day names
    re.compile(r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\b", re.IGNORECASE),
]

HOUR_PATTERNS = [
    # 14h / 14 h / 14h30 / 14 h 30
    re.compile(r"\b([01]?\d|2[0-3])\s*h\s*([0-5]\d)?\b", re.IGNORECASE),
    # 14:30 / 9:05
    re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b"),
    # 14 heures / 14 heure / 9 heures 30
    re.compile(r"\b([01]?\d|2[0-3])\s*heures?\s*([0-5]\d)?\b", re.IGNORECASE),
    # midi / minuit
    re.compile(r"\b(midi|minuit)\b", re.IGNORECASE),
]

# Avoid duplicates and keep stable ordering
def _unique_sorted(entities: List[Entity]) -> List[Entity]:
    return sorted(set(entities), key=lambda x: (x[0], x[1], x[2]))


def _find_spans(text: str, patterns: List[re.Pattern], label: str) -> List[Entity]:
    ents: List[Entity] = []
    for pat in patterns:
        for m in pat.finditer(text):
            start, end = m.span()
            if end > start:
                ents.append((start, end, label))
    return ents


def predict_rules(text: str, use_dates: bool = True, use_hours: bool = True) -> List[Entity]:
    ents: List[Entity] = []
    if use_dates:
        ents.extend(_find_spans(text, DATE_PATTERNS, "DATE"))
    if use_hours:
        ents.extend(_find_spans(text, HOUR_PATTERNS, "HOUR"))
    return _unique_sorted(ents)


def _ensure_outdir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(out_path: Path, rows: Iterable[Dict]) -> None:
    _ensure_outdir(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--out", type=str, default="", help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_dates", action="store_true", help="Disable DATE rules")
    ap.add_argument("--no_hours", action="store_true", help="Disable HOUR rules")
    args = ap.parse_args()

    docs = load_winer(args.winer_root)
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("No documents found. Check --winer_root path.")

    out_path = Path(args.out) if args.out else Path("results/rules_date_hour_winer.jsonl")

    rows: List[Dict] = []
    t_all0 = time.perf_counter()

    for d in docs:
        t0 = time.perf_counter()
        ents = predict_rules(
            d["text"],
            use_dates=not args.no_dates,
            use_hours=not args.no_hours,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rows.append(
            {
                "doc_id": d["doc_id"],
                "model": "RULES_DATE_HOUR",
                "entities": ents,
                "elapsed_ms": round(elapsed_ms, 3),
            }
        )

    total_ms = (time.perf_counter() - t_all0) * 1000.0
    _write_jsonl(out_path, rows)

    avg_ms = total_ms / max(1, len(rows))
    print(f"Docs:  {len(rows)}")
    print(f"Out:   {out_path}")
    print(f"Total: {total_ms:.2f} ms | Avg: {avg_ms:.2f} ms/doc")


if __name__ == "__main__":
    main()
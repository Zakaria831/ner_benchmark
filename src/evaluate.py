# src/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Optional

import pandas as pd

from load_winer import load_winer

Span = Tuple[int, int]
Entity = Tuple[int, int, str]

WINER_LABELS = ["Person", "Location", "Organization", "Date", "Event", "Product", "Hour"]


# ----------------------------
# Label mapping to WiNER schema
# ----------------------------
def normalize_label(model_name: str, raw_label: str) -> Optional[str]:
    """
    Returns a WiNER label, or None if label is unsupported / should be ignored.
    """
    if raw_label is None:
        return None
    lab = raw_label.strip()

    # Rules baseline
    if model_name.startswith("RULES"):
        if lab in ("DATE", "Date"):
            return "Date"
        if lab in ("HOUR", "TIME", "Hour"):
            return "Hour"
        return None

    # spaCy labels (usually: PERSON, LOC, ORG, DATE, TIME, GPE...)
    if model_name.startswith("spacy"):
        up = lab.upper()
        if up in ("PERSON",):
            return "Person"
        if up in ("LOC", "GPE"):
            return "Location"
        if up in ("ORG",):
            return "Organization"
        if up in ("DATE",):
            return "Date"
        if up in ("TIME",):
            return "Hour"
        return None

    # HF / Flair common labels (often PER/LOC/ORG/MISC)
    up = lab.upper()
    if up == "PER":
        return "Person"
    if up == "LOC":
        return "Location"
    if up == "ORG":
        return "Organization"

    # Some models may output DATE/TIME
    if up == "DATE":
        return "Date"
    if up == "TIME":
        return "Hour"

    # MISC is not part of WiNER tagset; treat as unsupported (ignore)
    if up == "MISC":
        return None

    # If already in WiNER format
    if lab in WINER_LABELS:
        return lab

    return None


def guess_family(model_field: str) -> str:
    m = model_field.lower()
    if m.startswith("flair/"):
        return "Flair"
    if "camembert" in m or "bert" in m or "distil" in m:
        return "HuggingFace"
    if m.startswith("fr_") or "spacy" in m:
        return "spaCy"
    if "heideltime" in m:
        return "HeidelTime"
    if "rules" in m:
        return "Rules"
    return "Other"


# ----------------------------
# IO helpers
# ----------------------------
def read_predictions_jsonl(path: Path) -> Dict[str, List[Entity]]:
    """
    Reads your JSONL predictions:
    {"doc_id": "...", "model": "...", "entities": [[s,e,label], ...], ...}

    Returns: doc_id -> list of (s,e,label)
    """
    out: Dict[str, List[Entity]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            ents = []
            for e in obj.get("entities", []):
                if len(e) != 3:
                    continue
                s, t, lab = int(e[0]), int(e[1]), str(e[2])
                ents.append((s, t, lab))
            out[doc_id] = ents
    return out


def build_gold_index(winer_root: str) -> Dict[str, List[Entity]]:
    docs = load_winer(winer_root)
    gold = {}
    for d in docs:
        # d["entities"] are (start,end,label) already from BRAT
        gold[d["doc_id"]] = [(int(s), int(e), str(lab)) for (s, e, lab) in d["entities"]]
    return gold


# ----------------------------
# Nested entity analysis
# ----------------------------
def is_nested(a: Span, b: Span) -> bool:
    """
    True if spans are nested (one strictly contains the other).
    """
    (s1, e1), (s2, e2) = a, b
    return (s1 < s2 and e2 < e1) or (s2 < s1 and e1 < e2)


def any_overlap(a: Span, b: Span) -> bool:
    (s1, e1), (s2, e2) = a, b
    return max(s1, s2) < min(e1, e2)


def nested_gold_spans(gold_ents: List[Entity]) -> Set[Span]:
    spans = [(s, e) for (s, e, _) in gold_ents]
    nested: Set[Span] = set()
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if is_nested(spans[i], spans[j]):
                nested.add(spans[i])
                nested.add(spans[j])
    return nested


def overlap_rate(pred_ents: List[Entity]) -> float:
    spans = [(s, e) for (s, e, _) in pred_ents]
    if len(spans) < 2:
        return 0.0
    overlaps = 0
    total_pairs = 0
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            total_pairs += 1
            if any_overlap(spans[i], spans[j]):
                overlaps += 1
    return overlaps / total_pairs if total_pairs else 0.0


# ----------------------------
# Metrics (exact match)
# ----------------------------
def as_set(ents: Iterable[Entity], model_name: str) -> Set[Entity]:
    """
    Normalize labels to WiNER schema and return a set of entities.
    """
    normed: Set[Entity] = set()
    for s, e, lab in ents:
        wl = normalize_label(model_name, lab)
        if wl is None:
            continue
        if e > s:
            normed.add((int(s), int(e), wl))
    return normed


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_one(
    model_name: str,
    gold_by_doc: Dict[str, List[Entity]],
    pred_by_doc: Dict[str, List[Entity]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - summary_df: micro/macro + nesting stats
      - per_label_df: P/R/F1 per WiNER label + support sizes
    """
    # Accumulators per label
    tp_l = {lab: 0 for lab in WINER_LABELS}
    fp_l = {lab: 0 for lab in WINER_LABELS}
    fn_l = {lab: 0 for lab in WINER_LABELS}

    # Nesting stats
    gold_nested_total = 0
    gold_total = 0
    nested_tp = 0
    nested_fn = 0

    # Prediction overlap (proxy for nesting support)
    pred_overlap_rates: List[float] = []

    all_doc_ids = set(gold_by_doc.keys()) | set(pred_by_doc.keys())

    for doc_id in all_doc_ids:
        gold_raw = gold_by_doc.get(doc_id, [])
        pred_raw = pred_by_doc.get(doc_id, [])

        gold_set = as_set(gold_raw, "WINER_GOLD")  # gold already in WiNER labels
        pred_set = as_set(pred_raw, model_name)

        # nesting gold spans
        gold_nested_sp = nested_gold_spans(list(gold_set))
        gold_total += len(gold_set)
        gold_nested_total += len(gold_nested_sp)

        # exact match TP/FP/FN
        inter = gold_set & pred_set
        only_pred = pred_set - gold_set
        only_gold = gold_set - pred_set

        # per label counts
        for (s, e, lab) in inter:
            tp_l[lab] += 1
        for (s, e, lab) in only_pred:
            fp_l[lab] += 1
        for (s, e, lab) in only_gold:
            fn_l[lab] += 1

        # nested detection (exact match): treat each nested gold span as positive
        if gold_nested_sp:
            for (s, e, lab) in gold_set:
                if (s, e) in gold_nested_sp:
                    if (s, e, lab) in pred_set:
                        nested_tp += 1
                    else:
                        nested_fn += 1

        pred_overlap_rates.append(overlap_rate(list(pred_set)))

    # Per-label dataframe
    rows = []
    for lab in WINER_LABELS:
        p, r, f1 = prf(tp_l[lab], fp_l[lab], fn_l[lab])
        support = tp_l[lab] + fn_l[lab]  # gold count for this label
        pred_count = tp_l[lab] + fp_l[lab]
        rows.append(
            {
                "model": model_name,
                "label": lab,
                "precision": p,
                "recall": r,
                "f1": f1,
                "gold_support": support,
                "pred_count": pred_count,
                "tp": tp_l[lab],
                "fp": fp_l[lab],
                "fn": fn_l[lab],
            }
        )
    per_label_df = pd.DataFrame(rows)

    # Micro
    tp_micro = sum(tp_l.values())
    fp_micro = sum(fp_l.values())
    fn_micro = sum(fn_l.values())
    p_micro, r_micro, f1_micro = prf(tp_micro, fp_micro, fn_micro)

    # Macro over labels (unweighted)
    macro_p = per_label_df["precision"].mean()
    macro_r = per_label_df["recall"].mean()
    macro_f1 = per_label_df["f1"].mean()

    # Nested stats
    nested_rate_gold = (gold_nested_total / gold_total) if gold_total else 0.0
    nested_det_rate = (nested_tp / (nested_tp + nested_fn)) if (nested_tp + nested_fn) else 0.0
    pred_overlap_avg = sum(pred_overlap_rates) / len(pred_overlap_rates) if pred_overlap_rates else 0.0

    summary_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "family": guess_family(model_name),
                "micro_precision": p_micro,
                "micro_recall": r_micro,
                "micro_f1": f1_micro,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "macro_f1": macro_f1,
                "gold_entities_total": gold_total,
                "gold_nested_entity_rate": nested_rate_gold,
                "nested_exactmatch_recall": nested_det_rate,
                "pred_overlap_rate_avg": pred_overlap_avg,
                "notes": (
                    "pred_overlap_rate_avg≈0 => model outputs flat entities (no nesting). "
                    "WiNER includes nesting; most NER tools ignore it."
                ),
            }
        ]
    )

    return summary_df, per_label_df


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True, help="Path to WiNER-fr root folder")
    ap.add_argument("--pred_dir", type=str, required=True, help="Directory containing *.jsonl predictions")
    ap.add_argument("--out_dir", type=str, default="results_eval", help="Output folder for CSV reports")
    args = ap.parse_args()

    gold = build_gold_index(args.winer_root)

    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(pred_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found in {pred_dir}")

    all_summary = []
    all_per_label = []

    for f in jsonl_files:
        pred_by_doc = read_predictions_jsonl(f)

        # model name: use filename stem as stable id
        model_name = f.stem

        # Special: if your rules file name doesn't start with RULES, set it
        if "rules" in model_name.lower():
            model_name = "RULES_DATE_HOUR"
        if model_name.lower().startswith("spacy_"):
            model_name = "spacy_" + model_name[len("spacy_"):]
        if model_name.lower().startswith("hf_"):
            model_name = "hf_" + model_name[len("hf_"):]

        summary_df, per_label_df = evaluate_one(model_name, gold, pred_by_doc)
        all_summary.append(summary_df)
        all_per_label.append(per_label_df)

    summary = pd.concat(all_summary, ignore_index=True)
    per_label = pd.concat(all_per_label, ignore_index=True)

    # Extra “best model per label” table (by F1)
    best_per_label = (
        per_label.sort_values(["label", "f1"], ascending=[True, False])
        .groupby("label", as_index=False)
        .head(1)[["label", "model", "f1", "precision", "recall", "gold_support"]]
        .rename(columns={"model": "best_model"})
    )

    # Save
    summary_path = out_dir / "summary_models.csv"
    per_label_path = out_dir / "per_label_metrics.csv"
    best_label_path = out_dir / "best_model_per_label.csv"

    summary.to_csv(summary_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    best_per_label.to_csv(best_label_path, index=False)

    # Console view (nice for quick analysis)
    print("\n=== MODEL SUMMARY (sorted by micro_f1) ===")
    print(summary.sort_values("micro_f1", ascending=False)[
        ["model", "family", "micro_f1", "macro_f1", "gold_nested_entity_rate", "nested_exactmatch_recall", "pred_overlap_rate_avg"]
    ].to_string(index=False))

    print("\n=== BEST MODEL PER ENTITY TYPE (by F1) ===")
    print(best_per_label.sort_values("label").to_string(index=False))

    print(f"\nSaved:\n- {summary_path}\n- {per_label_path}\n- {best_label_path}")


if __name__ == "__main__":
    main()
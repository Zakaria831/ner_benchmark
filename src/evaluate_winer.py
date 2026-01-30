from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set, Optional, Any

import pandas as pd

Span = Tuple[int, int]
Entity = Tuple[int, int, str]

WINER_LABELS = ["Person", "Location", "Organization", "Date", "Event", "Product", "Hour"]



# Label normalization
def _clean_label(raw_label: Any) -> str:
    lab = str(raw_label).strip()
    # strip BIO if present
    if len(lab) > 2 and lab[1] == "-" and lab[0] in ("B", "I"):
        lab = lab.split("-", 1)[1].strip()
    return lab


def normalize_label(model_name: str, raw_label: Any) -> Optional[str]:
    """
    Map system-specific labels to WiNER labels.
    Return None for unsupported labels.
    """
    if raw_label is None:
        return None

    lab = _clean_label(raw_label)
    if not lab:
        return None

    # already WiNER
    if lab in WINER_LABELS:
        return lab

    m = (model_name or "").lower()
    up = lab.upper()

    # Rules baseline
    if "rules" in m:
        if up == "DATE":
            return "Date"
        if up in ("HOUR", "TIME"):
            return "Hour"
        return None

    # spaCy (FR often PER/LOC/ORG, sometimes PERSON/GPE)
    if "spacy" in m or m.startswith("fr_"):
        if up in ("PER", "PERSON"):
            return "Person"
        if up in ("LOC", "GPE", "LOCATION"):
            return "Location"
        if up in ("ORG", "ORGANIZATION"):
            return "Organization"
        if up == "DATE":
            return "Date"
        if up in ("TIME", "HOUR"):
            return "Hour"
        return None

    # HF / Flair common
    if up in ("PER", "PERSON"):
        return "Person"
    if up in ("LOC", "LOCATION"):
        return "Location"
    if up in ("ORG", "ORGANIZATION"):
        return "Organization"
    if up == "DATE":
        return "Date"
    if up in ("TIME", "HOUR"):
        return "Hour"

    # MISC can't be mapped to Event/Product reliably
    if up == "MISC":
        return None

    return None


def guess_family(model_field: str) -> str:
    m = (model_field or "").lower()
    if m.startswith("flair/") or m.startswith("flair_") or "flair" in m:
        return "Flair"
    if "camembert" in m or "bert" in m or "distil" in m or m.startswith("hf_"):
        return "HuggingFace"
    if "spacy" in m or m.startswith("fr_"):
        return "spaCy"
    if "heideltime" in m:
        return "HeidelTime"
    if "rules" in m:
        return "Rules"
    return "Other"



# IO: Predictions
def read_predictions_jsonl(path: Path) -> Dict[str, List[Entity]]:
    """
    Reads JSONL predictions:
      {"doc_id":"...", "entities":[[s,e,label], ...], ...}
    Also tolerates entities as dict: {"start":..,"end":..,"label":..}
    """
    out: Dict[str, List[Entity]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["doc_id"]

            ents: List[Entity] = []
            for e in obj.get("entities", []):
                if isinstance(e, dict):
                    s = int(e.get("start", -1))
                    t = int(e.get("end", -1))
                    lab = e.get("label", "")
                elif isinstance(e, (list, tuple)) and len(e) == 3:
                    s, t, lab = int(e[0]), int(e[1]), e[2]
                else:
                    continue

                if t > s and s >= 0:
                    ents.append((s, t, str(lab)))
            out[doc_id] = ents
    return out


# Gold loader with RELATIVE doc_id
# (must match make_splits_winer.py doc_id format)
def load_winer_gold_rel(winer_root: str) -> Dict[str, List[Entity]]:
    """
    Load WiNER BRAT gold but use doc_id = relative path without extension:
      2016/01/article123
    This avoids doc_id collisions and matches your new splits.
    """
    root = Path(winer_root).resolve()
    gold: Dict[str, List[Entity]] = {}

    for txt_path in root.rglob("*.txt"):
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            continue

        rel_noext = txt_path.relative_to(root).with_suffix("")
        doc_id = str(rel_noext).replace("\\", "/")

        # parse ann
        entities: List[Entity] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[1].strip()
            if not meta:
                continue
            label = meta.split()[0].strip()

            rest = meta[len(label):].strip()
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
                    entities.append((s, e, label))

        gold[doc_id] = entities

    return gold



# Nesting helpers
def is_nested(a: Span, b: Span) -> bool:
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



# Metrics
def as_set(ents: Iterable[Entity], model_name: str) -> Set[Entity]:
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


def _normalize_model_name_from_file(stem: str) -> str:
    s = stem.strip()
    low = s.lower()
    if "rules" in low:
        return "RULES_DATE_HOUR"
    return s


def evaluate_one(
    model_name: str,
    gold_by_doc: Dict[str, List[Entity]],
    pred_by_doc: Dict[str, List[Entity]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tp_l = {lab: 0 for lab in WINER_LABELS}
    fp_l = {lab: 0 for lab in WINER_LABELS}
    fn_l = {lab: 0 for lab in WINER_LABELS}

    gold_nested_total = 0
    gold_total = 0
    nested_tp = 0
    nested_fn = 0

    pred_overlap_rates: List[float] = []

    all_doc_ids = set(gold_by_doc.keys()) | set(pred_by_doc.keys())

    for doc_id in all_doc_ids:
        gold_raw = gold_by_doc.get(doc_id, [])
        pred_raw = pred_by_doc.get(doc_id, [])

        gold_set = as_set(gold_raw, "WINER_GOLD")  # gold already WiNER-like
        pred_set = as_set(pred_raw, model_name)

        gold_nested_sp = nested_gold_spans(list(gold_set))
        gold_total += len(gold_set)
        gold_nested_total += len(gold_nested_sp)

        inter = gold_set & pred_set
        only_pred = pred_set - gold_set
        only_gold = gold_set - pred_set

        for (_, _, lab) in inter:
            tp_l[lab] += 1
        for (_, _, lab) in only_pred:
            fp_l[lab] += 1
        for (_, _, lab) in only_gold:
            fn_l[lab] += 1

        if gold_nested_sp:
            for (s, e, lab) in gold_set:
                if (s, e) in gold_nested_sp:
                    if (s, e, lab) in pred_set:
                        nested_tp += 1
                    else:
                        nested_fn += 1

        pred_overlap_rates.append(overlap_rate(list(pred_set)))

    per_label_rows = []
    for lab in WINER_LABELS:
        p, r, f1 = prf(tp_l[lab], fp_l[lab], fn_l[lab])
        support = tp_l[lab] + fn_l[lab]
        pred_count = tp_l[lab] + fp_l[lab]
        per_label_rows.append(
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
    per_label_df = pd.DataFrame(per_label_rows)

    tp_micro = sum(tp_l.values())
    fp_micro = sum(fp_l.values())
    fn_micro = sum(fn_l.values())
    p_micro, r_micro, f1_micro = prf(tp_micro, fp_micro, fn_micro)

    macro_p = per_label_df["precision"].mean()
    macro_r = per_label_df["recall"].mean()
    macro_f1 = per_label_df["f1"].mean()

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
                    "nested_exactmatch_recall is strict (span+label exact) on nested gold entities. "
                    "Most NER tools output flat entities."
                ),
            }
        ]
    )

    return summary_df, per_label_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--winer_root", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results_eval")
    args = ap.parse_args()

    gold = load_winer_gold_rel(args.winer_root)

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
        model_name = _normalize_model_name_from_file(f.stem)

        summary_df, per_label_df = evaluate_one(model_name, gold, pred_by_doc)
        all_summary.append(summary_df)
        all_per_label.append(per_label_df)

    summary = pd.concat(all_summary, ignore_index=True)
    per_label = pd.concat(all_per_label, ignore_index=True)

    best_per_label = (
        per_label.sort_values(["label", "f1"], ascending=[True, False])
        .groupby("label", as_index=False)
        .head(1)[["label", "model", "f1", "precision", "recall", "gold_support"]]
        .rename(columns={"model": "best_model"})
    )

    summary_path = out_dir / "summary_models.csv"
    per_label_path = out_dir / "per_label_metrics.csv"
    best_label_path = out_dir / "best_model_per_label.csv"

    summary.to_csv(summary_path, index=False)
    per_label.to_csv(per_label_path, index=False)
    best_per_label.to_csv(best_label_path, index=False)

    print("\n=== MODEL SUMMARY (sorted by micro_f1) ===")
    print(
        summary.sort_values("micro_f1", ascending=False)[
            ["model", "family", "micro_f1", "macro_f1", "gold_nested_entity_rate", "nested_exactmatch_recall", "pred_overlap_rate_avg"]
        ].to_string(index=False)
    )

    print("\n=== BEST MODEL PER ENTITY TYPE (by F1) ===")
    print(best_per_label.sort_values("label").to_string(index=False))

    print(f"\nSaved:\n- {summary_path}\n- {per_label_path}\n- {best_label_path}")


if __name__ == "__main__":
    main()
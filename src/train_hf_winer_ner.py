# train_hf_winer_ner.py

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def _filter_kwargs_for_callable(fn, kwargs: dict) -> dict:
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def compute_class_weights(train_labels: List[List[int]], num_labels: int) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.int64)
    for seq in train_labels:
        for x in seq:
            x = int(x)
            if x == -100:
                continue
            counts[x] += 1

    total = counts.sum()
    weights = np.zeros(num_labels, dtype=np.float32)
    for i in range(num_labels):
        if counts[i] > 0:
            weights[i] = total / (num_labels * counts[i])
        else:
            weights[i] = 0.0

    nonzero = weights[weights > 0]
    if nonzero.size > 0:
        weights = weights / float(nonzero.mean())

    return torch.tensor(weights, dtype=torch.float32)


class WeightedTokenTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            w = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=w, ignore_index=-100)

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics(id2label: Dict[int, str], wb):
    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=-1)
        labels = p.label_ids

        true_labels: List[List[str]] = []
        true_preds: List[List[str]] = []

        for pred_seq, lab_seq in zip(preds, labels):
            seq_true = []
            seq_pred = []
            for pred_id, lab_id in zip(pred_seq, lab_seq):
                lab_id = int(lab_id)
                if lab_id == -100:
                    continue
                seq_true.append(id2label[lab_id])
                seq_pred.append(id2label[int(pred_id)])
            true_labels.append(seq_true)
            true_preds.append(seq_pred)

        prec = precision_score(true_labels, true_preds)
        rec = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)

        if wb is not None:
            report = classification_report(true_labels, true_preds, digits=4)
            wb.log({"seqeval_report": wb.Html(f"<pre>{report}</pre>")})

        return {"precision": prec, "recall": rec, "f1": f1}

    return compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True, help=".../dataset (save_to_disk)")
    ap.add_argument("--label2id", type=str, required=True)
    ap.add_argument("--id2label", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_batch", type=int, default=2)
    ap.add_argument("--eval_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--logging_steps", type=int, default=50)

    ap.add_argument("--best_metric", type=str, default="eval_loss")
    ap.add_argument("--greater_is_better", action="store_true")

    ap.add_argument("--use_class_weights", action="store_true")

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="ner-benchmark-winer")
    ap.add_argument("--wandb_run_name", type=str, default="run")

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    label2id_path = Path(args.label2id)
    id2label_path = Path(args.id2label)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")
    if not label2id_path.exists():
        raise SystemExit(f"label2id not found: {label2id_path}")
    if not id2label_path.exists():
        raise SystemExit(f"id2label not found: {id2label_path}")

    wb = None
    if args.use_wandb:
        try:
            import wandb as _wandb
            wb = _wandb
            os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
            wb.init(project=args.wandb_project, name=args.wandb_run_name)
        except Exception as e:
            print(f"[WARN] wandb disabled (import/init failed): {e}")
            wb = None

    ds = load_from_disk(str(dataset_dir))
    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    id2label_raw = json.loads(id2label_path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in id2label_raw.items()}
    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(ds["train"]["labels"], num_labels=num_labels)
        if wb is not None:
            nz = class_weights[class_weights > 0]
            wb.log({"class_weights_mean": float(nz.mean().cpu().numpy()) if nz.numel() else 0.0})

    ta_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.best_metric,
        greater_is_better=args.greater_is_better,
        fp16=bool(args.fp16),
        seed=args.seed,
        report_to=(["wandb"] if wb is not None else []),
    )
    ta_kwargs = _filter_kwargs_for_callable(TrainingArguments.__init__, ta_kwargs)
    training_args = TrainingArguments(**ta_kwargs)

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(id2label, wb),
        class_weights=class_weights,
    )

    trainer.train()

    test_metrics = trainer.evaluate(ds["test"])
    print("\n=== TEST METRICS ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    best_dir = output_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    if wb is not None:
        wb.log({f"test/{k}": v for k, v in test_metrics.items()})
        wb.finish()

    print(f"\nSaved best model to: {best_dir}")
    print(f"Saved test metrics to: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
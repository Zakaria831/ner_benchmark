#!/bin/bash
#SBATCH --job-name=hf_distilcamembert_winer
#SBATCH --output=logs/hf_distilcamembert_winer_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue07
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00

set -e

echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p logs results results/predictions data_hf models

eval "$(/info/etu/m2/s2506967/anaconda3/bin/conda shell.bash hook)"
conda activate covea_env
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python -V

PROJECT_ROOT="/info/etu/m2/s2506967/Covea_project/ner_benchmark"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
cd "$PROJECT_ROOT"

nvidia-smi || true

# ---------------------------
# CONFIG DISTILCAMEMBERT NER
# ---------------------------
MODEL_NAME="cmarkea/distilcamembert-base"
RUN_TAG="distilcamembert_priority"

DATA_DIR="$PROJECT_ROOT/data_hf/winer_${RUN_TAG}"
OUT_MODEL_DIR="$PROJECT_ROOT/models/${RUN_TAG}"

# 1) Build dataset (tokenization depends on model tokenizer)
python "$PROJECT_ROOT/src/build_hf_dataset_winer.py" \
  --splits_json "$PROJECT_ROOT/splits/winer_splits.json" \
  --model_name "$MODEL_NAME" \
  --out_dir "$DATA_DIR" \
  --max_length 512 \
  --stride 128

# 2) Train (faster / bigger batch possible)
python "$PROJECT_ROOT/src/train_hf_winer_ner.py" \
  --base_model "$MODEL_NAME" \
  --dataset_dir "$DATA_DIR/dataset" \
  --label2id "$DATA_DIR/label2id.json" \
  --id2label "$DATA_DIR/id2label.json" \
  --output_dir "$OUT_MODEL_DIR" \
  --epochs 20 \
  --lr 3e-5 \
  --train_batch 8 \
  --eval_batch 8 \
  --grad_accum 2 \
  --weight_decay 0.01 \
  --fp16 \
  --seed 13 \
  --save_total_limit 5 \
  --logging_steps 50 \
  --best_metric eval_loss \
  --use_class_weights \
  --use_wandb \
  --wandb_project ner-benchmark-winer \
  --wandb_run_name distilcamembert_priority

# 3) Predict on TEST from BRAT
python "$PROJECT_ROOT/src/run_hf_trained.py" \
  --dataset_dir "$DATA_DIR/dataset" \
  --winer_root "$PROJECT_ROOT/data/WiNER-fr" \
  --model_dir "$OUT_MODEL_DIR/best_model" \
  --out_jsonl "$PROJECT_ROOT/results/predictions/hf_${RUN_TAG}_ft_TEST.jsonl" \
  --max_length 512 \
  --stride 128 \
  --model_name_in_jsonl "${RUN_TAG}_ft"

echo "=== Job finished at $(date) ==="
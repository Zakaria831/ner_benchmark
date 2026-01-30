#!/bin/bash
#SBATCH --job-name=hf_distilcamembert_ner_winer
#SBATCH --output=logs/hf_distilcamembert_ner_winer_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue01
#SBATCH --mem=20G
#SBATCH --cpus-per-task=10
#SBATCH --time=08:00:00

set -e

echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p logs results


eval "$(/info/etu/m2/s2506967/anaconda3/bin/conda shell.bash hook)"

conda activate covea_env
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python -V


PROJECT_ROOT="/info/etu/m2/s2506967/Covea_project/ner_benchmark" 
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

cd "$PROJECT_ROOT"
nvidia-smi || true

python src/run_span_trained_winer_test.py --winer_root data/WiNER-fr --model_dir ./models/spanmarker_camembert_winer --out results/predictions/spanmarker_camembert_winer.jsonl

echo "=== Job finished at $(date) ==="
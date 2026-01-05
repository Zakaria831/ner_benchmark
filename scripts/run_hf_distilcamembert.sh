#!/bin/bash
#SBATCH --job-name=hf_distilcamembert_ner_winer
#SBATCH --output=logs/hf_distilcamembert_ner_winer_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue05
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00

set -e

echo "=== Job started on $(hostname) at $(date) ==="
mkdir -p logs results


eval "$(/info/etu/m2/s2506967/anaconda3/bin/conda shell.bash hook)"

conda activate covea_env
echo "Conda env: $CONDA_DEFAULT_ENV"
which python
python -V


PROJECT_ROOT="/info/etu/m2/s2506967/Covea_project/NER-Project" 
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"


nvidia-smi || true

python "$PROJECT_ROOT/src/run_hf.py" \
  --winer_root "$PROJECT_ROOT/data/WiNER-fr" \
  --model cmarkea/distilcamembert-base-ner \
  --max_length 512 \
  --stride 128

echo "=== Job finished at $(date) ==="
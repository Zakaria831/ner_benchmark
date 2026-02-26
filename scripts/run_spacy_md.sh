#!/bin/bash
#SBATCH --job-name=spacy_md_winer
#SBATCH --output=logs/spacy_md_winer_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpue02
#SBATCH --mem=10G
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


nvidia-smi || true

python "$PROJECT_ROOT/src/run_spacy.py" \
  --winer_root "$PROJECT_ROOT/data/WiNER-fr" \
  --model fr_core_news_md \
  --use_gpu \
  --batch_size 8 \
  --n_process 1 \
  --ids_file splits/test_ids.txt

echo "=== Job finished at $(date) ==="
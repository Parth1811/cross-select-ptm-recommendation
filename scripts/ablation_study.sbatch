#!/bin/bash -l
# FILENAME: ablation_study.sbatch
# Architecture ablation: 6 variants sharing identical batches

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --partition=smallgpu
#SBATCH --mem=192G
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --job-name=ablation_study
#SBATCH --output=/home/patil185/joboutput/ablation_study.%j.out
#SBATCH --error=/home/patil185/joboutput/ablation_study.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

export HF_HOME="/scratch/gautschi/patil185/hub"
export TORCH_HOME="/scratch/gautschi/patil185/torch_cache"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/artifacts"

export PTM_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_model_embeddings"
export PTM_RAW_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_models"
export PTM_DATASET_ROOT="/home/patil185/ptm-recommendation-with-transformers/artifacts/extracted/datasets"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

mkdir -p /home/patil185/joboutput artifacts

echo "Node: $(hostname)"
echo "Job start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

pip install --user -e ".[dev]" 2>&1 | tail -3

echo "Running ablation study (7 variants)..."
python -m cross_select.cli.multi_train experiment=ablation_study

echo ""
echo "Job end: $(date)"

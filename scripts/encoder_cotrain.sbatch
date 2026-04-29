#!/bin/bash -l
# FILENAME: encoder_cotrain.sbatch
# Encoder co-training: raw 8192-dim → trainable encoder → scorer

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --partition=smallgpu
#SBATCH --mem=192G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=encoder_cotrain
#SBATCH --output=/home/patil185/joboutput/encoder_cotrain.%j.out
#SBATCH --error=/home/patil185/joboutput/encoder_cotrain.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

export HF_HOME="/scratch/gautschi/patil185/hub"
export TORCH_HOME="/scratch/gautschi/patil185/torch_cache"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/artifacts"

OLD_REPO="/home/patil185/ptm-recommendation-with-transformers"
export PTM_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_model_embeddings_new"
export PTM_RAW_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_models"
export PTM_DATASET_ROOT="${OLD_REPO}/artifacts/extracted/datasets"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

mkdir -p /home/patil185/joboutput artifacts

echo "Node: $(hostname)"
echo "Job start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

pip install --user -e ".[dev]" 2>&1 | tail -3

python -m cross_select.cli.train experiment=encoder_cotrain

echo "Job end: $(date)"

#!/bin/bash -l
# FILENAME: parc_benchmark.sbatch
# PARC Benchmark Replication: Cross-Select vs Model Spider

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --partition=smallgpu
#SBATCH --mem=16G
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --job-name=parc_benchmark
#SBATCH --output=/home/patil185/joboutput/parc_benchmark.%j.out
#SBATCH --error=/home/patil185/joboutput/parc_benchmark.%j.err

set -euo pipefail

module load cuda

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Cache dirs on scratch (fast local storage)
export HF_HOME="/scratch/gautschi/patil185/hub"
export TORCH_HOME="/scratch/gautschi/patil185/torch_cache"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/artifacts"

# Point data config to the old repo's extracted artifacts
export PTM_MODEL_TOKENS_DIR="/home/patil185/ptm-recommendation-with-transformers/artifacts/extracted/parc_model_embeddings"
export PTM_DATASET_ROOT="/home/patil185/ptm-recommendation-with-transformers/artifacts/extracted/datasets"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

# WandB API key
export WANDB_API_KEY="${WANDB_API_KEY:-}"

mkdir -p /home/patil185/joboutput artifacts

echo "Node: $(hostname)"
echo "Job start: $(date)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Install package in editable mode
pip install --user -e ".[dev]" 2>&1 | tail -3

echo "Running PARC benchmark replication..."
python -m cross_select.cli.multi_train experiment=parc_benchmark

echo ""
echo "Job end: $(date)"

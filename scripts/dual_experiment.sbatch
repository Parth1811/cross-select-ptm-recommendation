#!/bin/bash -l
# FILENAME: dual_experiment.sbatch
# Two experiments:
# (1) Encoder co-train: raw 8192-dim → trainable encoder → scorer
# (2) Embedding pipeline: pre-computed 512-dim → standard scorer

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --partition=smallgpu
#SBATCH --mem=192G
#SBATCH --qos=normal
#SBATCH --time=08:00:00
#SBATCH --job-name=dual_experiment
#SBATCH --output=/home/patil185/joboutput/dual_experiment.%j.out
#SBATCH --error=/home/patil185/joboutput/dual_experiment.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

export HF_HOME="/scratch/gautschi/patil185/hub"
export TORCH_HOME="/scratch/gautschi/patil185/torch_cache"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/artifacts"

OLD_REPO="/home/patil185/ptm-recommendation-with-transformers"

# Embeddings: symlink → parc_model_embeddings_new (weight-space, 32 unique)
export PTM_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_model_embeddings"
export PTM_RAW_MODEL_TOKENS_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_models"
export PTM_DATASET_ROOT="${OLD_REPO}/artifacts/extracted/datasets"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

mkdir -p /home/patil185/joboutput artifacts

echo "Node: $(hostname)"
echo "Job start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

pip install --user -e ".[dev]" 2>&1 | tail -3

echo ""
echo "=== Data verification ==="
echo "Embeddings: $(ls ${PTM_MODEL_TOKENS_DIR}/*_embedding.npz 2>/dev/null | wc -l) files"
echo "Raw vectors: $(ls ${PTM_RAW_MODEL_TOKENS_DIR}/*.npz 2>/dev/null | wc -l) files"
echo "Dataset shards: $(find ${PTM_DATASET_ROOT} -name '*.npz' 2>/dev/null | wc -l) files"

# ── Experiment 1: Encoder co-training ──
echo ""
echo "=========================================="
echo "EXPERIMENT 1: Encoder Co-Training (raw 8192-dim)"
echo "=========================================="
python -m cross_select.cli.train experiment=encoder_cotrain || {
    echo "ERROR: Experiment 1 failed with exit code $?"
}

# ── Experiment 2: Embedding pipeline (weight-space 512-dim) ──
echo ""
echo "=========================================="
echo "EXPERIMENT 2: Embedding Pipeline (512-dim)"
echo "=========================================="
python -m cross_select.cli.train experiment=embedding_pipeline || {
    echo "ERROR: Experiment 2 failed with exit code $?"
}

echo ""
echo "Job end: $(date)"

#!/bin/bash -l
# FILENAME: dual_experiment.sbatch
# Full pipeline: extract → embed → run two experiments
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
PARC_MODELS_DIR="${OLD_REPO}/parc/models"
AE_CHECKPOINT="${OLD_REPO}/artifacts/models/model_autoencoder/ModelAutoEncoder_best_20251124_113402.pt"

# Output dirs for extracted data
RAW_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_models"
EMB_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_model_embeddings"
DATASET_ROOT="${OLD_REPO}/artifacts/extracted/datasets"

export PTM_MODEL_TOKENS_DIR="${EMB_DIR}"
export PTM_RAW_MODEL_TOKENS_DIR="${RAW_DIR}"
export PTM_DATASET_ROOT="${DATASET_ROOT}"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

mkdir -p /home/patil185/joboutput artifacts "${RAW_DIR}" "${EMB_DIR}"

echo "Node: $(hostname)"
echo "Job start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

pip install --user -e ".[dev]" 2>&1 | tail -3

# ── Step 1: Extract raw 8192-dim vectors ──
RAW_COUNT=$(ls "${RAW_DIR}"/*.npz 2>/dev/null | wc -l)
if [ "${RAW_COUNT}" -lt 28 ]; then
    echo ""
    echo "=========================================="
    echo "STEP 1: Extracting raw 8192-dim vectors"
    echo "=========================================="
    python scripts/extract_parc_raw_vectors.py \
        --parc-models-dir "${PARC_MODELS_DIR}" \
        --output-dir "${RAW_DIR}" \
        --skip-existing
else
    echo "STEP 1: SKIP — ${RAW_COUNT} raw vectors already exist"
fi

# ── Step 2: Generate 512-dim embeddings via autoencoder ──
EMB_COUNT=$(ls "${EMB_DIR}"/*_embedding.npz 2>/dev/null | wc -l)
if [ "${EMB_COUNT}" -lt 28 ]; then
    echo ""
    echo "=========================================="
    echo "STEP 2: Generating 512-dim embeddings"
    echo "=========================================="
    python scripts/extract_model_embeddings.py \
        --parc-models-dir "${PARC_MODELS_DIR}" \
        --autoencoder-checkpoint "${AE_CHECKPOINT}" \
        --output-dir "${EMB_DIR}" \
        --skip-existing
else
    echo "STEP 2: SKIP — ${EMB_COUNT} embeddings already exist"
fi

# Verify
echo ""
echo "=== Data verification ==="
echo "Raw vectors: $(ls ${RAW_DIR}/*.npz 2>/dev/null | wc -l) files"
echo "Embeddings: $(ls ${EMB_DIR}/*_embedding.npz 2>/dev/null | wc -l) files"
echo "Dataset shards: $(find ${DATASET_ROOT} -name '*.npz' 2>/dev/null | wc -l) files"

# ── Experiment 1: Encoder co-training ──
echo ""
echo "=========================================="
echo "EXPERIMENT 1: Encoder Co-Training (raw 8192-dim)"
echo "=========================================="
python -m cross_select.cli.train experiment=encoder_cotrain || {
    echo "ERROR: Experiment 1 failed with exit code $?"
}

# ── Experiment 2: Old pipeline with embeddings ──
echo ""
echo "=========================================="
echo "EXPERIMENT 2: Embedding Pipeline (512-dim)"
echo "=========================================="
python -m cross_select.cli.train experiment=embedding_pipeline || {
    echo "ERROR: Experiment 2 failed with exit code $?"
}

echo ""
echo "Job end: $(date)"

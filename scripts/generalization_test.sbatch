#!/bin/bash -l
# FILENAME: generalization_test.sbatch
# Q1-Q4 generalization test with weight-space embeddings.
# Hold out: resnet50 (8 models), cub200 + stanford_dogs (2 datasets).
# Runs Cross-Select + Spider on each quadrant via multi_train.

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --gpus-per-node=1
#SBATCH --partition=smallgpu
#SBATCH --mem=192G
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=generalization
#SBATCH --output=/home/patil185/joboutput/generalization.%j.out
#SBATCH --error=/home/patil185/joboutput/generalization.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

export HF_HOME="/scratch/gautschi/patil185/hub"
export TORCH_HOME="/scratch/gautschi/patil185/torch_cache"
export WANDB_DIR="${SLURM_SUBMIT_DIR}/artifacts"

OLD_REPO="/home/patil185/ptm-recommendation-with-transformers"
EMB_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_model_embeddings_new"
DATASET_ROOT="${OLD_REPO}/artifacts/extracted/datasets"

export PTM_MODEL_TOKENS_DIR="${EMB_DIR}"
export PTM_DATASET_ROOT="${DATASET_ROOT}"
export PTM_GT_PATH="${SLURM_SUBMIT_DIR}/constants/ground_truth_rankings.json"

mkdir -p /home/patil185/joboutput artifacts

echo "Node: $(hostname)"
echo "Job start: $(date)"

pip install --user -e ".[dev]" 2>&1 | tail -3

HELD_MODELS="[resnet50_caltech101,resnet50_cifar10,resnet50_cub200,resnet50_imagenet,resnet50_nabird,resnet50_oxford_pets,resnet50_stanford_dogs,resnet50_voc2007]"
HELD_DATASETS="[cub200,stanford_dogs]"
COMMON="model.num_layers=4 trainer.epochs=100"

# ── Q1: Known models × Known datasets ──
echo ""
echo "=========================================="
echo "Q1: Known Models × Known Datasets"
echo "=========================================="
python -m cross_select.cli.multi_train experiment=parc_benchmark \
  quadrant.id=1 quadrant.name=q1_known_known \
  "quadrant.held_out_models=${HELD_MODELS}" \
  "quadrant.held_out_datasets=${HELD_DATASETS}" \
  wandb.name=gen-q1-known-known \
  ${COMMON} || echo "ERROR: Q1 failed ($?)"

# ── Q2: Unknown models × Known datasets ──
echo ""
echo "=========================================="
echo "Q2: Unknown Models × Known Datasets"
echo "=========================================="
python -m cross_select.cli.multi_train experiment=parc_benchmark \
  quadrant.id=2 quadrant.name=q2_unknown_models \
  "quadrant.held_out_models=${HELD_MODELS}" \
  quadrant.held_out_datasets=[] \
  wandb.name=gen-q2-unknown-models \
  ${COMMON} || echo "ERROR: Q2 failed ($?)"

# ── Q3: Known models × Unknown datasets ──
echo ""
echo "=========================================="
echo "Q3: Known Models × Unknown Datasets"
echo "=========================================="
python -m cross_select.cli.multi_train experiment=parc_benchmark \
  quadrant.id=3 quadrant.name=q3_unknown_datasets \
  quadrant.held_out_models=[] \
  "quadrant.held_out_datasets=${HELD_DATASETS}" \
  wandb.name=gen-q3-unknown-datasets \
  ${COMMON} || echo "ERROR: Q3 failed ($?)"

# ── Q4: Unknown models × Unknown datasets ──
echo ""
echo "=========================================="
echo "Q4: Unknown Models × Unknown Datasets"
echo "=========================================="
python -m cross_select.cli.multi_train experiment=parc_benchmark \
  quadrant.id=4 quadrant.name=q4_unknown_unknown \
  "quadrant.held_out_models=${HELD_MODELS}" \
  "quadrant.held_out_datasets=${HELD_DATASETS}" \
  wandb.name=gen-q4-unknown-unknown \
  ${COMMON} || echo "ERROR: Q4 failed ($?)"

echo ""
echo "Job end: $(date)"

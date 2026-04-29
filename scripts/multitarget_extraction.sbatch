#!/bin/bash -l
# FILENAME: multitarget_extraction.sbatch
# Extract multi-target PARC embeddings (Strategy C fix)

#SBATCH --account=davisjam
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --partition=shared
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --job-name=parc_multitarget
#SBATCH --output=/home/patil185/joboutput/parc_multitarget.%j.out
#SBATCH --error=/home/patil185/joboutput/parc_multitarget.%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

PROBES_DIR="/home/patil185/ptm-recommendation-with-transformers/parc/cache/probes/fixed_budget_500"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_multitarget"

mkdir -p /home/patil185/joboutput "${OUTPUT_DIR}"

echo "Node: $(hostname)"
echo "Job start: $(date)"

pip install --user -e ".[dev]" 2>&1 | tail -3

# Extract with mean aggregation (same dim as existing 512-dim embeddings)
echo ""
echo "=== Multi-target extraction (mean, 512-dim) ==="
python scripts/extract_parc_multitarget.py \
    --probes-dir "${PROBES_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --aggregation mean \
    --dim 512

# Also extract concat version for comparison
CONCAT_DIR="${SLURM_SUBMIT_DIR}/artifacts/extracted/parc_multitarget_concat"
mkdir -p "${CONCAT_DIR}"

echo ""
echo "=== Multi-target extraction (concat, 3584-dim) ==="
python scripts/extract_parc_multitarget.py \
    --probes-dir "${PROBES_DIR}" \
    --output-dir "${CONCAT_DIR}" \
    --aggregation concat \
    --dim 512

# Verify uniqueness
echo ""
echo "=== Uniqueness check ==="
python -c "
import numpy as np
from pathlib import Path
for d in ['${OUTPUT_DIR}', '${CONCAT_DIR}']:
    vecs = {}
    for f in sorted(Path(d).glob('*_embedding.npz')):
        vecs[f.stem] = np.load(f)['embedding']
    unique = len(set(v.tobytes() for v in vecs.values()))
    print(f'{Path(d).name}: {unique}/{len(vecs)} unique embeddings, dim={next(iter(vecs.values())).shape}')
"

echo ""
echo "Job end: $(date)"

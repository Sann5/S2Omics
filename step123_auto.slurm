#!/bin/bash
#SBATCH --job-name=s2omics_auto_step13_array
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=1024GB
#SBATCH --time=24:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=goerkemkadir.solun@dqbm.uzh.ch
#SBATCH --partition=lowprio

set -euo pipefail

ROOT_DIR="/scratch/gsolun"
PROJECT_DIR="${ROOT_DIR}/S2Omics"
WORK_DIR="${PROJECT_DIR}/outputs_1"
NDPI_DIR="${ROOT_DIR}/Data/NSCLC1/HES1"
# Samples that were NOT manually preprocessed -> full-slide auto conversion.
# Build this list ONCE on the login node BEFORE sbatch (excludes the manual set):
#   comm -23 <(ls -1 ${ROOT_DIR}/Data/NSCLC1/HES1/*.ndpi | LC_ALL=C sort) \
#            <(LC_ALL=C sort ${WORK_DIR}/manual_preprocessed_inputs.txt) \
#            > ${WORK_DIR}/auto_inputs.txt
INPUT_LIST="${WORK_DIR}/auto_inputs.txt"

# Model settings
FOUNDATION_MODEL="uni"
CKPT_PATH="${PROJECT_DIR}/checkpoints/uni/"
DEVICE="cuda:0"
DOWN_SAMP_STEP=5
BATCH_SIZE=16
NUM_WORKERS=2
TARGET_LEVEL=0

# === PIPELINE STEPS TO EXECUTE ===
# run_batch.py step mapping:
#   1 = NDPI conversion + histology preprocess (p0 + p1)
#   2 = superpixel QC / masking (p2)
#   3 = feature extraction (p3)
#   4 = joint segmentation (p4)
#
# These samples have NO manual crops, so we START AT STEP 1 to auto-convert the
# full slide and preprocess, then mask (step 2) and extract features (step 3).
# Outputs go to the SAME work-dir as the manual run so the later joint step 4
# clusters manual + non-manual samples together. Keep DOWN_SAMP_STEP, masking,
# and work-dir identical to step123.slurm for comparable results.
START_STEP=1
END_STEP=3

# === STEP 2 MASKING === (must match step123.slurm for comparability)
MASKING_METHOD="victor"
VICTOR_MEAN_THRESHOLD=0.85
VICTOR_MAX_ITERATIONS=5
VICTOR_SIGMA=20
VICTOR_SUPERPIXEL_THRESHOLD=0.5
# Standard H&E = dark tissue on bright background, so positive-contrast stays
# OFF. To enable it, add --victor-positive-contrast to the python command below.

CLUSTERING_METHOD="kmeans"
N_CLUSTERS=20
RESOLUTION=1.0

NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-20}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# Fail fast (on every array task) if the exclusion list was not built yet.
if [[ ! -f "${INPUT_LIST}" ]]; then
  echo "[ERROR] Input list not found: ${INPUT_LIST}" >&2
  echo "        Build it ONCE on the login node before submitting:" >&2
  echo "          comm -23 <(ls -1 ${NDPI_DIR}/*.ndpi | LC_ALL=C sort) \\" >&2
  echo "                   <(LC_ALL=C sort ${WORK_DIR}/manual_preprocessed_inputs.txt) \\" >&2
  echo "                   > ${INPUT_LIST}" >&2
  exit 1
fi

# Load module and activate environment
module load miniforge3/25.3.0-3
source activate s2omics

# Stream Python print output to SLURM logs without buffering.
export PYTHONUNBUFFERED=1

# Print execution plan
echo "=========================================="
echo "S2Omics Pipeline Configuration (auto / non-manual samples)"
echo "=========================================="
echo "Steps to execute: ${START_STEP}-${END_STEP}"
echo "  - Step 1: NDPI conversion + histology preprocess (full slide, level ${TARGET_LEVEL})"
echo "  - Step 2: Superpixel QC (masking: ${MASKING_METHOD})"
echo "  - Step 3: Feature extraction (model: ${FOUNDATION_MODEL})"
echo "Down-sampling step: ${DOWN_SAMP_STEP}"
echo "Input list: ${INPUT_LIST}"
echo "Output directory: ${WORK_DIR}"
echo "=========================================="
echo ""

echo "[INFO] Launching run_batch.py (unbuffered logging enabled)"

python -u run_batch.py \
  --input-list "${INPUT_LIST}" \
  --work-dir "${WORK_DIR}" \
  --target-level "${TARGET_LEVEL}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --ckpt-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --masking-method "${MASKING_METHOD}" \
  --victor-mean-threshold "${VICTOR_MEAN_THRESHOLD}" \
  --victor-max-iterations "${VICTOR_MAX_ITERATIONS}" \
  --victor-sigma "${VICTOR_SIGMA}" \
  --victor-superpixel-threshold "${VICTOR_SUPERPIXEL_THRESHOLD}" \
  --clustering-method "${CLUSTERING_METHOD}" \
  --n-clusters "${N_CLUSTERS}" \
  --resolution "${RESOLUTION}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}"

#!/bin/bash
#SBATCH --job-name=s2omics_step45_globalpca
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=goerkemkadir.solun@dqbm.uzh.ch
#SBATCH --partition=lowprio

set -euo pipefail

ROOT_DIR="/scratch/gsolun"
PROJECT_DIR="${ROOT_DIR}/S2Omics"
INPUT_GLOB="${ROOT_DIR}/Data/NSCLC2/data/*.ndpi"
WORK_DIR="${PROJECT_DIR}/outputs"

# Model settings
FOUNDATION_MODEL="uni"
CKPT_PATH="${PROJECT_DIR}/checkpoints/uni/"
DEVICE="cuda:0"
DOWN_SAMP_STEP=10
BATCH_SIZE=16
NUM_WORKERS=2
TARGET_LEVEL=0

# === PIPELINE STEPS TO EXECUTE ===
# Set START_STEP and END_STEP to define the range of steps to run
# Options: 1=NDPI conversion, 2=preprocess, 3=feature extraction, 4=segmentation, 5=merge clusters
START_STEP=4
END_STEP=5

# === STEP 4/5 SPECIFIC SETTINGS ===
CLUSTERING_METHOD="kmeans"
N_CLUSTERS=20
RESOLUTION=1.0
TARGET_N_CLUSTERS=15
N_PCA_COMPONENTS=80

# === GLOBAL PCA SETTINGS (only applies if step 4 is in the execution range) ===
# Set PCA_MODE to "per-sample" (default) or "global" (fit across multiple samples)
PCA_MODE="per-sample"
GLOBAL_PCA_MODEL_PATH="${WORK_DIR}/global_pca_${FOUNDATION_MODEL}_downsamp_${DOWN_SAMP_STEP}.pickle"

# === SAMPLE DISCOVERY SETTINGS ===
# When START_STEP >= 4, samples are discovered from existing output folders.
# Optional: limit sample discovery to samples with this prefix; leave empty to include all
SAMPLE_PREFIX="LUNG-NSCLC2-"

NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-20}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# Global PCA must be fit on one shared sample set.
# If running step45-only (START_STEP >= 4) with global PCA, keep only task 0 active.
if [[ ${START_STEP} -ge 4 && "${PCA_MODE}" == "global" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    echo "[INFO] Skipping array task ${SLURM_ARRAY_TASK_ID} for shared global PCA run."
    exit 0
  fi
  NUM_TASKS=1
  TASK_ID=0
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

# Load module and activate environment
module load miniforge3/25.3.0-3
source activate s2omics

# Stream Python print output to SLURM logs without buffering.
export PYTHONUNBUFFERED=1

# Print execution plan
echo "=========================================="
echo "S2Omics Pipeline Configuration"
echo "=========================================="
echo "Steps to execute: ${START_STEP}-${END_STEP}"
if [[ ${START_STEP} -le 4 && ${END_STEP} -ge 4 ]]; then
  echo "  - Step 4: Histology segmentation (PCA mode: ${PCA_MODE})"
fi
if [[ ${START_STEP} -le 5 && ${END_STEP} -ge 5 ]]; then
  echo "  - Step 5: Merge over clusters"
fi
if [[ ${START_STEP} -le 3 && ${END_STEP} -ge 3 ]]; then
  echo "  - Step 3: Feature extraction (model: ${FOUNDATION_MODEL})"
fi
if [[ ${START_STEP} -le 2 && ${END_STEP} -ge 2 ]]; then
  echo "  - Step 2: Superpixel QC"
fi
if [[ ${START_STEP} -le 1 && ${END_STEP} -ge 1 ]]; then
  echo "  - Step 1: Histology preprocessing"
fi
echo "Output directory: ${WORK_DIR}"
echo "=========================================="
echo ""

echo "[INFO] Launching run_batch.py (unbuffered logging enabled)"

python -u run_batch.py \
  --input-glob "${INPUT_GLOB}" \
  --work-dir "${WORK_DIR}" \
  --target-level "${TARGET_LEVEL}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --ckpt-path "${CKPT_PATH}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --clustering-method "${CLUSTERING_METHOD}" \
  --n-clusters "${N_CLUSTERS}" \
  --resolution "${RESOLUTION}" \
  --target-n-clusters "${TARGET_N_CLUSTERS}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --task-id "${TASK_ID}" \
  --num-tasks "${NUM_TASKS}" \
  --skip-existing \
  --pca-mode "${PCA_MODE}" \
  --n-pca-components "${N_PCA_COMPONENTS}" \
  --global-pca-model-path "${GLOBAL_PCA_MODEL_PATH}" \
  ${SAMPLE_PREFIX:+--sample-prefix "${SAMPLE_PREFIX}"}

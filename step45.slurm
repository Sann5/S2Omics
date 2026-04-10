#!/bin/bash
#SBATCH --job-name=s2omics_step45_single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=goerkemkadir.solun@dqbm.uzh.ch
#SBATCH --partition=lowprio

set -euo pipefail

ROOT_DIR="/scratch/gsolun"
PROJECT_DIR="${ROOT_DIR}/S2Omics"
WORK_DIR="${PROJECT_DIR}/outputs_1"

# Model settings
FOUNDATION_MODEL="uni"
DOWN_SAMP_STEP=10
NUM_WORKERS=2

# === PIPELINE STEPS TO EXECUTE ===
# Options: 1=NDPI conversion, 2=preprocess, 3=feature extraction, 4=segmentation, 5=merge clusters
START_STEP=4
END_STEP=5

# === STEP 4/5 SPECIFIC SETTINGS ===
CLUSTERING_METHOD="kmeans"
N_CLUSTERS=20
RESOLUTION=1.0
TARGET_N_CLUSTERS=15

# === GLOBAL PCA SETTINGS ===
# Global PCA is fit once in this single non-array run.
PCA_MODE="global"
N_PCA_COMPONENTS=80
GLOBAL_PCA_MODEL_PATH="${WORK_DIR}/global_pca_${FOUNDATION_MODEL}_downsamp_${DOWN_SAMP_STEP}.pickle"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module load miniforge3/25.3.0-3
source activate s2omics

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "S2Omics Step 4-5 Configuration"
echo "=========================================="
echo "Steps to execute: ${START_STEP}-${END_STEP}"
echo "PCA mode: ${PCA_MODE}"
echo "Output directory: ${WORK_DIR}"
echo "=========================================="
echo ""

echo "[INFO] Launching run_batch.py for steps 4-5"

python -u run_batch.py \
  --work-dir "${WORK_DIR}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --clustering-method "${CLUSTERING_METHOD}" \
  --n-clusters "${N_CLUSTERS}" \
  --resolution "${RESOLUTION}" \
  --target-n-clusters "${TARGET_N_CLUSTERS}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --pca-mode "${PCA_MODE}" \
  --n-pca-components "${N_PCA_COMPONENTS}" \
  --global-pca-model-path "${GLOBAL_PCA_MODEL_PATH}"

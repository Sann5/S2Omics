#!/bin/bash
#SBATCH --job-name=s2omics_step4_single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=goerkemkadir.solun@dqbm.uzh.ch
#SBATCH --partition=lowprio

set -euo pipefail

ROOT_DIR="/scratch/gsolun"
PROJECT_DIR="${ROOT_DIR}/S2Omics"
WORK_DIR="${PROJECT_DIR}/outputs_1"

# Model settings
FOUNDATION_MODEL="uni"
DOWN_SAMP_STEP=7
NUM_WORKERS=2

# === PIPELINE STEPS TO EXECUTE ===
# Options: 1=NDPI conversion, 2=preprocess, 3=feature extraction, 4=segmentation
START_STEP=4
END_STEP=4

# === STEP 4 SPECIFIC SETTINGS ===
CLUSTERING_METHOD="kmeans"
N_CLUSTERS=20
RESOLUTION=1.0

# === JOINT PCA + HARMONY + JOINT CLUSTERING ===
# Step 4 always uses global PCA + Harmony batch correction + joint KMeans
# across all samples simultaneously (required for globally comparable cluster IDs).
# --global-pca-model-path is optional: if the file exists it is loaded as a
# pre-fitted PCA; otherwise PCA is fit inside step 4 on the concatenated embeddings.
N_PCA_COMPONENTS=80
GLOBAL_PCA_MODEL_PATH="${WORK_DIR}/global_pca_${FOUNDATION_MODEL}_downsamp_${DOWN_SAMP_STEP}.pickle"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module load miniforge3/25.3.0-3
source activate s2omics

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "S2Omics Step 4 Configuration"
echo "=========================================="
echo "Steps to execute: ${START_STEP}-${END_STEP}"
echo "Mode: joint segmentation (global PCA + Harmony + joint KMeans)"
echo "Output directory: ${WORK_DIR}"
echo "=========================================="
echo ""

echo "[INFO] Launching run_batch.py for step 4 (joint clustering across all samples)"

python -u run_batch.py \
  --work-dir "${WORK_DIR}" \
  --foundation-model "${FOUNDATION_MODEL}" \
  --down-samp-step "${DOWN_SAMP_STEP}" \
  --clustering-method "${CLUSTERING_METHOD}" \
  --n-clusters "${N_CLUSTERS}" \
  --resolution "${RESOLUTION}" \
  --start-step "${START_STEP}" \
  --end-step "${END_STEP}" \
  --num-workers "${NUM_WORKERS}" \
  --n-pca-components "${N_PCA_COMPONENTS}" \
  --global-pca-model-path "${GLOBAL_PCA_MODEL_PATH}"

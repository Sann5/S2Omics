#!/bin/bash
#SBATCH --job-name=s2omics_manual_preprocess
#SBATCH --cpus-per-task=12
#SBATCH --mem=512GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=goerkemkadir.solun@dqbm.uzh.ch
#SBATCH --partition=lowprio

set -euo pipefail

ROOT_DIR="/scratch/gsolun"
PROJECT_DIR="${ROOT_DIR}/S2Omics"

# Cluster paths -- VERIFY these match where the HES1 .ndpi files actually live.
# step123.slurm uses ${ROOT_DIR}/Data/NSCLC1/larges; the manual list expects HES1.
export NDPI_DIR="${ROOT_DIR}/Data/NSCLC1/HES1"
export WORK_DIR="${PROJECT_DIR}/outputs_1"
export TARGET_LEVEL=0
export PYTHON_BIN=python

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

module load miniforge3/25.3.0-3
source activate s2omics

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "S2Omics manual preprocessing (steps 0 + 1)"
echo "NDPI_DIR:  ${NDPI_DIR}"
echo "WORK_DIR:  ${WORK_DIR}"
echo "Level:     ${TARGET_LEVEL}"
echo "=========================================="

# Pass WORK_DIR and TARGET_LEVEL positionally too (script reads $1/$2 first).
bash run_manual_sample_preprocessing.sh "${WORK_DIR}" "${TARGET_LEVEL}"

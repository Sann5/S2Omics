#!/usr/bin/env bash
set -euo pipefail

# Collect every sample's p4_segmentation folder into demo/outputs/, preserving
# the <sample>/S2Omics_output/p4_segmentation/ layout used by the rest of demo/.
#
# Usage:
#   ./collect_p4_segmentation.sh [SRC] [DST]
# Defaults (override via args or SRC=/DST= env vars):
#   SRC = /scratch/gsolun/S2Omics/outputs_1   (where the pipeline wrote outputs)
#   DST = <repo>/demo/outputs
#
# Samples without a p4_segmentation folder (step 4 not finished) are skipped and
# reported -- so the "copied" count also tells you how many samples finished step 4.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-${1:-/scratch/gsolun/S2Omics/outputs_1}}"
DST="${DST:-${2:-${PROJECT_DIR}/demo/outputs}}"

if [[ ! -d "${SRC}" ]]; then
  echo "[ERROR] Source directory not found: ${SRC}" >&2
  exit 1
fi

mkdir -p "${DST}"
shopt -s nullglob

copied=0
skipped=0
for sample_dir in "${SRC}"/*/; do
  sample="$(basename "${sample_dir}")"
  src_p4="${sample_dir}S2Omics_output/p4_segmentation"
  if [[ ! -d "${src_p4}" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  dst_p4="${DST}/${sample}/S2Omics_output/p4_segmentation"
  mkdir -p "$(dirname "${dst_p4}")"
  rsync -a "${src_p4}/" "${dst_p4}/"
  echo "[COPY] ${sample}"
  copied=$((copied + 1))
done

echo ""
echo "Copied p4_segmentation for ${copied} sample(s); skipped ${skipped} without one."
echo "Source:      ${SRC}"
echo "Destination: ${DST}"

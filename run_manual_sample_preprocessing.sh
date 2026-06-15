#!/usr/bin/env bash
set -euo pipefail

# Manually selected NDPI conversion (step 0) and H&E preprocessing (step 1).
# Edit samples below, crop values are percentages in: top bottom left right order.
# Each selected crop becomes the canonical pipeline input for its sample.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NDPI_DIR="${NDPI_DIR:-/Users/gorkemkadirsolun/Library/CloudStorage/GoogleDrive-gorkemkadirsolun@gmail.com/My Drive/Job_Work/Bodenmiller/Data/NSCLC1/HES1}"
WORK_DIR="${1:-${WORK_DIR:-/Users/gorkemkadirsolun/Library/CloudStorage/GoogleDrive-gorkemkadirsolun@gmail.com/My Drive/Job_Work/Bodenmiller/Data/NSCLC1_S2Omics_Preprocessed}}"
TARGET_LEVEL="${2:-${TARGET_LEVEL:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Format: "exact_filename.ndpi crop_top crop_bottom crop_left crop_right"
# Add one line per image to process. Partial sample identifiers are not accepted.
SAMPLES=(
  "IMMU-NSCLC-0093-FIXT-01-HES-01_#_8c5674364a8d6c923149deea024c64cc.ndpi 0 0 22 0"  # Manual crop: remove 22 percent from the left.
  "IMMU-NSCLC-0106-FIXT-01-HES-01_#_8c4db8d7b2ede316f7f8faa832763e6c.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-0872-FIXT-02-HES-01_#_cf0ed8a2ea625d135b787b07bd516384.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1148-FIXT-01-HES-01_#_a1552804f47f6d2c6d229361af361d99.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1350-FIXT-01-HES-01_#_15eef50dfb31e15f750bd4e0098578d0.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1352-FIXT-01-HES-01_#_688829c34cfcb37dd56791be58667bb6.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1353-FIXT-01-HES-01_#_PATIENT_WITHDRAWAL_#_688829c34cfcb37dd56791be58667bb6.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1364-FIXT-01-HES-01_#_1be01241c908304cbf2028597792d789.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1367-FIXT-01-HES-01_#_d2711b9531b53d8cf6d26323d7e87d7e.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1382-FIXT-01-HES-01_#_5bfda1a440c4160e4963d9fde202044c.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1397-FIXT-01-HES-01_#_1d3b536dee1629a35b6701c3244e7ee5.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1406-FIXT-01-HES-01_#_28f837a34c0efc602fe45f7cc5ceb693.ndpi 0 0 35 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1415-FIXT-01-HES-01_#_9324316471b45aad6c1082d7b8f59d42.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1419-FIXT-01-HES-01_#_9324316471b45aad6c1082d7b8f59d42.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1420-FIXT-01-HES-01_#_769560642cd4d837941529cfe2ebf295.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1677-FIXT-01-HES-01_#_3c1f2de423bd5092395ba4f362dced22.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1711-FIXT-01-HES-01_#_2f60e97e5e82579c6d83cf2acffc8231.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1758-FIXT-01-HES-01_#_3643d656168f0b1a93a01f04a345216d.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1832-FIXT-01-HES-01_#_5cc8eb621979ba91a710d90c60d25dc6.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1837-FIXT-02-HES-01_#_bf9ac0791684c77d49fb6e3dcbfa6234.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1847-FIXT-01-HES-01_#_4549bae4d4e2095cc828d60bdc9d6296.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1850-FIXT-01-HES-01_#_b4ff983c70bd040bae8e4e5640b0d123.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1852-FIXT-01-HES-01_#_986380ac8488b560d4441713686c189f.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1858-FIXT-01-HES-01_#_f323451d90c7e39b630e2fc9a9eb031d.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1864-FIXT-01-HES-01_#_2b1b3891a9b6310f2460777775192804.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1881-FIXT-01-HES-01_#_4b0caa74387fd689ae844bf529b264d4.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1945-FIXT-01-HES-01_#_ef560d2969b61e0d134584835370322b.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1953-FIXT-01-HES-01_#_703425259521a08bc6456cf6474bd099.ndpi 0 0 60 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1954-FIXT-01-HES-01_#_aaf74c27a3221d83d3250c105813423e.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-1976-FIXT-01-HES-01_#_e7a0ae42a6a5388017eeb40579cdcb6e.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-1978-FIXT-01-HES-01_#_c0cccdd3befcb2b752edb2e4283c3c57.ndpi 0 0 40 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-2021-FIXT-01-HES-01_#_a759748a703dae3dd14a556c0e648de0.ndpi 0 0 0 40"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2022-FIXT-01-HES-01_#_b9c7e87f7b03a0b43118993ea9593854.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-2025-FIXT-01-HES-01_#_b9c7e87f7b03a0b43118993ea9593854.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2026-FIXT-01-HES-01_#_a759748a703dae3dd14a556c0e648de0.ndpi 0 0 75 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-2036-FIXT-02-HES-01_#_77beee5343a13c08354998438b5cb290.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2077-FIXT-01-HES-01_#_7ef5247bc8465db788f24efba278cc89.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2132-FIXT-02-HES-01_#_9c1ba794871b27f2b1a5aec42bb5a915.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2141-FIXT-01-HES-01_#_7c0d6ab9bb590acc3f9d5d95b0548d87.ndpi 0 0 50 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-2154-FIXT-01-HES-01_#_dc056bbcb16839f1f4907898b4fc4a61.ndpi 0 0 0 70"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2166-FIXT-01-HES-01_#_ece77a7533f1336b83e1689e13beef24.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2181-FIXT-01-HES-01_#_76b916ae0d0d601908592cd0cd87d2fb.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2192-FIXT-01-HES-01_#_776a59b871cdbb48e66e88c6ee07ce7c.ndpi 0 0 0 65"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2225-FIXT-01-HES-01_#_34d4c9f24a7fabb713cf78ba4a4e8410.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2235-FIXT-01-HES-01_#_963cbfc0b405c0373e8867fcb04002d2.ndpi 0 0 0 50"  # 1:2: keep left tissue; crop right half.
  "IMMU-NSCLC-2236-FIXT-01-HES-01_#_fe43fba2e80573907741189a8bbede04.ndpi 0 0 45 0"  # 2:2: keep right tissue; crop left half.
  "IMMU-NSCLC-2439-FIXT-01-HES-01_#_0778ac76ebdf6177a331356ce928b61e.ndpi 0 0 0 30"  # 1:2: keep left tissue; crop right half.
)

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  printf 'Usage: %s [work_dir] [target_level]\n' "$(basename "$0")"
  printf '\nEdit the SAMPLES array inside the script before running it.\n'
  printf 'Entry format: "exact_filename.ndpi crop_top crop_bottom crop_left crop_right"\n'
  printf 'Environment overrides: NDPI_DIR, WORK_DIR, TARGET_LEVEL, PYTHON_BIN\n'
  exit 0
fi

if [[ ! -d "${NDPI_DIR}" ]]; then
  printf '[ERROR] NDPI directory not found: %s\n' "${NDPI_DIR}" >&2
  exit 1
fi

if [[ "${#SAMPLES[@]}" -eq 0 ]]; then
  printf '[ERROR] No samples configured. Edit the SAMPLES array in %s first.\n' "$(basename "$0")" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"
cd "${PROJECT_DIR}"
shopt -s nullglob

processed=0
skipped=0
input_list="${WORK_DIR}/manual_preprocessed_inputs.txt"
: > "${input_list}"

for entry in "${SAMPLES[@]}"; do
  read -r filename crop_top crop_bottom crop_left crop_right extra <<< "${entry}"
  if [[ -z "${filename:-}" || -z "${crop_right:-}" || -n "${extra:-}" ]]; then
    printf '[ERROR] Invalid SAMPLES entry: "%s"\n' "${entry}" >&2
    printf '        Required format: "exact_filename.ndpi crop_top crop_bottom crop_left crop_right"\n' >&2
    exit 1
  fi

  if [[ "${filename}" != *.ndpi || "$(basename "${filename}")" != "${filename}" ]]; then
    printf '[ERROR] SAMPLES must contain a complete .ndpi filename, not a sample ID or path: "%s"\n' "${filename}" >&2
    exit 1
  fi

  ndpi_path="${NDPI_DIR}/${filename}"
  if [[ ! -f "${ndpi_path}" ]]; then
    printf '[ERROR] NDPI file not found: %s\n' "${ndpi_path}" >&2
    exit 1
  fi

  sample_name="${filename%.ndpi}"
  save_folder="${WORK_DIR}/${sample_name}/S2Omics_output"
  crop_parameters_path="${save_folder}/manual_crop_parameters.txt"
  crop_parameters="ndpi=${filename}
target_level=${TARGET_LEVEL}
crop_top=${crop_top}
crop_bottom=${crop_bottom}
crop_left=${crop_left}
crop_right=${crop_right}"

  if [[ -d "${save_folder}" ]]; then
    if [[ ! -f "${crop_parameters_path}" ]]; then
      printf '[ERROR] Existing output has no manual crop metadata: %s\n' "${save_folder}" >&2
      printf '        Use a different WORK_DIR or clear this sample output before manual preprocessing.\n' >&2
      exit 1
    fi
    if [[ "$(cat "${crop_parameters_path}")" != "${crop_parameters}" ]]; then
      printf '[ERROR] A different manual crop already exists for %s in this output root.\n' "${sample_name}" >&2
      printf '        Use a different WORK_DIR for alternate crops.\n' >&2
      exit 1
    fi
    if [[ -f "${save_folder}/p0_ndpi_conversion/he-raw.tiff" &&
          -f "${save_folder}/p0_ndpi_conversion/pixel-size-raw.txt" &&
          -f "${save_folder}/p1_preprocess/he-scaled.tiff" &&
          -f "${save_folder}/p1_preprocess/he.tiff" ]]; then
      printf '[SKIP]  %s: matching preprocessing already exists.\n' "${sample_name}"
      printf '%s\n' "${ndpi_path}" >> "${input_list}"
      ((skipped += 1))
      continue
    fi
  fi

  mkdir -p "${save_folder}"
  printf '%s\n' "${crop_parameters}" > "${crop_parameters_path}"

  printf '[START] %s (%s; top=%s bottom=%s left=%s right=%s)\n' \
    "${sample_name}" "${filename}" "${crop_top}" "${crop_bottom}" "${crop_left}" "${crop_right}"
  "${PYTHON_BIN}" -m s2omics.p0_ndpi_conversion \
    "${ndpi_path}" \
    --save-folder "${save_folder}" \
    --level "${TARGET_LEVEL}" \
    --crop-top "${crop_top}" \
    --crop-bottom "${crop_bottom}" \
    --crop-left "${crop_left}" \
    --crop-right "${crop_right}"
  "${PYTHON_BIN}" -c \
    'import sys; from s2omics.p1_histology_preprocess import histology_preprocess; histology_preprocess(sys.argv[1])' \
    "${save_folder}"
  printf '[DONE]  %s -> %s\n' "${sample_name}" "${save_folder}"
  printf '%s\n' "${ndpi_path}" >> "${input_list}"
  ((processed += 1))
done

printf '\nFinished manual preprocessing for %d new sample(s); skipped %d completed sample(s).\n' "${processed}" "${skipped}"
printf 'Output root: %s\n' "${WORK_DIR}"
printf 'Resume input list: %s\n' "${input_list}"

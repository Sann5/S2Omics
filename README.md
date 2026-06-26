# S2Omics

Smart spatial omics (S2-omics) is an end-to-end workflow that selects regions of
interest (ROIs) for spatial-omics experiments directly from H&E histology images, and
virtually reconstructs spatial molecular profiles across whole tissue sections.

This repository is an extension of the original S2Omics project
([ddb-qiwang/S2Omics](https://github.com/ddb-qiwang/S2Omics), Yuan et al., *Nat Cell Biol*
2025 — <https://doi.org/10.1038/s41556-025-01811-w>). On top of the upstream pipeline this
copy adds:

- NDPI conversion for Hamamatsu whole-slide images (p0),
- a batch driver (run_batch.py) and SLURM scripts for running hundreds of slides on the UZH/DQBM cluster,
- an alternative "Victor" tissue-masking method,
- a survival-analysis module that correlates H&E-derived clusters with clinical outcomes and IMC cell-type densities,

all applied to an NSCLC (IMMUcan) cohort.

---

## 1. Repository map

```text
S2Omics/
├── s2omics/                      #   all the pipeline CODE
│   ├── step_paths.py             #   single source of truth for per-step output folder names
│   ├── p0_ndpi_conversion.py     #   step p0: Hamamatsu .ndpi  → he-raw.tiff + pixel size
│   ├── p1_histology_preprocess.py#   step p1: rescale to 0.5 µm/px + pad to ×256 → he.tiff
│   ├── p2_superpixel_quality_control.py  # step p2: tissue/background superpixel QC mask
│   ├── p3_feature_extraction.py  #   step p3: foundation-model (UNI/Virchow/GigaPath) embeddings  [GPU]
│   ├── s1_utils.py               #   shared I/O + idempotency helpers
│   ├── s2_label_broadcasting.py  #   model class (autoencoder+classifier) used by p6
│   ├── HistoSweep/               #   QC / tissue-masking engine used by p2 (+ victorMasking.py)
│   ├── multiple_sections/        #   p4 joint segmentation, p5 ROI/FOV selection, p6 label broadcasting
│   ├── color_list*.txt           #   cluster colour palettes (needed by plotting code)
│   ├── survival analysis.ipynb   #   survival-analysis notebook (intended entrypoint)
│   └── survival analysis.py      #   survival-analysis library (KM / Cox / IMC correlation)
│
├── run_batch.py                  #   MAIN driver: NDPI→TIFF + steps 1–4 over many slides
├── run_roi_selection_multiple.py #   steps 1–4 then ROI selection across sections
├── run_label_broadcasting.py     #   transfer spatial-omics annotations onto an H&E slide (p6)
├── victor_masking.py             #   thin wrapper around HistoSweep/victorMasking.py
│
├── *.slurm  +  *.sh              #   cluster orchestration (see §4.2)
│
├── requirements.txt              #   runtime deps (modern GCC/CUDA)
├── requirements_old_gcc.txt      #   runtime deps for old-GCC servers
│
├── checkpoints/                  #   foundation-model weights (NOT in git, download separately)
│   ├── uni/  gigapath/  virchow2/  hipt/
│
├── demo/                         #   INPUT data + collected OUTPUTS (NOT in git, download separately)
│   ├── *.ndpi                    #   raw whole-slide images (NSCLC demo slides)
│   ├── Tutorial_1..4_.../        #   upstream tutorial inputs (he-raw.jpg + pixel-size-raw.txt)
│   ├── s2omics_demo_samples_*.txt#   sample lists (one .ndpi absolute path per line)
│   ├── outputs/                  #   collected per-sample p4_segmentation results (~231 samples)
│   ├── survival/                 #   clinical + IMC density tables + cluster summaries
│   └── victor_mask_eval/         #   masking QC previews
│
├── docs/                         #   Sphinx / ReadTheDocs documentation source
```
---

## 2. Environment setup

Requires conda and (for the feature-extraction / segmentation steps) a CUDA GPU.

```bash
git clone https://github.com/ddb-qiwang/S2Omics   # or your fork remote
cd S2Omics

conda create -n s2omics python=3.11
conda activate s2omics

pip install -r requirements.txt
# Old-GCC servers instead: pip install -r requirements_old_gcc.txt
#   (that file pins torch 2.0.1 + CUDA-11 wheels but OMITS openslide-python, install it manually)

# Register a Jupyter kernel (for the tutorial / survival notebooks)
python -m ipykernel install --user --name s2omics --display-name s2omics
```

Notes:

- OpenSlide native library is required to read .ndpi slides — the pip wheel alone is not
  always enough. Install the C library too: `brew install openslide` (macOS) or
  `apt install openslide-tools` (Linux).
- PyTorch / CUDA: requirements.txt pins torch==2.4.1 / torchvision==0.19.1. On
  H100 / SM90 clusters you may need the CUDA-12.1 wheels explicitly:
  `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121`.
- Survival analysis uses statsmodels (pinned) and may also need lifelines, which is
  not in either requirements file — `pip install lifelines` if needed.
- Use Python 3.10–3.11, avoid 3.12+.

---

## 3. Getting the data and model checkpoints

Neither the model weights nor the demo data are in git. Download both from the project
Google Drive and place them under the repo root:

> <https://drive.google.com/drive/folders/1z1nk0sF_e25LKMyHxJVMtROFjuWet2G_>

```text
checkpoints/uni/pytorch_model.bin          # UNI (default, ~1.2 GB)
checkpoints/gigapath/pytorch_model.bin     # Prov-GigaPath (~2.0 GB)
checkpoints/virchow2/pytorch_model.bin     # Virchow2 (~2.5 GB)
checkpoints/hipt/vit256_small_dino.pth + vit4k_xs_dino.pth   # HIPT (two-stage)
demo/...                                    # tutorial inputs + NSCLC slides + survival tables
```

`checkpoints/<model>/` must contain a local `pytorch_model.bin` — there is no auto-download.

### Input data format (per slide)

- `he-raw.jpg` (or `.tiff`) — raw H&E image. *(For .ndpi slides this is produced by step p0.)*
- `pixel-size-raw.txt` — a one-line file with the pixel size in microns/pixel (e.g. 0.5).
- `annotation_file.csv` *(optional, only for label broadcasting)* — columns
  `super_pixel_x`, `super_pixel_y`, `annotation`.

---

## 4. The pipeline & how to run it

### Conceptual steps

| Step | Module                                     | What it does                                                              | GPU |
| ---- | ------------------------------------------ | ------------------------------------------------------------------------ | --- |
| p0   | `s2omics/p0_ndpi_conversion.py`            | .ndpi → flat RGB he-raw.tiff + effective µm/px                            | –   |
| p1   | `s2omics/p1_histology_preprocess.py`       | rescale to 0.5 µm/px, pad to a multiple of 256 → he.tiff                  | –   |
| p2   | `s2omics/p2_superpixel_quality_control.py` | tile into superpixels, compute tissue/background mask (s2omics or victor) | –   |
| p3   | `s2omics/p3_feature_extraction.py`         | foundation-model (UNI/Virchow/GigaPath) embedding per superpixel          |     |
| p4   | `s2omics/multiple_sections/p4_*`           | joint segmentation: global PCA + Harmony batch-correction + clustering    |     |
| p5   | `s2omics/multiple_sections/p5_*`           | ROI / FOV selection across sections (rectangle default, circle variant)  | –   |
| p6   | `s2omics/multiple_sections/p6_*`           | broadcast spatial-omics cell-type labels onto the whole slide            |     |

> run_batch.py step numbering: in the batch driver step 1 = p0 + p1, step 2 = p2,
> step 3 = p3, step 4 = p4. Steps 1–3 run per sample, step 4 is JOINT — it pools the
> QC'd embeddings of *all* samples, fits one global PCA (80 components by default) + Harmony,
> then clusters everyone together so cluster IDs are comparable across slides. ROI selection
> (p5) and label broadcasting (p6) are run by the separate run_roi_selection_multiple.py /
> run_label_broadcasting.py drivers.

### 4.1 Run a batch locally

Full pipeline (steps 1→4) over a folder of NDPIs:

```bash
python run_batch.py \
  --input-glob '/data/wsi/**/*.ndpi' \      # or --input-list slides.txt (one .ndpi path per line)
  --work-dir   /out/run1 \
  --foundation-model uni --ckpt-path ./checkpoints/uni/ \
  --device cuda:0 --down-samp-step 5 \
  --masking-method victor \
  --clustering-method kmeans --n-clusters 20
```

Re-cluster only (step 4 on already-processed samples, no NDPIs needed):

```bash
python run_batch.py --work-dir /out/run1 --start-step 4 --end-step 4 \
  --foundation-model uni --down-samp-step 5 \
  --global-pca-model-path /out/run1/global_pca_uni_downsamp_5.pickle
```

Key arguments:

- Inputs: `--input-glob` and/or `--input-list` (merged, de-duped, sorted). Step-4-only
  mode discovers existing sample folders under `--work-dir` instead.
- Steps: `--start-step` / `--end-step` (1–4).
- Step 2 masking: `--masking-method {s2omics,victor}` + `--density-thresh`, `--clean-background-flag`,
  `--min-size`, `--patch-size`, and Victor knobs (`--victor-mean-threshold 0.85`,
  `--victor-sigma 20`, `--victor-superpixel-threshold 0.5`, `--victor-positive-contrast`).
- Step 3: `--foundation-model {uni,virchow,gigapath}`, `--ckpt-path`, `--device`,
  `--batch-size`, `--down-samp-step` (default 10 ≈ 1 % of superpixels, must match in step 4).
- Step 4: `--clustering-method {kmeans,fcm,agglo,bisect,birch,louvain,leiden}`,
  `--n-clusters`, `--resolution` (leiden/louvain), `--n-pca-components` (default 80),
  `--if-evaluate`, `--global-pca-model-path`.
- Robustness: per-sample try/except (one bad slide doesn't kill the batch), `--stop-on-error`
  to override. `--task-id`/`--num-tasks` for round-robin SLURM array sharding.

### 4.2 Run on the SLURM cluster

The cluster scripts target the UZH/DQBM lowprio partition with scratch at
/scratch/gsolun. Samples are split into a manually-cropped set and an auto/full-slide
set, both written to the same outputs_1 work-dir so a single joint step 4 clusters
everyone together. Run in this order:

```bash
# 1. Steps 0+1 on the 47 manually-cropped slides → writes manual_preprocessed_inputs.txt (CPU only)
sbatch manual_preprocess.slurm

# 2. Steps 2–3 on those manual slides (GPU array job, starts at step 2 to preserve manual crops)
sbatch step123.slurm

# 3. Build the auto list (all HES1 slides minus the manual ones), then steps 1–3 on the rest
comm -23 <(ls -1 /scratch/gsolun/Data/NSCLC1/HES1/*.ndpi | LC_ALL=C sort) \
         <(LC_ALL=C sort outputs_1/manual_preprocessed_inputs.txt) > outputs_1/auto_inputs.txt
sbatch step123_auto.slurm

# 4. After ALL step123 array tasks finish: single JOINT segmentation over every sample
sbatch step4.slurm

# 5. Gather each sample's p4_segmentation into the repo's demo/outputs/
./collect_p4_segmentation.sh
```

### 4.3 ROI selection & label broadcasting

```bash
# ROI selection across consecutive sections
python run_roi_selection_multiple.py \
  --save_folder_list /out/s1/S2Omics_output /out/s2/S2Omics_output \
  --foundation_model uni --clustering_method kmeans --n_clusters 20 \
  --roi_size 6.5 6.5 --num_roi 0          # num_roi 0 = auto

# Broadcast spatial-omics annotations onto a whole-slide H&E
python run_label_broadcasting.py \
  --WSI_save_folder /out/wsi/S2Omics_output \
  --SO_save_folder  /out/so/S2Omics_output \
  --SO_annotation_csv /data/annotation_file.csv \
  --need_preprocess --need_feature_extraction --device cuda:0
```

> Label broadcasting requires embeddings extracted with `down_samp_step=1` (the driver
> enforces this). ROI selection writes all its outputs into the first folder of
> `--save_folder_list`.

---

## 5. Where the outputs are stored

Every step writes into its own subfolder under a per-sample root, conventionally
`<work-dir>/<sample>/S2Omics_output/` (folder names are centralised in
[s2omics/step_paths.py](s2omics/step_paths.py)):

```text
<work-dir>/<sample>/S2Omics_output/
├── p0_ndpi_conversion/   he-raw.tiff, pixel-size-raw.txt
├── p1_preprocess/        he-scaled.tiff, he.tiff           (he.tiff = canonical input for p2/p3)
├── p2_qc/                shapes.pickle, qc_preserve_indicator.pickle (boolean tissue mask),
│                         qc_parameters.json, HistoSweep_output/{mask.png, mask-small.png}
├── p3_features/          <model>_embeddings_downsamp_<step>_part_<N>.pickle, num_patches.pickle,
│                         feature_extraction_complete_*.pickle
└── p4_segmentation/      cluster_image.pickle (2D cluster-label image, −1 = background),
                          cluster_image_num_clusters_<N>.jpg (legend labels 1..N),
                          joint_histology_segmentation_complete.pickle, [clustering_metrics.pickle]
```

- The sample folder is named after the slide (e.g.
  `IMMU-NSCLC-0096-FIXT-02-HES-01_#_<hash>`).
- On the cluster, results live under /scratch/gsolun/S2Omics/outputs_1,
  collect_p4_segmentation.sh rsyncs each p4_segmentation/ into the repo's demo/outputs/
  (~231 samples collected there).
- Idempotency: every step writes a completion marker and records its parameters. Reruns
  with the same outputs + parameters are skipped — to force a rerun, delete the relevant
  `*_complete.pickle` / output pickles or change a parameter.

---

## 6. Survival analysis

s2omics/survival analysis.ipynb (notebook, the intended entrypoint) and its library
s2omics/survival analysis.py link the H&E-derived clusters to clinical outcomes and IMC data.

Typical flow:

1. Summarize clusters → per-patient tissue area + cluster frequencies → cluster_summary.csv.
2. Merge with clinical data (demo/survival/survival.txt, TAB-separated, keyed on
   patient_id, fuzzy slide-ID ↔ clinical-ID matching) → merged_clinical_cluster_summary.csv.
3. Kaplan–Meier (log-rank) and LASSO Cox models per treatment subgroup, with C-index
   diagnostics, univariate forest/volcano plots.
4. IMC correlation: correlate H&E cluster frequencies against IMC cell-type / neighbourhood
   density tables (the four `*_density*_p1.csv` files in demo/survival/, joined on
   immucan_id), producing significance heatmaps, r-heatmaps, and bubble plots.

Inputs live in demo/survival/ (clinical + IMC tables + precomputed cluster_summary.csv /
merged_clinical_cluster_summary.csv). Figures render inline in the notebook (no savefig).

---

## 7. Things to know

- Step 4 is global, not per-sample — its clustering depends on the full set of samples in the
  run. Array shards each cluster only their own subset, so the final joint clustering must be the
  single step4.slurm job.
- GPU is needed for p3/p4/p6. p3 auto-falls back to CPU if CUDA is unavailable (slow). p0/p1/p2
  are CPU-only.
- Victor masking expects dark tissue on a bright background (standard H&E), add
  `--victor-positive-contrast` for the inverse.
- s2omics/s2_label_broadcasting.py is a model class, not the broadcasting driver — the real
  logic is s2omics/multiple_sections/p6_cell_label_broadcasting.py.
- Some HistoSweep/ files (preprocess.py, rescale.py, additionalPlots.py, saveParameters.py)
  are legacy/standalone and not wired into the live pipeline (scaling/padding is done by p1).
- The filenames survival analysis.py / .ipynb literally contain a space — quote them in the
  shell and import via importlib.

---

## Upstream project, docs & tutorials

- Upstream code: <https://github.com/ddb-qiwang/S2Omics>
- Paper: Yuan, M., Jin, K., Yan, H. et al. *Smart spatial omics (S2-omics) optimizes region of
  interest selection to capture molecular heterogeneity in diverse tissues.* Nat Cell Biol (2025).
  <https://doi.org/10.1038/s41556-025-01811-w>
- ReadTheDocs: <https://s2omics.readthedocs.io/en/latest/> (source under docs/)
- Tutorials (upstream notebooks, under docs/source/notebooks/): (1) VisiumHD ROI selection on
  colorectal cancer, (2) CosMx FOV selection on kidney, (3) consecutive breast-cancer ROI selection,
  (4) TMA circular ROI selection.

<div align="center">
    <img src="/docs/source/images/S2Omics_pipeline.png" alt="S2Omics pipeline" width="80%">
</div>

"""Per-step output folder layout for the S2Omics pipeline.

Every pipeline step writes into its own subfolder under the per-sample
``save_folder`` root (typically ``<sample>/S2Omics_output``). Centralising the
folder names here gives both producers and consumers a single source of truth.
"""

import os


P0_NDPI_CONVERSION = 'p0_ndpi_conversion'
P1_PREPROCESS = 'p1_preprocess'
P2_QC = 'p2_qc'
P3_FEATURES = 'p3_features'
P4_SEGMENTATION = 'p4_segmentation'
P4_5_BACKGROUND_FILTER = 'p4_5_background_filter'
P5_MERGE = 'p5_merge'  # single-section only
P5_ROI_SELECTION = 'p5_roi_selection'  # multi-section only
P6_ROI_SELECTION = 'p6_roi_selection'  # single-section only
P6_LABEL_BROADCASTING = 'p6_label_broadcasting'  # multi-section only
P7_LABEL_BROADCASTING = 'p7_label_broadcasting'  # single-section only

HISTOSWEEP_SUBFOLDER = 'HistoSweep_output'  # nested under p2_qc/


def step_dir(save_folder, step_name, create=True):
    """Return ``<save_folder>/<step_name>/`` (with trailing slash).

    Creates the directory by default so callers can write into it immediately.
    """
    out = os.path.join(save_folder, step_name)
    if create:
        os.makedirs(out, exist_ok=True)
    return out + '/'

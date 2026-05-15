import json
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from . import step_paths
from .HistoSweep.computeMetrics import compute_metrics_memory_optimized
from .HistoSweep.densityFiltering import compute_low_density_mask
from .HistoSweep.textureAnalysis import run_texture_analysis
from .HistoSweep.ratioFiltering import run_ratio_filtering
from .HistoSweep.generateMask import generate_final_mask
from .HistoSweep.UTILS import get_image_filename, load_image
from .s1_utils import save_pickle


def _qc_signature(
        density_thresh,
        clean_background_flag,
        min_size,
        patch_size,
        masking_method,
        victor_mean_threshold,
        victor_max_iterations,
        victor_sigma,
        victor_positive_contrast,
        victor_superpixel_threshold):
    return json.dumps(
        {
            "density_thresh": density_thresh,
            "clean_background_flag": clean_background_flag,
            "min_size": min_size,
            "patch_size": patch_size,
            "masking_method": masking_method,
            "victor_mean_threshold": victor_mean_threshold,
            "victor_max_iterations": victor_max_iterations,
            "victor_sigma": victor_sigma,
            "victor_positive_contrast": victor_positive_contrast,
            "victor_superpixel_threshold": victor_superpixel_threshold,
        },
        sort_keys=True,
    )


def patchify(x, patch_size):
    shape_ori = np.array(x.shape[:2])
    shape_ext = (
            (shape_ori + patch_size - 1)
            // patch_size * patch_size)
    pad_w = shape_ext[0] - x.shape[0]
    pad_h = shape_ext[1] - x.shape[1]
    print(pad_w, pad_h)
    x = np.pad(x, ((0, pad_w), (0, pad_h), (0, 0)), mode='edge')
    patch_index_mask = np.zeros(np.shape(x)[:2])
    tiles_shape = np.array(x.shape[:2]) // patch_size
    tiles = []
    counter = 0
    for i0 in range(tiles_shape[0]):
        a0 = i0 * patch_size
        b0 = a0 + patch_size
        for i1 in range(tiles_shape[1]):
            a1 = i1 * patch_size
            b1 = a1 + patch_size
            tiles.append(x[a0:b0, a1:b1])
            patch_index_mask[a0:b0, a1:b1] = counter
            counter += 1

    shapes = dict(
            original=shape_ori,
            padded=shape_ext,
            tiles=tiles_shape)
    patch_index_mask = patch_index_mask[:np.shape(x)[0] - pad_w, :np.shape(x)[1] - pad_h]
    return tiles, shapes, patch_index_mask


def superpixel_quality_control(save_folder,
                               density_thresh=100,
                               clean_background_flag=False,
                               min_size=10, patch_size=16,
                               masking_method="s2omics",
                               victor_mean_threshold=0.85,
                               victor_max_iterations=5,
                               victor_sigma=20,
                               victor_positive_contrast=False,
                               victor_superpixel_threshold=0.5,
                               show_image=False):
    """Run superpixel QC.

    Reads ``he.*`` from ``save_folder/p1_preprocess/`` and writes outputs
    (shapes/qc pickles, HistoSweep masks and plots) into
    ``save_folder/p2_qc/`` and ``save_folder/p2_qc/HistoSweep_output/``.

    clean_background_flag: Whether to remove small speckles in superpixel mask
    masking_method: "s2omics" for the default mask or "victor" for Victor's
        grayscale/Gaussian/Otsu mask.
    """
    if masking_method not in {"s2omics", "victor"}:
        raise ValueError(
            "masking_method must be either 's2omics' or 'victor', "
            f"got {masking_method!r}."
        )

    p1_dir = step_paths.step_dir(save_folder, step_paths.P1_PREPROCESS, create=False)
    p2_dir = step_paths.step_dir(save_folder, step_paths.P2_QC)
    histosweep_dir = os.path.join(p2_dir, step_paths.HISTOSWEEP_SUBFOLDER)
    os.makedirs(histosweep_dir, exist_ok=True)

    shapes_output = p2_dir + 'shapes.pickle'
    qc_output = p2_dir + 'qc_preserve_indicator.pickle'
    qc_signature_output = p2_dir + 'qc_parameters.json'
    signature = _qc_signature(
        density_thresh=density_thresh,
        clean_background_flag=clean_background_flag,
        min_size=min_size,
        patch_size=patch_size,
        masking_method=masking_method,
        victor_mean_threshold=victor_mean_threshold,
        victor_max_iterations=victor_max_iterations,
        victor_sigma=victor_sigma,
        victor_positive_contrast=victor_positive_contrast,
        victor_superpixel_threshold=victor_superpixel_threshold,
    )
    if os.path.exists(shapes_output) and os.path.exists(qc_output):
        if os.path.exists(qc_signature_output):
            with open(qc_signature_output, "r", encoding="utf-8") as f:
                existing_signature = f.read()
            if existing_signature == signature:
                print(
                    "Skipping QC; outputs already exist with matching parameters: "
                    f"'{shapes_output}', '{qc_output}'."
                )
                return
            print("QC outputs exist, but masking/QC parameters changed. Regenerating QC.")
        elif masking_method == "s2omics":
            print(
                "Skipping QC; outputs already exist: "
                f"'{shapes_output}', '{qc_output}'."
            )
            return
        else:
            print("Legacy QC outputs exist. Regenerating QC with Victor masking.")

    image = load_image(get_image_filename(p1_dir + 'he'))
    _, shapes, _ = patchify(image, patch_size)
    save_pickle(shapes, shapes_output)

    # HistoSweep functions construct paths as f"{prefix}/{output_dir}/...".
    # Pass the p2 directory as prefix so HistoSweep outputs nest under p2_qc/.
    histosweep_prefix = p2_dir.rstrip('/')

    if masking_method == "s2omics":
        he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = (
            compute_metrics_memory_optimized(image, patch_size=patch_size)
        )

        # identify low density superpixels
        mask1_lowdensity = compute_low_density_mask(
            z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
        print('Total selected for density filtering: ', mask1_lowdensity.sum())

        mask1_updated = run_texture_analysis(
            prefix=histosweep_prefix,
            image=image,
            tissue_mask=mask1_lowdensity,
            output_dir=step_paths.HISTOSWEEP_SUBFOLDER,
            patch_size=patch_size,
            glcm_levels=64,
        )

        mask2 = run_ratio_filtering(ratio_norm_, mask1_updated)[0]
        print(mask2.shape)
    else:
        mask_shape = tuple(shapes['tiles'])
        mask1_updated = np.zeros(mask_shape, dtype=bool)
        mask2 = np.zeros(mask_shape, dtype=bool)

    generate_final_mask(
        prefix=histosweep_prefix,
        he=image,
        mask1_updated=mask1_updated,
        mask2=mask2,
        output_dir=step_paths.HISTOSWEEP_SUBFOLDER,
        clean_background=clean_background_flag,
        super_pixel_size=patch_size,
        minSize=min_size,
        masking_method=masking_method,
        victor_mean_threshold=victor_mean_threshold,
        victor_max_iterations=victor_max_iterations,
        victor_sigma=victor_sigma,
        victor_positive_contrast=victor_positive_contrast,
        victor_superpixel_threshold=victor_superpixel_threshold,
    )

    print("Running successfully!")

    # convert mask-small.png into a boolean array, persist it as a pickle
    mask_small_path = os.path.join(histosweep_dir, 'mask-small.png')
    img = Image.open(mask_small_path)
    if show_image:
        plt.imshow(img)

    arr = np.array(img)
    threshold = 128
    mask = arr > threshold  # True for white, False for black

    save_pickle(mask, qc_output)
    with open(qc_signature_output, "w", encoding="utf-8") as f:
        f.write(signature)

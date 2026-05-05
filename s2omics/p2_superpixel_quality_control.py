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
                               show_image=False):
    """Run superpixel QC.

    Reads ``he.*`` from ``save_folder/p1_preprocess/`` and writes outputs
    (shapes/qc pickles, HistoSweep masks and plots) into
    ``save_folder/p2_qc/`` and ``save_folder/p2_qc/HistoSweep_output/``.

    clean_background_flag: Whether to remove small speckles in superpixel mask
    """
    p1_dir = step_paths.step_dir(save_folder, step_paths.P1_PREPROCESS, create=False)
    p2_dir = step_paths.step_dir(save_folder, step_paths.P2_QC)
    histosweep_dir = os.path.join(p2_dir, step_paths.HISTOSWEEP_SUBFOLDER)
    os.makedirs(histosweep_dir, exist_ok=True)

    shapes_output = p2_dir + 'shapes.pickle'
    qc_output = p2_dir + 'qc_preserve_indicator.pickle'
    if os.path.exists(shapes_output) and os.path.exists(qc_output):
        print(
            "Skipping QC; outputs already exist: "
            f"'{shapes_output}', '{qc_output}'."
        )
        return

    image = load_image(get_image_filename(p1_dir + 'he'))
    _, shapes, _ = patchify(image, patch_size)
    save_pickle(shapes, shapes_output)

    he_std_norm_image_, he_std_image_, z_v_norm_image_, z_v_image_, ratio_norm_, ratio_norm_image_ = (
        compute_metrics_memory_optimized(image, patch_size=patch_size)
    )

    # identify low density superpixels
    mask1_lowdensity = compute_low_density_mask(
        z_v_image_, he_std_image_, ratio_norm_, density_thresh=density_thresh)
    print('Total selected for density filtering: ', mask1_lowdensity.sum())

    # HistoSweep functions construct paths as f"{prefix}/{output_dir}/...".
    # Pass the p2 directory as prefix so HistoSweep outputs nest under p2_qc/.
    histosweep_prefix = p2_dir.rstrip('/')

    mask1_lowdensity_update = run_texture_analysis(
        prefix=histosweep_prefix,
        image=image,
        tissue_mask=mask1_lowdensity,
        output_dir=step_paths.HISTOSWEEP_SUBFOLDER,
        patch_size=patch_size,
        glcm_levels=64,
    )

    mask2_lowratio, otsu_thresh = run_ratio_filtering(ratio_norm_, mask1_lowdensity_update)
    print(mask2_lowratio.shape)

    generate_final_mask(
        prefix=histosweep_prefix,
        he=image,
        mask1_updated=mask1_lowdensity_update,
        mask2=mask2_lowratio,
        output_dir=step_paths.HISTOSWEEP_SUBFOLDER,
        clean_background=clean_background_flag,
        super_pixel_size=patch_size,
        minSize=min_size,
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

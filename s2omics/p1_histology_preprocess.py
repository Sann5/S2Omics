import os
from time import time

from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt

from . import step_paths
from .s1_utils import (
        crop_image, load_image, save_image, get_image_filename,
        read_string)


def rescale_image(img, scale):
    if img.ndim == 2:
        scale = [scale, scale]
    elif img.ndim == 3:
        scale = [scale, scale, 1]
    else:
        raise ValueError('Unrecognized image ndim')
    img = rescale(img, scale, preserve_range=True)
    return img


def adjust_margins(img, pad, pad_value=None):
    extent = np.stack([[0, 0], img.shape[:2]]).T
    # make size divisible by pad without changing coords
    remainder = (extent[:, 1] - extent[:, 0]) % pad
    complement = (pad - remainder) % pad
    extent[:, 1] += complement
    if pad_value is None:
        mode = 'edge'
    else:
        mode = 'constant'
    img = crop_image(
            img, extent, mode=mode, constant_values=pad_value)
    return img


def histology_preprocess(save_folder, show_image=False):
    """Rescale and pad the raw H&E image.

    Reads ``he-raw.*`` and ``pixel-size-raw.txt`` from ``save_folder/p0_ndpi_conversion/``,
    writes ``he-scaled.tiff`` and ``he.tiff`` into ``save_folder/p1_preprocess/``.
    """
    p0_dir = step_paths.step_dir(save_folder, step_paths.P0_NDPI_CONVERSION, create=False)
    p1_dir = step_paths.step_dir(save_folder, step_paths.P1_PREPROCESS)

    scaled_output = p1_dir + 'he-scaled.tiff'
    padded_output = p1_dir + 'he.tiff'
    if os.path.exists(scaled_output) and os.path.exists(padded_output):
        print(f"Skipping H&E preprocessing; outputs already exist: '{scaled_output}', '{padded_output}'.")
        return

    pixel_size_raw = float(read_string(p0_dir + 'pixel-size-raw.txt'))
    pixel_size = 0.5
    scale = pixel_size_raw / pixel_size

    img = load_image(get_image_filename(p0_dir + 'he-raw'))
    img = img.astype(np.float32)
    print(f'Rescaling image (scale: {scale:.3f})...')
    t0 = time()
    img = rescale_image(img, scale)
    print(int(time() - t0), 'sec')
    img = img.astype(np.uint8)
    save_image(img, scaled_output)

    pad = 256
    img = adjust_margins(img, pad=pad, pad_value=255)
    save_image(img, padded_output)
    print('Preprocessed H&E image saved!')

    if show_image:
        plt.imshow(img)

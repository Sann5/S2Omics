import numpy as np
from skimage.filters import gaussian, threshold_otsu


def _to_normalized_grayscale(image):
    """Convert RGB/RGBA/grayscale image data to float grayscale in [0, 1]."""
    arr = np.asarray(image)
    source_dtype = arr.dtype

    if arr.ndim == 2:
        gray = arr.astype(np.float32)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        gray = arr[..., 0].astype(np.float32)
    elif arr.ndim == 3:
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    else:
        raise ValueError(f"Expected 2D grayscale or 3D RGB image, got shape {arr.shape}.")

    if np.issubdtype(source_dtype, np.integer):
        max_value = float(np.iinfo(source_dtype).max)
    else:
        finite_max = float(np.nanmax(gray)) if gray.size else 1.0
        max_value = 255.0 if finite_max > 1.0 and finite_max <= 255.0 else finite_max

    if max_value > 0:
        gray = gray / max_value

    return np.clip(gray, 0.0, 1.0)


def smooth_and_threshold(
    image,
    mean_threshold=0.85,
    max_iterations=5,
    sigma=20,
    positive_contrast=False,
):
    """Victor's grayscale -> Gaussian smoothing -> Otsu tissue mask.

    ``positive_contrast=False`` is the expected setting for standard H&E images,
    where tissue is darker than the bright slide background. Set it to ``True``
    only for images where tissue is brighter than the background.

    ``mean_threshold`` is interpreted against the estimated background:
    background should be brighter than this threshold for standard H&E and
    darker than this threshold for positive-contrast images.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1.")
    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    gray = _to_normalized_grayscale(image)
    smoothed_image = gray
    threshold_value = None
    tissue_mask = np.ones(gray.shape, dtype=bool)

    for _ in range(max_iterations):
        smoothed_image = gaussian(smoothed_image, sigma=sigma, preserve_range=True)
        if np.isclose(float(np.min(smoothed_image)), float(np.max(smoothed_image))):
            threshold_value = float(mean_threshold)
            bright_mask = smoothed_image > mean_threshold
            tissue_mask = bright_mask if positive_contrast else ~bright_mask
            break

        threshold_value = threshold_otsu(smoothed_image)
        bright_mask = smoothed_image > threshold_value
        tissue_mask = bright_mask if positive_contrast else ~bright_mask

        background_values = gray[~tissue_mask]
        if not background_values.size:
            break

        background_mean = float(np.mean(background_values))
        if positive_contrast:
            background_is_separated = background_mean < mean_threshold
        else:
            background_is_separated = background_mean > mean_threshold
        if background_is_separated:
            break

    return tissue_mask.astype(bool), threshold_value


def compute_victor_superpixel_mask(
    he,
    super_pixel_size=16,
    mean_threshold=0.85,
    max_iterations=5,
    sigma=20,
    positive_contrast=False,
    superpixel_threshold=0.5,
):
    """Return a superpixel-level tissue mask using Victor's Otsu workflow."""
    if not 0.0 <= superpixel_threshold <= 1.0:
        raise ValueError("superpixel_threshold must be between 0 and 1.")

    image_height, image_width = he.shape[:2]
    num_super_pixels_y = image_height // super_pixel_size
    num_super_pixels_x = image_width // super_pixel_size

    if num_super_pixels_y == 0 or num_super_pixels_x == 0:
        raise ValueError(
            "Image is smaller than one superpixel; "
            f"image shape={he.shape}, super_pixel_size={super_pixel_size}."
        )

    he_crop = he[
        : num_super_pixels_y * super_pixel_size,
        : num_super_pixels_x * super_pixel_size,
        ...,
    ]
    tissue_mask, threshold_value = smooth_and_threshold(
        he_crop,
        mean_threshold=mean_threshold,
        max_iterations=max_iterations,
        sigma=sigma,
        positive_contrast=positive_contrast,
    )

    tissue_fraction = tissue_mask.reshape(
        num_super_pixels_y,
        super_pixel_size,
        num_super_pixels_x,
        super_pixel_size,
    ).mean(axis=(1, 3))

    return tissue_fraction >= superpixel_threshold, threshold_value

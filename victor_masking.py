import numpy as np

from s2omics.HistoSweep.victorMasking import smooth_and_threshold as _smooth_and_threshold


def smooth_and_threshold(
    image: np.array,
    mean_threshold: float,
    max_iterations: int = 5,
    sigma: int = 20,
    positive_contrast: bool = True,
):
    """Compatibility wrapper for Victor's integrated masking implementation."""
    mask, threshold_value = _smooth_and_threshold(
        image=image,
        mean_threshold=mean_threshold,
        max_iterations=max_iterations,
        sigma=sigma,
        positive_contrast=positive_contrast,
    )
    return mask.astype(int), threshold_value

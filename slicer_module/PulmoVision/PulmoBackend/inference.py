"""
PulmoVision Backend - Inference (Placeholder)

This module currently implements simple placeholder segmentation functions.
Later, this will be replaced or augmented by a 3D U-Net–based inference pipeline.

All functions operate on NumPy arrays, typically preprocessed volumes with
values in [0, 1].
"""

import numpy as np


def percentile_threshold_segmentation(volume, percentile=99.0):
    """
    Segment a volume by thresholding at a given intensity percentile.

    Intended as a placeholder for a real tumor segmentation model.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed volume, dtype float32, shape (H, W, D).
        Values are assumed to be in a finite numeric range (e.g., [0, 1]).
    percentile : float, optional
        Percentile (0–100) to use as the threshold. Voxels >= this value
        are labeled as 1, others as 0.

    Returns
    -------
    mask : np.ndarray
        Binary mask, same shape as input, dtype uint8 (0 or 1).
    """
    vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError("percentile_threshold_segmentation expects a 3D volume (H, W, D)")

    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100]")

    thresh = float(np.percentile(vol, percentile))
    mask = (vol >= thresh).astype(np.uint8)
    return mask


def run_placeholder_segmentation(volume,
                                 method="percentile",
                                 **kwargs):
    """
    Entry point for placeholder segmentation.

    Parameters
    ----------
    volume : np.ndarray
        Preprocessed CT volume (typically output of preprocess_ct),
        expected shape (H, W, D), dtype float32.
    method : str, optional
        Placeholder segmentation method. Currently supported:
        - "percentile": uses percentile_threshold_segmentation
    **kwargs :
        Additional parameters forwarded to the underlying method.
        For method == "percentile", you can pass `percentile=...`.

    Returns
    -------
    mask : np.ndarray
        Binary segmentation mask, same shape as input, dtype uint8.
    """
    if method == "percentile":
        return percentile_threshold_segmentation(volume, **kwargs)
    else:
        raise ValueError(f"Unsupported placeholder segmentation method: {method!r}")


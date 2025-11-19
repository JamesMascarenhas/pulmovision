"""
PulmoVision Backend - Postprocessing

This module provides simple postprocessing for segmentation masks.
In v1, this mainly enforces a clean binary mask. More advanced steps
(e.g., largest-component selection) can be added later.
"""

import numpy as np


def ensure_binary(mask, threshold=0.5):
    """
    Convert a mask to a clean binary array of 0/1.

    Parameters
    ----------
    mask : np.ndarray
        Input mask, any numeric dtype, shape (H, W, D).
    threshold : float, optional
        Threshold used if the mask is not already binary. Values >= threshold
        become 1, others become 0.

    Returns
    -------
    bin_mask : np.ndarray
        Binary mask, same shape as input, dtype uint8, values in {0, 1}.
    """
    arr = np.asarray(mask)

    if arr.ndim != 3:
        raise ValueError("ensure_binary expects a 3D volume (H, W, D)")

    # If already 0/1 or 0/255 integer mask, normalize to {0,1}
    unique_vals = np.unique(arr)
    if unique_vals.size <= 2:
        # Treat anything >0 as 1
        bin_mask = (arr > 0).astype(np.uint8)
        return bin_mask

    # Otherwise apply a numeric threshold
    bin_mask = (arr.astype(np.float32) >= float(threshold)).astype(np.uint8)
    return bin_mask


def postprocess_mask(mask,
                     keep_largest_component=False,
                     min_size_voxels=0):
    """
    Basic postprocessing for a segmentation mask.

    Currently:
    - Ensures binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Input mask, shape (H, W, D).
    keep_largest_component : bool, optional
        Placeholder flag for future functionality. Currently ignored.
    min_size_voxels : int, optional
        Placeholder for minimum component size. Currently ignored.

    Returns
    -------
    clean_mask : np.ndarray
        Binary mask, same shape as input, dtype uint8.
    """
    clean_mask = ensure_binary(mask)

    # NOTE: v1 ignores keep_largest_component and min_size_voxels.
    # These can be implemented later with a 3D connected-component analysis.

    return clean_mask


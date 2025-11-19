"""
PulmoVision Backend - Preprocessing

This module provides basic preprocessing utilities for CT lung volumes:
- CT windowing (center/width)
- Intensity normalization

All functions operate on NumPy arrays, typically 3D volumes of shape (H, W, D).
"""

import numpy as np


def window_ct(volume, window_center, window_width):
    """
    Apply CT windowing to a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        Input CT volume as a NumPy array. Expected shape (H, W, D).
        Values are assumed to be in HU (or similar).
    window_center : float
        Window center (level).
    window_width : float
        Window width.

    Returns
    -------
    windowed : np.ndarray
        Windowed volume, same shape as input, dtype float32.
        Values are clipped to [center - width/2, center + width/2].
    """
    vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError("window_ct expects a 3D volume (H, W, D)")

    if window_width <= 0:
        raise ValueError("window_width must be > 0")

    low = window_center - (window_width / 2.0)
    high = window_center + (window_width / 2.0)

    windowed = np.clip(vol, low, high)
    return windowed.astype(np.float32)


def normalize_minmax(volume, in_min=None, in_max=None,
                     out_min=0.0, out_max=1.0):
    """
    Min–max normalize a volume to a desired output range.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (e.g., result of window_ct). Any numeric dtype, shape (H, W, D).
    in_min : float, optional
        Minimum intensity for normalization. If None, uses volume.min().
    in_max : float, optional
        Maximum intensity for normalization. If None, uses volume.max().
    out_min : float, optional
        Lower bound of output range (default 0.0).
    out_max : float, optional
        Upper bound of output range (default 1.0).

    Returns
    -------
    norm_vol : np.ndarray
        Normalized volume as float32, with values in [out_min, out_max].
        If in_max == in_min, returns a constant volume filled with out_min.
    """
    vol = np.asarray(volume, dtype=np.float32)

    if vol.ndim != 3:
        raise ValueError("normalize_minmax expects a 3D volume (H, W, D)")

    if in_min is None:
        in_min = float(vol.min())
    if in_max is None:
        in_max = float(vol.max())

    if in_max <= in_min:
        # Degenerate case: no contrast; avoid divide-by-zero
        return np.full_like(vol, fill_value=out_min, dtype=np.float32)

    scale = (out_max - out_min) / (in_max - in_min)
    norm_vol = (vol - in_min) * scale + out_min
    return norm_vol.astype(np.float32)


def preprocess_ct(volume,
                  window_center=None,
                  window_width=None,
                  normalize=True,
                  norm_out_min=0.0,
                  norm_out_max=1.0):
    """
    Convenience function to preprocess a CT volume.

    Steps:
    1. (Optional) Apply CT windowing using center/width.
    2. (Optional) Min–max normalize to [norm_out_min, norm_out_max].

    Parameters
    ----------
    volume : np.ndarray
        Input CT volume, expected shape (H, W, D).
    window_center : float, optional
        Window center (level). If None, no windowing is applied.
    window_width : float, optional
        Window width. Must be provided if window_center is not None.
    normalize : bool, optional
        If True, apply min–max normalization after windowing.
    norm_out_min : float, optional
        Lower bound of normalization range (default 0.0).
    norm_out_max : float, optional
        Upper bound of normalization range (default 1.0).

    Returns
    -------
    vol_pre : np.ndarray
        Preprocessed volume, dtype float32, shape (H, W, D).
    """
    vol = np.asarray(volume)

    if vol.ndim != 3:
        raise ValueError("preprocess_ct expects a 3D volume (H, W, D)")

    vol_pre = vol.astype(np.float32)

    # Step 1: windowing (if requested)
    if window_center is not None:
        if window_width is None:
            raise ValueError("window_width must be provided when window_center is not None")
        vol_pre = window_ct(vol_pre, window_center, window_width)

    # Step 2: normalization (if requested)
    if normalize:
        vol_pre = normalize_minmax(vol_pre,
                                   in_min=None,
                                   in_max=None,
                                   out_min=norm_out_min,
                                   out_max=norm_out_max)

    return vol_pre


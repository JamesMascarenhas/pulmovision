"""Lightweight radiomics-style feature extraction for PulmoVision.

The functions in this module rely only on NumPy to keep dependencies minimal
while still providing a handful of commonly used descriptors:
- Volume/voxel counts and surface area (based on exposed faces).
- First-order HU statistics within the mask.
- Simple texture metrics computed from a small gray-level co-occurrence matrix.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

ArrayLike = np.ndarray


def _surface_area(mask: ArrayLike, spacing: Tuple[float, float, float]) -> float:
    """Estimate surface area of a binary mask using exposed voxel faces.

    Each voxel contributes a face whenever it borders background along any axis.
    The surface area is the sum of these exposed faces scaled by in-plane spacings.
    """

    mask_bool = mask.astype(bool)
    if mask_bool.size == 0:
        return 0.0

    # Pad with zeros so edges are treated as boundaries
    padded = np.pad(mask_bool, 1, mode="constant", constant_values=False)

    area = 0.0
    face_spacings = [
        spacing[1] * spacing[2],  # faces orthogonal to H axis
        spacing[0] * spacing[2],  # faces orthogonal to W axis
        spacing[0] * spacing[1],  # faces orthogonal to D axis
    ]

    for axis, face_area in enumerate(face_spacings):
        diff = np.abs(np.diff(padded, axis=axis))
        area += face_area * diff.sum()

    return float(area)


def _quantize(values: ArrayLike, bins: int) -> Tuple[ArrayLike, float, float]:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        # Avoid zero range; keep bin index centered
        vmax = vmin + 1.0
    scaled = (values - vmin) / (vmax - vmin)
    quantized = np.clip(np.floor(scaled * (bins - 1)), 0, bins - 1).astype(np.int32)
    return quantized, vmin, vmax


def _glcm_features(masked_volume: ArrayLike, bins: int = 8, offsets: Iterable[Tuple[int, int, int]] = ((1, 0, 0),)) -> Dict[str, float]:
    """Compute simple GLCM-based texture features (contrast, energy).

    Parameters
    ----------
    masked_volume : np.ndarray
        1D array of intensity values inside the mask.
    bins : int
        Number of gray levels to quantize the intensities to.
    offsets : iterable of 3-tuples
        Offsets in (H, W, D) indexing used to build co-occurrence counts.
    """

    if masked_volume.size < 2:
        return {"GLCM Contrast": float("nan"), "GLCM Energy": float("nan")}

    quantized, vmin, vmax = _quantize(masked_volume, bins)

    # Reshape back into an arbitrary cube using a simple heuristic
    # (layout only matters for adjacency counting).
    length = masked_volume.size
    cube_dim = int(round(length ** (1.0 / 3))) or 1
    padded_size = cube_dim ** 3
    if padded_size < length:
        cube_dim += 1
        padded_size = cube_dim ** 3

    data = np.full(padded_size, -1, dtype=np.int32)
    data[:length] = quantized
    cube = data.reshape((cube_dim, cube_dim, cube_dim))

    glcm = np.zeros((bins, bins), dtype=np.int64)

    for offset in offsets:
        dh, dw, dd = offset
        src = cube
        dst = cube
        if dh > 0:
            src = src[:-dh, :, :]
            dst = dst[dh:, :, :]
        if dw > 0:
            src = src[:, :-dw, :]
            dst = dst[:, dw:, :]
        if dd > 0:
            src = src[:, :, :-dd]
            dst = dst[:, :, dd:]

        valid = (src >= 0) & (dst >= 0)
        if not np.any(valid):
            continue
        src_vals = src[valid]
        dst_vals = dst[valid]
        np.add.at(glcm, (src_vals, dst_vals), 1)

    total = glcm.sum()
    if total == 0:
        return {"GLCM Contrast": float("nan"), "GLCM Energy": float("nan")}

    probs = glcm / float(total)
    i_idx, j_idx = np.indices(glcm.shape)
    contrast = float(np.sum(((i_idx - j_idx) ** 2) * probs))
    energy = float(np.sum(probs**2))

    return {"GLCM Contrast": contrast, "GLCM Energy": energy}


def compute_radiomics_features(
    volume_hwd: ArrayLike,
    mask_hwd: ArrayLike,
    *,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    texture_bins: int = 8,
    texture_offsets: Iterable[Tuple[int, int, int]] = ((1, 0, 0),),
) -> Dict[str, float]:
    """Compute a small collection of radiomics-style features.

    Parameters
    ----------
    volume_hwd : np.ndarray
        CT volume in (H, W, D) layout.
    mask_hwd : np.ndarray
        Binary mask aligned with ``volume_hwd``.
    voxel_spacing : tuple of float, optional
        Physical spacing (H, W, D). Defaults to isotropic 1 mm.
    texture_bins : int, optional
        Number of bins for GLCM quantization.
    texture_offsets : iterable of tuple, optional
        Neighbor offsets for the GLCM calculation.
    """

    vol = np.asarray(volume_hwd)
    mask = np.asarray(mask_hwd)

    if vol.shape != mask.shape:
        raise ValueError(
            f"Volume and mask shapes must match; got {vol.shape} and {mask.shape}"
        )

    mask_bool = mask.astype(bool)
    voxel_spacing = tuple(float(s) for s in voxel_spacing)

    voxel_count = int(mask_bool.sum())
    voxel_volume_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])
    volume_mm3 = voxel_volume_mm3 * voxel_count
    volume_ml = volume_mm3 / 1000.0
    surface_area = _surface_area(mask_bool, voxel_spacing)

    if voxel_count > 0:
        masked_values = vol[mask_bool]
        mean_hu = float(masked_values.mean())
        std_hu = float(masked_values.std())
        max_hu = float(masked_values.max())
        min_hu = float(masked_values.min())
        median_hu = float(np.median(masked_values))
        texture = _glcm_features(masked_values, bins=texture_bins, offsets=texture_offsets)
    else:
        mean_hu = std_hu = max_hu = min_hu = median_hu = float("nan")
        texture = {"GLCM Contrast": float("nan"), "GLCM Energy": float("nan")}

    features: Dict[str, float] = {
        "Voxels": voxel_count,
        "Volume (mm^3)": volume_mm3,
        "Volume (mL)": volume_ml,
        "Surface Area (mm^2)": surface_area,
        "Mean HU": mean_hu,
        "Std HU": std_hu,
        "Min HU": min_hu,
        "Max HU": max_hu,
        "Median HU": median_hu,
    }
    features.update(texture)

    return features
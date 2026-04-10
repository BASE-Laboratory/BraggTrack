"""Post-processing helpers for labeled 3D masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes, label


def remove_small_objects(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Remove labeled regions with fewer than *min_size* voxels."""
    labels = np.asarray(labels)
    out = labels.copy()
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl <= 0:
            continue
        if np.count_nonzero(labels == lbl) < min_size:
            out[out == lbl] = 0
    return out


def fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes in a binary 3D mask.

    A 1-voxel ``False`` border is added before flood-filling so that
    foreground objects touching the volume boundary do not confuse the
    background connectivity.
    """
    mask = np.asarray(mask, dtype=bool)
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    filled = binary_fill_holes(padded)
    return filled[1:-1, 1:-1, 1:-1]


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Remap positive labels to consecutive integers starting at 1."""
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for new_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        out[labels == old_id] = new_id
    return out

"""Post-processing helpers for labeled 3D masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes


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


def merge_nearby_labels(
    labels: np.ndarray,
    volume: np.ndarray,
    max_centroid_distance: float,
) -> np.ndarray:
    """Merge adjacent labeled regions whose intensity-weighted centroids are close.

    Two labels are candidates for merging when they are spatially adjacent
    (share a face via 6-connectivity dilation) **and** the Euclidean distance
    between their intensity-weighted centroids is below *max_centroid_distance*.
    Merging is greedy — closest pairs first — and iterates until no more
    merges are possible.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    labels = np.asarray(labels, dtype=np.int32).copy()
    volume = np.asarray(volume, dtype=np.float64)
    struct = generate_binary_structure(3, 1)  # 6-connectivity

    changed = True
    while changed:
        changed = False
        unique_ids = [i for i in np.unique(labels) if i > 0]
        if len(unique_ids) < 2:
            break

        centroids: dict[int, np.ndarray] = {}
        for lid in unique_ids:
            mask = labels == lid
            coords = np.argwhere(mask)
            weights = volume[mask]
            total = weights.sum()
            if total > 0:
                centroids[lid] = (coords * weights[:, None]).sum(axis=0) / total
            else:
                centroids[lid] = coords.mean(axis=0)

        merge_pairs: list[tuple[float, int, int]] = []
        for lid in unique_ids:
            dilated = binary_dilation(labels == lid, structure=struct)
            neighbor_ids = set(np.unique(labels[dilated])) - {0, lid}
            for nid in neighbor_ids:
                if nid < lid:
                    continue
                dist = float(np.linalg.norm(centroids[lid] - centroids[nid]))
                if dist < max_centroid_distance:
                    merge_pairs.append((dist, lid, nid))

        merge_pairs.sort()
        merged_this_round: set[int] = set()
        for _, a, b in merge_pairs:
            if a in merged_this_round or b in merged_this_round:
                continue
            labels[labels == b] = a
            merged_this_round.add(b)
            changed = True

    return labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Remap positive labels to consecutive integers starting at 1."""
    labels = np.asarray(labels)
    out = np.zeros_like(labels)
    for new_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        out[labels == old_id] = new_id
    return out

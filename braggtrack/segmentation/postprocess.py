"""Post-processing helpers for labeled 3D masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes


def remove_small_objects(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Remove labeled regions with fewer than *min_size* voxels."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.copy()
    max_lbl = int(labels.max())
    if max_lbl <= 0:
        return np.zeros_like(labels)
    clipped = np.where(labels > 0, labels, 0)
    counts = np.bincount(clipped.ravel(), minlength=max_lbl + 1)
    keep = counts >= min_size
    keep[0] = False
    return np.where(keep[clipped], labels, 0)


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


def _six_connected_label_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Return sorted ``(a, b)`` pairs (`0 < a < b`) sharing a 6-connected face."""
    pairs: set[tuple[int, int]] = set()
    for sl_a, sl_b in (
        (labels[:-1, :, :], labels[1:, :, :]),
        (labels[:, :-1, :], labels[:, 1:, :]),
        (labels[:, :, :-1], labels[:, :, 1:]),
    ):
        diff = (sl_a != sl_b) & (sl_a > 0) & (sl_b > 0)
        if not np.any(diff):
            continue
        a_vals = sl_a[diff]
        b_vals = sl_b[diff]
        lo = np.minimum(a_vals, b_vals)
        hi = np.maximum(a_vals, b_vals)
        uniq = np.unique(np.stack([lo, hi], axis=1), axis=0)
        for a_id, b_id in uniq:
            pairs.add((int(a_id), int(b_id)))
    return pairs


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
    labels = np.asarray(labels, dtype=np.int32).copy()
    volume = np.asarray(volume, dtype=np.float64)
    if labels.size == 0 or int(labels.max()) < 2:
        return labels

    fg_coords = np.argwhere(labels > 0)
    if len(fg_coords) < 2:
        return labels
    z_fg = fg_coords[:, 0].astype(np.float64)
    y_fg = fg_coords[:, 1].astype(np.float64)
    x_fg = fg_coords[:, 2].astype(np.float64)
    w_fg = volume[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]]

    changed = True
    while changed:
        changed = False
        cur_fg = labels[fg_coords[:, 0], fg_coords[:, 1], fg_coords[:, 2]]
        unique_ids = [int(i) for i in np.unique(cur_fg) if i > 0]
        if len(unique_ids) < 2:
            break

        adj_pairs = _six_connected_label_pairs(labels)
        if not adj_pairs:
            break

        max_id = max(unique_ids)
        counts = np.bincount(cur_fg, minlength=max_id + 1)
        w_sum = np.bincount(cur_fg, weights=w_fg, minlength=max_id + 1)
        wz_sum = np.bincount(cur_fg, weights=w_fg * z_fg, minlength=max_id + 1)
        wy_sum = np.bincount(cur_fg, weights=w_fg * y_fg, minlength=max_id + 1)
        wx_sum = np.bincount(cur_fg, weights=w_fg * x_fg, minlength=max_id + 1)
        gz_sum = np.bincount(cur_fg, weights=z_fg, minlength=max_id + 1)
        gy_sum = np.bincount(cur_fg, weights=y_fg, minlength=max_id + 1)
        gx_sum = np.bincount(cur_fg, weights=x_fg, minlength=max_id + 1)

        centroids: dict[int, np.ndarray] = {}
        for lid in unique_ids:
            tot = float(w_sum[lid])
            if tot > 0.0:
                centroids[lid] = np.array([wz_sum[lid] / tot, wy_sum[lid] / tot, wx_sum[lid] / tot], dtype=np.float64)
            else:
                cnt = max(1, int(counts[lid]))
                centroids[lid] = np.array([gz_sum[lid] / cnt, gy_sum[lid] / cnt, gx_sum[lid] / cnt], dtype=np.float64)

        merge_pairs: list[tuple[float, int, int]] = []
        for lid, nid in sorted(adj_pairs):
            dist = float(np.linalg.norm(centroids[lid] - centroids[nid]))
            if dist < max_centroid_distance:
                merge_pairs.append((dist, lid, nid))

        merge_pairs.sort()
        merged_this_round: set[int] = set()
        remap = np.arange(max_id + 1, dtype=np.int32)
        for _, a, b in merge_pairs:
            if a in merged_this_round or b in merged_this_round:
                continue
            remap[b] = a
            merged_this_round.add(b)
            changed = True

        if changed:
            labels = np.where(labels > 0, remap[np.maximum(labels, 0)], 0)

    return labels


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    """Remap positive labels to consecutive integers starting at 1."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros_like(labels)
    max_lbl = int(labels.max())
    if max_lbl <= 0:
        return np.zeros_like(labels)
    pos = np.unique(labels[labels > 0])
    remap = np.zeros(max_lbl + 1, dtype=labels.dtype)
    remap[pos] = np.arange(1, len(pos) + 1, dtype=labels.dtype)
    return np.where(labels > 0, remap[np.maximum(labels, 0)], 0)

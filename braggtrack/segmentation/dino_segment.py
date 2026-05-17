"""DINO-based 3D segmentation: patch feature clustering + 3D stitching.

Replaces the hand-tuned LoG + watershed pipeline with frozen DINOv3 patch
features clustered via HDBSCAN, producing instance labels that generalise
across beamlines and detectors without per-instrument parameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DinoSegmentationResult:
    """Output of :func:`segment_dino`, field-compatible with ClassicalSegmentationResult."""

    threshold: float
    seed_count: int
    component_count: int
    labeled_volume: np.ndarray
    response: np.ndarray


def _extract_slice_features(
    volume: np.ndarray,
    encoder: object,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Extract patch features for each slice along *axis*.

    Returns ``(features, slice_hw)`` where *features* has shape
    ``(n_slices, H_patches, W_patches, D)`` and *slice_hw* is the
    pixel-level ``(H, W)`` of each slice.
    """
    n_slices = volume.shape[axis]
    feature_maps: list[np.ndarray] = []
    for i in range(n_slices):
        slc = np.take(volume, i, axis=axis)
        fmap = encoder.extract_patch_features(slc)  # type: ignore[union-attr]
        feature_maps.append(fmap)

    slice_hw = (volume.shape[1] if axis == 0 else volume.shape[0], volume.shape[2] if axis != 2 else volume.shape[1])
    return np.stack(feature_maps, axis=0), slice_hw


def _cluster_feature_map(
    features: np.ndarray,
    *,
    n_components_pca: int = 16,
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> np.ndarray:
    """Cluster a 2D feature map ``(H_p, W_p, D)`` into instance labels.

    Uses PCA dimensionality reduction followed by HDBSCAN.
    Returns ``(H_p, W_p)`` int array, 0 = background/noise.
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    h_p, w_p, d = features.shape
    n_patches = h_p * w_p
    flat = features.reshape(-1, d)

    n_comp = min(n_components_pca, d, n_patches)
    if n_comp < 2:
        # Too few patches to cluster — assign all to a single region.
        return np.ones((h_p, w_p), dtype=np.int32)

    reduced = PCA(n_components=n_comp).fit_transform(flat)

    effective_min_cluster = max(2, min(min_cluster_size, n_patches // 2))
    clusterer = HDBSCAN(
        min_cluster_size=effective_min_cluster,
        min_samples=max(1, min(min_samples, effective_min_cluster - 1)),
    )
    raw_labels = clusterer.fit_predict(reduced)
    # HDBSCAN labels: -1 = noise, 0..K = clusters. Shift to 1-based.
    labels = np.where(raw_labels >= 0, raw_labels + 1, 0)

    # If HDBSCAN assigned everything to noise, treat all patches as one region.
    if not np.any(labels > 0):
        return np.ones((h_p, w_p), dtype=np.int32)

    return labels.reshape(h_p, w_p).astype(np.int32)


def _upsample_labels(
    patch_labels: np.ndarray,
    target_shape: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """Nearest-neighbor upsample patch-resolution labels to pixel resolution."""
    h_p, w_p = patch_labels.shape
    out = np.zeros(target_shape, dtype=np.int32)
    for py in range(h_p):
        for px in range(w_p):
            y0 = py * patch_size
            x0 = px * patch_size
            y1 = min(y0 + patch_size, target_shape[0])
            x1 = min(x0 + patch_size, target_shape[1])
            out[y0:y1, x0:x1] = patch_labels[py, px]
    return out


def _stitch_slices_3d(
    per_slice_labels: list[np.ndarray],
    *,
    min_overlap_fraction: float = 0.3,
) -> np.ndarray:
    """Stitch 2D per-slice labels into a consistent 3D label volume.

    Two labels on adjacent slices are merged when their spatial overlap
    exceeds *min_overlap_fraction* of the smaller region.  Uses a
    union-find for global relabelling.
    """
    if not per_slice_labels:
        return np.zeros((0, 0, 0), dtype=np.int32)

    h, w = per_slice_labels[0].shape
    n_slices = len(per_slice_labels)

    # Assign globally unique label offsets per slice.
    offset = 0
    global_slices: list[np.ndarray] = []
    for sl in per_slice_labels:
        shifted = np.where(sl > 0, sl + offset, 0)
        global_slices.append(shifted.astype(np.int32))
        mx = int(sl.max())
        offset += mx

    total_labels = offset + 1

    # Union-find.
    parent = list(range(total_labels))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Merge labels across adjacent slices by overlap.
    for i in range(n_slices - 1):
        sl_a = global_slices[i]
        sl_b = global_slices[i + 1]
        pairs_a = sl_a.ravel()
        pairs_b = sl_b.ravel()
        # Only look at pixels where both slices have a label.
        mask = (pairs_a > 0) & (pairs_b > 0)
        if not mask.any():
            continue

        la = pairs_a[mask]
        lb = pairs_b[mask]
        unique_pairs, counts = np.unique(np.stack([la, lb], axis=1), axis=0, return_counts=True)
        for (lid_a, lid_b), cnt in zip(unique_pairs, counts):
            size_a = int(np.count_nonzero(sl_a == lid_a))
            size_b = int(np.count_nonzero(sl_b == lid_b))
            min_size = min(size_a, size_b)
            if min_size > 0 and cnt / min_size >= min_overlap_fraction:
                union(int(lid_a), int(lid_b))

    # Flatten union-find and relabel sequentially.
    volume_3d = np.stack(global_slices, axis=0)
    flat = volume_3d.ravel()
    root_map = np.zeros(total_labels, dtype=np.int32)
    for lbl in range(total_labels):
        root_map[lbl] = find(lbl)

    flat = root_map[flat]
    # Relabel to sequential.
    unique_roots = np.unique(flat[flat > 0])
    new_map = np.zeros(total_labels, dtype=np.int32)
    for new_id, old_root in enumerate(unique_roots, start=1):
        new_map[old_root] = new_id

    flat = new_map[flat]
    return flat.reshape(n_slices, h, w).astype(np.int32)


def segment_dino(
    volume: np.ndarray,
    *,
    backend: str | None = None,
    model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    torch_device: str | None = None,
    n_components_pca: int = 16,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    threshold_fraction: float = 1.0,
    min_overlap_fraction: float = 0.3,
    axis: int = 0,
) -> DinoSegmentationResult:
    """Segment a 3D volume using DINOv3 patch-level features + HDBSCAN.

    Parameters
    ----------
    volume
        Raw 3-D intensity cube (z, y, x), typically float64.
    backend
        DINO backend: ``"auto"``, ``"mock"``, or ``"torch"``.
    model_name
        HuggingFace model ID for the torch backend.
    n_components_pca
        Number of PCA components for dimensionality reduction.
    min_cluster_size, min_samples
        HDBSCAN density parameters.
    threshold_fraction
        Multiply Otsu threshold by this for foreground masking.
    min_overlap_fraction
        Minimum overlap for stitching 2D labels across slices.
    axis
        Axis to slice along (0 = mu/z, typically the narrowest).
    """
    from braggtrack.segmentation.otsu import otsu_threshold
    from braggtrack.semantic.dino import make_patch_encoder

    volume = np.asarray(volume, dtype=np.float64)
    encoder = make_patch_encoder(backend, model_name=model_name, torch_device=torch_device)

    raw_threshold = otsu_threshold(volume.ravel())
    threshold = raw_threshold * threshold_fraction

    features, slice_hw = _extract_slice_features(volume, encoder, axis=axis)

    per_slice_labels: list[np.ndarray] = []
    for i in range(features.shape[0]):
        patch_labels = _cluster_feature_map(
            features[i],
            n_components_pca=n_components_pca,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        full_labels = _upsample_labels(patch_labels, slice_hw, encoder.patch_size)
        per_slice_labels.append(full_labels)

    labels_3d = _stitch_slices_3d(per_slice_labels, min_overlap_fraction=min_overlap_fraction)

    # Apply foreground mask from raw intensity.
    foreground = volume >= threshold
    labels_3d = np.where(foreground, labels_3d, 0).astype(np.int32)

    component_count = len(np.unique(labels_3d[labels_3d > 0]))
    return DinoSegmentationResult(
        threshold=threshold,
        seed_count=component_count,
        component_count=component_count,
        labeled_volume=labels_3d,
        response=np.zeros_like(volume),
    )

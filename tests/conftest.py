"""Shared test fixtures for BraggTrack."""

from __future__ import annotations


def make_spot(
    mu: float,
    chi: float,
    d: float,
    intensity: float = 100.0,
    voxels: int = 10,
    eig: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> dict:
    """Create a synthetic spot dictionary for testing."""
    return {
        "label": 1,
        "voxel_count": voxels,
        "integrated_intensity": intensity,
        "centroid_mu": mu,
        "centroid_chi": chi,
        "centroid_d": d,
        "eig_1": eig[0],
        "eig_2": eig[1],
        "eig_3": eig[2],
    }

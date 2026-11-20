"""Tests for BraggTrack Layer 2 `basecoat` volume/OME-Zarr delegation and import boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import basecoat
import numpy as np
from basecoat.layers import check_layer_imports

from braggtrack.io import (
    AxisCalibration,
    LazyVolume,
    export_ome_zarr_volume,
    extract_axis_calibration,
    load_primary_volume,
    open_lazy_volume,
    otsu_threshold_chunked,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBasecoatLayer2Migration(unittest.TestCase):
    def test_layer2_import_boundaries(self) -> None:
        violations = check_layer_imports(REPO_ROOT)
        self.assertEqual(violations, [])

    def test_axis_calibration_basecoat_roundtrip_including_negative_steps(self) -> None:
        cal = AxisCalibration(
            mu_origin=12.5,
            mu_step=-0.02,
            chi_origin=-1.4,
            chi_step=0.05,
            d_origin=4.71,
            d_step=0.0005,
            calibrated=True,
        )
        axes = cal.to_basecoat_axes()
        self.assertEqual(tuple(ax.name for ax in axes), ("mu", "d", "chi"))
        self.assertEqual(tuple(ax.unit for ax in axes), ("deg", "angstrom", "deg"))
        recovered = AxisCalibration.from_basecoat_axes(axes)
        self.assertEqual(recovered, cal)

    def test_ome_zarr_volume_and_spot_labels_export_and_lazy_streaming(self) -> None:
        rng = np.random.default_rng(7)
        cube = rng.poisson(3.0, size=(16, 20, 24)).astype(np.float32)
        cube[4:8, 6:10, 8:14] += 250.0
        cube[10:14, 12:16, 14:18] += 180.0

        labels = np.zeros_like(cube, dtype=np.int32)
        labels[4:8, 6:10, 8:14] = 1
        labels[10:14, 12:16, 14:18] = 2

        cal = AxisCalibration(
            mu_origin=10.0,
            mu_step=0.01,
            chi_origin=-2.0,
            chi_step=0.02,
            d_origin=2.035,
            d_step=0.0004,
            calibrated=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp) / "dfxm_scan.ome.zarr"
            out = export_ome_zarr_volume(
                cube,
                store,
                spot_labels=labels,
                label_names={0: "background", 1: "grain_A", 2: "grain_B"},
                calibration=cal,
                multiscale_levels=2,
            )
            self.assertEqual(out, store)

            vset_l1 = basecoat.read_ome_zarr(store, level=1, as_set=True)
            self.assertIn("spots", vset_l1.labels)
            self.assertTrue(set(np.unique(vset_l1.labels["spots"].data)).issubset({0, 1, 2}))

            ext_cal = extract_axis_calibration(store)
            self.assertEqual(ext_cal, cal)

            dense = load_primary_volume(store)
            np.testing.assert_allclose(dense, cube, rtol=1e-6, atol=1e-6)

            with open_lazy_volume(store) as lazy:
                self.assertIsInstance(lazy, LazyVolume)
                self.assertEqual(lazy.shape, (16, 20, 24))
                roi = lazy.read_roi(slice(4, 8), slice(6, 10), slice(8, 14))
                np.testing.assert_allclose(roi, cube[4:8, 6:10, 8:14], rtol=1e-6)
                thresh_lazy = otsu_threshold_chunked(lazy, chunk_z=4)
                thresh_arr = otsu_threshold_chunked(cube, chunk_z=4)
                self.assertAlmostEqual(thresh_lazy, thresh_arr, places=5)
                bc_vol = lazy.to_basecoat_volume(calibration=cal)
                self.assertEqual(bc_vol.shape, (16, 20, 24))


if __name__ == "__main__":
    unittest.main()

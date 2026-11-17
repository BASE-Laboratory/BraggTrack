"""Unit tests for LazyVolume ROI streaming, chunked Otsu, crystallography, and ImageD11/NeXus exports."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from braggtrack.cli import track_dataset
from braggtrack.cli._utils import write_csv
from braggtrack.io import (
    AxisCalibration,
    DetectorGeometry,
    LazyVolume,
    annotate_crystallographic_columns,
    export_imaged11_flt,
    export_nexus_process,
    open_lazy_volume,
    otsu_threshold_chunked,
    read_imaged11_flt,
)
from braggtrack.segmentation import otsu_threshold
from braggtrack.semantic import crop_spot_cube


class LazyAndExportTests(unittest.TestCase):
    def test_lazy_volume_roi_and_chunked_otsu_parity(self) -> None:
        rng = np.random.default_rng(42)
        vol = (rng.exponential(scale=2.0, size=(20, 32, 28)) * 10.0).astype(np.float32)
        vol[6:10, 12:18, 10:15] += 250.0

        labels = np.zeros_like(vol, dtype=np.int32)
        labels[6:10, 12:18, 10:15] = 1
        bbox = {
            "bbox_min_z": 6,
            "bbox_max_z": 9,
            "bbox_min_y": 12,
            "bbox_max_y": 17,
            "bbox_min_x": 10,
            "bbox_max_x": 14,
        }

        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "test_vol.h5"
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol, chunks=(5, 16, 14))

            with open_lazy_volume(h5_path) as lazy:
                self.assertEqual(lazy.shape, vol.shape)
                self.assertEqual(lazy.dataset_path, "/entry_0000/ESRF-ID03/pco_nf/data")

                roi_lazy, mask_lazy = crop_spot_cube(lazy, labels, 1, bbox, margin=2)
                roi_dense, mask_dense = crop_spot_cube(vol.astype(np.float64), labels, 1, bbox, margin=2)
                np.testing.assert_allclose(roi_lazy, roi_dense)
                np.testing.assert_array_equal(mask_lazy, mask_dense)

                thr_chunked = otsu_threshold_chunked(lazy, chunk_z=5)
                thr_dense = otsu_threshold(vol)
                self.assertAlmostEqual(thr_chunked, thr_dense, places=5)

        # Constant-volume edge case
        const_lazy = LazyVolume.from_array(np.full((4, 4, 4), 7.5))
        self.assertEqual(otsu_threshold_chunked(const_lazy), 7.5)

    def test_detector_geometry_bragg_law_identities(self) -> None:
        geom = DetectorGeometry(
            distance_mm=100.0,
            pixel_size_y_mm=0.1,
            pixel_size_x_mm=0.1,
            beam_center_y_px=500.0,
            beam_center_x_px=500.0,
            wavelength_angstrom=0.5,
            omega_origin_deg=-10.0,
            omega_step_deg=0.5,
        )
        # Offset by dx = 30 mm, dy = 40 mm => r = 50 mm at L = 100 mm => tan(2theta) = 0.5
        scat = geom.spot_to_scattering(z_idx=20.0, y_px=900.0, x_px=800.0)
        expected_tth = math.degrees(math.atan2(50.0, 100.0))
        expected_d = 0.5 / (2.0 * math.sin(0.5 * math.radians(expected_tth)))
        expected_q = (2.0 * math.pi) / expected_d

        self.assertAlmostEqual(scat["omega_deg"], 0.0, places=5)
        self.assertAlmostEqual(scat["two_theta_deg"], expected_tth, places=5)
        self.assertAlmostEqual(scat["d_spacing_angstrom"], expected_d, places=5)
        self.assertAlmostEqual(scat["q_mag_inv_angstrom"], expected_q, places=5)

        # Verify rotational invariance of |Q| across arbitrary omega
        scat_rot = geom.spot_to_scattering(z_idx=73.0, y_px=900.0, x_px=800.0)
        self.assertAlmostEqual(scat_rot["q_mag_inv_angstrom"], expected_q, places=5)

        # Verify calibration inversion parity
        cal = AxisCalibration(
            mu_origin=-10.0, mu_step=0.5, chi_origin=0.0, chi_step=0.02, d_origin=1.0, d_step=0.01, calibrated=True
        )
        mu_phys, chi_phys, d_phys = cal.voxel_to_physical(20.0, 900.0, 800.0)
        annotated = annotate_crystallographic_columns(
            [{"centroid_mu": mu_phys, "centroid_chi": chi_phys, "centroid_d": d_phys}],
            geom,
            calibration=cal,
        )
        self.assertAlmostEqual(annotated[0]["d_spacing_angstrom"], expected_d, places=5)

    def test_imaged11_and_nexus_exports_via_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seg_dir = root / "seg"
            for s_idx in range(3):
                sd = seg_dir / f"scan{s_idx + 1:04d}"
                sd.mkdir(parents=True)
                write_csv(
                    sd / "features.csv",
                    [
                        {
                            "label": 1,
                            "voxel_count": 30,
                            "integrated_intensity": 1200.0 + 10.0 * s_idx,
                            "centroid_mu": 5.0,
                            "centroid_chi": 620.0,
                            "centroid_d": 580.0 + 0.5 * s_idx,
                            "eig_1": 2.0,
                            "eig_2": 1.4,
                            "eig_3": 1.1,
                            "bbox_min_z": 3,
                            "bbox_max_z": 7,
                            "bbox_min_y": 577,
                            "bbox_max_y": 583,
                            "bbox_min_x": 617,
                            "bbox_max_x": 623,
                        }
                    ],
                )
                (sd / "summary.json").write_text(json.dumps({"calibration": {"calibrated": False}}))

            geom_path = root / "detector.json"
            geom_path.write_text(
                json.dumps(
                    {
                        "distance_mm": 120.0,
                        "pixel_size_y_mm": 0.075,
                        "pixel_size_x_mm": 0.075,
                        "beam_center_y_px": 512.0,
                        "beam_center_x_px": 512.0,
                        "wavelength_angstrom": 0.31,
                        "omega_origin_deg": 0.0,
                        "omega_step_deg": 0.1,
                    }
                )
            )
            flt_path = root / "peaks.flt"
            nxs_path = root / "results.nxs"
            trk_dir = root / "tracking"

            rc = track_dataset.main(
                [
                    str(seg_dir),
                    "--outdir",
                    str(trk_dir),
                    "--detect-events",
                    "--detector-json",
                    str(geom_path),
                    "--export-imaged11",
                    str(flt_path),
                    "--export-nexus",
                    str(nxs_path),
                ]
            )
            self.assertEqual(rc, 0)

            hdr, flt_rows = read_imaged11_flt(flt_path)
            self.assertEqual(len(flt_rows), 3)
            self.assertAlmostEqual(float(hdr["wavelength"]), 0.31, places=5)
            self.assertTrue(all(r["tth"] > 0.0 and r["d_spacing"] > 0.0 for r in flt_rows))

            with h5py.File(nxs_path, "r") as h5:
                self.assertEqual(h5.attrs["NX_class"], "NXroot")
                self.assertEqual(h5["/entry"].attrs["NX_class"], "NXentry")
                self.assertEqual(h5["/entry/braggtrack"].attrs["NX_class"], "NXprocess")
                self.assertEqual(int(h5["/entry/braggtrack/tracks"].attrs["row_count"]), 3)
                self.assertIn("two_theta_deg", h5["/entry/braggtrack/tracks"])
                self.assertIn("strain", h5["/entry/braggtrack/kinematics"])
                self.assertIn("delta_d", h5["/entry/braggtrack/kinematics"])

            # Direct export helper check
            nxs_direct = export_nexus_process(trk_dir, root / "direct.nxs")
            self.assertTrue(nxs_direct.exists())
            flt_direct = export_imaged11_flt(
                [{"centroid_mu": 5.0, "centroid_chi": 620.0, "centroid_d": 580.0, "integrated_intensity": 500.0}],
                root / "direct.flt",
            )
            self.assertTrue(flt_direct.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from braggtrack.io import (
    AxisCalibration,
    BeamlineAdapter,
    discover_primary_volume_path,
    extract_axis_calibration,
    extract_scan_metadata,
    load_primary_volume,
    resolve_dataset_root,
)
from braggtrack.segmentation import extract_instance_table
from braggtrack.tracking import compute_grain_kinematics, kinematics_to_table


class TestAxisCalibrationAndESRFDiscovery(unittest.TestCase):
    def test_axis_calibration_roundtrip_and_transform(self) -> None:
        cal = AxisCalibration(
            mu_origin=-1.5,
            mu_step=0.05,
            chi_origin=10.0,
            chi_step=0.02,
            d_origin=2.035,
            d_step=-1.5e-4,
            calibrated=True,
        )
        mu, chi, d = cal.voxel_to_physical(z=10.0, y=20.0, x=30.0)
        self.assertAlmostEqual(mu, -1.0)
        self.assertAlmostEqual(chi, 10.6)
        self.assertAlmostEqual(d, 2.032)

        restored = AxisCalibration.from_dict(cal.to_dict())
        self.assertEqual(restored, cal)
        self.assertTrue(restored.calibrated)

        with self.assertRaises(ValueError):
            AxisCalibration(d_step=0.0)

    def test_bundled_esrf_id03_scans_load_real_detector_cubes(self) -> None:
        root = resolve_dataset_root(None)
        adapter = BeamlineAdapter(root)
        seq = adapter.build_sequence()
        self.assertEqual(len(seq.scans), 3)
        self.assertTrue(seq.is_monotonic())

        for scan in seq.scans:
            self.assertEqual(scan.volume_dataset_path, "/entry_0000/ESRF-ID03/pco_nf/data")
            self.assertIsNotNone(scan.start_time)
            self.assertEqual(scan.title, "Lima 2D detector acquisition")

        vol = load_primary_volume(seq.scans[0].file_path)
        self.assertEqual(vol.shape, (100, 250, 250))

    def test_auto_discover_fallback_for_custom_h5_layout_and_axes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            h5_path = Path(tmp) / "custom_nx.h5"
            cube = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
            with h5py.File(h5_path, "w") as h5:
                grp = h5.create_group("/entry_custom")
                grp.attrs["NX_class"] = "NXentry"
                grp.create_dataset("start_time", data="2026-01-15T12:00:00Z")
                grp.create_dataset("title", data="custom_operando_scan")
                data_grp = grp.create_group("instrument/detector")
                data_grp.create_dataset("frames", data=cube)
                data_grp.create_dataset("mu", data=np.linspace(0.0, 0.4, 5))
                data_grp.create_dataset("chi", data=np.linspace(-1.0, -0.4, 7))
                data_grp.create_dataset("d", data=np.linspace(2.05, 2.06, 6))

            ds_path = discover_primary_volume_path(h5_path)
            self.assertEqual(ds_path, "/entry_custom/instrument/detector/frames")
            loaded = load_primary_volume(h5_path)
            self.assertEqual(loaded.shape, (5, 6, 7))

            meta = extract_scan_metadata(h5_path)
            self.assertEqual(meta["title"], "custom_operando_scan")
            self.assertIsNotNone(meta["start_time"])

            cal = extract_axis_calibration(h5_path)
            self.assertTrue(cal.calibrated)
            self.assertAlmostEqual(cal.mu_step, 0.1, places=6)
            self.assertAlmostEqual(cal.chi_origin, -1.0, places=6)
            self.assertAlmostEqual(cal.d_origin, 2.05, places=6)

    def test_calibrated_features_preserve_voxel_bbox_and_fix_crop_offset_strain(self) -> None:
        shape = (16, 32, 32)
        zz, yy, xx = np.indices(shape, dtype=np.float64)

        # Two scans where a spot shifts by +2 voxels along y (d-axis) in a cropped ROI.
        # Uncalibrated voxel strain would be 2.0 / 10.0 = +0.20 (20% artificial!),
        # whereas calibrated physical strain with d_origin=2.000 A, d_step=0.001 A/px
        # is (2 * 0.001) / (2.000 + 10 * 0.001) = 0.002 / 2.010 = 9.95e-4.
        cal = AxisCalibration(
            mu_origin=5.0,
            mu_step=0.05,
            chi_origin=-2.0,
            chi_step=0.04,
            d_origin=2.000,
            d_step=0.001,
            calibrated=True,
        )

        track_rows = []
        for scan_idx, y_center in enumerate((10.0, 12.0)):
            vol = 100.0 * np.exp(-0.5 * ((zz - 8.0) ** 2 + (yy - y_center) ** 2 + (xx - 15.0) ** 2))
            labels = (vol >= 10.0).astype(np.int32)
            rows = extract_instance_table(labels, vol, calibration=cal)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            # Bounding box indices MUST stay integer voxel coordinates for cropping!
            self.assertIsInstance(row["bbox_min_y"], int)
            self.assertGreaterEqual(row["bbox_min_y"], 6)
            self.assertLessEqual(row["bbox_max_y"], 16)
            row["track_id"] = 1
            row["scan_idx"] = scan_idx
            track_rows.append(row)

        gk = compute_grain_kinematics(track_rows)
        self.assertEqual(len(gk), 1)
        expected_strain = (2.0 * 0.001) / (2.000 + 10.0 * 0.001)
        self.assertAlmostEqual(float(gk[0].strain[1]), expected_strain, places=4)
        self.assertAlmostEqual(float(gk[0].delta_d[1]), 0.002, places=5)

        # Default table preserves strict 18-column contract; include_delta_d=True adds delta_d.
        tbl_default = kinematics_to_table(gk)
        self.assertNotIn("delta_d", tbl_default[0])
        tbl_ext = kinematics_to_table(gk, include_delta_d=True)
        self.assertIn("delta_d", tbl_ext[0])

        # Verify segment_dataset CLI --calibration-json support on synthetic h5.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cal_path = tmp_path / "cal.json"
            cal_path.write_text(json.dumps(cal.to_dict()))
            scan_dir = tmp_path / "data" / "scan0001"
            scan_dir.mkdir(parents=True)
            vol = 100.0 * np.exp(-0.5 * ((zz - 8.0) ** 2 + (yy - 10.0) ** 2 + (xx - 15.0) ** 2))
            with h5py.File(scan_dir / "scan.h5", "w") as h5:
                h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol.astype(np.float32))

            from braggtrack.cli import segment_dataset

            out_seg = tmp_path / "seg_out"
            rc = segment_dataset.main(
                [str(tmp_path / "data"), "--outdir", str(out_seg), "--calibration-json", str(cal_path)]
            )
            self.assertEqual(rc, 0)
            seg_summary = json.loads((out_seg / "segmentation_summary.json").read_text())
            self.assertTrue(seg_summary[0]["calibrated"])


if __name__ == "__main__":
    unittest.main()

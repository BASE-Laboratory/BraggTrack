from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from braggtrack.cli import detect_events
from braggtrack.cli._utils import write_csv
from braggtrack.semantic import make_patch_encoder
from braggtrack.tracking import (
    PositionShapeCost,
    PrecursorConfig,
    SplitMergeConfig,
    build_tracks,
    compute_fracture_precursors,
    compute_grain_kinematics,
    compute_tracking_metrics,
    events_to_table,
    generate_fracture_scenario,
    precursors_to_table,
    tracks_to_table,
)


class TestEventsAndPrecursors(unittest.TestCase):
    def test_fracture_scenario_recovers_split_merge_and_top_precursor(self) -> None:
        volumes, label_volumes, scan_tables, expected = generate_fracture_scenario(n_scans=5, seed=42)
        sm_cfg = SplitMergeConfig(max_distance=20.0)
        G = build_tracks(
            scan_tables,
            cost_fn=PositionShapeCost(),
            split_merge_config=sm_cfg,
            scan_timestamps={i: f"2026-01-15T12:0{i}:00Z" for i in range(5)},
        )
        metrics = compute_tracking_metrics(G, n_scans=5)
        self.assertEqual(metrics["split_count"], 1)
        self.assertEqual(metrics["merge_count"], 1)

        lineage_events = G.graph["lineage_events"]
        splits = [e for e in lineage_events if e.event_type == "split"]
        merges = [e for e in lineage_events if e.event_type == "merge"]
        self.assertEqual(len(splits), 1)
        self.assertEqual(len(merges), 1)

        split_ev = splits[0]
        self.assertEqual(split_ev.scan_idx_from, expected["split_scan_from"])
        self.assertEqual(split_ev.scan_idx_to, expected["split_scan_to"])
        self.assertEqual(len(split_ev.parent_track_ids), 1)
        self.assertEqual(len(split_ev.child_track_ids), 2)
        # Daughter track IDs must be distinct from each other and from parent.
        self.assertEqual(len(set(split_ev.child_track_ids)), 2)
        self.assertNotIn(split_ev.parent_track_ids[0], split_ev.child_track_ids)

        merge_ev = merges[0]
        self.assertEqual(merge_ev.scan_idx_from, expected["merge_scan_from"])
        self.assertEqual(merge_ev.scan_idx_to, expected["merge_scan_to"])
        self.assertEqual(len(merge_ev.parent_track_ids), 2)
        self.assertEqual(len(merge_ev.child_track_ids), 1)

        # Verify diverging daughter kinematics after split.
        track_rows = tracks_to_table(G)
        gk_by_tid = {gk.track_id: gk for gk in compute_grain_kinematics(track_rows)}
        c1, c2 = split_ev.child_track_ids
        self.assertIn(c1, gk_by_tid)
        self.assertIn(c2, gk_by_tid)
        # Daughters drift in opposite directions along d across scans 2..4.
        self.assertLess(float(gk_by_tid[c1].strain[-1]) * float(gk_by_tid[c2].strain[-1]), 0.0)

        # Verify fracture precursor ranking with 3D ROI sub-peak + mock patch encoder.
        patch_enc = make_patch_encoder("mock")
        candidates, obs_scores = compute_fracture_precursors(
            track_rows,
            PrecursorConfig(score_threshold=0.35, min_persistence=2),
            volumes=volumes,
            label_volumes=label_volumes,
            patch_encoder=patch_enc,
            lineage_events=lineage_events,
        )
        self.assertGreaterEqual(len(candidates), 1)
        top = candidates[0]
        self.assertEqual(top.rank, 1)
        self.assertEqual(top.track_id, split_ev.parent_track_ids[0])
        self.assertTrue(top.subsequent_split)
        self.assertGreaterEqual(top.persistence_scans, 2)
        self.assertGreaterEqual(top.max_subpeak_count, 2)

        # Control grain (located at chi ~ 36, d ~ 24) must NOT be ranked as a precursor.
        control_tids = {
            int(r["track_id"])
            for r in track_rows
            if abs(float(r["centroid_chi"]) - 36.0) < 2.0 and abs(float(r["centroid_d"]) - 24.0) < 2.0
        }
        self.assertTrue(control_tids)
        ranked_tids = {c.track_id for c in candidates}
        self.assertTrue(control_tids.isdisjoint(ranked_tids))

        # Verify table serializers.
        self.assertEqual(len(events_to_table(lineage_events)), 2)
        self.assertEqual(len(precursors_to_table(candidates)), len(candidates))
        self.assertGreater(len(obs_scores), 0)

    def test_anti_noise_intensity_guard_rejects_tiny_spurious_speck(self) -> None:
        """A tiny background speck appearing next to a steady grain must NOT trigger a false split."""
        scan0 = [
            {
                "label": 1,
                "centroid_mu": 10.0,
                "centroid_chi": 10.0,
                "centroid_d": 10.0,
                "integrated_intensity": 1000.0,
                "voxel_count": 50,
                "eig_1": 1.0,
                "eig_2": 1.0,
                "eig_3": 1.0,
            }
        ]
        scan1 = [
            {
                "label": 1,
                "centroid_mu": 10.1,
                "centroid_chi": 10.0,
                "centroid_d": 10.0,
                "integrated_intensity": 995.0,
                "voxel_count": 50,
                "eig_1": 1.0,
                "eig_2": 1.0,
                "eig_3": 1.0,
            },
            # Spurious 2% intensity noise speck nearby: fails min_fragment_share=0.10!
            {
                "label": 2,
                "centroid_mu": 12.0,
                "centroid_chi": 11.0,
                "centroid_d": 10.0,
                "integrated_intensity": 20.0,
                "voxel_count": 8,
                "eig_1": 0.5,
                "eig_2": 0.5,
                "eig_3": 0.5,
            },
        ]
        G = build_tracks([scan0, scan1], split_merge_config=SplitMergeConfig(max_distance=15.0))
        self.assertEqual(G.graph["lineage_events"], [])

    def test_single_frame_spike_filtered_by_persistence(self) -> None:
        """A transient 1-scan anisotropy spike must be rejected when min_persistence=2."""
        rows = [
            {"track_id": 1, "scan_idx": 0, "label": 1, "eig_1": 1.0, "eig_2": 1.0, "eig_3": 1.0},
            {"track_id": 1, "scan_idx": 1, "label": 1, "eig_1": 6.0, "eig_2": 1.0, "eig_3": 1.0},  # transient spike
            {"track_id": 1, "scan_idx": 2, "label": 1, "eig_1": 1.0, "eig_2": 1.0, "eig_3": 1.0},
        ]
        candidates, _ = compute_fracture_precursors(
            rows,
            PrecursorConfig(score_threshold=0.40, min_persistence=2),
        )
        self.assertEqual(candidates, [])

    def test_detect_events_cli_end_to_end(self) -> None:
        volumes, label_volumes, scan_tables, _ = generate_fracture_scenario(n_scans=4, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            seg_dir = tmp_path / "seg"
            out_dir = tmp_path / "events"
            for idx, (vol, labs, rows) in enumerate(zip(volumes, label_volumes, scan_tables)):
                sname = f"scan{idx + 1:04d}"
                (data_dir / sname).mkdir(parents=True)
                with h5py.File(data_dir / sname / "scan.h5", "w") as h5:
                    h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol.astype(np.float32))
                (seg_dir / sname).mkdir(parents=True)
                write_csv(seg_dir / sname / "features.csv", rows)
                np.savez_compressed(seg_dir / sname / "labels.npz", labels=labs)

            rc = detect_events.main(
                [
                    str(data_dir),
                    "--segdir",
                    str(seg_dir),
                    "--outdir",
                    str(out_dir),
                    "--backend",
                    "mock",
                ]
            )
            self.assertEqual(rc, 0)
            summary = json.loads((out_dir / "events_summary.json").read_text())
            self.assertEqual(summary["split_count"], 1)
            self.assertEqual(summary["merge_count"], 1)
            self.assertTrue((out_dir / "events.csv").exists())
            self.assertTrue((out_dir / "precursors.csv").exists())


if __name__ == "__main__":
    unittest.main()

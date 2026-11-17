"""Unit and CLI tests for braggtrack.viz plotting functions and braggtrack-report HTML explorer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from braggtrack.cli import report_dataset, run_pipeline
from braggtrack.tracking import (
    PrecursorConfig,
    SplitMergeConfig,
    build_tracks,
    compute_fracture_precursors,
    compute_grain_kinematics,
    generate_fracture_scenario,
    tracks_to_table,
)
from braggtrack.viz import (
    generate_html_report,
    plot_kinematics_dashboard,
    plot_lineage_dag,
    plot_precursor_timeline,
    plot_scan_projection_overlay,
    plot_spot_mip_card,
)


class VizAndReportTests(unittest.TestCase):
    def test_all_matplotlib_plots_render_and_save(self) -> None:
        volumes, label_volumes, scan_tables, _ = generate_fracture_scenario()
        G = build_tracks(scan_tables, split_merge_config=SplitMergeConfig(max_distance=20.0))
        track_rows = tracks_to_table(G)
        gks = compute_grain_kinematics(track_rows)
        candidates, obs_scores = compute_fracture_precursors(
            track_rows,
            PrecursorConfig(score_threshold=0.35, min_persistence=2),
            volumes=volumes,
            label_volumes=label_volumes,
            lineage_events=G.graph["lineage_events"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fig1, _ = plot_lineage_dag(G, save_path=out / "dag.png")
            plt.close(fig1)
            self.assertTrue((out / "dag.png").stat().st_size > 1000)

            fig2, _ = plot_kinematics_dashboard(
                gks, events=G.graph["lineage_events"], calibrated=True, save_path=out / "kin.png"
            )
            plt.close(fig2)
            self.assertTrue((out / "kin.png").stat().st_size > 1000)

            fig3, _ = plot_precursor_timeline(obs_scores, candidates=candidates, save_path=out / "prec.png")
            plt.close(fig3)
            self.assertTrue((out / "prec.png").stat().st_size > 1000)

            spot0 = scan_tables[0][0]
            fig4, _ = plot_spot_mip_card(
                volumes[0], label_volumes[0], int(spot0["label"]), spot0, save_path=out / "mip.png"
            )
            plt.close(fig4)
            self.assertTrue((out / "mip.png").stat().st_size > 1000)

            fig5, _ = plot_scan_projection_overlay(
                volumes[0], label_volumes[0], scan_tables[0], save_path=out / "proj.png"
            )
            plt.close(fig5)
            self.assertTrue((out / "proj.png").stat().st_size > 1000)

    def test_pipeline_with_report_and_exports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ds = base / "operando"
            for i in range(3):
                sdir = ds / f"scan{i + 1:04d}"
                sdir.mkdir(parents=True)
                vol = np.zeros((16, 28, 28), dtype=np.float32)
                vol[5:10, 10 + i : 15 + i, 12:17] = 180.0
                with h5py.File(sdir / "pco_nf.h5", "w") as h5:
                    entry = h5.create_group("entry_0000")
                    entry.attrs["NX_class"] = "NXentry"
                    entry.create_dataset("start_time", data=f"2026-05-16T10:0{i}:00Z")
                    h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol)

            out_dir = base / "pipeline_out"
            nxs_out = base / "summary.nxs"
            flt_out = base / "peaks.flt"
            rc = run_pipeline.main(
                [
                    str(ds),
                    "--outdir",
                    str(out_dir),
                    "--backend",
                    "mock",
                    "--report",
                    "--export-nexus",
                    str(nxs_out),
                    "--export-imaged11",
                    str(flt_out),
                ]
            )
            self.assertEqual(rc, 0)
            html_file = out_dir / "report" / "report.html"
            self.assertTrue(html_file.exists())
            html_text = html_file.read_text(encoding="utf-8")
            self.assertIn("BraggTrack 4D Explorer", html_text)
            self.assertIn("data:image/png;base64,", html_text)
            self.assertTrue((out_dir / "report" / "figures" / "lineage_dag.png").exists())
            self.assertTrue(nxs_out.exists())
            self.assertTrue(flt_out.exists())

            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertIn("report", manifest["stage_timings_sec"])
            self.assertIsNotNone(manifest["artifacts"]["html_report"])

            # Standalone report CLI smoke check
            standalone_html = generate_html_report(out_dir, base / "standalone.html", dataset_root=ds)
            self.assertTrue(standalone_html.exists())
            rc_rep = report_dataset.main([str(out_dir), "--outdir", str(base / "rep2"), "--no-figures"])
            self.assertEqual(rc_rep, 0)


if __name__ == "__main__":
    unittest.main()

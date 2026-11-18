from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from braggtrack.cli import run_pipeline
from braggtrack.tracking import generate_fracture_scenario


class TestRunPipelineAndAblationCI(unittest.TestCase):
    def test_run_pipeline_with_config_produces_manifest_and_all_artifacts(self) -> None:
        volumes, _, _, _ = generate_fracture_scenario(n_scans=4, seed=11)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "operando"
            for idx, vol in enumerate(volumes):
                sdir = data_dir / f"scan{idx + 1:04d}"
                sdir.mkdir(parents=True)
                with h5py.File(sdir / "pco_nf.h5", "w") as h5:
                    grp = h5.create_group("/entry_0000")
                    grp.create_dataset("start_time", data=f"2026-02-01T10:0{idx}:00Z")
                    grp.create_dataset("title", data="synth_operando")
                    h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol.astype(np.float32))

            cfg_path = tmp_path / "pipeline_config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "backend": "mock",
                        "cost_beta": 0.25,
                        "smooth_window": 3,
                        "detect_events": True,
                        "segmentation_extra_args": ["--merge-distance", "0"],
                    }
                )
            )
            out_dir = tmp_path / "pipeline_out"
            rc = run_pipeline.main([str(data_dir), "--outdir", str(out_dir), "--config", str(cfg_path)])
            self.assertEqual(rc, 0)

            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], "manifest.v1")
            self.assertEqual(len(manifest["inputs"]), 4)
            for inp in manifest["inputs"]:
                self.assertEqual(len(inp["sha256"]), 64)

            for stage in ("validate", "segment", "embed", "track", "events"):
                self.assertIn(stage, manifest["stage_timings_sec"])

            for art_key in ("kinematics_csv", "kinematics_summary", "events_csv", "precursors_csv"):
                self.assertTrue(Path(manifest["artifacts"][art_key]).exists(), msg=f"Missing {art_key}")

            # Test ablation_semantic.py --bootstrap-samples on the generated segmentation + embeddings.
            import subprocess
            import sys

            ablation_out = tmp_path / "ablation.json"
            proc_ab = subprocess.run(
                [
                    sys.executable,
                    "scripts/ablation_semantic.py",
                    "--indir",
                    str(out_dir / "segmentation"),
                    "--embedding-dir",
                    str(out_dir / "embedding"),
                    "--betas",
                    "0,0.5",
                    "--bootstrap-samples",
                    "20",
                    "--output",
                    str(ablation_out),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc_ab.returncode, 0, msg=proc_ab.stdout + proc_ab.stderr)
            rep = json.loads(ablation_out.read_text())
            self.assertEqual(rep["bootstrap_samples"], 20)
            self.assertEqual(len(rep["runs"]), 2)
            for run in rep["runs"]:
                self.assertIn("fragmentation_ratio_ci95", run)
                self.assertEqual(len(run["fragmentation_ratio_ci95"]), 2)
                self.assertIn("id_switch_rate_ci95", run)

    def test_run_pipeline_default_skips_embedding_stage(self) -> None:
        volumes, _, _, _ = generate_fracture_scenario(n_scans=4, seed=19)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "operando_classical"
            for idx, vol in enumerate(volumes):
                sdir = data_dir / f"scan{idx + 1:04d}"
                sdir.mkdir(parents=True)
                with h5py.File(sdir / "pco_nf.h5", "w") as h5:
                    h5.create_dataset("/entry_0000/ESRF-ID03/pco_nf/data", data=vol.astype(np.float32))

            out_dir = tmp_path / "pipeline_classical_out"
            rc = run_pipeline.main([str(data_dir), "--outdir", str(out_dir)])
            self.assertEqual(rc, 0)
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["config"]["cost_beta"], 0.0)
            self.assertNotIn("embed", manifest["stage_timings_sec"])
            self.assertIsNone(manifest["artifacts"]["embedding_summary"])
            self.assertTrue(Path(manifest["artifacts"]["kinematics_csv"]).exists())


if __name__ == "__main__":
    unittest.main()

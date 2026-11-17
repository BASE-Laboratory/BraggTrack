from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.benchmark_suite import main as benchmark_main  # noqa: E402
from scripts.benchmark_suite import run_suite  # noqa: E402


class BenchmarkSuiteTests(unittest.TestCase):
    def test_run_suite_geo_and_mock_smoke(self) -> None:
        payload = run_suite(["geo", "mock"], trials=1)
        self.assertEqual(payload["schema_version"], "benchmark_suite.v1")
        self.assertEqual(payload["backends"], ["geo", "mock"])
        scenarios = payload["scenarios"]
        for expected in (
            "crossing_morphology",
            "intensity_jitter",
            "mosaic_broadening",
            "beam_dropout_no_gap",
            "beam_dropout_gap2",
            "chemomechanical_fracture",
        ):
            self.assertIn(expected, scenarios)
            self.assertIn("geo", scenarios[expected])
            self.assertIn("mock", scenarios[expected])

        # Gap-closing recovers both staggered 1-scan dropouts on geometry backend
        self.assertEqual(scenarios["beam_dropout_no_gap"]["geo"]["mean_gap_closed_count"], 0.0)
        self.assertEqual(scenarios["beam_dropout_gap2"]["geo"]["mean_gap_closed_count"], 2.0)
        self.assertLess(
            scenarios["beam_dropout_gap2"]["geo"]["mean_fragmentation_ratio"],
            scenarios["beam_dropout_no_gap"]["geo"]["mean_fragmentation_ratio"],
        )

    def test_cli_writes_json_and_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_out = tmp_path / "bench.json"
            plot_out = tmp_path / "bench.png"
            rc = benchmark_main(
                [
                    "--backends",
                    "geo,mock",
                    "--trials",
                    "1",
                    "--json-out",
                    str(json_out),
                    "--plot-out",
                    str(plot_out),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(json_out.exists())
            self.assertTrue(plot_out.exists())
            self.assertGreater(plot_out.stat().st_size, 5_000)
            parsed = json.loads(json_out.read_text())
            self.assertEqual(parsed["trials"], 1)


if __name__ == "__main__":
    unittest.main()

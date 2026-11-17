"""Report, multi-scan gap-closing, and crystallographic export acceptance gate.

Acceptance criteria:
  1. Multi-scan gap closing (--max-gap 2) reconnects transient 1-scan beam dropouts
     with zero track fragmentation on a controlled dropout sequence.
  2. End-to-end pipeline (braggtrack-run --report --detector-json --export-imaged11
     --export-nexus) executes on bundled ESRF-ID03 operando data and emits:
       - Interactive self-contained HTML explorer (report/report.html) + static PNGs
       - ImageD11 columnfile (.flt) with physical (tth, eta, d_spacing) columns
       - NeXus NXprocess HDF5 summary (.nxs) carrying tracks, kinematics, events, and precursors.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braggtrack.cli._utils import write_csv
from braggtrack.io import DetectorGeometry, read_imaged11_flt, resolve_dataset_root

ACCEPTANCE_OUTDIR = Path("artifacts/report_export_acceptance")
DATASET_ROOT = resolve_dataset_root(None)


def _check_gap_closing_cli(failures: list[str]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        seg_dir = tmp_root / "seg"
        trk_dir = tmp_root / "trk"

        # Two spots across 4 scans; spot 1 drops out at scan0002 (scan_idx=1) and reappears at scan0003
        scan_specs = [
            [(1, 10.0, 20.0, 30.0), (2, 10.0, 60.0, 60.0)],
            [(1, 10.0, 60.2, 60.1)],
            [(1, 10.1, 20.3, 30.2), (2, 10.0, 60.4, 60.2)],
            [(1, 10.2, 20.5, 30.4), (2, 10.1, 60.6, 60.3)],
        ]
        for idx, specs in enumerate(scan_specs):
            sdir = seg_dir / f"scan{idx + 1:04d}"
            sdir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "label": lbl,
                    "voxel_count": 20,
                    "integrated_intensity": 500.0,
                    "centroid_mu": mu,
                    "centroid_chi": chi,
                    "centroid_d": d,
                    "bbox_min_mu": int(mu) - 2,
                    "bbox_min_d": int(d) - 2,
                    "bbox_min_chi": int(chi) - 2,
                    "bbox_max_mu": int(mu) + 3,
                    "bbox_max_d": int(d) + 3,
                    "bbox_max_chi": int(chi) + 3,
                    "cov_00": 1.0,
                    "cov_01": 0.0,
                    "cov_02": 0.0,
                    "cov_11": 1.0,
                    "cov_12": 0.0,
                    "cov_22": 1.0,
                    "eig_1": 1.0,
                    "eig_2": 1.0,
                    "eig_3": 1.0,
                }
                for lbl, mu, chi, d in specs
            ]
            write_csv(sdir / "features.csv", rows)

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "braggtrack.cli.track_dataset",
                str(seg_dir),
                "--outdir",
                str(trk_dir),
                "--max-gap",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            failures.append(f"track_dataset --max-gap 2 failed: {proc.stderr.strip() or proc.stdout.strip()}")
            return {}

        summary = json.loads(proc.stdout)
        if int(summary.get("gap_closed_count", 0)) != 1:
            failures.append(f"Expected gap_closed_count==1, got {summary.get('gap_closed_count')}")
        if float(summary.get("fragmentation_ratio", 1.0)) != 0.0:
            failures.append(
                f"Expected fragmentation_ratio==0.0 after gap closing, got {summary.get('fragmentation_ratio')}"
            )
        return {
            "gap_closed_count": summary.get("gap_closed_count"),
            "total_tracks": summary.get("total_tracks"),
            "fragmentation_ratio": summary.get("fragmentation_ratio"),
        }


def _check_pipeline_report_and_exports(failures: list[str]) -> dict[str, object]:
    ACCEPTANCE_OUTDIR.mkdir(parents=True, exist_ok=True)
    det_json = ACCEPTANCE_OUTDIR / "detector_geometry.json"
    flt_path = ACCEPTANCE_OUTDIR / "operando_peaks.flt"
    nxs_path = ACCEPTANCE_OUTDIR / "operando_results.nxs"

    geom = DetectorGeometry(
        distance_mm=150.0,
        pixel_size_y_mm=0.075,
        pixel_size_x_mm=0.075,
        beam_center_y_px=20.0,
        beam_center_x_px=20.0,
        wavelength_angstrom=0.50,
        omega_origin_deg=10.0,
        omega_step_deg=0.05,
    )
    geom.save_json(det_json)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "braggtrack.cli.run_pipeline",
            str(DATASET_ROOT),
            "--outdir",
            str(ACCEPTANCE_OUTDIR),
            "--backend",
            "mock",
            "--max-gap",
            "1",
            "--detector-json",
            str(det_json),
            "--export-imaged11",
            str(flt_path),
            "--export-nexus",
            str(nxs_path),
            "--report",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        failures.append(f"run_pipeline --report failed: {proc.stderr.strip() or proc.stdout.strip()}")
        return {}

    html_path = ACCEPTANCE_OUTDIR / "report" / "report.html"
    if not html_path.exists() or html_path.stat().st_size < 5_000:
        failures.append(f"Missing or undersized HTML report: {html_path}")
    else:
        html_text = html_path.read_text(encoding="utf-8")
        if "report.v1" not in html_text or "data:image/png;base64," not in html_text:
            failures.append("report.html missing embedded report.v1 payload or base64 MIP thumbnails")

    fig_dir = ACCEPTANCE_OUTDIR / "report" / "figures"
    expected_figs = ("lineage_dag.png", "kinematics_dashboard.png", "precursor_timeline.png")
    for fname in expected_figs:
        fpath = fig_dir / fname
        if not fpath.exists() or fpath.stat().st_size < 1_000:
            failures.append(f"Missing or empty report figure: {fpath}")

    if not flt_path.exists():
        failures.append(f"Missing ImageD11 export: {flt_path}")
        flt_rows = []
    else:
        flt_params, flt_rows = read_imaged11_flt(flt_path)
        if "wavelength" not in flt_params:
            failures.append("ImageD11 .flt header missing wavelength parameter")
        if not flt_rows:
            failures.append("ImageD11 .flt export contained 0 rows")
        else:
            for col in ("sc", "fc", "omega", "tth", "eta", "d_spacing", "track_id"):
                if col not in flt_rows[0]:
                    failures.append(f"ImageD11 .flt missing column {col}")
            if float(flt_rows[0].get("d_spacing", 0.0)) <= 0.0:
                failures.append("ImageD11 .flt d_spacing must be positive when DetectorGeometry is provided")

    if not nxs_path.exists():
        failures.append(f"Missing NeXus HDF5 export: {nxs_path}")
    else:
        with h5py.File(nxs_path, "r") as h5:
            if h5.attrs.get("NX_class") != "NXroot":
                failures.append("NeXus root missing NX_class=NXroot")
            if "/entry/braggtrack" not in h5:
                failures.append("NeXus file missing /entry/braggtrack NXprocess group")
            else:
                grp = h5["/entry/braggtrack"]
                for sub in ("tracks", "kinematics", "events", "precursors"):
                    if sub not in grp:
                        failures.append(f"NeXus /entry/braggtrack missing {sub} table group")

    return {
        "html_report_bytes": html_path.stat().st_size if html_path.exists() else 0,
        "flt_row_count": len(flt_rows),
        "nexus_bytes": nxs_path.stat().st_size if nxs_path.exists() else 0,
    }


def main() -> int:
    failures: list[str] = []
    gap_report = _check_gap_closing_cli(failures)
    export_report = _check_pipeline_report_and_exports(failures)
    payload = {
        "gap_closing": gap_report,
        "report_and_exports": export_report,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

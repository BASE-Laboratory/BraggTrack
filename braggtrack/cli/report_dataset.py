"""Generate publication figures and an interactive HTML 4D beamtime explorer from BraggTrack outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from braggtrack.viz import (
    generate_html_report,
    plot_kinematics_dashboard,
    plot_lineage_dag,
    plot_precursor_timeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        default="artifacts/pipeline",
        help="Pipeline or tracking output directory containing tracks.csv / events / manifest",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional raw HDF5 dataset root for embedding 3-axis MIP spot thumbnails",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional report output directory (defaults to <artifact_dir>/report)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip static Matplotlib PNG/PDF generation and emit only the self-contained HTML explorer",
    )
    parser.add_argument(
        "--max-mip-tracks",
        type=int,
        default=16,
        help="Maximum priority tracks for lazy 3-view MIP thumbnail extraction",
    )
    return parser


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_root = Path(args.artifact_dir)
    if not in_root.is_dir():
        print(json.dumps({"error": "Artifact directory not found", "artifact_dir": str(in_root)}))
        return 1

    out_root = Path(args.outdir) if args.outdir else (in_root / "report")
    out_root.mkdir(parents=True, exist_ok=True)

    trk_dir = in_root / "tracking" if (in_root / "tracking").is_dir() else in_root
    evt_dir = in_root / "events" if (in_root / "events").is_dir() else trk_dir
    track_rows = _read_csv_rows(evt_dir / "tracks.csv") or _read_csv_rows(trk_dir / "tracks.csv")
    if not track_rows:
        print(json.dumps({"error": "No tracks.csv found under artifact directory", "artifact_dir": str(in_root)}))
        return 1

    html_path = generate_html_report(
        in_root,
        out_root / "report.html",
        dataset_root=args.dataset_root,
        max_mip_tracks=max(0, args.max_mip_tracks),
    )

    figures: dict[str, str] = {}
    if not args.no_figures:
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt

            fig_dir = out_root / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)
            event_rows = _read_csv_rows(evt_dir / "events.csv") or _read_csv_rows(trk_dir / "events.csv")
            prec_rows = _read_csv_rows(evt_dir / "precursors.csv") or _read_csv_rows(trk_dir / "precursors.csv")
            prec_obs_rows = _read_csv_rows(evt_dir / "precursor_observations.csv")

            calibrated = False
            ts_path = trk_dir / "tracking_summary.json"
            if ts_path.exists():
                try:
                    calibrated = bool(json.loads(ts_path.read_text()).get("calibrated_axes", False))
                except (OSError, json.JSONDecodeError):
                    calibrated = False

            fig_dag, _ = plot_lineage_dag(track_rows, events=event_rows, save_path=fig_dir / "lineage_dag.png")
            plt.close(fig_dag)
            figures["lineage_dag"] = str((fig_dir / "lineage_dag.png").resolve())

            fig_kin, _ = plot_kinematics_dashboard(
                track_rows,
                events=event_rows,
                calibrated=calibrated,
                save_path=fig_dir / "kinematics_dashboard.png",
            )
            plt.close(fig_kin)
            figures["kinematics_dashboard"] = str((fig_dir / "kinematics_dashboard.png").resolve())

            if prec_obs_rows:
                fig_prec, _ = plot_precursor_timeline(
                    prec_obs_rows,
                    candidates=prec_rows,
                    save_path=fig_dir / "precursor_timeline.png",
                )
                plt.close(fig_prec)
                figures["precursor_timeline"] = str((fig_dir / "precursor_timeline.png").resolve())
        except ImportError:
            pass

    summary: dict[str, Any] = {
        "schema_version": "report.v1",
        "html_report": str(html_path.resolve()),
        "figures": figures,
        "track_row_count": len(track_rows),
    }
    (out_root / "report_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for operando electrochemistry / stimulus synchronization and visualization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from braggtrack.cli._utils import write_csv
from braggtrack.io import (
    ExperimentSequence,
    ScanVolumeMeta,
    annotate_tracks_with_stimulus,
    load_operando_log,
    synchronize_to_scans,
)
from braggtrack.tracking import PositionShapeCost, SplitMergeConfig, build_tracks, tracks_to_table
from braggtrack.viz import generate_html_report, plot_operando_strain_and_events


def test_biologic_mpt_parsing_and_dq_dv_sync(tmp_path: Path) -> None:
    """Biologic EC-Lab .mpt ASCII format parses Nb header lines, Ewe/V, <I>/mA, and derives dQ/dV."""
    mpt_path = tmp_path / "ni92_cycle1.mpt"
    mpt_lines = [
        "EC-Lab ASCII file",
        "Nb header lines : 5",
        "Acquisition started on : 11/17/2026 10:00:00",
        "Electrode material : LiNi0.92Mn0.04Co0.04O2",
        "time/s\tEwe/V\t<I>/mA\t(Q-Qo)/mA.h\tcycle number",
        "0.0\t3.600\t0.50\t0.00\t1",
        "600.0\t3.950\t0.50\t60.00\t1",
        "1200.0\t4.180\t0.50\t165.00\t1",
        "1800.0\t4.210\t0.50\t205.00\t1",
        "2400.0\t4.300\t0.50\t222.00\t1",
    ]
    mpt_path.write_text("\n".join(mpt_lines), encoding="utf-8")

    log = load_operando_log(mpt_path)
    assert log.n_points == 5
    assert log.voltage_v is not None
    assert log.dq_dv is not None
    # H2 -> H3 plateau (4.18 -> 4.21 V, delta_Q = 40 mAh over 30 mV) produces largest dQ/dV peak
    assert float(log.dq_dv[3]) > float(log.dq_dv[1])

    t0 = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc)
    seq = ExperimentSequence(
        scans=tuple(
            ScanVolumeMeta(
                scan_name=f"scan{i + 1:04d}",
                file_path=tmp_path / f"s{i}.h5",
                sequence_index=i,
                start_time=t0 + timedelta(seconds=600 * i),
                end_time=t0 + timedelta(seconds=600 * i + 20),
            )
            for i in range(5)
        )
    )
    pts = synchronize_to_scans(log, seq)
    assert len(pts) == 5
    assert 4.17 < (pts[2].voltage_v or 0.0) < 4.20
    assert pts[3].cycle == 1


def test_csv_operando_annotation_plot_and_html_report(tmp_path: Path) -> None:
    """CSV operando log enriches tracks with d_strain_d_v and renders synchronized plot + HTML explorer."""
    csv_log = tmp_path / "echem.csv"
    csv_log.write_text(
        "scan_idx,voltage_V,current_mA,capacity_mAh_g,temperature_C\n"
        "0,3.80,0.4,40.0,25.0\n"
        "1,4.16,0.4,155.0,25.2\n"
        "2,4.22,0.4,198.0,25.6\n"
        "3,4.30,0.4,215.0,25.8\n",
        encoding="utf-8",
    )

    def _spot(lbl: int, mu: float, chi: float, d: float, inten: float) -> dict:
        return {
            "label": lbl,
            "voxel_count": 25,
            "integrated_intensity": inten,
            "centroid_mu": mu,
            "centroid_chi": chi,
            "centroid_d": d,
            "centroid_mu_err": 1e-3,
            "centroid_chi_err": 1e-3,
            "centroid_d_err": 2e-4,
            "eig_1": 1.4,
            "eig_2": 1.0,
            "eig_3": 0.7,
        }

    scans = [
        [_spot(1, 5.0, 5.0, 1.422, 1000.0)],
        [_spot(1, 5.1, 5.0, 1.415, 1000.0)],
        [_spot(1, 5.2, 4.7, 1.388, 560.0), _spot(2, 5.2, 5.3, 1.412, 440.0)],
        [_spot(1, 5.3, 4.6, 1.384, 550.0), _spot(2, 5.3, 5.4, 1.411, 450.0)],
    ]
    G = build_tracks(scans, PositionShapeCost(), split_merge_config=SplitMergeConfig(max_distance=5.0))
    rows = tracks_to_table(G)

    log = load_operando_log(csv_log)
    pts = synchronize_to_scans(log, 4)
    enriched = annotate_tracks_with_stimulus(rows, pts)
    assert any(r["d_strain_d_v"] != "" for r in enriched)

    art_dir = tmp_path / "art"
    write_csv(art_dir / "tracking" / "tracks.csv", enriched)
    html_out = generate_html_report(art_dir, art_dir / "report.html", operando_log=csv_log)
    html_txt = html_out.read_text(encoding="utf-8")
    assert "Operando Potential E (V)" in html_txt
    assert "4.22" in html_txt

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_path = tmp_path / "operando_fig.png"
    fig, _ = plot_operando_strain_and_events(
        enriched,
        pts,
        events=G.graph["lineage_events"],
        zoom_triggers=[{"scan_idx": 1, "track_id": 1, "precursor_score": 0.72}],
        calibrated=True,
        save_path=fig_path,
    )
    plt.close(fig)
    assert fig_path.exists() and fig_path.stat().st_size > 1000

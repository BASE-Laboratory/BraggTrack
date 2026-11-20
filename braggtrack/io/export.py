"""Interoperability exporters for ImageD11 columnfiles (.flt) and NeXus NXprocess HDF5."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import braggtrack

from .crystallography import DetectorGeometry, _recover_voxel_centroid
from .models import AxisCalibration
from .nexus import _require_h5py

IMAGED11_COLUMNS = (
    "sc",
    "fc",
    "omega",
    "Number_of_pixels",
    "avg_intensity",
    "sum_intensity",
    "tth",
    "eta",
    "d_spacing",
    "gx",
    "gy",
    "gz",
    "scan_idx",
    "spot_id",
    "track_id",
)


def export_imaged11_flt(
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    geometry: DetectorGeometry | None = None,
    calibration: AxisCalibration | None = None,
) -> Path:
    """Write segmented or tracked Bragg reflections as an ImageD11 ``.flt`` columnfile.

    In ImageD11 convention, ``sc`` is the slow (vertical, ``y``) pixel coordinate,
    ``fc`` is the fast (horizontal, ``x``) pixel coordinate, and ``(gx, gy, gz)``
    are reciprocal-space scattering vectors in ``1/d`` (``Q / 2*pi``) units.
    """
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    geom = geometry or DetectorGeometry()
    two_pi = 2.0 * math.pi

    lines: list[str] = [
        f"# ImageD11 columnfile exported by BraggTrack {braggtrack.__version__}",
        f"# wavelength = {geom.wavelength_angstrom:.6f}",
        f"# distance = {geom.distance_mm * 1000.0:.3f}",
        f"# y_center = {geom.beam_center_y_px:.3f}",
        f"# z_center = {geom.beam_center_x_px:.3f}",
        f"# y_size = {geom.pixel_size_y_mm * 1000.0:.3f}",
        f"# z_size = {geom.pixel_size_x_mm * 1000.0:.3f}",
        "#  " + "  ".join(IMAGED11_COLUMNS),
    ]

    for idx, row in enumerate(rows):
        z_idx, y_px, x_px = _recover_voxel_centroid(row, calibration)
        scat = geom.spot_to_scattering(z_idx, y_px, x_px)
        vox = max(1, int(float(row.get("voxel_count", 1))))
        sum_int = float(row.get("integrated_intensity", 0.0))
        avg_int = sum_int / float(vox)
        scan_idx = int(float(row.get("scan_idx", 0)))
        spot_id = int(float(row.get("label", row.get("spot_idx", idx + 1))))
        track_id = int(float(row.get("track_id", -1)))
        d_val = scat["d_spacing_angstrom"] if math.isfinite(scat["d_spacing_angstrom"]) else 0.0
        gx = scat["qx_inv_angstrom"] / two_pi
        gy = scat["qy_inv_angstrom"] / two_pi
        gz = scat["qz_inv_angstrom"] / two_pi

        lines.append(
            f"{y_px:10.4f} {x_px:10.4f} {scat['omega_deg']:10.4f} "
            f"{vox:8d} {avg_int:12.4f} {sum_int:14.4f} "
            f"{scat['two_theta_deg']:10.5f} {scat['eta_deg']:10.4f} {d_val:10.5f} "
            f"{gx:11.6f} {gy:11.6f} {gz:11.6f} "
            f"{scan_idx:6d} {spot_id:6d} {track_id:6d}"
        )

    dest.write_text("\n".join(lines) + "\n")
    return dest


def read_imaged11_flt(path: str | Path) -> tuple[dict[str, str], list[dict[str, float]]]:
    """Parse an ImageD11 ``.flt`` columnfile back into header parameters and numeric row dicts."""
    params: dict[str, str] = {}
    columns: list[str] = []
    rows: list[dict[str, float]] = []

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if "=" in body:
                k, v = body.split("=", 1)
                params[k.strip()] = v.strip()
            elif body.startswith("sc ") or body.startswith("sc\t"):
                columns = body.split()
            continue
        if not columns:
            raise ValueError("Missing '#  sc  fc ...' column header in ImageD11 file.")
        parts = line.split()
        if len(parts) != len(columns):
            raise ValueError(f"Expected {len(columns)} columns, got {len(parts)} in line: {line}")
        rows.append({col: float(val) for col, val in zip(columns, parts)})

    return params, rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_table_group(grp: Any, h5py: Any, rows: list[dict[str, Any]]) -> None:
    if not rows:
        grp.attrs["row_count"] = 0
        return
    grp.attrs["row_count"] = len(rows)
    str_dt = h5py.string_dtype(encoding="utf-8")
    keys = list(rows[0].keys())
    for key in keys:
        vals = [r.get(key, "") for r in rows]
        # Attempt numeric conversion first (float64 or int64), else UTF-8 strings.
        is_int = True
        is_float = True
        int_vals: list[int] = []
        float_vals: list[float] = []
        for v in vals:
            s = str(v).strip()
            if s == "" or s.lower() in ("true", "false") or ";" in s:
                is_int = False
                is_float = False
                break
            try:
                iv = int(s)
                int_vals.append(iv)
            except ValueError:
                is_int = False
            try:
                fv = float(s)
                float_vals.append(fv)
            except ValueError:
                is_float = False
                break
        if is_int:
            grp.create_dataset(key, data=np.asarray(int_vals, dtype=np.int64), compression="gzip")
        elif is_float:
            grp.create_dataset(key, data=np.asarray(float_vals, dtype=np.float64), compression="gzip")
        else:
            arr = np.asarray([str(v) for v in vals], dtype=object)
            grp.create_dataset(key, data=arr, dtype=str_dt)


def export_nexus_process(
    artifact_dir: str | Path,
    out_h5: str | Path,
    *,
    geometry: DetectorGeometry | None = None,
    calibration: AxisCalibration | None = None,
) -> Path:
    """Package BraggTrack pipeline/tracking outputs into a NeXus ``NXentry`` / ``NXprocess`` HDF5 file."""
    h5py = _require_h5py()
    root = Path(artifact_dir)
    dest = Path(out_h5)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Support either a pipeline root (containing tracking/ & events/) or a direct tracking output dir.
    trk_dir = root / "tracking" if (root / "tracking").is_dir() else root
    evt_dir = root / "events" if (root / "events").is_dir() else trk_dir

    track_rows: list[dict[str, Any]] = [dict(r) for r in _read_csv_rows(trk_dir / "tracks.csv")]
    if geometry is not None and track_rows:
        from .crystallography import annotate_crystallographic_columns

        track_rows = annotate_crystallographic_columns(track_rows, geometry, calibration=calibration)

    kin_rows = _read_csv_rows(trk_dir / "kinematics.csv")
    event_rows = _read_csv_rows(evt_dir / "events.csv")
    prec_rows = _read_csv_rows(evt_dir / "precursors.csv")

    provenance_json = "{}"
    for cand in (root / "run_manifest.json", trk_dir / "tracking_summary.json"):
        if cand.exists():
            provenance_json = cand.read_text()
            break

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(dest, "w") as h5:
        h5.attrs["NX_class"] = "NXroot"
        h5.attrs["default"] = "entry"

        entry = h5.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "braggtrack"
        entry.create_dataset("title", data="BraggTrack 4D Diffraction Tracking & Fracture Analysis", dtype=str_dt)

        proc = entry.create_group("braggtrack")
        proc.attrs["NX_class"] = "NXprocess"
        proc.create_dataset("program", data="braggtrack", dtype=str_dt)
        proc.create_dataset("version", data=str(braggtrack.__version__), dtype=str_dt)
        proc.create_dataset("date", data=datetime.now(timezone.utc).isoformat(), dtype=str_dt)
        proc.create_dataset("provenance_json", data=provenance_json, dtype=str_dt)

        if geometry is not None:
            geom_grp = proc.create_group("detector_geometry")
            geom_grp.attrs["NX_class"] = "NXparameters"
            for k, v in geometry.to_dict().items():
                geom_grp.create_dataset(k, data=float(v))

        tracks_grp = proc.create_group("tracks")
        tracks_grp.attrs["NX_class"] = "NXdata"
        _write_table_group(tracks_grp, h5py, track_rows)

        kin_grp = proc.create_group("kinematics")
        kin_grp.attrs["NX_class"] = "NXdata"
        _write_table_group(kin_grp, h5py, kin_rows)

        evt_grp = proc.create_group("events")
        evt_grp.attrs["NX_class"] = "NXcollection"
        _write_table_group(evt_grp, h5py, event_rows)

        prec_grp = proc.create_group("precursors")
        prec_grp.attrs["NX_class"] = "NXcollection"
        _write_table_group(prec_grp, h5py, prec_rows)

        # Attach summary metrics if available
        trk_summary_path = trk_dir / "tracking_summary.json"
        if trk_summary_path.exists():
            try:
                ts = json.loads(trk_summary_path.read_text())
                metrics_grp = proc.create_group("metrics")
                metrics_grp.attrs["NX_class"] = "NXparameters"
                for k, v in ts.items():
                    if isinstance(v, bool):
                        metrics_grp.create_dataset(k, data=bool(v))
                    elif isinstance(v, (int, float)):
                        metrics_grp.create_dataset(k, data=v)
            except (OSError, json.JSONDecodeError):
                pass

    return dest


def export_ome_zarr_volume(
    volume: np.ndarray,
    out_store: str | Path,
    *,
    spot_labels: np.ndarray | None = None,
    label_names: dict[int, str] | None = None,
    calibration: AxisCalibration | None = None,
    multiscale_levels: int = 2,
    chunks: tuple[int, int, int] = (64, 64, 64),
    overwrite: bool = True,
) -> Path:
    """Export a 3D detector/reciprocal-space cube and optional Bragg spot labels to OME-Zarr v0.4 via ``basecoat``."""
    import basecoat

    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"export_ome_zarr_volume requires a 3D volume, got shape={arr.shape}")

    cal = calibration or AxisCalibration()
    use_spatial_grid = spot_labels is not None and cal.mu_step > 0.0 and cal.d_step > 0.0 and cal.chi_step > 0.0
    axes = cal.to_basecoat_axes(spatial_grid=use_spatial_grid)
    prov = {
        "producer": "braggtrack",
        "braggtrack_version": str(braggtrack.__version__),
        "braggtrack_calibration": cal.to_dict(),
    }
    img_vol = basecoat.Volume(arr, axes, provenance=prov)

    if spot_labels is None:
        return basecoat.write_ome_zarr(
            img_vol,
            out_store,
            multiscale_levels=int(multiscale_levels),
            chunks=chunks,
            overwrite=overwrite,
        )

    lbl_arr = np.asarray(spot_labels)
    if lbl_arr.shape != arr.shape:
        raise ValueError(f"spot_labels shape {lbl_arr.shape} must match volume shape {arr.shape}")
    if not np.issubdtype(lbl_arr.dtype, np.integer):
        lbl_arr = lbl_arr.astype(np.int32)

    if not use_spatial_grid:
        # Negative rocking-scan steps cannot enter VolumeSet spatial pyramids; normalize positive grid scales and preserve exact signed steps in provenance.
        spatial_axes = (
            basecoat.AxisSpec(
                name="mu",
                kind="space",
                unit="deg" if cal.calibrated else "voxel",
                scale=abs(cal.mu_step),
                offset=cal.mu_origin,
            ),
            basecoat.AxisSpec(
                name="d",
                kind="space",
                unit="angstrom" if cal.calibrated else "voxel",
                scale=abs(cal.d_step),
                offset=cal.d_origin,
            ),
            basecoat.AxisSpec(
                name="chi",
                kind="space",
                unit="deg" if cal.calibrated else "voxel",
                scale=abs(cal.chi_step),
                offset=cal.chi_origin,
            ),
        )
        img_vol = basecoat.Volume(arr, spatial_axes, provenance=prov)
        axes = spatial_axes

    lbl_vol = basecoat.Volume(lbl_arr, axes, label_names=label_names, provenance=prov)
    vset = basecoat.VolumeSet(img_vol, labels={"spots": lbl_vol})
    return basecoat.write_ome_zarr(
        vset,
        out_store,
        multiscale_levels=int(multiscale_levels),
        chunks=chunks,
        overwrite=overwrite,
    )

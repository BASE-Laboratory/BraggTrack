"""Self-contained offline interactive HTML 4D Lineage, Kinematics & Fracture Explorer."""

from __future__ import annotations

import base64
import csv
import html
import json
import struct
import zlib
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.io import (
    LazyVolume,
    MissingH5DependencyError,
    discover_operando_scans,
    open_lazy_volume,
    resolve_dataset_root,
)
from braggtrack.semantic.mips import crop_spot_cube, orthogonal_mips
from braggtrack.tracking.kinematics import compute_grain_kinematics


def _encode_rgb_png_base64(rgb: np.ndarray) -> str:
    """Encode a ``(H, W, 3)`` ``uint8`` RGB array to a base64 PNG data URI using stdlib ``zlib``."""
    h, w, _ = rgb.shape
    raw_rows = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw_rows, 6))
        + _chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _mip_to_magma_like_rgb(mip: np.ndarray, size: int = 44) -> np.ndarray:
    """Nearest-neighbor resize a 2D MIP to ``(size, size)`` and apply a high-contrast magma-like LUT."""
    arr = np.asarray(mip, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    yi = np.linspace(0, arr.shape[0] - 1, size).astype(np.intp)
    xi = np.linspace(0, arr.shape[1] - 1, size).astype(np.intp)
    resized = arr[np.ix_(yi, xi)]
    lo = float(np.min(resized))
    hi = float(np.percentile(resized, 99.5)) if resized.size > 1 else float(np.max(resized))
    if hi <= lo:
        hi = lo + 1.0
    t = np.clip((resized - lo) / (hi - lo), 0.0, 1.0)
    # Smooth 4-stop magma-like ramp: #0b0924 -> #6b1d5c -> #dc493a -> #f9e784
    r = np.clip(11.0 + 235.0 * (t**0.75), 0.0, 255.0).astype(np.uint8)
    g = np.clip(9.0 + 222.0 * (t**1.65), 0.0, 255.0).astype(np.uint8)
    b = np.clip(36.0 + 140.0 * np.sin(np.pi * np.clip(t * 0.85, 0.0, 1.0)) + 80.0 * (t**3), 0.0, 255.0).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _stitch_three_mips_b64(mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray, tile: int = 40) -> str:
    """Stitch ``(d, chi) | (mu, d) | (mu, chi)`` MIPs horizontally with 2-px dividers into one base64 PNG."""
    t1 = _mip_to_magma_like_rgb(mip_mu, size=tile)
    t2 = _mip_to_magma_like_rgb(mip_chi, size=tile)
    t3 = _mip_to_magma_like_rgb(mip_d, size=tile)
    sep = np.full((tile, 2, 3), 30, dtype=np.uint8)
    strip = np.concatenate([t1, sep, t2, sep, t3], axis=1)
    return _encode_rgb_png_base64(strip)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _collect_mip_thumbnails(
    seg_dir: Path,
    dataset_root: Path | None,
    track_rows: list[dict[str, Any]],
    priority_tids: list[int],
    *,
    max_tracks: int,
) -> dict[str, str]:
    """Extract compact 3-view MIP strips keyed by ``'{track_id}:{scan_idx}'`` via lazy HDF5 ROI reads."""
    if not seg_dir.is_dir() or dataset_root is None or not dataset_root.is_dir():
        return {}
    try:
        scans_fs = discover_operando_scans(dataset_root)
    except Exception:
        return {}
    if not scans_fs:
        return {}

    scan_dirs = sorted(d for d in seg_dir.iterdir() if d.is_dir() and d.name.startswith("scan"))
    scan_by_name = {s.scan_name: s for s in scans_fs}
    selected = set(priority_tids[:max_tracks])
    wanted_by_scan: dict[int, list[dict[str, Any]]] = {}
    for r in track_rows:
        tid = int(float(r["track_id"]))
        if tid in selected and "bbox_min_z" in r and "label" in r:
            s_idx = int(float(r["scan_idx"]))
            wanted_by_scan.setdefault(s_idx, []).append(r)

    thumbs: dict[str, str] = {}
    with ExitStack() as stack:
        for s_idx, obs_list in wanted_by_scan.items():
            if s_idx >= len(scan_dirs):
                continue
            sd = scan_dirs[s_idx]
            lab_path = sd / "labels.npz"
            sf = scan_by_name.get(sd.name)
            if not lab_path.exists() or sf is None:
                continue
            labels = np.load(lab_path)["labels"]
            try:
                lazy_vol: LazyVolume = stack.enter_context(open_lazy_volume(sf.path))
            except (MissingH5DependencyError, KeyError, ValueError, OSError):
                continue
            if lazy_vol.shape != labels.shape:
                continue
            for obs in obs_list:
                lid = int(float(obs["label"]))
                tid = int(float(obs["track_id"]))
                masked_cube, _ = crop_spot_cube(lazy_vol, labels, lid, obs, margin=2)
                m_mu, m_chi, m_d = orthogonal_mips(masked_cube)
                thumbs[f"{tid}:{s_idx}"] = _stitch_three_mips_b64(m_mu, m_chi, m_d, tile=38)
    return thumbs


def build_explorer_payload(
    artifact_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    operando_log: str | Path | None = None,
    max_mip_tracks: int = 16,
) -> dict[str, Any]:
    """Assemble structured JSON payload for the interactive 4D explorer from pipeline/tracking outputs."""
    root = Path(artifact_dir)
    trk_dir = root / "tracking" if (root / "tracking").is_dir() else root
    evt_dir = root / "events" if (root / "events").is_dir() else trk_dir
    seg_dir = root / "segmentation" if (root / "segmentation").is_dir() else root.parent / "segmentation"

    manifest = _read_json(root / "run_manifest.json") or {}
    trk_summary = _read_json(trk_dir / "tracking_summary.json") or {}
    evt_summary = _read_json(evt_dir / "events_summary.json") or {}

    ds_root: Path | None = None
    if dataset_root is not None:
        ds_root = resolve_dataset_root(str(dataset_root))
    elif manifest.get("dataset_root"):
        cand = Path(str(manifest["dataset_root"]))
        if cand.is_dir():
            ds_root = cand

    raw_tracks = _read_csv(evt_dir / "tracks.csv") or _read_csv(trk_dir / "tracks.csv")
    raw_events = _read_csv(evt_dir / "events.csv") or _read_csv(trk_dir / "events.csv")
    raw_precursors = _read_csv(evt_dir / "precursors.csv") or _read_csv(trk_dir / "precursors.csv")
    raw_prec_obs = _read_csv(evt_dir / "precursor_observations.csv")

    track_rows: list[dict[str, Any]] = []
    for r in raw_tracks:
        parsed: dict[str, Any] = {}
        for k, v in r.items():
            if k in ("event", "lineage_role", "parent_track_ids"):
                parsed[k] = v
            elif k in ("track_id", "scan_idx", "spot_idx", "label", "voxel_count"):
                parsed[k] = int(float(v)) if v != "" else 0
            else:
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
        track_rows.append(parsed)

    gks = compute_grain_kinematics(track_rows) if track_rows else []
    gk_by_tid = {g.track_id: g for g in gks}

    prec_obs_by_key: dict[tuple[int, int], dict[str, float]] = {}
    for r in raw_prec_obs:
        key = (int(float(r["track_id"])), int(float(r["scan_idx"])))
        prec_obs_by_key[key] = {
            "score": float(r.get("precursor_score", 0.0)),
            "anisotropy_term": float(r.get("anisotropy_term", 0.0)),
            "spread_term": float(r.get("spread_term", 0.0)),
            "roi_dispersion_term": float(r.get("roi_dispersion_term", 0.0)),
            "subpeaks": int(float(r.get("roi_subpeak_count", 1))),
        }

    events_list: list[dict[str, Any]] = []
    split_tids: set[int] = set()
    merge_tids: set[int] = set()
    for r in raw_events:
        ptids = [int(x) for x in str(r.get("parent_track_ids", "")).split(";") if x.strip()]
        ctids = [int(x) for x in str(r.get("child_track_ids", "")).split(";") if x.strip()]
        etype = str(r.get("event_type", ""))
        if etype == "split":
            split_tids.update(ptids)
            split_tids.update(ctids)
        elif etype == "merge":
            merge_tids.update(ptids)
            merge_tids.update(ctids)
        events_list.append(
            {
                "event_type": etype,
                "split_subtype": str(r.get("split_subtype", "")),
                "parent_scan": int(float(r.get("scan_idx_from", r.get("parent_scan", 0)))),
                "child_scan": int(float(r.get("scan_idx_to", r.get("child_scan", 1)))),
                "parent_track_ids": ptids,
                "child_track_ids": ctids,
                "intensity_ratio": round(float(r.get("intensity_ratio", 1.0)), 3),
                "centroid_distance": round(float(r.get("mean_distance", r.get("centroid_distance", 0.0))), 3),
                "confidence": round(float(r.get("confidence", 0.0)), 3),
            }
        )

    precursors_list: list[dict[str, Any]] = []
    prec_tids: set[int] = set()
    for r in raw_precursors:
        tid = int(float(r["track_id"]))
        prec_tids.add(tid)
        precursors_list.append(
            {
                "rank": int(float(r.get("rank", 0))),
                "track_id": tid,
                "onset_scan": int(float(r.get("onset_scan", 0))),
                "peak_scan": int(float(r.get("peak_scan", 0))),
                "persistence_scans": int(float(r.get("persistence_scans", 0))),
                "peak_score": round(float(r.get("peak_score", 0.0)), 4),
                "max_anisotropy": round(float(r.get("max_anisotropy", 1.0)), 3),
                "max_subpeak_count": int(float(r.get("max_subpeak_count", 1))),
                "confidence": round(float(r.get("confidence", 0.0)), 4),
                "subsequent_split": str(r.get("subsequent_split", "")).lower() == "true",
            }
        )

    by_tid: dict[int, list[dict[str, Any]]] = {}
    for r in track_rows:
        by_tid.setdefault(int(r["track_id"]), []).append(r)

    n_scans = int(trk_summary.get("n_scans") or evt_summary.get("n_scans") or 0)
    if n_scans == 0 and track_rows:
        n_scans = max(int(r["scan_idx"]) for r in track_rows) + 1

    # Load synchronized operando stimulus from explicit log path or saved artifact JSON
    stimuli_list: list[dict[str, Any]] = []
    saved_stim = _read_json(root / "operando_stimulus.json")
    if operando_log is not None:
        from braggtrack.io import BeamlineAdapter, load_operando_log, synchronize_to_scans

        op_log = load_operando_log(operando_log)
        if ds_root is not None and ds_root.is_dir():
            try:
                seq = BeamlineAdapter(ds_root).build_sequence()
                pts = (
                    synchronize_to_scans(op_log, seq)
                    if len(seq.scans) == n_scans
                    else synchronize_to_scans(op_log, n_scans)
                )
            except Exception:
                pts = synchronize_to_scans(op_log, n_scans)
        else:
            pts = synchronize_to_scans(op_log, n_scans)
        stimuli_list = [p.to_dict() for p in pts]
    elif isinstance(saved_stim, dict) and isinstance(saved_stim.get("scans"), list):
        stimuli_list = list(saved_stim["scans"])
    stim_by_scan = {int(s["scan_idx"]): s for s in stimuli_list if "scan_idx" in s}

    priority_tids = sorted(
        by_tid.keys(),
        key=lambda tid: (
            0 if (tid in split_tids or tid in merge_tids or tid in prec_tids) else 1,
            -len(by_tid[tid]),
            -max(float(x.get("integrated_intensity", 0.0)) for x in by_tid[tid]),
            tid,
        ),
    )
    thumbs = _collect_mip_thumbnails(seg_dir, ds_root, track_rows, priority_tids, max_tracks=max_mip_tracks)

    tracks_payload: list[dict[str, Any]] = []
    for tid in sorted(by_tid.keys(), key=lambda t: (float(by_tid[t][0]["centroid_d"]), t)):
        obs_sorted = sorted(by_tid[tid], key=lambda x: int(x["scan_idx"]))
        gk = gk_by_tid.get(tid)
        scans = [int(x["scan_idx"]) for x in obs_sorted]
        has_gap = any(scans[i + 1] - scans[i] > 1 for i in range(len(scans) - 1))
        is_full = n_scans > 0 and min(scans) == 0 and max(scans) >= n_scans - 1

        obs_items: list[dict[str, Any]] = []
        for idx_o, o in enumerate(obs_sorted):
            s_i = int(o["scan_idx"])
            strain_pct = float(gk.strain[idx_o] * 100.0) if gk is not None and idx_o < len(gk.strain) else 0.0
            lin_strain_pct = (
                float(gk.lineage_strain[idx_o] * 100.0)
                if gk is not None and idx_o < len(gk.lineage_strain)
                else strain_pct
            )
            strain_err_pct = (
                float(gk.lineage_strain_err[idx_o] * 100.0)
                if gk is not None and idx_o < len(gk.lineage_strain_err)
                else 0.0
            )
            miso = (
                float(gk.misorientation_total[idx_o])
                if gk is not None and idx_o < len(gk.misorientation_total)
                else 0.0
            )
            aniso = float(gk.anisotropy[idx_o]) if gk is not None and idx_o < len(gk.anisotropy) else 1.0
            po = prec_obs_by_key.get((tid, s_i), {})
            stim = stim_by_scan.get(s_i, {})
            obs_items.append(
                {
                    "scan_idx": s_i,
                    "label": int(o.get("label", 0)),
                    "mu": round(float(o["centroid_mu"]), 4),
                    "chi": round(float(o["centroid_chi"]), 4),
                    "d": round(float(o["centroid_d"]), 4),
                    "intensity": round(float(o.get("integrated_intensity", 0.0)), 2),
                    "voxels": int(o.get("voxel_count", 0)),
                    "strain_pct": round(strain_pct, 4),
                    "lineage_strain_pct": round(lin_strain_pct, 4),
                    "strain_err_pct": round(strain_err_pct, 4),
                    "voltage_v": stim.get("voltage_v"),
                    "dq_dv": stim.get("dq_dv"),
                    "stress_mpa": stim.get("stress_mpa"),
                    "temperature_c": stim.get("temperature_c"),
                    "misorientation": round(miso, 4),
                    "anisotropy": round(aniso, 3),
                    "precursor_score": round(float(po.get("score", 0.0)), 4),
                    "roi_dispersion_term": round(float(po.get("roi_dispersion_term", 0.0)), 4),
                    "subpeaks": int(po.get("subpeaks", 1)),
                    "mip_b64": thumbs.get(f"{tid}:{s_i}"),
                }
            )

        tracks_payload.append(
            {
                "track_id": tid,
                "root_track_id": gk.root_track_id if gk is not None and gk.root_track_id > 0 else tid,
                "is_full": is_full,
                "has_gap": has_gap,
                "is_split": tid in split_tids,
                "is_merge": tid in merge_tids,
                "is_precursor": tid in prec_tids,
                "max_abs_strain_pct": round(max((abs(x["lineage_strain_pct"]) for x in obs_items), default=0.0), 4),
                "max_precursor": round(max((x["precursor_score"] for x in obs_items), default=0.0), 4),
                "observations": obs_items,
            }
        )

    scan_names = (
        trk_summary.get("scan_names") or evt_summary.get("scan_names") or [f"scan{i + 1:04d}" for i in range(n_scans)]
    )
    return {
        "title": f"BraggTrack 4D Explorer — {Path(manifest.get('dataset_root') or root).name}",
        "schema_version": "report.v1",
        "n_scans": n_scans,
        "scan_names": scan_names,
        "calibrated_axes": bool(trk_summary.get("calibrated_axes", False)),
        "backend": evt_summary.get("backend") or manifest.get("config", {}).get("backend") or "n/a",
        "model": evt_summary.get("model") or manifest.get("config", {}).get("model") or "n/a",
        "braggtrack_version": manifest.get("braggtrack_version", "0.5.0"),
        "total_duration_sec": manifest.get("total_duration_sec"),
        "kpis": {
            "total_tracks": len(tracks_payload),
            "full_length_tracks": sum(1 for t in tracks_payload if t["is_full"]),
            "gap_closed_count": int(trk_summary.get("gap_closed_count", evt_summary.get("gap_closed_count", 0))),
            "split_count": sum(1 for e in events_list if e["event_type"] == "split"),
            "merge_count": sum(1 for e in events_list if e["event_type"] == "merge"),
            "precursor_count": len(precursors_list),
        },
        "stimuli": stimuli_list,
        "tracks": tracks_payload,
        "events": events_list,
        "precursors": precursors_list,
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline';" />
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0f172a; --card: #1e293b; --border: #334155; --fg: #f8fafc; --muted: #94a3b8;
    --primary: #38bdf8; --split: #f97316; --merge: #10b981; --gap: #eab308; --prec: #ec4899;
  }
  body.light {
    --bg: #f8fafc; --card: #ffffff; --border: #cbd5e1; --fg: #0f172a; --muted: #475569;
    --primary: #0284c7; --split: #ea580c; --merge: #059669; --gap: #ca8a04; --prec: #db2777;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 18px 22px; background: var(--bg); color: var(--fg); font: 13.5px/1.45 system-ui, -apple-system, sans-serif; }
  .hdr { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
  .hdr h1 { margin: 0; font-size: 18px; font-weight: 700; }
  .meta { font-size: 12px; color: var(--muted); }
  .btn { cursor: pointer; border: 1px solid var(--border); background: var(--card); color: var(--fg); padding: 5px 11px; border-radius: 6px; font-size: 12px; }
  .btn.active { border-color: var(--primary); background: rgba(56,189,248,0.16); color: var(--primary); font-weight: 600; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 10px; margin-bottom: 14px; }
  .kpi { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .kpi .lbl { font-size: 11.5px; color: var(--muted); }
  .kpi .val { font-size: 20px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .grid-main { display: grid; grid-template-columns: 1.45fr 1fr; gap: 14px; margin-bottom: 14px; }
  @media (max-width: 980px) { .grid-main { grid-template-columns: 1fr; } }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .card h2 { margin: 0 0 8px 0; font-size: 14px; font-weight: 650; display: flex; justify-content: space-between; align-items: center; }
  .pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .dag-wrap { width: 100%; overflow: auto; max-height: 430px; border: 1px solid var(--border); border-radius: 6px; }
  svg text { fill: var(--muted); font-size: 11px; user-select: none; }
  .node-dot { cursor: pointer; transition: r 0.12s ease; }
  .node-dot:hover { r: 7px; }
  .mips-row { display: flex; gap: 8px; overflow-x: auto; padding: 6px 0; }
  .mip-item { flex: 0 0 auto; border: 1px solid var(--border); border-radius: 6px; padding: 4px; text-align: center; background: rgba(0,0,0,0.18); }
  .mip-item img { display: block; height: 42px; image-rendering: pixelated; border-radius: 3px; }
  .mip-item span { font-size: 11px; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  th { color: var(--muted); font-weight: 600; }
  tr.clickable:hover { background: rgba(56,189,248,0.09); cursor: pointer; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .b-split { background: rgba(249,115,22,0.18); color: var(--split); }
  .b-merge { background: rgba(16,185,129,0.18); color: var(--merge); }
  .b-prec { background: rgba(236,72,153,0.18); color: var(--prec); }
</style>
</head>
<body>
<div class="hdr">
  <div>
    <h1 id="page-title">BraggTrack 4D Explorer</h1>
    <div class="meta" id="page-meta"></div>
  </div>
  <button class="btn" id="theme-btn" type="button">Toggle Light / Dark</button>
</div>
<div class="kpis" id="kpi-strip"></div>
<div class="grid-main">
  <div class="card">
    <h2>
      <span>4D Trajectory &amp; Lineage DAG</span>
      <span class="meta">Click any track node to inspect</span>
    </h2>
    <div class="pills" id="filter-pills"></div>
    <div class="dag-wrap" id="dag-container"></div>
  </div>
  <div class="card">
    <h2>
      <span id="insp-title">Track Inspector</span>
      <span class="meta" id="insp-badges"></span>
    </h2>
    <div id="insp-charts"></div>
    <div id="insp-mips"></div>
  </div>
</div>
<div class="grid-main">
  <div class="card">
    <h2>Split &amp; Merge Lineage Events</h2>
    <div id="events-table"></div>
  </div>
  <div class="card">
    <h2>Persistent Fracture Precursor Candidates</h2>
    <div id="prec-table"></div>
  </div>
</div>
<script id="bt-data" type="application/json">__PAYLOAD_JSON__</script>
<script>
(() => {
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const DATA = JSON.parse(document.getElementById('bt-data').textContent);
  document.getElementById('page-title').textContent = DATA.title;
  const dur = DATA.total_duration_sec != null ? ` | Runtime: ${DATA.total_duration_sec}s` : '';
  document.getElementById('page-meta').textContent =
    `Scans: ${DATA.n_scans} | Axes: ${DATA.calibrated_axes ? 'Calibrated (mu, chi, d)' : 'Voxel'} | Backend: ${DATA.backend} (${DATA.model}) | v${DATA.braggtrack_version}${dur}`;

  document.getElementById('theme-btn').addEventListener('click', () => {
    document.body.classList.toggle('light');
  });

  const kpis = [
    ['Total Tracks', DATA.kpis.total_tracks, 'var(--fg)'],
    ['Full-Span Tracks', DATA.kpis.full_length_tracks, 'var(--primary)'],
    ['Gap Closures', DATA.kpis.gap_closed_count, 'var(--gap)'],
    ['1 -> 2+ Splits', DATA.kpis.split_count, 'var(--split)'],
    ['2+ -> 1 Merges', DATA.kpis.merge_count, 'var(--merge)'],
    ['Precursor Candidates', DATA.kpis.precursor_count, 'var(--prec)'],
  ];
  document.getElementById('kpi-strip').innerHTML = kpis.map(([l, v, c]) =>
    `<div class="kpi"><div class="lbl">${esc(l)}</div><div class="val" style="color:${esc(c)}">${esc(v)}</div></div>`
  ).join('');

  const filters = [
    ['all', 'All Tracks'],
    ['full', 'Full-Span'],
    ['split', 'Splits & Daughters'],
    ['merge', 'Merges'],
    ['precursor', 'Precursors'],
    ['gap', 'Gap-Closed'],
  ];
  let activeFilter = 'all';
  let selectedTid = DATA.tracks.length ? (
    (DATA.precursors[0] && DATA.precursors[0].track_id) ||
    (DATA.tracks.find(t => t.is_split) || DATA.tracks[0]).track_id
  ) : null;

  function filteredTracks() {
    return DATA.tracks.filter(t => {
      if (activeFilter === 'full') return t.is_full;
      if (activeFilter === 'split') return t.is_split;
      if (activeFilter === 'merge') return t.is_merge;
      if (activeFilter === 'precursor') return t.is_precursor;
      if (activeFilter === 'gap') return t.has_gap;
      return true;
    });
  }

  function renderPills() {
    const el = document.getElementById('filter-pills');
    el.innerHTML = filters.map(([id, label]) =>
      `<button class="btn ${activeFilter === id ? 'active' : ''}" data-f="${esc(id)}">${esc(label)}</button>`
    ).join('');
    el.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
      activeFilter = b.dataset.f;
      renderPills();
      renderDag();
    }));
  }

  function renderDag() {
    const box = document.getElementById('dag-container');
    const tracks = filteredTracks();
    if (!tracks.length) {
      box.innerHTML = '<div style="padding:18px;color:var(--muted)">No tracks match this filter.</div>';
      return;
    }
    const nScans = Math.max(1, DATA.n_scans);
    const padL = 52, padR = 26, padT = 26, rowH = 24;
    const width = Math.max(540, padL + padR + Math.max(1, nScans - 1) * 95);
    const height = padT + tracks.length * rowH + 22;
    const xOf = s => padL + (nScans > 1 ? (s / (nScans - 1)) * (width - padL - padR) : (width - padL - padR) / 2);
    const lane = new Map(tracks.map((t, idx) => [t.track_id, padT + idx * rowH]));

    let svg = `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`;
    const stimMap = new Map((DATA.stimuli || []).map(st => [st.scan_idx, st]));
    for (let s = 0; s < nScans; s++) {
      const x = xOf(s);
      const st = stimMap.get(s);
      const vTag = (st && st.voltage_v != null) ? ` (${Number(st.voltage_v).toFixed(2)}V)` : '';
      const scanLabel = esc((DATA.scan_names[s] || ('S' + s)) + vTag);
      svg += `<line x1="${x}" y1="${padT - 10}" x2="${x}" y2="${height - 16}" stroke="var(--border)" stroke-dasharray="3,3" stroke-width="1"/>`;
      svg += `<text x="${x}" y="15" text-anchor="middle">${scanLabel}</text>`;
    }

    // Event edges
    DATA.events.forEach(ev => {
      const col = ev.event_type === 'split' ? 'var(--split)' : 'var(--merge)';
      ev.parent_track_ids.forEach(ptid => {
        ev.child_track_ids.forEach(ctid => {
          if (!lane.has(ptid) || !lane.has(ctid)) return;
          svg += `<line x1="${xOf(ev.parent_scan)}" y1="${lane.get(ptid)}" x2="${xOf(ev.child_scan)}" y2="${lane.get(ctid)}" stroke="${col}" stroke-width="2.4"/>`;
        });
      });
    });

    tracks.forEach(t => {
      const y = lane.get(t.track_id);
      const isSel = t.track_id === selectedTid;
      svg += `<text x="8" y="${y + 4}" style="font-weight:${isSel ? '700' : '400'};fill:${isSel ? 'var(--primary)' : 'var(--muted)'}">T${Number(t.track_id)}</text>`;
      const obs = t.observations;
      for (let i = 0; i < obs.length - 1; i++) {
        const s0 = obs[i].scan_idx, s1 = obs[i + 1].scan_idx;
        const gap = s1 - s0 > 1;
        const col = gap ? 'var(--gap)' : (isSel ? 'var(--primary)' : '#64748b');
        svg += `<line x1="${xOf(s0)}" y1="${y}" x2="${xOf(s1)}" y2="${y}" stroke="${col}" stroke-width="${isSel ? 2.6 : 1.7}" ${gap ? 'stroke-dasharray="5,3"' : ''}/>`;
      }
      obs.forEach(o => {
        let fill = 'var(--primary)';
        if (t.is_split) fill = 'var(--split)';
        else if (t.is_merge) fill = 'var(--merge)';
        else if (t.is_precursor) fill = 'var(--prec)';
        const r = isSel ? 6 : 4.6;
        const linS = o.lineage_strain_pct != null ? o.lineage_strain_pct : o.strain_pct;
        const errS = o.strain_err_pct ? `±${o.strain_err_pct}` : '';
        const vInfo = o.voltage_v != null ? ` | V=${Number(o.voltage_v).toFixed(3)}V` : '';
        const tip = esc(`T${t.track_id} @ scan ${o.scan_idx}${vInfo} | root_strain=${linS}${errS}% | I=${o.intensity} | prec=${o.precursor_score}`);
        svg += `<circle class="node-dot" data-tid="${Number(t.track_id)}" cx="${xOf(o.scan_idx)}" cy="${y}" r="${r}" fill="${fill}" stroke="#0f172a" stroke-width="1">
          <title>${tip}</title>
        </circle>`;
      });
    });
    svg += '</svg>';
    box.innerHTML = svg;
    box.querySelectorAll('.node-dot').forEach(el => el.addEventListener('click', () => {
      selectedTid = Number(el.dataset.tid);
      renderDag();
      renderInspector();
    }));
  }

  function sparkSvg(points, color, yLabel) {
    const w = 360, h = 92, pL = 42, pR = 12, pT = 14, pB = 20;
    if (!points.length) return '';
    const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
    const minX = 0, maxX = Math.max(1, DATA.n_scans - 1);
    let minY = Math.min(...ys), maxY = Math.max(...ys);
    if (Math.abs(maxY - minY) < 1e-6) { minY -= 0.5; maxY += 0.5; }
    const xMap = x => pL + ((x - minX) / (maxX - minX)) * (w - pL - pR);
    const yMap = y => h - pB - ((y - minY) / (maxY - minY)) * (h - pT - pB);
    const poly = points.map(([x, y]) => `${xMap(x).toFixed(1)},${yMap(y).toFixed(1)}`).join(' ');
    const dots = points.map(([x, y]) => `<circle cx="${xMap(x).toFixed(1)}" cy="${yMap(y).toFixed(1)}" r="3.3" fill="${esc(color)}"/>`).join('');
    return `<div style="margin-bottom:8px">
      <div class="meta" style="margin-bottom:2px">${esc(yLabel)} (min ${minY.toFixed(3)}, max ${maxY.toFixed(3)})</div>
      <svg width="100%" viewBox="0 0 ${w} ${h}" style="border:1px solid var(--border);border-radius:6px">
        <text x="4" y="${pT + 6}">${maxY.toFixed(2)}</text>
        <text x="4" y="${h - pB}">${minY.toFixed(2)}</text>
        <polyline fill="none" stroke="${esc(color)}" stroke-width="2" points="${poly}"/>
        ${dots}
      </svg>
    </div>`;
  }

  function renderInspector() {
    const t = DATA.tracks.find(x => x.track_id === selectedTid);
    const titleEl = document.getElementById('insp-title');
    const badgeEl = document.getElementById('insp-badges');
    const chartsEl = document.getElementById('insp-charts');
    const mipsEl = document.getElementById('insp-mips');
    if (!t) {
      titleEl.textContent = 'Track Inspector';
      badgeEl.innerHTML = '';
      chartsEl.innerHTML = '<div class="meta">Select a track node on the left.</div>';
      mipsEl.innerHTML = '';
      return;
    }
    const rootNote = (t.root_track_id && t.root_track_id !== t.track_id) ? ` [root T${t.root_track_id}]` : '';
    titleEl.textContent = `Track T${t.track_id}${rootNote} (${t.observations.length} observations)`;
    const badges = [];
    if (t.is_split) badges.push('<span class="badge b-split">Split Lineage</span>');
    if (t.is_merge) badges.push('<span class="badge b-merge">Merge Lineage</span>');
    if (t.is_precursor) badges.push('<span class="badge b-prec">Precursor</span>');
    badgeEl.innerHTML = badges.join(' ');

    const vPts = (DATA.stimuli || []).filter(s => s.voltage_v != null).map(s => [s.scan_idx, Number(s.voltage_v)]);
    const strainPts = t.observations.map(o => [o.scan_idx, o.lineage_strain_pct != null ? o.lineage_strain_pct : o.strain_pct]);
    const intPts = t.observations.map(o => [o.scan_idx, o.intensity]);
    const precPts = t.observations.map(o => [o.scan_idx, o.precursor_score]);
    chartsEl.innerHTML =
      (vPts.length ? sparkSvg(vPts, 'var(--gap)', 'Operando Potential E (V)') : '') +
      sparkSvg(strainPts, 'var(--primary)', 'Ancestor-Referenced Strain (%)') +
      sparkSvg(intPts, 'var(--merge)', 'Integrated Intensity') +
      sparkSvg(precPts, 'var(--prec)', 'Fracture Precursor Score [0, 1]');

    const mips = t.observations.filter(o => o.mip_b64 && String(o.mip_b64).startsWith('data:image/png;base64,'));
    if (mips.length) {
      mipsEl.innerHTML = `<div class="meta" style="margin-bottom:3px">3-View Orthogonal MIPs (d-chi | mu-d | mu-chi)</div>
        <div class="mips-row">` + mips.map(o =>
        `<div class="mip-item"><img src="${esc(o.mip_b64)}" alt="MIP S${Number(o.scan_idx)}"/><span>Scan ${Number(o.scan_idx)} (subpeaks: ${Number(o.subpeaks)})</span></div>`
      ).join('') + '</div>';
    } else {
      mipsEl.innerHTML = '';
    }
  }

  function renderTables() {
    const evEl = document.getElementById('events-table');
    if (!DATA.events.length) {
      evEl.innerHTML = '<div class="meta">No split or merge events detected.</div>';
    } else {
      evEl.innerHTML = `<table><thead><tr><th>Type</th><th>Scans</th><th>Parents -&gt; Children</th><th>I_child / I_parent</th><th>Conf</th></tr></thead><tbody>` +
        DATA.events.map(e => {
          const cls = e.event_type === 'split' ? 'b-split' : 'b-merge';
          const focusTid = Number(e.parent_track_ids[0] || e.child_track_ids[0] || 0);
          const sub = e.split_subtype ? ` <span class="meta">(${esc(e.split_subtype)})</span>` : '';
          return `<tr class="clickable" data-tid="${focusTid}">
            <td><span class="badge ${cls}">${esc(e.event_type)}</span>${sub}</td>
            <td>${Number(e.parent_scan)} -&gt; ${Number(e.child_scan)}</td>
            <td>T${e.parent_track_ids.map(Number).join(',T')} -&gt; T${e.child_track_ids.map(Number).join(',T')}</td>
            <td>${esc(e.intensity_ratio)}</td>
            <td>${esc(e.confidence)}</td>
          </tr>`;
        }).join('') + '</tbody></table>';
      evEl.querySelectorAll('tr.clickable').forEach(r => r.addEventListener('click', () => {
        selectedTid = Number(r.dataset.tid);
        activeFilter = 'all';
        renderPills(); renderDag(); renderInspector();
      }));
    }

    const prEl = document.getElementById('prec-table');
    if (!DATA.precursors.length) {
      prEl.innerHTML = '<div class="meta">No persistent fracture precursor candidates above threshold.</div>';
    } else {
      prEl.innerHTML = `<table><thead><tr><th>Rank</th><th>Track</th><th>Onset..Peak</th><th>Persist</th><th>Peak Score</th><th>Subpeaks</th><th>Split?</th></tr></thead><tbody>` +
        DATA.precursors.map(p => `<tr class="clickable" data-tid="${Number(p.track_id)}">
          <td>#${Number(p.rank)}</td>
          <td><strong>T${Number(p.track_id)}</strong></td>
          <td>S${Number(p.onset_scan)}..S${Number(p.peak_scan)}</td>
          <td>${Number(p.persistence_scans)} scans</td>
          <td>${esc(p.peak_score)}</td>
          <td>${Number(p.max_subpeak_count)}</td>
          <td>${p.subsequent_split ? '<span class="badge b-split">Yes</span>' : 'No'}</td>
        </tr>`).join('') + '</tbody></table>';
      prEl.querySelectorAll('tr.clickable').forEach(r => r.addEventListener('click', () => {
        selectedTid = Number(r.dataset.tid);
        activeFilter = 'all';
        renderPills(); renderDag(); renderInspector();
      }));
    }
  }

  renderPills();
  renderDag();
  renderInspector();
  renderTables();
})();
</script>
</body>
</html>
"""


def generate_html_report(
    artifact_dir: str | Path,
    out_html: str | Path | None = None,
    *,
    dataset_root: str | Path | None = None,
    operando_log: str | Path | None = None,
    max_mip_tracks: int = 16,
) -> Path:
    """Render a self-contained interactive HTML 4D explorer from a BraggTrack output directory."""
    root = Path(artifact_dir)
    dest = Path(out_html) if out_html is not None else (root / "report.html")
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = build_explorer_payload(
        root,
        dataset_root=dataset_root,
        operando_log=operando_log,
        max_mip_tracks=max_mip_tracks,
    )
    safe_json = json.dumps(payload).replace("</", "<\\/")
    html_doc = _HTML_TEMPLATE.replace("__TITLE__", html.escape(str(payload["title"]))).replace(
        "__PAYLOAD_JSON__", safe_json
    )
    dest.write_text(html_doc, encoding="utf-8")
    return dest

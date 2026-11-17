"""Operando external stimulus log ingestion and scan synchronization.

Supports generic CSV/TSV potentiostat/load/thermal logs as well as Biologic
EC-Lab ``.mpt`` ASCII exports (``Nb header lines : N``, ``time/s``, ``Ewe/V``,
``<I>/mA``, ``(Q-Qo)/mA.h``, ``cycle number``).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .models import ExperimentSequence


_COL_ALIASES: dict[str, tuple[str, ...]] = {
    "scan_idx": ("scan_idx", "scan", "scan_index", "frame", "frame_idx"),
    "timestamp": ("timestamp", "iso_time", "datetime", "date_time", "utc_time", "abs_time"),
    "uts": ("uts", "epoch", "epoch_s", "unix_time"),
    "time_s": ("time_s", "elapsed_s", "time/s", "time", "t_s", "seconds"),
    "voltage_v": ("voltage_v", "voltage", "ewe/v", "<ewe>/v", "ecell/v", "v_cell", "potential_v"),
    "current_ma": ("current_ma", "current", "i/ma", "<i>/ma", "current/ma"),
    "capacity_mah_g": (
        "capacity_mah_g",
        "specific_capacity",
        "capacity",
        "q_mah_g",
        "(q-qo)/ma.h",
        "q charge/ma.h",
        "q discharge/ma.h",
        "dq/ma.h",
    ),
    "dq_dv": ("dq_dv", "dq/dv", "dqdv"),
    "cycle": ("cycle", "cycle_number", "cycle number", "half_cycle"),
    "stress_mpa": ("stress_mpa", "stress", "load_mpa", "sigma_mpa"),
    "temperature_c": ("temperature_c", "temperature", "temp_c", "t_degc", "temperature/°c", "temperature/c"),
}


@dataclass(frozen=True)
class OperandoStimulusLog:
    """Time-series external operando stimulus channels (electrochemistry, load, temperature)."""

    time_s: np.ndarray
    scan_idx: np.ndarray | None = None
    epoch_s: np.ndarray | None = None
    voltage_v: np.ndarray | None = None
    current_ma: np.ndarray | None = None
    capacity_mah_g: np.ndarray | None = None
    dq_dv: np.ndarray | None = None
    cycle: np.ndarray | None = None
    stress_mpa: np.ndarray | None = None
    temperature_c: np.ndarray | None = None
    source_path: str | None = None

    @property
    def n_points(self) -> int:
        return int(len(self.time_s))


@dataclass(frozen=True)
class ScanStimulusPoint:
    """Synchronized external stimulus state evaluated at one diffraction scan."""

    scan_idx: int
    scan_name: str
    elapsed_s: float
    timestamp: str | None = None
    voltage_v: float | None = None
    current_ma: float | None = None
    capacity_mah_g: float | None = None
    dq_dv: float | None = None
    cycle: int | None = None
    stress_mpa: float | None = None
    temperature_c: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_idx": self.scan_idx,
            "scan_name": self.scan_name,
            "elapsed_s": round(float(self.elapsed_s), 4),
            "timestamp": self.timestamp,
            "voltage_v": round(float(self.voltage_v), 5) if self.voltage_v is not None else None,
            "current_ma": round(float(self.current_ma), 5) if self.current_ma is not None else None,
            "capacity_mah_g": round(float(self.capacity_mah_g), 4) if self.capacity_mah_g is not None else None,
            "dq_dv": round(float(self.dq_dv), 4) if self.dq_dv is not None else None,
            "cycle": int(self.cycle) if self.cycle is not None else None,
            "stress_mpa": round(float(self.stress_mpa), 4) if self.stress_mpa is not None else None,
            "temperature_c": round(float(self.temperature_c), 3) if self.temperature_c is not None else None,
        }


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _match_columns(headers: list[str]) -> dict[str, str]:
    norm_to_orig = {_norm_col(h): h for h in headers if h is not None}
    matched: dict[str, str] = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in norm_to_orig:
                matched[canonical] = norm_to_orig[alias]
                break
    return matched


def _parse_iso_to_epoch(val: str) -> float | None:
    s = val.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _parse_float(val: str | None) -> float:
    if val is None:
        return math.nan
    s = val.strip().replace(",", ".") if "," in val and "." not in val else val.strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def _compute_regularized_dq_dv(
    voltage_v: np.ndarray,
    capacity: np.ndarray,
    *,
    min_dv: float = 1.5e-3,
) -> np.ndarray:
    """Compute causal backward-difference ``dQ/dV`` with mild voltage-step regularization."""
    n = len(voltage_v)
    out = np.zeros(n, dtype=np.float64)
    if n < 2:
        return out

    # Smooth voltage slightly with a 3-point causal moving average to suppress potentiostat bit noise
    v_smooth = np.empty(n, dtype=np.float64)
    q_smooth = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - 2)
        v_smooth[i] = float(np.nanmean(voltage_v[lo : i + 1]))
        q_smooth[i] = float(np.nanmean(capacity[lo : i + 1]))

    for i in range(1, n):
        dv = v_smooth[i] - v_smooth[i - 1]
        dq = abs(q_smooth[i] - q_smooth[i - 1])
        if abs(dv) >= min_dv and math.isfinite(dv) and math.isfinite(dq):
            out[i] = math.copysign(dq / abs(dv), dv)
        else:
            out[i] = out[i - 1]
    out[0] = out[1]
    return out


def load_operando_log(path: str | Path) -> OperandoStimulusLog:
    """Load an external operando CSV, TSV, or Biologic EC-Lab ``.mpt`` file."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Operando stimulus log '{file_path}' is empty.")

    # Detect Biologic EC-Lab ASCII header ("Nb header lines : N")
    header_idx = 0
    for line in lines[:30]:
        m = re.match(r"^\s*Nb\s+header\s+lines\s*:\s*(\d+)", line, flags=re.IGNORECASE)
        if m:
            header_idx = max(0, int(m.group(1)) - 1)
            break
    else:
        # Skip leading comment lines starting with '#'
        while header_idx < len(lines) and (not lines[header_idx].strip() or lines[header_idx].lstrip().startswith("#")):
            header_idx += 1

    if header_idx >= len(lines):
        raise ValueError(f"Could not locate table header in operando log '{file_path}'.")

    body_lines = [ln for ln in lines[header_idx:] if ln.strip()]
    header_line = body_lines[0]
    delimiter = "\t" if "\t" in header_line else (";" if ";" in header_line and "," not in header_line else ",")

    reader = csv.DictReader(body_lines, delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError(f"No column headers found in operando log '{file_path}'.")

    col_map = _match_columns(list(reader.fieldnames))
    raw_rows = list(reader)
    if not raw_rows:
        raise ValueError(f"Operando stimulus log '{file_path}' contains zero data rows.")

    n = len(raw_rows)

    def _extract_num(canon: str) -> np.ndarray | None:
        col = col_map.get(canon)
        if col is None:
            return None
        arr = np.array([_parse_float(r.get(col)) for r in raw_rows], dtype=np.float64)
        if np.all(np.isnan(arr)):
            return None
        return arr

    scan_arr = _extract_num("scan_idx")
    time_arr = _extract_num("time_s")
    uts_arr = _extract_num("uts")

    epoch_arr: np.ndarray | None = uts_arr
    if epoch_arr is None and "timestamp" in col_map:
        ts_col = col_map["timestamp"]
        parsed_ep = [_parse_iso_to_epoch(str(r.get(ts_col, ""))) for r in raw_rows]
        if any(x is not None for x in parsed_ep):
            epoch_arr = np.array([x if x is not None else math.nan for x in parsed_ep], dtype=np.float64)

    if time_arr is None:
        if epoch_arr is not None:
            valid0 = float(epoch_arr[np.isfinite(epoch_arr)][0])
            time_arr = epoch_arr - valid0
        elif scan_arr is not None:
            time_arr = scan_arr.astype(np.float64)
        else:
            time_arr = np.arange(n, dtype=np.float64)

    voltage_v = _extract_num("voltage_v")
    current_ma = _extract_num("current_ma")
    capacity = _extract_num("capacity_mah_g")
    dq_dv = _extract_num("dq_dv")
    cycle = _extract_num("cycle")
    stress_mpa = _extract_num("stress_mpa")
    temperature_c = _extract_num("temperature_c")

    # Derive Coulombic capacity (mAh) when current_mA & time_s exist without explicit capacity column
    if capacity is None and current_ma is not None and len(time_arr) >= 2:
        dt_h = np.diff(time_arr, prepend=time_arr[0]) / 3600.0
        dt_h = np.where(np.isfinite(dt_h) & (dt_h > 0.0), dt_h, 0.0)
        i_clean = np.where(np.isfinite(current_ma), np.abs(current_ma), 0.0)
        capacity = np.cumsum(i_clean * dt_h)

    if dq_dv is None and voltage_v is not None and capacity is not None:
        dq_dv = _compute_regularized_dq_dv(voltage_v, capacity)

    return OperandoStimulusLog(
        time_s=time_arr,
        scan_idx=scan_arr,
        epoch_s=epoch_arr,
        voltage_v=voltage_v,
        current_ma=current_ma,
        capacity_mah_g=capacity,
        dq_dv=dq_dv,
        cycle=cycle,
        stress_mpa=stress_mpa,
        temperature_c=temperature_c,
        source_path=str(file_path),
    )


def _interp_optional(x_target: np.ndarray, x_src: np.ndarray, y_src: np.ndarray | None) -> np.ndarray | None:
    if y_src is None:
        return None
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    if int(np.sum(mask)) == 0:
        return None
    xs = x_src[mask]
    ys = y_src[mask]
    order = np.argsort(xs, kind="mergesort")
    xs_sorted = xs[order]
    ys_sorted = ys[order]
    return np.interp(x_target, xs_sorted, ys_sorted)


def synchronize_to_scans(
    log: OperandoStimulusLog,
    sequence_or_count: ExperimentSequence | int,
    *,
    start_offset_s: float = 0.0,
    scan_cadence_s: float | None = None,
) -> list[ScanStimulusPoint]:
    """Interpolate external stimulus channels onto each diffraction scan's midpoint time or index."""
    if isinstance(sequence_or_count, int):
        n_scans = max(0, sequence_or_count)
        scan_names = [f"scan{i + 1:04d}" for i in range(n_scans)]
        scan_epochs: list[float | None] = [None] * n_scans
        scan_isos: list[str | None] = [None] * n_scans
    else:
        scans = sequence_or_count.scans
        n_scans = len(scans)
        scan_names = [s.scan_name for s in scans]
        scan_epochs = []
        scan_isos = []
        for s in scans:
            if s.start_time is not None and s.end_time is not None:
                st = s.start_time if s.start_time.tzinfo else s.start_time.replace(tzinfo=timezone.utc)
                et = s.end_time if s.end_time.tzinfo else s.end_time.replace(tzinfo=timezone.utc)
                mid_ep = 0.5 * (st.timestamp() + et.timestamp())
                scan_epochs.append(float(mid_ep))
                scan_isos.append(st.isoformat())
            elif s.start_time is not None:
                st = s.start_time if s.start_time.tzinfo else s.start_time.replace(tzinfo=timezone.utc)
                scan_epochs.append(float(st.timestamp()))
                scan_isos.append(st.isoformat())
            else:
                scan_epochs.append(None)
                scan_isos.append(None)

    if n_scans == 0:
        return []

    idx_vec = np.arange(n_scans, dtype=np.float64)

    # Choose synchronization coordinate: explicit scan_idx > absolute epoch > elapsed seconds
    if log.scan_idx is not None and np.any(np.isfinite(log.scan_idx)):
        x_src = log.scan_idx
        x_query = idx_vec
        elapsed_s = _interp_optional(x_query, x_src, log.time_s)
        if elapsed_s is None:
            elapsed_s = idx_vec * (scan_cadence_s or 60.0)
    elif log.epoch_s is not None and all(ep is not None for ep in scan_epochs):
        x_src = log.epoch_s
        x_query = np.array([float(ep) for ep in scan_epochs if ep is not None], dtype=np.float64)
        elapsed_s = x_query - x_query[0]
    elif all(ep is not None for ep in scan_epochs):
        ep_arr = np.array([float(ep) for ep in scan_epochs if ep is not None], dtype=np.float64)
        elapsed_s = (ep_arr - ep_arr[0]) + start_offset_s
        x_src = log.time_s
        x_query = elapsed_s
    else:
        if scan_cadence_s is not None:
            elapsed_s = start_offset_s + idx_vec * scan_cadence_s
        else:
            finite_t = log.time_s[np.isfinite(log.time_s)]
            t_min = float(finite_t[0]) if len(finite_t) else 0.0
            t_max = float(finite_t[-1]) if len(finite_t) else float(max(0, n_scans - 1) * 60.0)
            elapsed_s = np.linspace(t_min, t_max, n_scans) if n_scans > 1 else np.array([t_min])
        x_src = log.time_s
        x_query = elapsed_s

    v_interp = _interp_optional(x_query, x_src, log.voltage_v)
    i_interp = _interp_optional(x_query, x_src, log.current_ma)
    q_interp = _interp_optional(x_query, x_src, log.capacity_mah_g)
    dqdv_interp = _interp_optional(x_query, x_src, log.dq_dv)
    cyc_interp = _interp_optional(x_query, x_src, log.cycle)
    sig_interp = _interp_optional(x_query, x_src, log.stress_mpa)
    tmp_interp = _interp_optional(x_query, x_src, log.temperature_c)

    points: list[ScanStimulusPoint] = []
    for k in range(n_scans):
        points.append(
            ScanStimulusPoint(
                scan_idx=k,
                scan_name=scan_names[k],
                elapsed_s=float(elapsed_s[k]),
                timestamp=scan_isos[k],
                voltage_v=float(v_interp[k]) if v_interp is not None else None,
                current_ma=float(i_interp[k]) if i_interp is not None else None,
                capacity_mah_g=float(q_interp[k]) if q_interp is not None else None,
                dq_dv=float(dqdv_interp[k]) if dqdv_interp is not None else None,
                cycle=int(round(float(cyc_interp[k]))) if cyc_interp is not None else None,
                stress_mpa=float(sig_interp[k]) if sig_interp is not None else None,
                temperature_c=float(tmp_interp[k]) if tmp_interp is not None else None,
            )
        )
    return points


def annotate_tracks_with_stimulus(
    track_rows: list[dict[str, Any]],
    scan_stimuli: list[ScanStimulusPoint],
) -> list[dict[str, Any]]:
    """Return shallow-copied track rows enriched with synchronized stimulus and ``d_strain_d_v``."""
    if not track_rows:
        return []

    stim_by_scan = {pt.scan_idx: pt for pt in scan_stimuli}
    enriched: list[dict[str, Any]] = []
    prev_by_tid: dict[int, tuple[float | None, float | None]] = {}

    computed_strain_by_key: dict[tuple[int, int], float] = {}
    if "lineage_strain" not in track_rows[0] and "strain" not in track_rows[0] and "centroid_d" in track_rows[0]:
        from braggtrack.tracking.kinematics import compute_grain_kinematics

        for gk in compute_grain_kinematics(track_rows):
            lin = gk.lineage_strain if len(gk.lineage_strain) else gk.strain
            for idx_s, s_i in enumerate(gk.scan_indices):
                computed_strain_by_key[(gk.track_id, s_i)] = float(lin[idx_s])

    # Process in chronological order per track so d_strain_d_v is strictly causal
    indexed = sorted(
        enumerate(track_rows),
        key=lambda item: (int(float(item[1]["track_id"])), int(float(item[1]["scan_idx"]))),
    )
    out_by_orig_idx: dict[int, dict[str, Any]] = {}

    for orig_idx, row in indexed:
        r = dict(row)
        tid = int(float(r["track_id"]))
        s_idx = int(float(r["scan_idx"]))
        pt = stim_by_scan.get(s_idx)

        v_val = pt.voltage_v if pt is not None else None
        r["elapsed_s"] = round(pt.elapsed_s, 4) if pt is not None else ""
        r["voltage_v"] = round(v_val, 5) if v_val is not None else ""
        r["current_ma"] = round(pt.current_ma, 5) if (pt is not None and pt.current_ma is not None) else ""
        r["capacity_mah_g"] = round(pt.capacity_mah_g, 4) if (pt is not None and pt.capacity_mah_g is not None) else ""
        r["dq_dv"] = round(pt.dq_dv, 4) if (pt is not None and pt.dq_dv is not None) else ""
        r["stress_mpa"] = round(pt.stress_mpa, 4) if (pt is not None and pt.stress_mpa is not None) else ""
        r["temperature_c"] = round(pt.temperature_c, 3) if (pt is not None and pt.temperature_c is not None) else ""

        strain_val: float | None = computed_strain_by_key.get((tid, s_idx))
        if strain_val is None:
            for key in ("lineage_strain", "strain"):
                if key in r and r[key] != "":
                    try:
                        strain_val = float(r[key])
                        break
                    except (TypeError, ValueError):
                        pass

        d_eps_dv: float | str = ""
        if tid in prev_by_tid and v_val is not None and strain_val is not None:
            prev_v, prev_eps = prev_by_tid[tid]
            if prev_v is not None and prev_eps is not None and abs(v_val - prev_v) >= 2e-3:
                d_eps_dv = round((strain_val - prev_eps) / (v_val - prev_v), 6)
        r["d_strain_d_v"] = d_eps_dv
        prev_by_tid[tid] = (v_val, strain_val)
        out_by_orig_idx[orig_idx] = r

    for i in range(len(track_rows)):
        enriched.append(out_by_orig_idx[i])
    return enriched

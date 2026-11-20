"""NeXus/HDF5 metadata extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import AxisCalibration

if TYPE_CHECKING:
    import numpy as np

DEFAULT_VOLUME_CANDIDATES: tuple[str, ...] = (
    "/entry/data/data",
    "/entry1/data/data",
    "/entry_0000/ESRF-ID03/pco_nf/data",
    "/entry_0000/data/data",
    "/entry/darfix/data",
    "/entry_0000/darfix/data",
    "/entry/process/darfix/results/data",
    "/entry/instrument/detector/data",
    "/entry1/instrument/detector/data",
)


class MissingH5DependencyError(RuntimeError):
    """Raised when HDF5 dependencies are unavailable."""


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore

        return h5py
    except ModuleNotFoundError as exc:
        raise MissingH5DependencyError("h5py is required for HDF5/NeXus parsing. Install dependencies first.") from exc


def _open_h5_readonly(h5py: Any, path: str | Path, *, swmr: bool = False) -> Any:
    """Open an HDF5 file read-only, enabling SWMR when supported by the file superblock."""
    if not swmr:
        return h5py.File(path, "r")
    try:
        return h5py.File(path, "r", swmr=True)
    except ValueError:
        # Older HDF5 v0/v2 superblocks reject swmr=True with ValueError; fall back cleanly while keeping OSError intact.
        return h5py.File(path, "r")


def summarize_hdf5_tree(path: str | Path) -> list[tuple[str, str, tuple[int, ...] | None, str | None]]:
    """Return a flat summary of groups/datasets in an HDF5 file.

    Returns entries as ``(kind, name, shape, dtype)`` where kind is ``group`` or ``dataset``.
    """

    h5py = _require_h5py()
    entries: list[tuple[str, str, tuple[int, ...] | None, str | None]] = []

    with h5py.File(path, "r") as handle:

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Group):
                entries.append(("group", name, None, None))
            elif isinstance(obj, h5py.Dataset):
                entries.append(("dataset", name, tuple(obj.shape), str(obj.dtype)))

        handle.visititems(visitor)

    return entries


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        inner = value.item()
        if isinstance(inner, bytes):
            return inner.decode("utf-8", errors="replace")
        return inner
    return value


def _decode_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (bytes, str)):
        s = _decode_value(value)
        return [s] if isinstance(s, str) and s.strip() else []
    if hasattr(value, "__iter__"):
        out: list[str] = []
        for item in value:
            s = _decode_value(item)
            if isinstance(s, str) and s.strip():
                out.append(s)
        return out
    return []


def _candidate_entries(handle: Any, h5py: Any) -> list[str]:
    seen: list[str] = []

    def _add(prefix: str) -> None:
        norm = "/" + prefix.strip("/")
        if norm != "/" and norm in handle and isinstance(handle[norm], h5py.Group) and norm not in seen:
            seen.append(norm)

    default_attr = _decode_value(handle.attrs.get("default"))
    if isinstance(default_attr, str) and default_attr.strip():
        _add(default_attr)

    for std in ("/entry", "/entry1", "/entry_0000"):
        _add(std)

    for key in sorted(handle.keys()):
        obj = handle[key]
        if isinstance(obj, h5py.Group):
            nx_cls = _decode_value(obj.attrs.get("NX_class"))
            if key.startswith("entry") or nx_cls == "NXentry":
                _add(key)

    return seen


def _read_first(handle: Any, candidates: list[str], h5py: Any | None = None) -> Any | None:
    for key in candidates:
        if key in handle:
            obj = handle[key]
            if h5py is not None and isinstance(obj, h5py.Group):
                return key
            return _decode_value(obj[()])
    return None


def extract_scan_metadata(path: str | Path, *, swmr: bool = False) -> dict[str, Any]:
    """Extract common beamline metadata from a NeXus/HDF5 scan file.

    Probes standard NeXus ``/entry*`` groups (including ``/entry_0000`` used by
    ESRF LIMA/BLISS acquisitions).
    """

    h5py = _require_h5py()
    path = Path(path)

    with _open_h5_readonly(h5py, path, swmr=swmr) as handle:
        entries = _candidate_entries(handle, h5py) or ["/entry", "/entry1", "/entry_0000"]
        start_candidates = [f"{e}/start_time" for e in entries]
        end_candidates = [f"{e}/end_time" for e in entries]
        title_candidates = [f"{e}/title" for e in entries]
        sample_candidates: list[str] = []
        for e in entries:
            sample_candidates.extend([f"{e}/sample/name", f"{e}/sample"])

        entry_found = next((e for e in entries if e in handle), None)
        metadata = {
            "file": str(path),
            "entry": entry_found,
            "start_time": _read_first(handle, start_candidates, h5py),
            "end_time": _read_first(handle, end_candidates, h5py),
            "title": _read_first(handle, title_candidates, h5py),
            "sample_name": _read_first(handle, sample_candidates, h5py),
        }

    return metadata


_MOTOR_AXIS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "mu": ("mu", "omega", "obx", "rocking"),
    "chi": ("chi", "phi", "oby"),
    "d": ("d", "obpitch", "twotheta", "two_theta", "2theta", "tth"),
}


def extract_axis_calibration(path: str | Path, *, swmr: bool = False) -> AxisCalibration:
    """Extract linear ``(mu, chi, d)`` calibration from a NeXus/HDF5 file or OME-Zarr store if present.

    Probes:
    1. ``basecoat`` OME-Zarr v0.4 stores (``.ome.zarr`` / ``.zarr``).
    2. Explicit ``/entry*/calibration`` groups.
    3. NeXus ``NXdata`` ``@axes`` attributes on the primary 3D dataset's parent group (e.g. ``darfix``).
    4. 1D coordinate arrays under ``NXentry``, ``data``, ``measurement``, or ESRF BLISS
       ``instrument/positioners/{mu, chi, obpitch, twotheta, d}``.
    """
    import numpy as np

    path = Path(path)
    if path.name.lower().endswith((".ome.zarr", ".zarr")):
        import basecoat

        with basecoat.open_volume(path) as bc_lazy:
            return AxisCalibration.from_basecoat_axes(bc_lazy.axes, provenance=bc_lazy.provenance)

    h5py = _require_h5py()

    with _open_h5_readonly(h5py, path, swmr=swmr) as handle:
        vol_parent: str | None = None
        try:
            ds_path = _resolve_volume_dataset(handle, h5py, auto_discover=True)
            vol_parent = str(Path(ds_path).parent)
        except (KeyError, ValueError):
            vol_parent = None

        entries = _candidate_entries(handle, h5py)
        for e in entries:
            cal_grp_path = f"{e}/calibration"
            if cal_grp_path in handle and isinstance(handle[cal_grp_path], h5py.Group):
                grp = handle[cal_grp_path]
                raw: dict[str, Any] = {k: _decode_value(v) for k, v in grp.attrs.items()}
                for k in grp:
                    if isinstance(grp[k], h5py.Dataset) and grp[k].ndim == 0:
                        raw[k] = _decode_value(grp[k][()])
                if raw:
                    return AxisCalibration.from_dict(raw)

        # Check NeXus NXdata @axes attribute on parent group (e.g. darfix NXdata cubes: (z=mu, y=d/2theta, x=chi))
        if vol_parent and vol_parent != "/" and vol_parent in handle:
            pgrp = handle[vol_parent]
            nx_axes = _decode_str_list(pgrp.attrs.get("axes"))
            if len(nx_axes) == 3:
                nx_found: dict[str, tuple[float, float]] = {}
                # Voxel order (z, y, x) -> BraggTrack (mu, d, chi)
                for dim_idx, target_axis in ((0, "mu"), (1, "d"), (2, "chi")):
                    ax_ds_name = nx_axes[dim_idx]
                    if ax_ds_name != "." and ax_ds_name in pgrp and isinstance(pgrp[ax_ds_name], h5py.Dataset):
                        arr = np.asarray(pgrp[ax_ds_name][()], dtype=np.float64).ravel()
                        if arr.size >= 2 and float(arr[-1] - arr[0]) != 0.0:
                            nx_found[target_axis] = (float(arr[0]), float((arr[-1] - arr[0]) / (arr.size - 1)))
                if nx_found:
                    return AxisCalibration(
                        mu_origin=nx_found.get("mu", (0.0, 1.0))[0],
                        mu_step=nx_found.get("mu", (0.0, 1.0))[1],
                        chi_origin=nx_found.get("chi", (0.0, 1.0))[0],
                        chi_step=nx_found.get("chi", (0.0, 1.0))[1],
                        d_origin=nx_found.get("d", (0.0, 1.0))[0],
                        d_step=nx_found.get("d", (0.0, 1.0))[1],
                        calibrated=True,
                    )

        for e in entries:
            # Check 1D axis/motor arrays under NXentry, NXdata, measurement, or ESRF BLISS instrument/positioners
            axes_found: dict[str, tuple[float, float]] = {}
            for axis_name, synonyms in _MOTOR_AXIS_SYNONYMS.items():
                search_paths: list[str] = []
                for syn in synonyms:
                    search_paths.extend(
                        [
                            f"{e}/{syn}",
                            f"{e}/data/{syn}",
                            f"{e}/measurement/{syn}",
                            f"{e}/instrument/positioners/{syn}",
                        ]
                    )
                    if vol_parent and vol_parent != "/":
                        search_paths.append(f"{vol_parent.rstrip('/')}/{syn}")
                for sub in search_paths:
                    if sub in handle and isinstance(handle[sub], h5py.Dataset):
                        arr = np.asarray(handle[sub][()], dtype=np.float64).ravel()
                        if arr.size >= 2 and float(arr[-1] - arr[0]) != 0.0:
                            axes_found[axis_name] = (float(arr[0]), float((arr[-1] - arr[0]) / (arr.size - 1)))
                            break
            if axes_found:
                return AxisCalibration(
                    mu_origin=axes_found.get("mu", (0.0, 1.0))[0],
                    mu_step=axes_found.get("mu", (0.0, 1.0))[1],
                    chi_origin=axes_found.get("chi", (0.0, 1.0))[0],
                    chi_step=axes_found.get("chi", (0.0, 1.0))[1],
                    d_origin=axes_found.get("d", (0.0, 1.0))[0],
                    d_step=axes_found.get("d", (0.0, 1.0))[1],
                    calibrated=True,
                )

    return AxisCalibration()


def _resolve_volume_dataset(
    handle: Any,
    h5py: Any,
    candidates: list[str] | None = None,
    *,
    auto_discover: bool = True,
) -> str:
    import numpy as np

    ds_candidates = list(candidates) if candidates is not None else list(DEFAULT_VOLUME_CANDIDATES)
    for key in ds_candidates:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            ds = handle[key]
            if getattr(ds, "ndim", None) != 3:
                raise ValueError(f"Dataset '{key}' exists but is not 3D (ndim={getattr(ds, 'ndim', None)}).")
            return "/" + key.lstrip("/")

    if candidates is None and auto_discover:
        # Check NeXus @default -> NXdata @signal chain first (used by darfix / silx / Ewoks)
        for e in _candidate_entries(handle, h5py):
            eg = handle[e]
            def_data = _decode_value(eg.attrs.get("default"))
            if isinstance(def_data, str) and def_data.strip():
                dgrp_path = f"{e.rstrip('/')}/{def_data.strip('/')}"
                if dgrp_path in handle and isinstance(handle[dgrp_path], h5py.Group):
                    sig = _decode_value(handle[dgrp_path].attrs.get("signal"))
                    if isinstance(sig, str) and sig in handle[dgrp_path]:
                        sig_ds = handle[dgrp_path][sig]
                        if isinstance(sig_ds, h5py.Dataset) and sig_ds.ndim == 3:
                            return f"{dgrp_path}/{sig}"

        discovered: list[tuple[str, tuple[int, ...]]] = []

        def _visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and obj.ndim == 3 and np.issubdtype(obj.dtype, np.number):
                discovered.append(("/" + name.lstrip("/"), tuple(obj.shape)))

        handle.visititems(_visit)
        if discovered:
            return max(discovered, key=lambda item: int(np.prod(item[1])))[0]

    raise KeyError(f"No candidate 3D dataset found in file. Tried: {ds_candidates}")


def discover_primary_volume_path(
    path: str | Path,
    candidates: list[str] | None = None,
    *,
    auto_discover: bool = True,
    swmr: bool = False,
) -> str:
    """Return the HDF5 dataset path of the primary 3D detector volume."""
    h5py = _require_h5py()
    with _open_h5_readonly(h5py, path, swmr=swmr) as handle:
        return _resolve_volume_dataset(handle, h5py, candidates=candidates, auto_discover=auto_discover)


def load_primary_volume(
    path: str | Path,
    candidates: list[str] | None = None,
    *,
    auto_discover: bool = True,
    swmr: bool = False,
) -> np.ndarray:
    """Load a primary 3D detector volume from NeXus HDF5 or ``basecoat``-supported stores."""
    import numpy as np

    p = Path(path)
    if p.name.lower().endswith((".ome.zarr", ".zarr", ".tif", ".tiff", ".npy", ".raw")):
        import basecoat

        loaded = basecoat.read_volume(p)
        vol = loaded.image if isinstance(loaded, basecoat.VolumeSet) else loaded
        return np.asarray(vol.data, dtype=np.float64)

    h5py = _require_h5py()
    with _open_h5_readonly(h5py, p, swmr=swmr) as handle:
        ds_path = _resolve_volume_dataset(handle, h5py, candidates=candidates, auto_discover=auto_discover)
        return np.asarray(handle[ds_path][()], dtype=np.float64)

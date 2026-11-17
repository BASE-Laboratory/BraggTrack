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


def extract_scan_metadata(path: str | Path) -> dict[str, Any]:
    """Extract common beamline metadata from a NeXus/HDF5 scan file.

    Probes standard NeXus ``/entry*`` groups (including ``/entry_0000`` used by
    ESRF LIMA/BLISS acquisitions).
    """

    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as handle:
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


def extract_axis_calibration(path: str | Path) -> AxisCalibration:
    """Extract linear ``(mu, chi, d)`` calibration from a NeXus/HDF5 file if present.

    Checks ``/entry*/calibration`` attributes/datasets or 1D coordinate arrays
    ``mu``, ``chi``, and ``d``. Returns an uncalibrated identity
    :class:`AxisCalibration` when physical axis scales are absent.
    """
    import numpy as np

    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as handle:
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

            # Check 1D axis arrays under NXentry, NXdata, or the primary 3D dataset's parent group.
            axes_found: dict[str, tuple[float, float]] = {}
            for axis_name in ("mu", "chi", "d"):
                search_paths = [f"{e}/{axis_name}", f"{e}/data/{axis_name}", f"{e}/measurement/{axis_name}"]
                if vol_parent and vol_parent != "/":
                    search_paths.append(f"{vol_parent.rstrip('/')}/{axis_name}")
                for sub in search_paths:
                    if sub in handle and isinstance(handle[sub], h5py.Dataset):
                        arr = np.asarray(handle[sub][()], dtype=np.float64).ravel()
                        if arr.size >= 2:
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
) -> str:
    """Return the HDF5 dataset path of the primary 3D detector volume."""
    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        return _resolve_volume_dataset(handle, h5py, candidates=candidates, auto_discover=auto_discover)


def load_primary_volume(
    path: str | Path,
    candidates: list[str] | None = None,
    *,
    auto_discover: bool = True,
) -> np.ndarray:
    """Load a primary 3D detector volume from NeXus paths or largest-3D auto-discovery."""
    import numpy as np

    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        ds_path = _resolve_volume_dataset(handle, h5py, candidates=candidates, auto_discover=auto_discover)
        return np.asarray(handle[ds_path][()], dtype=np.float64)

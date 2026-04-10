"""NeXus/HDF5 metadata extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MissingH5DependencyError(RuntimeError):
    """Raised when HDF5 dependencies are unavailable."""


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore

        return h5py
    except ModuleNotFoundError as exc:
        raise MissingH5DependencyError(
            "h5py is required for HDF5/NeXus parsing. Install dependencies first."
        ) from exc


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


def _read_first(handle: Any, candidates: list[str]) -> Any | None:
    for key in candidates:
        if key in handle:
            value = handle[key][()]
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value
    return None


def extract_scan_metadata(path: str | Path) -> dict[str, Any]:
    """Extract common beamline metadata from a NeXus/HDF5 scan file.

    The function probes a small set of common key paths used by NeXus instrument files.
    """

    h5py = _require_h5py()
    path = Path(path)

    with h5py.File(path, "r") as handle:
        metadata = {
            "file": str(path),
            "entry": _read_first(handle, ["/entry", "/entry1"]),
            "start_time": _read_first(handle, ["/entry/start_time", "/entry1/start_time"]),
            "end_time": _read_first(handle, ["/entry/end_time", "/entry1/end_time"]),
            "title": _read_first(handle, ["/entry/title", "/entry1/title"]),
            "sample_name": _read_first(
                handle,
                [
                    "/entry/sample/name",
                    "/entry1/sample/name",
                    "/entry/sample",
                    "/entry1/sample",
                ],
            ),
        }

    return metadata


def load_primary_volume(path: str | Path, candidates: list[str] | None = None) -> list[list[list[float]]]:
    """Load a primary 3D detector volume from common NeXus dataset paths."""

    h5py = _require_h5py()
    ds_candidates = candidates or [
        "/entry/data/data",
        "/entry1/data/data",
        "/entry/instrument/detector/data",
        "/entry1/instrument/detector/data",
    ]

    with h5py.File(path, "r") as handle:
        for key in ds_candidates:
            if key in handle:
                data = handle[key][()]
                if getattr(data, "ndim", None) != 3:
                    raise ValueError(f"Dataset '{key}' exists but is not 3D (ndim={getattr(data, 'ndim', None)}).")
                return data.tolist()

    raise KeyError(f"No candidate 3D dataset found in file. Tried: {ds_candidates}")

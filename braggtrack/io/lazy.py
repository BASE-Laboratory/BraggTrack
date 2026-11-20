"""Chunked and ROI-lazy 3D volume access for multi-gigabyte HDF5 detector stacks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from skimage.filters import threshold_otsu

from .models import AxisCalibration
from .nexus import _require_h5py, _resolve_volume_dataset

_BASECOAT_VOLUME_SUFFIXES = (".ome.zarr", ".zarr", ".tif", ".tiff", ".npy", ".raw")


class _BasecoatLazyDatasetAdapter:
    """Sliceable adapter bridging ``basecoat.LazyVolume`` to :class:`LazyVolume`."""

    def __init__(self, bc_lazy: Any) -> None:
        self._bc = bc_lazy
        self.shape = bc_lazy.shape
        self.dtype = bc_lazy.dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        if key is Ellipsis or key == ():
            return np.asarray(self._bc.to_volume(max_bytes=4 * 1024**3).data)
        return np.asarray(self._bc.read_region(key).data)


class LazyVolume:
    """Lazy 3D volume wrapper supporting bounded ROI reads and z-slab streaming.

    Works seamlessly over an open ``h5py.Dataset``, a ``basecoat.LazyVolume``
    (``.ome.zarr``, ``.tif``, ``.npy``), or an in-memory ``numpy.ndarray``
    fallback so downstream stages (:func:`crop_spot_cube`, chunked Otsu
    thresholding) never materialize full ``float64`` detector cubes unless
    explicitly requested via ``vol[()]``.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        handle: Any | None = None,
        dataset_path: str | None = None,
    ) -> None:
        shape = tuple(int(x) for x in dataset.shape)
        if len(shape) != 3:
            if handle is not None:
                handle.close()
            raise ValueError(f"LazyVolume requires a 3D dataset, got shape={shape}")
        self._dataset = dataset
        self._handle = handle
        self._shape: tuple[int, int, int] = (shape[0], shape[1], shape[2])
        self._dtype = np.dtype(dataset.dtype)
        self._dataset_path = dataset_path

    @classmethod
    def from_array(cls, array: np.ndarray, *, dataset_path: str | None = None) -> LazyVolume:
        arr = np.asarray(array)
        return cls(arr, handle=None, dataset_path=dataset_path)

    @classmethod
    def from_basecoat(cls, source: Any, *, dataset_path: str | None = None) -> LazyVolume:
        """Wrap a ``basecoat.LazyVolume`` or ``basecoat.Volume`` without materializing out-of-core stores."""
        import basecoat

        if isinstance(source, basecoat.Volume):
            return cls.from_array(np.asarray(source.data), dataset_path=dataset_path)
        if isinstance(source, basecoat.LazyVolume):
            return cls(_BasecoatLazyDatasetAdapter(source), handle=source, dataset_path=dataset_path)
        raise TypeError(f"Expected basecoat.LazyVolume or basecoat.Volume, got {type(source).__name__}")

    def to_basecoat_volume(self, calibration: AxisCalibration | None = None) -> Any:
        """Materialize as a calibrated ``basecoat.Volume`` carrying ``(mu, d, chi)`` axes."""
        import basecoat

        cal = calibration or AxisCalibration()
        return basecoat.Volume(
            self.to_dense(),
            cal.to_basecoat_axes(),
            provenance={"producer": "braggtrack", "braggtrack_calibration": cal.to_dict()},
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    @property
    def ndim(self) -> int:
        return 3

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def dataset_path(self) -> str | None:
        return self._dataset_path

    def read_roi(self, z_slice: slice, y_slice: slice, x_slice: slice) -> np.ndarray:
        """Read a bounded 3D sub-block into memory as ``float64``."""
        return np.asarray(self._dataset[z_slice, y_slice, x_slice], dtype=np.float64)

    def __getitem__(self, key: Any) -> np.ndarray:
        return np.asarray(self._dataset[key], dtype=np.float64)

    def iter_z_chunks(self, chunk_z: int = 32) -> Iterator[np.ndarray]:
        """Yield contiguous ``(dz, ny, nx)`` ``float64`` slabs along axis 0."""
        step = max(1, int(chunk_z))
        nz = self._shape[0]
        for z0 in range(0, nz, step):
            z1 = min(nz, z0 + step)
            yield np.asarray(self._dataset[z0:z1, :, :], dtype=np.float64)

    def to_dense(self) -> np.ndarray:
        """Materialize the full 3D volume as ``float64``."""
        return np.asarray(self._dataset[()], dtype=np.float64)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> LazyVolume:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def open_lazy_volume(
    path: str | Path,
    candidates: list[str] | None = None,
    *,
    auto_discover: bool = True,
    swmr: bool = False,
) -> LazyVolume:
    """Open an HDF5/NeXus or ``basecoat``-supported store (``.ome.zarr``, ``.tif``, ``.npy``) lazily."""
    p = Path(path)
    if p.name.lower().endswith(_BASECOAT_VOLUME_SUFFIXES):
        import basecoat

        bc_lazy = basecoat.open_volume(p)
        return LazyVolume.from_basecoat(bc_lazy, dataset_path=str(p))

    from .nexus import _open_h5_readonly

    h5py = _require_h5py()
    handle = _open_h5_readonly(h5py, p, swmr=swmr)
    try:
        ds_path = _resolve_volume_dataset(handle, h5py, candidates=candidates, auto_discover=auto_discover)
        ds = handle[ds_path]
    except Exception:
        handle.close()
        raise
    return LazyVolume(ds, handle=handle, dataset_path=ds_path)


def otsu_threshold_chunked(
    volume: LazyVolume | np.ndarray,
    *,
    nbins: int = 256,
    chunk_z: int = 32,
) -> float:
    """Compute global Otsu threshold via streaming z-slabs in ``O(chunk)`` memory."""
    lazy = volume if isinstance(volume, LazyVolume) else LazyVolume.from_array(np.asarray(volume))
    nz, ny, nx = lazy.shape
    if nz * ny * nx == 0:
        raise ValueError("Otsu threshold requires at least one value.")

    vmin = float("inf")
    vmax = float("-inf")
    for slab in lazy.iter_z_chunks(chunk_z=chunk_z):
        s_min = float(np.min(slab))
        s_max = float(np.max(slab))
        if s_min < vmin:
            vmin = s_min
        if s_max > vmax:
            vmax = s_max

    if vmin == vmax:
        return float(vmin)

    edges = np.linspace(vmin, vmax, max(2, int(nbins)) + 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for slab in lazy.iter_z_chunks(chunk_z=chunk_z):
        c, _ = np.histogram(slab, bins=edges)
        counts += c

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return float(threshold_otsu(hist=(counts, bin_centers)))


def _histogram_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linearly interpolate quantile ``q in [0, 1]`` from a 1D histogram."""
    total = float(np.sum(counts))
    if total <= 0.0:
        return float(edges[0])
    target = float(np.clip(q, 0.0, 1.0)) * total
    cum = np.cumsum(counts, dtype=np.float64)
    idx = int(np.searchsorted(cum, target, side="left"))
    idx = min(max(idx, 0), len(counts) - 1)
    prev = float(cum[idx - 1]) if idx > 0 else 0.0
    c_bin = float(counts[idx])
    frac = (target - prev) / c_bin if c_bin > 0.0 else 0.0
    return float(edges[idx] + np.clip(frac, 0.0, 1.0) * (edges[idx + 1] - edges[idx]))


def detector_foreground_threshold_chunked(
    volume: LazyVolume | np.ndarray,
    *,
    nbins: int = 4096,
    chunk_z: int = 32,
    bg_sigma: float = 5.0,
) -> float:
    """Streaming z-slab Otsu threshold guarded by a lower-quartile read-noise floor."""
    lazy = volume if isinstance(volume, LazyVolume) else LazyVolume.from_array(np.asarray(volume))
    nz, ny, nx = lazy.shape
    total_vox = nz * ny * nx
    if total_vox == 0:
        raise ValueError("Otsu threshold requires at least one value.")

    vmin = float("inf")
    vmax = float("-inf")
    for slab in lazy.iter_z_chunks(chunk_z=chunk_z):
        s_min = float(np.min(slab))
        s_max = float(np.max(slab))
        if s_min < vmin:
            vmin = s_min
        if s_max > vmax:
            vmax = s_max

    if vmin == vmax:
        return float(vmin)

    edges = np.linspace(vmin, vmax, max(16, int(nbins)) + 1, dtype=np.float64)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for slab in lazy.iter_z_chunks(chunk_z=chunk_z):
        c, _ = np.histogram(slab, bins=edges)
        counts += c

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    raw_otsu = float(threshold_otsu(hist=(counts, bin_centers)))
    if bg_sigma <= 0.0 or total_vox < 4:
        return raw_otsu

    q25 = _histogram_quantile(counts, edges, 0.25)
    q50 = _histogram_quantile(counts, edges, 0.50)
    sigma_bg = 1.482602218505602 * max(0.0, q50 - q25)
    return float(max(raw_otsu, q50 + float(bg_sigma) * sigma_bg))

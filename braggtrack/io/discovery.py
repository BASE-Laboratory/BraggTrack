"""Filesystem discovery for operando scan datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScanFile:
    """A discovered scan file on disk."""

    scan_name: str
    path: Path


def discover_operando_scans(root: str | Path, pattern: str = "pco_nf_*.h5") -> list[ScanFile]:
    """Discover sequential scan directories and their HDF5 files.

    Parameters
    ----------
    root:
        Directory containing scan folders such as ``scan0001``.
    pattern:
        Glob pattern used within each scan folder.

    Returns
    -------
    list[ScanFile]
        Discovered files sorted by scan name then path.
    """

    root_path = Path(root)
    scans: list[ScanFile] = []

    for scan_dir in sorted(p for p in root_path.iterdir() if p.is_dir() and p.name.startswith("scan")):
        for h5_file in sorted(scan_dir.glob(pattern)):
            scans.append(ScanFile(scan_name=scan_dir.name, path=h5_file))

    return scans

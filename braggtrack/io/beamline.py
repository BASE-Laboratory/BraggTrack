"""Beamline-oriented loading and sequence construction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .discovery import ScanFile, discover_operando_scans
from .models import ExperimentSequence, ScanVolumeMeta
from .nexus import MissingH5DependencyError, extract_scan_metadata


class BeamlineAdapter:
    """Build BraggTrack contracts from local NeXus/HDF5 scan files."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def _parse_scan_index(scan_name: str) -> int:
        digits = "".join(ch for ch in scan_name if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not parse numeric index from scan name: {scan_name}")
        return int(digits)

    @staticmethod
    def _parse_datetime(raw: Any) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            cleaned = raw.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                return None
        return None

    def _scan_to_meta(self, scan_file: ScanFile) -> ScanVolumeMeta:
        metadata: dict[str, Any] = {}
        metadata_error: str | None = None

        try:
            metadata = extract_scan_metadata(scan_file.path)
        except MissingH5DependencyError as exc:
            metadata_error = str(exc)

        extras: dict[str, Any] = {}
        if metadata_error:
            extras["metadata_error"] = metadata_error

        return ScanVolumeMeta(
            scan_name=scan_file.scan_name,
            file_path=scan_file.path,
            sequence_index=self._parse_scan_index(scan_file.scan_name),
            start_time=self._parse_datetime(metadata.get("start_time")),
            end_time=self._parse_datetime(metadata.get("end_time")),
            sample_name=metadata.get("sample_name"),
            title=metadata.get("title"),
            extras=extras,
        )

    def build_sequence(self, pattern: str = "pco_nf_*.h5") -> ExperimentSequence:
        scans = discover_operando_scans(self.root, pattern=pattern)
        mapped = tuple(self._scan_to_meta(scan) for scan in scans)
        return ExperimentSequence(scans=tuple(sorted(mapped, key=lambda item: item.sequence_index)))

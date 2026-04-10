"""Core data contracts for BraggTrack datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AxisSpec:
    """Axis definition for reciprocal-space coordinates."""

    name: str
    units: str
    size: int | None = None


@dataclass(frozen=True)
class ScanVolumeMeta:
    """Metadata contract for one operando scan volume."""

    scan_name: str
    file_path: Path
    sequence_index: int
    axes: tuple[AxisSpec, ...] = (
        AxisSpec("mu", "degree"),
        AxisSpec("chi", "degree"),
        AxisSpec("d", "angstrom"),
    )
    start_time: datetime | None = None
    end_time: datetime | None = None
    sample_name: str | None = None
    title: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSequence:
    """Ordered set of scans composing one operando experiment."""

    scans: tuple[ScanVolumeMeta, ...]

    def is_monotonic(self) -> bool:
        """Whether scan sequence indexes are strictly increasing by 1."""

        if not self.scans:
            return True
        expected = self.scans[0].sequence_index
        for scan in self.scans:
            if scan.sequence_index != expected:
                return False
            expected += 1
        return True

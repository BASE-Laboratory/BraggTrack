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
class AxisCalibration:
    """Linear mapping from voxel indices ``(z, y, x)`` to physical ``(mu, d, chi)``."""

    mu_origin: float = 0.0
    mu_step: float = 1.0
    chi_origin: float = 0.0
    chi_step: float = 1.0
    d_origin: float = 0.0
    d_step: float = 1.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.mu_step == 0.0 or self.chi_step == 0.0 or self.d_step == 0.0:
            raise ValueError("AxisCalibration steps (mu_step, chi_step, d_step) must be non-zero.")

    def voxel_to_physical(self, z: float, y: float, x: float) -> tuple[float, float, float]:
        """Convert voxel coordinates ``(z, y, x)`` into physical ``(mu, chi, d)``."""
        mu = self.mu_origin + z * self.mu_step
        chi = self.chi_origin + x * self.chi_step
        d = self.d_origin + y * self.d_step
        return (float(mu), float(chi), float(d))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu_origin": self.mu_origin,
            "mu_step": self.mu_step,
            "chi_origin": self.chi_origin,
            "chi_step": self.chi_step,
            "d_origin": self.d_origin,
            "d_step": self.d_step,
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AxisCalibration:
        has_axis_keys = any(k in data for k in ("mu_origin", "mu_step", "chi_origin", "chi_step", "d_origin", "d_step"))
        calibrated = bool(data.get("calibrated", has_axis_keys))
        return cls(
            mu_origin=float(data.get("mu_origin", 0.0)),
            mu_step=float(data.get("mu_step", 1.0)),
            chi_origin=float(data.get("chi_origin", 0.0)),
            chi_step=float(data.get("chi_step", 1.0)),
            d_origin=float(data.get("d_origin", 0.0)),
            d_step=float(data.get("d_step", 1.0)),
            calibrated=calibrated,
        )


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
    volume_dataset_path: str | None = None
    calibration: AxisCalibration = field(default_factory=AxisCalibration)
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

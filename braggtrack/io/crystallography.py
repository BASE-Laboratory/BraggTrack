"""Transmission diffraction detector geometry and reciprocal-space (2theta, eta, |Q|) mapping."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import AxisCalibration


@dataclass(frozen=True)
class DetectorGeometry:
    """Area-detector diffraction geometry for crystallographic angle & Q conversion.

    Follows standard transmission 3DXRD conventions (e.g. ESRF / ImageD11):
    incident beam along ``+X_lab``, detector plane at ``X_lab = distance_mm``,
    horizontal detector coordinate ``x_px`` (BraggTrack ``chi`` axis),
    vertical detector coordinate ``y_px`` (BraggTrack ``d`` axis), and
    rotation stack ``z_idx`` (BraggTrack ``mu`` / ``omega`` axis).
    """

    distance_mm: float = 150.0
    pixel_size_y_mm: float = 0.075
    pixel_size_x_mm: float = 0.075
    beam_center_y_px: float = 1024.0
    beam_center_x_px: float = 1024.0
    wavelength_angstrom: float = 0.2898
    omega_origin_deg: float = 0.0
    omega_step_deg: float = 0.05

    def __post_init__(self) -> None:
        if self.distance_mm <= 0.0:
            raise ValueError("distance_mm must be positive.")
        if self.pixel_size_y_mm <= 0.0 or self.pixel_size_x_mm <= 0.0:
            raise ValueError("pixel_size_y_mm and pixel_size_x_mm must be positive.")
        if self.wavelength_angstrom <= 0.0:
            raise ValueError("wavelength_angstrom must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "distance_mm": float(self.distance_mm),
            "pixel_size_y_mm": float(self.pixel_size_y_mm),
            "pixel_size_x_mm": float(self.pixel_size_x_mm),
            "beam_center_y_px": float(self.beam_center_y_px),
            "beam_center_x_px": float(self.beam_center_x_px),
            "wavelength_angstrom": float(self.wavelength_angstrom),
            "omega_origin_deg": float(self.omega_origin_deg),
            "omega_step_deg": float(self.omega_step_deg),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectorGeometry:
        return cls(
            distance_mm=float(data.get("distance_mm", 150.0)),
            pixel_size_y_mm=float(data.get("pixel_size_y_mm", 0.075)),
            pixel_size_x_mm=float(data.get("pixel_size_x_mm", 0.075)),
            beam_center_y_px=float(data.get("beam_center_y_px", 1024.0)),
            beam_center_x_px=float(data.get("beam_center_x_px", 1024.0)),
            wavelength_angstrom=float(data.get("wavelength_angstrom", 0.2898)),
            omega_origin_deg=float(data.get("omega_origin_deg", 0.0)),
            omega_step_deg=float(data.get("omega_step_deg", 0.05)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DetectorGeometry:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("DetectorGeometry JSON must contain an object.")
        return cls.from_dict(raw)

    def save_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def spot_to_scattering(self, z_idx: float, y_px: float, x_px: float) -> dict[str, float]:
        """Convert detector voxel centroid ``(z_idx, y_px, x_px)`` to ``(omega, 2theta, eta, d, Q)``."""
        dx_mm = (float(x_px) - self.beam_center_x_px) * self.pixel_size_x_mm
        dy_mm = (float(y_px) - self.beam_center_y_px) * self.pixel_size_y_mm
        r_mm = math.hypot(dx_mm, dy_mm)
        path_len = math.hypot(self.distance_mm, r_mm)

        two_theta_rad = math.atan2(r_mm, self.distance_mm)
        two_theta_deg = math.degrees(two_theta_rad)
        eta_deg = math.degrees(math.atan2(dy_mm, dx_mm))
        omega_deg = self.omega_origin_deg + float(z_idx) * self.omega_step_deg
        omega_rad = math.radians(omega_deg)

        sin_theta = math.sin(0.5 * two_theta_rad)
        if sin_theta > 1e-12:
            d_angstrom = self.wavelength_angstrom / (2.0 * sin_theta)
        else:
            d_angstrom = float("inf")

        k_mag = (2.0 * math.pi) / self.wavelength_angstrom
        qlab_x = k_mag * (self.distance_mm / path_len - 1.0)
        qlab_y = k_mag * (dx_mm / path_len)
        qlab_z = k_mag * (dy_mm / path_len)

        cos_w = math.cos(omega_rad)
        sin_w = math.sin(omega_rad)
        qx = qlab_x * cos_w + qlab_y * sin_w
        qy = -qlab_x * sin_w + qlab_y * cos_w
        qz = qlab_z
        q_mag = math.sqrt(qx * qx + qy * qy + qz * qz)

        return {
            "omega_deg": round(omega_deg, 6),
            "two_theta_deg": round(two_theta_deg, 6),
            "eta_deg": round(eta_deg, 6),
            "d_spacing_angstrom": round(d_angstrom, 6) if math.isfinite(d_angstrom) else float("inf"),
            "qx_inv_angstrom": round(qx, 6),
            "qy_inv_angstrom": round(qy, 6),
            "qz_inv_angstrom": round(qz, 6),
            "q_mag_inv_angstrom": round(q_mag, 6),
        }


def _recover_voxel_centroid(row: dict[str, Any], calibration: AxisCalibration | None) -> tuple[float, float, float]:
    mu = float(row["centroid_mu"])
    chi = float(row["centroid_chi"])
    d_val = float(row["centroid_d"])
    if calibration is not None and calibration.calibrated:
        z_idx = (mu - calibration.mu_origin) / calibration.mu_step
        x_px = (chi - calibration.chi_origin) / calibration.chi_step
        y_px = (d_val - calibration.d_origin) / calibration.d_step
        return z_idx, y_px, x_px
    return mu, d_val, chi


def annotate_crystallographic_columns(
    rows: list[dict[str, Any]],
    geometry: DetectorGeometry,
    *,
    calibration: AxisCalibration | None = None,
) -> list[dict[str, Any]]:
    """Return shallow-copied rows augmented with ``(omega_deg, two_theta_deg, eta_deg, d_spacing_angstrom, q_*)``."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        z_idx, y_px, x_px = _recover_voxel_centroid(row, calibration)
        scat = geometry.spot_to_scattering(z_idx, y_px, x_px)
        enriched.append({**row, **scat})
    return enriched

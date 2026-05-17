"""Ground-truth validation of kinematics physics.

Constructs a deterministic multi-grain scenario with analytically known
physical evolution, runs it through the full pipeline (tracking → kinematics),
and asserts exact recovery of prescribed strain, misorientation, growth, and
shape evolution.

Scenario (5 scans, 4 grains well-separated so tracking is unambiguous):

  Grain A — Linear elastic loading
    d increases 0.1% per step from d₀=10.0
    Expected strain: [0, 0.001, 0.002, 0.003, 0.004]

  Grain B — Pure rotation (no strain)
    μ rotates +0.5°/step, χ rotates +0.3°/step, d constant
    Expected misorientation_total: [0, √(0.5²+0.3²), 2×..., 3×..., 4×...]

  Grain C — Dissolution (shrinking grain)
    Intensity drops linearly: 1000 → 800 → 600 → 400 → 200
    Volume drops linearly: 100 → 80 → 60 → 40 → 20
    Expected relative_intensity: [0, -0.2, -0.4, -0.6, -0.8]

  Grain D — Late nucleation (born at scan 2)
    Appears at scan index 2, grows: intensity 50 → 100 → 150
    Expected: 3 observations, strain computed from its own d₀
"""

import unittest

import numpy as np

from braggtrack.tracking.cost import PositionShapeCost
from braggtrack.tracking.kinematics import compute_grain_kinematics, summarize_kinematics
from braggtrack.tracking.lifecycle import build_tracks, tracks_to_table


N_SCANS = 5


def _spot(
    mu: float,
    chi: float,
    d: float,
    intensity: float = 100.0,
    voxels: int = 10,
    eig: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    return {
        "label": 1,
        "voxel_count": voxels,
        "integrated_intensity": intensity,
        "centroid_mu": mu,
        "centroid_chi": chi,
        "centroid_d": d,
        "eig_1": eig[0],
        "eig_2": eig[1],
        "eig_3": eig[2],
    }


def _build_scenario() -> list[list[dict]]:
    """Build the 5-scan, 4-grain ground truth scenario."""
    scans: list[list[dict]] = []

    for i in range(N_SCANS):
        frame: list[dict] = []

        # Grain A: linear elastic loading at (mu=10, chi=10)
        d_a = 10.0 * (1.0 + 0.001 * i)
        frame.append(_spot(10.0, 10.0, d_a, intensity=500, voxels=50))

        # Grain B: pure rotation at (mu=50+0.5*i, chi=50+0.3*i, d=20)
        frame.append(_spot(50.0 + 0.5 * i, 50.0 + 0.3 * i, 20.0, intensity=500, voxels=50))

        # Grain C: dissolution at (mu=90, chi=90, d=15)
        intensity_c = 1000.0 - 200.0 * i
        voxels_c = 100 - 20 * i
        frame.append(_spot(90.0, 90.0, 15.0, intensity=intensity_c, voxels=voxels_c))

        # Grain D: nucleation at scan 2, at (mu=30, chi=70, d=25)
        if i >= 2:
            intensity_d = 50.0 + 50.0 * (i - 2)
            d_d = 25.0 * (1.0 + 0.002 * (i - 2))
            frame.append(
                _spot(30.0, 70.0, d_d, intensity=intensity_d, voxels=30, eig=(3.0, 2.0, 1.0))
            )

        scans.append(frame)

    return scans


def _run_pipeline(scans: list[list[dict]]):
    """Run full tracking + kinematics pipeline."""
    cost_fn = PositionShapeCost()
    G = build_tracks(scans, cost_fn)
    table = tracks_to_table(G)
    return compute_grain_kinematics(table)


def _find_grain(results, mu_approx: float, chi_approx: float):
    """Find grain by approximate initial centroid position."""
    for gk in results:
        if abs(float(gk.centroid_mu[0]) - mu_approx) < 5.0 and abs(float(gk.centroid_chi[0]) - chi_approx) < 5.0:
            return gk
    raise ValueError(f"No grain near mu={mu_approx}, chi={chi_approx}")


class TestGroundTruthElasticLoading(unittest.TestCase):
    """Grain A: verify strain recovery from prescribed d-spacing evolution."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = _run_pipeline(_build_scenario())
        cls.grain_a = _find_grain(cls.results, 10.0, 10.0)

    def test_track_length(self) -> None:
        self.assertEqual(len(self.grain_a.scan_indices), N_SCANS)

    def test_strain_values(self) -> None:
        expected = np.array([0.001 * i for i in range(N_SCANS)])
        np.testing.assert_allclose(self.grain_a.strain, expected, atol=1e-12)

    def test_no_misorientation(self) -> None:
        np.testing.assert_allclose(self.grain_a.misorientation_total, 0.0, atol=1e-12)

    def test_constant_intensity(self) -> None:
        np.testing.assert_allclose(self.grain_a.relative_intensity, 0.0, atol=1e-12)


class TestGroundTruthRotation(unittest.TestCase):
    """Grain B: verify misorientation recovery from prescribed μ/χ drift."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = _run_pipeline(_build_scenario())
        cls.grain_b = _find_grain(cls.results, 50.0, 50.0)

    def test_track_length(self) -> None:
        self.assertEqual(len(self.grain_b.scan_indices), N_SCANS)

    def test_misorientation_mu(self) -> None:
        expected = np.array([0.5 * i for i in range(N_SCANS)])
        np.testing.assert_allclose(self.grain_b.misorientation_mu, expected, atol=1e-12)

    def test_misorientation_chi(self) -> None:
        expected = np.array([0.3 * i for i in range(N_SCANS)])
        np.testing.assert_allclose(self.grain_b.misorientation_chi, expected, atol=1e-12)

    def test_misorientation_total(self) -> None:
        step = np.sqrt(0.5**2 + 0.3**2)
        expected = np.array([step * i for i in range(N_SCANS)])
        np.testing.assert_allclose(self.grain_b.misorientation_total, expected, atol=1e-12)

    def test_no_strain(self) -> None:
        np.testing.assert_allclose(self.grain_b.strain, 0.0, atol=1e-12)


class TestGroundTruthDissolution(unittest.TestCase):
    """Grain C: verify growth/dissolution from prescribed intensity/volume decay."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = _run_pipeline(_build_scenario())
        cls.grain_c = _find_grain(cls.results, 90.0, 90.0)

    def test_track_length(self) -> None:
        self.assertEqual(len(self.grain_c.scan_indices), N_SCANS)

    def test_relative_intensity(self) -> None:
        # I = [1000, 800, 600, 400, 200], I₀ = 1000
        expected = np.array([0.0, -0.2, -0.4, -0.6, -0.8])
        np.testing.assert_allclose(self.grain_c.relative_intensity, expected, atol=1e-12)

    def test_relative_volume(self) -> None:
        # V = [100, 80, 60, 40, 20], V₀ = 100
        expected = np.array([0.0, -0.2, -0.4, -0.6, -0.8])
        np.testing.assert_allclose(self.grain_c.relative_volume, expected, atol=1e-12)

    def test_no_strain(self) -> None:
        np.testing.assert_allclose(self.grain_c.strain, 0.0, atol=1e-12)

    def test_no_misorientation(self) -> None:
        np.testing.assert_allclose(self.grain_c.misorientation_total, 0.0, atol=1e-12)


class TestGroundTruthNucleation(unittest.TestCase):
    """Grain D: verify late-born grain with correct reference frame."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = _run_pipeline(_build_scenario())
        cls.grain_d = _find_grain(cls.results, 30.0, 70.0)

    def test_track_length(self) -> None:
        self.assertEqual(len(self.grain_d.scan_indices), 3)

    def test_birth_scan(self) -> None:
        self.assertEqual(self.grain_d.scan_indices[0], 2)

    def test_strain_from_own_d0(self) -> None:
        # d₀ = 25.0, d = [25.0, 25.05, 25.10]
        # strain = [0, 0.002, 0.004]
        expected = np.array([0.0, 0.002, 0.004])
        np.testing.assert_allclose(self.grain_d.strain, expected, atol=1e-12)

    def test_growth(self) -> None:
        # intensity = [50, 100, 150], I₀ = 50
        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(self.grain_d.relative_intensity, expected, atol=1e-12)

    def test_anisotropy(self) -> None:
        # eig = (3, 2, 1), anisotropy = 3/1 = 3.0 for all observations
        np.testing.assert_allclose(self.grain_d.anisotropy, 3.0, atol=1e-12)

    def test_covariance_trace(self) -> None:
        # eig = (3, 2, 1), trace = 6.0
        np.testing.assert_allclose(self.grain_d.covariance_trace, 6.0, atol=1e-12)


class TestGroundTruthSummary(unittest.TestCase):
    """Verify summary statistics against known scenario."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.results = _run_pipeline(_build_scenario())
        cls.summary = summarize_kinematics(cls.results, n_scans=N_SCANS)

    def test_track_count(self) -> None:
        self.assertEqual(self.summary.n_tracks, 4)

    def test_full_track_count(self) -> None:
        # Grains A, B, C span all 5 scans; Grain D only 3
        self.assertEqual(self.summary.n_full_tracks, 3)

    def test_max_strain_grain_a(self) -> None:
        # Grain A max strain = 0.004
        grain_a = _find_grain(self.results, 10.0, 10.0)
        idx = self.summary.track_ids.index(grain_a.track_id)
        self.assertAlmostEqual(self.summary.max_strain[idx], 0.004)

    def test_total_misorientation_grain_b(self) -> None:
        # Grain B final misorientation = 4 * √(0.25 + 0.09)
        grain_b = _find_grain(self.results, 50.0, 50.0)
        idx = self.summary.track_ids.index(grain_b.track_id)
        expected = 4.0 * np.sqrt(0.5**2 + 0.3**2)
        self.assertAlmostEqual(self.summary.total_misorientation[idx], expected)

    def test_intensity_change_grain_c(self) -> None:
        # Grain C final relative intensity = -0.8
        grain_c = _find_grain(self.results, 90.0, 90.0)
        idx = self.summary.track_ids.index(grain_c.track_id)
        self.assertAlmostEqual(self.summary.intensity_change_frac[idx], -0.8)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for darfix NXdata and ESRF BLISS NeXus/HDF5 auto-discovery."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from braggtrack.io.nexus import (
    discover_primary_volume_path,
    extract_axis_calibration,
    extract_scan_metadata,
    load_primary_volume,
)


def test_darfix_nxdata_default_signal_and_axes_calibration(tmp_path: Path) -> None:
    h5_path = tmp_path / "darfix_processed.h5"
    vol = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    mu_axis = np.linspace(12.0, 12.4, 5)
    tth_axis = np.linspace(21.5, 21.75, 6)
    chi_axis = np.linspace(-1.5, 1.5, 7)

    with h5py.File(h5_path, "w") as f:
        f.attrs["default"] = "scan_01"
        entry = f.create_group("scan_01")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "rocking_curves"
        entry.create_dataset("title", data="ID03 darfix rocking curve")

        nxdata = entry.create_group("rocking_curves")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "intensity_cube"
        nxdata.attrs["axes"] = ["obx", "obpitch", "oby"]
        nxdata.create_dataset("intensity_cube", data=vol)
        nxdata.create_dataset("obx", data=mu_axis)
        nxdata.create_dataset("obpitch", data=tth_axis)
        nxdata.create_dataset("oby", data=chi_axis)

    ds_path = discover_primary_volume_path(h5_path)
    loaded = load_primary_volume(h5_path)
    assert ds_path == "/scan_01/rocking_curves/intensity_cube"
    np.testing.assert_allclose(loaded, vol)

    cal = extract_axis_calibration(h5_path)
    assert cal.calibrated is True
    assert cal.mu_origin == pytest.approx(12.0)
    assert cal.mu_step == pytest.approx(0.1)
    assert cal.d_origin == pytest.approx(21.5)
    assert cal.d_step == pytest.approx(0.05)
    assert cal.chi_origin == pytest.approx(-1.5)
    assert cal.chi_step == pytest.approx(0.5)

    meta = extract_scan_metadata(h5_path)
    assert meta["entry"] == "/scan_01"
    assert meta["title"] == "ID03 darfix rocking curve"


def test_esrf_bliss_instrument_positioners_calibration(tmp_path: Path) -> None:
    h5_path = tmp_path / "bliss_id03_scan.h5"
    vol = np.ones((4, 5, 6), dtype=np.float32)

    with h5py.File(h5_path, "w") as f:
        entry = f.create_group("entry_0000")
        entry.attrs["NX_class"] = "NXentry"
        det = entry.create_group("ESRF-ID03/pco_nf")
        det.create_dataset("data", data=vol)

        pos = entry.create_group("instrument/positioners")
        pos.create_dataset("mu", data=np.array([10.0, 10.02, 10.04, 10.06]))
        pos.create_dataset("chi", data=np.linspace(-0.5, 0.5, 6))
        pos.create_dataset("twotheta", data=np.linspace(18.0, 18.2, 5))

    ds_path = discover_primary_volume_path(h5_path)
    loaded = load_primary_volume(h5_path)
    assert ds_path == "/entry_0000/ESRF-ID03/pco_nf/data"
    assert loaded.shape == (4, 5, 6)

    cal = extract_axis_calibration(h5_path)
    assert cal.calibrated is True
    assert cal.mu_origin == pytest.approx(10.0)
    assert cal.mu_step == pytest.approx(0.02)
    assert cal.chi_origin == pytest.approx(-0.5)
    assert cal.chi_step == pytest.approx(0.2)
    assert cal.d_origin == pytest.approx(18.0)
    assert cal.d_step == pytest.approx(0.05)

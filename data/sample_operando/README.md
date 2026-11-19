# Sample operando dataset (bundled)

This directory holds **three small cropped ESRF-ID03 detector cubes** (`scan0001`–`scan0003`) shipped with BraggTrack for automated tests, CI verification, and quick-start tutorials.

## Provenance & Licence

- **Facility & Beamline:** European Synchrotron Radiation Facility (ESRF), Beamline **ID03** (Dark-Field X-ray Microscopy), near-field **PCO.edge 5.5 (`pco_nf`)** 16-bit sCMOS detector (`100 × 250 × 250` `uint16` cropped detector ROI, acquired `2026-04-08`).
- **Experiment & Proposal:** ESRF Proposal **`MA-650628`** (*operando* Dark-Field X-ray Microscopy of high-nickel layered oxide `LiNi0.8Mn0.1Co0.1O2` / `NMC811` cathodes; J. Le Houx, J. J. Mistry, Y. Li, L. Lesage, S. Staeck, R. Bird, D. Spencer-Jolly *et al.*, ChemRxiv, DOI: `10.26434/chemrxiv.15002900`). Full raw diffraction volumes are archived in the ESRF Data Portal under Proposal `MA-650628`.
- **Evidentiary Scope:** These three uncalibrated consecutive detector sub-volumes test NeXus/LIMA HDF5 ingestion, sCMOS dark-pedestal read-noise rejection, 3D watershed instance segmentation, and frame-to-frame tracking; they are too short (`n = 3` scans, no bundled potentiostat log) to evaluate multi-horizon GP forecasting or fracture-precursor classification ROC.
- **Licence:** Bundled cropped test scans in this directory are distributed under the **Creative Commons Attribution 4.0 International (`CC-BY-4.0`)** licence with permission of the experimental team.

## Layout convention

Any **dataset root** passed to the CLIs (`braggtrack-segment-dataset`, `braggtrack-run`, `braggtrack-inspect`, etc.) must contain one folder per scan:

```text
<dataset_root>/
  scan0001/
    *.h5          # NeXus / HDF5 volume (naming pattern configurable in discovery)
  scan0002/
  scan0003/
```

Folder names must start with `scan` and sort in experiment order. Numeric indices (e.g. `0001` → sequence index `1`) are parsed by `BeamlineAdapter`.

## Your own data

Keep large or proprietary experiments **outside** the repository (or under a path you add to `.gitignore`). Point tools at that root explicitly:

```bash
python -m braggtrack.cli.segment_dataset /path/to/my_experiment --outdir artifacts/week2
```

## See also

- [Architecture](../../docs/architecture.md) — how I/O and discovery fit the pipeline
- [Coding practices](../../docs/coding-practices.md) — extending discovery or CLIs

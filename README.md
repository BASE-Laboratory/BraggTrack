# BraggTrack

Semantic 4D kinematics and fracture tracking for operando diffraction using foundation vision models.

## Repository layout

- **`braggtrack/`** — Python package (I/O, segmentation, semantic embeddings, tracking, CLIs).
- **`data/sample_operando/`** — bundled example scans (`scan0001` … `scan0003`, each with `pco_nf_0000_cropped.h5`). See [`data/sample_operando/README.md`](data/sample_operando/README.md).
- **`docs/`** — [architecture](docs/architecture.md) and [coding practices](docs/coding-practices.md).
- **`artifacts/`** — generated outputs (gitignored).

Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md).

## Included example scans

Bundled under **`data/sample_operando/`**:

- `data/sample_operando/scan0001/pco_nf_0000_cropped.h5`
- `data/sample_operando/scan0002/pco_nf_0000_cropped.h5`
- `data/sample_operando/scan0003/pco_nf_0000_cropped.h5`

## Week 2 classical baseline status

Implemented segmentation stack:

- 3D LoG enhancement (`log_enhance_3d`)
- h-maxima seed detection (`h_maxima_seeds`)
- 3D seeded watershed-style region growing (`watershed_from_seeds`)
- post-processing (small-object cleanup + 3D hole fill)
- feature extraction table (centroid, bbox, integrated intensity, covariance, eigenvalues)

Run end-to-end segmentation artifacts on the three scans (omit the path to use the bundled sample automatically, or pass it explicitly):

```bash
python -m braggtrack.cli.segment_dataset --outdir artifacts/week2
python -m braggtrack.cli.segment_dataset data/sample_operando --outdir artifacts/week2
```

Outputs:

- `artifacts/week2/segmentation_summary.json`
- `artifacts/week2/segmentation_summary.csv`
- `artifacts/week2/scanXXXX/summary.json`
- `artifacts/week2/scanXXXX/features.csv`
- `artifacts/week2/qc/week2_visual_qc.ipynb`

Note: if `h5py` is not available, the CLI uses a deterministic synthetic fallback volume per scan file so CI and schema checks remain reproducible.

## CI/CD

GitHub Actions CI runs:

- `python scripts/ci_report.py`

`ci_report.py` executes unit tests, Week 1 acceptance, Week 2 smoke, and Week 2 acceptance checks, then prints explicit PASS/FAIL criteria and overall status.

Before opening a PR, run:

```bash
python scripts/pre_pr_check.py
```

This checks for unresolved conflict markers and (when `origin/main` exists) reports ahead/behind divergence with a rebase hint.
## Week 1 implementation status

Week 1 focuses on data contracts, beamline adaptation, and dataset validation.

Implemented modules:

- `braggtrack.io.models`: canonical contracts (`ScanVolumeMeta`, `ExperimentSequence`, axis definitions)
- `braggtrack.io.beamline`: beamline adapter to map discovered scan files into contract objects
- `braggtrack.io.validation`: sequence and metadata validation checks
- `braggtrack.cli.inspect_datasets`: scan discovery + metadata extraction report
- `braggtrack.cli.validate_dataset`: sequence validation report

## Week 2 baseline (segmentation)

Week 2 starts with a robust classical baseline, including **Otsu thresholding** as the standard first-pass separator.

Implemented modules:

- `braggtrack.segmentation.otsu`: pure-Python Otsu threshold estimation
- `braggtrack.segmentation.pipeline`: threshold segmentation + 3D connected component counting
- `braggtrack.segmentation.classical`: LoG enhancement + local-max seed extraction + seeded region growing
- `braggtrack.cli.segment_synthetic`: synthetic smoke test for segmentation harness
- `braggtrack.cli.segment_dataset`: dataset scan segmentation CLI

## Quick start

```bash
python -m braggtrack.cli.inspect_datasets
python -m braggtrack.cli.validate_dataset
python -m braggtrack.cli.segment_synthetic
python -m braggtrack.cli.segment_dataset --outdir artifacts/week2
```

Each command resolves the dataset root via `braggtrack.io.paths`: **bundled `data/sample_operando` when present**, otherwise the current working directory. Pass an explicit path to use another dataset root.

If `h5py` is unavailable in your environment, scan discovery and validation still run and include a clear warning in the output payload.

## CI/CD

GitHub Actions CI now runs:

- unit tests (`python -m unittest discover -s tests -v`)
- Week 1 acceptance checks (`python scripts/check_acceptance.py`)

The acceptance script verifies scan discovery/order/monotonicity and fails only on unmet acceptance criteria (warnings are reported but do not fail by themselves).

# BraggTrack

Semantic 4D kinematics and fracture tracking for operando diffraction using foundation vision models.

## Included example scans

This repository currently includes three sequential operando scan folders:

- `scan0001/pco_nf_0000_cropped.h5`
- `scan0002/pco_nf_0000_cropped.h5`
- `scan0003/pco_nf_0000_cropped.h5`

## Week 1 implementation status

Week 1 focuses on data contracts, beamline adaptation, and dataset validation.

Implemented modules:

- `braggtrack.io.models`: canonical contracts (`ScanVolumeMeta`, `ExperimentSequence`, axis definitions)
- `braggtrack.io.beamline`: beamline adapter to map discovered scan files into contract objects
- `braggtrack.io.validation`: sequence and metadata validation checks
- `braggtrack.cli.inspect_datasets`: scan discovery + metadata extraction report
- `braggtrack.cli.validate_dataset`: sequence validation report

## Quick start

```bash
python -m braggtrack.cli.inspect_datasets .
python -m braggtrack.cli.validate_dataset .
```

If `h5py` is unavailable in your environment, scan discovery and validation still run and include a clear warning in the output payload.

## CI/CD

GitHub Actions CI now runs:

- unit tests (`python -m unittest discover -s tests -v`)
- Week 1 acceptance checks (`python scripts/check_acceptance.py`)

The acceptance script verifies scan discovery/order/monotonicity and fails only on unmet acceptance criteria (warnings are reported but do not fail by themselves).

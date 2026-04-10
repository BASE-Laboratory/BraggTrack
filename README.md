# BraggTrack

Semantic 4D kinematics and fracture tracking for operando diffraction using foundation vision models.

## Included example scans

This repository currently includes three sequential operando scan folders:

- `scan0001/pco_nf_0000_cropped.h5`
- `scan0002/pco_nf_0000_cropped.h5`
- `scan0003/pco_nf_0000_cropped.h5`

## Week 2 classical baseline status

Implemented segmentation stack:

- 3D LoG enhancement (`log_enhance_3d`)
- h-maxima seed detection (`h_maxima_seeds`)
- 3D seeded watershed-style region growing (`watershed_from_seeds`)
- post-processing (small-object cleanup + 3D hole fill)
- feature extraction table (centroid, bbox, integrated intensity, covariance, eigenvalues)

Run end-to-end segmentation artifacts on the three scans:

```bash
python -m braggtrack.cli.segment_dataset . --outdir artifacts/week2
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

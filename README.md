<img width="1774" height="887" alt="Social_banner" src="https://github.com/user-attachments/assets/4c73be10-eb60-4e6b-a0cd-4b6ac00af46b" />

# BraggTrack

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BASE-Laboratory/BraggTrack/blob/main/notebooks/braggtrack_demo.ipynb)

Semantic 4D kinematics, grain lineage (`1 -> 2+` fracture splits & `2+ -> 1` coalescence merges), and persistent fracture precursor tracking for operando X-ray diffraction using classical 3D crystallography and foundation vision transformers.

## Key Capabilities

- **Beamline-ready NeXus/HDF5 I/O & calibration (`braggtrack.io`)** — Auto-discovers 3D detector cubes across standard NeXus paths, ESRF-ID03 LIMA files (`/entry_0000/ESRF-ID03/pco_nf/data`), and arbitrary HDF5 trees (`auto_discover=True`), with optional physical `(mu, chi, d)` coordinate calibration via `AxisCalibration`.
- **Stabilised 3D Bragg reflection segmentation (`braggtrack.segmentation`)** — Intensity-domain Otsu foreground masking with cross-scan rolling-median smoothing (`smooth_thresholds`), 3D LoG peak enhancement, robust `h`-maxima seeding, 3D watershed, nearby-basin merging, and physical or voxel-space covariance/eigenvalue extraction. Includes an experimental dense patch-feature + HDBSCAN slice-stitching segmenter (`--method dino`).
- **Multi-view foundation vision descriptors (`braggtrack.semantic`)** — Orthogonal $(\mu, \chi, d)$ maximum-intensity projections (MIPs) encoded with ungated **`facebook/dinov2-small`** by default (zero HuggingFace login required), drop-in support for **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`), or a deterministic zero-GPU `mock` backend for CI.
- **4D grain kinematics, split/merge lineage & fracture precursors (`braggtrack.tracking`)** — Vectorised geometry + semantic Hungarian association into a NetworkX trajectory DAG, per-grain lattice strain ($\Delta d / d_0$) and angular misorientation ($\Delta\mu$, $\Delta\chi$), intensity-partitioned `1 -> 2+` split and `2+ -> 1` merge DAG edges, and persistent multi-scan fracture precursor ranking combining 3D shape anisotropy, covariance spread growth, and spot ROI sub-peak / DINO patch-token dispersion.
- **One-command reproducible pipeline (`braggtrack-run`)** — Runs validation, segmentation, semantic embedding, tracking, kinematics, and fracture event analysis end-to-end, emitting SHA-256 input checksums, stage timings, and `run_manifest.json`.

---

## Installation

```bash
# Core package (numpy, scipy, scikit-image, scikit-learn, networkx, h5py)
pip install -e .

# With PyTorch + HuggingFace Transformers for real DINOv2 / DINOv3 backbones
pip install -e ".[torch]"

# With Jupyter, Matplotlib, and Pandas for interactive notebooks
pip install -e ".[notebook]"

# Full developer environment (torch + notebook + pytest + ruff)
pip install -e ".[dev]"
```

---

## Quick Start

BraggTrack ships with three real ESRF ID03 operando Ni-rich cathode scans under [`data/sample_operando/`](data/sample_operando/README.md) (`100 x 250 x 250` `uint16` cubes). Omit the dataset path to run on the bundled scans automatically, or pass your own directory of `scanXXXX/` folders:

### One-command end-to-end pipeline

```bash
braggtrack-run --outdir artifacts/pipeline
```

Produces:
- `artifacts/pipeline/validation_summary.json`
- `artifacts/pipeline/segmentation/` (`features.csv`, `labels.npz`, `summary.json`, QC notebook)
- `artifacts/pipeline/embedding/` (`embeddings.npz`, `embedding_manifest.json`)
- `artifacts/pipeline/tracking/` (`tracks.csv`, `kinematics.csv`, `kinematics_summary.json`, `tracking_summary.json`, QC notebook)
- `artifacts/pipeline/events/` (`events.csv`, `events.json`, `precursors.csv`, `events_summary.json`)
- `artifacts/pipeline/run_manifest.json` (SHA-256 input checksums, git SHA, config, stage timings)

### Stage-by-stage CLI commands

| CLI Entry Point | Module Equivalent | Purpose |
|---|---|---|
| `braggtrack-inspect` | `python -m braggtrack.cli.inspect_datasets` | Discover `scan*` folders, primary 3D HDF5 dataset paths, and metadata |
| `braggtrack-validate` | `python -m braggtrack.cli.validate_dataset` | Validate sequence ordering, monotonic timestamps, and axis contracts |
| `braggtrack-segment-synthetic` | `python -m braggtrack.cli.segment_synthetic` | Run fast synthetic 3D segmentation smoke test |
| `braggtrack-segment-dataset` | `python -m braggtrack.cli.segment_dataset` | Segment 3D Bragg spots across scans with smoothed thresholds & optional calibration |
| `braggtrack-embed-dataset` | `python -m braggtrack.cli.embed_dataset` | Compute orthogonal MIP embeddings (`auto`, `mock`, or `torch` DINOv2/DINOv3) |
| `braggtrack-track-dataset` | `python -m braggtrack.cli.track_dataset` | Build trajectory DAG, export `tracks.csv` + `kinematics.csv` (+ optional `--detect-events`) |
| `braggtrack-events` | `python -m braggtrack.cli.detect_events` | Resolve `1 -> 2+` splits, `2+ -> 1` merges, and ROI-based fracture precursors |
| `braggtrack-run` | `python -m braggtrack.cli.run_pipeline` | Execute the full pipeline with optional `--config config.json` and `run_manifest.json` |

---

## Physical Axis Calibration (`mu`, `chi`, `d`)

Cropped detector cubes store integer voxel indices `[0..N)`. For physically meaningful lattice strain $\varepsilon = (d(t) - d_0) / d_0$ and angular misorientation in degrees, supply an `AxisCalibration` JSON file (or CLI flags) when segmenting:

```json
{
  "mu_origin": 12.50,
  "mu_step": 0.02,
  "chi_origin": -3.20,
  "chi_step": 0.015,
  "d_origin": 2.042,
  "d_step": -0.00012
}
```

```bash
braggtrack-segment-dataset data/sample_operando \
  --calibration-json calibration.json \
  --outdir artifacts/segmentation
```

If HDF5 scans already include 1D `mu`, `chi`, `d` coordinate datasets or `/entry/calibration` attributes, `extract_axis_calibration()` detects them automatically.

---

## Vision Backbone Options (DINOv2 Default & Optional DINOv3)

- **Default (`facebook/dinov2-small`)**: Public and ungated on HuggingFace — works out of the box with `pip install -e ".[torch]"` without tokens or license gates.
- **DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`)**: Supported via `--model` / `--dino-model` after accepting Meta's license on HuggingFace and setting `HF_TOKEN` (or running `huggingface-cli login`):
  ```bash
  braggtrack-embed-dataset --backend torch --model facebook/dinov3-vitb16-pretrain-lvd1689m
  ```
- **Offline / CI (`mock`)**: Deterministic SHA-256 multi-view projection encoder requiring zero GPU or network access (`--backend mock` or `BRAGGTRACK_DINO_BACKEND=mock`).

---

## Notebooks

- [`notebooks/braggtrack_demo.ipynb`](notebooks/braggtrack_demo.ipynb) — End-to-end tutorial on the bundled ESRF-ID03 scans: discovery, stabilised 3D segmentation, physics & semantic tracking, per-grain kinematics, α/β ablation, and synthetic 4D grain fracture + precursor ranking.
- [`notebooks/dino_segmentation_comparison.ipynb`](notebooks/dino_segmentation_comparison.ipynb) — Side-by-side comparison of stabilised classical 3D watershed vs. experimental DINO patch-clustering segmentation.

---

## Testing, Acceptance Gates & Pre-PR Checks

```bash
# Lint & format check
ruff check braggtrack/ tests/ scripts/
ruff format --check braggtrack/ tests/ scripts/

# Unit test suite with coverage
pytest tests/ -v --cov=braggtrack

# Full CI gate runner (unit tests + IO, smoke, segmentation, tracking, embedding, DINO, and events acceptance)
python scripts/ci_report.py

# Pre-PR hygiene check
python scripts/pre_pr_check.py
```

---

## Troubleshooting & FAQ

- **`MissingH5DependencyError` or custom HDF5 dataset paths:** `h5py` is installed by default with `pip install -e .`. `load_primary_volume()` automatically inspects standard NeXus paths, `/entry_0000/ESRF-ID03/pco_nf/data`, and falls back to the largest 3D numeric dataset in the file. Pass `--allow-synthetic-fallback` only when testing schema plumbing on empty/metadata-only HDF5 stubs.
- **`401 Unauthorized` / gated HuggingFace model error:** You passed a gated model ID (such as `facebook/dinov3-*`) without authenticating. Either omit `--model` to use the ungated default (`facebook/dinov2-small`), or accept the model license on HuggingFace and export `HF_TOKEN`.
- **Unphysical strain percentages in `kinematics.csv`:** Check `"calibrated_axes"` in `kinematics_summary.json`. When `false`, `centroid_d` is measured in cropped detector pixels (`y` index) rather than Angstroms — pass `--calibration-json` (or `--d-origin` / `--d-step`) to compute true crystallographic $\Delta d / d_0$.

## Repository Layout & Contributing

- **`braggtrack/`** — Installable Python library (`io`, `segmentation`, `semantic`, `tracking`, `cli`).
- **`data/sample_operando/`** — Bundled ESRF-ID03 operando test scans (`scan0001`–`scan0003`).
- **`docs/`** — [Architecture](docs/architecture.md) and [Coding Practices](docs/coding-practices.md).
- **`scripts/`** — Acceptance gates, semantic α/β ablation (`ablation_semantic.py`), and CI runner (`ci_report.py`).
- **`artifacts/`** — Generated pipeline outputs (gitignored; never commit).

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [paper.md](paper.md) for the manuscript overview.

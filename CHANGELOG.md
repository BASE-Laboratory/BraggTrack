# Changelog

All notable changes to BraggTrack will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-17

### Added
- **ESRF-ID03 & generic NeXus HDF5 auto-discovery (`braggtrack.io.nexus`)**: Added `/entry_0000/ESRF-ID03/pco_nf/data` to primary volume candidates, largest-3D numeric dataset fallback (`discover_primary_volume_path`, `load_primary_volume(..., auto_discover=True)`), and `/entry*` / `NXentry` metadata discovery in `extract_scan_metadata`.
- **Physical reciprocal-space calibration (`AxisCalibration`)**: Added `AxisCalibration` contract (`voxel_to_physical`, JSON serialisation), HDF5 axis auto-detection (`extract_axis_calibration`), CLI calibration flags (`--calibration-json`, `--mu-origin/--mu-step`, `--chi-origin/--chi-step`, `--d-origin/--d-step`), and calibrated physical centroids/covariances/eigenvalues in `extract_instance_table` while preserving integer voxel bounding boxes.
- **Per-grain 4D kinematics CLI export**: Wired `compute_grain_kinematics` and `summarize_kinematics` into `braggtrack-track-dataset`, automatically emitting `kinematics.csv` (including `delta_d`) and `kinematics_summary.json` alongside `tracks.csv`.
- **Split (`1 -> 2+`) and merge (`2+ -> 1`) lineage DAG resolution (`braggtrack.tracking.events`)**: Added intensity-partitioning-gated split and merge detection (`resolve_splits_and_merges`, `SplitMergeConfig`, `LineageEvent`, `events_to_table`) with anti-noise minimum fragment share guards and distinct daughter/merged track ID propagation.
- **Persistent 4D fracture precursor scoring (`braggtrack.tracking.precursors`)**: Added `compute_fracture_precursors` combining 3D eigenvalue anisotropy, covariance trace growth, 3D spot ROI sub-peak bimodality, and upsampled MIP DINO patch-token cosine dispersion with multi-scan contiguous persistence filtering (`min_persistence=2`).
- **Synthetic 4D fracture ground-truth generator**: Added `generate_fracture_scenario` producing 4D intensity cubes, label masks, and ground-truth split, merge, persistent precursor, and isotropic control trajectories.
- **New CLI entry points**:
  - `braggtrack-events` (`braggtrack.cli.detect_events`) for standalone split/merge lineage and ROI fracture precursor ranking (`events.csv`, `events.json`, `precursors.csv`, `events_summary.json`).
  - `braggtrack-run` (`braggtrack.cli.run_pipeline`) for one-command reproducible end-to-end execution with JSON config support, stage timing profiling, SHA-256 input checksums, and `run_manifest.json`.
- **Semantic ablation bootstrap confidence intervals**: Added `--bootstrap-samples` to `scripts/ablation_semantic.py` reporting 95% bootstrap CIs (`fragmentation_ratio_ci95`, `id_switch_rate_ci95`).
- **Events & pipeline acceptance gate**: Added `scripts/check_events_acceptance.py` and wired it into `scripts/ci_report.py`.
- Rolling-median threshold smoother (`smooth_thresholds`) wired into `braggtrack-segment-dataset` (`--smooth-window 5`) and outlier frame detection (`flag_outlier_frames`) via MAD-based statistics.
- Label projection by intensity (`label_projection_by_intensity`) and MIP-floor masking via 2-D Otsu (`otsu_floor_from_mip`).
- Optional dependency groups (`[torch]`, `[notebook]`, `[dev]`), `BRAGGTRACK_DATA_ROOT` env var, Ruff configuration, and PEP 561 `py.typed` marker.

### Changed
- **Ungated vision backbone default**: Standardised `DEFAULT_DINO_MODEL` to `"facebook/dinov2-small"` across `semantic` and `segmentation` so local installs, CI, and Google Colab run zero-auth out of the box, while keeping gated DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`) fully supported via `--model` / `--dino-model` with universal dynamic prefix-token skipping (`cls` + register tokens).
- `segment_dataset`, `embed_dataset`, and `detect_events` now load real bundled ESRF-ID03 `(100, 250, 250)` volumes directly via `load_primary_volume` and require explicit `--allow-synthetic-fallback` before substituting synthetic blobs on unreadable HDF5 files.
- Seed floor uses robust peak reference (`p99.99`) and `seed_response_percentile=99.95`.

### Fixed
- Fixed silent fallback to fake `(24, 24, 24)` synthetic volumes when running CLI tools on bundled ESRF-ID03 NeXus/LIMA HDF5 files (`/entry_0000/ESRF-ID03/pco_nf/data`).
- Fixed uncalibrated detector-crop pixel offset distortion in crystallographic strain ($\Delta d / d_0$) when `AxisCalibration` is supplied.
- Fixed threshold domain mismatch (intensity Otsu vs. LoG response domain) and label projection highest-ID bias.

## [0.1.0] - 2025-12-01

### Added
- Initial release: discovery, segmentation (Otsu + connected components), tracking (Hungarian + lifecycle DAG)
- Week 1: beamline adapter, scan discovery, validation
- Week 2: classical LoG segmentation, h-maxima seeds, seeded watershed
- Week 3: position+shape cost, per-axis gating, NetworkX lifecycle graph
- Week 4: multi-view MIPs, mock/DINOv2 encoder, geometry+semantic cost, alpha/beta ablation
- CLI tools: inspect, validate, segment-synthetic, segment-dataset, track-dataset, embed-dataset
- Bundled ESRF-ID03 sample data (3 scans, 100×250×250 uint16)

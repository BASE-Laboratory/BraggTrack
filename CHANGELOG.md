# Changelog

All notable changes to BraggTrack will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-17

### Added
- **Analytical Poisson + quantization centroid/strain uncertainties & ancestor-referenced kinematics (`braggtrack.segmentation.features`, `braggtrack.tracking.kinematics`, `braggtrack.tracking.assignment`)**:
  - Emits per-spot `centroid_mu_err`, `centroid_chi_err`, and `centroid_d_err` combining Poisson photon-counting variance and `1/12` voxel quantization variance.
  - Propagates `strain_err` ($\sigma_\varepsilon$), ancestor-referenced `root_track_id`, `lineage_strain`, and `lineage_strain_err` across `1 -> 2+` fracture splits, plus causal velocities (`rate_mu`, `rate_chi`, `rate_d`, `strain_rate`) and Hungarian runner-up `assignment_margin`.
- **Operando electrochemistry & external stimulus synchronization (`braggtrack.io.operando_log`, `braggtrack.viz.operando`)**:
  - Parses `.csv`/`.tsv` and Biologic EC-Lab `.mpt` potentiostat logs (`load_operando_log`, `synchronize_stimulus_to_scans`), computing Savitzky–Golay regularized differential capacity $\mathrm{d}Q/\mathrm{d}V$ and per-grain electro-strain susceptibility `d_strain_d_v` ($\partial\varepsilon/\partial V$).
  - Added 3-panel publication plot `plot_operando_strain_and_events` and wired `--operando-log` / `--operando-tz-offset` into `braggtrack-report`, `braggtrack-run`, and `report.html`.
- **Batched `DINOv2` ViT inference & zero-config `darfix` / ESRF BLISS NeXus auto-calibration (`braggtrack.semantic.dino`, `braggtrack.io.nexus`)**:
  - Added `embed_batch` and `extract_patch_features_batch` across `MultiviewEncoder` and `PatchFeatureEncoder` (with CPU L3-friendly micro-batching), accelerating `braggtrack-embed-dataset`, `compute_fracture_precursors`, and `segment_dino` while guaranteeing bit-identical cosine parity.
  - Added `NXdata` `@default` / `@signal` / `@axes` (`darfix` processed HDF5) and ESRF BLISS `instrument/positioners` auto-calibration in `discover_primary_volume_path` and `extract_axis_calibration`.
- **Online causal streaming, Gaussian Process trajectory forecasting & Bayesian "Scout-and-Zoom" DFXM engine (`braggtrack.autonomous`, `braggtrack-watch`)**:
  - Added `forecast_1d_gp`, `forecast_grain_trajectory_gp`, `evaluate_acquisition`, and `score_candidates_gp` (`gp.py`), combining inverse-variance-weighted linear trend estimation with a heteroscedastic `Matérn-3/2` Gaussian Process (`alpha = sigma_pos**2`) for smooth `(mu, chi, d)` setpoint forecasting and `UCB` / `EI` / `max_variance` Bayesian target ranking.
  - Added `Bluesky-Adaptive` style `tell()` / `ask()` / `tell_zoom_outcome()` / `report()` methods and $O(1)$ bounded sliding-window incremental tracking (`sliding_window_scans=24`) in `CausalStreamingTracker` (`streaming.py`), maintaining constant per-scan latency across `1,000+` scan operando runs while preserving global track IDs and root strain anchors.
  - Added staged execution modes (`--mode {replay, shadow, watch, active}`), `--acquisition {ucb, ei, max_variance}`, diffractometer motor soft-limit clamping, electrochemical voltage bounds, per-lineage X-ray dose caps, synchrotron beam-dump suppression, and live HDF5 file-quiescence + `SWMR` polling in `braggtrack-watch`, emitting `bliss_dfxm_zoom.v1` commands (`zoom_triggers.jsonl`) and baseline replay metrics including `mean_setpoint_mae`.
- **Crystallographic physics & Red-Team hardening (`segmentation`, `semantic`, `tracking`, `viz`, `cli`)**:
  - Added outer-perimeter background pedestal subtraction (`background_level`, `net_intensity`), gain-aware Poisson counting variance (`detector_gain`), standardized Pearson rocking-curve excess kurtosis (`kurtosis_mu`, `kurtosis_chi`, `kurtosis_d`), dimensionless voxel eigenvalues (`vox_eig_1..3`), and boundary truncation flags (`touches_border`, `touches_rocking_border`) in `extract_instance_table`, plus `float32` classical segmentation memory reduction (`segment_classical`).
  - Added step-normalized physical-unit invariance across `PositionShapeCost`, `resolve_track_gaps`, and `resolve_splits_and_merges` so calibrated `AxisCalibration` runs (`deg`, `Å`) produce identical trajectories to uncalibrated voxel space, plus `mechanical_split` vs. `phase_split` (`H2 -> H3`) split classification and wall-clock `strain_rate_per_s`.
  - Added aspect-ratio-preserving square letterbox padding for global `DINOv2` MIP embeddings, strict `--config` key validation in `braggtrack-run`, and Content-Security-Policy (CSP) + HTML entity escaping against DOM XSS in `generate_html_report`.
- **Ecosystem positioning, autonomous facility bibliography & release workflow (`docs/positioning.md`, `.github/workflows/release.yml`)**:
  - Added `docs/positioning.md` positioning `BraggTrack` alongside `darfix`, `ImageD11`, `HEXRD`, `BraggNN`, `TrackMate 7`, `LapTrack`, `gpCAM`, `Bluesky-Adaptive`, `ANDiE`, and the Facility Autonomy Levels (FAL) taxonomy, updated `paper.md` and `paper.bib` with verified DOIs (`@GarrigaFerrer2023`, `@Isern2025`, `@Bernier2011`, `@Liu2022`, `@Fukai2023`, `@Ershov2022`, `@Drnec2025`, `@LeHoux2026`, `@Lu2025`, `@LeHouxFAL2026`, `@Szymanski2023`, `@McDannald2022`, `@Noack2021`, `@Maffettone2023`, `@Kalinin2021`), bumped metadata to `0.4.0`, and added tag-triggered sdist/wheel + Twine + GitHub Release / PyPI Trusted Publisher workflow.

## [0.3.0] - 2026-11-17

### Added
- **Multi-scan beam-dropout gap closing (`braggtrack.tracking.gap_closing`)**: Added Jaqaman-style augmented linear assignment (`GapClosingConfig`, `GapCloseRecord`, `resolve_track_gaps`) with diffusive displacement scaling (`pos_d2 / delta_scans`), bidirectional spurious jump-edge severing, semantic cosine gating, and chronological track ID propagation (`--max-gap` across `track_dataset`, `detect_events`, and `run_pipeline`).
- **Chunked & lazy HDF5 streaming (`braggtrack.io.lazy`)**: Added context-managed `LazyVolume` (`open_lazy_volume`, `read_roi`, `iter_z_chunks`) and exact two-pass `otsu_threshold_chunked`, wired into `segment_dataset` Pass 1, `embed_dataset`, `detect_events`, and `crop_spot_cube` to avoid full-volume `float64` allocations on gigabyte-scale detector stacks.
- **Crystallographic scattering geometry & downstream exports (`braggtrack.io.crystallography`, `braggtrack.io.export`)**: Added `DetectorGeometry` (`spot_to_scattering`, `annotate_crystallographic_columns`) converting spot centroids into `(omega_deg, two_theta_deg, eta_deg, d_spacing_angstrom, qx/qy/qz/q_mag_inv_angstrom)` via Bragg's law, plus direct `ImageD11` columnfile export/parsing (`export_imaged11_flt`, `read_imaged11_flt`) and self-describing NeXus `NXprocess` HDF5 archival export (`export_nexus_process`).
- **Publication visualization package & interactive HTML 4D Explorer (`braggtrack.viz`, `braggtrack-report`)**: Added `plot_spot_mip_card`, `plot_scan_projection_overlay`, `plot_lineage_dag`, `plot_kinematics_dashboard`, `plot_precursor_timeline`, and `generate_html_report` (zero-CDN offline `report.html` with stdlib-encoded base64 3-view MIP thumbnails, pan/zoom SVG Lineage DAG, filter pills, and synchronized strain/precursor inspector), plus the 9th CLI entry point `braggtrack-report` and `--report` flag on `braggtrack-run`.
- **Quantitative 5-scenario real DINOv2 benchmark suite (`scripts/benchmark_suite.py`, `docs/benchmarks.md`)**: Added multi-trial bootstrap-CI benchmark harness comparing **Geometry-Only**, **Mock-Hash**, and **Real `facebook/dinov2-small`** across crossing tilted reflections, beam-flux jitter, mosaic rocking broadening, staggered beam dropouts (`max_gap=2`), and 4D chemomechanical fracture/precursor contrast, alongside publication figures (`docs/figures/`) and an ESRF-ID03 `facebook/dinov2-small` case study.
- **9th CI acceptance gate (`scripts/check_report_and_export_acceptance.py`)**: Wired automated verification of `--max-gap`, `report.html` + static figures, `ImageD11` `.flt` round-trip parsing, and NeXus `NXprocess` HDF5 schema into `scripts/ci_report.py`.

### Fixed
- Fixed patch-grid shape mismatch and far-edge clipping in `segment_dino` when HuggingFace ViT processors resize non-multiple slices (e.g. `250x250 -> 224x224`, `16x16` tokens vs `250 // 14 = 17`), and honored `BRAGGTRACK_DINO_BACKEND` when `backend="auto"` in `braggtrack.semantic.dino`.
- Fixed `@Finegan2017` citation key/year mismatch (`2017`, DOI `10.1039/C7EE00385D`) and added `@Jaqaman2008` in `paper.bib` and `paper.md`; synced `CITATION.cff`, `CONTRIBUTING.md`, `docs/architecture.md`, and `README.md` to `v0.3.0`.

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

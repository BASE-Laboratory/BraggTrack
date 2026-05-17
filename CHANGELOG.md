# Changelog

All notable changes to BraggTrack will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Rolling-median threshold smoother (`smooth_thresholds`) for stable multi-frame segmentation
- Outlier frame detection (`flag_outlier_frames`) via MAD-based statistics
- Label projection by intensity (`label_projection_by_intensity`) — replaces broken `labels.max(axis=k)`
- MIP-floor masking via 2-D Otsu (`otsu_floor_from_mip`)
- Tri-axis segmented mask visualisation in demo notebook
- Semantic MIP gallery and PCA embedding space plots
- Google Colab support with auto-install cell and "Open in Colab" badge
- CI matrix (Python 3.10/3.11/3.12) with pip caching and notebook execution job
- Optional dependency groups: `[torch]`, `[notebook]`, `[dev]`
- `BRAGGTRACK_DATA_ROOT` env var for custom data locations
- Ruff linting and formatting configuration
- PEP 561 `py.typed` marker

### Changed
- Seed floor now uses robust peak reference (p99.99) instead of absolute max
- Default `seed_response_percentile` raised from 99.5 to 99.95
- `torch` and `transformers` moved to optional `[torch]` extra (bare install is lightweight)

### Fixed
- Critical threshold domain mismatch: intensity Otsu was applied to LoG response domain
- Spot count instability across scans (11/22/36 → 18/20/16 on bundled data)
- Label projection picking highest label ID instead of brightest voxel's label

## [0.1.0] - 2025-12-01

### Added
- Initial release: discovery, segmentation (Otsu + connected components), tracking (Hungarian + lifecycle DAG)
- Week 1: beamline adapter, scan discovery, validation
- Week 2: classical LoG segmentation, h-maxima seeds, seeded watershed
- Week 3: position+shape cost, per-axis gating, NetworkX lifecycle graph
- Week 4: multi-view MIPs, mock/DINOv2 encoder, geometry+semantic cost, alpha/beta ablation
- CLI tools: inspect, validate, segment-synthetic, segment-dataset, track-dataset, embed-dataset
- Bundled ESRF-ID03 sample data (3 scans, 100×250×250 uint16)

---
title: 'BraggTrack: Semantic 4D Kinematics and Fracture Tracking for Operando Diffraction'
tags:
  - Python
  - X-ray diffraction
  - operando crystallography
  - 3D segmentation
  - object tracking
  - grain fracture
  - materials science
authors:
  - name: James Le Houx
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: BASE Laboratory, School of Engineering, University of Greenwich, United Kingdom
    index: 1
date: 16 May 2026
bibliography: paper.bib
---

# Summary

BraggTrack is a Python library for automated 3D segmentation, semantic tracking,
kinematic strain extraction, and chemomechanical fracture detection across
sequential operando X-ray diffraction (XRD) volumes. Given a time series of
three-dimensional reciprocal-space or near-field diffraction cubes acquired
during *in situ* or *operando* experiments, BraggTrack segments individual
Bragg reflections, extracts calibrated geometric and foundation vision features,
and links observations across scans into a directed acyclic trajectory graph
capturing births, continuations, terminations, `1 -> 2+` grain fracture splits,
and `2+ -> 1` coalescence merges. In addition, BraggTrack ranks persistent
pre-fracture precursor candidates across scans by combining 3D reflection
anisotropy, covariance broadening, internal sub-peak bimodality, and frozen
vision-transformer patch-token dispersion. Designed for synchrotron NeXus/HDF5
data, BraggTrack ships with bundled operando scans from ESRF beamline ID03,
eight CLI entry points including a single-command reproducible pipeline runner
(`braggtrack-run`), and interactive Jupyter notebooks runnable on Google Colab.

# Statement of Need

Operando diffraction experiments at modern synchrotron sources generate
hundreds to thousands of three-dimensional diffraction volumes at sub-minute or
sub-second cadence, each containing tens to hundreds of Bragg reflections whose
positions, intensities, and 3D morphologies evolve during electrochemical
cycling, mechanical loading, or thermal treatment [@Simons2015; @Hayashi2019].
In battery electrodes such as Ni-rich layered oxides, chemomechanical
degradation manifests first as anisotropic intragranular strain broadening and
sub-domain formation before culminating in particle microcracking — observable
in reciprocal space as reflection broadening followed by `1 -> 2+` peak
splitting [@Finegan2019]. Quantifying per-grain lattice strain
($\varepsilon = \Delta d / d_0$), lattice rotation ($\Delta\mu$, $\Delta\chi$),
and fracture lineage across long scan sequences is essential for linking local
crystallography to macroscopic failure, as well as for solidification and
additive manufacturing studies [@Leung2018].

Existing workflows rely on frame-by-frame peak fitting, bespoke experiment
scripts, or crystallographic indexing packages such as GrainSpotter
[@Schmidt2014], 3DXRD orientation mappers [@Poulsen2004], and ImageD11
[@Wright2023] that focus on unit-cell orientation indexing rather than tracking
evolving, splitting, and broadening 3D intensity distributions through time.
Previously, researchers lacked an open-source, `pip`-installable toolkit
combining stabilised 3D spot segmentation, foundation-model visual descriptors,
calibrated per-grain kinematics, and explicit split/merge fracture lineage in a
reproducible software architecture.

BraggTrack addresses this need by separating beamline HDF5 ingestion and axis
calibration, 3D instance segmentation, multi-view vision-transformer encoding,
Hungarian and topological lineage tracking, and fracture precursor scoring into
cleanly tested, modular layers.

# Implementation

## Beamline ingestion and physical axis calibration

BraggTrack discovers sequential `scan*` directories, inspects NeXus/HDF5 trees
(including ESRF-ID03 LIMA detector hierarchies under `/entry_0000/...` and
generic HDF5 layouts via automatic largest-3D numeric dataset discovery), and
validates timestamp monotonicity. An `AxisCalibration` transform converts raw
detector indices into physical reciprocal-space coordinates
$(\mu, \chi, d)$ via HDF5 axis metadata or JSON/CLI calibration parameters while
preserving integer voxel bounding boxes for lossless 3D region-of-interest (ROI)
cropping. This ensures crystallographic strain $\Delta d / d_0$ is invariant to
detector window cropping.

## Stabilised 3D segmentation

BraggTrack's primary segmentation pipeline processes 3D volumes in four stages:

1. **Stabilised foreground thresholding.** Per-scan Otsu thresholds [@Otsu1979]
   on raw intensity histograms are smoothed across scans with a rolling median
   (`smooth_thresholds`) and checked for median-absolute-deviation outliers,
   eliminating spot-count jitter caused by transient beam fluctuations.

2. **3D LoG enhancement.** Gaussian smoothing followed by the negative discrete
   Laplacian highlights Bragg reflections as local maxima [@Lindeberg1998]
   independently of intensity-domain foreground masking.

3. **Robust seed detection.** `h`-maxima filtering [@Soille2003] selects seeds
   exceeding both a fraction of the 99.99th-percentile foreground response and a
   high response percentile with non-maximum suppression.

4. **Seeded watershed and basin refinement.** Watershed region growing
   [@Vincent1991] over the inverted LoG response is followed by small-object
   removal, 3D hole filling, and optional nearby-centroid merging to prevent
   over-segmentation of mosaic reflections. An experimental dense patch-feature
   + HDBSCAN slice-stitching segmenter (`segment_dino`) is also provided for
   cross-method comparison.

## Multi-view semantic descriptors

Each segmented 3D reflection is cropped, masked to its instance support, and
projected into three orthogonal maximum-intensity projections (MIPs) along
$\mu$, $\chi$, and $d$. By default, BraggTrack encodes these views using the
public, ungated DINOv2 vision transformer (`facebook/dinov2-small`)
[@Oquab2024] into an $L_2$-normalised 384-d descriptor, while also supporting
newer DINOv3 backbones (`facebook/dinov3-vitb16-pretrain-lvd1689m`) via dynamic
prefix/register token handling and a deterministic hash-based `mock` encoder for
zero-GPU continuous integration.

## Tracking, kinematics, and chemomechanical fracture events

Frame-to-frame 1-to-1 association solves the gated Hungarian assignment problem
[@Kuhn1955] over $\alpha \cdot C_{\text{geo}}(i, j) + \beta \cdot (1 -
\cos(\mathbf{f}_i, \mathbf{f}_j))$, constructing a NetworkX [@Hagberg2008]
`DiGraph`. From each trajectory, `compute_grain_kinematics` extracts time-series
strain $\Delta d / d_0$, rocking/azimuthal misorientation, integrated intensity
and volume evolution, and 3D eigenvalue anisotropy.

To capture grain fracture and coalescence, `resolve_splits_and_merges` augments
unmatched birth and termination nodes with explicit `1 -> 2+` split and
`2+ -> 1` merge edges gated by spatial proximity, integrated-intensity
conservation ($I_{\text{children}} / I_{\text{parent}}$), and minimum fragment
intensity share, assigning clean daughter track IDs so diverging post-fracture
strains are tracked independently. Complementarily,
`compute_fracture_precursors` scores pre-split intragranular degradation across
contiguous scans (`min_persistence`) using 3D eigenvalue anisotropy, covariance
trace growth, internal 3D sub-peak bimodality, and foreground DINO patch-token
cosine dispersion.

## Reproducibility and deployment

BraggTrack (`pip install braggtrack`) provides eight CLI commands, culminating
in `braggtrack-run`, which executes validation, segmentation, embedding,
tracking, kinematics, and event detection and writes a `run_manifest.json`
recording SHA-256 input checksums, configuration, software version, and per-stage
runtimes. Automated CI tests Python 3.10–3.12, enforces Ruff linting/formatting,
runs seven acceptance gates (`scripts/ci_report.py`), and executes the tutorial
notebook end-to-end on bundled ESRF-ID03 data.

# Acknowledgements

The authors acknowledge the European Synchrotron Radiation Facility (ESRF)
for provision of beamtime at beamline ID03. This work was supported by the
BASE Laboratory at the University of Greenwich.

# References

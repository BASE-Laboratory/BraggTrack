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
and `2+ -> 1` coalescence merges, as well as multi-scan gap closures across
transient beam dropouts. In addition, BraggTrack ranks persistent pre-fracture
precursor candidates across scans by combining 3D reflection anisotropy,
covariance broadening, internal sub-peak bimodality, and frozen
vision-transformer patch-token dispersion. Designed for synchrotron NeXus/HDF5
data, BraggTrack supports zero-configuration
`darfix` and ESRF BLISS motor auto-calibration, lazy HDF5 ROI streaming,
batched vision-transformer inference, analytical Poisson centroid/strain
uncertainties, operando electrochemistry synchronization (`--operando-log`),
crystallographic $(2\theta, \eta, d, |\mathbf{Q}|)$ conversion with direct
`ImageD11` (`.flt`) and NeXus `NXprocess` HDF5 export, publication plotting and
offline interactive HTML 4D lineage reports (`braggtrack-report`), causal
online streaming with velocity-extrapolated Dark-Field X-ray Microscopy (DFXM)
zoom triggering (`braggtrack-watch`), bundled operando scans from ESRF beamline
ID03, ten CLI entry points, and interactive Jupyter notebooks runnable on
Google Colab.

# Statement of Need

Operando diffraction experiments at modern synchrotron sources generate
hundreds to thousands of three-dimensional diffraction volumes at sub-minute or
sub-second cadence, each containing tens to hundreds of Bragg reflections whose
positions, intensities, and 3D morphologies evolve during electrochemical
cycling, mechanical loading, or thermal treatment [@Simons2015; @Hayashi2019; @Isern2025].
Yet brute-force grid automation faces a recognized crisis in reliability,
representativeness, and reproducibility [@Drnec2025], expending the majority of
incident flux on bulk elastic volume while missing spatially localized transient
failure events [@Lu2025]. In battery electrodes such as Ni-rich layered oxides,
chemomechanical degradation initiates deterministically at surface stress
concentrators as localized dislocation pileups, non-Gaussian rocking-curve tails,
and anisotropic sub-domain formation before culminating in particle microcracking
— observable in reciprocal space as reflection broadening followed by `1 -> 2+`
peak splitting [@Finegan2017; @LeHoux2026]. Quantifying per-grain lattice strain
($\varepsilon = \Delta d / d_0$), lattice rotation ($\Delta\mu$, $\Delta\chi$),
standardized rocking-curve excess kurtosis ($g_2$), and fracture lineage across
long scan sequences — and steering high-resolution DFXM zoom onto pre-fracture
grains online to maximize Entropy-Scaled Measurement Efficiency ($E_{\text{SME}}$)
[@Lu2025] — is essential for linking local crystallography to macroscopic
failure, as well as for solidification and additive manufacturing studies [@Leung2018].

Existing diffraction, autonomous steering, and tracking packages address
complementary parts of this workflow: crystallographic indexing suites such as
GrainSpotter [@Schmidt2014], 3DXRD orientation mappers [@Poulsen2004], HEXRD
[@Bernier2011], and ImageD11 [@Wright2023] determine unit-cell orientations and
static grain tensors; deep neural localizers such as BraggNN [@Liu2022]
accelerate 2D sub-pixel peak finding; DFXM analysis suites such as *darfix*
[@GarrigaFerrer2023] map pixel-wise rocking-curve moments once a single-grain
zoom scan has been acquired; adaptive powder/single-crystal steering frameworks
[@McDannald2022; @Szymanski2023] optimize scalar phase or magnetic peak
acquisition without 4D multi-grain fracture lineage; and bioimage
linear-assignment trackers such as TrackMate 7 [@Ershov2022] and LapTrack
[@Fukai2023] link cell trajectories without reciprocal-space Bragg strain
physics, integrated-intensity split conservation, or synchrotron beamline
metadata. Previously, researchers lacked an open-source, `pip`-installable
toolkit combining stabilised 3D spot segmentation, foundation-model visual
descriptors, calibrated per-grain kinematics with analytical uncertainties,
explicit mechanical vs. two-phase split/merge lineage, operando potentiostat
synchronization, and Facility Autonomy Level 3 (FAL L3) [@LeHouxFAL2026] causal
DFXM target selection.

BraggTrack addresses this need by separating beamline HDF5 ingestion and axis
calibration, operando stimulus synchronization, 3D instance segmentation,
batched multi-view vision-transformer encoding, step-normalized Hungarian and
topological lineage tracking, fracture precursor scoring, FAL L3 causal
streaming (`braggtrack.autonomous`), and interactive visualization
(`braggtrack.viz`) into cleanly tested, modular layers.

# Implementation

## Beamline ingestion, lazy HDF5 streaming, operando sync, and crystallographic mapping

BraggTrack discovers sequential `scan*` directories, inspects NeXus/HDF5 trees
(including `NXdata` `@default`/`@signal` hierarchies exported by *darfix*
[@GarrigaFerrer2023], ESRF-ID03 LIMA detector hierarchies under `/entry_0000/...`
[@Isern2025], and generic HDF5 layouts via automatic largest-3D numeric dataset
discovery with Single-Writer Multiple-Reader `SWMR` support), and validates
timestamp monotonicity. To handle gigabyte-scale detector stacks without
exhausting host memory, `LazyVolume` streams `z`-slabs for exact two-pass global
Otsu histogramming (`otsu_threshold_chunked`) and reads only tight 3D spot
bounding-box slices (`read_roi`) during semantic embedding and precursor analysis.

An `AxisCalibration` transform — supplied via JSON or auto-extracted from
NeXus `@axes` arrays and ESRF BLISS `instrument/positioners` metadata — converts
raw detector indices into physical reciprocal-space coordinates $(\mu, \chi, d)$
while preserving integer voxel bounding boxes for lossless 3D ROI cropping. When
operando electrochemistry logs (`.csv`, `.tsv`, or Biologic EC-Lab `.mpt`) are
provided via `--operando-log`, `braggtrack.io.operando_log` synchronizes cell
potential $E(t)$, current $I(t)$, specific capacity $Q(t)$, regularized
differential capacity $\mathrm{d}Q/\mathrm{d}V$, and per-grain electro-strain
susceptibility $\partial\varepsilon/\partial V$ onto scan timestamps.
Complementarily, `DetectorGeometry` maps transmission-diffraction spot centroids
into crystallographic scattering angle $2\theta$, azimuth $\eta$, $d$-spacing
($\lambda = 2 d \sin\theta$), and reciprocal-lattice vectors $\mathbf{Q}$
($\text{\AA}^{-1}$), exporting directly to `ImageD11` columnfiles (`.flt`)
[@Wright2023] and self-describing NeXus `NXprocess` HDF5 archives (`.nxs`).

## Stabilised 3D segmentation

BraggTrack's primary segmentation pipeline processes 3D volumes in `float32`
across four stages:

1. **Stabilised foreground thresholding.** Streaming per-scan Otsu thresholds
   [@Otsu1979] on raw intensity histograms are smoothed across scans with a
   rolling median (`smooth_thresholds`) and checked for median-absolute-deviation
   outliers, eliminating spot-count jitter caused by transient beam fluctuations.

2. **3D LoG enhancement.** Gaussian smoothing followed by the negative discrete
   Laplacian highlights Bragg reflections as local maxima [@Lindeberg1998]
   independently of intensity-domain foreground masking.

3. **Robust seed detection.** `h`-maxima filtering [@Soille2003] selects seeds
   exceeding both a fraction of the 99.99th-percentile foreground response and a
   high response percentile with non-maximum suppression.

4. **Seeded watershed, pedestal subtraction, and rocking-curve kurtosis.**
   Watershed region growing [@Vincent1991] over the inverted LoG response is
   followed by small-object removal, 3D hole filling, local exterior background
   pedestal subtraction, gain-aware analytical Poisson plus voxel-quantization
   centroid standard errors, standardized Pearson rocking-curve excess kurtosis
   ($g_2 = \mu_4 / \mu_2^2 - 3$), and rocking-scan boundary truncation flags
   (`touches_rocking_border`). An experimental batched dense patch-feature +
   HDBSCAN slice-stitching segmenter (`segment_dino`) is also provided for
   cross-method comparison.

## Batched multi-view semantic descriptors

Each segmented 3D reflection is cropped via `LazyVolume`, masked to its instance
support, and projected into three orthogonal maximum-intensity projections
(MIPs) along $\mu$, $\chi$, and $d$ with aspect-ratio-preserving square
letterbox padding. By default, BraggTrack batches all $3 N_{\text{spots}}$ views
per scan through the public, ungated DINOv2 vision transformer
(`facebook/dinov2-small`) [@Oquab2024] into a concatenated $L_2$-normalised
1152-d descriptor (`embed_batch`), while also supporting newer DINOv3 backbones
(`facebook/dinov3-vitb16-pretrain-lvd1689m`) via dynamic prefix/register token
handling and a deterministic hash-based `mock` encoder for zero-GPU continuous
integration.

## Tracking, gap closing, kinematics, and chemomechanical fracture events

Frame-to-frame 1-to-1 association solves the gated Hungarian assignment problem
[@Kuhn1955] over step-normalized $\alpha \cdot C_{\text{geo}}(i, j) + \beta \cdot
(1 - \cos(\mathbf{f}_i, \mathbf{f}_j))$ — guaranteeing exact trajectory
invariance under arbitrary physical `AxisCalibration` scales — while recording
per-link assignment cost margins (`assignment_margin` to the runner-up
candidate) and constructing a NetworkX [@Hagberg2008] `DiGraph`. Prior to
topological event detection, `resolve_track_gaps` solves an augmented multi-scan
linear assignment problem [@Jaqaman2008] with diffusive displacement scaling
($\Delta r^2 / \Delta t$) and semantic cosine gating to reconnect reflections
across transient beam dropouts (`--max-gap`) while severing spurious
cross-detector jumps. From each trajectory, `compute_grain_kinematics` extracts
local time-series strain $\Delta d / d_0$, ancestor-referenced `lineage_strain`
across fracture splits, propagated strain uncertainty $\sigma_\varepsilon$,
per-scan and wall-clock strain rates (`strain_rate`, `strain_rate_per_s`),
integrated intensity and volume evolution, and dimensionless 3D eigenvalue
anisotropy.

To capture grain fracture and coalescence, `resolve_splits_and_merges` augments
remaining unmatched birth and termination nodes with explicit `1 -> 2+` split
and `2+ -> 1` merge edges gated by step-normalized proximity,
integrated-intensity conservation ($I_{\text{children}} / I_{\text{parent}}$),
and minimum fragment intensity share, classifying splits into crystallographic
orientation cleavage (`mechanical_split`), first-order two-phase lattice jumps
(`phase_split`, e.g., `H2 -> H3` phase coexistence), or `mixed_split`
(\autoref{fig:lineage}). Complementarily, `compute_fracture_precursors` scores
pre-split intragranular degradation across contiguous scans (`min_persistence`)
using 3D eigenvalue anisotropy, covariance trace growth, internal 3D sub-peak
bimodality, and batched foreground DINO patch-token cosine dispersion.

![Four-dimensional chemomechanical fracture lineage, pre-split bimodal MIP evolution, persistent precursor decomposition (`facebook/dinov2-small`), and calibrated ESRF-ID03 Ni-rich cathode per-grain strain trajectories.\label{fig:lineage}](docs/figures/fracture_lineage_and_precursors.png)

## Causal streaming, Gaussian process forecasting, and autonomous DFXM targeting

Building on Gaussian process and Bayesian active-learning architectures established
across synchrotron, neutron, and microscopy user facilities [@Noack2021;
@McDannald2022; @Kalinin2021; @Szymanski2023], `braggtrack.autonomous`
(`CausalStreamingTracker` and `braggtrack-watch`) adopts the decoupled
`tell` / `ask` / `report` agent interface standardized by `Bluesky-Adaptive`
[@Maffettone2023]. Bounded $O(1)$ sliding-window DAG updates (`tell`) maintain
sub-second per-scan latency across `1,000+` scan operando campaigns without
future lookahead. During rapid electrochemical phase transitions (`H2 -> H3`),
`forecast_grain_trajectory_gp` combines inverse-variance-weighted linear trend
estimation with a heteroscedastic `Matérn-3/2` Gaussian process weighted by
analytical Poisson centroid variances ($\alpha_i = \sigma_{\text{pos},i}^2$) to
forecast smooth $(\mu, \chi, d)$ setpoints and posterior `1-sigma` rocking-window
widths across uniform or irregular scan timestamps. Candidate grains are ranked
(`ask`) via Upper Confidence Bound (`ucb`), Expected Improvement (`ei`), or pure
variance exploration (`max_variance`), with optional closed-loop conditioning on
high-resolution DFXM zoom outcomes (`tell_zoom_outcome`). Within the Facility
Autonomy Levels taxonomy [@LeHouxFAL2026; @Lu2025], `braggtrack-watch` supports
staged `replay`, live non-actuating `shadow`, and `active` closed-loop operation
under diffractometer soft limits, electrochemical voltage bounds, per-grain X-ray
dose caps, and synchrotron beam-dump suppression, emitting `bliss_dfxm_zoom.v1`
JSONL commands (`zoom_triggers.jsonl`) and baseline replay benchmarks
(`evaluate_causal_replay`).

## Quantitative evaluation, visualization, and reproducibility

Across controlled 4D benchmarks (`scripts/benchmark_suite.py`;
\autoref{fig:benchmarks}), multi-view `facebook/dinov2-small` descriptors
maintain $> 0.96$ adjacent-scan cosine similarity under shot noise, 45% beam-flux
jitter, and mosaic rocking-curve broadening (where byte-hash mocks collapse to
$\approx 0$), eliminate crossing ID switches between reflections with identical
3D covariance eigenvalues (`+45` vs. `-45` deg detector tilts), and recover
100% of staggered beam dropouts with `--max-gap 2`.

![Quantitative 4D diffraction benchmark suite comparing Geometry-Only, Mock-Hash, and Real `facebook/dinov2-small` across crossing tilted reflections, flux/mosaic noise invariance, and staggered beam-dropout recovery.\label{fig:benchmarks}](docs/figures/benchmark_comparison.png)

BraggTrack (`pip install braggtrack`) provides ten CLI commands, including
`braggtrack-report` (generating publication figures, 3-panel operando
electrochemistry $\mathrm{d}Q/\mathrm{d}V$ dashboards, and a zero-CDN offline
interactive HTML 4D Lineage & Strain Explorer), `braggtrack-watch` (live causal
streaming and offline autonomous DFXM replay), and `braggtrack-run`, which
executes validation, segmentation, embedding, tracking, kinematics, event
detection, reporting, and `ImageD11`/NeXus export while writing a
`run_manifest.json` with SHA-256 input checksums and per-stage runtimes.
Automated CI tests Python 3.10–3.12, enforces Ruff linting/formatting, runs nine
acceptance gates (`scripts/ci_report.py`), and executes the tutorial notebook
end-to-end on bundled ESRF-ID03 data.

# Acknowledgements

The authors acknowledge the European Synchrotron Radiation Facility (ESRF)
for provision of beamtime at beamline ID03. This work was supported by the
BASE Laboratory at the University of Greenwich.

# References

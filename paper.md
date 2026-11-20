---
title: 'BraggTrack: 4D Bragg Reflection Tracking, Kinematics, and Lineage for Operando X-ray Diffraction'
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
    orcid: 0000-0002-1576-0673
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: BASE Laboratory, School of Engineering, University of Greenwich, United Kingdom
    index: 1
  - name: The Faraday Institution, Harwell Campus, Didcot, United Kingdom
    index: 2
date: 19 November 2026
bibliography: paper.bib
---

# Summary

`BraggTrack` is an open-source Python library for automated 3D Bragg reflection
segmentation, multi-scan trajectory tracking, kinematic strain extraction, and
split/merge lineage analysis across sequential *in situ* and *operando* X-ray
diffraction (XRD) volumes. Given a time series of 3D reciprocal-space or
near-field NeXus/HDF5 detector stacks, `BraggTrack` segments individual
reflections above detector read-noise pedestals, extracts calibrated 3D moments
with analytical Poisson centroid uncertainties, and links observations across
scans into a directed acyclic trajectory graph (`DiGraph`) capturing births,
continuations, terminations, beam-dropout gap closures, `1 -> 2+` grain splits,
and `2+ -> 1` coalescence merges. It synchronizes external electrochemistry or
mechanical logs onto scan timestamps, exports directly to `ImageD11` (`.flt`)
and NeXus `NXprocess` HDF5 archives, renders offline interactive HTML 4D
reports, and provides a strictly causal streaming agent (`tell` / `ask` /
`report`) with Gaussian process (GP) setpoint forecasting for autonomous
Dark-Field X-ray Microscopy (DFXM) zoom targeting.

# Statement of Need

Modern synchrotron *operando* diffraction experiments acquire hundreds to
thousands of 3D diffraction volumes at sub-minute cadence, each containing tens
to hundreds of reflections whose positions, intensities, and shapes evolve
during battery cycling, mechanical loading, or solidification [@Simons2015;
@Hayashi2019; @Leung2018; @Isern2025]. Static grid scanning expends most
incident X-ray flux on elastic bulk grains while missing localized transient
degradation [@Drnec2025; @Lu2025]. In layered oxide battery cathodes,
chemomechanical failure initiates at surface stress concentrators as lattice
rotation, rocking-curve broadening, and sub-grain splitting (`1 -> 2+`)
[@Finegan2017; @LeHoux2026]. Tracking per-grain lattice strain
($\varepsilon = \Delta d / d_0$), angular misorientation ($\Delta\mu,
\Delta\chi$), and fracture lineage across long scan sequences—and steering
high-resolution DFXM zoom onto candidate grains online [@LeHouxFAL2026]—requires
linking reflections reliably through rapid lattice shifts and crowded
polycrystalline layers.

Existing packages solve adjacent parts of this problem: `GrainSpotter`
[@Schmidt2014], 3DXRD suites [@Poulsen2004], `HEXRD` [@Bernier2011], and
`ImageD11` [@Wright2023] index unit-cell orientations per scan; `BraggNN`
[@Liu2022] accelerates 2D peak localization; `darfix` [@GarrigaFerrer2023] maps
rocking-curve moments once a single-grain DFXM zoom scan is acquired; Bayesian
diffraction controllers [@McDannald2022; @Szymanski2023] steer scalar phase
acquisitions without multi-grain lineage; and bioimage trackers (`TrackMate 7`
[@Ershov2022], `LapTrack` [@Fukai2023]) lack reciprocal-space calibration,
collective rocking-slab drift registration, and Bragg intensity conservation.
`BraggTrack` bridges raw 3D NeXus/BLISS/`darfix` detector streams and
grain-resolved 4D kinematics and lineage in a modular, installable Python
package (`pip install "git+https://github.com/BASE-Laboratory/BraggTrack.git"`).

# Design and Implementation

1. **Lazy HDF5 ingestion and pedestal-safe 3D segmentation.** `LazyVolume`
   streams `z`-slabs from NeXus, *darfix*, and ESRF-ID03 LIMA HDF5 files with
   automatic motor calibration (`AxisCalibration`) and potentiostat log
   synchronization. Because unconstrained Otsu thresholding [@Otsu1979] splits
   the electronic read-noise peak on sparse sCMOS frames,
   `detector_foreground_threshold` enforces a `median + k · MAD` read-noise
   floor before rolling-median Otsu smoothing, `float32` 3D Laplacian-of-Gaussian
   enhancement [@Lindeberg1998], `h`-maxima [@Soille2003] seeded watershed
   [@Vincent1991], and gain-aware Poisson plus quantization error propagation.
2. **Classical optimal-transport tracking, drift registration, and lineage.**
   Frame-to-frame association solves the Jaqaman birth/death linear assignment
   problem [@Kuhn1955; @Jaqaman2008] in step-normalized coordinates using either
   `PositionShapeCost` or closed-form 3D Gaussian Bures–Wasserstein optimal
   transport (`WassersteinBuresCost`) [@Bhatia2019] with log-intensity
   conservation, preceded by optional sub-pixel 2D FFT rocking-slab phase
   correlation [@GuizarSicairos2008] and 6D constant-velocity Kalman prediction
   [@Kalman1960] for crowded layers. Multi-view DINOv2 descriptors [@Oquab2024]
   are available opt-in (`--cost-beta > 0`). Diffusive gap closing reconnects
   beam dropouts, while `cKDTree`-indexed lineage resolution links intensity-conserved
   `1 -> 2+` splits and `2+ -> 1` merges (`mechanical_split`, `phase_split`,
   `reversible_overlap`) without severing fast-moving continuation tracks,
   alongside heuristic multi-scan morphological precursor scoring (\autoref{fig:lineage}).
3. **Causal streaming and GP targeting.** Following `Bluesky-Adaptive`
   [@Maffettone2023; @Noack2021; @Kalinin2021], `CausalStreamingTracker`
   maintains bounded sliding-window state (`tell`) and fits trend-detrended
   heteroscedastic `Matérn-3/2` GPs in time or electrochemical $(Q, V)$ state
   space (`ask`) under diffractometer and dose safety limits (`bliss_dfxm_zoom.v1`).

![Four-dimensional split/merge lineage, morphological precursor decomposition, and per-grain strain trajectories.\label{fig:lineage}](docs/figures/fracture_lineage_and_precursors.png)

# Validation and Scope

Quantitative validation leads with controlled 4D synthetic benchmarks
(`scripts/benchmark_suite.py`; \autoref{fig:benchmarks}), where true
trajectories, `1 -> 2` splits, `2 -> 1` merges, and beam dropouts are known
exactly: both 3D `WassersteinBuresCost` and opt-in `facebook/dinov2-small`
descriptors eliminate crossing ID switches between tilted streaks (`+45°` vs.
`-45°`) with identical scalar eigenvalues, and Jaqaman gap closing (`--max-gap 2`)
recovers 100% of staggered dropouts.

Real-detector verification uses three cropped ESRF-ID03 `pco_nf` sCMOS scans
(`data/sample_operando/`, Proposal `MA-650628`, `CC-BY-4.0`). On these frames,
read-noise-guarded thresholding reduces empty-slice false foreground from 46.4%
(raw Otsu) to 0.0%, classical tracking (`--cost-beta 0`) matches DINOv2 links
identically at $5.6\times$ lower CPU runtime, and slab FFT registration
compensates collective rocking-layer shifts. Because three uncalibrated scans
cannot validate multi-horizon GP forecasting or empirical fracture-precursor
classification ROC, `BraggTrack` ships an $O(1)$-memory archive evaluator
(`braggtrack-eval-archive`) and a zero-hint blind human ground-truth annotator
for full multi-cycle *operando* campaigns.

![Synthetic 4D benchmark suite comparing Geometry-Only, Mock-Hash, and `facebook/dinov2-small` across crossing tilted reflections, flux/mosaic jitter, and beam-dropout recovery.\label{fig:benchmarks}](docs/figures/benchmark_comparison.png)

# Acknowledgements

This work was supported by The Faraday Institution Emerging Leader Fellowship
(`FIELF001`) and the BASE Laboratory at the University of Greenwich. We
acknowledge the European Synchrotron Radiation Facility (ESRF) for provision of
beamtime at beamline ID03 under Proposal `MA-650628`, and thank Jaimini Jessica
Mistry, Yaozhu Li, Louis Lesage, Steffen Staeck, Robert Bird, Dominic
Spencer-Jolly, Can Yildirim, and Carsten Detlefs for experimental collaboration
on the underlying ID03 measurements.

# References

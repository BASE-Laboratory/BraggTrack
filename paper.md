---
title: 'BraggTrack: Semantic 4D Kinematics and Fracture Tracking for Operando Diffraction'
tags:
  - Python
  - X-ray diffraction
  - operando crystallography
  - 3D segmentation
  - object tracking
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

BraggTrack is a Python library for automated detection, segmentation, and
tracking of Bragg diffraction spots across sequential operando X-ray
diffraction (XRD) volumes. Given a time series of three-dimensional
reciprocal-space intensity cubes acquired during *in situ* or *operando*
experiments, BraggTrack segments individual Bragg reflections, extracts
geometric and semantic features, and links observations across scans into
persistent tracks with lifecycle annotations (birth, continuation,
termination). The library combines classical image-processing techniques —
Laplacian-of-Gaussian (LoG) enhancement, h-maxima seed detection, and seeded
watershed segmentation — with modern self-supervised vision-transformer
embeddings that provide shape-aware identity descriptors for robust
cross-frame association. BraggTrack is designed for synchrotron beamline data
in NeXus/HDF5 format and ships with bundled sample data from the ESRF ID03
beamline, a command-line interface for batch processing, and an interactive
Jupyter notebook that runs on Google Colab without local installation.

# Statement of Need

Operando diffraction experiments at modern synchrotron sources generate
thousands of three-dimensional reciprocal-space volumes at sub-second
cadence, each containing tens to hundreds of Bragg reflections whose
positions, intensities, and shapes evolve as the sample undergoes
electrochemical cycling, mechanical loading, or thermal treatment
[@Simons2015; @Hayashi2019]. Extracting quantitative kinematics — tracking
which reflection belongs to which crystallographic grain across time, and
detecting when grains nucleate, fragment, or disappear — is a prerequisite
for understanding microstructural evolution during battery operation
[@Finegan2019], additive manufacturing [@Leung2018], or fatigue testing.

Existing workflows rely on manual inspection, commercial peak-fitting
packages that treat each frame independently, or bespoke scripts written per
experiment [@Schmidt2014; @Poulsen2004]. These approaches do not scale to
the data rates of fourth-generation synchrotron sources, lack temporal
coherence, and are difficult to reproduce. Three-dimensional X-ray
diffraction (3DXRD) grain-tracking tools such as those built on
ImageD11 [@Wright2023] and GrainSpotter [@Schmidt2014] focus on indexing
lattice orientations rather than tracking segmented intensity blobs through
time. There is no open-source, pip-installable library that performs both
segmentation and multi-frame tracking of Bragg spots with an extensible cost
function.

BraggTrack fills this gap. It provides a reproducible pipeline from raw HDF5
volumes to labelled tracks, with a modular architecture that separates
segmentation, feature extraction, cost-function design, and assignment into
independently testable layers. Researchers can swap in different segmentation
backends, plug in real or mock vision-transformer encoders, and tune the
geometry-versus-semantics tradeoff without modifying core tracking logic.

# Implementation

## Segmentation

BraggTrack's classical segmentation pipeline operates on each 3D intensity
volume independently:

1. **Foreground thresholding.** An Otsu threshold [@Otsu1979] computed on
   the raw intensity histogram defines the foreground mask. For multi-frame
   sequences, per-frame thresholds are smoothed with a rolling median to
   suppress sensitivity to transient beam drops or detector artefacts.

2. **LoG enhancement.** The volume is convolved with a Gaussian kernel and
   the negative discrete Laplacian is taken, producing a response map in
   which Bragg peaks appear as local maxima [@Lindeberg1998].

3. **Seed detection.** H-maxima filtering [@Soille2003] on the LoG response
   selects seed points that exceed both a fraction of a robust peak
   reference (99.99th percentile, not the absolute maximum, to avoid
   instability from single bright voxels) and a configurable response
   percentile within the foreground. Non-maximum suppression enforces a
   minimum separation between seeds.

4. **Seeded watershed.** Seeds are grown into labelled regions via the
   watershed transform [@Vincent1991] over the inverted LoG response,
   restricted to the intensity-domain foreground mask. Post-processing
   removes small components, fills holes, and relabels sequentially.

This separation of intensity-domain thresholding from response-domain seed
detection avoids a common pitfall in which an Otsu threshold calibrated for
raw intensities is applied to the LoG response, whose dynamic range and
distribution are fundamentally different.

## Feature extraction and semantic descriptors

For each segmented instance, BraggTrack computes an intensity-weighted
centroid (in the $\mu$, $\chi$, $d$ reciprocal-space convention), a bounding
box, voxel count, integrated intensity, covariance tensor, and principal-axis
eigenvalues. These geometric features feed a `PositionShapeCost` function
that combines squared centroid distance with squared eigenvalue distance and
supports per-axis gating.

Optionally, each instance is cropped and masked, and three orthogonal
maximum-intensity projections (MIPs) are computed. These 2D views are
encoded by a frozen DINOv2 vision transformer [@Oquab2024] into a 384-d unit
embedding vector that captures shape and texture information invariant to
minor intensity fluctuations. A `GeometrySemanticCost` function composes the
geometric and semantic terms as $\alpha \cdot C_\text{geo}(i,j) + \beta
\cdot (1 - \cos(\mathbf{f}_i, \mathbf{f}_j))$, allowing researchers to
ablate the contribution of each modality.

A deterministic mock encoder (hash-based) is provided for environments
without GPU access, enabling full pipeline execution in continuous
integration and on Google Colab.

## Tracking and lifecycle

Cross-frame association is performed by the Hungarian algorithm
[@Kuhn1955] on the cost matrix between consecutive scans.
Unmatched observations are marked as births (new reflections entering the
diffraction condition) or terminations (reflections leaving), while matched
pairs are continuations. The result is a directed acyclic graph (DAG)
implemented as a NetworkX [@Hagberg2008] `DiGraph`, where each node carries
a `TrackEvent` annotation and each edge represents a temporal link. Metrics
including fragmentation ratio, ID-switch rate, and full-length track count
are computed from the graph.

## Reproducibility and deployment

BraggTrack is packaged as a standard Python project (`pip install
braggtrack`) with optional dependency groups for PyTorch-based encoders and
notebook execution. Six CLI entry points expose every pipeline stage. A
GitHub Actions CI pipeline tests across Python 3.10–3.12, enforces Ruff
linting and formatting, and executes the demo notebook end-to-end. Bundled
ESRF-ID03 sample data (three 100×250×250 uint16 volumes) allows immediate
experimentation without external data downloads.

# Acknowledgements

The authors acknowledge the European Synchrotron Radiation Facility (ESRF)
for provision of beamtime at beamline ID03. This work was supported by the
BASE Laboratory at the University of Greenwich.

# References

# BraggTrack v0.4.0 Benchmarks, ESRF-ID03 Case Study & Autonomous DFXM Replay

This document reports reproducible multi-trial quantitative benchmarks for **BraggTrack v0.4.0**, comparing **Geometry-Only**, **Deterministic Mock-Hash** (`mock`), and **Pretrained Vision Transformer** (`torch` with ungated `facebook/dinov2-small`) tracking, gap-closing, and 4D chemomechanical fracture precursor scoring, followed by an end-to-end case study on bundled operando dark-field X-ray microscopy (DFXM) data from **ESRF ID03** and a closed-loop **Autonomous "Scout-and-Zoom" Causal Replay Benchmark**.

All synthetic benchmarks can be reproduced via:

```bash
python scripts/benchmark_suite.py \
    --include-torch \
    --model facebook/dinov2-small \
    --trials 5 \
    --bootstrap-samples 500 \
    --output-json artifacts/benchmarks/benchmark_results.json \
    --output-figure docs/figures/benchmark_comparison.png
```

![BraggTrack 5-scenario benchmark comparison across Geometry-Only, Mock-Hash, and Pretrained DINOv2](figures/benchmark_comparison.png)

---

## 1. Benchmark Methodology

To isolate when pretrained multi-view morphological features genuinely outperform classical 3D centroid + covariance tracking—and to verify that gains come from learned visual structure rather than arbitrary feature dimensionality—`scripts/benchmark_suite.py` evaluates three backends under identical segmentation and Hungarian/Jaqaman solver configurations:

1. **`geo` (Geometry-Only, $\alpha=1.0, \beta=0.0$):** Uses calibrated 3D reciprocal-space centroid displacement and log-eigenvalue shape distance (`Eigen-Shape`), with zero semantic weight.
2. **`mock` (Deterministic Hash Baseline, $\alpha=1.0, \beta=0.6$):** Appends `1152`-d `SHA-256` multi-view MIP vectors (`MockDinoEncoder`). Useful solely for deterministic offline CI plumbing; slight pixel changes completely decorrelate consecutive scan hashes.
3. **`torch` (Pretrained `facebook/dinov2-small`, $\alpha=1.0, \beta=0.6$):** Extracts `1152`-d L2-normalized multi-view orthogonal MIP embeddings (`3` projections $\times$ `384`-d ViT-S/14 features) and `16x16` spatial patch tokens for sub-spot precursor dispersion.

### Scenarios Evaluated (`N = 5` trials, `500` bootstrap resamples for 95% CIs)

| Scenario | Physical Beamline Regime | Primary Failure Mode Tested |
| :--- | :--- | :--- |
| **`crossing_morphology`** | Two sub-grains on an antipodal circle ($R = 4.2\text{ px}$) rotating $90.25^\circ$ between scans with orthogonal $+45^\circ$ vs $-45^\circ$ streak tilt (`3D` eigenvalues identical by symmetry) | Equidistant / near-degenerate 3D geometry causing catastrophic Hungarian ID swaps without 2D MIP orientation semantics |
| **`intensity_jitter`** | $\pm 45\%$ incident synchrotron beam-flux spikes + Poisson/Gaussian detector shot noise across 6 scans | Photometric invariance of L2-normalized ViT MIP embeddings vs pixel-hash fragility |
| **`mosaic_broadening`** | Progressive rocking-curve ($\mu$) mosaic broadening ($1.0\times \to 1.9\times$) under electrochemical cycling | Smooth morphological continuity ($\cos > 0.98$) as Bragg reflections broaden |
| **`beam_dropout_no_gap` vs `beam_dropout_gap2`** | Staggered 1-scan rocking-window dropouts across 4 grains over 6 scans (`max_gap=0` vs `max_gap=2`) | Track fragmentation and cross-grain jump hijacking vs Jaqaman diffusive gap-closing recovery |
| **`chemomechanical_fracture`** | 8-scan 4D grain evolution with persistent pre-split broadening + sub-peak bimodality (`scans 1-3`), `1 -> 2` split (`scan 3 -> 4`), `2 -> 1` merge (`scan 5 -> 6`), and isotropic control | Exact split/merge lineage recall and pre-fracture precursor score contrast against stable control grains |

---

## 2. Quantitative Multi-Trial Results (`facebook/dinov2-small`)

### 2.1 Tracking Identity, Gap-Closing & Embedding Stability

| Scenario | Backend | Mean ID-Switch Rate (95% CI) | Fragmentation Ratio (95% CI) | Gap Closures | Cosine Stability $\cos(\mathbf{z}_t, \mathbf{z}_{t+1})$ | Elapsed (`5` trials, CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`crossing_morphology`** | `geo` | `0.2500` (`[0.2500, 0.2500]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | — | `0.51 s` |
| **`crossing_morphology`** | `mock` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | `0.0160` | `0.50 s` |
| **`crossing_morphology`** | **`torch` (`dinov2-small`)** | **`0.0000` (`[0.0000, 0.0000]`)** | **`0.0000` (`[0.0000, 0.0000]`)** | `0.0` | **`0.9035`** | `43.80 s` |
| **`intensity_jitter`** | `geo` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | — | `0.04 s` |
| **`intensity_jitter`** | `mock` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | `0.0089` | `0.05 s` |
| **`intensity_jitter`** | **`torch` (`dinov2-small`)** | **`0.0000` (`[0.0000, 0.0000]`)** | **`0.0000` (`[0.0000, 0.0000]`)** | `0.0` | **`0.9908`** | `10.66 s` |
| **`mosaic_broadening`** | `geo` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | — | `0.05 s` |
| **`mosaic_broadening`** | `mock` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | `0.0` | `-0.0031` | `0.05 s` |
| **`mosaic_broadening`** | **`torch` (`dinov2-small`)** | **`0.0000` (`[0.0000, 0.0000]`)** | **`0.0000` (`[0.0000, 0.0000]`)** | `0.0` | **`0.9899`** | `10.85 s` |
| **`beam_dropout_no_gap` (`max_gap=0`)** | `geo` / `mock` / `torch` | `0.0833` (`[0.0833, 0.0833]`) | `0.5000` (`[0.5000, 0.5000]`) | `0.0` | `— / 0.0048 / 0.9879` | `0.07–17.83 s` |
| **`beam_dropout_gap2` (`max_gap=2`)** | `geo` | `0.0000` (`[0.0000, 0.0000]`) | `0.0000` (`[0.0000, 0.0000]`) | **`2.0`** | — | `0.07 s` |
| **`beam_dropout_gap2` (`max_gap=2`)** | `mock` | `0.0833` (`[0.0833, 0.0833]`) | `0.5000` (`[0.5000, 0.5000]`) | `0.0` | `0.0048` | `0.08 s` |
| **`beam_dropout_gap2` (`max_gap=2`)** | **`torch` (`dinov2-small`)** | **`0.0000` (`[0.0000, 0.0000]`)** | **`0.0000` (`[0.0000, 0.0000]`)** | **`2.0`** | **`0.9879`** | `17.88 s` |

### 2.2 4D Chemomechanical Fracture & Precursor Benchmark

| Scenario | Backend | `1 -> 2` Split Recall | `2 -> 1` Merge Recall | Top-1 Precursor Precision | Precursor Contrast $\Delta S$ vs Control (95% CI) | Elapsed (`5` trials, CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`chemomechanical_fracture`** | `geo` | `1.00` | `1.00` | `1.00` | `0.6449` (`[0.6448, 0.6450]`) | `0.13 s` |
| **`chemomechanical_fracture`** | `mock` | `1.00` | `1.00` | `1.00` | `0.5539` (`[0.5538, 0.5540]`) | `0.29 s` |
| **`chemomechanical_fracture`** | **`torch` (`dinov2-small`)** | **`1.00`** | **`1.00`** | **`1.00`** | **`0.5584` (`[0.5582, 0.5585]`)** | `74.82 s` |

![4D chemomechanical fracture lineage DAG, persistent precursor contrast, and MIP evolution](figures/fracture_lineage_and_precursors.png)

### Key Scientific Takeaways

1. **Orientation-Sensitive Disambiguation (`crossing_morphology`):** When two grains have identical 3D eigenvalue magnitudes ($\lambda_1, \lambda_2, \lambda_3$) and rotate slightly past $90^\circ$ on a shared Debye-Scherrer / rocking locus, **Geometry-Only** suffers a deterministic `25.0%` ID-switch rate because 3D scalar eigenvalues discard 2D detector streak orientation (`+45 deg` vs `-45 deg`). **`facebook/dinov2-small`** preserves high multi-view morphological similarity (`0.9035`) along true trajectories and eliminates ID switches (`0.0%`).
2. **Real Pretrained Embeddings vs Mock Hash (`beam_dropout_gap2` & `intensity_jitter`):** While `MockDinoEncoder` produces pseudo-random zero-mean cosine similarities (`-0.003` to `0.016`) whenever shot noise perturbs a single voxel—causing semantic gap-closing (`min_cosine_similarity=0.35`) to reject valid re-links (`0.0` gap closures, `50%` fragmentation)—pretrained **`facebook/dinov2-small`** maintains `0.988–0.991` cosine stability across `45%` flux spikes, progressive mosaic broadening, and multi-scan dropouts, recovering `100%` of dropped tracks (`0.0%` fragmentation, `2.0` gap closures).
3. **Early Precursor Contrast (`chemomechanical_fracture`):** Across `1 -> 2` grain splitting (`scan 3 -> 4`) and `2 -> 1` sub-grain coalescence (`scan 5 -> 6`), BraggTrack's multi-component precursor score flags the fracturing parent grain `3` scans before topological separation (`1.00` top-1 precision, `+0.56` to `+0.64` score contrast above isotropic control grains).

---

## 3. ESRF-ID03 Operando DFXM Case Study (`data/sample_operando/`)

We executed the complete **BraggTrack v0.4.0** batched pipeline with real **`facebook/dinov2-small`** embeddings, multi-scan gap closing (`--max-gap 1`), interactive HTML reporting (`--report`), `ImageD11` `.flt` export, and NeXus `NXprocess` HDF5 archival on the bundled 3-scan **ESRF-ID03** near-field PCO dataset (`3` scans $\times$ `100 x 250 x 250` `uint16` cubes, ~`20 MB` compressed HDF5):

```bash
braggtrack-run data/sample_operando \
    --outdir artifacts/esrf_id03_dinov2 \
    --backend torch \
    --model facebook/dinov2-small \
    --max-gap 1 \
    --report \
    --export-nexus artifacts/esrf_id03_dinov2/esrf_id03.nxs \
    --export-imaged11 artifacts/esrf_id03_dinov2/esrf_id03.flt
```

### 3.1 CPU Stage-Timing Profile (`v0.3.0` Sequential vs. `v0.4.0` Batched)

| Pipeline Stage | `v0.3.0` (CPU) | **`v0.4.0` Batched (CPU)** | Share of Run | Primary Workload |
| :--- | :---: | :---: | :---: | :--- |
| **`validate`** | `0.043 s` | **`0.044 s`** | `0.1%` | NeXus `/entry_0000/ESRF-ID03/pco_nf/data` discovery & SHA-256 verification |
| **`segment`** | `13.48 s` | **`12.23 s`** | `25.3%` | Chunked lazy Otsu (`LazyVolume`) + 3D LoG + watershed + Poisson $\sigma$ (`[18, 17, 15]` spots) |
| **`embed`** | `18.35 s` | **`15.70 s` (`-14.4%`)** | `32.4%` | Lazy ROI cropping + batched `embed_batch` (`facebook/dinov2-small`, `50` spots $\times$ `1152`-d) |
| **`track`** | `0.024 s` | **`0.029 s`** | `0.06%` | Hungarian matching (`assignment_margin`) + Jaqaman gap closing + `lineage_strain` & rates |
| **`events`** | `20.23 s` | **`17.80 s` (`-12.0%`)** | `36.8%` | `1 -> 2` / `2 -> 1` lineage + batched `extract_patch_features_batch` & $O(1)$-RAM `roi_cache` |
| **`report` + exports** | `3.04 s` | **`2.57 s`** | `5.3%` | Self-contained `report.html`, publication PNGs, `esrf_id03.flt`, and `esrf_id03.nxs` |
| **Total** | `55.17 s` | **`48.44 s` (`-12.2%`)** | **`100.0%`** | Full 3-scan operando DFXM workflow on CPU (no GPU required) |

### 3.2 Detected Lineage Events & Persistent Precursor Candidates

- **Split Candidates (`events.csv`):**
   - **Event 1 (`scan 0 -> 1`):** Parent **Track `6`** (`s0_i5`) $\to$ Daughters **Track `32`** (`s1_i11`) & **Track `34`** (`s1_i9`), intensity partition ratio `1.434`, mean daughter separation `18.64 px`, **DINOv2 multi-view cosine similarity `0.7561`**.
   - **Event 2 (`scan 1 -> 2`):** Parent **Track `21`** (`s1_i1`) $\to$ Daughters **Track `36`** (`s2_i1`) & **Track `44`** (`s2_i13`), intensity partition ratio `1.258`, mean daughter separation `20.11 px`, **DINOv2 multi-view cosine similarity `0.6837`**.
- **Top Persistent Precursor Tracks (`precursors.csv`, `min_persistence = 2` scans):**
   1. **Track `14`** (`rank 1`): Active across all `3` scans (`onset_scan=0`, `peak_scan=1`), `peak_score = 0.8789`, `mean_active_score = 0.7775`, `max_anisotropy = 5.45`, `max_spread_growth = +106%`, `confidence = 0.8555`.
   2. **Track `34`** (`rank 2`, split daughter of Track `6`): Active across `scans 1-2`, `peak_score = 0.9023`, extreme streak elongation (`max_anisotropy = 26.78`), `max_spread_growth = +129%`, `confidence = 0.8119`.
   3. **Track `27`** (`rank 3`): Active across `scans 1-2`, `peak_score = 0.6710`, `max_anisotropy = 16.00`, `confidence = 0.6534`.
   4. **Track `29`** (`rank 4`): Active across `scans 1-2`, `peak_score = 0.6668`, `max_anisotropy = 6.67`, `confidence = 0.6490`.

---

## 4. Autonomous "Scout-and-Zoom" DFXM Causal Replay Benchmark (`v0.4.0`)

To validate online closed-loop target selection before synchrotron beamtime, `evaluate_causal_replay()` (`braggtrack.autonomous.replay` / `braggtrack-watch --mode replay`) streams a 6-scan operando charging sequence (`3.95 V -> 4.27 V`, `3 min/scan` cadence, mimicking Ni-rich `H2 -> H3` $c$-axis contraction) strictly scan-by-scan through `CausalStreamingTracker` using real **`facebook/dinov2-small`** (`backend="torch"`).

The scene contains three competing reflections under a strict `zoom_budget = 1` acquisition budget:
- **Target Grain (Track `3`):** Develops persistent bimodal mosaicity across `scans 1–3` while drifting rapidly along $d$, then fractures (`1 -> 2` split) between `scan 3` (`4.21 V`) and `scan 4` (`4.24 V`).
- **Bright Distractor:** Higher integrated intensity than the target grain, remains isotropic and never fractures.
- **High-Strain Distractor:** Undergoes larger rigid-body lattice shift $|\varepsilon|$ without internal mosaicity broadening or fracture.

| Autonomous Policy (`zoom_budget = 1`) | Zooms Fired | Pre-Split Capture Recall | Pre-Split Precision | Lead Scans Before Split | Electrochemical Lead ($\Delta E$) | Velocity-Extrapolated Setpoint Hit Rate | Mean Causal Step Latency (`dinov2-small`, CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`braggtrack_autonomous`** | `1` | **`1.00`** | **`1.00`** | **`2.0` scans** | **`80.0 mV`** (`4.16 V` vs `4.24 V`) | **`1.00`** | **`1.39 s / scan`** (`max 1.71 s`) |
| `brightest_intensity` | `1` | `0.00` | `0.00` | `0.0` | — | `1.00` | — |
| `strain_only` | `1` | `0.00` | `0.00` | `0.0` | — | `1.00` | — |

At `1.39 s / scan` on CPU (`< 1%` of a typical `180 s` wide-field scouting scan), `CausalStreamingTracker` leaves ample margin for diffractometer slewing and objective-lens insertion while predicting velocity-compensated `bliss_dfxm_zoom.v1` motor setpoints before grain microcracking occurs.


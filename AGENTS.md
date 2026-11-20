# Agent & Contributor Guidelines (`BraggTrack`)

Read [ECOSYSTEM.md](https://github.com/BASE-Laboratory/basecoat/blob/main/ECOSYSTEM.md) before adding a module, dependency, file reader, or dataset constant. The master copy of `ECOSYSTEM.md` lives in `BASE-Laboratory/basecoat`.

## Ecosystem Layer

- **Repository:** `BASE-Laboratory/BraggTrack`
- **Licence:** `BSD-3-Clause`
- **Layer:** **Layer 2 — Perception & image tools**
- **Allowed ecosystem dependencies:** `basecoat` (Layer 1) **only**.
- **Banned imports (enforced by `[tool.basecoat.layers]`):** `multimodal_benchmark`, `xct_research`, `openimpala`, `sms`, `openlsrx`, `x_fuse`, `porousdiff`, `porousdiff_workflows`, `zarr`, `numcodecs`.

## Ownership Boundaries

- **Owns:** 3D Bragg spot segmentation (`braggtrack/segmentation/`), multi-scan trajectory & split/merge lineage tracking (`braggtrack/tracking/`), DINOv2/v3 multi-view semantic descriptors (`braggtrack/semantic/`), causal GP DFXM target selection & BLISS triggers (`braggtrack/autonomous/`), and `ImageD11` `.flt` / NeXus `NXprocess` export (`braggtrack/io/export.py`).
- **Delegates to `basecoat` (`v0.2`):** Calibrated `Volume` / `VolumeSet` / `AxisSpec` (`deg` and `angstrom` reciprocal-space axes via `AxisCalibration.to_basecoat_axes`), OME-Zarr v0.4 volume + mode-downsampled Bragg spot label export (`export_ome_zarr_volume`), and region-lazy `.ome.zarr` / `.tif` / `.npy` detector cube access (`open_lazy_volume`, `load_primary_volume`). Facility-specific NeXus `@signal` / `darfix` / BLISS SWMR HDF5 discovery stays in `braggtrack/io/nexus.py` until a second consumer requires it in `basecoat`.

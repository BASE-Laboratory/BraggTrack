#!/usr/bin/env python3
"""Build a standalone, zero-hint HTML Blind Annotator for independent human tracking ground truth.

Generates a single self-contained HTML application embedding per-scan detector MIPs,
per-rocking-band MIP slabs, zoomed spot crops, and randomized neutral spot codes
(``S0_01..``, ``S1_01..``) with **zero tracker predictions** so an independent annotator
(PI or student) can label ``1-to-1``, ``split``, ``merge``, ``disappeared``, and ``uncertain``
links and export ``ground_truth_links.json``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from braggtrack.io import discover_operando_scans, extract_axis_calibration
from braggtrack.io.discovery import ScanFile
from braggtrack.io.lazy import detector_foreground_threshold_chunked, open_lazy_volume
from braggtrack.segmentation import (
    extract_instance_table,
    relabel_sequential,
    remove_small_objects,
    segment_classical,
)
from braggtrack.tracking.drift import is_single_slice_speck


def _encode_gray_png_b64(img_u8: np.ndarray) -> str:
    """Encode a 2D uint8 grayscale array into a base64 PNG data URI using stdlib zlib."""
    arr = np.ascontiguousarray(img_u8, dtype=np.uint8)
    h, w = arr.shape
    raw_rows = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def _chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    idat = zlib.compress(raw_rows, level=6)
    png = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _normalize_sqrt_u8(arr2d: np.ndarray, pedestal: float, max_dim: int | None = None) -> np.ndarray:
    a = np.asarray(arr2d, dtype=np.float32)
    if max_dim is not None and max_dim >= 32:
        h, w = a.shape
        step = max(1, int(math.ceil(max(h, w) / float(max_dim))))
        if step > 1:
            pad_h = (-h) % step
            pad_w = (-w) % step
            if pad_h or pad_w:
                a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="edge")
            a = a.reshape(a.shape[0] // step, step, a.shape[1] // step, step).max(axis=(1, 3))
    net = np.maximum(0.0, a - float(pedestal))
    sq = np.sqrt(net)
    hi = float(np.percentile(sq, 99.85)) if sq.size > 0 else 1.0
    if hi <= 1e-6:
        hi = max(float(sq.max()), 1.0)
    scaled = np.clip(sq / hi, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def _group_rocking_bands(all_spots: list[list[dict[str, Any]]], gap_z: float = 3.5) -> list[dict[str, Any]]:
    z_vals: list[float] = []
    mu_vals: list[float] = []
    for sp_list in all_spots:
        for s in sp_list:
            z_c = 0.5 * (float(s["bbox_min_z"]) + float(s["bbox_max_z"]))
            z_vals.append(z_c)
            mu_vals.append(float(s["centroid_mu"]))
    if not z_vals:
        return [{"band_id": "all", "label": "All rocking slices", "z_min": 0, "z_max": 999999}]

    order = sorted(range(len(z_vals)), key=lambda i: z_vals[i])
    bands: list[list[int]] = [[order[0]]]
    for idx in order[1:]:
        if z_vals[idx] - z_vals[bands[-1][-1]] <= gap_z:
            bands[-1].append(idx)
        else:
            bands.append([idx])

    out: list[dict[str, Any]] = []
    for b_idx, members in enumerate(bands):
        zs = [z_vals[i] for i in members]
        mus = [mu_vals[i] for i in members]
        z_lo = max(0, int(math.floor(min(zs))) - 2)
        z_hi = int(math.ceil(max(zs))) + 3
        mu_lo, mu_hi = min(mus), max(mus)
        out.append(
            {
                "band_id": f"band_{b_idx}",
                "label": f"Layer {b_idx + 1}: mu [{mu_lo:.1f}, {mu_hi:.1f}] (z={z_lo}..{z_hi}, {len(members)} obs)",
                "z_min": z_lo,
                "z_max": z_hi,
                "mu_min": round(mu_lo, 2),
                "mu_max": round(mu_hi, 2),
            }
        )
    return out


def build_blind_annotator_payload(
    root: Path,
    *,
    start_scan: int = 0,
    num_scans: int = 3,
    bg_sigma: float = 5.0,
    smooth_window: int = 5,
    seed_peak_fraction: float = 0.08,
    seed_response_percentile: float = 95.0,
    min_spot_voxels: int = 6,
    anisotropic: bool = False,
    seed: int = 42,
    max_preview_dim: int | None = None,
) -> dict[str, Any]:
    """Segment a scan window and build a tracker-free blind annotation payload."""
    all_scans = discover_operando_scans(root)
    if not all_scans:
        flat_h5 = sorted({*Path(root).glob("*.h5"), *Path(root).glob("*.nxs"), *Path(root).glob("*.hdf5")})
        all_scans = [ScanFile(scan_name=p.stem, path=p) for p in flat_h5]
    selected = all_scans[start_scan : start_scan + num_scans]
    if len(selected) < 2:
        raise ValueError(f"Need at least 2 scans for blind annotation; found {len(selected)} in {root}")

    rng = random.Random(seed)
    volumes: list[np.ndarray] = []
    pedestals: list[float] = []
    raw_spot_tables: list[list[dict[str, Any]]] = []
    scan_names: list[str] = []
    raw_thresholds: list[float] = []
    cal = extract_axis_calibration(selected[0].path)

    for sc in selected:
        with open_lazy_volume(sc.path) as lv:
            raw_thr = float(detector_foreground_threshold_chunked(lv, bg_sigma=bg_sigma))
            vol = np.asarray(lv._dataset[()], dtype=np.float32)
        raw_thresholds.append(raw_thr)
        win_lo = max(0, len(raw_thresholds) - max(1, int(smooth_window)))
        thr = float(np.median(raw_thresholds[win_lo:]))
        ped = float(np.median(vol[::2, ::4, ::4]))
        seg = segment_classical(
            vol,
            threshold=thr,
            seed_peak_fraction=seed_peak_fraction,
            seed_response_percentile=seed_response_percentile,
            anisotropic_auto=anisotropic,
        )
        labels = relabel_sequential(remove_small_objects(seg.labeled_volume, min_size=min_spot_voxels))
        spots = extract_instance_table(labels, vol, calibration=cal)
        volumes.append(vol)
        pedestals.append(ped)
        raw_spot_tables.append(spots)
        scan_names.append(sc.scan_name)

    bands = _group_rocking_bands(raw_spot_tables)
    nz, ny, nx_det = volumes[0].shape

    scans_payload: list[dict[str, Any]] = []
    for local_s, (vol, ped, spots, s_name) in enumerate(
        zip(volumes, pedestals, raw_spot_tables, scan_names, strict=True)
    ):
        global_s = start_scan + local_s
        full_mip_b64 = _encode_gray_png_b64(_normalize_sqrt_u8(vol.max(axis=0), ped, max_dim=max_preview_dim))
        band_mips: dict[str, str] = {"all": full_mip_b64}
        for b in bands:
            zlo = max(0, min(nz - 1, int(b["z_min"])))
            zhi = max(zlo + 1, min(nz, int(b["z_max"])))
            band_mips[b["band_id"]] = _encode_gray_png_b64(
                _normalize_sqrt_u8(vol[zlo:zhi].max(axis=0), ped, max_dim=max_preview_dim)
            )

        perm = list(range(len(spots)))
        rng.shuffle(perm)
        code_for_idx = {sp_idx: f"S{global_s}_{rank + 1:02d}" for rank, sp_idx in enumerate(perm)}

        spot_records: list[dict[str, Any]] = []
        for sp_idx, row in enumerate(spots):
            mu_a = float(row["centroid_mu"])
            chi_a = float(row["centroid_chi"])
            d_a = float(row["centroid_d"])
            step_mu = max(abs(float(row.get("step_mu", 1.0))), 1e-12)
            step_chi = max(abs(float(row.get("step_chi", 1.0))), 1e-12)
            step_d = max(abs(float(row.get("step_d", 1.0))), 1e-12)

            y_px = 0.5 * (float(row["bbox_min_y"]) + float(row["bbox_max_y"]) - 1.0)
            x_px = 0.5 * (float(row["bbox_min_x"]) + float(row["bbox_max_x"]) - 1.0)
            z_peak = max(0, min(nz - 1, int(round(0.5 * (float(row["bbox_min_z"]) + float(row["bbox_max_z"]) - 1.0)))))

            n_near = 0
            for sp_b, other in enumerate(spots):
                if sp_b == sp_idx:
                    continue
                if abs(float(other["centroid_mu"]) - mu_a) / step_mu <= 4.5:
                    dist_det = math.hypot(
                        (float(other["centroid_chi"]) - chi_a) / step_chi,
                        (float(other["centroid_d"]) - d_a) / step_d,
                    )
                    if dist_det <= 55.0:
                        n_near += 1

            half_crop = 16
            yc, xc = int(round(y_px)), int(round(x_px))
            y0, y1 = max(0, yc - half_crop), min(ny, yc + half_crop)
            x0, x1 = max(0, xc - half_crop), min(nx_det, xc + half_crop)
            z0 = max(0, int(row["bbox_min_z"]))
            z1 = min(nz, max(z0 + 1, int(row["bbox_max_z"])))
            crop2d = vol[z0:z1, y0:y1, x0:x1].max(axis=0)
            crop_b64 = _encode_gray_png_b64(_normalize_sqrt_u8(crop2d, ped))

            assigned_band = "all"
            for b in bands:
                if int(b["z_min"]) <= z_peak < int(b["z_max"]):
                    assigned_band = b["band_id"]
                    break

            spot_records.append(
                {
                    "spot_idx": sp_idx,
                    "blind_code": code_for_idx[sp_idx],
                    "x_px": round(x_px, 2),
                    "y_px": round(y_px, 2),
                    "z_peak": z_peak,
                    "centroid_mu": round(mu_a, 3),
                    "centroid_chi": round(chi_a, 3),
                    "centroid_d": round(d_a, 3),
                    "step_mu": round(step_mu, 5),
                    "step_chi": round(step_chi, 5),
                    "step_d": round(step_d, 5),
                    "net_intensity": round(float(row.get("net_intensity", row["integrated_intensity"])), 1),
                    "n_voxels": int(row.get("voxel_count", row.get("n_voxels", 0))),
                    "is_single_slice_speck": bool(is_single_slice_speck(row)),
                    "layer_density": "crowded" if n_near >= 2 else "isolated",
                    "band_id": assigned_band,
                    "crop_b64": crop_b64,
                }
            )

        spot_records.sort(key=lambda r: str(r["blind_code"]))
        scans_payload.append(
            {
                "scan_idx": global_s,
                "scan_name": s_name,
                "detector_shape": [nz, ny, nx_det],
                "band_mips": band_mips,
                "spots": spot_records,
            }
        )

    return {
        "dataset_root": str(root),
        "bands": [{"band_id": "all", "label": "All rocking layers (full MIP)"}, *bands],
        "scans": scans_payload,
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>BraggTrack Blind Ground-Truth Annotator</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-[var(--background)] text-[var(--foreground)] antialiased p-4 space-y-4">
  <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg p-4 space-y-3">
    <div class="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 class="text-base font-semibold">BraggTrack Blind Ground-Truth Link Annotator</h1>
        <p class="text-xs text-[var(--muted-foreground)]">
          Zero tracker predictions are loaded. Spot codes are randomized per scan for independent human verification.
        </p>
      </div>
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <label class="flex items-center gap-1.5 text-[var(--muted-foreground)]">
          Annotator ID:
          <input id="annotatorId" type="text" placeholder="e.g. student_initials"
            class="px-2 py-1 rounded bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)] font-mono text-xs" />
        </label>
        <label class="px-2.5 py-1 rounded border border-[var(--border)] cursor-pointer hover:opacity-80">
          Import JSON
          <input id="importInput" type="file" accept=".json" class="hidden" />
        </label>
        <button id="exportBtn" class="px-3 py-1 rounded bg-[var(--primary)] text-[var(--primary-foreground)] font-medium hover:opacity-90">
          Export ground_truth_links.json
        </button>
      </div>
    </div>

    <div class="flex flex-wrap items-center gap-3 pt-2 border-t border-[var(--border)] text-xs">
      <label class="flex items-center gap-1.5">
        <span class="text-[var(--muted-foreground)]">Scan Pair:</span>
        <select id="pairSelect" class="px-2 py-1 rounded bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)]"></select>
      </label>
      <label class="flex items-center gap-1.5">
        <span class="text-[var(--muted-foreground)]">Rocking Layer Filter:</span>
        <select id="bandSelect" class="px-2 py-1 rounded bg-[var(--background)] border border-[var(--border)] text-[var(--foreground)]"></select>
      </label>
      <label class="flex items-center gap-1.5 cursor-pointer">
        <input id="hideSpecks" type="checkbox" class="accent-[var(--primary)]" />
        <span class="text-[var(--muted-foreground)]">Dim 1-slice specks</span>
      </label>
      <button id="blinkBtn" class="px-2.5 py-1 rounded border border-[var(--border)] hover:opacity-80">
        Hold to Blink t / t+1
      </button>
      <span id="progressBadge" class="ml-auto font-mono text-xs text-[var(--muted-foreground)]"></span>
    </div>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
    <div class="lg:col-span-5 bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 space-y-2">
      <div class="flex items-center justify-between text-xs">
        <span id="leftTitle" class="font-semibold">Scan t (Source)</span>
        <span class="text-[var(--muted-foreground)]">Click spot to select source (Shift+Click for merge)</span>
      </div>
      <canvas id="leftCanvas" width="380" height="380" class="w-full rounded border border-[var(--border)] cursor-crosshair"></canvas>
    </div>

    <div class="lg:col-span-5 bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 space-y-2">
      <div class="flex items-center justify-between text-xs">
        <span id="rightTitle" class="font-semibold">Scan t+1 (Target)</span>
        <span class="text-[var(--muted-foreground)]">Click target (Shift+Click for 1-&gt;2+ split)</span>
      </div>
      <canvas id="rightCanvas" width="380" height="380" class="w-full rounded border border-[var(--border)] cursor-crosshair"></canvas>
    </div>

    <div class="lg:col-span-2 bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 space-y-3 text-xs">
      <div class="font-semibold border-b border-[var(--border)] pb-1.5">Selected Inspection</div>
      <div id="sourceCard" class="space-y-1 text-[var(--muted-foreground)]">No source spot selected.</div>
      <div id="targetCard" class="space-y-1 pt-2 border-t border-[var(--border)] text-[var(--muted-foreground)]">No target spot selected.</div>

      <div class="pt-2 border-t border-[var(--border)] space-y-1.5">
        <div class="font-semibold">Record Link</div>
        <button data-ltype="1-to-1" class="linkBtn w-full py-1.5 px-2 rounded bg-[var(--primary)] text-[var(--primary-foreground)] font-medium text-left">
          1. 1-to-1 Continuation
        </button>
        <button data-ltype="split" class="linkBtn w-full py-1 px-2 rounded border border-[var(--border)] text-left hover:opacity-80">
          2. Split (1 -&gt; 2+)
        </button>
        <button data-ltype="merge" class="linkBtn w-full py-1 px-2 rounded border border-[var(--border)] text-left hover:opacity-80">
          3. Merge / Overlap (2+ -&gt; 1)
        </button>
        <button data-ltype="disappeared" class="linkBtn w-full py-1 px-2 rounded border border-[var(--border)] text-left hover:opacity-80">
          4. Disappeared (1 -&gt; 0)
        </button>
        <button data-ltype="uncertain" class="linkBtn w-full py-1 px-2 rounded border border-[var(--border)] text-left hover:opacity-80">
          5. Uncertain (Exclude)
        </button>
      </div>
    </div>
  </div>

  <div class="bg-[var(--card)] border border-[var(--border)] rounded-lg p-3 space-y-2 text-xs">
    <div class="flex items-center justify-between">
      <span class="font-semibold">Recorded Blind Annotations (<span id="linkCount">0</span>)</span>
      <button id="clearPairBtn" class="text-xs text-[var(--muted-foreground)] hover:underline">Clear current scan pair</button>
    </div>
    <div id="linksTableWrap" class="max-h-48 overflow-y-auto font-mono text-xs space-y-1"></div>
  </div>

  <script>
    const PAYLOAD = __PAYLOAD_JSON__;
    let currentPairIdx = 0;
    let currentBandId = "all";
    let blinkActive = false;
    let selectedSources = [];
    let selectedTargets = [];
    let recordedLinks = [];
    const imgCache = {};

    function loadImg(src) {
      if (imgCache[src]) return Promise.resolve(imgCache[src]);
      return new Promise((resolve) => {
        const im = new Image();
        im.onload = () => { imgCache[src] = im; resolve(im); };
        im.src = src;
      });
    }

    function pairScans() {
      return [PAYLOAD.scans[currentPairIdx], PAYLOAD.scans[currentPairIdx + 1]];
    }

    function visibleSpots(scanObj) {
      const dimSpecks = document.getElementById("hideSpecks").checked;
      return scanObj.spots.filter(sp => {
        if (currentBandId !== "all" && sp.band_id !== currentBandId) return false;
        if (dimSpecks && sp.is_single_slice_speck) return false;
        return true;
      });
    }

    function linkedSourceSet(scanFrom, scanTo) {
      const s = new Set();
      recordedLinks.forEach(l => {
        if (l.scan_from === scanFrom && l.scan_to === scanTo) l.source_spots.forEach(sp => s.add(sp));
      });
      return s;
    }

    function linkedTargetSet(scanFrom, scanTo) {
      const s = new Set();
      recordedLinks.forEach(l => {
        if (l.scan_from === scanFrom && l.scan_to === scanTo) l.target_spots.forEach(sp => s.add(sp));
      });
      return s;
    }

    async function renderCanvases() {
      const [sLeft, sRight] = pairScans();
      document.getElementById("leftTitle").textContent =
        blinkActive ? `BLINK: ${sRight.scan_name} (Scan ${sRight.scan_idx})` : `${sLeft.scan_name} (Scan ${sLeft.scan_idx})`;
      document.getElementById("rightTitle").textContent = `${sRight.scan_name} (Scan ${sRight.scan_idx})`;

      const leftMip = (blinkActive ? sRight : sLeft).band_mips[currentBandId] || sLeft.band_mips["all"];
      const rightMip = sRight.band_mips[currentBandId] || sRight.band_mips["all"];
      const [imL, imR] = await Promise.all([loadImg(leftMip), loadImg(rightMip)]);

      const doneSrc = linkedSourceSet(sLeft.scan_idx, sRight.scan_idx);
      const doneTgt = linkedTargetSet(sLeft.scan_idx, sRight.scan_idx);

      drawDetector(document.getElementById("leftCanvas"), imL, sLeft, visibleSpots(sLeft), selectedSources, doneSrc);
      drawDetector(document.getElementById("rightCanvas"), imR, sRight, visibleSpots(sRight), selectedTargets, doneTgt);

      const totalNonSpeck = sLeft.spots.filter(sp => !sp.is_single_slice_speck).length;
      const doneNonSpeck = sLeft.spots.filter(sp => !sp.is_single_slice_speck && doneSrc.has(sp.spot_idx)).length;
      document.getElementById("progressBadge").textContent =
        `Pair progress: ${doneSrc.size}/${sLeft.spots.length} spots (${doneNonSpeck}/${totalNonSpeck} multi-slice)`;
      renderInspectionCards(sLeft, sRight);
      renderLinksList();
    }

    function drawDetector(canvas, img, scanObj, spots, selectedIds, doneSet) {
      const ctx = canvas.getContext("2d");
      const [, ny, nx] = scanObj.detector_shape;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const sx = canvas.width / nx;
      const sy = canvas.height / ny;

      spots.forEach(sp => {
        const cx = (sp.x_px + 0.5) * sx;
        const cy = (sp.y_px + 0.5) * sy;
        const isSel = selectedIds.includes(sp.spot_idx);
        const isDone = doneSet.has(sp.spot_idx);
        ctx.beginPath();
        ctx.arc(cx, cy, isSel ? 9 : 6, 0, Math.PI * 2);
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.strokeStyle = isSel ? "#38bdf8" : (isDone ? "#22c55e" : (sp.layer_density === "crowded" ? "#f59e0b" : "#e2e8f0"));
        ctx.stroke();

        ctx.font = "10px monospace";
        ctx.fillStyle = isSel ? "#38bdf8" : (isDone ? "#4ade80" : "#f8fafc");
        ctx.fillText(sp.blind_code.split("_")[1], cx + 7, cy - 5);
      });
    }

    function renderSpotSummary(title, scanObj, spotIds) {
      if (!spotIds.length) return `<div class="text-[var(--muted-foreground)]">${title}: none</div>`;
      return spotIds.map(id => {
        const sp = scanObj.spots.find(x => x.spot_idx === id);
        if (!sp) return "";
        return `
          <div class="space-y-1">
            <div class="flex items-center gap-2">
              <img src="${sp.crop_b64}" class="w-10 h-10 rounded border border-[var(--border)]" style="image-rendering: pixelated;" />
              <div class="font-mono text-xs text-[var(--foreground)]">
                <div class="font-semibold">${sp.blind_code} (${sp.layer_density})</div>
                <div>mu=${sp.centroid_mu}</div>
                <div>chi=${sp.centroid_chi}, d=${sp.centroid_d}</div>
                <div>net=${sp.net_intensity} ADU (${sp.n_voxels} vox)</div>
              </div>
            </div>
          </div>`;
      }).join("");
    }

    function renderInspectionCards(sLeft, sRight) {
      document.getElementById("sourceCard").innerHTML = renderSpotSummary("Source", sLeft, selectedSources);
      document.getElementById("targetCard").innerHTML = renderSpotSummary("Target", sRight, selectedTargets);
    }

    function pickSpot(canvas, evt, scanObj) {
      const rect = canvas.getBoundingClientRect();
      const [, ny, nx] = scanObj.detector_shape;
      const xPx = ((evt.clientX - rect.left) / rect.width) * nx;
      const yPx = ((evt.clientY - rect.top) / rect.height) * ny;
      let best = null;
      let bestDist = 14.0;
      visibleSpots(scanObj).forEach(sp => {
        const d = Math.hypot(sp.x_px - xPx, sp.y_px - yPx);
        if (d < bestDist) { bestDist = d; best = sp; }
      });
      return best;
    }

    function addAnnotation(ltype) {
      const [sLeft, sRight] = pairScans();
      if (!selectedSources.length) return;
      if ((ltype === "1-to-1" || ltype === "merge") && selectedTargets.length !== 1) return;
      if (ltype === "split" && selectedTargets.length < 2) return;
      const targets = (ltype === "disappeared" || (ltype === "uncertain" && !selectedTargets.length)) ? [] : [...selectedTargets];

      recordedLinks = recordedLinks.filter(l =>
        !(l.scan_from === sLeft.scan_idx && l.scan_to === sRight.scan_idx &&
          l.source_spots.some(s => selectedSources.includes(s)))
      );
      const srcObjs = selectedSources.map(id => sLeft.spots.find(x => x.spot_idx === id)).filter(Boolean);
      const isCrowded = srcObjs.some(x => x.layer_density === "crowded");
      recordedLinks.push({
        scan_from: sLeft.scan_idx,
        scan_to: sRight.scan_idx,
        link_type: ltype,
        source_spots: [...selectedSources],
        target_spots: targets,
        source_codes: srcObjs.map(x => x.blind_code),
        target_codes: targets.map(id => (sRight.spots.find(x => x.spot_idx === id) || {}).blind_code || String(id)),
        layer_density: isCrowded ? "crowded" : "isolated",
      });
      selectedSources = [];
      selectedTargets = [];
      renderCanvases();
    }

    function renderLinksList() {
      document.getElementById("linkCount").textContent = String(recordedLinks.length);
      const wrap = document.getElementById("linksTableWrap");
      if (!recordedLinks.length) {
        wrap.innerHTML = `<div class="text-[var(--muted-foreground)]">No links recorded yet. Click a source spot on Scan t and a target spot on Scan t+1.</div>`;
        return;
      }
      wrap.innerHTML = recordedLinks.slice().reverse().map((l, revIdx) => {
        const idx = recordedLinks.length - 1 - revIdx;
        return `<div class="flex items-center justify-between py-0.5 border-b border-[var(--border)]">
          <span>[s${l.scan_from}-&gt;s${l.scan_to}] <b>${l.link_type}</b>: ${l.source_codes.join("+")} -&gt; ${l.target_codes.length ? l.target_codes.join("+") : "NONE"} (${l.layer_density})</span>
          <button onclick="removeLink(${idx})" class="text-[var(--muted-foreground)] hover:underline">remove</button>
        </div>`;
      }).join("");
    }

    window.removeLink = function(idx) {
      recordedLinks.splice(idx, 1);
      renderCanvases();
    };

    function init() {
      const pairSel = document.getElementById("pairSelect");
      for (let i = 0; i < PAYLOAD.scans.length - 1; i++) {
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = `${PAYLOAD.scans[i].scan_name} -> ${PAYLOAD.scans[i + 1].scan_name}`;
        pairSel.appendChild(opt);
      }
      pairSel.onchange = () => { currentPairIdx = Number(pairSel.value); selectedSources = []; selectedTargets = []; renderCanvases(); };

      const bandSel = document.getElementById("bandSelect");
      PAYLOAD.bands.forEach(b => {
        const opt = document.createElement("option");
        opt.value = b.band_id;
        opt.textContent = b.label;
        bandSel.appendChild(opt);
      });
      bandSel.onchange = () => { currentBandId = bandSel.value; renderCanvases(); };
      document.getElementById("hideSpecks").onchange = () => renderCanvases();

      const blinkBtn = document.getElementById("blinkBtn");
      blinkBtn.onmousedown = () => { blinkActive = true; renderCanvases(); };
      blinkBtn.onmouseup = blinkBtn.onmouseleave = () => { if (blinkActive) { blinkActive = false; renderCanvases(); } };

      document.getElementById("leftCanvas").onclick = (e) => {
        const [sLeft] = pairScans();
        const sp = pickSpot(e.target, e, sLeft);
        if (!sp) return;
        if (e.shiftKey) {
          if (!selectedSources.includes(sp.spot_idx)) selectedSources.push(sp.spot_idx);
        } else {
          selectedSources = [sp.spot_idx];
        }
        renderCanvases();
      };

      document.getElementById("rightCanvas").onclick = (e) => {
        const [, sRight] = pairScans();
        const sp = pickSpot(e.target, e, sRight);
        if (!sp) return;
        if (e.shiftKey) {
          if (!selectedTargets.includes(sp.spot_idx)) selectedTargets.push(sp.spot_idx);
        } else {
          selectedTargets = [sp.spot_idx];
        }
        renderCanvases();
      };

      document.querySelectorAll(".linkBtn").forEach(btn => {
        btn.onclick = () => addAnnotation(btn.dataset.ltype);
      });

      document.getElementById("clearPairBtn").onclick = () => {
        const [sLeft, sRight] = pairScans();
        recordedLinks = recordedLinks.filter(l => !(l.scan_from === sLeft.scan_idx && l.scan_to === sRight.scan_idx));
        renderCanvases();
      };

      document.getElementById("exportBtn").onclick = () => {
        const exportObj = {
          schema_version: "braggtrack_gt_links.v1",
          dataset_root: PAYLOAD.dataset_root,
          annotator_id: document.getElementById("annotatorId").value.trim() || "anonymous_human",
          created_at: new Date().toISOString(),
          scans: PAYLOAD.scans.map(s => ({
            scan_idx: s.scan_idx,
            scan_name: s.scan_name,
            spots: s.spots.map(({crop_b64, ...rest}) => rest),
          })),
          links: recordedLinks,
        };
        const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "ground_truth_links.json";
        a.click();
      };

      document.getElementById("importInput").onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
          const parsed = JSON.parse(reader.result);
          if (Array.isArray(parsed.links)) recordedLinks = parsed.links;
          if (parsed.annotator_id) document.getElementById("annotatorId").value = parsed.annotator_id;
          renderCanvases();
        };
        reader.readAsText(file);
      };

      window.addEventListener("themechange", () => renderCanvases());
      renderCanvases();
    }

    init();
  </script>
</body>
</html>
"""


def render_blind_annotator_html(payload: dict[str, Any]) -> str:
    return _HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build standalone zero-hint HTML Blind Annotator.")
    parser.add_argument("--root", type=Path, required=True, help="Experiment directory containing HDF5 scans")
    parser.add_argument("--start-scan", type=int, default=0, help="0-based starting scan index")
    parser.add_argument("--num-scans", type=int, default=3, help="Number of consecutive scans to include")
    parser.add_argument("--bg-sigma", type=float, default=5.0, help="Detector MAD background floor multiplier")
    parser.add_argument("--min-spot-voxels", type=int, default=6, help="Minimum connected-component voxels")
    parser.add_argument("--anisotropic", action="store_true", help="Enable aspect-aware anisotropic LoG seeding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for blind spot code permutation")
    parser.add_argument(
        "--max-preview-dim",
        type=int,
        default=None,
        help="Optional max pixel dimension for display MIP PNGs (preserves exact detector coordinates)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output self-contained .html path")
    args = parser.parse_args(argv)

    payload = build_blind_annotator_payload(
        args.root,
        start_scan=args.start_scan,
        num_scans=args.num_scans,
        bg_sigma=args.bg_sigma,
        min_spot_voxels=args.min_spot_voxels,
        anisotropic=args.anisotropic,
        seed=args.seed,
        max_preview_dim=args.max_preview_dim,
    )
    html_str = render_blind_annotator_html(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_str, encoding="utf-8")
    print(f"Wrote blind annotator HTML ({len(html_str) // 1024} KB) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute Week 4 multi-view MIP embeddings for segmented spots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from braggtrack.io import (
    MissingH5DependencyError,
    discover_operando_scans,
    load_primary_volume,
    resolve_dataset_root,
)
from braggtrack.semantic import crop_spot_cube, make_multiview_encoder, orthogonal_mips


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "root",
        nargs="?",
        default=None,
        help="Dataset root with scan folders (default: data/sample_operando if present, else .)",
    )
    p.add_argument("--segdir", default="artifacts/week2", help="Segmentation output with features.csv + labels.npz")
    p.add_argument("--outdir", default="artifacts/week4", help="Embedding output root")
    p.add_argument("--margin", type=int, default=2, help="Voxel padding around each spot bbox")
    p.add_argument(
        "--backend",
        choices=("auto", "mock", "torch"),
        default="auto",
        help="Embedding backend (mock needs no PyTorch; torch uses Dinov2-small)",
    )
    p.add_argument("--model", default="facebook/dinov2-small", help="HF model id when backend=torch")
    return p


def _synth_volume_from_file(path: Path, size: int = 24) -> np.ndarray:
    digest = hashlib.sha256(path.read_bytes()[:4096]).digest()
    seed_vals = [b for b in digest[:12]]
    volume = np.ones((size, size, size), dtype=np.float64)
    centers = [
        (4 + seed_vals[0] % 8, 4 + seed_vals[1] % 8, 4 + seed_vals[2] % 8),
        (10 + seed_vals[3] % 8, 10 + seed_vals[4] % 8, 10 + seed_vals[5] % 8),
        (6 + seed_vals[6] % 10, 6 + seed_vals[7] % 10, 6 + seed_vals[8] % 10),
    ]
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    for cz, cy, cx in centers:
        amp = 10.0 + (seed_vals[(cz + cy + cx) % len(seed_vals)] % 20)
        sigma_blob = 1.5
        d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += amp * np.exp(-d2 / (2.0 * sigma_blob ** 2))
    return volume


def _load_feature_rows(path: Path) -> list[dict[str, object]]:
    import csv

    rows: list[dict[str, object]] = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            typed: dict[str, object] = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except ValueError:
                    try:
                        typed[k] = float(v)
                    except ValueError:
                        typed[k] = v
            rows.append(typed)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    root = resolve_dataset_root(args.root)
    segdir = Path(args.segdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scans_fs = discover_operando_scans(root)
    scan_by_name = {s.scan_name: s for s in scans_fs}
    summaries: list[dict[str, object]] = []

    for scan_dir in sorted(d for d in segdir.iterdir() if d.is_dir() and d.name.startswith("scan")):
        name = scan_dir.name
        feat_path = scan_dir / "features.csv"
        lab_path = scan_dir / "labels.npz"
        if not feat_path.exists():
            continue
        if not lab_path.exists():
            print(json.dumps({"error": "Missing labels.npz — re-run segment_dataset", "scan": name}))
            return 1

        rows = _load_feature_rows(feat_path)
        labels_full = np.load(lab_path)["labels"]

        scan_file = scan_by_name.get(name)
        if scan_file is None:
            print(json.dumps({"error": "Scan not found under dataset root", "scan": name}))
            return 1

        try:
            volume = load_primary_volume(scan_file.path)
            if not isinstance(volume, np.ndarray):
                volume = np.asarray(volume, dtype=np.float64)
        except (MissingH5DependencyError, KeyError, ValueError):
            volume = _synth_volume_from_file(scan_file.path)

        if volume.shape != labels_full.shape:
            print(
                json.dumps(
                    {
                        "error": "Volume/labels shape mismatch",
                        "scan": name,
                        "volume": list(volume.shape),
                        "labels": list(labels_full.shape),
                    }
                )
            )
            return 1

        enc = make_multiview_encoder(
            args.backend,  # type: ignore[arg-type]
            model_name=args.model,
        )

        label_ids: list[int] = []
        vectors: list[np.ndarray] = []

        for row in rows:
            lid = int(row["label"])
            mip_mu, mip_chi, mip_d = orthogonal_mips(
                crop_spot_cube(volume, labels_full, lid, row, margin=max(0, args.margin))[0]
            )
            vec = enc.embed(mip_mu, mip_chi, mip_d)
            label_ids.append(lid)
            vectors.append(vec)

        if not vectors:
            dim = 384
        else:
            dim = int(vectors[0].shape[0])

        mat = np.stack(vectors, axis=0) if vectors else np.zeros((0, dim), dtype=np.float32)
        scan_out = outdir / name
        scan_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scan_out / "embeddings.npz",
            labels=np.array(label_ids, dtype=np.int32),
            vectors=mat.astype(np.float32, copy=False),
        )
        manifest = {
            "scan": name,
            "n_spots": len(label_ids),
            "dim": dim,
            "backend": args.backend,
            "model": args.model if args.backend == "torch" else "mock-hash",
            "schema_version": "week4.v1",
        }
        (scan_out / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2))
        summaries.append({**manifest, "embeddings": str(scan_out / "embeddings.npz")})

    (outdir / "embedding_summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for batched DINOv2 multiview and patch-feature inference."""

from __future__ import annotations

import numpy as np
import pytest

from braggtrack.semantic.dino import make_multiview_encoder, make_patch_encoder


def _random_triplets(n: int = 7, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(n):
        h = 16 + 2 * (i % 3)
        w = 18 + 2 * (i % 4)
        triplets.append(
            (
                rng.uniform(0.0, 100.0, size=(h, w)).astype(np.float32),
                rng.uniform(0.0, 100.0, size=(h, w + 2)).astype(np.float32),
                rng.uniform(0.0, 100.0, size=(h + 2, w)).astype(np.float32),
            )
        )
    return triplets


def test_mock_multiview_and_patch_batch_parity() -> None:
    triplets = _random_triplets(6)
    enc = make_multiview_encoder("mock")
    single = np.stack([enc.embed(a, b, c) for a, b, c in triplets], axis=0)
    batched = enc.embed_batch(triplets, batch_size=2)
    assert batched.shape == single.shape
    np.testing.assert_allclose(batched, single, atol=1e-6)
    assert enc.embed_batch([]).shape == (0, single.shape[1])

    penc = make_patch_encoder("mock")
    imgs = [t[0] for t in triplets]
    p_single = [penc.extract_patch_features(img) for img in imgs]
    p_batch = penc.extract_patch_features_batch(imgs, batch_size=3)
    assert len(p_batch) == len(p_single)
    for f_b, f_s in zip(p_batch, p_single, strict=True):
        np.testing.assert_allclose(f_b, f_s, atol=1e-6)


def test_torch_dino_multiview_and_patch_batch_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    triplets = _random_triplets(5, seed=7)
    try:
        enc = make_multiview_encoder("torch", torch_device="cpu")
        penc = make_patch_encoder("torch", torch_device="cpu")
    except Exception as exc:  # pragma: no cover - offline fallback if model cache missing
        pytest.skip(f"DINO model unavailable in environment: {exc}")

    single = np.stack([enc.embed(a, b, c) for a, b, c in triplets], axis=0)
    batched = enc.embed_batch(triplets, batch_size=2)
    assert batched.shape == single.shape
    np.testing.assert_allclose(batched, single, atol=1e-5)

    imgs = [np.pad(t[0], ((24, 24), (24, 24))) for t in triplets]
    p_single = [penc.extract_patch_features(img) for img in imgs]
    p_batch = penc.extract_patch_features_batch(imgs, batch_size=2)
    assert len(p_batch) == len(p_single)
    for f_b, f_s in zip(p_batch, p_single, strict=True):
        assert f_b.shape == f_s.shape
        np.testing.assert_allclose(f_b, f_s, atol=1e-5)

"""Frozen DINO-style embeddings for three orthogonal MIPs (Week 4).

Backend is selected with env ``BRAGGTRACK_DINO_BACKEND``:

* ``mock`` — deterministic CPU-only vector from image bytes (default when
  PyTorch is unavailable).
* ``torch`` — Hugging Face ``facebook/dinov2-small`` (one CLS embedding per view,
  concatenated and L2-normalised).
* ``auto`` — use ``torch`` if import succeeds, else ``mock``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal, Protocol

import numpy as np

BackendName = Literal["auto", "mock", "torch"]


def _resolve_backend(requested: BackendName) -> Literal["mock", "torch"]:
    if requested == "mock":
        return "mock"
    if requested == "torch":
        return "torch"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "mock"
    return "torch"


def _mock_embedding_from_mips(
    mip_mu: np.ndarray,
    mip_chi: np.ndarray,
    mip_d: np.ndarray,
    out_dim: int = 384,
) -> np.ndarray:
    """Deterministic pseudo-embedding for CI / no-torch environments."""
    parts = (
        np.asarray(mip_mu, dtype=np.float32).tobytes(),
        np.asarray(mip_chi, dtype=np.float32).tobytes(),
        np.asarray(mip_d, dtype=np.float32).tobytes(),
    )
    h = hashlib.sha256(b"".join(parts)).digest()
    seed = int.from_bytes(h[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(out_dim).astype(np.float32)
    nrm = float(np.linalg.norm(v))
    if nrm > 0:
        v /= nrm
    return v


def _mips_to_rgb_uint8(mip: np.ndarray) -> np.ndarray:
    m = np.asarray(mip, dtype=np.float64)
    lo, hi = float(np.percentile(m, 1.0)), float(np.percentile(m, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    u = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    g = (u * 255.0).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


class MultiviewEncoder(Protocol):
    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray:
        ...


class MockMultiviewEncoder:
    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray:
        return _mock_embedding_from_mips(mip_mu, mip_chi, mip_d)


class TorchDinoMultiviewEncoder:
    """Loads Dinov2 once; call :meth:`embed` per spot."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray:
        vecs: list[np.ndarray] = []
        with self._torch.no_grad():
            for mip in (mip_mu, mip_chi, mip_d):
                rgb = _mips_to_rgb_uint8(mip)
                inputs = self._proc(images=[rgb], return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                out = self._model(**inputs).last_hidden_state[:, 0, :].squeeze(0)
                x = out.float().cpu().numpy()
                n = float(np.linalg.norm(x))
                if n > 0:
                    x = x / n
                vecs.append(x.astype(np.float32))

        fused = np.concatenate(vecs, axis=0)
        nf = float(np.linalg.norm(fused))
        if nf > 0:
            fused = fused / nf
        return fused.astype(np.float32)


def _requested_backend(explicit: BackendName | None) -> BackendName:
    env_b = os.environ.get("BRAGGTRACK_DINO_BACKEND", "").strip().lower()
    if explicit is not None:
        return explicit
    if env_b in ("mock", "torch", "auto"):
        return env_b  # type: ignore[return-value]
    return "auto"


def make_multiview_encoder(
    backend: BackendName | None = None,
    *,
    model_name: str = "facebook/dinov2-small",
    torch_device: str | None = None,
) -> MultiviewEncoder:
    """Construct a reusable encoder (loads torch weights at most once)."""
    req = _requested_backend(backend)
    use = _resolve_backend(req)
    if use == "mock":
        return MockMultiviewEncoder()
    return TorchDinoMultiviewEncoder(model_name, torch_device)


def embed_multiview_mips(
    mip_mu: np.ndarray,
    mip_chi: np.ndarray,
    mip_d: np.ndarray,
    *,
    backend: BackendName | None = None,
    model_name: str = "facebook/dinov2-small",
    torch_device: str | None = None,
) -> np.ndarray:
    """Return a single L2-normalised concatenated feature vector."""
    return make_multiview_encoder(
        backend, model_name=model_name, torch_device=torch_device
    ).embed(mip_mu, mip_chi, mip_d)

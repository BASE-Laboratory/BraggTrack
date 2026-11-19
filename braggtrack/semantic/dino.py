"""Frozen DINO-style embeddings and patch-level feature extraction.

Backend is selected with env ``BRAGGTRACK_DINO_BACKEND``:

* ``mock`` — deterministic CPU-only vectors from image bytes (default when
  PyTorch is unavailable).
* ``torch`` — Hugging Face DINOv2 (ungated default) or DINOv3 model.
* ``auto`` — use ``torch`` if import succeeds, else ``mock``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any, Literal, Protocol

import numpy as np

BackendName = Literal["auto", "mock", "torch"]
DEFAULT_DINO_MODEL = "facebook/dinov2-small"
DEFAULT_DINO_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"


def _resolve_backend(requested: BackendName) -> Literal["mock", "torch"]:
    if requested == "mock":
        return "mock"
    if requested == "torch":
        return "torch"
    env_override = os.environ.get("BRAGGTRACK_DINO_BACKEND", "").strip().lower()
    if env_override in ("mock", "torch"):
        return "mock" if env_override == "mock" else "torch"
    try:
        import torch  # noqa: F401
    except ImportError:
        return "mock"
    return "torch"


def _load_hf_dino(model_name: str, *, revision: str | None = None) -> tuple[Any, Any]:
    from transformers import AutoImageProcessor, AutoModel

    resolved_rev = revision
    if resolved_rev is None and model_name == DEFAULT_DINO_MODEL:
        resolved_rev = DEFAULT_DINO_REVISION
    hf_kwargs: dict[str, Any] = {"revision": resolved_rev} if resolved_rev is not None else {}

    try:
        proc = AutoImageProcessor.from_pretrained(model_name, **hf_kwargs)
        model = AutoModel.from_pretrained(model_name, **hf_kwargs)
    except OSError as exc:
        msg = str(exc)
        if any(tok in msg.lower() for tok in ("gated", "401", "403", "authenticated")):
            raise OSError(
                f"Model '{model_name}' requires HuggingFace authentication. "
                f"Use the ungated default '{DEFAULT_DINO_MODEL}' or set HF_TOKEN after accepting the model license."
            ) from exc
        raise
    return proc, model


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


def _mips_to_rgb_uint8(mip: np.ndarray, *, preserve_aspect: bool = True) -> np.ndarray:
    m = np.asarray(mip, dtype=np.float64)
    if m.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    lo, hi = float(np.percentile(m, 1.0)), float(np.percentile(m, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    u = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
    g = (u * 255.0).astype(np.uint8)
    if preserve_aspect and g.ndim == 2 and g.shape[0] != g.shape[1]:
        h, w = g.shape
        # Pad onto a centered square canvas (with 8% margin for DINOv2's 256->224 center-crop)
        side = max(h, w, int(np.ceil(max(h, w) / 0.875)))
        canvas = np.zeros((side, side), dtype=np.uint8)
        y0 = (side - h) // 2
        x0 = (side - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = g
        g = canvas
    return np.stack([g, g, g], axis=-1)


class MultiviewEncoder(Protocol):
    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray: ...

    def embed_batch(
        self,
        mips_list: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        batch_size: int = 32,
    ) -> np.ndarray: ...


class MockMultiviewEncoder:
    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray:
        return _mock_embedding_from_mips(mip_mu, mip_chi, mip_d)

    def embed_batch(
        self,
        mips_list: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not mips_list:
            return np.zeros((0, 384), dtype=np.float32)
        return np.stack([self.embed(m0, m1, m2) for m0, m1, m2 in mips_list], axis=0)


class TorchDinoMultiviewEncoder:
    """Loads DINOv2/DINOv3 once; supports both single-spot :meth:`embed` and batched :meth:`embed_batch`."""

    def __init__(self, model_name: str = DEFAULT_DINO_MODEL, device: str | None = None) -> None:
        import torch

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc, self._model = _load_hf_dino(model_name)
        self._model.eval()
        self._model.to(self._device)
        self._hidden = int(getattr(self._model.config, "hidden_size", 384))

    def embed(self, mip_mu: np.ndarray, mip_chi: np.ndarray, mip_d: np.ndarray) -> np.ndarray:
        return self.embed_batch([(mip_mu, mip_chi, mip_d)], batch_size=3)[0]

    def embed_batch(
        self,
        mips_list: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        batch_size: int = 32,
    ) -> np.ndarray:
        if not mips_list:
            return np.zeros((0, 3 * self._hidden), dtype=np.float32)

        all_rgbs: list[np.ndarray] = []
        for mip_mu, mip_chi, mip_d in mips_list:
            all_rgbs.append(_mips_to_rgb_uint8(mip_mu))
            all_rgbs.append(_mips_to_rgb_uint8(mip_chi))
            all_rgbs.append(_mips_to_rgb_uint8(mip_d))

        step = max(1, int(batch_size))
        if str(self._device) == "cpu":
            step = min(step, 16)
        cls_chunks: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(all_rgbs), step):
                chunk = all_rgbs[start : start + step]
                inputs = self._proc(images=chunk, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                cls_tok = self._model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy()
                norms = np.linalg.norm(cls_tok, axis=-1, keepdims=True)
                norms = np.where(norms > 0.0, norms, 1.0)
                cls_chunks.append((cls_tok / norms).astype(np.float32))

        view_vecs = np.concatenate(cls_chunks, axis=0)  # (3 * N, D)
        fused = view_vecs.reshape(len(mips_list), 3 * view_vecs.shape[-1])
        fnorms = np.linalg.norm(fused, axis=-1, keepdims=True)
        fnorms = np.where(fnorms > 0.0, fnorms, 1.0)
        return (fused / fnorms).astype(np.float32)


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
    model_name: str = DEFAULT_DINO_MODEL,
    torch_device: str | None = None,
) -> MultiviewEncoder:
    """Construct a reusable encoder (loads torch weights at most once)."""
    req = _requested_backend(backend)
    use = _resolve_backend(req)
    if use == "mock":
        return MockMultiviewEncoder()
    return TorchDinoMultiviewEncoder(model_name, torch_device)


# ---------------------------------------------------------------------------
# Patch-level feature extraction (for ROI precursors and DINO segmentation)
# ---------------------------------------------------------------------------


class PatchFeatureEncoder(Protocol):
    """Interface for extracting spatially-resolved patch features."""

    def extract_patch_features(self, image_2d: np.ndarray) -> np.ndarray:
        """Return (H_patches, W_patches, D_features) feature map from a 2D grayscale image."""
        ...

    def extract_patch_features_batch(
        self,
        images_2d: Sequence[np.ndarray],
        *,
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Return list of (H_patches, W_patches, D_features) maps for a batch of 2D images."""
        ...

    @property
    def patch_size(self) -> int: ...

    @property
    def feature_dim(self) -> int: ...


class MockPatchEncoder:
    """Deterministic hash-based patch features for CI (no GPU)."""

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def feature_dim(self) -> int:
        return 384

    def extract_patch_features(self, image_2d: np.ndarray) -> np.ndarray:
        h, w = image_2d.shape[:2]
        h_p = max(1, h // self.patch_size)
        w_p = max(1, w // self.patch_size)
        seed = int.from_bytes(
            hashlib.sha256(np.asarray(image_2d, dtype=np.float32).tobytes()).digest()[:8],
            "little",
            signed=False,
        )
        rng = np.random.default_rng(seed)
        features = rng.standard_normal((h_p, w_p, self.feature_dim)).astype(np.float32)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (features / norms).astype(np.float32)

    def extract_patch_features_batch(
        self,
        images_2d: Sequence[np.ndarray],
        *,
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        return [self.extract_patch_features(img) for img in images_2d]


class TorchDinoPatchEncoder:
    """Extracts DINOv2/DINOv3 patch tokens as spatial feature maps (single or batched)."""

    def __init__(self, model_name: str = DEFAULT_DINO_MODEL, device: str | None = None) -> None:
        import torch

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc, self._model = _load_hf_dino(model_name)
        self._model.eval()
        self._model.to(self._device)
        self._patch_size = int(getattr(self._model.config, "patch_size", 14))
        self._feature_dim = int(getattr(self._model.config, "hidden_size", 384))

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def extract_patch_features(self, image_2d: np.ndarray) -> np.ndarray:
        return self.extract_patch_features_batch([image_2d], batch_size=1)[0]

    def extract_patch_features_batch(
        self,
        images_2d: Sequence[np.ndarray],
        *,
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        if not images_2d:
            return []
        rgbs = [_mips_to_rgb_uint8(img, preserve_aspect=False) for img in images_2d]
        step = max(1, int(batch_size))
        if str(self._device) == "cpu":
            step = min(step, 16)
        results: list[np.ndarray] = []
        with self._torch.no_grad():
            for start in range(0, len(rgbs), step):
                chunk = rgbs[start : start + step]
                inputs = self._proc(images=chunk, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                out = self._model(**inputs)
                bsz = int(inputs["pixel_values"].shape[0])
                h_img = int(inputs["pixel_values"].shape[2])
                w_img = int(inputs["pixel_values"].shape[3])
                h_p = h_img // self._patch_size
                w_p = w_img // self._patch_size
                expected_patches = h_p * w_p
                seq_len = int(out.last_hidden_state.shape[1])
                skip = max(0, seq_len - expected_patches)
                patch_tokens = out.last_hidden_state[:, skip : skip + expected_patches, :]
                fmaps = patch_tokens.reshape(bsz, h_p, w_p, -1).float().cpu().numpy()
                norms = np.linalg.norm(fmaps, axis=-1, keepdims=True)
                norms = np.where(norms > 0.0, norms, 1.0)
                normed = (fmaps / norms).astype(np.float32)
                for i in range(bsz):
                    results.append(normed[i])
        return results


def make_patch_encoder(
    backend: BackendName | None = None,
    *,
    model_name: str = DEFAULT_DINO_MODEL,
    torch_device: str | None = None,
) -> PatchFeatureEncoder:
    """Construct a reusable patch-level encoder."""
    req = _requested_backend(backend)
    use = _resolve_backend(req)
    if use == "mock":
        return MockPatchEncoder()
    return TorchDinoPatchEncoder(model_name, torch_device)


def embed_multiview_mips(
    mip_mu: np.ndarray,
    mip_chi: np.ndarray,
    mip_d: np.ndarray,
    *,
    backend: BackendName | None = None,
    model_name: str = DEFAULT_DINO_MODEL,
    torch_device: str | None = None,
) -> np.ndarray:
    """Return a single L2-normalised concatenated feature vector."""
    return make_multiview_encoder(backend, model_name=model_name, torch_device=torch_device).embed(
        mip_mu, mip_chi, mip_d
    )

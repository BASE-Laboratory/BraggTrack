"""Filesystem locations for bundled sample data and CLI defaults."""

from __future__ import annotations

from pathlib import Path

# braggtrack/io/paths.py → parents[2] is repository root when installed from source tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_OPERANDO_DIR = _REPO_ROOT / "data" / "sample_operando"


def sample_operando_root() -> Path:
    """Directory containing bundled ``scanNNNN/`` example folders (may be absent in sdist-only installs)."""
    return SAMPLE_OPERANDO_DIR


def default_dataset_root() -> Path:
    """Prefer bundled sample operando data when present; otherwise the process working directory."""
    sample = sample_operando_root()
    if sample.is_dir() and any(sample.glob("scan*/")):
        return sample
    return Path(".")


def resolve_dataset_root(explicit: str | None) -> Path:
    """Resolve CLI ``root`` argument: explicit path wins, else :func:`default_dataset_root`."""
    if explicit is not None:
        return Path(explicit)
    return default_dataset_root()

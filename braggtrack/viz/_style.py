"""Shared styling and optional-dependency guards for braggtrack.viz."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Colorblind-safe Okabe-Ito inspired scientific palette
OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermilion / split highlight
    "#009E73",  # bluish green / merge highlight
    "#CC79A7",  # reddish purple
    "#E69F00",  # amber / gap-close highlight
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#333333",  # charcoal
)


def _require_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "braggtrack.viz requires matplotlib. Install with `pip install braggtrack[notebook]`."
        ) from exc
    return matplotlib, plt


def apply_axis_style(ax: Any) -> None:
    """Apply crisp publication-ready spine and grid styling to a Matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)


def maybe_save(fig: Any, save_path: str | Path | None, *, dpi: int = 200) -> Path | None:
    if save_path is None:
        return None
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return out

"""BraggTrack — 4D grain kinematics, lineage, and fracture tracking for operando diffraction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("braggtrack")
except PackageNotFoundError:
    __version__ = "0.4.0.dev0"

__all__ = ["autonomous", "io", "segmentation", "semantic", "tracking", "viz"]

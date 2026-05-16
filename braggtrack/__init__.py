"""BraggTrack — semantic 4D kinematics and fracture tracking for operando diffraction."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("braggtrack")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

__all__ = ["io", "segmentation", "semantic", "tracking"]

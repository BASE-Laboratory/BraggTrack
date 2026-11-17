"""Publication plotting and interactive HTML 4D beamtime explorer for BraggTrack."""

from .html_report import build_explorer_payload, generate_html_report
from .kinematics import plot_kinematics_dashboard
from .lineage import plot_lineage_dag
from .mips import plot_scan_projection_overlay, plot_spot_mip_card
from .precursors import plot_precursor_timeline

__all__ = [
    "build_explorer_payload",
    "generate_html_report",
    "plot_kinematics_dashboard",
    "plot_lineage_dag",
    "plot_precursor_timeline",
    "plot_scan_projection_overlay",
    "plot_spot_mip_card",
]

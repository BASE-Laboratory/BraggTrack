"""Dataset validation for Week 1 data-contract checks."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ExperimentSequence


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    code: str
    message: str


REQUIRED_FIELDS = ("sample_name", "start_time", "end_time")


def validate_sequence(sequence: ExperimentSequence) -> list[ValidationIssue]:
    """Validate ordering and metadata completeness for an experiment sequence."""

    issues: list[ValidationIssue] = []

    if not sequence.scans:
        issues.append(ValidationIssue("error", "no_scans", "No scan files were discovered."))
        return issues

    if not sequence.is_monotonic():
        issues.append(
            ValidationIssue(
                "error",
                "non_monotonic_sequence",
                "Scan sequence indexes are not contiguous and increasing.",
            )
        )

    sample_names = {scan.sample_name for scan in sequence.scans if scan.sample_name}
    if len(sample_names) > 1:
        issues.append(
            ValidationIssue(
                "warning",
                "sample_name_changed",
                f"Multiple sample names were detected across scans: {sorted(sample_names)}",
            )
        )

    for scan in sequence.scans:
        for field in REQUIRED_FIELDS:
            value = getattr(scan, field)
            if value is None:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "missing_metadata",
                        f"{scan.scan_name}: missing required metadata field '{field}'.",
                    )
                )
        if scan.extras.get("metadata_error"):
            issues.append(
                ValidationIssue(
                    "warning",
                    "metadata_backend_unavailable",
                    f"{scan.scan_name}: {scan.extras['metadata_error']}",
                )
            )

    return issues

"""Week 1 acceptance checks for BraggTrack.

Checks:
1. All three sample scans are discovered and ordered as 1,2,3.
2. Sequence monotonicity is true.
3. Validation report can be produced (warnings allowed; errors fail).
"""

from __future__ import annotations

import json
from pathlib import Path

from braggtrack.io import BeamlineAdapter, validate_sequence


EXPECTED_SCAN_COUNT = 3
EXPECTED_INDEXES = [1, 2, 3]


def main() -> int:
    adapter = BeamlineAdapter(Path('.'))
    sequence = adapter.build_sequence()
    issues = validate_sequence(sequence)

    failures: list[str] = []

    if len(sequence.scans) != EXPECTED_SCAN_COUNT:
        failures.append(
            f"Expected {EXPECTED_SCAN_COUNT} scans, found {len(sequence.scans)}."
        )

    indexes = [scan.sequence_index for scan in sequence.scans]
    if indexes != EXPECTED_INDEXES:
        failures.append(f"Expected ordered indexes {EXPECTED_INDEXES}, got {indexes}.")

    if not sequence.is_monotonic():
        failures.append("Sequence is not monotonic.")

    if any(issue.level == 'error' for issue in issues):
        failures.append("Validation returned one or more error-level issues.")

    report = {
        'scan_count': len(sequence.scans),
        'indexes': indexes,
        'is_monotonic': sequence.is_monotonic(),
        'error_count': sum(1 for issue in issues if issue.level == 'error'),
        'warning_count': sum(1 for issue in issues if issue.level == 'warning'),
        'failures': failures,
    }

    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())

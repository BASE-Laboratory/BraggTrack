"""Pre-PR diagnostics for merge-conflict hygiene and branch sync hints."""

from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def main() -> int:
    failures = []

    conflict = _run(["git", "grep", "-nE", "^(<<<<<<<|=======|>>>>>>>)"])
    if conflict.returncode == 0:
        print("Conflict markers detected:")
        print(conflict.stdout.strip())
        failures.append("conflict_markers")
    else:
        print("No conflict markers detected in tracked files.")

    has_origin_main = _run(["git", "show-ref", "--verify", "--quiet", "refs/remotes/origin/main"]).returncode == 0
    if has_origin_main:
        ahead_behind = _run(["git", "rev-list", "--left-right", "--count", "HEAD...origin/main"])
        if ahead_behind.returncode == 0:
            ahead, behind = ahead_behind.stdout.strip().split()
            print(f"Branch divergence vs origin/main: ahead={ahead}, behind={behind}")
            if int(behind) > 0:
                print("Recommended: git fetch origin && git rebase origin/main")
        else:
            print("Could not compute divergence against origin/main.")
    else:
        print("origin/main not configured in this environment; cannot auto-check rebase state.")

    if failures:
        print("pre_pr_check=FAIL")
        return 1

    print("pre_pr_check=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

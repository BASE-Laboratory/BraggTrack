# Coding practices (BraggTrack)

Conventions for contributors and for AI-assisted edits in this repository.

## Scope and style

- **Minimal diffs:** change only what the task requires; avoid drive-by refactors or unrelated formatting.
- **Match existing code:** naming, imports, type hints, and test style should blend with neighbouring modules.
- **No committed artifacts:** outputs go under `artifacts/` (gitignored). Bundled **inputs** live under `data/sample_operando/`.

## Python

- **Version:** `requires-python >= 3.10` (see `pyproject.toml`).
- **Types:** Prefer explicit annotations on public functions and complex internals; use `Protocol` for pluggable interfaces (e.g. tracking `CostFunction`).
- **NumPy / SciPy:** Use vectorised operations on hot paths; delegate heavy distance work to `cdist` / BLAS-style matrix ops where possible.
- **Optional dependencies:** `h5py`, `torch`, `transformers` may be absent (CI installs a minimal set). Use feature checks or env-driven backends (e.g. mock DINO) so tests stay deterministic.

## Data and I/O

- **Dataset root:** always a directory containing `scan*` subfolders; discovery is defined in `braggtrack/io/discovery.py`.
- **Resolving CLI paths:** use `braggtrack.io.paths.resolve_dataset_root(args.root)` so “no argument” prefers bundled sample data when present.
- **Contracts:** serialised tables (`features.csv`, `embeddings.npz`) should stay backward-compatible or bump a `schema_version` field in JSON summaries.

## Testing

- **Framework:** `unittest` (`python -m unittest discover -s tests -v`).
- **Before a PR:** `python scripts/pre_pr_check.py` and `python scripts/ci_report.py` when dependencies are installed.
- **H5 / NeXus tests:** do not assume `h5py` is missing; use `unittest.mock` to exercise error paths (see `tests/test_nexus_dependency.py`).

## Documentation

- **User-facing workflow:** `README.md` quick starts.
- **Design depth:** `docs/architecture.md` (this repo’s structure and data flow).
- **Contributor onboarding:** `CONTRIBUTING.md` (links here and to architecture).

## Commits and PRs

- Prefer **clear, imperative** commit subjects (`feat(tracking): …`, `fix(io): …`, `docs: …`).
- PR description should state **what** changed and **why**, and mention any **schema or path** changes (e.g. sample data location).

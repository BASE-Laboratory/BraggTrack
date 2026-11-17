# Contributing to BraggTrack

Thanks for helping improve BraggTrack.

## Before you open a PR

1. Install the package in editable mode with developer dependencies (`pip install -e ".[dev]"` for full PyTorch, notebook, pytest, and Ruff tooling, or `pip install -e .` for the lightweight core).
2. Run **`ruff check braggtrack/ tests/ scripts/`** and **`ruff format --check braggtrack/ tests/ scripts/`**.
3. Run **`python scripts/pre_pr_check.py`** (conflict markers and optional sync hint against `origin/main`).
4. Run **`python scripts/ci_report.py`** (unit tests + all 9 acceptance gates covering IO, smoke, segmentation, tracking, embedding, DINO, events/pipeline, and interactive report + ImageD11/NeXus exports).

## Where things live

- **[Architecture](docs/architecture.md)** — package layers (`io`, `segmentation`, `semantic`, `tracking`, `viz`, `cli`), data flow, extension points.
- **[Benchmarks](docs/benchmarks.md)** — 5-scenario quantitative evaluation (`scripts/benchmark_suite.py`) and ESRF-ID03 `facebook/dinov2-small` case study.
- **[Coding practices](docs/coding-practices.md)** — style, testing, optional deps, CI expectations.
- **Sample scan data** — versioned under [`data/sample_operando/`](data/sample_operando/README.md) (not the repo root).

## Default dataset root

If you omit the dataset `root` argument on CLIs such as `segment_dataset`, the package uses **`data/sample_operando`** when that tree exists (editable install from a checkout); otherwise it falls back to the current directory. Override by passing an explicit path.

## Questions

Open a discussion or issue on the project repository, or tag maintainers in your PR.

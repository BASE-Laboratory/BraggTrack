# Contributing to BraggTrack

Thanks for helping improve BraggTrack.

## Before you open a PR

1. Install the package in editable mode with dev/runtime dependencies you need (`pip install -e .` plus `numpy`, `scipy`, `scikit-image`, `networkx`; add `h5py` / `torch` / `transformers` as required).
2. Run **`python scripts/pre_pr_check.py`** (conflict markers and optional sync hint against `origin/main`).
3. Run **`python scripts/ci_report.py`** (unit tests + Week 1–4 acceptance scripts).

## Where things live

- **[Architecture](docs/architecture.md)** — package layers, data flow, extension points.
- **[Coding practices](docs/coding-practices.md)** — style, testing, optional deps, CI expectations.
- **Sample scan data** — versioned under [`data/sample_operando/`](data/sample_operando/README.md) (not the repo root).

## Default dataset root

If you omit the dataset `root` argument on CLIs such as `segment_dataset`, the package uses **`data/sample_operando`** when that tree exists (editable install from a checkout); otherwise it falls back to the current directory. Override by passing an explicit path.

## Questions

Open a discussion or issue on the project repository, or tag maintainers in your PR.

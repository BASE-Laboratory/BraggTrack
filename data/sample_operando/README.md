# Sample operando dataset (bundled)

This directory holds **small example scans** shipped with BraggTrack for tests, CI, and local experimentation.

## Layout convention

Any **dataset root** passed to the CLIs (`segment_dataset`, `embed_dataset`, `inspect_datasets`, etc.) must contain one folder per scan:

```text
<dataset_root>/
  scan0001/
    *.h5          # NeXus / HDF5 volume (naming pattern configurable in discovery)
  scan0002/
  scan0003/
```

Folder names must start with `scan` and sort in experiment order. Numeric indices (e.g. `0001` → sequence index `1`) are parsed by `BeamlineAdapter`.

## Your own data

Keep large or proprietary experiments **outside** the repository (or under a path you add to `.gitignore`). Point tools at that root explicitly:

```bash
python -m braggtrack.cli.segment_dataset /path/to/my_experiment --outdir artifacts/week2
```

## See also

- [Architecture](../../docs/architecture.md) — how I/O and discovery fit the pipeline
- [Coding practices](../../docs/coding-practices.md) — extending discovery or CLIs

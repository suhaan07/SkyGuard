# SkyGuard agent instructions

## Project navigation
- Treat `SkyGuard/` as the repo root for this project.
- Keep path examples relative to the repo root unless the user explicitly asks for absolute paths.
- Refer to [`README.md`](README.md) and [`data_dictionary.md`](data_dictionary.md) for dataset structure and field definitions.

## Data loading convention
- Use `src.data.load_split("train")` / `load_split("test")`.
- Do **not** pass `data/train` or `data/test` into `load_split`; `src/data.py` already builds `data/<split>/<split>_*.csv|npz` internally.
- When path bugs appear, inspect `src/data.py` first:
  - `load_tabular`, `load_notes`, and `load_signals` construct the file paths.
  - `load_split` only orchestrates the three loaders.

## Notebook conventions
- Keep split names, CV folds, and feature builders consistent across notebooks so Stage 1 / Stage 2 / Stage 3 results remain comparable.
- Prefer updating the shared loader or feature module over duplicating path logic inside notebooks.
- If a notebook cell fails on file access, check whether the notebook is being run from the `SkyGuard/` root and whether the cell is using a split name instead of a folder path.

## What to change together
- If dataset locations change, update `src/data.py`, `README.md`, and any notebook examples together.
- If you add a new data convention, document it briefly here and link to the canonical source instead of copying the full details.

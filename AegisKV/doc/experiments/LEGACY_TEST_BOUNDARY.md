# Current and legacy test boundary

`AegisKV/experiments/` is the only current S1--S9 experiment implementation.
Its self-contained tests live in `AegisKV/experiments/tests/` and validate the
current three-layer JSON matrix plus controlled mechanism construction.

Some repository-global files under `test/unit/benchmark/` and `test/unit/srt/`
still reference historical paths such as `benchmark/AegisKV/`, `AegisKV/demo/`,
or earlier named experiment directories. Those files belong to archived
campaigns and are not evidence for the current suite. They have intentionally
not been rewritten to make old result contracts look current.

Use these commands for the current boundary:

```bash
uv run --extra benchmark pytest AegisKV/experiments/tests test/unit/srt/test_aegis_complete.py -q
uv run --extra benchmark python AegisKV/experiments/run.py validate
```

The repository-wide default pytest command remains a separate migration task:
each historical test must either be moved with its archived implementation or
rewritten against a current, equivalent contract. It must not be counted as a
failure of the current S1--S9 lineage without checking which path it imports.

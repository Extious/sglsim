# DP=2 Smoke 32852 Failure

The first DP=2 smoke attempt was submitted as Slurm job `32852` and then
cancelled after the server failed during import.

Observed failure:

```text
ModuleNotFoundError: No module named 'IPython'
```

Cause:

The Slurm script used `/usr/local/bin/python`, while the remote SGLang
environment expects the repository-managed `uv run python` environment.

Fix applied:

`slurm/aegiskv_dp2_smoke.sbatch` now launches the server, smoke client, and
summary script through:

```text
/home/comp/25482351/.local/bin/uv run python
```

Validation:

```text
cd ~/sglang
/home/comp/25482351/.local/bin/uv run python -c "import IPython, sglang"
```

passed on `hkbugpumanode`.

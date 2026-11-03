"""Runs any repo command on a Modal GPU, from the same uv lock file as a local install. Outputs land on the Modal volume `retina-runs` under the run name.

    uv run --extra modal modal run tools/modal_run.py --cmd "python tests/test_end_to_end.py OUT" --run test --gpu L4
    uv run --extra modal modal volume get retina-runs test ./test_out

OUT in the command is replaced by the run's folder on the volume. A Modal secret named `huggingface` with HF_TOKEN is needed for the gated DINOv3 weights.
"""
import pathlib
import shlex
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (
    modal.Image.debian_slim("3.12")
    .uv_sync(str(ROOT))
    .env({"HF_HOME": "/cache", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(str(ROOT), "/root/retina", ignore=["archive", "archive/**", ".venv", ".venv/**", "**/__pycache__", "*_out", "*_out/**", "mar_toy", "mar_toy/**"])
)
app = modal.App("retina-repro", image=image)
cache = modal.Volume.from_name("retina-cache", create_if_missing=True)
runs = modal.Volume.from_name("retina-runs", create_if_missing=True)
KW = dict(timeout=4 * 3600, volumes={"/cache": cache, "/runs": runs}, secrets=[modal.Secret.from_name("huggingface")])


def _run(cmd, run):
    out = f"/runs/{run}"
    pathlib.Path(out).mkdir(parents=True, exist_ok=True)
    args = [out if a == "OUT" else a for a in shlex.split(cmd)]
    rc = subprocess.run(args, cwd="/root/retina", env={**__import__("os").environ, "PYTHONPATH": "/root/retina"}).returncode
    cache.commit()
    runs.commit()
    return rc


@app.function(gpu="L4", **KW)
def l4(cmd, run):
    return _run(cmd, run)


@app.function(gpu="A10G", **KW)
def a10g(cmd, run):
    return _run(cmd, run)


@app.function(gpu="A100-40GB", **KW)
def a100(cmd, run):
    return _run(cmd, run)


@app.local_entrypoint()
def main(cmd: str, run: str = "run", gpu: str = "L4"):
    rc = {"L4": l4, "A10G": a10g, "A100": a100}[gpu.upper()].remote(cmd, run)
    print("exit code", rc)
    if rc:
        raise SystemExit(rc)

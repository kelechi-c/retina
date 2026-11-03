"""Runs every command in README.md, in order: each line of the bash blocks and each python block, from the repository root. Install, login and Modal lines are skipped; they need a person or another machine.

usage: uv run python tests/readme_check.py [--no-uv]

--no-uv drops the `uv run` prefix, for a machine whose environment is already synced (the Modal image).
"""
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time

import click

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP = ("git clone", "uv sync", "uv run hf auth", "uv run modal")


@click.command()
@click.option("--uv/--no-uv", default=True, show_default=True)
def main(uv):
    text = (ROOT / "README.md").read_text()
    steps = []
    for lang, body in re.findall(r"```(\w+)\n(.*?)```", text, re.S):
        if lang == "bash":
            steps += [("bash", l.strip()) for l in body.splitlines() if l.strip() and not l.strip().startswith(SKIP)]
        elif lang == "python":
            steps.append(("python", body))
    fails = []
    for kind, s in steps:
        if kind == "python":
            f = pathlib.Path(tempfile.mkdtemp()) / "readme_snippet.py"
            f.write_text(s)
            cmd = f"{'uv run ' if uv else ''}python {f}"
        else:
            cmd = s if uv else re.sub(r"^uv run ", "", s)
        print(f"\n$ {cmd if kind == 'bash' else 'python <README snippet>'}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, shell=True, cwd=ROOT, env={**os.environ, **({} if uv else {"PY": "python"})}).returncode
        print(f"-> exit {rc} in {time.time() - t0:.0f} s", flush=True)
        if rc:
            fails.append(cmd)
    print("\nREADME commands:", len(steps), "failed:", fails)
    print("README_PASSED" if not fails else "README_FAILED", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

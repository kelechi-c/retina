"""Samples pictures from MAR by label and saves them with their token counts.

usage: uv run python -m mar.sample OUT [--per 4] [--labels simple,dense] [--ckpt mar.pt] [--live]

Files are named <label>_<seed>_<tokens>.png; same_<label>.png holds one seed and layout under each label.
"""
import pathlib

import click

from retina import save_png

from .model import NAMES, Sampler


@click.command()
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--per", default=4, show_default=True, help="pictures per label")
@click.option("--labels", default=",".join(NAMES), show_default=True)
@click.option("--ckpt", default=None, help="a file from mar.train; default the released EMA weights")
@click.option("--live", is_flag=True, help="live weights instead of the EMA")
@click.option("--temp", default=.95, show_default=True)
@click.option("--cfg", default=1., show_default=True)
def main(out, per, labels, ckpt, live, temp, cfg):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = Sampler("cuda", ckpt, ema=not live)
    for name in labels.split(","):
        for j in range(per):
            p = s.generate(name, seed=1000 * NAMES.index(name) + j, temp=temp, cfg=cfg)
            save_png(s.decode(p), out / f"{name}_{j}_{len(p['seq'])}.png")
            print(name, j, len(p["seq"]), "tokens", flush=True)
    mid = s.layouts.index(sorted(s.layouts, key=lambda q: q[0])[len(s.layouts) // 2])
    for name in labels.split(","):
        save_png(s.decode(s.generate(name, li=mid, seed=7, temp=temp, cfg=cfg)), out / f"same_{name}.png")


if __name__ == "__main__":
    main()

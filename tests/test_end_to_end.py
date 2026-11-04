"""End to end on a GPU: the tokenizer's defaults, encode, decode, an exact pack round trip, simple images cheaper than busy ones, then one MAR sample per label.

usage: uv run python tests/test_end_to_end.py [OUT] [--no-mar]
"""
import pathlib
import sys
import time

import click
import torch

from mar.model import NAMES, Sampler
from retina import Tokenizer, load_img, save_png

ROOT = pathlib.Path(__file__).resolve().parent.parent


@click.command()
@click.argument("out", default="test_out", type=click.Path(file_okay=False))
@click.option("--mar/--no-mar", default=True, show_default=True, help="also sample the MAR model")
def main(out, mar):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    tok = Tokenizer()
    assert tok.net is not None, "the S1 refiner must load by default"
    assert tok.v.__class__.__name__ == "Flux2"
    fails, n = [], {}
    for f in sorted((ROOT / "assets" / "examples").glob("*/*.jpg")):
        x = load_img(str(f), 768).to(dev)
        t0 = time.time()
        t = tok.encode(x)
        enc = time.time() - t0
        r = tok.decode(t)
        p = Tokenizer.pack(t)
        rt = (r - tok.decode(Tokenizer.unpack(p))).abs().max().item()
        n[f.stem] = len(p["seq"])
        print(f"{f.stem:16s} {len(p['seq']):5d} tokens  layout {p['layout']}  round trip {rt:.1e}  encode {enc:.2f} s", flush=True)
        if p["seq"].shape[1] != 128 or rt != 0 or not torch.isfinite(r).all():
            fails.append(f.stem)
        save_png(r, out / f"{f.stem}.png")
    if not max(n["mondrian_piet"], n["mondrian_b144"]) * 8 < min(n["veronese"], n["bruegel"]):
        fails.append("Mondrians should cost under an eighth of the busy paintings")

    s = Sampler(dev) if mar else None
    for name in NAMES if mar else []:
        p = s.generate(name, seed=0)
        r = s.decode(p)
        print(f"MAR {name:9s} {len(p['seq']):5d} tokens", flush=True)
        if r.shape != (1, 3, 512, 512) or not torch.isfinite(r).all():
            fails.append(f"MAR {name}")
        save_png(r, out / f"mar_{name}.png")
    print("FAILURES", fails)
    print("TEST_PASSED" if not fails else "TEST_FAILED", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

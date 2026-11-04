"""The tokenizer's main options, each on a simple and a busy example: every setting must encode, decode to a finite image, pack and unpack exactly, and move the token count the way it says.

usage: uv run python tests/test_flags.py [--vaes flux2,flux]
"""
import pathlib
import sys

import click
import torch

from retina import Tokenizer, load_img

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETTINGS = {
    "default": {},
    "no S1 refiner": dict(learned=None),
    "original DReX": dict(drext=None),
    "gap 1.5": dict(gap=1.5),
    "gap 1.0": dict(gap=1.),
    "no PSNR floor": dict(floor=0.),
    "whole-image DReX": dict(tile=0),
    "no warp": dict(warp_max=0),
}


@click.command()
@click.option("--vaes", default="flux2", show_default=True, help="comma list of flux2, flux, qwen")
def main(vaes):
    dev, fails = "cuda", []
    imgs = {k: load_img(str(next((ROOT / "assets" / "examples").glob(f"*/{k}.jpg"))), 768).to(dev) for k in ("mondrian_b144", "veronese")}
    n = {}
    for vae in vaes.split(","):
        for name, kw in SETTINGS.items() if vae == "flux2" else {"default": {}}.items():
            tok = Tokenizer(vae, dev, **kw)
            for k, x in imgs.items():
                t = tok.encode(x)
                r = tok.decode(t)
                p = Tokenizer.pack(t)
                rt = (r - tok.decode(Tokenizer.unpack(p))).abs().max().item()
                n[vae, name, k] = len(p["seq"])
                ok = torch.isfinite(r).all().item() and r.shape == x.shape and rt == 0
                print(f"{vae:6s} {name:16s} {k:14s} {len(p['seq']):5d} tokens  warped {t['knots'] is not None!s:5s} S1 {'stats' in t!s:5s} round trip {rt:.0e}", flush=True)
                if not ok:
                    fails.append(f"{vae} {name} {k}")
            del tok
            torch.cuda.empty_cache()
        if not n[vae, "default", "mondrian_b144"] < n[vae, "default", "veronese"]:
            fails.append(f"{vae}: Mondrian not cheaper than Veronese")
    if "flux2" in vaes.split(","):
        g = lambda s, k: n["flux2", s, k]
        checks = {
            "no warp keeps the Mondrian unwarped and costs at least as much": g("no warp", "mondrian_b144") >= g("default", "mondrian_b144") - 2,
            "gap 1.5 widens the gap over gap 1.0": g("gap 1.5", "veronese") / g("gap 1.5", "mondrian_b144") >= g("gap 1.0", "veronese") / g("gap 1.0", "mondrian_b144"),
            "no S1 adds no statistics tokens": g("no S1 refiner", "mondrian_b144") <= g("default", "mondrian_b144"),
        }
        for c, ok in checks.items():
            print(("PASS " if ok else "FAIL ") + c, flush=True)
            if not ok:
                fails.append(c)
    print("FAILURES", fails)
    print("TEST_PASSED" if not fails else "TEST_FAILED", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

"""Tokenize one image, print what it costs, decode it, and check the packed sequence round-trips.

usage: uv run python examples/quickstart.py IMAGE [OUT.png] [--side 768]
"""
import click
import torch

from retina import Tokenizer, load_img, save_png


@click.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.argument("out", default="decoded.png")
@click.option("--side", default=768, show_default=True, help="square side; multiples of 256 are what the budget was fit on")
def main(image, out, side):
    tok = Tokenizer()  # FLUX.2-dev VAE, tuned DReX budget, warp below 200 tokens, S1 refiner at or below 600
    x = load_img(image, side).cuda()  # centre crop, [1,3,side,side] in [-1,1]
    t = tok.encode(x)
    p = Tokenizer.pack(t)  # one [N,128] sequence plus its layout, the form an AR model reads
    print(f"{image}: {len(p['seq'])} tokens (the plain FLUX.2 grid is {(side // 16) ** 2})")
    print(f"DReX {t['drex']:.2f}, warped {t['knots'] is not None}, refiner statistics {'stats' in t}")
    y = tok.decode(t)
    assert torch.equal(y, tok.decode(Tokenizer.unpack(p)))  # nothing is lost in packing
    save_png(y, out)
    print("saved", out)


if __name__ == "__main__":
    main()

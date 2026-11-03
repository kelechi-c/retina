"""Tokens per image, and how close each decode is to the plain FLUX.2 grid decode, over a folder of images grouped by subfolder.

usage: uv run python tests/evaluate.py ROOT OUT [--side 768] [--no-metrics] [--save]

ROOT/<group>/*.jpg|png. Writes OUT/eval.json (per image) and prints per group: median and p90 tokens, then DINOv3 patch and global cosine, LPIPS and PSNR against the grid decode. p90 is the worse tail: the 90th percentile of tokens and LPIPS, the 10th of similarity and PSNR. Images at the full grid decode identically, and their PSNR is reported as 120.
"""
import json
import pathlib

import click
import numpy as np
import torch
import torch.nn.functional as F

from retina import Tokenizer, load_img, save_png

DINO = "facebook/dinov3-vitb16-pretrain-lvd1689m"  # gated: request access on Hugging Face


class Compare:
    """Our decode against the grid decode: DINOv3 patch and global cosine at native resolution, LPIPS, PSNR. Both rounded to 8 bit first, as saved images."""

    def __init__(self, dev, side):
        import lpips
        from transformers import AutoModel

        self.dino = AutoModel.from_pretrained(DINO).eval().to(dev).requires_grad_(False)
        self.lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        self.mean = torch.tensor((.485, .456, .406), device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor((.229, .224, .225), device=dev).view(1, 3, 1, 1)
        self.patches = (side // 16) ** 2

    @torch.no_grad()
    def feats(self, v):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.dino(pixel_values=((v + 1) / 2 - self.mean) / self.std).last_hidden_state.float()
        return F.normalize(h[:, 0], dim=-1), F.normalize(h[:, -self.patches :], dim=-1)

    @torch.no_grad()
    def __call__(self, p, q):
        p, q = [((v.clamp(-1, 1) + 1) * 127.5).round() / 127.5 - 1 for v in (p, q)]
        (cp, pp), (cq, pq) = self.feats(p), self.feats(q)
        mse = ((p - q) / 2).pow(2).mean().clamp_min(1e-12)
        return dict(dino_patch=(pp * pq).sum(-1).mean().item(), dino_global=(cp * cq).sum().item(), lpips=self.lp(p, q).item(), psnr=(-10 * torch.log10(mse)).item())


@click.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False))
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--side", default=768, show_default=True, help="square side the images are resized to")
@click.option("--metrics/--no-metrics", default=True, show_default=True, help="compare against the grid decode")
@click.option("--save", is_flag=True, help="also save our decode and the grid decode")
def main(root, out, side, metrics, save):
    root, out, dev = pathlib.Path(root), pathlib.Path(out), "cuda"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in root.glob("*/*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    if not files:
        raise click.UsageError(f"no images under {root}/<group>/")
    tok = Tokenizer("flux2", dev)
    cmp = Compare(dev, side) if metrics else None
    res = {}
    for f in files:
        g, k = f.parent.name, f"{f.parent.name}/{f.stem}"
        x = load_img(str(f), side).to(dev)
        t = tok.encode(x)
        r = dict(group=g, tokens=len(Tokenizer.pack(t)["seq"]), counted=Tokenizer.length(t), drex=float(t["drex"]), warped=t["knots"] is not None, s1="stats" in t)
        if metrics or save:
            ours = tok.decode(t)
            with torch.no_grad():
                grid = tok.v.dec(tok.v.enc(x)).float().clamp(-1, 1)
            if metrics:
                r["vs_grid"] = cmp(ours, grid)
            if save:
                save_png(ours, out / f"{g}__{f.stem}_ours.png")
                save_png(grid, out / f"{g}__{f.stem}_grid.png")
        res[k] = r
        print(k, {q: (round(v, 3) if isinstance(v, float) else v) for q, v in r.items() if q != "vs_grid"}, flush=True)
    json.dump(res, open(out / "eval.json", "w"), indent=1)

    grid_tokens = (side // 16) ** 2
    print(f"\n{'group':16s} {'n':>3s} {'tokens p50/p90':>15s} {'grid/p50':>8s}" + ("  DINO patch     DINO global    LPIPS          PSNR" if metrics else ""))
    for g in sorted({r["group"] for r in res.values()}) + ["all"]:
        rs = [r for r in res.values() if g in ("all", r["group"])]
        tk = [r["tokens"] for r in rs]
        line = f"{g:16s} {len(rs):3d} {np.median(tk):7.0f}/{np.percentile(tk, 90):<7.0f} {grid_tokens / np.median(tk):7.1f}x"
        if metrics:
            for m, hi in (("dino_patch", False), ("dino_global", False), ("lpips", True), ("psnr", False)):
                v = [r["vs_grid"][m] for r in rs]
                line += f"  {np.median(v):6.3f}/{np.percentile(v, 90 if hi else 10):<6.3f}"
        print(line)


if __name__ == "__main__":
    main()

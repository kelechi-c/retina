"""Tokens and judge scores at every step, not stopping at the bars: how tokens fall as quality falls, and how far the bars sit from
sparsity. Saves the decodes of a few images per step for viewing. KLT comes from the 256 px training tar at every resolution.

usage: ladder.py RUN_DIR FLOOR.json --tar images.tar [--side 256] [--only substr ...]
"""
import argparse
import glob
import json
import os
import pathlib
import tarfile
from io import BytesIO

import numpy as np
import torch
from PIL import Image

import codec as C
import judges as J
import qvae as Q

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("floor")
ap.add_argument("--tar", required=True)
ap.add_argument("--n_tar", type=int, default=1500)
ap.add_argument("--side", type=int, default=256)
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--show", nargs="*", default=["00_Composition,_1921", "41_waldo_book", "02_Rock", "07_Claudia"])
args = ap.parse_args()
dev = "cuda"
run = pathlib.Path(args.run)
(run / "png").mkdir(parents=True, exist_ok=True)
BARS = json.load(open(args.floor))[str(args.side)]
RUNGS = [.1, .2, .4, .8, 1.2, 1.6, 2.4, 3.2, 4.8, 6.4]
FILES = sorted(glob.glob("data/*/*.jpg") + glob.glob("data2/*/*.jpg"))
if args.only:
    FILES = [f for f in FILES if any(s in f for s in args.only)]


def key(f):
    return f"{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)[:-4]}"


with tarfile.open(args.tar) as t:
    mem = [m for m in t.getmembers() if m.name.lower().endswith((".jpg", ".jpeg", ".png"))]
    tar = torch.stack([torch.from_numpy(np.asarray(Image.open(BytesIO(t.extractfile(m).read())).convert("RGB")).copy()).permute(2, 0, 1) for m in mem[:: max(1, len(mem) // args.n_tar)][: args.n_tar]])
vae = Q.load_vae(dev)
zs = torch.cat([Q.encode(vae, tar[i : i + 16].to(dev).float() / 127.5 - 1).float().cpu() for i in range(0, len(tar), 16)])
judges = [J.Dino(dev), J.Dists(dev), J.DreamSim(dev)]
res = {}
for lv in (2, 3) if args.side == 256 else (4,):
    tf = C.Wavelet(list(zs), lv, dev)
    sig = tf.ev.mean().sqrt().item()
    rows = res.setdefault(f"raw_l{lv}", {})
    for f in FILES:
        k = key(f)
        x = Q.load_img(f, args.side)[None].to(dev)
        for j in judges:
            j.set_ref(x)
        c = tf.fwd(Q.encode(vae, x).float()[0].clone())
        rows[k] = []
        for d in RUNGS:
            q = C.quant(c, d * sig)
            r = (Q.decode(vae, tf.inv(q * (d * sig))[None])[0].clamp(-1, 1) * 127.5).round() / 127.5
            e = {j.name: j.score(r)["tile_worst"] for j in judges}
            e["colour"] = J.colour(r.cpu(), x.cpu())["de_max"]
            rows[k].append(dict(step=d, scores=e, over={a: e[a] - BARS[k][a]["bar"] for a in e}, psnr=(-10 * torch.log10(((r - x) / 2).pow(2).mean())).item(), **C.tokens(q.cpu(), lv)))
            if any(s in f for s in args.show):
                Image.fromarray(((r[0].cpu() + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()).save(run / "png" / f"l{lv}_{d}__{k}.png")
        json.dump(res, open(run / "ladder.json", "w"))
    print("level", lv, "done", flush=True)
print("LADDER_DONE", flush=True)

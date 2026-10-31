"""The plain codec on the Qwen VAE in tokens, at 256. Raw or std-normalized latent, KLT, CDF 9/7, one dead-zone
step, no search and no learned part. KLT is fit on a tar of 256 px training images, which must not overlap the test set.

Each image takes the largest step at which every judge stays within its frozen bar, found by a ladder and bisection with real
decodes.

usage: run_codec.py RUN_DIR FLOOR.json --tar images.tar
"""
import argparse
import glob
import json
import math
import os
import pathlib
import tarfile
import time
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
ap.add_argument("--images", type=int, default=0, help="first n only, for a smoke test")
args = ap.parse_args()
dev = "cuda"
run = pathlib.Path(args.run)
(run / "q").mkdir(parents=True, exist_ok=True)
(run / "png").mkdir(exist_ok=True)
BARS = json.load(open(args.floor))[str(args.side)]
RUNGS = [.1, .14, .2, .28, .4, .56, .8, 1.1, 1.6, 2.2, 3.2, 4.5, 6.4]
ARMS = [("raw", 2), ("raw", 3), ("std", 2), ("std", 3)]
FILES = sorted(glob.glob("data/*/*.jpg") + glob.glob("data2/*/*.jpg"))[: args.images or None]


def key(f):
    return f"{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)[:-4]}"


def load_tar(path, n):
    """Every k-th image of a tar of 256 px training images, [N,3,256,256] uint8."""
    with tarfile.open(path) as t:
        mem = [m for m in t.getmembers() if m.name.lower().endswith((".jpg", ".jpeg", ".png"))]
        out = []
        for m in mem[:: max(1, len(mem) // n)][:n]:
            a = np.asarray(Image.open(BytesIO(t.extractfile(m).read())).convert("RGB")).copy()
            out.append(torch.from_numpy(a).permute(2, 0, 1))
    return torch.stack(out)


def u8(x):
    return (x.clamp(-1, 1) * 127.5).round() / 127.5


vae = Q.load_vae(dev)
m0, s0 = (v.to(dev) for v in Q.norm_stats(vae))
t0 = time.time()
tar = load_tar(args.tar, args.n_tar)
zs = []
for i in range(0, len(tar), 16):
    zs.append(Q.encode(vae, tar[i : i + 16].to(dev).float() / 127.5 - 1).float().cpu())
zs = torch.cat(zs)
print(f"encoded {len(zs)} training images in {time.time() - t0:.0f}s, latent {zs.shape}", flush=True)
TF = {}
for norm, lv in ARMS:
    z = (zs - m0.cpu()) / s0.cpu() if norm == "std" else zs
    TF[norm, lv] = C.Wavelet(list(z), lv, dev)
    tf = TF[norm, lv]
    print(f"arm {norm} levels {lv}: top 8 of KLT energy share {(tf.ev[:8].sum() / tf.ev.sum()).item():.3f}, sigma {tf.ev.mean().sqrt().item():.3f}", flush=True)
judges = [J.Dino(dev), J.Dists(dev), J.DreamSim(dev)]
res = dict(arms={}, timing={})
ckpt = run / "codec.json"
for norm, lv in ARMS:
    tf = TF[norm, lv]
    sig = tf.ev.mean().sqrt().item()
    rows = res["arms"].setdefault(f"{norm}_l{lv}", {})
    ts, nchk = [], []
    for f in FILES:
        k = key(f)
        x = Q.load_img(f, args.side)[None].to(dev)
        bar = BARS[k]
        t = time.time()
        z = Q.encode(vae, x).float()
        for j in judges:
            j.set_ref(x)
        zin = (z - m0) / s0 if norm == "std" else z
        c = tf.fwd(zin[0].clone())
        n = [0]

        def check(d):
            n[0] += 1
            q = C.quant(c, d * sig)
            zh = tf.inv(q * (d * sig))[None]
            zh = zh * s0 + m0 if norm == "std" else zh
            r = u8(Q.decode(vae, zh)[0])
            e = {j.name: j.score(r)["tile_worst"] for j in judges}
            e["colour"] = J.colour(r.cpu(), x.cpu())["de_max"]
            return all(e[a] <= bar[a]["bar"] for a in e), e, q, r

        lo, hi, keep = None, None, None
        for d in RUNGS:
            ok, e, q, r = check(d)
            if ok:
                lo, keep = d, (e, q, r)
            else:
                hi = d
                break
        if lo is not None and hi is not None:
            for _ in range(4):
                mid = math.sqrt(lo * hi)
                ok, e, q, r = check(mid)
                if ok:
                    lo, keep = mid, (e, q, r)
                else:
                    hi = mid
        ts.append(time.time() - t), nchk.append(n[0])
        if keep is None:  # never within the bars at the finest rung: count the finest rung and flag it
            ok, e, q, r = check(RUNGS[0])
            rows[k] = dict(miss=True, step=RUNGS[0], scores=e, **C.tokens(q.cpu(), lv))
        else:
            e, q, r = keep
            rows[k] = dict(miss=False, step=lo, scores=e, **C.tokens(q.cpu(), lv))
        np.savez_compressed(run / "q" / f"{norm}_l{lv}__{k}.npz", q=q.cpu().numpy().astype(np.int16), step=rows[k]["step"], sig=sig)
        Image.fromarray(((r[0].cpu() + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()).save(run / "png" / f"{norm}_l{lv}__{k}.png")
        json.dump(res, open(ckpt, "w"))
    res["timing"][f"{norm}_l{lv}"] = dict(s_per_image=float(np.median(ts)), checks=float(np.median(nchk)))
    print(norm, lv, res["timing"][f"{norm}_l{lv}"], "misses", sum(r["miss"] for r in rows.values()), "median t16x1 blocks", float(np.median([r["t16x1"]["blocks"] for r in rows.values()])), flush=True)
    json.dump(res, open(ckpt, "w"))
print("CODEC_DONE", flush=True)

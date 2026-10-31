"""The Qwen VAE's quality floor on the test images, at 1024, 512 and 256.

recon: encode and decode every image once, keep the 8-bit PNG a viewer would get.
score JUDGE: score that PNG against the original, plus known degradations of it, so each judge's margin comes from
something measured: JPEG q90 and noise of sigma 1 and 2 of 255 on the reconstruction, a 1 px shift, the same on the original,
and an unrelated image as a ceiling. Results resume per image.

usage: floor.py recon RUN_DIR | floor.py score RUN_DIR dinov3|dists|dreamsim|colour
"""
import glob
import io
import json
import os
import pathlib
import statistics as st
import sys
import time

import numpy as np
import torch
from PIL import Image

import judges as J
import qvae as Q

SIDES = [int(v) for v in os.environ.get("SCORE_SIDES", "1024,512,256").split(",")]
RECON_SIDES = [int(v) for v in os.environ.get("RECON_SIDES", "1024,512,256").split(",")]
dev = "cuda"
mode, run = sys.argv[1:3]
run = pathlib.Path(run)
FILES = sorted(glob.glob("data/*/*.jpg") + glob.glob("data2/*/*.jpg"))


def key(f):
    return f"{os.path.basename(os.path.dirname(f))}__{os.path.basename(f)[:-4]}"


def rec_path(f, side):
    return run / "recon" / str(side) / f"{key(f)}.png"


def to_u8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


def from_u8(a):
    return torch.from_numpy(np.asarray(a).copy()).permute(2, 0, 1).float() / 127.5 - 1


def jpeg(x, q=90):
    b = io.BytesIO()
    Image.fromarray(to_u8(x)).save(b, "JPEG", quality=q, subsampling=0)
    return from_u8(Image.open(b).convert("RGB"))


def noise(x, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    return from_u8(to_u8((x + torch.randn(x.shape, generator=g) * sigma / 127.5).clamp(-1, 1)))


def recon():
    vae = Q.load_vae(dev)
    log = {}
    for side in RECON_SIDES:
        (run / "recon" / str(side)).mkdir(parents=True, exist_ok=True)
        t0, ts = time.time(), []
        for f in FILES:
            p = rec_path(f, side)
            if p.exists():
                continue
            x = Q.load_img(f, side)[None].to(dev)
            torch.cuda.synchronize(); t = time.time()
            r, a = Q.decode(vae, Q.encode(vae, x))
            torch.cuda.synchronize(); ts.append(time.time() - t)
            Image.fromarray(to_u8(r[0])).save(p)
            log.setdefault(str(side), {})[key(f)] = dict(alpha_min=a.min().item(), alpha_mean=a.mean().item())
        print(f"recon {side}: {len(ts)} images, {time.time() - t0:.0f}s, median encode+decode {st.median(ts) if ts else 0:.3f}s", flush=True)
        json.dump(log, open(run / "recon_log.json", "w"))
    print("RECON_DONE", flush=True)


def score(name):
    if name != "colour":
        m = dict(dinov3=J.Dino, dists=J.Dists)[name](dev) if name != "dreamsim" else J.DreamSim(dev)
    pth = run / f"scores_{name}.json"
    res = json.load(open(pth)) if pth.exists() else dict(scores={}, timing={})
    for side in SIDES:
        S = res["scores"].setdefault(str(side), {})
        tr, ts = [], []
        for i, f in enumerate(FILES):
            if key(f) in S:
                continue
            x = Q.load_img(f, side)
            r = from_u8(Image.open(rec_path(f, side)).convert("RGB"))
            other = from_u8(Image.open(rec_path(FILES[(i + len(FILES) // 2) % len(FILES)], side)).convert("RGB"))
            V = dict(vae=r, vae_jpeg90=jpeg(r), vae_noise1=noise(r, 1, i), vae_noise2=noise(r, 2, i), vae_shift1=torch.roll(r, (1, 1), (1, 2)), orig_jpeg90=jpeg(x), orig_noise2=noise(x, 2, i), unrelated=other)
            row = {}
            if name == "colour":
                row = {k: J.colour(v[None], x[None]) for k, v in V.items()}
            else:
                torch.cuda.synchronize(); t = time.time()
                m.set_ref(x[None])
                torch.cuda.synchronize(); tr.append(time.time() - t)
                for k, v in V.items():
                    torch.cuda.synchronize(); t = time.time()
                    row[k] = m.score(v[None])
                    torch.cuda.synchronize(); ts.append(time.time() - t)
            S[key(f)] = row
            json.dump(res, open(pth, "w"))
        if ts:
            res["timing"][str(side)] = dict(set_ref_s=st.median(tr), score_s=st.median(ts), gb=torch.cuda.max_memory_allocated() / 2 ** 30)
            print(name, side, res["timing"][str(side)], flush=True)
            json.dump(res, open(pth, "w"))
    print("SCORE_DONE", name, flush=True)


recon() if mode == "recon" else score(sys.argv[3])

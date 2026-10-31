"""Qwen VAE checks on this machine. Shapes and token counts per resolution, alpha, one-frame decode, latent
normalization round trip, time and peak memory, all in bfloat16.

usage: vae_check.py OUT.json
"""
import glob
import json
import math
import sys
import time

import torch

import qvae as Q

dev = "cuda"
paths = {g: sorted(glob.glob(f"data*/{g}/*.jpg")) for g in ("mondrian", "waldo")}
imgs = dict(mondrian=paths["mondrian"][0], page=[p for p in paths["waldo"] if "waldo_book_page" in p][0])
out = dict(rows=[])


def run(vae, name, x, side, label):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.time()
    z = Q.encode(vae, x)
    torch.cuda.synchronize(); te = time.time() - t
    me = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t = time.time()
    r, a = Q.decode(vae, z)
    torch.cuda.synchronize(); td = time.time() - t
    md = torch.cuda.max_memory_allocated() / 2 ** 30
    psnr = -10 * math.log10(((r - x) / 2).pow(2).mean().item())
    row = dict(image=name, side=side, dtype=label, latent=list(z.shape), floats=z[0].numel(), tokens16=z[0].numel() // 16, enc_s=te, dec_s=td, enc_gb=me, dec_gb=md, psnr=psnr,
               alpha_min=a.min().item(), alpha_mean=a.mean().item(), alpha_max=a.max().item(), z_std=z.float().std().item(), z_abs=z.float().abs().max().item())
    print(row, flush=True)
    out["rows"].append(row)
    return z, r


for label, dt in (("bf16", torch.bfloat16),):
    vae = Q.load_vae(dev, dt)
    if label == "bf16":
        n = sum(p.numel() for p in vae.encoder.parameters()) / 1e6, sum(p.numel() for p in vae.decoder.parameters()) / 1e6
        print("params M enc, dec", n, flush=True)
        out["params_m"] = n
    keep = {}
    for side in (256, 512, 768, 1024):
        for name, p in imgs.items():
            x = Q.load_img(p, side)[None].to(dev)
            for rep in range(2):  # the first run pays for warmup
                try:
                    z, r = run(vae, name, x, side, label)
                except torch.OutOfMemoryError as e:
                    print("OOM", name, side, label, str(e)[:120], flush=True)
                    out["rows"].append(dict(image=name, side=side, dtype=label, oom=True))
                    torch.cuda.empty_cache()
                    break
            else:
                keep[(name, side)] = (z.float().cpu(), r.cpu())
    z = keep[max(keep, key=lambda k: k[1])][0]
    m, s = Q.norm_stats(vae)
    rt = ((z - m) / s) * s + m
    out["norm_roundtrip_max"] = (rt - z).abs().max().item()
    out["z_channel_std_range"] = [z.std((0, 2, 3)).min().item(), z.std((0, 2, 3)).max().item()]
    print("norm round trip max err", out["norm_roundtrip_max"], "channel std range", out["z_channel_std_range"], flush=True)
    del vae
    torch.cuda.empty_cache()
    json.dump(out, open(sys.argv[1], "w"))
print("PHASE0_DONE", flush=True)

"""Freeze the floor. bar = this image's plain Qwen VAE score + one margin per judge and resolution, on the worst 256 tile.
Primary margin: median over the images of (JPEG q90 of the VAE reconstruction - the reconstruction), which is noisy per image,
so it is not used per image, and clamped at 0 so the plain VAE always meets its own bar. Second set, for sensitivity only: median JPEG q90 of the original against the original. Colour: Delta E
max at 1/16 scale, margin 1.0, roughly one just-noticeable difference.

usage: freeze_floor.py RUN_DIR
"""
import hashlib
import json
import statistics as st
import sys

run = sys.argv[1]
S = {j: json.load(open(f"{run}/scores_{j}.json"))["scores"] for j in ("dinov3", "dists", "dreamsim", "colour")}
for name, mfn in (("floor", lambda r, m: r["vae_jpeg90"][m] - r["vae"][m]), ("floor_generous", lambda r, m: r["orig_jpeg90"][m])):
    F, M = {}, {}
    for side in ("1024", "512", "256"):
        M[side] = {j: max(0, st.median(mfn(r, "tile_worst") for r in S[j][side].values())) for j in ("dinov3", "dists", "dreamsim")}
        M[side]["colour"] = 1.0
        F[side] = {k: {j: dict(floor=S[j][side][k]["vae"]["de_max" if j == "colour" else "tile_worst"], margin=M[side][j], bar=S[j][side][k]["vae"]["de_max" if j == "colour" else "tile_worst"] + M[side][j]) for j in S} for k in S["dinov3"][side]}
    p = f"{run}/{name}.json"
    json.dump(F, open(p, "w"))
    print(name, {s: {j: round(v, 4) for j, v in m.items()} for s, m in M.items()}, "sha256", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])

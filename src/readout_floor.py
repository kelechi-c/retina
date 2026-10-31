"""Floor readout: the Qwen VAE floor per judge and group, and what each known degradation adds to it.

usage: readout_floor.py RUN_DIR
"""
import json
import statistics as st
import sys

run = sys.argv[1]
J = dict(dinov3="tile_worst", dists="tile_worst", dreamsim="tile_worst")
S = {j: json.load(open(f"{run}/scores_{j}.json")) for j in (*J, "colour")}
for j in S:
    print(j, "timing", {k: {a: round(b, 3) for a, b in v.items()} for k, v in S[j]["timing"].items()})
for side in ("1024", "512", "256"):
    print(f"\n=== {side} ===")
    for j, m in J.items():
        T = S[j]["scores"][side]
        print(f"{j} {m}, medians: " + "  ".join(f"{v} {st.median(r[v][m] for r in T.values()):.4f}" for v in next(iter(T.values()))))
        G = {}
        for k, r in T.items():
            G.setdefault(k.split("__")[0], []).append(r)
        print("  floor by group: " + "  ".join(f"{g} {st.median(r['vae'][m] for r in v):.3f} [{min(r['vae'][m] for r in v):.3f},{max(r['vae'][m] for r in v):.3f}]" for g, v in sorted(G.items())))
        for v in ("vae_jpeg90", "vae_noise1", "vae_noise2"):
            d = sorted(r[v][m] - r["vae"][m] for r in T.values())
            print(f"  {v} minus floor: median {st.median(d):+.4f}, range {d[0]:+.4f} {d[-1]:+.4f}, negative on {sum(x < 0 for x in d)}")
    T = S["colour"]["scores"][side]
    print("colour vae: psnr median %.1f min %.1f; dE mean median %.2f, dE max median %.2f worst %.2f; jpeg90 dE max median %.2f" % (st.median(r["vae"]["psnr"] for r in T.values()), min(r["vae"]["psnr"] for r in T.values()), st.median(r["vae"]["de_mean"] for r in T.values()), st.median(r["vae"]["de_max"] for r in T.values()), max(r["vae"]["de_max"] for r in T.values()), st.median(r["vae_jpeg90"]["de_max"] for r in T.values())))
T = S["dinov3"]["scores"]["1024"]
print("\nworst floors at 1024:")
for j, m in J.items():
    T = S[j]["scores"]["1024"]
    print(" ", j, [(k.split('__')[1][:18], round(r["vae"][m], 3)) for k, r in sorted(T.items(), key=lambda kv: -kv[1]["vae"][m])[:5]])

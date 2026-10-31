"""Codec readout: tokens per image at 256 by arm and block shape, group medians, page/Mondrian, rank against complexity tiers.

usage: readout_codec.py RUN_DIR
"""
import json
import random
import statistics as st
import sys

TIERS = dict(mondrian=2, simple=2, ornate=3, texture=4, dense=5, waldo=5)
OVER = {"22_Clear": 1, "29_mj_": 3, "10_Rough": 3, "13_Pebbles": 3}
run = sys.argv[1]
R = json.load(open(f"{run}/codec.json"))


def ranks(v):
    s = sorted(v)
    return [s.index(x) + (s.count(x) - 1) / 2 for x in v]


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5


print("timing", R["timing"])
for arm, rows in R["arms"].items():
    if len(rows) < 52:
        print(arm, "partial", len(rows))
    for shape in ("t16x1", "t4x2"):
        for count in ("blocks", "length"):
            G = {}
            for k, r in rows.items():
                G.setdefault(k.split("__")[0], []).append(r[shape][count])
            M = G["mondrian"]
            page = next((r[shape][count] for k, r in rows.items() if "41_waldo_book" in k), None)
            random.seed(0)
            b = sorted(page / st.median(random.choices(M, k=len(M))) for _ in range(4000)) if page else [0]
            tier = [next((t for p, t in OVER.items() if k.split("__")[1].startswith(p)), TIERS[k.split("__")[0]]) for k in rows]
            print(f"{arm} {shape} {count}: " + " ".join(f"{g} {st.median(v):.0f}" for g, v in sorted(G.items())) + f" | page {page} page/M {page / st.median(M):.1f} [{b[100]:.1f},{b[3899]:.1f}], clear 16x {sum(page / max(v, 1) >= 16 for v in M)}/{len(M)}; Spearman {spearman(tier, [r[shape][count] for r in rows.values()]):.2f}")
    print(f"  misses {[k.split('__')[1][:20] for k, r in rows.items() if r['miss']]}; median nonzero {st.median(r['nonzero'] for r in rows.values())}")

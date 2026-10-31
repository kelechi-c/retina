"""Ladder readout: median tokens and judge excess over the frozen bars per step and group.

usage: readout_ladder.py RUN_DIR
"""
import json
import statistics as st
import sys

R = json.load(open(f"{sys.argv[1]}/ladder.json"))
for arm, rows in R.items():
    print(f"== {arm}, {len(rows)} images")
    n = len(next(iter(rows.values())))
    for i in range(n):
        G = {}
        for k, v in rows.items():
            G.setdefault(k.split("__")[0], []).append(v[i])
        page = next((v[i] for k, v in rows.items() if "41_waldo_book" in k), None)
        M = G.get("mondrian")
        s = f"step {rows[next(iter(rows))][i]['step']:4}: t16x1 " + " ".join(f"{g[:4]} {st.median(a['t16x1']['blocks'] for a in v):.0f}" for g, v in sorted(G.items()))
        if page and M:
            m = st.median(a["t16x1"]["blocks"] for a in M)
            s += f" | page/M {page['t16x1']['blocks'] / max(m, 1):.1f}, t4x2 {page['t4x2']['blocks'] / max(st.median(a['t4x2']['blocks'] for a in M), 1):.1f}, length {page['t16x1']['length'] / max(st.median(a['t16x1']['length'] for a in M), 1):.1f}"
        print(s)
        A = [v[i] for v in rows.values()]
        print("    median over bar " + " ".join(f"{j} {st.median(a['over'][j] for a in A):+.3f}" for j in ("dinov3", "dists", "dreamsim", "colour")) + f"; pass all {sum(all(x <= 0 for x in a['over'].values()) for a in A)}/{len(A)}; PSNR {st.median(a['psnr'] for a in A):.1f}")

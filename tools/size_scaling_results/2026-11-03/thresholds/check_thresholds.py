"""Check the four 768-tied thresholds at other sides, with a GPU A/B where the sweep cannot answer.

  200-token warp cutoff     -> warp on vs off at 512/768/1024 around the cutoff
  600-token refiner limit   -> S1 stat refiner on vs off at 512/768/1024, low budgets
  22 dB floor               -> data-only (sweep already has native PSNR per grid)
  6-pixel filter window     -> data-only (BUDGET_SIDE reads texture at 768; check 1 of the run)

usage: python tools/check_thresholds.py OUT --images DIR [--per-group 3]
"""
import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXT = {".jpg", ".jpeg", ".png", ".webp"}

SIDES = (512, 768, 1024)
WARP_BUDGETS = {512: (64, 89, 128, 200, 256), 768: (64, 128, 200, 256, 400, 600), 1024: (128, 200, 356, 512, 800, 1200)}
S1_BUDGETS = {512: (128, 256, 512), 768: (128, 256, 600), 1024: (256, 600, 1000)}


def cells(tok, tokens):
    """Token budget -> cells per side, the unit Tokenizer.encode expects."""
    n = int(round(tokens ** .5)) * tok.v.pack
    return max(tok.v.pack, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--images", required=True)
    p.add_argument("--per-group", type=int, default=3)
    p.add_argument("--dev", default="cuda")
    a = p.parse_args()
    dev = a.dev if torch.cuda.is_available() else "cpu"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from retina import common as C, load_img
    from retina.tokenizer import LEARNED_MAX, WARP_MAX, Tokenizer

    root = pathlib.Path(a.images)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    per = {}
    picks = []
    for f in files:
        g = f.relative_to(root).parts[0] if len(f.relative_to(root).parts) > 1 else "images"
        if per.get(g, 0) < a.per_group:
            per[g] = per.get(g, 0) + 1
            picks.append((g, f))
    print("images", len(picks), dict(per), flush=True)

    plain = Tokenizer("flux2", dev, learned=None, floor=0.)
    s1 = Tokenizer("flux2", dev, floor=0.)
    s1.learned_max = 10 ** 9  # force the refiner on at every tested budget
    s1.s1_min = 0.
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    @torch.no_grad()
    def score(y, x):
        a8, b8 = C.to8(y), C.to8(x)
        psnr = float(-10 * torch.log10(((a8 - b8) / 2).pow(2).mean().clamp_min(1e-12)))
        return dict(psnr=psnr, lpips=float(lp(a8, b8).mean()))

    rows = []
    for i, (g, f) in enumerate(picks):
        for S in SIDES:
            x = load_img(str(f), S).to(dev)
            for nt in WARP_BUDGETS[S]:
                n = cells(plain, nt)
                rec = dict(id=f"file:{f.relative_to(root)}", group=g, side=S, tokens=(n // plain.v.pack) ** 2)
                for tag, wm in (("nowarp", 0), ("warp", 10 ** 9)):
                    plain.warp_max = wm
                    t = plain.encode(x, n=n)
                    rec[tag] = score(plain.decode(t), x)
                rows.append(rec)
            for nt in S1_BUDGETS[S]:
                n = cells(plain, nt)
                t = plain.encode(x, n=n)
                y = plain.decode(t)
                t1 = s1.encode(x, n=n)
                y1 = s1.decode(t1)  # uses the refiner when the encode attached stats
                rows.append(dict(id=f"file:{f.relative_to(root)}", group=g, side=S, tokens=(n // plain.v.pack) ** 2,
                                 refiner=score(y1, x), plaine=score(y, x), has_stats="stats" in t1))
        print(f"[{i + 1}/{len(picks)}] {f.name}", flush=True)
    json.dump(dict(rows=rows, warp_max=WARP_MAX, learned_max=LEARNED_MAX, sides=SIDES,
                   warp_budgets=WARP_BUDGETS, s1_budgets=S1_BUDGETS), open(out / "thresholds.json", "w"))

    # compact medians: gain = plain - variant (positive = variant better on that metric)
    def med(v):
        v = sorted(v)
        return v[len(v) // 2] if v else float("nan")

    lines = ["# 768-tied thresholds at other sides", ""]
    lines.append("## Warp on vs off (delta LPIPS plain-warp minus warp; positive = warp better), and PSNR")
    lines.append("| side | tokens | n | dLPIPS | dPSNR dB |")
    lines.append("|---|---|---|---|---|")
    for S in SIDES:
        seen = sorted({r["tokens"] for r in rows if "nowarp" in r and r["side"] == S})
        for nt in seen:
            v = [r for r in rows if "nowarp" in r and r["side"] == S and r["tokens"] == nt]
            if not v:
                continue
            dl = med([r["nowarp"]["lpips"] - r["warp"]["lpips"] for r in v])
            dp = med([r["warp"]["psnr"] - r["nowarp"]["psnr"] for r in v])
            lines.append(f"| {S} | {nt} | {len(v)} | {dl:+.4f} | {dp:+.2f} |")
    lines.append("")
    lines.append("## S1 refiner on vs off (delta LPIPS plain minus refiner; positive = refiner better)")
    lines.append("| side | tokens | n | dLPIPS | dPSNR dB | stats attached |")
    lines.append("|---|---|---|---|---|---|")
    for S in SIDES:
        seen = sorted({r["tokens"] for r in rows if "refiner" in r and r["side"] == S})
        for nt in seen:
            v = [r for r in rows if "refiner" in r and r["side"] == S and r["tokens"] == nt]
            if not v:
                continue
            dl = med([r["plaine"]["lpips"] - r["refiner"]["lpips"] for r in v])
            dp = med([r["refiner"]["psnr"] - r["plaine"]["psnr"] for r in v])
            lines.append(f"| {S} | {nt} | {len(v)} | {dl:+.4f} | {dp:+.2f} | {sum(r['has_stats'] for r in v)}/{len(v)} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print("THRESHOLDS_DONE", flush=True)


if __name__ == "__main__":
    main()

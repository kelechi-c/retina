"""Measure how the token budget should grow with image size, and fit a size term for it. Needs a CUDA GPU and the FLUX.2-dev VAE; read tools/size_scaling.md first.

The budget curve was fit at 768 px and returns the same token count at every size. For each image and each side S this script asks:
how few latent tokens does a decode at S need to be as good as the shipped 768 budget's decode is at 768?
It then fits, on training images only,

    log(N_S / N_768) = (g0 + g1 * t + g2 * d) * log(S / 768)

where d is the budget's DReX and t is 10 x its texture energy, both read from a 768 px copy by Tokenizer.budget. At 768 the term is zero, so the shipped behaviour there does not change. "As good" is judged two ways, by LPIPS and by PSNR, and two views: at the image's own size (native) and with both images resized to 768 (display, as if shown at the same size on screen).

usage:
    uv run python tools/fit_size_scaling.py OUT --images DIR [--procedural 40] [--wikiart 160] [--sizes 512,768,1024]
    uv run python tools/fit_size_scaling.py OUT --fit-only        # refit from OUT's measurements, no GPU
"""
import csv
import hashlib
import json
import math
import pathlib
import random
import sys
import time

import click
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXT = {".jpg", ".jpeg", ".png", ".webp"}
EXAMPLES = ["simple/mondrian_piet", "simple/mondrian_b144", "busy/bruegel", "busy/veronese"]
HIGHER_BETTER = {"psnr": True, "lpips": False}


# ---------- images ----------

def folder_images(root, need):
    from PIL import Image, ImageOps

    root = pathlib.Path(root)
    for f in sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT):
        try:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception as e:
            print("skip, unreadable:", f, repr(e)[:80], flush=True)
            continue
        if min(im.size) < need:
            print(f"skip, short side {min(im.size)} < {need}:", f, flush=True)
            continue
        rel = f.relative_to(root)
        yield dict(id=f"file:{rel}", source="folder", group=rel.parts[0] if len(rel.parts) > 1 else "folder"), im


def procedural_images(n, need):
    from mar import data as D

    for i in range(n):
        yield dict(id=f"proc:{i}", source="procedural", group="procedural"), D.procedural(10_000 + i, need)


def wikiart_images(n, need):
    """A spread of the four MAR labels, each image at least `need` px on its short side."""
    from mar import data as D
    from mar.model import NAMES

    quota, got = -(-n // 4), [0, 0, 0, 0]
    shards = random.Random(0).sample(range(72), 72)
    for key, _, label, raw in D.wikiart(D.STYLES, D.GENRES, shards, [1, 1, 1, 1], salt="size"):
        if got[label] >= quota:
            if min(got) >= quota:
                return
            continue
        try:
            im = D.opened(raw)
        except Exception:
            continue
        if min(im.size) < need:
            continue
        got[label] += 1
        yield dict(id=f"wikiart:{key}", source="wikiart", group=f"wikiart-{NAMES[label]}"), im


def held_out(i):
    return int(hashlib.md5(i.encode()).hexdigest(), 16) % 5 == 0  # one image in five, fixed by its id


# ---------- measuring ----------

class Meter:
    def __init__(self, tok, metrics, dev):
        self.tok, self.metrics, self.dev = tok, metrics, dev
        if "lpips" in metrics:
            import lpips

            self.lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    @torch.no_grad()
    def decode_at(self, x, n):
        """The shipped path at n cells per side, with no floor and no refiner: warp below the warp cutoff, as encode does."""
        tok, S = self.tok, x.shape[-1]
        warped = (n // tok.v.pack) ** 2 < tok.warp_max
        return tok.decode(tok._pack(*tok._code(x, n, warped), n, S)), warped

    @torch.no_grad()
    def score(self, y, x):
        """Both images rounded to 8 bit, as saved; native at their own size, display after resizing both to 768."""
        from retina import common as C

        out = {}
        for view in ("native", "display"):
            a, b = (y, x) if view == "native" or x.shape[-1] == 768 else (C.down(y, 768).clamp(-1, 1), C.down(x, 768).clamp(-1, 1))
            a, b = C.to8(a), C.to8(b)
            if "psnr" in self.metrics:
                out[f"psnr_{view}"] = float(-10 * torch.log10(((a - b) / 2).pow(2).mean().clamp_min(1e-12)))
            if "lpips" in self.metrics:
                out[f"lpips_{view}"] = float(self.lp(a, b).mean())
        return out


def grid_list(gmax, ratio):
    g, v = {2, gmax}, 2.0
    while v < gmax:
        g.add(int(round(v)))
        v *= ratio
    return sorted(x for x in g if 2 <= x <= gmax)


def needed(tokens, q, target, higher):
    """Fewest tokens whose decode reaches the target, on the envelope of the sweep (the best quality reachable with at most that many tokens).
    Interpolated in log tokens between the last grid that misses and the first that reaches. Status 'below': even the smallest grid reaches it; 'above': no grid does."""
    t, q = np.asarray(tokens, float), np.asarray(q, float)
    env = np.maximum.accumulate(q) if higher else np.minimum.accumulate(q)
    ok = env >= target if higher else env <= target
    if not ok.any():
        return float(t[-1]), "above"
    i = int(np.argmax(ok))
    if i == 0:
        return float(t[0]), "below"
    f = (target - env[i - 1]) / (env[i] - env[i - 1])
    return float(math.exp(math.log(t[i - 1]) + f * (math.log(t[i]) - math.log(t[i - 1])))), "ok"


def measure(out, tok, meter, sources, sizes, ratio, limit):
    from mar import data as D

    done = set()
    if (out / "images.jsonl").exists():
        done = {json.loads(l)["id"] for l in open(out / "images.jsonl")}
        print("resuming:", len(done), "images already measured", flush=True)
    keys = [f"{m}_{v}" for m in meter.metrics for v in ("native", "display")]
    pk = tok.v.cell * tok.v.pack
    t0, k = time.time(), 0
    for meta, im in sources:
        if limit and len(done) >= limit:
            break
        if meta["id"] in done:
            continue
        ti = time.time()
        x768 = D.to_tensor(im, 768).to(meter.dev)
        b, d, tex = tok.budget(x768)
        n768 = tok.size_of(b, 768)
        y, _ = meter.decode_at(x768, n768)
        ref = meter.score(y, x768)
        rec = dict(meta, held_out=held_out(meta["id"]), d=d, tex=tex, t=10 * tex, budget=b, n768=n768, tokens768=(n768 // tok.v.pack) ** 2, ref=ref, sizes={})
        rows = []
        for S in sizes:
            x = x768 if S == 768 else D.to_tensor(im, S).to(meter.dev)
            bS, dS, texS = tok.budget(x)  # read from a 768 px copy: should match d and tex closely
            gs = grid_list(S // pk, ratio)
            if S == 768 and n768 // tok.v.pack not in gs:
                gs = sorted(set(gs) | {n768 // tok.v.pack})
            sweep = []
            for g in gs:
                y, warped = meter.decode_at(x, g * tok.v.pack)
                s = meter.score(y, x)
                sweep.append(dict(g=g, tokens=g * g, warped=warped, **s))
                rows.append(dict(id=meta["id"], side=S, **sweep[-1]))
            need = {}
            for key in keys:
                m = key.split("_")[0]
                need[key] = needed([r["tokens"] for r in sweep], [r[key] for r in sweep], ref[key], HIGHER_BETTER[m])
            rec["sizes"][str(S)] = dict(d=dS, tex=texS, budget=bS, grid_max=(S // pk) ** 2, need={k_: dict(tokens=v[0], status=v[1]) for k_, v in need.items()})
        with open(out / "sweep.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(out / "images.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.add(meta["id"])
        k += 1
        n_ok = sum(v["status"] == "ok" for s in rec["sizes"].values() for v in s["need"].values())
        print(f"[{len(done)}] {meta['id'][:48]:48s} d {d:.3f} tex {tex:.4f} budget {b:6.0f} -> {rec['tokens768']:4d} tokens at 768 | {time.time() - ti:5.1f} s | {n_ok} of {len(sizes) * len(keys)} uncensored | {(time.time() - t0) / k:.1f} s/image", flush=True)


# ---------- fitting ----------

def fit(out, sizes):
    recs = [json.loads(l) for l in open(out / "images.jsonl")]
    keys = sorted({k for r in recs for k in r["sizes"]["768"]["need"]})
    others = [S for S in sizes if S != 768]
    res = dict(n_images=len(recs), n_train=sum(not r["held_out"] for r in recs), sizes=sizes, model="log(N_S/N_768) = (g0 + g1*t + g2*d) * log(S/768)", fits={})
    per_image = []
    for key in keys:
        X, Y, Xh, Yh, Sh, cens = [], [], [], [], [], {"below": 0, "above": 0, "ok": 0}
        for r in recs:
            n0 = r["sizes"]["768"]["need"][key]
            for S in others:
                if str(S) not in r["sizes"]:
                    continue
                nS = r["sizes"][str(S)]["need"][key]
                cens[nS["status"]] += 1
                if n0["status"] != "ok" or nS["status"] != "ok":
                    continue
                x = math.log(S / 768)
                y = math.log(nS["tokens"] / n0["tokens"])
                feats = [x, r["t"] * x, r["d"] * x]
                (Xh if r["held_out"] else X).append(feats)
                (Yh if r["held_out"] else Y).append(y)
                if r["held_out"]:
                    Sh.append(S)
                per_image.append(dict(id=r["id"], group=r["group"], held_out=r["held_out"], key=key, side=S, d=r["d"], t=r["t"], tokens768=n0["tokens"], tokensS=nS["tokens"], gamma=y / x))
        X, Y, Xh, Yh = map(np.asarray, (X, Y, Xh, Yh))
        if len(Y) < 4:
            res["fits"][key] = dict(error=f"only {len(Y)} usable training points", censored=cens)
            continue
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        glob = float((X[:, 0] * Y).sum() / (X[:, 0] ** 2).sum())
        pred = X @ coef
        r2 = 1 - ((Y - pred) ** 2).sum() / max(((Y - Y.mean()) ** 2).sum(), 1e-12)
        val = {}
        if len(Yh):
            for name, p in (("no scaling", np.zeros_like(Yh)), ("area (gamma 2)", 2 * Xh[:, 0]), ("one gamma", glob * Xh[:, 0]), ("fitted", Xh @ coef)):
                e = np.abs(p - Yh)
                val[name] = dict(median_abs_log_error=float(np.median(e)), p90=float(np.percentile(e, 90)),
                                 by_side={str(S): float(np.median(e[np.asarray(Sh) == S])) for S in others if (np.asarray(Sh) == S).any()})
        res["fits"][key] = dict(g0=float(coef[0]), g1=float(coef[1]), g2=float(coef[2]), one_gamma=glob, r2_train=float(r2), n_train_points=int(len(Y)), n_held_points=int(len(Yh)), censored=cens, held_out=val)
    # what the fits imply for the example paintings
    ex = json.load(open(out / "examples.json")) if (out / "examples.json").exists() else []
    for e in ex:
        e["implied"] = {}
        for key, f in res["fits"].items():
            if "g0" not in f:
                continue
            g = f["g0"] + f["g1"] * e["t"] + f["g2"] * e["d"]
            e["implied"][key] = {str(S): dict(gamma=g, budget=e["budget"] * (S / 768) ** g, cap=(S // 16) ** 2) for S in sizes}
    res["examples"] = ex
    usable = [k for k in ("lpips_native", "psnr_native", "lpips_display", "psnr_display") if "g0" in res["fits"].get(k, {})]
    res["primary"] = usable[0] if usable else None
    json.dump(res, open(out / "fit.json", "w"), indent=1)
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group", "held_out", "key", "side", "d", "t", "tokens768", "tokensS", "gamma"])
        w.writeheader()
        w.writerows(per_image)
    report(out, res, per_image, recs)
    return res


def report(out, res, per_image, recs):
    L = [f"# Size scaling of the token budget\n", f"{res['n_images']} images ({res['n_train']} training, {res['n_images'] - res['n_train']} held out), sides {', '.join(map(str, res['sizes']))}. Model: `{res['model']}`.\n"]
    # step 1 check: the budget's inputs at each side against 768
    L.append("## Budget inputs read at each side (should match 768)\n\n| side | median abs change in d | median abs change in texture |\n|---|---|---|")
    for S in res["sizes"]:
        dd = [abs(r["sizes"][str(S)]["d"] - r["d"]) for r in recs if str(S) in r["sizes"]]
        dt = [abs(r["sizes"][str(S)]["tex"] - r["tex"]) for r in recs if str(S) in r["sizes"]]
        L.append(f"| {S} | {np.median(dd):.4f} | {np.median(dt):.5f} |")
    sanity = [r["sizes"]["768"]["need"] for r in recs]
    L.append("\n## Sanity at 768: measured tokens against the shipped grid\n\n| criterion | median measured / shipped | share censored |\n|---|---|---|")
    for key in sorted(res["fits"]):
        ratio = [n[key]["tokens"] / r["tokens768"] for n, r in zip(sanity, recs) if n[key]["status"] == "ok"]
        cz = np.mean([n[key]["status"] != "ok" for n in sanity])
        L.append(f"| {key} | {np.median(ratio) if ratio else float('nan'):.3f} | {cz:.0%} |")
    L.append("\n## Fits\n\n| criterion | g0 | g1 (t) | g2 (d) | one gamma | R² train | points | censored below / above |\n|---|---|---|---|---|---|---|---|")
    for key, f in sorted(res["fits"].items()):
        if "g0" not in f:
            L.append(f"| {key} | {f['error']} | | | | | | {f['censored']['below']} / {f['censored']['above']} |")
            continue
        L.append(f"| {key} | {f['g0']:.3f} | {f['g1']:.3f} | {f['g2']:.3f} | {f['one_gamma']:.3f} | {f['r2_train']:.3f} | {f['n_train_points']} | {f['censored']['below']} / {f['censored']['above']} |")
    L.append("\n## Held out: median absolute error in log tokens (lower is better)\n\n| criterion | no scaling | area (gamma 2) | one gamma | fitted |\n|---|---|---|---|---|")
    for key, f in sorted(res["fits"].items()):
        v = f.get("held_out") or {}
        if v:
            L.append(f"| {key} | " + " | ".join(f"{v[n]['median_abs_log_error']:.3f}" for n in ("no scaling", "area (gamma 2)", "one gamma", "fitted")) + " |")
    L.append("\n## Gamma by group (median per image, training and held out)\n\n| criterion | group | side | n | median gamma | p10 | p90 |\n|---|---|---|---|---|---|---|")
    groups = sorted({(p["key"], p["group"], p["side"]) for p in per_image})
    for key, g, S in groups:
        v = [p["gamma"] for p in per_image if (p["key"], p["group"], p["side"]) == (key, g, S)]
        L.append(f"| {key} | {g} | {S} | {len(v)} | {np.median(v):.2f} | {np.percentile(v, 10):.2f} | {np.percentile(v, 90):.2f} |")
    if res["examples"] and res["primary"]:
        L.append(f"\n## Example paintings under {res['primary']}\n\n| image | d | t | gamma | " + " | ".join(f"budget at {S}" for S in res["sizes"]) + " |\n|---|---|---|---|" + "---|" * len(res["sizes"]))
        for e in res["examples"]:
            im = e["implied"].get(res["primary"], {})
            if im:
                g = next(iter(im.values()))["gamma"]
                L.append(f"| {e['name']} | {e['d']:.3f} | {e['t']:.3f} | {g:.2f} | " + " | ".join(f"{im[str(S)]['budget']:.0f} (cap {im[str(S)]['cap']})" for S in res["sizes"]) + " |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


# ---------- main ----------

@click.command()
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--images", "images_dir", default="", help="folder of images, searched recursively; the first folder level names the group")
@click.option("--procedural", default=0, show_default=True, help="Mondrian-style flats from mar.data")
@click.option("--wikiart", default=0, show_default=True, help="WikiArt images spread over the four MAR labels (downloads parquet shards)")
@click.option("--sizes", default="512,768,1024", show_default=True, help="sides to measure; multiples of 256, must include 768")
@click.option("--metrics", default="lpips,psnr", show_default=True)
@click.option("--ratio", default=1.12, show_default=True, help="step between swept grid sides; smaller is finer and slower")
@click.option("--limit", default=0, show_default=True, help="stop after this many images (0: all)")
@click.option("--vae", default="flux2", show_default=True)
@click.option("--drext", default="default", show_default=True, help="'default' for the shipped text-tuned DReX, 'none' for the original")
@click.option("--dev", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--fit-only", is_flag=True, help="refit from OUT/images.jsonl without measuring")
def main(out, images_dir, procedural, wikiart, sizes, metrics, ratio, limit, vae, drext, dev, fit_only):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({int(s) for s in sizes.split(",")})
    if 768 not in sizes or any(s % 256 for s in sizes):
        raise click.UsageError("--sizes must be multiples of 256 and include 768")
    if fit_only:
        fit(out, sizes)
        return
    if not (images_dir or procedural or wikiart):
        raise click.UsageError("give images: --images DIR, --procedural N and/or --wikiart N")
    import itertools

    from retina import load_img
    from retina.tokenizer import DREXT, Tokenizer

    metrics = [m for m in metrics.split(",") if m]
    tok = Tokenizer(vae, dev, learned=None, floor=0., drext=None if drext == "none" else DREXT if drext == "default" else drext)
    meter = Meter(tok, metrics, dev)
    json.dump(dict(sizes=sizes, metrics=metrics, ratio=ratio, vae=vae, drext=drext, images=images_dir, procedural=procedural, wikiart=wikiart), open(out / "settings.json", "w"), indent=1)
    ex = []
    for e in EXAMPLES:
        b, d, tex = tok.budget(load_img(str(ROOT / "assets/examples" / f"{e}.jpg"), 768).to(dev))
        ex.append(dict(name=e.split("/")[1], d=d, tex=tex, t=10 * tex, budget=b))
    json.dump(ex, open(out / "examples.json", "w"), indent=1)
    need = max(sizes)
    sources = itertools.chain(folder_images(images_dir, need) if images_dir else (), procedural_images(procedural, need), wikiart_images(wikiart, need) if wikiart else ())
    measure(out, tok, meter, sources, sizes, ratio, limit)
    fit(out, sizes)


if __name__ == "__main__":
    main()

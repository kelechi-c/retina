"""Does the encoder still need its PSNR floor? Encodes a folder of images with the floor on and off and reports where it changes anything, what that buys and what it costs. Needs a CUDA GPU and the FLUX.2-dev VAE; read tools/floor_test.md first.

Three policies are compared on every image:
    on      the shipped encoder: decode, check 22 dB, grow 1.4x per side until it passes
    off     no check: the budget's grid is final
    free    check only when the decode is needed anyway, i.e. when the image gets refiner statistics (600 tokens or fewer)

usage: uv run python tools/test_floor.py OUT --images DIR [--ocr] [--dreamsim] [--save] [--limit N]
       uv run python tools/test_floor.py OUT --report-only
"""
import collections
import json
import pathlib
import sys
import time

import click
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXT = {".jpg", ".jpeg", ".png", ".webp"}


def words(img_t):
    """Words Tesseract reads from an image in [-1,1]."""
    import pytesseract
    from PIL import Image

    a = ((img_t[0].clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return [w for w in pytesseract.image_to_string(Image.fromarray(a)).split() if any(c.isalnum() for c in w)]


def recovered(ref, got):
    """Share of the original's words found in the decode, counted with multiplicity."""
    if not ref:
        return None
    a, b = collections.Counter(ref), collections.Counter(got)
    return sum(min(v, b[k]) for k, v in a.items()) / sum(a.values())


class Run:
    def __init__(self, tok, dev, use_lpips, use_dreamsim, ocr):
        self.tok, self.dev, self.ocr = tok, dev, ocr
        self.lp = self.ds = None
        if use_lpips:
            import lpips

            self.lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        if use_dreamsim:
            from dreamsim import dreamsim

            self.ds, _ = dreamsim(pretrained=True, device=dev)
        self.calls = 0
        code = tok._code

        def counted(*a, **k):
            self.calls += 1
            return code(*a, **k)

        tok._code = counted

    def sync(self):
        if self.dev == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def encode(self, x, floor):
        from retina import Tokenizer

        self.tok.floor, self.calls = floor, 0
        self.sync()
        t0 = time.time()
        t = self.tok.encode(x)
        self.sync()
        dt = time.time() - t0
        return t, dict(seconds=dt, tries=self.calls, n=t["n"], packed=len(Tokenizer.pack(t)["seq"]), counted=Tokenizer.length(t), warped=t["knots"] is not None, stats="stats" in t)

    @torch.no_grad()
    def plain_decode_seconds(self, t):
        self.sync()
        t0 = time.time()
        self.tok._plain(t)
        self.sync()
        return time.time() - t0

    @torch.no_grad()
    def quality(self, y, x, grid):
        from retina import common as C

        y8, x8, g8 = C.to8(y), C.to8(x), C.to8(grid)
        q = dict(psnr=float(-10 * torch.log10(((y8 - x8) / 2).pow(2).mean().clamp_min(1e-12))),
                 psnr_grid=float(-10 * torch.log10(((y8 - g8) / 2).pow(2).mean().clamp_min(1e-12))))
        if self.lp is not None:
            q["lpips"], q["lpips_grid"] = float(self.lp(y8, x8).mean()), float(self.lp(y8, g8).mean())
        if self.ds is not None:
            f = lambda v: torch.nn.functional.interpolate((v + 1) / 2, size=224, mode="bicubic", antialias=True).clamp(0, 1)
            q["dreamsim"] = float(self.ds(f(y8), f(x8)).mean())
        return q


def measure(out, run, images_dir, side, limit, save):
    from retina import load_img, save_png

    done = set()
    if (out / "results.jsonl").exists():
        done = {json.loads(l)["id"] for l in open(out / "results.jsonl")}
        print("resuming:", len(done), "images already measured", flush=True)
    root = pathlib.Path(images_dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXT)
    t_start, k = time.time(), 0
    for f in files:
        rel = str(f.relative_to(root))
        if rel in done:
            continue
        if limit and len(done) >= limit:
            break
        group = f.relative_to(root).parts[0] if len(f.relative_to(root).parts) > 1 else "images"
        x = load_img(str(f), side).to(run.dev)
        with torch.no_grad():
            grid = run.tok.v.dec(run.tok.v.enc(x)).float().clamp(-1, 1)
        floor = 22.0
        t_on, on = run.encode(x, floor)
        t_off, off = run.encode(x, 0.0)
        on["quality"] = run.quality(run.tok.decode(t_on), x, grid)
        off["quality"] = run.quality(run.tok.decode(t_off), x, grid)
        # with the floor dropped, the one decode left in encode is only needed for the refiner statistics
        off["seconds_without_extra_decode"] = off["seconds"] - (0 if off["stats"] else run.plain_decode_seconds(t_off))
        rec = dict(id=rel, group=group, floor=floor, budget=t_on["budget"], drex=t_on["drex"], tex=t_on["tex"], on=on, off=off, moved=on["n"] != off["n"])
        if run.ocr:
            ref = words(x)
            rec["ocr"] = dict(words=len(ref), on=recovered(ref, words(run.tok.decode(t_on))), off=recovered(ref, words(run.tok.decode(t_off))), grid=recovered(ref, words(grid)))
        if save and rec["moved"]:
            d = out / "moved"
            d.mkdir(exist_ok=True)
            stem = rel.replace("/", "__").rsplit(".", 1)[0]
            save_png(run.tok.decode(t_on), d / f"{stem}__on_{on['packed']}.png")
            save_png(run.tok.decode(t_off), d / f"{stem}__off_{off['packed']}.png")
        with open(out / "results.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done.add(rel)
        k += 1
        print(f"[{len(done)}] {rel[:52]:52s} {group:12s} off {off['packed']:5d} -> on {on['packed']:5d} tokens ({on['tries']} tries) | PSNR {off['quality']['psnr']:5.1f} -> {on['quality']['psnr']:5.1f} | {on['seconds']:.2f} s vs {off['seconds_without_extra_decode']:.2f} s | {(time.time() - t_start) / k:.1f} s/image", flush=True)


def report(out, learned_max=600):
    R = [json.loads(l) for l in open(out / "results.jsonl")]
    for r in R:  # 'free': keep the check only where the decode is needed anyway (the budget's grid gets statistics)
        r["free"] = r["on"] if r["off"]["counted"] + 9 <= learned_max else r["off"]
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    has = lambda k: all(k in r["on"]["quality"] for r in R)
    L = ["# Does the encoder still need its PSNR floor?\n", f"{len(R)} images. Floor {R[0]['floor'] if R else ''} dB. Policies: **on** (shipped), **off** (the budget's grid is final), **free** (check only when the image gets refiner statistics, where the decode is needed anyway).\n"]
    L.append("## By group\n\n| group | n | moved by the floor | median extra steps | median tokens off → on | median encode s, on / off | median PSNR off → on, moved only |\n|---|---|---|---|---|---|---|")
    groups = sorted({r["group"] for r in R}) + ["all"]
    for g in groups:
        rs = [r for r in R if g in ("all", r["group"])]
        mv = [r for r in rs if r["moved"]]
        L.append(f"| {g} | {len(rs)} | {len(mv)} ({len(mv) / max(len(rs), 1):.0%}) | {med([r['on']['tries'] - 1 for r in mv]):.0f} | {med([r['off']['packed'] for r in rs]):.0f} → {med([r['on']['packed'] for r in rs]):.0f} | {med([r['on']['seconds'] for r in rs]):.3f} / {med([r['off']['seconds_without_extra_decode'] for r in rs]):.3f} | "
                 + (f"{med([r['off']['quality']['psnr'] for r in mv]):.1f} → {med([r['on']['quality']['psnr'] for r in mv]):.1f}" if mv else "–") + " |")
    L.append("\n## Policies, all images\n\n| policy | total tokens | median encode s | images under 22 dB | median LPIPS vs grid | worst LPIPS vs grid |" + (" pages reading at OCR .9+ | median OCR |" if any("ocr" in r for r in R) else "") + "\n|---|---|---|---|---|---|" + ("---|---|" if any("ocr" in r for r in R) else ""))
    for p in ("on", "off", "free"):
        secs = [r[p]["seconds"] if p == "on" or (p == "free" and r["free"] is r["on"]) else r["off"]["seconds_without_extra_decode"] for r in R]
        lp = [r[p]["quality"].get("lpips_grid", float("nan")) for r in R]
        row = f"| {p} | {sum(r[p]['packed'] for r in R):,} | {med(secs):.3f} | {sum(r[p]['quality']['psnr'] < 22 for r in R)} | {med(lp):.3f} | {max(lp) if lp else float('nan'):.3f} |"
        if any("ocr" in r for r in R):
            oc = [r["ocr"][p if p != "free" else ("on" if r["free"] is r["on"] else "off")] for r in R if "ocr" in r and r["ocr"]["words"] >= 20]
            oc = [v for v in oc if v is not None]
            row += f" {sum(v >= .9 for v in oc)} of {len(oc)} | {med(oc):.2f} |"
        L.append(row)
    if any("ocr" in r for r in R):
        gr = [r["ocr"]["grid"] for r in R if "ocr" in r and r["ocr"]["words"] >= 20 and r["ocr"]["grid"] is not None]
        L.append(f"\nFull grid for reference: {sum(v >= .9 for v in gr)} of {len(gr)} pages at OCR .9+, median {med(gr):.2f}. Pages with fewer than 20 words are left out of the OCR columns.")
    mv = sorted((r for r in R if r["moved"]), key=lambda r: -(r["off"]["quality"].get("lpips_grid", 0) - r["on"]["quality"].get("lpips_grid", 0)))
    L.append("\n## Every image the floor moved, most helped first\n\n| image | group | DReX | tokens off → on | PSNR off → on | LPIPS vs grid off → on |" + (" OCR off → on |" if any("ocr" in r for r in mv) else "") + "\n|---|---|---|---|---|---|" + ("---|" if any("ocr" in r for r in mv) else ""))
    for r in mv:
        a, b = r["off"]["quality"], r["on"]["quality"]
        row = f"| {r['id']} | {r['group']} | {r['drex']:.2f} | {r['off']['packed']} → {r['on']['packed']} | {a['psnr']:.1f} → {b['psnr']:.1f} | {a.get('lpips_grid', float('nan')):.3f} → {b.get('lpips_grid', float('nan')):.3f} |"
        if "ocr" in r and r["ocr"]["on"] is not None:
            row += f" {r['ocr']['off']:.2f} → {r['ocr']['on']:.2f} |"
        L.append(row)
    if not has("lpips"):
        L.append("\nLPIPS was off for this run, so the LPIPS columns read nan.")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


@click.command()
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--images", "images_dir", default="", help="folder of images, searched recursively; the first folder level names the group")
@click.option("--side", default=768, show_default=True)
@click.option("--ocr", is_flag=True, help="read words with Tesseract (needs the tesseract binary and pytesseract)")
@click.option("--dreamsim", is_flag=True, help="also score DreamSim against the original (uv sync --extra judges)")
@click.option("--lpips/--no-lpips", "use_lpips", default=True, show_default=True)
@click.option("--save", is_flag=True, help="save both decodes of every image the floor moved, to look at by eye")
@click.option("--limit", default=0, show_default=True)
@click.option("--vae", default="flux2", show_default=True)
@click.option("--drext", default="default", show_default=True, help="'default' for the shipped text-tuned DReX, 'none' for the original")
@click.option("--dev", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--report-only", is_flag=True)
def main(out, images_dir, side, ocr, dreamsim, use_lpips, save, limit, vae, drext, dev, report_only):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not report_only:
        if not images_dir:
            raise click.UsageError("give --images DIR")
        from retina.tokenizer import DREXT, Tokenizer

        tok = Tokenizer(vae, dev, drext=None if drext == "none" else DREXT if drext == "default" else drext)
        run = Run(tok, dev, use_lpips, dreamsim, ocr)
        json.dump(dict(images=images_dir, side=side, ocr=ocr, dreamsim=dreamsim, lpips=use_lpips, vae=vae, drext=drext), open(out / "settings.json", "w"), indent=1)
        measure(out, run, images_dir, side, limit, save)
    report(out)


if __name__ == "__main__":
    main()

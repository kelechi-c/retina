"""Does the 22 dB PSNR floor still earn its keep with the shipped default (text-tuned DReX + S1)?

A/B on a folder of images: encode with `floor=22` (shipped) and with `floor=0`, same tokenizer,
then decode both and score against the original. Records the token counts, the extra encode time,
how many grow iterations the floor used, and the quality delta.

usage: python tools/check_floor.py OUT --images DIR [--sides 768,512] [--limit 0]
"""
import argparse
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
EXT = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--images", default="")
    p.add_argument("--from-tar", default="", help="REPO:FILE on the HF Hub; the project's inputs.tar has the document/text test sets")
    p.add_argument("--sides", default="768")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dev", default="cuda")
    a = p.parse_args()
    dev = a.dev if torch.cuda.is_available() else "cpu"
    sides = [int(s) for s in a.sides.split(",")]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from retina import common as C, load_img
    from retina.tokenizer import Tokenizer

    root = pathlib.Path(a.images) if a.images else None
    if a.from_tar:
        import tarfile
        from huggingface_hub import hf_hub_download
        repo, fname = a.from_tar.split(":")
        tar = hf_hub_download(repo, fname, repo_type="dataset")
        root = pathlib.Path("/tmp/floor_images")
        root.mkdir(parents=True, exist_ok=True)
        n = 0
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not m.isfile() or not m.name.lower().endswith(tuple(EXT)):
                    continue
                parts = [x for x in m.name.split("/") if x and x != "."]
                base = parts[-1]
                if "DocLayNet" in base:
                    g = "doclaynet"
                elif base.startswith("funsd"):
                    g = "funsd"
                elif base.startswith("synthdog"):
                    g = "synthdog"
                elif base.startswith("text_"):
                    g = "text"
                else:
                    g = parts[-2] if len(parts) > 1 else "images"
                d = root / g
                d.mkdir(parents=True, exist_ok=True)
                (d / base).write_bytes(t.extractfile(m).read())
                n += 1
        print("extracted", n, "images from", a.from_tar, "->", root, flush=True)
    files = sorted(f for f in root.rglob("*") if f.suffix.lower() in EXT)
    if a.limit:
        files = files[: a.limit]
    print("images", len(files), "sides", sides, flush=True)

    tok = Tokenizer("flux2", dev)  # shipped default: tuned DReX, S1 refiner, floor 22
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

    @torch.no_grad()
    def score(y, x, view="native"):
        a8, b8 = C.to8(y), C.to8(x)
        psnr = float(-10 * torch.log10(((a8 - b8) / 2).pow(2).mean().clamp_min(1e-12)))
        return dict(psnr=psnr, lpips=float(lp(a8, b8).mean()))

    @torch.no_grad()
    def encode(x):
        calls = {"n": 0}
        orig = tok._psnr

        def counted(xx, cc):
            calls["n"] += 1
            return orig(xx, cc)

        tok._psnr = counted
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.perf_counter()
        t = tok.encode(x)
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.perf_counter() - t0
        tok._psnr = orig
        return t, dt, calls["n"]

    rows = []
    for i, f in enumerate(files):
        rel = f.relative_to(root)
        group = rel.parts[0] if len(rel.parts) > 1 else "images"
        gid = f"file:{rel}"
        for S in sides:
            x = load_img(str(f), S).to(dev)
            b, d, tex = tok.budget(x)
            tok.floor = 22.
            t1, dt1, calls = encode(x)
            tok.floor = 0.
            t2, dt2, _ = encode(x)
            tok.floor = 22.
            y1, y2 = tok.decode(t1), tok.decode(t2)
            s1, s2 = score(y1, x), score(y2, x)
            rows.append(dict(id=gid, group=group, side=S, d=d, tex=tex, budget=b,
                             tokens_floor=Tokenizer.length(t1), tokens_nf=Tokenizer.length(t2),
                             stats_floor=t1.get("stats") is not None, stats_nf=t2.get("stats") is not None,
                             psnr_floor=s1["psnr"], psnr_nf=s2["psnr"], lpips_floor=s1["lpips"], lpips_nf=s2["lpips"],
                             enc_s_floor=dt1, enc_s_nf=dt2, psnr_calls=calls))
        print(f"[{i + 1}/{len(files)}] {rel}", flush=True)
    json.dump(dict(rows=rows, sides=sides), open(out / "floor.json", "w"))

    def med(v):
        v = sorted(x for x in v if x is not None)
        return v[len(v) // 2] if v else float("nan")

    L = ["# Does the 22 dB PSNR floor still earn its keep?", ""]
    L.append("Shipped default `Tokenizer()` (text-tuned DReX, S1 refiner), same image, `floor=22` vs `floor=0`. ")
    L.append("Tokens are counted tokens (`Tokenizer.length`), not packed length.")
    L.append("")
    act = [r for r in rows if r["tokens_floor"] > r["tokens_nf"]]
    L.append(f"- {len(act)}/{len(rows)} (image, side) pairs change budget when the floor is on ({len(act) / max(len(rows), 1):.1%}).")
    L.append(f"- On those, median tokens {med([r['tokens_nf'] for r in act]):.0f} -> {med([r['tokens_floor'] for r in act]):.0f}, "
             f"median PSNR {med([r['psnr_nf'] for r in act]):.2f} -> {med([r['psnr_floor'] for r in act]):.2f} dB, "
             f"median LPIPS {med([r['lpips_nf'] for r in act]):.4f} -> {med([r['lpips_floor'] for r in act]):.4f}.")
    L.append(f"- Encode time overall: median {1000 * med([r['enc_s_nf'] for r in rows]):.0f} ms -> "
             f"{1000 * med([r['enc_s_floor'] for r in rows]):.0f} ms with the floor "
             f"({100 * (med([r['enc_s_floor'] for r in rows]) / max(med([r['enc_s_nf'] for r in rows]), 1e-9) - 1):+.0f}%).")
    L.append("")
    for S in sides:
        L.append(f"## side {S}")
        L.append("")
        L.append("| group | n | floor active | tokens nf -> floor | PSNR nf -> floor | LPIPS nf -> floor | enc ms nf -> floor |")
        L.append("|---|---|---|---|---|---|---|")
        groups = sorted({r["group"] for r in rows if r["side"] == S})
        for g in groups:
            v = [r for r in rows if r["side"] == S and r["group"] == g]
            va = [r for r in v if r["tokens_floor"] > r["tokens_nf"]]
            L.append(f"| {g} | {len(v)} | {len(va)} | {med([r['tokens_nf'] for r in v]):.0f} -> {med([r['tokens_floor'] for r in v]):.0f} | "
                     f"{med([r['psnr_nf'] for r in v]):.2f} -> {med([r['psnr_floor'] for r in v]):.2f} | "
                     f"{med([r['lpips_nf'] for r in v]):.4f} -> {med([r['lpips_floor'] for r in v]):.4f} | "
                     f"{1000 * med([r['enc_s_nf'] for r in v]):.0f} -> {1000 * med([r['enc_s_floor'] for r in v]):.0f} |")
        L.append("")
    L.append("## Floor-active cases, worst 10 by PSNR gained")
    L.append("")
    L.append("| image | side | tokens | PSNR nf -> floor | LPIPS nf -> floor | psnr calls |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(act, key=lambda r: r["psnr_floor"] - r["psnr_nf"], reverse=True)[:10]:
        L.append(f"| {r['id']} | {r['side']} | {r['tokens_nf']:.0f} -> {r['tokens_floor']:.0f} | "
                 f"{r['psnr_nf']:.2f} -> {r['psnr_floor']:.2f} | {r['lpips_nf']:.4f} -> {r['lpips_floor']:.4f} | {r['psnr_calls']} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("FLOOR_DONE", flush=True)


if __name__ == "__main__":
    main()

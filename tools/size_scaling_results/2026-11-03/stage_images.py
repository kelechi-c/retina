"""Stage the size-scaling image set from the project's curated fine-tune shards (private HF dataset).

The shards are the 6,000-image fine-tune pool built by the curate session (12 tars, 500 images
each, 1,200 per complexity quintile, Open Images / SA-1B / DIV2K / WikiArt / Met / synthetic).
They are disjoint from the test, eval and document images the size-scaling notes say not to use.

usage (inside the Modal container):
    python tools/size_stage_images.py --shards 0,3 --out /runs/size_images [--min-side 1024]
"""
import argparse
import glob
import os
import pathlib
import tarfile

KNOWN = {"oi", "sa1b", "div2k", "wikiart", "met", "synth", "lsdir"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", default="0,3")
    p.add_argument("--rebalance", default="", help="skip the download: rebalance an already staged folder instead")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", default="tensorkelechi/retina-inputs")
    p.add_argument("--min-side", type=int, default=1024)
    p.add_argument("--per-group", type=int, default=0, help="cap per source group (0: no cap)")
    a = p.parse_args()
    from huggingface_hub import hf_hub_download
    from PIL import Image

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, skipped, total = {}, 0, 0
    if a.rebalance:  # copy at most --per-group files per group from an existing staged folder
        src = pathlib.Path(a.rebalance)
        import shutil
        for g in sorted(p.name for p in src.iterdir() if p.is_dir()):
            files = sorted((src / g).iterdir())
            keep = files if not a.per_group else files[: a.per_group]
            (out / g).mkdir(parents=True, exist_ok=True)
            for f in keep:
                shutil.copy2(f, out / g / f.name)
                counts[g] = counts.get(g, 0) + 1
        print("rebalanced", sum(counts.values()), "images", flush=True)
        for g in sorted(counts):
            print(f"  {g:8s} {counts[g]}", flush=True)
        print("STAGE_DONE", flush=True)
        return
    for sh in [s.strip() for s in a.shards.split(",") if s.strip()]:
        name = f"curated/shard-{int(sh):05d}.tar"
        print("download", name, flush=True)
        tar = hf_hub_download(a.repo, name, repo_type="dataset")
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                base = os.path.basename(m.name)
                if base.startswith(".") or ".." in base:
                    continue
                f = t.extractfile(m)
                data = f.read()
                total += 1
                try:
                    import io
                    im = Image.open(io.BytesIO(data))
                    if min(im.size) < a.min_side:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
                stem = base.rsplit(".", 1)[0]
                group = stem.split("-")[0] if "-" in stem else "other"
                group = group if group in KNOWN else "other"
                if a.per_group and counts.get(group, 0) >= a.per_group:
                    skipped += 1
                    continue
                d = out / group
                d.mkdir(parents=True, exist_ok=True)
                (d / base).write_bytes(data)
                counts[group] = counts.get(group, 0) + 1
    print("staged", sum(counts.values()), "of", total, "members;", skipped, "skipped (< %d px or unreadable)" % a.min_side, flush=True)
    for g in sorted(counts):
        print(f"  {g:8s} {counts[g]}", flush=True)
    print("STAGE_DONE", flush=True)


if __name__ == "__main__":
    main()

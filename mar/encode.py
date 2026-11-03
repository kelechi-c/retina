"""Tokenizes the AR v2 buckets at 512 and each image's flip. Loading runs in worker threads, and images over a source's cap are skipped before they are decoded. Keeps a few originals with their decodes, the check that 512 tokens are worth modelling.

usage: python -m mar.encode OUT [--manifest M] [--caps 0/procedural=1500,...] [--p a,b,c,d] [--shards a-b] [--tars a-b] [--mirror DIR]
"""
import collections
import concurrent.futures as cf
import itertools
import json
import os
import pathlib
import random
import shutil
import time
from types import SimpleNamespace

import click
import torch

from retina import Tokenizer
from retina import common as C

from . import data as D


@click.command()
@click.argument("out")
@click.option("--manifest", default="", show_default=True, help="curated set manifest; its shards are on a private repo, so without it the curated source is skipped")
@click.option("--caps", default="0/procedural=1500,0/wikiart=1500,1/curated=1500,1/wikiart=1500,2/dtd=1000,2/curated=1000,2/wikiart=1000,3/curated=1500,3/wikiart=1500", show_default=True)
@click.option("--p", default=".54,.226,.048,.115", show_default=True, help="keep rate of WikiArt per bucket: its cap over its count, plus 5%%")
@click.option("--shards", default="0-71", show_default=True)
@click.option("--tars", default="0-11", show_default=True)
@click.option("--side", type=int, default=512, show_default=True)
@click.option("--workers", type=int, default=6, show_default=True)
@click.option("--mirror", default="", show_default=True)
@click.option("--skip", default="", show_default=True, help="tokens file on the private HF repo whose images are already encoded")
@click.option("--salt", default="", show_default=True, help="a new draw of WikiArt, for a second batch")
@click.option("--procedural_from", type=int, default=0, show_default=True)
@click.option("--max_s", type=float, default=0, show_default=True, help="stop and save after this many seconds, to fit a budget")
def main(**kw):
    a = SimpleNamespace(**kw)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    caps = {tuple(k.split("/")): int(v) for k, v in (x.split("=") for x in a.caps.split(","))}
    if not a.manifest:
        caps = {k: v for k, v in caps.items() if k[1] != "curated"}
    span = lambda t: range(int(t.split("-")[0]), int(t.split("-")[-1]) + 1)
    order = lambda t: random.Random(0).sample(list(span(t)), len(span(t)))  # shards run in style order, and a time limit may cut the last
    tok = Tokenizer("flux2", "cuda")
    src = itertools.chain(
        ((f"proc-{i}", "procedural", 0, D.procedural(i, a.side)) for i in range(a.procedural_from, a.procedural_from + caps.get(("0", "procedural"), 0))),
        D.dtd(None) if caps.get(("2", "dtd")) else iter(()),
        D.curated(span(a.tars), a.manifest) if a.manifest else iter(()),
        D.wikiart(D.STYLES, D.GENRES, order(a.shards), [float(x) for x in a.p.split(",")], a.salt),
    )
    taken = collections.Counter()
    skip = set()
    if a.skip:
        from huggingface_hub import hf_hub_download

        skip = {r["id"] for r in torch.load(hf_hub_download("tensorkelechi/retina-inputs", a.skip, repo_type="dataset"), weights_only=False)}
        print("already encoded", len(skip), flush=True)


    def wanted():
        """Takes from each source up to its cap; stops once every cap is met, so no shard is read for nothing."""
        for key, source, label, raw in src:
            k = str(label), source
            if key in skip:
                continue
            if taken[k] < caps.get(k, 0):
                taken[k] += 1
                yield key, source, label, raw
            if all(taken[c] >= v for c, v in caps.items()):
                return


    def load(item):
        key, source, label, raw = item
        return key, source, label, D.to_tensor(D.opened(raw), a.side)


    def save():
        torch.save(res, out / "tokens.pt")
        if a.mirror:
            os.makedirs(a.mirror, exist_ok=True)
            shutil.copy(out / "tokens.pt", pathlib.Path(a.mirror) / "tokens.pt")


    res, done, bad, t_enc, kept = [], collections.Counter(), 0, 0., collections.Counter()
    t1 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as pool:
        it, q = wanted(), collections.deque()
        while True:
            while len(q) < a.workers * 3:
                n = next(it, None)
                if n is None:
                    break
                q.append(pool.submit(load, n))
            if not q or (a.max_s and time.time() - t1 > a.max_s):
                break
            try:
                key, source, label, x = q.popleft().result()
                x = x.to("cuda")
                t2 = time.time()
                for flip in (0, 1):
                    p = Tokenizer.pack(tok.encode(x.flip(-1) if flip else x))
                    res.append(dict(id=key, source=source, label=label, flip=flip, seq=p["seq"].half().cpu(), layout=p["layout"]))
                t_enc += time.time() - t2
                done[label] += 1
                k = label, source
                if kept[k] < 3:
                    kept[k] += 1
                    d = tok.decode(Tokenizer.unpack(dict(seq=res[-2]["seq"].float().cuda(), layout=res[-2]["layout"])))
                    C.save_png(x, out / f"check_{label}_{source}_{kept[k]}_orig.png")
                    C.save_png(d, out / f"check_{label}_{source}_{kept[k]}_{len(res[-2]['seq'])}.png")
            except Exception as e:
                bad += 1
                print("bad", repr(e)[:160], flush=True)
            m = sum(done.values())
            if m % 500 == 0:
                print(m, dict(done), f"wall s/img {(time.time() - t1) / m:.3f} encode s/img {t_enc / m:.3f}", flush=True)
            if m % 3000 == 0:
                save()
    save()


    def quant(L):
        L = sorted(L)
        return [L[int(q * (len(L) - 1))] for q in (.05, .25, .5, .75, .95)] if L else []


    images = sum(done.values())
    report = dict(images=images, bad=bad, seconds=time.time() - t1, encode_s=t_enc, per_source={"/".join(k): v for k, v in taken.items()})
    for l in range(4):
        v = [len(r["seq"]) for r in res if r["label"] == l and not r["flip"]]
        w = [len(r["seq"]) for r in res if r["label"] == l and not r["flip"] and r["source"] != "curated"]
        report[f"bucket{l}"] = dict(n=len(v), all=quant(v), no_curated=quant(w))
        print("bucket", l, "n", len(v), "lengths p5 p25 p50 p75 p95", quant(v), "without curated", quant(w), flush=True)
    json.dump(report, open(out / "encode.json", "w"))
    print("encode", {k: report[k] for k in ("images", "bad", "seconds", "encode_s", "per_source")}, flush=True)
    print("ENCODE_DONE", flush=True)


if __name__ == "__main__":
    main()

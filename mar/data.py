"""MAR data: four content labels, assigned from source, style and genre before anything is tokenized. Yields (id, source, label, PIL image).

Labels: 0 simple, 1 moderate, 2 textured, 3 dense.
"""
import io
import itertools
import json
import os
import random
import tarfile

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

SIMPLE = {"Minimalism", "Color_Field_Painting"}
MODERATE = {"Abstract_Expressionism", "Cubism", "Analytical_Cubism", "Synthetic_Cubism", "Pop_Art"}
TEXTURED = {"Impressionism", "Pointillism"}
DENSE = {"Baroque", "Northern_Renaissance", "Mannerism_Late_Renaissance", "Romanticism"}
BINS = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3}  # curated bins come from DReX and texture scores, so the ladder is also read without them


# label order of huggan/wikiart, whose parquet files carry no names
STYLES = ["Abstract_Expressionism", "Action_painting", "Analytical_Cubism", "Art_Nouveau", "Baroque", "Color_Field_Painting", "Contemporary_Realism", "Cubism", "Early_Renaissance", "Expressionism", "Fauvism", "High_Renaissance", "Impressionism", "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art", "Post_Impressionism", "Realism", "Rococo", "Romanticism", "Symbolism", "Synthetic_Cubism", "Ukiyo_e"]
GENRES = ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting", "portrait", "religious_painting", "sketch_and_study", "still_life", "Unknown Genre"]


def wikiart_label(style, genre):
    """Frozen map. This dataset has no Neoplasticism, Suprematism or Abstract Art style, so the last two rules stand in through its genres."""
    if style in SIMPLE:
        return 0
    if style in MODERATE:
        return 1
    if style in TEXTURED or genre == "landscape":
        return 2
    if style in DENSE:
        return 3
    return 1 if genre == "abstract_painting" else None


def procedural(seed, side=512):
    """A Mondrian-style lattice or a few flat shapes, usually rotated. Fixed by the seed."""
    r = random.Random(seed)
    S = side * 2
    pal = [(235, 232, 222), (250, 250, 245), (200, 30, 35), (30, 60, 150), (240, 200, 40), (20, 20, 20)]
    if r.random() < .5:
        im = Image.new("RGB", (S * 2, S * 2), pal[r.randrange(2)])
        d = ImageDraw.Draw(im)
        xs = sorted(r.sample(range(0, S * 2, 40), r.randint(3, 8)))
        ys = sorted(r.sample(range(0, S * 2, 40), r.randint(3, 8)))
        xs, ys = [0, *xs, S * 2], [0, *ys, S * 2]
        for a, b in zip(xs, xs[1:]):
            for c, e in zip(ys, ys[1:]):
                if r.random() < .3:
                    d.rectangle((a, c, b, e), fill=pal[r.randrange(2, 5)])
        w = r.randint(6, 18)
        for x in xs:
            d.rectangle((x - w, 0, x + w, S * 2), fill=pal[5])
        for y in ys:
            d.rectangle((0, y - w, S * 2, y + w), fill=pal[5])
    else:
        im = Image.new("RGB", (S * 2, S * 2), tuple(r.randrange(256) for _ in range(3)))
        d = ImageDraw.Draw(im)
        for _ in range(r.randint(1, 6)):
            c = tuple(r.randrange(256) for _ in range(3))
            x, y, q = r.randrange(S // 2, S * 3 // 2), r.randrange(S // 2, S * 3 // 2), r.randrange(S // 12, S // 2)
            kind = r.randrange(3)
            if kind == 0:
                d.ellipse((x - q, y - q, x + q, y + q), fill=c)
            elif kind == 1:
                d.rectangle((x - q, y - q, x + q, y + q), fill=c)
            else:
                d.polygon([(x + r.randint(-q, q), y + r.randint(-q, q)) for _ in range(3)], fill=c)
    if r.random() < .7:
        im = im.rotate(r.uniform(-45, 45), Image.BICUBIC)
    u = S // 2
    return im.crop((u, u, u + S, u + S)).resize((side, side), Image.LANCZOS)


def wikiart(S, G, shards, p, salt=""):
    """Keeps each image of bucket l with probability p[l], by a fixed draw on its index."""
    for sh in shards:
        f = hf_hub_download("huggan/wikiart", f"data/train-{sh:05d}-of-00072.parquet", repo_type="dataset")
        i = -1
        for b in pq.ParquetFile(f).iter_batches(256):
            d = b.to_pydict()
            for j in range(len(d["style"])):
                i += 1
                l = wikiart_label(S[d["style"][j]], G[d["genre"][j]])
                if l is None or random.Random(f"{sh}/{i}{salt}").random() >= p[l]:
                    continue
                yield f"w{sh}-{i}", "wikiart", l, d["image"][j]["bytes"]


def curated(tars, manifest):
    """Curated images, minus its WikiArt, which the WikiArt source already covers."""
    M = {json.loads(l)["file"]: json.loads(l) for l in open(manifest)}
    for k in tars:
        f = hf_hub_download("tensorkelechi/retina-inputs", f"curated/shard-{k:05d}.tar", repo_type="dataset")
        with tarfile.open(f) as t:
            for m in t:
                r = M.get(os.path.basename(m.name))
                if r is None or r["set"] != "finetune" or r["source_dataset"] == "wikiart":
                    continue
                yield r["id"], "curated", BINS[int(r["bin"])], t.extractfile(m).read()


def dtd(limit=None, stride=5):
    """Every stride-th row, because the rows run class by class."""
    from datasets import load_dataset

    rows = itertools.chain(*(load_dataset("tanganke/dtd", split=x, streaming=True) for x in ("train", "test")))
    n = 0
    for i, r in enumerate(rows):
        if i % stride:
            continue
        if limit is not None and n >= limit:
            return
        n += 1
        yield f"dtd-{i}", "dtd", 2, r["image"].convert("RGB")


def opened(raw):
    """A worker's half of loading: raw bytes or a finished picture -> picture."""
    return raw if isinstance(raw, Image.Image) else Image.open(io.BytesIO(raw)).convert("RGB")


def to_tensor(im, side):
    import torch

    s = min(im.size)
    l, u = (im.width - s) // 2, (im.height - s) // 2
    a = np.asarray(im.crop((l, u, l + s, u + s)).resize((side, side), Image.LANCZOS)).copy()
    return torch.from_numpy(a).permute(2, 0, 1).float()[None] / 127.5 - 1

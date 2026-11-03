"""Trains MAR on retina tokens made by encode.py. At the end it samples with the EMA weights and re-encodes each picture, the check that generated length follows content.

usage: python -m mar.train OUT TOKENS [--steps N] [--load CKPT] [--more TOKENS2] [--until STEP] [--budget B]

--steps counts new steps. --load continues a checkpoint's weights, EMA, scaling and layouts; Adam restarts. --more adds images to training; the held-out images are always drawn from TOKENS. --until, with --load, trains up to that total step.
The released model: --budget 12000 on one A100, 30,000 steps, then --load and --more with the second batch of images, --until 100000. 9.8 it/s, about 3 hours.
"""
import json
import math
import pathlib
import random
import time
from types import SimpleNamespace

import click
import torch
import torch.nn.functional as F

from retina import Tokenizer, save_png

from .model import NAMES, NULL, W, build, coords, generate, slots


@click.command()
@click.argument("out")
@click.argument("tokens")
@click.option("--steps", type=int, default=30000, show_default=True)
@click.option("--load", default="", show_default=True)
@click.option("--more", default="", show_default=True)
@click.option("--until", type=int, default=0, show_default=True)
@click.option("--d", type=int, default=512, show_default=True)
@click.option("--layers", type=int, default=12, show_default=True)
@click.option("--head_w", type=int, default=1024, show_default=True)
@click.option("--head_l", type=int, default=3, show_default=True)
@click.option("--budget", type=int, default=6000, show_default=True, help="tokens per batch")
@click.option("--lr", type=float, default=2e-4, show_default=True)
@click.option("--warmup", type=int, default=2000, show_default=True)
@click.option("--ema", type=float, default=.9999, show_default=True)
@click.option("--samples", type=int, default=4, show_default=True, help="per label, at the end")
@click.option("--every", type=int, default=2000, show_default=True, help="checkpoint and held-out loss")
def main(**kw):
    a = SimpleNamespace(**kw)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    random.seed(0)
    torch.manual_seed(0)

    R = torch.load(a.tokens, weights_only=False)
    ids = sorted({r["id"] for r in R})
    random.shuffle(ids)
    ho = set(ids[: max(8, min(400, len(ids) // 20))])
    if a.more:
        R += torch.load(a.more, weights_only=False)
    ck = torch.load(a.load, weights_only=False) if a.load else None
    key = lambda r: (r["layout"]["n"], r["layout"]["warped"], r["layout"]["stats"])
    LAYOUTS = ck["LAYOUTS"] if ck else sorted({key(r) for r in R})
    n_all = len(R)
    R = [r for r in R if key(r) in LAYOUTS]  # a resumed model has a fixed layout head
    print("dropped for unseen layouts", n_all - len(R), flush=True)
    train, held = [r for r in R if r["id"] not in ho], [r for r in R if r["id"] in ho]
    print("train", len(train), "held", len(held), "layouts", len(LAYOUTS), flush=True)
    for r in R:
        r["slot"], r["seq"] = slots(r["layout"]), r["seq"].float()
    mu, sd = torch.zeros(4, W), torch.ones(4, W)
    for s in range(4):
        v = torch.cat([r["seq"][r["slot"] == s] for r in train[:4000]])
        if len(v) >= 8:
            mu[s], sd[s] = v.mean(0), v.std(0).clamp_min(1e-3)
    mu, sd = (ck["mu"].to(dev), ck["sd"].to(dev)) if ck else (mu.to(dev), sd.to(dev))
    BY = [[r for r in train if r["label"] == l] for l in range(4)]
    SIDE = R[0]["layout"]["side"]

    model = build(vars(a), len(LAYOUTS)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("MAR params", n_params, flush=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    start = 0
    if ck:
        start = ck["step"]
        if a.until:
            a.steps = a.until - start
        model.load_state_dict(ck["model"])
        ema = {k: v.to(dev) for k, v in ck["ema"].items()}


    def collate(B):
        N = max(len(r["seq"]) for r in B)
        y, seen, slot, co, pad = torch.zeros(len(B), N, W), torch.zeros(len(B), N, dtype=torch.bool), torch.zeros(len(B), N, dtype=torch.long), torch.zeros(len(B), N, 2), torch.ones(len(B), N, dtype=torch.bool)
        for i, r in enumerate(B):
            n = len(r["seq"])
            y[i, :n], slot[i, :n], co[i, :n], pad[i, :n] = r["seq"], r["slot"], coords(r["layout"], r["slot"]), False
            head = n - (r["layout"]["n"] // 2) ** 2
            order = torch.cat((torch.arange(head), head + torch.randperm(n - head)))
            # knots and statistics are written first, so some batches train them with nothing else known
            k = random.randrange(head) if head and random.random() < .3 else int((1 - math.cos(math.pi / 2 * random.random())) * n)
            seen[i, order[:k]] = True
        y, seen, slot, co, pad = y.to(dev), seen.to(dev), slot.to(dev), co.to(dev), pad.to(dev)
        label = torch.tensor([r["label"] for r in B], device=dev)
        lay = torch.tensor([LAYOUTS.index(key(r)) for r in B], device=dev)
        return (y - mu[slot]) / sd[slot], seen, slot, co, label, lay, pad


    def batch():
        """Equal mix of labels; about a.budget tokens per batch, sequences of similar length together."""
        pool = sorted((random.choice(BY[random.randrange(4)]) for _ in range(256)), key=lambda r: len(r["seq"]))
        i = random.randrange(len(pool))
        B, n = [], 0
        for r in pool[i:] + pool[:i]:
            n = max(n, len(r["seq"]))
            if n * (len(B) + 1) > a.budget:
                break
            B.append(r)
        return B or [pool[i]]


    def loss_of(B, drop=True):
        y, seen, slot, co, label, lay, pad = collate(B)
        if drop:  # 10% unconditional, for guidance
            label = torch.where(torch.rand(len(label), device=dev) < .1, torch.full_like(label, NULL), label)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(y, seen, slot, co, label, lay, pad)
            m = ~seen & ~pad
            if not m.any():
                m = ~pad
            zz, x1, sl = z[m].repeat(4, 1), y[m].repeat(4, 1), slot[m].repeat(4)
            x0, t = torch.randn_like(x1), torch.rand(len(x1), device=dev)
            v = model.head(x0 + (x1 - x0) * t[:, None], t, zz)
            e = (v - (x1 - x0)).pow(2).mean(-1)
            l = sum(lam * e[sl == s].mean() for s, lam in ((0, 2.), (1, .5), (2, .5), (3, 1.)) if (sl == s).any())
            ll = F.cross_entropy(model.h_layout(model.label(label)), lay)
        return l, ll


    @torch.no_grad()
    def held_loss(data, n=64, wrong=False):
        """Fixed noise and masks. wrong gives each sequence another label: if that costs nothing, the label does not reach the content."""
        model.eval()
        random.seed(1)
        torch.manual_seed(1)
        out_ = []
        for r in random.sample(data, min(n, len(data))):
            q = r if not wrong else {**r, "label": (r["label"] + 2) % 4}
            out_.append(loss_of([q], drop=False)[0].item())
        random.seed()
        model.train()
        return sum(out_) / len(out_)


    def save(step):
        torch.save(dict(model=model.state_dict(), ema=ema, mu=mu, sd=sd, LAYOUTS=LAYOUTS, NAMES=NAMES, side=SIDE, step=step, args=vars(a)), out / "mar.pt")


    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(.9, .95), weight_decay=.02)
    t0, avg, seen_tok = time.time(), None, 0
    torch.cuda.reset_peak_memory_stats()
    step = start
    for step in range(start + 1, start + a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1., (step - start) / a.warmup)
        B = batch()
        seen_tok += sum(len(r["seq"]) for r in B)
        l, ll = loss_of(B)
        opt.zero_grad(set_to_none=True)
        (l + ll).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].lerp_(v, 1 - a.ema) if v.is_floating_point() else ema[k].copy_(v)
        avg = l.item() if avg is None else .98 * avg + .02 * l.item()
        if step % 100 == 0:
            dt = time.time() - t0
            print(f"step {step} flow {avg:.4f} layout {ll.item():.3f} {(step - start) / dt:.2f} it/s tokens/s {seen_tok / dt:.0f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % a.every == 0:
            save(step)
            print(f"step {step} held {held_loss(held):.4f} wrong_label {held_loss(held, wrong=True):.4f} train {held_loss(train):.4f}", flush=True)
    train_s = time.time() - t0
    save(step)
    model.load_state_dict(ema)
    model.eval()
    res = dict(params=n_params, steps=step, budget=a.budget, train_s=train_s, ema_held=held_loss(held), samples=[])
    print({k: v for k, v in res.items() if k != "samples"}, flush=True)
    tok = Tokenizer("flux2", dev)


    def picture(p):
        t = Tokenizer.unpack(p)
        if t["knots"] is not None:
            t["knots"] = t["knots"].clamp(0, 1).cummax(1).values
        return tok.decode(t)


    for l in range(4):
        for j in range(a.samples):
            g = generate(model, mu, sd, LAYOUTS, SIDE, l, seed=1000 * l + j)
            x = picture(g)
            re = len(Tokenizer.pack(tok.encode(x))["seq"])
            res["samples"].append(dict(label=NAMES[l], j=j, tokens=len(g["seq"]), reencoded=re))
            print(res["samples"][-1], flush=True)
            save_png(x, out / f"{NAMES[l]}_{j}_{len(g['seq'])}.png")
    # same seed and layout under each label: if they match, the label is not reaching the content
    mid = LAYOUTS.index(sorted(LAYOUTS, key=lambda q: q[0])[len(LAYOUTS) // 2])
    for l in range(4):
        save_png(picture(generate(model, mu, sd, LAYOUTS, SIDE, l, li=mid, seed=7)), out / f"same_{NAMES[l]}.png")
    json.dump(res, open(out / "mar.json", "w"))
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()

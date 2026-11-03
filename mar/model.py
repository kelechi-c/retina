"""MAR on retina tokens: a masked autoregressive transformer that writes a variable-length token sequence for one of four content labels, each token drawn by a small flow-matching head.

The model first draws a layout (grid size, warped or not, statistics or not), which fixes how many tokens the picture gets. Knots and statistics come next, in order, then the latent tokens in random order over a cosine schedule.

usage:
    from mar.model import Sampler
    s = Sampler("cuda")                     # EMA weights from the Hugging Face repo
    p = s.generate("simple", seed=0)        # dict(seq [N,128], layout)
    image = s.decode(p)                     # [1,3,512,512] in [-1,1]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

W, NULL = 128, 4  # token width; label index used for the unconditional pass
NAMES = ["simple", "moderate", "textured", "dense"]
REPO, FILE = "tensorkelechi/retina-tokenizer", "mar_v2.pt"


def slots(L):
    """Normalization slot of each token: 0 knots, 1 and 2 statistics, 3 latents."""
    # statistics are 27 floats per 256 px tile: 2 tokens at 768, 1 at 512
    s = [0] * (1 if L["warped"] else 0) + [1, 2][: -(-27 * (L["side"] // 256) ** 2 // W)] * (1 if L["stats"] else 0)
    return torch.tensor(s + [3] * (L["n"] // 2) ** 2)


def coords(L, slot):
    """Row and column in 0..1, so grids of any size share positions. Knots and statistics get their own slot embedding instead."""
    g = L["n"] // 2
    c = torch.zeros(len(slot), 2)
    q = torch.arange(g**2)
    c[-g**2 :, 0], c[-g**2 :, 1] = (q // g + .5) / g, (q % g + .5) / g
    return c


def sinus(x, d):
    f = torch.exp(torch.arange(d // 2, device=x.device) * -math.log(1000.) / max(d // 2 - 1, 1))
    return torch.cat((torch.sin(x[..., None] * 2 * math.pi * f / f[-1] * .5), torch.cos(x[..., None] * 2 * math.pi * f / f[-1] * .5)), -1)


class Res(nn.Module):
    def __init__(s, w):
        super().__init__()
        s.n, s.a, s.b = nn.LayerNorm(w, elementwise_affine=False), nn.Linear(w, w), nn.Linear(w, w)
        s.c = nn.Linear(w, 3 * w)
        nn.init.zeros_(s.c.weight), nn.init.zeros_(s.c.bias)

    def forward(s, x, c):
        g, b, k = s.c(c).chunk(3, -1)
        return x + (1 + k) * s.b(F.silu(s.a(s.n(x) * (1 + g) + b)))


class Head(nn.Module):
    """Flow matching per token: x_t and t, conditioned on the transformer's output for this position."""

    def __init__(s, d, w=512, L=6):
        super().__init__()
        s.i, s.z, s.t = nn.Linear(W, w), nn.Linear(d, w), nn.Sequential(nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w))
        s.w, s.b, s.o = w, nn.ModuleList(Res(w) for _ in range(L)), nn.Linear(w, W)

    def forward(s, x, t, z):
        c = s.z(z) + s.t(sinus(t, s.w))
        h = s.i(x)
        for b in s.b:
            h = b(h, c)
        return s.o(h)


class MAR(nn.Module):
    def __init__(s, d, L, layouts, head_w, head_l):
        super().__init__()
        s.label, s.layout, s.slot = nn.Embedding(5, d), nn.Embedding(layouts, d), nn.Embedding(4, d)
        s.h_layout = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, layouts))
        s.tok, s.pos, s.mask = nn.Linear(W, d), nn.Linear(d, d), nn.Parameter(torch.zeros(d))
        s.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d, d // 64, 4 * d, 0., "gelu", batch_first=True, norm_first=True) for _ in range(L))
        s.norm, s.head, s.d = nn.LayerNorm(d), Head(d, head_w, head_l), d

    def forward(s, y, seen, slot, co, label, lay, pad):
        """y [b,N,W] normalized, seen [b,N] bool: which tokens are known. The label is added to every token and to the head's condition, so it cannot be lost among hundreds of tokens."""
        c = s.label(label)
        e = torch.where(seen[..., None], s.tok(y), s.mask.expand(*y.shape[:2], -1))
        e = e + s.slot(slot) + s.pos(sinus(co, s.d // 2).flatten(-2)) + c[:, None]
        h = torch.cat(((c + s.layout(lay))[:, None], e), 1)
        for b in s.blocks:
            h = b(h, src_key_padding_mask=F.pad(pad, (1, 0)))
        return s.norm(h)[:, 1:] + c[:, None]


def build(args, layouts):
    return MAR(args["d"], args["layers"], layouts, args["head_w"], args["head_l"])


@torch.no_grad()
def generate(model, mu, sd, layouts, side, label, li=None, cfg=1., temp=.95, seed=None, ar_steps=64, flow_steps=32):
    """label 0-3 -> dict(seq, layout), ready for Tokenizer.unpack. li fixes the layout index; otherwise the model draws it, and with it the token count."""
    dev = mu.device
    if seed is not None:
        torch.manual_seed(seed)
    lab = torch.tensor([label], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model.h_layout(model.label(lab)).float()
    li = li if li is not None else torch.multinomial(lg.softmax(-1), 1).item()
    n, w, st = layouts[li]
    L = dict(side=side, n=n, warped=w, stats=st)
    sl = slots(L).to(dev)
    N, head = len(sl), len(sl) - (n // 2) ** 2
    co = coords(L, sl.cpu()).to(dev)[None]
    y, seen = torch.zeros(1, N, W, device=dev), torch.zeros(1, N, dtype=torch.bool, device=dev)
    plan, perm, c0 = [torch.tensor([i], device=dev) for i in range(head)], head + torch.randperm(N - head, device=dev), 0
    K = min(ar_steps, len(perm))
    for k in range(K):
        e = len(perm) if k == K - 1 else round((1 - math.cos(math.pi / 2 * (k + 1) / K)) * len(perm))
        if e > c0:
            plan.append(perm[c0:e])
            c0 = e
    pad, ll = torch.zeros(1, N, dtype=torch.bool, device=dev), torch.tensor([li], device=dev)
    for idx in plan:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(y, seen, sl[None], co, lab, ll, pad)[0, idx]
            g = 1 + (cfg - 1) * seen.float().mean().item()  # guidance rises as the picture fills in
            zn = model(y, seen, sl[None], co, torch.full_like(lab, NULL), ll, pad)[0, idx] if g != 1 else None
            x = torch.randn(len(z), W, device=dev) * temp
            for i in range(flow_steps):
                t = torch.full((len(z),), i / flow_steps, device=dev)
                v = model.head(x, t, z).float()
                if zn is not None:
                    v = (u := model.head(x, t, zn).float()) + g * (v - u)
                x = x + v / flow_steps
        y[0, idx], seen[0, idx] = x.float(), True
    return dict(seq=y[0] * sd[sl] + mu[sl], layout=L)


class Sampler:
    """Loads a checkpoint (the EMA weights unless ema=False) and samples pictures by label."""

    def __init__(self, dev="cuda", ckpt=None, ema=True):
        ck = torch.load(ckpt or hf_hub_download(REPO, FILE), map_location=dev, weights_only=False)
        self.dev, self.mu, self.sd, self.layouts, self.args = dev, ck["mu"].to(dev), ck["sd"].to(dev), ck["LAYOUTS"], ck["args"]
        self.side = ck.get("side", 512)
        self.m = build(self.args, len(self.layouts)).to(dev).eval().requires_grad_(False)
        self.m.load_state_dict(ck["ema"] if ema and "ema" in ck else ck["model"])
        self.tok = None

    def generate(self, label, **kw):
        return generate(self.m, self.mu, self.sd, self.layouts, self.side, NAMES.index(label) if isinstance(label, str) else label, **kw)

    def decode(self, p):
        """Sampled sequence -> image. Sampled knots can step backwards, so they are made monotonic first."""
        from retina import Tokenizer

        self.tok = self.tok or Tokenizer("flux2", self.dev)
        t = Tokenizer.unpack(p)
        if t["knots"] is not None:
            t["knots"] = t["knots"].clamp(0, 1).cummax(1).values
        return self.tok.decode(t)

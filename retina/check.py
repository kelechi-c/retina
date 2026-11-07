"""v1.8 budget and acceptance check (ported from the fovea experiments, archive/experiments/fovea-v2-20261106).

Budget: about twenty content features from the same DReX calls as v1's budget (whole image and nine 256 px tiles), plus gradient,
texture and structure-tensor statistics, go through a ridge fit on the user's own accept/reject labels: one number, the token budget.
Check: one decode is described by LPIPS, MSE, sharpness and DISTS maps against the original, and a logistic model fitted on the user's
blind staircase answers gives the probability it would be accepted. When the content is concentrated (a small busy region on a calm
background), every busy 256 px window is also checked on its own, so a whole-image average cannot hide a destroyed island.
Weights: v18.json (fitted on all labelled images)."""
import json
import math
import pathlib

import torch
import torch.nn.functional as F

from . import split as X

KG = .03  # gradient energy floor per 32 px block: below it the original's gradient is noise, and losing it is not lost sharpness
SIDE = 768  # every map below is laid out for a 768 px image (24 x 24 blocks of 32 px)
WEIGHTS = json.loads((pathlib.Path(__file__).parent / "v18.json").read_text())


def grad(x, B=32):
    y = x.mean(1, keepdim=True)
    g = F.pad((y[..., :, 1:] - y[..., :, :-1]).abs(), (0, 1)) + F.pad((y[..., 1:, :] - y[..., :-1, :]).abs(), (0, 0, 0, 1))
    return F.avg_pool2d(g, B)[0, 0]


def structure(x, B=32):
    """per 32 px block: log structure-tensor energy and coherence (1 = one orientation: edges/text; 0 = isotropic: texture)"""
    y = x.mean(1, keepdim=True)
    kx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).view(1, 1, 3, 3) / 8
    gx, gy = F.conv2d(y, kx, padding=1), F.conv2d(y, kx.transpose(2, 3), padding=1)
    j11, j22, j12 = (F.avg_pool2d(v, B)[0, 0] for v in (gx * gx, gy * gy, gx * gy))
    tr, det = j11 + j22, j11 * j22 - j12 * j12
    disc = ((tr / 2) ** 2 - det).clamp_min(0).sqrt()
    l1, l2 = tr / 2 + disc, (tr / 2 - disc).clamp_min(0)
    return torch.log(tr + 1e-8), (l1 - l2) / (l1 + l2 + 1e-8)


def _q(t, q):
    return t.flatten().float().quantile(q).item()


class Linear:
    """standardised linear model from v18.json: ridge (budget, in log tokens) or logistic (acceptance)"""
    def __init__(self, w):
        self.cols = w.get("cols") or w["names"]
        self.w, self.mu, self.sd = (torch.tensor(w[k]) for k in ("w", "mu", "sd"))

    def z(self, f):
        x = torch.tensor([f[c] for c in self.cols], dtype=torch.float32)
        return (((x - self.mu) / self.sd) @ self.w[1:] + self.w[0]).item()


BUDGET, ACCEPT = Linear(WEIGHTS["budget"]), Linear(WEIGHTS["accept"])


def content(x, dw, d256, tex, b):
    """x [1,3,768,768]; dw whole-image DReX, d256 the nine tile scores, tex texture energy, b v1's budget -> (budget features, content maps)"""
    tx = F.avg_pool2d((x - X.structure(x)).abs().mean(1, keepdim=True), 32)[0, 0]
    go = grad(x); le, coh = structure(x)
    busy = go > KG
    d256 = torch.tensor(d256)
    f = dict(log_b=math.log(b), d=dw, log_tex=math.log(tex + 1e-5),
             d256_mean=d256.mean().item(), d256_max=d256.max().item(), d256_min=d256.min().item(), d256_std=d256.std().item(),
             d256_f3=(d256 > .3).float().mean().item(), d256_f5=(d256 > .5).float().mean().item(),
             tex_p90=math.log(_q(tx, .9) + 1e-5), tex_p50=math.log(_q(tx, .5) + 1e-5),
             go_mean=math.log(go.mean().item() + 1e-4), go_p90=math.log(_q(go, .9) + 1e-4), busy=busy.float().mean().item(),
             le_p50=_q(le, .5), le_p90=_q(le, .9),
             coh_busy=coh[busy].mean().item() if busy.any() else 0., edge_frac=(busy & (coh > .5)).float().mean().item(),
             texture_frac=(busy & (coh < .3)).float().mean().item())
    f["log_b_sq"] = f["log_b"] ** 2
    return f, dict(b=b, go=go.cpu(), tex32=tx.cpu(), d256=d256.view(3, 3))


def budget(f):
    return math.exp(BUDGET.z(f))


def _sl(a, b, s):
    return slice(a // s, max(a // s + 1, math.ceil(b / s)))


def _pool(m):
    return F.avg_pool2d(m[None, None], 4, 2)[0, 0].flatten() if min(m.shape) >= 4 else m.mean()[None]


def features(maps, ct, box, tokens, ds_region=None):
    """acceptance features for the region `box` ([y0,x0,y1,x1] in 768 px, None = whole image)"""
    y0, x0, y1, x1 = box or [0, 0, SIDE, SIDE]
    sy, sx = _sl(y0, y1, 32), _sl(x0, x1, 32)
    lp, mse, gc, go = maps["lp"][sy, sx], maps["mse"][sy, sx], maps["gc"][sy, sx], ct["go"][sy, sx]
    gw, cw = _pool(go), _pool(gc)
    r = torch.log((cw + KG) / (gw + KG)); m = gw >= gw.median()
    t = maps["dists_t128"][_sl(y0, y1, 128), _sl(x0, x1, 128)].flatten()
    if ds_region is None:
        ds_region = maps["dists"] if box is None else t.mean().item()
    return dict(lp_mean=lp.mean().item(), lp_p90=lp.flatten().quantile(.9).item(), lp_win=_pool(lp).max().item(),
                ds_region=ds_region, ds_tile_max=t.max().item(), ds_tile_p75=t.quantile(.75).item(),
                log_mse=math.log(mse.mean().item() + 1e-6), sharp=math.log((gc.mean().item() + KG) / (go.mean().item() + KG)),
                sharp_worst=r[m].min().item(), zoom=math.log(640 / max(y1 - y0, x1 - x0)), log_go=math.log(go.mean().item() + 1e-4),
                log_tex=math.log(ct["tex32"][sy, sx].mean().item() + 1e-4),
                d_max=ct["d256"][_sl(y0, y1, 256), _sl(x0, x1, 256)].max().item(), log_b=math.log(ct["b"]), log_tok=math.log(tokens))


def windows():
    """the 256 px windows at stride 128, as boxes"""
    return [[i, j, i + 256, j + 256] for i in range(0, SIDE - 255, 128) for j in range(0, SIDE - 255, 128)]


def _tiles(x, s, st=None):
    st = st or s
    return torch.cat([x[..., i:i + s, j:j + s] for i in range(0, SIDE - s + 1, st) for j in range(0, SIDE - s + 1, st)])


class Checker:
    """P(the user accepts this decode). q: accept above this; win: busy windows must also pass above this when content is concentrated
    (median / max window gradient energy below conc); busy windows are those with at least half the busiest window's texture energy."""
    def __init__(self, dev, q=.7, win=.5, conc=.25):
        import lpips
        from DISTS_pytorch import DISTS
        self.lp = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(dev).eval().requires_grad_(False)
        self.ds = DISTS().to(dev).eval().requires_grad_(False)
        self.q, self.win, self.conc = q, win, conc

    def _d(self, a, b):
        return self.ds(a, b, batch_average=False).flatten().float().cpu()

    @torch.no_grad()
    def __call__(self, x, c, ct, tokens):
        a, b = (x + 1) / 2, (c + 1) / 2
        m = dict(lp=F.avg_pool2d(self.lp(x, c), 32)[0, 0].cpu(), mse=F.avg_pool2d(((x - c) / 2).pow(2).mean(1, keepdim=True), 32)[0, 0].cpu(),
                 gc=grad(c).cpu(), dists=self._d(a, b).item(), dists_t128=self._d(_tiles(a, 128), _tiles(b, 128)).view(6, 6))
        p = 1 / (1 + math.exp(-ACCEPT.z(features(m, ct, None, tokens))))
        wb = windows()
        g = sorted(ct["go"][w[0] // 32:w[2] // 32, w[1] // 32:w[3] // 32].mean().item() for w in wb)
        if self.win and p > self.q and g[len(g) // 2] / max(g[-1], 1e-6) < self.conc:
            e = [ct["tex32"][w[0] // 32:w[2] // 32, w[1] // 32:w[3] // 32].mean().item() for w in wb]
            busy = [k for k in range(len(wb)) if e[k] >= .5 * max(e)]
            dw = self._d(_tiles(a, 256, 128)[busy], _tiles(b, 256, 128)[busy])
            pw = min(1 / (1 + math.exp(-ACCEPT.z(features(m, ct, wb[k], tokens, dw[j].item())))) for j, k in enumerate(busy))
            if pw <= self.win:
                p = min(p, self.q)
        return p
